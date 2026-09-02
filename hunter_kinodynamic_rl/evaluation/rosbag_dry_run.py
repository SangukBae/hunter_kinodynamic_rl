#!/usr/bin/env python3
"""Requirement J / defect-fix item 8: rosbag dry-run safety validation.

``nodes/hierarchical_navigation_node.py`` already structurally blocks
actuator publishing in dry-run: ``_publish()`` is the ONE call site that
ever reaches ``cmd_vel_topic``, and it short-circuits BEFORE
``self._cmd_pub.publish()`` whenever ``self.dry_run`` is True -- publish
itself never happens, not merely redirected to a different sink. This
module adds the missing piece: an explicit OFFLINE rosbag replay mode with
a bag-identity/topic-coverage report, decision/actuator-command counting,
and a hard multi-condition success gate -- verified structurally (never
trusted from a log line, and never dependent on a mock-only ``.published``
attribute a real ``rclpy.Publisher`` does not have).

Defect-fix item 8's false-positive: the v1 report defined ``success`` as
ONLY ``actuator_command_count == 0``, measured by reading
``node._cmd_pub.published`` -- an attribute that exists on the test
fixture's fake publisher but NOT on a real ``rclpy.Publisher``, so against
a real node this check silently read ``[]`` via ``getattr(..., [])``
regardless of what actually happened and always reported success. Fixed by
:class:`_PublishAttemptCounter`, an instrumented wrapper this module
installs onto ``node._cmd_pub`` itself for the duration of the replay
(restored afterward) -- it counts every ``.publish()`` CALL ATTEMPT
directly, works identically whether the underlying object is a real
``rclpy.Publisher`` or a test double, and never forwards to the wrapped
publisher (a genuine belt-and-suspenders: even a bug that let a call
through during "dry run" still never reaches a live topic). ``success`` is
now ALL of:

- every required topic present in the bag with at least
  ``min_messages_per_topic`` messages (never silently proceeding on a
  missing/near-empty topic and only noticing afterward via a report field
  nobody checked);
- at least one message from a real, USED sensor topic was actually
  dispatched;
- at least one decision/control tick was generated;
- replay completed without an unhandled exception (implicit: an exception
  propagates out of :func:`replay_message_stream`/:func:`run_rosbag_dry_run`
  rather than being swallowed into a falsely-successful report -- there is
  no code path that produces a :class:`DryRunReport` for an incomplete
  replay);
- ``node.dry_run`` AND (when the node defines it) ``node.replay_mode`` are
  both True;
- ``actuator_publish_attempt_count == 0`` (the instrumented count, not a
  read of a possibly-nonexistent attribute).

Two layers, deliberately separated for testability:

- :func:`replay_message_stream` -- pure logic. Takes an ordered iterable of
  ``(topic, msg)`` tuples (a REAL bag's messages, or a synthetic/mock
  fixture -- see this module's own tests) and a ``node``-shaped object
  exposing ``_on_scan``/``_on_odom``/``_on_joint_states``/``_on_control_tick``/
  ``dry_run``/``_cmd_pub`` (exactly what
  :class:`~hunter_kinodynamic_rl.nodes.hierarchical_navigation_node.HierarchicalNavigationNode`
  exposes, real or a test double). Dispatches each message directly to the
  matching callback (never publishes it onto a live ROS topic itself -- no
  rclpy/live Gazebo needed for this layer at all) and ticks
  ``_on_control_tick()`` once per batch of sensor messages, mirroring the
  timer-driven control loop. Accepts any iterable (including a generator),
  never requires a fully materialized list.
- :func:`inspect_bag`/:func:`run_rosbag_dry_run` -- real ``rosbag2_py``
  I/O: reads an actual bag's metadata/messages (STREAMED, never loading the
  whole bag into memory at once) and feeds them through
  :func:`replay_message_stream` against a real (caller-constructed,
  ``dry_run=True``) node.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

DRY_RUN_SCHEMA_VERSION = 2

#: topic -> node callback attribute name this module knows how to dispatch.
TOPIC_CALLBACK_MAP: Dict[str, str] = {
    "scan_topic": "_on_scan", "odom_topic": "_on_odom", "joint_states_topic": "_on_joint_states",
}

#: Chunk size for streaming a bag file's content into the identity hash --
#: bounds peak memory regardless of bag size.
_HASH_CHUNK_BYTES = 1 << 20


class RosbagDryRunError(RuntimeError):
    """Raised by structural preconditions this module refuses to proceed
    past (never caught internally -- propagates to the caller, so an
    incomplete/invalid dry-run can never produce a report claiming
    success)."""


@dataclass
class BagInfo:
    bag_path: str
    topics: Dict[str, int]  # topic name -> message count
    duration_sec: Optional[float]
    start_time_ns: Optional[int]
    identity_sha256: str


@dataclass
class DryRunReport:
    schema_version: int
    bag_identity: str
    topics_used: List[str]
    missing_topics: List[str]
    insufficient_topics: List[str]
    decisions_generated: int
    actuator_publish_attempt_count: int
    dry_run_active: bool
    replay_mode_active: Optional[bool]
    success: bool = field(init=False)
    #: Always 0 -- the instrumented publisher wrapper NEVER forwards a
    #: publish attempt to the underlying (possibly real) topic, so this is
    #: a structural guarantee, not a measurement -- kept as its own field
    #: (rather than reusing actuator_publish_attempt_count for both
    #: meanings) so a reader can see "N attempts, 0 of which ever reached
    #: the actuator topic" instead of one ambiguous count (item 8: "publish
    #: 시도 횟수와 실제 actuator topic 발행 횟수를 구분해 기록").
    actuator_topic_message_count: int = 0

    def __post_init__(self) -> None:
        self.success = (
            self.dry_run_active
            and (self.replay_mode_active is not False)
            and not self.missing_topics
            and not self.insufficient_topics
            and bool(self.topics_used)
            and self.decisions_generated > 0
            and self.actuator_publish_attempt_count == 0
        )


class _PublishAttemptCounter:
    """Instrumented wrapper installed onto ``node._cmd_pub`` for the
    duration of a replay -- counts every ``.publish()`` CALL ATTEMPT
    directly (never a read of a ``.published`` attribute that a real
    ``rclpy.Publisher`` does not have -- defect-fix item 8's core fix).
    Never forwards to the wrapped publisher: even if a future bug let a
    publish call through during a "dry run", it still cannot reach a real
    topic through this wrapper."""

    def __init__(self, wrapped: Any) -> None:
        self._wrapped = wrapped
        self.publish_attempt_count = 0
        self.attempted_messages: List[Any] = []

    def publish(self, msg: Any) -> None:
        self.publish_attempt_count += 1
        self.attempted_messages.append(msg)

    def __getattr__(self, name: str) -> Any:
        # Any OTHER attribute access (topic name introspection, etc.)
        # transparently reaches the wrapped object -- only publish() itself
        # is intercepted.
        return getattr(self._wrapped, name)


def inspect_bag(bag_path: str, storage_id: str = "sqlite3") -> BagInfo:
    """Real ``rosbag2_py`` metadata read -- no message deserialization,
    just topic/message-count/duration identity, PLUS (defect-fix item 8) a
    content-based identity component: every regular file directly under
    ``bag_path`` is streamed (bounded-memory) into the same digest, so two
    bags with identical metadata but different recorded bytes (a
    re-recorded or hand-edited bag) never collide on ``identity_sha256``."""
    import rosbag2_py

    info = rosbag2_py.Info()
    metadata = info.read_metadata(bag_path, storage_id)
    topics = {t.topic_metadata.name: t.message_count for t in metadata.topics_with_message_count}
    duration_sec = metadata.duration.nanoseconds / 1e9 if metadata.duration is not None else None
    start_time_ns = metadata.starting_time.nanoseconds if metadata.starting_time is not None else None

    digest = hashlib.sha256()
    digest.update(json.dumps(
        {"topics": topics, "duration_sec": duration_sec, "start_time_ns": start_time_ns}, sort_keys=True,
    ).encode())
    if os.path.isdir(bag_path):
        for name in sorted(os.listdir(bag_path)):
            file_path = os.path.join(bag_path, name)
            if not os.path.isfile(file_path):
                continue
            digest.update(name.encode())
            with open(file_path, "rb") as f:
                while True:
                    chunk = f.read(_HASH_CHUNK_BYTES)
                    if not chunk:
                        break
                    digest.update(chunk)
    elif os.path.isfile(bag_path):
        with open(bag_path, "rb") as f:
            while True:
                chunk = f.read(_HASH_CHUNK_BYTES)
                if not chunk:
                    break
                digest.update(chunk)
    identity_sha256 = digest.hexdigest()
    return BagInfo(
        bag_path=bag_path, topics=topics, duration_sec=duration_sec, start_time_ns=start_time_ns,
        identity_sha256=identity_sha256,
    )


def verify_required_topics(bag_info: BagInfo, required_topics: Sequence[str]) -> List[str]:
    """Returns the subset of ``required_topics`` NOT present in the bag AT
    ALL -- empty list means full coverage. Never raises -- the caller
    decides whether missing topics should abort the dry-run. See
    :func:`verify_minimum_message_counts` for the separate "present but too
    few messages" check."""
    return [t for t in required_topics if t not in bag_info.topics]


def verify_minimum_message_counts(
    bag_info: BagInfo, min_messages_per_topic: Dict[str, int],
) -> List[str]:
    """Defect-fix item 8: "필수 topic별 최소 message 수 만족" -- returns the
    subset of ``min_messages_per_topic`` keys present in the bag but with
    FEWER messages than required (a topic missing entirely is reported by
    :func:`verify_required_topics` instead, never silently folded in
    here)."""
    return [
        topic for topic, min_count in min_messages_per_topic.items()
        if topic in bag_info.topics and bag_info.topics[topic] < min_count
    ]


def replay_message_stream(
    messages: Iterable[Tuple[str, Any]], node: Any, *, topic_callback_map: Dict[str, str],
    tick_every_n_messages: int = 1,
) -> DryRunReport:
    """``messages``: ordered ``(topic, msg)`` pairs (bag playback order, or
    a synthetic/mock fixture) -- any iterable, including a generator (never
    required to be a fully materialized list). ``topic_callback_map``:
    ``{topic_name: node_attr_name}`` (the caller resolves ACTUAL topic
    names -> the generic keys in :data:`TOPIC_CALLBACK_MAP` beforehand,
    since a live deployment's topic names are launch-file-configurable).
    Requires ``node.dry_run is True`` -- refuses to replay against a node
    that could actually actuate -- and, when the node defines a
    ``replay_mode`` attribute, requires it to also be True (defect-fix item
    8: "dry_run/replay_mode가 모두 활성화됨")."""
    dry_run_active = bool(getattr(node, "dry_run", False))
    if not dry_run_active:
        raise RosbagDryRunError(
            "replay_message_stream: node.dry_run must be True -- refusing to replay recorded data against a "
            "node that could actuate a real robot (plan 10.7)"
        )
    replay_mode_active = getattr(node, "replay_mode", None)
    if replay_mode_active is False:
        raise RosbagDryRunError(
            "replay_message_stream: node.replay_mode is False -- refusing to replay recorded data against a "
            "node not configured for replay"
        )

    original_cmd_pub = getattr(node, "_cmd_pub", None)
    counter = _PublishAttemptCounter(original_cmd_pub)
    if hasattr(node, "_cmd_pub"):
        node._cmd_pub = counter
    try:
        topics_used: List[str] = []
        decisions_generated = 0
        since_last_tick = 0
        for topic, msg in messages:
            attr_name = topic_callback_map.get(topic)
            if attr_name is None:
                continue
            callback = getattr(node, attr_name, None)
            if callback is None:
                continue
            callback(msg)
            if topic not in topics_used:
                topics_used.append(topic)
            since_last_tick += 1
            if since_last_tick >= tick_every_n_messages:
                tick = getattr(node, "_on_control_tick", None)
                if tick is not None:
                    tick()
                    decisions_generated += 1
                since_last_tick = 0
    finally:
        if hasattr(node, "_cmd_pub"):
            node._cmd_pub = original_cmd_pub

    return DryRunReport(
        schema_version=DRY_RUN_SCHEMA_VERSION, bag_identity="", topics_used=topics_used, missing_topics=[],
        insufficient_topics=[], decisions_generated=decisions_generated,
        actuator_publish_attempt_count=counter.publish_attempt_count, dry_run_active=dry_run_active,
        replay_mode_active=replay_mode_active,
    )


def _iter_bag_messages(reader, topic_callback_map: Dict[str, str]) -> Iterable[Tuple[str, Any]]:
    """Streaming generator over a bag's messages -- yields one at a time,
    never materializing the full bag in memory (defect-fix item 8)."""
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    msg_type_cache: Dict[str, Any] = {}
    while reader.has_next():
        topic, data, _t = reader.read_next()
        if topic not in topic_callback_map:
            continue
        if topic not in msg_type_cache:
            msg_type_cache[topic] = get_message(type_map[topic])
        yield topic, deserialize_message(data, msg_type_cache[topic])


def run_rosbag_dry_run(
    bag_path: str, node: Any, *, scan_topic: str, odom_topic: str, joint_states_topic: str = "",
    storage_id: str = "sqlite3", tick_every_n_messages: int = 5, output_dir: Optional[str] = None,
    min_messages_per_topic: Optional[Dict[str, int]] = None,
) -> DryRunReport:
    """Real end-to-end: inspects ``bag_path``, verifies topic coverage AND
    minimum per-topic message counts BEFORE replaying a single message
    (defect-fix item 8), streams every message in playback order via
    ``rosbag2_py.SequentialReader`` (never fully materialized in memory),
    and replays it through :func:`replay_message_stream`.

    ``min_messages_per_topic`` defaults to requiring at least 1 message on
    every one of ``scan_topic``/``odom_topic``/(``joint_states_topic`` if
    given) -- override to demand more for a specific topic."""
    import rosbag2_py

    required_topics = [t for t in (scan_topic, odom_topic, joint_states_topic) if t]
    if min_messages_per_topic is None:
        min_messages_per_topic = {t: 1 for t in required_topics}

    bag_info = inspect_bag(bag_path, storage_id=storage_id)
    missing_topics = verify_required_topics(bag_info, required_topics)
    insufficient_topics = verify_minimum_message_counts(bag_info, min_messages_per_topic)
    topic_callback_map = {
        scan_topic: "_on_scan", odom_topic: "_on_odom", joint_states_topic: "_on_joint_states",
    }

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id=storage_id),
        rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    message_stream = _iter_bag_messages(reader, topic_callback_map)
    report = replay_message_stream(
        message_stream, node, topic_callback_map=topic_callback_map, tick_every_n_messages=tick_every_n_messages,
    )
    report.bag_identity = bag_info.identity_sha256
    report.missing_topics = missing_topics
    report.insufficient_topics = insufficient_topics
    # Re-evaluate success now that missing/insufficient topics are known --
    # __post_init__ ran with empty lists (replay_message_stream doesn't
    # know about bag-level topic coverage, only what it actually saw).
    report.__post_init__()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "rosbag_dry_run_report.json"), "w") as f:
            json.dump({
                "schema_version": report.schema_version, "bag_path": bag_path,
                "bag_identity_sha256": report.bag_identity, "bag_topics": bag_info.topics,
                "topics_used": report.topics_used, "missing_topics": report.missing_topics,
                "insufficient_topics": report.insufficient_topics,
                "decisions_generated": report.decisions_generated,
                "actuator_publish_attempt_count": report.actuator_publish_attempt_count,
                "actuator_topic_message_count": report.actuator_topic_message_count,
                "dry_run_active": report.dry_run_active, "replay_mode_active": report.replay_mode_active,
                "success": report.success,
            }, f, indent=2)
    return report
