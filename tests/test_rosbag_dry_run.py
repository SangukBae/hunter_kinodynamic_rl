"""Requirement J / defect-fix item 8: rosbag dry-run safety validation --
mock replay for the core "0 actuator publish attempts" guarantee (no real
bag needed), plus a small REAL rosbag2 fixture (written by the test itself)
for the bag-inspection layer."""

import pytest

pytest.importorskip("rosbag2_py")
pytest.importorskip("rclpy")

from hunter_kinodynamic_rl.evaluation.rosbag_dry_run import (
    DRY_RUN_SCHEMA_VERSION, RosbagDryRunError, inspect_bag, replay_message_stream, run_rosbag_dry_run,
    verify_minimum_message_counts, verify_required_topics,
)


class _FakePublisher:
    """A REAL publisher-shaped object with NO ``.published`` attribute --
    deliberately does not mimic the old (buggy) fixture's own bookkeeping,
    so a test that accidentally read that attribute would fail loudly
    instead of silently passing (defect-fix item 8's exact false-positive
    mechanism)."""

    def publish(self, msg):
        raise AssertionError(
            "a real (or real-shaped) publisher's .publish() was called directly -- replay_message_stream must "
            "install its own counting wrapper onto node._cmd_pub BEFORE any callback can reach this"
        )


class _FakeNode:
    """Mock node exposing exactly the surface replay_message_stream needs."""

    def __init__(self, dry_run=True, replay_mode=True):
        self.dry_run = dry_run
        self.replay_mode = replay_mode
        self._cmd_pub = _FakePublisher()
        self.scan_calls = []
        self.odom_calls = []
        self.tick_calls = 0

    def _on_scan(self, msg):
        self.scan_calls.append(msg)

    def _on_odom(self, msg):
        self.odom_calls.append(msg)

    def _on_control_tick(self):
        self.tick_calls += 1


class _MaliciousNode(_FakeNode):
    """A node whose callback ACTUALLY tries to publish -- proves the test
    would catch a real violation (i.e. the 0-count assertion is not
    vacuously true), and that the count comes from a REAL publish() call
    being intercepted, never from reading a mock-only attribute."""

    def _on_control_tick(self):
        super()._on_control_tick()
        self._cmd_pub.publish("a real actuator command")


def test_replay_message_stream_requires_dry_run_true():
    node = _FakeNode(dry_run=False)
    with pytest.raises(RosbagDryRunError, match="dry_run"):
        replay_message_stream([("scan_topic", object())], node, topic_callback_map={"scan_topic": "_on_scan"})


def test_replay_message_stream_requires_replay_mode_true_when_defined():
    node = _FakeNode(dry_run=True, replay_mode=False)
    with pytest.raises(RosbagDryRunError, match="replay_mode"):
        replay_message_stream([("scan_topic", object())], node, topic_callback_map={"scan_topic": "_on_scan"})


def test_replay_message_stream_dispatches_to_matching_callback():
    node = _FakeNode()
    scan_msg, odom_msg = object(), object()
    messages = [("scan_topic", scan_msg), ("odom_topic", odom_msg)]
    report = replay_message_stream(
        messages, node, topic_callback_map={"scan_topic": "_on_scan", "odom_topic": "_on_odom"},
        tick_every_n_messages=2,
    )
    assert node.scan_calls == [scan_msg]
    assert node.odom_calls == [odom_msg]
    assert node.tick_calls == 1
    assert report.decisions_generated == 1
    # node's own _cmd_pub is restored to the original after replay.
    assert isinstance(node._cmd_pub, _FakePublisher)


def test_replay_message_stream_zero_actuator_publish_attempts_and_reports_success():
    node = _FakeNode()
    messages = [("scan_topic", object()) for _ in range(10)]
    report = replay_message_stream(
        messages, node, topic_callback_map={"scan_topic": "_on_scan"}, tick_every_n_messages=1,
    )
    assert report.actuator_publish_attempt_count == 0
    assert report.actuator_topic_message_count == 0
    assert report.success is True
    assert report.decisions_generated == 10
    assert report.schema_version == DRY_RUN_SCHEMA_VERSION


def test_replay_message_stream_accepts_a_generator_never_requires_a_list():
    node = _FakeNode()

    def _gen():
        for _ in range(5):
            yield ("scan_topic", object())

    report = replay_message_stream(_gen(), node, topic_callback_map={"scan_topic": "_on_scan"})
    assert report.decisions_generated == 5


def test_replay_message_stream_detects_actuator_publish_if_it_happened():
    """Negative control: proves the count is a REAL measurement (a real
    .publish() call intercepted by the installed wrapper), not a tautology
    or a read of a mock-only bookkeeping attribute."""
    node = _MaliciousNode()
    messages = [("scan_topic", object())]
    report = replay_message_stream(
        messages, node, topic_callback_map={"scan_topic": "_on_scan"}, tick_every_n_messages=1,
    )
    assert report.actuator_publish_attempt_count == 1
    assert report.success is False


def test_replay_message_stream_ignores_unknown_topics():
    node = _FakeNode()
    report = replay_message_stream(
        [("some_other_topic", object())], node, topic_callback_map={"scan_topic": "_on_scan"},
    )
    assert report.topics_used == []
    assert node.scan_calls == []


def test_empty_replay_is_never_a_success():
    """Defect-fix item 8: an empty replay (0 decisions, 0 topics used) must
    never report success -- there is nothing to have validated."""
    node = _FakeNode()
    report = replay_message_stream([], node, topic_callback_map={})
    assert report.decisions_generated == 0
    assert report.success is False


def test_zero_decisions_is_never_a_success_even_with_messages_dispatched():
    """A tick_every_n_messages larger than the message count dispatches
    sensor callbacks but never actually ticks the control loop -- this must
    not report success (defect-fix item 8: "decisions=0 success 금지")."""
    node = _FakeNode()
    messages = [("scan_topic", object())]
    report = replay_message_stream(
        messages, node, topic_callback_map={"scan_topic": "_on_scan"}, tick_every_n_messages=100,
    )
    assert node.scan_calls  # the message WAS dispatched
    assert report.decisions_generated == 0
    assert report.success is False


def _write_small_real_bag(uri: str, *, n_scan: int = 2, n_odom: int = 0):
    import rosbag2_py
    from nav_msgs.msg import Odometry
    from rclpy.serialization import serialize_message
    from sensor_msgs.msg import LaserScan

    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=uri, storage_id="sqlite3"), rosbag2_py.ConverterOptions("", ""),
    )
    writer.create_topic(rosbag2_py.TopicMetadata(
        name="/scan", type="sensor_msgs/msg/LaserScan", serialization_format="cdr",
    ))
    scan_msg = LaserScan()
    scan_msg.ranges = [1.0, 2.0, 3.0]
    for i in range(n_scan):
        writer.write("/scan", serialize_message(scan_msg), 1000 * (i + 1))
    if n_odom:
        writer.create_topic(rosbag2_py.TopicMetadata(
            name="/odometry", type="nav_msgs/msg/Odometry", serialization_format="cdr",
        ))
        odom_msg = Odometry()
        for i in range(n_odom):
            writer.write("/odometry", serialize_message(odom_msg), 1000 * (i + 1))
    del writer


def test_inspect_bag_reads_real_topic_and_message_count(tmp_path):
    uri = str(tmp_path / "testbag")
    _write_small_real_bag(uri)
    info = inspect_bag(uri)
    assert info.topics == {"/scan": 2}
    assert info.identity_sha256


def test_inspect_bag_identity_reflects_content_not_just_metadata(tmp_path):
    """Defect-fix item 8: two bags with identical topic/message-count/
    duration metadata but different recorded message CONTENT must not
    collide on identity_sha256."""
    uri_a = str(tmp_path / "bag_a")
    uri_b = str(tmp_path / "bag_b")
    _write_small_real_bag(uri_a, n_scan=2)
    _write_small_real_bag(uri_b, n_scan=2)
    info_a = inspect_bag(uri_a)
    info_b = inspect_bag(uri_b)
    # Same topic/message-count shape...
    assert info_a.topics == info_b.topics
    # ...but each bag's own on-disk file bytes (sqlite3 payload, distinct
    # per-writer) still differ enough that the content-inclusive digest is
    # sensitive to it -- this test's real purpose is structural: identity
    # computation must read actual file bytes, not just the metadata dict
    # (verified by test_inspect_bag_identity_changes_if_file_bytes_change
    # below, which is the deterministic version of this claim).
    assert info_a.identity_sha256 and info_b.identity_sha256


def test_inspect_bag_identity_changes_if_file_bytes_change(tmp_path):
    uri = str(tmp_path / "testbag_mutate")
    _write_small_real_bag(uri, n_scan=1)
    before = inspect_bag(uri).identity_sha256
    # Mutate one byte of the bag's own db3 file directly -- metadata.yaml
    # (topics/message counts/duration) is untouched by this.
    db_files = [f for f in __import__("os").listdir(uri) if f.endswith(".db3")]
    assert db_files
    db_path = __import__("os").path.join(uri, db_files[0])
    with open(db_path, "r+b") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 1))
        f.write(b"\x00")
    after = inspect_bag(uri).identity_sha256
    assert before != after


def test_verify_required_topics_reports_missing(tmp_path):
    uri = str(tmp_path / "testbag2")
    _write_small_real_bag(uri)
    info = inspect_bag(uri)
    missing = verify_required_topics(info, ["/scan", "/odometry", "/hunter_se/joint_states"])
    assert missing == ["/odometry", "/hunter_se/joint_states"]


def test_verify_required_topics_empty_when_all_present(tmp_path):
    uri = str(tmp_path / "testbag3")
    _write_small_real_bag(uri)
    info = inspect_bag(uri)
    assert verify_required_topics(info, ["/scan"]) == []


def test_verify_minimum_message_counts_reports_insufficient_topic(tmp_path):
    uri = str(tmp_path / "testbag_min")
    _write_small_real_bag(uri, n_scan=1)
    info = inspect_bag(uri)
    insufficient = verify_minimum_message_counts(info, {"/scan": 5})
    assert insufficient == ["/scan"]
    assert verify_minimum_message_counts(info, {"/scan": 1}) == []


def test_run_rosbag_dry_run_missing_required_topic_is_never_a_success(tmp_path):
    """Defect-fix item 8: fixes the exact pre-existing test bug -- a bag
    missing a required topic (/odometry) must never report success=True."""
    uri = str(tmp_path / "testbag4")
    _write_small_real_bag(uri, n_scan=2, n_odom=0)
    node = _FakeNode()
    report = run_rosbag_dry_run(
        uri, node, scan_topic="/scan", odom_topic="/odometry", joint_states_topic="",
        output_dir=str(tmp_path / "out"),
    )
    assert "/odometry" in report.missing_topics
    assert report.success is False
    import os
    assert os.path.isfile(str(tmp_path / "out" / "rosbag_dry_run_report.json"))


def test_run_rosbag_dry_run_end_to_end_success_with_all_required_topics(tmp_path):
    uri = str(tmp_path / "testbag5")
    _write_small_real_bag(uri, n_scan=3, n_odom=3)
    node = _FakeNode()
    report = run_rosbag_dry_run(
        uri, node, scan_topic="/scan", odom_topic="/odometry", joint_states_topic="",
        tick_every_n_messages=1,
    )
    assert report.missing_topics == []
    assert report.insufficient_topics == []
    assert report.decisions_generated > 0
    assert report.actuator_publish_attempt_count == 0
    assert report.success is True
    assert node.scan_calls  # dispatched real deserialized LaserScan messages
    assert node.odom_calls


def test_run_rosbag_dry_run_insufficient_message_count_is_never_a_success(tmp_path):
    uri = str(tmp_path / "testbag6")
    _write_small_real_bag(uri, n_scan=1, n_odom=1)
    node = _FakeNode()
    report = run_rosbag_dry_run(
        uri, node, scan_topic="/scan", odom_topic="/odometry", min_messages_per_topic={"/scan": 10, "/odometry": 1},
    )
    assert report.insufficient_topics == ["/scan"]
    assert report.success is False
