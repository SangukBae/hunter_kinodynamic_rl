#!/usr/bin/env python3
"""Shared off-policy training loop: talks to environment_node.py's
Reset/Step/GetDimensions/Seed services AND its risk-telemetry side channel
(env/simulation/risk_telemetry.py) as a CLIENT, fills a ReplayBuffer, and
drives an Agent's ``train_step``. ``train_tqc.py`` (vanilla) and
``train_kinodynamic_tqc.py`` (risk-aware) are thin subclasses.

Seed ownership (section 3.4): THIS class owns the authoritative
:class:`SeedScheduler`, not the environment node -- the env node's own
internal scheduler is only a fallback for standalone debugging. The trainer
calls ``env.seed(scheduler.next_seed())`` before every ``/reset``, and
persists ``scheduler.state_dict()`` in every checkpoint (section 9), so a
resumed run continues the EXACT same episode-seed sequence regardless of
whether the environment node was restarted.
"""

from __future__ import annotations

import json
import os
import random
import time
from datetime import datetime
from typing import Optional

import math

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rcl_interfaces.srv import GetParameters, SetParameters
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter, parameter_value_to_python
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray

from drl_agent_interfaces.srv import GetDimensions, Reset, Seed, Step

from hunter_kinodynamic_rl.common.seed import enable_torch_determinism, seed_all
from hunter_kinodynamic_rl.config.schema import Profile
from hunter_kinodynamic_rl.env.randomization.domain_randomizer import sample_draw
from hunter_kinodynamic_rl.env.scenarios.seed_scheduler import SeedScheduler
from hunter_kinodynamic_rl.env.simulation import risk_telemetry as rt
from hunter_kinodynamic_rl.env.simulation import sensor_diagnostics as sd
from hunter_kinodynamic_rl.evaluation.fingerprint import (
    architecture_fingerprint,
    architecture_fingerprint_from_resolved_config,
    local_training_contract_fingerprint,
    local_training_contract_fingerprint_from_resolved_config,
)
from hunter_kinodynamic_rl.rl.checkpointing import manager as ckpt_manager
from hunter_kinodynamic_rl.rl.replay.buffer import ReplayBuffer, RiskTransition
from hunter_kinodynamic_rl.training.checkpoint_policy import checkpoint_due
from hunter_kinodynamic_rl.trajectory.action_space import (
    TrajectoryCommand, decode_action, trajectory_command_to_normalized,
)
from hunter_kinodynamic_rl.training.run_logger import RunLogger, run_directory


def telemetry_matches_step(t: Optional[rt.RiskTelemetry], expected_step_id: int,
                            expected_reset_generation: Optional[int]) -> bool:
    """Pure predicate (no ROS) factored out of
    EnvironmentClient._await_matching_telemetry purely so it's directly
    unit-testable without a live rclpy graph -- see tests/test_trainer_telemetry_sync.py.
    ``expected_reset_generation is None`` (impossible in normal operation
    now that ``reset()`` learns it directly and synchronously from
    Reset.srv's own response before any step can be taken -- kept as a
    defensive "can never match" fallback, never intentionally relied on)
    can never match anything."""
    return (
        t is not None and expected_reset_generation is not None
        and t.step_id == expected_step_id and t.reset_generation == expected_reset_generation
    )


def telemetry_is_new_reset_marker(t: Optional[rt.RiskTelemetry], known_generation: int) -> bool:
    """Pure predicate (no ROS) factored out of
    EnvironmentClient._await_new_reset_marker. A genuine RESET_MARKER
    message (step_id=0, InvalidReason.RESET_MARKER) whose reset_generation
    is STRICTLY GREATER than ``known_generation`` -- the highest generation
    THIS client had already observed before issuing its current ``/reset``
    call (see ``EnvironmentClient.reset()``).

    section P1-5 (shared-interface preservation): ``drl_agent_interfaces/srv/Reset.srv``
    is intentionally left UNMODIFIED -- reset_generation is learned
    PURELY from this package's own risk-telemetry broadcast topic
    (``/hunter_kinodynamic_rl/risk_telemetry``), never from a Reset.srv
    response field. A prior version of this module DID add a
    ``reset_generation`` field to Reset.srv's response specifically to
    close a residual race a purely-observational marker-polling scheme
    could not (a DIFFERENT, CONCURRENT client's marker landing between
    this client's own request and its own marker is indistinguishable
    from "my own marker, just slow to arrive" under ANY heuristic
    operating on a broadcast topic alone) -- code review found that
    change violated this project's "shared interfaces stay untouched"
    requirement, so it was reverted (see docs/SOURCE_MAP.md).

    RESULT: this closes the SAME race the previous scheme handled for the
    single-owner-client case (exactly one client calling ``/reset`` on a
    given environment_node instance at a time -- true of every shipped
    profile/launch path in this package: one trainer XOR one evaluation
    run owns a given environment_node process), by relying on
    ``environment_node.py``'s own ``/reset``-callback serialization
    (``MutuallyExclusiveCallbackGroup`` -- see that module's docstring): no
    second ``_on_reset`` invocation, and therefore no second marker, can
    begin publishing until this client's own in-flight request has
    returned its response. It does NOT reproduce the fully general,
    genuinely-concurrent-multi-client guarantee the reverted field
    provided -- two DIFFERENT clients issuing overlapping ``/reset`` calls
    against the SAME environment_node instance is an unsupported
    configuration for this package (documented, not silently
    best-effort)."""
    return (
        t is not None and t.step_id == 0 and t.invalid_reason == int(rt.InvalidReason.RESET_MARKER)
        and t.reset_generation > known_generation
    )


def sensor_diagnostics_matches_step(d: Optional[sd.SensorDiagnostics], expected_step_id: int,
                                     expected_reset_generation: Optional[int]) -> bool:
    """requirement 5: the ``sensor_diagnostics`` analogue of
    ``telemetry_matches_step`` -- same (reset_generation, step_id) pairing
    rule, on the SEPARATE sensor_diagnostics channel (never risk_telemetry's
    own cache), so a trainer can never accidentally pair a stale/mismatched
    diagnostics sample to the wrong episode/step."""
    return (
        d is not None and expected_reset_generation is not None
        and d.step_id == expected_step_id and d.reset_generation == expected_reset_generation
    )


class EnvServiceError(RuntimeError):
    """Raised when environment_node.py's services never come up, or a
    service call's response never arrives, within this client's bounded
    patience. Mirrors environment_node.py's own GazeboServiceError -- a
    dead env node / crashed Gazebo must surface as a clear, bounded
    failure the trainer can catch and stop on, never an unbounded hang
    (section 13: "environment node 종료/Gazebo failure/service exception을
    trainer가 감지하고 종료하도록 한다")."""


class _RiskTelemetryListener(Node):
    """A DEDICATED node + (separately, in EnvironmentClient) a DEDICATED
    executor for JUST the ``/hunter_kinodynamic_rl/risk_telemetry``
    subscription -- code review (risk-telemetry correctness bug): confirmed
    LIVE that sharing a node/executor between this subscription and
    ``/clock`` (which EnvironmentClient also subscribes to, for
    actual_dt_sec logging) makes ``rclpy.spin_once()`` essentially NEVER
    reach the telemetry callback within any bounded polling budget --
    Gazebo's ``/clock`` publishes at an extreme rate (~10,000+ messages
    observed in a 2-second poll window) and ``spin_once()`` services at
    most ONE ready entity per call, so a wait-set that also contains a
    constantly-ready ``/clock`` subscription starves every other
    subscription on the SAME node/executor of a fair chance to ever be
    picked, independent of total time budget. Isolating this subscription
    onto its own node with its OWN ``rclpy.executors.SingleThreadedExecutor``
    (never the implicit per-context default executor -- see
    EnvironmentClient's own comment on why ``rclpy.spin_once(node, ...)``'s
    convenience form does NOT actually isolate anything, since it routes
    through that same shared default executor) fixes this: confirmed live,
    matching succeeded on 100% of steps once isolated, vs. 0% before."""

    def __init__(self, node_name: str):
        super().__init__(node_name)
        self.latest_telemetry: Optional[rt.RiskTelemetry] = None
        self.create_subscription(Float32MultiArray, "/hunter_kinodynamic_rl/risk_telemetry",
                                  self._on_risk_telemetry, 10)
        # requirement 5: the SAME isolation rationale above (away from
        # /clock's extreme publish rate) applies here too -- sharing THIS
        # node/executor (never the /clock-carrying one) is fine, since the
        # documented starvation problem was specifically /clock's publish
        # rate, not general multi-subscription contention between two
        # per-step-rate topics.
        self.latest_sensor_diagnostics: Optional[sd.SensorDiagnostics] = None
        self.create_subscription(Float32MultiArray, "/hunter_kinodynamic_rl/sensor_diagnostics",
                                  self._on_sensor_diagnostics, 10)

    def _on_risk_telemetry(self, msg: Float32MultiArray) -> None:
        try:
            self.latest_telemetry = rt.decode(list(msg.data))
        except ValueError as e:
            self.get_logger().warn(f"[risk_telemetry] decode failed: {e}")

    def _on_sensor_diagnostics(self, msg: Float32MultiArray) -> None:
        try:
            self.latest_sensor_diagnostics = sd.decode(list(msg.data))
        except ValueError as e:
            self.get_logger().warn(f"[sensor_diagnostics] decode failed: {e}")


class EnvironmentClient(Node):
    """rclpy service-client + risk-telemetry subscriber wrapper around a
    running environment_node.py."""

    def __init__(self, node_name: str = "hunter_kinodynamic_train_client",
                 env_node_name: str = "hunter_kinodynamic_environment",
                 service_discovery_timeout_sec: float = 60.0,
                 call_timeout_sec: float = 30.0,
                 telemetry_wait_timeout_sec: float = 1.0,
                 reset_marker_wait_timeout_sec: float = 5.0):
        super().__init__(node_name)
        self._call_timeout_sec = call_timeout_sec
        self._telemetry_wait_timeout_sec = telemetry_wait_timeout_sec
        self._reset_marker_wait_timeout_sec = reset_marker_wait_timeout_sec
        self._reset_client = self.create_client(Reset, "reset")
        self._step_client = self.create_client(Step, "step")
        self._dims_client = self.create_client(GetDimensions, "get_dimensions")
        self._seed_client = self.create_client(Seed, "seed")
        self._set_params_client = self.create_client(SetParameters, f"/{env_node_name}/set_parameters")
        self._get_params_client = self.create_client(GetParameters, f"/{env_node_name}/get_parameters")
        for client, name in ((self._reset_client, "reset"), (self._step_client, "step"),
                              (self._dims_client, "get_dimensions"), (self._seed_client, "seed")):
            deadline = time.monotonic() + service_discovery_timeout_sec
            while not client.wait_for_service(timeout_sec=2.0):
                self.get_logger().info(f"waiting for /{name} service...")
                if time.monotonic() > deadline:
                    raise EnvServiceError(
                        f"/{name} service did not come up within {service_discovery_timeout_sec:.0f}s "
                        f"-- is environment_node.py (and Gazebo) actually running?"
                    )

        # See _RiskTelemetryListener's docstring: MUST be a separate node on
        # its OWN dedicated executor, never a subscription on `self`
        # (which also carries /clock/odometry/joint_states -- /clock's
        # extreme publish rate starves any other subscription sharing its
        # node/executor).
        self._telemetry_listener = _RiskTelemetryListener(f"{node_name}_telemetry")
        self._telemetry_executor = SingleThreadedExecutor()
        self._telemetry_executor.add_node(self._telemetry_listener)

        self._step_id = 0
        # code review (risk-telemetry correctness bug, and its P2 follow-up):
        # reset_generation is NOT a client-local counter -- it is LEARNED,
        # race-free, directly from Reset.srv's own response every reset()
        # call (see reset()'s docstring/comment). A locally-incremented
        # "starts at 0" assumption was a real, confirmed-live bug
        # (permanently desyncs from the server's actual process-lifetime
        # counter the moment this client is anything other than the FIRST
        # one to ever talk to that environment_node instance since its own
        # process start -- e.g. a restarted/resumed trainer, or a second
        # EnvironmentClient in the same process such as evaluation_node.py's,
        # connecting to an already-running node). None until the first
        # reset() call.
        self._reset_generation: Optional[int] = None
        self.telemetry_timeouts = 0
        self.telemetry_matched_count = 0
        self.reset_marker_timeouts = 0
        # requirement 5: independent counters from the risk-telemetry ones
        # above -- the two channels can (and do, e.g. under
        # COMPUTATION_EXCEPTION on just one of them) diverge in health.
        self.sensor_diagnostics_timeouts = 0
        self.sensor_diagnostics_matched_count = 0

        # Independent of the risk channel -- real robot pose/velocity/
        # steering for evaluation metrics (section 11/P1-4: never
        # approximate these from the normalized action). Subscribes the
        # SAME /odometry, joint_states, and /clock topics environment_node.py
        # itself consumes.
        self.declare_parameter("odom_topic", "/odometry")
        self.declare_parameter("joint_states_topic", "/hunter_se/joint_states")
        self._latest_xy = None
        self._latest_yaw = 0.0
        self.episode_path_length_m = 0.0
        self._latest_v_mps = 0.0
        self._latest_yaw_rate = 0.0
        self._latest_center_steering_rad = 0.0
        self._latest_sim_time_sec: Optional[float] = None
        self._episode_start_sim_time_sec: Optional[float] = None
        self.create_subscription(Odometry, self.get_parameter("odom_topic").value, self._on_odom, 10)
        self.create_subscription(JointState, self.get_parameter("joint_states_topic").value,
                                  self._on_joint_states, 10)
        self.create_subscription(Clock, "/clock", self._on_clock, 10)

    def _on_odom(self, msg: Odometry) -> None:
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        if self._latest_xy is not None:
            self.episode_path_length_m += math.hypot(x - self._latest_xy[0], y - self._latest_xy[1])
        self._latest_xy = (x, y)
        q = msg.pose.pose.orientation
        self._latest_yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self._latest_v_mps = msg.twist.twist.linear.x
        self._latest_yaw_rate = msg.twist.twist.angular.z

    def _on_joint_states(self, msg: JointState) -> None:
        """Mirrors environment_node.py's own steering computation EXACTLY
        (Ackermann center steering = mean of the two front wheel angles) --
        NOT odometry angular.z (that's yaw rate, a different physical
        quantity -- section 14/P1-4's explicit correction)."""
        try:
            left = float(msg.position[msg.name.index("front_left_steering")])
            right = float(msg.position[msg.name.index("front_right_steering")])
        except (ValueError, IndexError, TypeError):
            return
        self._latest_center_steering_rad = 0.5 * (left + right)

    def _on_clock(self, msg: Clock) -> None:
        self._latest_sim_time_sec = msg.clock.sec + msg.clock.nanosec * 1e-9

    @property
    def latest_v_mps(self) -> float:
        return self._latest_v_mps

    @property
    def latest_yaw_rate_rad_s(self) -> float:
        return self._latest_yaw_rate

    @property
    def latest_center_steering_rad(self) -> float:
        return self._latest_center_steering_rad

    @property
    def latest_pose(self) -> Optional[tuple]:
        """(x, y, yaw) from the most recent /odometry message -- None if
        none has been received yet (section P1-11: real pose for step
        logging, never approximated)."""
        if self._latest_xy is None:
            return None
        return (self._latest_xy[0], self._latest_xy[1], self._latest_yaw)

    @property
    def latest_sim_time_sec(self) -> Optional[float]:
        """Raw ABSOLUTE /clock reading (not episode-relative like
        episode_elapsed_sim_time_sec) -- section P1-11's per-step
        actual_dt_sec is computed as a difference of this across two
        consecutive steps, which the episode-relative property can't
        safely provide across a reset boundary."""
        return self._latest_sim_time_sec

    @property
    def episode_elapsed_sim_time_sec(self) -> Optional[float]:
        """None if /clock was never received (e.g. use_sim_time off) --
        callers must handle that explicitly, never silently substitute a
        wall-clock or step-count approximation."""
        if self._episode_start_sim_time_sec is None or self._latest_sim_time_sec is None:
            return None
        return self._latest_sim_time_sec - self._episode_start_sim_time_sec

    def _call(self, client, request):
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=self._call_timeout_sec)
        if not future.done():
            raise EnvServiceError(
                f"{client.srv_name}: no response within {self._call_timeout_sec:.0f}s -- "
                "environment_node.py may have crashed, or Gazebo has stalled"
            )
        result = future.result()
        if result is None:
            raise EnvServiceError(f"{client.srv_name}: call failed (exception in the service, or node exited)")
        return result

    def get_dimensions(self):
        return self._call(self._dims_client, GetDimensions.Request())

    def seed(self, value: int) -> bool:
        req = Seed.Request()
        req.seed = int(value)
        return self._call(self._seed_client, req).success

    def _set_string_parameter(self, name: str, value: str) -> bool:
        """Shared by :meth:`set_scenario_override` and
        :meth:`set_evaluation_contract_override` -- both are "set one
        STRING parameter on the live environment_node, take effect at its
        NEXT /reset" mechanisms, differing only in which parameter name
        and what environment_node.py does with it."""
        if not self._set_params_client.wait_for_service(timeout_sec=self._call_timeout_sec):
            raise EnvServiceError("environment_node's set_parameters service is unavailable")
        req = SetParameters.Request()
        req.parameters = [Parameter(name, Parameter.Type.STRING, value).to_parameter_msg()]
        result = self._call(self._set_params_client, req)
        return all(r.successful for r in result.results)

    def set_scenario_override(self, path: str) -> bool:
        """Sets environment_node.py's ``scenario_override_path`` parameter --
        the NEXT ``/reset`` places this EXACT fixed scenario instead of
        procedurally generating one (section 6/11's exact-benchmark-replay
        requirement). Pass ``""`` to return to normal (procedural/seeded)
        operation."""
        return self._set_string_parameter("scenario_override_path", path)

    def set_explicit_seed_mode(self, mode: str) -> bool:
        if mode not in ("train", "validation", "test"):
            raise ValueError(f"explicit seed mode must be train|validation|test, got {mode!r}")
        return self._set_string_parameter("explicit_seed_mode", mode)

    def set_evaluation_contract_override(self, path: str) -> bool:
        """section item-1 (round 2): sets environment_node.py's
        ``evaluation_contract_override_path`` parameter -- the NEXT
        ``/reset`` applies the ``reward``/``scenario``/``runtime``/
        ``evaluation`` sections from the file at ``path``
        (``evaluation/contract_override.py``'s own YAML shape) to the live
        environment, in place of whatever ITS OWN launch-time profile set
        for those sections -- see that module's docstring for why this is
        necessary for fair cross-algorithm benchmark comparisons. Pass
        ``""`` to restore the environment's own launch-time contract
        sections. Returns ``False`` (never raises) if the live
        environment_node rejected the parameter set -- callers MUST check
        this and fail fast, exactly like :meth:`set_scenario_override`."""
        return self._set_string_parameter("evaluation_contract_override_path", path)

    def get_remote_parameter(self, name: str):
        """section P0-1: reads a LIVE parameter directly from the connected
        environment_node -- e.g. its ``profile`` parameter, so a client
        (``evaluation_node.py``/``benchmark_runner.py``) can verify the
        SERVER is actually running under the profile a checkpoint expects,
        rather than only checking dimensions (which cannot distinguish two
        profiles that happen to produce the same state_dim/action_dim but
        decode ``action_space.mode`` differently -- e.g. legacy_waypoint vs
        trajectory, both 3-D). Raises :class:`EnvServiceError` if the
        parameter doesn't exist on the server (rather than returning a
        silently-wrong default)."""
        if not self._get_params_client.wait_for_service(timeout_sec=self._call_timeout_sec):
            raise EnvServiceError("environment_node's get_parameters service is unavailable")
        req = GetParameters.Request()
        req.names = [name]
        result = self._call(self._get_params_client, req)
        if not result.values or result.values[0].type == 0:  # PARAMETER_NOT_SET
            raise EnvServiceError(f"environment_node has no parameter named {name!r}")
        return parameter_value_to_python(result.values[0])

    def _spin_telemetry_once(self, timeout_sec: float) -> None:
        self._telemetry_executor.spin_once(timeout_sec=timeout_sec)

    def _await_new_reset_marker(self, known_generation: int) -> int:
        """section P1-5 (shared-interface preservation): learns THIS
        reset's ``reset_generation`` purely from the risk-telemetry
        broadcast (``/hunter_kinodynamic_rl/risk_telemetry``, this
        package's OWN topic) -- ``drl_agent_interfaces/srv/Reset.srv`` is
        never touched (see ``telemetry_is_new_reset_marker``'s docstring
        for the full single-owner-client rationale and its documented,
        narrower-than-fully-concurrent guarantee).

        Unlike the diagnostic-only wait this replaced, a timeout HERE
        means reset_generation for this episode is genuinely UNKNOWN --
        continuing would either silently reuse a stale value (corrupting
        every subsequent step's telemetry match for the whole episode) or
        fabricate one. Raises :class:`EnvServiceError` instead."""
        deadline = time.monotonic() + self._reset_marker_wait_timeout_sec
        while time.monotonic() < deadline:
            t = self._telemetry_listener.latest_telemetry
            if telemetry_is_new_reset_marker(t, known_generation):
                return t.reset_generation
            self._spin_telemetry_once(0.02)
        self.reset_marker_timeouts += 1
        raise EnvServiceError(
            f"EnvironmentClient.reset(): no NEW reset-marker telemetry (generation > {known_generation}) "
            f"arrived on /hunter_kinodynamic_rl/risk_telemetry within {self._reset_marker_wait_timeout_sec:.1f}s "
            "-- environment_node.py may have crashed, the risk-telemetry subscription may be stalled, or "
            "(if this is NOT the only client calling /reset on this environment_node instance) a "
            "concurrent-client race this package does not support was hit. reset_generation for this "
            "episode is unknown; refusing to guess."
        )

    def reset(self) -> np.ndarray:
        known_generation = self._reset_generation or 0
        result = self._call(self._reset_client, Reset.Request())
        self._step_id = 0
        # section P1-5: learned from the risk-telemetry broadcast, not a
        # Reset.srv response field -- see _await_new_reset_marker and
        # telemetry_is_new_reset_marker's docstrings for the full
        # rationale (drl_agent_interfaces/srv/Reset.srv stays byte-identical
        # to drl_agent's own, in line with this project's "reuse shared
        # infrastructure unmodified" requirement).
        self._reset_generation = self._await_new_reset_marker(known_generation)
        self.episode_path_length_m = 0.0
        self._latest_xy = None
        self._episode_start_sim_time_sec = self._latest_sim_time_sec
        return np.asarray(result.state, dtype=np.float32)

    @property
    def latest_xy(self):
        return self._latest_xy

    def _await_matching_telemetry(self, expected_step_id: int, timeout_sec: Optional[float] = None) -> rt.RiskTelemetry:
        """Bounded poll (the ISOLATED telemetry executor -- see
        _RiskTelemetryListener's docstring for why this must never share a
        node/executor with /clock) until the subscription cache's step_id
        AND reset_generation BOTH match. step_id ALONE is not enough
        (section P0-5): it resets to 0 every episode on both ends, so a
        message left over from the PREVIOUS episode's OWN step_id=K could
        otherwise be wrongly accepted as the CURRENT episode's step_id=K
        message if it's still sitting in the subscription cache when this
        episode reaches the same step number -- reset_generation (never
        reused across episodes, and now LEARNED from the server via
        _await_reset_marker rather than locally assumed) rules that out.
        Returns an explicit ``invalid`` telemetry (never a stale one) on
        timeout, incrementing ``telemetry_timeouts`` for the trainer to
        log; ``timeout_sec`` defaults to the profile-configured
        ``runtime.risk_telemetry_wait_timeout_sec`` (constructor arg)."""
        budget = timeout_sec if timeout_sec is not None else self._telemetry_wait_timeout_sec
        deadline = time.monotonic() + budget
        while time.monotonic() < deadline:
            t = self._telemetry_listener.latest_telemetry
            if telemetry_matches_step(t, expected_step_id, self._reset_generation):
                self.telemetry_matched_count += 1
                return t
            self._spin_telemetry_once(0.02)
        self.telemetry_timeouts += 1
        return rt.invalid(expected_step_id, reset_generation=self._reset_generation or 0,
                           reason=rt.InvalidReason.POLL_TIMEOUT)

    def _await_matching_sensor_diagnostics(self, expected_step_id: int,
                                            timeout_sec: Optional[float] = None) -> sd.SensorDiagnostics:
        """requirement 5: the ``sensor_diagnostics`` analogue of
        ``_await_matching_telemetry`` -- identical bounded-poll/staleness-
        rejection shape, on the separate diagnostics cache. Never pairs a
        stale/mismatched sample to the wrong step: returns an explicit
        ``invalid(..., reason=POLL_TIMEOUT)`` (never a stale one) if nothing
        matching arrives within budget, so a caller (e.g. RunLogger.log_step)
        can log ``null`` GT/noisy values with the reason instead of silently
        misattributing a leftover sample."""
        budget = timeout_sec if timeout_sec is not None else self._telemetry_wait_timeout_sec
        deadline = time.monotonic() + budget
        while time.monotonic() < deadline:
            d = self._telemetry_listener.latest_sensor_diagnostics
            if sensor_diagnostics_matches_step(d, expected_step_id, self._reset_generation):
                self.sensor_diagnostics_matched_count += 1
                return d
            self._spin_telemetry_once(0.02)
        self.sensor_diagnostics_timeouts += 1
        return sd.invalid(expected_step_id, reset_generation=self._reset_generation or 0,
                           reason=sd.DiagnosticsInvalidReason.POLL_TIMEOUT)

    @property
    def telemetry_valid_ratio(self) -> float:
        """Fraction of ``step()`` calls whose telemetry actually matched
        (regardless of the risk framework being enabled -- this measures
        CHANNEL health, i.e. is step_id/reset_generation sync working at
        all, not risk-label validity specifically)."""
        total = self.telemetry_matched_count + self.telemetry_timeouts
        if total == 0:
            return 1.0  # no steps taken yet -- vacuously healthy, never a false alarm
        return self.telemetry_matched_count / total

    @property
    def sensor_diagnostics_valid_ratio(self) -> float:
        """requirement 5: the sensor_diagnostics analogue of
        ``telemetry_valid_ratio`` -- channel health (sync working at all),
        independent of risk_telemetry's own ratio."""
        total = self.sensor_diagnostics_matched_count + self.sensor_diagnostics_timeouts
        if total == 0:
            return 1.0
        return self.sensor_diagnostics_matched_count / total

    def step(self, action):
        req = Step.Request()
        req.action = [float(a) for a in action]
        result = self._call(self._step_client, req)
        self._step_id += 1
        telemetry = self._await_matching_telemetry(self._step_id)
        diagnostics = self._await_matching_sensor_diagnostics(self._step_id)
        return (np.asarray(result.state, dtype=np.float32), float(result.reward),
                bool(result.done), bool(result.target), bool(result.collision),
                float(result.min_obstacle_dist_m), telemetry, diagnostics)

    def destroy_node(self) -> None:
        self._telemetry_executor.remove_node(self._telemetry_listener)
        self._telemetry_listener.destroy_node()
        super().destroy_node()


def _telemetry_to_risk_transition(telemetry: rt.RiskTelemetry, action_cfg, robot, max_candidates: int) -> RiskTransition:
    if not telemetry.valid:
        return RiskTransition(risk_target=None)
    candidate_kappa, candidate_v_ref, candidate_horizon = [], [], []
    candidate_risk, candidate_goal_progress = [], []
    for c in telemetry.candidates[:max_candidates]:
        candidate_kappa.append(c.kappa)
        candidate_v_ref.append(c.v_ref)
        candidate_horizon.append(c.horizon_m)
        candidate_risk.append(c.risk_score)
        candidate_goal_progress.append(c.goal_progress_m)
    return RiskTransition(
        risk_target=telemetry.risk_target, min_clearance_m=telemetry.min_clearance_m,
        ttc_sec=telemetry.ttc_sec, collision_within_horizon=telemetry.collision_within_horizon,
        stopping_margin_m=telemetry.stopping_margin_m, unrecoverable=telemetry.unrecoverable,
        steering_saturation=telemetry.steering_saturation,
        goal_progress_m=telemetry.goal_progress_m,
        safer_alternative_margin=telemetry.safer_alternative_margin,
        candidate_kappa=candidate_kappa, candidate_v_ref=candidate_v_ref,
        candidate_horizon=candidate_horizon, candidate_risk=candidate_risk,
        candidate_goal_progress=candidate_goal_progress,
        actor_candidate_index=telemetry.actor_candidate_index,
    )


def _candidates_to_normalized_actions(batch, action_cfg, robot, max_candidates: int):
    """(B, K) physical candidate_{kappa,v_ref,horizon} -> (B, K, 3) normalized
    actions, for the risk critic's candidate-augmented supervision (which
    consumes the SAME normalized-action space the actor outputs)."""
    import torch
    b, k = batch["candidate_kappa"].shape
    kappa_max = robot.max_curvature * action_cfg.kappa_scale
    v_max = action_cfg.v_max_mps if action_cfg.v_max_mps is not None else robot.max_forward_speed_mps

    def inv_lerp(value, lo, hi):
        if hi <= lo:
            return torch.zeros_like(value)
        return torch.clamp(2.0 * (value - lo) / (hi - lo) - 1.0, -1.0, 1.0)

    norm_kappa = inv_lerp(batch["candidate_kappa"], -kappa_max, kappa_max)
    norm_v = inv_lerp(batch["candidate_v_ref"], action_cfg.v_min_mps, v_max)
    norm_l = inv_lerp(batch["candidate_horizon"], action_cfg.horizon_length_min_m, action_cfg.horizon_length_max_m)
    return torch.stack([norm_kappa, norm_v, norm_l], dim=-1)


class TrainerBase:
    """Subclasses implement :meth:`build_agent` and :meth:`agent_train_step`."""

    risk_aware: bool = False

    def __init__(self, profile: Profile, run_root: str, resume: bool = False,
                 resume_run_dir: Optional[str] = None, resume_checkpoint_tag: str = "latest"):
        self.profile = profile
        seed_all(profile.training.seed)
        enable_torch_determinism(warn_only=True)

        self.run_dir = resume_run_dir if (resume and resume_run_dir) else run_directory(profile, run_root)
        self.logger = RunLogger(self.run_dir, profile, log_full_state=profile.training.log_full_state)

        rclpy.init(args=None)
        self.env = EnvironmentClient(
            telemetry_wait_timeout_sec=profile.runtime.risk_telemetry_wait_timeout_sec,
            reset_marker_wait_timeout_sec=profile.runtime.risk_telemetry_reset_marker_timeout_sec,
        )
        dims = self.env.get_dimensions()
        self.state_dim = dims.state_dim
        self.action_dim = dims.action_dim
        self.max_action = dims.max_action

        self.max_candidates = profile.counterfactual.num_candidates if profile.counterfactual.enabled else 0
        self.replay_buffer = ReplayBuffer(
            self.state_dim, self.action_dim, capacity=profile.hyperparameters.buffer_size,
            seed=profile.training.seed, max_candidates=max(1, self.max_candidates),
        )
        self.agent = self.build_agent()

        self.seed_scheduler = SeedScheduler(profile.training.seed, profile.scenario, mode="train")
        # section P1-13: periodic validation draws from its OWN pool,
        # never train/test (SeedScheduler's own pool-isolation guarantee)
        # -- a training run's "held-out" performance measure would be
        # meaningless if it could ever reuse a seed the replay buffer was
        # also trained on.
        self.validation_seed_scheduler = SeedScheduler(profile.training.seed, profile.scenario, mode="validation")
        self.global_step = 0
        self.episode_index = 0
        self.episode_reward = 0.0
        self.episode_len = 0
        self.best_eval_metric = -float("inf")
        # step at which the last periodic checkpoint was actually taken
        # (section P0-4) -- see _checkpoint_due()'s docstring for why this
        # replaces a bare `step % eval_freq == 0` cadence check.
        self._last_checkpoint_step = 0
        # section P1-11: previous /clock reading, for computing each
        # step's REAL elapsed sim-time (actual_dt_sec in log_step) as a
        # difference across two consecutive steps.
        self._prev_step_sim_time_sec: Optional[float] = None

        if resume and resume_run_dir:
            self._resume_from(resume_run_dir, resume_checkpoint_tag)

    def build_agent(self):
        raise NotImplementedError

    def agent_train_step(self, batch):
        raise NotImplementedError

    def _random_action(self) -> np.ndarray:
        return np.random.uniform(-1.0, 1.0, size=self.action_dim).astype(np.float32)

    def _new_episode(self) -> np.ndarray:
        seed = self.seed_scheduler.next_seed()
        set_mode = getattr(self.env, "set_explicit_seed_mode", None)
        if callable(set_mode) and not set_mode("train"):
            raise RuntimeError("environment rejected explicit_seed_mode='train'")
        if self.env.seed(seed) is False:
            raise RuntimeError(f"environment rejected training seed {seed}")
        state = self.env.reset()
        self.episode_reward = 0.0
        self.episode_len = 0
        self.episode_index += 1
        # section P1-8: sample_draw() is a PURE function of (seed, config) --
        # environment_node.py independently computes the exact same draw
        # server-side from this same seed (see its `_on_reset`'s
        # `elif self.profile.domain_randomization.enabled: draw =
        # sample_draw(seed, ...)`), so recomputing it here needs no new IPC
        # channel (and touches no read-only drl_agent_interfaces .srv) while
        # still logging the real per-episode applied values.
        draw = sample_draw(seed, self.profile.domain_randomization) if self.profile.domain_randomization.enabled \
            else None
        self.logger.log_episode_start(self.episode_index, seed, self.global_step, domain_rand_draw=draw)
        return state, seed

    def _run_validation_episodes(self) -> dict:
        """section P1-13: run ``profile.training.eval_episodes`` episodes
        with the agent in DETERMINISTIC mode, seeded from the VALIDATION
        pool -- called ONLY from ``run()``'s ``if done:`` block, at the
        SAME episode-boundary point checkpointing already happens (never
        mid-episode, for the exact same reason ``checkpoint_due`` gates
        checkpoints there -- see that function's docstring). Validation
        transitions are NEVER stored in the replay buffer (this is
        evaluation, not training data) and ``global_step``/``episode_index``
        are left untouched -- only ``self.env``'s live Gazebo state is
        borrowed for the duration, between two training episodes."""
        n = max(1, self.profile.training.eval_episodes)
        successes = 0
        collisions = 0
        total_reward = 0.0
        set_mode = getattr(self.env, "set_explicit_seed_mode", None)
        if callable(set_mode) and not set_mode("validation"):
            raise RuntimeError("environment rejected explicit_seed_mode='validation'")
        try:
            for _ in range(n):
                val_seed = self.validation_seed_scheduler.next_seed()
                if self.env.seed(val_seed) is False:
                    raise RuntimeError(f"environment rejected validation seed {val_seed}")
                state = self.env.reset()
                episode_reward = 0.0
                for _ in range(self.profile.training.episode_length_steps):
                    action = self.agent.select_action(state, deterministic=True)
                    state, reward, done, target, collision, _min_dist, _telemetry, _diag = self.env.step(action)
                    episode_reward += reward
                    if done:
                        successes += int(target)
                        collisions += int(collision)
                        break
                total_reward += episode_reward
        finally:
            if callable(set_mode) and not set_mode("train"):
                raise RuntimeError("environment rejected restoration of explicit_seed_mode='train'")
        return {
            "num_episodes": n,
            "success_rate": successes / n,
            "collision_rate": collisions / n,
            "mean_reward": total_reward / n,
        }

    def _store_transition(self, state, action, next_state, reward, done, telemetry: rt.RiskTelemetry) -> None:
        risk = _telemetry_to_risk_transition(
            telemetry, self.profile.action_space, self.agent_robot_config(), self.max_candidates,
        )
        self.replay_buffer.add(state, action, next_state, reward, done, risk=risk)

    def agent_robot_config(self):
        return self.profile.robot

    def _sample_batch_for_agent(self):
        batch = self.replay_buffer.sample_torch(self.profile.hyperparameters.batch_size, device=self.agent.device)
        if self.max_candidates > 0:
            batch["candidate_actions_normalized"] = _candidates_to_normalized_actions(
                batch, self.profile.action_space, self.agent_robot_config(), self.max_candidates,
            )
        return batch

    def _check_risk_telemetry_health(self, step: int, *, final: bool) -> None:
        """code review (risk-telemetry correctness bug): a risk-aware run
        must never finish looking successful while the risk-telemetry side
        channel was systematically broken -- e.g. every /step's telemetry
        timing out (a REAL failure mode, confirmed live: an entire smoke-
        test run finished with risk_valid=False on 100% of steps and
        risk_supervised_updates==0, silently, until this check existed).
        ONLY enforced for risk-aware trainers (``self.risk_aware``, True on
        KinodynamicTQCTrainer) with ``risk.enabled`` -- vanilla TQC/SAC
        never consume risk labels, so a low valid_ratio there is
        diagnostic-only, never a training-correctness problem.

        Always LOGS (checkpoint-meta-verifiable via ``_save_checkpoint``'s
        ``telemetry_valid_ratio``/``telemetry_timeouts`` fields, and via
        ``RunLogger.log_telemetry_health``'s steps.jsonl record) regardless
        of pass/fail. Raises ``RuntimeError`` (never just logs) on failure
        -- ``train_kinodynamic_tqc.py``'s existing
        ``finally: trainer.shutdown(failed=...)`` pattern already marks
        ``metadata.json`` ``status="failed"`` and propagates a non-zero
        process exit whenever ``run()`` raises, so this integrates with
        that existing failure-reporting path rather than inventing a new
        one."""
        if not (self.risk_aware and self.profile.risk.enabled):
            return
        valid_ratio = self.replay_buffer.valid_ratio()
        risk_supervised_updates = getattr(self.agent, "risk_supervised_updates", None)
        self.logger.log_telemetry_health(
            step, valid_ratio=valid_ratio, telemetry_matched=self.env.telemetry_matched_count,
            telemetry_timeouts=self.env.telemetry_timeouts, reset_marker_timeouts=self.env.reset_marker_timeouts,
            risk_supervised_updates=risk_supervised_updates,
        )
        if not final and step < self.profile.risk.telemetry_valid_ratio_grace_steps:
            return  # too early to judge -- avoid a noisy false alarm on a near-empty buffer
        if valid_ratio < self.profile.risk.min_telemetry_valid_ratio:
            raise RuntimeError(
                f"risk-telemetry health check FAILED at step {step}: replay buffer valid_ratio="
                f"{valid_ratio:.3f} < risk.min_telemetry_valid_ratio="
                f"{self.profile.risk.min_telemetry_valid_ratio:.3f} "
                f"(telemetry_matched={self.env.telemetry_matched_count}, "
                f"telemetry_timeouts={self.env.telemetry_timeouts}, "
                f"reset_marker_timeouts={self.env.reset_marker_timeouts}) -- this profile has "
                "features.risk_critic/risk.enabled=true, so the risk critic is training on a "
                "systematically-broken (or near-empty) label stream; refusing to let this run finish "
                "looking successful."
            )
        if final and risk_supervised_updates == 0:
            raise RuntimeError(
                f"risk-telemetry health check FAILED at final step {step}: this profile has "
                "features.risk_critic/risk.enabled=true but risk_supervised_updates==0 -- the risk "
                "critic never received a single supervised update for the ENTIRE run (every training "
                "batch had fewer than risk.min_valid_labels_per_batch valid labels). Refusing to let "
                "this run finish looking successful."
            )

    def run(self) -> dict:
        # ``max_timesteps`` is the TOTAL target step count, not "how many
        # MORE steps to run" -- a resumed run continues from
        # ``self.global_step + 1`` (restored by ``_resume_from``, 0 on a
        # fresh run) instead of restarting the loop counter at 1 (section
        # P1-1: "resume 후 global step이 되돌아가지 않음"). If the restored
        # step already meets or exceeds the target, this is a normal,
        # successful no-op completion, not an error.
        start_step = self.global_step + 1
        if start_step > self.profile.training.max_timesteps:
            self.logger.log_resume_noop(self.global_step, self.profile.training.max_timesteps)
            return {"final_step": self.global_step, "elapsed_sec": 0.0, "last_metrics": {}, "run_dir": self.run_dir}

        state, seed = self._new_episode()
        last_metrics = {}
        t0 = time.monotonic()

        for step in range(start_step, self.profile.training.max_timesteps + 1):
            self.global_step = step
            if step <= self.profile.training.timesteps_before_training:
                action = self._random_action()
            else:
                action = self.agent.select_action(state, deterministic=False)

            next_state, reward, done, target, collision, min_dist, telemetry, diagnostics = self.env.step(action)
            self.episode_reward += reward
            self.episode_len += 1

            # section P0-9: environment_node.py already truncates the
            # episode (done=True) on a stale-sensor step -- the trainer's
            # OWN responsibility is to never let that one uncertain
            # transition into the replay buffer as if it were an ordinary
            # continuing (or even ordinary terminal) sample: next_state may
            # not actually reflect the post-action world. Still logged
            # (log_step, telemetry.sensor_stale) for visibility, and the
            # episode-boundary handling below still runs normally (done is
            # already True by construction whenever this fires).
            if not telemetry.sensor_stale:
                self._store_transition(state, action, next_state, reward, done, telemetry)
            # section P1-11: real pose/decoded-trajectory/actual-dt for the
            # step log -- all best-effort (None when the underlying data
            # isn't available yet, e.g. before the first /odometry or
            # /clock message), never approximated from the normalized
            # action alone.
            trajectory_command = decode_action(action, self.profile.action_space, self.agent_robot_config())
            predict_risk = getattr(self.agent, "predict_risk", None)
            predicted_risk = predict_risk(state, action) if callable(predict_risk) else None
            now_sim_time = self.env.latest_sim_time_sec
            actual_dt_sec = (now_sim_time - self._prev_step_sim_time_sec
                              if now_sim_time is not None and self._prev_step_sim_time_sec is not None else None)
            self._prev_step_sim_time_sec = now_sim_time
            self.logger.log_step(step, self.episode_index, action, reward, telemetry, collision, target,
                                  pose=self.env.latest_pose, trajectory_command=trajectory_command,
                                  actual_dt_sec=actual_dt_sec, state=state, next_state=next_state,
                                  measured_velocity_mps=getattr(self.env, "latest_v_mps", None),
                                  measured_yaw_rate_rad_s=getattr(self.env, "latest_yaw_rate_rad_s", None),
                                  measured_steering_rad=getattr(self.env, "latest_center_steering_rad", None),
                                  predicted_risk=predicted_risk, sensor_diagnostics=diagnostics)
            state = next_state

            # Training/sampling MUST run here -- BEFORE the episode-boundary
            # checkpoint block below, not after (found via replay-buffer
            # sample-sequence determinism testing, section P0-1): a
            # checkpoint saved at step N must capture the replay buffer's
            # sampling RNG state AFTER step N's own sample() draw, not
            # before it. With the training block running AFTER the
            # checkpoint save, a checkpoint taken exactly on a step that
            # ALSO triggers training would persist the RNG state as it
            # stood BEFORE that step's draw -- a resumed run's very next
            # sample() call would then re-draw from that pre-step-N
            # position, one draw "behind" where an uninterrupted run's
            # replay-buffer RNG genuinely is immediately after step N,
            # causing every subsequent sampled batch (and therefore every
            # subsequent gradient step) to diverge from the uninterrupted
            # run despite the action sequence itself still matching
            # (action selection never touches the replay buffer's RNG).
            if (step > self.profile.training.timesteps_before_training
                    and len(self.replay_buffer) >= self.profile.hyperparameters.batch_size):
                batch = self._sample_batch_for_agent()
                last_metrics = self.agent_train_step(batch)
                self.logger.log_train_metrics(step, last_metrics, self.replay_buffer.valid_ratio())

            if step % self.profile.risk.telemetry_health_check_interval_steps == 0:
                self._check_risk_telemetry_health(step, final=False)

            if done:
                self.logger.log_episode_end(self.episode_index, seed, step, self.episode_reward, self.episode_len,
                                             target, collision,
                                             timeout=not (target or collision or telemetry.sensor_stale),
                                             unrecoverable=telemetry.unrecoverable,
                                             sensor_stale=telemetry.sensor_stale)
                # Checkpoint gate MUST run here -- after log_episode_end,
                # BEFORE _new_episode() below draws the next seed and
                # resets the env -- never on a bare `step % eval_freq == 0`
                # (section P0-4: see checkpoint_policy.checkpoint_due()'s
                # docstring for why a mid-episode save breaks deterministic
                # resume).
                if checkpoint_due(step, self._last_checkpoint_step, self.profile.training.eval_freq):
                    # Update BEFORE saving, not after: _save_checkpoint()
                    # writes self._last_checkpoint_step into THIS
                    # checkpoint's own meta (below) -- it must read back
                    # as "this checkpoint's step", not "the PREVIOUS
                    # checkpoint's step" (confirmed live: without this
                    # ordering, a resumed run's next periodic save fires
                    # immediately, using a stale cadence reference).
                    self._last_checkpoint_step = step
                    # section P1-13: validate BEFORE saving "periodic" so
                    # its own meta already reflects any updated
                    # best_eval_metric, then save a SEPARATE "best" file
                    # only when this validation actually improved on it --
                    # documented criterion: success_rate - collision_rate
                    # (higher is better; a policy that reaches the goal
                    # more often AND collides less often is unambiguously
                    # better, ties/partial trade-offs resolve via this one
                    # scalar rather than a multi-objective comparison).
                    validation_result = self._run_validation_episodes()
                    eval_metric = validation_result["success_rate"] - validation_result["collision_rate"]
                    is_new_best = eval_metric > self.best_eval_metric
                    if is_new_best:
                        self.best_eval_metric = eval_metric
                    self.logger.log_validation(step, validation_result, eval_metric, is_new_best)
                    self._save_checkpoint(tag="periodic")
                    if is_new_best:
                        self._save_checkpoint(tag="best")
                state, seed = self._new_episode()

        # A "final" save happens unconditionally when max_timesteps is
        # reached, even if that lands mid-episode (training is simply OVER
        # at this point, there is no "next episode" for a resume to
        # deterministically continue into the way there is for the
        # periodic in-loop saves above). Resuming from "final" still
        # starts a fresh episode with the next scheduled seed via run()'s
        # own unconditional pre-loop _new_episode() call -- a valid
        # continuation, just not a bit-identical replay of whatever
        # remained of the interrupted episode (which no uninterrupted run
        # could be compared against anyway, since max_timesteps is where
        # IT also stops).
        self._save_checkpoint(tag="final")
        self._check_risk_telemetry_health(self.profile.training.max_timesteps, final=True)
        return {"final_step": self.profile.training.max_timesteps, "elapsed_sec": time.monotonic() - t0,
                "last_metrics": last_metrics, "run_dir": self.run_dir}

    # ------------------------------------------------------------ checkpoint
    def _save_checkpoint(self, tag: str = "periodic") -> str:
        directory = os.path.join(self.run_dir, "checkpoints")
        filename = "latest" if tag == "periodic" else tag
        import dataclasses
        import torch

        # section item-3 (checkpoint generation/atomicity): model + manifest
        # + replay buffer are published as ONE atomic unit by
        # ckpt_manager.save_generation -- see that function's and this
        # module's own docstring for the staging-directory +
        # atomic-symlink-rename mechanism and the generation/hash
        # cross-verification load_generation performs. `generation`/
        # `pt_sha256`/`pt_size_bytes`/`replay_sha256`/`replay_size_bytes`
        # are computed and embedded by save_generation itself (never by
        # this method), so they are NOT set in `meta` below.
        meta = {
            "profile_name": self.profile.name,
            # The FULL resolved config, not just profile_name -- a run-level
            # configs/profile_snapshot.json also exists (RunLogger.__init__)
            # but is OVERWRITTEN on every process start (including a later
            # resume), so it cannot preserve what config was actually in
            # effect for an EARLIER checkpoint if the same-named profile's
            # YAML is edited between runs. Embedding it directly here makes
            # every checkpoint self-describing regardless of what the
            # profile file on disk looks like later.
            "resolved_config": dataclasses.asdict(self.profile),
            # defect 2 (hierarchical live executor): the SUBGOAL-DISTRIBUTION
            # contract this checkpoint was actually trained under
            # (scenario.goal_sampling_mode/goal_distance_range_m/
            # goal_direction_sectors_deg/goal_infeasible_fraction/
            # feasibility_check), separate from architecture_fingerprint --
            # two checkpoints can share an architecture fingerprint while
            # having trained under completely different subgoal
            # distributions. See evaluation.fingerprint's own docstring.
            "local_training_contract_fingerprint": local_training_contract_fingerprint(self.profile),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "training_steps": self.agent.training_steps,
            # Only present on rl.algorithms.kinodynamic_tqc.Agent -- None on
            # the vanilla agent (no risk critic, no warmup counter to save).
            "risk_supervised_updates": getattr(self.agent, "risk_supervised_updates", None),
            # code review (risk-telemetry correctness bug): verifiable proof
            # in the checkpoint manifest itself that the risk-telemetry side
            # channel was actually healthy for this run, not just that
            # training "completed" -- see TrainerBase._check_risk_telemetry_health.
            # getattr-defaulted: test doubles standing in for the real
            # EnvironmentClient (e.g. test_checkpoint_boundary_wiring.py's
            # _FakeEnv) don't need to satisfy this diagnostic-only surface.
            "telemetry_valid_ratio": self.replay_buffer.valid_ratio(),
            "telemetry_matched_count": getattr(self.env, "telemetry_matched_count", None),
            "telemetry_timeouts": getattr(self.env, "telemetry_timeouts", None),
            "reset_marker_timeouts": getattr(self.env, "reset_marker_timeouts", None),
            "global_step": self.global_step,
            "episode_index": self.episode_index,
            "last_checkpoint_step": self._last_checkpoint_step,
            "seed_scheduler": self.seed_scheduler.state_dict(),
            "validation_seed_scheduler": self.validation_seed_scheduler.state_dict(),
            "ent_coef_state": self.agent.ent_coef_state(),
            "python_random_state": random.getstate(),
            "numpy_random_state": [s if not isinstance(s, np.ndarray) else s.tolist()
                                    for s in np.random.get_state()],
            "torch_rng_state": torch.get_rng_state().tolist(),
            "torch_cuda_rng_state": (torch.cuda.get_rng_state().tolist()
                                      if torch.cuda.is_available() else None),
            # section P1-13: the documented best-checkpoint criterion is
            # success_rate - collision_rate from _run_validation_episodes()
            # (held-out VALIDATION-pool episodes, never train/test) --
            # updated (and a "best" checkpoint written alongside "latest")
            # only when a periodic validation pass actually improves on it.
            # See run()'s checkpoint_due block and
            # RunLogger.log_validation for the full wiring.
            "best_eval_metric": self.best_eval_metric,
            # No online observation/reward normalizer exists anywhere in
            # this system (LiDAR ranges are already bounded [0,
            # lidar_max_range_m], robot-state fields are already physically
            # scaled -- matches drl_agent's own un-normalized vanilla TQC
            # reference, section: "Vanilla TQC 수학을... 변경하지 않는다").
            # Explicit null (not an omitted key) so this is a documented
            # "nothing to restore" rather than a silent gap.
            "normalization_state": None,
            "schema_version": 2,
        }
        ckpt_manager.save_generation(directory, filename, self.agent.checkpoint_components(), meta,
                                      self.replay_buffer)
        return directory

    def _resume_from(self, run_dir: str, checkpoint_tag: str = "latest") -> None:
        import torch
        directory = os.path.join(run_dir, "checkpoints")
        # code review (checkpoint load/prune TOCTOU): load_generation_lease
        # keeps this generation's shared advisory lock held for this ENTIRE
        # `with` block -- including the ReplayBuffer.load() call at the
        # bottom, not just the model.pt/manifest.json/replay.npz-header read
        # load_generation_lease itself performs. Using load_generation (or
        # doing anything with result["replay_path"] after this block would
        # exit) reopens exactly the TOCTOU window this closes: a concurrent
        # prune_orphan_generations could otherwise delete this generation
        # (including replay.npz) between the lock's old release point and
        # this method's own later read of it. See
        # ckpt_manager.load_generation_lease's docstring.
        with ckpt_manager.load_generation_lease(
            directory, checkpoint_tag, self.agent.checkpoint_components(),
            map_location=str(self.agent.device),
        ) as result:
            manifest = result["manifest"]

            # Fail-fast compatibility checks (section P1-1: "architecture/action
            # dimension/profile compatibility를 fail-fast 검증한다") -- a silent
            # dimension mismatch would corrupt training with garbage gradients
            # instead of an immediate, legible error.
            required_fields = (
                "state_dim", "action_dim", "profile_name", "resolved_config",
                "local_training_contract_fingerprint",
            )
            missing = [name for name in required_fields if name not in manifest or manifest[name] is None]
            if missing:
                raise ValueError(
                    f"resume checkpoint is missing required compatibility field(s): {missing}"
                )

            ckpt_state_dim = manifest["state_dim"]
            ckpt_action_dim = manifest["action_dim"]
            if ckpt_state_dim != self.state_dim:
                raise ValueError(
                    f"resume dimension mismatch: checkpoint state_dim={ckpt_state_dim} != "
                    f"live environment state_dim={self.state_dim} (observation config changed?)"
                )
            if ckpt_action_dim != self.action_dim:
                raise ValueError(
                    f"resume dimension mismatch: checkpoint action_dim={ckpt_action_dim} != "
                    f"live environment action_dim={self.action_dim}"
                )
            ckpt_profile = manifest["profile_name"]
            if ckpt_profile != self.profile.name:
                raise ValueError(
                    f"resume profile mismatch: checkpoint was saved under profile {ckpt_profile!r}, "
                    f"resuming with {self.profile.name!r} -- use the SAME profile to resume, or accept "
                    f"the risk explicitly by editing this check if a deliberate architecture-preserving "
                    f"profile swap is intended."
                )
            recorded_config = manifest["resolved_config"]
            if architecture_fingerprint_from_resolved_config(recorded_config) != architecture_fingerprint(
                self.profile
            ):
                raise ValueError(
                    "resume architecture mismatch: checkpoint resolved_config does not match the current "
                    "Local policy architecture/action/observation contract"
                )
            recorded_training_contract = manifest["local_training_contract_fingerprint"]
            try:
                derived_training_contract = local_training_contract_fingerprint_from_resolved_config(
                    recorded_config
                )
            except KeyError as exc:
                raise ValueError(f"resume checkpoint has an incomplete Local training contract: {exc}") from exc
            current_training_contract = local_training_contract_fingerprint(self.profile)
            if recorded_training_contract != derived_training_contract:
                raise ValueError(
                    "resume Local training-contract metadata is internally inconsistent: "
                    f"recorded={recorded_training_contract}, derived={derived_training_contract}"
                )
            if recorded_training_contract != current_training_contract:
                raise ValueError(
                    "resume Local training distribution mismatch: checkpoint="
                    f"{recorded_training_contract}, current={current_training_contract}; start a fresh run "
                    "instead of relabeling old weights after changing the subgoal distribution"
                )

            self.agent.load_ent_coef_state(manifest.get("ent_coef_state"))
            self.agent.training_steps = manifest.get("training_steps", 0)
            if manifest.get("risk_supervised_updates") is not None and hasattr(self.agent, "risk_supervised_updates"):
                self.agent.risk_supervised_updates = manifest["risk_supervised_updates"]
            self.global_step = manifest.get("global_step", 0)
            self.episode_index = manifest.get("episode_index", 0)
            # Fallback for pre-P0-4 manifests missing this key: this checkpoint
            # WAS just saved at global_step (by construction, every save sets
            # it), so treating it as the last checkpoint step is exactly
            # correct, not just a safe default.
            self._last_checkpoint_step = manifest.get("last_checkpoint_step", self.global_step)
            self.best_eval_metric = manifest.get("best_eval_metric", -float("inf"))
            if "seed_scheduler" in manifest:
                self.seed_scheduler = SeedScheduler.from_state_dict(manifest["seed_scheduler"], self.profile.scenario)
            if "validation_seed_scheduler" in manifest:
                self.validation_seed_scheduler = SeedScheduler.from_state_dict(
                    manifest["validation_seed_scheduler"], self.profile.scenario)
            if manifest.get("python_random_state"):
                random.setstate(tuple(
                    tuple(x) if isinstance(x, list) else x for x in manifest["python_random_state"]
                ))
            if manifest.get("numpy_random_state"):
                state = manifest["numpy_random_state"]
                state[1] = np.array(state[1], dtype=np.uint32)
                np.random.set_state(tuple(state))
            if manifest.get("torch_rng_state"):
                torch.set_rng_state(torch.tensor(manifest["torch_rng_state"], dtype=torch.uint8))
            if manifest.get("torch_cuda_rng_state") and torch.cuda.is_available():
                torch.cuda.set_rng_state(torch.tensor(manifest["torch_cuda_rng_state"], dtype=torch.uint8))
            # section item-3: load_generation_lease() has ALREADY verified the
            # replay file exists and is generation/hash-consistent with
            # model.pt/manifest.json (see its own docstring) -- just load it
            # from the path it resolved, no re-verification needed here. This
            # read happens WHILE the lease's shared lock is still held (see
            # this method's own docstring above) -- the whole point of using
            # the lease instead of load_generation.
            self.replay_buffer = ReplayBuffer.load(result["replay_path"], seed=self.profile.training.seed)
        self.logger.log_resume(self.global_step, self.episode_index, checkpoint_tag, directory)

    def shutdown(self, failed: bool = False) -> None:
        self.logger.close(failed=failed)
        self.env.destroy_node()
        # A SIGINT during training can already have triggered rclpy's own
        # shutdown before this runs -- guard against calling it twice
        # ("rcl_shutdown already called").
        if rclpy.ok():
            rclpy.shutdown()
