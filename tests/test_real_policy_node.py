"""section P2-11: real_policy_node.py's safety/operational interface
(dry_run, replay_mode's dry_run pairing requirement, the software E-stop
kill-switch). RealPolicyNode.__init__ needs a live rclpy context, a real
checkpoint, and a loaded profile to fully construct -- these tests instead
build a BARE instance (RealPolicyNode.__new__, skipping Node.__init__ and
the whole ROS-wiring body) with just the attributes each method under test
actually reads, then call the bound methods directly. This is the same
"duck-typed harness" approach tests/test_obstacle_spawner.py uses for
similarly ROS-heavy production code.
"""

import dataclasses
import threading
import time

import numpy as np
import pytest

pytest.importorskip("rclpy")  # real_policy_node.py imports rclpy at module scope
pytest.importorskip("torch")  # ...and the RiskAgent/VanillaAgent kinodynamic_tqc/tqc modules at module scope

from hunter_kinodynamic_rl.config.loader import load_profile  # noqa: E402
from hunter_kinodynamic_rl.env.safety.action_guard import STOP_COMMAND, SafetyLimits  # noqa: E402
from hunter_kinodynamic_rl.nodes.real_policy_node import (  # noqa: E402
    CheckpointProfileMismatchError, RealPolicyNode, UnsafeDeploymentOverrideError, build_effective_profile,
    _safety_limits_from_profile,
)
from hunter_kinodynamic_rl.navigation.local_rl.controller import LocalPolicyController  # noqa: E402
from hunter_kinodynamic_rl.trajectory.pure_pursuit_adapter import VehicleCommand  # noqa: E402


def _manifest_for(training_profile_name: str) -> dict:
    profile = load_profile(training_profile_name)
    return {
        "profile_name": training_profile_name,
        "resolved_config": dataclasses.asdict(profile),
    }


# ------------------------------------------------------ build_effective_profile (section P2)
def test_missing_resolved_config_raises():
    """A pre-P0-1 checkpoint (schema_version=1) has no resolved_config --
    must refuse rather than silently build an unverified architecture and
    publish real robot commands from it."""
    requested = load_profile("real_hunter_safe")
    with pytest.raises(SystemExit, match="resolved_config"):
        build_effective_profile({"profile_name": "kinodynamic_tqc"}, "real_hunter_safe", requested)


def test_matching_checkpoint_and_deployment_profile_succeeds():
    """kinodynamic_tqc_improved.yaml (the active full-system profile
    real_hunter_safe.yaml is designed to deploy) must pass compatibility
    and restore the checkpoint's own architecture-determining features."""
    manifest = _manifest_for("kinodynamic_tqc_improved")
    requested = load_profile("real_hunter_safe")
    effective = build_effective_profile(manifest, "real_hunter_safe", requested)
    assert effective.features.risk_critic is True
    assert effective.features.counterfactual_risk is True
    assert effective.action_space.mode == "trajectory"


def test_frozen_baseline_checkpoint_is_rejected_by_improved_deployment_profile():
    """A baseline checkpoint cannot be relabelled as an improved-model result."""
    manifest = _manifest_for("kinodynamic_tqc_counterfactual")
    requested = load_profile("real_hunter_safe")
    with pytest.raises(CheckpointProfileMismatchError, match="robot.(steering_limit_deg|name|track_width_m)"):
        build_effective_profile(manifest, "real_hunter_safe", requested)


def test_risk_critic_architecture_mismatch_raises_not_warns():
    """The core P2 regression: a checkpoint trained WITHOUT a risk critic
    (kinodynamic_tqc, plain TQC network shape) deployed against a REQUESTED
    profile that declares features.risk_critic=true would previously only
    WARN, then go on to build a RiskAgent and attempt to load a checkpoint
    whose tensors don't match that architecture -- now it must raise before
    any agent is constructed."""
    manifest = _manifest_for("kinodynamic_tqc")  # features.risk_critic: false
    requested = dataclasses.replace(
        load_profile("real_hunter_safe"),
        features=dataclasses.replace(load_profile("real_hunter_safe").features, risk_critic=True),
    )
    with pytest.raises(CheckpointProfileMismatchError, match="risk_critic"):
        build_effective_profile(manifest, "real_hunter_safe", requested)


def test_action_space_mode_mismatch_raises():
    manifest = _manifest_for("legacy_waypoint_tqc")  # action_space.mode: legacy_waypoint
    requested = load_profile("real_hunter_safe")  # action_space.mode: trajectory
    with pytest.raises(CheckpointProfileMismatchError, match="action_space.mode"):
        build_effective_profile(manifest, "real_hunter_safe", requested)


def test_temporal_context_mismatch_raises():
    manifest = _manifest_for("kinodynamic_tqc")  # features.temporal_context: false
    base = load_profile("real_hunter_safe")
    requested = dataclasses.replace(base, features=dataclasses.replace(base.features, temporal_context=True))
    with pytest.raises(CheckpointProfileMismatchError, match="temporal_context"):
        build_effective_profile(manifest, "real_hunter_safe", requested)


def test_deployment_speed_cap_survives_a_matching_checkpoint():
    """The deployment-safety override (real_hunter_safe.yaml's conservative
    robot.max_forward_speed_mps cap) must be applied even though the
    checkpoint's OWN training robot config had a higher (sim) speed --
    proving safety overrides are layered independently of the architecture
    restoration, per build_effective_profile's docstring."""
    manifest = _manifest_for("kinodynamic_tqc_improved")
    requested = load_profile("real_hunter_safe")
    training = load_profile("kinodynamic_tqc_improved")
    assert requested.robot.max_forward_speed_mps < training.robot.max_forward_speed_mps  # sanity: sim is faster
    effective = build_effective_profile(manifest, "real_hunter_safe", requested)
    assert effective.robot.max_forward_speed_mps == pytest.approx(requested.robot.max_forward_speed_mps)


def test_deployment_min_safe_clearance_survives_a_matching_checkpoint():
    manifest = _manifest_for("kinodynamic_tqc_improved")
    requested = load_profile("real_hunter_safe")
    effective = build_effective_profile(manifest, "real_hunter_safe", requested)
    assert effective.risk.min_safe_clearance_m == pytest.approx(requested.risk.min_safe_clearance_m)


# --------------------------------- shrink/raise-only robot-tunable override policy (section P2, round 2)
#
# code review: build_effective_profile previously validated ONLY
# wheelbase_m/steering_limit_deg (exact) and the derived effective v_max,
# then took the requested profile's ENTIRE `robot` section verbatim into
# `effective` -- so steering_rate_deg_s/accel_limit_mps2/brake_decel_mps2/
# speed_lag_tau_sec/collision_radius_m could silently become MORE
# OPTIMISTIC than training (faster steering/accel/braking, less lag, a
# smaller footprint), all of which the LIVE risk rollout
# (dynamics/ackermann_rollout.py) and stopping-margin/guard check
# (dynamics/stopping_model.py, env/safety/action_guard.py) would then
# underestimate real risk under. These tests cover each field's ALLOWED
# (conservative) direction succeeding and its REJECTED (optimistic)
# direction raising, plus that non-tunable geometry/identity fields are
# exact-match now instead of silently overwritten.


def _real_hunter_safe_with_robot(**overrides) -> "Profile":
    base = load_profile("real_hunter_safe")
    return dataclasses.replace(base, robot=dataclasses.replace(base.robot, **overrides))


def test_real_hunter_safe_yaml_still_passes_end_to_end():
    """The shipped profile itself (only overriding max_forward_speed_mps)
    must keep working unchanged after this fix."""
    manifest = _manifest_for("kinodynamic_tqc_improved")
    requested = load_profile("real_hunter_safe")
    effective = build_effective_profile(manifest, "real_hunter_safe", requested)
    assert effective.robot.max_forward_speed_mps == pytest.approx(0.8)
    # every other robot field must be the CHECKPOINT's own (training-time)
    # value, not silently re-derived from the requested profile.
    training = load_profile("kinodynamic_tqc_improved")
    assert effective.robot.wheelbase_m == pytest.approx(training.robot.wheelbase_m)
    assert effective.robot.mass_kg == pytest.approx(training.robot.mass_kg)


@pytest.mark.parametrize("field,value", [
    ("steering_rate_deg_s", 199.0),   # <= training's 200.0: allowed (slower-or-equal)
    ("accel_limit_mps2", 1.9),        # <= training's 2.0: allowed
    ("brake_decel_mps2", 1.9),        # <= training's 2.0: allowed
    ("speed_lag_tau_sec", 0.06),      # >= training's 0.05: allowed
    ("collision_radius_m", 0.59),     # >= training's 0.58: allowed
])
def test_conservative_robot_tunable_override_is_accepted(field, value):
    manifest = _manifest_for("kinodynamic_tqc_improved")
    requested = _real_hunter_safe_with_robot(**{field: value})
    effective = build_effective_profile(manifest, "real_hunter_safe", requested)
    assert getattr(effective.robot, field) == pytest.approx(value)


@pytest.mark.parametrize("field,value,training_value", [
    ("steering_rate_deg_s", 250.0, 200.0),   # FASTER than training: rejected
    ("accel_limit_mps2", 3.0, 2.0),          # STRONGER accel than training: rejected
    ("brake_decel_mps2", 3.0, 2.0),          # STRONGER braking than training: rejected
    ("speed_lag_tau_sec", 0.01, 0.05),       # LESS lag than training: rejected
    ("collision_radius_m", 0.3, 0.58),       # SMALLER footprint than training: rejected
])
def test_optimistic_robot_tunable_override_raises_unsafe_override(field, value, training_value):
    manifest = _manifest_for("kinodynamic_tqc_improved")  # sanity-checked training values above
    training = load_profile("kinodynamic_tqc_improved")
    assert getattr(training.robot, field) == pytest.approx(training_value)
    requested = _real_hunter_safe_with_robot(**{field: value})
    with pytest.raises(UnsafeDeploymentOverrideError, match=field):
        build_effective_profile(manifest, "real_hunter_safe", requested)


def test_direct_max_forward_speed_mps_override_is_shrink_only_even_with_v_max_mps_set():
    """robot.max_forward_speed_mps must be validated DIRECTLY (not just via
    the derived effective-v_max check), because action_guard.guard()
    enforces it as a hard cap regardless of action_space.v_max_mps. Here
    action_space.v_max_mps is explicitly set (so the OLD effective-v_max
    check, which only fires when v_max_mps is None, would have missed a
    raised robot.max_forward_speed_mps entirely) -- must still raise."""
    manifest = _manifest_for("kinodynamic_tqc_improved")  # robot.max_forward_speed_mps == 1.3333333333
    base = load_profile("real_hunter_safe")
    requested = dataclasses.replace(
        base,
        robot=dataclasses.replace(base.robot, max_forward_speed_mps=5.0),
        action_space=dataclasses.replace(base.action_space, v_max_mps=0.5),  # unrelated to robot's own cap
    )
    with pytest.raises(UnsafeDeploymentOverrideError, match="max_forward_speed_mps"):
        build_effective_profile(manifest, "real_hunter_safe", requested)


@pytest.mark.parametrize("field,value", [
    ("mass_kg", 50.0),
    ("wheel_radius_m", 0.2),
    ("track_width_m", 0.6),
    ("length_m", 1.0),
    ("width_m", 0.5),
    ("height_m", 0.2),
    ("min_forward_speed_mps", 0.1),
])
def test_non_tunable_robot_field_drift_raises_architecture_mismatch(field, value):
    """These fields are never read by any inference-time (guard/risk/
    action-decode) code path -- confirmed by grep -- so a deployment
    profile changing them is now an explicit, rejected mismatch rather
    than a silent, effect-free override."""
    manifest = _manifest_for("kinodynamic_tqc_improved")
    requested = _real_hunter_safe_with_robot(**{field: value})
    with pytest.raises(CheckpointProfileMismatchError, match=field):
        build_effective_profile(manifest, "real_hunter_safe", requested)


# ------------------------------------------ shrink-only action-space override policy (section P2)
def _real_hunter_safe_with_action_space(**overrides) -> "Profile":
    base = load_profile("real_hunter_safe")
    return dataclasses.replace(base, action_space=dataclasses.replace(base.action_space, **overrides))


def test_widened_kappa_scale_raises_unsafe_override_not_architecture_mismatch():
    """The core P2 regression: kappa_scale is NOT an architecture field
    (action_space.mode is unchanged), so the OLD code accepted this
    silently -- widening it beyond the checkpoint's own training range
    means the actor's raw output now reaches curvatures it never explored.
    Must raise UnsafeDeploymentOverrideError (a distinct type from
    architecture mismatches, but still a CheckpointProfileMismatchError /
    SystemExit, so any caller catching the base type still stops)."""
    manifest = _manifest_for("kinodynamic_tqc_improved")  # kappa_scale defaults to 1.0
    requested = _real_hunter_safe_with_action_space(kappa_scale=1.0)
    build_effective_profile(manifest, "real_hunter_safe", requested)  # sanity: 1.0 == 1.0 is fine

    widened = _real_hunter_safe_with_action_space(kappa_scale=1.0 + 1e-3)
    with pytest.raises(UnsafeDeploymentOverrideError, match="kappa_scale"):
        build_effective_profile(manifest, "real_hunter_safe", widened)
    with pytest.raises(CheckpointProfileMismatchError, match="kappa_scale"):
        build_effective_profile(manifest, "real_hunter_safe", widened)  # also catchable via the base type


def test_lowered_v_min_raises_unsafe_override():
    manifest = _manifest_for("kinodynamic_tqc_improved")  # v_min_mps defaults to 0.0
    requested = _real_hunter_safe_with_action_space(v_min_mps=-0.1)
    with pytest.raises(UnsafeDeploymentOverrideError, match="v_min_mps"):
        build_effective_profile(manifest, "real_hunter_safe", requested)


def test_widened_horizon_length_min_raises_unsafe_override():
    manifest = _manifest_for("kinodynamic_tqc_improved")  # horizon_length_min_m defaults to 0.5
    requested = _real_hunter_safe_with_action_space(horizon_length_min_m=0.1)
    with pytest.raises(UnsafeDeploymentOverrideError, match="horizon_length_min_m"):
        build_effective_profile(manifest, "real_hunter_safe", requested)


def test_widened_horizon_length_max_raises_unsafe_override():
    manifest = _manifest_for("kinodynamic_tqc_improved")  # horizon_length_max_m defaults to 3.0
    requested = _real_hunter_safe_with_action_space(horizon_length_max_m=3.5)
    with pytest.raises(UnsafeDeploymentOverrideError, match="horizon_length_max_m"):
        build_effective_profile(manifest, "real_hunter_safe", requested)


def test_raised_effective_v_max_raises_unsafe_override():
    """A deployment profile that RAISES robot.max_forward_speed_mps beyond
    the checkpoint's own (the opposite of real_hunter_safe.yaml's actual,
    conservative 0.8 m/s cap) must be rejected -- proves this ISN'T just a
    one-directional "the field changed" check, it's actually directional."""
    manifest = _manifest_for("kinodynamic_tqc_improved")  # robot.max_forward_speed_mps == 1.3333333333
    base = load_profile("real_hunter_safe")
    requested = dataclasses.replace(base, robot=dataclasses.replace(base.robot, max_forward_speed_mps=5.0))
    with pytest.raises(UnsafeDeploymentOverrideError, match="v_max"):
        build_effective_profile(manifest, "real_hunter_safe", requested)


def test_lowered_min_safe_clearance_raises_unsafe_override():
    manifest = _manifest_for("kinodynamic_tqc_improved")  # risk.min_safe_clearance_m defaults to 0.3
    base = load_profile("real_hunter_safe")
    requested = dataclasses.replace(base, risk=dataclasses.replace(base.risk, min_safe_clearance_m=0.1))
    with pytest.raises(UnsafeDeploymentOverrideError, match="min_safe_clearance_m"):
        build_effective_profile(manifest, "real_hunter_safe", requested)


def test_shrinking_horizon_length_max_below_training_still_succeeds():
    """The actual real_hunter_safe.yaml shape (horizon_length_max_m=1.5,
    shorter than kinodynamic_tqc_improved's default 3.0) must keep
    working -- this is the ALLOWED direction, not a false positive from
    the new check."""
    manifest = _manifest_for("kinodynamic_tqc_improved")
    requested = load_profile("real_hunter_safe")
    effective = build_effective_profile(manifest, "real_hunter_safe", requested)
    assert effective.action_space.horizon_length_max_m == pytest.approx(1.5)


def test_legacy_yield_enabled_mismatch_raises_for_legacy_waypoint_mode():
    """legacy_yield_enabled is an exact-match field (toggling it is a
    semantics change, not a numeric shrink) -- only reachable when BOTH
    sides are legacy_waypoint mode (else action_space.mode itself would
    already have raised as an architecture mismatch)."""
    manifest = _manifest_for("legacy_waypoint_tqc")  # legacy_yield_enabled defaults to True
    base = load_profile("legacy_waypoint_tqc")
    requested = dataclasses.replace(
        base, action_space=dataclasses.replace(base.action_space, legacy_yield_enabled=False),
        runtime=dataclasses.replace(base.runtime, deployment="real_hardware"),
    )
    with pytest.raises(UnsafeDeploymentOverrideError, match="legacy_yield_enabled"):
        build_effective_profile(manifest, "legacy_waypoint_tqc", requested)


class _FakePublisher:
    def __init__(self):
        self.published = []

    def publish(self, msg):
        self.published.append(msg)


def _bare_node(dry_run: bool) -> RealPolicyNode:
    node = RealPolicyNode.__new__(RealPolicyNode)
    node.dry_run = dry_run
    node._estopped = False
    node._cmd_pub = _FakePublisher()
    return node


def test_publish_reaches_cmd_vel_when_not_dry_run():
    node = _bare_node(dry_run=False)
    node._publish(VehicleCommand(speed_mps=1.0, steering_rad=0.1))
    assert len(node._cmd_pub.published) == 1
    assert node._cmd_pub.published[0].linear.x == pytest.approx(1.0)


def test_publish_never_reaches_cmd_vel_when_dry_run():
    """The core P2-11 regression: dry_run must be the ONE gate between the
    computed command and the actual publisher call, with NO path around it."""
    node = _bare_node(dry_run=True)
    node._publish(VehicleCommand(speed_mps=2.0, steering_rad=-0.2))
    assert node._cmd_pub.published == []


def test_on_estop_latches_true_and_clears_on_false():
    node = _bare_node(dry_run=False)
    node.get_logger = lambda: _NullLogger()

    class _Msg:
        data = True

    node._on_estop(_Msg())
    assert node._estopped is True

    _Msg.data = False
    node._on_estop(_Msg())
    assert node._estopped is False


def test_estopped_control_tick_publishes_stop_and_skips_inference():
    """Once latched, the E-stop must short-circuit BEFORE any sensor-data
    check or inference call -- the tick must publish STOP_COMMAND even if
    fresh sensor data is available (an operator stop overrides everything
    else, not just the "no sensor data yet" fallback path)."""
    node = _bare_node(dry_run=False)
    node._estopped = True
    node._latest_scan = ("sentinel-should-never-be-read",)
    node._latest_odom = ("sentinel-should-never-be-read",)

    def _boom(*a, **kw):
        raise AssertionError("inference/observation pipeline ran despite E-stop being engaged")
    node.agent = type("FakeAgent", (), {"select_action": staticmethod(_boom)})()

    node._on_control_tick()
    assert len(node._cmd_pub.published) == 1
    published = node._cmd_pub.published[0]
    assert published.linear.x == pytest.approx(STOP_COMMAND.speed_mps)
    assert published.angular.z == pytest.approx(STOP_COMMAND.steering_rad)


class _NullLogger:
    def error(self, *a, **kw):
        pass

    def warn(self, *a, **kw):
        pass

    def info(self, *a, **kw):
        pass


# ---------------------------------------------------------------- section P0-3
def test_safety_limits_wired_from_effective_profile():
    """The core P0-3 regression: real_hunter_safe.yaml's own
    risk.min_safe_clearance_m=0.5 (deliberately more conservative than the
    0.3 m sim default) must actually reach the guard's
    ``min_obstacle_stop_distance_m`` -- previously a bare, default-
    constructed ``SafetyLimits()`` silently ignored it."""
    profile = load_profile("real_hunter_safe")
    assert profile.risk.min_safe_clearance_m == pytest.approx(0.5)
    limits = _safety_limits_from_profile(profile)
    assert isinstance(limits, SafetyLimits)
    assert limits.min_obstacle_stop_distance_m == pytest.approx(0.5)
    assert limits.max_sensor_age_sec == pytest.approx(profile.runtime.sensor_freshness_timeout_sec)
    assert limits.max_odom_age_sec == pytest.approx(profile.runtime.sensor_freshness_timeout_sec)
    assert limits.max_command_age_sec == pytest.approx(profile.runtime.watchdog_command_timeout_sec)


def _full_bare_node(**overrides) -> RealPolicyNode:
    """Builds a bare RealPolicyNode with every attribute `_on_control_tick`
    reads, using real_hunter_safe.yaml's own profile shape (loaded once,
    never constructing a live rclpy Node). A default no-op agent returns a
    finite, correctly-shaped zero action unless overridden."""
    profile = load_profile("real_hunter_safe")
    node = RealPolicyNode.__new__(RealPolicyNode)
    node.profile = profile
    node.dry_run = False
    node.replay_mode = False
    node._estopped = False
    node._cmd_pub = _FakePublisher()
    node._diag_pub = _FakePublisher()
    node.get_logger = lambda: _NullLogger()
    node.goal_x = 3.0
    node.goal_y = 1.0
    node._latest_steering_rad = 0.0
    node._local_controller = LocalPolicyController(profile)
    node._safety_limits = _safety_limits_from_profile(profile)
    node._inference_timeouts = 0
    node._inference_errors = 0
    node._last_successful_inference_time = None
    node._last_command_time = None
    node._inference_worker = None  # item-5: default thread mode
    node._inference_thread = None
    node._inference_consecutive_timeouts = 0

    n_ranges = 360
    ranges = np.full(n_ranges, profile.observation.lidar_max_range_m, dtype=np.float32)
    now = time.monotonic()
    node._latest_scan = (ranges, -np.pi, 2 * np.pi / n_ranges)
    node._latest_scan_time = now
    node._latest_odom = (0.0, 0.0, 0.0, 0.0, 0.0)
    node._latest_odom_time = now

    class _ZeroAgent:
        @staticmethod
        def select_action(observation, deterministic=True):
            return np.zeros(3, dtype=np.float32)

    node.agent = _ZeroAgent()
    for key, value in overrides.items():
        setattr(node, key, value)
    return node


def test_control_tick_publishes_real_command_on_healthy_tick():
    node = _full_bare_node()
    node._on_control_tick()
    assert len(node._cmd_pub.published) == 1
    assert node._inference_timeouts == 0
    assert node._inference_errors == 0
    assert node._last_successful_inference_time is not None


def test_stale_odometry_forces_safe_stop():
    """section P0-3: odometry staleness must be checked INDEPENDENTLY of
    LiDAR staleness -- scan stays fresh, only odom is stale."""
    node = _full_bare_node()
    node._latest_odom_time = time.monotonic() - 10.0  # far beyond max_odom_age_sec
    node._on_control_tick()
    published = node._cmd_pub.published[-1]
    assert published.linear.x == pytest.approx(STOP_COMMAND.speed_mps)


def test_inference_timeout_publishes_safe_stop_and_counts_it():
    """section P0-3: a policy call that never returns within
    runtime.policy_inference_timeout_sec must not block the tick forever --
    the tick gives up, publishes a safe stop, and the timeout is counted."""
    class _HangingAgent:
        @staticmethod
        def select_action(observation, deterministic=True):
            time.sleep(5.0)  # far longer than any reasonable timeout budget
            return np.zeros(3, dtype=np.float32)

    node = _full_bare_node(agent=_HangingAgent())
    node.profile = dataclasses.replace(
        node.profile, runtime=dataclasses.replace(node.profile.runtime, policy_inference_timeout_sec=0.05))
    node._on_control_tick()
    assert node._inference_timeouts == 1
    published = node._cmd_pub.published[-1]
    assert published.linear.x == pytest.approx(STOP_COMMAND.speed_mps)
    assert node._last_successful_inference_time is None


def test_inference_exception_publishes_safe_stop_and_counts_it():
    class _BoomAgent:
        @staticmethod
        def select_action(observation, deterministic=True):
            raise RuntimeError("simulated inference crash")

    node = _full_bare_node(agent=_BoomAgent())
    node._on_control_tick()
    assert node._inference_errors == 1
    published = node._cmd_pub.published[-1]
    assert published.linear.x == pytest.approx(STOP_COMMAND.speed_mps)


def test_nan_action_publishes_safe_stop_without_reaching_trajectory_executor():
    class _NanAgent:
        @staticmethod
        def select_action(observation, deterministic=True):
            return np.array([float("nan"), 0.0, 0.0], dtype=np.float32)

    node = _full_bare_node(agent=_NanAgent())
    node._on_control_tick()
    published = node._cmd_pub.published[-1]
    assert published.linear.x == pytest.approx(STOP_COMMAND.speed_mps)
    # a rejected action must never be adopted as prev_action for the NEXT tick
    assert node._local_controller.prev_action == [0.0, 0.0, 0.0]


def test_wrong_shape_action_publishes_safe_stop():
    class _WrongShapeAgent:
        @staticmethod
        def select_action(observation, deterministic=True):
            return np.zeros(2, dtype=np.float32)  # ACTION_DIM is 3

    node = _full_bare_node(agent=_WrongShapeAgent())
    node._on_control_tick()
    published = node._cmd_pub.published[-1]
    assert published.linear.x == pytest.approx(STOP_COMMAND.speed_mps)


def test_unexpected_exception_in_pipeline_still_publishes_safe_stop():
    """section P0-3: an exception ANYWHERE in the pipeline (not just
    inference) must still result in a published safe stop, never a
    silently-skipped tick."""
    node = _full_bare_node()
    node._local_controller = None  # guaranteed AttributeError deep in the pipeline
    node._on_control_tick()
    assert len(node._cmd_pub.published) == 1
    published = node._cmd_pub.published[0]
    assert published.linear.x == pytest.approx(STOP_COMMAND.speed_mps)


def test_watchdog_thread_republishes_stop_after_command_goes_stale():
    """section P0-3: the watchdog runs on an independent Python thread, not
    an rclpy timer -- it must keep working even while nothing calls
    `_on_control_tick` at all (simulating a genuinely hung control loop)."""
    node = _full_bare_node()
    node.profile = dataclasses.replace(
        node.profile,
        runtime=dataclasses.replace(
            node.profile.runtime, watchdog_command_timeout_sec=0.05, real_policy_watchdog_period_sec=0.02),
    )
    node._last_command_time = time.monotonic()
    node._start_watchdog_thread()
    try:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not node._cmd_pub.published:
            time.sleep(0.02)
    finally:
        node._watchdog_stop_event.set()
        node._watchdog_thread.join(timeout=1.0)
    assert node._cmd_pub.published, "watchdog thread never published a stop for a stale command"
    published = node._cmd_pub.published[-1]
    assert published.linear.x == pytest.approx(STOP_COMMAND.speed_mps)


# --------------------------------------------------------------- item-5
def test_thread_mode_single_flight_does_not_spawn_a_new_thread_on_repeated_timeouts():
    """The core item-5 regression: a genuinely (not just slow) wedged
    select_action must NOT cause a new worker thread to be spawned on every
    subsequent control tick -- previously this leaked one new daemon thread
    per tick for as long as the hang lasted (10/sec at the default control
    rate). At most ONE extra thread should ever exist, no matter how many
    ticks elapse while it's still hung."""
    release = threading.Event()

    class _PermanentHangAgent:
        @staticmethod
        def select_action(observation, deterministic=True):
            release.wait(10.0)  # far longer than this test's timeout budget
            return np.zeros(3, dtype=np.float32)

    node = _full_bare_node(agent=_PermanentHangAgent())
    node.profile = dataclasses.replace(
        node.profile, runtime=dataclasses.replace(node.profile.runtime, policy_inference_timeout_sec=0.02))
    try:
        threads_before = threading.active_count()
        for _ in range(20):
            node._on_control_tick()
        threads_after = threading.active_count()
        assert node._inference_timeouts == 20
        assert node._inference_consecutive_timeouts == 20
        # exactly the one hung worker thread, never 20
        assert threads_after - threads_before <= 1
        assert node._inference_thread is not None and node._inference_thread.is_alive()
        for published in node._cmd_pub.published:
            assert published.linear.x == pytest.approx(STOP_COMMAND.speed_mps)
    finally:
        release.set()
        if node._inference_thread is not None:
            node._inference_thread.join(timeout=2.0)


def test_thread_mode_recovers_once_the_previously_hung_thread_finally_returns():
    release = threading.Event()

    class _EventuallyReturnsAgent:
        @staticmethod
        def select_action(observation, deterministic=True):
            release.wait(5.0)
            return np.zeros(3, dtype=np.float32)

    node = _full_bare_node(agent=_EventuallyReturnsAgent())
    node.profile = dataclasses.replace(
        node.profile, runtime=dataclasses.replace(node.profile.runtime, policy_inference_timeout_sec=0.02))
    node._on_control_tick()
    assert node._inference_timeouts == 1
    assert node._inference_thread is not None and node._inference_thread.is_alive()

    release.set()
    node._inference_thread.join(timeout=2.0)
    assert not node._inference_thread.is_alive()

    node._on_control_tick()  # the stale thread is done -- a NEW one may now start
    assert node._inference_timeouts == 1  # this tick completed within budget, no new timeout
    assert node._inference_consecutive_timeouts == 0
    assert node._last_successful_inference_time is not None


def test_exception_during_inference_resets_the_consecutive_timeout_counter():
    class _BoomAgent:
        @staticmethod
        def select_action(observation, deterministic=True):
            raise RuntimeError("simulated inference crash")

    node = _full_bare_node(agent=_BoomAgent())
    node._inference_consecutive_timeouts = 2
    node._on_control_tick()
    assert node._inference_errors == 1
    assert node._inference_consecutive_timeouts == 0


def test_estop_short_circuits_immediately_even_while_a_previous_inference_is_hung():
    """An operator E-stop must never be delayed by a hung inference call --
    it must not attempt to wait on (or touch) the stale worker thread at
    all."""
    release = threading.Event()

    class _HangingAgent:
        @staticmethod
        def select_action(observation, deterministic=True):
            release.wait(5.0)
            return np.zeros(3, dtype=np.float32)

    node = _full_bare_node(agent=_HangingAgent())
    node.profile = dataclasses.replace(
        node.profile, runtime=dataclasses.replace(node.profile.runtime, policy_inference_timeout_sec=0.02))
    try:
        node._on_control_tick()
        assert node._inference_thread is not None and node._inference_thread.is_alive()

        node._estopped = True
        start = time.monotonic()
        node._on_control_tick()
        elapsed = time.monotonic() - start
        assert elapsed < 0.1
        published = node._cmd_pub.published[-1]
        assert published.linear.x == pytest.approx(STOP_COMMAND.speed_mps)
    finally:
        release.set()
        if node._inference_thread is not None:
            node._inference_thread.join(timeout=2.0)


def test_dry_run_never_publishes_under_repeated_inference_timeouts():
    release = threading.Event()

    class _HangingAgent:
        @staticmethod
        def select_action(observation, deterministic=True):
            release.wait(5.0)
            return np.zeros(3, dtype=np.float32)

    node = _full_bare_node(dry_run=True, agent=_HangingAgent())
    node.profile = dataclasses.replace(
        node.profile, runtime=dataclasses.replace(node.profile.runtime, policy_inference_timeout_sec=0.02))
    try:
        for _ in range(5):
            node._on_control_tick()
        assert node._cmd_pub.published == []
        assert node._inference_timeouts == 5
    finally:
        release.set()
        if node._inference_thread is not None:
            node._inference_thread.join(timeout=2.0)


def test_control_tick_remains_responsive_under_gil_contention():
    """A competing CPU-bound thread hogging the GIL must not prevent
    control ticks from completing within a bounded multiple of the
    configured timeout -- the watchdog/E-stop/dry_run paths must stay
    responsive rather than the whole process effectively wedging."""
    stop_busy = threading.Event()

    def _busy_loop():
        while not stop_busy.is_set():
            pass  # tight CPU-bound loop, contends for the GIL

    busy_thread = threading.Thread(target=_busy_loop, daemon=True)
    busy_thread.start()
    try:
        node = _full_bare_node()  # real, fast _ZeroAgent inference
        node.profile = dataclasses.replace(
            node.profile, runtime=dataclasses.replace(node.profile.runtime, policy_inference_timeout_sec=0.5))
        start = time.monotonic()
        for _ in range(5):
            node._on_control_tick()
        elapsed = time.monotonic() - start
        assert elapsed < 5.0  # generous: 5 ticks * 0.5s budget, contention or not
        assert len(node._cmd_pub.published) == 5
        assert node._inference_timeouts == 0
    finally:
        stop_busy.set()
        busy_thread.join(timeout=2.0)


def test_watchdog_logs_policy_health_warning_after_sustained_inference_failure():
    """item-5: last_successful_inference_time -- not just last_command_time
    -- drives a distinct, throttled diagnostic when the POLICY itself has
    stopped producing usable actions, even though the robot is already kept
    safely stopped regardless (command-freshness watchdog disabled here via
    a very long timeout, to isolate this specific check)."""
    node = _full_bare_node()
    node.profile = dataclasses.replace(
        node.profile,
        runtime=dataclasses.replace(
            node.profile.runtime, policy_health_timeout_sec=0.05, real_policy_watchdog_period_sec=0.02,
            watchdog_command_timeout_sec=1000.0,
        ),
    )
    node._last_command_time = time.monotonic()
    node._last_successful_inference_time = time.monotonic() - 10.0

    errors = []

    class _CapturingLogger:
        def error(self, msg):
            errors.append(msg)

        def warn(self, *a, **kw):
            pass

        def info(self, *a, **kw):
            pass

    node.get_logger = lambda: _CapturingLogger()
    node._start_watchdog_thread()
    try:
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and not errors:
            time.sleep(0.02)
    finally:
        node._watchdog_stop_event.set()
        node._watchdog_thread.join(timeout=1.0)
    assert any("no SUCCESSFUL policy inference" in m for m in errors)
    # never actuates -- this is a diagnostic-only signal
    assert node._cmd_pub.published == []


def test_watchdog_thread_respects_dry_run():
    """The watchdog must go through `_publish` (dry_run-gated), never a raw
    publisher bypass -- section P2-11's "no actuation in dry_run, ever"
    guarantee must hold for this path too."""
    node = _full_bare_node(dry_run=True)
    node.profile = dataclasses.replace(
        node.profile,
        runtime=dataclasses.replace(
            node.profile.runtime, watchdog_command_timeout_sec=0.05, real_policy_watchdog_period_sec=0.02),
    )
    node._last_command_time = time.monotonic()
    node._start_watchdog_thread()
    time.sleep(0.3)
    node._watchdog_stop_event.set()
    node._watchdog_thread.join(timeout=1.0)
    assert node._cmd_pub.published == []
