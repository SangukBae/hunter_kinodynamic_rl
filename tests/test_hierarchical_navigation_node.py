"""Phase 5/6 hierarchical_navigation_node.py (plan section 10.6): mission
lifecycle wiring on top of the Phase 4 HierarchyCoordinator/PartialMap/
Global-RL stack. Mirrors tests/test_real_policy_node.py's own "bare
instance" harness (HierarchicalNavigationNode.__new__, skipping Node.__init__
and the ROS-wiring body) since full construction needs a live rclpy
context, a real checkpoint, and topic subscriptions."""

import dataclasses
import math
import time

import numpy as np
import pytest

pytest.importorskip("rclpy")  # module imports rclpy at module scope
pytest.importorskip("torch")  # ...and the RiskAgent/VanillaAgent kinodynamic_tqc/tqc modules at module scope

from nav_msgs.msg import Odometry  # noqa: E402
from sensor_msgs.msg import LaserScan  # noqa: E402

from hunter_kinodynamic_rl.config.loader import load_profile  # noqa: E402
from hunter_kinodynamic_rl.env.safety.action_guard import STOP_COMMAND  # noqa: E402
from hunter_kinodynamic_rl.nodes.hierarchical_navigation_node import (  # noqa: E402
    HierarchicalNavigationNode, _build_localization_backend,
)
from hunter_kinodynamic_rl.navigation.hierarchy.coordinator import HierarchyCoordinator  # noqa: E402
from hunter_kinodynamic_rl.navigation.local_rl.controller import LocalPolicyController  # noqa: E402
from hunter_kinodynamic_rl.navigation.localization.gazebo_odom_backend import GazeboOdomLocalizationBackend  # noqa: E402
from hunter_kinodynamic_rl.navigation.localization.lio_adapter import LioLocalizationBackend  # noqa: E402
from hunter_kinodynamic_rl.navigation.localization.wheel_imu_backend import WheelImuLocalizationBackend  # noqa: E402
from hunter_kinodynamic_rl.navigation.mapping.partial_map import PartialMap  # noqa: E402
from hunter_kinodynamic_rl.navigation.mission.mission_frame import MissionFrame, PoseXYYaw  # noqa: E402
from hunter_kinodynamic_rl.navigation.ros.mission_map_node import yaw_to_quaternion  # noqa: E402
from hunter_kinodynamic_rl.navigation.global_rl.subgoal_sampler import build_candidate_set  # noqa: E402
from hunter_kinodynamic_rl.training.train_hierarchical_dqn import hierarchy_config_from  # noqa: E402


class _NullLogger:
    def info(self, *a, **k): pass
    def warn(self, *a, **k): pass
    def error(self, *a, **k): pass


class _FakePublisher:
    def __init__(self):
        self.published = []

    def publish(self, msg):
        self.published.append(msg)


class _ZeroLocalAgent:
    @staticmethod
    def select_action(observation, deterministic=True):
        return np.zeros(3, dtype=np.float32)


def _bare_node(profile_name: str = "hierarchical_phase5_b", **overrides) -> HierarchicalNavigationNode:
    profile = load_profile(profile_name)
    node = HierarchicalNavigationNode.__new__(HierarchicalNavigationNode)
    node.profile = profile
    node.dry_run = False
    node.replay_mode = False
    node._estopped = False
    node.get_logger = lambda: _NullLogger()
    node._cmd_pub = _FakePublisher()
    node.mission_frame = MissionFrame()
    node.partial_map = PartialMap(profile.mapping)
    node.coordinator = HierarchyCoordinator(hierarchy_config_from(profile.hierarchy))
    node.candidates = build_candidate_set(profile.global_rl)
    node._local_controller = LocalPolicyController(profile)
    node.local_agent = _ZeroLocalAgent()
    node.global_agent = None
    node.memory_enabled = False
    node.feasibility_enabled = False
    node.global_risk_enabled = False
    node.graph = None
    node.node_manager = None
    node.dead_end_detector = None
    node.feasibility_config = None
    node.max_nodes = 0
    node._safety_limits = None
    from hunter_kinodynamic_rl.env.safety.action_guard import SafetyLimits
    node._safety_limits = SafetyLimits(
        max_sensor_age_sec=profile.runtime.sensor_freshness_timeout_sec,
        max_odom_age_sec=profile.runtime.sensor_freshness_timeout_sec,
        max_command_age_sec=profile.runtime.watchdog_command_timeout_sec,
        min_obstacle_stop_distance_m=profile.risk.min_safe_clearance_m,
    )
    node._policy_inference_timeout_sec = 0.5
    node._latest_scan = None
    node._latest_scan_time = None
    node._latest_scan_receipt_time = None
    node._latest_odom = None
    node._latest_odom_time = None
    node._latest_odom_receipt_time = None
    node._latest_steering_rad = 0.0
    node._last_command_time = None
    node._last_successful_inference_time = None
    from collections import deque
    node._speed_history = deque(maxlen=50)
    node._step_counter = 0
    node._rng = np.random.RandomState(0)
    node._final_goal_mission = None
    node._inference_thread = None
    node._goal_x = 10.0
    node._goal_y = 0.0
    node._wheel_imu_initialized = False
    for key, value in overrides.items():
        setattr(node, key, value)
    return node


def test_control_tick_stops_before_mission_frame_initialized():
    node = _bare_node()
    node._on_control_tick()
    # not dry_run -- a zero-velocity STOP_COMMAND is still published (it is
    # a real, safe command), just never a nonzero one.
    assert len(node._cmd_pub.published) == 1
    assert node._cmd_pub.published[0].linear.x == 0.0
    assert node._cmd_pub.published[0].angular.z == 0.0
    assert node._last_command_time is not None


def test_mission_starts_and_first_subgoal_activates():
    node = _bare_node()
    node.mission_frame.initialize(PoseXYYaw(0.0, 0.0, 0.0))
    node._final_goal_mission = node.mission_frame.odom_to_mission(10.0, 0.0)
    node.coordinator.start_mission(node._final_goal_mission[0], node._final_goal_mission[1])
    node._latest_odom = (0.0, 0.0, 0.0, 0.0, 0.0)
    node._latest_odom_time = time.monotonic()
    node._latest_odom_receipt_time = time.monotonic()  # fresh receipt -- localization not stale
    # Provide a scan so _on_control_tick's precondition passes.
    ranges = np.full(360, 10.0, dtype=np.float32)
    node._latest_scan = (ranges, -np.pi, 2 * np.pi / 360)
    node._latest_scan_time = node._latest_odom_time
    node._latest_scan_receipt_time = time.monotonic()

    from hunter_kinodynamic_rl.navigation.localization.odom_backend import OdomLocalizationBackend
    node._localization = OdomLocalizationBackend()
    node._localization.update(0.0, 0.0, 0.0, node._latest_odom_time, confidence=1.0, valid=True)

    node._on_control_tick()
    assert node.coordinator.active_subgoal_mission is not None


class _RecordingLocalAgent:
    def __init__(self):
        self.observations = []

    def select_action(self, observation, deterministic=True):
        self.observations.append(observation.copy())
        return np.zeros(3, dtype=np.float32)


def test_control_tick_builds_robot_relative_observation_at_nonorigin_pose():
    """Defect-fix item 1 regression, exercised through the REAL
    _on_control_tick call path (not a reimplementation): mission starts at
    a non-origin/non-zero-yaw pose, the robot then odometers to a
    DIFFERENT pose, and the active subgoal is mission-frame -- the
    observation the local agent actually receives must reflect
    distance/heading computed in the ROBOT's own frame (matching
    MissionFrame.mission_to_robot + LocalPolicyController.robot_relative_state),
    never the historical bug's frame-mismatched distance/heading (a
    robot-frame subgoal paired with a raw odom/mission-frame RobotState
    pose)."""
    from hunter_kinodynamic_rl.common.geometry import goal_distance_and_heading

    node = _bare_node()
    agent = _RecordingLocalAgent()
    node.local_agent = agent
    start_pose = PoseXYYaw(x=2.0, y=-3.0, yaw=0.3)
    node.mission_frame.initialize(start_pose)
    node._final_goal_mission = node.mission_frame.odom_to_mission(20.0, -3.0)
    node.coordinator.start_mission(node._final_goal_mission[0], node._final_goal_mission[1])

    odom_x, odom_y, odom_yaw = 5.0, 1.0, 1.1
    node._latest_odom = (odom_x, odom_y, odom_yaw, 0.6, -0.05)
    now = time.monotonic()
    node._latest_odom_time = now
    node._latest_odom_receipt_time = now
    ranges = np.full(360, 10.0, dtype=np.float32)
    node._latest_scan = (ranges, -np.pi, 2 * np.pi / 360)
    node._latest_scan_time = now
    node._latest_scan_receipt_time = now

    from hunter_kinodynamic_rl.navigation.localization.odom_backend import OdomLocalizationBackend
    node._localization = OdomLocalizationBackend()
    node._localization.update(odom_x, odom_y, odom_yaw, now, confidence=1.0, valid=True)

    node._on_control_tick()

    assert len(agent.observations) == 1
    active_subgoal_mission = node.coordinator.active_subgoal_mission
    assert active_subgoal_mission is not None
    robot_pose_mission = node.mission_frame.odom_pose_to_mission(PoseXYYaw(odom_x, odom_y, odom_yaw))
    expected_subgoal_robot = MissionFrame.mission_to_robot(active_subgoal_mission, robot_pose_mission)
    expected_dist, expected_heading = goal_distance_and_heading(0.0, 0.0, 0.0, *expected_subgoal_robot)

    obs = agent.observations[0]
    lidar_dim = node.profile.observation.lidar_bins * (
        node.profile.observation.frame_stack if node.profile.features.temporal_context else 1
    )
    tail = obs[lidar_dim:]
    assert tail[0] == pytest.approx(expected_dist, abs=1e-4)
    assert tail[1] == pytest.approx(expected_heading, abs=1e-4)
    # Sanity check this test actually exercises the regression: the buggy
    # (mismatched-frame) distance/heading -- computed from the RAW
    # odom/mission pose instead of the origin -- must differ from the
    # correct value for this non-trivial pose, or the assertions above
    # would trivially pass even with the bug still present.
    buggy_dist, buggy_heading = goal_distance_and_heading(
        robot_pose_mission.x, robot_pose_mission.y, robot_pose_mission.yaw, *expected_subgoal_robot,
    )
    assert (buggy_dist, buggy_heading) != pytest.approx((expected_dist, expected_heading))


def _make_odometry_msg(x: float, y: float, yaw: float, stamp_sec: float) -> Odometry:
    msg = Odometry()
    msg.header.stamp.sec = int(stamp_sec)
    msg.header.stamp.nanosec = int(round((stamp_sec - int(stamp_sec)) * 1e9))
    msg.pose.pose.position.x = x
    msg.pose.pose.position.y = y
    qx, qy, qz, qw = yaw_to_quaternion(yaw)
    msg.pose.pose.orientation.x, msg.pose.pose.orientation.y = qx, qy
    msg.pose.pose.orientation.z, msg.pose.pose.orientation.w = qz, qw
    return msg


def test_final_goal_fixed_correctly_for_nonzero_start_pose_and_yaw():
    """Regression (High finding): goal_x/goal_y are the user's relative
    goal in the robot's OWN frame at mission start -- by construction that
    IS the MissionFrame origin, so the mission-frame final goal must equal
    (goal_x, goal_y) EXACTLY for ANY start pose/yaw, never
    mission_frame.odom_to_mission(goal_x, goal_y) (which only happens to
    be a no-op when the start pose is exactly (0,0,0))."""
    node = _bare_node()
    node._goal_x, node._goal_y = 25.0, 10.0
    node._localization = GazeboOdomLocalizationBackend()

    msg = _make_odometry_msg(x=7.0, y=-3.0, yaw=1.2, stamp_sec=1.0)  # non-zero start pose AND yaw
    node._on_odom(msg)

    assert node.mission_frame.initialized
    assert node._final_goal_mission == pytest.approx((25.0, 10.0))
    # Sanity check that this test actually exercises the regression: the
    # (buggy) odom_to_mission-transformed value must be DIFFERENT from the
    # correct one for this non-zero start pose/yaw, or this assertion
    # would pass even with the bug still present.
    buggy_result = node.mission_frame.odom_to_mission(25.0, 10.0)
    assert buggy_result != pytest.approx((25.0, 10.0))


def test_final_goal_is_a_noop_transform_only_at_zero_start_pose():
    """Documents WHY the bug went unnoticed: at start pose (0,0,0), the
    buggy odom_to_mission() call and the correct direct assignment produce
    the SAME value."""
    node = _bare_node()
    node._goal_x, node._goal_y = 25.0, 10.0
    node._localization = GazeboOdomLocalizationBackend()
    msg = _make_odometry_msg(x=0.0, y=0.0, yaw=0.0, stamp_sec=1.0)
    node._on_odom(msg)
    assert node._final_goal_mission == pytest.approx((25.0, 10.0))
    assert node.mission_frame.odom_to_mission(25.0, 10.0) == pytest.approx((25.0, 10.0))


def test_scan_and_odom_receipt_time_tracked_separately_from_message_stamp():
    """Regression (High finding): action_guard.guard()'s now_sec must share
    a clock domain with last_sensor_time_sec/last_odom_time_sec -- this
    node tracks a SEPARATE time.monotonic() receipt timestamp for exactly
    that purpose, distinct from the message header stamp used for
    pose_at()/is_pose_usable() synchronization elsewhere."""
    node = _bare_node()
    node._localization = GazeboOdomLocalizationBackend()
    node.mission_frame.initialize(PoseXYYaw(0.0, 0.0, 0.0))

    # A message header stamp deliberately far from wall-clock "now" --
    # the common Gazebo sim-time convention (small values near epoch 0).
    odom_msg = _make_odometry_msg(x=0.0, y=0.0, yaw=0.0, stamp_sec=5.0)
    before = time.monotonic()
    node._on_odom(odom_msg)
    after = time.monotonic()

    assert node._latest_odom_time == pytest.approx(5.0)  # message-stamp domain, unchanged
    assert before <= node._latest_odom_receipt_time <= after  # monotonic receipt domain
    assert abs(node._latest_odom_time - node._latest_odom_receipt_time) > 1.0  # genuinely different domains

    scan_msg = LaserScan()
    scan_msg.header.stamp.sec = 7
    scan_msg.header.stamp.nanosec = 0
    scan_msg.ranges = [10.0] * 360
    scan_msg.angle_min = -math.pi
    scan_msg.angle_increment = 2 * math.pi / 360
    scan_msg.range_max = 12.0
    scan_msg.range_min = 0.1
    before_scan = time.monotonic()
    node._on_scan(scan_msg)
    after_scan = time.monotonic()
    assert node._latest_scan_time == pytest.approx(7.0)
    assert before_scan <= node._latest_scan_receipt_time <= after_scan


def test_control_tick_guard_call_uses_monotonic_receipt_time_not_message_stamp():
    """End-to-end check that _on_control_tick actually PASSES the receipt-
    time fields (not the message-stamp fields) into decode_and_guard --
    captures the real call's kwargs via a spy rather than inferring it
    from pass/fail behavior, so this test fails loudly (not just
    coincidentally) if the wiring regresses."""
    node = _bare_node()
    node.mission_frame.initialize(PoseXYYaw(0.0, 0.0, 0.0))
    node._final_goal_mission = node.mission_frame.odom_to_mission(10.0, 0.0)
    node.coordinator.start_mission(node._final_goal_mission[0], node._final_goal_mission[1])
    node._latest_odom = (0.0, 0.0, 0.0, 0.0, 0.0)
    # Message-stamp domain deliberately far from monotonic "now" -- the
    # captured call must NOT use these two values as its freshness inputs.
    node._latest_odom_time = 0.001
    node._latest_odom_receipt_time = time.monotonic()
    ranges = np.full(360, 10.0, dtype=np.float32)
    node._latest_scan = (ranges, -np.pi, 2 * np.pi / 360)
    node._latest_scan_time = 0.001
    node._latest_scan_receipt_time = time.monotonic()

    from hunter_kinodynamic_rl.navigation.localization.odom_backend import OdomLocalizationBackend
    node._localization = OdomLocalizationBackend()
    node._localization.update(0.0, 0.0, 0.0, node._latest_odom_time, confidence=1.0, valid=True)

    captured = {}
    original = LocalPolicyController.decode_and_guard

    def _spy(self, *args, **kwargs):
        captured.update(kwargs)
        return original(self, *args, **kwargs)

    node._local_controller.decode_and_guard = _spy.__get__(node._local_controller, LocalPolicyController)
    node._on_control_tick()

    assert captured, "decode_and_guard was never called"
    assert captured["last_sensor_time_sec"] == node._latest_scan_receipt_time
    assert captured["last_odom_time_sec"] == node._latest_odom_receipt_time
    assert captured["last_sensor_time_sec"] != node._latest_scan_time
    assert captured["last_odom_time_sec"] != node._latest_odom_time


def test_localization_backend_factory_selects_matching_type():
    for backend_name, expected_type in (
        ("gazebo_odom", GazeboOdomLocalizationBackend),
        ("odom", GazeboOdomLocalizationBackend),
        ("wheel_imu", WheelImuLocalizationBackend),
        ("lio", LioLocalizationBackend),
    ):
        profile = load_profile("hierarchical_phase5_b")
        profile = dataclasses.replace(
            profile, localization=dataclasses.replace(profile.localization, backend=backend_name),
        )
        backend = _build_localization_backend(profile)
        assert isinstance(backend, expected_type), backend_name


def test_localization_backend_factory_rejects_unsupported_backend():
    """Defense in depth: even if a caller bypasses LocalizationConfig.validate()'s
    own rejection, the factory itself must never silently fall back to
    GazeboOdomLocalizationBackend for an unrecognized name."""
    profile = load_profile("hierarchical_phase5_b")
    profile = dataclasses.replace(
        profile, localization=dataclasses.replace(profile.localization, backend="lidar_odom"),
    )
    with pytest.raises(ValueError):
        _build_localization_backend(profile)


def test_wheel_imu_backend_dispatch_dead_reckons_from_odometry_twist():
    """The wheel_imu backend has no on_odometry_msg() of its own -- _on_odom
    must dispatch to initialize()/integrate() instead, using the SAME
    bridged odometry topic's twist as its only live (v, yaw_rate) source in
    this simulation-only repository (documented stand-in, see
    wheel_imu_backend.py)."""
    node = _bare_node()
    node._localization = WheelImuLocalizationBackend()
    node._wheel_imu_initialized = False

    first = _make_odometry_msg(x=0.0, y=0.0, yaw=0.0, stamp_sec=0.0)
    first.twist.twist.linear.x = 1.0
    node._on_odom(first)
    assert node._wheel_imu_initialized
    pose0 = node._localization.latest_pose()
    assert pose0.x == pytest.approx(0.0)

    second = _make_odometry_msg(x=0.0, y=0.0, yaw=0.0, stamp_sec=1.0)
    second.twist.twist.linear.x = 1.0
    node._on_odom(second)
    pose1 = node._localization.latest_pose()
    # 1 m/s for 1s of dead-reckoned integration -- moved forward, NOT still
    # at the origin (confirms integrate() actually ran, not just parsed).
    assert pose1.x == pytest.approx(1.0, abs=1e-6)


def test_localization_valid_detects_staleness_after_odom_receipt_stops():
    """Regression (Medium finding): the pose's own message stamp compared
    against ITSELF (via self._latest_odom_time) never detects staleness --
    _localization_valid() must ALSO check monotonic receipt time, which
    genuinely reflects "how long ago did odom last arrive"."""
    node = _bare_node()
    node._localization = GazeboOdomLocalizationBackend()
    node._localization.update(0.0, 0.0, 0.0, 1.0, confidence=1.0, valid=True)
    node._latest_odom_receipt_time = time.monotonic() - 10.0  # long past pose_timeout_sec
    assert not node._localization_valid()


def test_localization_valid_true_for_fresh_receipt():
    node = _bare_node()
    node._localization = GazeboOdomLocalizationBackend()
    node._localization.update(0.0, 0.0, 0.0, 1.0, confidence=1.0, valid=True)
    node._latest_odom_receipt_time = time.monotonic()
    assert node._localization_valid()


def test_localization_valid_false_when_receipt_never_recorded():
    node = _bare_node()
    node._localization = GazeboOdomLocalizationBackend()
    node._localization.update(0.0, 0.0, 0.0, 1.0, confidence=1.0, valid=True)
    node._latest_odom_receipt_time = None
    assert not node._localization_valid()


def test_effective_localization_confidence_forced_to_zero_when_stale():
    node = _bare_node()
    node._localization = GazeboOdomLocalizationBackend()
    node._localization.update(0.0, 0.0, 0.0, 1.0, confidence=0.9, valid=True)
    node._latest_odom_receipt_time = time.monotonic() - 10.0
    assert node._effective_localization_confidence() == 0.0


def test_effective_localization_confidence_passes_through_when_fresh():
    node = _bare_node()
    node._localization = GazeboOdomLocalizationBackend()
    node._localization.update(0.0, 0.0, 0.0, 1.0, confidence=0.9, valid=True)
    node._latest_odom_receipt_time = time.monotonic()
    assert node._effective_localization_confidence() == pytest.approx(0.9)


def test_maybe_select_next_subgoal_blocks_global_decision_when_odom_receipt_stale():
    """Regression (Medium finding): stale localization must ALSO block
    Global subgoal selection/activation, not just the physical command --
    the receipt-aware effective confidence feeds HierarchyCoordinator's
    EXISTING degraded-localization state machine (never confirms a fresh
    subgoal activation from a stale reading, refuses to activate even a
    just-selected/queued candidate)."""
    node = _bare_node()
    node.mission_frame.initialize(PoseXYYaw(0.0, 0.0, 0.0))
    goal = node.mission_frame.odom_to_mission(10.0, 0.0)
    node.coordinator.start_mission(goal[0], goal[1])
    node._final_goal_mission = goal
    node._localization = GazeboOdomLocalizationBackend()
    node._localization.update(0.0, 0.0, 0.0, 1.0, confidence=1.0, valid=True)
    node._latest_odom_receipt_time = time.monotonic() - 10.0  # stale
    node._latest_odom_time = 1.0

    activated = node._maybe_select_next_subgoal(PoseXYYaw(0.0, 0.0, 0.0))
    assert not activated
    assert node.coordinator.localization_degraded
    assert node.coordinator.active_subgoal_mission is None


def test_maybe_select_next_subgoal_activates_normally_when_localization_fresh():
    """Control case for the previous test -- confirms the fix did not
    over-block normal operation when localization genuinely is fresh."""
    node = _bare_node()
    node.mission_frame.initialize(PoseXYYaw(0.0, 0.0, 0.0))
    goal = node.mission_frame.odom_to_mission(10.0, 0.0)
    node.coordinator.start_mission(goal[0], goal[1])
    node._final_goal_mission = goal
    node._localization = GazeboOdomLocalizationBackend()
    node._localization.update(0.0, 0.0, 0.0, 1.0, confidence=1.0, valid=True)
    node._latest_odom_receipt_time = time.monotonic()
    node._latest_odom_time = 1.0

    activated = node._maybe_select_next_subgoal(PoseXYYaw(0.0, 0.0, 0.0))
    assert activated
    assert not node.coordinator.localization_degraded
    assert node.coordinator.active_subgoal_mission is not None


def test_maybe_select_next_subgoal_never_enqueues_while_stale_across_repeated_ticks():
    """Regression: a stale-localization Global "decision" every tick used
    to still build an observation, select a candidate, and
    coordinator.enqueue_subgoal() it -- activate_next_subgoal() refuses to
    POP while degraded, so nothing ever drained that queue, and stale
    candidates piled up one per tick for as long as odom stayed silent.
    Calling _maybe_select_next_subgoal() repeatedly while stale must leave
    pending_subgoal_count at 0 the entire time."""
    node = _bare_node()
    node.mission_frame.initialize(PoseXYYaw(0.0, 0.0, 0.0))
    goal = node.mission_frame.odom_to_mission(10.0, 0.0)
    node.coordinator.start_mission(goal[0], goal[1])
    node._final_goal_mission = goal
    node._localization = GazeboOdomLocalizationBackend()
    node._localization.update(0.0, 0.0, 0.0, 1.0, confidence=1.0, valid=True)
    node._latest_odom_receipt_time = time.monotonic() - 10.0  # stale
    node._latest_odom_time = 1.0

    for _ in range(5):
        activated = node._maybe_select_next_subgoal(PoseXYYaw(0.0, 0.0, 0.0))
        assert not activated
        assert node.coordinator.pending_subgoal_count == 0
    assert node.coordinator.localization_degraded
    assert node.coordinator.active_subgoal_mission is None


def test_maybe_select_next_subgoal_activates_a_fresh_candidate_after_recovery_never_a_stale_one():
    """Companion to the above: after several stale ticks (queue stays
    empty throughout, per the previous test), recovering localization must
    activate a subgoal computed from the CURRENT pose -- there is no
    leftover stale candidate sitting in the queue to be popped first."""
    node = _bare_node()
    node.mission_frame.initialize(PoseXYYaw(0.0, 0.0, 0.0))
    goal = node.mission_frame.odom_to_mission(10.0, 0.0)
    node.coordinator.start_mission(goal[0], goal[1])
    node._final_goal_mission = goal
    node._localization = GazeboOdomLocalizationBackend()
    node._localization.update(0.0, 0.0, 0.0, 1.0, confidence=1.0, valid=True)

    # Several stale ticks first.
    node._latest_odom_receipt_time = time.monotonic() - 10.0
    node._latest_odom_time = 1.0
    for _ in range(3):
        assert not node._maybe_select_next_subgoal(PoseXYYaw(0.0, 0.0, 0.0))
    assert node.coordinator.pending_subgoal_count == 0
    assert node.coordinator.localization_degraded

    # Localization recovers.
    node._latest_odom_receipt_time = time.monotonic()
    activated = node._maybe_select_next_subgoal(PoseXYYaw(0.0, 0.0, 0.0))
    assert activated
    assert not node.coordinator.localization_degraded
    assert node.coordinator.active_subgoal_mission is not None
    assert node.coordinator.pending_subgoal_count == 0


def test_dry_run_never_publishes_any_command_including_nonzero():
    """Requirement J: dry-run/rosbag-replay mode must structurally block
    ALL actuator command publishes, not just STOP -- proves it by driving a
    genuinely nonzero (non-stop) command through ``_publish()`` while
    ``dry_run=True`` and asserting the fake publisher NEVER receives
    anything, regardless of command content."""
    from hunter_kinodynamic_rl.trajectory.pure_pursuit_adapter import VehicleCommand

    node = _bare_node(dry_run=True)
    nonzero_command = VehicleCommand(speed_mps=1.5, steering_rad=0.2)
    for _ in range(5):
        node._publish(nonzero_command)
    assert node._cmd_pub.published == []


def test_dry_run_false_does_publish_a_real_command():
    """Companion to the above -- confirms the fake publisher itself
    actually records a publish when NOT in dry-run (so the previous test's
    empty list is a genuine block, not just an inert fake)."""
    from hunter_kinodynamic_rl.trajectory.pure_pursuit_adapter import VehicleCommand

    node = _bare_node(dry_run=False)
    node._publish(VehicleCommand(speed_mps=1.0, steering_rad=0.1))
    assert len(node._cmd_pub.published) == 1


def test_replay_mode_requires_dry_run_true_in_bare_construction_contract():
    """The real __init__ raises SystemExit for replay_mode=True with
    dry_run=False (plan 10.7) -- this test documents/locks that contract
    at the source level (the bare-instance harness bypasses __init__
    entirely, so it cannot exercise the raise itself)."""
    import inspect

    from hunter_kinodynamic_rl.nodes import hierarchical_navigation_node as module
    source = inspect.getsource(module.HierarchicalNavigationNode.__init__)
    assert "replay_mode and not self.dry_run" in source


def test_hierarchical_navigation_node_reuses_action_guard_never_reimplements_it():
    """Structural check: the control tick must route through
    LocalPolicyController.decode_and_guard (== env.safety.action_guard.guard()),
    never a bespoke safety check -- verified by confirming the node module
    never imports/defines a second sanitize/guard function of its own."""
    import inspect

    from hunter_kinodynamic_rl.nodes import hierarchical_navigation_node as module
    source = inspect.getsource(module)
    assert "def guard(" not in source
    assert "def sanitize_command(" not in source
