#!/usr/bin/env python3
"""``ros2 run hunter_kinodynamic_rl hierarchical_navigation_node.py`` --
Phase 5/6 real-Hunter / rosbag-dry-run hierarchical mission node (plan
section 10.6/10.7/24/25).

Safety-hardened successor to ``hierarchical_environment_node.py`` (Phase
4's live-Gazebo wiring, which never grew the real-hardware safety layer):
this node adds the SAME watchdog/E-stop/timeout-boxed-inference/dry-run
pattern ``real_policy_node.py`` already uses for the local-only path, driving
``HierarchyCoordinator`` + ``GlobalDQNAgent``/heuristic + ``LocalPolicyController``
instead of a single flat local policy. Every physical-command publish goes
through ``env.safety.action_guard.guard()`` -- never bypassed.

Responsibilities (plan 10.6): start/health-check the localization backend,
initialize the mission frame from the first usable pose, accept a relative
final goal, initialize partial/visited/topological memory, run Global
inference + action mask, run Local inference + safety guard, publish
mission/subgoal status, stop/recover on degraded localization, and enforce
policy/command timeouts, invalid-subgoal rejection, proximity e-stop, and
dry-run/replay actuation blocking. It deliberately never uses full
ground-truth world data, a simulator obstacle list, or benchmark-only
privileged labels (plan 10.6's negative list) -- only the ONLINE
``PartialMap``/``TopologicalGraph`` this node itself accumulates.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np
import rclpy
import tf2_ros
from geometry_msgs.msg import PointStamped, TransformStamped, Twist
from nav_msgs.msg import Odometry, OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState, LaserScan
from std_msgs.msg import Bool

from hunter_kinodynamic_rl.config.loader import load_profile
from hunter_kinodynamic_rl.config.schema import Profile
from hunter_kinodynamic_rl.env.safety.action_guard import STOP_COMMAND, SafetyLimits
from hunter_kinodynamic_rl.navigation.global_rl.action_mask import compute_action_mask
from hunter_kinodynamic_rl.navigation.global_rl.observation import build_global_observation, resolve_map_channel_names
from hunter_kinodynamic_rl.navigation.global_rl.subgoal_sampler import build_candidate_set, candidate_endpoint_mission
from hunter_kinodynamic_rl.navigation.hierarchy.coordinator import HierarchyCoordinator
from hunter_kinodynamic_rl.navigation.hierarchy.feasibility import compute_feasibility_features
from hunter_kinodynamic_rl.navigation.hierarchy.local_feasibility_evaluator import (
    FeasibilityEvaluatorTelemetry, FrozenLocalFeasibilityEvaluator, LocalSensorSnapshot,
)
from hunter_kinodynamic_rl.navigation.local_rl.controller import LocalPolicyController
from hunter_kinodynamic_rl.navigation.localization.gazebo_odom_backend import GazeboOdomLocalizationBackend
from hunter_kinodynamic_rl.navigation.localization.interface import LocalizationBackend, is_pose_usable
from hunter_kinodynamic_rl.navigation.localization.lio_adapter import LioLocalizationBackend
from hunter_kinodynamic_rl.navigation.localization.wheel_imu_backend import WheelImuLocalizationBackend
from hunter_kinodynamic_rl.navigation.mapping.partial_map import PartialMap
from hunter_kinodynamic_rl.navigation.memory.dead_end_detector import DeadEndDetector
from hunter_kinodynamic_rl.navigation.memory.node_manager import TopologicalNodeManager, compute_candidate_topology_features
from hunter_kinodynamic_rl.navigation.memory.route_history import RouteHistory
from hunter_kinodynamic_rl.navigation.memory.topological_graph import TopologicalGraph
from hunter_kinodynamic_rl.navigation.mission.mission_frame import MissionFrame, PoseXYYaw
from hunter_kinodynamic_rl.navigation.ros.mission_map_node import build_occupancy_grid_msg, channel_to_occupancy_data, \
    yaw_to_quaternion
from hunter_kinodynamic_rl.rl.algorithms.kinodynamic_tqc.agent import Agent as RiskAgent
from hunter_kinodynamic_rl.rl.algorithms.tqc.agent import Agent as VanillaAgent
from hunter_kinodynamic_rl.rl.checkpointing import manager as ckpt_manager
from hunter_kinodynamic_rl.robot.limits import wheel_angles_to_center_steering
from hunter_kinodynamic_rl.trajectory.action_space import ACTION_DIM
from hunter_kinodynamic_rl.training.train_hierarchical_dqn import (
    feasibility_config_from, hierarchy_config_from, memory_config_from,
)

SENSOR_QOS = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
LATCHED_QOS = QoSProfile(
    depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


def _twist_from(command) -> Twist:
    msg = Twist()
    msg.linear.x = command.speed_mps
    msg.angular.z = command.steering_rad
    return msg


def _heuristic_select_action(candidates, obs) -> int:
    """Same goal-seeking fallback ``hierarchical_environment_node.py`` uses
    when no trained Global checkpoint is loaded -- kept identical so this
    node is runnable end to end without one."""
    goal_bearing = obs.scalar_tensor[1] * math.pi
    best_index, best_score = None, None
    for candidate in candidates:
        if not obs.action_mask[candidate.index]:
            continue
        score = abs(math.atan2(math.sin(candidate.angle_rad - goal_bearing), math.cos(candidate.angle_rad - goal_bearing)))
        if candidate.is_fallback:
            score += math.pi
        if best_score is None or score < best_score:
            best_index, best_score = candidate.index, score
    return best_index


def _build_localization_backend(profile: Profile) -> LocalizationBackend:
    """Constructs the localization backend named by
    ``profile.localization.backend`` -- plan section 10.6's "localization
    backend를 교체해도 navigation core를 수정하지 않는다" requirement, made
    real for this node rather than always hardcoding
    ``GazeboOdomLocalizationBackend`` regardless of config. A pure factory
    (no ROS I/O of its own) so it is directly unit-testable; the CALLER
    (``__init__``/``_on_odom``) is responsible for wiring the right topic
    subscription and feeding-message dispatch for whichever concrete type
    comes back -- see ``SUPPORTED_LOCALIZATION_BACKENDS`` in
    ``config/schema.py`` for which backends this repository can actually
    feed from a live topic today."""
    backend_name = profile.localization.backend
    if backend_name in ("gazebo_odom", "odom"):
        return GazeboOdomLocalizationBackend(use_covariance_confidence=(backend_name == "gazebo_odom"))
    if backend_name == "wheel_imu":
        return WheelImuLocalizationBackend()
    if backend_name == "lio":
        return LioLocalizationBackend()
    raise ValueError(
        f"hierarchical_navigation_node: unsupported localization.backend {backend_name!r} -- see "
        "config.schema.SUPPORTED_LOCALIZATION_BACKENDS"
    )


class HierarchicalNavigationNode(Node):
    def __init__(self, **node_kwargs) -> None:
        super().__init__("hierarchical_navigation_node", **node_kwargs)

        self.declare_parameter("profile", "hierarchical_phase4")
        self.declare_parameter("local_checkpoint_dir", "")
        self.declare_parameter("local_checkpoint_name", "")
        self.declare_parameter("global_checkpoint_dir", "")
        self.declare_parameter("global_checkpoint_name", "final")
        self.declare_parameter("goal_x", 10.0)
        self.declare_parameter("goal_y", 0.0)
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("odom_topic", "")
        # Only consulted when localization.backend == "lio" -- LIO-SAM
        # publishes its own odometry on a topic DIFFERENT from Gazebo's
        # raw /odometry (the sibling LIO-SAM package's typical mapping
        # output topic name; override if the launch configures a
        # different one).
        self.declare_parameter("lio_odom_topic", "/lio_sam/mapping/odometry")
        self.declare_parameter("joint_states_topic", "/hunter_se/joint_states")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("estop_topic", "/hunter_kinodynamic_rl/hierarchical_navigation/estop")
        self.declare_parameter("publish_period_sec", 1.0)
        self.declare_parameter("mission_frame_id", "mission")
        self.declare_parameter("odom_frame_id", "odom")
        self.declare_parameter("policy_inference_timeout_sec", 0.5)
        # plan 10.7: "dry-run/replay mode에서 actuation 완전 차단". dry_run
        # runs the FULL pipeline (mission, mapping, memory, Global/Local
        # inference, safety guard, diagnostics) every tick but never
        # publishes to cmd_vel_topic -- for validating the pipeline against
        # live sensors/a live robot without ever being able to move it.
        # replay_mode REQUIRES dry_run=true (raises at startup otherwise):
        # replaying recorded data must never be able to actuate a real robot.
        self.declare_parameter("dry_run", False)
        self.declare_parameter("replay_mode", False)

        # Cached once (never re-read from get_parameter() per odom message,
        # both to avoid the small per-message parameter-service lookup and
        # so a bare-constructed test instance can set these two plain
        # attributes directly without needing a live rclpy parameter
        # context). goal_x/goal_y are the mission's user-supplied relative
        # goal in the robot's OWN frame at mission start (see the
        # mission-frame-fixing comment in _on_odom below).
        self._goal_x = float(self.get_parameter("goal_x").value)
        self._goal_y = float(self.get_parameter("goal_y").value)

        self.profile = load_profile(self.get_parameter("profile").value)
        if not self.profile.global_rl.enabled:
            raise RuntimeError(
                f"profile {self.profile.name!r} does not enable global_rl -- "
                "hierarchical_navigation_node requires a hierarchical profile"
            )
        self.dry_run = bool(self.get_parameter("dry_run").value)
        self.replay_mode = bool(self.get_parameter("replay_mode").value)
        if self.replay_mode and not self.dry_run:
            raise SystemExit(
                "-p replay_mode:=true requires -p dry_run:=true -- replaying recorded data must never be able "
                "to actuate a real robot (plan section 10.7)"
            )

        odom_topic = self.get_parameter("odom_topic").value or self.profile.localization.odom_topic
        self._mission_frame_id = self.get_parameter("mission_frame_id").value
        self._odom_frame_id = self.get_parameter("odom_frame_id").value

        self.mission_frame = MissionFrame()
        self.partial_map = PartialMap(self.profile.mapping)
        self.coordinator = HierarchyCoordinator(hierarchy_config_from(self.profile.hierarchy))
        self.candidates = build_candidate_set(self.profile.global_rl)
        # plan 10.6: "localization backend를 교체해도 navigation core를
        # 수정하지 않는다" -- this node only ever calls
        # latest_pose()/pose_at() through the LocalizationBackend Protocol
        # everywhere else; _build_localization_backend() is the ONE place
        # that knows which concrete type profile.localization.backend
        # names (see SUPPORTED_LOCALIZATION_BACKENDS in config/schema.py).
        self._localization = _build_localization_backend(self.profile)
        # wheel_imu dead-reckons from a local (0,0,0) origin with no
        # separate "initialize from a real pose" step of its own (unlike
        # gazebo_odom/lio, which parse an absolute pose straight out of
        # their message) -- lazily initialized on the FIRST odom message,
        # tracked here since WheelImuLocalizationBackend itself has no
        # "am I initialized yet" query.
        self._wheel_imu_initialized = False
        self._local_controller = LocalPolicyController(self.profile)

        self.memory_enabled = bool(self.profile.memory.enabled and self.profile.global_rl.topology_feedback_enabled)
        self.feasibility_enabled = bool(
            self.profile.feasibility.enabled and self.profile.global_rl.feasibility_feedback_enabled
        )
        self.global_risk_enabled = bool(
            self.profile.feasibility.enabled and self.profile.global_rl.global_risk_feedback_enabled
        )
        self.graph: Optional[TopologicalGraph] = None
        self.node_manager: Optional[TopologicalNodeManager] = None
        self.dead_end_detector: Optional[DeadEndDetector] = None
        if self.memory_enabled:
            self.graph = TopologicalGraph(max_nodes=self.profile.memory.max_nodes)
            node_manager_cfg, dead_end_cfg = memory_config_from(self.profile.memory)
            self.node_manager = TopologicalNodeManager(self.graph, RouteHistory(), node_manager_cfg)
            self.dead_end_detector = DeadEndDetector(dead_end_cfg)
        self.feasibility_config = (
            feasibility_config_from(self.profile.feasibility)
            if (self.feasibility_enabled or self.global_risk_enabled) else None
        )
        self.max_nodes = self.profile.memory.max_nodes_in_observation if self.memory_enabled else 0

        history_len = self.profile.observation.frame_stack if self.profile.features.temporal_context else 1
        state_dim = self.profile.observation.lidar_bins * history_len + self.profile.observation.robot_state_dim
        if self.profile.features.risk_critic:
            self.local_agent = RiskAgent(state_dim, ACTION_DIM, 1.0, self.profile.hyperparameters,
                                          self.profile.risk, self.profile.counterfactual)
        else:
            self.local_agent = VanillaAgent(state_dim, ACTION_DIM, 1.0, self.profile.hyperparameters)
        local_ckpt_dir = (
            self.get_parameter("local_checkpoint_dir").value
            or self.profile.hierarchical_training.local_checkpoint_dir
        )
        local_ckpt_name = (
            self.get_parameter("local_checkpoint_name").value
            or self.profile.hierarchical_training.local_checkpoint_name
        )
        if local_ckpt_dir:
            result = ckpt_manager.load_generation(
                local_ckpt_dir, local_ckpt_name, self.local_agent.checkpoint_components(),
                map_location=str(self.local_agent.device),
            )
            self.get_logger().info(f"loaded local checkpoint: {result['loaded']} (skipped: {result['skipped']})")
        else:
            self.get_logger().warn("no local_checkpoint_dir provided -- local agent is UNTRAINED")

        self.global_agent = None
        global_ckpt_dir = self.get_parameter("global_checkpoint_dir").value
        if global_ckpt_dir:
            from hunter_kinodynamic_rl.navigation.global_rl.agent import GlobalDQNAgent
            map_channels = len(resolve_map_channel_names(self.profile.global_rl))
            self.global_agent = GlobalDQNAgent(self.profile.global_rl, map_channels, device="cpu", max_nodes=self.max_nodes)
            result = ckpt_manager.load_generation(
                global_ckpt_dir, self.get_parameter("global_checkpoint_name").value,
                self.global_agent.checkpoint_components(), map_location="cpu",
            )
            self.get_logger().info(f"loaded Global checkpoint: {result['loaded']} (skipped: {result['skipped']})")
        else:
            self.get_logger().warn(
                "no global_checkpoint_dir provided -- using the built-in goal-seeking heuristic for subgoal "
                "selection, never a trained Global policy"
            )

        self._safety_limits = SafetyLimits(
            max_sensor_age_sec=self.profile.runtime.sensor_freshness_timeout_sec,
            max_odom_age_sec=self.profile.runtime.sensor_freshness_timeout_sec,
            max_command_age_sec=self.profile.runtime.watchdog_command_timeout_sec,
            min_obstacle_stop_distance_m=self.profile.risk.min_safe_clearance_m,
        )
        self._policy_inference_timeout_sec = float(self.get_parameter("policy_inference_timeout_sec").value)

        self._latest_scan: Optional[Tuple[np.ndarray, float, float]] = None
        # Message header stamp -- SAME clock domain as the localization
        # backend's own PoseEstimate.stamp_sec (Gazebo-bridged sim time,
        # typically), used ONLY for stamp-synchronized lookups
        # (pose_at()/is_pose_usable() below, mirroring
        # navigation/ros/mission_map_node.py's own "never mix this with
        # wall-clock" rule -- see that module's docstring).
        self._latest_scan_time: Optional[float] = None
        # time.monotonic() at RECEIPT -- a SEPARATE clock domain, used ONLY
        # for the real-time safety-guard freshness checks below (mirrors
        # real_policy_node.py's own _latest_scan_time convention exactly).
        # Deliberately never fed into is_pose_usable()/pose_at() (those
        # need the message-stamp domain instead) and never compared
        # against self._latest_scan_time (that would itself be the exact
        # clock-domain-mixing bug this split exists to prevent).
        self._latest_scan_receipt_time: Optional[float] = None
        self._latest_odom: Optional[Tuple[float, float, float, float, float]] = None
        self._latest_odom_time: Optional[float] = None
        self._latest_odom_receipt_time: Optional[float] = None
        self._latest_steering_rad = 0.0
        self._last_command_time: Optional[float] = None
        self._last_successful_inference_time: Optional[float] = None
        self._speed_history: Deque[Tuple[float, float]] = deque(maxlen=50)
        self._step_counter = 0
        self._rng = np.random.RandomState(0)
        self._final_goal_mission: Optional[Tuple[float, float]] = None
        self._estopped = False
        self._inference_thread: Optional[threading.Thread] = None

        # Requirement E: the real LocalFeasibilityEvaluator production
        # path. Constructed once, bound to THIS node's own frozen
        # self.local_agent + a snapshot_provider reading its own live
        # sensor state -- never None whenever feasibility/global-risk
        # feedback is enabled, so compute_feasibility_features() below is
        # never called with local_evaluator=None for a profile that
        # declared it needs real predicted_action_risk/progress_preserving.
        self.feasibility_evaluator_telemetry = FeasibilityEvaluatorTelemetry()
        self.local_evaluator: Optional[FrozenLocalFeasibilityEvaluator] = None
        if self.feasibility_enabled or self.global_risk_enabled:
            self.local_evaluator = FrozenLocalFeasibilityEvaluator(
                self.profile, self.local_agent,
                inference_timeout_sec=self.profile.feasibility.local_evaluator_inference_timeout_sec,
                max_snapshot_age_sec=self.profile.feasibility.local_evaluator_max_snapshot_age_sec,
                snapshot_provider=self._local_feasibility_snapshot,
                telemetry=self.feasibility_evaluator_telemetry,
            )

        self._tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self._mission_map_pub = self.create_publisher(OccupancyGrid, "/mission_map", LATCHED_QOS)
        self._goal_pub = self.create_publisher(PointStamped, "/mission_goal", LATCHED_QOS)
        self._cmd_pub = self.create_publisher(Twist, self.get_parameter("cmd_vel_topic").value, 10)

        # "lio" reads a DIFFERENT topic (LIO-SAM's own odometry output,
        # never Gazebo's raw /odometry) -- every other supported backend
        # feeds from odom_topic. Same message type (nav_msgs/Odometry),
        # same callback -- _on_odom() dispatches on self._localization's
        # concrete type, not on which topic it happened to arrive from.
        subscribed_odom_topic = (
            self.get_parameter("lio_odom_topic").value
            if isinstance(self._localization, LioLocalizationBackend) else odom_topic
        )
        self.create_subscription(Odometry, subscribed_odom_topic, self._on_odom, SENSOR_QOS)
        self.create_subscription(LaserScan, self.get_parameter("scan_topic").value, self._on_scan, SENSOR_QOS)
        self.create_subscription(JointState, self.get_parameter("joint_states_topic").value,
                                  self._on_joint_states, SENSOR_QOS)
        self.create_subscription(Bool, self.get_parameter("estop_topic").value, self._on_estop, 10)
        self.create_timer(self.profile.runtime.time_delta_sec, self._on_control_tick)
        self.create_timer(float(self.get_parameter("publish_period_sec").value), self._publish_visualization)

        self.get_logger().info(
            f"hierarchical_navigation_node started: profile={self.profile.name}, dry_run={self.dry_run}, "
            f"replay_mode={self.replay_mode}"
        )
        # Started LAST (mirrors real_policy_node.py's own ordering
        # rationale exactly): a plain Python thread outside rclpy's
        # executor/callback-group machinery, so it keeps re-publishing
        # STOP_COMMAND even if _on_control_tick itself is hung.
        self._start_watchdog_thread()

    # ------------------------------------------------------------------ safety
    def _on_estop(self, msg: Bool) -> None:
        self._estopped = bool(msg.data)
        if self._estopped:
            self.get_logger().warn("[hierarchical_navigation_node] E-STOP engaged")

    def _publish(self, command) -> None:
        """The ONE call site that reaches ``cmd_vel_topic`` -- ``dry_run``
        (and therefore ``replay_mode``) short-circuits HERE, never earlier,
        so every other tick of the pipeline still runs for pipeline
        validation without being able to move a real robot (plan 10.7)."""
        if not self.dry_run:
            self._cmd_pub.publish(_twist_from(command))

    def _start_watchdog_thread(self) -> None:
        self._watchdog_stop_event = threading.Event()

        def _loop() -> None:
            period = self.profile.runtime.real_policy_watchdog_period_sec
            timeout = self.profile.runtime.watchdog_command_timeout_sec
            while not self._watchdog_stop_event.wait(period):
                now = time.monotonic()
                if self._last_command_time is not None and (now - self._last_command_time) > timeout:
                    self._publish(STOP_COMMAND)

        self._watchdog_thread = threading.Thread(target=_loop, name="hierarchical_navigation_watchdog", daemon=True)
        self._watchdog_thread.start()

    def destroy_node(self) -> None:
        stop_event = getattr(self, "_watchdog_stop_event", None)
        if stop_event is not None:
            stop_event.set()
        super().destroy_node()

    def _infer_local_action_with_timeout(self, observation: np.ndarray) -> Optional[np.ndarray]:
        """Single-flight, thread-bounded local-policy inference -- mirrors
        ``real_policy_node.py``'s ``_infer_with_timeout_thread`` exactly
        (same documented CPython limitation: a genuinely hung call is not
        forcibly killed, but THIS tick gives up waiting and the node stays
        responsive)."""
        if self._inference_thread is not None and self._inference_thread.is_alive():
            self.get_logger().error(
                "[hierarchical_navigation_node] previous local inference still running -- safe stop this tick"
            )
            return None
        result: dict = {}

        def _worker() -> None:
            try:
                result["action"] = self.local_agent.select_action(observation, deterministic=True)
            except Exception as e:  # noqa: BLE001 -- must never propagate into the timer callback
                result["error"] = e

        thread = threading.Thread(target=_worker, daemon=True)
        self._inference_thread = thread
        thread.start()
        thread.join(self._policy_inference_timeout_sec)
        if thread.is_alive():
            self.get_logger().error(
                f"[hierarchical_navigation_node] local inference exceeded {self._policy_inference_timeout_sec:.2f}s"
            )
            return None
        self._inference_thread = None
        if "error" in result:
            self.get_logger().error(f"[hierarchical_navigation_node] local inference raised: {result['error']!r}")
            return None
        return result.get("action")

    # ------------------------------------------------------------------ sensor callbacks
    def _on_odom(self, msg: Odometry) -> None:
        self._latest_odom_receipt_time = time.monotonic()
        stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if isinstance(self._localization, WheelImuLocalizationBackend):
            # No live wheel-encoder/IMU driver exists in this simulation-
            # only repository (see wheel_imu_backend.py's own docstring) --
            # the bridged odometry topic's own twist is the only live
            # (v, yaw_rate) source available, used as a documented,
            # honest stand-in. Dead-reckoning has no absolute frame, so it
            # is seeded at a LOCAL (0,0,0) origin -- consistent with
            # MissionFrame's own contract (the mission frame IS the
            # robot's pose at the moment localization first becomes
            # usable, never an externally known absolute position).
            if not self._wheel_imu_initialized:
                self._localization.initialize(0.0, 0.0, 0.0, stamp_sec)
                self._wheel_imu_initialized = True
            else:
                previous_stamp = self._latest_odom_time if self._latest_odom_time is not None else stamp_sec
                dt_sec = max(0.0, stamp_sec - previous_stamp)
                self._localization.integrate(msg.twist.twist.linear.x, msg.twist.twist.angular.z, dt_sec, stamp_sec)
        elif isinstance(self._localization, LioLocalizationBackend):
            self._localization.on_lio_odometry_msg(msg, stamp_sec)
        else:
            self._localization.on_odometry_msg(msg, stamp_sec)
        v = msg.twist.twist.linear.x
        yaw_rate = msg.twist.twist.angular.z
        speed_mps = math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)
        if not math.isfinite(speed_mps):
            speed_mps = float("inf")
        self._speed_history.append((stamp_sec, speed_mps))
        pose = self._localization.latest_pose()
        self._latest_odom = (pose.x, pose.y, pose.yaw, v, yaw_rate)
        self._latest_odom_time = stamp_sec

        if not self.mission_frame.initialized and is_pose_usable(
            pose, now_sec=pose.stamp_sec, timeout_sec=self.profile.localization.pose_timeout_sec,
            min_confidence=self.profile.localization.minimum_confidence,
        ):
            self.mission_frame.initialize(PoseXYYaw(x=pose.x, y=pose.y, yaw=pose.yaw))
            # plan section 3.1 / GoalManager's own contract
            # (navigation/mission/goal_manager.py's module docstring):
            # goal_x/goal_y are the user's relative goal in the ROBOT'S OWN
            # frame AT MISSION START -- by construction that IS the
            # MissionFrame origin, so it is already numerically identical
            # to the mission-frame goal. It is never an odom-frame point,
            # so mission_frame.odom_to_mission() must NOT be applied here
            # (that transform is for converting an ODOM-frame point into
            # mission frame) -- doing so previously rotated/translated the
            # goal by the start pose for any non-zero start pose/yaw,
            # silently breaking the final goal for every mission except one
            # starting at (0,0,0). Matches mission_map_node.py's own
            # goal_manager.set_goal(goal_x, goal_y) call exactly.
            self._final_goal_mission = (self._goal_x, self._goal_y)
            self.coordinator.start_mission(self._final_goal_mission[0], self._final_goal_mission[1])
            self.get_logger().info(f"mission frame initialized; final_goal_mission={self._final_goal_mission}")

    def _on_joint_states(self, msg: JointState) -> None:
        try:
            left = float(msg.position[msg.name.index("front_left_steering")])
            right = float(msg.position[msg.name.index("front_right_steering")])
        except (ValueError, IndexError, TypeError):
            return
        self._latest_steering_rad = wheel_angles_to_center_steering(
            left, right,
            self.profile.robot.wheelbase_m,
            self.profile.robot.track_width_m,
        )

    def _on_scan(self, msg: LaserScan) -> None:
        self._latest_scan_receipt_time = time.monotonic()
        self._latest_scan = (np.asarray(msg.ranges, dtype=np.float64), msg.angle_min, msg.angle_increment)
        self._latest_scan_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if not self.mission_frame.initialized:
            return
        pose = self._localization.pose_at(self._latest_scan_time, max_dt_sec=self.profile.localization.pose_timeout_sec)
        if not is_pose_usable(
            pose, now_sec=self._latest_scan_time, timeout_sec=self.profile.localization.pose_timeout_sec,
            min_confidence=self.profile.localization.minimum_confidence,
        ):
            return
        robot_pose_mission = self.mission_frame.odom_pose_to_mission(PoseXYYaw(x=pose.x, y=pose.y, yaw=pose.yaw))
        n = len(msg.ranges)
        if n == 0:
            return
        beam_yaws_robot = msg.angle_min + msg.angle_increment * np.arange(n, dtype=np.float64)
        beam_angles_mission = robot_pose_mission.yaw + beam_yaws_robot
        self.partial_map.integrate_scan(
            (robot_pose_mission.x, robot_pose_mission.y), beam_angles_mission,
            np.asarray(msg.ranges, dtype=np.float64), float(msg.range_max), range_min=float(msg.range_min),
        )
        self.partial_map.record_visit(robot_pose_mission.x, robot_pose_mission.y, self._step_counter)
        self._step_counter += 1

    def _speed_at(self, stamp_sec: float) -> float:
        if not self._speed_history:
            return float("inf")
        stamp, speed = min(self._speed_history, key=lambda item: abs(item[0] - stamp_sec))
        if abs(stamp - stamp_sec) > self.profile.localization.pose_timeout_sec or not math.isfinite(speed):
            return float("inf")
        return speed

    def _local_feasibility_snapshot(self) -> Optional[LocalSensorSnapshot]:
        """Requirement E: ``snapshot_provider`` for ``self.local_evaluator``
        -- reads THIS node's own live odom/steering state plus a frozen
        copy of ``self._local_controller``'s CURRENT temporal state
        (defect-fix item 6: the REAL control loop's own LiDAR frame
        history and last committed action, never a second independently-
        accumulated stack) exactly once per call and returns ``None`` (a
        tracked fallback, never a fabricated reading) whenever either the
        temporal context isn't seeded yet or odom hasn't arrived yet.
        Staleness itself is checked by the evaluator against
        ``stamp_monotonic`` (the odom receipt time -- the temporal
        context has no timestamp of its own since it mirrors whatever the
        control loop has already integrated up to this instant)."""
        odom = self._latest_odom
        odom_receipt = self._latest_odom_receipt_time
        scan_receipt = self._latest_scan_receipt_time
        if odom is None or odom_receipt is None or scan_receipt is None:
            return None
        temporal_context = self._local_controller.snapshot_temporal_context()
        if temporal_context is None:
            return None
        _, _, _, v, yaw_rate = odom
        return LocalSensorSnapshot(
            lidar_frame=temporal_context.lidar_frame, prev_action=temporal_context.prev_action,
            speed_mps=v, yaw_rate_rps=yaw_rate, steering_rad=self._latest_steering_rad,
            stamp_monotonic=min(scan_receipt, odom_receipt),
        )

    def _localization_valid(self) -> bool:
        """Regression fix: ``now_sec=self._latest_odom_time`` used to
        compare the pose's own message stamp against ITSELF (both come
        from the same odom message), so this returned True forever once
        ANY odom had ever arrived, even long after odom RECEIPT stopped --
        stale localization silently kept looking "valid" to both the
        Global action mask and the physical-command guard. The
        ``is_pose_usable`` call below still only ever compares two values
        in the SAME (message-stamp) clock domain -- its own age check
        stays a no-op self-comparison, deliberately -- but it correctly
        catches ``pose.valid``/non-finite/low-confidence readings. The
        SEPARATE, explicit RECEIPT-time check below (monotonic domain,
        matching ``_latest_odom_receipt_time``'s own domain, never fed
        into ``is_pose_usable``) is what actually detects "odom has not
        been received in real wall-clock time recently"."""
        pose = self._localization.latest_pose()
        if not is_pose_usable(
            pose, now_sec=pose.stamp_sec, timeout_sec=self.profile.localization.pose_timeout_sec,
            min_confidence=self.profile.localization.minimum_confidence,
        ):
            return False
        if self._latest_odom_receipt_time is None:
            return False
        return (time.monotonic() - self._latest_odom_receipt_time) <= self.profile.localization.pose_timeout_sec

    def _effective_localization_confidence(self) -> float:
        """The backend's own reported confidence, FORCED to ``0.0``
        whenever odom hasn't been RECEIVED (real wall-clock terms) within
        ``pose_timeout_sec`` -- a backend's cached ``PoseEstimate.confidence``
        never decays on its own once messages stop arriving entirely
        (``latest_pose()`` just keeps returning the last computed value
        forever), so staleness needs this separate receipt-time check.
        Feeds directly into ``HierarchyCoordinator``'s ALREADY-REVIEWED
        degraded-localization state machine
        (``activate_next_subgoal``/``record_local_tick``'s
        ``localization_confidence`` parameter -- see ``coordinator.py``'s
        own docstring: degraded confidence cancels the active subgoal,
        sets ``stop_required``, and refuses to activate ANY subgoal,
        queued or not, until confidence recovers) -- this is a NEW input
        to that EXISTING gate, never a second parallel one."""
        if not self._localization_valid():
            return 0.0
        return self._localization.latest_pose().confidence

    # ------------------------------------------------------------------ Global decision
    def _maybe_select_next_subgoal(self, robot_pose_mission: PoseXYYaw) -> bool:
        if not self.coordinator.stop_required or self.coordinator.mission_done:
            return not self.coordinator.stop_required
        localization_confidence = self._effective_localization_confidence()
        if not self._localization_valid():
            # Stale localization must block the ENTIRE Global decision
            # (plan's "stale/invalid localization이면 invalid subgoal/
            # replanning/safe stop" contract), not just activation --
            # building an observation/selecting a candidate/enqueuing it
            # from a pose already known to be untrustworthy would
            # otherwise pile a stale candidate into the coordinator's
            # queue every tick: activate_next_subgoal refuses to POP
            # while degraded, so nothing ever drains it, and the OLDEST
            # (most stale) candidate would activate FIFO-first the moment
            # localization recovers. Returning here BEFORE
            # coordinator.enqueue_subgoal() is what actually prevents that
            # -- the call below (queue is always empty at this point; see
            # this method's own drain-per-call invariant) only updates the
            # coordinator's degraded-localization gate/stop_required state,
            # consistent with how record_local_tick() already handles this
            # for an ACTIVE subgoal.
            self.coordinator.activate_next_subgoal(
                robot_pose_mission, now_step=self._step_counter, now_time_sec=time.monotonic(),
                localization_confidence=localization_confidence,
            )
            return False
        mask = compute_action_mask(
            self.candidates, self.partial_map, robot_pose_mission, self.profile.global_rl,
            self.profile.robot.max_curvature, localization_valid=True,
        )
        feasibility_features = None
        if self.feasibility_enabled or self.global_risk_enabled:
            feasibility_features = compute_feasibility_features(
                self.candidates, self.partial_map, robot_pose_mission, self.profile.robot.max_curvature,
                self.feasibility_config, local_evaluator=self.local_evaluator,
            )
        topology_features = None
        node_tensor = None
        node_validity_mask = None
        if self.memory_enabled and self.graph is not None and self.node_manager is not None:
            topology_features = compute_candidate_topology_features(
                self.candidates, self.graph, robot_pose_mission, self.node_manager.config.node_merge_radius_m,
            )
            node_tensor, node_validity_mask = self.graph.to_fixed_tensor(
                self.max_nodes, robot_pose_mission, self._final_goal_mission,
                goal_distance_norm_m=self.profile.global_rl.goal_distance_norm_m,
                recency_norm_steps=self.profile.memory.node_recency_norm_steps, now_step=self._step_counter,
            )
        obs = build_global_observation(
            self.partial_map, robot_pose_mission, self._final_goal_mission, self.candidates, mask,
            speed_mps=self._speed_at(self._latest_odom_time or 0.0), previous_action_index=self.profile.global_rl.fallback_index,
            elapsed_mission_ratio=0.0, config=self.profile.global_rl,
            topology_candidate_features=topology_features,
            feasibility_candidate_features=(feasibility_features if self.feasibility_enabled else None),
            global_risk_candidate_features=(
                feasibility_features[:, 3:4] if (self.global_risk_enabled and feasibility_features is not None) else None
            ),
            node_tensor=node_tensor, node_validity_mask=node_validity_mask,
        )
        if self.global_agent is not None:
            action = self.global_agent.select_action(obs, epsilon=0.0, rng=self._rng)
        else:
            action = _heuristic_select_action(self.candidates, obs)
        candidate = self.candidates[action]
        endpoint_mission = candidate_endpoint_mission(candidate, robot_pose_mission)
        self.coordinator.enqueue_subgoal(*endpoint_mission)

        def _is_valid(x: float, y: float) -> bool:
            cell = self.partial_map.world_to_cell(x, y)
            return True if cell is None else not bool(self.partial_map.channels().inflated[cell])

        activated = self.coordinator.activate_next_subgoal(
            robot_pose_mission, now_step=self._step_counter, now_time_sec=time.monotonic(),
            is_valid=_is_valid, localization_confidence=localization_confidence,
        )
        if self.memory_enabled and self.node_manager is not None:
            self.node_manager.maybe_create_or_update_node(robot_pose_mission, now_step=self._step_counter)
        return activated

    # ------------------------------------------------------------------ control loop
    def _on_control_tick(self) -> None:
        now = time.monotonic()
        if self._estopped:
            self._publish(STOP_COMMAND)
            self._last_command_time = now
            return
        if not self.mission_frame.initialized or self._latest_scan is None or self._latest_odom is None:
            self._publish(STOP_COMMAND)
            self._last_command_time = now
            return
        try:
            x, y, yaw, v, yaw_rate = self._latest_odom
            robot_pose_mission = self.mission_frame.odom_pose_to_mission(PoseXYYaw(x=x, y=y, yaw=yaw))

            if self.coordinator.mission_done:
                self._publish(STOP_COMMAND)
                self._last_command_time = now
                return
            subgoal_active = self._maybe_select_next_subgoal(robot_pose_mission)
            if not subgoal_active or self.coordinator.stop_required:
                self._publish(STOP_COMMAND)
                self._last_command_time = now
                return

            active_subgoal = self.coordinator.active_subgoal_mission
            subgoal_robot = MissionFrame.mission_to_robot(active_subgoal, robot_pose_mission)
            ranges, angle_min, angle_increment = self._latest_scan
            # defect-fix item 1: subgoal_robot is already ROBOT-frame
            # (MissionFrame.mission_to_robot above) -- RobotState's pose
            # MUST be the origin to match, never the odom/mission-frame
            # (x, y, yaw) this tick's odometry reports (see
            # LocalPolicyController.build_observation's docstring).
            robot_state = LocalPolicyController.robot_relative_state(
                v=v, yaw_rate=yaw_rate, steering=self._latest_steering_rad,
            )
            local_obs = self._local_controller.build_observation(
                ranges, angle_min, angle_increment, robot_state, subgoal_robot[0], subgoal_robot[1],
            )
            raw_action = self._infer_local_action_with_timeout(local_obs.observation)
            action_arr = LocalPolicyController.validate_action(raw_action) if raw_action is not None else None
            if action_arr is None:
                self._publish(STOP_COMMAND)
                self._last_command_time = now
                return
            self._last_successful_inference_time = now
            self._local_controller.commit_action(action_arr)
            # action_guard.guard()'s now_sec/last_*_time_sec arguments MUST
            # all share ONE clock domain (its own docstring, and
            # is_pose_usable()'s -- see that function's explicit warning
            # about a real historical bug in this codebase,
            # evaluation/nav2_mppi_runner.py's module docstring). `now`
            # here is time.monotonic() (this tick's own wall-clock
            # reading, matching self._last_command_time), so the sensor/
            # odom freshness inputs must be the MONOTONIC RECEIPT
            # timestamps (self._latest_{scan,odom}_receipt_time) -- never
            # self._latest_scan_time/self._latest_odom_time, which are
            # message header stamps (Gazebo-bridged sim time when running
            # against simulation) and previously got compared directly
            # against this wall-clock `now`, silently producing either a
            # permanent stale-stop or a stale sensor passing freshness
            # depending on how far the two clocks had drifted apart.
            _, safe_command = self._local_controller.decode_and_guard(
                action_arr, current_steering_rad=self._latest_steering_rad, safety_limits=self._safety_limits,
                nearest_obstacle_dist_m=local_obs.nearest_obstacle_dist_m,
                last_sensor_time_sec=self._latest_scan_receipt_time or 0.0,
                last_command_time_sec=self._last_command_time or now,
                now_sec=now, last_odom_time_sec=self._latest_odom_receipt_time,
                localization_valid=self._localization_valid(),
            )
            self.coordinator.record_local_tick(
                robot_pose_mission, now_step=self._step_counter, now_time_sec=now, speed_mps=v,
                clearance_m=local_obs.nearest_obstacle_dist_m,
                localization_confidence=self._effective_localization_confidence(),
            )
        except Exception as e:  # noqa: BLE001 -- last-resort fail-safe
            self.get_logger().error(f"[hierarchical_navigation_node] control tick raised {e!r} -- safe stop")
            self._publish(STOP_COMMAND)
            self._last_command_time = now
            return

        self._publish(safe_command)
        self._last_command_time = now

    # ------------------------------------------------------------------ visualization
    def _publish_visualization(self) -> None:
        if not self.mission_frame.initialized:
            return
        stamp = self.get_clock().now().to_msg()
        origin = self.mission_frame.origin
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = self._odom_frame_id
        t.child_frame_id = self._mission_frame_id
        t.transform.translation.x, t.transform.translation.y = origin.x, origin.y
        qx, qy, qz, qw = yaw_to_quaternion(origin.yaw)
        t.transform.rotation.x, t.transform.rotation.y = qx, qy
        t.transform.rotation.z, t.transform.rotation.w = qz, qw
        self._tf_broadcaster.sendTransform(t)

        channels = self.partial_map.channels()
        data = channel_to_occupancy_data(channels.occupied, channels.free, channels.unknown, channels.observed_uncertain)
        self._mission_map_pub.publish(build_occupancy_grid_msg(
            data, channels.resolution_m, channels.origin_x, channels.origin_y, self.partial_map.size_cells,
            self._mission_frame_id, stamp,
        ))
        if self._final_goal_mission is not None:
            msg = PointStamped()
            msg.header.frame_id = self._mission_frame_id
            msg.header.stamp = stamp
            msg.point.x, msg.point.y = self._final_goal_mission
            self._goal_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = HierarchicalNavigationNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
