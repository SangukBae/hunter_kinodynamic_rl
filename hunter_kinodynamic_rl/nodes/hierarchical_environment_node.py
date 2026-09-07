#!/usr/bin/env python3
"""``ros2 run hunter_kinodynamic_rl hierarchical_environment_node.py`` --
Phase 4 live-Gazebo INFERENCE node adapter (plan section 8, "ROS adapter
skeleton"). Combines Phase 1's ``mission_map_node.py`` (localization +
partial-map accumulation + RViz publishing) with Phase 2's
``LocalPolicyController`` (frozen local kinodynamic TQC inference + safety
guard, mirroring ``real_policy_node.py``'s control-tick shape) and this
phase's ``HierarchyCoordinator`` + Global RL action mask/observation, so a
TRAINED Global checkpoint + a FROZEN local checkpoint can be driven live
against Gazebo end to end.

**Scope boundary (explicit, matches Phase 3's own precedent of not live-
verifying Gazebo this session)**: this node is structurally complete --
every ROS I/O wire-up, coordinate transform, and coordinator/controller call
uses the SAME already-unit-tested APIs the ROS-free training loop
(``training/train_hierarchical_dqn.py``) and ``real_policy_node.py``/
``mission_map_node.py`` use -- but it has NOT been exercised against a live
Gazebo instance. Treat it as the wiring point a future session verifies
live, not as a component this session's Docker test run covers.

INFERENCE ONLY: never trains either policy. The local policy checkpoint is
always frozen; the Global checkpoint (if provided) is loaded once and never
updated by this node -- training happens exclusively via
``training/train_hierarchical_dqn.py`` / ``hierarchical_train_node.py``.
"""

from __future__ import annotations

import math
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

from hunter_kinodynamic_rl.config.loader import load_profile
from hunter_kinodynamic_rl.env.safety.action_guard import STOP_COMMAND, SafetyLimits
from hunter_kinodynamic_rl.navigation.global_rl.action_mask import compute_action_mask
from hunter_kinodynamic_rl.navigation.global_rl.observation import build_global_observation
from hunter_kinodynamic_rl.navigation.global_rl.subgoal_sampler import build_candidate_set, candidate_endpoint_mission
from hunter_kinodynamic_rl.navigation.hierarchy.coordinator import HierarchyCoordinator
from hunter_kinodynamic_rl.navigation.local_rl.controller import LocalPolicyController
from hunter_kinodynamic_rl.navigation.localization.gazebo_odom_backend import GazeboOdomLocalizationBackend
from hunter_kinodynamic_rl.navigation.localization.interface import is_pose_usable
from hunter_kinodynamic_rl.navigation.mapping.partial_map import PartialMap
from hunter_kinodynamic_rl.navigation.mission.mission_frame import MissionFrame, PoseXYYaw
from hunter_kinodynamic_rl.navigation.ros.mission_map_node import (
    build_occupancy_grid_msg, channel_to_occupancy_data, yaw_to_quaternion,
)
from hunter_kinodynamic_rl.rl.algorithms.kinodynamic_tqc.agent import Agent as RiskAgent
from hunter_kinodynamic_rl.rl.algorithms.tqc.agent import Agent as VanillaAgent
from hunter_kinodynamic_rl.rl.checkpointing import manager as ckpt_manager
from hunter_kinodynamic_rl.robot.limits import wheel_angles_to_center_steering
from hunter_kinodynamic_rl.trajectory.action_space import ACTION_DIM
from hunter_kinodynamic_rl.training.train_hierarchical_dqn import hierarchy_config_from

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
    """Fallback Global "policy" used when no ``global_checkpoint_dir`` is
    configured -- picks the VALID candidate whose direction-alignment
    feature (``candidate_tensor[:, 3]`` is curvature difficulty; alignment
    with the goal is better read from the observation's own goal-bearing
    scalar combined with each candidate's angle) points most directly at
    the final goal. Never selects an invalid candidate -- exists purely so
    this node is runnable end to end without a trained Global checkpoint,
    not as a research contribution."""
    goal_bearing = obs.scalar_tensor[1] * math.pi
    best_index, best_score = None, None
    for candidate in candidates:
        if not obs.action_mask[candidate.index]:
            continue
        score = abs(math.atan2(math.sin(candidate.angle_rad - goal_bearing), math.cos(candidate.angle_rad - goal_bearing)))
        if candidate.is_fallback:
            score += math.pi  # last resort, never preferred over any real candidate
        if best_score is None or score < best_score:
            best_index, best_score = candidate.index, score
    return best_index


class HierarchicalEnvironmentNode(Node):
    def __init__(self, **node_kwargs) -> None:
        super().__init__("hierarchical_environment_node", **node_kwargs)

        self.declare_parameter("profile", "hierarchical_phase4")
        self.declare_parameter("local_checkpoint_dir", "")
        self.declare_parameter("local_checkpoint_name", "")
        self.declare_parameter("global_checkpoint_dir", "")
        self.declare_parameter("global_checkpoint_name", "final")
        self.declare_parameter("goal_x", 10.0)
        self.declare_parameter("goal_y", 0.0)
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("odom_topic", "")
        self.declare_parameter("joint_states_topic", "/hunter_se/joint_states")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("publish_period_sec", 1.0)
        self.declare_parameter("mission_frame_id", "mission")
        self.declare_parameter("odom_frame_id", "odom")

        self.profile = load_profile(self.get_parameter("profile").value)
        if not self.profile.global_rl.enabled:
            raise RuntimeError(
                f"profile {self.profile.name!r} does not enable global_rl -- "
                "hierarchical_environment_node requires a Phase 4 profile"
            )
        if (
            self.profile.global_rl.topology_feedback_enabled or self.profile.global_rl.feasibility_feedback_enabled
            or self.profile.global_rl.global_risk_feedback_enabled
        ):
            raise RuntimeError(
                f"profile {self.profile.name!r} enables a Phase 5 ablation flag (topology/feasibility/"
                "global-risk feedback) -- this node never computes those candidate features (it is Phase 4's "
                "live-Gazebo INFERENCE adapter, kept scoped to the Phase 4 MVP observation contract). Use "
                "nodes/hierarchical_navigation_node.py for Phase 5 profiles."
            )

        odom_topic = self.get_parameter("odom_topic").value or self.profile.localization.odom_topic
        self._mission_frame_id = self.get_parameter("mission_frame_id").value
        self._odom_frame_id = self.get_parameter("odom_frame_id").value

        self.mission_frame = MissionFrame()
        self.partial_map = PartialMap(self.profile.mapping)
        self.coordinator = HierarchyCoordinator(hierarchy_config_from(self.profile.hierarchy))
        self.candidates = build_candidate_set(self.profile.global_rl)
        self._localization = GazeboOdomLocalizationBackend(
            use_covariance_confidence=(self.profile.localization.backend == "gazebo_odom"),
        )
        self._local_controller = LocalPolicyController(self.profile)

        history_len = self.profile.observation.frame_stack if self.profile.features.temporal_context else 1
        state_dim = self.profile.observation.lidar_bins * history_len + self.profile.observation.robot_state_dim
        if self.profile.features.risk_critic:
            self.local_agent = RiskAgent(state_dim, ACTION_DIM, 1.0, self.profile.hyperparameters,
                                          self.profile.risk, self.profile.counterfactual)
        else:
            self.local_agent = VanillaAgent(state_dim, ACTION_DIM, 1.0, self.profile.hyperparameters)
        # (directory, tag) pair -- NEVER a combined file path -- matching
        # HierarchicalTrainingConfig.local_checkpoint_dir/local_checkpoint_name's
        # own contract (see that field's docstring for why a single
        # "model.pt path" string is the wrong shape for
        # ckpt_manager.load_generation, which needs the directory + tag
        # separately to resolve the tag's generation symlink itself).
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
            import torch  # local import: only needed on this optional path

            from hunter_kinodynamic_rl.navigation.global_rl.agent import GlobalDQNAgent
            from hunter_kinodynamic_rl.navigation.global_rl.observation import resolve_map_channel_names
            map_channels = len(resolve_map_channel_names(self.profile.global_rl))
            max_nodes = (
                self.profile.memory.max_nodes_in_observation
                if self.profile.global_rl.topology_feedback_enabled else 0
            )
            self.global_agent = GlobalDQNAgent(self.profile.global_rl, map_channels, device="cpu", max_nodes=max_nodes)
            result = ckpt_manager.load_generation(
                global_ckpt_dir, self.get_parameter("global_checkpoint_name").value,
                self.global_agent.checkpoint_components(), map_location="cpu",
            )
            self.get_logger().info(f"loaded Global checkpoint: {result['loaded']} (skipped: {result['skipped']})")
            self._torch = torch
        else:
            self.get_logger().warn(
                "no global_checkpoint_dir provided -- using the built-in goal-seeking heuristic for "
                "subgoal selection, never a trained Global policy"
            )

        self._safety_limits = SafetyLimits(
            max_sensor_age_sec=self.profile.runtime.sensor_freshness_timeout_sec,
            max_odom_age_sec=self.profile.runtime.sensor_freshness_timeout_sec,
            max_command_age_sec=self.profile.runtime.watchdog_command_timeout_sec,
            min_obstacle_stop_distance_m=self.profile.risk.min_safe_clearance_m,
        )

        self._latest_scan: Optional[Tuple[np.ndarray, float, float]] = None
        # Message header stamp (mission_map_node.py's "never mix with wall/
        # monotonic clock" convention) -- used ONLY for pose_at()/
        # is_pose_usable() timestamp synchronization below.
        self._latest_scan_time: Optional[float] = None
        # time.monotonic() at RECEIPT -- a SEPARATE clock domain, used ONLY
        # for real-time staleness detection (safety guard freshness,
        # _localization_valid() below) -- mirrors
        # hierarchical_navigation_node.py's identical split (see that
        # module's own docstring for why comparing a message stamp against
        # itself, or against time.monotonic(), each silently defeats
        # staleness detection in a different way).
        self._latest_scan_receipt_time: Optional[float] = None
        self._latest_odom: Optional[Tuple[float, float, float, float, float]] = None
        self._latest_odom_time: Optional[float] = None
        self._latest_odom_receipt_time: Optional[float] = None
        self._latest_steering_rad = 0.0
        self._last_command_time: Optional[float] = None
        self._speed_history: Deque[Tuple[float, float]] = deque(maxlen=50)
        self._step_counter = 0
        self._rng = np.random.RandomState(0)
        self._final_goal_mission: Optional[Tuple[float, float]] = None

        self._tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self._mission_map_pub = self.create_publisher(OccupancyGrid, "/mission_map", LATCHED_QOS)
        self._goal_pub = self.create_publisher(PointStamped, "/mission_goal", LATCHED_QOS)
        self._cmd_pub = self.create_publisher(Twist, self.get_parameter("cmd_vel_topic").value, 10)

        self.create_subscription(Odometry, odom_topic, self._on_odom, SENSOR_QOS)
        self.create_subscription(LaserScan, self.get_parameter("scan_topic").value, self._on_scan, SENSOR_QOS)
        self.create_subscription(JointState, self.get_parameter("joint_states_topic").value,
                                  self._on_joint_states, SENSOR_QOS)
        self.create_timer(self.profile.runtime.time_delta_sec, self._on_control_tick)
        self.create_timer(float(self.get_parameter("publish_period_sec").value), self._publish_visualization)

        self.get_logger().info(f"hierarchical_environment_node started: profile={self.profile.name}")

    # ------------------------------------------------------------------ sensor callbacks
    def _on_odom(self, msg: Odometry) -> None:
        self._latest_odom_receipt_time = time.monotonic()
        stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
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
            # goal_x/goal_y are the user's relative goal in the ROBOT'S OWN
            # frame AT MISSION START -- by construction that IS the
            # MissionFrame origin (see navigation/mission/goal_manager.py's
            # module docstring), so it is already the mission-frame goal
            # and must NOT be passed through mission_frame.odom_to_mission()
            # (that transform is for an ODOM-frame point, which this is
            # not) -- doing so silently rotated/translated the goal by the
            # start pose for any non-zero start pose/yaw. Matches
            # mission_map_node.py's own goal_manager.set_goal(goal_x,
            # goal_y) call exactly.
            self._final_goal_mission = (
                float(self.get_parameter("goal_x").value), float(self.get_parameter("goal_y").value),
            )
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

    def _localization_valid(self) -> bool:
        """``is_pose_usable(pose, now_sec=self._latest_odom_time, ...)``
        used to compare the pose's own message stamp against ITSELF (both
        come from the same odom message) -- True forever once ANY odom had
        ever arrived, even long after odom RECEIPT stopped. The
        ``is_pose_usable`` call below still only ever compares two values
        in the SAME (message-stamp) clock domain -- it correctly still
        catches ``pose.valid``/non-finite/low-confidence readings, its
        "age" check just stays a deliberate no-op self-comparison. The
        separate RECEIPT-time check (monotonic domain) is what actually
        detects "odom has not arrived in real wall-clock time recently"."""
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
        """The backend's own reported confidence, forced to ``0.0``
        whenever ``_localization_valid()`` is False (receipt-stale or
        otherwise unusable) -- feeds ``HierarchyCoordinator``'s existing
        degraded-localization state machine so stale localization also
        blocks Global subgoal selection/activation, not just the physical
        command (mirrors hierarchical_navigation_node.py's identical
        fix)."""
        if not self._localization_valid():
            return 0.0
        return self._localization.latest_pose().confidence

    # ------------------------------------------------------------------ Global decision
    def _maybe_select_next_subgoal(self, robot_pose_mission: PoseXYYaw) -> bool:
        """Runs the Global decision when the coordinator has no active
        subgoal. Returns whether a subgoal is now active."""
        if not self.coordinator.stop_required or self.coordinator.mission_done:
            return not self.coordinator.stop_required
        localization_confidence = self._effective_localization_confidence()
        if not self._localization_valid():
            # Stale localization must block the ENTIRE Global decision, not
            # just activation -- see hierarchical_navigation_node.py's
            # identical fix for the full rationale (a stale candidate
            # enqueued every tick would otherwise pile up in the
            # coordinator's queue, un-drained while degraded, and activate
            # FIFO-oldest-first the moment localization recovers). The call
            # below only updates the coordinator's degraded-localization
            # gate (queue is always empty at this point).
            self.coordinator.activate_next_subgoal(
                robot_pose_mission, now_step=self._step_counter, now_time_sec=time.monotonic(),
                localization_confidence=localization_confidence,
            )
            return False
        mask = compute_action_mask(
            self.candidates, self.partial_map, robot_pose_mission, self.profile.global_rl,
            self.profile.robot.max_curvature, localization_valid=True,
        )
        obs = build_global_observation(
            self.partial_map, robot_pose_mission, self._final_goal_mission, self.candidates, mask,
            speed_mps=self._speed_at(self._latest_odom_time or 0.0), previous_action_index=self.profile.global_rl.fallback_index,
            elapsed_mission_ratio=0.0, config=self.profile.global_rl,
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

        return self.coordinator.activate_next_subgoal(
            robot_pose_mission, now_step=self._step_counter, now_time_sec=time.monotonic(),
            is_valid=_is_valid, localization_confidence=localization_confidence,
        )

    # ------------------------------------------------------------------ control loop
    def _publish(self, command) -> None:
        self._cmd_pub.publish(_twist_from(command))

    def _on_control_tick(self) -> None:
        now = time.monotonic()
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
            raw_action = self.local_agent.select_action(local_obs.observation, deterministic=True)
            action_arr = LocalPolicyController.validate_action(raw_action)
            if action_arr is None:
                self._publish(STOP_COMMAND)
                self._last_command_time = now
                return
            self._local_controller.commit_action(action_arr)
            # now_sec/last_*_time_sec must all share ONE clock domain
            # (is_pose_usable()'s own documented contract) -- `now` here is
            # time.monotonic(), so the freshness inputs must be the
            # MONOTONIC RECEIPT timestamps, never the message-stamp
            # self._latest_scan_time/self._latest_odom_time (mirrors
            # hierarchical_navigation_node.py's identical fix).
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
        except Exception as e:  # noqa: BLE001 -- last-resort fail-safe, mirrors real_policy_node.py
            self.get_logger().error(f"[hierarchical_environment_node] control tick raised {e!r} -- safe stop")
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
    node = HierarchicalEnvironmentNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
