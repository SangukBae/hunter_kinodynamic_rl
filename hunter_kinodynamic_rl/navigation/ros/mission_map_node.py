#!/usr/bin/env python3
"""``ros2 run hunter_kinodynamic_rl mission_map_node.py`` -- Phase 1
verification node (plan section 5.9's "RViz에서 mission pose, final goal,
occupancy, unknown, visited를 확인할 수 있다"). Subscribes to ``/odometry`` +
a 360-degree ``/scan``, accumulates the mission :class:`PartialMap`, and
publishes ``nav_msgs/OccupancyGrid`` + a mission-frame goal point + the
``odom -> mission`` TF for RViz. Never touches ``/cmd_vel`` or any
simulator/ground-truth state -- read-only w.r.t. the running simulation.

**Clock domain** (code review fix): every timestamp compared here is a
message header stamp -- either a scan's own stamp or an odometry pose's own
stamp -- NEVER ``time.time()``/wall-clock mixed with a Gazebo-bridged sim
timestamp. That exact mixing bug ("a use_sim_time mismatch between Nav2's
clock and Gazebo-bridged /odometry's sim-time stamps") was previously found
and fixed live in this codebase's own Nav2-MPPI baseline (see
``evaluation/nav2_mppi_runner.py``'s module docstring) -- this node avoids
it structurally by never reading ``self.get_clock().now()``/``time.time()``
for any freshness comparison, and looking up the pose AT the scan's own
timestamp (:meth:`OdomLocalizationBackend.pose_at`) rather than whatever
odometry reading happened to be most recently RECEIVED.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, Tuple

import numpy as np
import rclpy
import tf2_ros
from geometry_msgs.msg import PointStamped, Pose, TransformStamped
from nav_msgs.msg import Odometry, OccupancyGrid, MapMetaData
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan

from hunter_kinodynamic_rl.config.loader import load_profile
from hunter_kinodynamic_rl.navigation.localization.gazebo_odom_backend import GazeboOdomLocalizationBackend
from hunter_kinodynamic_rl.navigation.localization.interface import is_pose_usable
from hunter_kinodynamic_rl.navigation.mapping.partial_map import MapChannels, PartialMap
from hunter_kinodynamic_rl.navigation.mapping.rolling_map import RollingCrop, crop_rolling
from hunter_kinodynamic_rl.navigation.mission.goal_manager import GoalManager, GoalManagerConfig
from hunter_kinodynamic_rl.navigation.mission.mission_frame import MissionFrame, PoseXYYaw

OCCUPIED_VALUE = 100
FREE_VALUE = 0
AMBIGUOUS_VALUE = 50
UNKNOWN_VALUE = -1

# section item-1 (code review): the real /scan and /odometry publishers in
# this codebase (pointcloud_to_laserscan, the Ignition odometry bridge; also
# environment_node.py's OWN subscription to both) use BEST_EFFORT --
# subscribing RELIABLE (the rclpy default) is incompatible per the DDS QoS
# spec and can silently starve the callback entirely with no error raised
# anywhere. Confirmed as a real, previously-hit bug in this exact codebase
# (evaluation/nav2_mppi_runner.py's docstring: "a QoS mismatch that
# silently starved the /scan subscription").
SENSOR_QOS = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

LATCHED_QOS = QoSProfile(
    depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


def channel_to_occupancy_data(
    occupied: np.ndarray, free: np.ndarray, unknown: np.ndarray, observed_uncertain: np.ndarray,
) -> np.ndarray:
    """Row-major ``int8`` OccupancyGrid payload from the four exclusive,
    exhaustive boolean channels (see ``partial_map.py``'s module docstring
    for why there are four, not three) -- every cell is written by exactly
    one of the four assignments below; there is no implicit/default value
    relied on for correctness."""
    data = np.zeros(occupied.shape, dtype=np.int8)
    data[unknown] = UNKNOWN_VALUE
    data[free] = FREE_VALUE
    data[observed_uncertain] = AMBIGUOUS_VALUE
    data[occupied] = OCCUPIED_VALUE
    return data.reshape(-1)


def build_occupancy_grid_msg(
    data: np.ndarray, resolution_m: float, origin_x: float, origin_y: float,
    size_cells: int, frame_id: str, stamp,
) -> OccupancyGrid:
    msg = OccupancyGrid()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.info = MapMetaData()
    msg.info.resolution = float(resolution_m)
    msg.info.width = int(size_cells)
    msg.info.height = int(size_cells)
    origin = Pose()
    origin.position.x = float(origin_x)
    origin.position.y = float(origin_y)
    origin.position.z = 0.0
    origin.orientation.w = 1.0
    msg.info.origin = origin
    msg.data = data.astype(np.int8).tolist()
    return msg


def yaw_to_quaternion(yaw: float):
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


class MissionMapNode(Node):
    def __init__(self, **node_kwargs) -> None:
        # **node_kwargs (e.g. cli_args=[...]) forwarded straight to
        # rclpy.node.Node -- lets tests construct this node with parameter
        # overrides (a different `profile`) without needing a live
        # `ros2 run --ros-args -p ...` process.
        super().__init__("mission_map_node", **node_kwargs)

        self.declare_parameter("profile", "hierarchical_phase1")
        self.declare_parameter("goal_x", 5.0)
        self.declare_parameter("goal_y", 0.0)
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("odom_topic", "")  # empty -> profile.localization.odom_topic
        self.declare_parameter("publish_period_sec", 1.0)
        self.declare_parameter("lidar_offset_x", 0.0)
        self.declare_parameter("lidar_offset_y", 0.0)
        self.declare_parameter("lidar_yaw_offset_rad", 0.0)
        self.declare_parameter("odom_frame_id", "odom")
        self.declare_parameter("mission_frame_id", "mission")

        profile_name = self.get_parameter("profile").value
        self.profile = load_profile(profile_name)

        odom_topic = self.get_parameter("odom_topic").value or self.profile.localization.odom_topic
        self._odom_frame_id = self.get_parameter("odom_frame_id").value
        self._mission_frame_id = self.get_parameter("mission_frame_id").value
        self._lidar_offset = (
            float(self.get_parameter("lidar_offset_x").value),
            float(self.get_parameter("lidar_offset_y").value),
        )
        self._lidar_yaw_offset = float(self.get_parameter("lidar_yaw_offset_rad").value)

        self.mission_frame = MissionFrame()
        self.goal_manager = GoalManager(
            GoalManagerConfig(
                position_tolerance_m=self.profile.mission.position_tolerance_m,
                heading_tolerance_rad=self.profile.mission.heading_tolerance_rad,
                require_low_speed_on_goal=self.profile.mission.require_low_speed_on_goal,
                goal_speed_threshold_mps=self.profile.mission.goal_speed_threshold_mps,
            ),
            reset_memory_on_goal_change=self.profile.mission.reset_memory_on_goal_change,
        )
        self.partial_map = PartialMap(self.profile.mapping)
        # section item-7 (code review): LocalizationConfig.backend actually
        # selects behaviour -- "gazebo_odom" derives confidence from the
        # message's own covariance trace, "odom" trusts the reading
        # unconditionally (confidence pinned to 1.0). Both still parse the
        # same nav_msgs/Odometry wire format (Phase 1 has only one live
        # transport); Phase 6's wheel+IMU/LiDAR-odometry/LIO backends will
        # be genuinely different adapters, not just a confidence-model
        # toggle on this one.
        self._localization = GazeboOdomLocalizationBackend(
            use_covariance_confidence=(self.profile.localization.backend == "gazebo_odom"),
        )
        # (stamp_sec, speed_mps) samples from /odometry's own twist, looked
        # up by scan timestamp via `_speed_at` -- mirrors the pose_at
        # pattern (never just "whatever was last received") since the
        # low-speed-on-goal gate must reflect the robot's speed AT the
        # moment being checked, not an arbitrary earlier/later reading.
        self._speed_history: Deque[Tuple[float, float]] = deque(maxlen=50)
        self._step_counter = 0
        self._tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self._mission_map_pub = self.create_publisher(OccupancyGrid, "/mission_map", LATCHED_QOS)
        self._rolling_map_pub = self.create_publisher(OccupancyGrid, "/rolling_map", LATCHED_QOS)
        self._visited_map_pub = self.create_publisher(OccupancyGrid, "/visited_map", LATCHED_QOS)
        self._inflated_map_pub = self.create_publisher(OccupancyGrid, "/inflated_map", LATCHED_QOS)
        self._goal_pub = self.create_publisher(PointStamped, "/mission_goal", LATCHED_QOS)

        self.create_subscription(Odometry, odom_topic, self._on_odometry, SENSOR_QOS)
        self.create_subscription(LaserScan, self.get_parameter("scan_topic").value, self._on_scan, SENSOR_QOS)

        publish_period = float(self.get_parameter("publish_period_sec").value)
        self.create_timer(publish_period, self._publish_all)

        self.get_logger().info(
            f"mission_map_node started: profile={self.profile.name} odom_topic={odom_topic} "
            f"scan_topic={self.get_parameter('scan_topic').value} "
            f"localization.backend={self.profile.localization.backend} "
            f"localization.publish_mission_tf={self.profile.localization.publish_mission_tf}"
        )

    # ------------------------------------------------------------------ subscriptions
    def _on_odometry(self, msg: Odometry) -> None:
        stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self._localization.on_odometry_msg(msg, stamp_sec)
        speed_mps = math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)
        # section item-1 (code review round 3): a NaN/Inf twist -- e.g. a
        # corrupted message, or hypot() itself producing NaN from a NaN
        # input -- must never be stored as-is. `abs(float("nan")) >
        # threshold` is False in Python, so an unguarded NaN here silently
        # bypassed `GoalManager.check_reached`'s low-speed-on-goal gate
        # entirely (confirmed live: speed_is_nan=True -> goal_reached=True).
        # Forced to +inf (definitely-moving / unknown-but-untrusted), the
        # same fail-closed value `_speed_at` already returns for a
        # missing/stale sample -- never silently treated as "stopped".
        if not math.isfinite(speed_mps):
            speed_mps = float("inf")
        self._speed_history.append((stamp_sec, speed_mps))

        if not self.mission_frame.initialized:
            pose = self._localization.latest_pose()
            # section item-4 (code review round 2): the SAME shared gate
            # `_on_scan` uses for every ordinary map update -- previously
            # only `valid`/finiteness was checked here, so a first
            # odometry reading with a huge covariance (hence low
            # `is_pose_usable`-computed confidence) could still permanently
            # anchor the whole mission's coordinate frame; every LATER map
            # update would then correctly get rejected by the confidence
            # gate, but the origin itself -- and therefore every mission-
            # frame coordinate for the rest of the mission -- would already
            # be wrong. `now_sec=pose.stamp_sec` makes the age term exactly
            # 0 (a reading is always "fresh" relative to its own
            # timestamp), so this only adds the valid/finite/confidence
            # checks on top of what was already done.
            if is_pose_usable(
                pose, now_sec=pose.stamp_sec, timeout_sec=self.profile.localization.pose_timeout_sec,
                min_confidence=self.profile.localization.minimum_confidence,
            ):
                self.mission_frame.initialize(PoseXYYaw(x=pose.x, y=pose.y, yaw=pose.yaw))
                self.goal_manager.set_goal(
                    float(self.get_parameter("goal_x").value), float(self.get_parameter("goal_y").value),
                )
                self.get_logger().info(
                    f"mission frame initialized at ({pose.x:.3f}, {pose.y:.3f}, {pose.yaw:.3f}); "
                    f"relative goal=({self.goal_manager.relative_goal[0]:.3f}, {self.goal_manager.relative_goal[1]:.3f})"
                )

    def _speed_at(self, stamp_sec: float, max_dt_sec: float) -> float:
        """Nearest recorded ``/odometry`` speed sample within ``max_dt_sec``
        of ``stamp_sec``. When no sample is close enough (or none exist
        yet), returns ``inf`` rather than ``0.0`` -- a MISSING speed
        reading must never be silently treated as "stopped", which would
        let a fast-moving robot satisfy the low-speed-on-goal gate purely
        because its speed was never actually measured (fail toward
        REJECTING a spurious goal-reached, matching this codebase's
        stale-sensor-means-explicit-stop convention elsewhere)."""
        if not self._speed_history:
            return float("inf")
        stamp, speed = min(self._speed_history, key=lambda item: abs(item[0] - stamp_sec))
        if abs(stamp - stamp_sec) > max_dt_sec:
            return float("inf")
        # Defense in depth (round 3): `_on_odometry` already forces a
        # non-finite reading to +inf before storing it, but never trust a
        # single ingestion-time guard alone for a safety-relevant value --
        # a NaN speed must never reach the low-speed-on-goal gate as NaN.
        if not math.isfinite(speed):
            return float("inf")
        return speed

    def _current_speed_at(self, stamp_sec: float) -> float:
        return self._speed_at(stamp_sec, max_dt_sec=self.profile.localization.pose_timeout_sec)

    def _on_scan(self, msg: LaserScan) -> None:
        if not self.mission_frame.initialized:
            return
        scan_stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # section item-2 (code review): look up the pose AT the scan's own
        # timestamp (interpolated from history) rather than "whatever
        # odometry happened to be latest when this callback fired" -- scan
        # and odometry are independent topics with no ordering guarantee.
        # `now_sec=scan_stamp_sec` keeps the freshness comparison entirely
        # within the message-timestamp clock domain (see module docstring).
        pose = self._localization.pose_at(scan_stamp_sec, max_dt_sec=self.profile.localization.pose_timeout_sec)
        if not is_pose_usable(
            pose, now_sec=scan_stamp_sec, timeout_sec=self.profile.localization.pose_timeout_sec,
            min_confidence=self.profile.localization.minimum_confidence,
        ):
            return

        robot_pose_mission = self.mission_frame.odom_pose_to_mission(PoseXYYaw(x=pose.x, y=pose.y, yaw=pose.yaw))
        sensor_origin_xy = self.mission_frame.robot_to_mission(self._lidar_offset, robot_pose_mission)

        n = len(msg.ranges)
        if n == 0:
            return
        beam_yaws_robot = msg.angle_min + msg.angle_increment * np.arange(n, dtype=np.float64)
        beam_angles_mission = robot_pose_mission.yaw + self._lidar_yaw_offset + beam_yaws_robot
        ranges = np.asarray(msg.ranges, dtype=np.float64)

        self.partial_map.integrate_scan(
            sensor_origin_xy, beam_angles_mission, ranges, float(msg.range_max), range_min=float(msg.range_min),
        )
        self.partial_map.record_visit(robot_pose_mission.x, robot_pose_mission.y, self._step_counter)
        self._step_counter += 1

        self.goal_manager.check_reached(robot_pose_mission, speed_mps=self._current_speed_at(scan_stamp_sec))

    # ------------------------------------------------------------------ publishing
    def _publish_all(self) -> None:
        if not self.mission_frame.initialized:
            return
        stamp = self.get_clock().now().to_msg()
        if self.profile.localization.publish_mission_tf:
            self._publish_tf(stamp)

        channels = self.partial_map.channels()
        self._mission_map_pub.publish(self._channels_to_grid(channels, self._mission_frame_id, stamp))

        inflated_data = np.where(channels.inflated, OCCUPIED_VALUE, np.where(channels.unknown, UNKNOWN_VALUE, FREE_VALUE))
        self._inflated_map_pub.publish(
            build_occupancy_grid_msg(
                inflated_data.astype(np.int8).reshape(-1), channels.resolution_m, channels.origin_x,
                channels.origin_y, self.partial_map.size_cells, self._mission_frame_id, stamp,
            )
        )

        pose = self._localization.latest_pose()
        if pose.valid:
            robot_pose_mission = self.mission_frame.odom_pose_to_mission(PoseXYYaw(x=pose.x, y=pose.y, yaw=pose.yaw))
            crop = crop_rolling(
                self.partial_map, (robot_pose_mission.x, robot_pose_mission.y),
                self.profile.mapping.rolling_size_cells,
            )
            self._rolling_map_pub.publish(self._crop_to_grid(crop, self._mission_frame_id, stamp))

        visited_data = np.clip(channels.visited * 100.0, 0, 100).astype(np.int8).reshape(-1)
        self._visited_map_pub.publish(
            build_occupancy_grid_msg(
                visited_data, channels.resolution_m, channels.origin_x, channels.origin_y,
                self.partial_map.size_cells, self._mission_frame_id, stamp,
            )
        )

        goal_msg = PointStamped()
        goal_msg.header.frame_id = self._mission_frame_id
        goal_msg.header.stamp = stamp
        gx, gy = self.goal_manager.mission_goal
        goal_msg.point.x, goal_msg.point.y, goal_msg.point.z = gx, gy, 0.0
        self._goal_pub.publish(goal_msg)

    def _channels_to_grid(self, channels: MapChannels, frame_id: str, stamp) -> OccupancyGrid:
        data = channel_to_occupancy_data(channels.occupied, channels.free, channels.unknown, channels.observed_uncertain)
        return build_occupancy_grid_msg(
            data, channels.resolution_m, channels.origin_x, channels.origin_y,
            self.partial_map.size_cells, frame_id, stamp,
        )

    def _crop_to_grid(self, crop: RollingCrop, frame_id: str, stamp) -> OccupancyGrid:
        data = channel_to_occupancy_data(crop.occupied, crop.free, crop.unknown, crop.observed_uncertain)
        return build_occupancy_grid_msg(
            data, crop.resolution_m, crop.origin_x, crop.origin_y, crop.size_cells, frame_id, stamp,
        )

    def _publish_tf(self, stamp) -> None:
        origin = self.mission_frame.origin
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = self._odom_frame_id
        t.child_frame_id = self._mission_frame_id
        t.transform.translation.x = origin.x
        t.transform.translation.y = origin.y
        t.transform.translation.z = 0.0
        qx, qy, qz, qw = yaw_to_quaternion(origin.yaw)
        t.transform.rotation.x, t.transform.rotation.y = qx, qy
        t.transform.rotation.z, t.transform.rotation.w = qz, qw
        self._tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = MissionMapNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
