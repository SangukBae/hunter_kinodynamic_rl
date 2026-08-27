"""ROS-node-level coverage for MissionMapNode -- requires a built ROS
workspace (rclpy + nav_msgs + sensor_msgs), self-skips cleanly on a bare
host checkout, mirroring test_environment_node.py's pattern.

Every stamp used below is an arbitrary MESSAGE-domain float -- deliberately
including values that look nothing like wall-clock ``time.time()`` (e.g.
``12345.678``) alongside huge wall-clock-epoch-shaped values, to prove the
node's freshness/synchronization logic genuinely only ever compares message
timestamps against each other and never reads any external clock itself
(the sim-time/wall-time mixing bug this file's tests previously masked by
always using ``time.time()`` for both odometry and scan)."""

import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("nav_msgs")
pytest.importorskip("sensor_msgs")

from rclpy.qos import ReliabilityPolicy  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from sensor_msgs.msg import LaserScan  # noqa: E402

from hunter_kinodynamic_rl.navigation.mission.goal_manager import MissionStatus  # noqa: E402
from hunter_kinodynamic_rl.navigation.ros.mission_map_node import (  # noqa: E402
    MissionMapNode, SENSOR_QOS, build_occupancy_grid_msg, channel_to_occupancy_data, yaw_to_quaternion,
)


def test_sensor_qos_is_best_effort():
    """Code review fix (item 1): the real /scan and /odometry publishers in
    this codebase use BEST_EFFORT -- a RELIABLE subscriber is incompatible
    per DDS QoS rules and can silently starve the callback entirely."""
    assert SENSOR_QOS.reliability == ReliabilityPolicy.BEST_EFFORT


def test_channel_to_occupancy_data_mapping():
    occupied = np.array([[True, False], [False, False]])
    free = np.array([[False, True], [False, False]])
    unknown = np.array([[False, False], [True, False]])
    observed_uncertain = np.array([[False, False], [False, True]])
    data = channel_to_occupancy_data(occupied, free, unknown, observed_uncertain)
    assert data.tolist() == [100, 0, -1, 50]


def test_build_occupancy_grid_msg_shape_and_metadata():
    data = np.zeros(9, dtype=np.int8)
    msg = build_occupancy_grid_msg(data, resolution_m=0.5, origin_x=-1.0, origin_y=-1.0, size_cells=3, frame_id="mission", stamp=rclpy.time.Time().to_msg())
    assert msg.info.resolution == pytest.approx(0.5)
    assert msg.info.width == 3
    assert msg.info.height == 3
    assert msg.header.frame_id == "mission"
    assert len(msg.data) == 9


def test_yaw_to_quaternion_zero_is_identity():
    qx, qy, qz, qw = yaw_to_quaternion(0.0)
    assert (qx, qy, qz) == pytest.approx((0.0, 0.0, 0.0))
    assert qw == pytest.approx(1.0)


def _stamped(sec: float):
    from builtin_interfaces.msg import Time as TimeMsg
    t = TimeMsg()
    t.sec = int(sec)
    t.nanosec = int(round((sec - int(sec)) * 1e9))
    return t


def _odom_msg(x: float, y: float, yaw: float, stamp_sec: float, speed_mps: float = 0.0, covariance=None) -> Odometry:
    msg = Odometry()
    msg.header.stamp = _stamped(stamp_sec)
    msg.pose.pose.position.x = x
    msg.pose.pose.position.y = y
    msg.pose.pose.orientation.z = float(np.sin(yaw / 2.0))
    msg.pose.pose.orientation.w = float(np.cos(yaw / 2.0))
    msg.pose.covariance = [0.0] * 36 if covariance is None else covariance
    msg.twist.twist.linear.x = speed_mps
    return msg


def _scan_msg(stamp_sec: float, n: int = 40, range_val: float = 5.0, range_max: float = 10.0, range_min: float = 0.05) -> LaserScan:
    msg = LaserScan()
    msg.header.stamp = _stamped(stamp_sec)
    msg.angle_min = -np.pi
    msg.angle_increment = 2 * np.pi / n
    msg.range_max = range_max
    msg.range_min = range_min
    msg.ranges = [range_val] * n
    return msg


def _node(cli_args=None):
    if not rclpy.ok():
        rclpy.init()
    return MissionMapNode(cli_args=cli_args) if cli_args else MissionMapNode()


@pytest.fixture
def node():
    n = _node()
    yield n
    n.destroy_node()


def test_node_scan_subscription_uses_sensor_qos(node):
    sub = next(s for s in node.subscriptions if s.topic_name == "/scan")
    assert sub.qos_profile.reliability == ReliabilityPolicy.BEST_EFFORT


def test_node_odom_subscription_uses_sensor_qos(node):
    sub = next(s for s in node.subscriptions if s.topic_name == "/odometry")
    assert sub.qos_profile.reliability == ReliabilityPolicy.BEST_EFFORT


def test_node_initializes_mission_frame_on_first_valid_odometry(node):
    assert not node.mission_frame.initialized
    node._on_odometry(_odom_msg(1.0, 2.0, 0.3, stamp_sec=12345.678))
    assert node.mission_frame.initialized
    assert node.goal_manager.active


def test_node_ignores_scan_before_mission_frame_initialized(node):
    node._on_scan(_scan_msg(stamp_sec=12345.678))
    assert node.partial_map.observed.sum() == 0


@pytest.mark.parametrize("sim_time_base", [12345.678, 1_788_000_000.0, 0.05])
def test_node_integrates_scan_after_valid_localization_any_clock_magnitude(node, sim_time_base):
    """The core regression test for the clock-domain-mixing fix: this must
    succeed identically whether message stamps look like small Gazebo
    sim-time values or huge wall-clock-epoch values -- the node reads no
    external clock for this decision at all."""
    node._on_odometry(_odom_msg(0.0, 0.0, 0.0, stamp_sec=sim_time_base))
    node._on_scan(_scan_msg(stamp_sec=sim_time_base + 0.02, range_val=3.0))
    assert node.partial_map.observed.sum() > 0


def _single_beam_scan_msg(stamp_sec: float, range_val: float, n: int = 40, range_max: float = 10.0) -> LaserScan:
    """A scan where every beam is a no-return EXCEPT the one pointing along
    the sensor's own +x axis (angle 0) -- isolates exactly one occupied
    cell so a test can check precisely where it landed."""
    msg = LaserScan()
    msg.header.stamp = _stamped(stamp_sec)
    msg.angle_min = -np.pi
    msg.angle_increment = 2 * np.pi / n
    msg.range_max = range_max
    msg.range_min = 0.05
    ranges = [float("inf")] * n
    zero_angle_index = round((0.0 - msg.angle_min) / msg.angle_increment) % n
    ranges[zero_angle_index] = range_val
    msg.ranges = ranges
    return msg


def test_node_synchronizes_scan_against_pose_at_scan_timestamp_not_latest_received(node):
    """Code review fix (item 2): the pose used for a scan must be looked up
    AT the scan's own timestamp (interpolated), not whatever odometry
    reading happened to be most recently received. Feeds two odometry
    samples bracketing the scan's timestamp with different x positions
    (0.3s apart on each side -- within the default pose_timeout_sec=0.5s
    interpolation gap bound, unlike unrealistically-far-apart odometry);
    the sensor origin used for mapping must reflect the INTERPOLATED
    midpoint, not the later (in wall-arrival-order) sample's raw x."""
    node._on_odometry(_odom_msg(0.0, 0.0, 0.0, stamp_sec=10.7))
    node._on_odometry(_odom_msg(2.0, 0.0, 0.0, stamp_sec=11.3))
    # Scan timestamped exactly halfway between the two odometry samples.
    node._on_scan(_single_beam_scan_msg(stamp_sec=11.0, range_val=3.0))
    channels = node.partial_map.channels()
    occ_row, occ_col = np.argwhere(channels.occupied)[0]
    ox, _ = node.partial_map.cell_to_world(occ_row, occ_col)
    # Sensor origin interpolated x=1.0 + 3m hit -> occupied cell near x=4.0,
    # NOT x=2.0+3=5.0 (which "latest received" would have produced).
    assert ox == pytest.approx(4.0, abs=node.partial_map.resolution_m * 2)


def test_node_rejects_stale_localization_for_map_update_pure_message_domain(node):
    """Odometry and scan are both timestamped in an arbitrary sim-time-like
    domain (no wall-clock value anywhere) -- the huge gap between them
    alone must trigger rejection, proving staleness is judged purely from
    message timestamps."""
    node._on_odometry(_odom_msg(0.0, 0.0, 0.0, stamp_sec=10.0))
    node._on_scan(_scan_msg(stamp_sec=500.0))  # 490s gap, default timeout 0.5s
    assert node.partial_map.observed.sum() == 0


def test_node_does_not_reinitialize_mission_frame_on_subsequent_odometry(node):
    node._on_odometry(_odom_msg(1.0, 1.0, 0.0, stamp_sec=100.0))
    origin_first = node.mission_frame.origin
    node._on_odometry(_odom_msg(5.0, 5.0, 1.0, stamp_sec=101.0))
    assert node.mission_frame.origin == origin_first


def test_node_odometry_with_non_finite_pose_does_not_initialize_mission_frame(node):
    node._on_odometry(_odom_msg(float("nan"), 0.0, 0.0, stamp_sec=10.0))
    assert not node.mission_frame.initialized


def test_node_low_confidence_odometry_does_not_initialize_mission_frame(node):
    """Code review fix (round 2, item 4): the first odometry reading must
    pass the SAME confidence gate as every later map update, not just
    valid/finite -- a huge-covariance first reading must never permanently
    anchor the mission origin."""
    huge_cov = [0.0] * 36
    huge_cov[0] = 1e9  # x variance -> ~0 confidence via confidence_from_covariance
    node._on_odometry(_odom_msg(1.0, 1.0, 0.0, stamp_sec=10.0, covariance=huge_cov))
    assert not node.mission_frame.initialized

    # A later, GOOD reading must still be able to initialize it.
    node._on_odometry(_odom_msg(2.0, 2.0, 0.0, stamp_sec=10.05))
    assert node.mission_frame.initialized
    assert node.mission_frame.origin.x == pytest.approx(2.0)


def test_node_fast_moving_robot_near_goal_is_not_marked_reached(node):
    """Code review fix (round 2, item 3): _last_speed_mps was never
    updated, so the low-speed-on-goal gate always saw speed=0.0 --
    hierarchical_phase1's require_low_speed_on_goal=true would have let a
    fast-moving robot register as goal-reached."""
    node._on_odometry(_odom_msg(0.0, 0.0, 0.0, stamp_sec=1.0))  # mission origin
    assert node.goal_manager.mission_goal == (5.0, 0.0)  # default goal_x/goal_y params
    node._on_odometry(_odom_msg(4.9, 0.0, 0.0, stamp_sec=1.2, speed_mps=2.0))  # within tolerance, FAST
    node._on_scan(_scan_msg(stamp_sec=1.2))
    assert node.goal_manager.status != MissionStatus.REACHED


def test_node_slow_robot_near_goal_is_marked_reached(node):
    node._on_odometry(_odom_msg(0.0, 0.0, 0.0, stamp_sec=1.0))
    node._on_odometry(_odom_msg(4.9, 0.0, 0.0, stamp_sec=1.2, speed_mps=0.02))  # within tolerance, SLOW
    node._on_scan(_scan_msg(stamp_sec=1.2))
    assert node.goal_manager.status == MissionStatus.REACHED


def test_node_missing_speed_sample_does_not_falsely_satisfy_low_speed_gate(node):
    """_speed_at must fail toward "moving" (inf), not "stopped" (0.0), when
    no speed sample is close enough to the query timestamp."""
    node._on_odometry(_odom_msg(0.0, 0.0, 0.0, stamp_sec=1.0))
    # Scan far in the future relative to any recorded speed sample --
    # localization itself would ALSO reject this as stale, so directly
    # exercise _speed_at instead of the full _on_scan path.
    assert node._speed_at(stamp_sec=1000.0, max_dt_sec=0.5) == float("inf")


@pytest.mark.parametrize("bad_speed", [float("nan"), float("inf"), float("-inf")])
def test_node_non_finite_twist_speed_is_stored_as_inf_not_raw(node, bad_speed):
    """Code review fix (round 3): a NaN/Inf twist.linear must never be
    stored as-is -- `abs(float("nan")) > threshold` is False in Python, so
    an unguarded NaN silently bypassed the low-speed-on-goal gate entirely
    (confirmed live in the review: speed_is_nan=True -> goal_reached=True).
    Forced to +inf at ingestion (fail toward "moving")."""
    node._on_odometry(_odom_msg(0.0, 0.0, 0.0, stamp_sec=1.0, speed_mps=bad_speed))
    _, stored_speed = node._speed_history[-1]
    assert stored_speed == float("inf")


def test_node_nan_twist_speed_does_not_bypass_low_speed_goal_gate(node):
    """End-to-end reproduction of the exact scenario the review reported:
    a robot within goal tolerance but reporting a NaN speed must NOT be
    marked reached."""
    node._on_odometry(_odom_msg(0.0, 0.0, 0.0, stamp_sec=1.0))  # mission origin
    node._on_odometry(_odom_msg(4.9, 0.0, 0.0, stamp_sec=1.2, speed_mps=float("nan")))
    node._on_scan(_scan_msg(stamp_sec=1.2))
    assert node.goal_manager.status != MissionStatus.REACHED


def test_speed_at_defends_against_a_non_finite_value_already_in_history(node):
    """Defense in depth: even if a non-finite value somehow ends up in
    _speed_history (bypassing the _on_odometry ingestion guard), _speed_at
    must still never return it as-is."""
    node._speed_history.append((5.0, float("nan")))
    assert node._speed_at(stamp_sec=5.0, max_dt_sec=1.0) == float("inf")


def test_node_backend_config_selects_confidence_mode(tmp_path):
    """Code review fix (item 7): localization.backend actually changes node
    behaviour -- 'odom' pins confidence to 1.0 regardless of covariance."""
    profile_path = tmp_path / "odom_backend.yaml"
    profile_path.write_text("localization:\n  backend: odom\n")
    n = _node(cli_args=["--ros-args", "-p", f"profile:={profile_path}"])
    try:
        assert n.profile.localization.backend == "odom"
        assert n._localization._use_covariance_confidence is False
    finally:
        n.destroy_node()


def test_node_backend_gazebo_odom_is_default_confidence_mode(node):
    assert node.profile.localization.backend == "gazebo_odom"
    assert node._localization._use_covariance_confidence is True


def test_node_publish_mission_tf_false_skips_tf_publish(tmp_path):
    profile_path = tmp_path / "no_tf.yaml"
    profile_path.write_text("localization:\n  publish_mission_tf: false\n")
    n = _node(cli_args=["--ros-args", "-p", f"profile:={profile_path}"])
    try:
        n._on_odometry(_odom_msg(0.0, 0.0, 0.0, stamp_sec=1.0))
        assert n.mission_frame.initialized
        calls = []
        n._tf_broadcaster.sendTransform = lambda t: calls.append(t)
        n._publish_all()
        assert calls == []
    finally:
        n.destroy_node()


def test_node_publish_mission_tf_true_publishes_tf(node):
    node._on_odometry(_odom_msg(0.0, 0.0, 0.0, stamp_sec=1.0))
    assert node.profile.localization.publish_mission_tf is True
    calls = []
    node._tf_broadcaster.sendTransform = lambda t: calls.append(t)
    node._publish_all()
    assert len(calls) == 1
