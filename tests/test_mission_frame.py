import math

import pytest

from hunter_kinodynamic_rl.navigation.mission.mission_frame import MissionFrame, MissionFrameError, PoseXYYaw


def test_uninitialized_frame_raises():
    frame = MissionFrame()
    assert not frame.initialized
    with pytest.raises(MissionFrameError):
        frame.origin
    with pytest.raises(MissionFrameError):
        frame.mission_to_odom(1.0, 2.0)


def test_identity_start_pose_is_identity_transform():
    frame = MissionFrame()
    frame.initialize(PoseXYYaw(x=0.0, y=0.0, yaw=0.0))
    assert frame.mission_to_odom(3.0, -2.0) == pytest.approx((3.0, -2.0))
    assert frame.odom_to_mission(3.0, -2.0) == pytest.approx((3.0, -2.0))


@pytest.mark.parametrize("start_yaw", [0.0, math.pi / 4, math.pi / 2, math.pi, -math.pi / 3, 2.9])
@pytest.mark.parametrize("start_xy", [(0.0, 0.0), (5.0, -3.0), (-10.0, 12.0)])
def test_mission_odom_round_trip_arbitrary_start_pose(start_xy, start_yaw):
    frame = MissionFrame()
    frame.initialize(PoseXYYaw(x=start_xy[0], y=start_xy[1], yaw=start_yaw))
    for x, y, yaw in [(0.0, 0.0, 0.0), (5.0, 2.0, 1.0), (-3.0, -7.0, -2.5)]:
        ox, oy, oyaw = frame.mission_to_odom(x, y, yaw)
        mx, my, myaw = frame.odom_to_mission(ox, oy, oyaw)
        assert mx == pytest.approx(x, abs=1e-9)
        assert my == pytest.approx(y, abs=1e-9)
        assert math.cos(myaw) == pytest.approx(math.cos(yaw), abs=1e-9)
        assert math.sin(myaw) == pytest.approx(math.sin(yaw), abs=1e-9)


def test_relative_goal_matches_odom_frame_goal_at_mission_start():
    """The whole point of the mission frame: a user goal (gx, gy) given in
    the robot's OWN frame at t=0 equals mission_to_odom(gx, gy) computed
    from that exact start pose -- CLAUDE.md/plan section 3.1's formula."""
    x0, y0, yaw0 = 2.0, -1.0, 0.7
    gx, gy = 25.0, 10.0
    frame = MissionFrame()
    frame.initialize(PoseXYYaw(x=x0, y=y0, yaw=yaw0))
    odom_goal = frame.mission_to_odom(gx, gy)
    expected_x = x0 + math.cos(yaw0) * gx - math.sin(yaw0) * gy
    expected_y = y0 + math.sin(yaw0) * gx + math.cos(yaw0) * gy
    assert odom_goal[0] == pytest.approx(expected_x)
    assert odom_goal[1] == pytest.approx(expected_y)


def test_goal_stays_fixed_in_mission_frame_after_robot_moves_and_rotates():
    frame = MissionFrame()
    frame.initialize(PoseXYYaw(x=0.0, y=0.0, yaw=0.0))
    goal_mission = (10.0, 0.0)

    pose_a = PoseXYYaw(x=0.0, y=0.0, yaw=0.0)
    pose_b = PoseXYYaw(x=3.0, y=4.0, yaw=math.pi / 2)

    # The mission-frame goal coordinate itself never changes...
    assert goal_mission == (10.0, 0.0)
    # ...even though its robot-relative bearing/distance does, as the robot
    # moves and rotates.
    robot_view_a = frame.mission_to_robot(goal_mission, pose_a)
    robot_view_b = frame.mission_to_robot(goal_mission, pose_b)
    assert robot_view_a != pytest.approx(robot_view_b)
    assert robot_view_a == pytest.approx((10.0, 0.0))


def test_mission_to_robot_round_trip():
    pose = PoseXYYaw(x=1.5, y=-2.5, yaw=1.234)
    point_mission = (7.0, -3.0)
    robot_pt = MissionFrame.mission_to_robot(point_mission, pose)
    back = MissionFrame.robot_to_mission(robot_pt, pose)
    assert back[0] == pytest.approx(point_mission[0])
    assert back[1] == pytest.approx(point_mission[1])


def test_reset_clears_initialization():
    frame = MissionFrame()
    frame.initialize(PoseXYYaw(x=1.0, y=1.0, yaw=0.0))
    assert frame.initialized
    frame.reset()
    assert not frame.initialized
    with pytest.raises(MissionFrameError):
        frame.origin


def test_reinitialize_without_reset_raises():
    """Code review fix (item 8): a second initialize() call must not
    silently overwrite the origin mid-mission -- an explicit reset() is
    required first."""
    frame = MissionFrame()
    frame.initialize(PoseXYYaw(x=1.0, y=1.0, yaw=0.0))
    with pytest.raises(MissionFrameError):
        frame.initialize(PoseXYYaw(x=99.0, y=99.0, yaw=0.0))
    # The original origin must be unchanged after the rejected re-init.
    assert frame.origin == PoseXYYaw(x=1.0, y=1.0, yaw=0.0)


def test_reset_then_reinitialize_succeeds():
    frame = MissionFrame()
    frame.initialize(PoseXYYaw(x=1.0, y=1.0, yaw=0.0))
    frame.reset()
    frame.initialize(PoseXYYaw(x=2.0, y=2.0, yaw=0.0))
    assert frame.origin == PoseXYYaw(x=2.0, y=2.0, yaw=0.0)


@pytest.mark.parametrize("field", ["x", "y", "yaw"])
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_initialize_rejects_non_finite_start_pose(field, bad_value):
    """Code review fix (item 3): a NaN/Inf start pose must never anchor the
    whole mission's coordinate frame."""
    kwargs = dict(x=0.0, y=0.0, yaw=0.0)
    kwargs[field] = bad_value
    frame = MissionFrame()
    with pytest.raises(ValueError):
        frame.initialize(PoseXYYaw(**kwargs))
    assert not frame.initialized
