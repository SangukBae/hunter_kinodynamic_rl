import math

import pytest

from hunter_kinodynamic_rl.navigation.mission.goal_manager import GoalManager, GoalManagerConfig, MissionStatus
from hunter_kinodynamic_rl.navigation.mission.mission_frame import PoseXYYaw


def _manager(**overrides):
    cfg = GoalManagerConfig(
        position_tolerance_m=0.5, heading_tolerance_rad=math.pi,
        require_low_speed_on_goal=False, goal_speed_threshold_mps=0.1,
    )
    for k, v in overrides.items():
        cfg = GoalManagerConfig(**{**cfg.__dict__, k: v})
    return GoalManager(cfg)


def test_relative_goal_before_set_raises():
    gm = _manager()
    with pytest.raises(RuntimeError):
        gm.relative_goal
    assert gm.status == MissionStatus.INACTIVE


def test_set_goal_activates_mission_and_stores_goal():
    gm = _manager()
    reset_needed = gm.set_goal(5.0, -2.0)
    assert reset_needed is False  # first set from INACTIVE is not a "change"
    assert gm.status == MissionStatus.ACTIVE
    assert gm.relative_goal == (5.0, -2.0)
    assert gm.mission_goal == (5.0, -2.0)


def test_set_goal_change_while_active_signals_memory_reset_by_default():
    gm = _manager()
    gm.set_goal(5.0, 0.0)
    reset_needed = gm.set_goal(1.0, 1.0)
    assert reset_needed is True
    assert gm.mission_goal == (1.0, 1.0)


def test_set_goal_change_respects_reset_memory_flag_disabled():
    gm = GoalManager(GoalManagerConfig(), reset_memory_on_goal_change=False)
    gm.set_goal(5.0, 0.0)
    assert gm.set_goal(1.0, 1.0) is False


def test_check_reached_no_op_when_inactive():
    gm = _manager()
    assert gm.check_reached(PoseXYYaw(0.0, 0.0, 0.0)) is False


def test_check_reached_within_position_tolerance():
    gm = _manager()
    gm.set_goal(1.0, 0.0)
    assert gm.check_reached(PoseXYYaw(x=0.9, y=0.0, yaw=0.0)) is True
    assert gm.status == MissionStatus.REACHED


def test_check_reached_outside_position_tolerance_stays_active():
    gm = _manager()
    gm.set_goal(10.0, 0.0)
    assert gm.check_reached(PoseXYYaw(x=0.0, y=0.0, yaw=0.0)) is False
    assert gm.status == MissionStatus.ACTIVE


def test_check_reached_respects_heading_tolerance():
    gm = _manager(heading_tolerance_rad=0.1)
    gm.set_goal(1.0, 0.0)
    # Robot is at the goal position but facing 90 degrees off from the goal bearing.
    assert gm.check_reached(PoseXYYaw(x=0.95, y=0.0, yaw=math.pi / 2)) is False


def test_check_reached_requires_low_speed_when_configured():
    gm = _manager(require_low_speed_on_goal=True, goal_speed_threshold_mps=0.1)
    gm.set_goal(1.0, 0.0)
    assert gm.check_reached(PoseXYYaw(x=0.95, y=0.0, yaw=0.0), speed_mps=0.5) is False
    assert gm.check_reached(PoseXYYaw(x=0.95, y=0.0, yaw=0.0), speed_mps=0.05) is True


def test_check_reached_once_reached_further_calls_are_noop():
    gm = _manager()
    gm.set_goal(1.0, 0.0)
    assert gm.check_reached(PoseXYYaw(x=0.9, y=0.0, yaw=0.0)) is True
    assert gm.status == MissionStatus.REACHED
    # A now-active-elsewhere pose must not resurrect the mission.
    assert gm.check_reached(PoseXYYaw(x=0.9, y=0.0, yaw=0.0)) is False


def test_cancel_sets_cancelled_status_and_blocks_further_reach():
    gm = _manager()
    gm.set_goal(1.0, 0.0)
    gm.cancel()
    assert gm.status == MissionStatus.CANCELLED
    assert gm.check_reached(PoseXYYaw(x=1.0, y=0.0, yaw=0.0)) is False


def test_distance_and_bearing_changes_with_robot_pose_goal_itself_fixed():
    gm = _manager()
    gm.set_goal(10.0, 0.0)
    d1, b1 = gm.distance_and_bearing(PoseXYYaw(x=0.0, y=0.0, yaw=0.0))
    d2, b2 = gm.distance_and_bearing(PoseXYYaw(x=5.0, y=0.0, yaw=0.0))
    assert d1 == pytest.approx(10.0)
    assert d2 == pytest.approx(5.0)
    assert gm.mission_goal == (10.0, 0.0)
