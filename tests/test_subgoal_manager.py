import math

import pytest

from hunter_kinodynamic_rl.navigation.hierarchy.subgoal_manager import (
    FAILURE_SUBGOAL_STATUSES, TERMINAL_SUBGOAL_STATUSES, SubgoalManager, SubgoalManagerConfig, SubgoalStatus,
)
from hunter_kinodynamic_rl.navigation.mission.mission_frame import PoseXYYaw


def _manager(**overrides) -> SubgoalManager:
    cfg = SubgoalManagerConfig(
        position_tolerance_m=0.5, heading_tolerance_rad=math.pi,
        require_low_speed_on_reach=False, goal_speed_threshold_mps=0.15,
    )
    for k, v in overrides.items():
        cfg = SubgoalManagerConfig(**{**cfg.__dict__, k: v})
    return SubgoalManager(cfg)


def test_subgoal_mission_xy_before_activate_raises():
    sm = _manager()
    with pytest.raises(RuntimeError):
        sm.subgoal_mission_xy
    assert sm.status is None
    assert sm.active is False


def test_activate_sets_active_and_stores_subgoal():
    sm = _manager()
    sid = sm.activate(3.0, 1.0, now_step=0, now_time_sec=0.0, final_goal_distance_m=10.0, subgoal_distance_m=3.16)
    assert sid == 1
    assert sm.active is True
    assert sm.status == SubgoalStatus.ACTIVE
    assert sm.subgoal_mission_xy == (3.0, 1.0)
    assert sm.local_steps == 0


def test_activate_while_active_raises():
    sm = _manager()
    sm.activate(3.0, 0.0, now_step=0, now_time_sec=0.0, final_goal_distance_m=10.0, subgoal_distance_m=3.0)
    with pytest.raises(RuntimeError):
        sm.activate(4.0, 0.0, now_step=1, now_time_sec=0.1, final_goal_distance_m=10.0, subgoal_distance_m=4.0)


def test_check_reached_false_when_not_active():
    sm = _manager()
    assert sm.check_reached(PoseXYYaw(0.0, 0.0, 0.0)) is False


def test_check_reached_within_position_tolerance():
    sm = _manager()
    sm.activate(1.0, 0.0, now_step=0, now_time_sec=0.0, final_goal_distance_m=5.0, subgoal_distance_m=1.0)
    assert sm.check_reached(PoseXYYaw(x=0.9, y=0.0, yaw=0.0)) is True


def test_check_reached_outside_position_tolerance():
    sm = _manager()
    sm.activate(5.0, 0.0, now_step=0, now_time_sec=0.0, final_goal_distance_m=10.0, subgoal_distance_m=5.0)
    assert sm.check_reached(PoseXYYaw(x=0.0, y=0.0, yaw=0.0)) is False


def test_check_reached_respects_heading_tolerance():
    sm = _manager(heading_tolerance_rad=0.1)
    sm.activate(1.0, 0.0, now_step=0, now_time_sec=0.0, final_goal_distance_m=5.0, subgoal_distance_m=1.0)
    assert sm.check_reached(PoseXYYaw(x=0.95, y=0.0, yaw=math.pi / 2)) is False


def test_check_reached_requires_low_speed_when_configured():
    sm = _manager(require_low_speed_on_reach=True, goal_speed_threshold_mps=0.1)
    sm.activate(1.0, 0.0, now_step=0, now_time_sec=0.0, final_goal_distance_m=5.0, subgoal_distance_m=1.0)
    assert sm.check_reached(PoseXYYaw(x=0.95, y=0.0, yaw=0.0), speed_mps=0.5) is False
    assert sm.check_reached(PoseXYYaw(x=0.95, y=0.0, yaw=0.0), speed_mps=0.05) is True


def test_check_reached_rejects_nan_speed_when_low_speed_required():
    """A NaN/Inf speed reading must fail-toward-REJECTING reached, mirroring
    GoalManager's own round-3 NaN-speed regression fix."""
    sm = _manager(require_low_speed_on_reach=True, goal_speed_threshold_mps=0.1)
    sm.activate(1.0, 0.0, now_step=0, now_time_sec=0.0, final_goal_distance_m=5.0, subgoal_distance_m=1.0)
    assert sm.check_reached(PoseXYYaw(x=0.95, y=0.0, yaw=0.0), speed_mps=float("nan")) is False


def test_record_tick_while_not_active_raises():
    sm = _manager()
    with pytest.raises(RuntimeError):
        sm.record_tick(step_delta_m=0.1, final_goal_distance_m=1.0, subgoal_distance_m=1.0)


def test_finish_while_not_active_raises():
    sm = _manager()
    with pytest.raises(RuntimeError):
        sm.finish(SubgoalStatus.REACHED, "x", now_step=0, now_time_sec=0.0)


def test_finish_rejects_non_terminal_status():
    sm = _manager()
    sm.activate(1.0, 0.0, now_step=0, now_time_sec=0.0, final_goal_distance_m=5.0, subgoal_distance_m=1.0)
    with pytest.raises(ValueError):
        sm.finish(SubgoalStatus.ACTIVE, "x", now_step=1, now_time_sec=0.1)


def test_finish_accumulates_every_required_stat_field():
    sm = _manager()
    sm.activate(5.0, 0.0, now_step=0, now_time_sec=0.0, final_goal_distance_m=20.0, subgoal_distance_m=5.0)
    sm.record_tick(
        step_delta_m=1.0, final_goal_distance_m=19.0, subgoal_distance_m=4.0,
        clearance_m=0.8, predicted_risk=0.2, emergency_stop=False, steering_saturated=False,
        newly_explored_cells=10,
    )
    sm.record_tick(
        step_delta_m=1.0, final_goal_distance_m=18.0, subgoal_distance_m=3.0,
        clearance_m=0.3, predicted_risk=0.6, emergency_stop=True, steering_saturated=True,
        newly_explored_cells=5,
    )
    result = sm.finish(SubgoalStatus.REACHED, "subgoal_tolerance_reached", now_step=10, now_time_sec=1.0)

    assert result.subgoal_id == 1
    assert result.status == SubgoalStatus.REACHED
    assert result.reason == "subgoal_tolerance_reached"
    assert result.local_steps == 2
    assert result.elapsed_time_sec == pytest.approx(1.0)
    assert result.path_length_m == pytest.approx(2.0)
    assert result.start_final_goal_distance_m == pytest.approx(20.0)
    assert result.end_final_goal_distance_m == pytest.approx(18.0)
    assert result.start_subgoal_distance_m == pytest.approx(5.0)
    assert result.end_subgoal_distance_m == pytest.approx(3.0)
    assert result.minimum_clearance_m == pytest.approx(0.3)
    assert result.mean_predicted_risk == pytest.approx(0.4)
    assert result.max_predicted_risk == pytest.approx(0.6)
    assert result.emergency_stop_count == 1
    assert result.steering_saturation_count == 1
    assert result.newly_explored_cells == 15
    assert sm.status == SubgoalStatus.REACHED
    assert sm.active is False


def test_finish_without_any_clearance_or_risk_samples_reports_none():
    """The plan's explicit '없으면 None' contract -- never a fabricated
    0.0/inf sentinel when the caller never supplied a sample."""
    sm = _manager()
    sm.activate(2.0, 0.0, now_step=0, now_time_sec=0.0, final_goal_distance_m=10.0, subgoal_distance_m=2.0)
    sm.record_tick(step_delta_m=0.5, final_goal_distance_m=9.5, subgoal_distance_m=1.5)
    result = sm.finish(SubgoalStatus.FAILED_TIMEOUT, "local_option_timeout", now_step=5, now_time_sec=0.5)
    assert result.minimum_clearance_m is None
    assert result.mean_predicted_risk is None
    assert result.max_predicted_risk is None


def test_record_tick_ignores_negative_step_delta():
    sm = _manager()
    sm.activate(2.0, 0.0, now_step=0, now_time_sec=0.0, final_goal_distance_m=10.0, subgoal_distance_m=2.0)
    sm.record_tick(step_delta_m=-1.0, final_goal_distance_m=10.0, subgoal_distance_m=2.0)
    result = sm.finish(SubgoalStatus.FAILED_BLOCKED, "x", now_step=1, now_time_sec=0.1)
    assert result.path_length_m == pytest.approx(0.0)


@pytest.mark.parametrize("status,reason", [
    (SubgoalStatus.FAILED_BLOCKED, "subgoal_endpoint_occupied_or_inflated"),
    (SubgoalStatus.FAILED_TIMEOUT, "local_option_timeout"),
    (SubgoalStatus.FAILED_NO_PROGRESS, "no_progress_within_window"),
    (SubgoalStatus.FAILED_HIGH_RISK, "local_risk_threshold_exceeded"),
    (SubgoalStatus.CANCELLED_BY_REPLAN, "localization_confidence_degraded"),
])
def test_every_failure_and_cancel_status_is_a_distinct_terminal_reason(status, reason):
    sm = _manager()
    sm.activate(1.0, 0.0, now_step=0, now_time_sec=0.0, final_goal_distance_m=5.0, subgoal_distance_m=1.0)
    result = sm.finish(status, reason, now_step=1, now_time_sec=0.1)
    assert result.status == status
    assert result.reason == reason
    assert status in TERMINAL_SUBGOAL_STATUSES
    if status != SubgoalStatus.CANCELLED_BY_REPLAN:
        assert status in FAILURE_SUBGOAL_STATUSES


def test_reached_is_not_in_failure_statuses():
    assert SubgoalStatus.REACHED not in FAILURE_SUBGOAL_STATUSES
    assert SubgoalStatus.CANCELLED_BY_REPLAN not in FAILURE_SUBGOAL_STATUSES


def test_reactivate_after_finish_starts_a_new_subgoal_id():
    sm = _manager()
    sm.activate(1.0, 0.0, now_step=0, now_time_sec=0.0, final_goal_distance_m=5.0, subgoal_distance_m=1.0)
    sm.finish(SubgoalStatus.FAILED_TIMEOUT, "local_option_timeout", now_step=1, now_time_sec=0.1)
    sid2 = sm.activate(2.0, 0.0, now_step=1, now_time_sec=0.1, final_goal_distance_m=6.0, subgoal_distance_m=2.0)
    assert sid2 == 2
    assert sm.subgoal_mission_xy == (2.0, 0.0)
    assert sm.local_steps == 0
