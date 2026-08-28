import math

import pytest

from hunter_kinodynamic_rl.navigation.hierarchy.coordinator import HierarchyCoordinator, HierarchyCoordinatorConfig
from hunter_kinodynamic_rl.navigation.hierarchy.failure_recovery import FailureRecoveryConfig
from hunter_kinodynamic_rl.navigation.hierarchy.replanning import ReplanningConfig
from hunter_kinodynamic_rl.navigation.hierarchy.subgoal_manager import SubgoalManagerConfig, SubgoalStatus
from hunter_kinodynamic_rl.navigation.mission.goal_manager import GoalManagerConfig
from hunter_kinodynamic_rl.navigation.mission.mission_frame import PoseXYYaw


def _coordinator(**overrides) -> HierarchyCoordinator:
    cfg = HierarchyCoordinatorConfig(
        goal=GoalManagerConfig(position_tolerance_m=0.5, require_low_speed_on_goal=False),
        subgoal=SubgoalManagerConfig(position_tolerance_m=0.4, require_low_speed_on_reach=False),
        replanning=ReplanningConfig(
            local_option_timeout_steps=80, no_progress_window_steps=8, no_progress_min_delta_m=0.05,
            consecutive_emergency_stop_limit=5, local_risk_threshold=0.8, localization_min_confidence=0.3,
        ),
        recovery=FailureRecoveryConfig(max_retries_per_subgoal=1),
    )
    for k, v in overrides.items():
        cfg = HierarchyCoordinatorConfig(**{**cfg.__dict__, k: v})
    return HierarchyCoordinator(cfg)


def _drive_straight_x(coordinator, start_x, target_x, *, max_steps=200, **tick_kwargs):
    """Steps the robot in a straight line along +x/-x from ``start_x``
    toward ``target_x`` in fixed increments, calling record_local_tick each
    step, until the coordinator reports subgoal_reached/subgoal_failed/
    mission_reached/mission_done or max_steps is exhausted. Returns the
    final pose and the step count actually taken."""
    direction = 1.0 if target_x >= start_x else -1.0
    x = start_x
    step_size = 0.1
    for i in range(1, max_steps + 1):
        x = x + direction * step_size if abs(target_x - x) > step_size else target_x
        pose = PoseXYYaw(x=x, y=0.0, yaw=0.0)
        coordinator.record_local_tick(pose, now_step=i, now_time_sec=i * 0.1, **tick_kwargs)
        if coordinator.mission_done or coordinator.subgoal_reached or coordinator.subgoal_failed:
            return pose, i
    return PoseXYYaw(x=x, y=0.0, yaw=0.0), max_steps


# ---------------------------------------------------------------- basic lifecycle

def test_stop_required_before_any_subgoal_is_activated():
    c = _coordinator()
    c.start_mission(10.0, 0.0)
    assert c.stop_required is True


def test_activate_first_subgoal_clears_stop_required():
    c = _coordinator()
    c.start_mission(10.0, 0.0)
    c.enqueue_subgoal(3.0, 0.0)
    ok = c.activate_next_subgoal(PoseXYYaw(0.0, 0.0, 0.0), now_step=0, now_time_sec=0.0)
    assert ok is True
    assert c.stop_required is False
    assert c.active_subgoal_mission == (3.0, 0.0)
    assert c.final_goal_mission == (10.0, 0.0)


def test_record_local_tick_raises_when_stop_required():
    c = _coordinator()
    c.start_mission(10.0, 0.0)
    with pytest.raises(RuntimeError):
        c.record_local_tick(PoseXYYaw(0.0, 0.0, 0.0), now_step=0, now_time_sec=0.0)


def test_start_mission_clears_a_still_active_subgoal_from_the_previous_mission():
    """code review finding: restarting the mission while the PREVIOUS
    mission's subgoal was still ACTIVE (never reached/failed/cancelled)
    used to leave that stale subgoal active -- a brand new mission's local
    policy could be handed a subgoal belonging to an entirely different
    mission, and stop_required would stay False across the restart."""
    c = _coordinator()
    c.start_mission(10.0, 0.0)
    c.enqueue_subgoal(3.0, 0.0)
    c.activate_next_subgoal(PoseXYYaw(0.0, 0.0, 0.0), now_step=0, now_time_sec=0.0)
    assert c.active_subgoal_mission == (3.0, 0.0)
    assert c.stop_required is False

    c.start_mission(20.0, 5.0, now_step=100, now_time_sec=10.0)
    assert c.active_subgoal_mission is None
    assert c.stop_required is True
    assert c.subgoal_manager.status is None
    assert c.subgoal_manager.active is False
    assert c.final_goal_mission == (20.0, 5.0)
    # No stale distance bookkeeping survives the restart either.
    assert c.previous_final_goal_distance == pytest.approx(0.0)
    assert c.previous_subgoal_distance == pytest.approx(0.0)


def test_start_mission_clears_a_terminated_subgoal_from_the_previous_mission():
    """Same guarantee when the previous mission ended cleanly (subgoal
    already REACHED, not ACTIVE) -- reset() must still be safe/idempotent
    and leave no readable subgoal behind for the new mission."""
    c = _coordinator()
    c.start_mission(3.0, 0.0)
    c.enqueue_subgoal(3.0, 0.0)
    c.activate_next_subgoal(PoseXYYaw(0.0, 0.0, 0.0), now_step=0, now_time_sec=0.0)
    _drive_straight_x(c, 0.0, 3.0)
    assert c.mission_reached is True

    c.start_mission(1.0, 1.0)
    assert c.active_subgoal_mission is None
    assert c.stop_required is True
    assert c.mission_reached is False


# --------------------------------------------------- subgoal reached != mission reached

def test_subgoal_reached_does_not_end_the_mission():
    c = _coordinator()
    c.start_mission(10.0, 0.0)
    c.enqueue_subgoal(3.0, 0.0)
    c.activate_next_subgoal(PoseXYYaw(0.0, 0.0, 0.0), now_step=0, now_time_sec=0.0)
    _pose, _step = _drive_straight_x(c, 0.0, 3.0)

    assert c.subgoal_reached is True
    assert c.subgoal_manager.status == SubgoalStatus.REACHED
    assert c.last_subgoal_result.status == SubgoalStatus.REACHED
    # The critical assertion: reaching a SUBGOAL must never be mistaken for
    # mission success.
    assert c.mission_reached is False
    assert c.mission_done is False
    assert c.stop_required is True  # awaiting the next subgoal


def test_only_final_goal_reach_sets_mission_reached():
    c = _coordinator()
    c.start_mission(3.0, 0.0)
    c.enqueue_subgoal(3.0, 0.0)  # subgoal coincides with the final goal this time
    c.activate_next_subgoal(PoseXYYaw(0.0, 0.0, 0.0), now_step=0, now_time_sec=0.0)
    _pose, _step = _drive_straight_x(c, 0.0, 3.0)

    assert c.mission_reached is True
    assert c.mission_done is True
    # Reaching the final goal also finishes the currently-active subgoal
    # (as REACHED) -- but the flag that actually terminates the MISSION is
    # mission_reached, backed by GoalManager, never subgoal_reached alone.
    assert c.subgoal_reached is True
    assert c.last_subgoal_result.reason == "final_goal_reached_during_subgoal"


def test_mission_never_reached_from_an_intermediate_subgoal_far_from_final_goal():
    c = _coordinator()
    c.start_mission(100.0, 0.0)
    c.enqueue_subgoal(3.0, 0.0)
    c.activate_next_subgoal(PoseXYYaw(0.0, 0.0, 0.0), now_step=0, now_time_sec=0.0)
    _pose, _step = _drive_straight_x(c, 0.0, 3.0)
    assert c.subgoal_reached is True
    assert c.mission_reached is False


# ------------------------------------------------------- multi-subgoal sequencing

def test_consecutive_subgoal_sequence_runs_to_completion():
    """Completion criterion: Global RL 없이 정해진 subgoal sequence를 여러 개
    연속 실행."""
    c = _coordinator()
    c.start_mission(9.0, 0.0)
    c.enqueue_subgoal(3.0, 0.0)
    c.enqueue_subgoal(6.0, 0.0)
    c.enqueue_subgoal(9.0, 0.0)
    c.activate_next_subgoal(PoseXYYaw(0.0, 0.0, 0.0), now_step=0, now_time_sec=0.0)

    pose = PoseXYYaw(0.0, 0.0, 0.0)
    step = 0
    reached_subgoals = 0
    for _ in range(3):
        pose, step = _drive_straight_x(c, pose.x, c.active_subgoal_mission[0], max_steps=100)
        assert c.subgoal_reached is True
        reached_subgoals += 1
        if c.mission_reached:
            break
        c.activate_next_subgoal(pose, now_step=step, now_time_sec=step * 0.1)

    assert reached_subgoals == 3
    assert c.mission_reached is True
    assert c.mission_done is True


# ------------------------------------------------------------- replanning triggers

def test_timeout_trigger_marks_subgoal_failed_with_timeout_reason():
    c = _coordinator(replanning=ReplanningConfig(local_option_timeout_steps=5, no_progress_window_steps=100))
    c.start_mission(100.0, 0.0)
    c.enqueue_subgoal(3.0, 0.0)
    c.activate_next_subgoal(PoseXYYaw(0.0, 0.0, 0.0), now_step=0, now_time_sec=0.0)
    for i in range(1, 6):
        if c.stop_required:
            break
        c.record_local_tick(PoseXYYaw(0.0, 0.0, 0.0), now_step=i, now_time_sec=i * 0.1)  # never moves
        if c.subgoal_failed:
            assert c.last_subgoal_result.status == SubgoalStatus.FAILED_TIMEOUT
            assert c.last_subgoal_result.reason == "local_option_timeout"
            return
    pytest.fail("expected a FAILED_TIMEOUT trigger within the configured timeout budget")


def test_blocked_trigger_marks_subgoal_failed_with_blocked_reason():
    c = _coordinator(recovery=FailureRecoveryConfig(max_retries_per_subgoal=0))
    c.start_mission(100.0, 0.0)
    c.enqueue_subgoal(3.0, 0.0)
    c.activate_next_subgoal(PoseXYYaw(0.0, 0.0, 0.0), now_step=0, now_time_sec=0.0)
    c.record_local_tick(PoseXYYaw(0.1, 0.0, 0.0), now_step=1, now_time_sec=0.1, subgoal_endpoint_blocked=True)
    assert c.subgoal_failed is True
    assert c.last_subgoal_result.status == SubgoalStatus.FAILED_BLOCKED
    assert c.last_subgoal_result.reason == "subgoal_endpoint_occupied_or_inflated"


def test_no_progress_trigger_marks_subgoal_failed_with_no_progress_reason():
    c = _coordinator(replanning=ReplanningConfig(
        local_option_timeout_steps=1000, no_progress_window_steps=5, no_progress_min_delta_m=0.2,
    ))
    c.start_mission(100.0, 0.0)
    c.enqueue_subgoal(3.0, 0.0)
    c.activate_next_subgoal(PoseXYYaw(0.0, 0.0, 0.0), now_step=0, now_time_sec=0.0)
    for i in range(1, 10):
        if c.stop_required:
            break
        # Crawl forward far too slowly to ever satisfy no_progress_min_delta_m
        # across the window.
        c.record_local_tick(PoseXYYaw(0.001 * i, 0.0, 0.0), now_step=i, now_time_sec=i * 0.1)
        if c.subgoal_failed:
            assert c.last_subgoal_result.status == SubgoalStatus.FAILED_NO_PROGRESS
            assert c.last_subgoal_result.reason == "no_progress_within_window"
            return
    pytest.fail("expected a FAILED_NO_PROGRESS trigger")


def test_high_risk_trigger_marks_subgoal_failed_with_high_risk_reason():
    c = _coordinator(recovery=FailureRecoveryConfig(max_retries_per_subgoal=0))
    c.start_mission(100.0, 0.0)
    c.enqueue_subgoal(3.0, 0.0)
    c.activate_next_subgoal(PoseXYYaw(0.0, 0.0, 0.0), now_step=0, now_time_sec=0.0)
    c.record_local_tick(PoseXYYaw(0.1, 0.0, 0.0), now_step=1, now_time_sec=0.1, predicted_risk=0.95)
    assert c.subgoal_failed is True
    assert c.last_subgoal_result.status == SubgoalStatus.FAILED_HIGH_RISK
    assert c.last_subgoal_result.reason == "local_risk_threshold_exceeded"


def test_localization_confidence_trigger_cancels_by_replan():
    c = _coordinator(recovery=FailureRecoveryConfig(max_retries_per_subgoal=0))
    c.start_mission(100.0, 0.0)
    c.enqueue_subgoal(3.0, 0.0)
    c.activate_next_subgoal(PoseXYYaw(0.0, 0.0, 0.0), now_step=0, now_time_sec=0.0)
    c.record_local_tick(PoseXYYaw(0.1, 0.0, 0.0), now_step=1, now_time_sec=0.1, localization_confidence=0.05)
    assert c.subgoal_failed is True
    assert c.last_subgoal_result.status == SubgoalStatus.CANCELLED_BY_REPLAN
    assert c.last_subgoal_result.reason == "localization_confidence_degraded"


def test_localization_degraded_blocks_physical_motion_even_with_a_candidate_still_queued():
    """code review (round 2) finding: a degraded-confidence cancel used to
    call the SAME recovery path as every other failure -- if the queue
    already had a next candidate, it was activated IMMEDIATELY in the same
    tick, handing the local policy a fresh subgoal (stop_required back to
    False) while localization was STILL reporting degraded confidence.
    "Never confirm success from a bad pose" is not the same guarantee as
    "no physical motion while degraded"; this test is the latter."""
    c = _coordinator(recovery=FailureRecoveryConfig(max_retries_per_subgoal=0))
    c.start_mission(100.0, 0.0)
    c.enqueue_subgoal(3.0, 0.0)
    c.enqueue_subgoal(5.0, 0.0)  # a candidate is READY -- must still not be handed out
    c.activate_next_subgoal(PoseXYYaw(0.0, 0.0, 0.0), now_step=0, now_time_sec=0.0)
    c.record_local_tick(PoseXYYaw(0.1, 0.0, 0.0), now_step=1, now_time_sec=0.1, localization_confidence=0.05)

    assert c.localization_degraded is True
    assert c.active_subgoal_mission is None
    assert c.stop_required is True
    assert c.mission_done is False  # not aborted either -- just paused
    assert c.pending_subgoal_count == 1  # the queued candidate was never touched


def test_localization_degraded_pause_clears_once_confidence_recovers():
    c = _coordinator(recovery=FailureRecoveryConfig(max_retries_per_subgoal=0))
    c.start_mission(100.0, 0.0)
    c.enqueue_subgoal(3.0, 0.0)
    c.enqueue_subgoal(5.0, 0.0)
    c.activate_next_subgoal(PoseXYYaw(0.0, 0.0, 0.0), now_step=0, now_time_sec=0.0)
    c.record_local_tick(PoseXYYaw(0.1, 0.0, 0.0), now_step=1, now_time_sec=0.1, localization_confidence=0.05)
    assert c.localization_degraded is True

    # Still degraded -- must stay refused.
    still_paused = c.activate_next_subgoal(
        PoseXYYaw(0.1, 0.0, 0.0), now_step=2, now_time_sec=0.2, localization_confidence=0.1,
    )
    assert still_paused is False
    assert c.localization_degraded is True
    assert c.stop_required is True

    # Confidence recovers -- the NEXT queued candidate is activated.
    ok = c.activate_next_subgoal(
        PoseXYYaw(0.1, 0.0, 0.0), now_step=3, now_time_sec=0.3, localization_confidence=0.9,
    )
    assert ok is True
    assert c.localization_degraded is False
    assert c.stop_required is False
    assert c.active_subgoal_mission == (5.0, 0.0)


def test_localization_degraded_pause_still_bounded_by_mission_timeout():
    """A localization that never recovers must not stall the mission
    forever -- the mission timeout still applies while paused, checked on
    each activate_next_subgoal() poll (record_local_tick can't run at all
    while stop_required is True)."""
    c = _coordinator(mission_timeout_steps=5)
    c.start_mission(100.0, 0.0, now_step=0, now_time_sec=0.0)
    c.enqueue_subgoal(3.0, 0.0)
    c.activate_next_subgoal(PoseXYYaw(0.0, 0.0, 0.0), now_step=0, now_time_sec=0.0)
    c.record_local_tick(PoseXYYaw(0.1, 0.0, 0.0), now_step=1, now_time_sec=0.1, localization_confidence=0.05)
    assert c.localization_degraded is True

    ok = c.activate_next_subgoal(
        PoseXYYaw(0.1, 0.0, 0.0), now_step=6, now_time_sec=0.6, localization_confidence=0.05,
    )
    assert ok is False
    assert c.mission_timed_out is True
    assert c.mission_done is True


def test_nan_confidence_pose_inside_final_goal_tolerance_never_confirms_mission_success():
    """NaN confidence must fail toward degraded localization, matching the
    Phase 1/GoalManager lesson that NaN comparisons are never safe."""
    c = _coordinator(recovery=FailureRecoveryConfig(max_retries_per_subgoal=0))
    c.start_mission(3.0, 0.0)
    c.enqueue_subgoal(3.0, 0.0)
    c.enqueue_subgoal(5.0, 0.0)
    c.activate_next_subgoal(PoseXYYaw(0.0, 0.0, 0.0), now_step=0, now_time_sec=0.0)

    c.record_local_tick(
        PoseXYYaw(3.0, 0.0, 0.0), now_step=1, now_time_sec=0.1,
        localization_confidence=float("nan"),
    )

    assert c.mission_reached is False
    assert c.localization_degraded is True
    assert c.active_subgoal_mission is None
    assert c.stop_required is True
    assert c.pending_subgoal_count == 1
    assert c.last_subgoal_result.status == SubgoalStatus.CANCELLED_BY_REPLAN
    assert c.last_subgoal_result.reason == "localization_confidence_degraded"


def test_localization_degraded_pause_does_not_clear_on_nan_confidence():
    c = _coordinator(recovery=FailureRecoveryConfig(max_retries_per_subgoal=0))
    c.start_mission(100.0, 0.0)
    c.enqueue_subgoal(3.0, 0.0)
    c.enqueue_subgoal(5.0, 0.0)
    c.activate_next_subgoal(PoseXYYaw(0.0, 0.0, 0.0), now_step=0, now_time_sec=0.0)
    c.record_local_tick(PoseXYYaw(0.1, 0.0, 0.0), now_step=1, now_time_sec=0.1, localization_confidence=0.05)
    assert c.localization_degraded is True

    ok = c.activate_next_subgoal(
        PoseXYYaw(0.1, 0.0, 0.0), now_step=2, now_time_sec=0.2,
        localization_confidence=float("nan"),
    )

    assert ok is False
    assert c.localization_degraded is True
    assert c.active_subgoal_mission is None
    assert c.stop_required is True
    assert c.pending_subgoal_count == 1


def test_initial_activation_with_low_confidence_is_refused_and_queue_preserved():
    c = _coordinator()
    c.start_mission(100.0, 0.0)
    c.enqueue_subgoal(3.0, 0.0)

    ok = c.activate_next_subgoal(
        PoseXYYaw(0.0, 0.0, 0.0), now_step=0, now_time_sec=0.0,
        localization_confidence=0.05,
    )

    assert ok is False
    assert c.localization_degraded is True
    assert c.active_subgoal_mission is None
    assert c.stop_required is True
    assert c.pending_subgoal_count == 1


def test_initial_activation_with_nan_confidence_is_refused_and_queue_preserved():
    c = _coordinator()
    c.start_mission(100.0, 0.0)
    c.enqueue_subgoal(3.0, 0.0)

    ok = c.activate_next_subgoal(
        PoseXYYaw(0.0, 0.0, 0.0), now_step=0, now_time_sec=0.0,
        localization_confidence=float("nan"),
    )

    assert ok is False
    assert c.localization_degraded is True
    assert c.active_subgoal_mission is None
    assert c.stop_required is True
    assert c.pending_subgoal_count == 1


def test_low_confidence_pose_inside_final_goal_tolerance_never_confirms_mission_success():
    """code review finding: a degraded-confidence pose that happens to
    fall INSIDE the final goal's tolerance radius must never be allowed
    to confirm mission_reached (GPS-denied/drifting localization could
    otherwise report a false "arrival" purely from noise). Must cancel by
    replan (and pause, see the dedicated tests above) instead."""
    c = _coordinator(recovery=FailureRecoveryConfig(max_retries_per_subgoal=0))
    c.start_mission(3.0, 0.0)  # final goal
    c.enqueue_subgoal(3.0, 0.0)  # subgoal coincides with the final goal
    c.activate_next_subgoal(PoseXYYaw(0.0, 0.0, 0.0), now_step=0, now_time_sec=0.0)
    # Pose is WITHIN both the subgoal's and the final goal's tolerance --
    # would otherwise confirm mission_reached on this exact tick.
    c.record_local_tick(PoseXYYaw(3.0, 0.0, 0.0), now_step=1, now_time_sec=0.1, localization_confidence=0.05)

    assert c.mission_reached is False
    assert c.mission_done is False
    assert c.subgoal_reached is False
    assert c.subgoal_failed is True
    assert c.last_subgoal_result.status == SubgoalStatus.CANCELLED_BY_REPLAN
    assert c.last_subgoal_result.reason == "localization_confidence_degraded"
    assert c.stop_required is True


def test_confidence_at_or_above_minimum_does_not_block_a_genuine_final_goal_reach():
    """Sanity counterpart: with HEALTHY confidence, the exact same pose
    DOES confirm mission success -- proves the previous test's rejection
    is caused by the degraded confidence, not by some unrelated bug."""
    c = _coordinator()
    c.start_mission(3.0, 0.0)
    c.enqueue_subgoal(3.0, 0.0)
    c.activate_next_subgoal(PoseXYYaw(0.0, 0.0, 0.0), now_step=0, now_time_sec=0.0)
    c.record_local_tick(PoseXYYaw(3.0, 0.0, 0.0), now_step=1, now_time_sec=0.1, localization_confidence=0.9)
    assert c.mission_reached is True


def test_five_terminal_reasons_are_all_distinct():
    reasons = {
        "timeout": "local_option_timeout",
        "blocked": "subgoal_endpoint_occupied_or_inflated",
        "no_progress": "no_progress_within_window",
        "high_risk": "local_risk_threshold_exceeded",
        "cancelled_by_replan": "localization_confidence_degraded",
    }
    assert len(set(reasons.values())) == 5


# ------------------------------------------------------------------- recovery

def test_retry_same_subgoal_before_advancing_to_next():
    c = _coordinator(
        replanning=ReplanningConfig(local_option_timeout_steps=3, no_progress_window_steps=1000),
        recovery=FailureRecoveryConfig(max_retries_per_subgoal=1),
    )
    c.start_mission(100.0, 0.0)
    c.enqueue_subgoal(3.0, 0.0)
    c.enqueue_subgoal(6.0, 0.0)
    c.activate_next_subgoal(PoseXYYaw(0.0, 0.0, 0.0), now_step=0, now_time_sec=0.0)

    for i in range(1, 4):
        c.record_local_tick(PoseXYYaw(0.0, 0.0, 0.0), now_step=i, now_time_sec=i * 0.1)
    # First attempt timed out and was retried -- SAME coordinates, a NEW id
    # (the retry is already active by the time this loop returns).
    assert c.last_subgoal_result.status == SubgoalStatus.FAILED_TIMEOUT
    assert c.last_subgoal_result.subgoal_id == 1
    assert c.active_subgoal_mission == (3.0, 0.0)

    for i in range(4, 7):
        c.record_local_tick(PoseXYYaw(0.0, 0.0, 0.0), now_step=i, now_time_sec=i * 0.1)
    # Retry budget exhausted -- advances to the NEXT queued candidate.
    assert c.active_subgoal_mission == (6.0, 0.0)
    assert c.pending_subgoal_count == 0


def test_abort_mission_when_queue_exhausted_and_no_retry_left():
    c = _coordinator(
        replanning=ReplanningConfig(local_option_timeout_steps=2, no_progress_window_steps=1000),
        recovery=FailureRecoveryConfig(max_retries_per_subgoal=0),
    )
    c.start_mission(100.0, 0.0)
    c.enqueue_subgoal(3.0, 0.0)
    c.activate_next_subgoal(PoseXYYaw(0.0, 0.0, 0.0), now_step=0, now_time_sec=0.0)
    for i in range(1, 3):
        c.record_local_tick(PoseXYYaw(0.0, 0.0, 0.0), now_step=i, now_time_sec=i * 0.1)
    assert c.mission_failed is True
    assert c.mission_done is True
    assert c.stop_required is True


def test_invalid_candidate_is_never_activated_and_the_next_one_is_tried():
    c = _coordinator()
    c.start_mission(100.0, 0.0)
    c.enqueue_subgoal(2.0, 0.0)  # will be rejected
    c.enqueue_subgoal(5.0, 0.0)  # valid

    def is_valid(x, y):
        return x != 2.0

    ok = c.activate_next_subgoal(PoseXYYaw(0.0, 0.0, 0.0), now_step=0, now_time_sec=0.0, is_valid=is_valid)
    assert ok is True
    assert c.active_subgoal_mission == (5.0, 0.0)


def test_mission_fails_when_every_candidate_is_invalid():
    c = _coordinator()
    c.start_mission(100.0, 0.0)
    c.enqueue_subgoal(2.0, 0.0)
    c.enqueue_subgoal(4.0, 0.0)
    ok = c.activate_next_subgoal(PoseXYYaw(0.0, 0.0, 0.0), now_step=0, now_time_sec=0.0, is_valid=lambda x, y: False)
    assert ok is False
    assert c.mission_failed is True
    assert c.stop_required is True


# ---------------------------------------------------------------- mission timeout

def test_mission_timeout_is_distinct_from_mission_failed():
    c = _coordinator(mission_timeout_steps=3)
    c.start_mission(100.0, 0.0, now_step=0, now_time_sec=0.0)
    c.enqueue_subgoal(50.0, 0.0)
    c.activate_next_subgoal(PoseXYYaw(0.0, 0.0, 0.0), now_step=0, now_time_sec=0.0)
    for i in range(1, 5):
        if c.mission_done:
            break
        c.record_local_tick(PoseXYYaw(0.0, 0.0, 0.0), now_step=i, now_time_sec=i * 0.1)
    assert c.mission_timed_out is True
    assert c.mission_failed is False
    assert c.mission_reached is False
    assert c.mission_done is True


# ---------------------------------------------------------------------- distance tracking

def test_previous_distances_track_the_prior_ticks_values():
    c = _coordinator()
    c.start_mission(10.0, 0.0)
    c.enqueue_subgoal(3.0, 0.0)
    c.activate_next_subgoal(PoseXYYaw(0.0, 0.0, 0.0), now_step=0, now_time_sec=0.0)
    c.record_local_tick(PoseXYYaw(1.0, 0.0, 0.0), now_step=1, now_time_sec=0.1)
    prev1_final, prev1_sub = c.previous_final_goal_distance, c.previous_subgoal_distance
    assert prev1_final == pytest.approx(10.0)
    assert prev1_sub == pytest.approx(3.0)
    c.record_local_tick(PoseXYYaw(2.0, 0.0, 0.0), now_step=2, now_time_sec=0.2)
    assert c.previous_final_goal_distance == pytest.approx(9.0)
    assert c.previous_subgoal_distance == pytest.approx(2.0)
