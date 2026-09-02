"""``HierarchyCoordinator.force_terminate_active_subgoal`` (item 2): the
executor-level safety net that guarantees every ACTIVATED subgoal ends in
one of the six terminal ``SubgoalStatus`` values, even for conditions
``evaluate_replanning`` (which only runs inside ``record_local_tick``) can
never see on its own -- stale localization/scan before a tick was even
attempted, a policy-inference timeout/error, a malformed raw action, or an
executor's own control-loop budget exhausting with nothing else having
fired. Pure Python, no ROS/Gazebo needed."""

from __future__ import annotations

import pytest

from hunter_kinodynamic_rl.navigation.hierarchy.coordinator import HierarchyCoordinator, HierarchyCoordinatorConfig
from hunter_kinodynamic_rl.navigation.hierarchy.failure_recovery import FailureRecoveryConfig
from hunter_kinodynamic_rl.navigation.hierarchy.subgoal_manager import SubgoalStatus
from hunter_kinodynamic_rl.navigation.mission.mission_frame import PoseXYYaw


def _coordinator(**recovery_overrides) -> HierarchyCoordinator:
    return HierarchyCoordinator(HierarchyCoordinatorConfig(
        recovery=FailureRecoveryConfig(**recovery_overrides) if recovery_overrides else FailureRecoveryConfig(),
    ))


def _activated_coordinator(**recovery_overrides) -> HierarchyCoordinator:
    coordinator = _coordinator(**recovery_overrides)
    coordinator.start_mission(10.0, 0.0, now_step=0, now_time_sec=0.0)
    coordinator.enqueue_subgoal(5.0, 0.0)
    activated = coordinator.activate_next_subgoal(PoseXYYaw(0.0, 0.0, 0.0), now_step=0, now_time_sec=0.0)
    assert activated
    return coordinator


def test_raises_when_no_subgoal_active():
    coordinator = _coordinator()
    with pytest.raises(RuntimeError, match="no subgoal is ACTIVE"):
        coordinator.force_terminate_active_subgoal(
            SubgoalStatus.FAILED_TIMEOUT, "max_local_steps_exhausted", now_step=0, now_time_sec=0.0,
        )


def test_localization_degraded_sets_sticky_flag_and_skips_recovery():
    coordinator = _activated_coordinator(max_retries_per_subgoal=0)
    result = coordinator.force_terminate_active_subgoal(
        SubgoalStatus.CANCELLED_BY_REPLAN, "localization_confidence_degraded", now_step=3, now_time_sec=0.3,
    )
    assert result.status == SubgoalStatus.CANCELLED_BY_REPLAN
    assert result.reason == "localization_confidence_degraded"
    assert coordinator.localization_degraded is True
    assert coordinator.stop_required is True
    # Recovery must NEVER run for this reason -- no ADVANCE_NEXT/ABORT
    # decision, no mission_failed side effect from _apply_recovery.
    assert coordinator.mission_failed is False
    assert coordinator.last_subgoal_result is result


def test_localization_degraded_blocks_reactivation_until_confidence_recovers():
    coordinator = _activated_coordinator()
    coordinator.force_terminate_active_subgoal(
        SubgoalStatus.CANCELLED_BY_REPLAN, "localization_confidence_degraded", now_step=1, now_time_sec=0.1,
    )
    coordinator.enqueue_subgoal(5.0, 0.0)
    # A caller polling with a still-degraded reading gets refused.
    assert coordinator.activate_next_subgoal(
        PoseXYYaw(0.0, 0.0, 0.0), now_step=2, now_time_sec=0.2, localization_confidence=0.0,
    ) is False
    assert coordinator.stop_required is True
    # Recovers once a healthy confidence reading is reported.
    assert coordinator.activate_next_subgoal(
        PoseXYYaw(0.0, 0.0, 0.0), now_step=3, now_time_sec=0.3, localization_confidence=1.0,
    ) is True
    assert coordinator.localization_degraded is False


def test_other_reason_runs_recovery_retry_then_aborts_when_exhausted():
    coordinator = _activated_coordinator(max_retries_per_subgoal=1)
    pose = PoseXYYaw(0.0, 0.0, 0.0)

    # First timeout: FAILED_TIMEOUT is retryable (default retryable set),
    # retry_count=0 < max_retries=1 -> RETRY_SAME re-activates the SAME
    # candidate immediately.
    first = coordinator.force_terminate_active_subgoal(
        SubgoalStatus.FAILED_TIMEOUT, "max_local_steps_exhausted", pose, now_step=5, now_time_sec=0.5,
    )
    assert first.status == SubgoalStatus.FAILED_TIMEOUT
    assert coordinator.subgoal_manager.active is True, "retry must reactivate a subgoal"
    assert coordinator.mission_failed is False

    # Second timeout on the retried subgoal: retry budget exhausted and no
    # next candidate queued -> ABORT_MISSION.
    second = coordinator.force_terminate_active_subgoal(
        SubgoalStatus.FAILED_TIMEOUT, "max_local_steps_exhausted", pose, now_step=10, now_time_sec=1.0,
    )
    assert second.status == SubgoalStatus.FAILED_TIMEOUT
    assert coordinator.mission_failed is True
    assert coordinator.stop_required is True


def test_non_localization_reason_omitted_pose_falls_back_to_last_recorded_pose():
    """``robot_pose_mission`` is optional -- when omitted, recovery falls
    back to the coordinator's own last-recorded pose (set at activation
    time) rather than requiring every caller to re-thread it. A
    CANCELLED_BY_REPLAN reason is never in FailureRecoveryPolicy's
    retryable set, so with no next candidate queued this still aborts the
    mission -- exercised here to confirm the fallback path runs the SAME
    recovery decision as passing the pose explicitly would, rather than
    silently no-op'ing or crashing on a None pose."""
    coordinator = _activated_coordinator(max_retries_per_subgoal=1)
    result = coordinator.force_terminate_active_subgoal(
        SubgoalStatus.CANCELLED_BY_REPLAN, "scan_stale", now_step=0, now_time_sec=0.0,
    )
    assert result.status == SubgoalStatus.CANCELLED_BY_REPLAN
    assert result.reason == "scan_stale"
    assert coordinator.mission_failed is True
    assert coordinator.subgoal_manager.active is False


def test_elapsed_time_sec_reflects_exactly_the_option_local_clock_given():
    coordinator = _activated_coordinator()
    result = coordinator.force_terminate_active_subgoal(
        SubgoalStatus.FAILED_TIMEOUT, "max_local_steps_exhausted", PoseXYYaw(0.0, 0.0, 0.0),
        now_step=35, now_time_sec=3.5,
    )
    assert result.local_steps == 0  # no record_local_tick was ever called
    assert result.elapsed_time_sec == pytest.approx(3.5)


def test_reached_status_sets_subgoal_reached_flag():
    coordinator = _activated_coordinator()
    result = coordinator.force_terminate_active_subgoal(
        SubgoalStatus.REACHED, "final_goal_reached_during_subgoal", now_step=1, now_time_sec=0.1,
    )
    assert result.status == SubgoalStatus.REACHED
    assert coordinator.subgoal_reached is True
    assert coordinator.subgoal_failed is False
