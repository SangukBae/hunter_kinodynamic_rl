"""Phase 4 Global RL: option-level Global reward (plan section 8.7/14/15)."""

from hunter_kinodynamic_rl.config.schema import GlobalRLConfig
from hunter_kinodynamic_rl.navigation.global_rl.reward import compute_global_reward, local_failure_penalty
from hunter_kinodynamic_rl.navigation.hierarchy.subgoal_manager import SubgoalStatus


def test_goal_reward_only_on_mission_reached():
    cfg = GlobalRLConfig()
    reached = compute_global_reward(
        cfg, mission_reached=True, start_final_goal_distance_m=1.0, end_final_goal_distance_m=0.1,
        newly_explored_cells=0,
    )
    not_reached = compute_global_reward(
        cfg, mission_reached=False, start_final_goal_distance_m=1.0, end_final_goal_distance_m=0.1,
        newly_explored_cells=0,
    )
    assert reached.goal == cfg.goal_reward
    assert not_reached.goal == 0.0


def test_progress_reward_signed_by_distance_change():
    cfg = GlobalRLConfig()
    closer = compute_global_reward(
        cfg, mission_reached=False, start_final_goal_distance_m=10.0, end_final_goal_distance_m=6.0,
        newly_explored_cells=0,
    )
    farther = compute_global_reward(
        cfg, mission_reached=False, start_final_goal_distance_m=6.0, end_final_goal_distance_m=10.0,
        newly_explored_cells=0,
    )
    assert closer.final_progress > 0.0
    assert farther.final_progress < 0.0


def test_exploration_reward_is_clipped_and_never_dominates_goal_reward():
    cfg = GlobalRLConfig(goal_reward=10.0, exploration_reward_scale=1.0, exploration_reward_clip=2.0)
    huge_exploration = compute_global_reward(
        cfg, mission_reached=False, start_final_goal_distance_m=0.0, end_final_goal_distance_m=0.0,
        newly_explored_cells=10_000,
    )
    assert huge_exploration.exploration == cfg.exploration_reward_clip
    assert huge_exploration.exploration < cfg.goal_reward


def test_repeated_deadend_penalty_only_applied_when_flagged():
    cfg = GlobalRLConfig(repeated_deadend_penalty=5.0)
    first_visit = compute_global_reward(
        cfg, mission_reached=False, start_final_goal_distance_m=0.0, end_final_goal_distance_m=0.0,
        newly_explored_cells=0, repeated_deadend=False,
    )
    repeat_visit = compute_global_reward(
        cfg, mission_reached=False, start_final_goal_distance_m=0.0, end_final_goal_distance_m=0.0,
        newly_explored_cells=0, repeated_deadend=True,
    )
    assert first_visit.repeated_deadend == 0.0
    assert repeat_visit.repeated_deadend == cfg.repeated_deadend_penalty
    assert repeat_visit.total < first_visit.total


def test_local_failure_penalty_is_reason_specific():
    cfg = GlobalRLConfig(
        local_failure_penalty_timeout=1.0, local_failure_penalty_no_progress=2.0,
        local_failure_penalty_blocked=3.0, local_failure_penalty_high_risk=4.0,
        local_failure_penalty_cancelled_by_replan=5.0,
    )
    assert local_failure_penalty(cfg, None) == 0.0
    assert local_failure_penalty(cfg, SubgoalStatus.REACHED) == 0.0
    assert local_failure_penalty(cfg, SubgoalStatus.FAILED_TIMEOUT) == 1.0
    assert local_failure_penalty(cfg, SubgoalStatus.FAILED_NO_PROGRESS) == 2.0
    assert local_failure_penalty(cfg, SubgoalStatus.FAILED_BLOCKED) == 3.0
    assert local_failure_penalty(cfg, SubgoalStatus.FAILED_HIGH_RISK) == 4.0
    assert local_failure_penalty(cfg, SubgoalStatus.CANCELLED_BY_REPLAN) == 5.0


def test_risk_and_elapsed_penalties_scale_with_inputs():
    cfg = GlobalRLConfig(risk_penalty_scale=2.0, elapsed_penalty_per_local_step=0.5)
    low = compute_global_reward(
        cfg, mission_reached=False, start_final_goal_distance_m=0.0, end_final_goal_distance_m=0.0,
        newly_explored_cells=0, risk_integral=1.0, local_steps=2,
    )
    high = compute_global_reward(
        cfg, mission_reached=False, start_final_goal_distance_m=0.0, end_final_goal_distance_m=0.0,
        newly_explored_cells=0, risk_integral=5.0, local_steps=20,
    )
    assert low.risk == 2.0
    assert low.elapsed == 1.0
    assert high.risk == 10.0
    assert high.elapsed == 10.0
    assert high.total < low.total


def test_total_matches_sum_of_signed_components():
    cfg = GlobalRLConfig()
    breakdown = compute_global_reward(
        cfg, mission_reached=True, start_final_goal_distance_m=8.0, end_final_goal_distance_m=2.0,
        newly_explored_cells=100, revisit_amount=3.0, repeated_deadend=True,
        subgoal_status=SubgoalStatus.FAILED_TIMEOUT, risk_integral=1.0, local_steps=10,
    )
    expected = (
        breakdown.goal + breakdown.final_progress + breakdown.exploration
        - breakdown.repeated_revisit - breakdown.repeated_deadend - breakdown.local_failure
        - breakdown.risk - breakdown.elapsed
    )
    assert breakdown.total == expected
