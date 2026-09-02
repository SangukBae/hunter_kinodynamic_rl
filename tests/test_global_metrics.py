"""Phase 5 Global (mission-level) metrics (plan section 9.11/10.4/21)."""

import pytest

from hunter_kinodynamic_rl.evaluation.global_metrics import (
    aggregate, excess_path_ratio, global_decision_rate, global_spl, localization_drift_sensitivity, revisit_ratio,
    subgoal_success_rate, unnecessary_exploration_ratio,
)


def _episode(**overrides):
    base = dict(
        success=True, route_length_m=20.0, shortest_feasible_path_length_m=15.0, navigation_time_sec=40.0,
        control_elapsed_time_sec=40.0, time_to_goal_sec=40.0, time_to_termination_sec=40.0,
        wall_elapsed_time_sec=0.5,
        num_global_subgoals=5, subgoal_success_count=4, subgoal_attempt_count=5, revisit_distance_m=2.0,
        dead_end_entries=1, repeated_dead_end_entries=0, backtracking_distance_m=3.0, explored_area_m2=100.0,
        visited_area_m2=40.0, local_planner_failure_count=1, global_replan_count=2,
        localization_drift_error_m=None,
    )
    base.update(overrides)
    return base


def test_global_spl_hand_computed():
    """L* = 10, route = 20, success -> SPL = 10 / max(20, 10) = 0.5."""
    ep = _episode(success=True, route_length_m=20.0, shortest_feasible_path_length_m=10.0)
    assert global_spl(ep) == pytest.approx(0.5)


def test_global_spl_perfect_path_is_one():
    ep = _episode(success=True, route_length_m=10.0, shortest_feasible_path_length_m=10.0)
    assert global_spl(ep) == pytest.approx(1.0)


def test_global_spl_failed_episode_is_zero_not_none():
    ep = _episode(success=False, shortest_feasible_path_length_m=10.0)
    assert global_spl(ep) == 0.0


def test_global_spl_none_when_shortest_path_unknown():
    ep = _episode(shortest_feasible_path_length_m=None)
    assert global_spl(ep) is None


def test_excess_path_ratio_only_defined_for_success():
    ep_success = _episode(success=True, route_length_m=15.0, shortest_feasible_path_length_m=10.0)
    assert excess_path_ratio(ep_success) == pytest.approx(0.5)
    ep_fail = _episode(success=False, route_length_m=15.0, shortest_feasible_path_length_m=10.0)
    assert excess_path_ratio(ep_fail) is None


def test_revisit_ratio_and_unnecessary_exploration_ratio():
    ep = _episode(route_length_m=10.0, revisit_distance_m=2.5, explored_area_m2=100.0, visited_area_m2=30.0)
    assert revisit_ratio(ep) == pytest.approx(0.25)
    assert unnecessary_exploration_ratio(ep) == pytest.approx(0.7)


def test_subgoal_success_rate():
    ep = _episode(subgoal_success_count=3, subgoal_attempt_count=4)
    assert subgoal_success_rate(ep) == pytest.approx(0.75)
    assert subgoal_success_rate(_episode(subgoal_attempt_count=0)) is None


def test_aggregate_valid_counts_distinguish_missing_from_zero():
    episodes = [
        _episode(shortest_feasible_path_length_m=10.0, route_length_m=10.0, success=True),
        _episode(shortest_feasible_path_length_m=None),
    ]
    result = aggregate(episodes)
    assert result["global_spl_valid_count"] == 1
    assert result["num_episodes"] == 2


def test_aggregate_empty_returns_empty_dict():
    assert aggregate([]) == {}


def test_aggregate_success_rate_and_totals():
    episodes = [_episode(success=True, dead_end_entries=2), _episode(success=False, dead_end_entries=1)]
    result = aggregate(episodes)
    assert result["final_goal_success_rate"] == pytest.approx(0.5)
    assert result["dead_end_entries_total"] == 3


def test_time_to_goal_excludes_failures_and_termination_time_keeps_them():
    episodes = [
        _episode(success=True, time_to_goal_sec=12.0, time_to_termination_sec=12.0),
        _episode(success=False, time_to_goal_sec=None, time_to_termination_sec=30.0),
    ]
    result = aggregate(episodes)
    assert result["time_to_goal_sec_mean"] == pytest.approx(12.0)
    assert result["time_to_goal_sec_valid_count"] == 1
    assert result["time_to_termination_sec_mean"] == pytest.approx(21.0)
    assert result["time_to_termination_sec_valid_count"] == 2


def test_localization_drift_sensitivity_zero_when_no_degradation():
    ideal = {"final_goal_success_rate": 0.9}
    drifted = {"final_goal_success_rate": 0.9}
    assert localization_drift_sensitivity(ideal, drifted) == pytest.approx(0.0)


def test_localization_drift_sensitivity_reflects_relative_drop():
    ideal = {"final_goal_success_rate": 0.8}
    drifted = {"final_goal_success_rate": 0.4}
    assert localization_drift_sensitivity(ideal, drifted) == pytest.approx(0.5)


def test_localization_drift_sensitivity_none_when_ideal_never_succeeded():
    ideal = {"final_goal_success_rate": 0.0}
    drifted = {"final_goal_success_rate": 0.0}
    assert localization_drift_sensitivity(ideal, drifted) is None


def test_global_decision_rate_hand_computed():
    ep = _episode(num_global_subgoals=4, cumulative_local_steps=200)
    assert global_decision_rate(ep) == pytest.approx(4 / 200)


def test_global_decision_rate_none_when_no_local_steps():
    ep = _episode(cumulative_local_steps=0)
    assert global_decision_rate(ep) is None


def test_aggregate_includes_loop_count_and_decision_rate():
    episodes = [
        _episode(loop_count=2, num_global_subgoals=4, cumulative_local_steps=100),
        _episode(loop_count=0, num_global_subgoals=2, cumulative_local_steps=50),
    ]
    summary = aggregate(episodes)
    assert summary["loop_count_mean"] == pytest.approx(1.0)
    assert summary["loop_count_total"] == 2
    assert summary["global_decision_rate_mean"] == pytest.approx(((4 / 100) + (2 / 50)) / 2)
    assert summary["global_decision_rate_valid_count"] == 2
