"""Global (mission-level, hierarchical) evaluation metrics (plan section
9.11/10.4/21) -- mirrors ``evaluation/metrics.py``'s own pure-function,
``*_valid_count``-paired aggregation style exactly, but scoped to Global/
mission-level navigation instead of the LOCAL per-episode metrics that
module already covers (kept unchanged, used TOGETHER with this module, not
replaced by it -- plan 10.4: "기존 Local metric은 함께 유지한다").

Each mission's episode dict is expected to carry (fields the runner --
``long_horizon_benchmark.py`` -- is responsible for populating from the
ONLINE PartialMap/HierarchyCoordinator/TopologicalGraph, never from
simulator ground truth except ``shortest_feasible_path_length_m`` itself,
which is explicitly a PRIVILEGED evaluation-only quantity, plan 3.2's
"scenario solvability/SPL 계산" carve-out):

    success: bool
    route_length_m: float                       -- actual odometry-based traveled distance
    shortest_feasible_path_length_m: Optional[float]  -- GT L* (long_horizon_solvability), None if not computed
    navigation_time_sec: Optional[float]
    num_global_subgoals: int
    subgoal_success_count: int
    subgoal_attempt_count: int
    revisit_distance_m: float                    -- route distance spent re-traversing already-visited cells
    dead_end_entries: int
    repeated_dead_end_entries: int
    backtracking_distance_m: float
    explored_area_m2: float                      -- PartialMap observed-cell area
    visited_area_m2: float                       -- PartialMap visited-cell area (subset of explored)
    local_planner_failure_count: int             -- subgoal FAILURE_SUBGOAL_STATUSES count
    global_replan_count: int                     -- CANCELLED_BY_REPLAN count
    localization_drift_error_m: Optional[float]  -- final-pose drift vs. ideal odometry, drift-profile runs only
"""

from __future__ import annotations

from typing import Dict, List, Optional


def _mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _rate(episodes: List[dict], key: str) -> float:
    if not episodes:
        return 0.0
    return sum(1 for e in episodes if e.get(key)) / len(episodes)


def global_spl(episode: dict) -> Optional[float]:
    """Obstacle-aware Global SPL for ONE mission --
    ``success * L* / max(route_length, L*)`` -- ``None`` (never 0.0) when
    ``shortest_feasible_path_length_m`` was not computed for this episode
    (excluded from the mean via ``global_spl_valid_count``, never silently
    treated as a 0-length/failed episode)."""
    l_star = episode.get("shortest_feasible_path_length_m")
    if l_star is None:
        return None
    if not episode.get("success"):
        return 0.0
    route = max(episode.get("route_length_m", 0.0), 1e-6)
    l_star = max(l_star, 1e-6)
    return l_star / max(route, l_star)


def excess_path_ratio(episode: dict) -> Optional[float]:
    """``(route_length - L*) / L*`` -- only meaningful for a SUCCESSFUL
    episode with a known ``L*`` (an unsuccessful episode's route never
    reached the goal, so comparing its length to the shortest FEASIBLE path
    is not "excess", it is "incomplete"). ``None`` otherwise."""
    if not episode.get("success"):
        return None
    l_star = episode.get("shortest_feasible_path_length_m")
    if l_star is None or l_star <= 0.0:
        return None
    route = episode.get("route_length_m", 0.0)
    return (route - l_star) / l_star


def revisit_ratio(episode: dict) -> Optional[float]:
    route = episode.get("route_length_m", 0.0)
    if route <= 0.0:
        return None
    return episode.get("revisit_distance_m", 0.0) / route


def unnecessary_exploration_ratio(episode: dict) -> Optional[float]:
    explored = episode.get("explored_area_m2", 0.0)
    if explored <= 0.0:
        return None
    visited = episode.get("visited_area_m2", 0.0)
    return max(0.0, 1.0 - (visited / explored))


def subgoal_success_rate(episode: dict) -> Optional[float]:
    attempts = episode.get("subgoal_attempt_count", 0)
    if attempts <= 0:
        return None
    return episode.get("subgoal_success_count", 0) / attempts


def global_decision_rate(episode: dict) -> Optional[float]:
    """Requirement D: "Global/Local decision rate가 분리되어 로그로
    확인된다" -- Global decisions (subgoal selections) per Local control
    tick. ``None`` (never 0.0) when no local steps ran at all (nothing to
    take a rate over)."""
    local_steps = episode.get("cumulative_local_steps", 0)
    if local_steps <= 0:
        return None
    return episode.get("num_global_subgoals", 0) / local_steps


def aggregate(episodes: List[dict]) -> Dict[str, float]:
    """Every ``*_mean``/``*_worst`` metric that can be undefined for some
    episodes is paired with an explicit ``*_valid_count`` (mirrors
    ``evaluation.metrics.aggregate``'s own JSON-safe convention)."""
    n = len(episodes)
    if n == 0:
        return {}

    route_lengths = [e.get("route_length_m", 0.0) for e in episodes]
    goal_times = [e["time_to_goal_sec"] for e in episodes if e.get("time_to_goal_sec") is not None]
    termination_times = [
        e["time_to_termination_sec"] for e in episodes if e.get("time_to_termination_sec") is not None
    ]
    control_times = [
        e["control_elapsed_time_sec"] for e in episodes if e.get("control_elapsed_time_sec") is not None
    ]
    wall_times = [e["wall_elapsed_time_sec"] for e in episodes if e.get("wall_elapsed_time_sec") is not None]
    n_subgoals = [e.get("num_global_subgoals", 0) for e in episodes]
    spls = [s for s in (global_spl(e) for e in episodes) if s is not None]
    excess_ratios = [r for r in (excess_path_ratio(e) for e in episodes) if r is not None]
    revisit_ratios = [r for r in (revisit_ratio(e) for e in episodes) if r is not None]
    unnecessary_ratios = [r for r in (unnecessary_exploration_ratio(e) for e in episodes) if r is not None]
    subgoal_rates = [r for r in (subgoal_success_rate(e) for e in episodes) if r is not None]
    dead_end_entries = [e.get("dead_end_entries", 0) for e in episodes]
    repeated_dead_end_entries = [e.get("repeated_dead_end_entries", 0) for e in episodes]
    loop_counts = [e.get("loop_count", 0) for e in episodes]
    decision_rates = [r for r in (global_decision_rate(e) for e in episodes) if r is not None]
    backtracking = [e.get("backtracking_distance_m", 0.0) for e in episodes]
    explored_areas = [e.get("explored_area_m2", 0.0) for e in episodes]
    local_failures = [e.get("local_planner_failure_count", 0) for e in episodes]
    global_replans = [e.get("global_replan_count", 0) for e in episodes]
    drift_errors = [e["localization_drift_error_m"] for e in episodes if e.get("localization_drift_error_m") is not None]

    # item 8: Local safety telemetry (OptionTelemetry/SubgoalResult,
    # connected in long_horizon_benchmark.run_ablation_mission) --
    # aggregated with the SAME *_mean/*_valid_count convention as every
    # other optional-per-episode metric above.
    min_clearances = [e["min_clearance_m"] for e in episodes if e.get("min_clearance_m") is not None]
    min_ttcs = [e["min_time_to_collision_sec"] for e in episodes if e.get("min_time_to_collision_sec") is not None]
    min_stopping_margins = [
        e["min_stopping_margin_m"] for e in episodes if e.get("min_stopping_margin_m") is not None
    ]
    steering_saturation_rates = [
        e["steering_saturation_rate"] for e in episodes if e.get("steering_saturation_rate") is not None
    ]
    emergency_stop_counts = [e.get("emergency_stop_count", 0) for e in episodes]
    predicted_risk_means = [e["predicted_risk_mean"] for e in episodes if e.get("predicted_risk_mean") is not None]
    predicted_risk_maxes = [e["predicted_risk_max"] for e in episodes if e.get("predicted_risk_max") is not None]
    risk_integrals = [e.get("risk_integral", 0.0) for e in episodes]
    inference_timeouts = [e.get("inference_timeout_count", 0) for e in episodes]
    inference_errors = [e.get("inference_error_count", 0) for e in episodes]
    command_timeouts = [e.get("command_timeout_count", 0) for e in episodes]
    collision_counts = [e.get("collision_count", 0) for e in episodes]

    return {
        "num_episodes": n,
        "final_goal_success_rate": _rate(episodes, "success"),
        "global_spl_mean": _mean(spls) if spls else None,
        "global_spl_valid_count": len(spls),
        "total_route_length_m_mean": _mean(route_lengths),
        "excess_path_ratio_mean": _mean(excess_ratios) if excess_ratios else None,
        "excess_path_ratio_valid_count": len(excess_ratios),
        "time_to_goal_sec_mean": _mean(goal_times) if goal_times else None,
        "time_to_goal_sec_valid_count": len(goal_times),
        "time_to_termination_sec_mean": _mean(termination_times) if termination_times else None,
        "time_to_termination_sec_valid_count": len(termination_times),
        "control_elapsed_time_sec_mean": _mean(control_times) if control_times else None,
        "control_elapsed_time_sec_valid_count": len(control_times),
        "wall_elapsed_time_sec_mean": _mean(wall_times) if wall_times else None,
        "wall_elapsed_time_sec_valid_count": len(wall_times),
        "num_global_subgoals_mean": _mean(n_subgoals),
        "subgoal_success_rate_mean": _mean(subgoal_rates) if subgoal_rates else None,
        "subgoal_success_rate_valid_count": len(subgoal_rates),
        "revisit_ratio_mean": _mean(revisit_ratios) if revisit_ratios else None,
        "revisit_ratio_valid_count": len(revisit_ratios),
        "dead_end_entries_mean": _mean(dead_end_entries),
        "dead_end_entries_total": sum(dead_end_entries),
        "repeated_dead_end_entries_mean": _mean(repeated_dead_end_entries),
        "repeated_dead_end_entries_total": sum(repeated_dead_end_entries),
        "loop_count_mean": _mean(loop_counts),
        "loop_count_total": sum(loop_counts),
        "global_decision_rate_mean": _mean(decision_rates) if decision_rates else None,
        "global_decision_rate_valid_count": len(decision_rates),
        "backtracking_distance_m_mean": _mean(backtracking),
        "explored_area_m2_mean": _mean(explored_areas),
        "unnecessary_exploration_ratio_mean": _mean(unnecessary_ratios) if unnecessary_ratios else None,
        "unnecessary_exploration_ratio_valid_count": len(unnecessary_ratios),
        "local_planner_failure_count_mean": _mean(local_failures),
        "local_planner_failure_count_total": sum(local_failures),
        "global_replan_count_mean": _mean(global_replans),
        "global_replan_count_total": sum(global_replans),
        "localization_drift_error_m_mean": _mean(drift_errors) if drift_errors else None,
        "localization_drift_error_m_valid_count": len(drift_errors),
        # item 8: Local safety telemetry summary.
        "min_clearance_m_mean": _mean(min_clearances) if min_clearances else None,
        "min_clearance_m_worst": min(min_clearances) if min_clearances else None,
        "min_clearance_m_valid_count": len(min_clearances),
        "min_time_to_collision_sec_mean": _mean(min_ttcs) if min_ttcs else None,
        "min_time_to_collision_sec_worst": min(min_ttcs) if min_ttcs else None,
        "min_time_to_collision_sec_valid_count": len(min_ttcs),
        "min_stopping_margin_m_mean": _mean(min_stopping_margins) if min_stopping_margins else None,
        "min_stopping_margin_m_worst": min(min_stopping_margins) if min_stopping_margins else None,
        "min_stopping_margin_m_valid_count": len(min_stopping_margins),
        "steering_saturation_rate_mean": _mean(steering_saturation_rates) if steering_saturation_rates else None,
        "steering_saturation_rate_valid_count": len(steering_saturation_rates),
        "emergency_stop_count_mean": _mean(emergency_stop_counts),
        "emergency_stop_count_total": sum(emergency_stop_counts),
        "predicted_risk_mean_mean": _mean(predicted_risk_means) if predicted_risk_means else None,
        "predicted_risk_mean_valid_count": len(predicted_risk_means),
        "predicted_risk_max_worst": max(predicted_risk_maxes) if predicted_risk_maxes else None,
        "predicted_risk_max_valid_count": len(predicted_risk_maxes),
        "risk_integral_mean": _mean(risk_integrals),
        "risk_integral_total": sum(risk_integrals),
        "inference_timeout_count_mean": _mean(inference_timeouts),
        "inference_timeout_count_total": sum(inference_timeouts),
        "inference_error_count_total": sum(inference_errors),
        "command_timeout_count_total": sum(command_timeouts),
        "collision_count_mean": _mean(collision_counts),
        "collision_count_total": sum(collision_counts),
        "collision_free_rate": sum(1 for c in collision_counts if c == 0) / n,
    }


def localization_drift_sensitivity(ideal_metrics: Dict[str, float], drifted_metrics: Dict[str, float]) -> Optional[float]:
    """Relative drop in ``final_goal_success_rate`` between an IDEAL
    (near-zero-noise) localization run and a NOISY/DRIFTING one over the
    SAME benchmark scenario set (plan 10.5's Phase A vs. B/C/D/E sweep) --
    ``0.0`` means no measurable degradation, ``1.0`` means total collapse
    (drifted success rate is 0 while ideal succeeded at least once). ``None``
    when the ideal run itself never succeeded (nothing to normalize against
    -- a 0/0 comparison is undefined, not a perfect 0.0 sensitivity)."""
    ideal_rate = ideal_metrics.get("final_goal_success_rate", 0.0)
    if ideal_rate <= 0.0:
        return None
    drifted_rate = drifted_metrics.get("final_goal_success_rate", 0.0)
    return max(0.0, min(1.0, (ideal_rate - drifted_rate) / ideal_rate))
