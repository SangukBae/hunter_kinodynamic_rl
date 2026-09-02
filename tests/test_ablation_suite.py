"""Requirements F/G: A-G ablation suite runner + acceptance report."""

import numpy as np
import pytest

from hunter_kinodynamic_rl.config.schema import (
    GlobalFeasibilityConfig, GlobalRLConfig, HierarchyConfig, LongHorizonWorldConfig, MappingConfig, MemoryConfig,
)
from hunter_kinodynamic_rl.evaluation.ablation_suite import (
    ABLATION_SUITE_SCHEMA_VERSION, evaluate_acceptance_report, run_ablation_suite,
)
from hunter_kinodynamic_rl.evaluation.long_horizon_benchmark import build_fixed_benchmark_manifest

_ROBOT_RADIUS_M = 0.45
_MIN_TURN_RADIUS_M = 1.2
_WHEELBASE_M = 0.65


def _small_world_cfg() -> LongHorizonWorldConfig:
    return LongHorizonWorldConfig(
        enabled=True, size_m=12.0, resolution_m=0.25, corridor_width_min_m=2.0, corridor_width_max_m=2.5,
        room_count_range=[2, 3], dead_end_count_range=[0, 1], loop_count_range=[0, 1],
        alternative_route_min_count=0, start_goal_geodesic_min_m=4.0, generation_attempt_limit=50,
        train_seed_range=[0, 999], validation_seed_range=[1000, 1999], test_seed_range=[2000, 2999],
    )


class _FixedActionAgent:
    def select_action(self, obs, *, epsilon, rng):
        valid = np.flatnonzero(obs.action_mask)
        return int(valid[0])


def _episode(**overrides):
    base = dict(
        success=True, route_length_m=20.0, shortest_feasible_path_length_m=15.0, navigation_time_sec=40.0,
        control_elapsed_time_sec=40.0, time_to_goal_sec=40.0, time_to_termination_sec=40.0,
        wall_elapsed_time_sec=0.5, num_global_subgoals=5, subgoal_success_count=4, subgoal_attempt_count=5,
        revisit_distance_m=2.0, dead_end_entries=1, repeated_dead_end_entries=2, loop_count=1,
        cumulative_local_steps=200, backtracking_distance_m=3.0, explored_area_m2=100.0, visited_area_m2=40.0,
        local_planner_failure_count=1, global_replan_count=2, localization_drift_error_m=None,
        collision_count=0,
    )
    base.update(overrides)
    return base


def test_evaluate_acceptance_report_rejects_bad_benchmark_kind():
    with pytest.raises(ValueError):
        evaluate_acceptance_report({}, benchmark_kind="bogus")


def test_missing_labels_report_insufficient_data():
    report = evaluate_acceptance_report({"E": [_episode()] * 25}, benchmark_kind="smoke")
    for comp in report.comparisons:
        if comp.candidate == "D" or comp.baseline == "C":
            assert comp.verdict == "insufficient_data"


def test_d_vs_c_improvement_detected():
    c_episodes = [_episode(repeated_dead_end_entries=5) for _ in range(25)]
    d_episodes = [_episode(repeated_dead_end_entries=1) for _ in range(25)]
    report = evaluate_acceptance_report({"C": c_episodes, "D": d_episodes}, benchmark_kind="smoke")
    dead_end_comp = next(c for c in report.comparisons if c.candidate == "D" and c.metric == "repeated_dead_end_entries_mean")
    assert dead_end_comp.verdict == "improved"


def test_d_vs_c_regression_detected():
    c_episodes = [_episode(repeated_dead_end_entries=1) for _ in range(25)]
    d_episodes = [_episode(repeated_dead_end_entries=5) for _ in range(25)]
    report = evaluate_acceptance_report({"C": c_episodes, "D": d_episodes}, benchmark_kind="smoke")
    dead_end_comp = next(c for c in report.comparisons if c.candidate == "D" and c.metric == "repeated_dead_end_entries_mean")
    assert dead_end_comp.verdict == "regressed"


def test_f_vs_e_non_regression_maintained_when_equal():
    e_episodes = [_episode(collision_count=1) for _ in range(25)]
    f_episodes = [_episode(collision_count=1) for _ in range(25)]
    report = evaluate_acceptance_report({"E": e_episodes, "F": f_episodes}, benchmark_kind="smoke")
    collision_comp = next(c for c in report.comparisons if c.candidate == "F" and c.metric == "collision_count_mean")
    assert collision_comp.verdict == "maintained"


def test_f_vs_e_regression_when_collisions_increase():
    e_episodes = [_episode(collision_count=0) for _ in range(25)]
    f_episodes = [_episode(collision_count=2) for _ in range(25)]
    report = evaluate_acceptance_report({"E": e_episodes, "F": f_episodes}, benchmark_kind="smoke")
    collision_comp = next(c for c in report.comparisons if c.candidate == "F" and c.metric == "collision_count_mean")
    assert collision_comp.verdict == "regressed"


def test_insufficient_samples_never_reports_improved():
    c_episodes = [_episode(repeated_dead_end_entries=5) for _ in range(3)]  # below min_valid_samples
    d_episodes = [_episode(repeated_dead_end_entries=1) for _ in range(3)]
    report = evaluate_acceptance_report(
        {"C": c_episodes, "D": d_episodes}, benchmark_kind="smoke", min_valid_samples=20,
    )
    dead_end_comp = next(c for c in report.comparisons if c.candidate == "D" and c.metric == "repeated_dead_end_entries_mean")
    assert dead_end_comp.verdict == "insufficient_data"
    assert dead_end_comp.candidate_valid_count == 3


def test_schema_version_and_labels_present():
    report = evaluate_acceptance_report({"A": [_episode()]}, benchmark_kind="formal", min_valid_samples=1)
    assert report.schema_version == ABLATION_SUITE_SCHEMA_VERSION
    assert report.labels_present == ["A"]
    assert report.benchmark_kind == "formal"


def test_run_ablation_suite_same_manifest_and_skips_missing_agent():
    cfg = _small_world_cfg()
    manifest = build_fixed_benchmark_manifest(cfg, _ROBOT_RADIUS_M, _MIN_TURN_RADIUS_M, _WHEELBASE_M, num_scenarios=2, seed=3)
    mission_kwargs = dict(
        global_cfg=GlobalRLConfig(enabled=True, replay_capacity=10), hierarchy_cfg=HierarchyConfig(mission_timeout_steps=1500),
        mapping_cfg=MappingConfig(mission_size_cells=128), memory_cfg=MemoryConfig(max_nodes=8, max_nodes_in_observation=4),
        feasibility_cfg=GlobalFeasibilityConfig(), long_horizon_cfg=cfg, robot_radius_m=_ROBOT_RADIUS_M,
        min_turning_radius_m=_MIN_TURN_RADIUS_M, wheelbase_m=_WHEELBASE_M, max_options=6, max_local_steps=40,
        mission_timeout_steps=1500,
    )
    result = run_ablation_suite(
        manifest, labels=("A", "B", "D", "E"), agents={"B": _FixedActionAgent(), "D": _FixedActionAgent()},
        # No agent for E, no local_evaluator for E either -- must be skipped, never silently run.
        seed=0, benchmark_kind="smoke", min_valid_samples=1, **mission_kwargs,
    )
    assert set(result["labels_run"]) == {"A", "B", "D"}
    assert "E" in result["labels_skipped"]
    assert result["num_scenarios"] == 2
    for label in ("A", "B", "D"):
        assert len(result["episodes"][label]) == 2
        assert result["summaries"][label]["num_episodes"] == 2
    assert result["acceptance_report"].benchmark_kind == "smoke"


def test_run_ablation_suite_never_substitutes_missing_agent_with_another_labels_agent():
    cfg = _small_world_cfg()
    manifest = build_fixed_benchmark_manifest(cfg, _ROBOT_RADIUS_M, _MIN_TURN_RADIUS_M, _WHEELBASE_M, num_scenarios=1, seed=1)
    mission_kwargs = dict(
        global_cfg=GlobalRLConfig(enabled=True, replay_capacity=10), hierarchy_cfg=HierarchyConfig(mission_timeout_steps=1500),
        mapping_cfg=MappingConfig(mission_size_cells=128), memory_cfg=MemoryConfig(), feasibility_cfg=GlobalFeasibilityConfig(),
        long_horizon_cfg=cfg, robot_radius_m=_ROBOT_RADIUS_M, min_turning_radius_m=_MIN_TURN_RADIUS_M,
        wheelbase_m=_WHEELBASE_M, max_options=6, max_local_steps=40, mission_timeout_steps=1500,
    )
    result = run_ablation_suite(
        manifest, labels=("B", "D"), agents={"B": _FixedActionAgent()},  # D has no agent
        seed=0, benchmark_kind="smoke", min_valid_samples=1, **mission_kwargs,
    )
    assert result["labels_run"] == ["B"]
    assert "no agent supplied" in result["labels_skipped"]["D"]


def test_run_ablation_suite_formal_raises_instead_of_skipping_missing_agent():
    """Defect-fix item 10: a formal A-G suite must fail the WHOLE run when
    any requested label is missing its agent, never silently narrow itself
    (label skip is a smoke-only behavior)."""
    cfg = _small_world_cfg()
    manifest = build_fixed_benchmark_manifest(cfg, _ROBOT_RADIUS_M, _MIN_TURN_RADIUS_M, _WHEELBASE_M, num_scenarios=1, seed=1)
    mission_kwargs = dict(
        global_cfg=GlobalRLConfig(enabled=True, replay_capacity=10), hierarchy_cfg=HierarchyConfig(mission_timeout_steps=1500),
        mapping_cfg=MappingConfig(mission_size_cells=128), memory_cfg=MemoryConfig(), feasibility_cfg=GlobalFeasibilityConfig(),
        long_horizon_cfg=cfg, robot_radius_m=_ROBOT_RADIUS_M, min_turning_radius_m=_MIN_TURN_RADIUS_M,
        wheelbase_m=_WHEELBASE_M, max_options=6, max_local_steps=40, mission_timeout_steps=1500,
    )
    with pytest.raises(RuntimeError, match="formal benchmark cannot skip label"):
        run_ablation_suite(
            manifest, labels=("B", "D"), agents={"B": _FixedActionAgent()},  # D has no agent
            seed=0, benchmark_kind="formal", min_valid_samples=1, **mission_kwargs,
        )


def test_run_ablation_suite_formal_raises_for_missing_feasibility_evaluator():
    cfg = _small_world_cfg()
    manifest = build_fixed_benchmark_manifest(cfg, _ROBOT_RADIUS_M, _MIN_TURN_RADIUS_M, _WHEELBASE_M, num_scenarios=1, seed=1)
    mission_kwargs = dict(
        global_cfg=GlobalRLConfig(enabled=True, replay_capacity=10), hierarchy_cfg=HierarchyConfig(mission_timeout_steps=1500),
        mapping_cfg=MappingConfig(mission_size_cells=128), memory_cfg=MemoryConfig(), feasibility_cfg=GlobalFeasibilityConfig(),
        long_horizon_cfg=cfg, robot_radius_m=_ROBOT_RADIUS_M, min_turning_radius_m=_MIN_TURN_RADIUS_M,
        wheelbase_m=_WHEELBASE_M, max_options=6, max_local_steps=40, mission_timeout_steps=1500,
    )
    with pytest.raises(RuntimeError, match="formal benchmark cannot skip label"):
        run_ablation_suite(
            manifest, labels=("E",), agents={"E": _FixedActionAgent()},  # no local_evaluator for E
            seed=0, benchmark_kind="formal", min_valid_samples=1, **mission_kwargs,
        )
