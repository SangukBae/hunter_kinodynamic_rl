"""Requirement I: localization sweep pairing + aggregation -- deterministic
synthetic episodes, no live long-duration sweep execution."""

import pytest

from hunter_kinodynamic_rl.evaluation.localization_sweep import (
    IDEAL_CONDITION, ScenarioPairingError, aggregate_localization_sweep, drift_curve, verify_scenario_pairing,
)


def _episode(scenario_id, success=True, route_length_m=20.0, l_star=15.0):
    return {
        "scenario_id": scenario_id, "success": success, "route_length_m": route_length_m,
        "shortest_feasible_path_length_m": l_star, "navigation_time_sec": 10.0, "control_elapsed_time_sec": 10.0,
        "time_to_goal_sec": 10.0 if success else None, "time_to_termination_sec": 10.0,
        "wall_elapsed_time_sec": 0.5, "num_global_subgoals": 3, "subgoal_success_count": 3,
        "subgoal_attempt_count": 3, "revisit_distance_m": 0.0, "dead_end_entries": 0,
        "repeated_dead_end_entries": 0, "backtracking_distance_m": 0.0, "explored_area_m2": 10.0,
        "visited_area_m2": 5.0, "local_planner_failure_count": 0, "global_replan_count": 0,
        "localization_drift_error_m": None,
    }


def _paired_episodes(scenario_ids, **overrides):
    return [_episode(sid, **overrides) for sid in scenario_ids]


def test_verify_scenario_pairing_passes_for_matched_ids():
    ids = ["s0", "s1", "s2"]
    results = {"ideal": _paired_episodes(ids), "noisy": _paired_episodes(ids, success=False)}
    verify_scenario_pairing(results)  # must not raise


def test_verify_scenario_pairing_rejects_mismatched_ids():
    results = {
        "ideal": _paired_episodes(["s0", "s1", "s2"]),
        "noisy": _paired_episodes(["s0", "s1", "s3"]),  # s2 missing, s3 extra
    }
    with pytest.raises(ScenarioPairingError):
        verify_scenario_pairing(results)


def test_verify_scenario_pairing_rejects_missing_scenario_id_field():
    results = {"ideal": [{"success": True}]}
    with pytest.raises(ScenarioPairingError):
        verify_scenario_pairing(results)


def test_verify_scenario_pairing_rejects_empty_input():
    with pytest.raises(ScenarioPairingError):
        verify_scenario_pairing({})


def test_aggregate_localization_sweep_rejects_broken_pairing():
    results = {"ideal": _paired_episodes(["s0", "s1"]), "drifting": _paired_episodes(["s0", "s2"])}
    with pytest.raises(ScenarioPairingError):
        aggregate_localization_sweep(results)


def test_aggregate_localization_sweep_computes_drift_sensitivity():
    ids = [f"s{i}" for i in range(10)]
    ideal = _paired_episodes(ids, success=True)
    # Half the episodes fail under drift -- success rate drops from 1.0 to 0.5.
    drifting = [_episode(sid, success=(i % 2 == 0)) for i, sid in enumerate(ids)]
    results = {"ideal": ideal, "drifting": drifting}
    sweep = aggregate_localization_sweep(
        results, noise_params={"ideal": {"drift_std_m": 0.0}, "drifting": {"drift_std_m": 0.5}},
        seeds={"ideal": 0, "drifting": 0},
    )
    assert sweep.conditions == ["drifting", "ideal"]
    assert sweep.summaries["ideal"]["final_goal_success_rate"] == pytest.approx(1.0)
    assert sweep.summaries["drifting"]["final_goal_success_rate"] == pytest.approx(0.5)
    assert sweep.drift_sensitivity["drifting"] == pytest.approx(0.5)
    assert IDEAL_CONDITION not in sweep.drift_sensitivity


def test_aggregate_localization_sweep_without_ideal_condition_has_no_drift_sensitivity():
    ids = ["s0", "s1"]
    results = {"noisy": _paired_episodes(ids), "drifting": _paired_episodes(ids)}
    sweep = aggregate_localization_sweep(results)
    assert sweep.drift_sensitivity == {}


def test_drift_curve_sorted_by_magnitude_and_excludes_missing_params():
    ids = [f"s{i}" for i in range(6)]
    results = {
        "ideal": _paired_episodes(ids, success=True),
        "noisy": _paired_episodes(ids, success=True),
        "drifting_large": [_episode(sid, success=(i < 2)) for i, sid in enumerate(ids)],
    }
    sweep = aggregate_localization_sweep(
        results,
        noise_params={"ideal": {"drift_std_m": 0.0}, "noisy": {"drift_std_m": 0.1},
                      "drifting_large": {"drift_std_m": 1.0}},
    )
    curve = drift_curve(sweep)
    magnitudes = [row["drift_magnitude"] for row in curve]
    assert magnitudes == sorted(magnitudes)
    assert {row["condition"] for row in curve} == {"noisy", "drifting_large"}


def test_drift_curve_excludes_condition_without_drift_magnitude_key():
    ids = ["s0", "s1"]
    results = {"ideal": _paired_episodes(ids), "wheel_imu": _paired_episodes(ids)}
    sweep = aggregate_localization_sweep(results, noise_params={"ideal": {}, "wheel_imu": {}})
    assert drift_curve(sweep) == []
