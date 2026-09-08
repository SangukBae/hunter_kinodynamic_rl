"""Config, metric and formal-matrix gates for paper evidence."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from hunter_kinodynamic_rl.config.tractor import (
    load_tractor_contract, tractor_profile_model_mismatches,
)
from hunter_kinodynamic_rl.config.loader import load_profile
from hunter_kinodynamic_rl.evaluation.tractor_artifacts import (
    paired_method_effect, paired_nested_method_effect, validate_complete_matrix,
)
from hunter_kinodynamic_rl.evaluation.tractor_metrics import (
    binary_calibration_metrics, candidate_ranking_metrics, occupancy_confusion,
)
from hunter_kinodynamic_rl.training.tractor_preflight import run_tractor_preflight


CONFIG_ROOT = str(Path(__file__).resolve().parents[1] / "config")


def test_registered_variants_have_distinct_and_exact_member_contracts():
    variants = {name: load_tractor_contract(CONFIG_ROOT, name) for name in ("a7", "a8", "a9")}
    assert (variants["a7"]["model"].residual_members, variants["a7"]["model"].risk_members) == (0, 1)
    assert (variants["a8"]["model"].residual_members, variants["a8"]["model"].risk_members) == (1, 1)
    assert (variants["a9"]["model"].residual_members, variants["a9"]["model"].risk_members) == (3, 3)
    assert len({item["model"].fingerprint() for item in variants.values()}) == 3


def test_static_and_dynamic_profiles_are_matched_except_obstacle_motion_count():
    static = load_profile("tractor_local_static", CONFIG_ROOT)
    dynamic = load_profile("tractor_local_dynamic", CONFIG_ROOT)
    assert static.robot == dynamic.robot
    assert static.action_space == dynamic.action_space
    assert static.observation == dynamic.observation
    assert static.features == dynamic.features
    assert static.scenario.dynamic_obstacle_count == 0
    assert dynamic.scenario.dynamic_obstacle_count == 2
    static.scenario.dynamic_obstacle_count = 2
    assert static.scenario == dynamic.scenario


def test_tractor_rollout_physics_matches_the_live_collection_profiles():
    model = load_tractor_contract(CONFIG_ROOT, "a7")["model"]
    for profile_name in ("tractor_local_static", "tractor_local_dynamic"):
        profile = load_profile(profile_name, CONFIG_ROOT)
        assert tractor_profile_model_mismatches(profile, model) == {}
        changed = replace(model, speed_lag_tau_sec=model.speed_lag_tau_sec + 0.01)
        assert "speed_lag_tau_sec" in tractor_profile_model_mismatches(profile, changed)


def test_frozen_protocol_passes_development_but_formal_requires_data_artifacts():
    development = run_tractor_preflight(config_root=CONFIG_ROOT, mode="development", variant="a7")
    assert development.ok
    assert development.protocol_status == "frozen"
    assert development.protocol_version == "tractor_protocol_v2"
    assert development.scenario_count == 616
    formal = run_tractor_preflight(config_root=CONFIG_ROOT, mode="formal", variant="a7")
    assert not formal.ok
    assert formal.protocol_status == "frozen"
    assert formal.formal_implementation_ready
    assert not formal.formal_implementation_gaps
    assert formal.scenario_feasibility_verified
    assert formal.formal_source_identity_ok is False
    assert any("formal source identity failed" in error for error in formal.errors)
    assert any("--scenario-manifest" in error for error in formal.errors)
    assert any("dataset validation skipped" in error for error in formal.errors)


def test_calibration_metrics_keep_denominators_and_event_counts():
    metrics = binary_calibration_metrics([0.1, 0.7, 0.8, np.nan], [0, 1, 0, 1], bins=2)
    assert metrics["count"] == 3
    assert metrics["event_count"] == 1
    assert metrics["brier"] == pytest.approx((0.01 + 0.09 + 0.64) / 3)
    assert metrics["nll"] > 0.0
    assert metrics["ece"] >= 0.0


def test_candidate_ranking_reports_regret_ndcg_and_unsafe_count():
    metrics = candidate_ranking_metrics(
        score=[[0.9, 0.1, 0.2], [0.0, 1.0, 0.5]],
        utility=[[1.0, 2.0, -1.0], [1.0, -2.0, 0.0]],
        valid=[[True, True, True], [True, True, False]],
    )
    assert metrics["row_count"] == 2
    assert metrics["mean_regret"] == pytest.approx(2.0)
    assert metrics["unsafe_top1_count"] == 1
    assert 0.0 <= metrics["ndcg"] <= 1.0


def test_occupancy_confusion_uses_explicit_valid_mask():
    report = occupancy_confusion(
        np.asarray([[0, 1], [2, 3]]), np.asarray([[0, 2], [2, 1]]),
        np.asarray([[True, True], [False, True]]),
    )
    assert report["valid_cells"] == 3
    assert sum(sum(row) for row in report["confusion"]) == 3


def _record(method, seed, scenario, value):
    return {
        "experiment_id": f"{method}-{seed}-{scenario}", "method_id": method,
        "seed": seed, "scenario_id": scenario, "split_id": "locked_test",
        "model_fingerprint": f"model-{method}", "data_fingerprint": "data",
        "training_fingerprint": f"training-{method}", "complete": True,
        "metrics": {"success_rate": value}, "artifact_sha256": "hash",
    }


def test_matrix_gate_rejects_missing_or_duplicate_rows_and_pairs_effects():
    records = [
        _record(method, seed, scenario, 0.8 if method == "A7" else 0.6)
        for method in ("A7", "B1") for seed in (1, 2) for scenario in ("s1", "s2")
    ]
    assert validate_complete_matrix(records, ["A7", "B1"], [1, 2], ["s1", "s2"])["ok"]
    effect = paired_method_effect(records, "A7", "B1", "success_rate")
    assert effect["count"] == 2
    assert effect["scenario_pair_count"] == 4
    assert effect["replication_unit"] == "independent_training_seed"
    assert effect["mean"] == pytest.approx(0.2)
    invalid = validate_complete_matrix(records[:-1] + [records[0]], ["A7", "B1"], [1, 2], ["s1", "s2"])
    assert not invalid["ok"]
    assert any("missing" in error for error in invalid["errors"])
    assert any("duplicate" in error for error in invalid["errors"])


def test_nested_hypothesis_effect_keeps_training_seed_as_replication_unit():
    records = []
    for method in ("A7", "B8"):
        for seed in (1, 2):
            for scenario in ("s1", "s2"):
                records.append({
                    "method_id": method, "seed": seed, "scenario_id": scenario,
                    "metrics": {"h3_risk": {"brier": 0.1 if method == "A7" else 0.2}},
                })
    effect = paired_nested_method_effect(
        records, "A7", "B8", ("metrics", "h3_risk", "brier"),
    )
    assert effect["count"] == 2
    assert effect["scenario_pair_count"] == 4
    assert effect["mean"] == pytest.approx(-0.1)
