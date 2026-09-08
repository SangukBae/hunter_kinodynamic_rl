import json
from pathlib import Path

import pytest

import hunter_kinodynamic_rl.config.tractor as tractor_config
from hunter_kinodynamic_rl.evaluation.run_paper_comparison_campaign import (
    _axis_stratified_effects, campaign_status, collect_campaign_data, load_campaign,
    prepare_campaign,
)
from hunter_kinodynamic_rl.evaluation.evaluate_paper_comparison import evaluate_method_seed
from hunter_kinodynamic_rl.evaluation.fit_comparison_calibration import fit_comparison_calibration
from hunter_kinodynamic_rl.training.collect_formal_comparison_data import collect_formal_comparison_data
from hunter_kinodynamic_rl.training.train_comparison_baselines import train_comparison_baseline
from hunter_kinodynamic_rl.training.train_tractor_tqc import _parser, train_from_sequence_replay


CONFIG_ROOT = str(Path(__file__).resolve().parents[1] / "config")


def test_prepare_freezes_the_complete_11_method_5_seed_matrix(tmp_path):
    result = prepare_campaign(tmp_path / "campaign", CONFIG_ROOT)
    assert result["methods"] == [
        "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "A7", "A8", "A9",
    ]
    assert result["seeds"] == [11, 23, 37, 53, 71]
    assert result["expected_locked_records"] == 11 * 5 * 176
    assert set(result["method_contract_sha256"]) == set(result["methods"])
    assert all(len(digest) == 64 for digest in result["method_contract_sha256"].values())
    assert result["formal_implementation_ready"]
    assert not result["formal_implementation_gaps"]
    assert result["evidence_status"] == "prepared_untrained"
    loaded, paths = load_campaign(tmp_path / "campaign", CONFIG_ROOT)
    assert loaded["campaign_manifest_sha256"] == result["campaign_manifest_sha256"]
    assert paths["scenario_manifest"].is_file()
    status = campaign_status(tmp_path / "campaign", CONFIG_ROOT)
    assert status["training_expected"] == 55
    assert status["calibration_expected"] == 20
    assert status["evaluation_expected"] == 55
    assert not status["formal_dataset_ready"]
    assert status["formal_implementation_ready"]


def test_formal_collection_fails_closed_if_readiness_regresses(tmp_path, monkeypatch):
    prepare_campaign(tmp_path / "campaign", CONFIG_ROOT)
    monkeypatch.setattr(
        tractor_config, "FORMAL_RESEARCH_IMPLEMENTATION_GAPS", ("fixture_regression",),
    )
    with pytest.raises(RuntimeError, match="implementation gaps"):
        collect_campaign_data(tmp_path / "campaign", config_root=CONFIG_ROOT)


def test_direct_formal_entry_points_cannot_bypass_the_campaign_readiness_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(
        tractor_config, "FORMAL_RESEARCH_IMPLEMENTATION_GAPS", ("fixture_regression",),
    )
    calls = (
        lambda: collect_formal_comparison_data(
            scenario_manifest_path="missing", dataset_root=tmp_path / "data",
        ),
        lambda: train_from_sequence_replay(
            variant="a7", dataset_root="missing", run_root=tmp_path / "run-a7",
            seed=1, updates=1, formal_data=True,
        ),
        lambda: train_comparison_baseline(
            method_id="B1", dataset_root="missing", run_root=tmp_path / "run-b1",
            seed=1, updates=1, formal_data=True,
        ),
        lambda: fit_comparison_calibration(
            method_id="A7", checkpoint_root="missing", dataset_root="missing",
            scenario_manifest_path="missing", output_root=tmp_path / "cal", seed=1,
        ),
        lambda: evaluate_method_seed(
            method_id="A7", seed=1, checkpoint_root="missing", dataset_root="missing",
            scenario_manifest_path="missing", output_jsonl=tmp_path / "eval.jsonl",
        ),
    )
    for call in calls:
        with pytest.raises(RuntimeError, match="formal implementation gaps"):
            call()


def test_training_cli_and_formal_budgets_fail_closed_before_dataset_access(tmp_path):
    parsed = _parser().parse_args(["--dataset-root", "data", "collect", "--episodes", "1"])
    assert parsed.command == "collect"
    assert parsed.episodes == 1

    with pytest.raises(ValueError, match="stage3 requires exactly 30000"):
        train_from_sequence_replay(
            variant="a7", dataset_root="missing", run_root=tmp_path / "run-a7",
            seed=1, updates=1, training_stage="stage3", formal_data=True,
            config_root=CONFIG_ROOT,
        )
    with pytest.raises(ValueError, match="frozen total update budget"):
        train_comparison_baseline(
            method_id="B1", dataset_root="missing", run_root=tmp_path / "run-b1",
            seed=1, updates=1, formal_data=True, config_root=CONFIG_ROOT,
        )


def test_campaign_manifest_tampering_fails_before_any_run(tmp_path):
    prepare_campaign(tmp_path / "campaign", CONFIG_ROOT)
    manifest_path = tmp_path / "campaign" / "campaign_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["seeds"] = [11]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        load_campaign(tmp_path / "campaign", CONFIG_ROOT)


def test_prepare_refuses_to_reuse_nonempty_campaign_root(tmp_path):
    root = tmp_path / "campaign"
    root.mkdir()
    (root / "foreign.txt").write_text("do not overwrite")
    with pytest.raises(FileExistsError, match="non-empty"):
        prepare_campaign(root, CONFIG_ROOT)


def test_axis_stratified_aggregate_preserves_independent_seed_replication():
    records = []
    for method in ("B1", "A7"):
        for seed in (11, 23):
            for scenario, vehicle, domain in (
                ("s-id", "vehicle_nominal", "id"),
                ("s-ood", "low_friction", "ood"),
            ):
                candidate = method == "A7"
                records.append({
                    "experiment_id": f"{method}-{seed}-{scenario}",
                    "method_id": method, "seed": seed, "scenario_id": scenario,
                    "vehicle_axis": vehicle, "sensor_axis": "sensor_nominal",
                    "localization_axis": "localization_nominal", "system_domain": domain,
                    "metrics": {
                        "success_rate": 0.9 if candidate else 0.8,
                        "collision_rate": 0.01 if candidate else 0.02,
                        "time_to_goal_sec_mean": 10.0 if candidate else 11.0,
                        "min_clearance_m_mean": 0.4 if candidate else 0.3,
                    },
                })
    report = _axis_stratified_effects(records, ["B1", "A7"])
    effect = report["vehicle_axis"]["low_friction"]["paired_effects_vs_B1"]["A7"]
    assert effect["success_rate"]["count"] == 2
    assert effect["success_rate"]["scenario_pair_count"] == 2
    assert report["system_domain"]["id"]["scenario_count"] == 1


def test_axis_stratified_aggregate_fails_closed_without_axis_identity():
    with pytest.raises(RuntimeError, match="missing frozen system axes"):
        _axis_stratified_effects([
            {"experiment_id": "missing", "method_id": "B1", "seed": 1,
             "scenario_id": "s", "metrics": {}}
        ], ["B1"])
