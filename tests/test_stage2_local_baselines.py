import dataclasses
import json
from pathlib import Path

import pytest

from hunter_kinodynamic_rl.config.loader import load_profile
from hunter_kinodynamic_rl.evaluation.fingerprint import training_profile_fingerprint
from hunter_kinodynamic_rl.evaluation.stage2_local_baselines import (
    DEFAULT_TRAINING_SEEDS,
    STAGE2_PROFILE_BY_LABEL,
    Stage2ContractError,
    aggregate_stage2,
    benchmark_artifact_path,
    prepare_stage2,
    stage2_status,
    training_complete,
    training_root,
)


def test_prepare_creates_immutable_six_by_five_contract(tmp_path):
    contract = prepare_stage2(tmp_path)
    assert contract["training_seeds"] == list(DEFAULT_TRAINING_SEEDS)
    assert len(contract["profiles"]) == 6 * 5
    assert contract["promotion_policy"] == {
        "label": "L5", "seed": 0, "checkpoint_tag": "final", "selection_uses_test_metrics": False,
    }
    assert prepare_stage2(tmp_path) == contract

    profile_path = Path(contract["profiles"]["L0/seed_0"]["path"])
    profile_path.write_text(profile_path.read_text() + "\n# changed\n")
    with pytest.raises(Stage2ContractError, match="refusing to rewrite"):
        prepare_stage2(tmp_path)


def test_prepare_rejects_less_than_five_seeds(tmp_path):
    with pytest.raises(Stage2ContractError, match="at least 5"):
        prepare_stage2(tmp_path, seeds=(0, 1, 2, 3))


def _summary(value):
    return {
        "feasible_subgoal_success_rate": value,
        "collision_rate": 1.0 - value,
        "timeout_rate": 0.0,
        "high_risk_failure_rate": 0.0,
        "infeasible_false_success_rate": 0.0,
        "infeasible_collision_rate": 0.0,
        "infeasible_high_risk_rate": 0.0,
        "infeasible_safe_termination_rate": 1.0,
        "time_to_goal_sec_mean": 2.0,
        "time_to_termination_sec_mean": 3.0,
    }


def _write_complete_artifacts(root, contract):
    scenario_hash = contract["benchmark"]["scenario_manifest_sha256"]
    for label in STAGE2_PROFILE_BY_LABEL:
        for seed in contract["training_seeds"]:
            key = f"{label}/seed_{seed}"
            path = benchmark_artifact_path(root, label, seed)
            path.parent.mkdir(parents=True, exist_ok=True)
            run_dir = training_root(root, label, seed) / "run"
            (run_dir / "configs").mkdir(parents=True, exist_ok=True)
            final_dir = run_dir / "checkpoints" / "final"
            final_dir.mkdir(parents=True, exist_ok=True)
            generation = f"{label}-{seed}"
            profile = load_profile(contract["profiles"][key]["path"])
            (final_dir / "manifest.json").write_text(json.dumps({
                "global_step": contract["profiles"][key]["max_timesteps"],
                "generation": generation,
                "pt_sha256": "a" * 64,
                "profile_name": profile.name,
                "resolved_config": dataclasses.asdict(profile),
                "training_environment_attestation": {
                    "profile_name": profile.name,
                    "training_profile_fingerprint_sha256": training_profile_fingerprint(profile),
                    "urdf_robot_name": profile.robot.name,
                },
            }))
            path.write_text(json.dumps({
                "benchmark_kind": "formal",
                "num_scenarios": contract["benchmark"]["num_scenarios"],
                "scenario_manifest_sha256": scenario_hash,
                "profile_name": contract["profiles"][key]["profile_name"],
                "checkpoint_generation": generation,
                "checkpoint_sha256": "a" * 64,
                "summary": _summary(0.5 + seed / 10.0),
            }))


def test_aggregate_requires_complete_shared_formal_matrix(tmp_path):
    contract = prepare_stage2(tmp_path)
    with pytest.raises(Stage2ContractError, match="incomplete"):
        aggregate_stage2(tmp_path)

    _write_complete_artifacts(tmp_path, contract)
    aggregate = aggregate_stage2(tmp_path)
    assert aggregate["training_seed_count"] == 5
    assert len(aggregate["rows"]) == 30
    assert aggregate["aggregates"]["L3"]["feasible_subgoal_success_rate"]["mean"] == pytest.approx(0.7)

    bad_path = benchmark_artifact_path(tmp_path, "L4", 3)
    bad = json.loads(bad_path.read_text())
    bad["scenario_manifest_sha256"] = "wrong"
    bad_path.write_text(json.dumps(bad))
    with pytest.raises(Stage2ContractError, match="shared scenario"):
        aggregate_stage2(tmp_path)


def test_status_reports_all_pending_rows_after_prepare(tmp_path):
    prepare_stage2(tmp_path)
    report = stage2_status(tmp_path)
    assert len(report["rows"]) == 30
    assert not any(row["training_complete"] for row in report["rows"].values())
    assert not any(row["benchmark_complete"] for row in report["rows"].values())


def test_training_complete_rejects_same_named_checkpoint_from_a_different_reward_contract(tmp_path):
    contract = prepare_stage2(tmp_path)
    key = "L2/seed_0"
    profile_path = Path(contract["profiles"][key]["path"])
    profile = load_profile(str(profile_path))
    stale = dataclasses.replace(
        profile,
        reward=dataclasses.replace(profile.reward, collision_penalty=profile.reward.collision_penalty - 1.0),
    )
    run_dir = training_root(tmp_path, "L2", 0) / "stale"
    final_dir = run_dir / "checkpoints" / "final"
    final_dir.mkdir(parents=True)
    (run_dir / "configs").mkdir()
    (final_dir / "manifest.json").write_text(json.dumps({
        "global_step": profile.training.max_timesteps,
        "profile_name": profile.name,
        "resolved_config": dataclasses.asdict(stale),
        "training_environment_attestation": {
            "profile_name": profile.name,
            "training_profile_fingerprint_sha256": training_profile_fingerprint(profile),
            "urdf_robot_name": profile.robot.name,
        },
    }))

    assert training_complete(run_dir, profile_path) is False
