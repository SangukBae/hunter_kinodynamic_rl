"""Requirement B / defect-fix items 2, 3, 4, 5: acceptance gate +
local_frozen promotion -- synthetic fixtures only, no live benchmark
execution."""

import dataclasses
import hashlib
import json
import os

import pytest

from hunter_kinodynamic_rl.config.loader import load_profile
from hunter_kinodynamic_rl.evaluation.fingerprint import architecture_fingerprint, local_training_contract_fingerprint
from hunter_kinodynamic_rl.evaluation.local_promotion import (
    DEFAULT_PROMOTION_TAG, LocalAcceptanceConfig, evaluate_local_acceptance, load_local_acceptance_config,
    promote_local_checkpoint, validate_promoted_local_checkpoint,
)
from hunter_kinodynamic_rl.evaluation.local_subgoal_benchmark import BENCHMARK_SCHEMA_VERSION

_PROFILE = load_profile("kinodynamic_tqc_arbitrary_subgoal")
_RESOLVED_CONFIG = dataclasses.asdict(_PROFILE)
_ARCH_FP = architecture_fingerprint(_PROFILE)
_CONTRACT_FP = local_training_contract_fingerprint(_PROFILE)


def _good_summary():
    return {
        "feasible_subgoal_success_rate": 0.8, "collision_rate": 0.05, "timeout_rate": 0.1,
        "high_risk_failure_rate": 0.02,
        "infeasible_valid_count": 3, "infeasible_false_success_rate": 0.0, "infeasible_collision_rate": 0.0,
        "infeasible_high_risk_rate": 0.0, "infeasible_safe_termination_rate": 1.0,
    }


def _artifact(summary=None, num_scenarios=20, benchmark_kind="formal", **overrides):
    a = {
        "schema_version": BENCHMARK_SCHEMA_VERSION, "benchmark_kind": benchmark_kind, "checkpoint_generation": "gen-1",
        "checkpoint_sha256": "abc123", "architecture_fingerprint": _ARCH_FP,
        "local_training_contract_fingerprint": _CONTRACT_FP, "scenario_manifest_sha256": "manifest-sha",
        "num_scenarios": num_scenarios, "summary": summary or _good_summary(),
    }
    a.update(overrides)
    return a


def _checkpoint_manifest(**overrides):
    m = {
        "generation": "gen-1", "pt_sha256": "abc123", "state_dim": 128, "action_dim": 3,
        "resolved_config": _RESOLVED_CONFIG, "local_training_contract_fingerprint": _CONTRACT_FP,
    }
    m.update(overrides)
    return m


def test_default_acceptance_config_validates():
    cfg = LocalAcceptanceConfig()
    cfg.validate()


def test_out_of_range_threshold_rejected():
    cfg = LocalAcceptanceConfig(min_feasible_subgoal_success_rate=1.5)
    with pytest.raises(Exception):
        cfg.validate()


def test_load_local_acceptance_config_default_file_exists_and_validates():
    cfg = load_local_acceptance_config()
    cfg.validate()
    assert cfg.min_scenarios == 20
    assert cfg.min_infeasible_scenarios >= 1


def test_load_local_acceptance_config_unknown_field_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("min_scenarios: 5\nbogus_field: 1\n")
    with pytest.raises(Exception):
        load_local_acceptance_config(str(path))


def test_good_artifact_passes_acceptance():
    cfg = LocalAcceptanceConfig()
    report = evaluate_local_acceptance(_artifact(), cfg)
    assert report.accepted, report.reasons


def test_old_schema_version_artifact_rejected():
    """Defect-fix item 2/3: an artifact from the old schema (still using
    the removed infeasible_goal_rejection_rate field) must be rejected
    outright, never silently reinterpreted under the new field names."""
    artifact = _artifact()
    artifact["schema_version"] = 1
    report = evaluate_local_acceptance(artifact, LocalAcceptanceConfig())
    assert not report.accepted
    assert any("schema_version" in r for r in report.reasons)


def test_failed_checkpoint_artifact_rejected():
    summary = _good_summary()
    summary["feasible_subgoal_success_rate"] = 0.0
    report = evaluate_local_acceptance(_artifact(summary=summary), LocalAcceptanceConfig())
    assert not report.accepted
    assert any("feasible_subgoal_success_rate" in r for r in report.reasons)


def test_none_feasible_success_rate_is_rejected_not_treated_as_pass():
    """0 feasible scenarios (every scenario in the benchmark was
    infeasible) must fail the gate, never be silently treated as passing
    because the comparison against None is vacuously skipped."""
    summary = _good_summary()
    summary["feasible_subgoal_success_rate"] = None
    report = evaluate_local_acceptance(_artifact(summary=summary), LocalAcceptanceConfig())
    assert not report.accepted


def test_too_few_scenarios_rejected():
    report = evaluate_local_acceptance(_artifact(num_scenarios=5), LocalAcceptanceConfig(min_scenarios=20))
    assert not report.accepted
    assert any("num_scenarios" in r for r in report.reasons)


def test_too_few_infeasible_scenarios_rejected_never_fabricates_a_pass():
    """Defect-fix item 2: min_infeasible_scenarios gate -- a benchmark run
    that happened to draw 0 infeasible scenarios cannot support ANY
    infeasible-conditional conclusion and must fail promotion outright."""
    summary = _good_summary()
    summary["infeasible_valid_count"] = 0
    summary["infeasible_false_success_rate"] = None
    summary["infeasible_collision_rate"] = None
    summary["infeasible_high_risk_rate"] = None
    summary["infeasible_safe_termination_rate"] = None
    report = evaluate_local_acceptance(_artifact(summary=summary), LocalAcceptanceConfig(min_infeasible_scenarios=1))
    assert not report.accepted
    assert any("infeasible_valid_count" in r for r in report.reasons)


def test_smoke_artifact_never_accepted():
    report = evaluate_local_acceptance(_artifact(benchmark_kind="smoke"), LocalAcceptanceConfig())
    assert not report.accepted
    assert any("smoke" in r.lower() or "formal" in r.lower() for r in report.reasons)


def test_missing_required_field_rejected():
    artifact = _artifact()
    del artifact["checkpoint_sha256"]
    report = evaluate_local_acceptance(artifact, LocalAcceptanceConfig())
    assert not report.accepted


def test_promote_rejects_incomplete_checkpoint_manifest():
    manifest = _checkpoint_manifest()
    del manifest["local_training_contract_fingerprint"]
    result = promote_local_checkpoint(
        _artifact(), manifest, source_checkpoint_dir="/tmp/x", source_checkpoint_tag="final",
        target_dir="/tmp/y", copy_files=False,
    )
    assert not result.accepted
    assert any("missing required field" in r for r in result.reasons)


def test_promote_rejects_on_failed_acceptance():
    summary = _good_summary()
    summary["collision_rate"] = 0.9
    result = promote_local_checkpoint(
        _artifact(summary=summary), _checkpoint_manifest(), source_checkpoint_dir="/tmp/x",
        source_checkpoint_tag="final", target_dir="/tmp/y", copy_files=False,
    )
    assert not result.accepted


def test_promote_rejects_checkpoint_identity_mismatch():
    result = promote_local_checkpoint(
        _artifact(), _checkpoint_manifest(generation="gen-DIFFERENT"), source_checkpoint_dir="/tmp/x",
        source_checkpoint_tag="final", target_dir="/tmp/y", copy_files=False,
    )
    assert not result.accepted
    assert any("checkpoint_generation" in r for r in result.reasons)


def test_promote_rejects_training_contract_mismatch():
    result = promote_local_checkpoint(
        _artifact(), _checkpoint_manifest(local_training_contract_fingerprint="OTHER"),
        source_checkpoint_dir="/tmp/x", source_checkpoint_tag="final", target_dir="/tmp/y", copy_files=False,
    )
    assert not result.accepted
    assert any("local_training_contract_fingerprint" in r for r in result.reasons)


def test_promote_rejects_architecture_fingerprint_mismatch_from_resolved_config():
    """Defect-fix item 4: architecture identity is recomputed from the
    checkpoint's own resolved_config, never trusted as a bare manifest
    field -- an artifact claiming a fingerprint the resolved_config
    couldn't have produced is rejected."""
    result = promote_local_checkpoint(
        _artifact(architecture_fingerprint="SOMETHING-ELSE"), _checkpoint_manifest(),
        source_checkpoint_dir="/tmp/x", source_checkpoint_tag="final", target_dir="/tmp/y", copy_files=False,
    )
    assert not result.accepted
    assert any("architecture_fingerprint" in r for r in result.reasons)


def test_promote_rejects_non_positive_state_or_action_dim():
    result = promote_local_checkpoint(
        _artifact(), _checkpoint_manifest(state_dim=0), source_checkpoint_dir="/tmp/x",
        source_checkpoint_tag="final", target_dir="/tmp/y", copy_files=False,
    )
    assert not result.accepted
    assert any("state_dim" in r for r in result.reasons)


def test_promote_accepts_when_everything_matches_dry_run():
    result = promote_local_checkpoint(
        _artifact(), _checkpoint_manifest(), source_checkpoint_dir="/tmp/x", source_checkpoint_tag="final",
        target_dir="/tmp/y", copy_files=False,
    )
    assert result.accepted


def _write_real_checkpoint(gen_dir, *, generation="gen-uuid-1", pt_bytes=b"fake-model-bytes", **manifest_overrides):
    gen_dir.mkdir(parents=True)
    (gen_dir / "model.pt").write_bytes(pt_bytes)
    pt_sha256 = hashlib.sha256(pt_bytes).hexdigest()
    manifest = _checkpoint_manifest(generation=generation, pt_sha256=pt_sha256, **manifest_overrides)
    (gen_dir / "manifest.json").write_text(json.dumps(manifest))
    return manifest, pt_sha256


def test_promote_actually_copies_files_and_writes_manifest(tmp_path):
    source_dir = tmp_path / "checkpoints"
    gen_dir = source_dir / "gen-uuid-1"
    manifest, pt_sha256 = _write_real_checkpoint(gen_dir)
    os.symlink("gen-uuid-1", source_dir / "final")

    target_dir = tmp_path / "local_frozen" / "checkpoints"
    result = promote_local_checkpoint(
        _artifact(checkpoint_generation="gen-uuid-1", checkpoint_sha256=pt_sha256), None,
        source_checkpoint_dir=str(source_dir), source_checkpoint_tag="final",
        target_dir=str(target_dir), target_tag=DEFAULT_PROMOTION_TAG, copy_files=True,
    )
    assert result.accepted, result.reasons
    assert os.path.isdir(result.target_dir)
    assert os.path.isfile(os.path.join(result.target_dir, "model.pt"))
    assert os.path.isfile(os.path.join(result.target_dir, "promotion_manifest.json"))
    final_link = os.path.join(str(target_dir), DEFAULT_PROMOTION_TAG)
    assert os.path.islink(final_link)
    assert os.path.realpath(final_link) == os.path.realpath(result.target_dir)

    validation = validate_promoted_local_checkpoint(str(target_dir), DEFAULT_PROMOTION_TAG)
    assert validation.promoted, validation.reasons


def test_promote_rejects_when_caller_supplied_manifest_disagrees_with_on_disk(tmp_path):
    """Defect-fix item 4: a caller-supplied checkpoint_manifest is a
    cross-check, never trusted over the on-disk manifest.json."""
    source_dir = tmp_path / "checkpoints"
    gen_dir = source_dir / "gen-uuid-1"
    _manifest, pt_sha256 = _write_real_checkpoint(gen_dir)
    os.symlink("gen-uuid-1", source_dir / "final")

    wrong_caller_manifest = _checkpoint_manifest(generation="gen-uuid-1", pt_sha256="not-the-real-sha")
    result = promote_local_checkpoint(
        _artifact(checkpoint_generation="gen-uuid-1", checkpoint_sha256=pt_sha256), wrong_caller_manifest,
        source_checkpoint_dir=str(source_dir), source_checkpoint_tag="final",
        target_dir=str(tmp_path / "local_frozen" / "checkpoints"), copy_files=True,
    )
    assert not result.accepted
    assert any("disagrees with the on-disk checkpoint" in r for r in result.reasons)


def test_promote_rejects_tampered_model_pt_on_disk(tmp_path):
    """Defect-fix item 4: model.pt's actual sha256 must match manifest.json's
    recorded pt_sha256 -- a checkpoint file swapped out after being saved
    (or a manifest.json hand-edited independently of it) is rejected."""
    source_dir = tmp_path / "checkpoints"
    gen_dir = source_dir / "gen-uuid-1"
    _write_real_checkpoint(gen_dir)
    (gen_dir / "model.pt").write_bytes(b"TAMPERED-BYTES")  # after the manifest was written
    os.symlink("gen-uuid-1", source_dir / "final")

    result = promote_local_checkpoint(
        _artifact(checkpoint_generation="gen-uuid-1"), None,
        source_checkpoint_dir=str(source_dir), source_checkpoint_tag="final",
        target_dir=str(tmp_path / "local_frozen" / "checkpoints"), copy_files=True,
    )
    assert not result.accepted
    assert any("sha256" in r for r in result.reasons)


def test_promote_refuses_to_overwrite_a_different_existing_promoted_checkpoint(tmp_path):
    """Defect-fix item 4: promotion must never clobber an existing promoted
    checkpoint at a colliding destination -- the OLD promoted checkpoint
    must be preserved when a NEW, different promotion attempt would
    otherwise land on the same basename."""
    target_dir = tmp_path / "local_frozen" / "checkpoints"
    # Pre-existing (differently-generationed) promoted checkpoint occupying
    # the exact destination basename this promotion would also use.
    collide_dir = target_dir / "gen-uuid-1_promoted"
    collide_dir.mkdir(parents=True)
    (collide_dir / "model.pt").write_bytes(b"OLD")
    (collide_dir / "promotion_manifest.json").write_text(json.dumps({
        "checkpoint_generation": "gen-uuid-1", "checkpoint_sha256": "OLD-SHA",
    }))

    source_dir = tmp_path / "checkpoints"
    gen_dir = source_dir / "gen-uuid-1"
    _manifest, pt_sha256 = _write_real_checkpoint(gen_dir)
    os.symlink("gen-uuid-1", source_dir / "final")

    result = promote_local_checkpoint(
        _artifact(checkpoint_generation="gen-uuid-1", checkpoint_sha256=pt_sha256), None,
        source_checkpoint_dir=str(source_dir), source_checkpoint_tag="final", target_dir=str(target_dir),
        copy_files=True,
    )
    assert not result.accepted
    # The pre-existing (different) promoted checkpoint is untouched.
    assert (collide_dir / "model.pt").read_bytes() == b"OLD"


def test_validate_promoted_local_checkpoint_rejects_missing_marker(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    gen_dir = ckpt_dir / "gen-1"
    gen_dir.mkdir(parents=True)
    os.symlink("gen-1", ckpt_dir / "final")
    validation = validate_promoted_local_checkpoint(str(ckpt_dir), "final")
    assert not validation.promoted


def test_validate_promoted_local_checkpoint_rejects_empty_marker(tmp_path):
    """Defect-fix item 4: an empty {} promotion_manifest.json (e.g. from an
    interrupted write) is never treated as promoted."""
    ckpt_dir = tmp_path / "checkpoints"
    gen_dir = ckpt_dir / "gen-1"
    gen_dir.mkdir(parents=True)
    (gen_dir / "promotion_manifest.json").write_text("{}")
    os.symlink("gen-1", ckpt_dir / "final")
    validation = validate_promoted_local_checkpoint(str(ckpt_dir), "final")
    assert not validation.promoted
    assert any("empty" in r for r in validation.reasons)


def test_validate_promoted_local_checkpoint_rejects_stale_tampered_marker(tmp_path):
    """A promotion_manifest.json whose recorded identity no longer matches
    the checkpoint files sitting next to it (edited independently, or the
    files were swapped after promotion) is rejected."""
    ckpt_dir = tmp_path / "checkpoints"
    gen_dir = ckpt_dir / "gen-1"
    _manifest, pt_sha256 = _write_real_checkpoint(gen_dir, generation="gen-1")
    tampered_marker = {
        "promoted_at_unix": 0.0, "source_checkpoint_dir": "x", "source_checkpoint_tag": "final",
        "checkpoint_generation": "gen-1", "checkpoint_sha256": "NOT-THE-REAL-SHA",
        "local_training_contract_fingerprint": _CONTRACT_FP, "architecture_fingerprint": _ARCH_FP,
        "acceptance": {}, "benchmark_artifact_summary": {}, "benchmark_num_scenarios": 20,
        "benchmark_scenario_manifest_sha256": "x",
    }
    (gen_dir / "promotion_manifest.json").write_text(json.dumps(tampered_marker))
    os.symlink("gen-1", ckpt_dir / "final")
    validation = validate_promoted_local_checkpoint(str(ckpt_dir), "final")
    assert not validation.promoted
    assert any("stale or tampered" in r for r in validation.reasons)


def test_validate_promoted_local_checkpoint_accepts_a_real_promotion(tmp_path):
    source_dir = tmp_path / "checkpoints"
    gen_dir = source_dir / "gen-uuid-2"
    _manifest, pt_sha256 = _write_real_checkpoint(gen_dir, generation="gen-uuid-2")
    os.symlink("gen-uuid-2", source_dir / "final")

    target_dir = tmp_path / "local_frozen" / "checkpoints"
    result = promote_local_checkpoint(
        _artifact(checkpoint_generation="gen-uuid-2", checkpoint_sha256=pt_sha256), None,
        source_checkpoint_dir=str(source_dir), source_checkpoint_tag="final", target_dir=str(target_dir),
        copy_files=True,
    )
    assert result.accepted, result.reasons
    validation = validate_promoted_local_checkpoint(str(target_dir), DEFAULT_PROMOTION_TAG)
    assert validation.promoted, validation.reasons
