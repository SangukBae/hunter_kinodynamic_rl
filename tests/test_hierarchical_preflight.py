"""Requirement C / defect-fix item 11: Global hierarchical training
preflight validator."""

import json
import os

import pytest

from hunter_kinodynamic_rl.training.hierarchical_preflight import GlobalPreflightError, run_global_preflight


def test_ros_free_dry_profile_passes_without_live_or_ros():
    report = run_global_preflight("hierarchical_phase4", live=False, resume=False)
    assert report.ok, report.errors
    assert report.hierarchical_architecture_fingerprint
    assert report.global_replay_schema_version is not None


def test_non_hierarchical_profile_fails():
    report = run_global_preflight("kinodynamic_tqc_arbitrary_subgoal", live=False, require_ros=False)
    assert not report.ok
    assert any("hierarchical_training" in e or "global_rl" in e for e in report.errors)


def test_missing_profile_fails_cleanly():
    report = run_global_preflight("this_profile_does_not_exist", live=False, require_ros=False)
    assert not report.ok


def test_live_fresh_run_without_promoted_local_checkpoint_fails(tmp_path, monkeypatch):
    # hierarchical_phase4's local_checkpoint_dir points at a relative path
    # that (in this sandboxed test) certainly has no promotion_manifest.json.
    report = run_global_preflight(
        "hierarchical_phase4", live=True, resume=False, require_promoted_local=True, require_ros=False,
    )
    assert not report.ok
    assert any("PROMOTED" in e for e in report.errors)


def test_live_fresh_run_with_require_promoted_local_false_only_warns(tmp_path):
    report = run_global_preflight(
        "hierarchical_phase4", live=True, resume=False, require_promoted_local=False, require_ros=False,
    )
    assert not any("PROMOTED" in e for e in report.errors)


def test_live_resume_run_does_not_require_promotion_gate():
    report = run_global_preflight(
        "hierarchical_phase4", live=True, resume=True, require_promoted_local=True, require_ros=False,
    )
    assert not any("fresh (non-resume)" in e for e in report.errors)


def test_promoted_checkpoint_dir_passes_gate(tmp_path):
    """Defect-fix item 4: is_local_checkpoint_promoted now requires a
    COMPLETE, self-consistent checkpoint (manifest.json + model.pt whose
    actual sha256 matches) plus a promotion_manifest.json that agrees with
    it -- a bare empty {} marker (the old fixture here) is no longer
    treated as "promoted" (see test_local_promotion.py's own dedicated
    empty-marker-rejection test); this fixture builds a real one instead."""
    import dataclasses
    import hashlib
    import json

    from hunter_kinodynamic_rl.config.loader import load_profile
    from hunter_kinodynamic_rl.evaluation.fingerprint import (
        architecture_fingerprint, local_training_contract_fingerprint,
    )
    from hunter_kinodynamic_rl.evaluation.local_promotion import is_local_checkpoint_promoted

    ckpt_profile = load_profile("kinodynamic_tqc_arbitrary_subgoal")
    resolved_config = dataclasses.asdict(ckpt_profile)
    arch_fp = architecture_fingerprint(ckpt_profile)
    contract_fp = local_training_contract_fingerprint(ckpt_profile)

    ckpt_dir = tmp_path / "checkpoints"
    gen_dir = ckpt_dir / "gen-1"
    gen_dir.mkdir(parents=True)
    pt_bytes = b"fake-model-bytes"
    (gen_dir / "model.pt").write_bytes(pt_bytes)
    pt_sha256 = hashlib.sha256(pt_bytes).hexdigest()
    checkpoint_manifest = {
        "generation": "gen-1", "pt_sha256": pt_sha256, "state_dim": 128, "action_dim": 3,
        "resolved_config": resolved_config, "local_training_contract_fingerprint": contract_fp,
    }
    (gen_dir / "manifest.json").write_text(json.dumps(checkpoint_manifest))
    (gen_dir / "promotion_manifest.json").write_text(json.dumps({
        "promoted_at_unix": 0.0, "source_checkpoint_dir": "x", "source_checkpoint_tag": "final",
        "checkpoint_generation": "gen-1", "checkpoint_sha256": pt_sha256,
        "local_training_contract_fingerprint": contract_fp, "architecture_fingerprint": arch_fp,
        "acceptance": {}, "benchmark_artifact_summary": {}, "benchmark_num_scenarios": 20,
        "benchmark_scenario_manifest_sha256": "x",
    }))
    os.symlink("gen-1", ckpt_dir / "final")

    profile = load_profile("hierarchical_phase4")
    profile.hierarchical_training = dataclasses.replace(
        profile.hierarchical_training, local_checkpoint_dir=str(ckpt_dir), local_checkpoint_name="final",
    )

    assert is_local_checkpoint_promoted(str(ckpt_dir), "final")


def test_raise_if_failed_raises():
    report = run_global_preflight("this_profile_does_not_exist", live=False, require_ros=False)
    with pytest.raises(GlobalPreflightError):
        report.raise_if_failed()


def test_feasibility_tier_profile_rejected_without_live():
    """Defect-fix item 11: an E/F/G-tier profile (feasibility/global-risk
    feedback) has no real LocalFeasibilityEvaluator in the ROS-free
    stand-in -- live=False must be rejected outright, never silently run
    with zero-filled feasibility features."""
    report = run_global_preflight("hierarchical_phase5_e", live=False, require_ros=False)
    assert not report.ok
    assert any("feasibility/global-risk feedback" in e for e in report.errors)


def test_feasibility_tier_profile_allowed_with_live():
    report = run_global_preflight(
        "hierarchical_phase5_e", live=True, resume=False, require_promoted_local=False, require_ros=False,
    )
    assert not any("feasibility/global-risk feedback" in e for e in report.errors)


def test_non_feasibility_profile_not_rejected_without_live():
    report = run_global_preflight("hierarchical_phase4", live=False, require_ros=False)
    assert not any("feasibility/global-risk feedback" in e for e in report.errors)


def test_promotion_reasons_surfaced_when_not_promoted():
    """Defect-fix item 11: the structured validator's reasons must be
    surfaced in the failure message, not just a generic 'not promoted'."""
    report = run_global_preflight(
        "hierarchical_phase4", live=True, resume=False, require_promoted_local=True, require_ros=False,
    )
    assert not report.ok
    assert report.local_checkpoint_promotion_reasons
    assert any(report.local_checkpoint_promotion_reasons[0] in e for e in report.errors)


def test_promoted_checkpoint_architecture_mismatch_rejected(tmp_path):
    """Defect-fix item 11: a promoted checkpoint whose recorded
    architecture_fingerprint doesn't match THIS profile's own must be
    rejected, even though it structurally passes promotion validation."""
    import dataclasses
    import hashlib
    import json

    from hunter_kinodynamic_rl.config.loader import load_profile

    profile = load_profile("hierarchical_phase4")
    gen_dir = tmp_path / "gen-1"
    gen_dir.mkdir(parents=True)
    pt_bytes = b"fake"
    (gen_dir / "model.pt").write_bytes(pt_bytes)
    pt_sha256 = hashlib.sha256(pt_bytes).hexdigest()
    (gen_dir / "manifest.json").write_text(json.dumps({
        "generation": "gen-1", "pt_sha256": pt_sha256, "state_dim": 1, "action_dim": 1,
        "resolved_config": {}, "local_training_contract_fingerprint": "x",
    }))
    (gen_dir / "promotion_manifest.json").write_text(json.dumps({
        "promoted_at_unix": 0.0, "source_checkpoint_dir": "x", "source_checkpoint_tag": "final",
        "checkpoint_generation": "gen-1", "checkpoint_sha256": pt_sha256,
        "local_training_contract_fingerprint": "x", "architecture_fingerprint": "WRONG-FINGERPRINT",
        "acceptance": {}, "benchmark_artifact_summary": {}, "benchmark_num_scenarios": 20,
        "benchmark_scenario_manifest_sha256": "x",
    }))
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    os.symlink(str(gen_dir), ckpt_dir / "final")

    profile = dataclasses.replace(
        profile, hierarchical_training=dataclasses.replace(
            profile.hierarchical_training, local_checkpoint_dir=str(ckpt_dir), local_checkpoint_name="final",
        ),
    )
    from hunter_kinodynamic_rl.training import hierarchical_preflight as preflight_module
    original = preflight_module.load_profile
    try:
        preflight_module.load_profile = lambda name, *a, **k: profile if name == "hierarchical_phase4" else original(name, *a, **k)
        report = run_global_preflight("hierarchical_phase4", live=True, resume=False, require_ros=False)
    finally:
        preflight_module.load_profile = original
    assert not report.ok
    assert any("architecture_fingerprint" in e for e in report.errors)


def test_resume_checkpoint_identity_mismatch_rejected(tmp_path):
    gen_dir = tmp_path / "run" / "checkpoints" / "gen-1"
    gen_dir.mkdir(parents=True)
    (gen_dir / "manifest.json").write_text(json.dumps({
        "hierarchical_architecture_fingerprint": "SOMETHING-ELSE",
    }))
    os.symlink("gen-1", tmp_path / "run" / "checkpoints" / "latest")

    report = run_global_preflight(
        "hierarchical_phase4", live=False, resume=True, resume_run_dir=str(tmp_path / "run"), require_ros=False,
    )
    assert not report.ok
    assert any("hierarchical_architecture_fingerprint" in e for e in report.errors)


def test_ros_gz_interfaces_reported():
    report = run_global_preflight("hierarchical_phase4", live=False, require_ros=False)
    assert isinstance(report.ros_gz_interfaces_available, bool)
