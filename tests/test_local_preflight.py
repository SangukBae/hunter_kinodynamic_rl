"""Requirement A / defect-fix item 11: Local TQC training preflight
validator."""

import json
import os

import pytest

from hunter_kinodynamic_rl.training import preflight as preflight_module
from hunter_kinodynamic_rl.training.preflight import (
    PreflightError, _binomial_confidence_half_width, _sectors_cover_full_circle, run_local_preflight,
)


def test_sectors_cover_full_circle_single_interval():
    assert _sectors_cover_full_circle([[-180.0, 180.0]])


def test_sectors_cover_full_circle_four_quadrants():
    assert _sectors_cover_full_circle([[-180.0, -90.0], [-90.0, 0.0], [0.0, 90.0], [90.0, 180.0]])


def test_sectors_with_gap_do_not_cover_full_circle():
    assert not _sectors_cover_full_circle([[-60.0, 60.0], [60.0, 150.0], [-150.0, -60.0]])


def test_sectors_empty_do_not_cover():
    assert not _sectors_cover_full_circle([])


def test_arbitrary_subgoal_profile_passes_preflight_without_ros():
    report = run_local_preflight(
        "kinodynamic_tqc_arbitrary_subgoal", require_ros=False, sample_count=60,
        run_root="/tmp/hunter_kinodynamic_rl_preflight_test",
    )
    assert report.full_circle_goal_direction
    assert report.goal_infeasible_fraction_configured > 0.0
    assert report.goal_infeasible_fraction_measured is not None
    assert report.state_dim is not None and report.action_dim == 3
    assert report.resolved_config_hash
    assert report.local_training_contract_fingerprint
    assert report.ok, report.errors


def test_arbitrary_subgoal_smoke_profile_passes_the_same_distribution_gate():
    report = run_local_preflight(
        "smoke_test_arbitrary_subgoal", require_ros=False, sample_count=60,
        run_root="/tmp/hunter_kinodynamic_rl_preflight_test",
    )
    assert report.ok, report.errors
    assert report.full_circle_goal_direction
    assert report.goal_infeasible_fraction_configured == pytest.approx(0.15)


def test_uniform_world_profile_fails_full_circle_requirement():
    # kinodynamic_tqc_risk (or any pre-existing non-band profile) never
    # opted into the arbitrary-subgoal distribution -- preflight must
    # reject it, not silently treat it as compatible.
    report = run_local_preflight("kinodynamic_tqc_risk", require_ros=False, sample_count=10,
                                  run_root="/tmp/hunter_kinodynamic_rl_preflight_test")
    assert not report.ok
    assert any("goal_sampling_mode" in e for e in report.errors)


def test_missing_profile_fails_cleanly():
    report = run_local_preflight("this_profile_does_not_exist", require_ros=False,
                                  run_root="/tmp/hunter_kinodynamic_rl_preflight_test")
    assert not report.ok
    assert report.errors


def test_resume_true_without_resume_run_dir_fails():
    report = run_local_preflight(
        "kinodynamic_tqc_arbitrary_subgoal", require_ros=False, sample_count=5, resume=True, resume_run_dir=None,
        run_root="/tmp/hunter_kinodynamic_rl_preflight_test",
    )
    assert not report.ok
    assert any("nothing to resume" in e for e in report.errors)


def test_resume_true_with_nonexistent_dir_fails():
    report = run_local_preflight(
        "kinodynamic_tqc_arbitrary_subgoal", require_ros=False, sample_count=5, resume=True,
        resume_run_dir="/tmp/hunter_kinodynamic_rl_this_dir_does_not_exist_xyz",
        run_root="/tmp/hunter_kinodynamic_rl_preflight_test",
    )
    assert not report.ok
    assert not report.would_reuse_existing_run_dir


def test_fresh_start_never_reports_reuse_even_if_resume_run_dir_given():
    report = run_local_preflight(
        "kinodynamic_tqc_arbitrary_subgoal", require_ros=False, sample_count=5, resume=False,
        resume_run_dir="/tmp/hunter_kinodynamic_rl_preflight_test",
        run_root="/tmp/hunter_kinodynamic_rl_preflight_test",
    )
    assert not report.would_reuse_existing_run_dir
    assert any("IGNORED" in w for w in report.warnings)


def test_require_ros_true_fails_without_rclpy_importable_or_reports_available():
    report = run_local_preflight("kinodynamic_tqc_arbitrary_subgoal", require_ros=True, sample_count=5,
                                  run_root="/tmp/hunter_kinodynamic_rl_preflight_test")
    if not report.ros_available:
        assert not report.ok


def test_raise_if_failed_raises_preflight_error():
    report = run_local_preflight("this_profile_does_not_exist", require_ros=False,
                                  run_root="/tmp/hunter_kinodynamic_rl_preflight_test")
    with pytest.raises(PreflightError):
        report.raise_if_failed()


def test_ok_report_does_not_raise():
    report = run_local_preflight("kinodynamic_tqc_arbitrary_subgoal", require_ros=False, sample_count=60,
                                  run_root="/tmp/hunter_kinodynamic_rl_preflight_test")
    report.raise_if_failed()  # must not raise


def test_sample_count_zero_is_rejected():
    """Defect-fix item 11: sample_count must be > 0."""
    report = run_local_preflight(
        "kinodynamic_tqc_arbitrary_subgoal", require_ros=False, sample_count=0,
        run_root="/tmp/hunter_kinodynamic_rl_preflight_test",
    )
    assert not report.ok
    assert any("sample_count" in e for e in report.errors)


def test_negative_sample_count_is_rejected():
    report = run_local_preflight(
        "kinodynamic_tqc_arbitrary_subgoal", require_ros=False, sample_count=-5,
        run_root="/tmp/hunter_kinodynamic_rl_preflight_test",
    )
    assert not report.ok


def test_zero_confidence_z_is_rejected():
    report = run_local_preflight(
        "kinodynamic_tqc_arbitrary_subgoal", require_ros=False, sample_count=60,
        infeasible_fraction_confidence_z=0.0, run_root="/tmp/hunter_kinodynamic_rl_preflight_test",
    )
    assert not report.ok


def test_target_015_measured_00_boundary_bug_is_fixed(monkeypatch):
    """Defect-fix item 11's exact regression: at the historical default
    (target=0.15, fixed tolerance=0.15), a measured=0.0 (complete generator
    failure) computed abs(0.0-0.15)=0.15, and the old check was strictly
    `> tolerance`, so 0.15 > 0.15 was False -- silently PASSING a
    completely broken generator. The binomial-confidence check must catch
    this."""
    monkeypatch.setattr(preflight_module, "_measure_infeasible_fraction", lambda profile, n: 0.0)
    report = run_local_preflight(
        "kinodynamic_tqc_arbitrary_subgoal", require_ros=False, sample_count=300,
        run_root="/tmp/hunter_kinodynamic_rl_preflight_test",
    )
    assert report.goal_infeasible_fraction_configured == pytest.approx(0.15)
    assert not report.ok
    assert any("empirically measured infeasible fraction" in e for e in report.errors)


def test_measurement_within_confidence_interval_passes(monkeypatch):
    """A measurement genuinely close to the target (well within the
    binomial confidence half-width) must NOT be flagged -- proves the fix
    isn't just "always fail", only a real miscalibration is caught."""
    target = 0.15
    monkeypatch.setattr(preflight_module, "_measure_infeasible_fraction", lambda profile, n: target)
    report = run_local_preflight(
        "kinodynamic_tqc_arbitrary_subgoal", require_ros=False, sample_count=300,
        run_root="/tmp/hunter_kinodynamic_rl_preflight_test",
    )
    assert report.ok, report.errors


def test_binomial_confidence_half_width_is_finite_and_positive():
    hw = _binomial_confidence_half_width(0.15, 300, 4.0)
    assert 0.0 < hw < 0.15  # tighter than the old flat 0.15 tolerance at this sample size


def test_binomial_confidence_half_width_zero_samples_is_infinite():
    assert _binomial_confidence_half_width(0.15, 0, 4.0) == float("inf")


def test_ros_gz_interfaces_reported():
    report = run_local_preflight(
        "kinodynamic_tqc_arbitrary_subgoal", require_ros=False, sample_count=5,
        run_root="/tmp/hunter_kinodynamic_rl_preflight_test",
    )
    assert isinstance(report.ros_gz_interfaces_available, bool)


def test_require_ros_true_fails_if_ros_gz_interfaces_unavailable(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def _fake_import(name, *a, **k):
        if name == "ros_gz_interfaces":
            raise ImportError("simulated missing ros_gz_interfaces")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    report = run_local_preflight(
        "kinodynamic_tqc_arbitrary_subgoal", require_ros=True, sample_count=5,
        run_root="/tmp/hunter_kinodynamic_rl_preflight_test",
    )
    assert not report.ros_gz_interfaces_available
    if not report.ros_available:
        # rclpy import is also faked out by the same builtins patch only if
        # it too goes through __import__ normally -- either way, some
        # ros_gz_interfaces-specific failure reason must be present.
        pass
    assert any("ros_gz_interfaces" in e for e in report.errors)


def test_resume_checkpoint_identity_mismatch_is_rejected(tmp_path):
    """Defect-fix item 11: resuming into a run directory whose existing
    checkpoint was trained under a DIFFERENT local_training_contract_fingerprint
    must fail, never silently continue training under a new distribution."""
    run_dir = tmp_path / "run"
    gen_dir = run_dir / "checkpoints" / "gen-1"
    gen_dir.mkdir(parents=True)
    (gen_dir / "manifest.json").write_text(json.dumps({
        "local_training_contract_fingerprint": "SOMETHING-ELSE-ENTIRELY",
    }))
    os.symlink("gen-1", run_dir / "checkpoints" / "latest")

    report = run_local_preflight(
        "kinodynamic_tqc_arbitrary_subgoal", require_ros=False, sample_count=5, resume=True,
        resume_run_dir=str(run_dir), run_root="/tmp/hunter_kinodynamic_rl_preflight_test",
    )
    assert not report.ok
    assert any("local_training_contract_fingerprint" in e for e in report.errors)


def test_resume_checkpoint_identity_match_passes(tmp_path):
    from hunter_kinodynamic_rl.config.loader import load_profile
    from hunter_kinodynamic_rl.evaluation.fingerprint import local_training_contract_fingerprint

    profile = load_profile("kinodynamic_tqc_arbitrary_subgoal")
    real_fp = local_training_contract_fingerprint(profile)

    run_dir = tmp_path / "run"
    gen_dir = run_dir / "checkpoints" / "gen-1"
    gen_dir.mkdir(parents=True)
    (gen_dir / "manifest.json").write_text(json.dumps({"local_training_contract_fingerprint": real_fp}))
    os.symlink("gen-1", run_dir / "checkpoints" / "latest")

    report = run_local_preflight(
        "kinodynamic_tqc_arbitrary_subgoal", require_ros=False, sample_count=60, resume=True,
        resume_run_dir=str(run_dir), run_root="/tmp/hunter_kinodynamic_rl_preflight_test",
    )
    assert report.ok, report.errors


def test_resume_with_no_checkpoints_dir_yet_only_warns(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    report = run_local_preflight(
        "kinodynamic_tqc_arbitrary_subgoal", require_ros=False, sample_count=60, resume=True,
        resume_run_dir=str(run_dir), run_root="/tmp/hunter_kinodynamic_rl_preflight_test",
    )
    assert report.ok, report.errors
    assert any("no checkpoints/ directory" in w for w in report.warnings)
