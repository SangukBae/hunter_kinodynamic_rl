"""Defect-fix item 10: formal A-G live ablation suite -- fail-fast checks
only (no live Gazebo). Every guard below fires before this function ever
touches Gazebo (profiles load -> formal-mode checks -> LiveGazeboLocalExecutor
construction, in that order), so these are exercised directly without a
live simulator."""

import pytest

from hunter_kinodynamic_rl.evaluation.run_live_ablation_suite import FORMAL_MIN_SCENARIOS, run_live_ablation_suite


def test_formal_below_min_scenarios_raises():
    with pytest.raises(RuntimeError, match="num_scenarios"):
        run_live_ablation_suite(labels=("A",), num_scenarios=FORMAL_MIN_SCENARIOS - 1, formal=True)


def test_formal_requires_test_mode():
    with pytest.raises(RuntimeError, match="mode='test'"):
        run_live_ablation_suite(labels=("A",), num_scenarios=20, mode="train", formal=True)


def test_formal_rejects_max_options_override():
    with pytest.raises(RuntimeError, match="max_options"):
        run_live_ablation_suite(labels=("A",), num_scenarios=20, max_options=1, formal=True)


def test_formal_rejects_max_local_steps_override():
    with pytest.raises(RuntimeError, match="max_options"):
        run_live_ablation_suite(labels=("A",), num_scenarios=20, max_local_steps=5, formal=True)


def test_formal_requires_promoted_local_checkpoint(tmp_path, monkeypatch):
    """A formal A-G run must refuse to proceed against a Local checkpoint
    that isn't actually PROMOTED -- exercised by pointing
    hierarchical_phase5_a's local_checkpoint_dir at an empty temp
    directory (never promoted)."""
    import dataclasses

    from hunter_kinodynamic_rl.config import loader as config_loader

    original_load_profile = config_loader.load_profile

    def _patched_load_profile(name, *a, **k):
        profile = original_load_profile(name, *a, **k)
        profile = dataclasses.replace(
            profile, hierarchical_training=dataclasses.replace(
                profile.hierarchical_training, local_checkpoint_dir=str(tmp_path), local_checkpoint_name="final",
            ),
        )
        return profile

    monkeypatch.setattr(config_loader, "load_profile", _patched_load_profile)
    with pytest.raises(RuntimeError, match="PROMOTED"):
        run_live_ablation_suite(labels=("A",), num_scenarios=20, formal=True)


def test_non_formal_mode_does_not_enforce_min_scenarios_or_promotion():
    # Not asserting success (still needs live Gazebo past this point) --
    # only that formal=False (default) does NOT raise for the formal-only
    # reasons before reaching the (expected, in a bare-host/no-Gazebo test)
    # Gazebo connection error.
    with pytest.raises(Exception) as exc_info:
        run_live_ablation_suite(labels=("A",), num_scenarios=1, formal=False)
    assert "num_scenarios" not in str(exc_info.value)
    assert "PROMOTED" not in str(exc_info.value)
