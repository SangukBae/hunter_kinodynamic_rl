"""Requirement D: formal A/B benchmark mode -- fail-fast checks only (no
live Gazebo). Both guards below fire before this function ever touches
Gazebo (profile load -> formal-mode checks -> LiveGazeboLocalExecutor
construction, in that order), so these are exercised directly without a
live simulator."""

import pytest

from hunter_kinodynamic_rl.evaluation.run_live_hierarchical_benchmark import (
    FORMAL_MIN_SCENARIOS, run_live_hierarchical_benchmark,
)


def test_formal_without_global_checkpoint_dir_raises():
    with pytest.raises(RuntimeError, match="global_checkpoint_dir"):
        run_live_hierarchical_benchmark(
            "hierarchical_phase4", num_scenarios=20, global_checkpoint_dir="", formal=True,
        )


def test_formal_below_min_scenarios_raises():
    with pytest.raises(RuntimeError, match="num_scenarios"):
        run_live_hierarchical_benchmark(
            "hierarchical_phase4", num_scenarios=FORMAL_MIN_SCENARIOS - 1,
            global_checkpoint_dir="/tmp/does_not_matter", formal=True,
        )


def test_formal_min_scenarios_constant_is_20():
    assert FORMAL_MIN_SCENARIOS == 20


def test_formal_requires_test_mode():
    """Defect-fix item 10: formal must always run mode='test', never a
    train/validation-pool seed."""
    with pytest.raises(RuntimeError, match="mode='test'"):
        run_live_hierarchical_benchmark(
            "hierarchical_phase4", num_scenarios=20, global_checkpoint_dir="/tmp/does_not_matter",
            mode="train", formal=True,
        )


def test_formal_rejects_max_options_override():
    """Defect-fix item 10: a smoke-sized max_options/max_local_steps
    override must never be silently accepted under formal=True."""
    with pytest.raises(RuntimeError, match="max_options"):
        run_live_hierarchical_benchmark(
            "hierarchical_phase4", num_scenarios=20, global_checkpoint_dir="/tmp/does_not_matter",
            max_options=1, formal=True,
        )


def test_formal_rejects_max_local_steps_override():
    with pytest.raises(RuntimeError, match="max_options"):
        run_live_hierarchical_benchmark(
            "hierarchical_phase4", num_scenarios=20, global_checkpoint_dir="/tmp/does_not_matter",
            max_local_steps=5, formal=True,
        )


def test_formal_requires_promoted_local_checkpoint(tmp_path, monkeypatch):
    """Defect-fix item 10: a formal A/B run must refuse to proceed against
    a Local checkpoint that isn't actually PROMOTED -- exercised by
    pointing hierarchical_phase4's own local_checkpoint_dir at an empty
    temp directory (never promoted)."""
    import dataclasses

    from hunter_kinodynamic_rl.config import loader as config_loader

    original_load_profile = config_loader.load_profile

    def _patched_load_profile(name, *a, **k):
        profile = original_load_profile(name, *a, **k)
        if name == "hierarchical_phase4":
            profile = dataclasses.replace(
                profile, hierarchical_training=dataclasses.replace(
                    profile.hierarchical_training, local_checkpoint_dir=str(tmp_path), local_checkpoint_name="final",
                ),
            )
        return profile

    monkeypatch.setattr(config_loader, "load_profile", _patched_load_profile)
    with pytest.raises(RuntimeError, match="PROMOTED"):
        run_live_hierarchical_benchmark(
            "hierarchical_phase4", num_scenarios=20, global_checkpoint_dir="/tmp/does_not_matter", formal=True,
        )


def test_non_formal_mode_does_not_enforce_scenario_count_or_checkpoint():
    # Not asserting success (still needs live Gazebo past this point) --
    # only that formal=False (default) does NOT raise for THESE reasons
    # before reaching the (expected, in a bare-host/no-Gazebo test) Gazebo
    # connection error.
    with pytest.raises(Exception) as exc_info:
        run_live_hierarchical_benchmark(
            "hierarchical_phase4", num_scenarios=1, global_checkpoint_dir="", formal=False,
        )
    assert "global_checkpoint_dir" not in str(exc_info.value)
    assert "num_scenarios" not in str(exc_info.value) or "formal=True" not in str(exc_info.value)
