"""section item-1 (round 2): evaluation/contract_override.py's write/load/
apply round-trip -- pure, ROS-free (the file-based delivery mechanism
itself; environment_node.py's own wiring is covered by
test_environment_node.py)."""

import pytest

from hunter_kinodynamic_rl.config.loader import load_profile
from hunter_kinodynamic_rl.evaluation.contract_override import (
    CONTRACT_SECTION_NAMES, apply_evaluation_contract, load_evaluation_contract_override,
    write_evaluation_contract_override,
)


def test_write_then_load_round_trips_the_four_contract_sections(tmp_path):
    profile = load_profile("evaluation_id")
    path = str(tmp_path / "override.yaml")
    write_evaluation_contract_override(path, profile)

    sections = load_evaluation_contract_override(path)
    assert set(sections.keys()) == set(CONTRACT_SECTION_NAMES)
    assert sections["reward"] == profile.reward
    assert sections["scenario"] == profile.scenario
    assert sections["runtime"] == profile.runtime
    assert sections["evaluation"] == profile.evaluation


def test_apply_evaluation_contract_replaces_only_the_four_sections():
    training_profile = load_profile("kinodynamic_tqc")
    eval_profile = load_profile("evaluation_id")
    sections = {name: getattr(eval_profile, name) for name in CONTRACT_SECTION_NAMES}

    effective = apply_evaluation_contract(training_profile, sections)

    assert effective.reward == eval_profile.reward
    assert effective.scenario == eval_profile.scenario
    assert effective.runtime == eval_profile.runtime
    assert effective.evaluation == eval_profile.evaluation
    # Everything architecture-owned (and training-loop-only) stays from
    # the TRAINING profile, untouched.
    assert effective.action_space == training_profile.action_space
    assert effective.features == training_profile.features
    assert effective.observation == training_profile.observation
    assert effective.robot == training_profile.robot
    assert effective.training == training_profile.training


def test_load_evaluation_contract_override_rejects_an_unknown_top_level_section(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("reward:\n  goal_threshold_m: 0.5\nnot_a_real_section:\n  foo: 1\n")
    with pytest.raises(ValueError, match="not_a_real_section"):
        load_evaluation_contract_override(str(path))


def test_load_evaluation_contract_override_rejects_an_unknown_key_within_a_section(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("reward:\n  this_key_does_not_exist: 1\n")
    with pytest.raises(Exception):  # ConfigError from _section_from_dict
        load_evaluation_contract_override(str(path))


def test_load_evaluation_contract_override_defaults_missing_sections():
    """A YAML that only sets SOME sections -- the rest default to each
    section dataclass's own defaults (never crash on a partial file)."""
    import tempfile
    import os as os_mod
    with tempfile.TemporaryDirectory() as d:
        path = os_mod.path.join(d, "partial.yaml")
        with open(path, "w") as f:
            f.write("reward:\n  goal_threshold_m: 0.7\n")
        sections = load_evaluation_contract_override(path)
        assert sections["reward"].goal_threshold_m == pytest.approx(0.7)
        from hunter_kinodynamic_rl.config.schema import EvaluationConfig, RuntimeConfig, ScenarioConfig
        assert sections["scenario"] == ScenarioConfig()
        assert sections["runtime"] == RuntimeConfig()
        assert sections["evaluation"] == EvaluationConfig()


def test_apply_then_fingerprint_matches_the_source_evaluation_profile():
    """End-to-end sanity: write -> load -> apply -> fingerprint must equal
    fingerprinting the SOURCE evaluation profile directly -- the whole
    point of this round trip is to deliver an IDENTICAL contract to a live
    environment_node."""
    from hunter_kinodynamic_rl.evaluation.fingerprint import evaluation_contract_fingerprint

    training_profile = load_profile("kinodynamic_tqc")
    eval_profile = load_profile("evaluation_id")

    import tempfile
    import os as os_mod
    with tempfile.TemporaryDirectory() as d:
        path = os_mod.path.join(d, "override.yaml")
        write_evaluation_contract_override(path, eval_profile)
        sections = load_evaluation_contract_override(path)
        effective = apply_evaluation_contract(training_profile, sections)

    assert evaluation_contract_fingerprint(effective) == evaluation_contract_fingerprint(eval_profile)


def test_contract_section_names_matches_fingerprint_evaluation_contract_sections():
    """The two lists (this module's CONTRACT_SECTION_NAMES and
    fingerprint.py's EVALUATION_CONTRACT_SECTIONS) must name the exact
    SAME set of sections -- otherwise a section could be fingerprinted
    without ever being deliverable, or delivered without ever being
    verified."""
    from hunter_kinodynamic_rl.evaluation.fingerprint import EVALUATION_CONTRACT_SECTIONS

    assert set(CONTRACT_SECTION_NAMES) == set(EVALUATION_CONTRACT_SECTIONS)
