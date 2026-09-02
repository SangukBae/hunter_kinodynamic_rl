"""Phase 5 ablation profiles (plan section 9.9/9.11): A-G must all validate
and produce DISTINCT hierarchical architecture fingerprints, so a checkpoint
trained under one ablation can never be silently confused with another.

Requirement F: B and C used to be byte-identical (Phase 4's ``visited`` map
channel was never behind its own toggle) -- ``GlobalRLConfig.include_visited_channel``
fixed this (False for B, True for C), so C now has its own profile file and
is included in every fingerprint-distinctness check below."""

import pytest

from hunter_kinodynamic_rl.config.loader import load_profile
from hunter_kinodynamic_rl.evaluation.fingerprint import (
    HIERARCHICAL_ARCHITECTURE_SECTIONS, hierarchical_architecture_fingerprint,
)
from hunter_kinodynamic_rl.navigation.global_rl.observation import resolve_map_channel_names

_PROFILE_NAMES = {
    "A": "hierarchical_phase5_a", "B": "hierarchical_phase5_b", "C": "hierarchical_phase5_c",
    "D": "hierarchical_phase5_d", "E": "hierarchical_phase5_e", "F": "hierarchical_phase5_f",
    "G": "hierarchical_phase5_g",
}


def test_every_ablation_profile_validates():
    for label, name in _PROFILE_NAMES.items():
        profile = load_profile(name)
        assert profile.name == name


def test_every_profile_uses_the_canonical_local_checkpoint_tag_including_phase4():
    """Defect-fix item 5: the Runbook promotes to "final"
    (evaluation.local_promotion.promote_local_checkpoint's own default
    target_tag), and every Phase 5 A-G profile already reads "final" --
    Phase 4 used to read "best" (a tag promote_local_checkpoint never
    writes), silently pointing at whatever stale/legacy directory happened
    to be named "best" instead of the actually-promoted checkpoint. Every
    profile that consumes a promoted Local checkpoint must agree on ONE
    canonical tag, and it must be "final"."""
    from hunter_kinodynamic_rl.evaluation.local_promotion import DEFAULT_PROMOTION_TAG

    assert DEFAULT_PROMOTION_TAG == "final"
    all_profile_names = {"hierarchical_phase4", *_PROFILE_NAMES.values()}
    for name in all_profile_names:
        profile = load_profile(name)
        assert profile.hierarchical_training.local_checkpoint_name == DEFAULT_PROMOTION_TAG, (
            f"{name}.hierarchical_training.local_checkpoint_name="
            f"{profile.hierarchical_training.local_checkpoint_name!r} != canonical tag {DEFAULT_PROMOTION_TAG!r}"
        )


def test_ablation_a_disables_global_rl_and_hierarchical_training():
    profile = load_profile("hierarchical_phase5_a")
    assert not profile.global_rl.enabled
    assert not profile.hierarchical_training.enabled
    # long_horizon_world/mapping/mission/hierarchy stay enabled so the SAME
    # benchmark manifest still generates for a local-only comparison run.
    assert profile.long_horizon_world.enabled


def test_ablation_b_is_phase4_equivalent_except_no_visited_channel():
    profile = load_profile("hierarchical_phase5_b")
    assert profile.global_rl.enabled
    assert not profile.global_rl.include_visited_channel
    assert not profile.memory.enabled
    assert not profile.feasibility.enabled
    assert not profile.global_rl.topology_feedback_enabled
    assert not profile.global_rl.feasibility_feedback_enabled
    assert not profile.global_rl.global_risk_feedback_enabled


def test_ablation_c_adds_only_visited_channel():
    profile = load_profile("hierarchical_phase5_c")
    assert profile.global_rl.enabled
    assert profile.global_rl.include_visited_channel
    assert not profile.memory.enabled
    assert not profile.feasibility.enabled
    assert not profile.global_rl.topology_feedback_enabled
    assert not profile.global_rl.feasibility_feedback_enabled
    assert not profile.global_rl.global_risk_feedback_enabled


def test_ablation_b_and_c_have_different_map_channel_counts():
    b = load_profile("hierarchical_phase5_b").global_rl
    c = load_profile("hierarchical_phase5_c").global_rl
    assert len(resolve_map_channel_names(b)) == len(resolve_map_channel_names(c)) - 1
    assert "visited" not in resolve_map_channel_names(b)
    assert "visited" in resolve_map_channel_names(c)


def test_ablation_d_adds_only_topology():
    profile = load_profile("hierarchical_phase5_d")
    assert profile.memory.enabled and profile.global_rl.topology_feedback_enabled
    assert not profile.global_rl.feasibility_feedback_enabled
    assert not profile.global_rl.global_risk_feedback_enabled


def test_ablation_e_adds_topology_and_feasibility():
    profile = load_profile("hierarchical_phase5_e")
    assert profile.global_rl.topology_feedback_enabled
    assert profile.global_rl.feasibility_feedback_enabled
    assert not profile.global_rl.global_risk_feedback_enabled


def test_ablation_f_adds_only_global_risk():
    profile = load_profile("hierarchical_phase5_f")
    assert not profile.global_rl.topology_feedback_enabled
    assert not profile.global_rl.feasibility_feedback_enabled
    assert profile.global_rl.global_risk_feedback_enabled


def test_ablation_g_is_full_system():
    profile = load_profile("hierarchical_phase5_g")
    assert profile.global_rl.topology_feedback_enabled
    assert profile.global_rl.feasibility_feedback_enabled
    assert profile.global_rl.global_risk_feedback_enabled


def test_ablation_fingerprints_are_all_distinct():
    """plan 9.11: ablation checkpoint와 결과가 서로 다른 architecture
    fingerprint로 구분된다 -- all 7 labels now (requirement F closed the
    former B==C exception)."""
    fingerprints = {}
    for label in ("A", "B", "C", "D", "E", "F", "G"):
        profile = load_profile(_PROFILE_NAMES[label])
        fingerprints[label] = hierarchical_architecture_fingerprint(profile)
    assert len(set(fingerprints.values())) == len(fingerprints)  # all 7 distinct


def test_hierarchical_architecture_fingerprint_never_touches_local_only_sections():
    """The Phase 5 fingerprint must be a SEPARATE contract from the
    LOCAL-only architecture_fingerprint -- never silently folded together
    (would break comparability of every already-recorded local-only
    fingerprint)."""
    from hunter_kinodynamic_rl.evaluation.fingerprint import ARCHITECTURE_SECTIONS
    assert set(HIERARCHICAL_ARCHITECTURE_SECTIONS).isdisjoint(set(ARCHITECTURE_SECTIONS))


def test_fingerprint_changes_if_a_single_reward_weight_changes():
    profile = load_profile("hierarchical_phase5_g")
    fp_before = hierarchical_architecture_fingerprint(profile)
    profile.global_rl.predicted_risk_penalty_scale += 1.0
    fp_after = hierarchical_architecture_fingerprint(profile)
    assert fp_before != fp_after
