"""Regression tests for the P0 counterfactual-candidate-generation fix:
the previous "append stop last, truncate to num_candidates" order silently
dropped the stop candidate in BOTH production counterfactual profiles
(smoke_test: num_candidates=4; kinodynamic_tqc_counterfactual:
num_candidates=8) and produced asymmetric left/right coverage."""

import pytest

from hunter_kinodynamic_rl.config.loader import load_profile
from hunter_kinodynamic_rl.config.schema import CounterfactualConfig, RobotConfig
from hunter_kinodynamic_rl.trajectory.action_space import TrajectoryCommand
from hunter_kinodynamic_rl.risk.counterfactual_sampler import (
    ScoredCandidate, best_candidate, progress_preserving_candidates, safer_alternative_margin,
)
from hunter_kinodynamic_rl.risk.labels import RiskLabel
from hunter_kinodynamic_rl.trajectory.trajectory_sampler import ACTOR_CANDIDATE_INDEX, generate_candidates


def make_robot(**overrides) -> RobotConfig:
    defaults = dict(
        name="hunter_se", wheelbase_m=0.547696, track_width_m=0.503404,
        wheel_radius_m=0.136, length_m=0.76, width_m=0.4, height_m=0.14,
        mass_kg=42.0, steering_limit_deg=21.58, max_forward_speed_mps=2.0,
        min_forward_speed_mps=0.0, accel_limit_mps2=6.0, brake_decel_mps2=6.0,
        steering_rate_deg_s=200.0, speed_lag_tau_sec=0.0,
    )
    defaults.update(overrides)
    return RobotConfig(**defaults)


def _has_stop_candidate(candidates):
    return any(c.v_ref == pytest.approx(0.0) for c in candidates)


def _has_left_and_right(base_kappa, candidates, tol=1e-9):
    return (any(c.kappa > base_kappa + tol for c in candidates)
            and any(c.kappa < base_kappa - tol for c in candidates))


@pytest.mark.parametrize("profile_name", ["smoke_test", "kinodynamic_tqc_counterfactual"])
def test_production_profile_includes_stop_and_symmetric_alternatives(profile_name):
    profile = load_profile(profile_name)
    robot = profile.robot
    base = TrajectoryCommand(kappa=0.0, v_ref=1.0, horizon_m=1.5)
    candidates = generate_candidates(base, robot, profile.counterfactual)

    assert candidates[ACTOR_CANDIDATE_INDEX] == base
    assert len(candidates) <= profile.counterfactual.num_candidates
    assert _has_stop_candidate(candidates), (
        f"{profile_name}: stop candidate missing from the final (possibly truncated) list "
        f"-- num_candidates={profile.counterfactual.num_candidates}"
    )
    assert _has_left_and_right(base.kappa, candidates), (
        f"{profile_name}: left/right steering alternatives are not both represented"
    )


def test_smoke_test_num_candidates_matches_config():
    profile = load_profile("smoke_test")
    assert profile.counterfactual.num_candidates == 8


def test_kinodynamic_tqc_counterfactual_num_candidates_matches_config():
    profile = load_profile("kinodynamic_tqc_counterfactual")
    assert profile.counterfactual.num_candidates == 8


def test_actor_near_steering_limit_does_not_crash_or_duplicate_actor_silently():
    robot = make_robot()
    kappa_max = robot.max_curvature
    base = TrajectoryCommand(kappa=kappa_max * 0.98, v_ref=1.0, horizon_m=1.5)  # near the limit
    cf_cfg = CounterfactualConfig(enabled=True, num_candidates=8,
                                   kappa_offsets_frac=[-1.0, -0.5, 0.5, 1.0],
                                   speed_fractions=[0.5, 1.0], include_stop_candidate=True)
    candidates = generate_candidates(base, robot, cf_cfg)

    assert candidates[0] == base
    for c in candidates:
        assert abs(c.kappa) <= kappa_max + 1e-9
    # No two candidates are bit-for-bit identical (dedup removed the clamped
    # duplicate that a +1.0*kappa_max offset would otherwise create here).
    seen = set()
    for c in candidates:
        key = (round(c.kappa, 9), round(c.v_ref, 9), round(c.horizon_m, 9))
        assert key not in seen, f"duplicate candidate survived generation: {c}"
        seen.add(key)


def test_v_ref_zero_base_does_not_explode_into_duplicate_speed_variants():
    robot = make_robot()
    base = TrajectoryCommand(kappa=0.0, v_ref=0.0, horizon_m=1.5)  # actor already chose to stop
    cf_cfg = CounterfactualConfig(enabled=True, num_candidates=8,
                                   kappa_offsets_frac=[-1.0, -0.5, 0.5, 1.0],
                                   speed_fractions=[0.5, 1.0], include_stop_candidate=True)
    candidates = generate_candidates(base, robot, cf_cfg)

    seen = set()
    for c in candidates:
        key = (round(c.kappa, 9), round(c.v_ref, 9), round(c.horizon_m, 9))
        assert key not in seen, f"v_ref=0 base produced duplicate speed-fraction candidates: {c}"
        seen.add(key)
    assert all(math_finite(c) for c in candidates)


def math_finite(c: TrajectoryCommand) -> bool:
    import math
    return math.isfinite(c.kappa) and math.isfinite(c.v_ref) and math.isfinite(c.horizon_m)


def test_include_stop_candidate_false_omits_stop():
    robot = make_robot()
    base = TrajectoryCommand(kappa=0.0, v_ref=1.0, horizon_m=1.5)
    cf_cfg = CounterfactualConfig(enabled=True, num_candidates=8,
                                   kappa_offsets_frac=[-1.0, 1.0], speed_fractions=[1.0],
                                   include_stop_candidate=False)
    candidates = generate_candidates(base, robot, cf_cfg)
    assert not _has_stop_candidate(candidates) or base.v_ref == pytest.approx(0.0)


def test_candidate_ordering_reserves_stop_before_steering_alternatives_under_truncation():
    """A tight num_candidates budget (actor + stop only) must still keep
    the stop candidate -- proving it is reserved BEFORE the steering/speed
    grid, not appended after and vulnerable to truncation."""
    robot = make_robot()
    base = TrajectoryCommand(kappa=0.0, v_ref=1.0, horizon_m=1.5)
    cf_cfg = CounterfactualConfig(enabled=True, num_candidates=2,
                                   kappa_offsets_frac=[-1.0, -0.5, 0.5, 1.0],
                                   speed_fractions=[0.5, 1.0], include_stop_candidate=True)
    candidates = generate_candidates(base, robot, cf_cfg)
    assert len(candidates) == 2
    assert candidates[0] == base
    assert candidates[1].v_ref == pytest.approx(0.0)


def test_steering_pairs_ordered_by_descending_magnitude():
    """+-1.0 (hard escape) must appear before +-0.5 (mild) when truncation
    forces a choice -- the most informative alternatives survive first."""
    robot = make_robot()
    base = TrajectoryCommand(kappa=0.0, v_ref=1.0, horizon_m=1.5)
    kappa_max = robot.max_curvature
    # actor(1) + stop(1) + exactly one pair at 1 speed_fraction(2) = 4 total, so
    # the +-0.5 pair is forced out by truncation if ordering is correct.
    cf_cfg = CounterfactualConfig(enabled=True, num_candidates=4,
                                   kappa_offsets_frac=[-1.0, -0.5, 0.5, 1.0],
                                   speed_fractions=[1.0], horizon_fractions=[1.0],
                                   include_stop_candidate=True)
    candidates = generate_candidates(base, robot, cf_cfg)
    assert len(candidates) == 4
    # index 0=actor, 1=stop, 2&3 must be the +-1.0 pair (largest magnitude), not +-0.5.
    non_actor_non_stop = candidates[2:]
    for c in non_actor_non_stop:
        assert abs(c.kappa) == pytest.approx(kappa_max, rel=1e-6)


def _scored(command, risk_score, progress):
    return ScoredCandidate(
        command=command,
        risk=RiskLabel(1.0, 2.0, False, 1.0, False, risk_score),
        goal_progress_m=progress,
    )


def test_l_candidates_are_generated_and_clamped_to_action_bounds():
    robot = make_robot()
    base = TrajectoryCommand(kappa=0.0, v_ref=1.0, horizon_m=1.5)
    cf_cfg = CounterfactualConfig(
        enabled=True, num_candidates=8, kappa_offsets_frac=[-1.0, 1.0],
        speed_fractions=[1.0], horizon_fractions=[0.1, 10.0], include_stop_candidate=True,
    )
    candidates = generate_candidates(base, robot, cf_cfg)
    horizons = {round(c.horizon_m, 6) for c in candidates}
    assert 0.5 in horizons
    assert 3.0 in horizons


def test_zero_progress_stop_cannot_beat_progress_preserving_actor():
    actor = TrajectoryCommand(kappa=0.0, v_ref=1.0, horizon_m=1.5)
    stop = TrajectoryCommand(kappa=0.0, v_ref=0.0, horizon_m=1.5)
    safe_turn = TrajectoryCommand(kappa=0.2, v_ref=0.8, horizon_m=1.5)
    scored = [
        _scored(actor, risk_score=0.8, progress=1.0),
        _scored(stop, risk_score=0.0, progress=0.0),
        _scored(safe_turn, risk_score=0.3, progress=0.85),
    ]
    cfg = CounterfactualConfig(enabled=True, min_progress_ratio=0.8, max_progress_loss_m=0.25)
    eligible = progress_preserving_candidates(scored, cfg)
    assert stop not in [c.command for c in eligible]
    assert best_candidate(scored, cfg).command == safe_turn
    assert safer_alternative_margin(scored, cfg) == pytest.approx(0.5)
