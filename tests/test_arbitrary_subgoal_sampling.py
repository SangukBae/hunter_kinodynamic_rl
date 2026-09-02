"""hierarchical Phase 2 (detailed spec "Local TQC 통합"): arbitrary
short-range subgoal training distribution --
``scenario.goal_sampling_mode="robot_relative_band"`` in
``env/scenarios/procedural_generator.generate_scenario``.

Covers:
  - goal lands within the configured distance range and one of the
    configured direction sectors (relative to start_yaw), for many seeds.
  - default ``goal_sampling_mode="uniform_world"`` is BYTE-IDENTICAL to the
    pre-existing goal draw (regression guard for every other profile).
  - ``goal_sampling_mode="robot_relative_band"`` requires
    ``start_pose.heading_mode="legacy_random"`` -- fails fast, never
    silently reorders the scenario draw sequence.
  - ``goal_infeasible_fraction`` actually produces some scenarios whose
    goal is NOT grid-BFS-reachable from the start (blocked/unreachable
    subgoals in the training distribution), while ``0.0`` (default) never
    does.
  - the active-subgoal observation contract
    (``navigation/local_rl/controller.py``) never reads a final goal --
    confirms by construction that the arbitrary subgoal produced here is
    exactly what the caller would hand ``LocalPolicyController``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hunter_kinodynamic_rl.config.schema import ConfigError, ScenarioConfig, StartPoseConfig
from hunter_kinodynamic_rl.env.scenarios.procedural_generator import (
    generate_scenario, is_reachable,
)

BAND_CFG_KWARGS = dict(
    world_size_m=16.0, min_obstacles=2, max_obstacles=10,
    goal_sampling_mode="robot_relative_band",
    goal_distance_range_m=[2.0, 6.0],
    goal_direction_sectors_deg=[[-60.0, 60.0], [60.0, 150.0], [-150.0, -60.0]],
)


def _angle_in_any_sector(relative_angle_deg: float, sectors) -> bool:
    for lo, hi in sectors:
        if lo <= relative_angle_deg <= hi:
            return True
    return False


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 5, 13, 42, 100, 207, 999, 5000])
def test_band_mode_goal_within_distance_and_sector(seed):
    cfg = ScenarioConfig(**BAND_CFG_KWARGS)
    spec = generate_scenario(seed, cfg, robot_radius=0.45, start_pose_cfg=StartPoseConfig())

    dx, dy = spec.goal_x - spec.start_x, spec.goal_y - spec.start_y
    distance = math.hypot(dx, dy)
    lo, hi = cfg.goal_distance_range_m
    assert lo - 1e-6 <= distance <= hi + 1e-6

    absolute_angle = math.atan2(dy, dx)
    relative_deg = math.degrees(
        (absolute_angle - spec.start_yaw + math.pi) % (2 * math.pi) - math.pi
    )
    assert _angle_in_any_sector(relative_deg, cfg.goal_direction_sectors_deg), (
        f"goal at relative angle {relative_deg:.1f} deg not in any configured sector"
    )


def test_band_mode_is_seed_deterministic():
    cfg = ScenarioConfig(**BAND_CFG_KWARGS)
    a = generate_scenario(777, cfg, robot_radius=0.45, start_pose_cfg=StartPoseConfig())
    b = generate_scenario(777, cfg, robot_radius=0.45, start_pose_cfg=StartPoseConfig())
    assert (a.start_x, a.start_y, a.start_yaw, a.goal_x, a.goal_y) == (
        b.start_x, b.start_y, b.start_yaw, b.goal_x, b.goal_y,
    )


def test_default_uniform_world_mode_matches_legacy_rng_draw_order():
    """goal_sampling_mode's default must not perturb the pre-existing
    uniform_world draw for ANY existing profile -- byte-identical RNG
    stream to a ScenarioConfig built without the new fields at all."""
    legacy_cfg = ScenarioConfig(world_size_m=12.0, min_obstacles=2, max_obstacles=8)
    new_field_cfg = ScenarioConfig(
        world_size_m=12.0, min_obstacles=2, max_obstacles=8, goal_sampling_mode="uniform_world",
    )
    for seed in (0, 1, 42, 999):
        a = generate_scenario(seed, legacy_cfg, robot_radius=0.3)
        b = generate_scenario(seed, new_field_cfg, robot_radius=0.3)
        assert a == b


def test_band_mode_requires_legacy_random_heading_mode():
    cfg = ScenarioConfig(**BAND_CFG_KWARGS)
    with pytest.raises(RuntimeError, match="legacy_random"):
        generate_scenario(
            0, cfg, robot_radius=0.45, start_pose_cfg=StartPoseConfig(heading_mode="goal_biased"),
        )


def test_goal_infeasible_fraction_zero_never_yields_unreachable_goal():
    cfg = ScenarioConfig(**{**BAND_CFG_KWARGS, "goal_infeasible_fraction": 0.0})
    for seed in range(30):
        spec = generate_scenario(seed, cfg, robot_radius=0.45, start_pose_cfg=StartPoseConfig())
        assert is_reachable(
            (spec.start_x, spec.start_y), (spec.goal_x, spec.goal_y),
            spec.static_obstacles, cfg.world_size_m, robot_radius=0.45,
        )


def test_goal_infeasible_fraction_one_always_yields_a_verified_unreachable_goal():
    """item 3 fix: force_infeasible_ok now constructs a real goal-encircling
    obstacle ring (_build_goal_blocking_ring) instead of merely skipping
    the reachability check -- every accepted scenario with
    intended_infeasible=True must be independently re-verifiable as
    grid-BFS-unreachable (never just trust the internal bookkeeping)."""
    cfg = ScenarioConfig(**{**BAND_CFG_KWARGS, "goal_infeasible_fraction": 1.0})
    for seed in range(40):
        spec = generate_scenario(seed, cfg, robot_radius=0.45, start_pose_cfg=StartPoseConfig())
        assert spec.intended_infeasible is True
        assert spec.realized_infeasible is True
        assert spec.infeasibility_kind == "goal_encircled_blocked"
        assert not is_reachable(
            (spec.start_x, spec.start_y), (spec.goal_x, spec.goal_y),
            spec.static_obstacles, cfg.world_size_m, robot_radius=0.45,
        ), f"seed {seed}: intended_infeasible scenario is still grid-BFS-reachable"
        # start position/footprint must stay collision-free -- the ring
        # must never have been allowed to swallow the start too.
        assert is_reachable(
            (spec.start_x, spec.start_y), (spec.start_x, spec.start_y),
            spec.static_obstacles, cfg.world_size_m, robot_radius=0.45,
        ), f"seed {seed}: start position itself is not collision-free"


def test_goal_infeasible_fraction_realistic_sample_has_no_false_positive_negatives():
    """500-seed sample at the profile's actual configured ratio (0.15,
    kinodynamic_tqc_arbitrary_subgoal.yaml): every scenario the generator
    LABELS as intended_infeasible must be independently verified as
    actually grid-unreachable (zero false-positive negative labels) --
    the exact statistic the live report found broken (0 actually-
    unreachable outcomes observed in a 500-seed sample under the old
    skip-the-check behaviour). Also checks the realized fraction lands in
    a reasonable statistical band around the configured 0.15 (each
    scenario draw independently decides force_infeasible_ok via its own
    rng.random() call, a Bernoulli trial per seed -- not exactly 15% of
    500, but not wildly off either)."""
    cfg = ScenarioConfig(**{**BAND_CFG_KWARGS, "goal_infeasible_fraction": 0.15, "max_obstacles": 6})
    n = 500
    intended_count = 0
    false_positive_negatives = 0
    for seed in range(n):
        spec = generate_scenario(seed, cfg, robot_radius=0.45, start_pose_cfg=StartPoseConfig())
        if spec.intended_infeasible:
            intended_count += 1
            assert spec.realized_infeasible is True
            if is_reachable(
                (spec.start_x, spec.start_y), (spec.goal_x, spec.goal_y),
                spec.static_obstacles, cfg.world_size_m, robot_radius=0.45,
            ):
                false_positive_negatives += 1
        else:
            assert spec.realized_infeasible is False
            assert spec.infeasibility_kind is None
            assert is_reachable(
                (spec.start_x, spec.start_y), (spec.goal_x, spec.goal_y),
                spec.static_obstacles, cfg.world_size_m, robot_radius=0.45,
            ), f"seed {seed}: target-positive scenario is not actually feasible"
    assert false_positive_negatives == 0, (
        f"{false_positive_negatives}/{intended_count} intended-infeasible scenarios were still reachable"
    )
    ratio = intended_count / n
    # Generous statistical band (0.05-0.30 around a configured 0.15) --
    # this is a smoke check that the fraction has a real, roughly-correct
    # effect, not a tight binomial confidence interval.
    assert 0.05 <= ratio <= 0.30, f"realized infeasible fraction {ratio:.3f} far from configured 0.15"


def test_goal_infeasible_fraction_is_deterministic_per_seed():
    cfg = ScenarioConfig(**{**BAND_CFG_KWARGS, "goal_infeasible_fraction": 0.15})
    for seed in (0, 7, 123, 4321):
        a = generate_scenario(seed, cfg, robot_radius=0.45, start_pose_cfg=StartPoseConfig())
        b = generate_scenario(seed, cfg, robot_radius=0.45, start_pose_cfg=StartPoseConfig())
        assert a == b


def test_infeasibility_metadata_never_referenced_by_observation_or_reward_code():
    """item 3: intended_infeasible/realized_infeasible/infeasibility_kind
    are privileged, evaluation-only fields (mirrors
    LongHorizonWorld.topology_metadata's own information-boundary
    contract) -- must never be read by the observation builder or reward
    calculator. A cheap, effective static guard against a future
    accidental leak: these field names must not appear anywhere in either
    module's source."""
    import inspect

    from hunter_kinodynamic_rl.env.observation import observation_builder
    from hunter_kinodynamic_rl.env.rewards import reward_calculator

    forbidden = ("intended_infeasible", "realized_infeasible", "infeasibility_kind")
    for module in (observation_builder, reward_calculator):
        source = inspect.getsource(module)
        for name in forbidden:
            assert name not in source, f"{module.__name__} must never reference ScenarioSpec.{name}"


@pytest.mark.parametrize("bad_field,bad_value", [
    ("goal_sampling_mode", "not_a_mode"),
    ("goal_distance_range_m", [6.0, 2.0]),
    ("goal_distance_range_m", [0.0, 6.0]),
    ("goal_direction_sectors_deg", []),
    ("goal_direction_sectors_deg", [[60.0, -60.0]]),
    ("goal_infeasible_fraction", 1.5),
    ("goal_infeasible_fraction", -0.1),
])
def test_invalid_arbitrary_subgoal_config_rejected(bad_field, bad_value):
    cfg = ScenarioConfig(**{bad_field: bad_value})
    with pytest.raises(ConfigError):
        cfg.validate()
