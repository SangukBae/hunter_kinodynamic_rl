"""Coverage for the safe start-position/initial-yaw sampling port (drl_agent
-> hunter_kinodynamic_rl requirement 1): ``config/schema.py``'s
``StartPoseConfig``, ``env/scenarios/safe_start.py::sample_start_yaw``, and
their wiring into ``env/scenarios/procedural_generator.py::generate_scenario``.
"""

import math

import numpy as np
import pytest

from hunter_kinodynamic_rl.config.schema import ConfigError, StartPoseConfig
from hunter_kinodynamic_rl.env.scenarios.procedural_generator import (
    StaticObstacle, STATIC_OBSTACLE_RADIUS_RANGE_M, ScenarioConfig, generate_scenario,
)
from hunter_kinodynamic_rl.env.scenarios.safe_start import sample_start_yaw

ROBOT_RADIUS = 0.45


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# --------------------------------------------------------------- schema
def test_start_pose_config_defaults_are_legacy_random_and_validate():
    cfg = StartPoseConfig()
    cfg.validate()
    assert cfg.heading_mode == "legacy_random"


@pytest.mark.parametrize("kwargs", [
    {"heading_mode": "not_a_mode"},
    {"min_wall_clearance_m": -1.0},
    {"min_obstacle_clearance_m": -1.0},
    {"front_safety_distance_m": 0.0},
    {"front_cone_half_angle_rad": 0.0},
    {"front_cone_half_angle_rad": 4.0},
    {"max_sampling_attempts": 0},
    {"goal_bias_prob": 1.5},
    {"goal_bias_spread_rad": -0.1},
    {"free_space_probe_count": 0},
])
def test_start_pose_config_rejects_invalid_values(kwargs):
    with pytest.raises(ConfigError):
        StartPoseConfig(**kwargs).validate()


# ------------------------------------------------- generate_scenario: legacy
def test_legacy_random_heading_mode_matches_pre_existing_rng_draw_order():
    """heading_mode='legacy_random' (the default) must draw start_yaw with
    ``rng.uniform(-pi, pi)`` immediately after (start_x, start_y) -- i.e.
    reproduce EXACTLY what generate_scenario did before this feature
    existed, for every profile that never sets start_pose at all.

    Uses min_obstacles=max_obstacles=0 so the very first draw always
    succeeds (no is_reachable/min_obstacles retry can consume extra RNG
    draws) -- this isolates the RNG DRAW ORDER claim from
    generate_scenario's unrelated retry machinery, which a fragile
    seed-by-seed replay would otherwise intermittently trip over whenever a
    seed happens to need a retry."""
    cfg = ScenarioConfig(world_size_m=12.0, min_obstacles=0, max_obstacles=0)
    for seed in (0, 1, 7, 42, 999):
        rng = np.random.RandomState(seed)
        half = cfg.world_size_m / 2.0 - ROBOT_RADIUS
        expected_start = rng.uniform(-half, half, size=2)
        expected_yaw = rng.uniform(-np.pi, np.pi)

        spec = generate_scenario(seed, cfg, robot_radius=ROBOT_RADIUS)
        assert spec.start_x == pytest.approx(expected_start[0])
        assert spec.start_y == pytest.approx(expected_start[1])
        assert spec.start_yaw == pytest.approx(expected_yaw)


# --------------------------------------------------------- determinism
@pytest.mark.parametrize("mode", ["random_rejected", "goal_biased", "free_space_biased"])
def test_same_seed_reproduces_identical_start_goal_yaw_and_obstacles(mode):
    cfg = ScenarioConfig(world_size_m=12.0, min_obstacles=2, max_obstacles=5)
    sp_cfg = StartPoseConfig(heading_mode=mode)
    spec_a = generate_scenario(123, cfg, robot_radius=ROBOT_RADIUS, start_pose_cfg=sp_cfg)
    spec_b = generate_scenario(123, cfg, robot_radius=ROBOT_RADIUS, start_pose_cfg=sp_cfg)
    assert spec_a == spec_b


# ------------------------------------------------------- footprint / clearance
@pytest.mark.parametrize("mode", ["random_rejected", "goal_biased", "free_space_biased"])
@pytest.mark.parametrize("seed", list(range(30)))
def test_generated_start_yaw_always_clears_front_cone_of_every_obstacle(mode, seed):
    cfg = ScenarioConfig(world_size_m=12.0, min_obstacles=2, max_obstacles=5)
    sp_cfg = StartPoseConfig(heading_mode=mode, front_safety_distance_m=0.8, front_cone_half_angle_rad=0.5236)
    spec = generate_scenario(seed, cfg, robot_radius=ROBOT_RADIUS, start_pose_cfg=sp_cfg)
    for obs in spec.static_obstacles:
        dist = math.hypot(obs.x - spec.start_x, obs.y - spec.start_y) - obs.radius - ROBOT_RADIUS
        if dist >= sp_cfg.front_safety_distance_m:
            continue
        bearing = _wrap(math.atan2(obs.y - spec.start_y, obs.x - spec.start_x) - spec.start_yaw)
        assert abs(bearing) > sp_cfg.front_cone_half_angle_rad, (
            f"seed={seed} mode={mode}: start_yaw={spec.start_yaw} faces obstacle at "
            f"({obs.x},{obs.y}) which is only {dist:.3f}m clear, bearing={bearing:.3f}"
        )


def test_generated_start_yaw_never_points_straight_at_the_wall():
    cfg = ScenarioConfig(world_size_m=12.0, min_obstacles=0, max_obstacles=0)
    sp_cfg = StartPoseConfig(heading_mode="random_rejected", min_wall_clearance_m=0.5,
                              front_safety_distance_m=1.0)
    half = cfg.world_size_m / 2.0
    for seed in range(30):
        spec = generate_scenario(seed, cfg, robot_radius=ROBOT_RADIUS, start_pose_cfg=sp_cfg)
        fx = spec.start_x + sp_cfg.front_safety_distance_m * math.cos(spec.start_yaw)
        fy = spec.start_y + sp_cfg.front_safety_distance_m * math.sin(spec.start_yaw)
        limit = half - sp_cfg.min_wall_clearance_m
        assert -limit - 1e-6 <= fx <= limit + 1e-6, f"seed={seed}: heading drives at the wall (fx={fx})"
        assert -limit - 1e-6 <= fy <= limit + 1e-6, f"seed={seed}: heading drives at the wall (fy={fy})"


def test_sample_start_yaw_rejects_a_heading_facing_a_close_obstacle_directly():
    cfg = StartPoseConfig(heading_mode="random_rejected", front_safety_distance_m=1.0,
                           front_cone_half_angle_rad=0.5, max_sampling_attempts=50)
    obstacles = [StaticObstacle(x=1.0, y=0.0, radius=0.2)]
    yaw, _attempts = sample_start_yaw(
        np.random.RandomState(0), cfg, 0.0, 0.0, 5.0, 5.0, obstacles,
        world_half_extent_m=6.0, robot_radius=ROBOT_RADIUS,
    )
    bearing_to_obstacle = abs(_wrap(math.atan2(0.0, 1.0) - yaw))
    assert bearing_to_obstacle > cfg.front_cone_half_angle_rad


def test_sample_start_yaw_rejects_a_heading_facing_the_boundary():
    cfg = StartPoseConfig(heading_mode="random_rejected", min_wall_clearance_m=0.5,
                           front_safety_distance_m=1.0, max_sampling_attempts=50)
    yaw, _attempts = sample_start_yaw(
        np.random.RandomState(1), cfg, 5.5, 0.0, -5.0, -5.0, [],
        world_half_extent_m=6.0, robot_radius=ROBOT_RADIUS,
    )
    fx = 5.5 + cfg.front_safety_distance_m * math.cos(yaw)
    assert fx <= 6.0 - 0.5 + 1e-6


# --------------------------------------------------------------- fail-fast
def test_sample_start_yaw_returns_none_when_totally_surrounded():
    """If every direction is unsafe (obstacles ring the start position within
    the front-safety distance, front cone wide enough to cover all of
    them), sample_start_yaw must return None -- never silently pick an
    unsafe heading."""
    cfg = StartPoseConfig(heading_mode="free_space_biased", front_safety_distance_m=2.0,
                           front_cone_half_angle_rad=3.0, max_sampling_attempts=5)
    obstacles = [
        StaticObstacle(x=1.5 * math.cos(a), y=1.5 * math.sin(a), radius=0.3)
        for a in np.linspace(-math.pi, math.pi, 16, endpoint=False)
    ]
    result = sample_start_yaw(
        np.random.RandomState(3), cfg, 0.0, 0.0, 5.0, 5.0, obstacles,
        world_half_extent_m=6.0, robot_radius=ROBOT_RADIUS,
    )
    assert result is None


def test_generate_scenario_raises_not_silently_degrades_when_no_safe_heading_exists_anywhere():
    """Every possible (start_x, start_y) generate_scenario could draw is
    boxed in on all sides by a world too small to ever satisfy the
    configured front-safety distance -- generate_scenario must raise
    RuntimeError (its own existing max_attempts-exhaustion contract), never
    return a ScenarioSpec with an unsafe heading."""
    cfg = ScenarioConfig(world_size_m=2.2, min_obstacles=0, max_obstacles=0)
    sp_cfg = StartPoseConfig(heading_mode="random_rejected", front_safety_distance_m=5.0,
                              min_wall_clearance_m=0.5, max_sampling_attempts=3)
    with pytest.raises(RuntimeError):
        generate_scenario(0, cfg, robot_radius=0.3, start_pose_cfg=sp_cfg, max_attempts=5)


# --------------------------------------------------------- goal/free-space bias
def test_goal_biased_heading_tends_toward_the_goal_bearing_more_than_uniform_random():
    cfg = ScenarioConfig(world_size_m=16.0, min_obstacles=0, max_obstacles=0)
    sp_cfg = StartPoseConfig(heading_mode="goal_biased", goal_bias_prob=1.0, goal_bias_spread_rad=0.2)
    deviations = []
    for seed in range(40):
        spec = generate_scenario(seed, cfg, robot_radius=ROBOT_RADIUS, start_pose_cfg=sp_cfg)
        goal_heading = math.atan2(spec.goal_y - spec.start_y, spec.goal_x - spec.start_x)
        deviations.append(abs(_wrap(spec.start_yaw - goal_heading)))
    # goal_bias_prob=1.0 with a small spread and no obstacles to force
    # rejection -- every heading should land within the configured spread.
    assert max(deviations) <= sp_cfg.goal_bias_spread_rad + 1e-6


# ------------------------------------------------------ radius quantizer plumbing
def test_static_radius_quantizer_is_applied_before_any_clearance_check():
    cfg = ScenarioConfig(world_size_m=12.0, min_obstacles=2, max_obstacles=4)
    quantized_values = {0.2, 0.35, 0.5}

    def quantizer(r: float) -> float:
        return min(v for v in sorted(quantized_values) if v >= r - 1e-9)

    for seed in range(15):
        spec = generate_scenario(seed, cfg, robot_radius=ROBOT_RADIUS, static_radius_quantizer=quantizer)
        for obs in spec.static_obstacles:
            assert obs.radius in quantized_values


def test_static_obstacle_radius_range_constant_matches_documented_bounds():
    assert STATIC_OBSTACLE_RADIUS_RANGE_M == (0.15, 0.5)


# --------------------------------------- start-position wall clearance (requirement 1)
@pytest.mark.parametrize("mode", ["legacy_random", "random_rejected", "goal_biased", "free_space_biased"])
@pytest.mark.parametrize("seed", list(range(25)))
def test_start_position_clears_wall_by_robot_radius_plus_min_wall_clearance(mode, seed):
    cfg = ScenarioConfig(world_size_m=12.0, min_obstacles=0, max_obstacles=3)
    sp_cfg = StartPoseConfig(heading_mode=mode, min_wall_clearance_m=0.4)
    spec = generate_scenario(seed, cfg, robot_radius=ROBOT_RADIUS, start_pose_cfg=sp_cfg)
    half = cfg.world_size_m / 2.0
    required = ROBOT_RADIUS + sp_cfg.min_wall_clearance_m
    assert abs(spec.start_x) <= half - required + 1e-6, f"seed={seed} mode={mode}: start_x={spec.start_x}"
    assert abs(spec.start_y) <= half - required + 1e-6, f"seed={seed} mode={mode}: start_y={spec.start_y}"


def test_zero_wall_clearance_reproduces_legacy_start_inset_exactly():
    """min_wall_clearance_m=0.0 (the field default, and every pre-existing
    profile) must sample the exact same start position as before this
    feature existed -- start_inset_half collapses to the old robot_radius-only
    inset_half."""
    cfg = ScenarioConfig(world_size_m=10.0, min_obstacles=0, max_obstacles=0)
    sp_cfg = StartPoseConfig(min_wall_clearance_m=0.0)
    for seed in (0, 5, 11, 50):
        rng = np.random.RandomState(seed)
        half = cfg.world_size_m / 2.0 - ROBOT_RADIUS
        expected_start = rng.uniform(-half, half, size=2)
        spec = generate_scenario(seed, cfg, robot_radius=ROBOT_RADIUS, start_pose_cfg=sp_cfg)
        assert spec.start_x == pytest.approx(expected_start[0])
        assert spec.start_y == pytest.approx(expected_start[1])


def test_generate_scenario_fails_fast_when_wall_clearance_leaves_no_start_region():
    """A world small enough that robot_radius + min_wall_clearance_m alone
    (never mind any obstacle) leaves no room to place the start position at
    all must raise immediately, not silently fall back to an unsafe
    (unclearanced) start position."""
    cfg = ScenarioConfig(world_size_m=1.0, min_obstacles=0, max_obstacles=0)
    sp_cfg = StartPoseConfig(min_wall_clearance_m=0.4)
    with pytest.raises(RuntimeError, match="min_wall_clearance_m"):
        generate_scenario(0, cfg, robot_radius=0.3, start_pose_cfg=sp_cfg)


def test_generate_scenario_still_fails_fast_on_robot_radius_alone_with_zero_wall_clearance():
    cfg = ScenarioConfig(world_size_m=0.5, min_obstacles=0, max_obstacles=0)
    sp_cfg = StartPoseConfig(min_wall_clearance_m=0.0)
    with pytest.raises(RuntimeError):
        generate_scenario(0, cfg, robot_radius=0.3, start_pose_cfg=sp_cfg)


@pytest.mark.parametrize("mode", ["random_rejected", "goal_biased", "free_space_biased"])
@pytest.mark.parametrize("seed", list(range(25)))
def test_projected_heading_footprint_clears_wall_by_robot_radius_plus_clearance(mode, seed):
    """The front-safety-distance projection used to reject a heading that
    drives straight at the boundary must also hold the ROBOT'S FOOTPRINT
    (not just its center point) back from the wall -- i.e. by robot_radius
    on top of min_wall_clearance_m."""
    cfg = ScenarioConfig(world_size_m=14.0, min_obstacles=0, max_obstacles=3)
    sp_cfg = StartPoseConfig(heading_mode=mode, min_wall_clearance_m=0.3, front_safety_distance_m=0.9)
    spec = generate_scenario(seed, cfg, robot_radius=ROBOT_RADIUS, start_pose_cfg=sp_cfg)
    half = cfg.world_size_m / 2.0
    fx = spec.start_x + sp_cfg.front_safety_distance_m * math.cos(spec.start_yaw)
    fy = spec.start_y + sp_cfg.front_safety_distance_m * math.sin(spec.start_yaw)
    limit = half - sp_cfg.min_wall_clearance_m - ROBOT_RADIUS
    assert -limit - 1e-6 <= fx <= limit + 1e-6, f"seed={seed} mode={mode}: fx={fx} limit={limit}"
    assert -limit - 1e-6 <= fy <= limit + 1e-6, f"seed={seed} mode={mode}: fy={fy} limit={limit}"
