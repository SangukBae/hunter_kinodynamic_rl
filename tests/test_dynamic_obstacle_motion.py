import math

import pytest

from hunter_kinodynamic_rl.env.humans.dynamic_obstacle_motion import (
    MotionPattern, RandomWaypointState, apply_pattern, assign_pattern, position_at_elapsed_time,
)
from hunter_kinodynamic_rl.env.scenarios.procedural_generator import DynamicObstacleSpec


def test_assign_pattern_is_deterministic():
    assert assign_pattern(seed=5, index=0) == assign_pattern(seed=5, index=0)


def test_assign_pattern_varies_by_index():
    patterns = {assign_pattern(seed=5, index=i) for i in range(20)}
    assert len(patterns) > 1


def test_crossing_pattern_is_perpendicular_to_heading():
    spec = DynamicObstacleSpec(x0=0.0, y0=0.0, vx=0.0, vy=0.0, radius=0.3)
    out = apply_pattern(spec, MotionPattern.CROSSING, start_xy=(0, 0), goal_xy=(1, 0),
                         speed_mps=1.0, seed=0, index=0)
    assert abs(out.vx) < 1e-6
    assert abs(out.vy) == pytest.approx(1.0, abs=1e-6)


def test_head_on_pattern_opposes_heading():
    spec = DynamicObstacleSpec(x0=0.0, y0=0.0, vx=0.0, vy=0.0, radius=0.3)
    out = apply_pattern(spec, MotionPattern.HEAD_ON, start_xy=(0, 0), goal_xy=(1, 0),
                         speed_mps=1.0, seed=0, index=0)
    assert out.vx == pytest.approx(-1.0, abs=1e-6)


def test_parallel_pattern_matches_heading():
    spec = DynamicObstacleSpec(x0=0.0, y0=0.0, vx=0.0, vy=0.0, radius=0.3)
    out = apply_pattern(spec, MotionPattern.PARALLEL, start_xy=(0, 0), goal_xy=(1, 0),
                         speed_mps=1.0, seed=0, index=0)
    assert out.vx == pytest.approx(1.0, abs=1e-6)


def test_constant_velocity_pattern_is_a_no_op():
    spec = DynamicObstacleSpec(x0=1.0, y0=2.0, vx=0.5, vy=-0.3, radius=0.3)
    out = apply_pattern(spec, MotionPattern.CONSTANT_VELOCITY, start_xy=(0, 0), goal_xy=(1, 0),
                         speed_mps=1.0, seed=0, index=0)
    assert out == spec


def test_random_waypoint_moves_toward_target_and_replans():
    state = RandomWaypointState(x=0.0, y=0.0, speed_mps=1.0, half_extent_m=5.0,
                                 replan_period_sec=0.5, seed=1)
    positions = [state.tick(0.1) for _ in range(20)]
    assert any(math.hypot(x, y) > 0.05 for x, y in positions)


def test_random_waypoint_is_deterministic_given_seed():
    s1 = RandomWaypointState(x=0.0, y=0.0, speed_mps=1.0, half_extent_m=5.0, seed=42)
    s2 = RandomWaypointState(x=0.0, y=0.0, speed_mps=1.0, half_extent_m=5.0, seed=42)
    for _ in range(30):
        assert s1.tick(0.1) == s2.tick(0.1)


# ------------------------------------------------- P0-8: elapsed-time positioning
def test_position_at_elapsed_time_matches_linear_extrapolation():
    spec0 = DynamicObstacleSpec(x0=1.0, y0=2.0, vx=0.5, vy=-0.3, radius=0.3)
    x, y = position_at_elapsed_time(spec0, elapsed_sec=4.0)
    assert x == pytest.approx(1.0 + 0.5 * 4.0)
    assert y == pytest.approx(2.0 - 0.3 * 4.0)


def test_position_at_elapsed_time_at_zero_elapsed_returns_spawn_point():
    spec0 = DynamicObstacleSpec(x0=3.0, y0=-1.0, vx=1.0, vy=1.0, radius=0.3)
    x, y = position_at_elapsed_time(spec0, elapsed_sec=0.0)
    assert (x, y) == (3.0, -1.0)


def test_position_at_elapsed_time_avoids_iterative_floating_point_drift():
    """The core P0-8 regression: the OLD per-tick approach
    (``x += vx * dt_sec``, repeated once per step) accumulates
    floating-point summation error over many ticks -- a well-known effect
    (``sum(0.1 for _ in range(10)) != 1.0`` in plain Python float64).
    Computing position directly from the spawn spec + TOTAL elapsed time
    (this function) is immune to it by construction: it is a single
    multiply-and-add, not N repeated additions."""
    spec0 = DynamicObstacleSpec(x0=0.0, y0=0.0, vx=0.7, vy=-1.3, radius=0.3)
    dt = 0.1
    n_steps = 100_000

    x_iterative = spec0.x0
    for _ in range(n_steps):
        x_iterative += spec0.vx * dt

    total_elapsed = n_steps * dt
    x_direct, _ = position_at_elapsed_time(spec0, total_elapsed)
    x_analytic = spec0.x0 + spec0.vx * total_elapsed

    assert x_direct == x_analytic  # position_at_elapsed_time IS this exact formula
    assert x_iterative != x_analytic  # genuine floating-point drift after 100k additions
