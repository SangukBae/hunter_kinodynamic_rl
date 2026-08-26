"""Ackermann-aware scenario feasibility gate (code review: the legacy
grid-BFS connectivity check in procedural_generator.is_reachable ignores
heading and minimum turning radius entirely). Covers both the standalone
search (env/scenarios/ackermann_feasibility.py) and its wiring into
generate_scenario's accept/reject/retry loop -- including the requirement
that an infeasible layout can NEVER be returned (and therefore can never
reach environment_node.py's /reset -> replay buffer path)."""

import pytest

from hunter_kinodynamic_rl.config.schema import ScenarioConfig
from hunter_kinodynamic_rl.env.scenarios import procedural_generator
from hunter_kinodynamic_rl.env.scenarios.ackermann_feasibility import is_ackermann_feasible
from hunter_kinodynamic_rl.env.scenarios.procedural_generator import StaticObstacle, generate_scenario

WHEELBASE_M = 0.547696


def test_straight_ahead_goal_is_ackermann_feasible():
    assert is_ackermann_feasible(
        start_x=0.0, start_y=0.0, start_yaw=0.0,
        goal_x=3.0, goal_y=0.0, goal_radius_m=0.42,
        obstacles=[], world_size_m=12.0, robot_radius=0.3,
        min_turning_radius_m=1.5, wheelbase_m=WHEELBASE_M,
    )


def test_goal_directly_behind_start_is_infeasible_for_a_near_zero_curvature_robot():
    """With an (unrealistically) huge minimum turning radius, the robot can
    barely curve away from a straight line at all -- a goal placed behind it
    in a small, bounded world cannot be reached within the search's bounded
    expansion budget. This is the geometric case the old 4-connected grid BFS
    would happily accept (the free space IS topologically connected) but a
    real Ackermann vehicle cannot execute."""
    assert not is_ackermann_feasible(
        start_x=0.0, start_y=0.0, start_yaw=0.0,
        goal_x=-2.0, goal_y=0.0, goal_radius_m=0.3,
        obstacles=[], world_size_m=6.0, robot_radius=0.3,
        min_turning_radius_m=50.0, wheelbase_m=WHEELBASE_M,
    )


def test_goal_already_within_radius_is_trivially_feasible():
    assert is_ackermann_feasible(
        start_x=1.0, start_y=1.0, start_yaw=0.7,
        goal_x=1.1, goal_y=1.05, goal_radius_m=0.42,
        obstacles=[], world_size_m=12.0, robot_radius=0.3,
        min_turning_radius_m=1.5, wheelbase_m=WHEELBASE_M,
    )


def test_start_already_inside_an_obstacle_is_infeasible():
    assert not is_ackermann_feasible(
        start_x=0.0, start_y=0.0, start_yaw=0.0,
        goal_x=3.0, goal_y=0.0, goal_radius_m=0.42,
        obstacles=[StaticObstacle(x=0.0, y=0.0, radius=0.5)],
        world_size_m=12.0, robot_radius=0.3,
        min_turning_radius_m=1.5, wheelbase_m=WHEELBASE_M,
    )


def test_start_outside_world_bounds_is_infeasible_even_within_goal_radius():
    """code review: is_ackermann_feasible had NO explicit start world-bound
    check, and its "start already within goal_radius" shortcut ran BEFORE
    the (pre-existing) start-obstacle check -- so a start OUTSIDE the world
    that happens to be close to the goal would return True, a contract
    violation for a function whose job is "is this start->goal pair
    actually driveable". world_size_m=6.0 -> half_extent=3.0; start_x=5.0
    is well outside it, yet within goal_radius_m of the goal."""
    assert not is_ackermann_feasible(
        start_x=5.0, start_y=0.0, start_yaw=0.0,
        goal_x=5.1, goal_y=0.0, goal_radius_m=0.42,
        obstacles=[], world_size_m=6.0, robot_radius=0.3,
        min_turning_radius_m=1.5, wheelbase_m=WHEELBASE_M,
    )


def test_start_inside_an_obstacle_within_goal_radius_is_still_infeasible():
    """The SAME shortcut-ordering bug, for the pre-existing start-obstacle
    check: a start sitting inside an obstacle but within goal_radius_m of
    the goal must still be rejected, not trivially accepted."""
    assert not is_ackermann_feasible(
        start_x=0.0, start_y=0.0, start_yaw=0.0,
        goal_x=0.1, goal_y=0.0, goal_radius_m=0.42,
        obstacles=[StaticObstacle(x=0.0, y=0.0, radius=0.5)],
        world_size_m=12.0, robot_radius=0.3,
        min_turning_radius_m=1.5, wheelbase_m=WHEELBASE_M,
    )


def test_start_inside_bounds_and_already_at_clear_goal_is_still_feasible():
    """Regression guard for the reordering above: a NORMAL start (inside
    the world, clear of obstacles) already within goal_radius_m of a clear
    goal must remain trivially feasible -- unaffected by the new
    up-front start checks."""
    assert is_ackermann_feasible(
        start_x=1.0, start_y=1.0, start_yaw=0.7,
        goal_x=1.1, goal_y=1.05, goal_radius_m=0.42,
        obstacles=[], world_size_m=12.0, robot_radius=0.3,
        min_turning_radius_m=1.5, wheelbase_m=WHEELBASE_M,
    )


def test_a_dense_obstacle_wall_blocking_the_only_corridor_is_infeasible():
    wall = [StaticObstacle(x=1.5, y=y, radius=0.3) for y in [-1.2, -0.6, 0.0, 0.6, 1.2]]
    assert not is_ackermann_feasible(
        start_x=0.0, start_y=0.0, start_yaw=0.0,
        goal_x=4.0, goal_y=0.0, goal_radius_m=0.3,
        obstacles=wall, world_size_m=8.0, robot_radius=0.3,
        min_turning_radius_m=1.5, wheelbase_m=WHEELBASE_M,
    )


@pytest.mark.parametrize("bad_kwargs", [
    dict(min_turning_radius_m=0.0, wheelbase_m=WHEELBASE_M),
    dict(min_turning_radius_m=-1.0, wheelbase_m=WHEELBASE_M),
    dict(min_turning_radius_m=1.5, wheelbase_m=0.0),
    dict(min_turning_radius_m=1.5, wheelbase_m=WHEELBASE_M, goal_radius_m=0.0),
])
def test_invalid_physical_parameters_raise_not_silently_degrade(bad_kwargs):
    kwargs = dict(
        start_x=0.0, start_y=0.0, start_yaw=0.0, goal_x=3.0, goal_y=0.0,
        goal_radius_m=0.42, obstacles=[], world_size_m=12.0, robot_radius=0.3,
        min_turning_radius_m=1.5, wheelbase_m=WHEELBASE_M,
    )
    kwargs.update(bad_kwargs)
    with pytest.raises(ValueError):
        is_ackermann_feasible(**kwargs)


# --------------------------------------------------------------- generate_scenario wiring


def test_generate_scenario_ackermann_check_requires_turning_radius_and_wheelbase():
    """No silent no-op: selecting feasibility_check='ackermann' without
    supplying the robot geometry it needs must fail LOUDLY and immediately,
    never fall back to the weaker grid_bfs check unnoticed."""
    cfg = ScenarioConfig(world_size_m=10.0, min_obstacles=0, max_obstacles=2, feasibility_check="ackermann")
    with pytest.raises(RuntimeError, match="feasibility_check"):
        generate_scenario(0, cfg, robot_radius=0.3)
    with pytest.raises(RuntimeError, match="feasibility_check"):
        generate_scenario(0, cfg, robot_radius=0.3, min_turning_radius_m=1.5)  # wheelbase_m still missing
    with pytest.raises(RuntimeError, match="feasibility_check"):
        generate_scenario(0, cfg, robot_radius=0.3, wheelbase_m=WHEELBASE_M)  # min_turning_radius_m still missing


def test_generate_scenario_grid_bfs_default_needs_no_robot_geometry():
    """Backward compatibility: the default feasibility_check ('grid_bfs')
    must keep working exactly as before with no new required arguments."""
    cfg = ScenarioConfig(world_size_m=10.0, min_obstacles=0, max_obstacles=2)
    spec = generate_scenario(0, cfg, robot_radius=0.3)
    assert spec.seed == 0


def test_generate_scenario_ackermann_mode_only_ever_returns_ackermann_feasible_layouts():
    """Every scenario generate_scenario('ackermann') actually returns must
    independently pass is_ackermann_feasible -- proving the gate is really
    wired into the accept path, not just validated and ignored."""
    cfg = ScenarioConfig(world_size_m=10.0, min_obstacles=0, max_obstacles=3, feasibility_check="ackermann")
    min_turning_radius_m = 1.5
    for seed in range(10):
        spec = generate_scenario(
            seed, cfg, robot_radius=0.3,
            min_turning_radius_m=min_turning_radius_m, wheelbase_m=WHEELBASE_M,
        )
        assert is_ackermann_feasible(
            spec.start_x, spec.start_y, spec.start_yaw, spec.goal_x, spec.goal_y, 0.42,
            spec.static_obstacles, cfg.world_size_m, 0.3, min_turning_radius_m, WHEELBASE_M,
        )


def test_generate_scenario_never_returns_a_layout_when_ackermann_check_always_fails(monkeypatch):
    """The core replay-buffer-safety guarantee: if NO layout can ever pass
    the Ackermann feasibility gate, generate_scenario must raise -- never
    silently fall back to returning a grid_bfs-only-reachable (but
    Ackermann-infeasible) scenario that would otherwise flow into
    environment_node.py's /reset -> episode -> replay buffer pipeline."""
    monkeypatch.setattr(procedural_generator, "is_ackermann_feasible", lambda *a, **k: False)
    cfg = ScenarioConfig(world_size_m=10.0, min_obstacles=0, max_obstacles=2, feasibility_check="ackermann")
    with pytest.raises(RuntimeError, match="no feasible layout"):
        generate_scenario(
            0, cfg, robot_radius=0.3, max_attempts=5,
            min_turning_radius_m=1.5, wheelbase_m=WHEELBASE_M,
        )


def test_generate_scenario_ackermann_mode_is_strictly_more_conservative_than_grid_bfs():
    """A pathological min_turning_radius (larger than the whole world) makes
    the Ackermann gate reject virtually every layout the grid-BFS-only path
    would accept -- proving the two checks are not equivalent (the new gate
    is doing real, additional work), while grid_bfs mode is unaffected by
    the (unused) turning-radius parameter."""
    cfg_bfs = ScenarioConfig(world_size_m=10.0, min_obstacles=0, max_obstacles=2, feasibility_check="grid_bfs")
    cfg_ackermann = ScenarioConfig(world_size_m=10.0, min_obstacles=0, max_obstacles=2,
                                    feasibility_check="ackermann")
    huge_turning_radius = 500.0

    # grid_bfs never even looks at the turning radius -- always succeeds quickly.
    spec = generate_scenario(1, cfg_bfs, robot_radius=0.3)
    assert spec.seed == 1

    with pytest.raises(RuntimeError, match="no feasible layout"):
        generate_scenario(
            1, cfg_ackermann, robot_radius=0.3, max_attempts=5,
            min_turning_radius_m=huge_turning_radius, wheelbase_m=WHEELBASE_M,
        )


# --------------------------------------------- guarantee-level / resolution honesty
# (code review: the module's own docstring previously overclaimed "never a
# false positive" -- these tests make the actual, documented limitation
# concrete and regression-checkable instead of just prose.)


def test_coarse_path_sample_resolution_can_miss_a_thin_obstacle_between_samples():
    """The documented false-positive ("tunneling") risk, made concrete: a
    small obstacle sits exactly at the MIDPOINT of a straight primitive's
    arc, between the start and a goal placed exactly at that primitive's
    endpoint. With only num_path_samples=1 (the primitive's own endpoint
    is the ONLY point ever checked), this obstacle is invisible to the
    collision check -- a genuine false positive (a real Hunter SE driving
    this straight line WOULD hit it). This is not a hypothetical: it is
    exactly the risk the module docstring's "Known false-positive risk"
    section describes."""
    obstacle = StaticObstacle(x=0.25, y=0.0, radius=0.05)
    kwargs = dict(
        start_x=0.0, start_y=0.0, start_yaw=0.0,
        goal_x=0.5, goal_y=0.0, goal_radius_m=0.05,
        obstacles=[obstacle], world_size_m=10.0, robot_radius=0.05,
        min_turning_radius_m=1.0, wheelbase_m=WHEELBASE_M,
        primitive_speed_mps=1.0, primitive_duration_sec=0.5,
    )
    # Coarse: only the primitive's endpoint (x=0.5) is ever sampled --
    # distance from there to the obstacle (0.25) is 0.25m, well outside
    # obstacle.radius + robot_radius = 0.1m, so the collision is MISSED.
    coarse_result = is_ackermann_feasible(num_path_samples=1, **kwargs)
    assert coarse_result is True  # the (documented) false positive

    # Fine: samples at x=0.1,0.2,0.3,0.4,0.5 -- x=0.2 and x=0.3 are each
    # only 0.05m from the obstacle's center, well inside the 0.1m
    # clearance radius, so the SAME obstacle is now correctly caught.
    fine_result = is_ackermann_feasible(num_path_samples=5, **kwargs)
    assert fine_result is False


def test_default_path_sample_resolution_catches_the_same_thin_obstacle():
    """The module's actual DEFAULT (num_path_samples=5, chosen specifically
    to tighten this trade-off -- see the module docstring) must catch the
    same obstacle the coarse case above misses, without the caller having
    to pass an explicit override."""
    obstacle = StaticObstacle(x=0.25, y=0.0, radius=0.05)
    result = is_ackermann_feasible(
        start_x=0.0, start_y=0.0, start_yaw=0.0,
        goal_x=0.5, goal_y=0.0, goal_radius_m=0.05,
        obstacles=[obstacle], world_size_m=10.0, robot_radius=0.05,
        min_turning_radius_m=1.0, wheelbase_m=WHEELBASE_M,
        primitive_speed_mps=1.0, primitive_duration_sec=0.5,
        # num_path_samples deliberately omitted -- exercises the real default.
    )
    assert result is False


def test_goal_inside_an_obstacle_is_rejected_immediately_not_via_search_exhaustion():
    """Explicit up-front goal-clearance check (code review ask): a goal
    placed inside an obstacle's footprint must be rejected regardless of
    how the search would otherwise explore -- checked here by placing the
    obstacle exactly AT the goal center with a radius that swallows it."""
    obstacle = StaticObstacle(x=3.0, y=0.0, radius=0.5)
    assert not is_ackermann_feasible(
        start_x=0.0, start_y=0.0, start_yaw=0.0,
        goal_x=3.0, goal_y=0.0, goal_radius_m=0.2,
        obstacles=[obstacle], world_size_m=12.0, robot_radius=0.3,
        min_turning_radius_m=1.5, wheelbase_m=WHEELBASE_M,
    )


def test_goal_outside_world_bounds_is_rejected():
    assert not is_ackermann_feasible(
        start_x=0.0, start_y=0.0, start_yaw=0.0,
        goal_x=100.0, goal_y=0.0, goal_radius_m=0.42,
        obstacles=[], world_size_m=12.0, robot_radius=0.3,
        min_turning_radius_m=1.5, wheelbase_m=WHEELBASE_M,
    )


def test_goal_clearance_check_does_not_override_a_genuine_start_at_goal_shortcut():
    """The goal precheck runs BEFORE the trivial start-already-at-goal
    shortcut, but must not break it for the ordinary case (goal clear,
    start already within goal_radius)."""
    assert is_ackermann_feasible(
        start_x=1.0, start_y=1.0, start_yaw=0.3,
        goal_x=1.05, goal_y=1.0, goal_radius_m=0.42,
        obstacles=[], world_size_m=12.0, robot_radius=0.3,
        min_turning_radius_m=1.5, wheelbase_m=WHEELBASE_M,
    )
