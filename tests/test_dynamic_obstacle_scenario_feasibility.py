"""section P1-3: procedural dynamic-obstacle placement must never spawn a
dynamic obstacle overlapping the robot's own start pose, the goal, or a
static obstacle -- previously drawn completely independently of all three,
so an initial (t=0) overlap was possible and undetected. See
env/scenarios/procedural_generator.py's ``_place_dynamic_obstacles``.
"""

import math

import pytest

from hunter_kinodynamic_rl.config.schema import ScenarioConfig
from hunter_kinodynamic_rl.env.scenarios.procedural_generator import (
    ScenarioSpec, generate_scenario,
)

ROBOT_RADIUS = 0.45
DYNAMIC_RADIUS = 0.3


def _min_pairwise_clearance(spec: ScenarioSpec) -> float:
    """Smallest (center-distance - both radii) gap among (start, goal,
    static, dynamic) -- must stay >= 0 for every dynamic-obstacle pair
    checked here (start/goal are treated as circles of ROBOT_RADIUS)."""
    worst = math.inf
    for dyn in spec.dynamic_obstacles:
        d_start = math.hypot(dyn.x0 - spec.start_x, dyn.y0 - spec.start_y) - ROBOT_RADIUS - dyn.radius
        d_goal = math.hypot(dyn.x0 - spec.goal_x, dyn.y0 - spec.goal_y) - ROBOT_RADIUS - dyn.radius
        worst = min(worst, d_start, d_goal)
        for s in spec.static_obstacles:
            worst = min(worst, math.hypot(dyn.x0 - s.x, dyn.y0 - s.y) - s.radius - dyn.radius)
        for other in spec.dynamic_obstacles:
            if other is dyn:
                continue
            worst = min(worst, math.hypot(dyn.x0 - other.x0, dyn.y0 - other.y0) - other.radius - dyn.radius)
    return worst


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 5, 13, 42, 100, 207, 999, 5000])
def test_dynamic_obstacles_never_overlap_start_goal_or_static_at_spawn(seed):
    """A broad seed sweep (including 207, called out explicitly in the
    governing review) -- every generated scenario's dynamic obstacles must
    be collision-free against start/goal/static AND each other at t=0."""
    cfg = ScenarioConfig(world_size_m=12.0, min_obstacles=2, max_obstacles=4, dynamic_obstacle_count=3)
    spec = generate_scenario(seed, cfg, robot_radius=ROBOT_RADIUS)
    assert _min_pairwise_clearance(spec) >= -1e-9


def test_dynamic_obstacle_count_is_always_fully_satisfied():
    """The placement retry must never silently return FEWER dynamic
    obstacles than requested -- either all of them are placed clear, or
    the whole scenario-generation attempt is redrawn."""
    cfg = ScenarioConfig(world_size_m=12.0, min_obstacles=0, max_obstacles=2, dynamic_obstacle_count=3)
    for seed in range(20):
        spec = generate_scenario(seed, cfg, robot_radius=ROBOT_RADIUS)
        assert len(spec.dynamic_obstacles) == 3


def test_same_seed_reproduces_identical_dynamic_obstacle_layout():
    """Determinism: identical seed -> byte-identical dynamic obstacle
    positions/velocities, even with the new retry loop in the mix."""
    cfg = ScenarioConfig(world_size_m=12.0, min_obstacles=1, max_obstacles=3, dynamic_obstacle_count=2)
    spec_a = generate_scenario(42, cfg, robot_radius=ROBOT_RADIUS)
    spec_b = generate_scenario(42, cfg, robot_radius=ROBOT_RADIUS)
    assert spec_a.dynamic_obstacles == spec_b.dynamic_obstacles
    assert spec_a.static_obstacles == spec_b.static_obstacles


def test_dynamic_obstacle_clearance_respects_configured_margin():
    """A larger dynamic_obstacle_min_clearance_m must be reflected in the
    actual generated layout's minimum gap, not just accepted and ignored."""
    cfg = ScenarioConfig(world_size_m=16.0, min_obstacles=0, max_obstacles=1, dynamic_obstacle_count=1,
                          dynamic_obstacle_min_clearance_m=1.5)
    for seed in range(10):
        spec = generate_scenario(seed, cfg, robot_radius=ROBOT_RADIUS)
        assert _min_pairwise_clearance(spec) >= 1.5 - 1e-9


# --------------------------------------------------------------- item-7
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 13, 42, 100, 207, 999, 5000])
def test_dynamic_obstacle_full_footprint_never_spawns_outside_the_world_boundary(seed):
    """The core item-7 regression (explicitly including seed=4): a dynamic
    obstacle's spawn CENTER must be inset from the world boundary by its
    own radius -- not just checked as a point -- so its FULL circular
    footprint stays inside [-half, half]^2, never poking past the edge.
    Previously the center was drawn over the whole [-half, half] range with
    no margin for the obstacle's own radius."""
    cfg = ScenarioConfig(world_size_m=12.0, min_obstacles=1, max_obstacles=3, dynamic_obstacle_count=3)
    half = cfg.world_size_m / 2.0
    spec = generate_scenario(seed, cfg, robot_radius=ROBOT_RADIUS)
    for dyn in spec.dynamic_obstacles:
        assert abs(dyn.x0) + dyn.radius <= half + 1e-9, (
            f"seed={seed}: dynamic obstacle x0={dyn.x0}, radius={dyn.radius} pokes past "
            f"world half_extent={half}"
        )
        assert abs(dyn.y0) + dyn.radius <= half + 1e-9, (
            f"seed={seed}: dynamic obstacle y0={dyn.y0}, radius={dyn.radius} pokes past "
            f"world half_extent={half}"
        )


@pytest.mark.parametrize("seed", [0, 4, 13, 207, 999])
def test_static_obstacle_full_footprint_never_spawns_outside_the_world_boundary(seed):
    """Same bug, same fix, in the static-obstacle placement path (which
    draws a variable per-obstacle radius up to 0.5 m)."""
    cfg = ScenarioConfig(world_size_m=12.0, min_obstacles=2, max_obstacles=4, dynamic_obstacle_count=0)
    half = cfg.world_size_m / 2.0
    spec = generate_scenario(seed, cfg, robot_radius=ROBOT_RADIUS)
    for obs in spec.static_obstacles:
        assert abs(obs.x) + obs.radius <= half + 1e-9
        assert abs(obs.y) + obs.radius <= half + 1e-9


def test_dynamic_obstacle_inset_raises_when_world_is_too_small_for_its_own_footprint():
    """Mirrors generate_scenario's own robot-radius inset-room check --
    a world too small to inset a dynamic obstacle by even its own radius is
    a config error, raised immediately (never silently degraded back to the
    un-inset -- and therefore boundary-violating -- placement range)."""
    cfg = ScenarioConfig(world_size_m=0.4, min_obstacles=0, max_obstacles=0, dynamic_obstacle_count=1)
    with pytest.raises(RuntimeError, match="dynamic obstacle radius"):
        generate_scenario(0, cfg, robot_radius=0.1, max_attempts=1)


# --------------------------------------------------- item-7: initial feasibility policy
def test_dynamic_obstacle_initial_feasibility_check_is_wired_into_the_retry_loop(monkeypatch):
    """DEFAULT policy (dynamic_obstacle_initial_feasibility_check=True):
    generate_scenario must re-run is_reachable with dynamic obstacles'
    t=0 positions folded in as temporary occupied regions -- not just the
    pre-existing static-only pass."""
    from hunter_kinodynamic_rl.env.scenarios import procedural_generator as pg

    real_is_reachable = pg.is_reachable
    call_log = []

    def _spy_is_reachable(start_xy, goal_xy, obstacles, world_size_m, robot_radius):
        result = real_is_reachable(start_xy, goal_xy, obstacles, world_size_m, robot_radius)
        call_log.append((len(obstacles), result))
        return result

    monkeypatch.setattr(pg, "is_reachable", _spy_is_reachable)
    cfg = ScenarioConfig(world_size_m=12.0, min_obstacles=0, max_obstacles=0, dynamic_obstacle_count=2)
    spec = generate_scenario(0, cfg, robot_radius=ROBOT_RADIUS)
    # is_reachable must have been called AGAIN with MORE obstacles than the
    # static-only count once dynamic obstacles were placed -- proof the
    # combined (static + dynamic-as-temporary-static) list was actually
    # checked, not just the static-only pass.
    static_only_calls = [n for n, _ in call_log if n == len(spec.static_obstacles)]
    combined_calls = [n for n, _ in call_log if n == len(spec.static_obstacles) + len(spec.dynamic_obstacles)]
    assert static_only_calls, "expected the pre-existing static-only reachability call"
    assert combined_calls, (
        "expected an ADDITIONAL reachability call including dynamic obstacles' t=0 positions "
        "(dynamic_obstacle_initial_feasibility_check=True is the default)"
    )


def test_dynamic_obstacle_initial_feasibility_check_actually_rejects_a_blocked_layout(monkeypatch):
    """The stronger regression: not just that the combined check is
    CALLED, but that a FAILING combined result actually causes
    generate_scenario to redraw (and, if every attempt fails it, raise) --
    forces is_reachable to report "blocked" specifically for the combined
    (static+dynamic) call, while leaving the static-only pass genuinely
    real, so only the item-7 check itself can be responsible for the
    resulting exhaustion."""
    from hunter_kinodynamic_rl.env.scenarios import procedural_generator as pg

    real_is_reachable = pg.is_reachable

    # Gate: the FIRST is_reachable call within each generate_scenario
    # attempt is the pre-existing static-only pass (real result, so the
    # attempt can genuinely get as far as placing dynamic obstacles); any
    # SUBSEQUENT call with a LARGER obstacle count (i.e. the item-7
    # combined static+dynamic pass) always reports "blocked".
    state = {"seen_static_only": None}

    def _spy(start_xy, goal_xy, obstacles, world_size_m, robot_radius):
        if state["seen_static_only"] is None or len(obstacles) <= state["seen_static_only"]:
            state["seen_static_only"] = len(obstacles)
            return real_is_reachable(start_xy, goal_xy, obstacles, world_size_m, robot_radius)
        return False  # the combined (dynamic-obstacles-included) call always "blocked"

    monkeypatch.setattr(pg, "is_reachable", _spy)
    cfg = ScenarioConfig(world_size_m=12.0, min_obstacles=0, max_obstacles=0, dynamic_obstacle_count=1)
    with pytest.raises(RuntimeError):
        generate_scenario(0, cfg, robot_radius=ROBOT_RADIUS, max_attempts=5)


def test_dynamic_obstacle_initial_feasibility_check_can_be_disabled():
    """Setting dynamic_obstacle_initial_feasibility_check=False opts back
    into the pre-item-7 behavior -- generate_scenario must NOT re-run
    is_reachable/is_ackermann_feasible with dynamic obstacles folded in."""
    from hunter_kinodynamic_rl.env.scenarios import procedural_generator as pg

    real_is_reachable = pg.is_reachable
    call_log = []

    def _spy_is_reachable(start_xy, goal_xy, obstacles, world_size_m, robot_radius):
        call_log.append(len(obstacles))
        return real_is_reachable(start_xy, goal_xy, obstacles, world_size_m, robot_radius)

    import pytest as _pytest
    mp = _pytest.MonkeyPatch()
    mp.setattr(pg, "is_reachable", _spy_is_reachable)
    try:
        cfg = ScenarioConfig(world_size_m=12.0, min_obstacles=0, max_obstacles=0, dynamic_obstacle_count=2,
                              dynamic_obstacle_initial_feasibility_check=False)
        spec = generate_scenario(0, cfg, robot_radius=ROBOT_RADIUS)
        combined_len = len(spec.static_obstacles) + len(spec.dynamic_obstacles)
        assert combined_len not in call_log, (
            "is_reachable must not be called with dynamic obstacles included when the "
            "initial-feasibility check is explicitly disabled"
        )
    finally:
        mp.undo()


def test_infeasible_dynamic_obstacle_density_raises_not_silently_degrades():
    """An impossible request (more dynamic obstacles + clearance margin
    than a tiny world can ever fit) must fail loudly (RuntimeError from the
    outer generate_scenario retry-exhaustion path), never silently return
    fewer/overlapping obstacles."""
    cfg = ScenarioConfig(world_size_m=2.0, min_obstacles=0, max_obstacles=0, dynamic_obstacle_count=5,
                          dynamic_obstacle_min_clearance_m=1.0,
                          dynamic_obstacle_placement_attempts=5)
    with pytest.raises(RuntimeError):
        generate_scenario(0, cfg, robot_radius=ROBOT_RADIUS, max_attempts=5)
