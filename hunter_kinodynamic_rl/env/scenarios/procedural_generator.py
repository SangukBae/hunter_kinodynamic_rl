"""Procedural training-environment generator (section 25-28): random
start/goal/static-obstacle layout per episode, seeded and train/val/test
seed-range-separated (section 31).

section item-5 (round 2, documentation accuracy): the DEFAULT feasibility
check (:func:`is_reachable`, ``scenario.feasibility_check: "grid_bfs"``) is
an APPROXIMATE 4-connected occupancy-grid connectivity check -- it ignores
the robot's heading and minimum turning radius entirely, so a layout it
accepts can still be geometrically UNDRIVABLE for a real Ackermann-
constrained robot (a corridor wide enough to fit through in a straight
line, but too narrow to actually turn into, is grid-BFS-reachable and
Ackermann-infeasible at the same time). It is NOT a solvability guarantee
-- callers that need one must explicitly opt into the stronger (and
slower) check via ``scenario.feasibility_check: "ackermann"``, which
additionally requires :func:`~hunter_kinodynamic_rl.env.scenarios.ackermann_feasibility.is_ackermann_feasible`
to accept the layout (a bounded Hybrid-A*-style search over the robot's
own bicycle-model kinematics -- see ``generate_scenario``'s own docstring).

Pure Python/numpy -- no Gazebo. ``env/spawning/obstacle_spawner.py`` is the
thin ROS layer that actually places the ``StaticObstacle``/dynamic-obstacle
specs this module produces into the running simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from hunter_kinodynamic_rl.config.schema import ScenarioConfig
from hunter_kinodynamic_rl.env.scenarios.ackermann_feasibility import is_ackermann_feasible


@dataclass(frozen=True)
class StaticObstacle:
    x: float
    y: float
    radius: float


@dataclass(frozen=True)
class DynamicObstacleSpec:
    x0: float
    y0: float
    vx: float
    vy: float
    radius: float
    # Optional NAMED motion pattern (section P1-3): set only by fixed
    # benchmark scenarios that want a geometry-resolved pattern (e.g.
    # "crossing" relative to THIS scenario's start/goal) instead of raw
    # (vx, vy). None (the default, and always the case for procedurally
    # generated obstacles) means "use vx/vy as-is" for a fixed scenario, or
    # "let the seed-based assign_pattern() choose" for a procedural one --
    # see environment_node.py::_spawn_scenario_obstacles.
    motion_pattern: Optional[str] = None


@dataclass(frozen=True)
class ScenarioSpec:
    seed: int
    start_x: float
    start_y: float
    start_yaw: float
    goal_x: float
    goal_y: float
    static_obstacles: List[StaticObstacle] = field(default_factory=list)
    dynamic_obstacles: List[DynamicObstacleSpec] = field(default_factory=list)


class SeedSplitError(ValueError):
    pass


def seed_split(seed: int, cfg: ScenarioConfig) -> str:
    """Which of train/validation/test a seed belongs to. Raises if the seed
    is in none of the three ranges -- section 31's non-negotiable: "test
    scenario/seed는 training replay buffer에 절대로 들어가면 안 된다" starts
    with never being ambiguous about which pool a seed is in."""
    if cfg.train_seed_range[0] <= seed <= cfg.train_seed_range[1]:
        return "train"
    if cfg.validation_seed_range[0] <= seed <= cfg.validation_seed_range[1]:
        return "validation"
    if cfg.test_seed_range[0] <= seed <= cfg.test_seed_range[1]:
        return "test"
    raise SeedSplitError(f"seed {seed} is outside all configured train/validation/test ranges")


def _occupancy_grid(obstacles: List[StaticObstacle], world_size_m: float, robot_radius: float,
                     resolution_m: float = 0.25) -> np.ndarray:
    n = max(2, int(round(world_size_m / resolution_m)))
    half = world_size_m / 2.0
    grid = np.zeros((n, n), dtype=bool)  # True == occupied
    xs = np.linspace(-half, half, n)
    ys = np.linspace(-half, half, n)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    for obs in obstacles:
        occupied = (gx - obs.x) ** 2 + (gy - obs.y) ** 2 <= (obs.radius + robot_radius) ** 2
        grid |= occupied
    return grid


def _place_dynamic_obstacles(
    rng: np.random.RandomState, cfg: ScenarioConfig, start_xy, goal_xy,
    static_obstacles: List[StaticObstacle], robot_radius: float,
) -> Optional[List[DynamicObstacleSpec]]:
    """section P1-3: places every dynamic obstacle's t=0 position with a
    real clearance check against the robot's start, the goal, and every
    static obstacle (previously drawn completely independently -- initial
    overlap, e.g. a dynamic obstacle spawning directly on the robot's own
    start pose, was possible and undetected). Also keeps newly-placed
    dynamic obstacles clear of ALREADY-placed ones this same call, so two
    dynamic obstacles don't spawn on top of each other either.

    Returns ``None`` (never a partial/overlapping list) if any ONE
    obstacle's clearance requirement can't be satisfied within
    ``cfg.dynamic_obstacle_placement_attempts`` tries -- the caller (
    ``generate_scenario``) treats that as "this whole scenario-generation
    attempt failed" and retries with a fresh draw from the SAME ``rng``
    stream (preserving full seed-determinism), exactly like the existing
    static-obstacle-count/reachability retry path.

    section item-7: the spawn CENTER is drawn INSET from the world
    boundary by the obstacle's own ``radius`` (mirroring how start/goal are
    already inset by ``robot_radius`` above) -- previously drawn over the
    FULL ``[-half, half]`` range with no margin for the obstacle's own
    footprint, so a fraction of its circular footprint could end up spawned
    outside the world boundary entirely (e.g. a center drawn at
    ``x0=half-0.05`` with ``radius=0.3`` sticks ``0.25`` m past the edge).
    A world too small to inset by even one obstacle radius is a config
    error, raised immediately (mirroring ``generate_scenario``'s own
    ``inset_half <= 0`` check for the robot), never silently degraded back
    to the un-inset range."""
    half = cfg.world_size_m / 2.0
    margin = cfg.dynamic_obstacle_min_clearance_m
    radius = 0.3
    inset_half = half - radius
    if inset_half <= 0.0:
        raise RuntimeError(
            f"generate_scenario: dynamic obstacle radius={radius} leaves no inset room in a "
            f"world_size_m={cfg.world_size_m} world (half_extent={half}) -- world_size_m is too small "
            "for a dynamic obstacle's own footprint"
        )
    placed: List[DynamicObstacleSpec] = []
    for _ in range(cfg.dynamic_obstacle_count):
        for _attempt in range(cfg.dynamic_obstacle_placement_attempts):
            x0, y0 = rng.uniform(-inset_half, inset_half, size=2)
            if np.hypot(x0 - start_xy[0], y0 - start_xy[1]) < robot_radius + radius + margin:
                continue
            if np.hypot(x0 - goal_xy[0], y0 - goal_xy[1]) < robot_radius + radius + margin:
                continue
            if any(np.hypot(x0 - s.x, y0 - s.y) < s.radius + radius + margin for s in static_obstacles):
                continue
            if any(np.hypot(x0 - p.x0, y0 - p.y0) < p.radius + radius + margin for p in placed):
                continue
            placed.append(DynamicObstacleSpec(
                x0=float(x0), y0=float(y0),
                vx=float(rng.uniform(-0.5, 0.5)), vy=float(rng.uniform(-0.5, 0.5)),
                radius=radius,
            ))
            break
        else:
            return None  # exhausted attempts for this one obstacle -- fail this scenario attempt
    return placed


def _to_cell(x: float, y: float, world_size_m: float, n: int) -> Tuple[int, int]:
    half = world_size_m / 2.0
    i = int(round((x + half) / world_size_m * (n - 1)))
    j = int(round((y + half) / world_size_m * (n - 1)))
    return max(0, min(n - 1, i)), max(0, min(n - 1, j))


def is_reachable(start_xy, goal_xy, obstacles: List[StaticObstacle], world_size_m: float, robot_radius: float) -> bool:
    """4-connected BFS flood-fill on a coarse occupancy grid -- an
    approximate feasibility gate (section 28), not a path planner."""
    grid = _occupancy_grid(obstacles, world_size_m, robot_radius)
    n = grid.shape[0]
    start_cell = _to_cell(start_xy[0], start_xy[1], world_size_m, n)
    goal_cell = _to_cell(goal_xy[0], goal_xy[1], world_size_m, n)
    if grid[start_cell] or grid[goal_cell]:
        return False

    visited = np.zeros_like(grid)
    stack = [start_cell]
    visited[start_cell] = True
    while stack:
        i, j = stack.pop()
        if (i, j) == goal_cell:
            return True
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < n and 0 <= nj < n and not visited[ni, nj] and not grid[ni, nj]:
                visited[ni, nj] = True
                stack.append((ni, nj))
    return False


def generate_scenario(seed: int, cfg: ScenarioConfig, robot_radius: float = 0.3,
                       min_start_goal_distance_m: float = 2.0,
                       max_attempts: int = 50,
                       min_turning_radius_m: Optional[float] = None,
                       wheelbase_m: Optional[float] = None,
                       goal_radius_m: float = 0.42) -> ScenarioSpec:
    """Retries with a fresh sub-draw until a feasible layout is found, or
    raises after ``max_attempts`` (a max_obstacles set absurdly high for
    world_size_m is a config error, not something to silently degrade).

    Start/goal are sampled INSET from the world boundary by ``robot_radius``
    (section P0-2: "Robot footprint를 고려하고 start/goal을 안전하게
    inset하라") -- without this, a spawned start/goal pose could sit with
    the robot's own footprint already overlapping the wall. A
    ``robot_radius`` that leaves no usable inset area (>= half the world
    extent) is a config error, raised immediately rather than silently
    degrading to the un-inset range.

    An attempt is only accepted once it ALSO placed at least
    ``cfg.min_obstacles`` obstacles: the per-obstacle clearance rejection
    below (too close to start/goal) can silently place FEWER than
    ``min_obstacles`` were it not checked here, breaking callers (e.g.
    smoke-test profiles) that rely on ``min_obstacles`` as a hard guarantee
    that risk labels have something to be tested against (section P0-2:
    "min_obstacles가 배치 실패로 깨지지 않도록 재시도 또는 명시적 실패
    처리를 구현하라").

    ``cfg.feasibility_check == "ackermann"`` additionally requires
    :func:`env.scenarios.ackermann_feasibility.is_ackermann_feasible` to
    accept the layout -- the plain grid BFS above (``is_reachable``) ignores
    heading and the robot's minimum turning radius entirely (code review),
    so a layout it accepts can still be geometrically undrivable. Selecting
    "ackermann" without also supplying a valid ``min_turning_radius_m``/
    ``wheelbase_m`` is a caller bug -- this raises immediately rather than
    silently falling back to the weaker grid-only check (no silent no-op
    config flag)."""
    if cfg.feasibility_check == "ackermann" and (
        min_turning_radius_m is None or min_turning_radius_m <= 0.0
        or wheelbase_m is None or wheelbase_m <= 0.0
    ):
        raise RuntimeError(
            "generate_scenario: scenario.feasibility_check='ackermann' requires the caller to pass "
            f"a valid min_turning_radius_m/wheelbase_m (got min_turning_radius_m={min_turning_radius_m}, "
            f"wheelbase_m={wheelbase_m}) -- derive both from the active RobotConfig "
            "(min_turning_radius_m = 1 / robot.max_curvature); refusing to silently degrade to grid_bfs"
        )

    rng = np.random.RandomState(seed)
    half = cfg.world_size_m / 2.0
    inset_half = half - robot_radius
    if inset_half <= 0.0:
        raise RuntimeError(
            f"generate_scenario: robot_radius={robot_radius} leaves no inset room in a "
            f"world_size_m={cfg.world_size_m} world (half_extent={half}) -- world_size_m is too small "
            "for this robot's footprint"
        )

    for _attempt in range(max_attempts):
        start_x, start_y = rng.uniform(-inset_half, inset_half, size=2)
        start_yaw = rng.uniform(-np.pi, np.pi)
        for _ in range(20):
            goal_x, goal_y = rng.uniform(-inset_half, inset_half, size=2)
            if np.hypot(goal_x - start_x, goal_y - start_y) >= min_start_goal_distance_m:
                break

        num_obstacles = rng.randint(cfg.min_obstacles, cfg.max_obstacles + 1)
        obstacles: List[StaticObstacle] = []
        for _ in range(num_obstacles):
            radius = float(rng.uniform(0.15, 0.5))
            # section item-7: same radius-inclusive-footprint fix as
            # dynamic obstacles below -- the CENTER must be inset by the
            # obstacle's own radius, not drawn over the full
            # [-half, half] range (which lets part of its footprint sit
            # outside the world boundary). max(..., 0.0) is a defensive
            # floor only -- world_size_m is always >> a 0.5 m max radius in
            # every shipped profile, but never crash rng.uniform on a
            # degenerate low>high range if it weren't.
            obstacle_inset_half = max(half - radius, 0.0)
            ox, oy = rng.uniform(-obstacle_inset_half, obstacle_inset_half, size=2)
            if np.hypot(ox - start_x, oy - start_y) < robot_radius + radius + 0.5:
                continue
            if np.hypot(ox - goal_x, oy - goal_y) < robot_radius + radius + 0.5:
                continue
            obstacles.append(StaticObstacle(x=float(ox), y=float(oy), radius=radius))

        if len(obstacles) < cfg.min_obstacles:
            continue
        if not is_reachable((start_x, start_y), (goal_x, goal_y), obstacles, cfg.world_size_m, robot_radius):
            continue
        if cfg.feasibility_check == "ackermann" and not is_ackermann_feasible(
            start_x, start_y, start_yaw, goal_x, goal_y, goal_radius_m,
            obstacles, cfg.world_size_m, robot_radius, min_turning_radius_m, wheelbase_m,
        ):
            continue
        # section P1-3: bounded-retry, clearance-checked placement -- see
        # _place_dynamic_obstacles's docstring. None means this attempt
        # could not place every dynamic obstacle without overlapping the
        # robot start, the goal, a static obstacle, or an already-placed
        # dynamic obstacle -- redraw the WHOLE scenario attempt (continue),
        # never silently keep a partial/overlapping list.
        dynamic_obstacles = _place_dynamic_obstacles(
            rng, cfg, (start_x, start_y), (goal_x, goal_y), obstacles, robot_radius,
        )
        if dynamic_obstacles is None:
            continue
        # section item-7: DEFAULT policy is to guarantee a feasible INITIAL
        # scenario -- re-run the same feasibility check(s), this time
        # treating every dynamic obstacle's t=0 position/radius as a
        # temporary occupied region too (never stored as a StaticObstacle
        # in the returned ScenarioSpec -- purely a feasibility-check
        # input). Says nothing about the obstacle's SUBSEQUENT motion:
        # a dynamic obstacle threatening/colliding with the robot LATER
        # in the episode is the deliberate point of this curriculum stage,
        # not something this check restricts. cfg.dynamic_obstacle_initial_feasibility_check=False
        # opts back into the pre-item-7 behavior (dynamic obstacles excluded
        # from the initial feasibility check) for a caller that explicitly
        # wants a scenario allowed to start already partially blocked.
        if cfg.dynamic_obstacle_initial_feasibility_check and dynamic_obstacles:
            combined_obstacles = obstacles + [
                StaticObstacle(x=d.x0, y=d.y0, radius=d.radius) for d in dynamic_obstacles
            ]
            if not is_reachable(
                (start_x, start_y), (goal_x, goal_y), combined_obstacles, cfg.world_size_m, robot_radius,
            ):
                continue
            if cfg.feasibility_check == "ackermann" and not is_ackermann_feasible(
                start_x, start_y, start_yaw, goal_x, goal_y, goal_radius_m,
                combined_obstacles, cfg.world_size_m, robot_radius, min_turning_radius_m, wheelbase_m,
            ):
                continue
        return ScenarioSpec(
            seed=seed, start_x=float(start_x), start_y=float(start_y), start_yaw=float(start_yaw),
            goal_x=float(goal_x), goal_y=float(goal_y),
            static_obstacles=obstacles, dynamic_obstacles=dynamic_obstacles,
        )

    raise RuntimeError(
        f"generate_scenario: no feasible layout found for seed={seed} after {max_attempts} attempts "
        f"(world_size_m={cfg.world_size_m}, min_obstacles={cfg.min_obstacles}, "
        f"max_obstacles={cfg.max_obstacles} may be too dense, min_obstacles too high for the "
        "start/goal clearance rejection rate, or dynamic_obstacle_count/dynamic_obstacle_min_clearance_m "
        "too demanding for this world_size_m/obstacle density -- see _place_dynamic_obstacles)"
    )
