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

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from hunter_kinodynamic_rl.config.schema import ScenarioConfig, StartPoseConfig
from hunter_kinodynamic_rl.env.scenarios.ackermann_feasibility import is_ackermann_feasible
from hunter_kinodynamic_rl.env.scenarios.safe_start import sample_start_yaw

# The exact (min, max) static-obstacle radius range generate_scenario draws
# from below -- exposed as a module constant (rather than a buried literal)
# so config/schema.py's ObstaclePoolConfig cross-validation can check its
# own static_size_classes_m actually covers it (a pool class list that
# tops out below this max would leave some episodes' largest obstacles
# with no pool slot big enough to spawn safely).
STATIC_OBSTACLE_RADIUS_RANGE_M = (0.15, 0.5)
# Every DynamicObstacleSpec generate_scenario places uses this exact,
# fixed radius (never drawn from a range) -- exposed so obstacle-pool code
# doesn't need its own duplicated magic number for the dynamic marker's
# baked-in pool geometry.
DYNAMIC_OBSTACLE_RADIUS_M = 0.3


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
    # Diagnostic only (never read by any feasibility/physics logic) --
    # number of sample_start_yaw attempts this scenario's heading actually
    # needed, for reset-time structured logging. None for
    # heading_mode="legacy_random" (no sampling loop ever runs) or for a
    # fixed-benchmark ScenarioSpec (benchmark_loader.py never sets this).
    heading_sample_attempts: Optional[int] = None
    # item 3 (evaluation-only privileged metadata -- plan section 6.6's
    # "즉시 도달 불가능한 subgoal도 학습/평가 분포에 포함" negative-episode
    # requirement): whether this attempt was CHOSEN (by
    # scenario.goal_infeasible_fraction) to deliberately construct a
    # blocked/unreachable goal, and whether the constructed geometry was
    # actually VERIFIED infeasible (is_reachable()/is_ackermann_feasible()
    # both explicitly re-checked as False -- see
    # _build_goal_blocking_ring's caller in generate_scenario) before this
    # ScenarioSpec is returned. These two fields are always equal in
    # practice: generate_scenario NEVER returns a "chosen infeasible but
    # not actually verified infeasible" spec -- an attempt whose barrier
    # construction doesn't verify is redrawn (continue), never returned
    # with a false realized_infeasible. Kept as two separate fields (not
    # collapsed into one) purely so a caller/analysis script can still
    # tell "was this negative-episode selection even attempted" apart from
    # "did it actually succeed", without re-deriving that from
    # infeasibility_kind. NEVER read by observation_builder.py or
    # reward_calculator.py -- these are reporting/analysis fields only
    # (mirrors heading_sample_attempts' own "diagnostic only" contract
    # above); see tests/test_procedural_generator.py's own leakage guard.
    intended_infeasible: bool = False
    realized_infeasible: bool = False
    # "goal_encircled_blocked" (the only kind this generator currently
    # constructs -- a solid ring of obstacles fully surrounding the goal
    # point, verified grid-BFS-unreachable from start) or None (feasible,
    # or infeasible-fraction not selected this attempt).
    infeasibility_kind: Optional[str] = None


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
    radius = DYNAMIC_OBSTACLE_RADIUS_M
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


#: Static radius the goal-encircling ring uses for every one of its
#: obstacles -- comfortably mid-range within STATIC_OBSTACLE_RADIUS_RANGE_M
#: so it also passes any pool-catalog size-class check ordinary static
#: obstacles pass.
_INFEASIBLE_RING_OBSTACLE_RADIUS_M = 0.35


def _build_goal_blocking_ring(
    rng: np.random.RandomState, start_xy: Tuple[float, float], goal_xy: Tuple[float, float],
    half: float, robot_radius: float, clearance_m: float,
) -> Optional[List[StaticObstacle]]:
    """item 3: a REAL, decision-verified-infeasible goal, not merely a
    skipped feasibility check (code review finding: the previous
    ``force_infeasible_ok`` path only skipped ``is_reachable``/
    ``is_ackermann_feasible`` for the drawn goal -- the SAME sparse,
    typically-still-connected obstacle layout every other attempt gets, so
    almost every "infeasible" episode was, geometrically, still perfectly
    solvable; a 500-seed sample of this profile's negative episodes found
    ZERO actually grid-unreachable outcomes).

    Constructs a solid ring of obstacles fully surrounding ``goal_xy`` --
    ring radius ``R`` chosen so the goal point itself stays clear (``R >
    ring obstacle radius + robot_radius``, i.e. the goal cell is never
    itself occupied -- a genuinely UNREACHABLE goal, not a degenerate
    "goal spawned inside an obstacle" one) while staying well short of
    ``start_xy`` (``R`` is capped at a fraction of ``dist(start, goal)``,
    so the ring can never also swallow the start position). Adjacent ring
    obstacles are spaced tightly enough (relative to their own
    grid-inflated ``radius + robot_radius`` footprint, well under
    ``is_reachable``'s fixed 0.25 m grid resolution) that no rasterized
    gap survives anywhere around the circumference.

    Returns ``None`` (never a partial/unverified ring) if no ring radius
    satisfies BOTH constraints for this ``(start_xy, goal_xy)`` pair (goal
    too close to start for any ring that also clears the goal cell, or the
    ring would extend past the world boundary) -- the caller
    (``generate_scenario``) treats that exactly like every other
    infeasibility-construction failure: redraw the whole scenario attempt
    from the same ``rng`` stream, never silently fall back to the
    skip-the-check behaviour this replaces."""
    sx, sy = start_xy
    gx, gy = goal_xy
    dist = math.hypot(gx - sx, gy - sy)
    r_obs = _INFEASIBLE_RING_OBSTACLE_RADIUS_M
    # Lower bound: goal cell itself must stay free of every ring obstacle's
    # own (radius + robot_radius)-inflated footprint, plus a small margin
    # so floating-point/grid-quantization can never flip it.
    r_min = r_obs + robot_radius + 0.05
    # Upper bound: the ring must stay comfortably clear of the START point
    # too (never also encircling/blocking the robot's own spawn) and
    # inside the world boundary. The CLOSEST point on a circle of radius R
    # centered at goal to an external point `dist` away is exactly
    # `dist - R` (the circle/segment intersection toward that point) --
    # the true tight bound, not `0.5 * dist` (needlessly halves the usable
    # ring-radius range and made short-distance goals, e.g. near
    # goal_distance_range_m's lower end, geometrically impossible to ring
    # at all). A modest fixed margin (not the full scattered-obstacle
    # clearance_m design margin -- this is a solid barrier, not a
    # navigable-around obstacle) keeps floating-point/grid-quantization
    # from ever flipping the start cell itself.
    r_max = min(
        dist - (r_obs + robot_radius + max(0.1, 0.2 * clearance_m)),
        half - (r_obs + max(abs(gx), abs(gy))) if half > 0.0 else 0.0,
    )
    if not (r_min < r_max):
        return None
    ring_radius_m = float(rng.uniform(r_min, r_max))

    # Adjacent centers spaced at well under half of one obstacle's own
    # grid-inflated reach -- comfortably tighter than is_reachable's fixed
    # 0.25 m grid resolution, so the rasterized ring is always solid.
    arc_spacing_m = max(0.10, 0.5 * (r_obs + robot_radius))
    circumference_m = 2.0 * math.pi * ring_radius_m
    n_ring = max(8, int(math.ceil(circumference_m / arc_spacing_m)))

    ring: List[StaticObstacle] = []
    phase = float(rng.uniform(0.0, 2.0 * math.pi))
    for i in range(n_ring):
        theta = phase + (2.0 * math.pi * i) / n_ring
        cx = gx + ring_radius_m * math.cos(theta)
        cy = gy + ring_radius_m * math.sin(theta)
        if abs(cx) > half or abs(cy) > half:
            return None  # a ring point fell outside the world -- construction failed for this attempt
        ring.append(StaticObstacle(x=float(cx), y=float(cy), radius=r_obs))
    return ring


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
                       goal_radius_m: float = 0.42,
                       start_pose_cfg: Optional[StartPoseConfig] = None,
                       static_radius_quantizer: Optional[Callable[[float], float]] = None) -> ScenarioSpec:
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

    The START position specifically is inset FURTHER, by
    ``start_pose_cfg.min_wall_clearance_m`` on top of ``robot_radius`` (goal
    sampling is unaffected) -- ``min_wall_clearance_m=0.0`` (the field
    default, and every pre-existing profile) makes this identical to the
    robot_radius-only inset above. A combination that leaves no usable start
    region raises immediately, the same way the robot_radius-only case does.
    ``safe_start.sample_start_yaw``'s own front-safety-distance projection
    (for every ``heading_mode`` other than ``legacy_random``) separately
    enforces the same ``robot_radius + min_wall_clearance_m`` margin against
    the wall the robot would be facing, not just the position it starts at.

    ``start_pose_cfg`` (default ``None`` -> ``StartPoseConfig()``, whose own
    default ``heading_mode="legacy_random"`` reproduces the pre-existing
    behaviour exactly -- see that class's docstring) controls how
    ``start_yaw`` is chosen. Any OTHER ``heading_mode`` changes this
    function's internal draw order: the heading is sampled by
    :func:`~hunter_kinodynamic_rl.env.scenarios.safe_start.sample_start_yaw`
    AFTER static obstacles are placed for this attempt (never before, unlike
    ``legacy_random``), so it can reject a candidate heading that points
    into a nearby obstacle or straight at the world boundary within a
    bounded number of attempts, falling back to a deterministic (still
    clearance-checked) sweep if every random attempt is rejected. If even
    the deterministic fallback finds no safe heading for this attempt's
    ``(start_x, start_y)``, the WHOLE scenario attempt is redrawn (mirroring
    every other infeasibility path in this function) -- ``generate_scenario``
    only ever raises once its own ``max_attempts`` budget is exhausted,
    never silently accepts an unsafe heading.

    ``static_radius_quantizer`` (default ``None`` -- radii stay continuous,
    unchanged behaviour), when given, is applied to EVERY static obstacle's
    drawn radius before any clearance/feasibility check runs -- used by
    ``environment_node.py`` when ``obstacle_pool.enabled`` to snap each
    radius to a pool-compatible size class UP FRONT, so the feasibility
    checks below and the physical geometry actually spawned into Gazebo are
    always for the exact same radius (see env/spawning/obstacle_pool.py's
    module docstring).

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
    start_pose_cfg = start_pose_cfg or StartPoseConfig()
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
    band_mode = cfg.goal_sampling_mode == "robot_relative_band"
    if band_mode and start_pose_cfg.heading_mode != "legacy_random":
        raise RuntimeError(
            "generate_scenario: scenario.goal_sampling_mode='robot_relative_band' requires "
            f"start_pose.heading_mode='legacy_random' (got {start_pose_cfg.heading_mode!r}) -- every other "
            "heading_mode samples start_yaw AFTER the goal would need to be placed relative to it; refusing "
            "to silently reorder the scenario draw sequence"
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
    # requirement 1 (start-pose wall clearance): the START position itself
    # must additionally clear the wall by start_pose_cfg.min_wall_clearance_m
    # on top of robot_radius -- previously only robot_radius was inset here,
    # so a start point could be sampled with the robot's footprint already
    # inside the configured clearance margin. min_wall_clearance_m=0.0 (the
    # default, and every pre-existing profile) makes start_inset_half ==
    # inset_half exactly, so this is byte-identical for those profiles
    # (see test_legacy_random_heading_mode_matches_pre_existing_rng_draw_order).
    # Goal sampling deliberately keeps using the un-widened inset_half above
    # -- only the start position carries a wall-clearance requirement.
    start_inset_half = half - (robot_radius + start_pose_cfg.min_wall_clearance_m)
    if start_inset_half <= 0.0:
        raise RuntimeError(
            f"generate_scenario: robot_radius={robot_radius} + "
            f"start_pose.min_wall_clearance_m={start_pose_cfg.min_wall_clearance_m} leaves no inset room "
            f"in a world_size_m={cfg.world_size_m} world (half_extent={half}) -- world_size_m is too "
            "small for this robot's footprint plus the configured start-pose wall clearance"
        )

    for _attempt in range(max_attempts):
        start_x, start_y = rng.uniform(-start_inset_half, start_inset_half, size=2)
        legacy_heading = start_pose_cfg.heading_mode == "legacy_random"
        heading_sample_attempts: Optional[int] = None
        if legacy_heading:
            start_yaw = rng.uniform(-np.pi, np.pi)

        force_infeasible_ok = False
        if band_mode:
            # section 6.6 (arbitrary short-range subgoal training): goal is
            # drawn relative to start_yaw (already known -- band_mode
            # requires legacy_heading, checked above), not uniformly over
            # the world. A candidate outside the inset world bounds is
            # rejected and redrawn within this same bounded inner loop
            # (mirroring the uniform_world loop's own 20-attempt budget);
            # exhausting it redraws the WHOLE scenario attempt, same as
            # every other infeasibility path in this function.
            found_goal = False
            for _ in range(20):
                lo_deg, hi_deg = cfg.goal_direction_sectors_deg[rng.randint(len(cfg.goal_direction_sectors_deg))]
                offset_rad = np.deg2rad(rng.uniform(lo_deg, hi_deg))
                distance = rng.uniform(*cfg.goal_distance_range_m)
                candidate_x = start_x + distance * np.cos(start_yaw + offset_rad)
                candidate_y = start_y + distance * np.sin(start_yaw + offset_rad)
                if -inset_half <= candidate_x <= inset_half and -inset_half <= candidate_y <= inset_half:
                    goal_x, goal_y = candidate_x, candidate_y
                    found_goal = True
                    break
            if not found_goal:
                continue
            force_infeasible_ok = bool(cfg.goal_infeasible_fraction > 0.0
                                        and rng.random() < cfg.goal_infeasible_fraction)
        else:
            for _ in range(20):
                goal_x, goal_y = rng.uniform(-inset_half, inset_half, size=2)
                if np.hypot(goal_x - start_x, goal_y - start_y) >= min_start_goal_distance_m:
                    break

        num_obstacles = rng.randint(cfg.min_obstacles, cfg.max_obstacles + 1)
        obstacles: List[StaticObstacle] = []
        obstacle_clearance_m = start_pose_cfg.min_obstacle_clearance_m
        for _ in range(num_obstacles):
            radius = float(rng.uniform(*STATIC_OBSTACLE_RADIUS_RANGE_M))
            if static_radius_quantizer is not None:
                radius = float(static_radius_quantizer(radius))
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
            if np.hypot(ox - start_x, oy - start_y) < robot_radius + radius + obstacle_clearance_m:
                continue
            if np.hypot(ox - goal_x, oy - goal_y) < robot_radius + radius + obstacle_clearance_m:
                continue
            obstacles.append(StaticObstacle(x=float(ox), y=float(oy), radius=radius))

        if len(obstacles) < cfg.min_obstacles:
            continue

        # item 3: force_infeasible_ok used to just SKIP the reachability
        # check below for the drawn goal -- the exact same sparse, usually-
        # still-connected obstacle layout every other attempt gets, so
        # almost none of these were actually unreachable (a 500-seed sample
        # found zero). Now it constructs a solid ring of obstacles fully
        # surrounding the goal and VERIFIES (never assumes) the result is
        # genuinely grid-unreachable before accepting it -- see
        # _build_goal_blocking_ring's own docstring. A construction/
        # verification failure redraws the whole attempt, same as every
        # other infeasibility path in this function; it never falls back to
        # returning an unverified "infeasible" episode.
        infeasibility_kind: Optional[str] = None
        if force_infeasible_ok:
            ring = _build_goal_blocking_ring(
                rng, (start_x, start_y), (goal_x, goal_y), half, robot_radius, obstacle_clearance_m,
            )
            if ring is None:
                continue
            obstacles = obstacles + ring
            if is_reachable((start_x, start_y), (goal_x, goal_y), obstacles, cfg.world_size_m, robot_radius):
                continue
            if cfg.feasibility_check == "ackermann" and is_ackermann_feasible(
                start_x, start_y, start_yaw, goal_x, goal_y, goal_radius_m,
                obstacles, cfg.world_size_m, robot_radius, min_turning_radius_m, wheelbase_m,
            ):
                continue
            infeasibility_kind = "goal_encircled_blocked"
        elif not is_reachable((start_x, start_y), (goal_x, goal_y), obstacles, cfg.world_size_m, robot_radius):
            continue

        if not legacy_heading:
            # section (safe start/yaw): obstacles for THIS attempt are now
            # known -- sample a heading that clears them (and the world
            # boundary) within a bounded number of attempts, with a
            # deterministic, still clearance-checked fallback. None means
            # even the fallback found no safe heading for this
            # (start_x, start_y) -- redraw the WHOLE scenario attempt,
            # exactly like every other infeasibility path in this loop
            # (never silently accept an unsafe heading).
            sampled = sample_start_yaw(
                rng, start_pose_cfg, start_x, start_y, goal_x, goal_y, obstacles, half, robot_radius,
            )
            if sampled is None:
                continue
            start_yaw, heading_sample_attempts = sampled

        if not force_infeasible_ok and cfg.feasibility_check == "ackermann" and not is_ackermann_feasible(
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
        if not force_infeasible_ok and cfg.dynamic_obstacle_initial_feasibility_check and dynamic_obstacles:
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
            heading_sample_attempts=heading_sample_attempts,
            intended_infeasible=force_infeasible_ok, realized_infeasible=(infeasibility_kind is not None),
            infeasibility_kind=infeasibility_kind,
        )

    raise RuntimeError(
        f"generate_scenario: no feasible layout found for seed={seed} after {max_attempts} attempts "
        f"(world_size_m={cfg.world_size_m}, min_obstacles={cfg.min_obstacles}, "
        f"max_obstacles={cfg.max_obstacles} may be too dense, min_obstacles too high for the "
        "start/goal clearance rejection rate, or dynamic_obstacle_count/dynamic_obstacle_min_clearance_m "
        "too demanding for this world_size_m/obstacle density -- see _place_dynamic_obstacles)"
    )
