"""Versioned TRACTOR simulation curriculum and conflict-aware scene generator.

This module is pure Python/numpy.  It does not mutate the frozen
legacy environment files and is used only when ``environment_v2.enabled``
is explicitly selected by a profile.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from hunter_kinodynamic_rl.config.schema import (
    EnvironmentV2Config, RobotConfig, ScenarioConfig, StartPoseConfig,
)
from hunter_kinodynamic_rl.env.scenarios.ackermann_feasibility import is_ackermann_feasible
from hunter_kinodynamic_rl.env.scenarios.footprint_geometry import (
    circle_to_oriented_rectangle_clearance,
)
from hunter_kinodynamic_rl.env.scenarios.procedural_generator import (
    DynamicObstacleSpec, ScenarioSpec, StaticObstacle, generate_scenario, is_reachable,
)


@dataclass(frozen=True)
class CurriculumStage:
    level: int
    max_static_obstacles: int
    dynamic_obstacle_count: int
    dynamic_speed_ratio_cap: float


def curriculum_level(cfg: EnvironmentV2Config, episode_index: int, mode: str) -> int:
    if mode != "train":
        return cfg.evaluation_level
    level = 0
    for index, boundary in enumerate(cfg.curriculum_episode_boundaries):
        if episode_index >= boundary:
            level = index
        else:
            break
    return level


def curriculum_stage(cfg: EnvironmentV2Config, episode_index: int, mode: str) -> CurriculumStage:
    level = curriculum_level(cfg, episode_index, mode)
    return CurriculumStage(
        level=level,
        max_static_obstacles=cfg.curriculum_static_limits[level],
        dynamic_obstacle_count=cfg.curriculum_dynamic_counts[level],
        dynamic_speed_ratio_cap=cfg.curriculum_speed_ratio_caps[level],
    )


def closest_approach_metrics(
    ego_xy: Tuple[float, float], ego_velocity_xy: Tuple[float, float],
    obstacle_xy: Tuple[float, float], obstacle_velocity_xy: Tuple[float, float],
) -> Tuple[float, float]:
    """Return non-negative time and distance at closest point of approach."""
    rx, ry = obstacle_xy[0] - ego_xy[0], obstacle_xy[1] - ego_xy[1]
    rvx = obstacle_velocity_xy[0] - ego_velocity_xy[0]
    rvy = obstacle_velocity_xy[1] - ego_velocity_xy[1]
    speed_sq = rvx * rvx + rvy * rvy
    if speed_sq <= 1e-12:
        return float("inf"), math.hypot(rx, ry)
    tcpa = max(0.0, -(rx * rvx + ry * rvy) / speed_sq)
    return tcpa, math.hypot(rx + rvx * tcpa, ry + rvy * tcpa)


def _path_frame(base: ScenarioSpec):
    dx, dy = base.goal_x - base.start_x, base.goal_y - base.start_y
    distance = max(math.hypot(dx, dy), 1e-6)
    ux, uy = dx / distance, dy / distance
    return (ux, uy), (-uy, ux), distance, math.atan2(dy, dx)


def _world_point(base: ScenarioSpec, along_m: float, lateral_m: float) -> Tuple[float, float]:
    forward, lateral, _, _ = _path_frame(base)
    return (
        base.start_x + forward[0] * along_m + lateral[0] * lateral_m,
        base.start_y + forward[1] * along_m + lateral[1] * lateral_m,
    )


def _shape_dimensions(shape: str, rng: np.random.RandomState) -> Tuple[float, float, float]:
    if shape == "cylinder":
        radius = float(rng.uniform(0.18, 0.45))
        return radius, 0.0, 0.0
    if shape == "cart":
        length, width = float(rng.uniform(0.70, 0.95)), float(rng.uniform(0.42, 0.60))
    elif shape == "l_shape":
        length, width = float(rng.uniform(0.65, 0.95)), float(rng.uniform(0.65, 0.95))
    else:
        length, width = float(rng.uniform(0.45, 0.90)), float(rng.uniform(0.25, 0.55))
    return 0.5 * math.hypot(length, width), length, width


def _candidate_static(
    x: float, y: float, yaw: float, shape: str, rng: np.random.RandomState,
    dimensions: Optional[Tuple[float, float]] = None,
) -> StaticObstacle:
    if dimensions is None:
        radius, length, width = _shape_dimensions(shape, rng)
    else:
        length, width = dimensions
        radius = 0.5 * math.hypot(length, width)
    return StaticObstacle(
        x=float(x), y=float(y), radius=float(radius), shape=shape,
        length_m=float(length), width_m=float(width), yaw_rad=float(yaw),
    )


def _clear_for_static(
    obstacle: StaticObstacle, placed: Sequence[StaticObstacle], base: ScenarioSpec,
    robot: RobotConfig, cfg: EnvironmentV2Config, world_size_m: float,
) -> bool:
    if abs(obstacle.x) + obstacle.radius >= world_size_m / 2.0:
        return False
    if abs(obstacle.y) + obstacle.radius >= world_size_m / 2.0:
        return False
    for xy, yaw in (
        ((base.start_x, base.start_y), base.start_yaw),
        ((base.goal_x, base.goal_y), _path_frame(base)[3]),
    ):
        if cfg.use_oriented_robot_footprint:
            clearance = circle_to_oriented_rectangle_clearance(
                (obstacle.x, obstacle.y), obstacle.radius, xy, yaw,
                robot.length_m, robot.width_m, cfg.footprint_padding_m,
            )
        else:
            clearance = math.hypot(obstacle.x - xy[0], obstacle.y - xy[1]) - (
                obstacle.radius + robot.collision_radius_m + cfg.footprint_padding_m
            )
        if clearance < 0.0:
            return False
    return all(
        math.hypot(obstacle.x - other.x, obstacle.y - other.y)
        >= 0.85 * (obstacle.radius + other.radius)
        for other in placed
    )


def _topology_candidates(
    topology: str, base: ScenarioSpec, count: int, rng: np.random.RandomState,
) -> List[StaticObstacle]:
    _, _, distance, heading = _path_frame(base)
    candidates: List[StaticObstacle] = []

    def add(along, lateral, shape="box", yaw=heading, dims=None):
        x, y = _world_point(base, along, lateral)
        candidates.append(_candidate_static(x, y, yaw, shape, rng, dims))

    if topology == "corridor":
        for along in np.linspace(0.25 * distance, 0.80 * distance, max(1, count // 2)):
            add(float(along), 1.15, "box", heading, (0.75, 0.28))
            add(float(along), -1.15, "box", heading, (0.75, 0.28))
    elif topology == "doorway":
        # Two lateral wall segments with a 1.4 m opening around the path.
        add(0.55 * distance, 1.22, "box", heading + math.pi / 2.0, (1.04, 0.28))
        add(0.55 * distance, -1.22, "box", heading + math.pi / 2.0, (1.04, 0.28))
    elif topology == "intersection":
        for along in (0.38 * distance, 0.67 * distance):
            add(along, 1.25, "box", heading, (0.65, 0.40))
            add(along, -1.25, "box", heading, (0.65, 0.40))
    elif topology == "s_curve":
        for index, along in enumerate(np.linspace(0.25 * distance, 0.78 * distance, max(2, count))):
            add(float(along), 0.78 if index % 2 == 0 else -0.78, "cylinder")
    elif topology == "warehouse":
        for along in np.linspace(0.22 * distance, 0.82 * distance, max(1, count // 2)):
            add(float(along), 1.35, "cart", heading)
            add(float(along), -1.35, "cart", heading)
    return candidates[:count]


def _make_static_layout(
    base: ScenarioSpec, topology: str, target_count: int, rng: np.random.RandomState,
    scenario_cfg: ScenarioConfig, environment_cfg: EnvironmentV2Config, robot: RobotConfig,
) -> List[StaticObstacle]:
    placed: List[StaticObstacle] = []
    for candidate in _topology_candidates(topology, base, target_count, rng):
        if _clear_for_static(candidate, placed, base, robot, environment_cfg, scenario_cfg.world_size_m):
            placed.append(candidate)
    half = scenario_cfg.world_size_m / 2.0
    for _ in range(300):
        if len(placed) >= target_count:
            break
        shape = environment_cfg.obstacle_shapes[int(rng.randint(len(environment_cfg.obstacle_shapes)))]
        radius, length, width = _shape_dimensions(shape, rng)
        x, y = rng.uniform(-half + radius, half - radius, size=2)
        candidate = StaticObstacle(
            x=float(x), y=float(y), radius=radius, shape=shape,
            length_m=length, width_m=width, yaw_rad=float(rng.uniform(-math.pi, math.pi)),
        )
        if _clear_for_static(candidate, placed, base, robot, environment_cfg, scenario_cfg.world_size_m):
            placed.append(candidate)
    return placed


def _motion_heading(pattern: str, path_heading: float, rng: np.random.RandomState) -> float:
    if pattern in ("crossing", "sine_swerve", "bounce"):
        return path_heading + (math.pi / 2.0 if rng.rand() < 0.5 else -math.pi / 2.0)
    if pattern in ("head_on", "cut_in"):
        return path_heading + math.pi + (float(rng.uniform(-0.45, 0.45)) if pattern == "cut_in" else 0.0)
    return path_heading


def _clear_dynamic_spawn(
    x: float, y: float, radius: float, dynamic: Sequence[DynamicObstacleSpec],
    static: Sequence[StaticObstacle], base: ScenarioSpec, robot: RobotConfig, world_size_m: float,
) -> bool:
    half = world_size_m / 2.0
    if abs(x) + radius >= half or abs(y) + radius >= half:
        return False
    if math.hypot(x - base.start_x, y - base.start_y) < radius + robot.collision_radius_m + 0.35:
        return False
    if math.hypot(x - base.goal_x, y - base.goal_y) < radius + robot.collision_radius_m + 0.20:
        return False
    if any(math.hypot(x - obs.x, y - obs.y) < radius + obs.radius + 0.15 for obs in static):
        return False
    return not any(math.hypot(x - obs.x0, y - obs.y0) < radius + obs.radius + 0.20 for obs in dynamic)


def _make_dynamic_obstacles(
    base: ScenarioSpec, static: Sequence[StaticObstacle], stage: CurriculumStage,
    rng: np.random.RandomState, scenario_cfg: ScenarioConfig, environment_cfg: EnvironmentV2Config,
    robot: RobotConfig,
) -> List[DynamicObstacleSpec]:
    if stage.dynamic_obstacle_count <= 0 or stage.dynamic_speed_ratio_cap <= 0.0:
        return []
    forward, _, path_distance, path_heading = _path_frame(base)
    ego_speed = max(0.35, min(0.65 * robot.max_forward_speed_mps, robot.max_forward_speed_mps))
    ego_velocity = (ego_speed * forward[0], ego_speed * forward[1])
    ratio_lo = min(environment_cfg.dynamic_speed_ratio_range[0], stage.dynamic_speed_ratio_cap)
    ratio_hi = min(environment_cfg.dynamic_speed_ratio_range[1], stage.dynamic_speed_ratio_cap)
    ratio_lo = max(min(ratio_lo, ratio_hi), 0.05)
    conflict_count = min(
        stage.dynamic_obstacle_count,
        int(round(stage.dynamic_obstacle_count * environment_cfg.conflict_fraction)),
    )
    dynamic: List[DynamicObstacleSpec] = []
    half = scenario_cfg.world_size_m / 2.0
    for index in range(stage.dynamic_obstacle_count):
        placed = False
        for attempt in range(320):
            pattern = environment_cfg.motion_patterns[int(rng.randint(len(environment_cfg.motion_patterns)))]
            interaction = environment_cfg.interaction_modes[int(rng.randint(len(environment_cfg.interaction_modes)))]
            speed = float(rng.uniform(ratio_lo, ratio_hi)) * robot.max_forward_speed_mps
            velocity_heading = _motion_heading(pattern, path_heading, rng)
            vx, vy = speed * math.cos(velocity_heading), speed * math.sin(velocity_heading)
            shape = environment_cfg.obstacle_shapes[int(rng.randint(len(environment_cfg.obstacle_shapes)))]
            radius, length, width = _shape_dimensions(shape, rng)
            target_ttc: Optional[float] = None
            target_dcpa: Optional[float] = None
            wants_conflict = index < conflict_count and attempt < 200
            if wants_conflict:
                max_path_ttc = max(0.7, 0.82 * path_distance / ego_speed)
                if max_path_ttc < environment_cfg.conflict_ttc_range_sec[0]:
                    continue
                t_lo = environment_cfg.conflict_ttc_range_sec[0]
                t_hi = min(environment_cfg.conflict_ttc_range_sec[1], max_path_ttc)
                if t_hi <= 0.0:
                    continue
                target_ttc = float(rng.uniform(max(0.25, min(t_lo, t_hi)), max(0.25, t_hi)))
                target_dcpa = float(rng.uniform(*environment_cfg.conflict_dcpa_range_m))
                collision_x = base.start_x + ego_velocity[0] * target_ttc
                collision_y = base.start_y + ego_velocity[1] * target_ttc
                rvx, rvy = vx - ego_velocity[0], vy - ego_velocity[1]
                rv_norm = math.hypot(rvx, rvy)
                if rv_norm <= 1e-5:
                    continue
                side = -1.0 if rng.rand() < 0.5 else 1.0
                offset_x, offset_y = side * (-rvy / rv_norm) * target_dcpa, side * (rvx / rv_norm) * target_dcpa
                x = collision_x + offset_x - vx * target_ttc
                y = collision_y + offset_y - vy * target_ttc
            else:
                x, y = rng.uniform(-half + radius, half - radius, size=2)
            if not _clear_dynamic_spawn(x, y, radius, dynamic, static, base, robot, scenario_cfg.world_size_m):
                continue
            accel = float(rng.uniform(*environment_cfg.dynamic_accel_limit_range_mps2))
            turn_rate = float(rng.uniform(*environment_cfg.dynamic_turn_rate_range_rad_s))
            realized_ttc, realized_dcpa = closest_approach_metrics(
                (base.start_x, base.start_y), ego_velocity, (float(x), float(y)), (vx, vy),
            )
            dynamic.append(DynamicObstacleSpec(
                x0=float(x), y0=float(y), vx=float(vx), vy=float(vy), radius=radius,
                motion_pattern=pattern, shape=shape, length_m=length, width_m=width,
                yaw_rad=float(velocity_heading), accel_limit_mps2=accel,
                turn_rate_rad_s=turn_rate, interaction_mode=interaction,
                target_ttc_sec=realized_ttc if wants_conflict else None,
                target_dcpa_m=realized_dcpa if wants_conflict else None,
            ))
            placed = True
            break
        if not placed:
            raise RuntimeError(
                f"tractor_env_v2: could not place dynamic obstacle {index + 1}/"
                f"{stage.dynamic_obstacle_count} for seed={base.seed}"
            )
    return dynamic


def _initial_dynamic_layout_feasible(
    base: ScenarioSpec, static: Sequence[StaticObstacle], dynamic: Sequence[DynamicObstacleSpec],
    scenario_cfg: ScenarioConfig, robot: RobotConfig, goal_radius_m: float,
) -> bool:
    """Apply the legacy initial-feasibility contract to a v2 dynamic layout.

    Dynamic obstacles remain dynamic in the returned scenario.  Circular
    t=0 envelopes are used only for this rejection filter, matching
    :func:`procedural_generator.generate_scenario`'s established policy.
    """
    combined = list(static) + [
        StaticObstacle(x=item.x0, y=item.y0, radius=item.radius)
        for item in dynamic
    ]
    if not is_reachable(
        (base.start_x, base.start_y), (base.goal_x, base.goal_y), combined,
        scenario_cfg.world_size_m, robot.collision_radius_m,
    ):
        return False
    if scenario_cfg.feasibility_check != "ackermann":
        return True
    return is_ackermann_feasible(
        base.start_x, base.start_y, base.start_yaw,
        base.goal_x, base.goal_y, goal_radius_m,
        combined, scenario_cfg.world_size_m, robot.collision_radius_m,
        1.0 / robot.max_curvature, robot.wheelbase_m,
    )


def _decorate_legacy_fallback(
    obstacles: Sequence[StaticObstacle], rng: np.random.RandomState,
    environment_cfg: EnvironmentV2Config,
) -> List[StaticObstacle]:
    """Give a feasibility-verified fallback layout v2 physical shapes.

    Every new geometry fits inside the legacy obstacle's already-verified
    circle, so the fallback cannot invalidate that path check.
    """
    decorated = []
    for obstacle in obstacles:
        shape = environment_cfg.obstacle_shapes[int(rng.randint(len(environment_cfg.obstacle_shapes)))]
        if shape == "cylinder":
            length = width = 0.0
        else:
            length, width = 1.35 * obstacle.radius, 0.75 * obstacle.radius
        decorated.append(dataclasses.replace(
            obstacle, shape=shape, length_m=length, width_m=width,
            yaw_rad=float(rng.uniform(-math.pi, math.pi)),
        ))
    return decorated


def generate_v2_scenario(
    seed: int, scenario_cfg: ScenarioConfig, environment_cfg: EnvironmentV2Config,
    robot: RobotConfig, start_pose_cfg: StartPoseConfig, goal_radius_m: float,
    episode_index: int, mode: str, *, _layout_retry: int = 0,
) -> ScenarioSpec:
    """Generate one deterministic curriculum scene from seed and episode index."""
    if not environment_cfg.enabled:
        raise ValueError("generate_v2_scenario requires environment_v2.enabled=true")
    stage = curriculum_stage(environment_cfg, episode_index, mode)
    base_cfg = dataclasses.replace(
        scenario_cfg, min_obstacles=0, max_obstacles=0, dynamic_obstacle_count=0,
        # A full-layout retry is reached only after an originally feasible
        # scene could not admit a feasible dynamic placement.  Do not let a
        # retry silently change that episode into a deliberately goal-blocked
        # negative example merely because the mixed retry seed differs.
        goal_infeasible_fraction=(0.0 if _layout_retry else scenario_cfg.goal_infeasible_fraction),
    )
    base = generate_scenario(
        seed, base_cfg, robot_radius=robot.collision_radius_m,
        min_turning_radius_m=(1.0 / robot.max_curvature) if robot.max_curvature > 0.0 else None,
        wheelbase_m=robot.wheelbase_m, goal_radius_m=goal_radius_m,
        start_pose_cfg=start_pose_cfg,
    )
    rng = np.random.RandomState((int(seed) ^ 0x5EEDBEEF) & 0xFFFFFFFF)
    topology = environment_cfg.topology_families[int(rng.randint(len(environment_cfg.topology_families)))]
    minimum = min(scenario_cfg.min_obstacles, stage.max_static_obstacles)
    target_static = (
        int(rng.randint(minimum, stage.max_static_obstacles + 1))
        if stage.max_static_obstacles > minimum else minimum
    )
    static = (
        list(base.static_obstacles) if base.realized_infeasible
        else _make_static_layout(base, topology, target_static, rng, scenario_cfg, environment_cfg, robot)
    )
    if base.realized_infeasible:
        topology = "goal_blocked"
    if len(static) < minimum:
        raise RuntimeError(
            f"tractor_env_v2: placed {len(static)} static obstacles, below required {minimum} for seed={seed}"
        )
    # Retain the package's established conservative path gates.  Shape-aware
    # start/goal clearance above is exact for the Hunter rectangle; path
    # feasibility intentionally uses circumscribed radii as a safe fallback.
    geometry_feasible = base.realized_infeasible or is_reachable(
        (base.start_x, base.start_y), (base.goal_x, base.goal_y), static,
        scenario_cfg.world_size_m, robot.collision_radius_m,
    )
    if geometry_feasible and not base.realized_infeasible and scenario_cfg.feasibility_check == "ackermann" and static:
        geometry_feasible = is_ackermann_feasible(
        base.start_x, base.start_y, base.start_yaw, base.goal_x, base.goal_y, goal_radius_m,
        static, scenario_cfg.world_size_m, robot.collision_radius_m,
        1.0 / robot.max_curvature, robot.wheelbase_m,
        )
    if not geometry_feasible and not base.realized_infeasible:
        fallback_cfg = dataclasses.replace(
            scenario_cfg, min_obstacles=target_static, max_obstacles=target_static,
            dynamic_obstacle_count=0, goal_infeasible_fraction=0.0,
        )
        base = generate_scenario(
            seed, fallback_cfg, robot_radius=robot.collision_radius_m,
            min_turning_radius_m=(1.0 / robot.max_curvature) if robot.max_curvature > 0.0 else None,
            wheelbase_m=robot.wheelbase_m, goal_radius_m=goal_radius_m,
            start_pose_cfg=start_pose_cfg,
        )
        static = _decorate_legacy_fallback(base.static_obstacles, rng, environment_cfg)
        topology = "clutter_fallback"
    if base.realized_infeasible or not scenario_cfg.dynamic_obstacle_initial_feasibility_check:
        dynamic = _make_dynamic_obstacles(
            base, static, stage, rng, scenario_cfg, environment_cfg, robot,
        )
    else:
        dynamic = None
        last_placement_error = None
        # Reuse the scenario contract's bounded layout-attempt budget.  Each
        # redraw consumes the same seed-derived RNG stream, so generation is
        # deterministic while never silently accepting a t=0 blocked layout.
        for _ in range(max(1, scenario_cfg.dynamic_obstacle_placement_attempts)):
            try:
                candidate = _make_dynamic_obstacles(
                    base, static, stage, rng, scenario_cfg, environment_cfg, robot,
                )
            except RuntimeError as error:
                last_placement_error = error
                continue
            if _initial_dynamic_layout_feasible(
                base, static, candidate, scenario_cfg, robot, goal_radius_m,
            ):
                dynamic = candidate
                break
        if dynamic is None:
            full_layout_attempts = max(1, scenario_cfg.dynamic_obstacle_placement_attempts)
            if _layout_retry + 1 < full_layout_attempts:
                mixed_seed = (
                    int(seed) ^ (0x9E3779B9 * (_layout_retry + 1))
                ) & 0xFFFFFFFF
                replacement = generate_v2_scenario(
                    mixed_seed, scenario_cfg, environment_cfg, robot, start_pose_cfg,
                    goal_radius_m, episode_index, mode, _layout_retry=_layout_retry + 1,
                )
                # The externally registered seed remains the episode identity;
                # the deterministic retry schedule is part of this generator's
                # versioned source contract and is replayed from that seed.
                return dataclasses.replace(replacement, seed=seed)
            detail = "" if last_placement_error is None else f"; last placement error: {last_placement_error}"
            raise RuntimeError(
                "tractor_env_v2: could not sample a static+dynamic t=0 feasible layout "
                f"after {full_layout_attempts} deterministic full-layout attempts "
                f"for seed={seed}{detail}"
            )
    return dataclasses.replace(
        base, static_obstacles=static, dynamic_obstacles=dynamic,
        environment_version=environment_cfg.contract_version,
        curriculum_level=stage.level, topology=topology,
        conflict_obstacle_count=sum(obs.target_ttc_sec is not None for obs in dynamic),
    )
