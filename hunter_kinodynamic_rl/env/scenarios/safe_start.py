"""Safe initial-heading sampling for a procedurally generated start pose
(requirement 1 of the drl_agent -> hunter_kinodynamic_rl port -- see
``config/schema.py``'s ``StartPoseConfig`` and
``procedural_generator.generate_scenario``, which is this module's only
caller).

Pure Python/numpy -- no Gazebo, no ROS. Called from INSIDE
``generate_scenario``'s own retry loop, after that attempt's static
obstacles are already known, so the filter below can reject a heading that
points into one of them.

Scope note: only STATIC obstacles (and the world boundary) are considered
here -- dynamic obstacles are placed later in ``generate_scenario`` (after
the heading is fixed) and move immediately once the episode starts, so
their t=0 position is not a meaningful "is this a safe place to be
FACING" signal the way a static wall/obstacle is.

Algorithm (mirrors drl_agent's own inward-safe-heading sampler in spirit,
adapted to this package's Ackermann/StaticObstacle types):

1. A candidate heading is proposed according to ``cfg.heading_mode``
   (uniform random / biased toward the goal bearing / biased toward the
   probe direction with the most surrounding clearance).
2. The candidate is REJECTED if any obstacle sits within
   ``cfg.front_safety_distance_m`` of the start position AND within
   ``cfg.front_cone_half_angle_rad`` of the candidate heading (a
   robot-footprint-inclusive distance: obstacle radius + robot_radius is
   subtracted before comparing against front_safety_distance_m), OR if a
   point ``front_safety_distance_m`` ahead along the candidate heading
   would land within ``cfg.min_wall_clearance_m + robot_radius`` of the
   world boundary (i.e. the candidate points the robot's FOOTPRINT, not
   just its center point, at a wall it would reach almost immediately).
3. Up to ``cfg.max_sampling_attempts`` candidates are tried. If none pass,
   a DETERMINISTIC fallback sweeps 36 evenly-spaced headings (still
   filtered by the exact same safety check) and picks the best one by the
   active mode's own preference (nearest-to-goal for ``goal_biased``,
   highest-clearance otherwise).
4. If not even one of the 36 fallback headings is safe, returns ``None`` --
   the caller treats this exactly like any other scenario-generation
   infeasibility (redraw the whole attempt), so an unsafe heading is never
   silently returned.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

from hunter_kinodynamic_rl.common.geometry import wrap_to_pi
from hunter_kinodynamic_rl.config.schema import StartPoseConfig

if TYPE_CHECKING:
    # Deferred to a type-checking-only import: procedural_generator.py
    # imports sample_start_yaw from THIS module at module load time, so a
    # real (runtime) import back of StaticObstacle here would be a circular
    # import. `from __future__ import annotations` (above) already makes
    # every annotation in this file a lazily-evaluated string, so this
    # guard is all that's needed for type-checkers without ever executing
    # at runtime.
    from hunter_kinodynamic_rl.env.scenarios.procedural_generator import StaticObstacle

_FALLBACK_HEADING_COUNT = 36


def _is_heading_safe(
    start_x: float, start_y: float, yaw: float,
    obstacles: List[StaticObstacle], world_half_extent_m: float, robot_radius: float,
    cfg: StartPoseConfig,
) -> bool:
    for obs in obstacles:
        dx, dy = obs.x - start_x, obs.y - start_y
        dist = math.hypot(dx, dy) - obs.radius - robot_radius
        if dist >= cfg.front_safety_distance_m:
            continue
        bearing = wrap_to_pi(math.atan2(dy, dx) - yaw)
        if abs(bearing) <= cfg.front_cone_half_angle_rad:
            return False  # a too-close obstacle sits within the front cone
    fx = start_x + cfg.front_safety_distance_m * math.cos(yaw)
    fy = start_y + cfg.front_safety_distance_m * math.sin(yaw)
    # The projected point is the robot's CENTER at that future pose, so the
    # boundary must additionally be kept back by robot_radius (the robot's
    # own footprint) on top of the configured clearance margin -- previously
    # only min_wall_clearance_m was subtracted here, which let a heading
    # pass this check with the robot's footprint already inside the margin.
    limit = world_half_extent_m - cfg.min_wall_clearance_m - robot_radius
    if not (-limit <= fx <= limit and -limit <= fy <= limit):
        return False  # heading drives straight at the boundary
    return True


def _min_obstacle_or_wall_clearance(
    start_x: float, start_y: float, yaw: float,
    obstacles: List[StaticObstacle], world_half_extent_m: float, robot_radius: float,
) -> float:
    """Clearance score used to rank candidates in ``free_space_biased`` mode
    and to pick the best safe fallback heading in every mode except
    ``goal_biased`` -- the smaller of (a) the nearest obstacle's
    footprint-inclusive distance anywhere near this heading and (b) how far
    a point projected ``robot_radius``-scaled ahead stays from the world
    boundary. Larger is safer/more open."""
    obstacle_clearance = math.inf
    for obs in obstacles:
        dx, dy = obs.x - start_x, obs.y - start_y
        bearing = wrap_to_pi(math.atan2(dy, dx) - yaw)
        if abs(bearing) > math.pi / 2.0:
            continue  # behind the robot -- not relevant to "facing this way"
        obstacle_clearance = min(obstacle_clearance, math.hypot(dx, dy) - obs.radius - robot_radius)
    # Ray-to-box-boundary distance: how far (start_x, start_y) can travel
    # along `yaw` before exiting the [-half, half]^2 world -- a direct,
    # exact measure of "how much open room is there in this direction"
    # (never a proxy/approximation).
    dx, dy = math.cos(yaw), math.sin(yaw)
    half = world_half_extent_m
    wall_clearance = math.inf
    if dx > 1e-9:
        wall_clearance = min(wall_clearance, (half - start_x) / dx)
    elif dx < -1e-9:
        wall_clearance = min(wall_clearance, (-half - start_x) / dx)
    if dy > 1e-9:
        wall_clearance = min(wall_clearance, (half - start_y) / dy)
    elif dy < -1e-9:
        wall_clearance = min(wall_clearance, (-half - start_y) / dy)
    # Match obstacle_clearance's units (raw distance minus robot_radius) so
    # the min() below trades the two off on a consistent, footprint-inclusive
    # basis instead of comparing a footprint-inclusive obstacle distance
    # against a bare geometric ray-to-boundary distance.
    wall_clearance -= robot_radius
    return min(obstacle_clearance, wall_clearance)


def sample_start_yaw(
    rng: np.random.RandomState, cfg: StartPoseConfig,
    start_x: float, start_y: float, goal_x: float, goal_y: float,
    obstacles: List[StaticObstacle], world_half_extent_m: float, robot_radius: float,
) -> Optional[Tuple[float, int]]:
    """Returns ``(yaw, attempts_used)`` for the first (or best fallback)
    heading that passes ``_is_heading_safe``, or ``None`` if not even the
    deterministic fallback sweep finds one. Never called for
    ``cfg.heading_mode == "legacy_random"`` (the caller keeps the original
    unconditional draw for that mode)."""
    goal_heading = math.atan2(goal_y - start_y, goal_x - start_x)

    def _propose(attempt_rng: np.random.RandomState) -> float:
        if cfg.heading_mode == "goal_biased":
            if attempt_rng.uniform(0.0, 1.0) < cfg.goal_bias_prob:
                return wrap_to_pi(goal_heading + attempt_rng.uniform(
                    -cfg.goal_bias_spread_rad, cfg.goal_bias_spread_rad))
            return float(attempt_rng.uniform(-math.pi, math.pi))
        if cfg.heading_mode == "free_space_biased":
            probes = attempt_rng.uniform(-math.pi, math.pi, size=cfg.free_space_probe_count)
            scores = np.array([
                _min_obstacle_or_wall_clearance(start_x, start_y, float(p), obstacles, world_half_extent_m,
                                                 robot_radius)
                for p in probes
            ])
            finite = np.where(np.isfinite(scores), scores, np.max(scores[np.isfinite(scores)], initial=1.0))
            weights = np.exp(finite - np.max(finite))
            weights = weights / weights.sum()
            return float(attempt_rng.choice(probes, p=weights))
        return float(attempt_rng.uniform(-math.pi, math.pi))  # "random_rejected"

    for attempt in range(1, cfg.max_sampling_attempts + 1):
        candidate = _propose(rng)
        if _is_heading_safe(start_x, start_y, candidate, obstacles, world_half_extent_m, robot_radius, cfg):
            return candidate, attempt

    # Deterministic fallback: no more RNG draws -- an evenly-spaced sweep,
    # filtered by the exact same safety check, ranked by the active mode's
    # own preference among the ones that pass.
    fallback_headings = [
        wrap_to_pi(-math.pi + 2.0 * math.pi * i / _FALLBACK_HEADING_COUNT)
        for i in range(_FALLBACK_HEADING_COUNT)
    ]
    safe_fallbacks = [
        h for h in fallback_headings
        if _is_heading_safe(start_x, start_y, h, obstacles, world_half_extent_m, robot_radius, cfg)
    ]
    if not safe_fallbacks:
        return None
    if cfg.heading_mode == "goal_biased":
        best = min(safe_fallbacks, key=lambda h: abs(wrap_to_pi(h - goal_heading)))
    else:
        best = max(
            safe_fallbacks,
            key=lambda h: _min_obstacle_or_wall_clearance(
                start_x, start_y, h, obstacles, world_half_extent_m, robot_radius),
        )
    return best, cfg.max_sampling_attempts
