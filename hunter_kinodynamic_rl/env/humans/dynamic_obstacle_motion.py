"""Generic moving-obstacle motion patterns (section 27 -- NOT pedestrian-
specific; a ``GenericMovingObstacle`` abstraction swappable for a HuNav
pedestrian or another robot later).

Six patterns, deterministic given a seed:
  crossing            -- perpendicular to the robot's start->goal line
  head_on             -- anti-parallel to the robot's start->goal direction
  cut_in              -- starts ahead and to one side, converges INTO the path
  parallel            -- parallel to start->goal, offset laterally (no crossing)
  random_waypoint     -- periodically re-targets a new random point (stateful)
  constant_velocity   -- the DynamicObstacleSpec's own (vx, vy), unchanged

``crossing``/``head_on``/``cut_in``/``parallel`` are constant-velocity for
their whole episode (the geometry is baked into the initial (vx, vy) once);
only ``random_waypoint`` needs a per-tick state update, handled by
:class:`RandomWaypointState`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from hunter_kinodynamic_rl.env.scenarios.procedural_generator import DynamicObstacleSpec


class MotionPattern(str, Enum):
    CROSSING = "crossing"
    HEAD_ON = "head_on"
    CUT_IN = "cut_in"
    PARALLEL = "parallel"
    RANDOM_WAYPOINT = "random_waypoint"
    CONSTANT_VELOCITY = "constant_velocity"


_ALL_PATTERNS = tuple(MotionPattern)


def assign_pattern(seed: int, index: int) -> MotionPattern:
    """Deterministic per-obstacle pattern choice from (episode seed, obstacle index)."""
    rng = np.random.RandomState((seed * 1000003 + index) & 0xFFFFFFFF)
    return _ALL_PATTERNS[rng.randint(0, len(_ALL_PATTERNS))]


def position_at_elapsed_time(spec0: DynamicObstacleSpec, elapsed_sec: float) -> tuple:
    """Constant-velocity obstacle position computed DIRECTLY from the
    ORIGINAL spawn-time spec and the TOTAL elapsed simulation time since
    episode start (section P0-8) -- not iteratively accumulated tick by
    tick (``x += vx * dt_sec``, repeated every step). Iterative
    accumulation drifts over an episode from two compounding sources: (1)
    plain floating-point summation error, and (2) the PREDICTED per-tick
    dt used to move the obstacle BEFORE physics advances (the real
    duration is only known AFTER, via /clock -- see
    environment_node.py::_tick_dynamic_obstacles) never being bit-exact.
    Anchoring to the immutable spawn-time spec and the REAL observed
    elapsed-time-so-far means only the CURRENT tick's small
    dt-prediction is ever uncertain; it is self-corrected at the start of
    the next tick once /clock reports the real value, so error never
    accumulates across the whole episode."""
    return spec0.x0 + spec0.vx * elapsed_sec, spec0.y0 + spec0.vy * elapsed_sec


def apply_pattern(
    spec: DynamicObstacleSpec, pattern: MotionPattern,
    start_xy, goal_xy, speed_mps: float, seed: int, index: int,
) -> DynamicObstacleSpec:
    """Recompute (x0, y0, vx, vy) for the given pattern relative to the
    robot's start->goal line, keeping the spec's own radius. The obstacle's
    (x0, y0) from procedural_generator's uniform draw is reused as its
    lateral/longitudinal ANCHOR point (already collision-free vs. static
    obstacles/start/goal by construction), reprojected onto a heading that
    matches the requested pattern."""
    sx, sy = start_xy
    gx, gy = goal_xy
    heading = math.atan2(gy - sy, gx - sx)

    if pattern == MotionPattern.CROSSING:
        vh = heading + math.pi / 2.0
    elif pattern == MotionPattern.HEAD_ON:
        vh = heading + math.pi
    elif pattern == MotionPattern.PARALLEL:
        vh = heading
    elif pattern == MotionPattern.CUT_IN:
        rng = np.random.RandomState((seed * 7919 + index) & 0xFFFFFFFF)
        vh = heading + math.pi + rng.uniform(-0.6, 0.6)
    else:  # RANDOM_WAYPOINT, CONSTANT_VELOCITY -- keep the spec's own (vx, vy)
        return spec

    vx = speed_mps * math.cos(vh)
    vy = speed_mps * math.sin(vh)
    return DynamicObstacleSpec(x0=spec.x0, y0=spec.y0, vx=vx, vy=vy, radius=spec.radius)


@dataclass
class RandomWaypointState:
    """Stateful ticker for MotionPattern.RANDOM_WAYPOINT: re-targets a new
    random point within [-half, half]^2 every ``replan_period_sec``, moving
    at constant ``speed_mps`` toward the current target."""

    x: float
    y: float
    speed_mps: float
    half_extent_m: float
    replan_period_sec: float = 4.0
    seed: int = 0
    _target: Optional[tuple] = None
    _t_since_replan: float = 0.0
    _rng: Optional[np.random.RandomState] = None

    def __post_init__(self):
        if self._rng is None:
            self._rng = np.random.RandomState(self.seed)
        if self._target is None:
            self._target = self._sample_target()

    def _sample_target(self):
        return (
            float(self._rng.uniform(-self.half_extent_m, self.half_extent_m)),
            float(self._rng.uniform(-self.half_extent_m, self.half_extent_m)),
        )

    def tick(self, dt_sec: float) -> tuple:
        self._t_since_replan += dt_sec
        if self._t_since_replan >= self.replan_period_sec:
            self._target = self._sample_target()
            self._t_since_replan = 0.0
        tx, ty = self._target
        dx, dy = tx - self.x, ty - self.y
        dist = math.hypot(dx, dy)
        step = self.speed_mps * dt_sec
        if dist > 1e-6:
            self.x += dx / dist * min(step, dist)
            self.y += dy / dist * min(step, dist)
        return self.x, self.y
