"""Generic moving-obstacle motion patterns (section 27 -- NOT pedestrian-
specific; a ``GenericMovingObstacle`` abstraction swappable for a HuNav
pedestrian or another robot later).

Legacy patterns plus opt-in v2 acceleration/turning patterns, deterministic
given a seed:
  crossing            -- perpendicular to the robot's start->goal line
  head_on             -- anti-parallel to the robot's start->goal direction
  cut_in              -- starts ahead and to one side, converges INTO the path
  parallel            -- parallel to start->goal, offset laterally (no crossing)
  random_waypoint     -- periodically re-targets a new random point (stateful)
  constant_velocity   -- the DynamicObstacleSpec's own (vx, vy), unchanged
  stop_go / accelerate_decelerate -- bounded longitudinal transients
  sine_swerve / bounce / u_turn   -- bounded non-linear heading changes

``crossing``/``head_on``/``cut_in``/``parallel`` are constant-velocity for
their whole episode (the geometry is baked into the initial (vx, vy) once);
Legacy v1 keeps that exact behavior through :class:`RandomWaypointState`;
v2 routes all patterns through :class:`KinematicMotionState`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
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
    STOP_GO = "stop_go"
    ACCELERATE_DECELERATE = "accelerate_decelerate"
    SINE_SWERVE = "sine_swerve"
    BOUNCE = "bounce"
    U_TURN = "u_turn"


# Frozen v1 selection set. Adding the v2 enum values to this tuple would
# silently change the seeded pattern assigned to every legacy procedural
# obstacle, so v2 patterns are selected only by tractor_environment_v2.py.
_LEGACY_PATTERNS = (
    MotionPattern.CROSSING, MotionPattern.HEAD_ON, MotionPattern.CUT_IN,
    MotionPattern.PARALLEL, MotionPattern.RANDOM_WAYPOINT, MotionPattern.CONSTANT_VELOCITY,
)


def assign_pattern(seed: int, index: int) -> MotionPattern:
    """Deterministic per-obstacle pattern choice from (episode seed, obstacle index)."""
    rng = np.random.RandomState((seed * 1000003 + index) & 0xFFFFFFFF)
    return _LEGACY_PATTERNS[rng.randint(0, len(_LEGACY_PATTERNS))]


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
    return replace(spec, vx=vx, vy=vy, yaw_rad=vh)


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

    def current_velocity(self) -> tuple:
        """Ground-truth velocity implied by the current target and state."""
        tx, ty = self._target
        dx, dy = tx - self.x, ty - self.y
        distance = math.hypot(dx, dy)
        if distance <= 1e-9:
            return 0.0, 0.0
        return self.speed_mps * dx / distance, self.speed_mps * dy / distance

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


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class KinematicMotionState:
    """Acceleration- and turn-rate-limited v2 obstacle motion.

    This state is used only by ``tractor_env_v2``.  It keeps the old
    constant-velocity and RandomWaypoint implementations untouched for v1.
    """

    x: float
    y: float
    vx: float
    vy: float
    yaw: float
    nominal_speed_mps: float
    nominal_heading_rad: float
    pattern: MotionPattern
    interaction_mode: str
    accel_limit_mps2: float
    turn_rate_rad_s: float
    half_extent_m: float
    radius_m: float
    seed: int = 0
    elapsed_sec: float = 0.0
    _target: Optional[tuple] = None
    _rng: Optional[np.random.RandomState] = None

    @classmethod
    def from_spec(cls, spec: DynamicObstacleSpec, half_extent_m: float, seed: int):
        speed = math.hypot(spec.vx, spec.vy)
        heading = math.atan2(spec.vy, spec.vx) if speed > 1e-9 else spec.yaw_rad
        return cls(
            x=spec.x0, y=spec.y0, vx=spec.vx, vy=spec.vy, yaw=heading,
            nominal_speed_mps=speed, nominal_heading_rad=heading,
            pattern=MotionPattern(spec.motion_pattern or MotionPattern.CONSTANT_VELOCITY.value),
            interaction_mode=spec.interaction_mode,
            accel_limit_mps2=max(spec.accel_limit_mps2, 1e-6),
            turn_rate_rad_s=max(spec.turn_rate_rad_s, 1e-6),
            half_extent_m=half_extent_m, radius_m=spec.radius, seed=seed,
        )

    def __post_init__(self):
        if self._rng is None:
            self._rng = np.random.RandomState(self.seed)

    def _random_waypoint_heading(self) -> float:
        if self._target is None or math.hypot(self._target[0] - self.x, self._target[1] - self.y) < 0.25:
            inset = max(0.0, self.half_extent_m - self.radius_m)
            self._target = (
                float(self._rng.uniform(-inset, inset)),
                float(self._rng.uniform(-inset, inset)),
            )
        return math.atan2(self._target[1] - self.y, self._target[0] - self.x)

    def _desired_motion(self, ego_xy: Optional[tuple]) -> tuple:
        speed = self.nominal_speed_mps
        heading = self.nominal_heading_rad
        if self.pattern == MotionPattern.STOP_GO:
            speed = self.nominal_speed_mps if (self.elapsed_sec % 4.0) < 2.4 else 0.0
        elif self.pattern == MotionPattern.ACCELERATE_DECELERATE:
            speed = self.nominal_speed_mps * (0.55 + 0.45 * math.sin(1.1 * self.elapsed_sec))
        elif self.pattern == MotionPattern.SINE_SWERVE:
            heading += 0.55 * math.sin(0.9 * self.elapsed_sec + (self.seed % 17) * 0.1)
        elif self.pattern == MotionPattern.U_TURN:
            turn_progress = max(0.0, self.elapsed_sec - 1.5)
            heading += min(math.pi, self.turn_rate_rad_s * turn_progress)
        elif self.pattern == MotionPattern.RANDOM_WAYPOINT:
            heading = self._random_waypoint_heading()

        if ego_xy is not None and self.interaction_mode in ("yielding", "reciprocal"):
            dx, dy = ego_xy[0] - self.x, ego_xy[1] - self.y
            distance = math.hypot(dx, dy)
            if distance < 2.2:
                if self.interaction_mode == "yielding":
                    speed *= max(0.0, min(1.0, (distance - 0.75) / 1.45))
                else:
                    away = math.atan2(-dy, -dx)
                    blend = max(0.0, min(1.0, (2.2 - distance) / 1.45))
                    heading = _wrap_angle(heading + blend * _wrap_angle(away - heading))
        return max(0.0, speed), _wrap_angle(heading)

    def tick(self, dt_sec: float, ego_xy: Optional[tuple] = None) -> tuple:
        if not math.isfinite(dt_sec) or dt_sec <= 0.0:
            raise ValueError("KinematicMotionState.tick requires finite dt_sec > 0")
        self.elapsed_sec += dt_sec
        desired_speed, desired_heading = self._desired_motion(ego_xy)
        current_speed = math.hypot(self.vx, self.vy)
        max_dv = self.accel_limit_mps2 * dt_sec
        speed = current_speed + max(-max_dv, min(max_dv, desired_speed - current_speed))
        max_dyaw = self.turn_rate_rad_s * dt_sec
        yaw_error = _wrap_angle(desired_heading - self.yaw)
        self.yaw = _wrap_angle(self.yaw + max(-max_dyaw, min(max_dyaw, yaw_error)))
        self.vx, self.vy = speed * math.cos(self.yaw), speed * math.sin(self.yaw)
        self.x += self.vx * dt_sec
        self.y += self.vy * dt_sec

        inset = max(0.0, self.half_extent_m - self.radius_m)
        hit_x = self.x < -inset or self.x > inset
        hit_y = self.y < -inset or self.y > inset
        if hit_x:
            self.x = max(-inset, min(inset, self.x))
            self.vx = -self.vx
        if hit_y:
            self.y = max(-inset, min(inset, self.y))
            self.vy = -self.vy
        if hit_x or hit_y:
            self.yaw = math.atan2(self.vy, self.vx)
            self.nominal_heading_rad = self.yaw
            if self.pattern == MotionPattern.RANDOM_WAYPOINT:
                self._target = None
        return self.x, self.y, self.vx, self.vy, self.yaw


def realized_velocity(previous_xy, current_xy, dt_sec: float) -> tuple:
    """Compute a label from realized displacement and measured time."""
    if not math.isfinite(dt_sec) or dt_sec <= 0.0:
        raise ValueError("realized velocity requires a finite positive dt")
    px, py = previous_xy
    cx, cy = current_xy
    if not all(math.isfinite(value) for value in (px, py, cx, cy)):
        raise ValueError("realized velocity positions must be finite")
    return (cx - px) / dt_sec, (cy - py) / dt_sec


def motion_substep_durations(total_dt_sec: float, count: int) -> tuple:
    """Split one control interval without changing its total duration."""
    if not math.isfinite(total_dt_sec) or total_dt_sec <= 0.0:
        raise ValueError("total_dt_sec must be finite and > 0")
    if count < 1:
        raise ValueError("motion substep count must be >= 1")
    dt = total_dt_sec / count
    return tuple(dt for _ in range(count))
