"""World-boundary risk contribution (code review P0-2): the procedural
training world is a virtual ``[-half, half]^2`` square (``scenario.world_size_m``)
with NO physical Gazebo wall geometry -- obstacle markers are spawned into an
otherwise open world (see ``env/scenarios/procedural_generator.py``), so
driving past this boundary is invisible to LiDAR-based collision detection
and to every obstacle-based clearance/TTC/risk function in this package
UNLESS explicitly modelled here. Without this module, a trajectory that
drives straight into a wall silently scored ``risk_target=0`` (no obstacle
was ever in front of it) -- see ``tests/test_boundary_risk.py``'s
regression test for exactly that scenario.

Modelled as 4 axis-aligned half-planes in WORLD frame. Candidate rollouts
(``dynamics/ackermann_rollout.Rollout``) are produced in the robot-LOCAL
frame (origin = robot's current pose, x-forward), so every rollout point is
first rotated+translated back to world frame via the robot's current WORLD
pose before being compared against the boundary -- mirroring
``env/simulation/privileged_snapshot.py``'s world-to-local conversion for
obstacles, just inverted.
"""

from __future__ import annotations

import math
from typing import Iterator, List, Optional, Tuple

from hunter_kinodynamic_rl.common.geometry import to_world_frame
from hunter_kinodynamic_rl.dynamics.ackermann_rollout import Rollout

RobotPose = Tuple[float, float, float]  # (x, y, yaw), WORLD frame


def distance_to_boundary_m(world_x: float, world_y: float, half_extent_m: float) -> float:
    """Distance from a WORLD-frame point to the NEAREST of the 4 walls of
    the ``[-half, half]^2`` square (ego radius NOT subtracted here -- see
    ``boundary_clearance_at`` for the ego-radius-aware version). Negative
    once the point is outside the square (not clamped at 0), so callers can
    distinguish "grazing the wall" from "already past it" -- the same
    convention ``future_clearance.clearance_at`` uses for obstacles."""
    return half_extent_m - max(abs(world_x), abs(world_y))


def boundary_clearance_at(
    local_x: float, local_y: float, ego_radius: float, robot_pose: RobotPose, half_extent_m: float,
) -> float:
    rx, ry, ryaw = robot_pose
    dx, dy = to_world_frame(local_x, local_y, ryaw)
    return distance_to_boundary_m(rx + dx, ry + dy, half_extent_m) - ego_radius


def min_boundary_clearance(
    rollout: Rollout, ego_radius: float, robot_pose: RobotPose, half_extent_m: float,
    *, horizon_sec: Optional[float] = None, check_t0: bool = True,
) -> float:
    """Mirrors ``future_clearance.min_clearance``'s contract: ``+inf`` when
    there is nothing to check. section P0-2: ``check_t0`` (default True --
    every rollout in this codebase starts at the robot-LOCAL origin ``(0,
    0)`` by construction, see ``dynamics/ackermann_rollout.py``, so this
    needs no caller-supplied state) additionally checks the CURRENT instant
    (t=0), not just future rollout points; ``horizon_sec`` (default
    ``None`` -- the whole rollout) clips to only points with ``t_sec <=
    horizon_sec``, so a rollout that extends past the caller's intended
    horizon can't report a clearance value from beyond it."""
    best = math.inf
    if check_t0:
        best = min(best, boundary_clearance_at(0.0, 0.0, ego_radius, robot_pose, half_extent_m))
    for p in rollout.points:
        if horizon_sec is not None and p.t_sec > horizon_sec:
            continue
        c = boundary_clearance_at(p.state.x, p.state.y, ego_radius, robot_pose, half_extent_m)
        if c < best:
            best = c
    return best


def boundary_time_to_exit_or_none(
    rollout: Rollout, ego_radius: float, robot_pose: RobotPose, half_extent_m: float, horizon_sec: float,
    *, check_t0: bool = True,
) -> Optional[float]:
    """section P0-2: the unambiguous primitive -- ``None`` means "never
    exits within ``[0, horizon_sec]``", distinct from "exits exactly at
    ``horizon_sec``" (both of which ``boundary_time_to_exit`` alone would
    report as the same clamped value). See ``ttc.time_to_collision_or_none``
    for the identical rationale."""
    if check_t0 and boundary_clearance_at(0.0, 0.0, ego_radius, robot_pose, half_extent_m) <= 0.0:
        return 0.0
    for p in rollout.points:
        if p.t_sec > horizon_sec:
            continue
        if boundary_clearance_at(p.state.x, p.state.y, ego_radius, robot_pose, half_extent_m) <= 0.0:
            return p.t_sec
    return None


def boundary_time_to_exit(
    rollout: Rollout, ego_radius: float, robot_pose: RobotPose, half_extent_m: float, horizon_sec: float,
    *, check_t0: bool = True,
) -> float:
    """Mirrors ``ttc.time_to_collision``'s contract: first ``t_sec`` at
    which the robot's footprint reaches/crosses a wall, else ``horizon_sec``
    (never infinite -- a finite, comparable number for every candidate).
    Ambiguous at the exact-horizon boundary by construction (see this
    module's docstring pattern in ``ttc.py``) -- use
    :func:`boundary_time_to_exit_or_none` when the boolean distinction
    matters."""
    t = boundary_time_to_exit_or_none(rollout, ego_radius, robot_pose, half_extent_m, horizon_sec, check_t0=check_t0)
    return horizon_sec if t is None else t


# ---------------------------------------------------------------- item-6: continuous
def _world_timed_points(
    rollout: Rollout, robot_pose: RobotPose, check_t0: bool,
) -> List[Tuple[float, float, float]]:
    rx, ry, ryaw = robot_pose
    pts: List[Tuple[float, float, float]] = []
    if check_t0:
        pts.append((0.0, rx, ry))  # rollout t=0 is the robot-local origin, by construction
    for p in rollout.points:
        dx, dy = to_world_frame(p.state.x, p.state.y, ryaw)
        pts.append((p.t_sec, rx + dx, ry + dy))
    return pts


def _world_segments(
    rollout: Rollout, robot_pose: RobotPose, horizon_sec: Optional[float], check_t0: bool,
) -> Iterator[Tuple[float, float, float, float, float, float]]:
    """Same clip-at-horizon contract as ``future_clearance._segments``, in
    WORLD frame."""
    pts = _world_timed_points(rollout, robot_pose, check_t0)
    for (t1, x1, y1), (t2, x2, y2) in zip(pts, pts[1:]):
        if horizon_sec is not None and t1 > horizon_sec:
            continue
        if horizon_sec is not None and t2 > horizon_sec:
            span = t2 - t1
            frac = 1.0 if span <= 0.0 else (horizon_sec - t1) / span
            x2 = x1 + frac * (x2 - x1)
            y2 = y1 + frac * (y2 - y1)
            t2 = horizon_sec
        yield t1, x1, y1, t2, x2, y2


def _boundary_clearance_world_xy(world_x: float, world_y: float, ego_radius: float, half_extent_m: float) -> float:
    return distance_to_boundary_m(world_x, world_y, half_extent_m) - ego_radius


def _segment_zero_crossing_breakpoints(x0: float, y0: float, x1: float, y1: float) -> List[float]:
    """``distance_to_boundary_m(x(s), y(s))`` is piecewise-LINEAR in ``s``
    (``x(s)``/``y(s)`` affine, ``max(|x|, |y|)`` of two affine functions is
    piecewise-linear) -- its minimum over ``[0, 1]`` can only occur at
    ``s=0``, ``s=1``, or a breakpoint where ``x(s)=0`` or ``y(s)=0`` (where
    ``|x(s)|`` or ``|y(s)|`` itself has a kink). Returns those interior
    breakpoints (each in ``(0, 1)``), sorted, endpoints not included."""
    breakpoints = []
    dx, dy = x1 - x0, y1 - y0
    if abs(dx) > 1e-12:
        s = -x0 / dx
        if 0.0 < s < 1.0:
            breakpoints.append(s)
    if abs(dy) > 1e-12:
        s = -y0 / dy
        if 0.0 < s < 1.0:
            breakpoints.append(s)
    return sorted(breakpoints)


def min_boundary_clearance_continuous(
    rollout: Rollout, ego_radius: float, robot_pose: RobotPose, half_extent_m: float,
    *, horizon_sec: Optional[float] = None, check_t0: bool = True,
) -> float:
    """section item-6: the CONTINUOUS counterpart of
    :func:`min_boundary_clearance` -- finds the EXACT minimum boundary
    clearance anywhere along each continuous segment (not just at its
    sampled endpoints), via the piecewise-linear breakpoint analysis in
    :func:`_segment_zero_crossing_breakpoints`. Strictly <= the discrete
    version's result; ``+inf`` when there is nothing to check (mirrors
    :func:`min_boundary_clearance`)."""
    pts = _world_timed_points(rollout, robot_pose, check_t0)
    if not pts:
        return math.inf
    best = math.inf
    t1_first, x1_first, y1_first = pts[0]
    if horizon_sec is None or t1_first <= horizon_sec:
        best = min(best, _boundary_clearance_world_xy(x1_first, y1_first, ego_radius, half_extent_m))
    for _t1, x1, y1, _t2, x2, y2 in _world_segments(rollout, robot_pose, horizon_sec, check_t0):
        candidate_s = [0.0, 1.0] + _segment_zero_crossing_breakpoints(x1, y1, x2, y2)
        dx, dy = x2 - x1, y2 - y1
        for s in candidate_s:
            x, y = x1 + s * dx, y1 + s * dy
            c = _boundary_clearance_world_xy(x, y, ego_radius, half_extent_m)
            if c < best:
                best = c
    return best


def boundary_time_to_exit_or_none_continuous(
    rollout: Rollout, ego_radius: float, robot_pose: RobotPose, half_extent_m: float, horizon_sec: float,
    *, check_t0: bool = True,
) -> Optional[float]:
    """section item-6: the CONTINUOUS counterpart of
    :func:`boundary_time_to_exit_or_none` -- finds the EARLIEST time
    anywhere along the continuous (piecewise-linear-in-clearance) path at
    which the boundary clearance reaches <= 0, via exact linear
    interpolation within whichever piece it first crosses in."""
    pts = _world_timed_points(rollout, robot_pose, check_t0)
    if not pts:
        return None
    t1_first, x1_first, y1_first = pts[0]
    if horizon_sec is None or t1_first <= horizon_sec:
        if _boundary_clearance_world_xy(x1_first, y1_first, ego_radius, half_extent_m) <= 0.0:
            return t1_first
    for t1, x1, y1, t2, x2, y2 in _world_segments(rollout, robot_pose, horizon_sec, check_t0):
        dx, dy = x2 - x1, y2 - y1
        pieces_s = [0.0] + _segment_zero_crossing_breakpoints(x1, y1, x2, y2) + [1.0]
        for sa, sb in zip(pieces_s, pieces_s[1:]):
            xa, ya = x1 + sa * dx, y1 + sa * dy
            xb, yb = x1 + sb * dx, y1 + sb * dy
            ca = _boundary_clearance_world_xy(xa, ya, ego_radius, half_extent_m)
            cb = _boundary_clearance_world_xy(xb, yb, ego_radius, half_extent_m)
            if ca <= 0.0:
                return t1 + sa * (t2 - t1)
            if cb <= 0.0:
                frac = ca / (ca - cb) if ca != cb else 0.0
                s_hit = sa + frac * (sb - sa)
                return t1 + s_hit * (t2 - t1)
    return None


def boundary_time_to_exit_continuous(
    rollout: Rollout, ego_radius: float, robot_pose: RobotPose, half_extent_m: float, horizon_sec: float,
    *, check_t0: bool = True,
) -> float:
    """Mirrors :func:`boundary_time_to_exit`'s clamped-at-``horizon_sec``
    contract, backed by the continuous check."""
    t = boundary_time_to_exit_or_none_continuous(
        rollout, ego_radius, robot_pose, half_extent_m, horizon_sec, check_t0=check_t0)
    return horizon_sec if t is None else t
