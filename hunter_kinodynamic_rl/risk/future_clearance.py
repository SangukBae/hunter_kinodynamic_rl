"""Minimum future clearance between an ego rollout and a set of (privileged)
dynamic obstacles -- the base signal collision/TTC/risk-score all build on.

section P0-2 (clearance/TTC correctness): a :class:`Rollout`'s own points
are, by construction (see ``dynamics/ackermann_rollout.py``'s
``rollout_constant_target`` docstring), samples at ``t in (dt, 2*dt, ...,
horizon_sec]`` -- they NEVER include ``t=0`` (the current, pre-action
instant). Checking ONLY ``rollout.points`` therefore cannot detect an
ALREADY-existing overlap at the current instant (e.g. an obstacle that has
already drifted into contact, or a bad reset) -- it would only be caught
once the FIRST future sample also happens to still show it, which is not
guaranteed. Passing ``t0_state`` (the exact :class:`VehicleState` the
rollout was generated FROM) closes this: every function below checks it
FIRST, at ``t_sec=0.0``, before considering any rollout point. Optional
(default ``None``, preserving the exact prior rollout-only behavior) purely
so existing callers/tests that don't have a convenient ``VehicleState`` on
hand keep working unchanged; every PRODUCTION call site
(``risk/trajectory_risk.py``) passes it.
"""

from __future__ import annotations

import math
from typing import Iterator, List, Optional, Sequence, Tuple

from hunter_kinodynamic_rl.dynamics.ackermann_rollout import Rollout
from hunter_kinodynamic_rl.risk import segment_math
from hunter_kinodynamic_rl.risk.labels import DynamicObstacle
from hunter_kinodynamic_rl.robot.interface import VehicleState


def clearance_at(ego_x: float, ego_y: float, ego_radius: float, obstacle: DynamicObstacle, t_sec: float) -> float:
    ox, oy = obstacle.position_at(t_sec)
    center_dist = math.hypot(ego_x - ox, ego_y - oy)
    return center_dist - ego_radius - obstacle.radius


def min_clearance(
    rollout: Rollout, ego_radius: float, obstacles: Sequence[DynamicObstacle],
    *, horizon_sec: Optional[float] = None, t0_state: Optional[VehicleState] = None,
) -> float:
    """Minimum clearance against ALL obstacles, over ``t=0`` (if
    ``t0_state`` given) plus every rollout point with ``t_sec <=
    horizon_sec`` (the WHOLE rollout if ``horizon_sec`` is ``None`` --
    preserves prior behavior for callers that don't clip to a horizon
    shorter than the rollout itself). ``+inf`` when there are no obstacles,
    or nothing to check (no ``t0_state`` and no in-horizon rollout points)
    -- callers treat that as "no constraint", not "perfectly safe at
    distance 0"."""
    if not obstacles:
        return math.inf
    best = math.inf
    if t0_state is not None:
        for obstacle in obstacles:
            c = clearance_at(t0_state.x, t0_state.y, ego_radius, obstacle, 0.0)
            if c < best:
                best = c
    for point in rollout.points:
        if horizon_sec is not None and point.t_sec > horizon_sec:
            continue
        for obstacle in obstacles:
            c = clearance_at(point.state.x, point.state.y, ego_radius, obstacle, point.t_sec)
            if c < best:
                best = c
    return best


def _timed_points(rollout: Rollout, t0_state: Optional[VehicleState]) -> List[Tuple[float, float, float]]:
    """The ordered ``(t_sec, x, y)`` sequence continuous checks walk
    consecutive PAIRS of -- ``t0_state`` (if given) first, then every
    rollout point in order."""
    pts: List[Tuple[float, float, float]] = []
    if t0_state is not None:
        pts.append((0.0, t0_state.x, t0_state.y))
    pts.extend((p.t_sec, p.state.x, p.state.y) for p in rollout.points)
    return pts


def _segments(
    rollout: Rollout, t0_state: Optional[VehicleState], horizon_sec: Optional[float],
) -> Iterator[Tuple[float, float, float, float, float, float]]:
    """Yields ``(t1, x1, y1, t2, x2, y2)`` for each CONSECUTIVE pair of
    points (t0_state -> first rollout point, then point -> point), clipped
    so that no segment extends past ``horizon_sec`` (a segment whose start
    is already past the horizon is dropped entirely; one that straddles the
    horizon is clipped to end exactly at it, via linear interpolation, so
    the horizon boundary itself is treated identically to the discrete
    functions' own ``t_sec <= horizon_sec`` semantics -- horizon-inclusive)."""
    pts = _timed_points(rollout, t0_state)
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


def min_clearance_continuous(
    rollout: Rollout, ego_radius: float, obstacles: Sequence[DynamicObstacle],
    *, horizon_sec: Optional[float] = None, t0_state: Optional[VehicleState] = None,
) -> float:
    """section item-6: the CONTINUOUS counterpart of :func:`min_clearance`
    -- instead of checking only ``t=0`` (if given) and each discrete
    rollout sample, walks every CONSECUTIVE PAIR of points as a segment and
    finds the EXACT closest approach to each obstacle anywhere along it
    (see ``segment_math.py``'s module docstring for the exact affine-motion
    math and its honesty caveat about linear-interpolating the ego's
    curved path between samples). Strictly <= the discrete
    :func:`min_clearance`'s result (checking a continuum can only find an
    equal-or-closer approach than checking its endpoints alone), t=0-
    inclusive and horizon-inclusive exactly like every other function in
    this module. ``+inf`` when there are no obstacles or no segments to
    check (mirrors :func:`min_clearance`)."""
    if not obstacles:
        return math.inf
    best = math.inf
    pts = _timed_points(rollout, t0_state)
    if not pts:
        return math.inf
    # The very first point itself (t=0 or the first rollout sample) must
    # still be checked even if there is no SEGMENT after it (a single-point
    # rollout) or before it (segments only cover pairs).
    t1_first, x1_first, y1_first = pts[0]
    if horizon_sec is None or t1_first <= horizon_sec:
        for obstacle in obstacles:
            c = clearance_at(x1_first, y1_first, ego_radius, obstacle, t1_first)
            if c < best:
                best = c
    for t1, x1, y1, t2, x2, y2 in _segments(rollout, t0_state, horizon_sec):
        for obstacle in obstacles:
            ox1, oy1 = obstacle.position_at(t1)
            ox2, oy2 = obstacle.position_at(t2)
            # Relative position P(s) = ego(s) - obstacle(s), affine in s
            # since both endpoints are exact/linearly-interpolated.
            _s, min_center_dist = segment_math.closest_approach_on_segment(
                x1 - ox1, y1 - oy1, x2 - ox2, y2 - oy2,
            )
            c = min_center_dist - ego_radius - obstacle.radius
            if c < best:
                best = c
    return best


def time_to_collision_or_none_continuous(
    rollout: Rollout, ego_radius: float, obstacles: Sequence[DynamicObstacle], horizon_sec: float,
    *, t0_state: Optional[VehicleState] = None,
) -> Optional[float]:
    """section item-6: the CONTINUOUS counterpart of
    :func:`~hunter_kinodynamic_rl.risk.ttc.time_to_collision_or_none` --
    finds the EARLIEST time anywhere along the continuous rollout (not just
    at a sample instant) at which any obstacle's clearance reaches <= 0.
    See :func:`min_clearance_continuous`'s docstring for the segment-
    interpolation rationale; t=0-inclusive and horizon-inclusive exactly
    like the discrete primitive."""
    if not obstacles:
        return None
    pts = _timed_points(rollout, t0_state)
    if not pts:
        return None
    t1_first, x1_first, y1_first = pts[0]
    if horizon_sec is None or t1_first <= horizon_sec:
        for obstacle in obstacles:
            if clearance_at(x1_first, y1_first, ego_radius, obstacle, t1_first) <= 0.0:
                return t1_first
    for t1, x1, y1, t2, x2, y2 in _segments(rollout, t0_state, horizon_sec):
        best_s_for_segment: Optional[float] = None
        for obstacle in obstacles:
            ox1, oy1 = obstacle.position_at(t1)
            ox2, oy2 = obstacle.position_at(t2)
            s_hit = segment_math.first_crossing_below_radius(
                x1 - ox1, y1 - oy1, x2 - ox2, y2 - oy2, ego_radius + obstacle.radius,
            )
            if s_hit is not None and (best_s_for_segment is None or s_hit < best_s_for_segment):
                best_s_for_segment = s_hit
        if best_s_for_segment is not None:
            return t1 + best_s_for_segment * (t2 - t1)
    return None


def clearance_timeseries(
    rollout: Rollout, ego_radius: float, obstacles: Sequence[DynamicObstacle],
    *, t0_state: Optional[VehicleState] = None,
) -> List[float]:
    """Per-sample minimum clearance against the closest obstacle -- used by
    trajectory_risk.py's aggregation and for plotting/analysis. Includes a
    leading ``t=0`` sample (against ``t0_state``) when provided, so the
    series' first entry is the CURRENT clearance, not the first future
    sample."""
    out = []
    if t0_state is not None:
        out.append(math.inf if not obstacles else min(
            clearance_at(t0_state.x, t0_state.y, ego_radius, obstacle, 0.0) for obstacle in obstacles
        ))
    for point in rollout.points:
        if not obstacles:
            out.append(math.inf)
            continue
        out.append(min(
            clearance_at(point.state.x, point.state.y, ego_radius, obstacle, point.t_sec)
            for obstacle in obstacles
        ))
    return out
