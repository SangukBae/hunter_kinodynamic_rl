"""Time-to-collision against a rolled-out ego trajectory + privileged
dynamic obstacles -- first sampled time at which clearance <= 0.

section P0-2 (clearance/TTC correctness): two related bugs fixed here.

1. Rollout points never include ``t=0`` (see future_clearance.py's module
   docstring) -- ``t0_state`` (optional, default ``None`` for backward
   compatibility) lets callers also check the CURRENT instant, so an
   already-existing overlap is detected immediately rather than only once
   a later sample happens to still show it.
2. The float-returning ``time_to_collision`` is fundamentally AMBIGUOUS at
   the boundary: "no collision found" and "collision found exactly at
   ``horizon_sec``" both return the same value (``horizon_sec``), so
   ``collision_within_horizon``'s old ``< horizon_sec`` comparison silently
   classified a genuine exactly-at-horizon collision as "no collision".
   ``time_to_collision_or_none`` (``None`` = no collision) is the
   unambiguous primitive every caller that needs the boolean should use
   directly; ``time_to_collision``/``collision_within_horizon`` are kept as
   thin, explicitly-documented compatibility wrappers for existing callers
   that only need one clamped float (e.g. TTC aux-supervision
   normalisation, which is fine with the ambiguity since it only consumes
   the number, never the boundary case)."""

from __future__ import annotations

from typing import Optional, Sequence

from hunter_kinodynamic_rl.dynamics.ackermann_rollout import Rollout
from hunter_kinodynamic_rl.risk.future_clearance import clearance_at, time_to_collision_or_none_continuous
from hunter_kinodynamic_rl.risk.labels import DynamicObstacle
from hunter_kinodynamic_rl.robot.interface import VehicleState


def time_to_collision_or_none(
    rollout: Rollout, ego_radius: float, obstacles: Sequence[DynamicObstacle], horizon_sec: float,
    *, t0_state: Optional[VehicleState] = None,
) -> Optional[float]:
    """The first ``t_sec`` in ``[0, horizon_sec]`` (INCLUSIVE of both
    endpoints) at which any obstacle's clearance drops to <= 0, or ``None``
    if no such time exists -- the only way to tell "no collision" and "a
    collision exactly at the horizon" apart, since both would otherwise
    collapse to the same sentinel value."""
    if not obstacles:
        return None
    if t0_state is not None:
        for obstacle in obstacles:
            if clearance_at(t0_state.x, t0_state.y, ego_radius, obstacle, 0.0) <= 0.0:
                return 0.0
    for point in rollout.points:
        if point.t_sec > horizon_sec:
            continue  # section P0-2: a rollout may extend past horizon_sec (e.g. L-derived > ttc_horizon_sec)
        for obstacle in obstacles:
            if clearance_at(point.state.x, point.state.y, ego_radius, obstacle, point.t_sec) <= 0.0:
                return point.t_sec
    return None


def time_to_collision(
    rollout: Rollout, ego_radius: float, obstacles: Sequence[DynamicObstacle], horizon_sec: float,
    *, t0_state: Optional[VehicleState] = None,
) -> float:
    """Returns the first ``t_sec`` at which any obstacle's clearance drops to
    <= 0, or ``horizon_sec`` if no collision is found within the rollout
    (i.e. TTC is clamped at the horizon, never reported as infinite -- a
    finite, comparable number for every candidate, matching
    aux_prediction's TTC-head normalisation convention in drl_agent). See
    this module's docstring: ambiguous at the exact-horizon boundary by
    construction -- use :func:`time_to_collision_or_none` when that
    distinction matters (e.g. the collision BOOLEAN, not just a numeric
    TTC value)."""
    t = time_to_collision_or_none(rollout, ego_radius, obstacles, horizon_sec, t0_state=t0_state)
    return horizon_sec if t is None else t


def collision_within_horizon(
    rollout: Rollout, ego_radius: float, obstacles: Sequence[DynamicObstacle], horizon_sec: float,
    *, t0_state: Optional[VehicleState] = None,
) -> bool:
    """section P0-2: derived from :func:`time_to_collision_or_none`'s
    None-vs-not-None distinction, NEVER from comparing
    :func:`time_to_collision`'s clamped float against ``horizon_sec`` (that
    comparison cannot distinguish "no collision" from "collision exactly at
    the horizon" -- both produce the same clamped value)."""
    return time_to_collision_or_none(rollout, ego_radius, obstacles, horizon_sec, t0_state=t0_state) is not None


def time_to_collision_continuous(
    rollout: Rollout, ego_radius: float, obstacles: Sequence[DynamicObstacle], horizon_sec: float,
    *, t0_state: Optional[VehicleState] = None,
) -> float:
    """section item-6: the CONTINUOUS (segment-based, not just sampled-
    endpoint) counterpart of :func:`time_to_collision` -- see
    ``future_clearance.time_to_collision_or_none_continuous``'s docstring.
    Same clamped-at-``horizon_sec`` convention as :func:`time_to_collision`
    (ambiguous at the exact-horizon boundary by the same construction; use
    :func:`collision_within_horizon_continuous` when that distinction
    matters)."""
    t = time_to_collision_or_none_continuous(rollout, ego_radius, obstacles, horizon_sec, t0_state=t0_state)
    return horizon_sec if t is None else t


def collision_within_horizon_continuous(
    rollout: Rollout, ego_radius: float, obstacles: Sequence[DynamicObstacle], horizon_sec: float,
    *, t0_state: Optional[VehicleState] = None,
) -> bool:
    """section item-6: the CONTINUOUS counterpart of
    :func:`collision_within_horizon` -- catches a genuine collision that
    occurs strictly BETWEEN two consecutive rollout samples (both
    individually clear), not just one that lands on a sampled instant."""
    return time_to_collision_or_none_continuous(
        rollout, ego_radius, obstacles, horizon_sec, t0_state=t0_state) is not None
