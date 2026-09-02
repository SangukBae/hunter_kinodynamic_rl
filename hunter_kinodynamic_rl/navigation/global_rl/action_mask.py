"""Global candidate action mask (plan section 8.4).

Invalid conditions, checked against the CURRENT online :class:`PartialMap`
only (never simulator ground truth -- plan section 3.2's information
boundary applies to the action mask exactly as much as to the observation):

- the DECLARED endpoint (the candidate's literal robot-relative
  ``(radius, angle)`` point) lies in a known occupied/inflated cell
- the required curvature to reach that endpoint via a single constant-
  curvature arc EXCEEDS the vehicle's own turning limit -- i.e. no
  feasible single-arc Ackermann rollout reaches the endpoint at all (see
  :func:`_ackermann_feasible_arc_points_robot_frame`'s docstring for why
  this must be an outright rejection rather than a clamp-and-truncate:
  clamping the curvature changes WHERE the arc actually ends, silently
  under-checking the segment between the clamped arc's short end and the
  candidate's real declared endpoint)
- the FULL feasible rollout -- swept all the way to the candidate's
  endpoint, never truncated at an arbitrary sample budget -- crosses a
  known occupied/inflated cell. The curve is sampled adaptively from map
  resolution and each adjacent sample pair is grid-traced, so long arcs do
  not skip between sparse fixed sample points.
- non-finite (NaN/Inf) or non-positive candidate geometry
- localization or map state is invalid
- the robot's OWN current footprint is already in collision

An UNKNOWN endpoint (never observed) is NOT one of these conditions -- it
stays valid (exploration target), exactly matching the plan's explicit
"UNKNOWN이라는 이유만으로 invalid 처리하지 말 것". The fallback candidate
(``candidates[-1].is_fallback``) is ALWAYS valid, unconditionally -- it is
excluded from every check above.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from hunter_kinodynamic_rl.config.schema import GlobalRLConfig
from hunter_kinodynamic_rl.navigation.global_rl.subgoal_sampler import SubgoalCandidate, candidate_endpoint_mission
from hunter_kinodynamic_rl.navigation.mapping.partial_map import PartialMap
from hunter_kinodynamic_rl.navigation.mapping.raytracing import trace_clipped
from hunter_kinodynamic_rl.navigation.mapping.visited_map import rasterize_circle_cells
from hunter_kinodynamic_rl.navigation.mission.mission_frame import MissionFrame, PoseXYYaw

# Below this, sin(angle_rad) is treated as exactly zero -- distinguishes
# "straight ahead" (cos > 0) from "directly behind" (cos < 0); see
# _ackermann_feasible_arc_points_robot_frame's docstring.
_SIN_NEAR_ZERO = 1e-9


def _robot_footprint_in_collision(partial_map: PartialMap, inflated: np.ndarray, robot_pose_mission: PoseXYYaw,
                                   footprint_radius_m: float) -> bool:
    center = partial_map.world_to_cell(robot_pose_mission.x, robot_pose_mission.y)
    if center is None:
        # The robot's own position is outside the accumulated map entirely
        # -- nothing known-occupied there, so this specific check cannot
        # fire (it is not evidence of a collision).
        return False
    radius_cells = footprint_radius_m / partial_map.resolution_m
    for r, c in rasterize_circle_cells(center[0], center[1], radius_cells):
        if partial_map.in_bounds(r, c) and inflated[r, c]:
            return True
    return False


def _ackermann_feasible_arc_points_robot_frame(
    candidate: SubgoalCandidate, max_curvature: float, sample_count: int, max_step_m: float,
) -> Optional[List[Tuple[float, float]]]:
    """Robot-body-frame points (x forward, y left) along the constant-
    curvature arc a real Ackermann vehicle -- starting at the origin,
    heading along +x -- must sweep to reach ``candidate``'s declared
    endpoint EXACTLY (never truncated at the candidate's straight-line
    ``radius_m`` when the actual arc needs more distance than that to get
    there), or ``None`` when no single feasible forward arc reaches it at
    all.

    Geometry: for a candidate at polar angle ``phi = candidate.angle_rad``
    and distance ``r = candidate.radius_m``, the standard pure-pursuit
    curvature of the circle from the origin (heading 0) through that point
    is ``kappa = 2*sin(phi)/r`` (equivalent to the usual
    ``2*dy/(dx^2+dy^2)`` form since ``dx=r*cos(phi)``, ``dy=r*sin(phi)``,
    ``dx^2+dy^2=r^2``). The chord from the origin to the point subtends an
    angle ``theta = 2*phi`` at the circle's center (a standard circular-arc
    identity -- the tangent at the start makes angle ``theta/2`` with the
    chord), so the ARC LENGTH actually needed to travel from the origin to
    the point along that circle is ``s = theta / kappa = r * phi / sin(phi)``
    -- NOT ``r`` itself (the straight-line chord length) unless ``phi`` is
    already 0. A previous version of this function used the unclamped
    curvature but only sampled out to ``r``, silently leaving the arc's
    true remaining segment (from ``r`` out to the real ``s``) unchecked --
    confirmed exploitable: a 90-degree/6m candidate needs ``s ~= 9.42 m``
    of arc to actually reach its endpoint, so a known obstacle placed
    between 6m and 9.42m along the real arc passed the mask as valid.

    Two cases are rejected outright (returns ``None``) rather than clamped:

    - ``abs(kappa) > max_curvature``: reaching the endpoint via ONE arc
      requires a tighter turn than the vehicle can make. Clamping
      ``kappa`` down to ``max_curvature`` would sweep a DIFFERENT, shorter
      arc that never actually reaches the declared endpoint either --
      exactly the same silently-incomplete-rollout bug this function
      exists to fix, just introduced a different way. Rejecting is the
      conservative, physically honest answer: this candidate is not
      reachable via a single feasible arc, full stop.
    - ``sin(phi)`` within :data:`_SIN_NEAR_ZERO` of 0 AND ``cos(phi) < 0``
      (the candidate is directly, or almost directly, BEHIND the robot):
      ``kappa -> 0`` in this limit, but the arc length needed to reach it,
      ``s = r*phi/sin(phi)``, diverges to infinity (traveling forward along
      an ever-flatter, ever-longer loop to come back around to a point
      behind the start) -- geometrically, no FORWARD arc reaches directly
      behind the vehicle at all; reaching it needs reversing or a
      multi-maneuver path, which is out of scope for a single short-arc
      rollout check. (``sin(phi) ~ 0`` with ``cos(phi) > 0`` -- straight
      ahead -- is the well-behaved ``kappa=0``, ``s=r`` limit instead.)
    """
    phi = candidate.angle_rad
    r = candidate.radius_m
    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)

    if abs(sin_phi) < _SIN_NEAR_ZERO:
        if cos_phi < 0.0:
            return None  # directly behind -- unreachable via any forward arc
        kappa = 0.0
        arc_length = r
    else:
        kappa = (2.0 / r) * sin_phi
        if max_curvature <= 0.0 or abs(kappa) > max_curvature:
            return None
        arc_length = r * phi / sin_phi

    if not (math.isfinite(arc_length) and arc_length > 0.0):
        return None

    # sample_count is the profile-authored minimum. Long arcs get additional
    # samples so adjacent points stay below the map-resolution-derived step;
    # the grid trace in _rollout_hits_inflated then checks cells between
    # consecutive samples as well.
    effective_sample_count = max(int(sample_count), 2)
    if math.isfinite(max_step_m) and max_step_m > 0.0:
        effective_sample_count = max(effective_sample_count, int(math.ceil(arc_length / max_step_m)) + 1)

    points: List[Tuple[float, float]] = []
    for t in np.linspace(0.0, 1.0, effective_sample_count):
        s = t * arc_length
        if abs(kappa) < 1e-12:
            points.append((s, 0.0))
        else:
            points.append((math.sin(kappa * s) / kappa, (1.0 - math.cos(kappa * s)) / kappa))
    return points


def _world_to_unbounded_cell(partial_map: PartialMap, x: float, y: float) -> Tuple[int, int]:
    col = int(np.floor((x - partial_map.origin_x) / partial_map.resolution_m))
    row = int(np.floor((y - partial_map.origin_y) / partial_map.resolution_m))
    return row, col


def _rollout_hits_inflated(
    partial_map: PartialMap, inflated: np.ndarray, robot_pose_mission: PoseXYYaw,
    points_robot_frame: List[Tuple[float, float]],
) -> bool:
    prev_cell: Optional[Tuple[int, int]] = None
    for point_robot in points_robot_frame:
        x, y = MissionFrame.robot_to_mission(point_robot, robot_pose_mission)
        cell = _world_to_unbounded_cell(partial_map, x, y)
        if prev_cell is None:
            cells = trace_clipped(cell[0], cell[1], cell[0], cell[1], partial_map.size_cells, partial_map.size_cells)
        else:
            cells = trace_clipped(
                prev_cell[0], prev_cell[1], cell[0], cell[1], partial_map.size_cells, partial_map.size_cells,
            )
        for rr, cc in cells:
            if inflated[rr, cc]:
                return True
        prev_cell = cell
    return False


def compute_action_mask(
    candidates: Sequence[SubgoalCandidate], partial_map: PartialMap, robot_pose_mission: PoseXYYaw,
    config: GlobalRLConfig, robot_max_curvature: float, *, localization_valid: bool = True, map_valid: bool = True,
) -> np.ndarray:
    """Returns a ``bool`` array, one entry per candidate, in the SAME order
    as ``candidates`` (see :func:`subgoal_sampler.build_candidate_set``).

    ``robot_max_curvature`` (1/m, i.e. ``RobotConfig.max_curvature`` --
    ``tan(steering_limit)/wheelbase``) is REQUIRED, not defaulted: silently
    falling back to "no turning limit" would defeat the whole point of the
    Ackermann rollout check below."""
    n = len(candidates)
    mask = np.zeros(n, dtype=bool)
    channels = partial_map.channels()
    inflated = channels.inflated

    globally_blocked = (not localization_valid) or (not map_valid) or _robot_footprint_in_collision(
        partial_map, inflated, robot_pose_mission, config.robot_footprint_radius_m,
    )

    for candidate in candidates:
        if candidate.is_fallback:
            mask[candidate.index] = True
            continue
        if globally_blocked:
            mask[candidate.index] = False
            continue
        if not (math.isfinite(candidate.radius_m) and math.isfinite(candidate.angle_rad)):
            mask[candidate.index] = False
            continue
        if candidate.radius_m <= 0.0:
            mask[candidate.index] = False
            continue
        endpoint_mission = candidate_endpoint_mission(candidate, robot_pose_mission)
        if not (math.isfinite(endpoint_mission[0]) and math.isfinite(endpoint_mission[1])):
            mask[candidate.index] = False
            continue
        endpoint_cell = partial_map.world_to_cell(*endpoint_mission)
        if endpoint_cell is not None and inflated[endpoint_cell]:
            mask[candidate.index] = False
            continue
        arc_points = _ackermann_feasible_arc_points_robot_frame(
            candidate, robot_max_curvature, config.rollout_sample_count, partial_map.resolution_m * 0.5,
        )
        if arc_points is None:
            mask[candidate.index] = False
            continue
        mask[candidate.index] = not _rollout_hits_inflated(partial_map, inflated, robot_pose_mission, arc_points)
    return mask
