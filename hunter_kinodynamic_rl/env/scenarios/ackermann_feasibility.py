"""Ackermann-aware scenario feasibility gate (code review: the existing
``procedural_generator.is_reachable`` is a 4-connected occupancy-grid BFS --
it proves the free space is topologically connected but knows nothing about
heading, minimum turning radius, or the nonholonomic constraint, so it can
(and, live, does) accept layouts a real Hunter SE cannot actually drive
between, e.g. a goal that is geometrically "around the corner" but requires a
tighter U-turn than the robot's steering lock allows).

WHAT THIS ACTUALLY IS -- read before citing it as a guarantee anywhere
(code review: an earlier version of this docstring overclaimed "never a
false positive", which is NOT accurate -- see "Known false-positive risk"
below): a small, bounded Hybrid-A*-style search (Dolgov et al. 2010) over
the package's OWN exact bicycle-model kinematics (``dynamics/bicycle_model.step``
-- the same closed-form arc integration used by every rollout/risk module
here, so "feasible" means feasible under the identical kinematics the rest
of the system already assumes), used as a SCENARIO REJECTION FILTER inside
``procedural_generator.generate_scenario``'s existing retry loop -- it is
NOT a formal/complete kinodynamic motion planner, has no soundness or
completeness proof, and its "feasible"/"infeasible" verdict is only as
trustworthy as the discretization choices below. Treat a "feasible" verdict
as "a coarse, bounded search found A plausible path", and an "infeasible"
verdict as "this search didn't find one within its budget" -- neither is a
mathematical certificate.

Not a Dubins/Reeds-Shepp closed-form solver -- no such library is available
in this environment (checked: neither ``dubins`` nor ``reeds_shepp``
importable, and this package's ``package.xml`` intentionally has no new
runtime deps for a training-time gate), and a hand-rolled closed-form Dubins
solver is easy to get subtly wrong in exactly the ways that would matter
here. A discretized motion-primitive search directly over already-tested
kinematics is smaller and easier to verify by inspection/testing than a
hand-derived closed-form solver would be.

Known false-NEGATIVE risk (rejecting an actually-feasible layout): the
motion-primitive set is coarse (3 fixed steering choices x one fixed speed x
one fixed duration, plus a coarse xy/yaw discretization for the visited-set
dedup) -- a real Hybrid-A* / Dubins path that needs an intermediate steering
angle, a different speed, or falls between two dedup cells this search
happens to treat as "already visited" can be missed. This class of error is
CHEAP: it just costs one extra retry in ``generate_scenario``'s existing
retry loop with a fresh procedural draw.

Known false-POSITIVE risk (accepting an actually-infeasible layout) -- THE
MORE SERIOUS FAILURE MODE, since it can hand the trainer/evaluator an
unsolvable episode: each motion primitive's collision check only samples
``num_path_samples`` DISCRETE points along its arc (not a continuous sweep),
so an obstacle small enough (or precisely enough placed) to fall entirely
BETWEEN two consecutive samples can be missed ("tunneling"). Concretely,
consecutive samples along a primitive are spaced roughly
``primitive_speed_mps * primitive_duration_sec / num_path_samples`` apart
(≈0.5m·1.0/5 ≈ 0.1m at this module's defaults) -- an obstacle whose radius
is small relative to that spacing, positioned near the arc but between two
samples, is the concrete scenario this search can get wrong. This is
mitigated (not eliminated) by keeping ``num_path_samples`` reasonably high
relative to the primitive's arc length (see that parameter's own docstring)
and by every OTHER independent safety layer downstream still applying in
full regardless (the runtime collision detector / safety guard / episode
truncation-on-collision all still run against the REAL simulated episode;
this gate only decides which SCENARIOS are worth attempting, it plays no
role in detecting a collision once an episode is actually running) -- see
``tests/test_ackermann_feasibility.py``'s resolution-sensitivity tests for
a concrete, checked example of a thin obstacle a coarse resolution misses
that a finer one catches.

Deliberately forward-only (3 motion primitives per expansion: full-left,
straight, full-right, at a fixed nominal speed) -- Hunter SE's action space
never reverses (``robot.min_forward_speed_mps >= 0`` always), so a search
that could reverse would accept layouts the trained policy could never
actually execute.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Tuple

from hunter_kinodynamic_rl.dynamics import bicycle_model
from hunter_kinodynamic_rl.robot.interface import VehicleState

_YAW_BINS = 16
_YAW_BIN_SIZE = 2.0 * math.pi / _YAW_BINS


def _yaw_cell(yaw: float) -> int:
    normalized = yaw % (2.0 * math.pi)
    return int(normalized / _YAW_BIN_SIZE) % _YAW_BINS


def _xy_cell(x: float, y: float, resolution_m: float) -> Tuple[int, int]:
    return int(round(x / resolution_m)), int(round(y / resolution_m))


def _clear_of_obstacles(x: float, y: float, obstacles: Iterable, robot_radius: float) -> bool:
    for obs in obstacles:
        if math.hypot(x - obs.x, y - obs.y) < obs.radius + robot_radius:
            return False
    return True


def _in_bounds(x: float, y: float, half_extent: float) -> bool:
    return -half_extent <= x <= half_extent and -half_extent <= y <= half_extent


def is_ackermann_feasible(
    start_x: float, start_y: float, start_yaw: float,
    goal_x: float, goal_y: float, goal_radius_m: float,
    obstacles: Iterable, world_size_m: float, robot_radius: float,
    min_turning_radius_m: float, wheelbase_m: float,
    *,
    primitive_speed_mps: float = 1.0,
    primitive_duration_sec: float = 0.5,
    xy_resolution_m: float = 0.5,
    # code review: raised from 3 -> 5 (a tighter per-primitive arc-length
    # sample spacing, see the module docstring's "Known false-positive
    # risk" section for the exact tunneling-vs-spacing relationship this
    # trades off against search cost) -- still cheap (linear cost per
    # primitive), and tests/test_ackermann_feasibility.py directly checks
    # the resolution-sensitivity this implies rather than just asserting it.
    num_path_samples: int = 5,
    max_expansions: int = 4000,
) -> bool:
    """Bounded forward search: can the robot ACTUALLY drive from
    ``(start_x, start_y, start_yaw)`` to within ``goal_radius_m`` of
    ``(goal_x, goal_y)`` obeying ``min_turning_radius_m`` (Ackermann steering
    lock) and staying clear of ``obstacles``/the world boundary the whole
    way? Returns False (not raises) on a normal search exhaustion -- only
    raises on a caller-supplied physically-invalid parameter (a config bug),
    never as a way to report "no path found". See the module docstring's
    "WHAT THIS ACTUALLY IS" section for the exact guarantee level (none,
    formally) -- this is a scenario REJECTION FILTER, not a certified
    planner."""
    if min_turning_radius_m <= 0.0:
        raise ValueError(f"min_turning_radius_m must be > 0, got {min_turning_radius_m}")
    if wheelbase_m <= 0.0:
        raise ValueError(f"wheelbase_m must be > 0, got {wheelbase_m}")
    if goal_radius_m <= 0.0:
        raise ValueError(f"goal_radius_m must be > 0, got {goal_radius_m}")

    half_extent = world_size_m / 2.0
    # code review: explicit, up-front goal clearance/boundary check --
    # previously only implicit (every SEARCH-REACHED endpoint's own
    # clearance was checked as part of that primitive's segment, so an
    # unreachable-anyway goal would eventually be rejected by search
    # exhaustion regardless, but at the cost of the FULL max_expansions
    # budget, and without ever making "the goal itself is the problem"
    # legible). Checked BEFORE the trivial start-already-at-goal shortcut:
    # a goal that is itself inside an obstacle or outside the world is
    # infeasible regardless of how close the start happens to be to it.
    if not _in_bounds(goal_x, goal_y, half_extent):
        return False
    if not _clear_of_obstacles(goal_x, goal_y, obstacles, robot_radius):
        return False
    # code review: explicit, up-front START bounds/clearance check -- MUST
    # precede the "start already within goal_radius" shortcut just below,
    # symmetric with the goal check above. Previously this function had NO
    # start-bounds check at all, and its start-obstacle check ran AFTER the
    # shortcut -- so a caller-supplied start that is outside the world (or,
    # for the pre-existing obstacle check, sitting inside an obstacle) but
    # happens to already be within goal_radius_m of the goal would return
    # True: a scenario the robot could never have actually been placed at
    # in the first place, which is a contract violation for a function
    # whose whole job is "is THIS start->goal pair actually driveable".
    if not _in_bounds(start_x, start_y, half_extent):
        return False
    if not _clear_of_obstacles(start_x, start_y, obstacles, robot_radius):
        return False

    if math.hypot(start_x - goal_x, start_y - goal_y) <= goal_radius_m:
        return True

    max_steering = math.atan(wheelbase_m / min_turning_radius_m)
    steering_choices = (-max_steering, 0.0, max_steering)

    start = VehicleState(x=start_x, y=start_y, yaw=start_yaw, v=primitive_speed_mps)
    visited = {(*_xy_cell(start_x, start_y, xy_resolution_m), _yaw_cell(start_yaw))}
    frontier: List[VehicleState] = [start]
    expansions = 0

    while frontier and expansions < max_expansions:
        next_frontier: List[VehicleState] = []
        for state in frontier:
            for steering in steering_choices:
                expansions += 1
                segment_clear = True
                endpoint = state
                for i in range(1, num_path_samples + 1):
                    frac = i / num_path_samples
                    endpoint = bicycle_model.step(
                        state, primitive_speed_mps, steering,
                        primitive_duration_sec * frac, wheelbase_m,
                    )
                    if not _in_bounds(endpoint.x, endpoint.y, half_extent) or not _clear_of_obstacles(
                        endpoint.x, endpoint.y, obstacles, robot_radius,
                    ):
                        segment_clear = False
                        break
                if not segment_clear:
                    continue
                if math.hypot(endpoint.x - goal_x, endpoint.y - goal_y) <= goal_radius_m:
                    return True
                cell = (*_xy_cell(endpoint.x, endpoint.y, xy_resolution_m), _yaw_cell(endpoint.yaw))
                if cell in visited:
                    continue
                visited.add(cell)
                next_frontier.append(endpoint)
                if expansions >= max_expansions:
                    break
            if expansions >= max_expansions:
                break
        frontier = next_frontier
    return False
