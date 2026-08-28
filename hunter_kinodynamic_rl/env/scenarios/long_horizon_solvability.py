"""Pure grid solvability checks for a ``LongHorizonWorld``'s occupancy (plan
section 7.3 items 6-9): footprint inflation, connectivity, weighted shortest
path (Dijkstra), and a grid-based Ackermann feasibility filter.

Every function here is a pure function of ``(occupancy, resolution_m,
origin_xy, ...)`` -- no ``LongHorizonWorld``/dataclass coupling, so a test
(or ``evaluation``/teacher code) can independently recompute
``shortest_path_length_m`` from a returned world's own ``occupancy`` and get
the EXACT same answer ``long_horizon_generator`` itself got: the generator
calls the SAME functions in this module for its own internal start/goal
search, so there is only ever one implementation to drift out of sync with
itself (see ``tests/test_long_horizon_solvability.py``'s
independent-recomputation tests).

Deliberately separate from ``env/scenarios/ackermann_feasibility.py`` (the
existing circular-obstacle-list-based check the small arena/circle
procedural generator uses) -- a rasterized wall-segment maze is a fine
OCCUPANCY GRID, not a small list of circles, so the collision test and the
practical search-parameter scale (a 40 m maze vs. a 12 m arena) are both
different enough to warrant a dedicated implementation rather than
threading a grid-lookup callback through code this package otherwise treats
as a stable, independently-tested Phase-1/2 utility.
"""

from __future__ import annotations

import heapq
import math
from typing import Optional, Tuple

import numpy as np

from hunter_kinodynamic_rl.dynamics import bicycle_model
from hunter_kinodynamic_rl.navigation.mapping.visited_map import rasterize_circle_cells
from hunter_kinodynamic_rl.robot.interface import VehicleState

_SQRT2 = math.sqrt(2.0)
# 8-connected neighbor offsets + their true Euclidean step weight (in grid
# cells; multiplied by resolution_m by the caller) -- diagonal moves cost
# sqrt(2), never treated as unit cost (which would systematically
# underestimate diagonal-heavy shortest paths).
_NEIGHBORS_8 = (
    (-1, -1, _SQRT2), (-1, 0, 1.0), (-1, 1, _SQRT2),
    (0, -1, 1.0), (0, 1, 1.0),
    (1, -1, _SQRT2), (1, 0, 1.0), (1, 1, _SQRT2),
)
_YAW_BINS = 16
_YAW_BIN_SIZE = 2.0 * math.pi / _YAW_BINS


def world_to_cell(
    x: float, y: float, resolution_m: float, origin_x: float, origin_y: float, height: int, width: int,
) -> Optional[Tuple[int, int]]:
    col = int(math.floor((x - origin_x) / resolution_m))
    row = int(math.floor((y - origin_y) / resolution_m))
    if 0 <= row < height and 0 <= col < width:
        return row, col
    return None


def cell_to_world(row: int, col: int, resolution_m: float, origin_x: float, origin_y: float) -> Tuple[float, float]:
    x = origin_x + (col + 0.5) * resolution_m
    y = origin_y + (row + 0.5) * resolution_m
    return x, y


def inflate_occupancy(occupancy: np.ndarray, resolution_m: float, inflation_radius_m: float) -> np.ndarray:
    """Dilate ``occupancy`` (True == occupied) by ``inflation_radius_m`` --
    e.g. the robot's own collision-footprint radius, so every downstream
    check here treats the robot as a point against the INFLATED grid rather
    than needing its own footprint-aware geometry. Reuses
    ``navigation.mapping.visited_map.rasterize_circle_cells`` (the exact
    same circle rasterization ``PartialMap._inflate`` uses) rather than a
    second, independent implementation."""
    if inflation_radius_m <= 0.0:
        return occupancy.copy()
    radius_cells = inflation_radius_m / resolution_m
    inflated = occupancy.copy()
    h, w = occupancy.shape
    for r, c in zip(*np.nonzero(occupancy)):
        for rr, cc in rasterize_circle_cells(int(r), int(c), radius_cells):
            if 0 <= rr < h and 0 <= cc < w:
                inflated[rr, cc] = True
    return inflated


def dijkstra_distances(free: np.ndarray, resolution_m: float, source_rc: Tuple[int, int]) -> np.ndarray:
    """8-connected weighted Dijkstra shortest-path DISTANCE (meters) from
    ``source_rc`` to every cell of ``free`` (True == passable). Unreachable
    cells stay ``np.inf``. ``source_rc`` itself must be passable -- raises
    ``ValueError`` otherwise (a caller bug: an occupied/out-of-bounds source
    has no meaningful distance field, this is not a "no path exists"
    outcome, which callers signal via an all-``inf`` reachable set instead)."""
    h, w = free.shape
    sr, sc = source_rc
    if not (0 <= sr < h and 0 <= sc < w) or not free[sr, sc]:
        raise ValueError(f"dijkstra_distances: source {source_rc} is out of bounds or occupied")
    dist = np.full((h, w), np.inf, dtype=np.float64)
    dist[sr, sc] = 0.0
    visited = np.zeros((h, w), dtype=bool)
    heap = [(0.0, sr, sc)]
    while heap:
        d, r, c = heapq.heappop(heap)
        if visited[r, c]:
            continue
        visited[r, c] = True
        for dr, dc, step_w in _NEIGHBORS_8:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or not free[nr, nc] or visited[nr, nc]:
                continue
            nd = d + step_w * resolution_m
            if nd < dist[nr, nc]:
                dist[nr, nc] = nd
                heapq.heappush(heap, (nd, nr, nc))
    return dist


def shortest_path_length_m(
    occupancy: np.ndarray, resolution_m: float, origin_xy: Tuple[float, float],
    start_xy: Tuple[float, float], goal_xy: Tuple[float, float], inflation_radius_m: float = 0.0,
) -> Optional[float]:
    """``None`` when start/goal fall outside the grid, on an (inflated)
    occupied cell, or are simply not connected -- never raises for an
    ordinary "no path" outcome (only :func:`dijkstra_distances`'s own
    programmer-error case can raise, and this function never triggers it:
    both endpoints are checked clear/in-bounds first)."""
    origin_x, origin_y = origin_xy
    h, w = occupancy.shape
    free = ~inflate_occupancy(occupancy, resolution_m, inflation_radius_m)
    start_cell = world_to_cell(start_xy[0], start_xy[1], resolution_m, origin_x, origin_y, h, w)
    goal_cell = world_to_cell(goal_xy[0], goal_xy[1], resolution_m, origin_x, origin_y, h, w)
    if start_cell is None or goal_cell is None:
        return None
    if not free[start_cell] or not free[goal_cell]:
        return None
    dist = dijkstra_distances(free, resolution_m, start_cell)
    d = float(dist[goal_cell])
    return None if math.isinf(d) else d


def is_connected(
    occupancy: np.ndarray, resolution_m: float, origin_xy: Tuple[float, float],
    start_xy: Tuple[float, float], goal_xy: Tuple[float, float], inflation_radius_m: float = 0.0,
) -> bool:
    return shortest_path_length_m(
        occupancy, resolution_m, origin_xy, start_xy, goal_xy, inflation_radius_m,
    ) is not None


def _yaw_cell(yaw: float) -> int:
    normalized = yaw % (2.0 * math.pi)
    return int(normalized / _YAW_BIN_SIZE) % _YAW_BINS


def is_ackermann_feasible_grid(
    start_x: float, start_y: float, start_yaw: float,
    goal_x: float, goal_y: float, goal_radius_m: float,
    occupancy: np.ndarray, resolution_m: float, origin_xy: Tuple[float, float], inflation_radius_m: float,
    min_turning_radius_m: float, wheelbase_m: float,
    *,
    primitive_speed_mps: float = 1.5,
    primitive_duration_sec: float = 1.0,
    xy_resolution_m: float = 1.0,
    num_path_samples: int = 5,
    # A long-horizon maze's start/goal geodesic distance can be 20-30+ m
    # (vs. this package's small circular-obstacle arena, where a few
    # thousand expansions comfortably covers a <15 m search) -- a path that
    # long needs ~15-25 chained primitive steps, and empirically (see
    # docs/verification/) 6000 expansions was observed to exhaust before
    # finding an otherwise-feasible route through an ordinary 2-2.5 m wide
    # corridor maze, rejecting a world that a larger budget accepts. 20000
    # was the smallest tested value that reliably found the same route.
    max_expansions: int = 20000,
) -> bool:
    """Bounded forward Hybrid-A*-style search (same algorithm family as
    ``env.scenarios.ackermann_feasibility.is_ackermann_feasible`` -- see
    that module's docstring for the exact guarantee level this class of
    search provides: a scenario REJECTION FILTER, never a certified
    planner, with both false-negative -- cheap, costs one retry -- and
    false-positive -- mitigated, not eliminated, by every other downstream
    safety layer -- failure modes). A deliberately SEPARATE implementation
    (module docstring) checking against a fine occupancy GRID instead of a
    circular-obstacle list, with coarser default primitive/resolution
    parameters sized for a long-horizon maze's ~10x larger spatial scale.

    Returns ``False`` (never raises) on ordinary search exhaustion -- only
    raises on a caller-supplied physically-invalid parameter."""
    if min_turning_radius_m <= 0.0:
        raise ValueError(f"min_turning_radius_m must be > 0, got {min_turning_radius_m}")
    if wheelbase_m <= 0.0:
        raise ValueError(f"wheelbase_m must be > 0, got {wheelbase_m}")
    if goal_radius_m <= 0.0:
        raise ValueError(f"goal_radius_m must be > 0, got {goal_radius_m}")

    origin_x, origin_y = origin_xy
    inflated = inflate_occupancy(occupancy, resolution_m, inflation_radius_m)
    h, w = inflated.shape

    def clear(x: float, y: float) -> bool:
        cell = world_to_cell(x, y, resolution_m, origin_x, origin_y, h, w)
        return cell is not None and not inflated[cell]

    # Explicit, up-front goal/start clearance checks BEFORE the trivial
    # already-at-goal shortcut (mirrors ackermann_feasibility.is_ackermann_feasible's
    # own ordering rationale -- a start/goal that is itself off-grid or
    # occupied must never report feasible regardless of their mutual distance).
    if not clear(goal_x, goal_y):
        return False
    if not clear(start_x, start_y):
        return False
    if math.hypot(start_x - goal_x, start_y - goal_y) <= goal_radius_m:
        return True

    max_steering = math.atan(wheelbase_m / min_turning_radius_m)
    steering_choices = (-max_steering, 0.0, max_steering)

    def _xy_cell(x: float, y: float) -> Tuple[int, int]:
        return int(round(x / xy_resolution_m)), int(round(y / xy_resolution_m))

    start = VehicleState(x=start_x, y=start_y, yaw=start_yaw, v=primitive_speed_mps)
    visited = {(*_xy_cell(start_x, start_y), _yaw_cell(start_yaw))}
    frontier = [start]
    expansions = 0

    while frontier and expansions < max_expansions:
        next_frontier = []
        for state in frontier:
            for steering in steering_choices:
                expansions += 1
                segment_clear = True
                endpoint = state
                for i in range(1, num_path_samples + 1):
                    frac = i / num_path_samples
                    endpoint = bicycle_model.step(
                        state, primitive_speed_mps, steering, primitive_duration_sec * frac, wheelbase_m,
                    )
                    if not clear(endpoint.x, endpoint.y):
                        segment_clear = False
                        break
                if not segment_clear:
                    continue
                if math.hypot(endpoint.x - goal_x, endpoint.y - goal_y) <= goal_radius_m:
                    return True
                cell = (*_xy_cell(endpoint.x, endpoint.y), _yaw_cell(endpoint.yaw))
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
