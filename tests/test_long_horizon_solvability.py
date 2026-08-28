"""Coverage for ``env/scenarios/long_horizon_solvability.py`` -- pure grid
inflation/connectivity/shortest-path/Ackermann-feasibility checks, tested
independently of the generator (small synthetic occupancy grids built by
hand)."""

import math

import numpy as np
import pytest

from hunter_kinodynamic_rl.env.scenarios import long_horizon_solvability as S

RES = 0.25
ORIGIN = (-5.0, -5.0)
N = 40  # 10 m x 10 m grid at 0.25 m resolution


def _empty_grid() -> np.ndarray:
    return np.zeros((N, N), dtype=bool)


def _wall_at_row(grid: np.ndarray, row: int, gap_col_range=None) -> None:
    for col in range(N):
        if gap_col_range is not None and gap_col_range[0] <= col < gap_col_range[1]:
            continue
        grid[row, col] = True


# --------------------------------------------------------------- cell/world coordinate round trip
def test_world_to_cell_and_cell_to_world_round_trip():
    for row in (0, 10, N - 1):
        for col in (0, 10, N - 1):
            x, y = S.cell_to_world(row, col, RES, *ORIGIN)
            cell = S.world_to_cell(x, y, RES, *ORIGIN, N, N)
            assert cell == (row, col)


def test_world_to_cell_out_of_bounds_returns_none():
    assert S.world_to_cell(1000.0, 1000.0, RES, *ORIGIN, N, N) is None
    assert S.world_to_cell(-1000.0, -1000.0, RES, *ORIGIN, N, N) is None


# --------------------------------------------------------------- inflation
def test_inflate_occupancy_grows_occupied_region_by_radius():
    grid = _empty_grid()
    grid[20, 20] = True
    inflated = S.inflate_occupancy(grid, RES, inflation_radius_m=0.5)
    assert inflated[20, 20]
    # 0.5 m / 0.25 m = 2 cells radius -- a cell 2 away should be inflated too.
    assert inflated[20, 22] or inflated[22, 20]
    # Far away must remain untouched.
    assert not inflated[0, 0]


def test_inflate_occupancy_zero_radius_is_a_copy_not_the_same_array():
    grid = _empty_grid()
    grid[5, 5] = True
    inflated = S.inflate_occupancy(grid, RES, inflation_radius_m=0.0)
    assert np.array_equal(inflated, grid)
    inflated[5, 5] = False
    assert grid[5, 5]  # original untouched -- confirms it's a copy


# --------------------------------------------------------------- connectivity / shortest path
def test_shortest_path_length_open_field_matches_straight_line_distance():
    grid = _empty_grid()
    start = (-3.0, -3.0)
    goal = (3.0, -3.0)
    d = S.shortest_path_length_m(grid, RES, ORIGIN, start, goal)
    assert d is not None
    # 8-connected grid distance is close to, never less than, straight-line.
    assert d >= 6.0 - RES
    assert d < 6.0 + 2 * RES


def test_shortest_path_length_none_when_wall_fully_blocks():
    grid = _empty_grid()
    _wall_at_row(grid, row=20)  # solid wall, no gap
    d = S.shortest_path_length_m(grid, RES, ORIGIN, (-3.0, -3.0), (3.0, 3.0))
    assert d is None


def test_shortest_path_length_routes_through_a_gap():
    grid = _empty_grid()
    _wall_at_row(grid, row=20, gap_col_range=(18, 22))
    d = S.shortest_path_length_m(grid, RES, ORIGIN, (-3.0, -3.0), (3.0, 3.0))
    assert d is not None
    assert d > 6.0  # must be longer than the direct euclidean distance


def test_shortest_path_length_matches_a_hand_computed_single_gap_detour():
    """Genuinely INDEPENDENT verification of the Dijkstra implementation
    itself (code review: recomputing shortest_path_length_m from a
    generator-produced world's own occupancy is regression protection, not
    an independent check on the algorithm -- see
    long_horizon_generator's module docstring step 8). This fixture's
    expected distance is derived by hand from known geometry, never by
    calling any package code:

    A single-cell gap (row=20, col=20 only) in an otherwise solid wall
    forces the path through exactly one known point -- the gap cell's own
    center, computed directly via cell_to_world (pure coordinate
    arithmetic, not a path-planning call). Start (-3, -3) and goal (3, 3)
    are chosen so BOTH straight-line legs (start -> gap point, gap point ->
    goal) are pure 45-degree diagonals -- the one case where an
    8-connected grid's distance metric reproduces true Euclidean distance
    EXACTLY (no octile-vs-Euclidean approximation error at all), so the
    hand-computed
    ``hypot(gap - start) + hypot(goal - gap)`` sum can be asserted against
    the algorithm's output to float precision, not just "within some
    tolerance"."""
    grid = _empty_grid()
    _wall_at_row(grid, row=20, gap_col_range=(20, 21))  # exactly one open cell
    start, goal = (-3.0, -3.0), (3.0, 3.0)
    gap_x, gap_y = S.cell_to_world(20, 20, RES, *ORIGIN)
    hand_computed = math.hypot(gap_x - start[0], gap_y - start[1]) + math.hypot(goal[0] - gap_x, goal[1] - gap_y)

    d = S.shortest_path_length_m(grid, RES, ORIGIN, start, goal)
    assert d is not None
    assert abs(d - hand_computed) < 1e-9


def test_shortest_path_length_none_when_start_or_goal_occupied():
    grid = _empty_grid()
    grid[20, 20] = True
    x, y = S.cell_to_world(20, 20, RES, *ORIGIN)
    assert S.shortest_path_length_m(grid, RES, ORIGIN, (x, y), (3.0, 3.0)) is None
    assert S.shortest_path_length_m(grid, RES, ORIGIN, (-3.0, -3.0), (x, y)) is None


def test_shortest_path_length_none_when_out_of_grid_bounds():
    grid = _empty_grid()
    assert S.shortest_path_length_m(grid, RES, ORIGIN, (-3.0, -3.0), (1000.0, 1000.0)) is None


def test_is_connected_matches_shortest_path_length_none_check():
    grid = _empty_grid()
    _wall_at_row(grid, row=20)
    assert not S.is_connected(grid, RES, ORIGIN, (-3.0, -3.0), (3.0, 3.0))
    grid2 = _empty_grid()
    assert S.is_connected(grid2, RES, ORIGIN, (-3.0, -3.0), (3.0, 3.0))


def test_dijkstra_distances_raises_on_occupied_source():
    grid = _empty_grid()
    grid[10, 10] = True  # occupied
    free = ~grid
    with pytest.raises(ValueError):
        S.dijkstra_distances(free, RES, (10, 10))


def test_dijkstra_distances_unreachable_cells_are_infinite():
    grid = _empty_grid()
    _wall_at_row(grid, row=20)
    dist = S.dijkstra_distances(~grid, RES, (0, 0))
    assert math.isinf(dist[39, 39])


# --------------------------------------------------------------- Ackermann grid feasibility
def test_ackermann_grid_feasible_open_field_straight_line():
    grid = _empty_grid()
    assert S.is_ackermann_feasible_grid(
        -3.0, -3.0, 0.0, 3.0, -3.0, 0.5, grid, RES, ORIGIN, 0.3, min_turning_radius_m=1.0, wheelbase_m=0.6,
    )


def test_ackermann_grid_infeasible_when_wall_fully_blocks():
    grid = _empty_grid()
    _wall_at_row(grid, row=20)
    assert not S.is_ackermann_feasible_grid(
        -3.0, -3.0, 0.0, 3.0, 3.0, 0.5, grid, RES, ORIGIN, 0.3, min_turning_radius_m=1.0, wheelbase_m=0.6,
    )


def test_ackermann_grid_feasible_through_a_wide_gap():
    grid = _empty_grid()
    _wall_at_row(grid, row=20, gap_col_range=(14, 26))
    assert S.is_ackermann_feasible_grid(
        -3.0, -3.0, math.pi / 2, 3.0, 3.0, 0.5, grid, RES, ORIGIN, 0.3, min_turning_radius_m=1.0, wheelbase_m=0.6,
    )


def test_ackermann_grid_start_or_goal_occupied_is_infeasible():
    grid = _empty_grid()
    grid[20, 20] = True
    x, y = S.cell_to_world(20, 20, RES, *ORIGIN)
    assert not S.is_ackermann_feasible_grid(
        x, y, 0.0, 3.0, 3.0, 0.5, grid, RES, ORIGIN, 0.0, min_turning_radius_m=1.0, wheelbase_m=0.6,
    )
    assert not S.is_ackermann_feasible_grid(
        -3.0, -3.0, 0.0, x, y, 0.5, grid, RES, ORIGIN, 0.0, min_turning_radius_m=1.0, wheelbase_m=0.6,
    )


@pytest.mark.parametrize("kwargs", [
    {"min_turning_radius_m": 0.0},
    {"min_turning_radius_m": -1.0},
    {"wheelbase_m": 0.0},
    {"goal_radius_m": 0.0},
])
def test_ackermann_grid_invalid_parameters_raise(kwargs):
    grid = _empty_grid()
    base = dict(min_turning_radius_m=1.0, wheelbase_m=0.6, goal_radius_m=0.5)
    base.update(kwargs)
    goal_radius_m = base.pop("goal_radius_m")
    with pytest.raises(ValueError):
        S.is_ackermann_feasible_grid(
            -3.0, -3.0, 0.0, 3.0, -3.0, goal_radius_m, grid, RES, ORIGIN, 0.0, **base,
        )


def test_ackermann_grid_already_within_goal_radius_is_trivially_feasible():
    grid = _empty_grid()
    assert S.is_ackermann_feasible_grid(
        0.0, 0.0, 0.0, 0.1, 0.1, 0.5, grid, RES, ORIGIN, 0.0, min_turning_radius_m=1.0, wheelbase_m=0.6,
    )
