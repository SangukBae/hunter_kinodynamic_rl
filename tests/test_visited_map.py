import numpy as np
import pytest

from hunter_kinodynamic_rl.config.schema import MappingConfig
from hunter_kinodynamic_rl.navigation.mapping.partial_map import PartialMap
from hunter_kinodynamic_rl.navigation.mapping.visited_map import (
    increment_saturating, normalize_counts, rasterize_circle_cells,
)


def _config(**overrides):
    base = dict(
        resolution_m=0.5, mission_size_cells=20, rolling_size_cells=10,
        free_log_odds_delta=-0.4, occupied_log_odds_delta=0.85,
        free_threshold=-0.2, occupied_threshold=0.2,
        log_odds_min=-4.0, log_odds_max=4.0, inflation_radius_m=0.45,
        visit_radius_m=0.5, visited_count_saturation=10, failure_count_saturation=5,
    )
    base.update(overrides)
    return MappingConfig(**base)


def test_rasterize_circle_cells_contains_center():
    cells = rasterize_circle_cells(5, 5, 2.0)
    assert (5, 5) in cells


def test_rasterize_circle_cells_zero_radius_returns_center_only():
    assert rasterize_circle_cells(3, 3, 0.0) == [(3, 3)]


def test_rasterize_circle_cells_symmetric_about_center():
    cells = set(rasterize_circle_cells(0, 0, 2.0))
    for r, c in cells:
        assert (-r, -c) in cells


def test_normalize_counts_saturates_to_one():
    counts = np.array([0, 5, 100], dtype=np.uint16)
    normalized = normalize_counts(counts, saturation=10)
    assert normalized[0] == pytest.approx(0.0)
    assert normalized[1] == pytest.approx(0.5)
    assert normalized[2] == pytest.approx(1.0)


def test_increment_saturating_never_exceeds_max_value():
    arr = np.zeros((3, 3), dtype=np.uint8)
    cells = [(1, 1)] * 500
    increment_saturating(arr, cells, max_value=5)
    assert arr[1, 1] == 5


def test_increment_saturating_never_exceeds_dtype_max():
    arr = np.zeros((2, 2), dtype=np.uint8)
    cells = [(0, 0)] * 1000
    increment_saturating(arr, cells, max_value=1000)  # requested cap > uint8 range
    assert arr[0, 0] == 255


def test_partial_map_record_visit_sets_visited_count_and_last_visit_step():
    pm = PartialMap(_config())
    pm.record_visit(0.0, 0.0, step=7)
    cell = pm.world_to_cell(0.0, 0.0)
    assert pm.visited_count[cell] >= 1
    assert pm.last_visit_step[cell] == 7


def test_partial_map_record_visit_out_of_bounds_is_noop():
    pm = PartialMap(_config())
    pm.record_visit(10_000.0, 10_000.0, step=1)
    assert pm.visited_count.sum() == 0


def test_partial_map_record_visit_saturates_normalized_channel():
    pm = PartialMap(_config(visited_count_saturation=3))
    for step in range(10):
        pm.record_visit(0.0, 0.0, step=step)
    channels = pm.channels()
    assert channels.visited.max() == pytest.approx(1.0)


def test_partial_map_record_failure_independent_of_visited_count():
    pm = PartialMap(_config())
    pm.record_failure(0.0, 0.0)
    cell = pm.world_to_cell(0.0, 0.0)
    assert pm.failure_count[cell] >= 1
    assert pm.visited_count[cell] == 0


def test_partial_map_record_failure_out_of_bounds_is_noop():
    pm = PartialMap(_config())
    pm.record_failure(10_000.0, 10_000.0)
    assert pm.failure_count.sum() == 0
