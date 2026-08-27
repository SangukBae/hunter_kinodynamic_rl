import numpy as np
import pytest

from hunter_kinodynamic_rl.config.schema import MappingConfig
from hunter_kinodynamic_rl.navigation.mapping.partial_map import PartialMap
from hunter_kinodynamic_rl.navigation.mapping.rolling_map import crop_rolling


def _config(**overrides):
    base = dict(
        resolution_m=0.5, mission_size_cells=20, rolling_size_cells=8,
        free_log_odds_delta=-0.4, occupied_log_odds_delta=0.85,
        free_threshold=-0.2, occupied_threshold=0.2,
        log_odds_min=-4.0, log_odds_max=4.0, inflation_radius_m=0.45,
        visit_radius_m=0.5, visited_count_saturation=10, failure_count_saturation=5,
    )
    base.update(overrides)
    return MappingConfig(**base)


def test_crop_size_matches_requested_size_cells():
    pm = PartialMap(_config())
    crop = crop_rolling(pm, (0.0, 0.0), 8)
    assert crop.occupied.shape == (8, 8)
    assert crop.size_cells == 8


def test_crop_centered_on_observed_obstacle_finds_it():
    pm = PartialMap(_config())
    pm.integrate_beam((0.0, 0.0), angle_mission=0.0, range_m=1.0, range_max=10.0)
    crop = crop_rolling(pm, (0.0, 0.0), 8)
    assert crop.occupied.sum() == 1
    assert not np.any(crop.occupied & crop.free)


def test_crop_far_outside_mission_map_is_entirely_unknown():
    pm = PartialMap(_config())
    pm.integrate_beam((0.0, 0.0), angle_mission=0.0, range_m=1.0, range_max=10.0)
    crop = crop_rolling(pm, (1000.0, 1000.0), 8)
    assert crop.unknown.all()
    assert not crop.occupied.any()
    assert not crop.free.any()


def test_crop_near_edge_pads_out_of_bounds_region_as_unknown():
    pm = PartialMap(_config(mission_size_cells=10))
    # Center the crop at the very corner of the mission map so half the
    # requested window necessarily falls outside it.
    corner_x, corner_y = pm.cell_to_world(0, 0)
    crop = crop_rolling(pm, (corner_x, corner_y), 8)
    assert crop.unknown.sum() > 0


def test_crop_origin_reflects_requested_center():
    pm = PartialMap(_config())
    crop = crop_rolling(pm, (0.0, 0.0), 8)
    # The crop's own origin must place (0, 0) inside its extent.
    assert crop.origin_x <= 0.0 <= crop.origin_x + crop.size_cells * crop.resolution_m
    assert crop.origin_y <= 0.0 <= crop.origin_y + crop.size_cells * crop.resolution_m


def test_crop_rejects_non_positive_size():
    pm = PartialMap(_config())
    with pytest.raises(ValueError):
        crop_rolling(pm, (0.0, 0.0), 0)


def test_crop_carries_observed_uncertain_channel():
    pm = PartialMap(_config())
    cell = pm.world_to_cell(0.0, 0.0)
    pm.log_odds[cell] = 0.0  # strictly between free_threshold and occupied_threshold
    pm.observed[cell] = True
    crop = crop_rolling(pm, (0.0, 0.0), 8)
    assert crop.observed_uncertain.sum() >= 1
    assert not np.any(crop.observed_uncertain & crop.free)
    assert not np.any(crop.observed_uncertain & crop.occupied)


def test_crop_carries_inflated_channel():
    pm = PartialMap(_config(inflation_radius_m=1.0))
    pm.integrate_beam((0.0, 0.0), angle_mission=0.0, range_m=1.0, range_max=10.0)
    crop = crop_rolling(pm, (0.0, 0.0), 8)
    assert crop.inflated.sum() >= crop.occupied.sum()
    assert np.all(crop.inflated[crop.occupied])
