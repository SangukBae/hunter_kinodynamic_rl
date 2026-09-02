import math

import numpy as np
import pytest

from hunter_kinodynamic_rl.config.schema import MappingConfig
from hunter_kinodynamic_rl.navigation.mapping.partial_map import PartialMap


def _config(**overrides):
    base = dict(
        resolution_m=0.5, mission_size_cells=20, rolling_size_cells=10,
        free_log_odds_delta=-0.4, occupied_log_odds_delta=0.85,
        free_threshold=-0.2, occupied_threshold=0.2,
        log_odds_min=-4.0, log_odds_max=4.0, inflation_radius_m=0.45,
        visit_radius_m=0.5, visited_count_saturation=100, failure_count_saturation=50,
    )
    base.update(overrides)
    return MappingConfig(**base)


def test_world_to_cell_round_trip_near_center():
    pm = PartialMap(_config())
    cell = pm.world_to_cell(0.0, 0.0)
    assert cell is not None
    x, y = pm.cell_to_world(*cell)
    assert x == pytest.approx(0.25)
    assert y == pytest.approx(0.25)


def test_world_to_cell_out_of_bounds_returns_none():
    pm = PartialMap(_config())
    assert pm.world_to_cell(1000.0, 1000.0) is None


def test_valid_hit_beam_marks_passthrough_free_and_endpoint_occupied():
    pm = PartialMap(_config())
    pm.integrate_beam((0.0, 0.0), angle_mission=0.0, range_m=3.0, range_max=10.0)
    channels = pm.channels()
    assert channels.occupied.sum() == 1
    assert channels.free.sum() >= 1
    # The occupied cell must be roughly 3m along +x from the sensor origin.
    occ_row, occ_col = np.argwhere(channels.occupied)[0]
    ox, oy = pm.cell_to_world(occ_row, occ_col)
    assert ox == pytest.approx(3.0, abs=pm.resolution_m)
    assert oy == pytest.approx(0.0, abs=pm.resolution_m)


def test_nan_beam_makes_no_map_update():
    pm = PartialMap(_config())
    pm.integrate_beam((0.0, 0.0), angle_mission=0.3, range_m=float("nan"), range_max=10.0)
    assert pm.observed.sum() == 0


def test_negative_inf_beam_makes_no_map_update():
    pm = PartialMap(_config())
    pm.integrate_beam((0.0, 0.0), angle_mission=0.3, range_m=float("-inf"), range_max=10.0)
    assert pm.observed.sum() == 0


def test_positive_inf_beam_is_treated_as_max_range_free_ray():
    """Code review fix: many real LiDAR drivers (and this project's own
    Gazebo bridge) publish a "no return" beam as +inf rather than exactly
    range_max. Previously skipped identically to NaN, leaving open space
    permanently UNKNOWN; must now behave exactly like a genuine max-range
    beam -- FREE ray out to range_max, no occupied endpoint."""
    pm_inf = PartialMap(_config())
    pm_inf.integrate_beam((0.0, 0.0), angle_mission=0.0, range_m=float("inf"), range_max=10.0)
    pm_max = PartialMap(_config())
    pm_max.integrate_beam((0.0, 0.0), angle_mission=0.0, range_m=10.0, range_max=10.0)
    channels_inf = pm_inf.channels()
    channels_max = pm_max.channels()
    assert channels_inf.occupied.sum() == 0
    assert channels_inf.free.sum() > 0
    assert np.array_equal(channels_inf.free, channels_max.free)


def test_non_positive_range_makes_no_map_update():
    pm = PartialMap(_config())
    pm.integrate_beam((0.0, 0.0), angle_mission=0.3, range_m=0.0, range_max=10.0)
    assert pm.observed.sum() == 0


def test_range_below_range_min_makes_no_map_update():
    pm = PartialMap(_config())
    pm.integrate_beam((0.0, 0.0), angle_mission=0.0, range_m=0.05, range_max=10.0, range_min=0.2)
    assert pm.observed.sum() == 0


def test_range_at_or_above_range_min_is_integrated():
    pm = PartialMap(_config())
    pm.integrate_beam((0.0, 0.0), angle_mission=0.0, range_m=0.5, range_max=10.0, range_min=0.2)
    assert pm.observed.sum() > 0


def test_max_range_beam_marks_free_but_no_occupied_endpoint():
    pm = PartialMap(_config())
    pm.integrate_beam((0.0, 0.0), angle_mission=0.0, range_m=10.0, range_max=10.0)
    channels = pm.channels()
    assert channels.occupied.sum() == 0
    assert channels.free.sum() > 0


def test_channels_are_mutually_exclusive_across_a_full_synthetic_scan():
    pm = PartialMap(_config())
    n = 60
    angles = np.linspace(-math.pi, math.pi, n, endpoint=False)
    rng = np.random.default_rng(0)
    ranges = rng.uniform(0.5, 10.0, size=n)
    ranges[::7] = float("nan")
    ranges[::11] = 10.0  # force some max-range beams
    ranges[::13] = float("inf")  # force some +inf no-return beams
    pm.integrate_scan((0.0, 0.0), angles, ranges, range_max=10.0)
    channels = pm.channels()
    assert not np.any(channels.occupied & channels.free)
    assert not np.any(channels.occupied & channels.unknown)
    assert not np.any(channels.free & channels.unknown)
    assert np.array_equal(channels.unknown, ~pm.observed)


def test_four_channels_exhaustively_and_disjointly_partition_every_cell():
    """Code review fix: occupied/free/observed_uncertain/unknown together
    must cover EVERY cell exactly once -- the map's real (4-state, not
    3-state) contract."""
    pm = PartialMap(_config())
    n = 40
    angles = np.linspace(-math.pi, math.pi, n, endpoint=False)
    rng = np.random.default_rng(1)
    ranges = rng.uniform(0.5, 10.0, size=n)
    pm.integrate_scan((0.0, 0.0), angles, ranges, range_max=10.0)
    channels = pm.channels()
    masks = (channels.occupied, channels.free, channels.observed_uncertain, channels.unknown)
    # Exhaustive: every cell is covered by at least one mask.
    union = np.zeros_like(channels.unknown)
    for m in masks:
        union |= m
    assert np.all(union)
    # Disjoint: no cell is covered by more than one mask.
    total = sum(m.astype(np.int32) for m in masks)
    assert np.all(total == 1)


def test_observed_uncertain_cell_is_neither_free_nor_occupied():
    pm = PartialMap(_config(free_threshold=-0.2, occupied_threshold=0.2))
    cell = pm.world_to_cell(0.0, 0.0)
    assert cell is not None
    pm.log_odds[cell] = 0.0  # strictly between free_threshold and occupied_threshold
    pm.observed[cell] = True
    channels = pm.channels()
    assert channels.observed_uncertain[cell]
    assert not channels.free[cell]
    assert not channels.occupied[cell]
    assert not channels.unknown[cell]


def test_unknown_is_not_computed_as_one_minus_occupied():
    """Regression guard for the spec's explicit anti-pattern: UNKNOWN must
    equal NOT-observed, never 1-occupied (which would silently merge FREE
    into UNKNOWN)."""
    pm = PartialMap(_config())
    pm.integrate_beam((0.0, 0.0), angle_mission=0.0, range_m=3.0, range_max=10.0)
    channels = pm.channels()
    naive_unknown = ~channels.occupied
    assert not np.array_equal(channels.unknown, naive_unknown)
    assert channels.free.sum() > 0


def test_ray_extending_past_grid_boundary_is_clipped_not_crashing():
    pm = PartialMap(_config(mission_size_cells=10))  # grid spans roughly [-2.5, 2.5]
    # Sensor near the map edge (still in-bounds), beam pointed further outward
    # than the grid actually extends.
    pm.integrate_beam((2.0, 0.0), angle_mission=0.0, range_m=8.0, range_max=10.0)
    channels = pm.channels()
    assert channels.occupied.sum() + channels.free.sum() > 0
    assert not np.any(channels.occupied & channels.free)


def test_hit_endpoint_outside_map_bounds_never_marks_occupied():
    """Code review fix: the PHYSICAL hit endpoint (5m away) lies well
    outside a map that only spans roughly [-2.5, 2.5]. Clipping the ray to
    the grid boundary and marking that boundary cell OCCUPIED would
    fabricate a wall at the map edge for an obstacle that is actually
    further out, entirely outside the map -- must instead only mark FREE up
    to the boundary, exactly like the max-range/no-hit case."""
    pm = PartialMap(_config(mission_size_cells=10))  # grid spans roughly [-2.5, 2.5]
    pm.integrate_beam((0.0, 0.0), angle_mission=0.0, range_m=5.0, range_max=10.0)
    channels = pm.channels()
    assert channels.occupied.sum() == 0
    assert channels.free.sum() > 0


def test_hit_endpoint_inside_map_bounds_still_marks_occupied():
    """Companion to the above: a hit whose endpoint genuinely IS inside the
    map must still be marked occupied -- the fix must not suppress
    legitimate in-bounds occupied endpoints."""
    pm = PartialMap(_config(mission_size_cells=20))  # grid spans roughly [-5, 5]
    pm.integrate_beam((0.0, 0.0), angle_mission=0.0, range_m=3.0, range_max=10.0)
    channels = pm.channels()
    assert channels.occupied.sum() == 1


def test_sensor_origin_outside_grid_does_not_crash():
    pm = PartialMap(_config(mission_size_cells=10))
    pm.integrate_beam((1000.0, 1000.0), angle_mission=0.0, range_m=3.0, range_max=10.0)
    # No crash, and since the whole ray is outside the grid, nothing observed.
    assert pm.observed.sum() == 0


def test_log_odds_saturates_within_configured_bounds():
    pm = PartialMap(_config(log_odds_min=-1.0, log_odds_max=1.0, occupied_log_odds_delta=0.85))
    for _ in range(20):
        pm.integrate_beam((0.0, 0.0), angle_mission=0.0, range_m=1.0, range_max=10.0)
    assert pm.log_odds.max() <= 1.0 + 1e-6
    assert pm.log_odds.min() >= -1.0 - 1e-6


def test_integrate_scan_rejects_mismatched_lengths():
    pm = PartialMap(_config())
    with pytest.raises(ValueError):
        pm.integrate_scan((0.0, 0.0), np.zeros(3), np.zeros(5), range_max=10.0)


def test_inflated_channel_always_includes_occupied():
    pm = PartialMap(_config(inflation_radius_m=0.5))
    pm.integrate_beam((0.0, 0.0), angle_mission=0.0, range_m=3.0, range_max=10.0)
    channels = pm.channels()
    assert np.all(channels.inflated[channels.occupied])


def test_inflated_channel_grows_with_inflation_radius():
    small = PartialMap(_config(inflation_radius_m=0.0))
    large = PartialMap(_config(inflation_radius_m=1.5))
    for pm in (small, large):
        pm.integrate_beam((0.0, 0.0), angle_mission=0.0, range_m=3.0, range_max=10.0)
    assert large.channels().inflated.sum() > small.channels().inflated.sum()


def test_zero_inflation_radius_makes_inflated_equal_occupied():
    pm = PartialMap(_config(inflation_radius_m=0.0))
    pm.integrate_beam((0.0, 0.0), angle_mission=0.0, range_m=3.0, range_max=10.0)
    channels = pm.channels()
    assert np.array_equal(channels.inflated, channels.occupied)


# ---------------------------------------------------------------------------
# item 7: PartialMap concurrency -- a live executor's scan CALLBACK (a
# background ROS spin thread) writes via integrate_scan()/record_visit()
# while the option/main thread concurrently reads via channels()/
# observed_count(). Stress-tests the lock added to PartialMap for exactly
# this: no torn reads, no crash, no lost updates, across many concurrent
# writer/reader iterations.
# ---------------------------------------------------------------------------

def test_concurrent_scan_writes_and_channel_reads_never_produce_a_torn_partition():
    import threading

    pm = PartialMap(_config(mission_size_cells=60))
    n_beams = 72
    angles = np.linspace(-math.pi, math.pi, n_beams, endpoint=False)
    stop = threading.Event()
    errors = []

    def _writer():
        rng = np.random.RandomState(0)
        while not stop.is_set():
            ranges = rng.uniform(0.5, 9.5, size=n_beams)
            try:
                pm.integrate_scan((0.0, 0.0), angles, ranges, range_max=10.0)
                pm.record_visit(0.0, 0.0, step=1)
            except Exception as e:  # noqa: BLE001 -- captured for the assertion below, not swallowed
                errors.append(e)
                return

    def _reader():
        while not stop.is_set():
            try:
                channels = pm.channels()
                # Exhaustive 4-way partition invariant (module docstring):
                # every cell belongs to EXACTLY one of these four -- if a
                # reader ever observes a torn snapshot (some arrays from
                # before a concurrent write, some from after), this sum
                # would not equal exactly 1 everywhere.
                partition_sum = (
                    channels.occupied.astype(np.int32) + channels.free.astype(np.int32)
                    + channels.unknown.astype(np.int32) + channels.observed_uncertain.astype(np.int32)
                )
                if not np.all(partition_sum == 1):
                    errors.append(AssertionError("torn 4-way map partition observed under concurrent access"))
                    return
                pm.observed_count()
            except Exception as e:  # noqa: BLE001 -- captured for the assertion below, not swallowed
                errors.append(e)
                return

    threads = [threading.Thread(target=_writer) for _ in range(2)] + [threading.Thread(target=_reader) for _ in range(3)]
    for t in threads:
        t.start()
    stop.wait(1.0)
    stop.set()
    for t in threads:
        t.join(timeout=5.0)
        assert not t.is_alive(), "a writer/reader thread failed to terminate"

    assert errors == [], f"concurrent access produced errors: {errors}"
    # Sanity: the writers actually ran and produced real map content --
    # this test would trivially "pass" with zero real concurrency exercised
    # if integrate_scan silently no-op'd.
    assert pm.observed_count() > 0
