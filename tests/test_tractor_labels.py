"""Scripted static/dynamic/boundary and severity label fixtures."""

import numpy as np
import pytest

from hunter_kinodynamic_rl.risk.tractor_labels import (
    ObstacleTrackBatch, generate_candidate_labels,
)
from hunter_kinodynamic_rl.rl.replay.sequence_schema import (
    CAUSE_BOUNDARY_UNKNOWN, CAUSE_DYNAMIC, CAUSE_STATIC,
)


def _generate(candidate_xy, speed, tracks, radii, causes, half_extent=5.0):
    samples = candidate_xy.shape[1]
    return generate_candidate_labels(
        timestamps_sec=np.arange(1, samples + 1) * 0.1,
        candidate_xy_m=candidate_xy,
        candidate_speed_mps=speed,
        candidate_valid=np.ones(speed.shape, dtype=bool),
        candidate_horizon_sec=np.full(candidate_xy.shape[0], samples * 0.1),
        obstacle_tracks=ObstacleTrackBatch(
            tracks, np.asarray(radii), np.asarray(causes),
            np.ones(tracks.shape[:2], dtype=bool),
        ),
        ego_radius_m=0.5, world_half_extent_m=half_extent,
        goal_xy_m=np.asarray([4.0, 0.0]), dt_out_sec=0.2,
        horizon_steps=3, brake_decel_mps2=2.0,
    )


def test_static_event_time_cause_and_censor_tuple():
    xy = np.asarray([[[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.5, 0.0]]])
    speed = np.ones((1, 4))
    static = np.asarray([[[2.0, 0.0]] * 4])
    labels = _generate(xy, speed, static, [0.5], [CAUSE_STATIC])
    assert labels.event_observed.tolist() == [True]
    assert labels.event_step.tolist() == [1]
    assert labels.event_cause.tolist() == [CAUSE_STATIC]
    assert labels.censor_step.tolist() == [1]


def test_dynamic_labels_use_realized_track_positions_not_a_velocity_command():
    xy = np.asarray([[[0.0, 0.0], [0.2, 0.0], [0.4, 0.0], [0.6, 0.0]]])
    speed = np.full((1, 4), 0.5)
    # The realized track crosses at sample 3. No commanded velocity is an
    # input to the label API, so a waypoint command cannot override truth.
    dynamic = np.asarray([[[3.0, 2.0], [2.0, 1.0], [0.4, 0.8], [0.6, 0.0]]])
    labels = _generate(xy, speed, dynamic, [0.2], [CAUSE_DYNAMIC])
    assert labels.event_observed[0]
    assert labels.event_step[0] == 1
    assert labels.event_cause[0] == CAUSE_DYNAMIC


def test_boundary_wins_exact_collision_tie_over_dynamic_and_static():
    xy = np.asarray([[[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.5, 0.0]]])
    speed = np.ones((1, 4))
    # At final sample, half_extent=2 gives boundary margin 0. Dynamic and
    # static tracks are also tangent; priority must be boundary > dynamic > static.
    tracks = np.asarray([
        [[9.0, 9.0], [9.0, 9.0], [9.0, 9.0], [2.5, 0.0]],
        [[9.0, 9.0], [9.0, 9.0], [9.0, 9.0], [2.5, 0.0]],
    ])
    labels = _generate(
        xy, speed, tracks, [0.5, 0.5], [CAUSE_STATIC, CAUSE_DYNAMIC], half_extent=2.0
    )
    assert labels.event_cause[0] == CAUSE_BOUNDARY_UNKNOWN


def test_stopping_margin_uses_speed_at_matching_closest_time():
    xy = np.asarray([[[0.0, 0.0], [0.2, 0.0], [0.4, 0.0], [0.6, 0.0]]])
    speed = np.asarray([[0.1, 0.2, 0.3, 1.0]])
    static = np.asarray([[[3.0, 0.0]] * 4])
    labels = _generate(xy, speed, static, [0.5], [CAUSE_STATIC])
    # Bin 1 contains t=0.3,0.4; t=0.4 is closest and has v=1.0.
    expected = labels.clearance_m[0, 1] - 1.0 ** 2 / (2.0 * 2.0)
    assert labels.stopping_margin_m[0, 1] == pytest.approx(expected)


def test_no_event_is_right_censored_at_true_horizon():
    xy = np.asarray([[[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0]]])
    speed = np.zeros((1, 4))
    no_obstacles = np.empty((0, 4, 2))
    labels = _generate(xy, speed, no_obstacles, [], [])
    assert not labels.event_observed[0]
    assert (labels.event_step[0], labels.event_cause[0], labels.censor_step[0]) == (-1, -1, 1)
