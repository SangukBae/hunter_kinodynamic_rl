"""Coverage for the GT/noisy observation diagnostics wire format
(requirement 5): env/simulation/sensor_diagnostics.py.

Pure-function module -- no ROS needed, runs on a bare host checkout.
"""

import dataclasses

import pytest

from hunter_kinodynamic_rl.env.simulation import sensor_diagnostics as sd


def _sample(**overrides) -> sd.SensorDiagnostics:
    base = dict(
        step_id=5, valid=True, reset_generation=2, episode_id=1710, sim_timestamp_sec=12.5,
        gt_x=1.0, gt_y=2.0, gt_yaw=0.1, noisy_x=1.05, noisy_y=1.98, noisy_yaw=0.11,
        gt_v_mps=0.5, gt_yaw_rate_radps=0.01, gt_steering_rad=0.02,
        noisy_v_mps=0.52, noisy_yaw_rate_radps=0.015, noisy_steering_rad=0.025,
        drift_x_m=0.03, drift_y_m=-0.02, drift_yaw_rad=0.005,
        localization_latency_steps=2, lidar_beam_count=80, lidar_dropout_count=3,
        lidar_perturbation_mean_m=0.04, lidar_perturbation_max_m=0.3,
        invalid_reason=int(sd.DiagnosticsInvalidReason.NONE),
    )
    base.update(overrides)
    return sd.SensorDiagnostics(**base)


# --------------------------------------------------------------- encode/decode
def test_encode_decode_round_trips_every_field():
    d = _sample()
    decoded = sd.decode(sd.encode(d))
    assert decoded == d


def test_encode_length_matches_the_fixed_wire_length():
    assert len(sd.encode(_sample())) == 27


def test_decode_rejects_wrong_length():
    with pytest.raises(ValueError, match="length"):
        sd.decode([1.0, 2.0, 3.0])


def test_decode_rejects_schema_version_mismatch():
    payload = sd.encode(_sample())
    payload[0] = 999.0
    with pytest.raises(ValueError, match="schema_version"):
        sd.decode(payload)


def test_invalid_helper_produces_a_not_valid_record_with_nan_fields():
    d = sd.invalid(step_id=3, reset_generation=1, episode_id=42, sim_timestamp_sec=5.0,
                    reason=sd.DiagnosticsInvalidReason.COMPUTATION_EXCEPTION)
    assert d.valid is False
    assert d.step_id == 3
    assert d.reset_generation == 1
    assert d.episode_id == 42
    assert d.invalid_reason == int(sd.DiagnosticsInvalidReason.COMPUTATION_EXCEPTION)
    import math
    assert math.isnan(d.gt_x)
    assert math.isnan(d.noisy_x)


def test_invalid_round_trips_through_encode_decode():
    d = sd.invalid(step_id=0, reset_generation=7, episode_id=1, sim_timestamp_sec=float("nan"))
    decoded = sd.decode(sd.encode(d))
    assert decoded.valid is False
    assert decoded.step_id == 0
    assert decoded.reset_generation == 7


def test_reset_marker_convention_step_id_zero_is_a_normal_valid_value():
    """Unlike risk_telemetry, step_id=0 here is just this episode's own
    /reset snapshot -- a perfectly normal, valid=True record, not a
    dedicated marker type."""
    d = _sample(step_id=0)
    decoded = sd.decode(sd.encode(d))
    assert decoded.step_id == 0
    assert decoded.valid is True


# ------------------------------------------------------- lidar_perturbation_stats
def test_lidar_perturbation_stats_identical_arrays_is_all_zero():
    gt = [1.0, 2.0, 3.0, 4.0]
    dropout, mean, mx = sd.lidar_perturbation_stats(gt, gt, max_range_m=10.0)
    assert dropout == 0
    assert mean == pytest.approx(0.0)
    assert mx == pytest.approx(0.0)


def test_lidar_perturbation_stats_computes_mean_and_max_deviation():
    gt = [1.0, 2.0, 3.0, 4.0]
    noisy = [1.1, 2.0, 3.5, 4.0]
    dropout, mean, mx = sd.lidar_perturbation_stats(gt, noisy, max_range_m=10.0)
    assert dropout == 0
    assert mean == pytest.approx((0.1 + 0.0 + 0.5 + 0.0) / 4.0)
    assert mx == pytest.approx(0.5)


def test_lidar_perturbation_stats_counts_a_beam_as_dropped_only_when_noisy_is_max_range_and_gt_is_not():
    max_range = 10.0
    gt = [1.0, 5.0, max_range, 3.0]  # index 2 was ALREADY at max_range in ground truth
    noisy = [max_range, 5.0, max_range, max_range]  # index 0 and 3 newly saturate to max_range
    dropout, _mean, _mx = sd.lidar_perturbation_stats(gt, noisy, max_range_m=max_range)
    assert dropout == 2  # index 2 must NOT count (gt was already at max_range)


def test_lidar_perturbation_stats_empty_arrays_returns_zero():
    assert sd.lidar_perturbation_stats([], [], max_range_m=10.0) == (0, 0.0, 0.0)


def test_lidar_perturbation_stats_rejects_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        sd.lidar_perturbation_stats([1.0, 2.0], [1.0], max_range_m=10.0)


def test_diagnostics_is_frozen_dataclass_immutable():
    d = _sample()
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.gt_x = 99.0
