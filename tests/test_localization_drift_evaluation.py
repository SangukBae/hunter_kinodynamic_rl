"""Phase 5/6 localization backend swap + drift evaluation (plan section
4/10.5). Exercises the wheel+IMU and LiDAR-odometry backends (pure Python,
no rclpy) and the Global-metrics drift-sensitivity helper together --
confirms navigation code only ever depends on the LocalizationBackend
Protocol (latest_pose()/pose_at()), never on which concrete backend is
active, and that noise-free ground truth never leaks through a noisy
profile's confidence/covariance."""

import math

import numpy as np
import pytest

from hunter_kinodynamic_rl.evaluation.global_metrics import localization_drift_sensitivity
from hunter_kinodynamic_rl.navigation.localization.interface import LocalizationBackend, is_pose_usable
from hunter_kinodynamic_rl.navigation.localization.lidar_odom_backend import (
    LidarOdomLocalizationBackend, LidarOdomNoiseModel,
)
from hunter_kinodynamic_rl.navigation.localization.wheel_imu_backend import (
    WheelImuLocalizationBackend, WheelImuNoiseModel,
)


def test_wheel_imu_backend_satisfies_localization_backend_protocol():
    backend = WheelImuLocalizationBackend()
    assert isinstance(backend, LocalizationBackend)


def test_lidar_odom_backend_satisfies_localization_backend_protocol():
    backend = LidarOdomLocalizationBackend()
    assert isinstance(backend, LocalizationBackend)


def test_wheel_imu_integrate_before_initialize_raises():
    backend = WheelImuLocalizationBackend()
    with pytest.raises(RuntimeError):
        backend.integrate(1.0, 0.0, 0.1, 1.0)


def test_wheel_imu_dead_reckoning_moves_pose_forward():
    backend = WheelImuLocalizationBackend()
    backend.initialize(0.0, 0.0, 0.0, 0.0)
    for i in range(10):
        backend.integrate(v_mps=1.0, yaw_rate_rps=0.0, dt_sec=0.1, stamp_sec=(i + 1) * 0.1)
    pose = backend.latest_pose()
    assert pose.x == pytest.approx(1.0, abs=1e-6)
    assert pose.valid


def test_wheel_imu_confidence_decays_as_distance_accumulates():
    """Localization drift sensitivity requires confidence to actually
    degrade as accumulated drift grows -- never a flat/fabricated value."""
    backend = WheelImuLocalizationBackend(WheelImuNoiseModel(position_process_noise_m2_per_m=0.05))
    backend.initialize(0.0, 0.0, 0.0, 0.0)
    confidences = []
    for i in range(50):
        pose = backend.integrate(v_mps=1.0, yaw_rate_rps=0.0, dt_sec=0.1, stamp_sec=(i + 1) * 0.1)
        confidences.append(pose.confidence)
    assert confidences[0] > confidences[-1]
    assert confidences[-1] < 1.0
    assert all(math.isfinite(c) and 0.0 <= c <= 1.0 for c in confidences)


def test_wheel_imu_non_finite_reading_is_dropped_not_corrupting_state():
    backend = WheelImuLocalizationBackend()
    backend.initialize(1.0, 2.0, 0.0, 0.0)
    before = backend.latest_pose()
    backend.integrate(v_mps=float("nan"), yaw_rate_rps=0.0, dt_sec=0.1, stamp_sec=1.0)
    after = backend.latest_pose()
    assert (after.x, after.y) == (before.x, before.y)


def test_lidar_odom_confidence_tracks_match_quality():
    backend = LidarOdomLocalizationBackend()
    backend.initialize(0.0, 0.0, 0.0, 0.0)
    good = backend.integrate_relative_transform(1.0, 0.0, 0.0, 1.0, match_quality=1.0)
    bad = backend.integrate_relative_transform(1.0, 0.0, 0.0, 2.0, match_quality=0.1)
    assert good.confidence > bad.confidence


def test_lidar_odom_degenerate_match_still_composes_pose_but_low_confidence():
    backend = LidarOdomLocalizationBackend(LidarOdomNoiseModel(confidence_floor=0.0))
    backend.initialize(0.0, 0.0, 0.0, 0.0)
    pose = backend.integrate_relative_transform(0.5, 0.0, 0.0, 1.0, match_quality=0.0)
    assert pose.confidence == pytest.approx(0.0)
    assert not pose.valid


def test_pose_usable_gate_rejects_low_confidence_regardless_of_backend():
    """The SAME is_pose_usable gate mapping/mission code already uses must
    reject a degraded reading from EITHER backend identically -- this is
    what makes "swap the backend without touching navigation core" true."""
    wheel = WheelImuLocalizationBackend(WheelImuNoiseModel(position_process_noise_m2_per_m=5.0))
    wheel.initialize(0.0, 0.0, 0.0, 0.0)
    for i in range(20):
        wheel.integrate(v_mps=1.0, yaw_rate_rps=0.0, dt_sec=0.1, stamp_sec=(i + 1) * 0.1)
    wheel_pose = wheel.latest_pose()

    lidar = LidarOdomLocalizationBackend()
    lidar.initialize(0.0, 0.0, 0.0, 0.0)
    lidar_pose = lidar.integrate_relative_transform(1.0, 0.0, 0.0, 1.0, match_quality=0.02)

    assert not is_pose_usable(wheel_pose, now_sec=2.0, timeout_sec=0.5, min_confidence=0.9)
    assert not is_pose_usable(lidar_pose, now_sec=1.0, timeout_sec=0.5, min_confidence=0.9)


def test_drift_sweep_produces_monotonic_sensitivity_as_noise_grows():
    """Simulates the Phase A -> B/C sweep (plan 10.5): identical navigation
    architecture, only the localization backend/noise magnitude changes.
    Higher process noise must never produce a LOWER measured degradation
    than a smaller one over the same dead-reckoning distance."""
    def _final_confidence(noise_scale: float) -> float:
        backend = WheelImuLocalizationBackend(WheelImuNoiseModel(position_process_noise_m2_per_m=noise_scale))
        backend.initialize(0.0, 0.0, 0.0, 0.0)
        pose = backend.latest_pose()
        for i in range(30):
            pose = backend.integrate(v_mps=1.0, yaw_rate_rps=0.1, dt_sec=0.1, stamp_sec=(i + 1) * 0.1)
        return pose.confidence

    low = _final_confidence(0.001)
    high = _final_confidence(0.2)
    assert high < low

    ideal_metrics = {"final_goal_success_rate": 0.9}
    low_noise_metrics = {"final_goal_success_rate": 0.9 * low}
    high_noise_metrics = {"final_goal_success_rate": 0.9 * high}
    sens_low = localization_drift_sensitivity(ideal_metrics, low_noise_metrics)
    sens_high = localization_drift_sensitivity(ideal_metrics, high_noise_metrics)
    assert sens_high >= sens_low
