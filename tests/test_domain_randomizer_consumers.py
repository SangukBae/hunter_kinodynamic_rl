"""Regression tests for the P1-10 domain-randomization CONSUMERS
(env/randomization/domain_randomizer.py's apply_lidar_noise/
apply_odometry_noise/should_drop_sensor_frame/steering_lag_alpha) -- the
sampling side (sample_draw) already had coverage; these functions previously
had NONE, matching the fact that nothing in environment_node.py called them
either (a real, disclosed gap this round closes)."""

import numpy as np
import pytest

from hunter_kinodynamic_rl.config.schema import RobotConfig
from hunter_kinodynamic_rl.env.randomization.domain_randomizer import (
    GAZEBO_APPLIED_FIELDS, MODEL_ONLY_FIELDS, OBSERVATION_ONLY_FIELDS, RandomizationDraw, apply_lidar_noise,
    apply_odometry_noise, classify_draw_fields, rate_limit_speed, should_drop_sensor_frame, steering_lag_alpha,
)


def _robot(accel=6.0, brake=6.0) -> RobotConfig:
    return RobotConfig(
        name="hunter_se", wheelbase_m=0.547696, track_width_m=0.503404,
        wheel_radius_m=0.136, length_m=0.76, width_m=0.4, height_m=0.14,
        mass_kg=42.0, steering_limit_deg=21.58, max_forward_speed_mps=2.0,
        min_forward_speed_mps=0.0, accel_limit_mps2=accel, brake_decel_mps2=brake,
        steering_rate_deg_s=200.0, speed_lag_tau_sec=0.0,
    )


def _no_noise_draw() -> RandomizationDraw:
    return RandomizationDraw()  # every field at its "no randomization" default


# --------------------------------------------------------------- LiDAR noise
def test_lidar_noise_is_identity_with_a_zero_draw():
    ranges = np.full(80, 5.0, dtype=np.float32)
    rng = np.random.RandomState(0)
    out = apply_lidar_noise(ranges, _no_noise_draw(), rng, max_range_m=10.0)
    np.testing.assert_array_equal(out, ranges)


def test_lidar_noise_perturbs_ranges_when_std_positive():
    ranges = np.full(80, 5.0, dtype=np.float32)
    draw = RandomizationDraw(lidar_range_noise_std_m=0.5)
    rng = np.random.RandomState(0)
    out = apply_lidar_noise(ranges, draw, rng, max_range_m=10.0)
    assert not np.array_equal(out, ranges)
    assert out.shape == ranges.shape


def test_lidar_noise_clips_to_valid_range():
    ranges = np.full(10, 9.9, dtype=np.float32)
    draw = RandomizationDraw(lidar_range_noise_std_m=5.0)  # huge noise, must still clip
    rng = np.random.RandomState(0)
    out = apply_lidar_noise(ranges, draw, rng, max_range_m=10.0)
    assert np.all(out >= 0.0) and np.all(out <= 10.0)


def test_lidar_dropout_sets_dropped_beams_to_max_range():
    ranges = np.full(1000, 2.0, dtype=np.float32)
    draw = RandomizationDraw(lidar_dropout_prob=1.0)  # every beam drops
    rng = np.random.RandomState(0)
    out = apply_lidar_noise(ranges, draw, rng, max_range_m=10.0)
    np.testing.assert_array_equal(out, np.full(1000, 10.0, dtype=np.float32))


def test_lidar_dropout_zero_never_drops():
    ranges = np.full(1000, 2.0, dtype=np.float32)
    draw = RandomizationDraw(lidar_dropout_prob=0.0)
    rng = np.random.RandomState(0)
    out = apply_lidar_noise(ranges, draw, rng, max_range_m=10.0)
    np.testing.assert_array_equal(out, ranges)


def test_lidar_noise_is_deterministic_given_seeded_rng():
    ranges = np.full(80, 5.0, dtype=np.float32)
    draw = RandomizationDraw(lidar_range_noise_std_m=0.3, lidar_dropout_prob=0.1)
    out1 = apply_lidar_noise(ranges, draw, np.random.RandomState(42), max_range_m=10.0)
    out2 = apply_lidar_noise(ranges, draw, np.random.RandomState(42), max_range_m=10.0)
    np.testing.assert_array_equal(out1, out2)


# ----------------------------------------------------------- odometry noise
def test_odometry_noise_is_identity_with_zero_std():
    v, yaw_rate = apply_odometry_noise(1.5, 0.2, _no_noise_draw(), np.random.RandomState(0))
    assert (v, yaw_rate) == (1.5, 0.2)


def test_odometry_noise_perturbs_when_std_positive():
    draw = RandomizationDraw(odometry_noise_std=0.2)
    v, yaw_rate = apply_odometry_noise(1.5, 0.2, draw, np.random.RandomState(1))
    assert v != 1.5 or yaw_rate != 0.2


# ------------------------------------------------------------ sensor dropout
def test_should_drop_sensor_frame_never_drops_at_zero_prob():
    draw = RandomizationDraw(sensor_frame_drop_prob=0.0)
    rng = np.random.RandomState(0)
    assert all(not should_drop_sensor_frame(draw, rng) for _ in range(1000))


def test_should_drop_sensor_frame_always_drops_at_prob_one():
    draw = RandomizationDraw(sensor_frame_drop_prob=1.0)
    rng = np.random.RandomState(0)
    assert all(should_drop_sensor_frame(draw, rng) for _ in range(100))


# ------------------------------------------------------------- steering lag
def test_steering_lag_alpha_is_one_with_zero_delay():
    assert steering_lag_alpha(steering_delay_sec=0.0, time_delta_sec=0.1) == pytest.approx(1.0)


def test_steering_lag_alpha_decreases_with_longer_delay():
    fast = steering_lag_alpha(steering_delay_sec=0.05, time_delta_sec=0.1)
    slow = steering_lag_alpha(steering_delay_sec=0.5, time_delta_sec=0.1)
    assert slow < fast
    assert 0.0 <= slow <= 1.0
    assert 0.0 <= fast <= 1.0


def test_steering_lag_alpha_clamped_to_one_when_tau_below_one_tick():
    # tau < time_delta_sec -> the actuator would reach target FASTER than
    # one tick anyway -- alpha must clamp at 1.0, not exceed it.
    assert steering_lag_alpha(steering_delay_sec=0.01, time_delta_sec=0.1) == pytest.approx(1.0)


# ------------------------------------------------------------- speed rate limit (P1-10 extension)
def test_rate_limit_speed_reaches_target_instantly_when_within_one_ticks_authority():
    robot = _robot(accel=10.0, brake=10.0)  # 10 m/s^2 * 0.1s = 1.0 m/s of authority this tick
    assert rate_limit_speed(0.0, 0.5, robot, dt_sec=0.1) == pytest.approx(0.5)


def test_rate_limit_speed_clips_to_the_accel_limit_when_speeding_up():
    robot = _robot(accel=1.0, brake=10.0)  # 1.0 m/s^2 * 0.1s = 0.1 m/s of authority
    out = rate_limit_speed(0.0, 2.0, robot, dt_sec=0.1)
    assert out == pytest.approx(0.1)


def test_rate_limit_speed_uses_brake_decel_when_slowing_down():
    robot = _robot(accel=10.0, brake=0.5)  # 0.5 m/s^2 * 0.1s = 0.05 m/s of authority braking
    out = rate_limit_speed(1.0, 0.0, robot, dt_sec=0.1)
    assert out == pytest.approx(0.95)


def test_rate_limit_speed_never_overshoots_the_target():
    robot = _robot(accel=100.0, brake=100.0)  # huge authority -- should snap exactly to target
    assert rate_limit_speed(0.0, 1.23, robot, dt_sec=0.1) == pytest.approx(1.23)


def test_rate_limit_speed_is_monotonic_progress_toward_a_lower_target():
    robot = _robot(accel=2.0, brake=2.0)
    v = 2.0
    for _ in range(5):
        new_v = rate_limit_speed(v, 0.0, robot, dt_sec=0.1)
        assert new_v <= v
        assert new_v >= 0.0
        v = new_v


# ------------------------------------------------------ draw field classification (P1-10 extension)
def test_classify_draw_fields_covers_every_randomizationdraw_field_exactly_once():
    classification = classify_draw_fields(_no_noise_draw())
    assert set(classification) == set(RandomizationDraw.__dataclass_fields__)
    assert MODEL_ONLY_FIELDS | GAZEBO_APPLIED_FIELDS | OBSERVATION_ONLY_FIELDS == set(classification)
    # No field is double-classified.
    assert not (MODEL_ONLY_FIELDS & GAZEBO_APPLIED_FIELDS)
    assert not (MODEL_ONLY_FIELDS & OBSERVATION_ONLY_FIELDS)
    assert not (GAZEBO_APPLIED_FIELDS & OBSERVATION_ONLY_FIELDS)


def test_classify_draw_fields_mass_and_wheel_radius_are_model_only():
    classification = classify_draw_fields(_no_noise_draw())
    assert classification["mass_scale"] == "model_only"
    assert classification["wheel_radius_scale"] == "model_only"


def test_classify_draw_fields_friction_and_velocity_response_are_gazebo_applied():
    classification = classify_draw_fields(_no_noise_draw())
    assert classification["friction_scale"] == "gazebo_applied"
    assert classification["velocity_response_scale"] == "gazebo_applied"


def test_classify_draw_fields_sensor_axes_are_observation_only():
    classification = classify_draw_fields(_no_noise_draw())
    for name in ("lidar_range_noise_std_m", "lidar_dropout_prob", "odometry_noise_std",
                 "sensor_frame_drop_prob"):
        assert classification[name] == "observation_only"
