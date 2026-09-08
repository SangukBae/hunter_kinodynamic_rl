"""Fixed-benchmark dynamics/sensor override application (section P1-3):
supported keys must produce a REAL, measurable RobotConfig/control-loop
change; unsupported keys must raise, never be silently ignored."""

import pytest

from hunter_kinodynamic_rl.config.schema import RobotConfig
from hunter_kinodynamic_rl.env.randomization.domain_randomizer import (
    apply_dynamics_overrides, check_sensor_overrides_supported, command_latency_steps,
)


def make_robot(**overrides) -> RobotConfig:
    defaults = dict(
        name="hunter_se", wheelbase_m=0.547696, track_width_m=0.503404,
        wheel_radius_m=0.136, length_m=0.76, width_m=0.4, height_m=0.14,
        mass_kg=42.0, steering_limit_deg=21.58, max_forward_speed_mps=2.0,
        min_forward_speed_mps=0.0, accel_limit_mps2=6.0, brake_decel_mps2=6.0,
        steering_rate_deg_s=200.0, speed_lag_tau_sec=0.0, collision_radius_m=0.45,
    )
    defaults.update(overrides)
    return RobotConfig(**defaults)


def test_friction_scale_override_changes_accel_and_brake():
    robot = make_robot()
    overridden = apply_dynamics_overrides(robot, {"friction_scale": 0.5})
    assert overridden.accel_limit_mps2 == pytest.approx(3.0)
    assert overridden.brake_decel_mps2 == pytest.approx(3.0)


def test_empty_overrides_is_identity():
    robot = make_robot()
    assert apply_dynamics_overrides(robot, {}) == robot


def test_unsupported_dynamics_key_raises_not_silently_ignored():
    robot = make_robot()
    with pytest.raises(ValueError, match="unsupported"):
        apply_dynamics_overrides(robot, {"steering_delay_sec": 0.15})


def test_command_latency_override_does_not_raise_as_a_robotconfig_key():
    """command_latency_sec is a SUPPORTED override key, handled separately
    by command_latency_steps() (it's a control-loop property, not a
    RobotConfig field) -- apply_dynamics_overrides must not reject it."""
    robot = make_robot()
    apply_dynamics_overrides(robot, {"command_latency_sec": 0.1})  # no raise


def test_command_latency_steps_computation():
    assert command_latency_steps({"command_latency_sec": 0.3}, time_delta_sec=0.1) == 3
    assert command_latency_steps({}, time_delta_sec=0.1) == 0


def test_command_latency_negative_raises():
    with pytest.raises(ValueError):
        command_latency_steps({"command_latency_sec": -0.1}, time_delta_sec=0.1)


def test_sensor_overrides_empty_is_supported():
    check_sensor_overrides_supported({})  # no raise


def test_fixed_sensor_and_localization_axes_are_explicitly_whitelisted():
    check_sensor_overrides_supported(
        {"lidar_range_noise_std_m": 0.05, "lidar_dropout_prob": 0.02},
        {"odometry_noise_std": 0.01},
    )
    with pytest.raises(ValueError, match="unsupported fixed"):
        check_sensor_overrides_supported({"invented_noise": 0.05})
    with pytest.raises(ValueError, match="probabilities"):
        check_sensor_overrides_supported({"lidar_dropout_prob": 1.5})
