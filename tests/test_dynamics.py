import math

import pytest

from hunter_kinodynamic_rl.config.schema import DynamicsConfig, RobotConfig
from hunter_kinodynamic_rl.dynamics import ackermann_rollout, bicycle_model, stopping_model
from hunter_kinodynamic_rl.robot.hunter_se import HunterSE
from hunter_kinodynamic_rl.robot.interface import VehicleState
from hunter_kinodynamic_rl.robot.limits import curvature_to_steering, steering_to_curvature


def make_robot() -> RobotConfig:
    return RobotConfig(
        name="hunter_se", wheelbase_m=0.547696, track_width_m=0.503404,
        wheel_radius_m=0.136, length_m=0.76, width_m=0.4, height_m=0.14,
        mass_kg=42.0, steering_limit_deg=21.58, max_forward_speed_mps=2.0,
        min_forward_speed_mps=0.0, accel_limit_mps2=6.0, brake_decel_mps2=6.0,
        steering_rate_deg_s=200.0, speed_lag_tau_sec=0.0,
    )


def test_straight_rollout_no_lateral_drift():
    robot = make_robot()
    dyn = DynamicsConfig(horizon_sec=1.0, dt_sec=0.1, model_actuator_lag=False)
    state = VehicleState(v=1.0, steering=0.0)
    out = ackermann_rollout.rollout_constant_target(state, 1.0, 0.0, robot, dyn)
    final = out.final_state
    assert final.y == pytest.approx(0.0, abs=1e-9)
    assert final.x == pytest.approx(1.0, abs=1e-6)
    assert final.yaw == pytest.approx(0.0, abs=1e-9)


def test_positive_steering_curves_left():
    robot = make_robot()
    dyn = DynamicsConfig(horizon_sec=1.0, dt_sec=0.05, model_actuator_lag=False)
    state = VehicleState(v=1.0, steering=0.0)
    out = ackermann_rollout.rollout_constant_target(state, 1.0, math.radians(15.0), robot, dyn)
    final = out.final_state
    assert final.y > 0.0
    assert final.yaw > 0.0


def test_negative_steering_curves_right():
    robot = make_robot()
    dyn = DynamicsConfig(horizon_sec=1.0, dt_sec=0.05, model_actuator_lag=False)
    state = VehicleState(v=1.0, steering=0.0)
    out = ackermann_rollout.rollout_constant_target(state, 1.0, math.radians(-15.0), robot, dyn)
    final = out.final_state
    assert final.y < 0.0
    assert final.yaw < 0.0


def test_zero_velocity_stays_put():
    robot = make_robot()
    dyn = DynamicsConfig(horizon_sec=1.0, dt_sec=0.1, model_actuator_lag=False)
    state = VehicleState(v=0.0, steering=0.3)
    out = ackermann_rollout.rollout_constant_target(state, 0.0, 0.3, robot, dyn)
    final = out.final_state
    assert final.x == pytest.approx(0.0, abs=1e-9)
    assert final.y == pytest.approx(0.0, abs=1e-9)


def test_steering_saturation_clamped_by_robot_limits():
    robot = make_robot()
    hunter = HunterSE(robot)
    over_limit = math.radians(45.0)
    clamped = hunter.clamp_steering(over_limit)
    assert clamped == pytest.approx(robot.steering_limit_rad)


def test_curvature_steering_roundtrip():
    robot = make_robot()
    for deg in (-20.0, -5.0, 0.0, 5.0, 20.0):
        steer = math.radians(deg)
        kappa = steering_to_curvature(steer, robot.wheelbase_m)
        back = curvature_to_steering(kappa, robot.wheelbase_m)
        assert back == pytest.approx(steer, abs=1e-9)


def test_max_curvature_matches_steering_limit():
    robot = make_robot()
    hunter = HunterSE(robot)
    kappa_max = robot.max_curvature
    steer_at_max = hunter.curvature_to_steering(kappa_max)
    assert steer_at_max == pytest.approx(robot.steering_limit_rad, abs=1e-6)


def test_rollout_is_deterministic():
    robot = make_robot()
    dyn = DynamicsConfig(horizon_sec=2.0, dt_sec=0.1, model_actuator_lag=True)
    state = VehicleState(v=0.5, steering=0.1)
    out1 = ackermann_rollout.rollout_constant_target(state, 1.5, 0.2, robot, dyn)
    out2 = ackermann_rollout.rollout_constant_target(state, 1.5, 0.2, robot, dyn)
    for p1, p2 in zip(out1.points, out2.points):
        assert p1.state == p2.state


def test_actuator_lag_braking_covers_less_than_instant_stop():
    """A 2 m/s -> 0 brake at 6 m/s^2 should cover ~v0^2/(2*decel) = 0.333 m,
    NOT zero (which a naive instant-stop model would predict)."""
    robot = make_robot()
    dyn = DynamicsConfig(horizon_sec=1.0, dt_sec=1.0 / 15.0, model_actuator_lag=True)
    state = VehicleState(v=2.0, steering=0.0)
    out = ackermann_rollout.rollout_constant_target(state, 0.0, 0.0, robot, dyn)
    expected = stopping_model.stopping_distance_m(2.0, robot.brake_decel_mps2)
    assert out.final_state.x == pytest.approx(expected, rel=0.1)
    assert out.final_state.x > 0.0


def test_bicycle_step_matches_closed_form_circle():
    robot = make_robot()
    v, steering, dt = 2.0, math.radians(21.58), 0.01
    state = VehicleState()
    n = int(round(1.0 / dt))
    for _ in range(n):
        state = bicycle_model.step(state, v, steering, dt, robot.wheelbase_m)
    kappa = steering_to_curvature(steering, robot.wheelbase_m)
    radius = 1.0 / kappa
    arc = v * 1.0
    expected_x = radius * math.sin(arc / radius)
    expected_y = radius * (1.0 - math.cos(arc / radius))
    assert state.x == pytest.approx(expected_x, abs=1e-6)
    assert state.y == pytest.approx(expected_y, abs=1e-6)


def test_stopping_distance_zero_at_zero_speed():
    assert stopping_model.stopping_distance_m(0.0, 6.0) == 0.0


def test_stopping_distance_scales_with_v_squared():
    d1 = stopping_model.stopping_distance_m(1.0, 6.0)
    d2 = stopping_model.stopping_distance_m(2.0, 6.0)
    assert d2 == pytest.approx(4.0 * d1)
