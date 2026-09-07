"""Regression tests for Hunter SE center/wheel/CAN steering conversions."""

import math
from pathlib import Path

import pytest

from hunter_kinodynamic_rl.config.loader import DEFAULT_RL_PROFILE, load_profile
from hunter_kinodynamic_rl.robot.hunter_se import HunterSE
from hunter_kinodynamic_rl.robot.limits import (
    center_steering_to_inner_wheel_angle,
    center_steering_to_wheel_angles,
    inner_wheel_angle_to_center_steering,
    wheel_angles_to_center_steering,
)


WHEELBASE_M = 0.550
TRACK_WIDTH_M = 0.460
CENTER_LIMIT_RAD = math.radians(19.0666770666)
INNER_LIMIT_RAD = math.radians(22.0)
CONFIG_ROOT = str(Path(__file__).resolve().parents[1] / "config")


def test_exactly_two_hunter_se_robot_configs_are_public():
    robot_dir = Path(CONFIG_ROOT) / "robot"
    assert {path.name for path in robot_dir.glob("hunter_se*.yaml")} == {
        "hunter_se.yaml",
        "hunter_se_improved.yaml",
    }

    baseline = load_profile("kinodynamic_tqc", config_root=CONFIG_ROOT)
    improved = load_profile("kinodynamic_tqc_improved", config_root=CONFIG_ROOT)
    real = load_profile("real_hunter_safe", config_root=CONFIG_ROOT)
    assert baseline.robot.name == "hunter_se"
    assert baseline.robot.collision_radius_m == pytest.approx(0.53)
    assert improved.robot.name == "hunter_se_improved"
    assert improved.robot.collision_radius_m == pytest.approx(0.58)
    assert improved.robot.track_width_m == pytest.approx(0.49238)
    assert improved.robot.steering_limit_deg == pytest.approx(18.8883201008)
    assert real.robot.name == "hunter_se_improved"


def test_new_rl_sessions_default_to_improved_while_baseline_stays_explicit():
    default_profile = load_profile(DEFAULT_RL_PROFILE, config_root=CONFIG_ROOT)
    baseline = load_profile("kinodynamic_tqc", config_root=CONFIG_ROOT)

    assert DEFAULT_RL_PROFILE == "kinodynamic_tqc_improved"
    assert default_profile.robot.name == "hunter_se_improved"
    assert default_profile.scenario.goal_sampling_mode == "robot_relative_band"
    assert default_profile.scenario.goal_direction_sectors_deg == [[-180.0, 180.0]]
    assert default_profile.scenario.goal_infeasible_fraction == pytest.approx(0.15)
    assert baseline.robot.name == "hunter_se"


@pytest.mark.parametrize("center_rad", [
    -CENTER_LIMIT_RAD,
    math.radians(-10.0),
    0.0,
    math.radians(10.0),
    CENTER_LIMIT_RAD,
])
def test_center_wheel_round_trip(center_rad):
    left, right = center_steering_to_wheel_angles(
        center_rad, WHEELBASE_M, TRACK_WIDTH_M
    )
    recovered = wheel_angles_to_center_steering(
        left, right, WHEELBASE_M, TRACK_WIDTH_M
    )
    assert recovered == pytest.approx(center_rad, abs=1e-12)


def test_manual_center_limit_maps_to_22_degree_inner_wheel_both_directions():
    positive = center_steering_to_inner_wheel_angle(
        CENTER_LIMIT_RAD, WHEELBASE_M, TRACK_WIDTH_M
    )
    negative = center_steering_to_inner_wheel_angle(
        -CENTER_LIMIT_RAD, WHEELBASE_M, TRACK_WIDTH_M
    )
    assert positive == pytest.approx(INNER_LIMIT_RAD, abs=1e-12)
    assert negative == pytest.approx(-INNER_LIMIT_RAD, abs=1e-12)


@pytest.mark.parametrize("inner_rad", [
    -INNER_LIMIT_RAD,
    math.radians(-12.0),
    0.0,
    math.radians(12.0),
    INNER_LIMIT_RAD,
])
def test_inner_wheel_can_adapter_round_trip(inner_rad):
    center = inner_wheel_angle_to_center_steering(
        inner_rad, WHEELBASE_M, TRACK_WIDTH_M
    )
    recovered = center_steering_to_inner_wheel_angle(
        center, WHEELBASE_M, TRACK_WIDTH_M
    )
    assert recovered == pytest.approx(inner_rad, abs=1e-12)


def test_wheel_angle_mean_is_not_used_as_ackermann_inverse():
    left, right = center_steering_to_wheel_angles(
        CENTER_LIMIT_RAD, WHEELBASE_M, TRACK_WIDTH_M
    )
    naive_mean = 0.5 * (left + right)
    recovered = wheel_angles_to_center_steering(
        left, right, WHEELBASE_M, TRACK_WIDTH_M
    )
    assert math.degrees(naive_mean - CENTER_LIMIT_RAD) == pytest.approx(
        0.33502079675, abs=1e-9
    )
    assert recovered == pytest.approx(CENTER_LIMIT_RAD, abs=1e-12)


def test_hunter_model_clamps_outgoing_center_command_before_can_conversion():
    robot = HunterSE(load_profile("kinodynamic_tqc").robot)
    inner = robot.center_steering_to_inner_wheel(math.radians(45.0))
    assert inner == pytest.approx(INNER_LIMIT_RAD, abs=1e-12)


def test_improved_hunter_geometry_preserves_22_degree_can_inner_limit():
    robot = HunterSE(
        load_profile(
            "kinodynamic_tqc_improved", config_root=CONFIG_ROOT
        ).robot
    )
    inner = robot.center_steering_to_inner_wheel(math.radians(45.0))
    assert inner == pytest.approx(INNER_LIMIT_RAD, abs=1e-12)


@pytest.mark.parametrize("wheelbase,track", [(0.0, TRACK_WIDTH_M), (WHEELBASE_M, -0.1)])
def test_invalid_geometry_is_rejected(wheelbase, track):
    with pytest.raises(ValueError):
        center_steering_to_wheel_angles(0.1, wheelbase, track)
