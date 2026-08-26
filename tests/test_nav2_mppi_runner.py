"""Regression tests for the Nav2-MPPI classical baseline's PURE logic
(section 34/35) -- collision_threshold_m/nearest_obstacle_distance_m are
ROS-free (only Nav2MPPIRunner's constructor touches rclpy/nav2_msgs, lazily,
see its docstring), so this module is importable and testable on a bare
host checkout, unlike most of this package's live-Gazebo-facing code.
"""

import math

import numpy as np
import pytest

from hunter_kinodynamic_rl.config.loader import load_profile
from hunter_kinodynamic_rl.config.schema import ConfigError, ObservationConfig, RobotConfig
from hunter_kinodynamic_rl.evaluation.nav2_mppi_runner import (
    collision_threshold_m, nearest_obstacle_distance_m, navigation_time_sec,
)


def test_collision_threshold_matches_robot_plus_margin():
    profile = load_profile("evaluation_id")
    expected = profile.robot.collision_radius_m + profile.observation.collision_margin_m
    assert collision_threshold_m(profile) == pytest.approx(expected)


def test_nearest_obstacle_distance_reports_the_closest_full_360_reading():
    profile = load_profile("evaluation_id")
    n = 400
    angle_min = -math.pi
    angle_increment = 2 * math.pi / n
    ranges = np.full(n, 9.0, dtype=np.float32)
    ranges[200] = 1.5  # a single close reading somewhere in the sweep
    dist = nearest_obstacle_distance_m(ranges, angle_min, angle_increment, profile, robot_x=0.0, robot_y=0.0)
    assert dist == pytest.approx(1.5, abs=0.05)


def test_nearest_obstacle_distance_is_capped_by_the_world_boundary_near_the_edge():
    """A robot near the world edge with an all-clear scan must still report
    the BOUNDARY distance as the binding constraint -- mirrors
    environment_node.py's own P0-2 fix (world boundary counts as an
    obstacle for collision purposes)."""
    profile = load_profile("evaluation_id")
    n = 400
    ranges = np.full(n, profile.observation.lidar_max_range_m, dtype=np.float32)  # nothing detected
    half_extent = profile.scenario.world_size_m / 2.0
    robot_x = half_extent - 0.1  # 0.1 m from the boundary
    dist = nearest_obstacle_distance_m(ranges, -math.pi, 2 * math.pi / n, profile, robot_x=robot_x, robot_y=0.0)
    assert dist == pytest.approx(0.1, abs=1e-3)


def test_min_turning_r_in_nav2_mppi_params_matches_hunter_se_geometry():
    """config/nav2_mppi/nav2_mppi_params.yaml's AckermannConstraints.
    min_turning_r is a pinned LITERAL (Nav2 params are static YAML, not
    Python) derived from config/robot/hunter_se.yaml's wheelbase_m/
    steering_limit_deg -- this proves that literal hasn't drifted from the
    robot config it claims to match, by re-deriving it from RobotConfig.max_curvature
    (kappa_max = tan(steering_limit)/wheelbase -> min_turning_r = 1/kappa_max)
    and parsing the actual pinned value out of the YAML file."""
    import os
    import re

    robot = load_profile("evaluation_id").robot
    expected_min_turning_r = 1.0 / robot.max_curvature

    here = os.path.dirname(os.path.abspath(__file__))
    params_path = os.path.normpath(os.path.join(here, "..", "config", "nav2_mppi", "nav2_mppi_params.yaml"))
    with open(params_path) as f:
        content = f.read()
    match = re.search(r"min_turning_r:\s*([0-9.]+)", content)
    assert match, "min_turning_r not found in nav2_mppi_params.yaml"
    pinned_value = float(match.group(1))
    assert pinned_value == pytest.approx(expected_min_turning_r, abs=1e-3)


def test_robot_config_max_curvature_matches_min_turning_r_pin_precisely():
    """Independent of the file-parsing test above: directly proves the
    schema-level derivation (max_curvature = tan(limit)/wheelbase) is what
    the pinned YAML literal encodes, using the exact hunter_se.yaml values
    quoted in nav2_mppi_params.yaml's header comment."""
    robot = RobotConfig(wheelbase_m=0.547696, steering_limit_deg=21.58,
                         max_forward_speed_mps=2.0, accel_limit_mps2=1.0, brake_decel_mps2=1.0)
    robot.validate()
    min_turning_r = 1.0 / robot.max_curvature
    assert min_turning_r == pytest.approx(1.3847, abs=1e-3)


# --------------------------------------------------------- P1-9: navigation_time_sec units
def test_navigation_time_prefers_clock_derived_sim_time():
    """section P1-9: this baseline's timing metric must match
    benchmark_runner.py's RL-agent metric (/clock-derived simulation time),
    never wall-clock -- fairness breaks the moment Gazebo's real-time-factor
    isn't exactly 1.0."""
    assert navigation_time_sec(episode_elapsed_sim_time_sec=12.5, wall_start=100.0, wall_now=999.0) == \
        pytest.approx(12.5)


def test_navigation_time_falls_back_to_wall_clock_only_when_clock_unavailable():
    """/clock was never received at all (episode_elapsed_sim_time_sec is
    None, e.g. use_sim_time off) -- wall time is the only option left, not
    silently reporting a nonsensical zero/None duration."""
    assert navigation_time_sec(episode_elapsed_sim_time_sec=None, wall_start=100.0, wall_now=107.5) == \
        pytest.approx(7.5)
