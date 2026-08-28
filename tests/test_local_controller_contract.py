"""LocalPolicyController contract tests -- pure Python, no rclpy/torch (the
controller itself has neither dependency; only ``config.loader.load_profile``
and pure trajectory/guard math are exercised here)."""

import numpy as np
import pytest

from hunter_kinodynamic_rl.common.geometry import goal_distance_and_heading
from hunter_kinodynamic_rl.config.loader import load_profile
from hunter_kinodynamic_rl.env.observation.observation_builder import RobotState
from hunter_kinodynamic_rl.env.safety.action_guard import STOP_COMMAND, SafetyLimits
from hunter_kinodynamic_rl.navigation.local_rl.controller import LocalPolicyController
from hunter_kinodynamic_rl.trajectory.action_space import ACTION_DIM


def _profile():
    return load_profile("smoke_test")  # action_space.mode=trajectory, temporal_context=True, robot_state_dim=8


def _scan(profile, n_ranges=360):
    ranges = np.full(n_ranges, profile.observation.lidar_max_range_m, dtype=np.float32)
    return ranges, -np.pi, 2 * np.pi / n_ranges


def _limits():
    return SafetyLimits(max_sensor_age_sec=0.5, max_command_age_sec=0.5, min_obstacle_stop_distance_m=0.3,
                         max_odom_age_sec=0.5)


def test_build_observation_shape_matches_lidar_bins_times_history_plus_robot_state_dim():
    profile = _profile()
    controller = LocalPolicyController(profile)
    ranges, angle_min, angle_increment = _scan(profile)
    robot_state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.0, yaw_rate=0.0, steering=0.0)
    result = controller.build_observation(ranges, angle_min, angle_increment, robot_state, 3.0, 1.0)

    history_len = profile.observation.frame_stack if profile.features.temporal_context else 1
    expected_len = profile.observation.lidar_bins * history_len + profile.observation.robot_state_dim
    assert result.observation.shape == (expected_len,)


def test_observation_goal_terms_are_computed_from_the_passed_subgoal_not_leaked_elsewhere():
    """Section 6.3/6.9: local observation goal distance/bearing must be
    computed from whatever (x, y) the caller passes as the ACTIVE SUBGOAL
    -- this is the only goal argument the API accepts at all, so a final
    mission goal can never reach it except by the caller mistakenly
    passing it as this argument (structurally impossible to leak
    otherwise, since the module has no separate final-goal state)."""
    profile = _profile()
    controller = LocalPolicyController(profile)
    ranges, angle_min, angle_increment = _scan(profile)
    robot_state = RobotState(x=1.0, y=2.0, yaw=0.3, v=0.5, yaw_rate=0.0, steering=0.0)

    subgoal_x, subgoal_y = 5.0, -2.0
    result = controller.build_observation(ranges, angle_min, angle_increment, robot_state, subgoal_x, subgoal_y)

    history_len = profile.observation.frame_stack if profile.features.temporal_context else 1
    lidar_len = profile.observation.lidar_bins * history_len
    tail = result.observation[lidar_len:]

    expected_dist, expected_heading = goal_distance_and_heading(1.0, 2.0, 0.3, subgoal_x, subgoal_y)
    assert tail[0] == pytest.approx(expected_dist, abs=1e-4)
    assert tail[1] == pytest.approx(expected_heading, abs=1e-4)

    # A DIFFERENT goal (standing in for what a final mission goal would look
    # like) must change the observation -- proving the tail is genuinely
    # goal-dependent, not a stale/ignored argument.
    result_other_goal = controller.build_observation(
        ranges, angle_min, angle_increment, robot_state, 50.0, 50.0,
    )
    other_tail = result_other_goal.observation[lidar_len:]
    assert not np.allclose(tail[:2], other_tail[:2])


def test_frame_stack_persists_across_ticks_until_reset():
    profile = _profile()
    controller = LocalPolicyController(profile)
    ranges, angle_min, angle_increment = _scan(profile)
    robot_state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.0, yaw_rate=0.0, steering=0.0)

    r1 = controller.build_observation(ranges, angle_min, angle_increment, robot_state, 1.0, 0.0)
    ranges2 = ranges * 0.5
    r2 = controller.build_observation(ranges2, angle_min, angle_increment, robot_state, 1.0, 0.0)
    # After the second tick the stacked lidar frame must differ from the
    # first (a genuinely different scan was pushed), proving continuity
    # across calls rather than re-seeding from scratch every tick.
    assert not np.allclose(r1.observation, r2.observation)

    controller.reset()
    r3 = controller.build_observation(ranges, angle_min, angle_increment, robot_state, 1.0, 0.0)
    # After reset(), the very next build_observation() reseeds the stack
    # (all frames = the fresh scan) -- byte-identical to a freshly
    # constructed controller's first observation.
    fresh = LocalPolicyController(profile)
    r_fresh = fresh.build_observation(ranges, angle_min, angle_increment, robot_state, 1.0, 0.0)
    assert np.allclose(r3.observation, r_fresh.observation)


def test_validate_action_accepts_well_formed_action():
    action = np.zeros(ACTION_DIM, dtype=np.float32)
    result = LocalPolicyController.validate_action(action)
    assert result is not None
    assert result.shape == (ACTION_DIM,)


def test_validate_action_rejects_wrong_shape():
    assert LocalPolicyController.validate_action(np.zeros(ACTION_DIM - 1)) is None


def test_validate_action_rejects_non_finite():
    action = np.zeros(ACTION_DIM, dtype=np.float32)
    action[0] = float("nan")
    assert LocalPolicyController.validate_action(action) is None
    action[0] = float("inf")
    assert LocalPolicyController.validate_action(action) is None


def test_validate_action_never_raises_on_a_ragged_nested_sequence():
    """code review finding: the documented "returns None, never raises"
    contract did not actually hold -- a ragged/inhomogeneous nested list
    makes np.asarray(..., dtype=float64) itself raise ValueError, which
    previously propagated straight out of this "pure, safe" API."""
    assert LocalPolicyController.validate_action([[1], [2, 3]]) is None


def test_validate_action_never_raises_on_a_plain_object_with_no_array_interface():
    assert LocalPolicyController.validate_action(object()) is None


def test_validate_action_never_raises_on_a_string():
    assert LocalPolicyController.validate_action("not an action") is None


def test_validate_action_never_raises_on_none():
    assert LocalPolicyController.validate_action(None) is None


def test_decode_and_guard_stops_on_invalid_localization():
    profile = _profile()
    controller = LocalPolicyController(profile)
    action = np.zeros(ACTION_DIM, dtype=np.float32)
    nominal, safe = controller.decode_and_guard(
        action, current_steering_rad=0.0, safety_limits=_limits(), nearest_obstacle_dist_m=10.0,
        last_sensor_time_sec=0.0, last_command_time_sec=0.0, now_sec=0.0, last_odom_time_sec=0.0,
        localization_valid=False, subgoal_valid=True,
    )
    assert nominal == STOP_COMMAND
    assert safe == STOP_COMMAND


def test_decode_and_guard_stops_on_invalid_subgoal():
    profile = _profile()
    controller = LocalPolicyController(profile)
    action = np.zeros(ACTION_DIM, dtype=np.float32)
    nominal, safe = controller.decode_and_guard(
        action, current_steering_rad=0.0, safety_limits=_limits(), nearest_obstacle_dist_m=10.0,
        last_sensor_time_sec=0.0, last_command_time_sec=0.0, now_sec=0.0, last_odom_time_sec=0.0,
        localization_valid=True, subgoal_valid=False,
    )
    assert nominal == STOP_COMMAND
    assert safe == STOP_COMMAND


def test_decode_and_guard_stops_on_stale_sensor_even_when_localization_and_subgoal_valid():
    profile = _profile()
    controller = LocalPolicyController(profile)
    action = np.zeros(ACTION_DIM, dtype=np.float32)
    _nominal, safe = controller.decode_and_guard(
        action, current_steering_rad=0.0, safety_limits=_limits(), nearest_obstacle_dist_m=10.0,
        last_sensor_time_sec=0.0, last_command_time_sec=100.0, now_sec=100.0, last_odom_time_sec=100.0,
        localization_valid=True, subgoal_valid=True,
    )
    assert safe.speed_mps == pytest.approx(STOP_COMMAND.speed_mps)


def test_decode_and_guard_healthy_tick_returns_matching_nominal_and_safe_when_nothing_forces_a_stop():
    profile = _profile()
    controller = LocalPolicyController(profile)
    action = np.zeros(ACTION_DIM, dtype=np.float32)
    nominal, safe = controller.decode_and_guard(
        action, current_steering_rad=0.0, safety_limits=_limits(), nearest_obstacle_dist_m=10.0,
        last_sensor_time_sec=0.0, last_command_time_sec=0.0, now_sec=0.0, last_odom_time_sec=0.0,
        localization_valid=True, subgoal_valid=True,
    )
    assert safe.speed_mps == pytest.approx(nominal.speed_mps)
    assert safe.steering_rad == pytest.approx(nominal.steering_rad)


def test_commit_action_updates_prev_action_used_by_next_observation():
    profile = _profile()
    controller = LocalPolicyController(profile)
    ranges, angle_min, angle_increment = _scan(profile)
    robot_state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.0, yaw_rate=0.0, steering=0.0)

    assert controller.prev_action == [0.0, 0.0, 0.0]
    action = np.array([0.4, -0.3, 0.2], dtype=np.float64)
    controller.commit_action(action)
    assert controller.prev_action == [pytest.approx(0.4), pytest.approx(-0.3), pytest.approx(0.2)]

    history_len = profile.observation.frame_stack if profile.features.temporal_context else 1
    lidar_len = profile.observation.lidar_bins * history_len
    result = controller.build_observation(ranges, angle_min, angle_increment, robot_state, 1.0, 0.0)
    # tail layout for robot_state_dim=8: [dist, heading_err, prev_a0, prev_a1, prev_a2, v, yaw_rate, steering]
    tail = result.observation[lidar_len:]
    assert tail[2] == pytest.approx(0.4, abs=1e-5)
    assert tail[3] == pytest.approx(-0.3, abs=1e-5)
    assert tail[4] == pytest.approx(0.2, abs=1e-5)


def test_reset_clears_prev_action():
    profile = _profile()
    controller = LocalPolicyController(profile)
    controller.commit_action(np.array([0.4, -0.3, 0.2]))
    controller.reset()
    assert controller.prev_action == [0.0, 0.0, 0.0]
