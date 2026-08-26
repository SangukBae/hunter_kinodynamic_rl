import math

import numpy as np
import pytest

from hunter_kinodynamic_rl.config.schema import (
    DomainRandomizationConfig, RewardConfig, RobotConfig, ScenarioConfig,
)
from hunter_kinodynamic_rl.env.observation.observation_builder import (
    RobotState, build_observation, build_robot_state_vector,
)
from hunter_kinodynamic_rl.env.randomization.domain_randomizer import apply_to_robot_config, sample_draw
from hunter_kinodynamic_rl.env.rewards.reward_calculator import compute_reward, is_goal_reached
from hunter_kinodynamic_rl.env.safety.action_guard import (
    SafetyLimits, apply_collision_proximity_stop, guard, sanitize_command,
)
from hunter_kinodynamic_rl.env.scenarios.procedural_generator import (
    SeedSplitError, generate_scenario, is_reachable, seed_split,
)
from hunter_kinodynamic_rl.sensing.scan_processor import bin_scan_sector, front_and_full_state
from hunter_kinodynamic_rl.sensing.temporal_stack import FrameStack
from hunter_kinodynamic_rl.trajectory.pure_pursuit_adapter import VehicleCommand


def make_robot() -> RobotConfig:
    return RobotConfig(
        name="hunter_se", wheelbase_m=0.547696, track_width_m=0.503404,
        wheel_radius_m=0.136, length_m=0.76, width_m=0.4, height_m=0.14,
        mass_kg=42.0, steering_limit_deg=21.58, max_forward_speed_mps=2.0,
        min_forward_speed_mps=0.0, accel_limit_mps2=6.0, brake_decel_mps2=6.0,
        steering_rate_deg_s=200.0, speed_lag_tau_sec=0.0,
    )


# ---------------------------------------------------------------- sensing
def test_frame_stack_warm_start_repeats_first_frame():
    fs = FrameStack(frame_dim=4, history_len=3)
    fs.reset(np.array([1, 2, 3, 4], dtype=np.float32))
    stacked = fs.stacked()
    assert stacked.shape == (12,)
    np.testing.assert_array_equal(stacked, np.tile([1, 2, 3, 4], 3))


def test_frame_stack_current_frame_first():
    fs = FrameStack(frame_dim=2, history_len=2)
    fs.reset(np.array([0, 0], dtype=np.float32))
    fs.push(np.array([9, 9], dtype=np.float32))
    stacked = fs.stacked()
    np.testing.assert_array_equal(stacked[:2], [9, 9])
    np.testing.assert_array_equal(stacked[2:], [0, 0])


def test_bin_scan_sector_detects_nearest_obstacle():
    ranges = np.full(360, 10.0, dtype=np.float32)
    ranges[0] = 1.0  # directly ahead (angle 0)
    binned = bin_scan_sector(ranges, angle_min=0.0, angle_increment=math.radians(1.0),
                              sector_center=0.0, sector_width=math.pi, num_bins=8, max_range=10.0)
    assert binned.min() == pytest.approx(1.0, abs=0.2)


def test_front_and_full_state_shapes():
    ranges = np.full(360, 5.0, dtype=np.float32)
    obs, full = front_and_full_state(ranges, angle_min=0.0, angle_increment=math.radians(1.0),
                                      num_bins=80, max_range=10.0)
    assert obs.shape == (80,)
    assert full.shape == (80,)


# ------------------------------------------------------------- observation
def test_observation_vector_default_dimension_matches_drl_agent_baseline_parity():
    """section P0-4: the DEFAULT (no robot_state_dim override -- what
    baseline_tqc.yaml/legacy_waypoint_tqc.yaml use) must match drl_agent's
    own 7D robot-state tail EXACTLY, so LiDAR(80) + robot_state(7) = 87,
    never the L-memory-extended 88."""
    robot_state = RobotState(x=0, y=0, yaw=0, v=1.0, yaw_rate=0.1, steering=0.05)
    vec = build_robot_state_vector(robot_state, goal_x=3.0, goal_y=4.0, prev_action_01=[0.1, -0.2, 0.7])
    assert vec.shape == (7,)
    assert vec[0] == pytest.approx(5.0)  # hypot(3,4)
    # The 3rd prev_action component (L/yield) is NOT included by default --
    # dropped entirely, never silently smuggled into one of the other slots.
    assert list(vec[2:4]) == [pytest.approx(0.1), pytest.approx(-0.2)]

    obs = build_observation(np.zeros(80, dtype=np.float32), vec)
    assert obs.shape == (87,)


def test_observation_vector_dimension_with_l_memory_opt_in():
    """section P0-4/P1-11: robot_state_dim=8 (kinodynamic_tqc*'s explicit
    opt-in) includes the 3rd previous-action (L/yield) component."""
    robot_state = RobotState(x=0, y=0, yaw=0, v=1.0, yaw_rate=0.1, steering=0.05)
    vec = build_robot_state_vector(
        robot_state, goal_x=3.0, goal_y=4.0, prev_action_01=[0.1, -0.2, 0.7], robot_state_dim=8)
    assert vec.shape == (8,)
    assert vec[0] == pytest.approx(5.0)  # hypot(3,4)
    assert vec[4] == pytest.approx(0.7)  # prev_a2 (L/yield) -- section P1-11

    obs = build_observation(np.zeros(80, dtype=np.float32), vec)
    assert obs.shape == (88,)


def test_observation_vector_prev_action_defaults_to_zero_when_l_missing():
    """Backward-compatible fallback for a 2-component prev_action (e.g. a
    fresh episode's initial np.zeros-of-unknown-length caller) -- prev_a2
    defaults to 0.0 rather than raising, when robot_state_dim=8."""
    robot_state = RobotState(x=0, y=0, yaw=0, v=1.0, yaw_rate=0.1, steering=0.05)
    vec = build_robot_state_vector(
        robot_state, goal_x=1.0, goal_y=0.0, prev_action_01=[0.1, -0.2], robot_state_dim=8)
    assert vec[4] == pytest.approx(0.0)


# ------------------------------------------------------------------ reward
def test_goal_reward_dominates_when_reached():
    cfg = RewardConfig()
    r = compute_reward(cfg, goal_distance_m=0.1, previous_goal_distance_m=0.5,
                        collided=False, reached_goal=True, action=[0, 0], previous_action=[0, 0])
    assert r.total == pytest.approx(cfg.goal_reached_reward)


def test_collision_penalty_applied():
    cfg = RewardConfig()
    r = compute_reward(cfg, goal_distance_m=5.0, previous_goal_distance_m=5.0,
                        collided=True, reached_goal=False, action=[0, 0], previous_action=[0, 0])
    assert r.total == pytest.approx(cfg.collision_penalty)


def test_progress_reward_positive_when_getting_closer():
    cfg = RewardConfig()
    r = compute_reward(cfg, goal_distance_m=3.0, previous_goal_distance_m=4.0,
                        collided=False, reached_goal=False, action=[0, 0], previous_action=[0, 0])
    assert r.progress > 0.0


def test_trajectory_smoothness_weight_penalizes_bending_energy():
    cfg = RewardConfig(trajectory_smoothness_weight=2.0, control_smoothness_weight=0.0)
    r = compute_reward(
        cfg, goal_distance_m=3.0, previous_goal_distance_m=3.0,
        collided=False, reached_goal=False, action=[0, 0, 0], previous_action=[0, 0, 0],
        trajectory_kappa=0.5, trajectory_horizon_m=2.0,
    )
    assert r.trajectory_smoothness == pytest.approx(-1.0)


def test_is_goal_reached_threshold():
    cfg = RewardConfig(goal_threshold_m=0.42)
    assert is_goal_reached(0.4, cfg) is True
    assert is_goal_reached(0.5, cfg) is False


# ----------------------------------------------------------------- safety
def test_sanitize_command_rejects_nan():
    robot = make_robot()
    bad = VehicleCommand(speed_mps=float("nan"), steering_rad=0.1)
    safe = sanitize_command(bad, robot)
    assert safe.speed_mps == 0.0 and safe.steering_rad == 0.0


def test_sanitize_command_clamps_bounds():
    robot = make_robot()
    over = VehicleCommand(speed_mps=100.0, steering_rad=10.0)
    safe = sanitize_command(over, robot)
    assert safe.speed_mps == pytest.approx(robot.max_forward_speed_mps)
    assert safe.steering_rad == pytest.approx(robot.steering_limit_rad)


def test_collision_proximity_stop_zeroes_forward_speed():
    cmd = VehicleCommand(speed_mps=1.0, steering_rad=0.1)
    limits = SafetyLimits(min_obstacle_stop_distance_m=0.3)
    stopped = apply_collision_proximity_stop(cmd, nearest_obstacle_distance_m=0.1, limits=limits)
    assert stopped.speed_mps == 0.0
    assert stopped.steering_rad == pytest.approx(0.1)  # steering preserved


def test_guard_stops_on_stale_sensor():
    robot = make_robot()
    limits = SafetyLimits(max_sensor_age_sec=0.5)
    cmd = VehicleCommand(speed_mps=1.0, steering_rad=0.0)
    out = guard(cmd, robot, limits, last_sensor_time_sec=0.0, last_command_time_sec=10.0, now_sec=10.0)
    assert out.speed_mps == 0.0


def test_guard_passes_through_when_all_fresh():
    robot = make_robot()
    limits = SafetyLimits()
    cmd = VehicleCommand(speed_mps=1.0, steering_rad=0.05)
    out = guard(cmd, robot, limits, last_sensor_time_sec=10.0, last_command_time_sec=10.0, now_sec=10.05)
    assert out.speed_mps == pytest.approx(1.0)


# --------------------------------------------------------------- scenarios
def test_seed_split_assigns_correct_pool():
    cfg = ScenarioConfig(train_seed_range=[0, 99], validation_seed_range=[100, 199], test_seed_range=[200, 299])
    assert seed_split(50, cfg) == "train"
    assert seed_split(150, cfg) == "validation"
    assert seed_split(250, cfg) == "test"


def test_seed_split_raises_outside_all_ranges():
    cfg = ScenarioConfig(train_seed_range=[0, 99], validation_seed_range=[100, 199], test_seed_range=[200, 299])
    with pytest.raises(SeedSplitError):
        seed_split(1000, cfg)


def test_generate_scenario_is_deterministic_for_a_seed():
    cfg = ScenarioConfig(world_size_m=10.0, min_obstacles=2, max_obstacles=4)
    s1 = generate_scenario(42, cfg, robot_radius=0.3)
    s2 = generate_scenario(42, cfg, robot_radius=0.3)
    assert s1 == s2


def test_generate_scenario_start_and_goal_are_reachable():
    cfg = ScenarioConfig(world_size_m=10.0, min_obstacles=0, max_obstacles=3)
    spec = generate_scenario(7, cfg, robot_radius=0.3)
    assert is_reachable((spec.start_x, spec.start_y), (spec.goal_x, spec.goal_y),
                         spec.static_obstacles, cfg.world_size_m, 0.3)


def test_generate_scenario_different_seeds_differ():
    cfg = ScenarioConfig(world_size_m=10.0, min_obstacles=0, max_obstacles=3)
    s1 = generate_scenario(1, cfg)
    s2 = generate_scenario(2, cfg)
    assert (s1.start_x, s1.start_y) != (s2.start_x, s2.start_y)


# ---------------------------------------------------------- randomization
def test_disabled_randomization_is_identity():
    cfg = DomainRandomizationConfig(enabled=False)
    draw = sample_draw(0, cfg)
    robot = make_robot()
    randomized = apply_to_robot_config(robot, draw)
    assert randomized == robot


def test_enabled_randomization_stays_within_configured_range():
    cfg = DomainRandomizationConfig(enabled=True, mass_scale_range=[0.8, 1.2])
    for seed in range(20):
        draw = sample_draw(seed, cfg)
        assert 0.8 <= draw.mass_scale <= 1.2


def test_randomization_is_deterministic_per_seed():
    cfg = DomainRandomizationConfig(enabled=True, mass_scale_range=[0.5, 1.5])
    d1 = sample_draw(123, cfg)
    d2 = sample_draw(123, cfg)
    assert d1 == d2


def test_randomized_robot_config_preserves_collision_radius():
    """Regression: apply_to_robot_config previously omitted
    collision_radius_m from the constructed RobotConfig, silently falling
    back to the dataclass default (0.45) instead of the BASE config's own
    value -- invisible whenever the base happened to equal the default, but
    a real bug for any robot.yaml that sets a different value."""
    robot = make_robot()
    custom_robot = RobotConfig(**{**robot.__dict__, "collision_radius_m": 0.99})
    cfg = DomainRandomizationConfig(enabled=True, mass_scale_range=[0.8, 1.2])
    draw = sample_draw(5, cfg)
    randomized = apply_to_robot_config(custom_robot, draw)
    assert randomized.collision_radius_m == pytest.approx(0.99)
