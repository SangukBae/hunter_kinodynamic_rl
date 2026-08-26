import pytest

from hunter_kinodynamic_rl.config.schema import (
    CounterfactualConfig, DynamicsConfig, RiskConfig, RobotConfig,
)
from hunter_kinodynamic_rl.dynamics import ackermann_rollout
from hunter_kinodynamic_rl.risk import future_clearance, trajectory_risk, ttc, unrecoverable_state
from hunter_kinodynamic_rl.risk.counterfactual_sampler import safer_alternative_margin, score_candidates
from hunter_kinodynamic_rl.risk.labels import DynamicObstacle
from hunter_kinodynamic_rl.risk.stopping_margin import stopping_margin_m
from hunter_kinodynamic_rl.robot.interface import VehicleState
from hunter_kinodynamic_rl.trajectory.action_space import TrajectoryCommand


def make_robot() -> RobotConfig:
    return RobotConfig(
        name="hunter_se", wheelbase_m=0.547696, track_width_m=0.503404,
        wheel_radius_m=0.136, length_m=0.76, width_m=0.4, height_m=0.14,
        mass_kg=42.0, steering_limit_deg=21.58, max_forward_speed_mps=2.0,
        min_forward_speed_mps=0.0, accel_limit_mps2=6.0, brake_decel_mps2=6.0,
        steering_rate_deg_s=200.0, speed_lag_tau_sec=0.0,
    )


def test_collision_trajectory_has_zero_ttc_within_horizon():
    robot = make_robot()
    dyn = DynamicsConfig(horizon_sec=2.0, dt_sec=0.1, model_actuator_lag=False)
    state = VehicleState(v=1.0, steering=0.0)
    rollout = ackermann_rollout.rollout_constant_target(state, 1.0, 0.0, robot, dyn)
    # Obstacle directly ahead, well inside the rollout's path.
    obstacle = DynamicObstacle(x0=1.0, y0=0.0, radius=0.3)
    t = ttc.time_to_collision(rollout, ego_radius=0.3, obstacles=[obstacle], horizon_sec=2.0)
    assert t < 2.0
    assert ttc.collision_within_horizon(rollout, 0.3, [obstacle], 2.0) is True


def test_non_collision_trajectory_has_full_horizon_ttc():
    robot = make_robot()
    dyn = DynamicsConfig(horizon_sec=2.0, dt_sec=0.1, model_actuator_lag=False)
    state = VehicleState(v=1.0, steering=0.0)
    rollout = ackermann_rollout.rollout_constant_target(state, 1.0, 0.0, robot, dyn)
    obstacle = DynamicObstacle(x0=10.0, y0=10.0, radius=0.3)  # far away
    t = ttc.time_to_collision(rollout, ego_radius=0.3, obstacles=[obstacle], horizon_sec=2.0)
    assert t == pytest.approx(2.0)
    assert ttc.collision_within_horizon(rollout, 0.3, [obstacle], 2.0) is False


def test_min_clearance_no_obstacles_is_infinite():
    robot = make_robot()
    dyn = DynamicsConfig(horizon_sec=1.0, dt_sec=0.1, model_actuator_lag=False)
    state = VehicleState(v=1.0, steering=0.0)
    rollout = ackermann_rollout.rollout_constant_target(state, 1.0, 0.0, robot, dyn)
    assert future_clearance.min_clearance(rollout, 0.3, []) == float("inf")


def test_min_clearance_decreases_as_obstacle_gets_closer():
    robot = make_robot()
    dyn = DynamicsConfig(horizon_sec=1.0, dt_sec=0.1, model_actuator_lag=False)
    state = VehicleState(v=1.0, steering=0.0)
    rollout = ackermann_rollout.rollout_constant_target(state, 1.0, 0.0, robot, dyn)
    far = future_clearance.min_clearance(rollout, 0.3, [DynamicObstacle(x0=5.0, y0=0.0, radius=0.1)])
    near = future_clearance.min_clearance(rollout, 0.3, [DynamicObstacle(x0=0.9, y0=0.0, radius=0.1)])
    assert near < far


def test_stopping_margin_positive_when_far_enough():
    robot = make_robot()
    margin = stopping_margin_m(current_speed_mps=1.0, distance_to_obstacle_m=10.0, robot=robot)
    assert margin > 0.0


def test_stopping_margin_negative_when_too_close():
    robot = make_robot()
    margin = stopping_margin_m(current_speed_mps=2.0, distance_to_obstacle_m=0.05, robot=robot)
    assert margin < 0.0


def test_assess_trajectory_collision_scores_maximal_risk():
    robot = make_robot()
    dyn = DynamicsConfig(horizon_sec=1.0, dt_sec=0.1, model_actuator_lag=False)
    risk_cfg = RiskConfig(enabled=True, ttc_horizon_sec=1.0, min_safe_clearance_m=0.3)
    state = VehicleState(v=1.0, steering=0.0)
    rollout = ackermann_rollout.rollout_constant_target(state, 1.0, 0.0, robot, dyn)
    obstacle = DynamicObstacle(x0=0.5, y0=0.0, radius=0.3)
    label = trajectory_risk.assess_trajectory(rollout, 0.3, [obstacle], 0.0, robot, risk_cfg)
    assert label.collision_within_horizon is True
    assert label.risk_score == pytest.approx(1.0)


def test_assess_trajectory_clear_path_scores_low_risk():
    robot = make_robot()
    dyn = DynamicsConfig(horizon_sec=1.0, dt_sec=0.1, model_actuator_lag=False)
    risk_cfg = RiskConfig(enabled=True, ttc_horizon_sec=1.0, min_safe_clearance_m=0.3)
    state = VehicleState(v=1.0, steering=0.0)
    rollout = ackermann_rollout.rollout_constant_target(state, 1.0, 0.0, robot, dyn)
    label = trajectory_risk.assess_trajectory(rollout, 0.3, [], 0.0, robot, risk_cfg)
    assert label.collision_within_horizon is False
    assert label.risk_score == pytest.approx(0.0)


def test_counterfactual_ranking_prefers_a_clear_lane():
    robot = make_robot()
    dyn = DynamicsConfig(horizon_sec=1.5, dt_sec=0.1, model_actuator_lag=False)
    risk_cfg = RiskConfig(enabled=True, ttc_horizon_sec=1.5, min_safe_clearance_m=0.3)
    cf_cfg = CounterfactualConfig(enabled=True, num_candidates=10,
                                   kappa_offsets_frac=[-1.0, -0.5, 0.5, 1.0],
                                   speed_fractions=[1.0], include_stop_candidate=True)
    state = VehicleState(v=1.0, steering=0.0)
    base = TrajectoryCommand(kappa=0.0, v_ref=1.0, horizon_m=1.5)  # straight ahead, into the obstacle
    obstacle = DynamicObstacle(x0=1.5, y0=0.0, radius=0.05)  # directly ahead; mild turn remains clear
    scored = score_candidates(
        base, state, 0.3, [obstacle], robot, dyn, risk_cfg, cf_cfg,
        goal_local_xy=(5.0, 0.0),
    )

    assert scored[0].command == base
    margin = safer_alternative_margin(scored, cf_cfg)
    assert margin > 0.0  # a progress-preserving steered alternative is safer than driving straight at it


def test_unrecoverable_state_when_all_candidates_collide():
    robot = make_robot()
    dyn = DynamicsConfig(horizon_sec=0.5, dt_sec=0.05, model_actuator_lag=False)
    risk_cfg = RiskConfig(enabled=True, ttc_horizon_sec=0.5, min_safe_clearance_m=0.3)
    cf_cfg = CounterfactualConfig(enabled=True, num_candidates=6, kappa_offsets_frac=[-0.1, 0.1],
                                   speed_fractions=[1.0], include_stop_candidate=False)
    state = VehicleState(v=2.0, steering=0.0)
    base = TrajectoryCommand(kappa=0.0, v_ref=2.0, horizon_m=1.0)
    # A wall of obstacles spanning the whole reachable lateral range at close range.
    obstacles = [DynamicObstacle(x0=0.3, y0=y, radius=0.5) for y in (-1.0, -0.5, 0.0, 0.5, 1.0)]
    scored = score_candidates(base, state, 0.3, obstacles, robot, dyn, risk_cfg, cf_cfg)
    assert unrecoverable_state.is_unrecoverable(scored) is True


def test_unrecoverable_state_false_when_escape_exists():
    robot = make_robot()
    dyn = DynamicsConfig(horizon_sec=1.0, dt_sec=0.1, model_actuator_lag=False)
    risk_cfg = RiskConfig(enabled=True, ttc_horizon_sec=1.0, min_safe_clearance_m=0.3)
    cf_cfg = CounterfactualConfig(enabled=True, num_candidates=6, kappa_offsets_frac=[-1.0, 1.0],
                                   speed_fractions=[1.0], include_stop_candidate=True)
    state = VehicleState(v=1.0, steering=0.0)
    base = TrajectoryCommand(kappa=0.0, v_ref=1.0, horizon_m=1.0)
    scored = score_candidates(base, state, 0.3, [], robot, dyn, risk_cfg, cf_cfg)
    assert unrecoverable_state.is_unrecoverable(scored) is False
