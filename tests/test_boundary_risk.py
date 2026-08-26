"""Regression tests for the world-boundary risk contribution (code review
P0-2): the procedural training world is a virtual square with NO physical
Gazebo wall geometry, so it was previously invisible to every
clearance/TTC/collision/counterfactual computation -- a trajectory that
drove straight into a wall silently scored ``risk_target=0``. See
``hunter_kinodynamic_rl/risk/boundary.py``'s module docstring for the full
rationale.
"""

import math

import pytest

from hunter_kinodynamic_rl.common.geometry import to_robot_frame, to_world_frame
from hunter_kinodynamic_rl.config.schema import DynamicsConfig, RiskConfig, RobotConfig
from hunter_kinodynamic_rl.dynamics import ackermann_rollout
from hunter_kinodynamic_rl.env.scenarios.procedural_generator import (
    ScenarioConfig, generate_scenario,
)
from hunter_kinodynamic_rl.risk import boundary
from hunter_kinodynamic_rl.risk.trajectory_risk import assess_trajectory
from hunter_kinodynamic_rl.robot.interface import VehicleState


def make_robot() -> RobotConfig:
    return RobotConfig(
        name="hunter_se", wheelbase_m=0.547696, track_width_m=0.503404,
        wheel_radius_m=0.136, length_m=0.76, width_m=0.4, height_m=0.14,
        mass_kg=42.0, steering_limit_deg=21.58, max_forward_speed_mps=2.0,
        min_forward_speed_mps=0.0, accel_limit_mps2=6.0, brake_decel_mps2=6.0,
        steering_rate_deg_s=200.0, speed_lag_tau_sec=0.0,
    )


# ------------------------------------------------------------- to_world_frame
def test_to_world_frame_inverts_to_robot_frame():
    for yaw in (0.0, 0.3, -1.2, math.pi / 2.0, math.pi):
        for rel in ((1.0, 0.0), (0.0, 1.0), (2.5, -3.1), (-4.0, -4.0)):
            local = to_robot_frame(rel[0], rel[1], yaw)
            back = to_world_frame(local[0], local[1], yaw)
            assert back[0] == pytest.approx(rel[0], abs=1e-9)
            assert back[1] == pytest.approx(rel[1], abs=1e-9)


# ---------------------------------------------------------- distance_to_boundary_m
def test_distance_to_boundary_positive_at_center_negative_outside():
    assert boundary.distance_to_boundary_m(0.0, 0.0, half_extent_m=5.0) == pytest.approx(5.0)
    assert boundary.distance_to_boundary_m(4.0, 0.0, half_extent_m=5.0) == pytest.approx(1.0)
    assert boundary.distance_to_boundary_m(6.0, 0.0, half_extent_m=5.0) == pytest.approx(-1.0)
    # Nearest wall is whichever axis is closer to its edge.
    assert boundary.distance_to_boundary_m(1.0, 4.5, half_extent_m=5.0) == pytest.approx(0.5)


def test_boundary_clearance_at_accounts_for_robot_world_pose_and_yaw():
    # Robot sits at world (4.5, 0, 0) facing +x -- 0.5m from the +x wall of a
    # half_extent=5.0 world. A rollout point 1.0m further "ahead" (local
    # +x) lands at world x=5.5, i.e. 0.5m PAST the wall.
    robot_pose = (4.5, 0.0, 0.0)
    clearance = boundary.boundary_clearance_at(1.0, 0.0, ego_radius=0.0, robot_pose=robot_pose, half_extent_m=5.0)
    assert clearance == pytest.approx(-0.5)


# ------------------------------------------------------------- assess_trajectory
def _straight_rollout(robot: RobotConfig, horizon_sec: float = 2.0, v: float = 1.0) -> "ackermann_rollout.Rollout":
    dyn = DynamicsConfig(horizon_sec=horizon_sec, dt_sec=0.1, model_actuator_lag=False)
    state = VehicleState(v=v, steering=0.0)
    return ackermann_rollout.rollout_constant_target(state, v, 0.0, robot, dyn)


def test_boundary_omitted_by_default_preserves_obstacle_only_behaviour():
    """Backward compatibility: callers that don't pass robot_world_pose /
    world_half_extent_m (the vast majority of tests/test_risk.py) must see
    IDENTICAL behaviour to before this feature existed."""
    robot = make_robot()
    risk_cfg = RiskConfig(enabled=True, ttc_horizon_sec=2.0, min_safe_clearance_m=0.3)
    rollout = _straight_rollout(robot)
    label = assess_trajectory(rollout, 0.3, [], 0.0, robot, risk_cfg)
    assert label.min_clearance_m == float("inf")
    assert label.collision_within_horizon is False
    assert label.risk_score == pytest.approx(0.0)


def test_boundary_alone_produces_collision_with_zero_obstacles():
    """The core P0-2 regression: a trajectory that runs straight into a
    wall, with NO obstacle anywhere near it, must be flagged as a real
    collision with maximal risk -- not silently risk_target=0 just because
    no obstacle happened to be in the way. This is the exact failure mode
    code review found."""
    robot = make_robot()
    risk_cfg = RiskConfig(enabled=True, ttc_horizon_sec=2.0, min_safe_clearance_m=0.3)
    # Robot at the world origin, driving straight (+x) at 1 m/s for 2s ->
    # travels ~2m. A half_extent of 1.0m means the +x wall is only 1m away,
    # well within the rollout's reach.
    rollout = _straight_rollout(robot, horizon_sec=2.0, v=1.0)
    label = assess_trajectory(
        rollout, ego_radius=0.3, obstacles=[], steering_rad=0.0, robot=robot, risk_cfg=risk_cfg,
        robot_world_pose=(0.0, 0.0, 0.0), world_half_extent_m=1.0,
    )
    assert label.collision_within_horizon is True
    assert label.risk_score == pytest.approx(1.0)
    assert label.min_clearance_m < 0.0


def test_boundary_dominates_when_wall_is_closer_than_any_obstacle():
    robot = make_robot()
    risk_cfg = RiskConfig(enabled=True, ttc_horizon_sec=2.0, min_safe_clearance_m=0.3)
    rollout = _straight_rollout(robot, horizon_sec=2.0, v=1.0)
    from hunter_kinodynamic_rl.risk.labels import DynamicObstacle
    far_obstacle = DynamicObstacle(x0=50.0, y0=0.0, radius=0.3)  # nowhere near the rollout

    without_boundary = assess_trajectory(
        rollout, 0.3, [far_obstacle], 0.0, robot, risk_cfg,
    )
    with_boundary = assess_trajectory(
        rollout, 0.3, [far_obstacle], 0.0, robot, risk_cfg,
        robot_world_pose=(0.0, 0.0, 0.0), world_half_extent_m=1.0,
    )
    assert without_boundary.collision_within_horizon is False
    assert with_boundary.collision_within_horizon is True
    assert with_boundary.min_clearance_m < without_boundary.min_clearance_m


def test_boundary_does_not_falsely_trigger_deep_inside_a_large_world():
    robot = make_robot()
    risk_cfg = RiskConfig(enabled=True, ttc_horizon_sec=2.0, min_safe_clearance_m=0.3)
    rollout = _straight_rollout(robot, horizon_sec=2.0, v=1.0)
    label = assess_trajectory(
        rollout, 0.3, [], 0.0, robot, risk_cfg,
        robot_world_pose=(0.0, 0.0, 0.0), world_half_extent_m=100.0,
    )
    assert label.collision_within_horizon is False
    assert label.risk_score == pytest.approx(0.0)


# --------------------------------------------------- procedural_generator inset
def test_generate_scenario_insets_start_and_goal_from_the_world_wall():
    robot_radius = 0.45
    cfg = ScenarioConfig(world_size_m=10.0, min_obstacles=0, max_obstacles=2)
    half = cfg.world_size_m / 2.0
    for seed in range(20):
        spec = generate_scenario(seed, cfg, robot_radius=robot_radius)
        assert abs(spec.start_x) <= half - robot_radius + 1e-9
        assert abs(spec.start_y) <= half - robot_radius + 1e-9
        assert abs(spec.goal_x) <= half - robot_radius + 1e-9
        assert abs(spec.goal_y) <= half - robot_radius + 1e-9


def test_generate_scenario_raises_when_robot_radius_leaves_no_inset_room():
    cfg = ScenarioConfig(world_size_m=1.0, min_obstacles=0, max_obstacles=0)
    with pytest.raises(RuntimeError):
        generate_scenario(0, cfg, robot_radius=1.0)  # robot_radius >= half_extent


def test_generate_scenario_guarantees_min_obstacles_or_raises():
    """min_obstacles must be a HARD guarantee, not a best-effort target --
    the per-obstacle start/goal clearance rejection must not silently
    return fewer than requested (section P0-2)."""
    cfg = ScenarioConfig(world_size_m=10.0, min_obstacles=3, max_obstacles=3)
    for seed in range(10):
        spec = generate_scenario(seed, cfg, robot_radius=0.3)
        assert len(spec.static_obstacles) >= cfg.min_obstacles
