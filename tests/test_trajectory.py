import math

import pytest

from hunter_kinodynamic_rl.config.schema import (
    ActionSpaceConfig, CounterfactualConfig, RobotConfig, TrajectoryConfig,
)
from hunter_kinodynamic_rl.trajectory import action_space, trajectory_executor, trajectory_sampler
from hunter_kinodynamic_rl.trajectory.trajectory_primitive import ConstantCurvatureArc, make_primitive


def make_robot() -> RobotConfig:
    return RobotConfig(
        name="hunter_se", wheelbase_m=0.547696, track_width_m=0.503404,
        wheel_radius_m=0.136, length_m=0.76, width_m=0.4, height_m=0.14,
        mass_kg=42.0, steering_limit_deg=21.58, max_forward_speed_mps=2.0,
        min_forward_speed_mps=0.0, accel_limit_mps2=6.0, brake_decel_mps2=6.0,
        steering_rate_deg_s=200.0, speed_lag_tau_sec=0.0,
    )


def test_trajectory_action_bounds_are_respected():
    robot = make_robot()
    cfg = ActionSpaceConfig(mode="trajectory", horizon_length_min_m=0.5, horizon_length_max_m=3.0)
    for a in ([-1, -1, -1], [1, 1, 1], [0, 0, 0], [0.3, -0.7, 0.9]):
        cmd = action_space.normalized_to_trajectory_command(a, cfg, robot)
        assert -robot.max_curvature - 1e-9 <= cmd.kappa <= robot.max_curvature + 1e-9
        assert 0.0 <= cmd.v_ref <= robot.max_forward_speed_mps + 1e-9
        assert 0.5 - 1e-9 <= cmd.horizon_m <= 3.0 + 1e-9


def test_trajectory_action_extremes_map_correctly():
    robot = make_robot()
    cfg = ActionSpaceConfig(mode="trajectory", v_min_mps=0.0)
    cmd_min = action_space.normalized_to_trajectory_command([-1, -1, -1], cfg, robot)
    cmd_max = action_space.normalized_to_trajectory_command([1, 1, 1], cfg, robot)
    assert cmd_min.kappa == pytest.approx(-robot.max_curvature)
    assert cmd_max.kappa == pytest.approx(robot.max_curvature)
    assert cmd_min.v_ref == pytest.approx(0.0)
    assert cmd_max.v_ref == pytest.approx(robot.max_forward_speed_mps)


def test_legacy_waypoint_action_bounds():
    cfg = ActionSpaceConfig(mode="legacy_waypoint", legacy_r_min_m=0.0, legacy_r_max_m=2.0,
                             legacy_theta_max_rad=0.524)
    cmd = action_space.normalized_to_legacy_waypoint_command([1, 1, 0], cfg)
    assert cmd.r == pytest.approx(2.0)
    assert cmd.theta == pytest.approx(0.524)


def test_trajectory_command_normalized_roundtrip():
    robot = make_robot()
    cfg = ActionSpaceConfig(mode="trajectory", v_min_mps=0.0)
    for a in ([-1, -1, -1], [1, 1, 1], [0, 0, 0], [0.3, -0.7, 0.9], [-0.2, 0.5, -0.4]):
        cmd = action_space.normalized_to_trajectory_command(a, cfg, robot)
        back = action_space.trajectory_command_to_normalized(cmd, cfg, robot)
        for original, recovered in zip(a, back):
            assert recovered == pytest.approx(original, abs=1e-5)


def test_wrong_action_length_raises():
    robot = make_robot()
    cfg = ActionSpaceConfig()
    with pytest.raises(ValueError):
        action_space.normalized_to_trajectory_command([0.0, 0.0], cfg, robot)


def test_zero_curvature_arc_is_a_straight_line():
    arc = ConstantCurvatureArc(kappa=0.0, v_ref=1.0, horizon_m=2.0)
    p = arc.point_at(2.0)
    assert p.x == pytest.approx(2.0)
    assert p.y == pytest.approx(0.0)
    assert p.yaw == pytest.approx(0.0)


def test_arc_path_length_matches_horizon():
    arc = make_primitive("constant_curvature_arc", kappa=0.3, v_ref=1.0, horizon_m=1.5)
    assert arc.path_length_m() == pytest.approx(1.5)
    pts = arc.sample(10)
    assert len(pts) == 10
    assert pts[0].s_m == pytest.approx(0.0)
    assert pts[-1].s_m == pytest.approx(1.5)


def test_arc_curvature_is_constant_along_path():
    arc = make_primitive("constant_curvature_arc", kappa=0.4, v_ref=1.0, horizon_m=2.0)
    for p in arc.sample(20):
        assert p.curvature == pytest.approx(0.4)


def test_pure_pursuit_adapter_produces_bounded_command():
    robot = make_robot()
    traj_cfg = TrajectoryConfig()
    cmd = trajectory_executor.execute(
        [0.5, 0.8, 0.5], ActionSpaceConfig(mode="trajectory"), traj_cfg, robot,
    )
    assert 0.0 <= cmd.speed_mps <= robot.max_forward_speed_mps + 1e-6
    assert abs(cmd.steering_rad) <= robot.steering_limit_rad + 1e-6


def test_legacy_waypoint_executor_produces_bounded_command():
    robot = make_robot()
    traj_cfg = TrajectoryConfig()
    cmd = trajectory_executor.execute(
        [0.2, 0.1, -1.0], ActionSpaceConfig(mode="legacy_waypoint"), traj_cfg, robot,
    )
    assert 0.0 <= cmd.speed_mps <= robot.max_forward_speed_mps + 1e-6
    assert abs(cmd.steering_rad) <= robot.steering_limit_rad + 1e-6


def test_v_ref_is_executed_directly_regardless_of_L():
    """v_ref is the policy's TARGET speed and must be executed AS COMMANDED
    (only clamped to the robot's own physical bounds) -- an L-derived speed
    cap would make the published command silently disagree with what the
    policy actually asked for, which this research direction explicitly
    forbids (pure_pursuit_adapter.py's module docstring). Same kappa/v_ref,
    only L varying (short vs. long): the EXECUTED SPEED must be identical;
    L instead affects the rollout/risk horizon (see tests/test_l_semantics.py),
    never the control-level speed."""
    robot = make_robot()
    traj_cfg = TrajectoryConfig()
    action_cfg = ActionSpaceConfig(mode="trajectory", v_min_mps=0.0)

    # action[1]=1.0 -> v_ref = robot.max_forward_speed_mps (2.0); action[2]
    # controls L within [horizon_length_min_m, horizon_length_max_m].
    short_l_action = [0.0, 1.0, -1.0]  # L = horizon_length_min_m (0.5m default)
    long_l_action = [0.0, 1.0, 1.0]    # L = horizon_length_max_m (3.0m default)

    short_cmd = trajectory_executor.execute(short_l_action, action_cfg, traj_cfg, robot)
    long_cmd = trajectory_executor.execute(long_l_action, action_cfg, traj_cfg, robot)

    assert short_cmd.speed_mps == pytest.approx(long_cmd.speed_mps, abs=1e-6)
    assert short_cmd.speed_mps == pytest.approx(robot.max_forward_speed_mps, abs=1e-6)


def test_v_min_zero_allows_a_full_commanded_stop():
    """v_ref=0 (a commanded stop) must be executed as exactly 0, not
    silently raised by any L-derived floor/cap."""
    robot = make_robot()
    traj_cfg = TrajectoryConfig()
    action_cfg = ActionSpaceConfig(mode="trajectory", v_min_mps=0.0)
    cmd = trajectory_executor.execute([0.0, -1.0, 1.0], action_cfg, traj_cfg, robot)
    assert cmd.speed_mps == pytest.approx(0.0, abs=1e-6)


def test_counterfactual_candidates_stay_within_physical_bounds():
    robot = make_robot()
    cf_cfg = CounterfactualConfig(num_candidates=8, kappa_offsets_frac=[-1.0, -0.5, 0.5, 1.0],
                                   speed_fractions=[0.5, 1.0], include_stop_candidate=True)
    base = action_space.TrajectoryCommand(kappa=0.1, v_ref=1.0, horizon_m=1.5)
    candidates = trajectory_sampler.generate_candidates(base, robot, cf_cfg)
    assert candidates[0] == base
    assert len(candidates) <= cf_cfg.num_candidates
    for c in candidates:
        assert math.isfinite(c.kappa) and math.isfinite(c.v_ref) and math.isfinite(c.horizon_m)
        assert abs(c.kappa) <= robot.max_curvature + 1e-9
        assert 0.0 <= c.v_ref <= robot.max_forward_speed_mps + 1e-9
        assert c.horizon_m > 0.0


def test_counterfactual_includes_stop_candidate():
    robot = make_robot()
    cf_cfg = CounterfactualConfig(num_candidates=20, kappa_offsets_frac=[0.5], speed_fractions=[1.0],
                                   include_stop_candidate=True)
    base = action_space.TrajectoryCommand(kappa=0.1, v_ref=1.0, horizon_m=1.5)
    candidates = trajectory_sampler.generate_candidates(base, robot, cf_cfg)
    assert any(c.v_ref == pytest.approx(0.0) for c in candidates)
