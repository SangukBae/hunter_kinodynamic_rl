"""Proves L (trajectory horizon) is NOT a dead action dimension: it must
change the rollout's TIME horizon, the risk assessment horizon, and
therefore the risk outcome for a fixed obstacle scene -- see
docs/TRACTOR_TQC_MODEL_SPEC.md's "L semantics" section for the full design."""

import math

import pytest

from hunter_kinodynamic_rl.config.schema import ActionSpaceConfig, DynamicsConfig, RiskConfig, RobotConfig, TrajectoryConfig
from hunter_kinodynamic_rl.dynamics.ackermann_rollout import (
    horizon_from_trajectory, l_derived_horizon_sec, rollout_trajectory_command,
)
from hunter_kinodynamic_rl.risk.labels import DynamicObstacle
from hunter_kinodynamic_rl.risk.trajectory_risk import assess_trajectory_command
from hunter_kinodynamic_rl.robot.interface import VehicleState
from hunter_kinodynamic_rl.trajectory import trajectory_executor
from hunter_kinodynamic_rl.trajectory.pure_pursuit_adapter import trajectory_command_to_vehicle_command


def make_robot() -> RobotConfig:
    return RobotConfig(
        name="hunter_se", wheelbase_m=0.547696, track_width_m=0.503404,
        wheel_radius_m=0.136, length_m=0.76, width_m=0.4, height_m=0.14,
        mass_kg=42.0, steering_limit_deg=21.58, max_forward_speed_mps=2.0,
        min_forward_speed_mps=0.0, accel_limit_mps2=6.0, brake_decel_mps2=6.0,
        steering_rate_deg_s=200.0, speed_lag_tau_sec=0.0,
    )


def test_horizon_scales_with_L_over_v_above_the_safety_floor():
    # L/v_ref = 2.0 / 1.0 = 2.0s, comfortably above min_safety_horizon_sec (1.0s)
    # here, so the L-derived term (not the floor) is what's being compared.
    cfg = DynamicsConfig(horizon_min_sec=0.5, horizon_max_sec=3.0, min_safety_horizon_sec=1.0, dt_sec=0.1)
    short = horizon_from_trajectory(horizon_m=1.2, v_ref_mps=1.0, dynamics_cfg=cfg)
    long = horizon_from_trajectory(horizon_m=3.0, v_ref_mps=1.0, dynamics_cfg=cfg)
    assert short < long
    assert short == pytest.approx(1.2)
    assert long == pytest.approx(3.0)   # clamped to horizon_max_sec (3.0/1.0=3.0)


def test_horizon_never_drops_below_the_L_independent_safety_floor():
    """The core P0 fix: no matter how small L is, the assessed horizon
    cannot shrink below min_safety_horizon_sec -- this is what stops a
    policy from picking a tiny L purely to hide a distant collision from
    risk assessment."""
    cfg = DynamicsConfig(horizon_min_sec=0.05, horizon_max_sec=3.0, min_safety_horizon_sec=1.5, dt_sec=0.05)
    tiny_l = horizon_from_trajectory(horizon_m=0.01, v_ref_mps=2.0, dynamics_cfg=cfg)
    assert tiny_l == pytest.approx(cfg.min_safety_horizon_sec)


def test_zero_speed_uses_max_horizon_not_divide_by_zero():
    cfg = DynamicsConfig(horizon_min_sec=0.5, horizon_max_sec=3.0, dt_sec=0.1)
    h = horizon_from_trajectory(horizon_m=1.0, v_ref_mps=0.0, dynamics_cfg=cfg)
    assert h == pytest.approx(cfg.horizon_max_sec)


def test_different_L_same_kappa_v_produces_different_rollout_endpoint():
    robot = make_robot()
    cfg = DynamicsConfig(horizon_min_sec=0.3, horizon_max_sec=3.0, dt_sec=0.05, model_actuator_lag=False)
    state = VehicleState(v=1.0, steering=0.0)
    short_rollout = rollout_trajectory_command(state, kappa=0.0, v_ref_mps=1.0, horizon_m=0.5, robot=robot, dynamics_cfg=cfg)
    long_rollout = rollout_trajectory_command(state, kappa=0.0, v_ref_mps=1.0, horizon_m=3.0, robot=robot, dynamics_cfg=cfg)
    assert short_rollout.final_state.x != pytest.approx(long_rollout.final_state.x)
    assert short_rollout.points[-1].t_sec < long_rollout.points[-1].t_sec


def test_different_L_changes_risk_outcome_for_an_obstacle_only_a_long_horizon_reaches():
    """An obstacle placed far enough that only the LONG-L candidate's
    rollout reaches it: short L must see it as safe (never rolled out that
    far), long L must detect the risk."""
    robot = make_robot()
    cfg = DynamicsConfig(horizon_min_sec=0.3, horizon_max_sec=3.0, dt_sec=0.05, model_actuator_lag=False)
    risk_cfg = RiskConfig(enabled=True, ttc_horizon_sec=3.0, min_safe_clearance_m=0.3)
    state = VehicleState(v=1.0, steering=0.0)
    obstacle = DynamicObstacle(x0=2.5, y0=0.0, radius=0.3)  # 2.5m ahead

    short_label = assess_trajectory_command(
        kappa=0.0, v_ref=1.0, horizon_m=0.5, initial_state=state, ego_radius=0.3,
        obstacles=[obstacle], robot=robot, dynamics_cfg=cfg, risk_cfg=risk_cfg,
    )
    long_label = assess_trajectory_command(
        kappa=0.0, v_ref=1.0, horizon_m=3.0, initial_state=state, ego_radius=0.3,
        obstacles=[obstacle], robot=robot, dynamics_cfg=cfg, risk_cfg=risk_cfg,
    )
    assert short_label.collision_within_horizon is False
    assert long_label.collision_within_horizon is True
    assert long_label.risk_score > short_label.risk_score


def test_ttc_horizon_never_exceeds_L_derived_rollout_horizon():
    robot = make_robot()
    cfg = DynamicsConfig(horizon_min_sec=0.3, horizon_max_sec=3.0, min_safety_horizon_sec=1.5,
                          dt_sec=0.05, model_actuator_lag=False)
    # risk_cfg requests a much longer TTC horizon than this candidate's derived horizon provides.
    risk_cfg = RiskConfig(enabled=True, ttc_horizon_sec=10.0, min_safe_clearance_m=0.3)
    state = VehicleState(v=1.0, steering=0.0)
    label = assess_trajectory_command(
        kappa=0.0, v_ref=1.0, horizon_m=0.5, initial_state=state, ego_radius=0.3,
        obstacles=[], robot=robot, dynamics_cfg=cfg, risk_cfg=risk_cfg,
    )
    # No collision -> TTC reports the (clamped) horizon it was actually assessed over (the
    # min_safety_horizon_sec floor, since L/v_ref=0.5 < 1.5), not risk_cfg's 10.0.
    assert label.time_to_collision_sec == pytest.approx(1.5, abs=1e-6)


def test_short_L_cannot_hide_a_distant_collision():
    """The regression test for the exact exploit code review found: an
    obstacle sits well within min_safety_horizon_sec's reach (at constant
    v_ref) but OUTSIDE what a tiny L/v_ref would cover on its own. Even
    though the policy commanded a minimal L, the L-independent safety floor
    must still catch this collision."""
    robot = make_robot()
    cfg = DynamicsConfig(horizon_min_sec=0.05, horizon_max_sec=3.0, min_safety_horizon_sec=1.5,
                          dt_sec=0.05, model_actuator_lag=False)
    risk_cfg = RiskConfig(enabled=True, ttc_horizon_sec=3.0, min_safe_clearance_m=0.3)
    state = VehicleState(v=1.0, steering=0.0)
    # At v_ref=1.0 m/s, a naive tiny-L horizon (e.g. L=0.05m -> 0.05s) would only
    # roll out ~0.05m -- nowhere near this obstacle at 1.2m. The 1.5s safety
    # floor rolls out 1.5m, which DOES reach it.
    obstacle = DynamicObstacle(x0=1.2, y0=0.0, radius=0.3)

    label = assess_trajectory_command(
        kappa=0.0, v_ref=1.0, horizon_m=0.05, initial_state=state, ego_radius=0.3,
        obstacles=[obstacle], robot=robot, dynamics_cfg=cfg, risk_cfg=risk_cfg,
    )
    assert label.collision_within_horizon is True
    assert label.risk_score == pytest.approx(1.0)


def test_lookahead_point_choice_is_provably_invariant_to_curvature_recovery():
    """The geometric fact the STEERING COMMIT WINDOW feature exists to work
    around: recovering curvature via Pure-Pursuit chord geometry from ANY
    point on an exact constant-curvature arc reproduces the arc's OWN
    kappa exactly, regardless of lookahead distance -- so sampling a
    different point on the arc (the pre-fix mechanism) can never make the
    published steering depend on L."""
    from hunter_kinodynamic_rl.trajectory import pure_pursuit
    from hunter_kinodynamic_rl.trajectory.trajectory_primitive import ConstantCurvatureArc
    from hunter_kinodynamic_rl.robot.limits import curvature_to_steering

    kappa, wheelbase = 0.3, 0.547696
    arc = ConstantCurvatureArc(kappa=kappa, v_ref=1.0, horizon_m=3.0)
    expected = curvature_to_steering(kappa, wheelbase)
    for s in (0.2, 1.0, 1.5, 2.9):
        wp = arc.point_at(s)
        steering = pure_pursuit.waypoint_to_command(
            wp.x, wp.y, wheelbase_m=wheelbase, steering_limit_rad=1.0,
            cruise_speed_mps=1.0, min_speed_mps=0.0, speed_steer_factor=0.0,
        )[1]
        assert steering == pytest.approx(expected, abs=1e-9)


def test_l_derived_horizon_sec_has_no_safety_floor_unlike_horizon_from_trajectory():
    """l_derived_horizon_sec is the raw, UNPADDED L/v_ref quantity used for
    the control-level steering commit window; horizon_from_trajectory pads
    it with min_safety_horizon_sec for risk purposes only. A tiny L must be
    able to make the control quantity fall to horizon_min_sec even when
    min_safety_horizon_sec is much larger, while horizon_from_trajectory
    stays floored."""
    cfg = DynamicsConfig(horizon_min_sec=0.2, horizon_max_sec=3.0, min_safety_horizon_sec=1.5, dt_sec=0.1)
    raw = l_derived_horizon_sec(horizon_m=0.05, v_ref_mps=2.0, dynamics_cfg=cfg)
    padded = horizon_from_trajectory(horizon_m=0.05, v_ref_mps=2.0, dynamics_cfg=cfg)
    assert raw == pytest.approx(cfg.horizon_min_sec)
    assert padded == pytest.approx(cfg.min_safety_horizon_sec)
    assert raw < padded


def test_same_kappa_v_ref_different_L_publishes_different_steering():
    """The core fix for the code-review finding: with the L-derived steering
    commit window engaged (dynamics_cfg + current_steering_rad supplied),
    two commands sharing (kappa, v_ref) but differing in L must publish
    DIFFERENT steering angles whenever the robot's current steering differs
    from the target -- proving L now has an observable effect on the
    actually-published command, not just the risk rollout."""
    robot = RobotConfig(
        name="hunter_se", wheelbase_m=0.547696, track_width_m=0.503404,
        wheel_radius_m=0.136, length_m=0.76, width_m=0.4, height_m=0.14,
        mass_kg=42.0, steering_limit_deg=21.58, max_forward_speed_mps=2.0,
        min_forward_speed_mps=0.0, accel_limit_mps2=6.0, brake_decel_mps2=6.0,
        steering_rate_deg_s=200.0, speed_lag_tau_sec=0.0,
    )
    traj_cfg = TrajectoryConfig(dt_sec=0.1)
    dyn_cfg = DynamicsConfig(horizon_min_sec=0.2, horizon_max_sec=3.0, min_safety_horizon_sec=0.2, dt_sec=0.1)
    current_steering = 0.0
    kappa, v_ref = 0.3, 1.0

    short_cmd = trajectory_command_to_vehicle_command(
        kappa, v_ref, horizon_m=0.5, robot=robot, trajectory_cfg=traj_cfg,
        dynamics_cfg=dyn_cfg, current_steering_rad=current_steering,
    )
    long_cmd = trajectory_command_to_vehicle_command(
        kappa, v_ref, horizon_m=3.0, robot=robot, trajectory_cfg=traj_cfg,
        dynamics_cfg=dyn_cfg, current_steering_rad=current_steering,
    )
    assert short_cmd.steering_rad != pytest.approx(long_cmd.steering_rad)
    # A LONGER commit window blends in LESS of the target this tick -> its
    # steering must stay strictly closer to the current (0.0) angle.
    assert abs(long_cmd.steering_rad) < abs(short_cmd.steering_rad)
    # Both must still make PROGRESS toward the target, same direction.
    assert short_cmd.steering_rad > 0.0 and long_cmd.steering_rad > 0.0
    # v_ref is untouched regardless of L (the forbidden design this feature
    # must not reintroduce).
    assert short_cmd.speed_mps == pytest.approx(long_cmd.speed_mps)
    assert short_cmd.speed_mps == pytest.approx(v_ref)


def test_steering_commit_blend_never_exceeds_the_actuator_rate_limit():
    robot = RobotConfig(
        name="hunter_se", wheelbase_m=0.547696, track_width_m=0.503404,
        wheel_radius_m=0.136, length_m=0.76, width_m=0.4, height_m=0.14,
        mass_kg=42.0, steering_limit_deg=21.58, max_forward_speed_mps=2.0,
        min_forward_speed_mps=0.0, accel_limit_mps2=6.0, brake_decel_mps2=6.0,
        steering_rate_deg_s=30.0, speed_lag_tau_sec=0.0,  # deliberately slow actuator
    )
    traj_cfg = TrajectoryConfig(dt_sec=0.1)
    dyn_cfg = DynamicsConfig(horizon_min_sec=0.05, horizon_max_sec=3.0, min_safety_horizon_sec=0.05, dt_sec=0.1)
    # Shortest possible L at high speed -> blend fraction saturates toward 1,
    # which would otherwise instantly jump to a large target steering angle.
    cmd = trajectory_command_to_vehicle_command(
        kappa=robot.max_curvature, v_ref=robot.max_forward_speed_mps, horizon_m=0.05,
        robot=robot, trajectory_cfg=traj_cfg, dynamics_cfg=dyn_cfg, current_steering_rad=0.0,
    )
    rate_limit_rad = math.radians(robot.steering_rate_deg_s) * traj_cfg.dt_sec
    assert abs(cmd.steering_rad) <= rate_limit_rad + 1e-9


def test_steering_commit_window_disabled_when_dynamics_cfg_or_current_steering_omitted():
    """Backward compatibility: omitting EITHER optional argument (the
    default for every caller not explicitly updated, e.g. the pre-existing
    tests in tests/test_trajectory.py) must reproduce the exact legacy
    instantaneous-steering value."""
    robot = RobotConfig(
        name="hunter_se", wheelbase_m=0.547696, track_width_m=0.503404,
        wheel_radius_m=0.136, length_m=0.76, width_m=0.4, height_m=0.14,
        mass_kg=42.0, steering_limit_deg=21.58, max_forward_speed_mps=2.0,
        min_forward_speed_mps=0.0, accel_limit_mps2=6.0, brake_decel_mps2=6.0,
        steering_rate_deg_s=200.0, speed_lag_tau_sec=0.0,
    )
    traj_cfg = TrajectoryConfig(dt_sec=0.1)
    dyn_cfg = DynamicsConfig(dt_sec=0.1)
    kappa, v_ref, horizon_m = 0.3, 1.0, 1.5

    legacy = trajectory_command_to_vehicle_command(kappa, v_ref, horizon_m, robot, traj_cfg)
    only_dynamics = trajectory_command_to_vehicle_command(
        kappa, v_ref, horizon_m, robot, traj_cfg, dynamics_cfg=dyn_cfg,
    )
    only_steering = trajectory_command_to_vehicle_command(
        kappa, v_ref, horizon_m, robot, traj_cfg, current_steering_rad=0.0,
    )
    assert only_dynamics.steering_rad == pytest.approx(legacy.steering_rad)
    assert only_steering.steering_rad == pytest.approx(legacy.steering_rad)


def test_executor_forwards_dynamics_and_current_steering_to_the_adapter():
    """trajectory_executor.execute must actually plumb dynamics_cfg/
    current_steering_rad through -- proven by observing the SAME
    same-kappa-v_ref-different-L divergence at the executor entrypoint
    every real caller (environment_node.py, real_policy_node.py) uses."""
    robot = RobotConfig(
        name="hunter_se", wheelbase_m=0.547696, track_width_m=0.503404,
        wheel_radius_m=0.136, length_m=0.76, width_m=0.4, height_m=0.14,
        mass_kg=42.0, steering_limit_deg=21.58, max_forward_speed_mps=2.0,
        min_forward_speed_mps=0.0, accel_limit_mps2=6.0, brake_decel_mps2=6.0,
        steering_rate_deg_s=200.0, speed_lag_tau_sec=0.0,
    )
    traj_cfg = TrajectoryConfig(dt_sec=0.1)
    dyn_cfg = DynamicsConfig(horizon_min_sec=0.2, horizon_max_sec=3.0, min_safety_horizon_sec=0.2, dt_sec=0.1)
    action_cfg = ActionSpaceConfig(mode="trajectory", horizon_length_min_m=0.5, horizon_length_max_m=3.0,
                                    v_min_mps=0.0)
    # action[0]=0.3 (fixed kappa direction), action[1]=1.0 (v_ref=v_max), only L (action[2]) differs.
    short_cmd = trajectory_executor.execute(
        [0.3, 1.0, -1.0], action_cfg, traj_cfg, robot,
        dynamics_cfg=dyn_cfg, current_steering_rad=0.0,
    )
    long_cmd = trajectory_executor.execute(
        [0.3, 1.0, 1.0], action_cfg, traj_cfg, robot,
        dynamics_cfg=dyn_cfg, current_steering_rad=0.0,
    )
    assert short_cmd.steering_rad != pytest.approx(long_cmd.steering_rad)
    assert short_cmd.speed_mps == pytest.approx(long_cmd.speed_mps)


def test_ackermann_rollout_feature_flag_gates_production_risk_call(monkeypatch):
    """features.ackermann_rollout must actually be READ somewhere in the
    production path -- this test just confirms the flag exists in the
    schema and defaults to True for trajectory-mode profiles; the real
    production gating is exercised in test_environment_node.py."""
    from hunter_kinodynamic_rl.config.loader import load_profile
    off_profile = load_profile("baseline_tqc")
    on_profile = load_profile("kinodynamic_tqc")
    assert off_profile.features.ackermann_rollout is False
    assert on_profile.features.ackermann_rollout is True
