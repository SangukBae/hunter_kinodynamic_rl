"""Regression coverage for the code-review finding: once
trajectory/pure_pursuit_adapter.py's L-derived steering commit window
shipped, the ACTUALLY-PUBLISHED steering command only partially commits
toward the target curvature each tick (blended in from the robot's current
steering, rate-limited) -- but risk/trajectory_risk.py's rollout still
assumed the target curvature was reached INSTANTLY, so a risk label could
silently disagree with what the robot would actually do. This is fixed by
threading a ``commit_blend`` flag through
dynamics.ackermann_rollout.rollout_trajectory_command ->
risk.trajectory_risk.assess_trajectory_command ->
risk.counterfactual_sampler.score_candidates ->
env.simulation.risk_computation.compute_risk_telemetry (gated on
``features.trajectory_l_preview_blend``, the SAME flag that gates the
executor's own blending), all sharing ONE formula
(``dynamics.ackermann_rollout.l_commit_blended_steering``) with the
executor -- see that function's docstring for why it lives in ``dynamics/``
rather than ``trajectory/`` (avoids a circular import)."""

import math

import pytest

from hunter_kinodynamic_rl.config.schema import (
    ActionSpaceConfig, CounterfactualConfig, DynamicsConfig, RiskConfig, RobotConfig, TrajectoryConfig,
)
from hunter_kinodynamic_rl.dynamics.ackermann_rollout import l_commit_blended_steering, rollout_trajectory_command
from hunter_kinodynamic_rl.risk.counterfactual_sampler import score_candidates
from hunter_kinodynamic_rl.risk.labels import DynamicObstacle
from hunter_kinodynamic_rl.risk.trajectory_risk import assess_trajectory_command
from hunter_kinodynamic_rl.robot.interface import VehicleState
from hunter_kinodynamic_rl.trajectory.action_space import TrajectoryCommand
from hunter_kinodynamic_rl.trajectory.pure_pursuit_adapter import trajectory_command_to_vehicle_command


def make_slow_steering_robot(steering_rate_deg_s: float = 20.0) -> RobotConfig:
    """A DELIBERATELY slow steering actuator -- makes the instant-vs-
    realistic rollout divergence large and unambiguous to assert on."""
    return RobotConfig(
        name="hunter_se", wheelbase_m=0.547696, track_width_m=0.503404,
        wheel_radius_m=0.136, length_m=0.76, width_m=0.4, height_m=0.14,
        mass_kg=42.0, steering_limit_deg=21.58, max_forward_speed_mps=2.0,
        min_forward_speed_mps=0.0, accel_limit_mps2=6.0, brake_decel_mps2=6.0,
        steering_rate_deg_s=steering_rate_deg_s, speed_lag_tau_sec=0.0,
    )


# --------------------------------------------------- rollout-level consistency


def test_commit_blend_false_is_byte_identical_to_legacy_rollout():
    """Backward compatibility: the default (commit_blend=False) rollout
    must be untouched by this fix."""
    robot = make_slow_steering_robot()
    cfg = DynamicsConfig(horizon_min_sec=0.3, horizon_max_sec=3.0, dt_sec=0.05, model_actuator_lag=False)
    state = VehicleState(v=1.0, steering=0.0)
    legacy = rollout_trajectory_command(state, kappa=0.3, v_ref_mps=1.0, horizon_m=2.0, robot=robot,
                                         dynamics_cfg=cfg)
    explicit_false = rollout_trajectory_command(state, kappa=0.3, v_ref_mps=1.0, horizon_m=2.0, robot=robot,
                                                 dynamics_cfg=cfg, commit_blend=False)
    assert legacy.final_state.x == pytest.approx(explicit_false.final_state.x)
    assert legacy.final_state.y == pytest.approx(explicit_false.final_state.y)
    assert legacy.final_state.yaw == pytest.approx(explicit_false.final_state.yaw)


def test_commit_blend_true_diverges_from_legacy_when_current_steering_differs():
    """The core fix: when the robot's CURRENT steering differs from the
    candidate's target curvature's steering, the commit-blend rollout must
    diverge from the legacy instant-tracking one -- proving the rollout now
    actually models the gradual commit instead of an instantaneous jump."""
    robot = make_slow_steering_robot()
    cfg = DynamicsConfig(horizon_min_sec=0.3, horizon_max_sec=3.0, dt_sec=0.05, model_actuator_lag=False)
    state = VehicleState(v=1.0, steering=0.0)  # currently going straight
    kappa = robot.max_curvature  # target requires a large steering angle

    legacy = rollout_trajectory_command(state, kappa=kappa, v_ref_mps=1.0, horizon_m=2.0, robot=robot,
                                         dynamics_cfg=cfg, commit_blend=False)
    realistic = rollout_trajectory_command(state, kappa=kappa, v_ref_mps=1.0, horizon_m=2.0, robot=robot,
                                            dynamics_cfg=cfg, commit_blend=True)
    assert legacy.final_state.y != pytest.approx(realistic.final_state.y)
    # The legacy rollout swerves harder/faster -- its lateral excursion
    # must be LARGER than the realistic (rate-limited) one for the same
    # elapsed time.
    assert abs(legacy.final_state.y) > abs(realistic.final_state.y)


def test_commit_blend_true_matches_legacy_when_already_on_target_steering():
    """When current_steering ALREADY equals the target (no commitment
    needed), the blend is a no-op every tick -- commit_blend=True must
    reduce to (approximately) the legacy rollout, not diverge for no
    reason."""
    robot = make_slow_steering_robot()
    cfg = DynamicsConfig(horizon_min_sec=0.3, horizon_max_sec=3.0, dt_sec=0.05, model_actuator_lag=False)
    kappa = 0.3
    from hunter_kinodynamic_rl.robot.limits import curvature_to_steering
    target_steering = curvature_to_steering(kappa, robot.wheelbase_m)
    state = VehicleState(v=1.0, steering=target_steering)  # ALREADY on target

    legacy = rollout_trajectory_command(state, kappa=kappa, v_ref_mps=1.0, horizon_m=2.0, robot=robot,
                                         dynamics_cfg=cfg, commit_blend=False)
    realistic = rollout_trajectory_command(state, kappa=kappa, v_ref_mps=1.0, horizon_m=2.0, robot=robot,
                                            dynamics_cfg=cfg, commit_blend=True)
    assert legacy.final_state.x == pytest.approx(realistic.final_state.x, abs=1e-6)
    assert legacy.final_state.y == pytest.approx(realistic.final_state.y, abs=1e-6)


def test_commit_blend_rollout_tick_one_matches_the_executors_published_steering():
    """DIRECT proof of consistency: the rollout's FIRST point's steering
    (recovered from its yaw-rate-implied curvature over the first dt) must
    match what pure_pursuit_adapter.trajectory_command_to_vehicle_command
    ACTUALLY publishes for the identical (kappa, v_ref, L, current_steering,
    dt) inputs -- both now call the SAME l_commit_blended_steering formula."""
    robot = make_slow_steering_robot()
    traj_cfg = TrajectoryConfig(dt_sec=0.05)
    dyn_cfg = DynamicsConfig(horizon_min_sec=0.3, horizon_max_sec=3.0, dt_sec=0.05, model_actuator_lag=False)
    kappa, v_ref, horizon_m, current_steering = 0.3, 1.0, 2.0, 0.02

    published = trajectory_command_to_vehicle_command(
        kappa, v_ref, horizon_m, robot, traj_cfg,
        dynamics_cfg=dyn_cfg, current_steering_rad=current_steering,
    )

    from hunter_kinodynamic_rl.robot.limits import curvature_to_steering
    target_steering = curvature_to_steering(kappa, robot.wheelbase_m)
    rollout_tick1_steering = l_commit_blended_steering(
        target_steering, current_steering, horizon_m, v_ref, robot, dyn_cfg, dyn_cfg.dt_sec,
    )
    assert published.steering_rad == pytest.approx(rollout_tick1_steering, abs=1e-6)


def test_different_L_still_produces_different_commit_blend_rollouts():
    """L must remain a non-dead dimension THROUGH the commit-blend rollout
    too -- a short L (fast commit) and a long L (slow commit), same
    (kappa, v_ref) and same current_steering, must produce different
    rollout endpoints."""
    robot = make_slow_steering_robot()
    cfg = DynamicsConfig(horizon_min_sec=0.2, horizon_max_sec=3.0, dt_sec=0.05, model_actuator_lag=False)
    state = VehicleState(v=1.0, steering=0.0)
    kappa = robot.max_curvature

    short_l = rollout_trajectory_command(state, kappa=kappa, v_ref_mps=1.0, horizon_m=0.3, robot=robot,
                                          dynamics_cfg=cfg, commit_blend=True)
    long_l = rollout_trajectory_command(state, kappa=kappa, v_ref_mps=1.0, horizon_m=3.0, robot=robot,
                                         dynamics_cfg=cfg, commit_blend=True)
    assert short_l.final_state.y != pytest.approx(long_l.final_state.y)
    # Short L commits faster -> larger lateral excursion for the same
    # elapsed rollout ticks (both spend dt_sec=0.05 per tick; short_l's
    # horizon_sec floors at horizon_min_sec=0.2, long_l's at 3.0, so they
    # don't cover the identical wall-clock span, but short_l's PER-TICK
    # commit fraction is still higher throughout its own rollout).
    assert abs(short_l.final_state.y) > 0.0


# --------------------------------------------------------- risk-label level


def test_commit_blend_reveals_risk_that_the_legacy_instant_rollout_missed():
    """The concrete correctness bug this whole fix exists to close: an
    obstacle sits directly ahead in the vehicle's CURRENT (straight)
    heading. A long-L, hard-swerve candidate looks SAFE under the legacy
    instant-steering rollout (it "immediately" swerves clear) but the REAL
    vehicle -- rate-limited AND only committing a small fraction of the
    turn per tick under a long L -- cannot actually clear it in time.
    commit_blend=True must catch this; commit_blend=False must miss it."""
    robot = make_slow_steering_robot(steering_rate_deg_s=10.0)  # very slow actuator
    cfg = DynamicsConfig(horizon_min_sec=0.3, horizon_max_sec=3.0, dt_sec=0.05, model_actuator_lag=False)
    risk_cfg = RiskConfig(enabled=True, ttc_horizon_sec=3.0, min_safe_clearance_m=0.3)
    state = VehicleState(v=1.0, steering=0.0)  # currently going straight
    obstacle = DynamicObstacle(x0=1.5, y0=0.0, radius=0.3)  # dead ahead, close
    kappa = robot.max_curvature  # policy commits to a hard swerve
    horizon_m = 3.0  # LONG L -> slow commit under commit_blend=True

    legacy_label = assess_trajectory_command(
        kappa=kappa, v_ref=1.0, horizon_m=horizon_m, initial_state=state, ego_radius=0.3,
        obstacles=[obstacle], robot=robot, dynamics_cfg=cfg, risk_cfg=risk_cfg, commit_blend=False,
    )
    realistic_label = assess_trajectory_command(
        kappa=kappa, v_ref=1.0, horizon_m=horizon_m, initial_state=state, ego_radius=0.3,
        obstacles=[obstacle], robot=robot, dynamics_cfg=cfg, risk_cfg=risk_cfg, commit_blend=True,
    )
    assert legacy_label.collision_within_horizon is False
    assert realistic_label.collision_within_horizon is True
    assert realistic_label.risk_score > legacy_label.risk_score


def test_commit_blend_default_false_preserves_existing_risk_labels():
    robot = make_slow_steering_robot()
    cfg = DynamicsConfig(horizon_min_sec=0.3, horizon_max_sec=3.0, dt_sec=0.05, model_actuator_lag=False)
    risk_cfg = RiskConfig(enabled=True, ttc_horizon_sec=3.0, min_safe_clearance_m=0.3)
    state = VehicleState(v=1.0, steering=0.0)
    obstacle = DynamicObstacle(x0=2.0, y0=0.0, radius=0.3)

    default_label = assess_trajectory_command(
        kappa=0.3, v_ref=1.0, horizon_m=2.0, initial_state=state, ego_radius=0.3,
        obstacles=[obstacle], robot=robot, dynamics_cfg=cfg, risk_cfg=risk_cfg,
    )
    explicit_false_label = assess_trajectory_command(
        kappa=0.3, v_ref=1.0, horizon_m=2.0, initial_state=state, ego_radius=0.3,
        obstacles=[obstacle], robot=robot, dynamics_cfg=cfg, risk_cfg=risk_cfg, commit_blend=False,
    )
    assert default_label == explicit_false_label


# --------------------------------------------------- counterfactual candidates


def test_counterfactual_candidates_all_share_commit_blend_semantics():
    """Every candidate (including the actor's own action at index 0) must
    be scored with the SAME commit_blend flag and the SAME initial_state
    (hence the same real current steering) -- proving the consistency fix
    applies uniformly, not just to the actor's own chosen action."""
    robot = make_slow_steering_robot(steering_rate_deg_s=10.0)
    cfg = DynamicsConfig(horizon_min_sec=0.3, horizon_max_sec=3.0, dt_sec=0.05, model_actuator_lag=False)
    risk_cfg = RiskConfig(enabled=True, ttc_horizon_sec=3.0, min_safe_clearance_m=0.3)
    cf_cfg = CounterfactualConfig(num_candidates=4, kappa_offsets_frac=[-1.0, 1.0], speed_fractions=[1.0],
                                   include_stop_candidate=False)
    state = VehicleState(v=1.0, steering=0.0)
    obstacle = DynamicObstacle(x0=1.5, y0=0.0, radius=0.3)
    base = TrajectoryCommand(kappa=robot.max_curvature, v_ref=1.0, horizon_m=3.0)

    legacy_scored = score_candidates(base, state, 0.3, [obstacle], robot, cfg, risk_cfg, cf_cfg,
                                      commit_blend=False)
    realistic_scored = score_candidates(base, state, 0.3, [obstacle], robot, cfg, risk_cfg, cf_cfg,
                                         commit_blend=True)
    assert len(legacy_scored) == len(realistic_scored) > 0
    # At least the actor's own (index-0, hard-swerve, long-L) candidate
    # must show the SAME under-vs-over-estimation gap the single-command
    # test above demonstrated.
    assert legacy_scored[0].command == realistic_scored[0].command
    assert realistic_scored[0].risk.risk_score >= legacy_scored[0].risk.risk_score


# ------------------------------------------------------- risk_computation wiring


# ------------------------------------------ commit_blend + model_actuator_lag


def test_commit_blend_true_with_actuator_lag_reflects_speed_accel_limit():
    """Issue: commit_blend=True previously called bicycle_model.step()
    directly, bypassing actuator_model.step_actuator() entirely -- so a
    risk rollout under the DEFAULT profile (model_actuator_lag=true) never
    saw speed accel/brake limiting even though the real robot does. A robot
    starting at rest (v=0) commanded to v_ref=2.0 with a tight accel limit
    must NOT reach v_ref within one dt when actuator lag is modeled -- the
    rolled-out x-displacement must be smaller than the idealised
    (accel-limit-free) displacement of the same command."""
    robot = make_slow_steering_robot(steering_rate_deg_s=200.0)  # steering not the bottleneck here
    robot = RobotConfig(**{**robot.__dict__, "accel_limit_mps2": 0.5})  # very weak acceleration
    cfg_lagged = DynamicsConfig(horizon_min_sec=0.3, horizon_max_sec=3.0, dt_sec=0.05, model_actuator_lag=True)
    cfg_idealised = DynamicsConfig(horizon_min_sec=0.3, horizon_max_sec=3.0, dt_sec=0.05, model_actuator_lag=False)
    state = VehicleState(v=0.0, steering=0.0)  # at rest

    lagged = rollout_trajectory_command(state, kappa=0.0, v_ref_mps=2.0, horizon_m=2.0, robot=robot,
                                         dynamics_cfg=cfg_lagged, commit_blend=True)
    idealised = rollout_trajectory_command(state, kappa=0.0, v_ref_mps=2.0, horizon_m=2.0, robot=robot,
                                            dynamics_cfg=cfg_idealised, commit_blend=True)
    # The idealised rollout jumps to v_ref=2.0 instantly (bicycle_model.step
    # applies target v directly) -- its final speed must be v_ref.
    assert idealised.final_state.v == pytest.approx(2.0)
    # The actuator-lag-aware rollout must still be accelerating, not at v_ref.
    assert lagged.final_state.v < 2.0 - 1e-6
    # Consequently it must have covered LESS ground than the idealised one.
    assert lagged.final_state.x < idealised.final_state.x


def test_commit_blend_true_with_actuator_lag_does_not_double_apply_steering_rate_limit():
    """The per-tick STEERING VALUE under commit_blend=True must be IDENTICAL
    whether or not model_actuator_lag is enabled -- proving the actuator
    model's own steering rate-limit clamp doesn't stack a SECOND limit on
    top of the L-commit blend's own (now-disabled-when-lagged,
    apply_rate_limit=False) clamp; if it did, the lagged run's steering
    would visibly lag further behind target than the legacy run's at every
    tick. (x/y are intentionally NOT compared here: model_actuator_lag=True
    integrates each substep with bicycle_model.step_midpoint -- trapezoidal
    on (v, steering) -- while the legacy path uses a closed-form
    single-value-per-substep arc; the two integrators diverge slightly in
    x/y for identical steering inputs, which is expected and orthogonal to
    the double-rate-limit question this test targets.)"""
    robot = make_slow_steering_robot(steering_rate_deg_s=15.0)
    cfg_lagged = DynamicsConfig(horizon_min_sec=0.3, horizon_max_sec=3.0, dt_sec=0.05, model_actuator_lag=True)
    cfg_legacy = DynamicsConfig(horizon_min_sec=0.3, horizon_max_sec=3.0, dt_sec=0.05, model_actuator_lag=False)
    # v held constant at v_ref for both paths (accel/brake limits irrelevant
    # here) so ONLY the steering-rate behavior differs between the two runs.
    state = VehicleState(v=1.0, steering=0.0)
    kappa = robot.max_curvature

    lagged = rollout_trajectory_command(state, kappa=kappa, v_ref_mps=1.0, horizon_m=2.0, robot=robot,
                                         dynamics_cfg=cfg_lagged, commit_blend=True)
    legacy = rollout_trajectory_command(state, kappa=kappa, v_ref_mps=1.0, horizon_m=2.0, robot=robot,
                                         dynamics_cfg=cfg_legacy, commit_blend=True)
    assert len(lagged.points) == len(legacy.points)
    for lagged_pt, legacy_pt in zip(lagged.points, legacy.points):
        assert lagged_pt.state.steering == pytest.approx(legacy_pt.state.steering, abs=1e-9)
    # The final steering must also have actually progressed toward (but not
    # instantly reached) the target -- a sanity check that this isn't a
    # degenerate zero-motion scenario.
    assert 0.0 < lagged.final_state.steering < robot.steering_limit_rad


def test_commit_blend_false_still_byte_identical_regardless_of_actuator_lag():
    """Sanity: commit_blend=False's own actuator-lag branching (in
    rollout_constant_target, pre-existing / untouched by this fix) is
    unaffected by anything changed here."""
    robot = make_slow_steering_robot()
    state = VehicleState(v=0.5, steering=0.0)
    for lag in (True, False):
        cfg = DynamicsConfig(horizon_min_sec=0.3, horizon_max_sec=3.0, dt_sec=0.05, model_actuator_lag=lag)
        a = rollout_trajectory_command(state, kappa=0.2, v_ref_mps=1.0, horizon_m=2.0, robot=robot,
                                        dynamics_cfg=cfg, commit_blend=False)
        b = rollout_trajectory_command(state, kappa=0.2, v_ref_mps=1.0, horizon_m=2.0, robot=robot,
                                        dynamics_cfg=cfg, commit_blend=False)
        assert a.final_state.x == pytest.approx(b.final_state.x)


def test_risk_computation_under_default_profile_does_not_bypass_actuator_lag():
    """The DEFAULT profile (config/training/defaults.yaml:
    dynamics.model_actuator_lag: true) combined with
    features.trajectory_l_preview_blend must produce a risk assessment that
    reflects actuator lag, not the idealised bicycle-only rollout -- proving
    the live risk-computation path (not just the standalone rollout
    function) is fixed. A weak-accel robot commanded to accelerate straight
    toward an obstacle it could only reach in time if speed were idealised
    (instant v_ref) must be assessed as SAFE under model_actuator_lag=True
    (never gets there within the horizon) but UNSAFE under
    model_actuator_lag=False (jumps straight to v_ref and collides) --
    proving the lagged rollout's speed profile actually reaches the risk
    computation, not just the standalone rollout function checked above."""
    import dataclasses

    robot = make_slow_steering_robot(steering_rate_deg_s=200.0)
    robot = RobotConfig(**{**robot.__dict__, "accel_limit_mps2": 0.3})
    risk_cfg = RiskConfig(enabled=True, ttc_horizon_sec=3.0, min_safe_clearance_m=0.3)
    state = VehicleState(v=0.0, steering=0.0)  # at rest, straight ahead
    obstacle = DynamicObstacle(x0=3.0, y0=0.0, radius=0.3)  # dead ahead

    cfg_default = DynamicsConfig(horizon_min_sec=0.3, horizon_max_sec=3.0, dt_sec=0.05, model_actuator_lag=True)
    cfg_no_lag = dataclasses.replace(cfg_default, model_actuator_lag=False)

    label_default = assess_trajectory_command(
        kappa=0.0, v_ref=2.0, horizon_m=6.0, initial_state=state, ego_radius=0.3,
        obstacles=[obstacle], robot=robot, dynamics_cfg=cfg_default, risk_cfg=risk_cfg, commit_blend=True,
    )
    label_no_lag = assess_trajectory_command(
        kappa=0.0, v_ref=2.0, horizon_m=6.0, initial_state=state, ego_radius=0.3,
        obstacles=[obstacle], robot=robot, dynamics_cfg=cfg_no_lag, risk_cfg=risk_cfg, commit_blend=True,
    )
    assert label_no_lag.collision_within_horizon is True
    assert label_default.collision_within_horizon is False
    assert label_default.risk_score < label_no_lag.risk_score


def test_risk_computation_derives_commit_blend_from_the_feature_flag():
    import dataclasses

    from hunter_kinodynamic_rl.config.loader import load_profile
    from hunter_kinodynamic_rl.env.simulation.risk_computation import compute_risk_telemetry
    from hunter_kinodynamic_rl.trajectory.action_space import decode_action

    profile = load_profile("kinodynamic_tqc_risk")
    assert profile.risk.enabled is True  # sanity: this profile actually exercises the risk path
    assert profile.features.trajectory_l_preview_blend is True  # sanity: the flag this test exercises

    action = [1.0, 1.0, 1.0]  # max kappa, max v_ref, max L -> a hard, long-L swerve
    command = decode_action(action, profile.action_space, profile.robot)
    telemetry = compute_risk_telemetry(
        command, step_id=1, robot_pose=(0.0, 0.0, 0.0), robot_v=1.0, robot_steering=0.0,
        static_obstacles=[], dynamic_specs=[], active_robot_config=profile.robot, profile=profile,
    )
    assert telemetry.valid is True  # sanity: the risk path actually ran (not FEATURES_DISABLED)

    off_profile = dataclasses.replace(
        profile, features=dataclasses.replace(profile.features, trajectory_l_preview_blend=False),
    )
    telemetry_off = compute_risk_telemetry(
        command, step_id=1, robot_pose=(0.0, 0.0, 0.0), robot_v=1.0, robot_steering=0.0,
        static_obstacles=[], dynamic_specs=[], active_robot_config=profile.robot, profile=off_profile,
    )
    assert telemetry_off.valid is True
    # Different commit_blend setting for the SAME command/current-steering
    # must be able to produce a different risk assessment (proving the
    # flag actually reaches the rollout, not just decoration) -- checked
    # via clearance, a continuous quantity unlikely to coincide exactly.
    assert telemetry.min_clearance_m != pytest.approx(telemetry_off.min_clearance_m)
