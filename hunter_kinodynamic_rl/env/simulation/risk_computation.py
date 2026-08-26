"""Pure risk-telemetry computation -- the ENTIRE body of
``environment_node.py``'s ``_compute_and_publish_risk``, minus the ROS
publish call, extracted here so the two P0 bugs code review found (static
obstacles omitted; state_t+1 used instead of the pre-action state_t
snapshot) have a single, directly-testable, ROS-free source of truth
instead of being buried in a stateful node method.

Callers (``environment_node.py``) MUST pass a PRE-ACTION snapshot (see
``build_privileged_obstacles``'s docstring) -- this module has no way to
enforce that itself, which is exactly why
``tests/test_risk_computation.py`` exists: it constructs BOTH a pre-action
and a post-action snapshot for the same synthetic scenario and asserts the
label differs, so a future accidental swap back to post-action state fails
loudly.
"""

from __future__ import annotations

import math
from typing import List

from hunter_kinodynamic_rl.config.schema import CounterfactualConfig, DynamicsConfig, Profile, RiskConfig, RobotConfig
from hunter_kinodynamic_rl.env.scenarios.procedural_generator import DynamicObstacleSpec, StaticObstacle
from hunter_kinodynamic_rl.env.simulation import risk_telemetry as rt
from hunter_kinodynamic_rl.env.simulation.privileged_snapshot import build_privileged_obstacles
from hunter_kinodynamic_rl.risk.counterfactual_sampler import best_candidate, safer_alternative_margin, score_candidates
from hunter_kinodynamic_rl.risk.trajectory_risk import assess_trajectory_command_with_rollout
from hunter_kinodynamic_rl.risk.unrecoverable_state import is_unrecoverable
from hunter_kinodynamic_rl.robot.interface import VehicleState
from hunter_kinodynamic_rl.robot.limits import steering_to_curvature
from hunter_kinodynamic_rl.trajectory.action_space import TrajectoryCommand
from hunter_kinodynamic_rl.trajectory.pure_pursuit_adapter import VehicleCommand


def _goal_local_xy(robot_pose, goal_world_xy):
    if goal_world_xy is None:
        return None
    rx, ry, yaw = robot_pose
    dx, dy = goal_world_xy[0] - rx, goal_world_xy[1] - ry
    c, s = math.cos(yaw), math.sin(yaw)
    return (c * dx + s * dy, -s * dx + c * dy)


def _rollout_goal_progress(initial_state, final_state, goal_local_xy) -> float:
    if goal_local_xy is None:
        return math.hypot(final_state.x - initial_state.x, final_state.y - initial_state.y)
    gx, gy = goal_local_xy
    return (math.hypot(gx - initial_state.x, gy - initial_state.y)
            - math.hypot(gx - final_state.x, gy - final_state.y))


def compute_risk_telemetry(
    command: TrajectoryCommand, step_id: int,
    robot_pose, robot_v: float, robot_steering: float,
    static_obstacles: List[StaticObstacle], dynamic_specs: List[DynamicObstacleSpec],
    active_robot_config: RobotConfig, profile: Profile,
    goal_world_xy=None,
) -> rt.RiskTelemetry:
    """``robot_pose``/``robot_v``/``robot_steering``/``static_obstacles``/
    ``dynamic_specs`` MUST all be a PRE-ACTION snapshot of the same instant
    ``command`` was decided from -- see this module's docstring.

    ``robot_steering`` doubles as the commit-blend rollout's starting
    steering (code review: risk label must match the nominal execution
    model) when ``features.trajectory_l_preview_blend`` is on -- it is
    ALREADY required to be the real, PRE-ACTION measured value (never a
    safety-guard-adjusted one), so reusing it here for that purpose adds
    no new risk of a guard intervention leaking into the label: a command
    the guard would go on to clamp/stop is still assessed as risky, using
    the SAME current steering the guard itself started from."""
    if not (profile.features.ackermann_rollout and profile.risk.enabled
            and profile.action_space.mode == "trajectory"):
        return rt.invalid(step_id)

    rx, ry, ryaw = robot_pose
    initial_state = VehicleState(x=0.0, y=0.0, yaw=0.0, v=robot_v, steering=robot_steering)
    obstacles = build_privileged_obstacles(rx, ry, ryaw, static_obstacles, dynamic_specs)
    ego_radius = active_robot_config.collision_radius_m
    world_half_extent_m = profile.scenario.world_size_m / 2.0
    commit_blend = profile.features.trajectory_l_preview_blend
    goal_local_xy = _goal_local_xy(robot_pose, goal_world_xy)

    if profile.counterfactual.enabled:
        scored = score_candidates(
            command, initial_state, ego_radius, obstacles, active_robot_config,
            profile.dynamics, profile.risk, profile.counterfactual,
            robot_world_pose=robot_pose, world_half_extent_m=world_half_extent_m,
            commit_blend=commit_blend,
            goal_local_xy=goal_local_xy, action_cfg=profile.action_space,
        )
        actor_label = scored[0].risk
        return rt.RiskTelemetry(
            step_id=step_id, valid=True, risk_target=actor_label.risk_score,
            min_clearance_m=actor_label.min_clearance_m, ttc_sec=actor_label.time_to_collision_sec,
            collision_within_horizon=actor_label.collision_within_horizon,
            stopping_margin_m=actor_label.stopping_margin_m,
            steering_saturation=actor_label.steering_saturation,
            goal_progress_m=scored[0].goal_progress_m,
            unrecoverable=is_unrecoverable(scored),
            safer_alternative_margin=safer_alternative_margin(scored, profile.counterfactual),
            actor_candidate_index=0,
            candidates=[
                rt.CandidateTelemetry(kappa=c.command.kappa, v_ref=c.command.v_ref,
                                      horizon_m=c.command.horizon_m, risk_score=c.risk.risk_score,
                                      goal_progress_m=c.goal_progress_m)
                for c in scored
            ],
        )

    label, rollout = assess_trajectory_command_with_rollout(
        command.kappa, command.v_ref, command.horizon_m, initial_state, ego_radius,
        obstacles, active_robot_config, profile.dynamics, profile.risk,
        robot_world_pose=robot_pose, world_half_extent_m=world_half_extent_m,
        commit_blend=commit_blend,
    )
    return rt.RiskTelemetry(
        step_id=step_id, valid=True, risk_target=label.risk_score,
        min_clearance_m=label.min_clearance_m, ttc_sec=label.time_to_collision_sec,
        collision_within_horizon=label.collision_within_horizon,
        stopping_margin_m=label.stopping_margin_m,
        steering_saturation=label.steering_saturation,
        goal_progress_m=_rollout_goal_progress(initial_state, rollout.final_state, goal_local_xy),
        unrecoverable=False,
        safer_alternative_margin=0.0, actor_candidate_index=0, candidates=[],
    )


def compute_common_evaluation_metrics(
    published_command: VehicleCommand, step_id: int,
    robot_pose, robot_v: float, robot_steering: float,
    static_obstacles: List[StaticObstacle], dynamic_specs: List[DynamicObstacleSpec],
    active_robot_config: RobotConfig, profile: Profile,
    goal_world_xy=None,
) -> rt.RiskTelemetry:
    """section item-1 (fixed-benchmark fairness): computed for EVERY model
    evaluated against a fixed benchmark scenario, REGARDLESS of
    ``action_space.mode``/``features.risk_critic``/``profile.risk.enabled``
    -- unlike :func:`compute_risk_telemetry` above (which is gated by all
    three, and reads the ARCHITECTURE-owned ``profile.risk``/
    ``profile.dynamics``/``profile.counterfactual`` sections), this uses
    ONLY the EVALUATION-CONTRACT-owned ``profile.evaluation.common_metrics_*``
    fields, so two different checkpoints (a risk-aware TQC trained with
    ``clearance_horizon_sec=2.0`` and a SAC baseline trained with no risk
    framework at all) are scored against the exact SAME yardstick when
    benchmarked through the same evaluation profile.

    Built from the REALIZED (published) ``VehicleCommand`` -- speed +
    steering, whatever the robot is ACTUALLY about to execute -- converted
    to an equivalent ``[kappa, v_ref]`` pair via ``steering_to_curvature``,
    rather than the abstract ``[kappa, v_ref, L]`` action. This is what
    makes it work for EVERY ``action_space.mode`` including
    ``legacy_waypoint`` (whose ``[r, theta, yield]`` action has no rollout
    concept of its own at all) -- ``horizon_m`` is a fixed placeholder
    (irrelevant to the result, see below) rather than derived from the
    action.

    Counterfactual candidates are ALSO generated and scored around the
    realized command (using ``profile.evaluation.common_metrics_num_candidates``
    etc, never ``profile.counterfactual``, which is architecture/training-owned
    and can legitimately differ between profiles) so ``unrecoverable``/
    ``safer_alternative_margin`` are comparable across models too -- a
    model whose OWN architecture never computes counterfactuals (e.g. SAC,
    or a TQC checkpoint trained with ``counterfactual.enabled=False``)
    still gets a real, common-yardstick unrecoverable-state reading here.

    The rollout horizon is pinned to a FIXED, non-L-dependent window: the
    throwaway ``DynamicsConfig`` below sets ``horizon_min_sec ==
    horizon_max_sec == min_safety_horizon_sec == common_metrics_horizon_sec``,
    which collapses ``ackermann_rollout.l_derived_horizon_sec``'s clamp to
    exactly that constant regardless of the (irrelevant, placeholder)
    ``horizon_m``/``v_ref`` inputs -- see that function's own docstring."""
    eval_cfg = profile.evaluation
    rx, ry, ryaw = robot_pose
    initial_state = VehicleState(x=0.0, y=0.0, yaw=0.0, v=robot_v, steering=robot_steering)
    obstacles = build_privileged_obstacles(rx, ry, ryaw, static_obstacles, dynamic_specs)
    ego_radius = active_robot_config.collision_radius_m
    world_half_extent_m = profile.scenario.world_size_m / 2.0
    goal_local_xy = _goal_local_xy(robot_pose, goal_world_xy)

    dynamics_cfg = DynamicsConfig(
        horizon_sec=eval_cfg.common_metrics_horizon_sec,
        horizon_min_sec=eval_cfg.common_metrics_horizon_sec,
        horizon_max_sec=eval_cfg.common_metrics_horizon_sec,
        min_safety_horizon_sec=eval_cfg.common_metrics_horizon_sec,
        dt_sec=eval_cfg.common_metrics_dt_sec,
        model_actuator_lag=False,
    )
    risk_cfg = RiskConfig(
        enabled=True,
        clearance_horizon_sec=eval_cfg.common_metrics_clearance_horizon_sec,
        ttc_horizon_sec=eval_cfg.common_metrics_ttc_horizon_sec,
        min_safe_clearance_m=eval_cfg.common_metrics_min_safe_clearance_m,
    )
    cf_cfg = CounterfactualConfig(
        enabled=True,
        num_candidates=eval_cfg.common_metrics_num_candidates,
        kappa_offsets_frac=list(eval_cfg.common_metrics_kappa_offsets_frac),
        speed_fractions=list(eval_cfg.common_metrics_speed_fractions),
        include_stop_candidate=eval_cfg.common_metrics_include_stop_candidate,
    )
    kappa = steering_to_curvature(published_command.steering_rad, active_robot_config.wheelbase_m)
    base = TrajectoryCommand(kappa=kappa, v_ref=max(0.0, published_command.speed_mps), horizon_m=1.0)

    scored = score_candidates(
        base, initial_state, ego_radius, obstacles, active_robot_config,
        dynamics_cfg, risk_cfg, cf_cfg,
        robot_world_pose=robot_pose, world_half_extent_m=world_half_extent_m,
        commit_blend=False,
        goal_local_xy=goal_local_xy, action_cfg=profile.action_space,
    )
    actor_label = scored[0].risk  # index 0 is always the realized command itself (trajectory_sampler convention)
    return rt.RiskTelemetry(
        step_id=step_id, valid=True, risk_target=actor_label.risk_score,
        min_clearance_m=actor_label.min_clearance_m, ttc_sec=actor_label.time_to_collision_sec,
        collision_within_horizon=actor_label.collision_within_horizon,
        stopping_margin_m=actor_label.stopping_margin_m,
        steering_saturation=actor_label.steering_saturation,
        goal_progress_m=scored[0].goal_progress_m,
        unrecoverable=is_unrecoverable(scored),
        safer_alternative_margin=safer_alternative_margin(scored, cf_cfg),
        actor_candidate_index=0,
        candidates=[
            rt.CandidateTelemetry(kappa=c.command.kappa, v_ref=c.command.v_ref,
                                  horizon_m=c.command.horizon_m, risk_score=c.risk.risk_score,
                                  goal_progress_m=c.goal_progress_m)
            for c in scored
        ],
    )
