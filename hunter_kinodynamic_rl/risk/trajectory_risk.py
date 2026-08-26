"""Aggregate a candidate trajectory's individual risk metrics (clearance,
TTC, stopping margin, collision, steering saturation) into one
:class:`~hunter_kinodynamic_rl.risk.labels.RiskLabel`.

The individual metrics are kept as SEPARATE, independently-inspectable
fields (section 17: "risk 정의를 단일 파일에 뒤섞지 말고 개별 metric과
aggregate function을 분리한다") -- this module only owns the combination
into one scalar ``risk_score``.

Combination rule: risk_score = max of the per-metric normalized risks (not a
weighted sum). A trajectory that is fine on 3 axes but about to collide is
still maximally risky -- averaging would dilute that, so the WORST axis
dominates (a conservative, safety-appropriate choice, matching how
Unrecoverable-State reasoning below treats "any candidate collides" as
disqualifying rather than partially-scoring it).
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

from hunter_kinodynamic_rl.config.schema import DynamicsConfig, RiskConfig, RobotConfig
from hunter_kinodynamic_rl.dynamics import ackermann_rollout
from hunter_kinodynamic_rl.dynamics.ackermann_rollout import Rollout
from hunter_kinodynamic_rl.risk import boundary as boundary_mod
from hunter_kinodynamic_rl.risk import future_clearance, ttc as ttc_mod
from hunter_kinodynamic_rl.risk.boundary import RobotPose
from hunter_kinodynamic_rl.risk.labels import DynamicObstacle, RiskLabel
from hunter_kinodynamic_rl.risk.stopping_margin import stopping_margin_m
from hunter_kinodynamic_rl.robot.interface import VehicleState


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def assess_trajectory(
    rollout: Rollout, ego_radius: float, obstacles: Sequence[DynamicObstacle],
    steering_rad: float, robot: RobotConfig, risk_cfg: RiskConfig,
    robot_world_pose: Optional[RobotPose] = None, world_half_extent_m: Optional[float] = None,
    initial_state: Optional[VehicleState] = None,
) -> RiskLabel:
    """``robot_world_pose``/``world_half_extent_m`` are OPTIONAL: when both
    are provided, the world boundary (section P0-2) is folded into
    ``clearance``/``collision_time``/``collided`` via elementwise min/OR
    against the obstacle-only values BEFORE any of the downstream risk-score
    math runs -- so stopping-margin, clearance-risk, TTC-risk and
    collision-risk all automatically account for the wall too, with no
    separate boundary-risk terms needed. Omitting them (the default) skips
    boundary consideration entirely -- e.g. real-hardware inference, which
    has no bounded procedural world to check against.

    ``initial_state`` is OPTIONAL (default ``None``, preserving the exact
    prior rollout-only behavior for callers that don't have a convenient
    :class:`VehicleState` on hand -- e.g. some standalone unit tests):
    when given, it is the ego state the rollout was generated FROM,
    ``t_sec=0`` -- see ``future_clearance.py``/``ttc.py``'s module
    docstrings for why checking it is necessary (rollout points never
    include ``t=0``). Every PRODUCTION call site
    (``assess_trajectory_command`` below) passes it."""
    # section item-6: CONTINUOUS (segment-based) checks, not just the
    # discrete sampled-endpoint ones -- a genuine collision/wall-crossing
    # occurring strictly BETWEEN two consecutive rollout samples (both
    # individually clear) would otherwise be invisible to the actual
    # production risk score. See future_clearance.py/ttc.py/boundary.py's
    # own "_continuous" function docstrings for the exact affine-segment
    # math; strictly <= the discrete result, so this can only ever find an
    # EQUAL-or-MORE-conservative risk assessment, never a less conservative
    # one.
    clearance = future_clearance.min_clearance_continuous(
        rollout, ego_radius, obstacles, horizon_sec=risk_cfg.clearance_horizon_sec, t0_state=initial_state,
    )
    collided = ttc_mod.collision_within_horizon_continuous(
        rollout, ego_radius, obstacles, risk_cfg.ttc_horizon_sec, t0_state=initial_state,
    )
    collision_time = ttc_mod.time_to_collision_continuous(
        rollout, ego_radius, obstacles, risk_cfg.ttc_horizon_sec, t0_state=initial_state,
    )

    if robot_world_pose is not None and world_half_extent_m is not None:
        boundary_clearance = boundary_mod.min_boundary_clearance_continuous(
            rollout, ego_radius, robot_world_pose, world_half_extent_m,
            horizon_sec=risk_cfg.clearance_horizon_sec,
        )
        boundary_collided = boundary_mod.boundary_time_to_exit_or_none_continuous(
            rollout, ego_radius, robot_world_pose, world_half_extent_m, risk_cfg.ttc_horizon_sec,
        ) is not None
        boundary_ttc = boundary_mod.boundary_time_to_exit_continuous(
            rollout, ego_radius, robot_world_pose, world_half_extent_m, risk_cfg.ttc_horizon_sec)
        clearance = min(clearance, boundary_clearance)
        collision_time = min(collision_time, boundary_ttc)
        collided = collided or boundary_collided

    final_v = rollout.final_state.v if rollout.points else 0.0
    nearest_ahead = clearance if clearance != float("inf") else float("inf")
    margin = stopping_margin_m(final_v, nearest_ahead, robot) if nearest_ahead != float("inf") else float("inf")

    saturated = abs(steering_rad) >= robot.steering_limit_rad - 1e-6

    clearance_risk = 0.0 if clearance == float("inf") else _clip01(
        1.0 - clearance / max(risk_cfg.min_safe_clearance_m, 1e-6)
    )
    ttc_risk = _clip01(1.0 - collision_time / max(risk_cfg.ttc_horizon_sec, 1e-6))
    margin_risk = 0.0 if margin == float("inf") else _clip01(
        -margin / max(risk_cfg.stopping_margin_safety_factor, 1e-6)
    )
    collision_risk = 1.0 if collided else 0.0

    risk_score = max(clearance_risk, ttc_risk, margin_risk, collision_risk)

    return RiskLabel(
        min_clearance_m=clearance,
        time_to_collision_sec=collision_time,
        collision_within_horizon=collided,
        stopping_margin_m=margin,
        steering_saturation=saturated,
        risk_score=risk_score,
    )


def assess_trajectory_command(
    kappa: float, v_ref: float, horizon_m: float, initial_state: VehicleState,
    ego_radius: float, obstacles: Sequence[DynamicObstacle],
    robot: RobotConfig, dynamics_cfg: DynamicsConfig, risk_cfg: RiskConfig,
    robot_world_pose: Optional[RobotPose] = None, world_half_extent_m: Optional[float] = None,
    commit_blend: bool = False,
) -> RiskLabel:
    """The L-CONSISTENT entry point: rolls out ``[kappa, v_ref, horizon_m]``
    using :func:`dynamics.ackermann_rollout.rollout_trajectory_command` (L
    -derived rollout horizon) and assesses it with an EFFECTIVE risk config
    whose ``ttc_horizon_sec`` is clamped to that same derived horizon -- so a
    short-L candidate is never scored against a longer TTC window than it
    was actually rolled out over (which would silently report "no collision"
    for a horizon the rollout never covered).

    ``commit_blend`` (code review: "risk label과 실제 실행 command가
    불일치" -- once ``trajectory/pure_pursuit_adapter.py``'s L-derived
    steering commit window shipped, the ACTUALLY-published steering only
    partially commits toward the target curvature each tick, while this
    rollout still assumed it was reached instantly): when True, forwarded
    to :func:`dynamics.ackermann_rollout.rollout_trajectory_command`'s own
    ``commit_blend`` flag, which rolls out using the SAME commit-window
    formula the executor uses, starting from ``initial_state.steering`` --
    so the risk label is computed from the ACTUAL nominal execution model,
    not an idealized instant-tracking one. Callers MUST pass
    ``initial_state.steering`` as the REAL, PRE-ACTION measured current
    steering (never a safety-guard-adjusted value -- see
    ``env/simulation/risk_computation.py``'s docstring: a guard
    intervention must never retroactively make a genuinely risky nominal
    action's label look safe). Default False preserves the exact legacy
    instant-tracking rollout (byte-identical) for any caller that hasn't
    opted in, or for the (real-hardware / no-executor-blend) case where
    the executor itself never applies this blend either."""
    label, _rollout = assess_trajectory_command_with_rollout(
        kappa, v_ref, horizon_m, initial_state, ego_radius, obstacles,
        robot, dynamics_cfg, risk_cfg, robot_world_pose=robot_world_pose,
        world_half_extent_m=world_half_extent_m, commit_blend=commit_blend,
    )
    return label


def assess_trajectory_command_with_rollout(
    kappa: float, v_ref: float, horizon_m: float, initial_state: VehicleState,
    ego_radius: float, obstacles: Sequence[DynamicObstacle],
    robot: RobotConfig, dynamics_cfg: DynamicsConfig, risk_cfg: RiskConfig,
    robot_world_pose: Optional[RobotPose] = None, world_half_extent_m: Optional[float] = None,
    commit_blend: bool = False,
) -> Tuple[RiskLabel, Rollout]:
    """Return the risk label and the exact rollout used to compute it.

    Counterfactual selection needs the rollout endpoint to measure goal
    progress.  Returning it here avoids running the same vehicle dynamics a
    second time solely for that progress calculation.
    """
    horizon_sec = ackermann_rollout.horizon_from_trajectory(horizon_m, v_ref, dynamics_cfg)
    rollout = ackermann_rollout.rollout_trajectory_command(
        initial_state, kappa, v_ref, horizon_m, robot, dynamics_cfg, commit_blend=commit_blend,
    )
    from hunter_kinodynamic_rl.robot.limits import curvature_to_steering
    steering = curvature_to_steering(kappa, robot.wheelbase_m)
    effective_risk_cfg = RiskConfig(
        enabled=risk_cfg.enabled,
        clearance_horizon_sec=min(risk_cfg.clearance_horizon_sec, horizon_sec),
        ttc_horizon_sec=min(risk_cfg.ttc_horizon_sec, horizon_sec),
        min_safe_clearance_m=risk_cfg.min_safe_clearance_m,
        stopping_margin_safety_factor=risk_cfg.stopping_margin_safety_factor,
        actor_lambda=risk_cfg.actor_lambda,
    )
    label = assess_trajectory(
        rollout, ego_radius, obstacles, steering, robot, effective_risk_cfg,
        robot_world_pose=robot_world_pose, world_half_extent_m=world_half_extent_m,
        initial_state=initial_state,
    )
    return label, rollout
