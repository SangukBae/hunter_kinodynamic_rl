"""Adapter: a [kappa, v_ref, L] trajectory command -> the existing Pure
Pursuit controller (trajectory/pure_pursuit.py, copied verbatim from
drl_agent -- see docs/IMPLEMENTATION_PLAN.md).

The STEERING command is computed by sampling a lookahead point on the
primitive's arc and running it through the shared
``pure_pursuit.waypoint_to_command`` chord geometry (section 12: "Pure
Pursuit 연결") so both this package and drl_agent track a commanded curvature
the same way. For a CONSTANT-CURVATURE primitive this recovered steering
angle is mathematically invariant to which point along the arc is sampled
as "lookahead" -- an accepted, documented property (a real geometric fact
about circular Pure Pursuit tracking, not a bug).

``v_ref`` is the policy's TARGET SPEED and is executed directly (only
clamped to the robot's own physical speed bounds via
``RobotLimits.clamp_speed`` -- never reduced as a function of ``L``). An
earlier version of this adapter additionally capped the executed speed by
``L / dynamics.horizon_min_sec`` specifically to keep L "non-dead" at the
control level -- removed: an L-derived cap on the actually-published speed
makes ``v_ref`` NOT the target speed (the command silently disagrees with
what the policy asked for), which is exactly the "arbitrary L-based speed
cap" this research direction now explicitly forbids. L remains very much
alive without it: it is the trajectory primitive's own arc-length extent
(``trajectory_primitive.ConstantCurvatureArc.horizon_m``), it directly
determines the risk/rollout TIME horizon
(``dynamics/ackermann_rollout.py::horizon_from_trajectory``, ``L/v_ref``
clipped to ``[horizon_min_sec, horizon_max_sec]`` with an L-independent
safety floor), and through that rollout horizon it determines stopping-
feasibility risk outputs (``stopping_margin_m`` etc., see
``risk/trajectory_risk.py``) -- see ``tests/test_l_semantics.py`` for the
regression coverage proving L changes the rollout endpoint and risk
outcome. None of that requires touching the published speed command.

STEERING COMMIT WINDOW (code review: L was "barely reflected" in the
actually-published command -- for an exact constant-curvature arc, a Pure
Pursuit lookahead-point recovery of curvature is PROVABLY invariant to which
point along the arc is sampled, so the ``lookahead_fraction`` machinery alone
never made the published steering depend on L at all; see
``tests/test_l_semantics.py``/``tests/test_trajectory.py`` for the geometric
proof-by-test). When the caller supplies the robot's ACTUAL current steering
angle and the run's :class:`~hunter_kinodynamic_rl.config.schema.DynamicsConfig`
(``dynamics_cfg``/``current_steering_rad``, both optional -- omitting either
preserves the exact legacy geometric-recovery behavior byte-for-byte), the
target steering angle is not applied in one instantaneous jump. Instead it is
blended in from the robot's CURRENT steering over the L-derived commit window
(``dynamics.ackermann_rollout.l_derived_horizon_sec`` -- deliberately the
UNPADDED L/v_ref quantity, not ``horizon_from_trajectory``'s safety-floored
one: the floor exists to stop risk-hiding, not to shape control response, so
reusing it here would mute L's effect at short horizons), further bounded by
the robot's real ``steering_rate_deg_s`` actuator limit so the blend can never
command a physically unrealizable angular rate. A short L means "I intend to
complete this maneuver quickly" -> a large fraction of the target steering is
applied this tick (closer to the legacy instant-jump). A long L means "I'm
committing to a gradual, longer-horizon curve" -> only a small fraction is
applied this tick, with the rest phased in over subsequent ticks as the policy
keeps re-issuing similar actions -- the SAME L that widens the risk-rollout
window also visibly softens the steering response, so the two are consistent
by construction. See ``tests/test_l_semantics.py`` for the regression proving
two candidates that share (kappa, v_ref) but differ in L publish different
steering commands under this path.

The blend formula itself, :func:`dynamics.ackermann_rollout.l_commit_blended_steering`,
lives in ``dynamics/`` (not here) so ``risk/trajectory_risk.py``'s
``assess_trajectory_command(..., commit_blend=True)`` can reuse the EXACT
SAME formula for the risk rollout (code review: risk labels previously
assumed the target curvature was reached instantly, disagreeing with what
this module actually publishes once L-commit blending is engaged) without
``dynamics`` importing FROM ``trajectory`` (circular).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from hunter_kinodynamic_rl.config.schema import DynamicsConfig, RobotConfig, TrajectoryConfig
from hunter_kinodynamic_rl.dynamics.ackermann_rollout import l_commit_blended_steering
from hunter_kinodynamic_rl.robot.limits import RobotLimits
from hunter_kinodynamic_rl.trajectory import pure_pursuit
from hunter_kinodynamic_rl.trajectory.trajectory_primitive import make_primitive


@dataclass(frozen=True)
class VehicleCommand:
    speed_mps: float
    steering_rad: float


def trajectory_command_to_vehicle_command(
    kappa: float, v_ref: float, horizon_m: float,
    robot: RobotConfig, trajectory_cfg: TrajectoryConfig,
    lookahead_fraction: float = 0.5,
    dynamics_cfg: Optional[DynamicsConfig] = None,
    current_steering_rad: Optional[float] = None,
) -> VehicleCommand:
    limits = RobotLimits(robot)
    primitive = make_primitive(trajectory_cfg.primitive, kappa, v_ref, horizon_m)
    lookahead_s = max(1e-3, lookahead_fraction) * horizon_m
    wp = primitive.point_at(lookahead_s)

    target_steering = pure_pursuit.waypoint_to_command(
        wp.x, wp.y,
        wheelbase_m=robot.wheelbase_m,
        steering_limit_rad=robot.steering_limit_rad,
        cruise_speed_mps=v_ref,  # only used for waypoint_to_command's internal speed calc (discarded below)
        min_speed_mps=0.0,
        speed_steer_factor=0.0,
    )[1]

    if dynamics_cfg is not None and current_steering_rad is not None:
        steering = l_commit_blended_steering(
            target_steering, float(current_steering_rad), horizon_m, v_ref,
            robot, dynamics_cfg, trajectory_cfg.dt_sec,
        )
    else:
        steering = target_steering

    return VehicleCommand(
        speed_mps=limits.clamp_speed(v_ref),
        steering_rad=limits.clamp_steering(steering),
    )
