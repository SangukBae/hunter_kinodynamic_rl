"""Single entrypoint a control loop calls each tick: normalized policy
action -> physical VehicleCommand, dispatching on ``action_space.mode`` so
callers (env node, real-robot inference node) never branch on the mode
themselves."""

from __future__ import annotations

from typing import Optional

from hunter_kinodynamic_rl.config.schema import ActionSpaceConfig, DynamicsConfig, RobotConfig, TrajectoryConfig
from hunter_kinodynamic_rl.robot.limits import RobotLimits
from hunter_kinodynamic_rl.trajectory import action_space, pure_pursuit, pure_pursuit_adapter
from hunter_kinodynamic_rl.trajectory.pure_pursuit_adapter import VehicleCommand


def execute(
    action, action_cfg: ActionSpaceConfig, trajectory_cfg: TrajectoryConfig, robot: RobotConfig,
    dynamics_cfg: Optional[DynamicsConfig] = None, current_steering_rad: Optional[float] = None,
) -> VehicleCommand:
    """``dynamics_cfg``/``current_steering_rad`` are forwarded to
    :func:`pure_pursuit_adapter.trajectory_command_to_vehicle_command`'s
    L-derived steering commit window (see its module docstring) -- both
    optional and both ignored outside ``action_cfg.mode == "trajectory"``;
    omitting either preserves the exact legacy instantaneous-steering
    behavior."""
    command = action_space.decode_action(action, action_cfg, robot)

    if isinstance(command, action_space.TrajectoryCommand):
        return pure_pursuit_adapter.trajectory_command_to_vehicle_command(
            command.kappa, command.v_ref, command.horizon_m, robot, trajectory_cfg,
            dynamics_cfg=dynamics_cfg, current_steering_rad=current_steering_rad,
        )

    if isinstance(command, action_space.LegacyWaypointCommand):
        # section P0-4 (baseline/reference parity): calls the SAME
        # hybrid_action_to_command drl_agent itself uses (verbatim-copied
        # into pure_pursuit.py -- see docs/IMPLEMENTATION_PLAN.md), with drl_agent's
        # own controller constants (action_cfg.legacy_*), instead of the
        # SUPERSEDED waypoint_to_command() low_speed_distance_m ramp this
        # branch previously hand-rolled (a real, confirmed parity gap: wrong
        # yield threshold, no MOVE-mode lookahead floor, wrong speed_steer_factor,
        # and a "floor" instead of a "cap" for the yield-mode speed limit).
        # The RAW normalized action is passed directly -- this function does
        # its own denormalization, matching drl_agent's own call convention
        # exactly (never operates on the already-decoded LegacyWaypointCommand,
        # which exists purely for the isinstance dispatch/telemetry logging
        # above and carries no post-floor information).
        limits = RobotLimits(robot)
        speed, steering, _theta, _info = pure_pursuit.hybrid_action_to_command(
            action,
            actions_low=[action_cfg.legacy_r_min_m, -action_cfg.legacy_theta_max_rad, -1.0],
            actions_high=[action_cfg.legacy_r_max_m, action_cfg.legacy_theta_max_rad, 1.0],
            wheelbase_m=robot.wheelbase_m,
            steering_limit_rad=robot.steering_limit_rad,
            cruise_speed_mps=robot.max_forward_speed_mps,
            speed_steer_factor=action_cfg.legacy_speed_steer_factor,
            yield_enabled=action_cfg.legacy_yield_enabled,
            yield_threshold=action_cfg.legacy_yield_threshold,
            lookahead_min_m=action_cfg.legacy_lookahead_min_m,
            v_move_min_mps=action_cfg.legacy_v_move_min_mps,
            yield_creep_speed_mps=action_cfg.legacy_yield_creep_mps,
        )
        return VehicleCommand(speed_mps=limits.clamp_speed(speed), steering_rad=limits.clamp_steering(steering))

    if isinstance(command, action_space.DirectControlCommand):
        limits = RobotLimits(robot)
        return VehicleCommand(
            speed_mps=limits.clamp_speed(command.speed_mps),
            steering_rad=limits.clamp_steering(command.steering_rad),
        )

    raise TypeError(f"unhandled decoded action type: {type(command)!r}")
