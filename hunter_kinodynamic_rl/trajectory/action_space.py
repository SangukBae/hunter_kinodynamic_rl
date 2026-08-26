"""Normalized [-1, 1] policy action <-> physical trajectory-command mapping.

Two contracts (selected by ``config.action_space.mode``), matching section 10
/ 11 of the research brief:

  trajectory (primary):
    action[0] -> kappa  in [-kappa_max, +kappa_max]   (kappa_max = robot.max_curvature)
    action[1] -> v_ref  in [0, v_max]                 (v_max = action_space.v_max_mps or robot.max_forward_speed_mps)
    action[2] -> L      in [horizon_length_min_m, horizon_length_max_m]

  legacy_waypoint (ablation A, same contract as drl_agent's hybrid action):
    action[0] -> r       in [legacy_r_min_m, legacy_r_max_m]
    action[1] -> theta   in [-legacy_theta_max_rad, +legacy_theta_max_rad]
    action[2] -> yield scalar in [-1, 1]

Both are pure functions: no ROS, no state, so they are covered directly by
unit tests and reused unchanged in nodes/, evaluation/, and real-robot deploy.
"""

from __future__ import annotations

from dataclasses import dataclass

from hunter_kinodynamic_rl.config.schema import ActionSpaceConfig, RobotConfig


ACTION_DIM = 3


@dataclass(frozen=True)
class TrajectoryCommand:
    kappa: float
    v_ref: float
    horizon_m: float


@dataclass(frozen=True)
class LegacyWaypointCommand:
    r: float
    theta: float
    yield_scalar: float


def _lerp(a_norm: float, lo: float, hi: float) -> float:
    a = max(-1.0, min(1.0, float(a_norm)))
    return 0.5 * (a + 1.0) * (hi - lo) + lo


def _inv_lerp(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    a = 2.0 * (float(value) - lo) / (hi - lo) - 1.0
    return max(-1.0, min(1.0, a))


def normalized_to_trajectory_command(action, action_cfg: ActionSpaceConfig,
                                      robot: RobotConfig) -> TrajectoryCommand:
    """action in [-1, 1]^3 -> physically bounded [kappa, v_ref, L]."""
    if len(action) != ACTION_DIM:
        raise ValueError(f"trajectory action must have length {ACTION_DIM}, got {len(action)}")
    kappa_max = robot.max_curvature * action_cfg.kappa_scale
    v_max = action_cfg.v_max_mps if action_cfg.v_max_mps is not None else robot.max_forward_speed_mps

    kappa = _lerp(action[0], -kappa_max, kappa_max)
    v_ref = _lerp(action[1], action_cfg.v_min_mps, v_max)
    horizon_m = _lerp(action[2], action_cfg.horizon_length_min_m, action_cfg.horizon_length_max_m)
    return TrajectoryCommand(kappa=kappa, v_ref=v_ref, horizon_m=horizon_m)


def normalized_to_legacy_waypoint_command(action, action_cfg: ActionSpaceConfig) -> LegacyWaypointCommand:
    if len(action) != ACTION_DIM:
        raise ValueError(f"legacy_waypoint action must have length {ACTION_DIM}, got {len(action)}")
    r = _lerp(action[0], action_cfg.legacy_r_min_m, action_cfg.legacy_r_max_m)
    theta = _lerp(action[1], -action_cfg.legacy_theta_max_rad, action_cfg.legacy_theta_max_rad)
    yield_scalar = max(-1.0, min(1.0, float(action[2])))
    return LegacyWaypointCommand(r=r, theta=theta, yield_scalar=yield_scalar)


def trajectory_command_to_normalized(cmd: TrajectoryCommand, action_cfg: ActionSpaceConfig,
                                      robot: RobotConfig):
    """Inverse of :func:`normalized_to_trajectory_command` -- physical
    [kappa, v_ref, L] -> normalized [-1, 1]^3. Used to turn a counterfactual
    CANDIDATE's physical command back into an action the risk critic (which
    consumes normalized actions, same as the actor's own output) can score
    (rl/algorithms/kinodynamic_tqc/agent.py's candidate-augmented supervision)."""
    kappa_max = robot.max_curvature * action_cfg.kappa_scale
    v_max = action_cfg.v_max_mps if action_cfg.v_max_mps is not None else robot.max_forward_speed_mps
    return (
        _inv_lerp(cmd.kappa, -kappa_max, kappa_max),
        _inv_lerp(cmd.v_ref, action_cfg.v_min_mps, v_max),
        _inv_lerp(cmd.horizon_m, action_cfg.horizon_length_min_m, action_cfg.horizon_length_max_m),
    )


def decode_action(action, action_cfg: ActionSpaceConfig, robot: RobotConfig):
    """Dispatch on ``action_cfg.mode`` -> the right command dataclass."""
    if action_cfg.mode == "trajectory":
        return normalized_to_trajectory_command(action, action_cfg, robot)
    if action_cfg.mode == "legacy_waypoint":
        return normalized_to_legacy_waypoint_command(action, action_cfg)
    raise ValueError(f"unknown action_space.mode {action_cfg.mode!r}")
