"""First-order actuator response: how commanded (v, steering) targets are
actually approached over time -- rate limits + optional speed lag.

Adapted from (not a verbatim copy of) drl_agent's
``common/pure_pursuit.ackermann_swept_path`` rate-limiting core: the
move-toward-target-by-max-delta step and accel/brake-by-sign selection are
the same idea, restructured here to operate on a full incremental control
SEQUENCE (not just one start/end target) so ackermann_rollout.py can drive it
with an arbitrary per-substep target profile (needed once a trajectory
primitive commands a non-constant speed near the horizon start). See
docs/IMPLEMENTATION_PLAN.md for the exact relationship to the original.
"""

from __future__ import annotations

from dataclasses import dataclass

from hunter_kinodynamic_rl.config.schema import RobotConfig


def move_towards(current: float, target: float, max_delta: float) -> float:
    delta = target - current
    if abs(delta) <= max_delta:
        return target
    return current + max_delta * (1.0 if delta > 0 else -1.0)


@dataclass
class ActuatorState:
    v: float = 0.0
    steering: float = 0.0
    # First-order lag internal state (only used when speed_lag_tau_sec > 0).
    v_lagged: float = 0.0


def step_actuator(state: ActuatorState, target_v: float, target_steering: float,
                   dt_sec: float, robot: RobotConfig) -> ActuatorState:
    """One control-tick actuator response: rate-limited steering, rate-limited
    (+ optionally lagged) speed. Mirrors hunter_se_cmd_prefilter's own
    shaping so a rollout using this predicts what the real prefilter will do,
    not an idealised instant-response robot."""
    rate = robot.accel_limit_mps2 if abs(target_v) >= abs(state.v) else robot.brake_decel_mps2
    v_rate_limited = move_towards(state.v, float(target_v), rate * dt_sec)
    steering = move_towards(state.steering, float(target_steering), robot.steering_rate_rad_s * dt_sec)

    if robot.speed_lag_tau_sec > 0.0:
        tau = robot.speed_lag_tau_sec
        alpha = 1.0 - pow(2.718281828459045, -dt_sec / tau)
        v_lagged = state.v_lagged + alpha * (v_rate_limited - state.v_lagged)
    else:
        v_lagged = v_rate_limited

    return ActuatorState(v=v_rate_limited, steering=steering, v_lagged=v_lagged)


def effective_speed(state: ActuatorState, robot: RobotConfig) -> float:
    """The speed value a downstream bicycle-model step should actually use:
    the lagged value when lag is modeled, else the rate-limited value."""
    return state.v_lagged if robot.speed_lag_tau_sec > 0.0 else state.v
