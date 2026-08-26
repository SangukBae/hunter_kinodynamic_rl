"""Kinematic bicycle model -- the physical core every trajectory/risk/rollout
module in this package sits on top of.

Convention (matches drl_agent's pure_pursuit.ackermann_rollout /
ackermann_swept_path, kept identical so the two systems' geometry agrees --
see docs/SOURCE_MAP.md): robot-local frame, x forward, y left, yaw CCW+.

    yaw_rate = v * tan(delta) / wheelbase
    kappa    = tan(delta) / wheelbase

This module only steps the KINEMATICS (instant v/steering -> next pose); how
v/steering approach a commanded target over time (accel limits, steering
rate, lag) is dynamics/actuator_model.py -- composed together in
dynamics/ackermann_rollout.py.
"""

from __future__ import annotations

import math

from hunter_kinodynamic_rl.robot.interface import VehicleState


def yaw_rate(v_mps: float, steering_rad: float, wheelbase_m: float) -> float:
    return v_mps * math.tan(steering_rad) / max(wheelbase_m, 1e-6)


def step(state: VehicleState, v_mps: float, steering_rad: float, dt_sec: float,
         wheelbase_m: float) -> VehicleState:
    """Exact bicycle-model integration over ``dt_sec`` at CONSTANT
    (v_mps, steering_rad) -- closed-form arc, not Euler, so a single call
    over a long dt is exact (matches pure_pursuit.ackermann_rollout's
    closed-form radius/sin/cos integration)."""
    v = float(v_mps)
    omega = yaw_rate(v, steering_rad, wheelbase_m)
    if abs(v) < 1e-9 or dt_sec <= 0.0:
        return state
    if abs(omega) < 1e-9:
        dx, dy = v * dt_sec, 0.0
        dyaw = 0.0
    else:
        radius = v / omega
        dyaw = omega * dt_sec
        dx = radius * math.sin(dyaw)
        dy = radius * (1.0 - math.cos(dyaw))
    cos_yaw, sin_yaw = math.cos(state.yaw), math.sin(state.yaw)
    world_dx = cos_yaw * dx - sin_yaw * dy
    world_dy = sin_yaw * dx + cos_yaw * dy
    return VehicleState(
        x=state.x + world_dx,
        y=state.y + world_dy,
        yaw=state.yaw + dyaw,
        v=v,
        steering=steering_rad,
    )


def step_midpoint(state: VehicleState, v_start: float, steering_start: float,
                   v_end: float, steering_end: float, dt_sec: float,
                   wheelbase_m: float) -> VehicleState:
    """Trapezoidal (midpoint-value) integration for a substep where v/steering
    are RAMPING linearly from (v_start, steering_start) to (v_end,
    steering_end) -- recovers the substep's true arc integral instead of a
    plain Euler under/over-shoot (see actuator_model.py's docstring for why
    this matters for braking/accelerating rollouts)."""
    v_mid = 0.5 * (v_start + v_end)
    steer_mid = 0.5 * (steering_start + steering_end)
    omega = yaw_rate(v_mid, steer_mid, wheelbase_m)
    if dt_sec <= 0.0:
        return state
    heading_mid = state.yaw + 0.5 * omega * dt_sec
    dx = v_mid * math.cos(heading_mid) * dt_sec
    dy = v_mid * math.sin(heading_mid) * dt_sec
    return VehicleState(
        x=state.x + dx,
        y=state.y + dy,
        yaw=state.yaw + omega * dt_sec,
        v=v_end,
        steering=steering_end,
    )
