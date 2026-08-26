"""Stopping-distance / stopping-time physics -- the building block for
risk/stopping_margin.py and the Unrecoverable-State check.

Kept separate from actuator_model.py's general rate-limited rollout because
these closed-form expressions are exact for constant-deceleration braking and
much cheaper than a full substep rollout when only the scalar distance/time
is needed (e.g. inside the counterfactual sampler's per-candidate scoring
loop).
"""

from __future__ import annotations

from hunter_kinodynamic_rl.config.schema import RobotConfig


def stopping_distance_m(v_mps: float, brake_decel_mps2: float) -> float:
    """v^2 / (2*decel) -- constant-deceleration braking distance. Always
    >= 0; a negative/zero speed stops instantly."""
    v = abs(float(v_mps))
    decel = max(float(brake_decel_mps2), 1e-6)
    return (v * v) / (2.0 * decel)


def stopping_time_sec(v_mps: float, brake_decel_mps2: float) -> float:
    v = abs(float(v_mps))
    decel = max(float(brake_decel_mps2), 1e-6)
    return v / decel


def stopping_distance_with_reaction(v_mps: float, brake_decel_mps2: float,
                                     reaction_time_sec: float = 0.0) -> float:
    """Adds a constant-speed "reaction"/latency phase before braking begins
    -- the more realistic quantity for a policy that must commit to a stop
    decision one control tick (or one command-latency window) early."""
    v = abs(float(v_mps))
    return v * max(0.0, reaction_time_sec) + stopping_distance_m(v, brake_decel_mps2)


def stopping_distance_from_config(v_mps: float, robot: RobotConfig,
                                   reaction_time_sec: float = 0.0) -> float:
    return stopping_distance_with_reaction(v_mps, robot.brake_decel_mps2, reaction_time_sec)
