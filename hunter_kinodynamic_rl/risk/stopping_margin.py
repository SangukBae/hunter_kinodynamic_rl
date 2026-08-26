"""Stopping margin: how much clearance is left over the physically required
braking distance -- the key "can Hunter SE actually still stop in time?"
quantity, distinct from a pure geometric-clearance check (section 17)."""

from __future__ import annotations

from hunter_kinodynamic_rl.config.schema import RobotConfig
from hunter_kinodynamic_rl.dynamics.stopping_model import stopping_distance_from_config


def stopping_margin_m(
    current_speed_mps: float, distance_to_obstacle_m: float, robot: RobotConfig,
    reaction_time_sec: float = 0.0,
) -> float:
    """``distance_to_obstacle - required_stopping_distance``. Positive means
    Hunter SE can still stop before reaching the obstacle at its CURRENT
    speed; negative means braking now is already not enough."""
    required = stopping_distance_from_config(current_speed_mps, robot, reaction_time_sec)
    return float(distance_to_obstacle_m) - required
