"""Pure clamp/conversion helpers derived from a RobotConfig -- no state, no
ROS. Kept separate from hunter_se.py so a robot swap can reuse these exact
kappa<->steering relations (they follow directly from the bicycle model, not
from anything Hunter-SE-specific)."""

from __future__ import annotations

import math

from hunter_kinodynamic_rl.config.schema import RobotConfig


def curvature_to_steering(kappa: float, wheelbase_m: float) -> float:
    """delta = atan(wheelbase * kappa) -- section 9 of the research brief."""
    return math.atan(wheelbase_m * kappa)


def steering_to_curvature(steering_rad: float, wheelbase_m: float) -> float:
    """kappa = tan(delta) / wheelbase."""
    if wheelbase_m <= 0.0:
        return 0.0
    return math.tan(steering_rad) / wheelbase_m


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class RobotLimits:
    """Bound-checking / clamping view over a :class:`RobotConfig`."""

    def __init__(self, config: RobotConfig):
        config.validate()
        self.config = config

    def clamp_speed(self, v_mps: float) -> float:
        return clamp(float(v_mps), self.config.min_forward_speed_mps, self.config.max_forward_speed_mps)

    def clamp_steering(self, steering_rad: float) -> float:
        limit = self.config.steering_limit_rad
        return clamp(float(steering_rad), -limit, limit)

    def clamp_curvature(self, kappa: float) -> float:
        limit = self.config.max_curvature
        return clamp(float(kappa), -limit, limit)

    def curvature_to_steering(self, kappa: float) -> float:
        return curvature_to_steering(self.clamp_curvature(kappa), self.config.wheelbase_m)

    def steering_to_curvature(self, steering_rad: float) -> float:
        return steering_to_curvature(self.clamp_steering(steering_rad), self.config.wheelbase_m)
