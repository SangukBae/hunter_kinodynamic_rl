"""Robot-agnostic interface -- swapping Hunter SE for another Ackermann UGV
(Scout, F1TENTH, ...) means writing one new class satisfying this Protocol,
touching nothing in dynamics/, trajectory/, risk/, or rl/ (section 61)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from hunter_kinodynamic_rl.config.schema import RobotConfig


@dataclass(frozen=True)
class VehicleState:
    """Minimal kinematic state used throughout dynamics/trajectory/risk."""

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    v: float = 0.0            # signed forward speed [m/s]
    steering: float = 0.0     # center steering angle [rad]


@runtime_checkable
class RobotModel(Protocol):
    """What every module downstream of the robot needs -- physical bounds
    plus curvature<->steering conversions. NOT a dynamics model (that's
    dynamics/bicycle_model.py, which takes a RobotModel as input)."""

    config: RobotConfig

    def clamp_speed(self, v_mps: float) -> float: ...
    def clamp_steering(self, steering_rad: float) -> float: ...
    def clamp_curvature(self, kappa: float) -> float: ...
    def curvature_to_steering(self, kappa: float) -> float: ...
    def steering_to_curvature(self, steering_rad: float) -> float: ...
