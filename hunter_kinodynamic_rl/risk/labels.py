"""Shared types for the risk framework: obstacles (as seen by PRIVILEGED
simulator ground truth -- section 18) and the composite risk label a
candidate trajectory ends up with.

IMPORTANT (section 18 / section 41): ``DynamicObstacle`` is a
training-time-only, privileged-information type. At real-robot inference
time, obstacle position/velocity is never known this precisely -- only
whatever the perception stack (LiDAR history) infers. Nothing in
env/simulation may hand a ``DynamicObstacle`` list to the POLICY forward
pass; it is only used to compute SUPERVISED risk targets and the
Gazebo-side counterfactual ranking used for training-time reward/label
shaping. real_hunter_safe.yaml's deployment path never constructs one.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DynamicObstacle:
    """Constant-velocity obstacle model, ROBOT-LOCAL frame at rollout t=0."""

    x0: float
    y0: float
    vx: float = 0.0
    vy: float = 0.0
    radius: float = 0.3

    def position_at(self, t_sec: float):
        return (self.x0 + self.vx * t_sec, self.y0 + self.vy * t_sec)


@dataclass(frozen=True)
class RiskLabel:
    """Composite, per-candidate-trajectory risk assessment."""

    min_clearance_m: float
    time_to_collision_sec: float
    collision_within_horizon: bool
    stopping_margin_m: float
    steering_saturation: bool
    risk_score: float  # normalized [0, 1], 1 = worst
