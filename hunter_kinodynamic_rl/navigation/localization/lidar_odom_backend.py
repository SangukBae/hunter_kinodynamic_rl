"""LiDAR-odometry :class:`LocalizationBackend` (plan section 4/10.5 Phase E:
"LiDAR odometry 또는 LIO"). Pure Python (ROS-free) -- performs pose
COMPOSITION and confidence mapping only; the actual scan-matching/ICP/NDT
algorithm that produces a relative transform between consecutive scans is
NOT implemented here (out of scope, and this repository has no such
algorithm today) -- this class is the integration point a future
scan-matcher calls via :meth:`integrate_relative_transform`, exactly
mirroring the plan's "Localization backend는 교체 가능한 interface" goal:
navigation code only ever depends on the
:class:`~hunter_kinodynamic_rl.navigation.localization.interface.LocalizationBackend`
Protocol, never on how a specific backend computed its pose.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from hunter_kinodynamic_rl.common.geometry import wrap_to_pi
from hunter_kinodynamic_rl.navigation.localization.interface import PoseEstimate
from hunter_kinodynamic_rl.navigation.localization.odom_backend import DEFAULT_HISTORY_SIZE, OdomLocalizationBackend


@dataclass(frozen=True)
class LidarOdomNoiseModel:
    #: Position-covariance-trace when match_quality == 0.0 (totally
    #: unreliable match). Scales down linearly to 0 as quality -> 1.0.
    max_position_variance_m2: float = 0.5
    max_heading_variance_rad2: float = 0.05
    confidence_floor: float = 0.0

    def validate(self) -> None:
        if self.max_position_variance_m2 < 0.0:
            raise ValueError("LidarOdomNoiseModel.max_position_variance_m2 must be >= 0")
        if self.max_heading_variance_rad2 < 0.0:
            raise ValueError("LidarOdomNoiseModel.max_heading_variance_rad2 must be >= 0")
        if not (0.0 <= self.confidence_floor <= 1.0):
            raise ValueError("LidarOdomNoiseModel.confidence_floor must be in [0, 1]")


class LidarOdomLocalizationBackend(OdomLocalizationBackend):
    def __init__(self, noise_model: LidarOdomNoiseModel = LidarOdomNoiseModel(), history_size: int = DEFAULT_HISTORY_SIZE) -> None:
        noise_model.validate()
        super().__init__(history_size)
        self._noise_model = noise_model
        self._x = 0.0
        self._y = 0.0
        self._yaw = 0.0
        self._initialized = False

    def initialize(self, x: float, y: float, yaw: float, stamp_sec: float) -> PoseEstimate:
        self._x, self._y, self._yaw = float(x), float(y), float(yaw)
        self._initialized = True
        return self.update(
            self._x, self._y, self._yaw, stamp_sec, covariance=np.zeros((3, 3)), confidence=1.0, valid=True,
        )

    def integrate_relative_transform(
        self, dx_robot: float, dy_robot: float, dyaw: float, stamp_sec: float, match_quality: float,
    ) -> PoseEstimate:
        """``dx_robot``/``dy_robot``/``dyaw`` are the scan-matcher's own
        relative-transform output, expressed in the PREVIOUS pose's robot
        frame (the standard scan-matcher convention: "how far did the robot
        move, in its own frame, between these two scans"). ``match_quality``
        in ``[0, 1]`` (1.0 == confident match, 0.0 == degenerate/rejected
        match, e.g. a feature-poor long corridor) drives BOTH the reported
        confidence and the covariance -- a non-finite input is DROPPED
        (returns the unchanged current pose), never silently composed into
        the running estimate."""
        if not self._initialized:
            raise RuntimeError("LidarOdomLocalizationBackend.integrate_relative_transform() called before initialize()")
        if not all(math.isfinite(v) for v in (dx_robot, dy_robot, dyaw, match_quality)):
            return self.latest_pose()

        quality = float(np.clip(match_quality, 0.0, 1.0))
        cos_y, sin_y = math.cos(self._yaw), math.sin(self._yaw)
        self._x += cos_y * dx_robot - sin_y * dy_robot
        self._y += sin_y * dx_robot + cos_y * dy_robot
        self._yaw = wrap_to_pi(self._yaw + dyaw)

        nm = self._noise_model
        position_variance = (1.0 - quality) * nm.max_position_variance_m2
        heading_variance = (1.0 - quality) * nm.max_heading_variance_rad2
        cov = np.diag([position_variance, position_variance, heading_variance])
        confidence = max(nm.confidence_floor, quality)
        return self.update(
            self._x, self._y, self._yaw, stamp_sec, covariance=cov, confidence=confidence, valid=quality > 0.0,
        )
