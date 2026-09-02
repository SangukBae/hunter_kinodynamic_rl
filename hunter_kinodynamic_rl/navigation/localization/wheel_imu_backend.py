"""Wheel odometry + IMU dead-reckoning :class:`LocalizationBackend` (plan
section 4/10.5 Phase D: "실제 Hunter wheel odom+IMU"). Pure Python
(ROS-free) so it is unit-testable against synthetic twist sequences exactly
like :mod:`odom_backend`.

HONEST LIMITATION: this is deliberately SIMPLE Euler dead-reckoning with a
growing-uncertainty covariance model, NOT a real EKF/UKF sensor-fusion
filter -- a real filter (bias estimation, IMU/wheel cross-correction) is
future work. What this backend DOES provide, correctly, is the
:class:`~hunter_kinodynamic_rl.navigation.localization.interface.LocalizationBackend`
Protocol contract (``latest_pose``/``pose_at``) with a CONFIDENCE that
actually decays as accumulated drift grows -- enough to exercise the
localization-drift-sensitivity evaluation (plan 10.5) and the
"localization backend를 교체해도 navigation core를 수정하지 않는다"
architecture claim without requiring a live wheel encoder/IMU driver in
this simulation-only repository.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from hunter_kinodynamic_rl.common.geometry import wrap_to_pi
from hunter_kinodynamic_rl.navigation.localization.interface import PoseEstimate
from hunter_kinodynamic_rl.navigation.localization.odom_backend import DEFAULT_HISTORY_SIZE, OdomLocalizationBackend


@dataclass(frozen=True)
class WheelImuNoiseModel:
    #: Position-covariance-trace growth per meter traveled (wheel slip).
    position_process_noise_m2_per_m: float = 0.01
    #: Heading-covariance growth per radian turned (steering calibration/slip).
    heading_process_noise_rad2_per_rad: float = 0.005
    #: Heading-covariance growth per second elapsed, even while stationary
    #: (IMU gyro bias drift).
    heading_process_noise_rad2_per_sec: float = 0.0005
    confidence_scale: float = 1.0

    def validate(self) -> None:
        for name in (
            "position_process_noise_m2_per_m", "heading_process_noise_rad2_per_rad",
            "heading_process_noise_rad2_per_sec", "confidence_scale",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"WheelImuNoiseModel.{name} must be >= 0")


class WheelImuLocalizationBackend(OdomLocalizationBackend):
    def __init__(self, noise_model: WheelImuNoiseModel = WheelImuNoiseModel(), history_size: int = DEFAULT_HISTORY_SIZE) -> None:
        noise_model.validate()
        super().__init__(history_size)
        self._noise_model = noise_model
        self._x = 0.0
        self._y = 0.0
        self._yaw = 0.0
        self._cov = np.zeros((3, 3), dtype=np.float64)
        self._initialized = False

    def initialize(self, x: float, y: float, yaw: float, stamp_sec: float) -> PoseEstimate:
        """Resets the dead-reckoning origin (e.g. the first valid reading
        at mission start) -- confidence starts at 1.0/zero covariance,
        exactly like a fresh Gazebo ground-truth pose, and only degrades
        from HERE as :meth:`integrate` accumulates drift."""
        self._x, self._y, self._yaw = float(x), float(y), float(yaw)
        self._cov = np.zeros((3, 3), dtype=np.float64)
        self._initialized = True
        return self.update(self._x, self._y, self._yaw, stamp_sec, covariance=self._cov.copy(), confidence=1.0, valid=True)

    def integrate(self, v_mps: float, yaw_rate_rps: float, dt_sec: float, stamp_sec: float) -> PoseEstimate:
        """Bicycle-model Euler integration (midpoint heading, to reduce
        first-order heading-integration bias) + covariance-trace growth
        proportional to distance traveled/heading turned/time elapsed. A
        non-finite or negative-``dt_sec`` reading is DROPPED (returns the
        unchanged current pose, never corrupts the running estimate) --
        mirrors ``OdomLocalizationBackend.update``'s own "only finite
        readings enter history" contract."""
        if not self._initialized:
            raise RuntimeError("WheelImuLocalizationBackend.integrate() called before initialize()")
        if not (math.isfinite(v_mps) and math.isfinite(yaw_rate_rps) and math.isfinite(dt_sec)) or dt_sec < 0.0:
            return self.latest_pose()

        distance = abs(v_mps) * dt_sec
        heading_delta = abs(yaw_rate_rps) * dt_sec
        mid_yaw = wrap_to_pi(self._yaw + 0.5 * yaw_rate_rps * dt_sec)
        self._x += v_mps * dt_sec * math.cos(mid_yaw)
        self._y += v_mps * dt_sec * math.sin(mid_yaw)
        self._yaw = wrap_to_pi(self._yaw + yaw_rate_rps * dt_sec)

        nm = self._noise_model
        self._cov[0, 0] += nm.position_process_noise_m2_per_m * distance
        self._cov[1, 1] += nm.position_process_noise_m2_per_m * distance
        self._cov[2, 2] += (
            nm.heading_process_noise_rad2_per_rad * heading_delta + nm.heading_process_noise_rad2_per_sec * dt_sec
        )
        trace = float(self._cov[0, 0] + self._cov[1, 1] + self._cov[2, 2])
        confidence = 1.0 / (1.0 + nm.confidence_scale * trace)
        return self.update(
            self._x, self._y, self._yaw, stamp_sec, covariance=self._cov.copy(), confidence=confidence, valid=True,
        )
