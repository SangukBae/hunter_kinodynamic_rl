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


def center_steering_to_wheel_angles(
    center_steering_rad: float,
    wheelbase_m: float,
    track_width_m: float,
) -> tuple[float, float]:
    """Convert a bicycle-model center angle to left/right Ackermann angles.

    Positive steering is a left turn.  The return order is ``(left, right)``;
    consequently the left wheel is the inner wheel for a positive command and
    the right wheel is inner for a negative command.
    """
    _validate_ackermann_geometry(wheelbase_m, track_width_m)
    if not math.isfinite(center_steering_rad):
        raise ValueError("center_steering_rad must be finite")
    curvature = math.tan(float(center_steering_rad)) / wheelbase_m
    numerator = wheelbase_m * curvature
    half_track_curvature = 0.5 * track_width_m * curvature
    left = math.atan2(numerator, 1.0 - half_track_curvature)
    right = math.atan2(numerator, 1.0 + half_track_curvature)
    return left, right


def wheel_angles_to_center_steering(
    left_steering_rad: float,
    right_steering_rad: float,
    wheelbase_m: float,
    track_width_m: float,
) -> float:
    """Recover the exact bicycle-model center angle from wheel feedback.

    Averaging the two joint angles is not an Ackermann inverse and introduces
    a steering-dependent bias.  Each measured wheel is first converted to
    centerline curvature; averaging those curvature estimates also makes the
    result tolerant of small encoder disagreement.
    """
    _validate_ackermann_geometry(wheelbase_m, track_width_m)
    left = float(left_steering_rad)
    right = float(right_steering_rad)
    if not (math.isfinite(left) and math.isfinite(right)):
        raise ValueError("left and right steering angles must be finite")

    tan_left = math.tan(left)
    tan_right = math.tan(right)
    left_denominator = wheelbase_m + 0.5 * track_width_m * tan_left
    right_denominator = wheelbase_m - 0.5 * track_width_m * tan_right
    if abs(left_denominator) <= 1e-12 or abs(right_denominator) <= 1e-12:
        raise ValueError("wheel angles are singular for the supplied geometry")
    left_curvature = tan_left / left_denominator
    right_curvature = tan_right / right_denominator
    center_curvature = 0.5 * (left_curvature + right_curvature)
    return math.atan(wheelbase_m * center_curvature)


def center_steering_to_inner_wheel_angle(
    center_steering_rad: float,
    wheelbase_m: float,
    track_width_m: float,
) -> float:
    """Map a signed center command to the signed inner-wheel CAN angle."""
    left, right = center_steering_to_wheel_angles(
        center_steering_rad, wheelbase_m, track_width_m
    )
    return left if center_steering_rad >= 0.0 else right


def inner_wheel_angle_to_center_steering(
    inner_wheel_steering_rad: float,
    wheelbase_m: float,
    track_width_m: float,
) -> float:
    """Map signed inner-wheel feedback/command to a bicycle center angle."""
    _validate_ackermann_geometry(wheelbase_m, track_width_m)
    inner = float(inner_wheel_steering_rad)
    if not math.isfinite(inner):
        raise ValueError("inner_wheel_steering_rad must be finite")
    if abs(inner) >= math.pi / 2.0:
        raise ValueError("inner wheel angle must have magnitude below pi/2")
    if abs(inner) <= 1e-15:
        return 0.0
    center_radius = wheelbase_m / math.tan(abs(inner)) + 0.5 * track_width_m
    center = math.atan(wheelbase_m / center_radius)
    return math.copysign(center, inner)


def _validate_ackermann_geometry(wheelbase_m: float, track_width_m: float) -> None:
    if not math.isfinite(wheelbase_m) or wheelbase_m <= 0.0:
        raise ValueError("wheelbase_m must be finite and > 0")
    if not math.isfinite(track_width_m) or track_width_m < 0.0:
        raise ValueError("track_width_m must be finite and >= 0")


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

    def center_steering_to_wheels(self, steering_rad: float) -> tuple[float, float]:
        """Convert a safety-clamped center command to both wheel angles."""
        return center_steering_to_wheel_angles(
            self.clamp_steering(steering_rad),
            self.config.wheelbase_m,
            self.config.track_width_m,
        )

    def wheel_steering_to_center(self, left_rad: float, right_rad: float) -> float:
        """Convert wheel feedback to center angle without hiding over-travel."""
        return wheel_angles_to_center_steering(
            left_rad, right_rad,
            self.config.wheelbase_m, self.config.track_width_m,
        )

    def center_steering_to_inner_wheel(self, steering_rad: float) -> float:
        """Adapter for Hunter CAN interfaces expressed as inner-wheel angle."""
        return center_steering_to_inner_wheel_angle(
            self.clamp_steering(steering_rad),
            self.config.wheelbase_m,
            self.config.track_width_m,
        )

    def inner_wheel_to_center_steering(self, inner_rad: float) -> float:
        """Adapter from Hunter inner-wheel feedback to center steering."""
        return inner_wheel_angle_to_center_steering(
            inner_rad, self.config.wheelbase_m, self.config.track_width_m
        )
