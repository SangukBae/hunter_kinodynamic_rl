"""``nav_msgs/Odometry`` -> :class:`PoseEstimate` adapter (the ``gazebo_odom``
backend named in ``LocalizationConfig.backend``). Requires ``rclpy``'s message
types to be importable (i.e. a built, sourced ROS workspace) -- ROS-free
mission/mapping code never imports this module directly, only
:mod:`hunter_kinodynamic_rl.navigation.localization.odom_backend`.
"""

from __future__ import annotations

import math

import numpy as np
from nav_msgs.msg import Odometry

from hunter_kinodynamic_rl.navigation.localization.odom_backend import OdomLocalizationBackend


def yaw_from_quaternion(qx: float, qy: float, qz: float, qw: float) -> float:
    """Standard yaw-from-quaternion extraction (Z-axis rotation only, valid
    for a ground vehicle's planar odometry)."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def confidence_from_covariance(cov_xy_yaw_trace: float, scale: float = 1.0) -> float:
    """Maps a nonnegative covariance-trace summary to ``(0, 1]`` -- a
    Gazebo-simulated odometry topic commonly reports an all-zero covariance
    (perfect confidence, trace 0 -> 1.0); a real/noisy backend reports a
    growing trace that this decays toward 0 without ever reaching it exactly
    (an unbounded-but-imperfect confidence is more honest than a hard cutoff
    here -- callers gate usability via ``minimum_confidence`` instead). A
    non-finite trace (corrupted covariance) maps to 0.0 confidence, never a
    misleadingly high one -- ``OdomLocalizationBackend.update`` separately
    forces the resulting pose invalid outright on a non-finite covariance,
    this is defense in depth for the confidence VALUE itself."""
    trace = float(cov_xy_yaw_trace)
    if not math.isfinite(trace):
        return 0.0
    trace = max(0.0, trace)
    return 1.0 / (1.0 + scale * trace)


class GazeboOdomLocalizationBackend(OdomLocalizationBackend):
    """Feeds :meth:`OdomLocalizationBackend.update` directly from
    ``nav_msgs/Odometry`` messages published by the Ignition
    odometry bridge (``/odometry``, see CLAUDE.md's Gazebo-ROS2 Bridge
    section).

    ``use_covariance_confidence`` distinguishes ``LocalizationConfig``'s two
    backend names: ``"gazebo_odom"`` (default, ``True``) derives confidence
    from the message's own covariance trace; ``"odom"`` (``False``) trusts
    the reading unconditionally (confidence pinned to 1.0) -- for a plain
    odometry source with no meaningful covariance estimate of its own."""

    def __init__(self, confidence_scale: float = 1.0, use_covariance_confidence: bool = True) -> None:
        super().__init__()
        self._confidence_scale = confidence_scale
        self._use_covariance_confidence = use_covariance_confidence

    def on_odometry_msg(self, msg: Odometry, stamp_sec: float) -> None:
        pos = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)
        cov_flat = np.asarray(msg.pose.covariance, dtype=np.float64).reshape(6, 6)
        # (x, y, yaw) sub-block out of ROS's 6x6 (x,y,z,roll,pitch,yaw) pose covariance.
        cov_xy_yaw = np.array(
            [
                [cov_flat[0, 0], cov_flat[0, 1], cov_flat[0, 5]],
                [cov_flat[1, 0], cov_flat[1, 1], cov_flat[1, 5]],
                [cov_flat[5, 0], cov_flat[5, 1], cov_flat[5, 5]],
            ],
            dtype=np.float64,
        )
        if self._use_covariance_confidence:
            trace = float(cov_xy_yaw[0, 0] + cov_xy_yaw[1, 1] + cov_xy_yaw[2, 2])
            confidence = confidence_from_covariance(trace, scale=self._confidence_scale)
        else:
            confidence = 1.0
        self.update(
            x=pos.x, y=pos.y, yaw=yaw, stamp_sec=stamp_sec,
            covariance=cov_xy_yaw, confidence=confidence, valid=True,
        )
