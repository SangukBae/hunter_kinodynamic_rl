"""LIO-SAM (or any ``nav_msgs/Odometry``-publishing LiDAR-Inertial
Odometry backend) adapter (plan section 4/10.5 Phase E). Requires
``rclpy``'s message types -- ROS-free code never imports this module
directly, mirroring
:mod:`~hunter_kinodynamic_rl.navigation.localization.gazebo_odom_backend`'s
own documented convention exactly. LIO-SAM's own mapping/odometry topic
publishes the SAME ``nav_msgs/Odometry`` message type Gazebo's bridge does,
so the message-PARSING logic is reused verbatim from that module (never
reimplemented here) -- only the confidence-scale default differs, since a
real LIO's covariance is typically a MEANINGFUL, nonzero estimate (unlike
Gazebo's commonly all-zero ground-truth covariance).
"""

from __future__ import annotations

import numpy as np
from nav_msgs.msg import Odometry

from hunter_kinodynamic_rl.navigation.localization.gazebo_odom_backend import (
    confidence_from_covariance, yaw_from_quaternion,
)
from hunter_kinodynamic_rl.navigation.localization.interface import PoseEstimate
from hunter_kinodynamic_rl.navigation.localization.odom_backend import DEFAULT_HISTORY_SIZE, OdomLocalizationBackend


class LioLocalizationBackend(OdomLocalizationBackend):
    """``confidence_scale`` defaults higher than
    ``GazeboOdomLocalizationBackend``'s (``5.0`` vs. ``1.0``) -- a
    real LIO's covariance trace is typically already a small, physically
    meaningful number (not Gazebo's frequently-all-zero ground truth), so a
    smaller scale would map even a healthy LIO estimate's covariance too
    close to 1.0 confidence to ever distinguish it from a degraded one."""

    def __init__(self, confidence_scale: float = 5.0, history_size: int = DEFAULT_HISTORY_SIZE) -> None:
        super().__init__(history_size)
        self._confidence_scale = confidence_scale

    def on_lio_odometry_msg(self, msg: Odometry, stamp_sec: float) -> PoseEstimate:
        pos = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)
        cov_flat = np.asarray(msg.pose.covariance, dtype=np.float64).reshape(6, 6)
        cov_xy_yaw = np.array(
            [
                [cov_flat[0, 0], cov_flat[0, 1], cov_flat[0, 5]],
                [cov_flat[1, 0], cov_flat[1, 1], cov_flat[1, 5]],
                [cov_flat[5, 0], cov_flat[5, 1], cov_flat[5, 5]],
            ],
            dtype=np.float64,
        )
        trace = float(cov_xy_yaw[0, 0] + cov_xy_yaw[1, 1] + cov_xy_yaw[2, 2])
        confidence = confidence_from_covariance(trace, scale=self._confidence_scale)
        return self.update(
            x=pos.x, y=pos.y, yaw=yaw, stamp_sec=stamp_sec, covariance=cov_xy_yaw, confidence=confidence, valid=True,
        )
