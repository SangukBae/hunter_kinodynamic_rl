"""Pure (ROS-free) odometry-fed :class:`LocalizationBackend`. A caller feeds
new readings via :meth:`update`; ``latest_pose()`` always returns the most
recent one, and :meth:`pose_at` looks up (interpolating a bounded pose
history) the pose closest to an ARBITRARY past timestamp -- e.g. a LiDAR
scan's own header stamp, which generally does not equal whatever odometry
reading happened to be "latest" when the scan callback ran (odom and scan
are independent topics arriving at different rates/order). Kept
ROS-independent so mapping/mission-frame logic can be unit tested against
synthetic pose sequences with no ``rclpy`` dependency at all -- see
:mod:`hunter_kinodynamic_rl.navigation.localization.gazebo_odom_backend`
for the thin ``nav_msgs/Odometry``-parsing wrapper around this class.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, Optional

import numpy as np

from hunter_kinodynamic_rl.common.geometry import wrap_to_pi
from hunter_kinodynamic_rl.navigation.localization.interface import INVALID_POSE, PoseEstimate

DEFAULT_HISTORY_SIZE = 50


def _is_finite_pose(
    x: float, y: float, yaw: float, stamp_sec: float, confidence: float, covariance: np.ndarray,
) -> bool:
    # section (code review round 2, item 2): `confidence` is included here
    # too, not just x/y/yaw/stamp/covariance -- a NaN confidence previously
    # slipped through every downstream `confidence < min_confidence` gate
    # unnoticed, since NaN compares False against everything in Python.
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(yaw) and math.isfinite(stamp_sec)
            and math.isfinite(confidence)):
        return False
    if not (0.0 <= confidence <= 1.0):
        return False
    return bool(np.all(np.isfinite(covariance)))


def _interpolate(before: PoseEstimate, after: PoseEstimate, stamp_sec: float) -> PoseEstimate:
    span = after.stamp_sec - before.stamp_sec
    t = 0.0 if span <= 0.0 else (stamp_sec - before.stamp_sec) / span
    t = max(0.0, min(1.0, t))
    x = before.x + t * (after.x - before.x)
    y = before.y + t * (after.y - before.y)
    # Shortest-path (wrapped) yaw interpolation -- a plain linear blend of
    # the raw yaw values breaks across the +-pi wraparound.
    yaw = wrap_to_pi(before.yaw + t * wrap_to_pi(after.yaw - before.yaw))
    covariance = before.covariance + t * (after.covariance - before.covariance)
    return PoseEstimate(
        x=x, y=y, yaw=yaw, stamp_sec=float(stamp_sec), covariance=covariance,
        confidence=min(before.confidence, after.confidence), valid=before.valid and after.valid,
    )


class OdomLocalizationBackend:
    def __init__(self, history_size: int = DEFAULT_HISTORY_SIZE) -> None:
        self._latest: PoseEstimate = INVALID_POSE
        self._history: Deque[PoseEstimate] = deque(maxlen=history_size)

    def latest_pose(self) -> PoseEstimate:
        return self._latest

    def pose_at(self, stamp_sec: float, max_dt_sec: float) -> PoseEstimate:
        """The pose closest to ``stamp_sec`` -- linearly interpolated
        between the two bracketing history samples ONLY when ``stamp_sec``
        is within ``max_dt_sec`` of BOTH of them, otherwise the single
        nearest sample IF it is within ``max_dt_sec`` (never silently
        extrapolated an unbounded distance). Only finite,
        successfully-``update()``-d samples are ever stored in history, so
        an interpolation result is never built from a NaN/Inf reading.

        section (code review round 2, item 1): a bracket existing on BOTH
        sides used to be sufficient to interpolate REGARDLESS of how far
        apart the two samples were -- e.g. odometry at t=0 and t=100
        bracketing a scan at t=50 would happily synthesize a pose halfway
        along a straight line across a 100s gap (a localization outage, a
        rosbag replay stall, ...), silently poisoning the map with a
        fabricated trajectory. Interpolation now requires EACH side to be
        within ``max_dt_sec`` of ``stamp_sec`` individually; a wide-but-
        one-sided-close bracket falls back to the single nearest sample
        (still gated by the same ``max_dt_sec`` bound below), and a
        wide-and-neither-side-close bracket is rejected outright.
        """
        before: Optional[PoseEstimate] = None
        after: Optional[PoseEstimate] = None
        for p in self._history:  # oldest -> newest (deque append order)
            if p.stamp_sec <= stamp_sec:
                before = p
            elif after is None:
                after = p
        if before is not None and after is not None:
            before_gap = stamp_sec - before.stamp_sec
            after_gap = after.stamp_sec - stamp_sec
            if before_gap <= max_dt_sec and after_gap <= max_dt_sec:
                return _interpolate(before, after, stamp_sec)
            nearest = before if before_gap <= after_gap else after
        else:
            nearest = after if before is None else before
        if nearest is None or abs(nearest.stamp_sec - stamp_sec) > max_dt_sec:
            return INVALID_POSE
        return nearest

    def update(
        self, x: float, y: float, yaw: float, stamp_sec: float,
        covariance: np.ndarray | None = None, confidence: float = 1.0, valid: bool = True,
    ) -> PoseEstimate:
        cov = np.zeros((3, 3), dtype=np.float64) if covariance is None else np.asarray(covariance, dtype=np.float64)
        finite = _is_finite_pose(float(x), float(y), float(yaw), float(stamp_sec), float(confidence), cov)
        pose = PoseEstimate(
            x=float(x), y=float(y), yaw=float(yaw), stamp_sec=float(stamp_sec),
            # section (code review, item 3): a NaN/Inf reading is FORCED
            # invalid regardless of what the caller passed for `valid` --
            # never left to every downstream consumer to independently
            # re-derive "is this actually usable". Now also covers a
            # NaN/Inf/out-of-[0,1] `confidence` (round 2, item 2).
            covariance=cov, confidence=float(confidence), valid=bool(valid) and finite,
        )
        self._latest = pose
        if finite:
            self._history.append(pose)
        return pose

    def invalidate(self) -> None:
        """Explicitly mark localization lost (e.g. backend health check
        failure) -- never left to a stale ``update()`` reading being
        silently reused past its actual validity. Also clears history, so
        ``pose_at`` never interpolates a synchronization gap across a
        genuine localization outage."""
        self._latest = INVALID_POSE
        self._history.clear()
