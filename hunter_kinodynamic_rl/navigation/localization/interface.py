"""Localization backend protocol (plan section 5.3) -- lets mapping/mission
code depend on ``latest_pose()``/``pose_at() -> PoseEstimate`` alone, never
on which real backend (Gazebo odom, wheel+IMU, LiDAR odometry, LIO -- see
Phase 6) produced it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class PoseEstimate:
    x: float
    y: float
    yaw: float
    stamp_sec: float
    covariance: np.ndarray = field(default_factory=lambda: np.zeros((3, 3), dtype=np.float64))
    confidence: float = 0.0
    valid: bool = False


INVALID_POSE = PoseEstimate(x=0.0, y=0.0, yaw=0.0, stamp_sec=0.0, valid=False, confidence=0.0)


@runtime_checkable
class LocalizationBackend(Protocol):
    def latest_pose(self) -> PoseEstimate: ...

    # section (code review round 2, item 5): part of the ABSTRACTION, not
    # a `GazeboOdomLocalizationBackend`-only implementation detail --
    # `mission_map_node.py` depends on `pose_at` directly, and every future
    # Phase 6 backend (wheel+IMU, LiDAR odometry, LIO) must provide the
    # same timestamp-synchronized lookup for the swap to be a real
    # backend-only change rather than a navigation-core rewrite.
    def pose_at(self, stamp_sec: float, max_dt_sec: float) -> PoseEstimate: ...


def is_pose_finite(pose: PoseEstimate) -> bool:
    """Every numeric field -- INCLUDING ``confidence`` -- must be finite,
    and ``confidence`` must additionally lie within ``[0, 1]``. A backend
    bug or a corrupted upstream message could otherwise produce a NaN/Inf
    pose that still has ``valid=True`` set, silently poisoning
    mission-frame initialization or map/rolling-crop coordinate math
    downstream. Checked independently of ``valid`` (defense in depth:
    backends are ALSO required to force ``valid=False`` on a non-finite
    reading themselves, see ``OdomLocalizationBackend.update``).

    section (code review round 2, item 2): ``confidence`` was previously
    NOT checked here -- a NaN ``confidence`` slipped through unnoticed,
    since ``NaN < min_confidence`` is ``False`` in Python (every comparison
    against NaN is False), so ``is_pose_usable``'s own
    ``confidence < min_confidence`` rejection silently never fired for a
    NaN confidence."""
    if not (math.isfinite(pose.x) and math.isfinite(pose.y) and math.isfinite(pose.yaw)
            and math.isfinite(pose.stamp_sec) and math.isfinite(pose.confidence)):
        return False
    if not (0.0 <= pose.confidence <= 1.0):
        return False
    return bool(np.all(np.isfinite(pose.covariance)))


def is_pose_usable(pose: PoseEstimate, now_sec: float, timeout_sec: float, min_confidence: float) -> bool:
    """A single, shared "may this pose be trusted for a map/goal update right
    now" gate -- used identically by mapping (reject a stale/invalid scan
    pose) and by the mission node (reject stop/goal checks on bad
    localization). Never silently falls back to a stale or non-finite
    reading.

    ``now_sec`` MUST be in the SAME clock domain as ``pose.stamp_sec`` --
    e.g. both message header stamps (as when synchronizing a scan against a
    pose via ``pose_at``), never one side wall-clock (``time.time()``) and
    the other Gazebo sim time, which would make every comparison here
    meaningless (confirmed live in this exact codebase's history, see
    ``evaluation/nav2_mppi_runner.py``'s module docstring: "a use_sim_time
    mismatch between Nav2's clock and Gazebo-bridged /odometry's sim-time
    stamps"). The age check is symmetric (``abs(age)``, not just
    ``age > timeout``) so a pose sampled slightly AFTER ``now_sec`` --
    routine when ``now_sec`` is a scan timestamp being synchronized against
    odometry rather than genuine wall/sim "now" -- is judged by the same
    bounded window instead of always failing."""
    if not pose.valid:
        return False
    if not is_pose_finite(pose):
        return False
    if pose.confidence < min_confidence:
        return False
    age = now_sec - pose.stamp_sec
    if abs(age) > timeout_sec:
        return False
    return True
