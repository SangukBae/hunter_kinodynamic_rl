"""Mission-start frame: a fixed coordinate frame captured at the robot's
pose the instant a mission begins, so a user-specified relative goal and the
accumulated partial map stay meaningful as the robot moves and rotates (see
``docs/IMPLEMENTATION_PLAN.md`` section 5.3).

All rotations reuse :mod:`hunter_kinodynamic_rl.common.geometry`'s
``to_robot_frame``/``to_world_frame`` -- mission_to_odom/robot_to_mission
rotate a *local* offset into a fixed frame (``to_world_frame``) and
odom_to_mission/mission_to_robot do the inverse (``to_robot_frame``), so
``MissionFrame`` is really that same transform applied twice: once between
odom and mission (origin = the captured start pose) and once between
mission and the current robot pose.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from hunter_kinodynamic_rl.common.geometry import to_robot_frame, to_world_frame, wrap_to_pi


@dataclass(frozen=True)
class PoseXYYaw:
    x: float
    y: float
    yaw: float


class MissionFrameError(RuntimeError):
    """Raised when a :class:`MissionFrame` is used before ``initialize()``."""


class MissionFrame:
    """Fixed mission-start frame. ``initialize()`` may be called exactly
    once per mission -- a second call without an intervening ``reset()``
    raises :class:`MissionFrameError` rather than silently overwriting the
    origin (a caller that races two initializations, e.g. two odometry
    callbacks both observing ``not initialized`` before either commits,
    must fail loudly, not quietly re-anchor the whole mission mid-flight).
    Every conversion method also raises :class:`MissionFrameError` until
    initialized."""

    def __init__(self) -> None:
        self._origin: Optional[PoseXYYaw] = None

    @property
    def initialized(self) -> bool:
        return self._origin is not None

    @property
    def origin(self) -> PoseXYYaw:
        if self._origin is None:
            raise MissionFrameError("MissionFrame.origin read before initialize()")
        return self._origin

    def initialize(self, start_pose: PoseXYYaw) -> None:
        if self._origin is not None:
            raise MissionFrameError(
                "MissionFrame.initialize() called while already initialized -- call reset() first "
                "to start a genuinely new mission"
            )
        if not (math.isfinite(start_pose.x) and math.isfinite(start_pose.y) and math.isfinite(start_pose.yaw)):
            raise ValueError(f"MissionFrame.initialize() requires a finite start_pose, got {start_pose}")
        self._origin = PoseXYYaw(x=start_pose.x, y=start_pose.y, yaw=wrap_to_pi(start_pose.yaw))

    def reset(self) -> None:
        """Explicit re-arm for a new mission (never implicit -- a stale
        origin silently reused across missions would corrupt every
        downstream goal/map coordinate)."""
        self._origin = None

    def mission_to_odom(self, x: float, y: float, yaw: Optional[float] = None) -> Tuple[float, float] | Tuple[float, float, float]:
        origin = self.origin
        wx, wy = to_world_frame(x, y, origin.yaw)
        ox, oy = origin.x + wx, origin.y + wy
        if yaw is None:
            return ox, oy
        return ox, oy, wrap_to_pi(origin.yaw + yaw)

    def odom_to_mission(self, x: float, y: float, yaw: Optional[float] = None) -> Tuple[float, float] | Tuple[float, float, float]:
        origin = self.origin
        dx, dy = x - origin.x, y - origin.y
        mx, my = to_robot_frame(dx, dy, origin.yaw)
        if yaw is None:
            return mx, my
        return mx, my, wrap_to_pi(yaw - origin.yaw)

    def odom_pose_to_mission(self, pose: PoseXYYaw) -> PoseXYYaw:
        x, y, yaw = self.odom_to_mission(pose.x, pose.y, pose.yaw)
        return PoseXYYaw(x=x, y=y, yaw=yaw)

    def mission_pose_to_odom(self, pose: PoseXYYaw) -> PoseXYYaw:
        x, y, yaw = self.mission_to_odom(pose.x, pose.y, pose.yaw)
        return PoseXYYaw(x=x, y=y, yaw=yaw)

    @staticmethod
    def mission_to_robot(point_mission_xy: Tuple[float, float], robot_pose_mission: PoseXYYaw) -> Tuple[float, float]:
        """A mission-frame point, expressed relative to ``robot_pose_mission``
        (itself given in mission frame) -- i.e. what the robot currently sees
        that point as, in its own body frame. Pure function of its two
        mission-frame arguments; does not require ``initialize()``."""
        dx = point_mission_xy[0] - robot_pose_mission.x
        dy = point_mission_xy[1] - robot_pose_mission.y
        return to_robot_frame(dx, dy, robot_pose_mission.yaw)

    @staticmethod
    def robot_to_mission(point_robot_xy: Tuple[float, float], robot_pose_mission: PoseXYYaw) -> Tuple[float, float]:
        """Inverse of :meth:`mission_to_robot`: a robot-body-frame point
        (e.g. a candidate subgoal), converted to mission frame given the
        robot's current mission-frame pose."""
        wx, wy = to_world_frame(point_robot_xy[0], point_robot_xy[1], robot_pose_mission.yaw)
        return robot_pose_mission.x + wx, robot_pose_mission.y + wy
