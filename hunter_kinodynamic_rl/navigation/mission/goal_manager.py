"""Relative final-goal bookkeeping (plan section 5.3's ``GoalManager``).

The user's relative goal ``(gx, gy)`` is defined in the robot's OWN frame at
the instant the mission starts -- by construction that is exactly the
:class:`~hunter_kinodynamic_rl.navigation.mission.mission_frame.MissionFrame`
origin, so the mission-frame goal is numerically identical to the
user-supplied relative goal; no separate transform is needed (unlike a goal
re-issued mid-mission, which would need a real robot-pose-at-that-moment
transform via ``MissionFrame.robot_to_mission``).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Optional, Tuple

from hunter_kinodynamic_rl.common.geometry import goal_distance_and_heading
from hunter_kinodynamic_rl.navigation.mission.mission_frame import PoseXYYaw


class MissionStatus(enum.Enum):
    INACTIVE = "inactive"
    ACTIVE = "active"
    REACHED = "reached"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class GoalManagerConfig:
    position_tolerance_m: float = 0.6
    heading_tolerance_rad: float = 3.141592653589793
    require_low_speed_on_goal: bool = True
    goal_speed_threshold_mps: float = 0.1


class GoalManager:
    """Owns the mission's single final goal and its lifecycle status. Does
    NOT own subgoals (plan Phase 2) -- Phase 1 only needs the final-goal
    contract."""

    def __init__(self, config: GoalManagerConfig, reset_memory_on_goal_change: bool = True) -> None:
        self._config = config
        self.reset_memory_on_goal_change = reset_memory_on_goal_change
        self._relative_goal: Optional[Tuple[float, float]] = None
        self._mission_goal: Optional[Tuple[float, float]] = None
        self._status: MissionStatus = MissionStatus.INACTIVE

    @property
    def status(self) -> MissionStatus:
        return self._status

    @property
    def active(self) -> bool:
        return self._status == MissionStatus.ACTIVE

    @property
    def relative_goal(self) -> Tuple[float, float]:
        if self._relative_goal is None:
            raise RuntimeError("GoalManager.relative_goal read before set_goal()")
        return self._relative_goal

    @property
    def mission_goal(self) -> Tuple[float, float]:
        if self._mission_goal is None:
            raise RuntimeError("GoalManager.mission_goal read before set_goal()")
        return self._mission_goal

    def set_goal(self, gx: float, gy: float) -> bool:
        """Set/replace the user relative final goal. Returns whether the
        caller should reset map/memory state (``reset_memory_on_goal_change``
        AND this is actually a goal change, not the initial set from
        INACTIVE)."""
        is_change = self._status != MissionStatus.INACTIVE
        self._relative_goal = (float(gx), float(gy))
        self._mission_goal = self._relative_goal
        self._status = MissionStatus.ACTIVE
        return is_change and self.reset_memory_on_goal_change

    def cancel(self) -> None:
        self._status = MissionStatus.CANCELLED

    def distance_and_bearing(self, robot_pose_mission: PoseXYYaw) -> Tuple[float, float]:
        gx, gy = self.mission_goal
        return goal_distance_and_heading(robot_pose_mission.x, robot_pose_mission.y, robot_pose_mission.yaw, gx, gy)

    def check_reached(self, robot_pose_mission: PoseXYYaw, speed_mps: Optional[float] = None) -> bool:
        """Evaluate final-goal tolerance; transitions to REACHED and returns
        True on success. A no-op (returns False) once the mission is no
        longer ACTIVE."""
        if self._status != MissionStatus.ACTIVE:
            return False
        dist, bearing = self.distance_and_bearing(robot_pose_mission)
        if dist > self._config.position_tolerance_m:
            return False
        if abs(bearing) > self._config.heading_tolerance_rad:
            return False
        if self._config.require_low_speed_on_goal:
            if speed_mps is None or abs(speed_mps) > self._config.goal_speed_threshold_mps:
                return False
        self._status = MissionStatus.REACHED
        return True
