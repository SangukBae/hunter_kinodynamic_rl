"""Active-subgoal lifecycle + running stats accumulator (plan section 6.4).

``SubgoalManager`` owns exactly ONE subgoal at a time -- the currently
ACTIVE one. ``HierarchyCoordinator`` calls :meth:`activate` when a new
subgoal starts, :meth:`record_tick` once per local control tick while it
is active, and :meth:`finish` when a terminal status is reached, which
freezes everything accumulated so far into an immutable :class:`SubgoalResult`
carrying every stat the plan requires.

Deliberately never touches the FINAL mission goal's own status (that is
``navigation.mission.goal_manager.GoalManager``'s job) -- a subgoal REACHED
is never treated as mission success here or anywhere in this module.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import Optional, Tuple

from hunter_kinodynamic_rl.common.geometry import goal_distance_and_heading
from hunter_kinodynamic_rl.navigation.mission.mission_frame import PoseXYYaw


class SubgoalStatus(enum.Enum):
    CREATED = "created"
    ACTIVE = "active"
    REACHED = "reached"
    FAILED_BLOCKED = "failed_blocked"
    FAILED_TIMEOUT = "failed_timeout"
    FAILED_NO_PROGRESS = "failed_no_progress"
    FAILED_HIGH_RISK = "failed_high_risk"
    CANCELLED_BY_REPLAN = "cancelled_by_replan"


#: Every status a subgoal can END in -- REACHED is the only non-failure
#: terminal outcome (see :data:`FAILURE_SUBGOAL_STATUSES` for the rest).
TERMINAL_SUBGOAL_STATUSES = frozenset({
    SubgoalStatus.REACHED, SubgoalStatus.FAILED_BLOCKED, SubgoalStatus.FAILED_TIMEOUT,
    SubgoalStatus.FAILED_NO_PROGRESS, SubgoalStatus.FAILED_HIGH_RISK, SubgoalStatus.CANCELLED_BY_REPLAN,
})

#: The subset of terminal statuses that represent the LOCAL policy failing
#: to reach the subgoal (as opposed to REACHED, or a system-initiated
#: CANCELLED_BY_REPLAN that is not the local policy's fault).
FAILURE_SUBGOAL_STATUSES = frozenset({
    SubgoalStatus.FAILED_BLOCKED, SubgoalStatus.FAILED_TIMEOUT,
    SubgoalStatus.FAILED_NO_PROGRESS, SubgoalStatus.FAILED_HIGH_RISK,
})


@dataclass(frozen=True)
class SubgoalManagerConfig:
    position_tolerance_m: float = 0.5
    heading_tolerance_rad: float = math.pi
    require_low_speed_on_reach: bool = False
    goal_speed_threshold_mps: float = 0.15

    def validate(self) -> None:
        if self.position_tolerance_m <= 0.0:
            raise ValueError("SubgoalManagerConfig.position_tolerance_m must be > 0")
        if not (0.0 <= self.heading_tolerance_rad <= math.pi):
            raise ValueError("SubgoalManagerConfig.heading_tolerance_rad must be in [0, pi]")
        if self.goal_speed_threshold_mps < 0.0:
            raise ValueError("SubgoalManagerConfig.goal_speed_threshold_mps must be >= 0")


@dataclass(frozen=True)
class SubgoalResult:
    """Every stat the plan (section 6.4) requires for one terminated
    subgoal. ``minimum_clearance_m``/``mean_predicted_risk``/
    ``max_predicted_risk`` are ``None`` when the caller never supplied any
    sample for that subgoal (e.g. no risk critic on this profile, or a
    real-hardware deployment that never computes clearance) -- exactly the
    "없으면 None" contract, never a fabricated 0.0/inf sentinel."""

    subgoal_id: int
    status: SubgoalStatus
    reason: str
    local_steps: int
    elapsed_time_sec: float
    path_length_m: float
    start_final_goal_distance_m: float
    end_final_goal_distance_m: float
    start_subgoal_distance_m: float
    end_subgoal_distance_m: float
    minimum_clearance_m: Optional[float]
    mean_predicted_risk: Optional[float]
    max_predicted_risk: Optional[float]
    emergency_stop_count: int
    steering_saturation_count: int
    newly_explored_cells: int


class SubgoalManager:
    """One active subgoal at a time. ``activate()`` may only be called while
    no subgoal is active (i.e. before the first subgoal, or after the prior
    one's :meth:`finish` -- the coordinator is responsible for sequencing
    this correctly; see ``coordinator.py``)."""

    def __init__(self, config: SubgoalManagerConfig) -> None:
        config.validate()
        self._config = config
        self._next_id = 0
        self._subgoal_xy: Optional[Tuple[float, float]] = None
        self._status: Optional[SubgoalStatus] = None
        self._reset_accumulators()

    def reset(self) -> None:
        """Explicit re-arm for a fresh mission (mirrors
        ``MissionFrame.reset()``) -- clears the active/most-recent subgoal
        back to pre-``activate()`` state (``status=None``,
        :attr:`subgoal_mission_xy` unreadable again) regardless of whether
        a subgoal is currently ACTIVE or already terminal. Never implicit:
        a stale subgoal silently surviving into a NEW mission would let
        that mission's local policy start chasing a subgoal that belongs
        to the PREVIOUS one (code review: ``HierarchyCoordinator.start_mission()``
        must call this)."""
        self._subgoal_xy = None
        self._status = None
        self._reset_accumulators()

    def _reset_accumulators(self) -> None:
        self._start_step = 0
        self._start_time_sec = 0.0
        self._start_final_goal_distance_m = 0.0
        self._start_subgoal_distance_m = 0.0
        self._last_final_goal_distance_m = 0.0
        self._last_subgoal_distance_m = 0.0
        self._path_length_m = 0.0
        self._local_steps = 0
        self._min_clearance: Optional[float] = None
        self._risk_samples: list = []
        self._emergency_stop_count = 0
        self._steering_saturation_count = 0
        self._newly_explored_cells = 0

    @property
    def status(self) -> Optional[SubgoalStatus]:
        return self._status

    @property
    def active(self) -> bool:
        return self._status == SubgoalStatus.ACTIVE

    @property
    def local_steps(self) -> int:
        return self._local_steps

    @property
    def subgoal_mission_xy(self) -> Tuple[float, float]:
        if self._subgoal_xy is None:
            raise RuntimeError("SubgoalManager.subgoal_mission_xy read before activate()")
        return self._subgoal_xy

    def activate(
        self, x_mission: float, y_mission: float, *, now_step: int, now_time_sec: float,
        final_goal_distance_m: float, subgoal_distance_m: float,
    ) -> int:
        """Begins tracking a new subgoal; returns its id (monotonically
        increasing across the manager's lifetime, unique per mission)."""
        if self._status == SubgoalStatus.ACTIVE:
            raise RuntimeError(
                "SubgoalManager.activate() called while a subgoal is still ACTIVE -- "
                "finish() it first (the coordinator must not overlap two active subgoals)"
            )
        self._next_id += 1
        self._subgoal_xy = (float(x_mission), float(y_mission))
        self._status = SubgoalStatus.ACTIVE
        self._reset_accumulators()
        self._start_step = int(now_step)
        self._start_time_sec = float(now_time_sec)
        self._start_final_goal_distance_m = float(final_goal_distance_m)
        self._start_subgoal_distance_m = float(subgoal_distance_m)
        self._last_final_goal_distance_m = float(final_goal_distance_m)
        self._last_subgoal_distance_m = float(subgoal_distance_m)
        return self._next_id

    def distance_to_subgoal(self, robot_pose_mission: PoseXYYaw) -> Tuple[float, float]:
        gx, gy = self.subgoal_mission_xy
        return goal_distance_and_heading(robot_pose_mission.x, robot_pose_mission.y, robot_pose_mission.yaw, gx, gy)

    def check_reached(self, robot_pose_mission: PoseXYYaw, speed_mps: Optional[float] = None) -> bool:
        """Pure predicate -- does NOT transition status itself (unlike
        ``GoalManager.check_reached``); the coordinator decides when a
        REACHED predicate becomes a terminal :meth:`finish` call, since
        reaching a subgoal must also be checked against replanning's other
        triggers in the same tick."""
        if not self.active:
            return False
        dist, bearing = self.distance_to_subgoal(robot_pose_mission)
        if dist > self._config.position_tolerance_m:
            return False
        if abs(bearing) > self._config.heading_tolerance_rad:
            return False
        if self._config.require_low_speed_on_reach:
            if speed_mps is None or not math.isfinite(speed_mps) or abs(speed_mps) > self._config.goal_speed_threshold_mps:
                return False
        return True

    def record_tick(
        self, *, step_delta_m: float, final_goal_distance_m: float, subgoal_distance_m: float,
        clearance_m: Optional[float] = None, predicted_risk: Optional[float] = None,
        emergency_stop: bool = False, steering_saturated: bool = False, newly_explored_cells: int = 0,
    ) -> None:
        if not self.active:
            raise RuntimeError("SubgoalManager.record_tick() called while not ACTIVE")
        self._local_steps += 1
        self._path_length_m += max(0.0, float(step_delta_m))
        self._last_final_goal_distance_m = float(final_goal_distance_m)
        self._last_subgoal_distance_m = float(subgoal_distance_m)
        if clearance_m is not None:
            self._min_clearance = clearance_m if self._min_clearance is None else min(self._min_clearance, clearance_m)
        if predicted_risk is not None:
            self._risk_samples.append(float(predicted_risk))
        if emergency_stop:
            self._emergency_stop_count += 1
        if steering_saturated:
            self._steering_saturation_count += 1
        self._newly_explored_cells += int(newly_explored_cells)

    def finish(self, status: SubgoalStatus, reason: str, *, now_step: int, now_time_sec: float) -> SubgoalResult:
        if not self.active:
            raise RuntimeError(f"SubgoalManager.finish() called while status={self._status} (must be ACTIVE)")
        if status not in TERMINAL_SUBGOAL_STATUSES:
            raise ValueError(f"finish() status must be one of {TERMINAL_SUBGOAL_STATUSES}, got {status}")
        result = SubgoalResult(
            subgoal_id=self._next_id, status=status, reason=str(reason),
            local_steps=self._local_steps,
            elapsed_time_sec=max(0.0, float(now_time_sec) - self._start_time_sec),
            path_length_m=self._path_length_m,
            start_final_goal_distance_m=self._start_final_goal_distance_m,
            end_final_goal_distance_m=self._last_final_goal_distance_m,
            start_subgoal_distance_m=self._start_subgoal_distance_m,
            end_subgoal_distance_m=self._last_subgoal_distance_m,
            minimum_clearance_m=self._min_clearance,
            mean_predicted_risk=(sum(self._risk_samples) / len(self._risk_samples)) if self._risk_samples else None,
            max_predicted_risk=max(self._risk_samples) if self._risk_samples else None,
            emergency_stop_count=self._emergency_stop_count,
            steering_saturation_count=self._steering_saturation_count,
            newly_explored_cells=self._newly_explored_cells,
        )
        self._status = status
        return result
