"""Multi-timescale hierarchy coordinator (plan section 6) -- decides WHERE
(which subgoal is active) while ``local_rl.controller.LocalPolicyController``
decides HOW to reach it. Pure Python, no ROS, no Global RL dependency: a
caller-populated queue of candidate subgoal coordinates drives "WHERE" until
Phase 4 adds a learned subgoal source with the same
:meth:`HierarchyCoordinator.activate_next_subgoal` contract this module
already exposes (its ``is_valid`` hook + :mod:`failure_recovery`'s
retry/advance/abort decision).

Owns:
  - the FINAL mission goal, via
    :class:`~hunter_kinodynamic_rl.navigation.mission.goal_manager.GoalManager`
  - the ACTIVE subgoal, via
    :class:`~hunter_kinodynamic_rl.navigation.hierarchy.subgoal_manager.SubgoalManager`
  - a pending queue of candidate subgoals (mission frame)
  - :mod:`replanning` + :mod:`failure_recovery`

Does NOT own sensor I/O, ``LocalPolicyController``, or command publishing.
A caller wires the two together per tick: read :attr:`stop_required` FIRST
and publish ``STOP_COMMAND`` without calling the local policy at all when
it's True; otherwise pass :attr:`active_subgoal_mission` (never
:attr:`final_goal_mission`) into ``LocalPolicyController.build_observation``,
then report the tick's outcome back via :meth:`record_local_tick`.

Degraded-localization safety contract (plan section 6, safety conditions --
"stale/invalid localization이면 physical motion 금지"): a
``record_local_tick`` call whose ``localization_confidence`` falls below
``hierarchy.localization_min_confidence`` always cancels the active subgoal
(``CANCELLED_BY_REPLAN``) and enters :attr:`localization_degraded` --
``stop_required`` then stays True and :meth:`activate_next_subgoal` refuses
to hand out ANY subgoal (queued candidates included) until a LATER call to
:meth:`activate_next_subgoal` reports a recovered confidence reading. A
caller whose sensor loop keeps sampling localization while
``stop_required`` is True should keep calling ``activate_next_subgoal``
each tick with its current confidence reading -- exactly like polling for
"is a next subgoal ready yet" -- so recovery is picked up as soon as it
happens, bounded by the mission timeout if it never does.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Callable, Deque, Optional, Tuple

from hunter_kinodynamic_rl.common.geometry import euclidean_distance, goal_distance_and_heading
from hunter_kinodynamic_rl.navigation.hierarchy.failure_recovery import (
    FailureRecoveryConfig, FailureRecoveryPolicy, RecoveryAction,
)
from hunter_kinodynamic_rl.navigation.hierarchy.replanning import ReplanningConfig, evaluate_replanning
from hunter_kinodynamic_rl.navigation.hierarchy.subgoal_manager import (
    SubgoalManager, SubgoalManagerConfig, SubgoalResult, SubgoalStatus,
)
from hunter_kinodynamic_rl.navigation.mission.goal_manager import GoalManager, GoalManagerConfig, MissionStatus
from hunter_kinodynamic_rl.navigation.mission.mission_frame import PoseXYYaw

#: ``Callable[[x_mission, y_mission], bool]`` -- True means the candidate is
#: usable (e.g. its endpoint is neither occupied nor inflated on the
#: current partial map). Kept as a type alias purely for readability at
#: call sites; never imported/enforced structurally.
SubgoalValidityCheck = Callable[[float, float], bool]


def _coerce_confidence(confidence: Optional[float]) -> Optional[float]:
    if confidence is None:
        return None
    try:
        return float(confidence)
    except (TypeError, ValueError):
        return float("nan")


def _confidence_is_healthy(confidence: Optional[float], min_confidence: float) -> bool:
    value = _coerce_confidence(confidence)
    return value is not None and math.isfinite(value) and value >= min_confidence


def _confidence_is_degraded(confidence: Optional[float], min_confidence: float) -> bool:
    value = _coerce_confidence(confidence)
    return value is not None and (not math.isfinite(value) or value < min_confidence)


@dataclass(frozen=True)
class HierarchyCoordinatorConfig:
    goal: GoalManagerConfig = field(default_factory=GoalManagerConfig)
    subgoal: SubgoalManagerConfig = field(default_factory=SubgoalManagerConfig)
    replanning: ReplanningConfig = field(default_factory=ReplanningConfig)
    recovery: FailureRecoveryConfig = field(default_factory=FailureRecoveryConfig)
    mission_timeout_steps: Optional[int] = None
    mission_timeout_sec: Optional[float] = None


class HierarchyCoordinator:
    def __init__(self, config: HierarchyCoordinatorConfig) -> None:
        self._config = config
        self._goal_manager = GoalManager(config.goal)
        self._subgoal_manager = SubgoalManager(config.subgoal)
        self._recovery = FailureRecoveryPolicy(config.recovery)
        self._queue: Deque[Tuple[float, float]] = deque()
        self._retry_count = 0
        self._current_candidate: Optional[Tuple[float, float]] = None
        self._distance_history: Deque[float] = deque(maxlen=config.replanning.no_progress_window_steps)
        self._consecutive_emergency_stops = 0
        self._last_pose: Optional[PoseXYYaw] = None
        self._mission_start_step = 0
        self._mission_start_time_sec = 0.0
        self._mission_failed = False
        self._mission_timed_out = False
        self._last_subgoal_result: Optional[SubgoalResult] = None
        self._subgoal_reached = False
        self._subgoal_failed = False
        self._previous_final_goal_distance = 0.0
        self._previous_subgoal_distance = 0.0
        self._last_final_goal_distance = 0.0
        self._last_subgoal_distance = 0.0
        self._localization_degraded = False

    # ---------------------------------------------------------- mission lifecycle
    def start_mission(self, final_goal_x: float, final_goal_y: float, *, now_step: int = 0, now_time_sec: float = 0.0) -> None:
        """Sets the FINAL mission goal (relative-to-mission-start, same
        convention as ``GoalManager.set_goal``) and resets every piece of
        Phase 2 state for a fresh mission. Does NOT activate a subgoal --
        call :meth:`enqueue_subgoal` then :meth:`activate_next_subgoal`
        (or a Phase 4 Global-RL equivalent) next.

        code review finding: a mission restarted WITHOUT first letting the
        previous one's subgoal reach a terminal state used to leave that
        stale subgoal ACTIVE (``active_subgoal_mission`` still the OLD
        coordinates, ``stop_required`` still False) -- the new mission's
        local policy could be handed a subgoal belonging to a completely
        different mission. ``SubgoalManager.reset()`` (never just
        ``finish()``, which requires ACTIVE and would raise if the prior
        mission already ended cleanly) unconditionally clears it back to
        pre-activation state, so ``stop_required`` is always True again
        immediately after ``start_mission()``."""
        self._goal_manager.set_goal(final_goal_x, final_goal_y)
        self._subgoal_manager.reset()
        self._mission_start_step = int(now_step)
        self._mission_start_time_sec = float(now_time_sec)
        self._mission_failed = False
        self._mission_timed_out = False
        self._retry_count = 0
        self._current_candidate = None
        self._queue.clear()
        self._distance_history.clear()
        self._consecutive_emergency_stops = 0
        self._last_pose = None
        self._last_subgoal_result = None
        self._subgoal_reached = False
        self._subgoal_failed = False
        self._previous_final_goal_distance = 0.0
        self._previous_subgoal_distance = 0.0
        self._last_final_goal_distance = 0.0
        self._last_subgoal_distance = 0.0
        self._localization_degraded = False

    def enqueue_subgoal(self, x_mission: float, y_mission: float) -> None:
        """Append one candidate subgoal (mission frame) to the pending
        sequence -- the "heuristic 또는 외부 입력" stand-in for a future
        Global RL action (plan section 6.1 goal 3)."""
        self._queue.append((float(x_mission), float(y_mission)))

    @property
    def pending_subgoal_count(self) -> int:
        return len(self._queue)

    @property
    def goal_manager(self) -> GoalManager:
        return self._goal_manager

    @property
    def subgoal_manager(self) -> SubgoalManager:
        return self._subgoal_manager

    @property
    def final_goal_mission(self) -> Tuple[float, float]:
        return self._goal_manager.mission_goal

    @property
    def active_subgoal_mission(self) -> Optional[Tuple[float, float]]:
        if not self._subgoal_manager.active:
            return None
        return self._subgoal_manager.subgoal_mission_xy

    @property
    def previous_final_goal_distance(self) -> float:
        return self._previous_final_goal_distance

    @property
    def previous_subgoal_distance(self) -> float:
        return self._previous_subgoal_distance

    @property
    def subgoal_reached(self) -> bool:
        """True only for the tick a subgoal transitioned to REACHED --
        never True for a final-mission-goal reach reported through
        :attr:`mission_reached` alone unless that same tick also finished
        the active subgoal (which it always does, see
        :meth:`record_local_tick`)."""
        return self._subgoal_reached

    @property
    def subgoal_failed(self) -> bool:
        return self._subgoal_failed

    @property
    def mission_reached(self) -> bool:
        return self._goal_manager.status == MissionStatus.REACHED

    @property
    def mission_failed(self) -> bool:
        return self._mission_failed

    @property
    def mission_timed_out(self) -> bool:
        return self._mission_timed_out

    @property
    def mission_done(self) -> bool:
        return self.mission_reached or self._mission_failed or self._mission_timed_out

    @property
    def stop_required(self) -> bool:
        """True whenever physical motion must NOT happen: no active
        subgoal (before the first one is activated, mid-replan while the
        next one hasn't been chosen yet, or the mission already ended) --
        plan section 6: "replanning 중에는 이전 command가 남지 않도록
        STOP_COMMAND를 명시적으로 반환/publish". Callers must check this
        BEFORE calling the local policy at all, every tick. While
        :attr:`localization_degraded` is True this is ALWAYS True too (no
        subgoal can be active during a degraded-localization pause -- see
        that property)."""
        return self.mission_done or not self._subgoal_manager.active

    @property
    def localization_degraded(self) -> bool:
        """True from the tick a subgoal is cancelled for
        ``"localization_confidence_degraded"`` until a LATER
        :meth:`activate_next_subgoal` call is made with a
        ``localization_confidence`` reading at/above
        ``hierarchy.localization_min_confidence``. While True,
        :meth:`activate_next_subgoal` refuses to activate ANY subgoal
        (queued candidates are left untouched) and :attr:`stop_required`
        is unconditionally True -- code review: "localization degraded
        상태에서 physical motion 금지" must hold even when the queue has
        another candidate ready, not just "a low-confidence pose can't
        confirm success" (already covered by :meth:`record_local_tick`'s
        own confidence gate)."""
        return self._localization_degraded

    @property
    def last_subgoal_result(self) -> Optional[SubgoalResult]:
        return self._last_subgoal_result

    # ---------------------------------------------------------- subgoal activation
    def activate_next_subgoal(
        self, robot_pose_mission: PoseXYYaw, *, now_step: int, now_time_sec: float,
        is_valid: Optional[SubgoalValidityCheck] = None,
        localization_confidence: Optional[float] = None,
    ) -> bool:
        """Pops the next queued candidate and activates it. ``is_valid``
        (e.g. a partial-map occupied/inflated check) is consulted for
        EVERY popped candidate -- an invalid one is discarded (never
        activated) and the next candidate tried instead, so an invalid
        subgoal is never even ACTIVATED and :attr:`stop_required` stays
        True until a valid one is found or the queue empties.

        ``localization_confidence`` is the gate for
        :attr:`localization_degraded` (code review: degraded localization
        must block physical motion, not just block CONFIRMING success --
        see that property's docstring). While degraded, this method
        refuses to activate anything -- not even an already-queued
        candidate -- until called again with a confidence reading at/above
        ``hierarchy.localization_min_confidence``; a caller that never
        recovers is still bounded by the mission timeout, checked here
        too, so a permanently degraded run does not stall forever with
        neither ``mission_reached`` nor ``mission_failed`` ever becoming
        True.

        Returns True iff a subgoal was activated; returns False (and sets
        ``mission_failed``) once the queue is exhausted with nothing valid
        left. A no-op returning False if the mission has already ended."""
        if self.mission_done:
            return False
        if self._check_mission_timeout(now_step, now_time_sec):
            self._mission_timed_out = True
            return False
        min_confidence = self._config.replanning.localization_min_confidence
        if self._localization_degraded:
            if not _confidence_is_healthy(localization_confidence, min_confidence):
                return False
            self._localization_degraded = False
        elif _confidence_is_degraded(localization_confidence, min_confidence):
            self._localization_degraded = True
            return False
        while self._queue:
            x, y = self._queue.popleft()
            if is_valid is not None and not is_valid(x, y):
                continue
            self._current_candidate = (x, y)
            final_goal_distance, _ = self._goal_manager.distance_and_bearing(robot_pose_mission)
            subgoal_distance, _ = goal_distance_and_heading(
                robot_pose_mission.x, robot_pose_mission.y, robot_pose_mission.yaw, x, y,
            )
            self._subgoal_manager.activate(
                x, y, now_step=now_step, now_time_sec=now_time_sec,
                final_goal_distance_m=final_goal_distance, subgoal_distance_m=subgoal_distance,
            )
            self._distance_history.clear()
            self._distance_history.append(subgoal_distance)
            self._consecutive_emergency_stops = 0
            self._last_pose = robot_pose_mission
            self._last_final_goal_distance = final_goal_distance
            self._last_subgoal_distance = subgoal_distance
            # Deliberately does NOT touch _subgoal_reached/_subgoal_failed:
            # activate_next_subgoal() can run INSIDE record_local_tick() (an
            # immediate retry/advance via _apply_recovery) -- resetting
            # those flags here would erase the very outcome THIS tick just
            # reported before the caller ever observes it. Both flags are
            # reset only at the top of the NEXT record_local_tick() call
            # (and by start_mission()).
            return True
        self._mission_failed = True
        return False

    def _retry_current_candidate(
        self, robot_pose_mission: PoseXYYaw, *, now_step: int, now_time_sec: float,
        is_valid: Optional[SubgoalValidityCheck],
    ) -> bool:
        if self._current_candidate is None:
            return False
        self._queue.appendleft(self._current_candidate)
        return self.activate_next_subgoal(robot_pose_mission, now_step=now_step, now_time_sec=now_time_sec, is_valid=is_valid)

    # ---------------------------------------------------------- per-tick update
    def record_local_tick(
        self, robot_pose_mission: PoseXYYaw, *, now_step: int, now_time_sec: float,
        speed_mps: Optional[float] = None, clearance_m: Optional[float] = None,
        predicted_risk: Optional[float] = None, emergency_stop: bool = False,
        steering_saturated: bool = False, newly_explored_cells: int = 0,
        subgoal_endpoint_blocked: bool = False, localization_confidence: Optional[float] = None,
        junction_detected: bool = False, subgoal_is_valid: Optional[SubgoalValidityCheck] = None,
    ) -> None:
        """Advances the coordinator by exactly one local control tick.
        ``robot_pose_mission`` must already be a MISSION-frame pose (the
        caller's localization backend + ``MissionFrame.odom_to_mission``
        composition, never raw odom). Raises if called while
        :attr:`stop_required` is True -- a caller that ticks under that
        condition has a bug in its own control loop, never something this
        method should silently tolerate."""
        if self.stop_required:
            raise RuntimeError(
                "HierarchyCoordinator.record_local_tick() called while stop_required is True "
                "(no active subgoal, or the mission already ended) -- check stop_required and "
                "publish STOP_COMMAND instead of ticking"
            )
        self._subgoal_reached = False
        self._subgoal_failed = False

        step_delta_m = 0.0 if self._last_pose is None else euclidean_distance(
            self._last_pose.x, self._last_pose.y, robot_pose_mission.x, robot_pose_mission.y,
        )
        self._last_pose = robot_pose_mission

        final_goal_distance, _ = self._goal_manager.distance_and_bearing(robot_pose_mission)
        subgoal_distance, _ = self._subgoal_manager.distance_to_subgoal(robot_pose_mission)
        self._previous_final_goal_distance = self._last_final_goal_distance
        self._previous_subgoal_distance = self._last_subgoal_distance
        self._last_final_goal_distance = final_goal_distance
        self._last_subgoal_distance = subgoal_distance

        self._subgoal_manager.record_tick(
            step_delta_m=step_delta_m, final_goal_distance_m=final_goal_distance,
            subgoal_distance_m=subgoal_distance, clearance_m=clearance_m, predicted_risk=predicted_risk,
            emergency_stop=emergency_stop, steering_saturated=steering_saturated,
            newly_explored_cells=newly_explored_cells,
        )
        self._distance_history.append(subgoal_distance)
        self._consecutive_emergency_stops = (self._consecutive_emergency_stops + 1) if emergency_stop else 0

        # code review finding: degraded localization confidence must be
        # checked BEFORE any reached/success determination -- previously
        # the final-goal check ran first, so a pose sampled from a
        # low-confidence (drifting/GPS-denied) localization estimate that
        # happened to land inside tolerance could confirm mission_reached
        # outright. A pose this untrustworthy must never be allowed to
        # confirm EITHER final-goal or subgoal success; it always
        # CANCELS the current subgoal instead (never REACHED, never
        # FAILED_*), independent of where the reported pose is.
        #
        # code review (round 2) finding: this used to call _apply_recovery
        # like every other failure/cancel trigger -- if the queue already
        # had a next candidate, RecoveryAction.ADVANCE_NEXT activated it
        # IMMEDIATELY, in this SAME tick, handing the local policy a fresh
        # active subgoal (stop_required back to False) while localization
        # was STILL reporting degraded confidence. "Never confirm success
        # from a bad pose" is necessary but not sufficient for
        # GPS-denied/drift safety -- "no physical motion while degraded"
        # is the stronger, actually-required guarantee. Deliberately does
        # NOT call _apply_recovery: sets `_localization_degraded` instead,
        # which `activate_next_subgoal` (see its own docstring) refuses to
        # clear until called again with a recovered confidence reading --
        # `stop_required` stays True for as many ticks as it takes.
        if _confidence_is_degraded(
            localization_confidence, self._config.replanning.localization_min_confidence,
        ):
            result = self._subgoal_manager.finish(
                SubgoalStatus.CANCELLED_BY_REPLAN, "localization_confidence_degraded",
                now_step=now_step, now_time_sec=now_time_sec,
            )
            self._last_subgoal_result = result
            self._subgoal_failed = True
            self._localization_degraded = True
            return

        # Final-goal reach takes priority over every subgoal-level outcome:
        # mission success is decided by GoalManager alone, never inferred
        # from a subgoal REACHED (plan section 6.9: "subgoal reached를
        # mission reached로 오인하지 마라").
        if self._goal_manager.check_reached(robot_pose_mission, speed_mps):
            self._last_subgoal_result = self._subgoal_manager.finish(
                SubgoalStatus.REACHED, "final_goal_reached_during_subgoal",
                now_step=now_step, now_time_sec=now_time_sec,
            )
            self._subgoal_reached = True
            return

        if self._check_mission_timeout(now_step, now_time_sec):
            self._last_subgoal_result = self._subgoal_manager.finish(
                SubgoalStatus.CANCELLED_BY_REPLAN, "mission_timeout",
                now_step=now_step, now_time_sec=now_time_sec,
            )
            self._mission_timed_out = True
            return

        reached = self._subgoal_manager.check_reached(robot_pose_mission, speed_mps)
        trigger = evaluate_replanning(
            self._config.replanning, reached=reached, local_steps=self._subgoal_manager.local_steps,
            subgoal_distance_history=tuple(self._distance_history),
            subgoal_endpoint_blocked=subgoal_endpoint_blocked,
            consecutive_emergency_stops=self._consecutive_emergency_stops,
            latest_predicted_risk=predicted_risk, localization_confidence=localization_confidence,
            junction_detected=junction_detected,
        )
        if trigger is None:
            return

        result = self._subgoal_manager.finish(
            trigger.status, trigger.reason, now_step=now_step, now_time_sec=now_time_sec,
        )
        self._last_subgoal_result = result
        if trigger.status == SubgoalStatus.REACHED:
            self._subgoal_reached = True
            return
        self._subgoal_failed = True
        self._apply_recovery(
            result, robot_pose_mission, now_step=now_step, now_time_sec=now_time_sec, is_valid=subgoal_is_valid,
        )

    # ---------------------------------------------------------- executor-level forced termination
    def force_terminate_active_subgoal(
        self, status: SubgoalStatus, reason: str, robot_pose_mission: Optional[PoseXYYaw] = None, *,
        now_step: int, now_time_sec: float, is_valid: Optional[SubgoalValidityCheck] = None,
    ) -> SubgoalResult:
        """Unconditionally finishes the currently ACTIVE subgoal with an
        explicit terminal ``status``/``reason``, for conditions an executor
        detects OUTSIDE any ``record_local_tick`` call -- i.e. conditions
        ``evaluate_replanning`` (which only runs INSIDE ``record_local_tick``)
        can never see on its own: stale/invalid localization or scan before
        a tick is even attempted, a policy-inference timeout, a malformed
        (NaN/Inf/wrong-shape) raw action, or the executor's own
        ``max_local_steps`` control-loop budget exhausting with none of
        ``record_local_tick``'s own triggers having fired yet.

        Every ``run_option()`` implementation MUST call this (never just
        return/break silently) whenever it aborts an ACTIVE subgoal for one
        of these reasons -- code review: an activated subgoal that a caller
        simply walks away from (loop budget exhausted, sensor never
        recovered) previously left ``last_subgoal_result=None`` forever,
        which every downstream consumer (mission-timeout bookkeeping, the
        benchmark's attempt/success counters, Global replay's
        ``failure_reason``) silently mis-happened. This method guarantees a
        real, non-``None`` :class:`SubgoalResult` exists after ANY
        activated subgoal ends, no matter which of the six ways it ends.

        Raises if no subgoal is currently ACTIVE (a caller bug -- there is
        nothing to terminate; check :attr:`stop_required` first, exactly
        like :meth:`record_local_tick`'s own contract).

        ``status=SubgoalStatus.CANCELLED_BY_REPLAN`` with
        ``reason="localization_confidence_degraded"`` is handled exactly
        like the equivalent path inside :meth:`record_local_tick`: it sets
        :attr:`localization_degraded` (so :attr:`stop_required` stays
        sticky True and no recovery/advance is attempted -- GPS-denied
        safety, see that property's docstring) instead of running the
        normal recovery decision. Every other status/reason runs the SAME
        :meth:`_apply_recovery` decision ``record_local_tick``'s own
        replanning-trigger path uses, so an executor-detected timeout is
        treated identically to a coordinator-detected one (retry/advance/
        abort by the SAME :class:`~hunter_kinodynamic_rl.navigation.hierarchy.failure_recovery.FailureRecoveryPolicy`)."""
        if not self._subgoal_manager.active:
            raise RuntimeError(
                "HierarchyCoordinator.force_terminate_active_subgoal() called while no subgoal is ACTIVE"
            )
        result = self._subgoal_manager.finish(status, reason, now_step=now_step, now_time_sec=now_time_sec)
        self._last_subgoal_result = result
        if status == SubgoalStatus.REACHED:
            self._subgoal_reached = True
            return result
        self._subgoal_failed = True
        if status == SubgoalStatus.CANCELLED_BY_REPLAN and reason == "localization_confidence_degraded":
            self._localization_degraded = True
            return result
        pose_for_recovery = robot_pose_mission if robot_pose_mission is not None else self._last_pose
        if pose_for_recovery is not None:
            self._apply_recovery(
                result, pose_for_recovery, now_step=now_step, now_time_sec=now_time_sec, is_valid=is_valid,
            )
        else:
            # No pose was ever recorded for this subgoal (it never
            # completed a single tick -- e.g. localization/scan was already
            # stale on option entry) -- there is nothing valid to re-plan
            # from; abort rather than guess a start-pose/cached-pose
            # fallback (code review: never synthesize a candidate from a
            # fallback pose for a subgoal that never got a real reading).
            self._mission_failed = True
        return result

    def _check_mission_timeout(self, now_step: int, now_time_sec: float) -> bool:
        cfg = self._config
        if cfg.mission_timeout_steps is not None and (now_step - self._mission_start_step) >= cfg.mission_timeout_steps:
            return True
        if cfg.mission_timeout_sec is not None and (now_time_sec - self._mission_start_time_sec) >= cfg.mission_timeout_sec:
            return True
        return False

    def _apply_recovery(
        self, result: SubgoalResult, robot_pose_mission: PoseXYYaw, *, now_step: int, now_time_sec: float,
        is_valid: Optional[SubgoalValidityCheck],
    ) -> None:
        action = self._recovery.decide(result, retry_count=self._retry_count, has_next_candidate=bool(self._queue))
        if action == RecoveryAction.RETRY_SAME:
            self._retry_count += 1
            self._retry_current_candidate(robot_pose_mission, now_step=now_step, now_time_sec=now_time_sec, is_valid=is_valid)
            return
        self._retry_count = 0
        if action == RecoveryAction.ADVANCE_NEXT:
            self.activate_next_subgoal(robot_pose_mission, now_step=now_step, now_time_sec=now_time_sec, is_valid=is_valid)
            return
        self._mission_failed = True
