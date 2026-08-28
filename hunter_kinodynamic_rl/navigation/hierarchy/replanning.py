"""Pure evaluation of the Phase 2 replanning triggers (plan section 6.5).

:func:`evaluate_replanning` decides WHETHER the currently active subgoal
should terminate THIS tick, and with which terminal
:class:`~hunter_kinodynamic_rl.navigation.hierarchy.subgoal_manager.SubgoalStatus`
and reason code -- from explicit inputs only. It never reaches into
``SubgoalManager``'s own running-stats accumulator or touches ROS; the
coordinator is responsible for supplying whatever summary value each check
needs, so every condition here stays a single, obvious, independently
testable comparison.

The safety-critical localization-confidence check runs before any success
condition: a low/non-finite confidence reading must never confirm a reached
state. After that, checks run in the order listed in the plan; the FIRST one
that fires wins (e.g. a tick that is simultaneously "reached" and "blocked"
is reported as reached -- success takes priority over a stale blocked reading
computed against the pose the robot is now leaving).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence

from hunter_kinodynamic_rl.navigation.hierarchy.subgoal_manager import SubgoalStatus


@dataclass(frozen=True)
class ReplanningConfig:
    local_option_timeout_steps: int = 200
    no_progress_window_steps: int = 40
    no_progress_min_delta_m: float = 0.1
    consecutive_emergency_stop_limit: int = 5
    local_risk_threshold: float = 0.8
    localization_min_confidence: float = 0.5

    def validate(self) -> None:
        if self.local_option_timeout_steps <= 0:
            raise ValueError("ReplanningConfig.local_option_timeout_steps must be > 0")
        if self.no_progress_window_steps <= 0:
            raise ValueError("ReplanningConfig.no_progress_window_steps must be > 0")
        if self.no_progress_min_delta_m < 0.0:
            raise ValueError("ReplanningConfig.no_progress_min_delta_m must be >= 0")
        if self.consecutive_emergency_stop_limit <= 0:
            raise ValueError("ReplanningConfig.consecutive_emergency_stop_limit must be > 0")
        if self.local_risk_threshold <= 0.0:
            raise ValueError("ReplanningConfig.local_risk_threshold must be > 0")
        if not (0.0 <= self.localization_min_confidence <= 1.0):
            raise ValueError("ReplanningConfig.localization_min_confidence must be in [0, 1]")


@dataclass(frozen=True)
class ReplanTrigger:
    status: SubgoalStatus
    reason: str


def _confidence_is_degraded(confidence: Optional[float], min_confidence: float) -> bool:
    if confidence is None:
        return False
    try:
        value = float(confidence)
    except (TypeError, ValueError):
        return True
    return (not math.isfinite(value)) or value < min_confidence


def junction_detected_stub(*_args, **_kwargs) -> bool:
    """Phase 5 stub (plan section 6.5: "새 junction 이벤트는 Phase 5 전까지
    optional/stub로만 둔다"). Always returns False -- a named hook a future
    topological-memory implementation can replace without touching
    :func:`evaluate_replanning`'s call sites or signature."""
    return False


def evaluate_replanning(
    config: ReplanningConfig, *,
    reached: bool,
    local_steps: int,
    subgoal_distance_history: Sequence[float],
    subgoal_endpoint_blocked: bool = False,
    consecutive_emergency_stops: int = 0,
    latest_predicted_risk: Optional[float] = None,
    localization_confidence: Optional[float] = None,
    junction_detected: bool = False,
) -> Optional[ReplanTrigger]:
    """``subgoal_distance_history`` is the active subgoal's distance-to-go
    sampled once per tick, oldest first (most recent last) -- the "no
    progress for a configured time" check looks at the last
    ``no_progress_window_steps`` samples and requires the distance to have
    decreased by at least ``no_progress_min_delta_m`` across that window;
    fewer samples than the window size means "not enough history yet",
    never a spurious trigger.

    Returns ``None`` when nothing should terminate the subgoal this tick.
    """
    if _confidence_is_degraded(localization_confidence, config.localization_min_confidence):
        return ReplanTrigger(SubgoalStatus.CANCELLED_BY_REPLAN, "localization_confidence_degraded")
    if reached:
        return ReplanTrigger(SubgoalStatus.REACHED, "subgoal_tolerance_reached")
    if local_steps >= config.local_option_timeout_steps:
        return ReplanTrigger(SubgoalStatus.FAILED_TIMEOUT, "local_option_timeout")
    if len(subgoal_distance_history) >= config.no_progress_window_steps:
        window = subgoal_distance_history[-config.no_progress_window_steps:]
        if (window[0] - window[-1]) < config.no_progress_min_delta_m:
            return ReplanTrigger(SubgoalStatus.FAILED_NO_PROGRESS, "no_progress_within_window")
    if subgoal_endpoint_blocked:
        return ReplanTrigger(SubgoalStatus.FAILED_BLOCKED, "subgoal_endpoint_occupied_or_inflated")
    if consecutive_emergency_stops >= config.consecutive_emergency_stop_limit:
        return ReplanTrigger(SubgoalStatus.FAILED_BLOCKED, "consecutive_emergency_stops")
    if latest_predicted_risk is not None and latest_predicted_risk > config.local_risk_threshold:
        return ReplanTrigger(SubgoalStatus.FAILED_HIGH_RISK, "local_risk_threshold_exceeded")
    if junction_detected:
        return ReplanTrigger(SubgoalStatus.CANCELLED_BY_REPLAN, "new_junction_detected")
    return None
