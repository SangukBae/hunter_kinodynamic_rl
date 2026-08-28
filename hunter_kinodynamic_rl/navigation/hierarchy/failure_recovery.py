"""Heuristic subgoal-sequence recovery (plan section 6, completion criterion
"subgoal 실패 후 다른 subgoal로 recovery/replan할 수 있다").

Given a just-terminated (non-REACHED) subgoal's
:class:`~hunter_kinodynamic_rl.navigation.hierarchy.subgoal_manager.SubgoalResult`,
:class:`FailureRecoveryPolicy` decides whether the coordinator should RETRY
the same subgoal, ADVANCE to the next candidate in its sequence, or ABORT
the mission -- with no Global RL involved; a fixed or externally-supplied
subgoal sequence stands in for it until Phase 4.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import FrozenSet

from hunter_kinodynamic_rl.navigation.hierarchy.subgoal_manager import SubgoalResult, SubgoalStatus


class RecoveryAction(enum.Enum):
    RETRY_SAME = "retry_same"
    ADVANCE_NEXT = "advance_next"
    ABORT_MISSION = "abort_mission"


def _default_retryable() -> FrozenSet[SubgoalStatus]:
    # FAILED_BLOCKED and FAILED_HIGH_RISK are NOT retried by default -- the
    # same subgoal endpoint is still occupied/inflated or still high-risk
    # immediately after a failed attempt, so retrying it is expected to
    # fail identically. FAILED_TIMEOUT/FAILED_NO_PROGRESS are transient
    # local-controller outcomes worth one more attempt before giving up on
    # that specific subgoal.
    return frozenset({SubgoalStatus.FAILED_TIMEOUT, SubgoalStatus.FAILED_NO_PROGRESS})


@dataclass(frozen=True)
class FailureRecoveryConfig:
    max_retries_per_subgoal: int = 1
    retryable_statuses: FrozenSet[SubgoalStatus] = field(default_factory=_default_retryable)

    def validate(self) -> None:
        if self.max_retries_per_subgoal < 0:
            raise ValueError("FailureRecoveryConfig.max_retries_per_subgoal must be >= 0")


class FailureRecoveryPolicy:
    def __init__(self, config: FailureRecoveryConfig) -> None:
        config.validate()
        self._config = config

    def decide(self, result: SubgoalResult, *, retry_count: int, has_next_candidate: bool) -> RecoveryAction:
        """``retry_count`` is how many times THIS SAME subgoal has already
        been retried (0 on its first failure). ``has_next_candidate`` tells
        this policy whether the coordinator's subgoal source has another
        candidate queued -- CANCELLED_BY_REPLAN (a system-initiated abort,
        e.g. localization confidence degraded) is never retried, it always
        either advances or aborts, since retrying the identical subgoal
        under the same degraded condition is not expected to help."""
        if result.status == SubgoalStatus.REACHED:
            raise ValueError("FailureRecoveryPolicy.decide() must only be called for a non-REACHED outcome")
        if (result.status in self._config.retryable_statuses
                and retry_count < self._config.max_retries_per_subgoal):
            return RecoveryAction.RETRY_SAME
        if has_next_candidate:
            return RecoveryAction.ADVANCE_NEXT
        return RecoveryAction.ABORT_MISSION
