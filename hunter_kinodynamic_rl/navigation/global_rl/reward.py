"""Option-level Global reward (plan section 8.7/14) -- computed exactly ONCE
per terminated Global subgoal option, from explicit inputs only (no ROS, no
map/hierarchy object reached into directly) so every component is
independently unit-testable.

::

    R_global = R_goal + R_final_progress + R_exploration
             - R_repeated_revisit - R_repeated_deadend - R_local_failure
             - R_risk - R_elapsed - R_predicted_risk

``R_predicted_risk`` (Phase 5, plan section 9.6/13's "global risk feedback")
is priced at DECISION time from the SELECTED candidate's own predicted risk
(``hierarchy.feasibility``'s ``predicted_action_risk`` feature) -- distinct
from ``R_risk``, which prices the REALIZED risk integral actually
experienced while the option ran. ``config.predicted_risk_penalty_scale``
defaults to ``0.0``, so a caller that never sets ``global_rl.global_risk_feedback_enabled``
gets ``R_predicted_risk == 0.0`` unconditionally -- this term is additive-
only and never changes Phase 4's reward value for any existing profile.

Section 8.7/15's two hard requirements this module enforces structurally:

- **Exploration never dominates goal/progress**: ``R_exploration`` is
  clipped to ``config.exploration_reward_clip`` BEFORE being added, so no
  amount of newly-explored area can outweigh a bounded ``R_goal``/
  ``R_final_progress``.
- **First dead-end vs. repeated dead-end**: this module never DETECTS a
  dead-end itself (that is a Phase 5 topological-memory concern, out of
  this phase's scope) -- it only prices whatever the caller already
  classified via ``repeated_deadend`` (True only for a REPEAT entry into a
  branch already recorded as failed). A caller that never sets it True for
  a branch's first visit gets zero ``R_repeated_deadend`` penalty on that
  first visit, by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from hunter_kinodynamic_rl.config.schema import GlobalRLConfig
from hunter_kinodynamic_rl.navigation.hierarchy.subgoal_manager import SubgoalStatus

_LOCAL_FAILURE_PENALTY_FIELD = {
    SubgoalStatus.FAILED_TIMEOUT: "local_failure_penalty_timeout",
    SubgoalStatus.FAILED_NO_PROGRESS: "local_failure_penalty_no_progress",
    SubgoalStatus.FAILED_BLOCKED: "local_failure_penalty_blocked",
    SubgoalStatus.FAILED_HIGH_RISK: "local_failure_penalty_high_risk",
    SubgoalStatus.CANCELLED_BY_REPLAN: "local_failure_penalty_cancelled_by_replan",
}


@dataclass(frozen=True)
class GlobalRewardBreakdown:
    total: float
    goal: float
    final_progress: float
    exploration: float
    repeated_revisit: float
    repeated_deadend: float
    local_failure: float
    risk: float
    elapsed: float
    predicted_risk: float


def local_failure_penalty(config: GlobalRLConfig, subgoal_status: Optional[SubgoalStatus]) -> float:
    """``0.0`` for ``REACHED``, ``None`` (no subgoal terminated this option --
    e.g. the very first Global decision of a mission), or any status not in
    the failure-penalty table; the configured penalty otherwise."""
    field_name = _LOCAL_FAILURE_PENALTY_FIELD.get(subgoal_status)
    return getattr(config, field_name) if field_name is not None else 0.0


def compute_global_reward(
    config: GlobalRLConfig, *,
    mission_reached: bool,
    start_final_goal_distance_m: float,
    end_final_goal_distance_m: float,
    newly_explored_cells: int,
    revisit_amount: float = 0.0,
    repeated_deadend: bool = False,
    subgoal_status: Optional[SubgoalStatus] = None,
    risk_integral: float = 0.0,
    local_steps: int = 0,
    predicted_risk_at_selection: float = 0.0,
) -> GlobalRewardBreakdown:
    goal = config.goal_reward if mission_reached else 0.0
    final_progress = config.progress_reward_scale * (start_final_goal_distance_m - end_final_goal_distance_m)
    exploration_raw = config.exploration_reward_scale * max(0.0, float(newly_explored_cells))
    exploration = min(exploration_raw, config.exploration_reward_clip)
    repeated_revisit = config.revisit_penalty_scale * max(0.0, float(revisit_amount))
    repeated_deadend_penalty = config.repeated_deadend_penalty if repeated_deadend else 0.0
    local_failure = local_failure_penalty(config, subgoal_status)
    risk = config.risk_penalty_scale * max(0.0, float(risk_integral))
    elapsed = config.elapsed_penalty_per_local_step * max(0, int(local_steps))
    predicted_risk = config.predicted_risk_penalty_scale * max(0.0, float(predicted_risk_at_selection))

    total = goal + final_progress + exploration - repeated_revisit - repeated_deadend_penalty - local_failure \
        - risk - elapsed - predicted_risk
    return GlobalRewardBreakdown(
        total=total, goal=goal, final_progress=final_progress, exploration=exploration,
        repeated_revisit=repeated_revisit, repeated_deadend=repeated_deadend_penalty,
        local_failure=local_failure, risk=risk, elapsed=elapsed, predicted_risk=predicted_risk,
    )
