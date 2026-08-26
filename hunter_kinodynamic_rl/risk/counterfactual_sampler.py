"""Counterfactual risk: roll out and score several ALTERNATIVE candidate
trajectories around the actor's chosen action, so training can ask "would a
different feasible action have been meaningfully safer?" (section 21).

This is training-time-only tooling (it needs :class:`DynamicObstacle`
privileged ground truth, see risk/labels.py's module docstring) -- never
called from the real-robot inference path.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Optional, Sequence, Tuple

from hunter_kinodynamic_rl.config.schema import (
    ActionSpaceConfig, CounterfactualConfig, DynamicsConfig, RiskConfig, RobotConfig,
)
from hunter_kinodynamic_rl.risk.boundary import RobotPose
from hunter_kinodynamic_rl.risk.labels import DynamicObstacle, RiskLabel
from hunter_kinodynamic_rl.risk.trajectory_risk import assess_trajectory_command_with_rollout
from hunter_kinodynamic_rl.robot.interface import VehicleState
from hunter_kinodynamic_rl.trajectory.action_space import TrajectoryCommand
from hunter_kinodynamic_rl.trajectory.trajectory_sampler import generate_candidates


@dataclass(frozen=True)
class ScoredCandidate:
    command: TrajectoryCommand
    risk: RiskLabel
    goal_progress_m: float = 0.0


def _goal_progress_m(initial_state: VehicleState, final_state: VehicleState,
                     goal_local_xy: Optional[Tuple[float, float]]) -> float:
    if goal_local_xy is None:
        # Standalone/tests without a goal still get a meaningful
        # stop-vs-moving discriminator: displacement along the rollout.
        return math.hypot(final_state.x - initial_state.x, final_state.y - initial_state.y)
    gx, gy = goal_local_xy
    before = math.hypot(gx - initial_state.x, gy - initial_state.y)
    after = math.hypot(gx - final_state.x, gy - final_state.y)
    return before - after


def score_candidates(
    base: TrajectoryCommand, initial_state: VehicleState, ego_radius: float,
    obstacles: Sequence[DynamicObstacle], robot: RobotConfig,
    dynamics_cfg: DynamicsConfig, risk_cfg: RiskConfig, cf_cfg: CounterfactualConfig,
    robot_world_pose: Optional[RobotPose] = None, world_half_extent_m: Optional[float] = None,
    commit_blend: bool = False,
    goal_local_xy: Optional[Tuple[float, float]] = None,
    action_cfg: Optional[ActionSpaceConfig] = None,
) -> List[ScoredCandidate]:
    """``commit_blend`` (code review: counterfactual candidate risk must
    follow the SAME nominal-execution semantics as the actor's own action)
    is forwarded, UNCHANGED, to every candidate's
    ``assess_trajectory_command`` call -- every candidate here shares the
    SAME ``initial_state`` (the ego's one actual current pose/steering
    right now), so they all correctly start their commit-blend rollout
    from the SAME real current steering; only each candidate's OWN
    kappa/v_ref/horizon (hence its OWN target steering) differs."""
    candidates = generate_candidates(base, robot, cf_cfg, action_cfg=action_cfg)
    scored = []
    for command in candidates:
        risk, rollout = assess_trajectory_command_with_rollout(
            command.kappa, command.v_ref, command.horizon_m, initial_state,
            ego_radius, obstacles, robot, dynamics_cfg, risk_cfg,
            robot_world_pose=robot_world_pose, world_half_extent_m=world_half_extent_m,
            commit_blend=commit_blend,
        )
        scored.append(ScoredCandidate(
            command=command, risk=risk,
            goal_progress_m=_goal_progress_m(initial_state, rollout.final_state, goal_local_xy),
        ))
    return scored


def progress_preserving_candidates(
    scored: Sequence[ScoredCandidate], cf_cfg: CounterfactualConfig,
) -> List[ScoredCandidate]:
    """Candidates whose goal progress is sufficiently close to the actor.

    The actor itself is always retained.  For positive actor progress, a
    candidate must satisfy both the relative and absolute progress limits;
    for a non-progressing actor, an alternative only needs to do no worse.
    """
    if not scored:
        return []
    actor_progress = scored[0].goal_progress_m
    if actor_progress > 0.0:
        threshold = max(
            actor_progress * cf_cfg.min_progress_ratio,
            actor_progress - cf_cfg.max_progress_loss_m,
        )
    else:
        threshold = actor_progress
    return [scored[0]] + [c for c in scored[1:] if c.goal_progress_m >= threshold]


def best_candidate(
    scored: Sequence[ScoredCandidate], cf_cfg: Optional[CounterfactualConfig] = None,
) -> ScoredCandidate:
    if not scored:
        raise ValueError("best_candidate requires at least one candidate")
    eligible = progress_preserving_candidates(
        scored, cf_cfg or CounterfactualConfig(enabled=True),
    )
    return min(eligible, key=lambda c: c.risk.risk_score)


def safer_alternative_margin(
    scored: Sequence[ScoredCandidate], cf_cfg: Optional[CounterfactualConfig] = None,
) -> float:
    """actor_risk - best_alternative_risk. Positive means a safer feasible
    alternative existed (the quantity the counterfactual risk penalty / loss
    is built from); 0 or negative means the actor already chose the safest
    (or joint-safest) option among the sampled candidates."""
    if not scored:
        return 0.0
    actor_risk = scored[0].risk.risk_score  # index 0 is always the actor's own action (see trajectory_sampler)
    best = best_candidate(scored, cf_cfg).risk.risk_score
    return actor_risk - best
