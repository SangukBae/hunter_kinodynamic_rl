"""Minimal reward function (section 24) -- goal / collision / progress /
time / smoothness. Deliberately simple: the research contribution lives in
the risk critic / counterfactual objective, not in reward shaping, and
collision avoidance is NOT asked to carry that burden alone via a single
penalty term (section 24: "collision avoidance를 reward penalty 하나에 전부
맡기지 않는다" -- the risk-aware actor penalty is the primary avoidance
signal; this collision_penalty is only the terminal outcome cost).
"""

from __future__ import annotations

from dataclasses import dataclass

from hunter_kinodynamic_rl.config.schema import RewardConfig


@dataclass(frozen=True)
class RewardBreakdown:
    goal: float
    collision: float
    progress: float
    step: float
    control_smoothness: float
    trajectory_smoothness: float

    @property
    def total(self) -> float:
        return (self.goal + self.collision + self.progress + self.step
                + self.control_smoothness + self.trajectory_smoothness)


def compute_reward(
    cfg: RewardConfig,
    goal_distance_m: float,
    previous_goal_distance_m: float,
    collided: bool,
    reached_goal: bool,
    action, previous_action,
    trajectory_kappa=None,
    trajectory_horizon_m=None,
) -> RewardBreakdown:
    goal_term = cfg.goal_reached_reward if reached_goal else 0.0
    collision_term = cfg.collision_penalty if collided else 0.0
    progress_term = 0.0
    if not reached_goal and not collided:
        progress_term = cfg.progress_weight * (previous_goal_distance_m - goal_distance_m)
    step_term = 0.0 if (reached_goal or collided) else cfg.step_penalty

    control_smoothness_term = 0.0
    if cfg.control_smoothness_weight > 0.0 and previous_action is not None:
        delta = sum((float(a) - float(b)) ** 2 for a, b in zip(action, previous_action))
        control_smoothness_term = -cfg.control_smoothness_weight * delta

    trajectory_smoothness_term = 0.0
    if cfg.trajectory_smoothness_weight > 0.0 and trajectory_kappa is not None:
        # Bending-energy proxy integral(kappa(s)^2 ds).  For the current
        # constant-curvature primitive this is exactly kappa^2 * L, remains
        # in physical units, and naturally generalizes to sampled
        # multi-segment primitives later.
        length_m = max(0.0, float(trajectory_horizon_m or 0.0))
        trajectory_smoothness_term = (
            -cfg.trajectory_smoothness_weight * float(trajectory_kappa) ** 2 * length_m
        )

    return RewardBreakdown(
        goal=goal_term, collision=collision_term, progress=progress_term,
        step=step_term, control_smoothness=control_smoothness_term,
        trajectory_smoothness=trajectory_smoothness_term,
    )


def is_goal_reached(goal_distance_m: float, cfg: RewardConfig) -> bool:
    return goal_distance_m <= cfg.goal_threshold_m
