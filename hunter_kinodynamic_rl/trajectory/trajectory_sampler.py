"""Generate a set of ALTERNATIVE [kappa, v_ref, L] candidates around a base
trajectory command -- the raw material risk/counterfactual_sampler.py scores.

Kept separate from the risk module: this file only knows about physically
valid Ackermann candidates (section 22 -- "steering limit 초과, negative
horizon, invalid speed, NaN을 생성하지 않는다"), never about obstacles/risk.

Candidate ORDERING (fixed, deterministic -- code review found the previous
"append stop last, then truncate to num_candidates" order silently DROPPED
the stop candidate whenever ``num_candidates`` was smaller than the full
steering x speed grid, which is exactly the case in both production
counterfactual profiles):

    index 0            : the actor's own action (ALWAYS present)
    index 1            : the stop candidate (if include_stop_candidate;
                          reserved BEFORE any steering/speed alternative, so
                          truncation can never remove it)
    next indices       : short/long L alternatives from horizon_fractions,
                          clamped to the configured action-space L bounds
    remaining indices  : steering-offset PAIRS (grouped by matching
                          +/-offset magnitude, most extreme pair first --
                          e.g. +-1.0 before +-0.5, so the most informative
                          "hard left / hard right" escape options survive
                          truncation ahead of milder ones) x speed_fractions
                          (in configured order), left-then-right within
                          each pair/speed combination. An offset with no
                          matching sign in ``kappa_offsets_frac`` is treated
                          as an unpaired singleton at its own priority slot.

Exact duplicates that arise after clamping to the robot's physical bounds
(e.g. an offset that clamps to the SAME kappa as the actor's own
near-steering-limit action, or multiple speed_fractions collapsing to the
same v_ref when base.v_ref=0) are removed BEFORE truncation, keeping only
the first (highest-priority) occurrence -- so truncation never wastes a
slot on a candidate that duplicates one already present.
"""

from __future__ import annotations

import math
from typing import List

from hunter_kinodynamic_rl.config.schema import ActionSpaceConfig, CounterfactualConfig, RobotConfig
from hunter_kinodynamic_rl.robot.limits import RobotLimits
from hunter_kinodynamic_rl.trajectory.action_space import TrajectoryCommand

ACTOR_CANDIDATE_INDEX = 0


def _offset_pairs(offsets: List[float]) -> List[List[float]]:
    """Group signed offsets into symmetric (-x, +x) pairs when both are
    configured, else singletons -- ordered by DESCENDING |offset| (most
    extreme steering alternatives first)."""
    remaining = set(offsets)
    seen = set()
    pairs: List[List[float]] = []
    for x in sorted(remaining, key=lambda v: -abs(v)):
        if x in seen:
            continue
        neg = -x
        if neg != x and neg in remaining and neg not in seen:
            pairs.append(sorted([x, neg]))  # negative (right) first, then positive (left)
            seen.add(x)
            seen.add(neg)
        else:
            pairs.append([x])
            seen.add(x)
    return pairs


def generate_candidates(
    base: TrajectoryCommand, robot: RobotConfig, cf_cfg: CounterfactualConfig,
    action_cfg: ActionSpaceConfig | None = None,
) -> List[TrajectoryCommand]:
    limits = RobotLimits(robot)
    action_cfg = action_cfg or ActionSpaceConfig()
    kappa_max = robot.max_curvature
    ordered: List[TrajectoryCommand] = [base]  # index 0: actor, ALWAYS present

    if cf_cfg.include_stop_candidate:
        ordered.append(TrajectoryCommand(kappa=base.kappa, v_ref=0.0, horizon_m=base.horizon_m))

    # L is a first-class policy action: counterfactual search must be able
    # to ask whether committing to the same curvature/speed for a shorter
    # or longer distance preserves progress while reducing future risk.
    for horizon_frac in cf_cfg.horizon_fractions:
        horizon_m = min(
            action_cfg.horizon_length_max_m,
            max(action_cfg.horizon_length_min_m, base.horizon_m * horizon_frac),
        )
        ordered.append(TrajectoryCommand(kappa=base.kappa, v_ref=base.v_ref, horizon_m=horizon_m))

    for pair in _offset_pairs(cf_cfg.kappa_offsets_frac):
        for speed_frac in cf_cfg.speed_fractions:
            v_ref = limits.clamp_speed(base.v_ref * speed_frac)
            for offset_frac in pair:
                kappa = limits.clamp_curvature(base.kappa + offset_frac * kappa_max)
                ordered.append(TrajectoryCommand(kappa=kappa, v_ref=v_ref, horizon_m=base.horizon_m))

    # Dedupe (keep first/highest-priority occurrence) BEFORE truncating, so
    # a clamped-to-duplicate candidate never silently displaces a genuinely
    # distinct lower-priority one.
    deduped: List[TrajectoryCommand] = []
    seen_keys = set()
    for c in ordered:
        key = (round(c.kappa, 9), round(c.v_ref, 9), round(c.horizon_m, 9))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(c)

    candidates = deduped[: cf_cfg.num_candidates]

    for c in candidates:
        if not (math.isfinite(c.kappa) and math.isfinite(c.v_ref) and math.isfinite(c.horizon_m)):
            raise ValueError(f"generated a non-finite candidate: {c}")
        if abs(c.kappa) > kappa_max + 1e-9:
            raise ValueError(f"candidate kappa {c.kappa} exceeds robot.max_curvature {kappa_max}")
        if c.horizon_m <= 0.0:
            raise ValueError(f"candidate horizon_m must be > 0, got {c.horizon_m}")
        if not (action_cfg.horizon_length_min_m - 1e-9 <= c.horizon_m
                <= action_cfg.horizon_length_max_m + 1e-9):
            raise ValueError(
                f"candidate horizon_m {c.horizon_m} out of configured bounds "
                f"[{action_cfg.horizon_length_min_m}, {action_cfg.horizon_length_max_m}]"
            )
        if not (0.0 <= c.v_ref <= robot.max_forward_speed_mps + 1e-9):
            raise ValueError(f"candidate v_ref {c.v_ref} out of [0, {robot.max_forward_speed_mps}]")

    return candidates
