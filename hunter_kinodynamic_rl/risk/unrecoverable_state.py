"""Approximate Unrecoverable-State check (section 23): given Hunter SE's
physical limits and the counterfactual candidate set, is there NO feasible
trajectory that avoids collision within the horizon?

This is a NECESSARY-condition approximation, not a full reachability
analysis (section 23 explicitly allows starting here): it only certifies
"unrecoverable" relative to the SAMPLED candidate set, so a denser sampler
(more kappa/speed offsets) can only ever make this MORE conservative
(report unrecoverable less often), never less safe. Swap in a proper
reachability check later behind the same function signature.
"""

from __future__ import annotations

from typing import Sequence

from hunter_kinodynamic_rl.risk.counterfactual_sampler import ScoredCandidate


def is_unrecoverable(scored: Sequence[ScoredCandidate]) -> bool:
    if not scored:
        return False
    return all(c.risk.collision_within_horizon for c in scored)
