"""Versioned on-disk field contract for the replay buffer's ``.npz`` save
format (section 39/7: "schema를 확장할 때는 기존 field를 무작정 수정하지
말고 명시적으로 versioning한다").

v1 -> v2: added explicit validity/counterfactual fields (section 5/6):
``valid`` (bool mask -- NEVER rely on NaN alone to signal "no label"),
``unrecoverable``, ``safer_alternative_margin``, and FIXED-width
per-candidate arrays (``candidate_kappa/v_ref/horizon/risk``,
``candidate_valid_mask``) sized to ``max_candidates``. A v1 file CANNOT be
loaded by v2 code -- :meth:`ReplayBuffer.load` raises with a clear message
rather than guessing how to backfill the new fields (section 7: "old schema
의 명확한 오류 또는 migration").

v2 -> v3: added ``actor_candidate_index`` (section P0-6: "candidate actions,
risk targets, validity mask, and actor candidate index" must all be part of
the stored transition). trajectory_sampler.ACTOR_CANDIDATE_INDEX fixes this
at 0 by construction (the actor's own action is always the first generated
candidate) -- storing it per-transition rather than trusting that constant
everywhere makes the invariant auditable from the replay data itself and
survives independently of any future change to candidate ordering.

v3 -> v4: preserves the raw steering-saturation label and actor/candidate
goal-progress values used by progress-preserving counterfactual selection.
The loader has an explicit v3 migration: missing scalar fields become
NaN/False and candidate progress becomes zero, while unknown versions still
fail loudly.
"""

from __future__ import annotations

SCHEMA_VERSION = 4

CORE_FIELDS = ("state", "action", "next_state", "reward", "done")

V2_RISK_FIELDS = (
    "risk_target", "valid", "min_clearance_m", "ttc_sec", "collision_within_horizon",
    "stopping_margin_m", "unrecoverable", "safer_alternative_margin",
)
V2_CANDIDATE_FIELDS = (
    "candidate_kappa", "candidate_v_ref", "candidate_horizon", "candidate_risk", "candidate_valid_mask",
)
V3_CANDIDATE_FIELDS = V2_CANDIDATE_FIELDS + ("actor_candidate_index",)
V3_FIELDS = CORE_FIELDS + V2_RISK_FIELDS + V3_CANDIDATE_FIELDS + ("ptr", "size", "max_candidates")
V4_RISK_FIELDS = V2_RISK_FIELDS + ("steering_saturation", "goal_progress_m")
V4_CANDIDATE_FIELDS = V3_CANDIDATE_FIELDS + ("candidate_goal_progress",)
V4_FIELDS = CORE_FIELDS + V4_RISK_FIELDS + V4_CANDIDATE_FIELDS + ("ptr", "size", "max_candidates")

FIELDS_BY_VERSION = {3: V3_FIELDS, 4: V4_FIELDS}


def fields_for_version(version: int):
    if version not in FIELDS_BY_VERSION:
        raise ValueError(f"unknown replay buffer schema version {version}; known: {sorted(FIELDS_BY_VERSION)}")
    return FIELDS_BY_VERSION[version]
