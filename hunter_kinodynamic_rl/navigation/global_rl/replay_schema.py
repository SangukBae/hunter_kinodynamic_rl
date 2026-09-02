"""Versioned on-disk field contract for :class:`GlobalReplayBuffer` (plan
section 8.8: "replay schema version과 map/channel metadata 저장") -- mirrors
``rl/replay/schema.py``'s own versioning convention for the (separate,
never-shared) LOCAL replay buffer.

v1: initial Phase 4 MVP schema -- one row per terminated Global option
(map/scalar/candidate state before and after, the selected discrete action,
option-level reward, SMDP bookkeeping fields).

v1 -> v2: v1 stored ``map_shape``/``n_scalars``/``n_candidates``/
``n_candidate_features`` (bare integers/shape) but never the actual channel
NAMES/ORDER those dimensions carry (``observation.MAP_CHANNEL_NAMES``/
``SCALAR_NAMES``/``CANDIDATE_FEATURE_NAMES``) -- a file whose shapes happened
to still match after a channel got reordered or renamed (e.g. inserting a
new map channel) would load and sample SILENTLY WRONG data, since nothing
compared the SAVED semantics against the CURRENT code's. v2 stores those
three name tuples and :meth:`GlobalReplayBuffer.load` rejects a mismatch
outright -- v1 has no migration (never shipped against a real training run,
only this package's own tests) and is rejected the same explicit way an
unknown future version would be.

v2 -> v3 (Phase 5, plan section 9): the channel/scalar/candidate-feature
name tuples are no longer FIXED module constants -- Phase 5's ablation
flags (``include_failure_channel``/``topology_feedback_enabled``/
``feasibility_feedback_enabled``/``global_risk_feedback_enabled``) make
them a function of the profile's ``GlobalRLConfig``
(``observation.resolve_map_channel_names``/``resolve_candidate_feature_names``).
v3 additionally stores the topological-memory node tensor
(``node_tensor``/``node_validity_mask``, empty ``(0, N_NODE_FEATURES)``/
``(0,)`` when topology feedback is disabled) alongside every transition.
v2 has no migration (same reasoning as v1's own retirement) and is rejected
the same explicit way.
"""

from __future__ import annotations

from hunter_kinodynamic_rl.navigation.hierarchy.subgoal_manager import SubgoalStatus

SCHEMA_VERSION = 3

CORE_FIELDS = (
    "map_state", "scalar_state", "candidate_features", "action_mask", "action", "option_reward",
    "next_map_state", "next_scalar_state", "next_candidate_features", "next_action_mask",
    "mission_done", "subgoal_success", "failure_reason", "local_steps", "exploration_gain", "risk_integral",
    "node_tensor", "node_validity_mask", "next_node_tensor", "next_node_validity_mask",
)

# failure_reason is stored as a small int code (never a variable-length
# string field) -- None ("no subgoal terminated this option") maps to -1;
# every SubgoalStatus member gets a stable index via enumeration order
# (never renumbered across a code change, since SubgoalStatus's own member
# order is itself a stable part of that module's contract).
FAILURE_REASON_CODES = {None: -1}
FAILURE_REASON_CODES.update({status: i for i, status in enumerate(SubgoalStatus)})
FAILURE_REASON_NAMES = {code: status for status, code in FAILURE_REASON_CODES.items()}


def encode_failure_reason(status) -> int:
    return FAILURE_REASON_CODES[status]


def decode_failure_reason(code: int):
    return FAILURE_REASON_NAMES[int(code)]
