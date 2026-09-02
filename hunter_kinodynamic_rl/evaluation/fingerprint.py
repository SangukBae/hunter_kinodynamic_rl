"""Config/benchmark fingerprinting (section item-2) -- distinguishes TWO
DIFFERENT questions that "compare profile names" alone cannot answer:

1. **architecture fingerprint**: does the LIVE environment_node's own
   resolved config actually match what a CHECKPOINT was trained under?
   Everything that determines the checkpoint's network shapes and the
   live environment's action-decode/observation-build semantics --
   ``action_space`` (mode + bounds), ``features``, ``observation`` (schema),
   ``robot`` (geometry), ``dynamics``, ``risk``/``counterfactual``
   (architecture-relevant fields the agent's own constructor consumes),
   ``hyperparameters``/``sac_hyperparameters``, ``algorithm``. A profile
   NAME match does not prove this -- the same-named profile's YAML file on
   disk could have been edited between when a checkpoint trained and when
   it is evaluated (section item-2: "동일 profile 이름의 YAML 내용을 변경한
   경우 fail-fast").
2. **evaluation-contract fingerprint**: are two DIFFERENT checkpoints (e.g.
   a SAC baseline and a risk-aware TQC) being benchmarked under IDENTICAL
   CONDITIONS? ``evaluation`` (benchmark/episode budget/common-metrics
   yardstick), ``reward`` (termination), ``scenario`` (world size) --
   exactly the sections ``nodes/evaluation_node.py::build_effective_profile``
   layers from the REQUESTED eval profile onto every checkpoint's own
   restored architecture, so this fingerprint is *expected* to be IDENTICAL
   across every model evaluated with the same ``--profile``.

Both are SHA-256 hex digests of a canonical (sorted-keys, compact
separators) JSON serialization of the relevant ``Profile`` section dicts --
deterministic and directly embeddable in ``summary.json``/checkpoint
manifests.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Dict, List, Optional

from hunter_kinodynamic_rl.config.schema import Profile

# section item-2: everything that determines a CHECKPOINT's own network
# shape / the live environment's action-decode & observation-build
# semantics. Never includes `evaluation`/`reward`/`scenario`/`runtime`
# (those are the evaluation-CONTRACT's job, see EVALUATION_CONTRACT_SECTIONS
# below) nor `training`/`domain_randomization` (infra/training-loop
# parameters with no effect on what a saved checkpoint's weights MEAN, and
# not part of the evaluation contract either -- `training.seed`/`max_timesteps`
# etc. are training-loop bookkeeping, never applied to a live evaluation
# environment at all).
ARCHITECTURE_SECTIONS = (
    "action_space", "features", "observation", "robot", "dynamics",
    "risk", "counterfactual", "hyperparameters", "sac_hyperparameters", "algorithm",
)

# section item-2 (round 2): the conditions every model benchmarked through
# the same requested `--profile` must share. Mirrors exactly what
# `evaluation_node.py::build_effective_profile` overrides from the
# REQUESTED profile (never the checkpoint's own resolved_config) --
# comparing this across two summary.json files proves (not just asserts)
# they were evaluated under identical terms. `runtime` was ADDED in the
# round-2 fairness pass: `time_delta_sec`/`deterministic_stepping`/
# `gazebo_max_step_size_sec` etc. materially change episode dynamics
# (physics step size, whether stepping is deterministic) just as much as
# `reward`/`scenario` do -- a checkpoint's own training-time runtime
# section must NEVER silently leak into evaluation, exactly like
# world_size_m or the reward function must not.
#
# `sensor_noise` (round 3, requirement 2): a checkpoint's own TRAINING-time
# sensor_noise section must equally never silently leak into evaluation --
# without this, two checkpoints trained under different sensor_noise
# settings would each keep their own training-time noise model active
# during a benchmark that never asked for it, which is exactly as unfair as
# leaking a different reward/scenario/runtime section would be. This is
# UNRELATED to (and must never be confused with) `env/randomization/
# domain_randomizer.py`'s own per-episode-randomized TRAIN-ONLY noise (that
# system's `sensor:` per-scenario overrides are explicitly REJECTED for
# fixed benchmarks by `check_sensor_overrides_supported` -- evaluation
# noise, if any, is controlled ONLY by the requested evaluation profile's
# own `sensor_noise` section, delivered here).
EVALUATION_CONTRACT_SECTIONS = ("evaluation", "reward", "scenario", "runtime", "sensor_noise")

# Phase 5 (plan section 9.11: "ablation checkpoint와 결과가 서로 다른
# architecture fingerprint로 구분된다") -- everything that determines a
# GLOBAL checkpoint's own network shape and observation/candidate-feature
# semantics. Deliberately a SEPARATE tuple/function from
# ARCHITECTURE_SECTIONS above, never appended to it: ARCHITECTURE_SECTIONS
# is the LOCAL (kinodynamic TQC) checkpoint's own fingerprint contract --
# folding global_rl/hierarchy/mapping/memory/feasibility into it would
# silently change what every EXISTING local-only fingerprint means and
# break comparability against anything already recorded under it.
HIERARCHICAL_ARCHITECTURE_SECTIONS = (
    "global_rl", "hierarchical_training", "hierarchy", "mapping", "mission", "memory", "feasibility",
)

# item 2 (defect): ARCHITECTURE_SECTIONS above deliberately EXCLUDES `scenario`
# (by design -- see EVALUATION_CONTRACT_SECTIONS/test_fingerprint.py's own
# "evaluation contract changes don't affect architecture" contract). That
# exclusion is correct for the evaluation-fairness use case but WRONG for
# gating whether a frozen Local checkpoint's own TRAINING distribution
# (what subgoal distances/directions/infeasible-fraction it actually saw)
# still matches what a live hierarchical caller (LiveGazeboLocalExecutor)
# is about to drive it against -- a checkpoint trained under a narrow
# goal_direction_sectors_deg is not safe to treat as arbitrary-subgoal-
# capable just because its action/observation/network ARCHITECTURE still
# matches. This is a SEPARATE, additional fingerprint -- never folded into
# ARCHITECTURE_SECTIONS itself (that would change what every existing
# architecture_fingerprint value means).
LOCAL_TRAINING_CONTRACT_FIELDS = (
    "goal_sampling_mode", "goal_distance_range_m", "goal_direction_sectors_deg",
    "goal_infeasible_fraction", "feasibility_check",
)


def hierarchical_architecture_fingerprint(profile: Profile) -> str:
    """SHA-256 of ``profile``'s Phase 4/5 hierarchical-architecture
    sections. Two profiles (or ablations A-G of the SAME base profile) with
    a DIFFERENT value here are guaranteed to produce different Global
    network shapes, observation contracts, or hierarchy/mapping semantics --
    exactly the "ablation checkpoint가 서로 다른 architecture fingerprint로
    구분된다" requirement. Two profiles with the SAME value are not
    guaranteed identical Global checkpoints (weights still depend on
    training data/seed), only identical ARCHITECTURE/CONTRACT."""
    profile_dict = dataclasses.asdict(profile)
    return sha256_of_obj(_sections_from_profile_dict(profile_dict, HIERARCHICAL_ARCHITECTURE_SECTIONS))


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def sha256_of_obj(obj: Any) -> str:
    return hashlib.sha256(_canonical_json(obj).encode("utf-8")).hexdigest()


def _sections_from_profile_dict(profile_dict: Dict[str, Any], sections) -> Dict[str, Any]:
    return {name: profile_dict.get(name, {}) for name in sections}


def architecture_fingerprint(profile: Profile) -> str:
    """SHA-256 of ``profile``'s architecture-relevant sections. Two
    profiles (or a profile and a checkpoint's own frozen
    ``resolved_config``) with the same fingerprint are guaranteed to
    produce IDENTICAL agent network shapes and action/observation
    semantics -- not just "the same profile name"."""
    profile_dict = dataclasses.asdict(profile)
    return sha256_of_obj(_sections_from_profile_dict(profile_dict, ARCHITECTURE_SECTIONS))


def evaluation_contract_fingerprint(profile: Profile) -> str:
    """SHA-256 of ``profile``'s evaluation-CONTRACT sections (benchmark
    selection, episode budget, common-metrics yardstick, reward/termination,
    world size). Every model evaluated through the same requested
    ``--profile`` must produce the SAME value here."""
    profile_dict = dataclasses.asdict(profile)
    return sha256_of_obj(_sections_from_profile_dict(profile_dict, EVALUATION_CONTRACT_SECTIONS))


def hierarchical_architecture_fingerprint_from_resolved_config(resolved_config: Dict[str, Any]) -> str:
    """Same as :func:`hierarchical_architecture_fingerprint`, but operating
    directly on a Global checkpoint manifest's raw ``resolved_config`` dict."""
    return sha256_of_obj(_sections_from_profile_dict(resolved_config, HIERARCHICAL_ARCHITECTURE_SECTIONS))


def architecture_fingerprint_from_resolved_config(resolved_config: Dict[str, Any]) -> str:
    """Same as :func:`architecture_fingerprint`, but operating directly on
    a checkpoint manifest's raw ``resolved_config`` dict (``dataclasses.asdict``
    output, e.g. straight from a loaded ``.json`` manifest) -- avoids
    needing to round-trip it back through ``profile_from_dict`` just to
    fingerprint it."""
    return sha256_of_obj(_sections_from_profile_dict(resolved_config, ARCHITECTURE_SECTIONS))


def _scenario_contract_dict(scenario_section: Dict[str, Any]) -> Dict[str, Any]:
    """Every field in :data:`LOCAL_TRAINING_CONTRACT_FIELDS` MUST be present
    -- a missing field is a defect-2 "field 누락도 mismatch로 처리" case, so
    this raises (never silently drops/defaults a field) rather than
    producing a fingerprint that would compare equal to another checkpoint
    that genuinely differs on the missing field."""
    missing = [f for f in LOCAL_TRAINING_CONTRACT_FIELDS if f not in scenario_section]
    if missing:
        raise KeyError(
            f"local_training_contract_fingerprint: scenario section is missing required field(s) {missing} "
            "-- refusing to compute a fingerprint that would silently ignore them"
        )
    return {name: scenario_section[name] for name in LOCAL_TRAINING_CONTRACT_FIELDS}


def local_training_contract_fingerprint(profile: Profile) -> str:
    """SHA-256 of ``profile.scenario``'s subgoal-distribution-defining
    fields (:data:`LOCAL_TRAINING_CONTRACT_FIELDS`) -- distinguishes a Local
    checkpoint's ACTUAL TRAINING distribution (what subgoal candidates it
    was ever exposed to) from its architecture (what tensor shapes it
    produces/consumes). Two checkpoints with the same
    :func:`architecture_fingerprint` can still have different values here
    (e.g. narrow vs. full-circle ``goal_direction_sectors_deg``) -- that is
    exactly the case this fingerprint exists to catch."""
    profile_dict = dataclasses.asdict(profile)
    scenario_section = profile_dict.get("scenario")
    if scenario_section is None:
        raise KeyError("local_training_contract_fingerprint: profile has no 'scenario' section")
    return sha256_of_obj(_scenario_contract_dict(scenario_section))


def local_training_contract_fingerprint_from_resolved_config(resolved_config: Dict[str, Any]) -> str:
    """Same as :func:`local_training_contract_fingerprint`, operating
    directly on a checkpoint manifest's raw ``resolved_config`` dict."""
    scenario_section = resolved_config.get("scenario")
    if scenario_section is None:
        raise KeyError("local_training_contract_fingerprint_from_resolved_config: resolved_config has no 'scenario' section")
    return sha256_of_obj(_scenario_contract_dict(scenario_section))


# item 4 (defect): the fields a Global checkpoint's own
# `local_checkpoint_manifest_summary` (written by
# nodes/hierarchical_train_node.py) and a currently-loaded Local
# checkpoint's own manifest (LiveGazeboLocalExecutor.checkpoint_manifest)
# must agree on before either a Global-resume or a live benchmark run is
# allowed to proceed against them together. A single shared helper --
# never re-derived separately by the resume path and the benchmark path --
# so the two call sites can never silently drift apart on what "identity"
# means.
LOCAL_CHECKPOINT_IDENTITY_FIELDS = (
    "generation", "pt_sha256", "state_dim", "action_dim", "local_training_contract_fingerprint",
)


def compare_local_checkpoint_identity(
    recorded_summary: Optional[Dict[str, Any]], current_summary: Optional[Dict[str, Any]],
) -> List[str]:
    """Returns a list of human-readable mismatch descriptions (empty list =
    compatible). ``recorded_summary`` is what a Global checkpoint's own
    manifest recorded about the Local checkpoint it was trained/benchmarked
    against; ``current_summary`` is the CURRENTLY loaded Local checkpoint's
    own manifest-derived summary. Missing on either side, or a missing
    field within an otherwise-present summary, is ALWAYS a mismatch --
    never silently skipped (defect 3/4: "필드 누락도 mismatch로 처리")."""
    if recorded_summary is None and current_summary is None:
        return []
    if recorded_summary is None:
        return ["local_checkpoint_manifest_summary: not recorded on the Global checkpoint, but a Local "
                "checkpoint IS currently loaded -- refusing to treat an unrecorded pairing as compatible"]
    if current_summary is None:
        return ["local_checkpoint_manifest_summary: Global checkpoint recorded one, but no Local checkpoint "
                "is currently loaded -- refusing to treat an unrecorded pairing as compatible"]
    mismatches = []
    for key in LOCAL_CHECKPOINT_IDENTITY_FIELDS:
        if key not in recorded_summary:
            mismatches.append(f"local_checkpoint.{key}: missing from recorded Global-checkpoint summary")
            continue
        if key not in current_summary:
            mismatches.append(f"local_checkpoint.{key}: missing from currently loaded Local manifest")
            continue
        recorded_value = recorded_summary.get(key)
        current_value = current_summary.get(key)
        if recorded_value != current_value:
            mismatches.append(f"local_checkpoint.{key}: recorded={recorded_value!r} current={current_value!r}")
    return mismatches
