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
from typing import Any, Dict

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


def architecture_fingerprint_from_resolved_config(resolved_config: Dict[str, Any]) -> str:
    """Same as :func:`architecture_fingerprint`, but operating directly on
    a checkpoint manifest's raw ``resolved_config`` dict (``dataclasses.asdict``
    output, e.g. straight from a loaded ``.json`` manifest) -- avoids
    needing to round-trip it back through ``profile_from_dict`` just to
    fingerprint it."""
    return sha256_of_obj(_sections_from_profile_dict(resolved_config, ARCHITECTURE_SECTIONS))
