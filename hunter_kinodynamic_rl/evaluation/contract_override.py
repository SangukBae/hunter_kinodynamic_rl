"""section item-1 (round 2): file-based delivery of the REQUESTED evaluation
profile's evaluation-CONTRACT sections (``reward``/``scenario``/``runtime``/
``evaluation``/``sensor_noise``) to a LIVE ``environment_node.py`` process, mirroring the
existing ``scenario_override_path`` mechanism (``env/scenarios/benchmark_loader.py``)
that already delivers exact scenario placement the same way.

Why this exists: ``evaluation_node.py`` reconstructs the checkpoint's own
training ARCHITECTURE (``action_space``/``features``/``observation``/
``robot``/``dynamics``/``risk``/``counterfactual``/``hyperparameters``/
``algorithm``) from its manifest, but a live ``environment_node.py`` is a
SEPARATE, already-launched ROS process whose own ``self.profile`` was
resolved ONCE at ITS launch (from whatever ``-p profile:=`` it was started
with -- see that module's own docstring for why it cannot be reconstructed
at runtime). Without an explicit delivery mechanism, the live environment
keeps using ITS OWN launch-time profile's world boundary
(``scenario.world_size_m``), reward/termination (``reward``), episode
timeout budget, common-metrics config (``evaluation``), and physics/timing
settings (``runtime`` -- ``time_delta_sec``, ``deterministic_stepping``,
...) and its OWN training-time sensor/localization noise model
(``sensor_noise``) for EVERY episode, regardless of what the REQUESTED
evaluation profile asks for -- silently unfair whenever two checkpoints
were trained under profiles with different values for any of these (world
size, reward shaping, physics step size, and whether/how much sensor noise
is injected all directly change episode difficulty/dynamics, not just
cosmetic config).

This module writes/reads a small YAML containing exactly those five
sections (never the architecture-owned ones -- see
``evaluation/fingerprint.py``'s own ``ARCHITECTURE_SECTIONS`` vs
``EVALUATION_CONTRACT_SECTIONS`` split, which this module's own section
list is kept in lock-step with). ``environment_node.py`` applies it
PER-EPISODE (at the top of ``/reset``, exactly like ``scenario_override_path``
is re-read every reset) rather than requiring a full node
reconstruction/relaunch -- a deliberately narrow, low-risk form of dynamic
reconfiguration: only ``reward``/``scenario``/``runtime``/``evaluation``/
``sensor_noise`` are ever replaced, never anything that would change the
checkpoint's own network shape or action/observation semantics.

``sensor_noise`` is completely independent of ``env/randomization/
domain_randomizer.py``'s per-episode-randomized, TRAIN-ONLY noise system --
that system's own ``sensor:`` per-fixed-scenario overrides remain rejected
outright for benchmark evaluation (``check_sensor_overrides_supported``);
only THIS section (the requested evaluation profile's fixed-shape
``sensor_noise`` model) may make a fixed benchmark noisy, and only when the
evaluation profile itself asks for it.
"""

from __future__ import annotations

import dataclasses
from typing import Dict

import yaml

from hunter_kinodynamic_rl.config.loader import _section_from_dict
from hunter_kinodynamic_rl.config.schema import (
    EvaluationConfig, Profile, RewardConfig, RuntimeConfig, ScenarioConfig, SensorNoiseConfig,
)

# Kept identical to evaluation/fingerprint.py's own EVALUATION_CONTRACT_SECTIONS
# (order doesn't matter for correctness -- both are just YAML top-level keys
# -- but the SET must match exactly, or a section could silently be
# fingerprinted without ever actually being deliverable, or vice versa).
# `sensor_noise` (requirement 2): the requested evaluation profile's own
# sensor/localization noise model must be what every checkpoint sees during
# a benchmark, never each checkpoint's own training-time setting -- see
# fingerprint.py's EVALUATION_CONTRACT_SECTIONS docstring for the full
# rationale and how this is distinct from domain_randomizer's train-only
# `sensor:` overrides.
CONTRACT_SECTION_NAMES = ("reward", "scenario", "runtime", "evaluation", "sensor_noise")
_CONTRACT_SECTION_TYPES = {
    "reward": RewardConfig, "scenario": ScenarioConfig, "runtime": RuntimeConfig, "evaluation": EvaluationConfig,
    "sensor_noise": SensorNoiseConfig,
}


def write_evaluation_contract_override(path: str, profile: Profile) -> None:
    """Serializes ``profile``'s own ``reward``/``scenario``/``runtime``/
    ``evaluation`` sections (and ONLY those) to ``path`` -- the file
    ``environment_node.py``'s own ``evaluation_contract_override_path``
    parameter points at."""
    data = {name: dataclasses.asdict(getattr(profile, name)) for name in CONTRACT_SECTION_NAMES}
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def load_evaluation_contract_override(path: str) -> Dict[str, object]:
    """Reads back a file :func:`write_evaluation_contract_override` wrote
    (or any hand-authored YAML with the same 4-section shape), returning
    ``{"reward": RewardConfig(...), "scenario": ScenarioConfig(...),
    "runtime": RuntimeConfig(...), "evaluation": EvaluationConfig(...)}``.
    Rejects an unknown top-level section (typo guard, matching
    ``config/loader.py``'s own convention) and an unknown key WITHIN a
    section (via the shared ``_section_from_dict``, which rejects those
    too) -- a silently-ignored override field is worse than a startup
    error here specifically, since it would make an evaluation run
    silently unfair rather than merely misconfigured."""
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping, got {type(data).__name__}")
    unknown = set(data.keys()) - set(CONTRACT_SECTION_NAMES)
    if unknown:
        raise ValueError(
            f"{path}: unknown evaluation-contract section(s) {sorted(unknown)}; only "
            f"{CONTRACT_SECTION_NAMES} are accepted"
        )
    return {
        name: _section_from_dict(_CONTRACT_SECTION_TYPES[name], data.get(name, {}) or {})
        for name in CONTRACT_SECTION_NAMES
    }


def apply_evaluation_contract(profile: Profile, sections: Dict[str, object]) -> Profile:
    """Returns a NEW ``Profile`` with ``reward``/``scenario``/``runtime``/
    ``evaluation``/``sensor_noise`` replaced by ``sections`` -- every architecture-owned
    section (``action_space``/``features``/``observation``/``robot``/
    ``dynamics``/``risk``/``counterfactual``/``hyperparameters``/
    ``sac_hyperparameters``/``algorithm``) and every training-loop-only
    section (``training``/``domain_randomization``) is carried over
    UNCHANGED from ``profile`` -- never touched by an evaluation-contract
    override."""
    return dataclasses.replace(profile, **sections)
