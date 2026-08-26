"""YAML -> :class:`~hunter_kinodynamic_rl.config.schema.Profile` loader.

Layering (later overrides earlier, deep-merged dict-of-dicts):

  1. ``config/robot/<robot>.yaml``               -- physical platform parameters
  2. ``config/training/defaults.yaml``           -- baseline hyperparameters/training knobs
  3. ``config/domain_randomization/default.yaml`` -- opt-in randomization RANGES (section
     P1-10) -- still ``enabled: false`` by default, so layering it in changes NOTHING for
     any profile that doesn't itself set ``domain_randomization.enabled: true``; a profile
     that does enable it gets these reasonable default ranges without repeating all 11
     range fields inline, and can still override individual ranges itself (layer 4 below)
  4. ``config/profiles/<profile>.yaml``          -- the ablation/experiment profile itself

Nothing here talks to ROS or the filesystem beyond plain file I/O, so it is
covered by ROS-free unit tests. Validation is fail-fast: :func:`load_profile`
raises :class:`~hunter_kinodynamic_rl.config.schema.ConfigError` immediately on
any structurally or physically invalid value -- never partway through training.
"""

from __future__ import annotations

import copy
import dataclasses
import os
from typing import Any, Dict, Optional

import yaml

from hunter_kinodynamic_rl.config.schema import (
    ActionSpaceConfig, AlgorithmConfig, ConfigError, CounterfactualConfig, DomainRandomizationConfig,
    DynamicsConfig, EvaluationConfig, FeatureFlags, ObservationConfig, Profile,
    RewardConfig, RiskConfig, RobotConfig, RuntimeConfig, SACHyperparameters, ScenarioConfig,
    TQCHyperparameters, TrainingConfig, TrajectoryConfig,
)

_SECTION_TYPES = {
    "robot": RobotConfig,
    "action_space": ActionSpaceConfig,
    "trajectory": TrajectoryConfig,
    "dynamics": DynamicsConfig,
    "observation": ObservationConfig,
    "risk": RiskConfig,
    "counterfactual": CounterfactualConfig,
    "hyperparameters": TQCHyperparameters,
    "sac_hyperparameters": SACHyperparameters,
    "algorithm": AlgorithmConfig,
    "features": FeatureFlags,
    "scenario": ScenarioConfig,
    "training": TrainingConfig,
    "evaluation": EvaluationConfig,
    "reward": RewardConfig,
    "domain_randomization": DomainRandomizationConfig,
    "runtime": RuntimeConfig,
}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` onto ``base`` (dicts merge key-wise,
    everything else -- including lists -- is replaced outright)."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_yaml(path: str) -> Dict[str, Any]:
    if not path or not os.path.isfile(path):
        return {}
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top-level YAML must be a mapping, got {type(data).__name__}")
    return data


def _section_from_dict(cls, data: Dict[str, Any]):
    """Build a dataclass instance from a dict, rejecting unknown keys (typo
    guard -- an ablation flag that silently does nothing is worse than a
    startup error)."""
    field_names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data.keys()) - field_names
    if unknown:
        raise ConfigError(f"{cls.__name__}: unknown key(s) {sorted(unknown)}; valid keys are {sorted(field_names)}")
    return cls(**data)


def profile_from_dict(name: str, merged: Dict[str, Any]) -> Profile:
    kwargs = {}
    unknown_sections = set(merged.keys()) - set(_SECTION_TYPES.keys())
    if unknown_sections:
        raise ConfigError(f"profile {name!r}: unknown section(s) {sorted(unknown_sections)}")
    for section, cls in _SECTION_TYPES.items():
        kwargs[section] = _section_from_dict(cls, merged.get(section, {}) or {})
    profile = Profile(name=name, **kwargs)
    profile.validate()
    return profile


def default_config_root() -> str:
    """The ``config/`` directory shipped with this package.

    Once ``colcon build`` installs this package, the importable Python
    package lands under ``install/hunter_kinodynamic_rl/local/lib/.../
    dist-packages/hunter_kinodynamic_rl/`` while ``config/`` (an
    ``install(DIRECTORY config ...)`` rule in CMakeLists.txt, not part of
    the Python package) lands under ``install/hunter_kinodynamic_rl/share/
    hunter_kinodynamic_rl/config/`` -- a SIBLING tree, not a relative parent
    of the Python module's own location. ``ament_index_python`` resolves
    that share directory; when it's unavailable (running from an unbuilt
    source checkout, e.g. these tests) fall back to the source-tree-relative
    path (package root's ``config/``, two levels up from this file).
    """
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory("hunter_kinodynamic_rl"), "config")
    except Exception:
        here = os.path.dirname(os.path.abspath(__file__))
        # hunter_kinodynamic_rl/hunter_kinodynamic_rl/config/loader.py -> package root/config
        candidate = os.path.join(here, "..", "..", "config")
        return os.path.normpath(candidate)


def load_profile(profile_name: str, config_root: Optional[str] = None) -> Profile:
    """Resolve and validate a profile by name (e.g. ``"kinodynamic_tqc"``) or
    by an explicit path to its YAML file.

    Robot layer is selected via the profile's own ``robot_file`` key
    (default ``"hunter_se.yaml"``) so a future robot swap only needs a new
    ``config/robot/<name>.yaml`` plus that one key in the profile.
    """
    root = config_root or default_config_root()

    if os.path.isfile(profile_name):
        profile_path = profile_name
        resolved_name = os.path.splitext(os.path.basename(profile_name))[0]
    else:
        profile_path = os.path.join(root, "profiles", f"{profile_name}.yaml")
        resolved_name = profile_name
    if not os.path.isfile(profile_path):
        raise ConfigError(f"profile not found: {profile_path}")

    profile_raw = _load_yaml(profile_path)
    robot_file = profile_raw.pop("robot_file", "hunter_se.yaml")

    # config/robot/<robot_file>.yaml is itself shaped as {"robot": {...fields}}
    # (a single top-level section, same shape every other layer uses), so it
    # is loaded directly -- not re-wrapped under an extra "robot" key.
    robot_layer = _load_yaml(os.path.join(root, "robot", robot_file))
    defaults_layer = _load_yaml(os.path.join(root, "training", "defaults.yaml"))
    domain_rand_layer = _load_yaml(os.path.join(root, "domain_randomization", "default.yaml"))

    merged = deep_merge(robot_layer, defaults_layer)
    merged = deep_merge(merged, domain_rand_layer)
    merged = deep_merge(merged, profile_raw)
    profile = profile_from_dict(resolved_name, merged)

    # Filesystem check (not part of Profile.validate(), which is pure/no I/O):
    # an evaluation profile naming a benchmark that doesn't actually have any
    # scenario files would silently evaluate on zero episodes (section P2:
    # "evaluation profile과 benchmark 존재").
    if profile.evaluation.benchmark:
        benchmark_dir = os.path.join(root, "benchmarks", profile.evaluation.benchmark)
        if not os.path.isdir(benchmark_dir):
            raise ConfigError(f"profile {resolved_name!r}: benchmark directory not found: {benchmark_dir}")
        if not any(f.endswith(".yaml") for f in os.listdir(benchmark_dir)):
            raise ConfigError(f"profile {resolved_name!r}: benchmark directory {benchmark_dir} has no scenario files")

    return profile
