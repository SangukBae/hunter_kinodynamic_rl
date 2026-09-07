"""Validation for the evidence boundary of v2 domain randomization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import yaml

from hunter_kinodynamic_rl.config.schema import Profile
from hunter_kinodynamic_rl.env.randomization.domain_randomizer import (
    GAZEBO_APPLIED_FIELDS, MODEL_ONLY_FIELDS, OBSERVATION_ONLY_FIELDS,
)


@dataclass(frozen=True)
class CalibrationReport:
    ok: bool
    status: str
    errors: Tuple[str, ...]
    warnings: Tuple[str, ...]


def validate_calibration_manifest(profile: Profile, config_root: str) -> CalibrationReport:
    errors = []
    warnings = []
    path = Path(config_root) / profile.environment_v2.calibration_manifest
    if not path.is_file():
        return CalibrationReport(False, "missing", (f"calibration manifest not found: {path}",), ())
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if data.get("schema_id") != "hunter_sim_calibration_v1":
        errors.append("calibration schema_id must be hunter_sim_calibration_v1")
    status = str(data.get("status", "missing"))
    if status not in ("engineering_prior", "measured"):
        errors.append("calibration status must be engineering_prior or measured")
    sources = data.get("source_artifacts") or []
    if status == "measured" and not sources:
        errors.append("measured calibration requires non-empty source_artifacts")
    if status != "measured":
        warnings.append("domain-randomization ranges are engineering priors, not measured Hunter SE evidence")
        if profile.environment_v2.require_measured_calibration:
            errors.append("environment_v2.require_measured_calibration=true but manifest is not measured")

    expected_ranges = data.get("domain_randomization_ranges") or {}
    for field_name in profile.domain_randomization.__dataclass_fields__:
        if field_name == "enabled":
            continue
        expected = expected_ranges.get(field_name)
        actual = getattr(profile.domain_randomization, field_name)
        if expected is None:
            errors.append(f"calibration manifest missing {field_name}")
        elif list(expected) != list(actual):
            errors.append(f"profile {field_name}={actual} differs from calibration manifest {expected}")

    contract = data.get("application_contract") or {}
    expected_contract = {
        "gazebo_applied": set(GAZEBO_APPLIED_FIELDS),
        "observation_only": set(OBSERVATION_ONLY_FIELDS),
        "model_only": set(MODEL_ONLY_FIELDS),
    }
    for category, expected in expected_contract.items():
        actual = set(contract.get(category) or [])
        if actual != expected:
            errors.append(
                f"application_contract.{category}={sorted(actual)} does not match code {sorted(expected)}"
            )
    if MODEL_ONLY_FIELDS:
        warnings.append(
            "mass_scale and wheel_radius_scale remain model-only; a Gazebo model plugin or respawned SDF "
            "is required before claiming plant-level randomization for those axes"
        )
    return CalibrationReport(not errors, status, tuple(errors), tuple(warnings))
