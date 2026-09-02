#!/usr/bin/env python3
"""Requirement I: Phase 6 localization sweep -- same scenario manifest under
ideal/noisy/drifting localization, pairing-verified aggregation.

Noise/drift injection itself is already implemented at the localization-
backend level (``navigation.localization.wheel_imu_backend.WheelImuNoiseModel``,
``navigation.localization.lidar_odom_backend.LidarOdomNoiseModel`` -- plan
10.5 Phase B/C) -- this module does not re-implement noise models, only:

1. :func:`verify_scenario_pairing` -- refuses to aggregate/compare sweep
   conditions whose episode sets do not cover the EXACT SAME scenario_ids
   (never a silent partial/mismatched comparison).
2. :func:`aggregate_localization_sweep` -- per-condition metric summaries +
   a drift-magnitude-vs-performance curve
   (:func:`evaluation.global_metrics.localization_drift_sensitivity` against
   the ``"ideal"`` condition), recording each condition's own noise/drift
   parameters and seed alongside its results.

Generic over WHICH episode-dict shape a caller supplies (Global mission
dicts from :mod:`evaluation.long_horizon_benchmark`, or Local subgoal
dicts from :mod:`evaluation.local_subgoal_benchmark`) and over which
ablation produced them -- a caller applies this once per
ablation/A-B-comparison, satisfying "A/B 및 high-level/low-level ablation에
동일 조건을 적용할 수 있어야 한다" by construction rather than by any
ablation-specific code here."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from hunter_kinodynamic_rl.evaluation.global_metrics import aggregate, localization_drift_sensitivity

SWEEP_SCHEMA_VERSION = 1
IDEAL_CONDITION = "ideal"


class ScenarioPairingError(ValueError):
    """Raised (never swallowed) when two sweep conditions' episode sets do
    not cover the identical scenario_id set -- comparing them would compare
    different scenarios under different labels, defeating the whole point
    of a controlled localization sweep."""


def verify_scenario_pairing(results: Dict[str, List[dict]]) -> None:
    """Every condition's episodes must share the EXACT SAME set of
    ``scenario_id`` values (episode dicts from either
    ``long_horizon_benchmark.run_ablation_mission`` or
    ``local_subgoal_benchmark.run_local_benchmark_episode`` both carry this
    key). Raises :class:`ScenarioPairingError` on any mismatch."""
    if not results:
        raise ScenarioPairingError("verify_scenario_pairing: no conditions supplied")
    id_sets: Dict[str, set] = {}
    for condition, episodes in results.items():
        missing = [e for e in episodes if "scenario_id" not in e]
        if missing:
            raise ScenarioPairingError(
                f"verify_scenario_pairing: condition {condition!r} has {len(missing)} episode(s) missing "
                "'scenario_id'"
            )
        id_sets[condition] = {e["scenario_id"] for e in episodes}
    reference_condition, reference_ids = next(iter(id_sets.items()))
    for condition, ids in id_sets.items():
        if ids != reference_ids:
            missing_here = reference_ids - ids
            extra_here = ids - reference_ids
            raise ScenarioPairingError(
                f"verify_scenario_pairing: condition {condition!r} scenario_id set does not match "
                f"{reference_condition!r} -- missing={sorted(missing_here)} extra={sorted(extra_here)}"
            )


@dataclass
class LocalizationSweepResult:
    schema_version: int
    conditions: List[str]
    noise_params: Dict[str, Any]
    seeds: Dict[str, int]
    summaries: Dict[str, Dict[str, Any]]
    drift_sensitivity: Dict[str, Optional[float]] = field(default_factory=dict)


def aggregate_localization_sweep(
    results: Dict[str, List[dict]], *, noise_params: Optional[Dict[str, Any]] = None,
    seeds: Optional[Dict[str, int]] = None,
) -> LocalizationSweepResult:
    """Fails fast (:class:`ScenarioPairingError`) if scenario pairing is
    broken across conditions -- never silently aggregates a mismatched
    comparison. ``noise_params``/``seeds`` are ``{condition: value}`` --
    recorded verbatim into the result for reproducibility (not derived or
    validated here; the caller that actually configured each condition's
    localization backend owns that)."""
    verify_scenario_pairing(results)
    noise_params = noise_params or {}
    seeds = seeds or {}
    summaries = {condition: aggregate(episodes) for condition, episodes in results.items()}

    drift_sensitivity: Dict[str, Optional[float]] = {}
    ideal_summary = summaries.get(IDEAL_CONDITION)
    if ideal_summary is not None:
        for condition, summary in summaries.items():
            if condition == IDEAL_CONDITION:
                continue
            drift_sensitivity[condition] = localization_drift_sensitivity(ideal_summary, summary)

    return LocalizationSweepResult(
        schema_version=SWEEP_SCHEMA_VERSION, conditions=sorted(results.keys()), noise_params=noise_params,
        seeds=seeds, summaries=summaries, drift_sensitivity=drift_sensitivity,
    )


def drift_curve(sweep: LocalizationSweepResult, *, drift_magnitude_key: str = "drift_std_m") -> List[Dict[str, Any]]:
    """Rearranges :attr:`LocalizationSweepResult.drift_sensitivity` into a
    ``[{"condition", "drift_magnitude", "sensitivity"}, ...]`` list sorted
    by drift magnitude -- directly plottable as a drift-magnitude-vs-
    performance curve (plan 10.10: "drift magnitude별 성능 저하 curve를
    생성할 수 있다"). A condition whose ``noise_params`` doesn't carry
    ``drift_magnitude_key`` is excluded (never fabricated as 0)."""
    rows = []
    for condition, sensitivity in sweep.drift_sensitivity.items():
        params = sweep.noise_params.get(condition, {})
        if drift_magnitude_key not in params:
            continue
        rows.append({
            "condition": condition, "drift_magnitude": params[drift_magnitude_key], "sensitivity": sensitivity,
        })
    rows.sort(key=lambda r: r["drift_magnitude"])
    return rows
