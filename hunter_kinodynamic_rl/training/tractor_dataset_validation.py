#!/usr/bin/env python3
"""Fail-closed validation report for a TRACTOR episode dataset."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import List

import numpy as np

from hunter_kinodynamic_rl.rl.replay import EpisodeStore, SequenceIndex
from hunter_kinodynamic_rl.training.tractor_scenario_plan import (
    validate_materialized_scenario_manifest,
)


@dataclass
class DatasetValidationReport:
    ok: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    episode_count: int = 0
    row_count: int = 0
    window_count: int = 0
    split_episode_counts: dict = field(default_factory=dict)
    split_window_counts: dict = field(default_factory=dict)
    scenario_counts: dict = field(default_factory=dict)
    scenario_family_counts: dict = field(default_factory=dict)
    obstacle_contract_counts: dict = field(default_factory=dict)
    scenario_geometry_count: int = 0
    termination_counts: dict = field(default_factory=dict)
    event_counts: dict = field(default_factory=dict)
    label_source_counts: dict = field(default_factory=dict)
    missing_rates: dict = field(default_factory=dict)
    contract_hashes: dict = field(default_factory=dict)
    index_sha256: str | None = None
    scenario_manifest_sha256: str | None = None


def validate_dataset(
    root: str | Path, loss_window: int = 16, *, formal: bool = False,
    scenario_manifest_path: str | Path | None = None, config_root: str | None = None,
) -> DatasetValidationReport:
    report = DatasetValidationReport()
    dataset_root = Path(root)
    split_episodes, scenarios, scenario_families, obstacle_contracts = (
        Counter(), Counter(), Counter(), Counter()
    )
    terminations = Counter()
    group_splits = {}
    geometry_splits = {}
    scenario_splits = {}
    scenario_occurrences = Counter()
    expected_scenarios = None
    expected_protocol_version = None
    if scenario_manifest_path is not None:
        try:
            manifest = validate_materialized_scenario_manifest(scenario_manifest_path, config_root)
            report.scenario_manifest_sha256 = str(manifest["artifact_manifest_sha256"])
            expected_protocol_version = str(manifest["protocol_version"])
            expected_scenarios = {
                str(item["scenario_id"]): item for item in manifest["entries"]
            }
        except Exception as error:
            report.errors.append(f"scenario manifest: {error}")
    elif formal:
        report.errors.append("formal dataset validation requires a frozen materialized scenario manifest")
    contract_sets = {
        "resolved_config": set(), "environment": set(), "robot": set(),
        "observation": set(), "action": set(), "trajectory": set(),
    }
    missing_numerator = Counter()
    missing_denominator = Counter()
    event_counts = Counter()
    label_sources = Counter()
    realized_lineage_errors = []
    try:
        if not (dataset_root / "episodes").is_dir():
            raise FileNotFoundError(f"dataset episodes directory does not exist: {dataset_root / 'episodes'}")
        store = EpisodeStore(dataset_root)
        paths = tuple(store.iter_paths())
        if not paths:
            report.errors.append("dataset contains no immutable episode chunks")
        for path in paths:
            header, columns = store.load(path.stem)
            report.episode_count += 1
            report.row_count += header.step_count
            split_episodes[header.split_id] += 1
            scenario_families[header.scenario_family] += 1
            obstacle_contracts[header.obstacle_contract] += 1
            scenario_occurrences[header.scenario_id] += 1
            prior_split = group_splits.setdefault(header.group_id, header.split_id)
            if prior_split != header.split_id:
                report.errors.append(
                    f"group {header.group_id!r} leaks across {prior_split!r} and {header.split_id!r}"
                )
            geometry_prior = geometry_splits.setdefault(
                header.scenario_geometry_sha256, header.split_id
            )
            if geometry_prior != header.split_id:
                report.errors.append(
                    f"scenario geometry {header.scenario_geometry_sha256!r} leaks across "
                    f"{geometry_prior!r} and {header.split_id!r}"
                )
            scenario_prior = scenario_splits.setdefault(header.scenario_id, header.split_id)
            if scenario_prior != header.split_id:
                report.errors.append(
                    f"scenario id {header.scenario_id!r} leaks across "
                    f"{scenario_prior!r} and {header.split_id!r}"
                )
            if expected_scenarios is not None:
                expected = expected_scenarios.get(header.scenario_id)
                if expected is None:
                    report.errors.append(f"episode uses unregistered scenario {header.scenario_id!r}")
                else:
                    comparisons = {
                        "split_id": header.split_id,
                        "family_id": header.scenario_family,
                        "obstacle_contract": header.obstacle_contract,
                        "seed": header.seed,
                        "scenario_geometry_sha256": header.scenario_geometry_sha256,
                        "group_id": header.group_id,
                    }
                    for name, observed in comparisons.items():
                        if observed != expected[name]:
                            report.errors.append(
                                f"scenario {header.scenario_id!r} {name}={observed!r} "
                                f"does not match frozen manifest value {expected[name]!r}"
                            )
                    if header.protocol_version != expected_protocol_version:
                        report.errors.append(
                            f"scenario {header.scenario_id!r} protocol_version does not match manifest"
                        )
            scenarios[header.scenario_id] += 1
            terminations[header.termination_reason] += 1
            contract_sets["resolved_config"].add(header.resolved_config_hash)
            contract_sets["environment"].add(header.environment_attestation_hash)
            contract_sets["robot"].add(header.robot_attestation_hash)
            contract_sets["observation"].add(header.observation_contract_hash)
            contract_sets["action"].add(header.action_contract_hash)
            contract_sets["trajectory"].add(header.trajectory_contract_hash)
            for field_name in (
                "motion_valid", "localization_valid", "sensor_freshness_valid",
                "previous_command_valid", "bellman_sample_valid",
            ):
                values = np.asarray(columns[field_name], dtype=bool)
                missing_numerator[field_name] += int(values.size - values.sum())
                missing_denominator[field_name] += int(values.size)
            if "collision_cause" in columns and "cause_valid" in columns:
                valid = np.asarray(columns["cause_valid"], dtype=bool)
                for cause in np.asarray(columns["collision_cause"])[valid]:
                    event_counts[str(int(cause))] += 1
            if "candidate_label_source" in columns:
                for source in np.asarray(columns["candidate_label_source"]).astype(str):
                    label_sources[str(source)] += 1
                if np.any(
                    np.asarray(columns["candidate_label_source"]).astype(str)
                    == "realized_timestamp_aligned_counterfactual_v1"
                ):
                    required_privileged = {
                        "privileged_snapshot_valid", "privileged_snapshot_timestamp_sec",
                        "privileged_ego_pose_world", "privileged_world_half_extent_m",
                        "privileged_obstacles_world",
                    }
                    missing_privileged = sorted(required_privileged - set(columns))
                    if missing_privileged:
                        realized_lineage_errors.append(
                            f"{header.episode_id}: missing realized-track lineage {missing_privileged}"
                        )
                    elif not np.asarray(columns["privileged_snapshot_valid"], dtype=bool).all():
                        realized_lineage_errors.append(
                            f"{header.episode_id}: realized labels contain invalid privileged snapshots"
                        )
        index = SequenceIndex.build(store, loss_window=loss_window)
        index.validate_split_isolation()
        report.window_count = len(index.windows)
        report.index_sha256 = index.sha256()
        report.split_window_counts = dict(sorted(Counter(w.split_id for w in index.windows).items()))
        if report.window_count == 0:
            report.errors.append(f"no valid reset-contained windows of length {loss_window}")
    except Exception as error:  # report artifact should survive validation failure
        report.errors.append(str(error))
    report.split_episode_counts = dict(sorted(split_episodes.items()))
    report.scenario_counts = dict(sorted(scenarios.items()))
    report.scenario_family_counts = dict(sorted(scenario_families.items()))
    report.obstacle_contract_counts = dict(sorted(obstacle_contracts.items()))
    report.scenario_geometry_count = len(geometry_splits)
    report.termination_counts = dict(sorted(terminations.items()))
    report.event_counts = dict(sorted(event_counts.items()))
    report.label_source_counts = dict(sorted(label_sources.items()))
    report.missing_rates = {
        name: count / max(1, missing_denominator[name])
        for name, count in sorted(missing_numerator.items())
    }
    report.contract_hashes = {
        name: sorted(values) for name, values in contract_sets.items()
    }
    for name, values in contract_sets.items():
        if len(values) > 1:
            report.errors.append(f"mixed {name} contract hashes: {sorted(values)}")
    required_splits = {"development", "calibration", "locked_test"}
    absent = sorted(required_splits - set(split_episodes))
    if absent:
        message = f"dataset does not contain all formal splits: {absent}"
        (report.errors if formal else report.warnings).append(message)
    if formal and expected_scenarios is not None:
        missing_scenarios = sorted(set(expected_scenarios) - set(scenario_occurrences))
        extra_scenarios = sorted(set(scenario_occurrences) - set(expected_scenarios))
        duplicate_scenarios = sorted(
            scenario_id for scenario_id, count in scenario_occurrences.items() if count != 1
        )
        if missing_scenarios:
            report.errors.append(f"formal dataset is missing planned scenarios: {missing_scenarios}")
        if extra_scenarios:
            report.errors.append(f"formal dataset has extra scenarios: {extra_scenarios}")
        if duplicate_scenarios:
            report.errors.append(
                f"formal dataset requires exactly one episode per planned scenario: {duplicate_scenarios}"
            )
        if set(obstacle_contracts) != {"static_only", "dynamic"}:
            report.errors.append("formal dataset must contain both static_only and dynamic episodes")
    formal_label_source = "realized_timestamp_aligned_counterfactual_v1"
    if formal and set(label_sources) != {formal_label_source}:
        report.errors.append(
            "formal candidate supervision requires only "
            f"{formal_label_source!r}, observed {sorted(label_sources)}"
        )
    if formal and realized_lineage_errors:
        report.errors.extend(realized_lineage_errors)
    elif "nominal_preaction_rollout_summary_v1" in label_sources:
        report.warnings.append(
            "candidate labels are nominal-rollout development supervision, not formal realized-track evidence"
        )
    report.ok = not report.errors
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--loss-window", type=int, default=16)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--scenario-manifest")
    parser.add_argument("--config-root")
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    report = validate_dataset(
        args.dataset_root, args.loss_window, formal=args.formal,
        scenario_manifest_path=args.scenario_manifest, config_root=args.config_root,
    )
    payload = json.dumps(asdict(report), indent=2, sort_keys=True) + "\n"
    if args.output_json:
        Path(args.output_json).write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
