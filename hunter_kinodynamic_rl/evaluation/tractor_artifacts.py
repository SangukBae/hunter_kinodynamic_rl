"""Fail-closed validation and aggregation of formal TRACTOR run artifacts."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Mapping, Sequence

from .tractor_metrics import paired_bootstrap_interval


REQUIRED_RUN_FIELDS = {
    "experiment_id", "method_id", "seed", "scenario_id", "split_id",
    "model_fingerprint", "data_fingerprint", "training_fingerprint",
    "complete", "metrics", "artifact_sha256",
}


def validate_complete_matrix(
    records: Sequence[Mapping], expected_methods: Sequence[str],
    expected_seeds: Sequence[int], expected_scenarios: Sequence[str],
) -> dict:
    errors = []
    seen = Counter()
    fingerprints = {"data": set(), "training_by_method": {}, "model_by_method": {}}
    for index, record in enumerate(records):
        missing = sorted(REQUIRED_RUN_FIELDS - set(record))
        if missing:
            errors.append(f"record {index} missing fields {missing}")
            continue
        key = (record["method_id"], int(record["seed"]), record["scenario_id"])
        seen[key] += 1
        if not record["complete"]:
            errors.append(f"incomplete registered run {key}")
        if record["split_id"] != "locked_test":
            errors.append(f"formal aggregate contains non-locked split for {key}")
        fingerprints["data"].add(record["data_fingerprint"])
        fingerprints["training_by_method"].setdefault(record["method_id"], set()).add(record["training_fingerprint"])
        fingerprints["model_by_method"].setdefault(record["method_id"], set()).add(record["model_fingerprint"])
    expected = {
        (method, int(seed), scenario)
        for method in expected_methods for seed in expected_seeds for scenario in expected_scenarios
    }
    missing_rows = sorted(expected - set(seen))
    duplicates = sorted(key for key, count in seen.items() if count != 1)
    if missing_rows:
        errors.append(f"missing seed-scenario rows: {missing_rows}")
    if duplicates:
        errors.append(f"duplicate seed-scenario rows: {duplicates}")
    if len(fingerprints["data"]) > 1:
        errors.append(f"mixed data fingerprints: {sorted(fingerprints['data'])}")
    for kind in ("training_by_method", "model_by_method"):
        for method, values in fingerprints[kind].items():
            if len(values) > 1:
                errors.append(f"mixed {kind} fingerprints for {method}: {sorted(values)}")
    return {"ok": not errors, "errors": errors, "expected_rows": len(expected), "observed_rows": len(records)}


def paired_method_effect(records: Sequence[Mapping], method: str, baseline: str, metric: str) -> dict:
    lookup = {
        (record["method_id"], int(record["seed"]), record["scenario_id"]): record
        for record in records
    }
    differences_by_seed = defaultdict(list)
    for (record_method, seed, scenario), record in lookup.items():
        if record_method != method:
            continue
        other = lookup.get((baseline, seed, scenario))
        if other is None:
            raise ValueError(f"missing paired baseline row for seed={seed}, scenario={scenario}")
        value, baseline_value = record["metrics"].get(metric), other["metrics"].get(metric)
        if value is None or baseline_value is None:
            raise ValueError(f"missing metric {metric!r} in paired rows")
        differences_by_seed[seed].append(float(value) - float(baseline_value))
    seed_differences = [
        sum(values) / len(values) for _seed, values in sorted(differences_by_seed.items())
    ]
    report = paired_bootstrap_interval(seed_differences, seed=1729)
    report["scenario_pair_count"] = sum(len(values) for values in differences_by_seed.values())
    report["replication_unit"] = "independent_training_seed"
    return report
