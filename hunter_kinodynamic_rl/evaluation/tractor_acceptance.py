#!/usr/bin/env python3
"""Evaluate locked TRACTOR results against the frozen promotion protocol."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from hunter_kinodynamic_rl.config.tractor import load_tractor_protocol
from hunter_kinodynamic_rl.evaluation.tractor_metrics import paired_bootstrap_interval


@dataclass
class AcceptanceReport:
    ok: bool = False
    protocol_version: str = ""
    protocol_sha256: str = ""
    method_id: str = ""
    baseline_id: str = ""
    seed_count: int = 0
    scenario_count_per_seed: int = 0
    gate_results: list = field(default_factory=list)
    runtime_results: list = field(default_factory=list)
    errors: list = field(default_factory=list)


def _seed_means(records: Sequence[Mapping], method: str, metric: str) -> dict[int, float]:
    grouped = defaultdict(list)
    for record in records:
        if record.get("method_id") != method:
            continue
        value = record.get("metrics", {}).get(metric)
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"{method} has missing/non-finite {metric} for {record.get('scenario_id')}")
        grouped[int(record["seed"])].append(float(value))
    return {seed: float(np.mean(values)) for seed, values in grouped.items()}


def _bound_pass(interval: Mapping, bound: str, threshold: float) -> bool:
    value = interval[bound]
    if value is None:
        return False
    return float(value) >= threshold if bound == "low" else float(value) <= threshold


def _finite_float(value, default: float = math.inf) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return default
    return converted if math.isfinite(converted) else default


def _integer(value, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def evaluate_acceptance(
    records: Sequence[Mapping], runtime_metrics: Mapping, *, config_root: str | None = None,
    method: str | None = None, baseline: str | None = None,
) -> AcceptanceReport:
    protocol = load_tractor_protocol(config_root)
    statistics = protocol["statistics"]
    support = protocol["support_gates"]
    method = method or protocol["primary_method"]
    baseline = baseline or protocol["primary_baseline"]
    report = AcceptanceReport(
        protocol_version=protocol["protocol_version"],
        protocol_sha256=protocol["frozen_payload_sha256"], method_id=method, baseline_id=baseline,
    )
    allowed_methods = {method, baseline}
    filtered = [record for record in records if record.get("method_id") in allowed_methods]
    required_split = statistics["required_split"]
    keys = defaultdict(int)
    scenarios_by_method_seed = defaultdict(set)
    families = set()
    for index, record in enumerate(filtered):
        try:
            seed = int(record["seed"])
            scenario = str(record["scenario_id"])
            split_id = str(record["split_id"])
            complete = bool(record["complete"])
            family = str(record["scenario_family"])
            obstacle_contract = str(record["obstacle_contract"])
        except (KeyError, TypeError, ValueError) as error:
            report.errors.append(f"record {index} has incomplete acceptance provenance: {error}")
            continue
        key = (str(record["method_id"]), seed, scenario)
        keys[key] += 1
        scenarios_by_method_seed[(key[0], seed)].add(scenario)
        families.add((family, obstacle_contract))
        if split_id != required_split:
            report.errors.append(f"record {key} is not from {required_split}")
        if not complete:
            report.errors.append(f"record {key} is incomplete")
    duplicates = sorted(key for key, count in keys.items() if count != 1)
    if duplicates:
        report.errors.append(f"duplicate method-seed-scenario rows: {duplicates}")
    method_seeds = {seed for name, seed in scenarios_by_method_seed if name == method}
    baseline_seeds = {seed for name, seed in scenarios_by_method_seed if name == baseline}
    if method_seeds != baseline_seeds:
        report.errors.append("method and baseline seed sets differ")
    report.seed_count = len(method_seeds)
    if report.seed_count < int(statistics["minimum_unique_seeds"]):
        report.errors.append(
            f"only {report.seed_count} independent seeds; "
            f"need {statistics['minimum_unique_seeds']}"
        )
    counts = [len(scenarios_by_method_seed[(method, seed)]) for seed in method_seeds]
    report.scenario_count_per_seed = min(counts, default=0)
    minimum_scenarios = int(support["minimum_locked_scenarios_per_seed"])
    if report.scenario_count_per_seed < minimum_scenarios:
        report.errors.append(
            f"only {report.scenario_count_per_seed} locked scenarios per seed; need {minimum_scenarios}"
        )
    for seed in method_seeds:
        if scenarios_by_method_seed[(method, seed)] != scenarios_by_method_seed[(baseline, seed)]:
            report.errors.append(f"method/baseline scenario sets differ for seed {seed}")
    if support["require_static_and_dynamic"]:
        obstacle_contracts = {item[1] for item in families}
        if not {"static_only", "dynamic"}.issubset(obstacle_contracts):
            report.errors.append("locked records must contain both static_only and dynamic scenarios")

    confidence = float(statistics["confidence"])
    samples = int(statistics["bootstrap_samples"])
    bootstrap_seed = int(statistics["bootstrap_seed"])
    for gate in protocol["headline_gates"]:
        result = {"id": gate["id"], "metric": gate["metric"], "passed": False}
        try:
            candidate = _seed_means(filtered, method, gate["metric"])
            reference = _seed_means(filtered, baseline, gate["metric"])
            paired_seeds = sorted(set(candidate) & set(reference))
            candidate_values = [candidate[seed] for seed in paired_seeds]
            absolute = paired_bootstrap_interval(
                candidate_values, confidence=confidence, bootstrap_samples=samples,
                seed=bootstrap_seed,
            )
            if gate["paired_transform"] == "additive_difference":
                effects = [candidate[seed] - reference[seed] for seed in paired_seeds]
            else:
                if any(abs(reference[seed]) <= 1e-12 for seed in paired_seeds):
                    raise ValueError("relative-change baseline contains zero")
                effects = [candidate[seed] / reference[seed] - 1.0 for seed in paired_seeds]
            paired = paired_bootstrap_interval(
                effects, confidence=confidence, bootstrap_samples=samples,
                seed=bootstrap_seed,
            )
            absolute_ok = True
            if gate["absolute_bound"] != "none":
                bound = "low" if gate["absolute_bound"] == "lower" else "high"
                absolute_ok = _bound_pass(absolute, bound, float(gate["absolute_threshold"]))
            paired_bound = "low" if gate["paired_bound"] == "lower" else "high"
            paired_ok = _bound_pass(paired, paired_bound, float(gate["paired_threshold"]))
            result.update({
                "absolute_interval": absolute, "paired_interval": paired,
                "paired_transform": gate["paired_transform"],
                "absolute_passed": absolute_ok, "paired_passed": paired_ok,
                "passed": absolute_ok and paired_ok,
            })
        except (KeyError, TypeError, ValueError) as error:
            result["error"] = str(error)
        report.gate_results.append(result)

    runtime = protocol["runtime_gates"]
    runtime_checks = [
        ("target_hardware", runtime_metrics.get("target_hardware") is True),
        ("timed_decisions", _integer(runtime_metrics.get("timed_decisions")) >= int(runtime["minimum_timed_decisions"])),
        ("p99_latency_ms", _finite_float(runtime_metrics.get("p99_latency_ms")) <= float(runtime["p99_latency_ms_max"])),
        ("deadline_miss_rate", _finite_float(runtime_metrics.get("deadline_miss_rate")) <= float(runtime["deadline_miss_rate_max"])),
    ]
    for name, passed in runtime_checks:
        report.runtime_results.append({"id": name, "passed": bool(passed)})
    report.ok = (
        not report.errors
        and bool(report.gate_results)
        and all(item["passed"] for item in report.gate_results)
        and all(item["passed"] for item in report.runtime_results)
    )
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-json", required=True)
    parser.add_argument("--runtime-json", required=True)
    parser.add_argument("--config-root")
    parser.add_argument("--method")
    parser.add_argument("--baseline")
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    records = json.loads(Path(args.records_json).read_text(encoding="utf-8"))
    runtime = json.loads(Path(args.runtime_json).read_text(encoding="utf-8"))
    report = evaluate_acceptance(
        records, runtime, config_root=args.config_root, method=args.method, baseline=args.baseline,
    )
    payload = json.dumps(asdict(report), indent=2, sort_keys=True) + "\n"
    if args.output_json:
        Path(args.output_json).write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
