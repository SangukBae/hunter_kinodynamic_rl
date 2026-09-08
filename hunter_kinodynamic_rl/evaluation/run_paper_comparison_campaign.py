#!/usr/bin/env python3
"""Fail-closed orchestration for the B1--B8/A7--A9 paper comparison."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile

from hunter_kinodynamic_rl.config.comparison import load_comparison_contract
from hunter_kinodynamic_rl.config.tractor import (
    canonical_sha256, formal_research_implementation_readiness, load_tractor_contract,
    require_formal_research_implementation_ready,
)
from hunter_kinodynamic_rl.evaluation.evaluate_paper_comparison import (
    PAPER_METHODS, evaluate_method_seed,
)
from hunter_kinodynamic_rl.evaluation.fit_comparison_calibration import (
    CALIBRATED_METHODS, fit_comparison_calibration,
)
from hunter_kinodynamic_rl.evaluation.tractor_acceptance import evaluate_acceptance
from hunter_kinodynamic_rl.evaluation.tractor_artifacts import (
    paired_method_effect, paired_nested_method_effect, validate_complete_matrix,
)
from hunter_kinodynamic_rl.training.collect_formal_comparison_data import (
    collect_formal_comparison_data,
)
from hunter_kinodynamic_rl.training.train_comparison_baselines import (
    train_comparison_baseline,
)
from hunter_kinodynamic_rl.training.train_tractor_tqc import train_from_sequence_replay
from hunter_kinodynamic_rl.training.tractor_dataset_validation import validate_dataset
from hunter_kinodynamic_rl.training.tractor_scenario_plan import (
    materialize_scenario_plan, validate_materialized_scenario_manifest,
)


CAMPAIGN_SCHEMA = "tractor_paper_comparison_campaign_v2"
SYSTEM_AXIS_FIELDS = ("vehicle_axis", "sensor_axis", "localization_axis", "system_domain")
HEADLINE_METRICS = (
    "success_rate", "collision_rate", "time_to_goal_sec_mean", "min_clearance_m_mean",
)


def _method_contract_sha256(config_root: str | None = None) -> dict[str, str]:
    result = {}
    for method in PAPER_METHODS:
        contract = (
            load_tractor_contract(config_root, method.lower())
            if method.startswith("A") else load_comparison_contract(config_root, method)
        )
        result[method] = str(contract["contract_sha256"])
    return result


def _require_formal_implementation_ready() -> None:
    require_formal_research_implementation_ready("formal paper campaign")


def _atomic_json(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"immutable artifact exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def _paths(campaign_root: str | Path) -> dict[str, Path]:
    root = Path(campaign_root).resolve()
    return {
        "root": root, "manifest": root / "campaign_manifest.json",
        "scenarios": root / "scenarios", "scenario_manifest": root / "scenarios" / "manifest.json",
        "dataset": root / "dataset", "runs": root / "runs",
        "calibration": root / "calibration", "evaluation": root / "evaluation",
        "aggregate": root / "aggregate",
    }


def prepare_campaign(campaign_root: str | Path, config_root: str | None = None) -> dict:
    paths = _paths(campaign_root)
    if paths["root"].exists() and any(paths["root"].iterdir()):
        raise FileExistsError(f"fresh campaign refuses non-empty root: {paths['root']}")
    paths["root"].mkdir(parents=True, exist_ok=True)
    scenarios = materialize_scenario_plan(paths["scenarios"], config_root)
    contract = load_tractor_contract(config_root, "a7")
    readiness = formal_research_implementation_readiness()
    payload = {
        "schema_id": CAMPAIGN_SCHEMA,
        "protocol_version": contract["protocol"]["protocol_version"],
        "protocol_sha256": contract["protocol_sha256"],
        "methods": list(PAPER_METHODS), "calibrated_methods": list(CALIBRATED_METHODS),
        "method_contract_sha256": _method_contract_sha256(config_root),
        "seeds": list(contract["campaign"]["seeds"]),
        "fixed_update_budget": int(contract["campaign"]["fixed_update_budget"]),
        "stage_update_budgets": dict(contract["campaign"]["stage_update_budgets"]),
        "batch_size": int(contract["campaign"]["batch_size"]),
        "scenario_plan_manifest_sha256": scenarios["plan_manifest_sha256"],
        "scenario_artifact_manifest_sha256": scenarios["artifact_manifest_sha256"],
        "expected_locked_records": (
            len(PAPER_METHODS) * len(contract["campaign"]["seeds"])
            * int(contract["protocol"]["support_gates"]["minimum_locked_scenarios_per_seed"])
        ),
        "formal_implementation_ready": bool(readiness["ready"]),
        "formal_implementation_gaps": list(readiness["gaps"]),
        "evidence_status": (
            "prepared_untrained" if readiness["ready"]
            else "prepared_blocked_implementation_gaps"
        ),
    }
    payload["campaign_manifest_sha256"] = canonical_sha256(payload)
    _atomic_json(paths["manifest"], payload)
    return {**payload, "campaign_root": str(paths["root"])}


def load_campaign(campaign_root: str | Path, config_root: str | None = None) -> tuple[dict, dict[str, Path]]:
    paths = _paths(campaign_root)
    payload = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    digest = payload.get("campaign_manifest_sha256")
    check = dict(payload)
    check.pop("campaign_manifest_sha256", None)
    if payload.get("schema_id") != CAMPAIGN_SCHEMA or digest != canonical_sha256(check):
        raise RuntimeError("campaign manifest schema or checksum mismatch")
    contract = load_tractor_contract(config_root, "a7")
    readiness = formal_research_implementation_readiness()
    expected = {
        "protocol_version": contract["protocol"]["protocol_version"],
        "protocol_sha256": contract["protocol_sha256"],
        "methods": list(PAPER_METHODS), "calibrated_methods": list(CALIBRATED_METHODS),
        "method_contract_sha256": _method_contract_sha256(config_root),
        "seeds": list(contract["campaign"]["seeds"]),
        "fixed_update_budget": int(contract["campaign"]["fixed_update_budget"]),
        "stage_update_budgets": dict(contract["campaign"]["stage_update_budgets"]),
        "formal_implementation_ready": bool(readiness["ready"]),
        "formal_implementation_gaps": list(readiness["gaps"]),
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise RuntimeError(f"campaign manifest changed frozen field {name!r}")
    scenarios = validate_materialized_scenario_manifest(paths["scenario_manifest"], config_root)
    if scenarios["artifact_manifest_sha256"] != payload["scenario_artifact_manifest_sha256"]:
        raise RuntimeError("campaign scenario artifacts changed")
    return payload, paths


def _selection(campaign: dict, methods, seeds) -> tuple[list[str], list[int]]:
    selected_methods = list(methods or campaign["methods"])
    selected_seeds = [int(seed) for seed in (seeds or campaign["seeds"])]
    unknown_methods = sorted(set(selected_methods) - set(campaign["methods"]))
    unknown_seeds = sorted(set(selected_seeds) - set(campaign["seeds"]))
    if unknown_methods or unknown_seeds:
        raise ValueError(f"unregistered campaign selection methods={unknown_methods}, seeds={unknown_seeds}")
    return selected_methods, selected_seeds


def collect_campaign_data(
    campaign_root: str | Path, *, profile_name: str = "tractor_local_dynamic",
    behavior_seed: int = 739391, config_root: str | None = None, resume: bool = False,
) -> dict:
    _require_formal_implementation_ready()
    _campaign, paths = load_campaign(campaign_root, config_root)
    return collect_formal_comparison_data(
        scenario_manifest_path=paths["scenario_manifest"], dataset_root=paths["dataset"],
        profile_name=profile_name, split_id="all", behavior_seed=behavior_seed,
        config_root=config_root, resume=resume,
    )


def train_campaign(
    campaign_root: str | Path, *, methods=None, seeds=None, updates: int | None = None,
    batch_size: int | None = None, device: str = "cpu", config_root: str | None = None,
    skip_complete: bool = False,
) -> dict:
    _require_formal_implementation_ready()
    campaign, paths = load_campaign(campaign_root, config_root)
    selected_methods, selected_seeds = _selection(campaign, methods, seeds)
    if updates is not None and int(updates) != int(campaign["fixed_update_budget"]):
        raise ValueError("formal campaign cannot override the frozen total update budget")
    if batch_size is not None and int(batch_size) != int(campaign["batch_size"]):
        raise ValueError("formal campaign cannot override the frozen batch size")
    data_report = validate_dataset(
        paths["dataset"], formal=True, scenario_manifest_path=paths["scenario_manifest"],
        config_root=config_root,
    )
    if not data_report.ok:
        raise RuntimeError(f"formal dataset is not ready: {data_report.errors}")
    results = []
    for method in selected_methods:
        for seed in selected_seeds:
            run_root = paths["runs"] / method / f"seed-{seed}"
            final_pointer = run_root / "checkpoints" / "final"
            if final_pointer.is_symlink():
                if skip_complete:
                    results.append({"method_id": method, "seed": seed, "status": "skipped_complete"})
                    continue
                raise FileExistsError(f"final checkpoint already exists: {final_pointer}")
            common = {
                "dataset_root": paths["dataset"],
                "seed": seed,
                "batch_size": batch_size or int(campaign["batch_size"]), "device": device,
                "config_root": config_root, "formal_data": True,
                "scenario_manifest_path": paths["scenario_manifest"],
            }
            if method.startswith("A"):
                phase_results = []
                previous_root = None
                for stage in ("stage3", "stage4", "stage5"):
                    phase_root = (
                        run_root if stage == "stage5" else run_root / "phases" / stage
                    )
                    phase_final = phase_root / "checkpoints" / "final"
                    if phase_final.is_symlink():
                        phase_results.append({"training_stage": stage, "status": "skipped_complete"})
                    else:
                        latest = phase_root / "checkpoints" / "latest"
                        phase_results.append(train_from_sequence_replay(
                            variant=method.lower(), run_root=phase_root,
                            updates=int(campaign["stage_update_budgets"][stage]),
                            training_stage=stage, resume=latest.is_symlink(),
                            warm_start_checkpoint_root=(
                                None if latest.is_symlink() or previous_root is None
                                else previous_root / "checkpoints"
                            ),
                            **common,
                        ))
                    previous_root = phase_root
                result = {"method_id": method, "seed": seed, "phases": phase_results}
            else:
                latest = run_root / "checkpoints" / "latest"
                result = train_comparison_baseline(
                    method_id=method, run_root=run_root,
                    updates=int(campaign["fixed_update_budget"]), resume=latest.is_symlink(),
                    **common,
                )
            results.append(result)
    return {"phase": "train", "runs": results, "performance_claim": "none_training_only"}


def calibrate_campaign(
    campaign_root: str | Path, *, methods=None, seeds=None, device: str = "cpu",
    config_root: str | None = None, skip_complete: bool = False,
) -> dict:
    _require_formal_implementation_ready()
    campaign, paths = load_campaign(campaign_root, config_root)
    requested = methods or campaign["calibrated_methods"]
    selected_methods, selected_seeds = _selection(campaign, requested, seeds)
    invalid = sorted(set(selected_methods) - set(CALIBRATED_METHODS))
    if invalid:
        raise ValueError(f"methods have no probability calibration contract: {invalid}")
    results = []
    for method in selected_methods:
        for seed in selected_seeds:
            artifact_id = f"{method.lower()}-seed-{seed}-platt-v1"
            output_root = paths["calibration"] / method / f"seed-{seed}"
            artifact = output_root / f"{artifact_id}.json"
            if artifact.is_file():
                if skip_complete:
                    results.append({"method_id": method, "seed": seed, "status": "skipped_complete"})
                    continue
                raise FileExistsError(f"calibration artifact already exists: {artifact}")
            results.append(fit_comparison_calibration(
                method_id=method,
                checkpoint_root=paths["runs"] / method / f"seed-{seed}" / "checkpoints",
                dataset_root=paths["dataset"], scenario_manifest_path=paths["scenario_manifest"],
                output_root=output_root, seed=seed, device=device, config_root=config_root,
                artifact_id=artifact_id,
            ))
    return {"phase": "calibrate", "runs": results, "performance_claim": "none_calibration_only"}


def evaluate_campaign(
    campaign_root: str | Path, *, methods=None, seeds=None, device: str = "cpu",
    profile_name: str = "tractor_local_dynamic", config_root: str | None = None,
    skip_complete: bool = False,
) -> dict:
    _require_formal_implementation_ready()
    campaign, paths = load_campaign(campaign_root, config_root)
    selected_methods, selected_seeds = _selection(campaign, methods, seeds)
    results = []
    for method in selected_methods:
        for seed in selected_seeds:
            output = paths["evaluation"] / method / f"seed-{seed}.jsonl"
            if output.is_file():
                if skip_complete:
                    results.append({"method_id": method, "seed": seed, "status": "skipped_complete"})
                    continue
                raise FileExistsError(f"evaluation artifact already exists: {output}")
            calibration = None
            if method in CALIBRATED_METHODS:
                calibration = (
                    paths["calibration"] / method / f"seed-{seed}"
                    / f"{method.lower()}-seed-{seed}-platt-v1.json"
                )
            results.append(evaluate_method_seed(
                method_id=method, seed=seed,
                checkpoint_root=paths["runs"] / method / f"seed-{seed}" / "checkpoints",
                calibration_artifact=calibration, dataset_root=paths["dataset"],
                scenario_manifest_path=paths["scenario_manifest"], output_jsonl=output,
                profile_name=profile_name, device=device, config_root=config_root,
            ))
    return {"phase": "evaluate", "runs": results, "evidence_scope": "locked_test_simulation"}


def _read_records(paths: dict[str, Path], campaign: dict) -> list[dict]:
    records = []
    for method in campaign["methods"]:
        for seed in campaign["seeds"]:
            path = paths["evaluation"] / method / f"seed-{seed}.jsonl"
            if not path.is_file():
                continue
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    stored_digest = record.get("artifact_sha256")
                    digest_payload = dict(record)
                    digest_payload.pop("artifact_sha256", None)
                    if stored_digest != canonical_sha256(digest_payload):
                        raise RuntimeError(f"evaluation record checksum mismatch: {path}")
                    records.append(record)
    return records


def _axis_stratified_effects(records: list[dict], methods: list[str]) -> dict:
    """Keep system-OOD claims separate instead of hiding them in one pooled mean."""
    missing = [
        record.get("experiment_id", f"row-{index}")
        for index, record in enumerate(records)
        if any(field not in record for field in SYSTEM_AXIS_FIELDS)
    ]
    if missing:
        raise RuntimeError(f"evaluation records are missing frozen system axes: {missing}")
    reports = {}
    for field in SYSTEM_AXIS_FIELDS:
        levels = sorted({str(record[field]) for record in records})
        reports[field] = {}
        for level in levels:
            selected = [record for record in records if str(record[field]) == level]
            reports[field][level] = {
                "record_count": len(selected),
                "scenario_count": len({record["scenario_id"] for record in selected}),
                "seed_count": len({int(record["seed"]) for record in selected}),
                "method_record_count": {
                    method: sum(record["method_id"] == method for record in selected)
                    for method in methods
                },
                "paired_effects_vs_B1": {
                    method: {
                        metric: paired_method_effect(selected, method, "B1", metric)
                        for metric in HEADLINE_METRICS
                    }
                    for method in methods if method != "B1"
                },
            }
    return reports


def _hypothesis_aggregate(records: list[dict], methods: list[str]) -> dict:
    support = {}
    for method in methods:
        selected = [record for record in records if record["method_id"] == method]
        support[method] = {
            "episode_count": len(selected),
            "h1_available_episodes": sum(
                bool(record["metrics"]["h1_prediction"].get("available"))
                for record in selected
            ),
            "h1_valid_cells": sum(
                int(record["metrics"]["h1_prediction"].get("valid_cells", 0))
                for record in selected
            ),
            "h2_available_episodes": sum(
                bool(record["metrics"]["h2_ranking"].get("available"))
                for record in selected
            ),
            "h2_valid_rows": sum(
                int(record["metrics"]["h2_ranking"].get("row_count", 0))
                for record in selected
            ),
            "h3_available_episodes": sum(
                bool(record["metrics"]["h3_risk"].get("available"))
                for record in selected
            ),
            "h3_candidate_count": sum(
                int(record["metrics"]["h3_risk"].get("count", 0))
                for record in selected
            ),
            "h3_event_count": sum(
                int(record["metrics"]["h3_risk"].get("event_count", 0))
                for record in selected
            ),
        }
    comparison_plan = {
        "H1": {
            "pairs": (("A7", "B3"), ("A7", "B4"), ("A7", "B5")),
            "metrics": ("occupancy_nll", "flow_epe", "tube_oob_mae"),
            "family": "h1_prediction",
        },
        "H2": {
            "pairs": (("A7", "B4"), ("A7", "B5"), ("A7", "B7")),
            "metrics": ("mean_regret", "ndcg", "unsafe_top1_rate"),
            "family": "h2_ranking",
        },
        "H3": {
            "pairs": (("A7", "B8"),),
            "metrics": ("brier", "nll", "ece"),
            "family": "h3_risk",
        },
    }
    effects = {}
    for hypothesis, plan in comparison_plan.items():
        effects[hypothesis] = {}
        for method, baseline in plan["pairs"]:
            effects[hypothesis][f"{method}_vs_{baseline}"] = {
                metric: paired_nested_method_effect(
                    records, method, baseline,
                    ("metrics", plan["family"], metric),
                )
                for metric in plan["metrics"]
            }
    return {"support_by_method": support, "paired_effects": effects}


def aggregate_campaign(
    campaign_root: str | Path, *, runtime_json: str | Path | None = None,
    config_root: str | None = None,
) -> dict:
    _require_formal_implementation_ready()
    campaign, paths = load_campaign(campaign_root, config_root)
    scenario_manifest = validate_materialized_scenario_manifest(
        paths["scenario_manifest"], config_root,
    )
    locked_scenarios = [
        item["scenario_id"] for item in scenario_manifest["entries"]
        if item["split_id"] == "locked_test"
    ]
    records = _read_records(paths, campaign)
    matrix = validate_complete_matrix(
        records, campaign["methods"], campaign["seeds"], locked_scenarios,
    )
    if not matrix["ok"]:
        raise RuntimeError(f"formal comparison matrix is incomplete: {matrix['errors']}")
    effects = {}
    for method in campaign["methods"]:
        if method == "B1":
            continue
        effects[method] = {
            metric: paired_method_effect(records, method, "B1", metric)
            for metric in HEADLINE_METRICS
        }
    acceptance = None
    if runtime_json is not None:
        runtime = json.loads(Path(runtime_json).read_text(encoding="utf-8"))
        acceptance = asdict(evaluate_acceptance(
            records, runtime, config_root=config_root, method="A7", baseline="B1",
        ))
    payload = {
        "schema_id": "tractor_paper_comparison_aggregate_v2",
        "campaign_manifest_sha256": campaign["campaign_manifest_sha256"],
        "matrix": matrix, "record_count": len(records),
        "paired_effects_vs_B1": effects,
        "system_axis_reports": _axis_stratified_effects(records, campaign["methods"]),
        "hypothesis_metrics": _hypothesis_aggregate(records, campaign["methods"]),
        "primary_acceptance_A7_vs_B1": acceptance,
        "evidence_scope": "locked_test_simulation_only",
    }
    payload["aggregate_sha256"] = canonical_sha256(payload)
    _atomic_json(paths["aggregate"] / "comparison.json", payload)
    return payload


def campaign_status(campaign_root: str | Path, config_root: str | None = None) -> dict:
    campaign, paths = load_campaign(campaign_root, config_root)
    training = calibration = evaluation = stage3 = stage4 = 0
    for method in campaign["methods"]:
        for seed in campaign["seeds"]:
            training += int((paths["runs"] / method / f"seed-{seed}" / "checkpoints" / "final").is_symlink())
            if method.startswith("A"):
                stage3 += int((
                    paths["runs"] / method / f"seed-{seed}" / "phases" / "stage3"
                    / "checkpoints" / "final"
                ).is_symlink())
                stage4 += int((
                    paths["runs"] / method / f"seed-{seed}" / "phases" / "stage4"
                    / "checkpoints" / "final"
                ).is_symlink())
            evaluation += int((paths["evaluation"] / method / f"seed-{seed}.jsonl").is_file())
            if method in CALIBRATED_METHODS:
                calibration += int((
                    paths["calibration"] / method / f"seed-{seed}"
                    / f"{method.lower()}-seed-{seed}-platt-v1.json"
                ).is_file())
    dataset = validate_dataset(
        paths["dataset"], formal=True, scenario_manifest_path=paths["scenario_manifest"],
        config_root=config_root,
    ) if paths["dataset"].exists() else None
    return {
        "campaign_root": str(paths["root"]),
        "formal_implementation_ready": bool(campaign["formal_implementation_ready"]),
        "formal_implementation_gaps": list(campaign["formal_implementation_gaps"]),
        "formal_dataset_ready": bool(dataset and dataset.ok),
        "training_complete": training, "training_expected": len(campaign["methods"]) * len(campaign["seeds"]),
        "tractor_stage3_complete": stage3,
        "tractor_stage3_expected": 3 * len(campaign["seeds"]),
        "tractor_stage4_complete": stage4,
        "tractor_stage4_expected": 3 * len(campaign["seeds"]),
        "calibration_complete": calibration,
        "calibration_expected": len(campaign["calibrated_methods"]) * len(campaign["seeds"]),
        "evaluation_complete": evaluation,
        "evaluation_expected": len(campaign["methods"]) * len(campaign["seeds"]),
        "aggregate_complete": (paths["aggregate"] / "comparison.json").is_file(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--config-root")
    parser.add_argument("--output-json")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    collect = subparsers.add_parser("collect")
    collect.add_argument("--profile", default="tractor_local_dynamic")
    collect.add_argument("--behavior-seed", type=int, default=739391)
    collect.add_argument("--resume", action="store_true")
    for command in ("train", "calibrate", "evaluate"):
        child = subparsers.add_parser(command)
        child.add_argument("--method", action="append")
        child.add_argument("--seed", type=int, action="append")
        child.add_argument("--device", default="cpu")
        child.add_argument("--skip-complete", action="store_true")
        if command == "train":
            child.add_argument("--updates", type=int)
            child.add_argument("--batch-size", type=int)
        if command == "evaluate":
            child.add_argument("--profile", default="tractor_local_dynamic")
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--runtime-json")
    subparsers.add_parser("status")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare_campaign(args.campaign_root, args.config_root)
    elif args.command == "collect":
        result = collect_campaign_data(
            args.campaign_root, profile_name=args.profile, behavior_seed=args.behavior_seed,
            config_root=args.config_root, resume=args.resume,
        )
    elif args.command == "train":
        result = train_campaign(
            args.campaign_root, methods=args.method, seeds=args.seed, updates=args.updates,
            batch_size=args.batch_size, device=args.device, config_root=args.config_root,
            skip_complete=args.skip_complete,
        )
    elif args.command == "calibrate":
        result = calibrate_campaign(
            args.campaign_root, methods=args.method, seeds=args.seed, device=args.device,
            config_root=args.config_root, skip_complete=args.skip_complete,
        )
    elif args.command == "evaluate":
        result = evaluate_campaign(
            args.campaign_root, methods=args.method, seeds=args.seed, device=args.device,
            profile_name=args.profile, config_root=args.config_root,
            skip_complete=args.skip_complete,
        )
    elif args.command == "aggregate":
        result = aggregate_campaign(
            args.campaign_root, runtime_json=args.runtime_json, config_root=args.config_root,
        )
    else:
        result = campaign_status(args.campaign_root, args.config_root)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        Path(args.output_json).write_text(payload, encoding="utf-8")
    print(payload, end="")
    return result


if __name__ == "__main__":
    main()
