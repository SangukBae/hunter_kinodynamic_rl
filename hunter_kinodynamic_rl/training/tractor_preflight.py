#!/usr/bin/env python3
"""Read-only readiness gate for TRACTOR development or formal runs."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import List

import yaml

from hunter_kinodynamic_rl.config.loader import default_config_root, load_profile
from hunter_kinodynamic_rl.config.tractor import (
    formal_research_implementation_readiness, load_tractor_contract,
    tractor_profile_model_mismatches,
)
from hunter_kinodynamic_rl.training.tractor_dataset_validation import validate_dataset
from hunter_kinodynamic_rl.training.tractor_scenario_plan import (
    build_scenario_plan, validate_materialized_scenario_manifest,
)


@dataclass
class TractorPreflightReport:
    ok: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    mode: str = "development"
    variant_id: str | None = None
    model_fingerprint: str | None = None
    contract_sha256: str | None = None
    protocol_status: str | None = None
    protocol_version: str | None = None
    protocol_sha256: str | None = None
    scenario_plan_sha256: str | None = None
    scenario_count: int = 0
    scenario_feasibility_verified: bool = False
    formal_implementation_ready: bool = False
    formal_implementation_gaps: List[str] = field(default_factory=list)
    materialized_scenario_manifest_ok: bool | None = None
    dataset_ok: bool | None = None
    torch_version: str | None = None
    cuda_available: bool = False


def run_tractor_preflight(
    *, config_root: str | None = None, dataset_root: str | None = None,
    scenario_manifest: str | None = None, mode: str = "development",
    device: str = "cpu", variant: str = "a7",
) -> TractorPreflightReport:
    report = TractorPreflightReport(mode=mode)
    if mode not in {"development", "formal"}:
        report.errors.append("mode must be development or formal")
        report.ok = False
        return report
    try:
        contract = load_tractor_contract(config_root, variant)
        model = contract["model"]
        report.variant_id = model.variant_id
        report.model_fingerprint = model.fingerprint()
        report.contract_sha256 = contract["contract_sha256"]
        report.protocol_status = str(contract["campaign"].get("protocol_status", "missing"))
        report.protocol_version = str(contract["protocol"]["protocol_version"])
        report.protocol_sha256 = str(contract["protocol_sha256"])
        expected = {
            "tractor_core_v1": (0, 1),
            "tractor_residual_v1": (1, 1),
            "tractor_ensemble_v1": (3, 3),
        }
        if model.variant_id not in expected:
            report.errors.append(f"unregistered variant_id {model.variant_id!r}")
        elif (model.residual_members, model.risk_members) != expected[model.variant_id]:
            report.errors.append("variant residual/risk member counts violate the registered A7/A8/A9 map")
        config_path = Path(config_root or default_config_root())
        static_profile = load_profile("tractor_local_static", str(config_path))
        dynamic_profile = load_profile("tractor_local_dynamic", str(config_path))
        if static_profile.scenario.dynamic_obstacle_count != 0 or dynamic_profile.scenario.dynamic_obstacle_count <= 0:
            report.errors.append("static/dynamic collection profiles do not separate obstacle motion")
        if static_profile.robot != dynamic_profile.robot or static_profile.action_space != dynamic_profile.action_space:
            report.errors.append("static/dynamic profiles changed robot or action contract")
        for profile_name, profile in (
            ("tractor_local_static", static_profile),
            ("tractor_local_dynamic", dynamic_profile),
        ):
            mismatches = tractor_profile_model_mismatches(profile, model)
            for field, (expected, observed) in sorted(mismatches.items()):
                report.errors.append(
                    f"{profile_name} physical contract mismatch for {field}: "
                    f"profile={expected!r}, model={observed!r}"
                )
        for manifest_name in ("scenarios_static.yaml", "scenarios_dynamic.yaml"):
            manifest = yaml.safe_load((contract["root"] / manifest_name).read_text())
            if manifest.get("schema_id") != "tractor_scenario_manifest_v1" or not manifest.get("scenarios"):
                report.errors.append(f"invalid or empty {manifest_name}")
        scenario_plan = build_scenario_plan(
            str(config_path), validate_feasibility=(mode == "formal"),
        )
        report.scenario_plan_sha256 = str(scenario_plan["manifest_sha256"])
        report.scenario_count = len(scenario_plan["entries"])
        report.scenario_feasibility_verified = mode == "formal"
        if report.scenario_plan_sha256 != contract["scenario_manifest_sha256"]:
            report.errors.append("scenario plan does not match the frozen contract")
        if scenario_manifest is None:
            if mode == "formal":
                report.errors.append("formal mode requires --scenario-manifest")
        else:
            try:
                validate_materialized_scenario_manifest(scenario_manifest, str(config_path))
                report.materialized_scenario_manifest_ok = True
            except Exception as error:
                report.materialized_scenario_manifest_ok = False
                report.errors.append(f"scenario manifest validation failed: {error}")
        if mode == "formal" and report.protocol_status != "frozen":
            report.errors.append("formal mode requires campaign.protocol_status=frozen")
        readiness = formal_research_implementation_readiness()
        report.formal_implementation_ready = bool(readiness["ready"])
        report.formal_implementation_gaps = list(readiness["gaps"])
        if mode == "formal" and not report.formal_implementation_ready:
            report.errors.extend(
                f"formal implementation gap: {gap}"
                for gap in report.formal_implementation_gaps
            )
    except Exception as error:
        report.errors.append(f"contract validation failed: {error}")
    try:
        import torch
        report.torch_version = torch.__version__
        report.cuda_available = torch.cuda.is_available()
        if device.startswith("cuda") and not report.cuda_available:
            report.errors.append(f"requested device {device!r} but CUDA is unavailable")
    except Exception as error:
        report.errors.append(f"PyTorch import failed: {error}")
    if dataset_root is None:
        message = "dataset validation skipped; pass --dataset-root before any training run"
        (report.errors if mode == "formal" else report.warnings).append(message)
    else:
        dataset = validate_dataset(
            dataset_root, loss_window=16, formal=(mode == "formal"),
            scenario_manifest_path=scenario_manifest, config_root=config_root,
        )
        report.dataset_ok = dataset.ok
        if not dataset.ok:
            report.errors.extend(f"dataset: {message}" for message in dataset.errors)
        report.warnings.extend(f"dataset: {message}" for message in dataset.warnings)
    report.ok = not report.errors
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-root")
    parser.add_argument("--dataset-root")
    parser.add_argument("--scenario-manifest")
    parser.add_argument("--mode", choices=("development", "formal"), default="development")
    parser.add_argument("--variant", choices=("a7", "a8", "a9"), default="a7")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    report = run_tractor_preflight(
        config_root=args.config_root, dataset_root=args.dataset_root,
        scenario_manifest=args.scenario_manifest,
        mode=args.mode, device=args.device, variant=args.variant,
    )
    payload = json.dumps(asdict(report), indent=2, sort_keys=True) + "\n"
    if args.output_json:
        Path(args.output_json).write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
