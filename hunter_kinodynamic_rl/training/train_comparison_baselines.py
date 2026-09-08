#!/usr/bin/env python3
"""Train B1--B8 on the same immutable sequence population as TRACTOR-TQC."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

from hunter_kinodynamic_rl.common.seed import enable_torch_determinism, seed_all
from hunter_kinodynamic_rl.config.comparison import BASELINE_METHODS, load_comparison_contract
from hunter_kinodynamic_rl.config.tractor import (
    canonical_sha256, require_formal_research_implementation_ready,
)
from hunter_kinodynamic_rl.evaluation.provenance import collect_package_provenance
from hunter_kinodynamic_rl.rl.algorithms.comparison_baselines import ComparisonAgent
from hunter_kinodynamic_rl.rl.algorithms.tractor_tqc import TractorAgent
from hunter_kinodynamic_rl.rl.checkpointing.tractor import (
    load_training_generation, save_training_generation,
)
from hunter_kinodynamic_rl.rl.replay import EpisodeStore, SequenceBuffer, SequenceIndex
from hunter_kinodynamic_rl.training.comparison_sequence_training import ComparisonSequenceTrainer
from hunter_kinodynamic_rl.training.tractor_dataset_validation import validate_dataset
from hunter_kinodynamic_rl.training.tractor_dataset_manifest import (
    formal_source_identity, validate_runtime_source_against_dataset,
)
from hunter_kinodynamic_rl.training.train_tractor_tqc import _json_line


def parameter_inventory(agent) -> dict[str, int]:
    groups = agent.parameter_groups()
    return {
        name: sum(parameter.numel() for parameter in parameters)
        for name, parameters in groups.items()
    }


def train_comparison_baseline(
    *, method_id: str, dataset_root: str | Path, run_root: str | Path,
    seed: int, updates: int, batch_size: int | None = None,
    device: str = "cpu", config_root: str | None = None,
    resume: bool = False, checkpoint_tag: str = "latest",
    checkpoint_interval: int | None = None, formal_data: bool = False,
    scenario_manifest_path: str | Path | None = None,
) -> dict:
    if formal_data:
        require_formal_research_implementation_ready("formal comparison-baseline training")
    if updates <= 0:
        raise ValueError("updates must be positive")
    contract = load_comparison_contract(config_root, method_id)
    if formal_data:
        if int(updates) != int(contract["campaign"]["fixed_update_budget"]):
            raise ValueError("formal baseline training requires the frozen total update budget")
        if batch_size is not None and int(batch_size) != int(contract["campaign"]["batch_size"]):
            raise ValueError("formal baseline training cannot override the registered batch size")
        if checkpoint_interval is not None and int(checkpoint_interval) != int(
            contract["campaign"]["checkpoint_interval_updates"]
        ):
            raise ValueError("formal baseline training cannot override the checkpoint interval")
    loss_window = int(contract["data"]["loss_window"])
    report = validate_dataset(
        dataset_root, loss_window=loss_window, formal=formal_data,
        scenario_manifest_path=scenario_manifest_path, config_root=config_root,
    )
    if not report.ok:
        raise RuntimeError(f"dataset validation failed: {report.errors}")
    provenance = collect_package_provenance(execution_file=__file__)
    source_identity = {
        "package_git_commit_sha": provenance["package_git_commit_sha"],
        "tracked_diff_sha256": provenance["tracked_diff_sha256"],
        "untracked_source_manifest_sha256": provenance["untracked_source_manifest_sha256"],
        "source_content_manifest_sha256": provenance["source_content_manifest_sha256"],
        "source_content_file_count": provenance["source_content_file_count"],
        "execution_module_sha256": provenance["execution_module_sha256"],
        "container_image_digest": os.environ.get(
            "HUNTER_CONTAINER_IMAGE_DIGEST", "unavailable-development",
        ),
    }
    if formal_data:
        source_identity = formal_source_identity(provenance)
        dataset_manifest = json.loads(
            (Path(dataset_root) / "dataset_manifest.json").read_text(encoding="utf-8")
        )
        validate_runtime_source_against_dataset(dataset_manifest, source_identity)
    checkpoint_identity = {
        "dataset_index_sha256": report.index_sha256,
        "dataset_manifest_sha256": report.dataset_manifest_sha256,
        "dataset_validation_sha256": canonical_sha256(asdict(report)),
        "scenario_manifest_sha256": report.scenario_manifest_sha256,
        "contract_sha256": contract["contract_sha256"],
        "protocol_version": contract["protocol"]["protocol_version"],
        "protocol_sha256": contract["protocol_sha256"],
        "model_fingerprint": contract["model"].fingerprint(),
        "method_id": str(method_id).upper(), "seed": seed,
        "training_stage": "stage5_baseline",
        "formal_dataset_validated": formal_data,
        "source_identity": source_identity,
        "device": str(device),
    }
    store = EpisodeStore(dataset_root)
    index = SequenceIndex.build(store, loss_window=loss_window)
    index.validate_split_isolation()
    run_path = Path(run_root)
    run_path.mkdir(parents=True, exist_ok=True)
    index_path = run_path / "sequence_index.json"
    if resume:
        restored = SequenceIndex.load(index_path)
        if restored.sha256() != index.sha256():
            raise RuntimeError("dataset/index changed since the checkpointed baseline run")
        index = restored
    elif index_path.exists():
        raise FileExistsError(f"fresh training run refuses existing index: {index_path}")
    else:
        index.save(index_path)

    seed_all(seed)
    enable_torch_determinism(warn_only=True)
    agent = ComparisonAgent(
        contract["model"], contract["input_model"], contract["agent"],
        device=device, target_seed=seed,
    )
    inventory = parameter_inventory(agent)
    parameter_match = None
    if agent.method_id == "B2":
        reference = TractorAgent(
            contract["input_model"], contract["agent"], device=device, target_seed=seed,
        )
        reference_count = sum(parameter.numel() for parameter in reference.online.parameters())
        baseline_count = sum(parameter.numel() for parameter in agent.online.parameters())
        relative_error = abs(baseline_count - reference_count) / reference_count
        tolerance = float(contract["parameter_match"]["relative_tolerance"])
        parameter_match = {
            "reference_method": "A7", "reference_count": reference_count,
            "baseline_count": baseline_count, "relative_error": relative_error,
            "tolerance": tolerance, "passed": relative_error <= tolerance,
        }
        if not parameter_match["passed"]:
            raise RuntimeError(f"B2 parameter-match gate failed: {parameter_match}")
        del reference

    sampler = SequenceBuffer(store, index, seed=seed)
    resume_manifest = None
    if resume:
        resume_manifest = load_training_generation(
            run_path / "checkpoints", checkpoint_tag, agent, sampler,
            expected_metadata={
                "dataset_index_sha256": index.sha256(),
                "contract_sha256": contract["contract_sha256"],
                "method_id": agent.method_id, "seed": seed,
                "training_stage": "stage5_baseline",
            },
        )
    trainer = ComparisonSequenceTrainer(agent, sampler)
    if resume_manifest is not None:
        trainer.risk_updates = int(
            resume_manifest.get("metadata", {}).get("risk_updates_applied", 0)
        )
    initial_update_step = agent.update_step
    target_update = int(updates)
    if initial_update_step > target_update:
        raise ValueError("checkpoint update step exceeds the registered total update budget")
    actual_batch = int(batch_size or contract["campaign"]["batch_size"])
    interval = int(checkpoint_interval or contract["campaign"]["checkpoint_interval_updates"])
    log_path = run_path / "logs" / "training.jsonl"
    started = time.monotonic()
    last_metrics = {}
    for _ in range(agent.update_step, target_update):
        last_metrics = trainer.update(actual_batch)
        _json_line(log_path, {"update_step": agent.update_step, **last_metrics})
        if agent.update_step > 0 and agent.update_step % interval == 0:
            save_training_generation(
                run_path / "checkpoints", "latest", agent, sampler,
                {**checkpoint_identity,
                 "risk_updates_applied": trainer.risk_updates,
                 "evidence_status": "training_in_progress_not_evaluation_evidence"},
            )
    if agent.method_id == "B8" and trainer.risk_updates == 0:
        raise RuntimeError("B8 scalar endpoint-risk head received no update")
    generation = save_training_generation(
        run_path / "checkpoints", "final", agent, sampler,
        {**checkpoint_identity,
            "parameter_inventory": inventory, "parameter_match": parameter_match,
            "risk_updates_applied": trainer.risk_updates,
            "evidence_status": "trained_not_held_out_evaluated",
        },
        parent_tag="latest",
    )
    return {
        "phase": "train", "method_id": agent.method_id, "seed": seed,
        "updates_applied": target_update - initial_update_step,
        "target_total_updates": target_update, "final_update_step": agent.update_step,
        "risk_updates_applied": trainer.risk_updates,
        "parameter_inventory": inventory, "parameter_match": parameter_match,
        "checkpoint_generation": generation, "run_root": str(run_path.resolve()),
        "elapsed_sec": time.monotonic() - started, "last_metrics": last_metrics,
        "performance_claim": "none_until_locked_test_evaluation",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=BASELINE_METHODS)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--checkpoint-tag", default="latest")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--config-root")
    parser.add_argument("--formal-data", action="store_true")
    parser.add_argument("--scenario-manifest")
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    result = train_comparison_baseline(
        method_id=args.method, dataset_root=args.dataset_root, run_root=args.run_root,
        seed=args.seed, updates=args.updates, batch_size=args.batch_size,
        checkpoint_interval=args.checkpoint_interval, checkpoint_tag=args.checkpoint_tag,
        resume=args.resume, device=args.device, config_root=args.config_root,
        formal_data=args.formal_data, scenario_manifest_path=args.scenario_manifest,
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        Path(args.output_json).write_text(payload, encoding="utf-8")
    print(payload, end="")
    return result


if __name__ == "__main__":
    main()
