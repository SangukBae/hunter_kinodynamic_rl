#!/usr/bin/env python3
"""Fit immutable calibration artifacts from the isolated calibration split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from hunter_kinodynamic_rl.config.comparison import load_comparison_contract
from hunter_kinodynamic_rl.config.tractor import (
    canonical_sha256, load_tractor_contract, require_formal_research_implementation_ready,
)
from hunter_kinodynamic_rl.evaluation.provenance import collect_package_provenance
from hunter_kinodynamic_rl.rl.algorithms.comparison_baselines import ComparisonAgent
from hunter_kinodynamic_rl.rl.algorithms.tractor_tqc import TractorAgent
from hunter_kinodynamic_rl.rl.checkpointing.tractor import (
    load_inference_weights, save_calibration_artifact,
)
from hunter_kinodynamic_rl.rl.networks.tractor.calibration import fit_platt
from hunter_kinodynamic_rl.rl.networks.tractor.contracts import CandidateSet
from hunter_kinodynamic_rl.rl.networks.tractor.selector import cumulative_event_probability
from hunter_kinodynamic_rl.rl.replay import EpisodeStore
from hunter_kinodynamic_rl.rl.replay.episode_store import sha256_file
from hunter_kinodynamic_rl.training.tractor_dataset_validation import validate_dataset
from hunter_kinodynamic_rl.training.tractor_dataset_manifest import (
    formal_source_identity, validate_runtime_source_against_dataset,
)
from hunter_kinodynamic_rl.training.tractor_sequence_training import _row_inputs


CALIBRATED_METHODS = ("B8", "A7", "A8", "A9")


def calibration_split_identity(
    dataset_root: str | Path, scenario_manifest_sha256: str,
) -> tuple[str, list[str]]:
    store = EpisodeStore(dataset_root)
    items = []
    for path in store.iter_paths():
        header, _columns = store.load(path.stem)
        if header.split_id == "calibration":
            items.append((header.episode_id, sha256_file(path)))
    if not items:
        raise RuntimeError("dataset contains no calibration episodes")
    return canonical_sha256({
        "split_id": "calibration", "episodes": sorted(items),
        "scenario_manifest_sha256": scenario_manifest_sha256,
    }), [episode_id for episode_id, _digest in sorted(items)]


def _candidate_set(columns, row: int, device: torch.device) -> CandidateSet:
    present = np.asarray(columns["candidate_present"])[row].astype(bool)
    present &= np.asarray(columns["candidate_model_valid"])[row].astype(bool)
    return CandidateSet(
        normalized_actions=torch.as_tensor(
            np.asarray(columns["candidate_actions_normalized"])[row:row + 1],
            dtype=torch.float32, device=device,
        ),
        present=torch.as_tensor(present[None], dtype=torch.bool, device=device),
        is_stop=torch.as_tensor(
            np.asarray(columns["candidate_is_stop"])[row:row + 1],
            dtype=torch.bool, device=device,
        ),
        source="calibration_split",
    )


def extract_episode_calibration_rows(agent, method_id: str, columns) -> tuple[list[float], list[float]]:
    """Return uncalibrated endpoint probabilities and observed-event targets once per row/candidate."""
    method_id = method_id.upper()
    if method_id not in CALIBRATED_METHODS:
        raise ValueError(f"calibration is only defined for {CALIBRATED_METHODS}")
    probabilities, targets = [], []
    scene_hidden = scene_valid = response_hidden = response_valid = None
    input_config = agent.input_config if method_id == "B8" else agent.model_config
    device = agent.device
    for row in range(len(np.asarray(columns["observation"]))):
        inputs = _row_inputs(
            columns, row, input_config, device,
            previous_scene_hidden=scene_hidden,
            previous_scene_hidden_valid=scene_valid,
            previous_response_hidden=response_hidden,
            previous_response_hidden_valid=response_valid,
        )
        candidates = _candidate_set(columns, row, device)
        with torch.inference_mode():
            if method_id == "B8":
                state = agent.online.encode(inputs)
                probability = agent.online.risk_probability_from_state(
                    state, candidates.normalized_actions,
                )[0]
            else:
                belief, context = agent.online.encode(inputs)
                output = agent.online.score(belief, context, candidates)
                probability = cumulative_event_probability(output.hazard).mean(dim=(1, 2))[0]
                scene_hidden = belief.next_scene_hidden
                scene_valid = belief.next_scene_hidden_valid
                response_hidden = belief.next_response_hidden
                response_valid = belief.next_response_hidden_valid
        valid = np.asarray(columns["candidate_event_label_valid"])[row].astype(bool) & candidates.present[0].cpu().numpy()
        target = np.asarray(columns["candidate_event_observed"])[row].astype(bool)
        probabilities.extend(probability[torch.as_tensor(valid, device=device)].cpu().tolist())
        targets.extend(target[valid].astype(np.float64).tolist())
    return probabilities, targets


def calibration_metrics(probability: torch.Tensor, target: torch.Tensor, bins: int = 10) -> dict[str, float | int]:
    if probability.shape != target.shape or probability.numel() == 0:
        raise ValueError("calibration metrics require non-empty shape-matched tensors")
    p, y = probability.double(), target.double()
    ece = torch.zeros((), dtype=torch.double)
    edges = torch.linspace(0.0, 1.0, bins + 1, dtype=torch.double)
    for index in range(bins):
        selected = (p >= edges[index]) & (p < edges[index + 1])
        if index == bins - 1:
            selected |= p == 1.0
        if selected.any():
            ece += selected.double().mean() * (p[selected].mean() - y[selected].mean()).abs()
    return {
        "sample_count": int(p.numel()), "event_count": int(y.sum()),
        "non_event_count": int(p.numel() - y.sum()),
        "brier": float((p - y).square().mean()), "ece_10_bin": float(ece),
    }


def fit_comparison_calibration(
    *, method_id: str, checkpoint_root: str | Path, dataset_root: str | Path,
    scenario_manifest_path: str | Path, output_root: str | Path, seed: int,
    checkpoint_tag: str = "final", device: str = "cpu", config_root: str | None = None,
    artifact_id: str | None = None, minimum_samples: int = 100,
) -> dict:
    require_formal_research_implementation_ready("formal comparison calibration")
    source_identity = formal_source_identity(
        collect_package_provenance(execution_file=__file__)
    )
    method_id = method_id.upper()
    if method_id not in CALIBRATED_METHODS:
        raise ValueError(f"method_id must be one of {CALIBRATED_METHODS}")
    if minimum_samples <= 1:
        raise ValueError("minimum_samples must be greater than one")
    variant = method_id.lower() if method_id.startswith("A") else None
    contract = (
        load_tractor_contract(config_root, variant)
        if variant is not None else load_comparison_contract(config_root, method_id)
    )
    report = validate_dataset(
        dataset_root, loss_window=int(contract["data"]["loss_window"]), formal=True,
        scenario_manifest_path=scenario_manifest_path, config_root=config_root,
    )
    if not report.ok:
        raise RuntimeError(f"formal calibration dataset validation failed: {report.errors}")
    dataset_manifest = json.loads(
        (Path(dataset_root) / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    validate_runtime_source_against_dataset(dataset_manifest, source_identity)
    if method_id == "B8":
        agent = ComparisonAgent(
            contract["model"], contract["input_model"], contract["agent"],
            device=device, target_seed=seed,
        )
    else:
        agent = TractorAgent(contract["model"], contract["agent"], device=device, target_seed=seed)
    manifest = load_inference_weights(checkpoint_root, checkpoint_tag, agent)
    metadata = manifest.get("metadata", {})
    expected_checkpoint_metadata = {
        "seed": int(seed), "dataset_index_sha256": report.index_sha256,
        "dataset_manifest_sha256": report.dataset_manifest_sha256,
        "scenario_manifest_sha256": report.scenario_manifest_sha256,
        "contract_sha256": contract["contract_sha256"],
        "protocol_sha256": contract["protocol_sha256"],
        "formal_dataset_validated": True,
    }
    for name, expected in expected_checkpoint_metadata.items():
        if metadata.get(name) != expected:
            raise RuntimeError(f"calibration checkpoint metadata mismatch for {name!r}")
    checkpoint_sha = str(manifest["training_payload_sha256"])
    store = EpisodeStore(dataset_root)
    split_sha, expected_episode_ids = calibration_split_identity(
        dataset_root, str(report.scenario_manifest_sha256),
    )
    episode_ids, probabilities, targets = [], [], []
    for path in store.iter_paths():
        header, columns = store.load(path.stem)
        if header.split_id != "calibration":
            continue
        episode_ids.append(header.episode_id)
        episode_probability, episode_target = extract_episode_calibration_rows(
            agent, method_id, columns,
        )
        probabilities.extend(episode_probability)
        targets.extend(episode_target)
    if sorted(episode_ids) != expected_episode_ids:
        raise RuntimeError("calibration episode iteration changed split identity")
    probability = torch.as_tensor(probabilities, dtype=torch.float64)
    target = torch.as_tensor(targets, dtype=torch.float64)
    before = calibration_metrics(probability, target)
    if before["sample_count"] < minimum_samples:
        raise RuntimeError(f"calibration has only {before['sample_count']} valid samples")
    if before["event_count"] == 0 or before["non_event_count"] == 0:
        raise RuntimeError("calibration requires both observed events and censored non-events")
    calibration = fit_platt(
        probability, target, source_checkpoint_sha256=checkpoint_sha, split_sha256=split_sha,
    )
    calibrated = calibration.apply(probability)
    after = calibration_metrics(calibrated, target)
    identity = artifact_id or f"{method_id.lower()}-seed-{seed}-platt-v1"
    artifact_path = save_calibration_artifact(
        output_root, identity, calibration, episode_ids=episode_ids,
        metrics={"before": before, "after": after},
        calibration_context_id=(
            "b8-scalar-endpoint-v1" if method_id == "B8" else "r5-post-aggregate-v1"
        ),
        provenance={
            **source_identity,
            "dataset_manifest_sha256": report.dataset_manifest_sha256,
            "scenario_manifest_sha256": report.scenario_manifest_sha256,
            "contract_sha256": contract["contract_sha256"],
        },
    )
    artifact_sha = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    return {
        "phase": "calibration", "method_id": method_id, "seed": seed,
        "checkpoint_sha256": checkpoint_sha, "calibration_split_sha256": split_sha,
        "calibration_artifact": str(artifact_path.resolve()),
        "calibration_artifact_sha256": artifact_sha,
        "metrics_before": before, "metrics_after": after,
        "performance_claim": "calibration_only_not_locked_test_evidence",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=CALIBRATED_METHODS)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--checkpoint-tag", default="final")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--scenario-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--artifact-id")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--config-root")
    parser.add_argument("--minimum-samples", type=int, default=100)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    result = fit_comparison_calibration(
        method_id=args.method, checkpoint_root=args.checkpoint_root,
        checkpoint_tag=args.checkpoint_tag, dataset_root=args.dataset_root,
        scenario_manifest_path=args.scenario_manifest, output_root=args.output_root,
        artifact_id=args.artifact_id, seed=args.seed, device=args.device,
        config_root=args.config_root, minimum_samples=args.minimum_samples,
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        Path(args.output_json).write_text(payload, encoding="utf-8")
    print(payload, end="")
    return result


if __name__ == "__main__":
    main()
