#!/usr/bin/env python3
"""Profile warmed TRACTOR encode/propose/score latency without training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from hunter_kinodynamic_rl.config.tractor import load_tractor_contract
from hunter_kinodynamic_rl.rl.networks.tractor import TractorInputs, TractorTQC


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _synthetic_input(config, device):
    scans = torch.full((1, config.t_obs, config.n_scan), config.max_range_m, device=device)
    tail = torch.zeros((1, config.tail_dim), device=device)
    tail[:, 0] = 3.0
    return TractorInputs(
        observation=torch.cat((scans.reshape(1, -1), tail), dim=-1),
        scan_valid=torch.ones_like(scans, dtype=torch.bool),
        motion_delta=torch.cat((
            torch.zeros((1, config.t_obs - 1, 3), device=device),
            torch.full((1, config.t_obs - 1, 1), 0.1, device=device),
        ), dim=-1),
        motion_valid=torch.ones((1, config.t_obs - 1), dtype=torch.bool, device=device),
        decision_timestamp_sec=torch.ones((1, 1), device=device),
        previous_intent_valid=torch.ones((1, 1), dtype=torch.bool, device=device),
        previous_command_published=torch.zeros((1, 2), device=device),
        previous_command_valid=torch.ones((1, 1), dtype=torch.bool, device=device),
        vehicle_response_valid=torch.ones((1, 3), dtype=torch.bool, device=device),
        localization_covariance=torch.eye(3, device=device).mul(1e-3).unsqueeze(0),
        localization_valid=torch.ones((1, 1), dtype=torch.bool, device=device),
        localization_confidence=torch.ones((1, 1), device=device),
        localization_confidence_valid=torch.ones((1, 1), dtype=torch.bool, device=device),
        sensor_freshness_sec=torch.zeros((1, 1), device=device),
        sensor_freshness_valid=torch.ones((1, 1), dtype=torch.bool, device=device),
        scene_reset=torch.ones((1, 1), dtype=torch.bool, device=device),
        response_reset=torch.ones((1, 1), dtype=torch.bool, device=device),
    )


def profile_latency(variant="a7", device="cpu", warmup=20, iterations=100, config_root=None):
    if warmup < 1 or iterations < 1:
        raise ValueError("warmup and iterations must be positive")
    contract = load_tractor_contract(config_root, variant)
    resolved_device = torch.device(device)
    model = TractorTQC(contract["model"]).to(resolved_device).eval()
    sample = _synthetic_input(contract["model"], resolved_device)
    if resolved_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved_device)
    stage = {"encode": [], "propose": [], "score": [], "end_to_end": []}
    with torch.inference_mode():
        for iteration in range(warmup + iterations):
            _synchronize(resolved_device)
            start = time.perf_counter_ns()
            belief, context = model.encode(sample)
            _synchronize(resolved_device)
            encoded = time.perf_counter_ns()
            candidates = model.propose(belief)
            _synchronize(resolved_device)
            proposed = time.perf_counter_ns()
            model.score(belief, context, candidates)
            _synchronize(resolved_device)
            finished = time.perf_counter_ns()
            if iteration >= warmup:
                stage["encode"].append((encoded - start) / 1e6)
                stage["propose"].append((proposed - encoded) / 1e6)
                stage["score"].append((finished - proposed) / 1e6)
                stage["end_to_end"].append((finished - start) / 1e6)
    summary = {
        name: {
            "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)),
            "p99_ms": float(np.percentile(values, 99)),
            "mean_ms": float(np.mean(values)),
        }
        for name, values in stage.items()
    }
    deadline = float(contract["runtime"]["decision_deadline_ms"])
    misses = np.asarray(stage["end_to_end"]) >= deadline
    component_parameters = {
        name: sum(parameter.numel() for parameter in module.parameters())
        for name, module in model.named_children()
    }
    return {
        "schema_id": "tractor_latency_v1", "evidence_level": "synthetic_process_profile",
        "variant": variant, "model_fingerprint": contract["model"].fingerprint(),
        "device": str(device), "torch_version": torch.__version__, "warmup": warmup,
        "iterations": iterations, "num_candidates": contract["model"].num_candidates,
        "horizon_steps": contract["model"].horizon_steps,
        "sparse_tube_samples": contract["model"].sparse_tube_samples,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "component_parameter_count": component_parameters,
        "precision": str(next(model.parameters()).dtype),
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(resolved_device))
            if resolved_device.type == "cuda" else None
        ),
        "summary": summary, "deadline_ms": deadline,
        "deadline_miss_count": int(misses.sum()), "deadline_miss_rate": float(misses.mean()),
        "target_hardware_evidence": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("a7", "a8", "a9"), default="a7")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--config-root")
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    report = profile_latency(args.variant, args.device, args.warmup, args.iterations, args.config_root)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        Path(args.output_json).write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
