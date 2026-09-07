#!/usr/bin/env python3
"""Run one trained method/seed over every frozen locked-test scenario."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from hunter_kinodynamic_rl.config.comparison import BASELINE_METHODS, load_comparison_contract
from hunter_kinodynamic_rl.config.loader import load_profile
from hunter_kinodynamic_rl.config.tractor import canonical_sha256, load_tractor_contract
from hunter_kinodynamic_rl.env.scenarios.benchmark_loader import load_scenario_file
from hunter_kinodynamic_rl.navigation.local_rl.tractor_policy import TractorPolicy
from hunter_kinodynamic_rl.rl.algorithms.comparison_baselines import ComparisonAgent
from hunter_kinodynamic_rl.rl.algorithms.tractor_tqc import TractorAgent
from hunter_kinodynamic_rl.rl.checkpointing.tractor import (
    load_calibration_artifact, load_inference_weights,
)
from hunter_kinodynamic_rl.training.tractor_dataset_validation import validate_dataset
from hunter_kinodynamic_rl.training.tractor_episode_collector import make_snapshot
from hunter_kinodynamic_rl.training.tractor_scenario_plan import (
    validate_materialized_scenario_manifest,
)
from hunter_kinodynamic_rl.training.tractor_sequence_training import snapshot_to_inputs
from hunter_kinodynamic_rl.evaluation.fit_comparison_calibration import (
    CALIBRATED_METHODS, calibration_split_identity,
)


PAPER_METHODS = BASELINE_METHODS + ("A7", "A8", "A9")


class _BaselineRuntimePolicy:
    def __init__(self, agent: ComparisonAgent):
        self.agent = agent

    def reset(self) -> None:
        pass

    def act(self, snapshot, snapshot_id: int) -> np.ndarray:
        del snapshot_id
        inputs = snapshot_to_inputs(snapshot, self.agent.input_config, self.agent.device)
        with torch.inference_mode():
            action = self.agent.online.deterministic_action(inputs)
        return action[0].cpu().numpy().astype(np.float32)


class _TractorRuntimePolicy:
    def __init__(self, agent: TractorAgent, selector, calibration):
        self.agent = agent
        self.policy = TractorPolicy(agent.online, selector, calibration, deployment=True)

    def reset(self) -> None:
        self.policy.reset()

    def act(self, snapshot, snapshot_id: int) -> np.ndarray:
        inputs = snapshot_to_inputs(snapshot, self.agent.model_config, self.agent.device)
        decision = self.policy.step(snapshot_id, inputs)
        if not decision.publish_allowed:
            # Normalized speed=-1 is the explicit physical stop; zeros would be mid-speed.
            return np.asarray([0.0, -1.0, 0.0], dtype=np.float32)
        return decision.selection.selected_action[0].cpu().numpy().astype(np.float32)


def _load_policy(
    method_id: str, seed: int, checkpoint_root: str | Path,
    checkpoint_tag: str, calibration_artifact: str | Path | None,
    dataset_root: str | Path, scenario_manifest_sha256: str,
    device: str, config_root: str | None,
):
    if method_id.startswith("A"):
        contract = load_tractor_contract(config_root, method_id.lower())
        agent = TractorAgent(contract["model"], contract["agent"], device=device, target_seed=seed)
    else:
        contract = load_comparison_contract(config_root, method_id)
        agent = ComparisonAgent(
            contract["model"], contract["input_model"], contract["agent"],
            device=device, target_seed=seed,
        )
    checkpoint_manifest = load_inference_weights(checkpoint_root, checkpoint_tag, agent)
    metadata = checkpoint_manifest.get("metadata", {})
    if int(metadata.get("seed", -1)) != int(seed):
        raise RuntimeError("checkpoint metadata seed does not match evaluation seed")
    if not metadata.get("formal_dataset_validated", False):
        raise RuntimeError("paper evaluation requires a checkpoint trained from validated formal data")
    calibration = None
    calibration_manifest = None
    if method_id in CALIBRATED_METHODS:
        if calibration_artifact is None:
            raise ValueError(f"{method_id} evaluation requires its calibration artifact")
        calibration, calibration_manifest = load_calibration_artifact(
            calibration_artifact,
            expected_checkpoint_sha256=checkpoint_manifest["training_payload_sha256"],
        )
        split_sha, _episode_ids = calibration_split_identity(
            dataset_root, scenario_manifest_sha256,
        )
        if calibration.split_sha256 != split_sha:
            raise RuntimeError("calibration artifact belongs to a different calibration split")
    runtime = (
        _TractorRuntimePolicy(agent, contract["selector"], calibration)
        if method_id.startswith("A") else _BaselineRuntimePolicy(agent)
    )
    return runtime, contract, checkpoint_manifest, calibration_manifest


def run_locked_episode(env, policy, profile, scenario, input_config, deadline_ms: float) -> dict:
    state, diagnostics = env.reset_with_diagnostics()
    if not diagnostics.valid:
        raise RuntimeError("locked evaluation reset has no synchronized sensor diagnostics")
    current = make_snapshot(
        state, input_config, diagnostics=diagnostics,
        pose_covariance=env.latest_pose_covariance, previous=None,
        nominal_dt_sec=profile.runtime.time_delta_sec,
    )
    policy.reset()
    latencies, clearances = [], []
    emergency_stops = invalid_telemetry = invalid_diagnostics = 0
    success = collision = timeout = False
    total_reward = 0.0
    steps = 0
    for step in range(profile.evaluation.max_episode_steps):
        started = time.perf_counter_ns()
        action = policy.act(current, step + 1)
        latency_ms = (time.perf_counter_ns() - started) * 1e-6
        if np.asarray(action).shape != (3,) or not np.isfinite(action).all():
            raise RuntimeError("policy produced a non-finite or non-trajectory action")
        latencies.append(latency_ms)
        next_state, reward, done, target, collided, _distance, telemetry, next_diagnostics = env.step(action)
        steps = step + 1
        total_reward += float(reward)
        invalid_telemetry += int(not telemetry.valid)
        invalid_diagnostics += int(not next_diagnostics.valid)
        emergency_stops += int(telemetry.emergency_stop)
        if telemetry.valid and math.isfinite(telemetry.min_clearance_m):
            clearances.append(float(telemetry.min_clearance_m))
        next_snapshot = make_snapshot(
            next_state, input_config, diagnostics=next_diagnostics,
            pose_covariance=env.latest_pose_covariance, previous=current,
            previous_action=action,
            previous_command=(telemetry.published_speed_mps, telemetry.published_steering_rad),
            nominal_dt_sec=profile.runtime.time_delta_sec,
        )
        current = next_snapshot
        success = success or bool(target)
        collision = collision or bool(collided)
        if done:
            timeout = not (success or collision)
            break
    else:
        timeout = not (success or collision)
    if not clearances:
        raise RuntimeError("locked evaluation produced no common-yardstick clearance samples")
    elapsed = env.episode_elapsed_sim_time_sec
    if elapsed is None or not math.isfinite(float(elapsed)):
        elapsed = steps * profile.runtime.time_delta_sec
    budget_sec = profile.evaluation.max_episode_steps * profile.runtime.time_delta_sec
    time_to_goal = float(elapsed) if success else float(budget_sec)
    complete = (
        steps > 0 and invalid_telemetry == 0 and invalid_diagnostics == 0
        and (success or collision or timeout)
    )
    latency_array = np.asarray(latencies, dtype=np.float64)
    return {
        "complete": complete,
        "metrics": {
            "success_rate": float(success), "collision_rate": float(collision),
            "time_to_goal_sec_mean": time_to_goal,
            "min_clearance_m_mean": float(min(clearances)),
            "timeout_rate": float(timeout), "path_length_m": float(env.episode_path_length_m),
            "total_reward": total_reward, "emergency_stop_count": emergency_stops,
            "decision_count": len(latencies),
            "mean_latency_ms": float(latency_array.mean()),
            "p99_latency_ms": float(np.quantile(latency_array, 0.99)),
            "deadline_miss_rate": float((latency_array > deadline_ms).mean()),
            "invalid_telemetry_count": invalid_telemetry,
            "invalid_diagnostics_count": invalid_diagnostics,
        },
    }


def _atomic_jsonl(path: Path, records: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(f"immutable evaluation artifact exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def evaluate_method_seed(
    *, method_id: str, seed: int, checkpoint_root: str | Path,
    dataset_root: str | Path, scenario_manifest_path: str | Path,
    output_jsonl: str | Path, calibration_artifact: str | Path | None = None,
    checkpoint_tag: str = "final", profile_name: str = "tractor_local_dynamic",
    device: str = "cpu", config_root: str | None = None,
) -> dict:
    method_id = method_id.upper()
    if method_id not in PAPER_METHODS:
        raise ValueError(f"method_id must be one of {PAPER_METHODS}")
    scenario_manifest = validate_materialized_scenario_manifest(
        scenario_manifest_path, config_root,
    )
    profile = load_profile(profile_name, config_root)
    report = validate_dataset(
        dataset_root, formal=True, scenario_manifest_path=scenario_manifest_path,
        config_root=config_root,
    )
    if not report.ok:
        raise RuntimeError(f"paper evaluation dataset validation failed: {report.errors}")
    policy, contract, checkpoint_manifest, calibration_manifest = _load_policy(
        method_id, seed, checkpoint_root, checkpoint_tag, calibration_artifact,
        dataset_root, scenario_manifest["artifact_manifest_sha256"], device, config_root,
    )
    if checkpoint_manifest["metadata"].get("dataset_index_sha256") != report.index_sha256:
        raise RuntimeError("checkpoint was trained against a different immutable dataset index")
    input_config = contract["model"] if method_id.startswith("A") else contract["input_model"]
    deadline_ms = float(contract["runtime"]["decision_deadline_ms"])
    locked_entries = [
        item for item in scenario_manifest["entries"] if item["split_id"] == "locked_test"
    ]
    if len(locked_entries) != int(contract["protocol"]["support_gates"]["minimum_locked_scenarios_per_seed"]):
        raise RuntimeError("locked scenario count does not match the frozen protocol")
    scenario_root = Path(scenario_manifest_path).resolve().parent
    import rclpy
    from hunter_kinodynamic_rl.training.trainer_base import (
        EnvironmentClient, validate_training_environment_contract,
    )
    rclpy.init(args=None)
    env = None
    records = []
    try:
        env = EnvironmentClient(
            node_name=f"hunter_paper_eval_{method_id.lower()}_{seed}",
            telemetry_wait_timeout_sec=profile.runtime.risk_telemetry_wait_timeout_sec,
            reset_marker_wait_timeout_sec=profile.runtime.risk_telemetry_reset_marker_timeout_sec,
            wheelbase_m=profile.robot.wheelbase_m, track_width_m=profile.robot.track_width_m,
        )
        attestation = validate_training_environment_contract(env, profile, env.get_dimensions())
        for entry in locked_entries:
            scenario_path = (scenario_root / entry["relative_path"]).resolve()
            if scenario_root != scenario_path and scenario_root not in scenario_path.parents:
                raise RuntimeError("locked scenario path escapes materialized root")
            if not env.set_scenario_override(str(scenario_path)):
                raise RuntimeError(f"environment rejected locked scenario {entry['scenario_id']}")
            scenario = load_scenario_file(str(scenario_path))
            outcome = run_locked_episode(
                env, policy, profile, scenario, input_config, deadline_ms,
            )
            record = {
                "experiment_id": f"tractor-paper-{method_id}-{seed}-{entry['scenario_id']}",
                "method_id": method_id, "seed": int(seed),
                "scenario_id": entry["scenario_id"], "split_id": "locked_test",
                "scenario_family": entry["family_id"],
                "obstacle_contract": entry["obstacle_contract"],
                "scenario_geometry_sha256": entry["scenario_geometry_sha256"],
                "model_fingerprint": contract["model"].fingerprint(),
                "data_fingerprint": report.index_sha256,
                "training_fingerprint": contract["contract_sha256"],
                "checkpoint_sha256": checkpoint_manifest["training_payload_sha256"],
                "environment_attestation_sha256": canonical_sha256(attestation),
                "calibration_artifact_sha256": (
                    None if calibration_manifest is None
                    else hashlib.sha256(Path(calibration_artifact).read_bytes()).hexdigest()
                ),
                "complete": outcome["complete"], "metrics": outcome["metrics"],
            }
            record["artifact_sha256"] = canonical_sha256(record)
            records.append(record)
    finally:
        if env is not None:
            try:
                env.set_scenario_override("")
            finally:
                env.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    _atomic_jsonl(Path(output_jsonl), records)
    return {
        "phase": "locked_test_evaluation", "method_id": method_id, "seed": seed,
        "record_count": len(records), "complete_count": sum(item["complete"] for item in records),
        "output_jsonl": str(Path(output_jsonl).resolve()),
        "scenario_manifest_sha256": scenario_manifest["artifact_manifest_sha256"],
        "evidence_scope": "locked_test_simulation_records",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=PAPER_METHODS)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--checkpoint-tag", default="final")
    parser.add_argument("--calibration-artifact")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--scenario-manifest", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--profile", default="tractor_local_dynamic")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--config-root")
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    result = evaluate_method_seed(
        method_id=args.method, seed=args.seed, checkpoint_root=args.checkpoint_root,
        checkpoint_tag=args.checkpoint_tag, calibration_artifact=args.calibration_artifact,
        dataset_root=args.dataset_root, scenario_manifest_path=args.scenario_manifest,
        output_jsonl=args.output_jsonl, profile_name=args.profile, device=args.device,
        config_root=args.config_root,
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        Path(args.output_json).write_text(payload, encoding="utf-8")
    print(payload, end="")
    return result


if __name__ == "__main__":
    main()
