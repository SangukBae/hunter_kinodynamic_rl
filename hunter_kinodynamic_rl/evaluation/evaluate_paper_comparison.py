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
from hunter_kinodynamic_rl.config.tractor import (
    canonical_sha256, load_tractor_contract, require_formal_research_implementation_ready,
)
from hunter_kinodynamic_rl.env.scenarios.benchmark_loader import load_scenario_file
from hunter_kinodynamic_rl.navigation.local_rl.tractor_policy import TractorPolicy
from hunter_kinodynamic_rl.rl.algorithms.comparison_baselines import ComparisonAgent
from hunter_kinodynamic_rl.rl.algorithms.tractor_tqc import TractorAgent
from hunter_kinodynamic_rl.rl.checkpointing.tractor import (
    load_calibration_artifact, load_inference_weights,
)
from hunter_kinodynamic_rl.rl.networks.tractor.contracts import CandidateSet
from hunter_kinodynamic_rl.rl.networks.tractor.selector import cumulative_event_probability
from hunter_kinodynamic_rl.rl.replay import EpisodeStore
from hunter_kinodynamic_rl.evaluation.tractor_metrics import (
    binary_calibration_metrics, candidate_ranking_metrics, occupancy_confusion,
)
from hunter_kinodynamic_rl.evaluation.provenance import collect_package_provenance
from hunter_kinodynamic_rl.training.tractor_dataset_validation import validate_dataset
from hunter_kinodynamic_rl.training.tractor_dataset_manifest import (
    formal_source_identity, validate_runtime_source_against_dataset,
)
from hunter_kinodynamic_rl.training.tractor_episode_collector import make_snapshot
from hunter_kinodynamic_rl.training.tractor_scenario_plan import (
    validate_materialized_scenario_manifest,
)
from hunter_kinodynamic_rl.training.tractor_sequence_training import _row_inputs, snapshot_to_inputs
from hunter_kinodynamic_rl.evaluation.fit_comparison_calibration import (
    CALIBRATED_METHODS, calibration_split_identity,
)


PAPER_METHODS = BASELINE_METHODS + ("A7", "A8", "A9")


class _BaselineRuntimePolicy:
    def __init__(self, agent: ComparisonAgent, calibration=None):
        self.agent = agent
        self.calibration = calibration

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
        self.calibration = calibration
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
    expected_stage = "stage5" if method_id.startswith("A") else "stage5_baseline"
    if metadata.get("training_stage") != expected_stage:
        raise RuntimeError("paper evaluation requires the registered final training stage")
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
        if method_id.startswith("A") else _BaselineRuntimePolicy(agent, calibration)
    )
    return runtime, contract, checkpoint_manifest, calibration_manifest


def _dataset_candidates(columns, row: int, device: torch.device) -> CandidateSet:
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
        source="locked_test_hypothesis_evaluator_v1",
    )


def _selective_risk_curve(probability: np.ndarray, outcome: np.ndarray) -> list[dict]:
    if probability.size == 0:
        return []
    order = np.argsort(probability, kind="stable")
    result = []
    for coverage in (0.25, 0.5, 0.75, 1.0):
        count = max(1, int(math.ceil(probability.size * coverage)))
        selected = order[:count]
        result.append({
            "coverage": coverage, "count": count,
            "event_count": int(outcome[selected].sum()),
            "event_rate": float(outcome[selected].mean()),
            "maximum_retained_risk": float(probability[selected].max()),
        })
    return result


def evaluate_hypotheses_episode(policy, method_id: str, columns, input_config) -> dict:
    """Evaluate H1--H3 from the immutable episode aligned to one live scenario."""
    agent = policy.agent
    device = agent.device
    scene_hidden = scene_valid = response_hidden = response_valid = None
    occupancy_pred, occupancy_target, occupancy_valid = [], [], []
    occupancy_nll_sum = occupancy_nll_count = 0
    flow_error_sum = flow_error_count = 0
    tube_error_sum = tube_error_count = 0
    ranking_score, ranking_utility, ranking_valid, ranking_unsafe = [], [], [], []
    risk_probability, risk_outcome = [], []
    cause_correct = time_correct = observed_count = 0
    cause_time_nll_sum = cause_time_nll_count = 0
    candidate_present_count = candidate_rank_valid_count = 0
    h1_supported = method_id.startswith("A") or method_id in {"B4", "B5"}
    h3_supported = method_id.startswith("A") or method_id == "B8"
    row_count = len(np.asarray(columns["observation"]))
    for row in range(row_count):
        inputs = _row_inputs(
            columns, row, input_config, device,
            previous_scene_hidden=scene_hidden,
            previous_scene_hidden_valid=scene_valid,
            previous_response_hidden=response_hidden,
            previous_response_hidden_valid=response_valid,
        )
        candidates = _dataset_candidates(columns, row, device)
        with torch.inference_mode():
            belief = context = output = None
            if method_id.startswith("A"):
                belief, context = agent.online.encode(inputs)
                output = agent.online.score(belief, context, candidates)
                quantiles = output.return_quantiles.mean(dim=(-1, -2))[0]
                endpoint = cumulative_event_probability(output.hazard).mean(dim=(1, 2))[0]
                score = quantiles - endpoint
                scene_hidden = belief.next_scene_hidden
                scene_valid = belief.next_scene_hidden_valid
                response_hidden = belief.next_response_hidden
                response_valid = belief.next_response_hidden_valid
            else:
                state = agent.online.encode(inputs)
                values = []
                for candidate_index in range(candidates.normalized_actions.shape[1]):
                    candidate_quantiles = agent.online.quantiles_from_state(
                        state, candidates.normalized_actions[:, candidate_index]
                    )
                    values.append(candidate_quantiles.mean(dim=(-1, -2)))
                score = torch.stack(values, dim=1)[0]
                endpoint = None
                if method_id == "B8":
                    endpoint = agent.online.risk_probability_from_state(
                        state, candidates.normalized_actions,
                    )[0]
                    score = score - endpoint
                if method_id in {"B4", "B5"}:
                    belief, _context = agent.online.encoder.belief(inputs)
                    scene_hidden = belief.next_scene_hidden
                    scene_valid = belief.next_scene_hidden_valid
                    response_hidden = belief.next_response_hidden
                    response_valid = belief.next_response_hidden_valid

            if h1_supported and belief is not None:
                prediction_sets = (
                    (belief.occupancy[0], np.asarray(columns["current_bev_class_target"])[row],
                     np.asarray(columns["current_bev_class_valid"])[row]),
                    (belief.future_occupancy[0], np.asarray(columns["future_bev_class_target"])[row],
                     np.asarray(columns["future_bev_class_valid"])[row]),
                )
                for probability_tensor, target_value, valid_value in prediction_sets:
                    probability_np = probability_tensor.detach().cpu().numpy()
                    predicted_np = probability_np.argmax(axis=-3)
                    target_np = np.asarray(target_value)
                    valid_np = np.asarray(valid_value, dtype=bool)
                    occupancy_pred.append(predicted_np.reshape(-1))
                    occupancy_target.append(target_np.reshape(-1))
                    occupancy_valid.append(valid_np.reshape(-1))
                    if valid_np.any():
                        selected_probability = np.moveaxis(probability_np, -3, -1)[valid_np]
                        selected_target = target_np[valid_np].astype(np.int64)
                        occupancy_nll_sum += float(
                            -np.log(np.clip(
                                selected_probability[np.arange(selected_target.size), selected_target],
                                1e-12, 1.0,
                            )).sum()
                        )
                        occupancy_nll_count += int(selected_target.size)
                flow_sets = (
                    (belief.dynamic_flow[0].cpu().numpy(),
                     np.asarray(columns["current_dynamic_flow_target"])[row],
                     np.asarray(columns["current_dynamic_flow_valid"])[row]),
                    (belief.future_dynamic_flow[0].cpu().numpy(),
                     np.asarray(columns["future_dynamic_flow_target"])[row],
                     np.asarray(columns["future_dynamic_flow_valid"])[row]),
                )
                for predicted_flow, target_flow, valid_flow in flow_sets:
                    valid_flow = np.asarray(valid_flow, dtype=bool)
                    if valid_flow.any():
                        error = np.linalg.norm(
                            np.moveaxis(predicted_flow, -3, -1)[valid_flow]
                            - np.moveaxis(target_flow, -3, -1)[valid_flow], axis=-1,
                        )
                        flow_error_sum += float(error.sum())
                        flow_error_count += int(error.size)
                if method_id.startswith("A"):
                    rollout = agent.online.rollout(candidates, context, belief.plant_latent)
                    tube = agent.online.tube_rasterizer(rollout)
                    predicted_oob = tube.oob_mass.mean(dim=1)[0].cpu().numpy()
                    target_oob = np.asarray(columns["tube_oob_mass_target"])[row]
                    valid_oob = np.asarray(columns["tube_coverage_valid"])[row].astype(bool)
                    valid_oob &= candidates.present[0].cpu().numpy()[:, None]
                    if valid_oob.any():
                        tube_error_sum += float(np.abs(predicted_oob - target_oob)[valid_oob].sum())
                        tube_error_count += int(valid_oob.sum())

        score_np = score.detach().cpu().numpy().reshape(-1)
        present_np = candidates.present[0].cpu().numpy()
        progress = np.asarray(columns["candidate_progress_m"])[row]
        event_valid = np.asarray(columns["candidate_event_label_valid"])[row].astype(bool)
        event = np.asarray(columns["candidate_event_observed"])[row].astype(bool)
        progress_valid = np.asarray(columns["candidate_progress_valid"])[row].astype(bool)
        valid_rank = present_np & event_valid & progress_valid & np.isfinite(progress)
        utility = np.where(valid_rank, progress - 10.0 * event.astype(np.float32), np.nan)
        ranking_score.append(score_np)
        ranking_utility.append(utility)
        ranking_valid.append(valid_rank)
        ranking_unsafe.append(event)
        candidate_present_count += int(present_np.sum())
        candidate_rank_valid_count += int(valid_rank.sum())

        if h3_supported and endpoint is not None:
            probability_np = endpoint.detach().cpu().numpy().reshape(-1)
            if policy.calibration is not None:
                probability_np = policy.calibration.apply(
                    torch.as_tensor(probability_np, dtype=torch.float64)
                ).cpu().numpy()
            valid_risk = present_np & event_valid & np.isfinite(probability_np)
            risk_probability.extend(probability_np[valid_risk].tolist())
            risk_outcome.extend(event[valid_risk].astype(np.float64).tolist())
            if method_id.startswith("A") and output is not None:
                mean_hazard = output.hazard.mean(dim=(1, 2))[0].cpu().numpy()
                for candidate in np.flatnonzero(valid_risk):
                    hazard = np.clip(mean_hazard[candidate], 1e-12, 1.0)
                    survival = np.clip(1.0 - hazard.sum(axis=-1), 1e-12, 1.0)
                    censor = int(np.asarray(columns["candidate_censor_step"])[row, candidate])
                    if event[candidate]:
                        event_step = int(np.asarray(columns["candidate_event_step"])[row, candidate])
                        event_cause = int(np.asarray(columns["candidate_event_cause"])[row, candidate])
                        log_likelihood = np.log(survival[:event_step]).sum()
                        log_likelihood += math.log(float(hazard[event_step, event_cause]))
                    else:
                        log_likelihood = np.log(survival[:censor + 1]).sum()
                    cause_time_nll_sum -= float(log_likelihood)
                    cause_time_nll_count += 1
                for candidate in np.flatnonzero(valid_risk & event):
                    predicted_step, predicted_cause = np.unravel_index(
                        np.argmax(mean_hazard[candidate]), mean_hazard[candidate].shape,
                    )
                    true_step = int(np.asarray(columns["candidate_event_step"])[row, candidate])
                    true_cause = int(np.asarray(columns["candidate_event_cause"])[row, candidate])
                    time_correct += int(predicted_step == true_step)
                    cause_correct += int(predicted_cause == true_cause)
                    observed_count += 1

    if h1_supported and occupancy_pred:
        h1 = occupancy_confusion(
            np.concatenate(occupancy_pred), np.concatenate(occupancy_target),
            np.concatenate(occupancy_valid),
        )
        h1.update({
            "available": True,
            "occupancy_nll": (
                None if occupancy_nll_count == 0 else occupancy_nll_sum / occupancy_nll_count
            ),
            "occupancy_nll_count": occupancy_nll_count,
            "flow_epe": None if flow_error_count == 0 else flow_error_sum / flow_error_count,
            "flow_valid_count": flow_error_count,
            "tube_oob_mae": None if tube_error_count == 0 else tube_error_sum / tube_error_count,
            "tube_valid_count": tube_error_count,
        })
    else:
        h1 = {"available": False, "reason": "method_has_no_dense_scene_prediction"}
    h2 = candidate_ranking_metrics(
        np.asarray(ranking_score), np.asarray(ranking_utility),
        np.asarray(ranking_valid), np.asarray(ranking_unsafe),
    )
    h2["available"] = h2["row_count"] > 0
    h2["candidate_present_count"] = candidate_present_count
    h2["candidate_valid_count"] = candidate_rank_valid_count
    h2["candidate_coverage"] = (
        None if candidate_present_count == 0
        else candidate_rank_valid_count / candidate_present_count
    )
    risk_probability_np = np.asarray(risk_probability, dtype=np.float64)
    risk_outcome_np = np.asarray(risk_outcome, dtype=np.float64)
    h3 = binary_calibration_metrics(risk_probability_np, risk_outcome_np)
    h3.update({
        "available": h3_supported and h3["count"] > 0,
        "cause_accuracy": None if observed_count == 0 else cause_correct / observed_count,
        "time_bin_accuracy": None if observed_count == 0 else time_correct / observed_count,
        "cause_time_event_count": observed_count,
        "cause_time_nll": (
            None if cause_time_nll_count == 0
            else cause_time_nll_sum / cause_time_nll_count
        ),
        "cause_time_nll_count": cause_time_nll_count,
        "selective_risk_curve": _selective_risk_curve(risk_probability_np, risk_outcome_np),
    })
    if not h3_supported:
        h3["reason"] = "method_has_no_probability_risk_head"
    return {"h1_prediction": h1, "h2_ranking": h2, "h3_risk": h3}


def validate_hypothesis_episode_support(method_id: str, report: dict) -> None:
    """Reject records whose method-required hypothesis evidence has no denominator."""
    h1 = report["h1_prediction"]
    h2 = report["h2_ranking"]
    h3 = report["h3_risk"]
    if not h2.get("available") or int(h2.get("row_count", 0)) <= 0:
        raise RuntimeError(f"{method_id} episode has no valid H2 candidate-ranking support")
    if method_id.startswith("A") or method_id in {"B4", "B5"}:
        if not h1.get("available") or int(h1.get("occupancy_nll_count", 0)) <= 0:
            raise RuntimeError(f"{method_id} episode has no valid H1 prediction support")
        if method_id.startswith("A") and int(h1.get("tube_valid_count", 0)) <= 0:
            raise RuntimeError(f"{method_id} episode has no valid H1 tube support")
    if method_id.startswith("A") or method_id == "B8":
        if not h3.get("available") or int(h3.get("count", 0)) <= 0:
            raise RuntimeError(f"{method_id} episode has no valid H3 risk support")


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
    require_formal_research_implementation_ready("formal locked-test evaluation")
    source_identity = formal_source_identity(
        collect_package_provenance(execution_file=__file__)
    )
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
    dataset_manifest = json.loads(
        (Path(dataset_root) / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    validate_runtime_source_against_dataset(dataset_manifest, source_identity)
    policy, contract, checkpoint_manifest, calibration_manifest = _load_policy(
        method_id, seed, checkpoint_root, checkpoint_tag, calibration_artifact,
        dataset_root, scenario_manifest["artifact_manifest_sha256"], device, config_root,
    )
    checkpoint_metadata = checkpoint_manifest["metadata"]
    expected_checkpoint_metadata = {
        "dataset_index_sha256": report.index_sha256,
        "dataset_manifest_sha256": report.dataset_manifest_sha256,
        "scenario_manifest_sha256": report.scenario_manifest_sha256,
        "contract_sha256": contract["contract_sha256"],
        "protocol_sha256": contract["protocol_sha256"],
    }
    for name, expected in expected_checkpoint_metadata.items():
        if checkpoint_metadata.get(name) != expected:
            raise RuntimeError(f"evaluation checkpoint metadata mismatch for {name!r}")
    if calibration_manifest is not None:
        calibration_provenance = calibration_manifest.get("provenance", {})
        for name, expected in (
            ("dataset_manifest_sha256", report.dataset_manifest_sha256),
            ("scenario_manifest_sha256", report.scenario_manifest_sha256),
            ("contract_sha256", contract["contract_sha256"]),
            ("source_content_manifest_sha256", source_identity["source_content_manifest_sha256"]),
        ):
            if calibration_provenance.get(name) != expected:
                raise RuntimeError(f"calibration artifact provenance mismatch for {name!r}")
    input_config = contract["model"] if method_id.startswith("A") else contract["input_model"]
    deadline_ms = float(contract["runtime"]["decision_deadline_ms"])
    locked_entries = [
        item for item in scenario_manifest["entries"] if item["split_id"] == "locked_test"
    ]
    if len(locked_entries) != int(contract["protocol"]["support_gates"]["minimum_locked_scenarios_per_seed"]):
        raise RuntimeError("locked scenario count does not match the frozen protocol")
    dataset_episodes = {}
    store = EpisodeStore(dataset_root)
    for path in store.iter_paths():
        header, columns = store.load(path.stem)
        if header.split_id != "locked_test":
            continue
        if header.scenario_id in dataset_episodes:
            raise RuntimeError(f"duplicate locked-test dataset episode {header.scenario_id}")
        dataset_episodes[header.scenario_id] = columns
    expected_locked_ids = {entry["scenario_id"] for entry in locked_entries}
    if set(dataset_episodes) != expected_locked_ids:
        raise RuntimeError("locked-test hypothesis dataset does not match the scenario matrix")
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
            hypothesis_metrics = evaluate_hypotheses_episode(
                policy, method_id, dataset_episodes[entry["scenario_id"]], input_config,
            )
            validate_hypothesis_episode_support(method_id, hypothesis_metrics)
            outcome["metrics"].update(hypothesis_metrics)
            record = {
                "experiment_id": f"tractor-paper-{method_id}-{seed}-{entry['scenario_id']}",
                "method_id": method_id, "seed": int(seed),
                "scenario_id": entry["scenario_id"], "split_id": "locked_test",
                "scenario_family": entry["family_id"],
                "obstacle_contract": entry["obstacle_contract"],
                "scenario_geometry_sha256": entry["scenario_geometry_sha256"],
                "vehicle_axis": entry["vehicle_axis"],
                "sensor_axis": entry["sensor_axis"],
                "localization_axis": entry["localization_axis"],
                "system_domain": entry["system_domain"],
                "model_fingerprint": contract["model"].fingerprint(),
                "data_fingerprint": report.index_sha256,
                "dataset_manifest_sha256": report.dataset_manifest_sha256,
                "training_fingerprint": contract["contract_sha256"],
                "checkpoint_sha256": checkpoint_manifest["training_payload_sha256"],
                "environment_attestation_sha256": canonical_sha256(attestation),
                "evaluation_source_identity": source_identity,
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
