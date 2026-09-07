#!/usr/bin/env python3
"""Collect the frozen scenario plan with realized timestamp-aligned labels."""

from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np

from hunter_kinodynamic_rl.common.seed import seed_all
from hunter_kinodynamic_rl.config.loader import default_config_root, load_profile
from hunter_kinodynamic_rl.config.tractor import (
    canonical_sha256, load_tractor_contract, tractor_profile_model_mismatches,
)
from hunter_kinodynamic_rl.evaluation.fingerprint import (
    architecture_fingerprint, training_profile_fingerprint,
)
from hunter_kinodynamic_rl.evaluation.provenance import collect_package_provenance
from hunter_kinodynamic_rl.rl.replay import EpisodeHeader, EpisodeStore
from hunter_kinodynamic_rl.training.tractor_dataset_validation import validate_dataset
from hunter_kinodynamic_rl.training.tractor_episode_collector import (
    REALIZED_LABEL_SOURCE, TractorEpisodeRecorder, make_snapshot,
)
from hunter_kinodynamic_rl.training.tractor_scenario_plan import (
    validate_materialized_scenario_manifest,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _header(
    *, entry: dict, profile, contract: dict, attestation: dict, provenance: dict,
    step_count: int, termination_reason: str, start_utc: str, end_utc: str,
) -> EpisodeHeader:
    dirty_digest = canonical_sha256({
        "tracked": provenance["tracked_diff_sha256"],
        "untracked": provenance["untracked_source_manifest_sha256"],
    })
    return EpisodeHeader(
        episode_id=f"formal-{entry['scenario_id']}", scenario_id=entry["scenario_id"],
        split_id=entry["split_id"], seed=int(entry["seed"]),
        software_commit=provenance["package_git_commit_sha"], dirty_state_digest=dirty_digest,
        container_image_digest=os.environ.get(
            "HUNTER_CONTAINER_IMAGE_DIGEST", "unavailable-development",
        ),
        resolved_config_hash=training_profile_fingerprint(profile),
        protocol_version=str(contract["protocol"]["protocol_version"]),
        environment_attestation_hash=canonical_sha256(attestation),
        robot_attestation_hash=canonical_sha256(dataclasses.asdict(profile.robot)),
        observation_contract_hash=canonical_sha256({
            "observation_dim": contract["model"].observation_dim,
            "t_obs": contract["model"].t_obs, "n_scan": contract["model"].n_scan,
            "tail_dim": contract["model"].tail_dim,
        }),
        action_contract_hash=architecture_fingerprint(profile),
        trajectory_contract_hash=canonical_sha256(dataclasses.asdict(profile.trajectory)),
        start_utc=start_utc, end_utc=end_utc, termination_reason=termination_reason,
        step_count=step_count,
        sensor_source="fixed_scenario_step_synchronized_sensor_diagnostics",
        localization_source="profile_observation_with_covariance",
        controller_source="seeded_uniform_behavior_policy_v1",
        clock_domain="gazebo_sim_time",
        group_id=entry["group_id"], scenario_family=entry["family_id"],
        obstacle_contract=entry["obstacle_contract"],
        scenario_geometry_sha256=entry["scenario_geometry_sha256"],
    )


def collect_formal_comparison_data(
    *, scenario_manifest_path: str | Path, dataset_root: str | Path,
    profile_name: str = "tractor_local_dynamic", split_id: str = "all",
    behavior_seed: int = 739391, config_root: str | None = None,
    max_steps_per_episode: int | None = None, resume: bool = False,
) -> dict:
    if split_id not in {"all", "development", "calibration", "locked_test"}:
        raise ValueError("split_id must be all, development, calibration, or locked_test")
    manifest = validate_materialized_scenario_manifest(scenario_manifest_path, config_root)
    profile = load_profile(profile_name, config_root)
    contract = load_tractor_contract(config_root, "a7")
    mismatches = tractor_profile_model_mismatches(profile, contract["model"])
    if mismatches:
        raise ValueError(f"formal collector profile/model physical contract differs: {mismatches}")
    if profile.action_space.mode != "trajectory":
        raise ValueError("formal comparison collection requires trajectory actions")
    expected_dim = profile.observation.lidar_bins * profile.observation.frame_stack
    expected_dim += profile.observation.robot_state_dim
    if expected_dim != contract["model"].observation_dim:
        raise ValueError("formal collection profile does not match A7's observation contract")
    entries = [
        item for item in manifest["entries"]
        if split_id == "all" or item["split_id"] == split_id
    ]
    scenario_root = Path(scenario_manifest_path).resolve().parent
    source_root = str(Path(config_root or default_config_root()).resolve().parent)
    provenance = collect_package_provenance(source_root, __file__)
    store = EpisodeStore(dataset_root)
    existing = {}
    for path in store.iter_paths():
        header, _columns = store.load(path.stem)
        existing[header.scenario_id] = header
    duplicates = sorted(set(existing) & {item["scenario_id"] for item in entries})
    if duplicates and not resume:
        raise FileExistsError(
            f"formal scenarios already collected; pass --resume to verify/skip: {duplicates[:5]}"
        )
    seed_all(behavior_seed)
    import rclpy
    from hunter_kinodynamic_rl.training.trainer_base import (
        EnvironmentClient, validate_training_environment_contract,
    )
    rclpy.init(args=None)
    env = None
    written = []
    label_totals = {"valid_candidate_labels": 0, "observed_events": 0, "valid_severity_bins": 0}
    try:
        env = EnvironmentClient(
            node_name="hunter_formal_comparison_collect_client",
            telemetry_wait_timeout_sec=profile.runtime.risk_telemetry_wait_timeout_sec,
            reset_marker_wait_timeout_sec=profile.runtime.risk_telemetry_reset_marker_timeout_sec,
            wheelbase_m=profile.robot.wheelbase_m, track_width_m=profile.robot.track_width_m,
        )
        attestation = validate_training_environment_contract(env, profile, env.get_dimensions())
        for entry in entries:
            if entry["scenario_id"] in existing:
                header = existing[entry["scenario_id"]]
                if (
                    header.split_id != entry["split_id"]
                    or header.scenario_geometry_sha256 != entry["scenario_geometry_sha256"]
                ):
                    raise RuntimeError(f"existing scenario lineage mismatch: {entry['scenario_id']}")
                continue
            scenario_path = (scenario_root / entry["relative_path"]).resolve()
            if scenario_root != scenario_path and scenario_root not in scenario_path.parents:
                raise RuntimeError("scenario path escapes materialized root")
            if not env.set_scenario_override(str(scenario_path)):
                raise RuntimeError(f"environment rejected scenario override {scenario_path}")
            rng = np.random.default_rng(int(entry["seed"]) ^ int(behavior_seed))
            start_utc = _utc_now()
            state, reset_diagnostics = env.reset_with_diagnostics()
            recorder = TractorEpisodeRecorder(profile, contract["model"])
            current = make_snapshot(
                state, contract["model"], diagnostics=reset_diagnostics,
                pose_covariance=env.latest_pose_covariance, previous=None,
                nominal_dt_sec=profile.runtime.time_delta_sec,
            )
            termination_reason = "infrastructure_failure"
            budget = max_steps_per_episode or profile.evaluation.max_episode_steps
            for _step in range(budget):
                action = rng.uniform(-1.0, 1.0, 3).astype(np.float32)
                next_state, reward, done, target, collision, _distance, telemetry, diagnostics = env.step(action)
                next_snapshot = make_snapshot(
                    next_state, contract["model"], diagnostics=diagnostics,
                    pose_covariance=env.latest_pose_covariance, previous=current,
                    previous_action=action,
                    previous_command=(telemetry.published_speed_mps, telemetry.published_steering_rad),
                    nominal_dt_sec=profile.runtime.time_delta_sec,
                )
                recorder.append(
                    current, action, next_snapshot, reward, done, target, collision, telemetry,
                )
                current = next_snapshot
                termination_reason = str(recorder.rows[-1]["termination_reason"])
                if done:
                    break
            if not recorder.rows[-1]["terminated"] and not recorder.rows[-1]["truncated"]:
                recorder.rows[-1].update({
                    "truncated": True, "termination_reason": "operator_stop",
                    "next_observation_valid": False, "bellman_sample_valid": False,
                })
                termination_reason = "operator_stop"
            label_report = recorder.finalize_realized_labels()
            for name in label_totals:
                label_totals[name] += int(label_report[name])
            columns = recorder.columns()
            header = _header(
                entry=entry, profile=profile, contract=contract, attestation=attestation,
                provenance=provenance, step_count=len(recorder.rows),
                termination_reason=termination_reason, start_utc=start_utc, end_utc=_utc_now(),
            )
            path, digest = store.append(header, dict(columns))
            written.append({
                "scenario_id": entry["scenario_id"], "path": str(path), "sha256": digest,
            })
    finally:
        if env is not None:
            try:
                env.set_scenario_override("")
            finally:
                env.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    available_scenarios = set(existing) | {item["scenario_id"] for item in written}
    required_scenarios = {item["scenario_id"] for item in manifest["entries"]}
    full_plan_present = available_scenarios == required_scenarios
    report = validate_dataset(
        dataset_root, loss_window=int(contract["data"]["loss_window"]),
        formal=full_plan_present, scenario_manifest_path=scenario_manifest_path,
        config_root=config_root,
    )
    if not report.ok:
        raise RuntimeError(f"formal comparison dataset failed validation: {report.errors}")
    return {
        "phase": "collect_formal_comparison_data", "split_id": split_id,
        "episodes_requested": len(entries), "episodes_written": len(written),
        "episodes_skipped": len(entries) - len(written),
        "dataset_root": str(Path(dataset_root).resolve()),
        "scenario_manifest_sha256": manifest["artifact_manifest_sha256"],
        "dataset_index_sha256": report.index_sha256,
        "label_source": REALIZED_LABEL_SOURCE, "label_totals": label_totals,
        "formal_dataset_complete": full_plan_present,
        "performance_claim": "none_data_collection_only",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-manifest", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--profile", default="tractor_local_dynamic")
    parser.add_argument(
        "--split", dest="split_id", default="all",
        choices=("all", "development", "calibration", "locked_test"),
    )
    parser.add_argument("--behavior-seed", type=int, default=739391)
    parser.add_argument("--max-steps-per-episode", type=int)
    parser.add_argument("--config-root")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    result = collect_formal_comparison_data(
        scenario_manifest_path=args.scenario_manifest, dataset_root=args.dataset_root,
        profile_name=args.profile, split_id=args.split_id, behavior_seed=args.behavior_seed,
        max_steps_per_episode=args.max_steps_per_episode, config_root=args.config_root,
        resume=args.resume,
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        Path(args.output_json).write_text(payload, encoding="utf-8")
    print(payload, end="")
    return result


if __name__ == "__main__":
    main()
