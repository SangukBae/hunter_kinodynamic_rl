#!/usr/bin/env python3
"""Dedicated Environment-v2 -> sequence replay -> TRACTOR-TQC runner.

``collect`` records immutable development episodes from a live environment.
``train`` freezes a sequence index and performs only development-split
updates. ``pipeline`` runs both phases in that order; it never trains while
the replay population is still changing, preserving exact sampler resume.
"""

from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import numpy as np

from hunter_kinodynamic_rl.common.seed import enable_torch_determinism, seed_all
from hunter_kinodynamic_rl.config.loader import default_config_root, load_profile
from hunter_kinodynamic_rl.config.tractor import (
    canonical_sha256, load_tractor_contract, tractor_profile_model_mismatches,
)
from hunter_kinodynamic_rl.env.scenarios.seed_scheduler import SeedScheduler
from hunter_kinodynamic_rl.env.scenarios.tractor_environment_v2 import curriculum_stage
from hunter_kinodynamic_rl.evaluation.fingerprint import (
    architecture_fingerprint, training_profile_fingerprint,
)
from hunter_kinodynamic_rl.evaluation.provenance import collect_package_provenance
from hunter_kinodynamic_rl.rl.algorithms.tractor_tqc import TractorAgent
from hunter_kinodynamic_rl.rl.checkpointing.tractor import (
    load_training_generation, save_training_generation,
)
from hunter_kinodynamic_rl.rl.replay import EpisodeHeader, EpisodeStore, SequenceBuffer, SequenceIndex
from hunter_kinodynamic_rl.training.tractor_dataset_validation import validate_dataset
from hunter_kinodynamic_rl.training.tractor_episode_collector import (
    LABEL_SOURCE, TractorEpisodeRecorder, make_snapshot,
)
from hunter_kinodynamic_rl.training.tractor_preflight import run_tractor_preflight
from hunter_kinodynamic_rl.training.tractor_sequence_training import TractorSequenceTrainer


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_line(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _validate_profile(profile, model_config) -> None:
    if not profile.environment_v2.enabled:
        raise ValueError("TRACTOR collection requires environment_v2.enabled=true")
    if profile.action_space.mode != "trajectory":
        raise ValueError("TRACTOR requires the normalized [kappa,v_ref,L] trajectory action")
    expected = profile.observation.lidar_bins * profile.observation.frame_stack + profile.observation.robot_state_dim
    if expected != model_config.observation_dim:
        raise ValueError(
            f"profile observation dimension {expected} != TRACTOR model {model_config.observation_dim}"
        )
    if profile.counterfactual.num_candidates != model_config.num_candidates:
        raise ValueError("profile/model candidate counts differ")
    mismatches = tractor_profile_model_mismatches(profile, model_config)
    if mismatches:
        details = ", ".join(
            f"{name}: profile={expected!r}, model={observed!r}"
            for name, (expected, observed) in sorted(mismatches.items())
        )
        raise ValueError(f"profile/model physical contract differs: {details}")


def _episode_header(
    *, profile, contract, attestation, provenance, episode_id: str, seed: int,
    episode_index: int, step_count: int, termination_reason: str,
    start_utc: str, end_utc: str,
) -> EpisodeHeader:
    stage = curriculum_stage(profile.environment_v2, episode_index, "train")
    geometry_identity = {
        "generator": profile.environment_v2.contract_version,
        "profile": training_profile_fingerprint(profile),
        "seed": int(seed), "stage": dataclasses.asdict(stage),
    }
    geometry_sha = canonical_sha256(geometry_identity)
    dirty_digest = canonical_sha256({
        "tracked": provenance["tracked_diff_sha256"],
        "untracked": provenance["untracked_source_manifest_sha256"],
    })
    robot_hash = canonical_sha256(dataclasses.asdict(profile.robot))
    return EpisodeHeader(
        episode_id=episode_id,
        scenario_id=f"envv2-train-{seed}-{episode_index:06d}",
        split_id="development", seed=int(seed),
        software_commit=provenance["package_git_commit_sha"],
        dirty_state_digest=dirty_digest,
        container_image_digest=os.environ.get("HUNTER_CONTAINER_IMAGE_DIGEST", "unavailable-development"),
        resolved_config_hash=training_profile_fingerprint(profile),
        protocol_version=str(contract["protocol"]["protocol_version"]),
        environment_attestation_hash=canonical_sha256(attestation),
        robot_attestation_hash=robot_hash,
        observation_contract_hash=canonical_sha256({
            "observation_dim": contract["model"].observation_dim,
            "t_obs": contract["model"].t_obs, "n_scan": contract["model"].n_scan,
            "tail_dim": contract["model"].tail_dim,
        }),
        action_contract_hash=architecture_fingerprint(profile),
        trajectory_contract_hash=canonical_sha256(dataclasses.asdict(profile.trajectory)),
        start_utc=start_utc, end_utc=end_utc,
        termination_reason=termination_reason, step_count=step_count,
        sensor_source="environment_v2_step_synchronized_sensor_diagnostics",
        localization_source="noisy_odometry_with_covariance",
        controller_source="uniform_random_development_behavior_policy",
        clock_domain="gazebo_sim_time_with_monotonic_fallback",
        group_id=geometry_sha,
        scenario_family=f"environment_v2_curriculum_level_{stage.level}",
        obstacle_contract="dynamic" if stage.dynamic_obstacle_count > 0 else "static_only",
        scenario_geometry_sha256=geometry_sha,
    )


def collect_development_episodes(
    *, profile_name: str, variant: str, dataset_root: str | Path,
    episodes: int, run_seed: int, config_root: str | None = None,
    max_steps_per_episode: int | None = None,
) -> dict:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    profile = load_profile(profile_name, config_root)
    contract = load_tractor_contract(config_root, variant)
    _validate_profile(profile, contract["model"])
    source_root = str(Path(config_root or default_config_root()).resolve().parent)
    provenance = collect_package_provenance(source_root, __file__)
    store = EpisodeStore(dataset_root)
    existing = len(tuple(store.iter_paths()))
    scheduler = SeedScheduler(run_seed, profile.scenario, "train", episode_index=existing)
    rng = np.random.default_rng(run_seed)
    seed_all(run_seed)
    import rclpy
    from hunter_kinodynamic_rl.training.trainer_base import (
        EnvironmentClient, validate_training_environment_contract,
    )
    rclpy.init(args=None)
    env = None
    written = []
    try:
        env = EnvironmentClient(
            node_name="hunter_tractor_collect_client",
            telemetry_wait_timeout_sec=profile.runtime.risk_telemetry_wait_timeout_sec,
            reset_marker_wait_timeout_sec=profile.runtime.risk_telemetry_reset_marker_timeout_sec,
            wheelbase_m=profile.robot.wheelbase_m, track_width_m=profile.robot.track_width_m,
        )
        attestation = validate_training_environment_contract(env, profile, env.get_dimensions())
        if not env.set_explicit_seed_mode("train"):
            raise RuntimeError("environment rejected explicit_seed_mode='train'")
        for offset in range(episodes):
            episode_index = existing + offset
            if not env.set_curriculum_episode_index(episode_index):
                raise RuntimeError("environment rejected curriculum_episode_index")
            seed = scheduler.next_seed()
            if not env.seed(seed):
                raise RuntimeError(f"environment rejected seed {seed}")
            start_utc = _utc_now()
            state = env.reset()
            recorder = TractorEpisodeRecorder(profile, contract["model"])
            current = make_snapshot(
                state, contract["model"], diagnostics=None,
                pose_covariance=env.latest_pose_covariance, previous=None,
                nominal_dt_sec=profile.runtime.time_delta_sec,
            )
            final_reason = "infrastructure_failure"
            budget = max_steps_per_episode or profile.training.episode_length_steps
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
                recorder.append(current, action, next_snapshot, reward, done, target, collision, telemetry)
                current = next_snapshot
                final_reason = str(recorder.rows[-1]["termination_reason"])
                if done:
                    break
            if not recorder.rows[-1]["terminated"] and not recorder.rows[-1]["truncated"]:
                # A caller-supplied shorter collection budget is an operator
                # truncation and must never bootstrap from an incomplete run.
                recorder.rows[-1].update({
                    "truncated": True,
                    "termination_reason": "operator_stop",
                    "next_observation_valid": False,
                    "bellman_sample_valid": False,
                })
                final_reason = "operator_stop"
            columns = recorder.columns()
            episode_id = f"dev-v2-{episode_index:06d}-{seed}"
            header = _episode_header(
                profile=profile, contract=contract, attestation=attestation,
                provenance=provenance, episode_id=episode_id, seed=seed,
                episode_index=episode_index, step_count=len(recorder.rows),
                termination_reason=final_reason, start_utc=start_utc, end_utc=_utc_now(),
            )
            path, digest = store.append(header, dict(columns))
            written.append({"episode_id": episode_id, "path": str(path), "sha256": digest})
    finally:
        if env is not None:
            env.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    report = validate_dataset(dataset_root, loss_window=int(contract["data"]["loss_window"]))
    if not report.ok:
        raise RuntimeError(f"collected dataset failed validation: {report.errors}")
    return {
        "phase": "collect", "profile": profile_name, "variant": variant,
        "episodes_written": len(written), "episodes": written,
        "dataset_root": str(Path(dataset_root).resolve()), "row_count": report.row_count,
        "window_count": report.window_count, "label_source": LABEL_SOURCE,
        "formal_evidence": False,
    }


def train_from_sequence_replay(
    *, variant: str, dataset_root: str | Path, run_root: str | Path,
    seed: int, updates: int, batch_size: int | None = None,
    device: str = "cpu", config_root: str | None = None,
    resume: bool = False, checkpoint_tag: str = "latest",
    checkpoint_interval: int | None = None,
    formal_data: bool = False, scenario_manifest_path: str | Path | None = None,
) -> dict:
    if updates <= 0:
        raise ValueError("updates must be positive")
    contract = load_tractor_contract(config_root, variant)
    loss_window = int(contract["data"]["loss_window"])
    report = validate_dataset(
        dataset_root, loss_window=loss_window, formal=formal_data,
        scenario_manifest_path=scenario_manifest_path, config_root=config_root,
    )
    if not report.ok:
        raise RuntimeError(f"dataset validation failed: {report.errors}")
    store = EpisodeStore(dataset_root)
    index = SequenceIndex.build(store, loss_window=loss_window)
    index.validate_split_isolation()
    run_path = Path(run_root)
    run_path.mkdir(parents=True, exist_ok=True)
    index_path = run_path / "sequence_index.json"
    if resume:
        restored_index = SequenceIndex.load(index_path)
        if restored_index.sha256() != index.sha256():
            raise RuntimeError("dataset/index changed since the checkpointed training run")
        index = restored_index
    elif index_path.exists():
        raise FileExistsError(f"fresh training run refuses existing index: {index_path}")
    else:
        index.save(index_path)
    seed_all(seed)
    enable_torch_determinism(warn_only=True)
    agent = TractorAgent(contract["model"], contract["agent"], device=device, target_seed=seed)
    sampler = SequenceBuffer(store, index, seed=seed)
    if resume:
        load_training_generation(run_path / "checkpoints", checkpoint_tag, agent, sampler)
    trainer = TractorSequenceTrainer(agent, sampler, contract["loss_weights"])
    target_update = agent.update_step + updates
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
                {
                    "dataset_index_sha256": index.sha256(), "contract_sha256": contract["contract_sha256"],
                    "seed": seed, "evidence_status": "training_in_progress_not_evaluation_evidence",
                },
            )
    if trainer.risk_updates == 0:
        raise RuntimeError("no risk-head update was applied; candidate label path is not healthy")
    generation = save_training_generation(
        run_path / "checkpoints", "final", agent, sampler,
        {
            "dataset_index_sha256": index.sha256(), "contract_sha256": contract["contract_sha256"],
            "seed": seed, "evidence_status": "trained_not_held_out_evaluated",
            "formal_dataset_validated": formal_data,
        },
    )
    return {
        "phase": "train", "variant": variant, "seed": seed,
        "updates_applied": updates, "final_update_step": agent.update_step,
        "risk_updates_applied": trainer.risk_updates,
        "checkpoint_generation": generation, "run_root": str(run_path.resolve()),
        "elapsed_sec": time.monotonic() - started, "last_metrics": last_metrics,
        "performance_claim": "none_until_held_out_evaluation",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-root")
    parser.add_argument("--profile", default="tractor_local_dynamic_v2")
    parser.add_argument("--variant", choices=("a7", "a8", "a9"), default="a7")
    parser.add_argument("--dataset-root", default="runtime/tractor_sequence_v2")
    parser.add_argument("--run-root", default="runtime/tractor_tqc")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-json")
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect = subparsers.add_parser("collect", help="collect immutable Environment-v2 episodes")
    collect.add_argument("--episodes", type=int, required=True)
    collect.add_argument("--max-steps-per-episode", type=int)
    train = subparsers.add_parser("train", help="train from a frozen sequence replay population")
    train.add_argument("--updates", type=int, required=True)
    train.add_argument("--batch-size", type=int)
    train.add_argument("--checkpoint-interval", type=int)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--checkpoint-tag", default="latest")
    train.add_argument("--formal-data", action="store_true")
    train.add_argument("--scenario-manifest")
    pipeline = subparsers.add_parser("pipeline", help="collect, freeze replay, then train")
    pipeline.add_argument("--episodes", type=int, required=True)
    pipeline.add_argument("--updates", type=int, required=True)
    pipeline.add_argument("--batch-size", type=int)
    pipeline.add_argument("--checkpoint-interval", type=int)
    pipeline.add_argument("--max-steps-per-episode", type=int)
    pipeline.add_argument("--formal-data", action="store_true")
    pipeline.add_argument("--scenario-manifest")
    subparsers.add_parser("preflight", help="read-only TRACTOR contract and device gate")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        result = dataclasses.asdict(run_tractor_preflight(
            config_root=args.config_root, dataset_root=args.dataset_root,
            mode="development", device=args.device, variant=args.variant,
        ))
    elif args.command == "collect":
        result = collect_development_episodes(
            profile_name=args.profile, variant=args.variant, dataset_root=args.dataset_root,
            episodes=args.episodes, run_seed=args.seed, config_root=args.config_root,
            max_steps_per_episode=args.max_steps_per_episode,
        )
    elif args.command == "train":
        result = train_from_sequence_replay(
            variant=args.variant, dataset_root=args.dataset_root, run_root=args.run_root,
            seed=args.seed, updates=args.updates, batch_size=args.batch_size,
            device=args.device, config_root=args.config_root, resume=args.resume,
            checkpoint_tag=args.checkpoint_tag, checkpoint_interval=args.checkpoint_interval,
            formal_data=args.formal_data, scenario_manifest_path=args.scenario_manifest,
        )
    else:
        collected = collect_development_episodes(
            profile_name=args.profile, variant=args.variant, dataset_root=args.dataset_root,
            episodes=args.episodes, run_seed=args.seed, config_root=args.config_root,
            max_steps_per_episode=args.max_steps_per_episode,
        )
        trained = train_from_sequence_replay(
            variant=args.variant, dataset_root=args.dataset_root, run_root=args.run_root,
            seed=args.seed, updates=args.updates, batch_size=args.batch_size,
            device=args.device, config_root=args.config_root,
            checkpoint_interval=args.checkpoint_interval,
            formal_data=args.formal_data, scenario_manifest_path=args.scenario_manifest,
        )
        result = {"phase": "pipeline", "collect": collected, "train": trained}
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        Path(args.output_json).write_text(payload, encoding="utf-8")
    print(payload, end="")
    return result


if __name__ == "__main__":
    main()
