#!/usr/bin/env python3
"""Prepare, resume, benchmark and promote the Stage-2 L0-L5 campaign."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import rclpy

from hunter_kinodynamic_rl.config.loader import load_profile
from hunter_kinodynamic_rl.evaluation.fingerprint import (
    architecture_fingerprint,
    architecture_fingerprint_from_resolved_config,
    local_training_contract_fingerprint,
    local_training_contract_fingerprint_from_resolved_config,
    training_profile_fingerprint,
    training_profile_fingerprint_from_resolved_config,
)
from hunter_kinodynamic_rl.evaluation.local_subgoal_benchmark import (
    load_local_benchmark_manifest,
    run_local_subgoal_benchmark,
)
from hunter_kinodynamic_rl.evaluation.stage2_local_baselines import (
    DEFAULT_TRAINING_SEEDS,
    STAGE2_PROFILE_BY_LABEL,
    Stage2ContractError,
    aggregate_stage2,
    benchmark_artifact_path,
    checkpoint_manifest,
    discover_run_dir,
    experiment_manifest_path,
    prepare_stage2,
    promote_preregistered_local,
    resume_checkpoint_tag,
    shared_benchmark_manifest_path,
    stage2_status,
    training_complete,
    training_root,
)
from hunter_kinodynamic_rl.nodes.evaluation_node import (
    build_agent,
    environment_client_kwargs,
    expected_dims,
    validate_live_environment,
)
from hunter_kinodynamic_rl.rl.checkpointing import manager as checkpoint_manager
from hunter_kinodynamic_rl.training.trainer_base import (
    EnvironmentClient,
    validate_training_environment_contract,
)


DEFAULT_STAGE_ROOT = "runtime/stage2_local_baselines"
DEFAULT_PROMOTION_DIR = "runtime/experiments/local_frozen/checkpoints"


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def _campaign_state(stage_root: str, status: str, **extra) -> None:
    payload = {"status": status, "pid": os.getpid(), "updated_at_unix": time.time(), **extra}
    _atomic_json(Path(stage_root).resolve() / "campaign_status.json", payload)


def _start_process(command: list[str], log_path: Path) -> tuple[subprocess.Popen, object]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("ab", buffering=0)
    process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    return process, handle


def _stop_process(process: subprocess.Popen | None, handle=None) -> None:
    if process is not None and process.poll() is None:
        for sig, timeout in ((signal.SIGINT, 10.0), (signal.SIGTERM, 5.0), (signal.SIGKILL, 1.0)):
            try:
                os.killpg(process.pid, sig)
                process.wait(timeout=timeout)
                break
            except ProcessLookupError:
                break
            except subprocess.TimeoutExpired:
                continue
    if handle is not None:
        handle.close()


def _assert_checkpoint_identity(profile, manifest: dict, manifest_path: Path) -> None:
    resolved = manifest.get("resolved_config") or {}
    attestation = manifest.get("training_environment_attestation") or {}
    try:
        checks = {
            "profile_name": manifest.get("profile_name") == profile.name,
            "architecture_fingerprint": (
                architecture_fingerprint_from_resolved_config(resolved) == architecture_fingerprint(profile)
            ),
            "local_training_contract_fingerprint": (
                local_training_contract_fingerprint_from_resolved_config(resolved)
                == local_training_contract_fingerprint(profile)
                == manifest.get("local_training_contract_fingerprint")
            ),
            "training_profile_fingerprint": (
                training_profile_fingerprint_from_resolved_config(resolved)
                == training_profile_fingerprint(profile)
            ),
            "training_environment_attestation": (
                isinstance(attestation, dict)
                and attestation.get("profile_name") == profile.name
                and attestation.get("urdf_robot_name") == profile.robot.name
                and attestation.get("training_profile_fingerprint_sha256")
                == training_profile_fingerprint(profile)
            ),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise Stage2ContractError(f"{manifest_path}: checkpoint identity is incomplete: {exc}") from exc
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise Stage2ContractError(f"{manifest_path}: checkpoint identity mismatch: {failed}")


def _prune_checkpoint_history(run_dir: Path, keep_last_n: int) -> None:
    """Bound campaign disk use while retaining every published tag."""
    checkpoint_dir = run_dir / "checkpoints"
    # The manager deliberately requires mark-then-sweep. A zero grace period
    # is safe here because training and benchmark have both stopped using
    # this run; per-generation locks still prevent deletion under a reader.
    checkpoint_manager.prune_orphan_generations(
        str(checkpoint_dir), keep_last_n=keep_last_n, min_age_sec=0.0,
    )
    checkpoint_manager.prune_orphan_generations(
        str(checkpoint_dir), keep_last_n=keep_last_n, min_age_sec=0.0,
    )


def _run_formal_benchmark(profile_path: Path, run_dir: Path, artifact_path: Path, manifest_path: Path) -> None:
    profile = load_profile(str(profile_path))
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_meta_path = checkpoint_dir / "final" / "manifest.json"
    with checkpoint_meta_path.open() as handle:
        checkpoint_meta = json.load(handle)
    _assert_checkpoint_identity(profile, checkpoint_meta, checkpoint_meta_path)

    rclpy.init(args=None)
    env = None
    try:
        env = EnvironmentClient(node_name="hunter_stage2_benchmark_client", **environment_client_kwargs(profile))
        live_dims = env.get_dimensions()
        # The normal TrainerBase path performs this before it creates an
        # agent/replay buffer.  Repeat it for the in-process formal benchmark
        # so a restarted campaign cannot benchmark an improved checkpoint
        # against a baseline Gazebo model (or a stale, merely shape-compatible
        # environment profile).
        validate_training_environment_contract(env, profile, live_dims)
        expected_state_dim, expected_action_dim = expected_dims(profile)
        if (live_dims.state_dim, live_dims.action_dim) != (expected_state_dim, expected_action_dim):
            raise Stage2ContractError(
                f"live dimensions {(live_dims.state_dim, live_dims.action_dim)} != "
                f"profile dimensions {(expected_state_dim, expected_action_dim)}"
            )
        validate_live_environment(
            env, profile, checkpoint_meta["profile_name"], live_dims,
            architecture_fingerprint_from_resolved_config(checkpoint_meta["resolved_config"]),
        )
        dims = SimpleNamespace(
            state_dim=live_dims.state_dim, action_dim=live_dims.action_dim, max_action=live_dims.max_action,
        )
        agent = build_agent(profile, dims)
        loaded = checkpoint_manager.load_generation(
            str(checkpoint_dir), "final", agent.checkpoint_components(), map_location=agent.device,
        )
        verified_meta = loaded["manifest"]
        artifact = run_local_subgoal_benchmark(
            profile,
            agent,
            manifest=load_local_benchmark_manifest(str(manifest_path)),
            checkpoint_generation=verified_meta["generation"],
            checkpoint_sha256=verified_meta["pt_sha256"],
            env=env,
            benchmark_kind="formal",
        )
        _atomic_json(artifact_path, artifact)
    finally:
        if env is not None:
            env.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _selected_rows(contract: dict, labels: list[str] | None, seeds: list[int] | None):
    chosen_labels = labels or list(STAGE2_PROFILE_BY_LABEL)
    invalid = sorted(set(chosen_labels) - set(STAGE2_PROFILE_BY_LABEL))
    if invalid:
        raise Stage2ContractError(f"unknown labels: {invalid}")
    chosen_seeds = seeds or list(contract["training_seeds"])
    invalid_seeds = sorted(set(chosen_seeds) - set(contract["training_seeds"]))
    if invalid_seeds:
        raise Stage2ContractError(f"seeds outside experiment contract: {invalid_seeds}")
    for label in chosen_labels:
        for seed in chosen_seeds:
            yield label, seed


def execute_campaign(args) -> None:
    if args.keep_orphan_generations < 0:
        raise Stage2ContractError("--keep-orphan-generations must be >= 0")
    root = Path(args.stage_root).resolve()
    contract = prepare_stage2(root)
    gazebo = gazebo_log = None
    if not args.no_launch_gazebo:
        gazebo, gazebo_log = _start_process(
            ["ros2", "launch", "hunter_se_gazebo", "simulate_hunter_se_ignition.launch.py",
             "rviz:=false", "headless:=true", "model_variant:=improved"],
            root / "logs" / "gazebo.log",
        )
        time.sleep(args.gazebo_startup_sec)
        if gazebo.poll() is not None:
            raise Stage2ContractError(f"Gazebo exited early; inspect {root / 'logs' / 'gazebo.log'}")
    try:
        for label, seed in _selected_rows(contract, args.labels, args.seeds):
            key = f"{label}/seed_{seed}"
            profile_path = Path(contract["profiles"][key]["path"])
            run_dir = discover_run_dir(root, label, seed)
            artifact_path = benchmark_artifact_path(root, label, seed)
            if run_dir is not None and training_complete(run_dir, profile_path) and artifact_path.is_file():
                print(f"SKIP {key}: training and formal benchmark already complete", flush=True)
                continue

            env_process = env_log = None
            try:
                env_process, env_log = _start_process(
                    ["ros2", "run", "hunter_kinodynamic_rl", "environment_node.py", "--ros-args",
                     "-p", f"profile:={profile_path}"],
                    root / "logs" / label / f"seed_{seed}_environment.log",
                )
                if not (run_dir is not None and training_complete(run_dir, profile_path)):
                    train_command = [
                        "ros2", "run", "hunter_kinodynamic_rl", "train_node.py", "--ros-args",
                        "-p", f"profile:={profile_path}", "-p", f"run_root:={training_root(root, label, seed)}",
                    ]
                    if run_dir is not None:
                        resume_tag = resume_checkpoint_tag(run_dir)
                        if resume_tag is None:
                            raise Stage2ContractError(
                                f"{key}: interrupted run has no resumable latest/best checkpoint: {run_dir}"
                            )
                        train_command += ["-p", f"resume_run_dir:={run_dir}", "-p", f"resume_checkpoint_tag:={resume_tag}"]
                    print(f"TRAIN {key}", flush=True)
                    train_log_path = root / "logs" / label / f"seed_{seed}_training.log"
                    train_process, train_log = _start_process(train_command, train_log_path)
                    train_code = train_process.wait()
                    train_log.close()
                    if train_code != 0:
                        raise Stage2ContractError(f"{key}: training exited {train_code}; inspect {train_log_path}")
                    run_dir = discover_run_dir(root, label, seed)
                if run_dir is None or not training_complete(run_dir, profile_path):
                    raise Stage2ContractError(f"{key}: final checkpoint did not reach max_timesteps")
                if not artifact_path.is_file():
                    print(f"BENCHMARK {key}", flush=True)
                    _run_formal_benchmark(
                        profile_path, run_dir, artifact_path, shared_benchmark_manifest_path(root),
                    )
                if args.prune_checkpoint_history:
                    _prune_checkpoint_history(run_dir, args.keep_orphan_generations)
            finally:
                _stop_process(env_process, env_log)
    finally:
        _stop_process(gazebo, gazebo_log)

    # Partial --labels/--seeds executions remain resumable but are not
    # allowed to aggregate or promote an incomplete matrix.
    if args.labels is None and args.seeds is None:
        aggregate_stage2(root)
        if args.promote:
            result = promote_preregistered_local(root, target_dir=args.promotion_dir)
            print(json.dumps(result.__dict__, indent=2, sort_keys=True), flush=True)
            if not result.accepted:
                raise Stage2ContractError(f"promotion rejected: {result.reasons}")


def _print_status(stage_root: str) -> None:
    report = stage2_status(stage_root)
    campaign_path = Path(stage_root).resolve() / "campaign_status.json"
    if campaign_path.is_file():
        with campaign_path.open() as handle:
            campaign = json.load(handle)
        print(f"campaign={campaign.get('status')} pid={campaign.get('pid')}")
    complete_train = sum(row["training_complete"] for row in report["rows"].values())
    complete_benchmark = sum(row["benchmark_complete"] for row in report["rows"].values())
    print(f"training={complete_train}/{len(report['rows'])} benchmark={complete_benchmark}/{len(report['rows'])}")
    for key, row in report["rows"].items():
        print(f"{key}: train={'done' if row['training_complete'] else 'pending'}, "
              f"benchmark={'done' if row['benchmark_complete'] else 'pending'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", default=DEFAULT_STAGE_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    subparsers.add_parser("status")
    subparsers.add_parser("aggregate")
    execute = subparsers.add_parser("execute")
    execute.add_argument("--labels", nargs="+", choices=tuple(STAGE2_PROFILE_BY_LABEL))
    execute.add_argument("--seeds", nargs="+", type=int)
    execute.add_argument("--no-launch-gazebo", action="store_true")
    execute.add_argument("--gazebo-startup-sec", type=float, default=10.0)
    execute.add_argument("--promote", action="store_true")
    execute.add_argument("--promotion-dir", default=DEFAULT_PROMOTION_DIR)
    execute.add_argument("--no-prune-checkpoint-history", dest="prune_checkpoint_history", action="store_false")
    execute.add_argument("--keep-orphan-generations", type=int, default=1)
    execute.set_defaults(prune_checkpoint_history=True)
    promote = subparsers.add_parser("promote")
    promote.add_argument("--promotion-dir", default=DEFAULT_PROMOTION_DIR)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            print(json.dumps(prepare_stage2(args.stage_root), indent=2, sort_keys=True))
        elif args.command == "status":
            _print_status(args.stage_root)
        elif args.command == "aggregate":
            print(json.dumps(aggregate_stage2(args.stage_root), indent=2, sort_keys=True))
        elif args.command == "promote":
            result = promote_preregistered_local(args.stage_root, target_dir=args.promotion_dir)
            print(json.dumps(result.__dict__, indent=2, sort_keys=True))
            if not result.accepted:
                return 2
        else:
            _campaign_state(args.stage_root, "running")
            execute_campaign(args)
            _campaign_state(args.stage_root, "completed")
        return 0
    except (Stage2ContractError, OSError, RuntimeError, KeyboardInterrupt) as exc:
        if args.command == "execute":
            _campaign_state(args.stage_root, "failed", error=str(exc))
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
