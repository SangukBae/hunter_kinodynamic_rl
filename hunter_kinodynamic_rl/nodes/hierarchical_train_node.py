#!/usr/bin/env python3
"""``ros2 run hunter_kinodynamic_rl hierarchical_train_node.py`` -- Phase 4
Global RL training entrypoint (plan section 8.9).

``live=False`` drives the ROS-free ``SimplifiedKinematicLocalExecutor`` for
fast deterministic tests. ``live=True`` constructs one long-lived
``LiveGazeboLocalExecutor`` and trains the Global DQN against the real frozen
Local TQC, live LiDAR/odometry and Gazebo services. The two modes share the
same ``HierarchicalTrainingLoop`` and checkpoint contract; a live run never
falls back to the simplified executor when its Local checkpoint is missing
or incompatible.

Local policy is ALWAYS frozen: this node's OWN checkpoint save/resume cycle
only ever covers the Global agent + Global replay buffer, mirroring
``training/trainer_base.py``'s "generation" checkpoint layout
(``rl.checkpointing.manager``) but scoped to Global-only components.
"""

from __future__ import annotations

import dataclasses
import json
import os
import uuid
from datetime import datetime
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node

from hunter_kinodynamic_rl.common.seed import enable_torch_determinism, seed_all
from hunter_kinodynamic_rl.config.loader import load_profile
from hunter_kinodynamic_rl.evaluation.fingerprint import (
    compare_local_checkpoint_identity,
    hierarchical_architecture_fingerprint,
    sha256_of_obj,
)
from hunter_kinodynamic_rl.navigation.memory.topological_graph import NODE_FEATURE_NAMES
from hunter_kinodynamic_rl.navigation.global_rl.replay_schema import SCHEMA_VERSION as GLOBAL_REPLAY_SCHEMA_VERSION
from hunter_kinodynamic_rl.rl.checkpointing import manager as ckpt_manager
from hunter_kinodynamic_rl.rl.checkpointing.rng_state import (
    capture_rng_state, load_rng_state_file, restore_rng_state, save_rng_state_file,
)
from hunter_kinodynamic_rl.training.train_hierarchical_dqn import HierarchicalTrainingLoop


def _min_turning_radius_m(robot) -> float:
    return 1.0 / robot.max_curvature


class ResumeIncompatibleError(RuntimeError):
    """Raised (never swallowed) when a resume checkpoint's own recorded
    architecture/schema/Local-checkpoint identity doesn't match the CURRENT
    profile/runtime -- item 4: fail fast rather than silently resuming
    training against a checkpoint whose network shapes, observation
    contract, or frozen Local policy have since changed."""


def _check_resume_compatibility(manifest: dict, profile, loop, live_executor) -> None:
    """Every one of item 4's required fail-fast comparisons, checked
    BEFORE any manifest state is applied to ``loop``/``agent``. Compares
    against values already embedded in the manifest at save time (see the
    ``meta.update(...)`` block below) -- never re-derives them a different
    way that could itself drift from what was actually checkpointed."""
    mismatches = []

    required_fields = (
        "hierarchical_architecture_fingerprint", "resolved_config", "resolved_config_hash",
        "map_channel_names", "candidate_feature_names", "node_feature_names",
        "max_nodes_in_observation", "memory_enabled", "feasibility_enabled",
        "global_risk_enabled", "loop_state", "training_steps", "total_optimizer_updates",
        "epsilon_schedule_basis", "global_replay_schema_version", "rng_state_path",
        "rng_state_sha256", "rng_state_schema_version", "local_checkpoint_manifest_summary",
    )
    missing = [name for name in required_fields if name not in manifest]
    if missing:
        mismatches.append(f"missing required manifest field(s): {missing}")

    current_fp = hierarchical_architecture_fingerprint(profile)
    recorded_fp = manifest.get("hierarchical_architecture_fingerprint")
    if recorded_fp != current_fp:
        mismatches.append(
            f"hierarchical_architecture_fingerprint: checkpoint={recorded_fp!r} current={current_fp!r}"
        )

    recorded_config = manifest.get("resolved_config")
    if recorded_config is not None:
        current_hash = sha256_of_obj(dataclasses.asdict(profile))
        recorded_hash = sha256_of_obj(recorded_config)
        if manifest.get("resolved_config_hash") != recorded_hash:
            mismatches.append(
                "resolved_config_hash: manifest="
                f"{manifest.get('resolved_config_hash')!r} derived={recorded_hash!r}"
            )
        if current_hash != recorded_hash:
            mismatches.append(f"resolved_config hash: checkpoint={recorded_hash} current={current_hash}")

    for field_name, current_value in (
        ("map_channel_names", list(loop.map_channel_names)),
        ("candidate_feature_names", list(loop.candidate_feature_names)),
        ("node_feature_names", list(NODE_FEATURE_NAMES) if loop.memory_enabled else []),
        ("max_nodes_in_observation", loop.max_nodes),
        ("memory_enabled", loop.memory_enabled),
        ("feasibility_enabled", loop.feasibility_enabled),
        ("global_risk_enabled", loop.global_risk_enabled),
    ):
        recorded_value = manifest.get(field_name)
        if field_name in manifest and recorded_value != current_value:
            mismatches.append(f"{field_name}: checkpoint={recorded_value!r} current={current_value!r}")

    if manifest.get("global_replay_schema_version") != GLOBAL_REPLAY_SCHEMA_VERSION:
        mismatches.append(
            "global_replay_schema_version: checkpoint="
            f"{manifest.get('global_replay_schema_version')!r} current={GLOBAL_REPLAY_SCHEMA_VERSION!r}"
        )
    if manifest.get("rng_state_schema_version") != 1:
        mismatches.append(
            f"rng_state_schema_version: checkpoint={manifest.get('rng_state_schema_version')!r} current=1"
        )
    if manifest.get("epsilon_schedule_basis") != "agent_training_steps":
        mismatches.append(
            "epsilon_schedule_basis: checkpoint="
            f"{manifest.get('epsilon_schedule_basis')!r} current='agent_training_steps'"
        )

    recorded_local = manifest.get("local_checkpoint_manifest_summary")
    if live_executor is not None:
        mismatches.extend(compare_local_checkpoint_identity(recorded_local, live_executor.checkpoint_manifest))
    elif recorded_local is not None:
        mismatches.append(
            "local_checkpoint_manifest_summary: checkpoint recorded a real Local checkpoint (live=True) "
            "but this resume is live=False -- refusing to silently drop the Local-checkpoint-driven replay "
            "history's own provenance"
        )

    if mismatches:
        raise ResumeIncompatibleError(
            "hierarchical_train_node resume: checkpoint is incompatible with the current profile/runtime "
            "(item 4 fail-fast) -- refusing to resume:\n  " + "\n  ".join(mismatches)
        )


def train_hierarchical_dqn(
    profile_name: str, run_root: str = "runtime/hierarchical_experiments",
    num_missions: int = 1000, resume_run_dir: Optional[str] = None, resume_checkpoint_tag: str = "latest",
    logger=print, live: bool = False, require_promoted_local: bool = True,
) -> dict:
    """ROS-free training driver -- importable and callable directly (e.g.
    from a test or a plain script) without ``rclpy``. ``main()`` below is
    the thin ``ros2 run`` wrapper around this function.

    ``live=False`` (default): unchanged -- drives
    ``SimplifiedKinematicLocalExecutor`` (ROS-free, no Gazebo). ``live=True``
    requires ``rclpy`` already initialized by the caller (``main()`` below
    does this) and a reachable Gazebo world already running the Phase 3
    long-horizon wall pool's target ``world_name``/``robot_entity_name`` --
    constructs ONE long-lived
    ``navigation.local_rl.live_gazebo_executor.LiveGazeboLocalExecutor`` and
    injects its ``bind_mission`` as ``HierarchicalTrainingLoop``'s
    ``local_executor_factory``, so every mission drives the REAL frozen
    Local TQC checkpoint (``profile.hierarchical_training.local_checkpoint_dir``/
    ``local_checkpoint_name``) over live Gazebo instead of the point-robot
    stand-in. Raises ``LocalCheckpointError`` immediately (never falls back
    to the ROS-free stand-in) if that checkpoint is missing/incompatible.

    ``require_promoted_local`` (requirement C, default True): for a FRESH
    (``resume_run_dir is None``) ``live=True`` run, requires the resolved
    Local checkpoint directory to carry a ``promotion_manifest.json``
    (:func:`evaluation.local_promotion.is_local_checkpoint_promoted`) --
    raises ``RuntimeError`` before touching Gazebo otherwise. A RESUMED run
    is governed instead by ``_check_resume_compatibility``'s own stricter
    Local-identity check against what the Global checkpoint itself already
    recorded, so this flag applies only to starting a brand-new Global run.
    Set False only for tests/dev iteration that intentionally drive an
    unpromoted checkpoint."""
    import torch

    from hunter_kinodynamic_rl.navigation.global_rl.agent import GlobalDQNAgent
    from hunter_kinodynamic_rl.navigation.global_rl.observation import resolve_map_channel_names

    profile = load_profile(profile_name)
    if not profile.hierarchical_training.enabled:
        raise SystemExit(
            f"profile {profile_name!r} does not enable hierarchical_training -- "
            "use a Phase 4 profile (e.g. hierarchical_phase4)"
        )
    seed_all(profile.training.seed)
    enable_torch_determinism(warn_only=True)

    run_dir = resume_run_dir or os.path.join(
        run_root, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{profile_name}_seed{profile.training.seed}",
    )
    os.makedirs(run_dir, exist_ok=True)

    # Local and Global devices are independently configurable.  The shipped
    # live Phase-4 profile selects CPU for both: CUDA-authored checkpoints
    # are remapped by LiveGazeboLocalExecutor and no CUDA runtime is brought
    # into the rclpy process, avoiding the previously observed publisher-
    # context corruption.  GPU remains an explicit opt-in, never an implicit
    # requirement for live=True.
    map_channels = len(resolve_map_channel_names(profile.global_rl))
    memory_enabled_for_agent = bool(profile.memory.enabled and profile.global_rl.topology_feedback_enabled)
    max_nodes_for_agent = profile.memory.max_nodes_in_observation if memory_enabled_for_agent else 0
    requested_global_device = profile.hierarchical_training.global_training_device
    if requested_global_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("hierarchical_training.global_training_device='cuda' but CUDA is unavailable")
    global_device = None if requested_global_device == "auto" else requested_global_device
    agent = GlobalDQNAgent(
        profile.global_rl, map_channels, device=global_device, max_nodes=max_nodes_for_agent)
    logger(f"GlobalDQNAgent device={agent.device}")

    live_executor = None
    local_executor_factory = None
    if live:
        from hunter_kinodynamic_rl.navigation.local_rl.live_gazebo_executor import LiveGazeboLocalExecutor

        if resume_run_dir is None and require_promoted_local:
            from hunter_kinodynamic_rl.evaluation.local_promotion import is_local_checkpoint_promoted

            if not is_local_checkpoint_promoted(
                profile.hierarchical_training.local_checkpoint_dir,
                profile.hierarchical_training.local_checkpoint_name,
            ):
                raise RuntimeError(
                    "hierarchical_train_node: fresh Global training run requires a PROMOTED Local checkpoint "
                    f"(no promotion_manifest.json found under "
                    f"{profile.hierarchical_training.local_checkpoint_dir!r}/"
                    f"{profile.hierarchical_training.local_checkpoint_name!r}) -- run the Local subgoal "
                    "benchmark and evaluation.local_promotion.promote_local_checkpoint() first, or pass "
                    "require_promoted_local=False to intentionally override for dev/testing"
                )

        live_executor = LiveGazeboLocalExecutor(
            profile, profile.long_horizon_world,
            profile.hierarchical_training.local_checkpoint_dir, profile.hierarchical_training.local_checkpoint_name,
        )
        local_executor_factory = live_executor.bind_mission

    # Requirement E: whenever this profile's ablation tier needs
    # feasibility/global-risk feedback, build a REAL LocalFeasibilityEvaluator
    # bound to the live executor's own frozen Local policy + live sensor
    # state -- HierarchicalTrainingLoop itself fail-fasts if this stays
    # None while those flags are on, so a live=False (ROS-free) run with
    # such a profile is rejected here explicitly, before even reaching
    # that constructor check, with a clearer live-specific message.
    needs_local_evaluator = (
        profile.feasibility.enabled
        and (profile.global_rl.feasibility_feedback_enabled or profile.global_rl.global_risk_feedback_enabled)
    )
    local_evaluator = None
    if needs_local_evaluator:
        if live_executor is None:
            raise RuntimeError(
                f"profile {profile_name!r} enables feasibility_feedback_enabled/global_risk_feedback_enabled "
                "but hierarchical_train_node was started without -p live:=true -- the ROS-free "
                "SimplifiedKinematicLocalExecutor stand-in has no real Local policy/sensors to build a "
                "LocalFeasibilityEvaluator from; run with -p live:=true or use a profile without these flags"
            )
        local_evaluator = live_executor.build_local_feasibility_evaluator(
            inference_timeout_sec=profile.feasibility.local_evaluator_inference_timeout_sec,
            max_snapshot_age_sec=profile.feasibility.local_evaluator_max_snapshot_age_sec,
        )

    loop = HierarchicalTrainingLoop(
        profile.global_rl, profile.hierarchy, profile.long_horizon_world, profile.mapping,
        profile.hierarchical_training, robot_radius_m=profile.robot.collision_radius_m,
        min_turning_radius_m=_min_turning_radius_m(profile.robot), wheelbase_m=profile.robot.wheelbase_m,
        seed=profile.training.seed, mode="train", memory_cfg=profile.memory, feasibility_cfg=profile.feasibility,
        local_executor_factory=local_executor_factory, local_evaluator=local_evaluator,
    )
    assert loop.max_nodes == max_nodes_for_agent, (
        f"loop.max_nodes ({loop.max_nodes}) != max_nodes_for_agent ({max_nodes_for_agent}) computed for the "
        "already-constructed GlobalDQNAgent -- the two max_nodes formulas have drifted apart"
    )
    # item 4: a DEDICATED action-selection RNG stream (never numpy's global
    # RNG) -- persisted/restored across resume exactly like every other
    # checkpointed RNG (see rl.checkpointing.rng_state and the save/resume
    # blocks below).
    rng = np.random.RandomState(profile.training.seed)

    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    total_updates = 0
    if resume_run_dir:
        with ckpt_manager.load_generation_lease(checkpoint_dir, resume_checkpoint_tag,
                                                 agent.checkpoint_components(),
                                                 map_location=str(agent.device)) as result:
            manifest = result["manifest"]
            # item 4: fail-fast BEFORE any manifest state is applied --
            # architecture fingerprint, resolved profile/config hash,
            # Global observation/candidate/node schema, and (live=True
            # only) the frozen Local checkpoint's own generation/sha256/dims
            # must all still match. See _check_resume_compatibility's own
            # docstring for the full list this enforces.
            _check_resume_compatibility(manifest, profile, loop, live_executor)
            loop.load_state_dict(manifest["loop_state"])
            agent.training_steps = manifest["training_steps"]
            total_updates = manifest["total_optimizer_updates"]
            loop.replay = type(loop.replay).load(result["replay_path"], seed=profile.training.seed)
            rng_state_path = manifest["rng_state_path"]
            rng_state_sha256 = manifest["rng_state_sha256"]
            if not os.path.isabs(rng_state_path):
                rng_state_path = os.path.join(checkpoint_dir, rng_state_path)
            from hunter_kinodynamic_rl.rl.checkpointing.manager import sha256_of_file
            if not os.path.isfile(rng_state_path):
                raise ResumeIncompatibleError(
                    f"hierarchical_train_node resume: RNG state file {rng_state_path!r} is missing"
                )
            actual_sha256 = sha256_of_file(rng_state_path)
            if actual_sha256 != rng_state_sha256:
                raise ResumeIncompatibleError(
                    f"hierarchical_train_node resume: RNG state SHA mismatch "
                    f"(recorded={rng_state_sha256!r}, actual={actual_sha256!r})"
                )
            restore_rng_state(load_rng_state_file(rng_state_path), action_rng=rng)
            logger(f"restored full RNG state (python/numpy/action_rng/torch) from {rng_state_path}")
        logger(f"resumed from {checkpoint_dir}/{resume_checkpoint_tag} at global_step={loop.global_step}")

    # detailed spec: "Global checkpoint/replay manifest에 Local checkpoint
    # generation/hash, profile/config hash, observation/action schema, seed
    # scheduler 상태를 기록한다" -- absent (None) for live=False, where
    # there is no real Local checkpoint driving the mission at all.
    local_checkpoint_manifest_summary = None
    if live_executor is not None:
        m = live_executor.checkpoint_manifest
        local_checkpoint_manifest_summary = {
            "local_checkpoint_dir": profile.hierarchical_training.local_checkpoint_dir,
            "local_checkpoint_name": profile.hierarchical_training.local_checkpoint_name,
            "generation": m.get("generation"), "pt_sha256": m.get("pt_sha256"),
            "state_dim": m.get("state_dim"), "action_dim": m.get("action_dim"),
            "local_training_contract_fingerprint": m.get("local_training_contract_fingerprint"),
        }

    total_reached = 0
    try:
        for mission_index in range(num_missions):
            epsilon = float(np.interp(
                agent.training_steps, [0, profile.global_rl.epsilon_decay_steps],
                [profile.global_rl.epsilon_start, profile.global_rl.epsilon_end],
            ))
            outcome = loop.run_mission(agent, epsilon=epsilon, rng=rng)
            total_reached += int(outcome.mission_reached)

            if len(loop.replay) >= max(profile.global_rl.batch_size, profile.global_rl.warmup_options):
                # item 9: "per_mission" (schema default, byte-identical to
                # pre-existing behaviour) always runs exactly one update;
                # "per_transition" scales the update count with how many
                # Global option-transitions THIS mission actually stored,
                # so a mission that ran many options doesn't collapse to
                # the same single gradient step as one that ran barely any
                # -- see HierarchicalTrainingConfig.training_update_cadence's
                # own docstring.
                if profile.hierarchical_training.training_update_cadence == "per_transition":
                    n_updates = max(1, round(
                        outcome.global_transitions * profile.hierarchical_training.updates_per_option_transition
                    ))
                else:
                    n_updates = 1
                last_metrics = None
                for _ in range(n_updates):
                    batch = loop.replay.sample_torch(profile.global_rl.batch_size, device=agent.device)
                    last_metrics = agent.train_step(batch)
                    total_updates += 1
                logger(
                    f"mission={mission_index} epsilon={epsilon:.3f} outcome={outcome} updates={n_updates} "
                    f"loss={last_metrics['loss']:.4f} success_rate={total_reached / (mission_index + 1):.3f}"
                )

            if (mission_index + 1) % max(1, num_missions // 10) == 0 or mission_index == num_missions - 1:
                # item 4: generated UPFRONT (not left to save_generation's
                # own default) so the RNG-state sidecar file can be named
                # after it and its path/sha256 embedded into `meta` BEFORE
                # save_generation publishes -- meta ends up inside the
                # SAME atomic manifest.json this generation id also names.
                generation = uuid.uuid4().hex
                rng_state_relpath = os.path.join(".rng_states", f"rng_state_{generation}.pt")
                rng_state_path = os.path.join(checkpoint_dir, rng_state_relpath)
                os.makedirs(os.path.dirname(rng_state_path), exist_ok=True)
                rng_state_sha256 = save_rng_state_file(rng_state_path, capture_rng_state(action_rng=rng))

                meta = loop.checkpoint_meta()
                resolved_config = dataclasses.asdict(profile)
                meta.update({
                    "profile_name": profile.name, "resolved_config": resolved_config,
                    "resolved_config_hash": sha256_of_obj(resolved_config),
                    "training_steps": agent.training_steps, "loop_state": loop.state_dict(),
                    "hierarchical_architecture_fingerprint": hierarchical_architecture_fingerprint(profile),
                    "local_checkpoint_manifest_summary": local_checkpoint_manifest_summary,
                    "live": live,
                    "rng_state_path": rng_state_relpath, "rng_state_sha256": rng_state_sha256,
                    "rng_state_schema_version": 1,
                    "global_replay_schema_version": GLOBAL_REPLAY_SCHEMA_VERSION,
                    "training_update_cadence": profile.hierarchical_training.training_update_cadence,
                    "updates_per_option_transition": profile.hierarchical_training.updates_per_option_transition,
                    "total_optimizer_updates": total_updates,
                    "epsilon_schedule_basis": "agent_training_steps",
                })
                ckpt_manager.save_generation(
                    checkpoint_dir, "latest", agent.checkpoint_components(), meta, loop.replay,
                    generation=generation,
                )
                with open(os.path.join(run_dir, "hierarchical_metadata.json"), "w") as f:
                    json.dump(meta, f, indent=2, default=str)
    finally:
        if live_executor is not None:
            live_executor.close()

    return {
        "run_dir": run_dir, "missions_run": num_missions, "success_rate": total_reached / max(1, num_missions),
        "global_step": loop.global_step,
    }


def main(args=None):
    rclpy.init(args=args)
    node = Node("hierarchical_train_node")
    node.declare_parameter("profile", "hierarchical_phase4")
    node.declare_parameter("run_root", "runtime/hierarchical_experiments")
    node.declare_parameter("num_missions", 1000)
    node.declare_parameter("resume", False)
    node.declare_parameter("resume_run_dir", "")
    node.declare_parameter("resume_checkpoint_tag", "latest")
    node.declare_parameter("live", False)
    node.declare_parameter("require_promoted_local", True)
    node.declare_parameter("dry_run", False)
    profile_name = node.get_parameter("profile").value
    run_root = node.get_parameter("run_root").value
    num_missions = int(node.get_parameter("num_missions").value)
    resume_run_dir = node.get_parameter("resume_run_dir").value or None
    resume_checkpoint_tag = node.get_parameter("resume_checkpoint_tag").value
    resume = bool(node.get_parameter("resume").value) and bool(resume_run_dir)
    live = bool(node.get_parameter("live").value)
    require_promoted_local = bool(node.get_parameter("require_promoted_local").value)
    dry_run = bool(node.get_parameter("dry_run").value)
    try:
        if dry_run:
            from hunter_kinodynamic_rl.training.hierarchical_preflight import _format_report, run_global_preflight

            report = run_global_preflight(
                profile_name, live=live, resume=resume, require_promoted_local=require_promoted_local,
            )
            node.get_logger().info("\n" + _format_report(report))
            if not report.ok:
                raise SystemExit(1)
            return
        result = train_hierarchical_dqn(
            profile_name, run_root=run_root, num_missions=num_missions,
            resume_run_dir=resume_run_dir if resume else None, resume_checkpoint_tag=resume_checkpoint_tag,
            logger=node.get_logger().info, live=live, require_promoted_local=require_promoted_local,
        )
        node.get_logger().info(f"hierarchical training finished: {result}")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
