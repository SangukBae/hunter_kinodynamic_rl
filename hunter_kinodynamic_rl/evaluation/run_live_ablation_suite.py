#!/usr/bin/env python3
"""``ros2 run hunter_kinodynamic_rl run_live_ablation_suite.py --ros-args -p num_scenarios:=20 -p labels:=A,B,C,D,E,F,G -p global_checkpoint_root:=... -p formal:=true``

Requirement G: live A-G ablation suite runner over live Gazebo + the real
frozen Local TQC -- composes :func:`evaluation.ablation_suite.run_ablation_suite`
with a SINGLE shared ``LiveGazeboLocalExecutor`` (every ``hierarchical_phase5_*``
profile freezes the SAME Local checkpoint/training profile, only their
``global_rl``/``memory``/``feasibility`` sections differ) and, for the
feasibility-tier labels (E/F/G), a
:class:`~hunter_kinodynamic_rl.navigation.hierarchy.local_feasibility_evaluator.FrozenLocalFeasibilityEvaluator`
built from that same executor.

Each requested label's own Global checkpoint is expected at
``<global_checkpoint_root>/<label>/<global_checkpoint_name>`` -- a label
whose checkpoint directory does not exist is SKIPPED (never silently run
against a heuristic or a different label's checkpoint, matching
``run_ablation_suite``'s own contract).

``formal=True`` enforces ``num_scenarios >= 20`` and writes
``benchmark_kind="formal"`` into the artifact -- otherwise ``"smoke"``.
NOT executed in this session (would require live Gazebo + trained
per-ablation checkpoints)."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Sequence

from hunter_kinodynamic_rl.evaluation.global_checkpoint_validation import (
    FORMAL_MIN_SCENARIOS, require_formal_budget_unmodified, require_formal_mode_is_test,
    require_promoted_local_checkpoint, validate_global_checkpoint_manifest,
)

_PROFILE_BY_LABEL = {
    "A": "hierarchical_phase5_a", "B": "hierarchical_phase5_b", "C": "hierarchical_phase5_c",
    "D": "hierarchical_phase5_d", "E": "hierarchical_phase5_e", "F": "hierarchical_phase5_f",
    "G": "hierarchical_phase5_g",
}


def run_live_ablation_suite(
    labels: Sequence[str] = ("A", "B", "C", "D", "E", "F", "G"), num_scenarios: int = 20, seed: int = 20000,
    mode: str = "test", global_checkpoint_root: str = "", global_checkpoint_name: str = "final",
    output_dir: str = "runtime/ablation_suite", logger=print, max_options: int = 0, max_local_steps: int = 0,
    formal: bool = False,
) -> dict:
    from hunter_kinodynamic_rl.config.loader import load_profile
    from hunter_kinodynamic_rl.evaluation.ablation_suite import evaluate_acceptance_report, run_ablation_suite
    from hunter_kinodynamic_rl.evaluation.long_horizon_benchmark import build_fixed_benchmark_manifest
    from hunter_kinodynamic_rl.navigation.local_rl.live_gazebo_executor import LiveGazeboLocalExecutor
    from hunter_kinodynamic_rl.rl.checkpointing import manager as ckpt_manager

    if formal and num_scenarios < FORMAL_MIN_SCENARIOS:
        raise RuntimeError(
            f"run_live_ablation_suite: formal=True requires num_scenarios >= {FORMAL_MIN_SCENARIOS}, "
            f"got {num_scenarios}"
        )
    require_formal_mode_is_test(formal, mode)
    require_formal_budget_unmodified(formal, max_options, max_local_steps)

    profiles = {label: load_profile(_PROFILE_BY_LABEL[label]) for label in labels}
    base_profile = next(iter(profiles.values()))
    min_turning_radius_m = 1.0 / base_profile.robot.max_curvature
    if formal:
        # Every hierarchical_phase5_* profile freezes the SAME Local
        # checkpoint/training profile (module docstring) -- one check
        # covers every label.
        require_promoted_local_checkpoint(
            base_profile.hierarchical_training.local_checkpoint_dir,
            base_profile.hierarchical_training.local_checkpoint_name, context="run_live_ablation_suite",
        )

    scenarios = build_fixed_benchmark_manifest(
        base_profile.long_horizon_world, base_profile.robot.collision_radius_m, min_turning_radius_m,
        base_profile.robot.wheelbase_m, num_scenarios=num_scenarios, seed=seed, mode=mode,
    )
    logger(f"built {len(scenarios)} fixed {mode!r}-pool scenarios shared across labels {list(labels)}")

    live_executor = LiveGazeboLocalExecutor(
        base_profile, base_profile.long_horizon_world,
        base_profile.hierarchical_training.local_checkpoint_dir,
        base_profile.hierarchical_training.local_checkpoint_name,
    )
    agents: Dict[str, object] = {}
    checkpoint_identity: Dict[str, dict] = {}
    local_evaluators: Dict[str, object] = {}
    evaluator_telemetry: Dict[str, object] = {}
    try:
        from hunter_kinodynamic_rl.navigation.global_rl.agent import GlobalDQNAgent
        from hunter_kinodynamic_rl.navigation.global_rl.observation import resolve_map_channel_names
        from hunter_kinodynamic_rl.navigation.hierarchy.local_feasibility_evaluator import (
            FeasibilityEvaluatorTelemetry,
        )

        for label in labels:
            profile = profiles[label]
            ckpt_dir = os.path.join(global_checkpoint_root, label) if global_checkpoint_root else ""
            if not ckpt_dir or not os.path.isdir(ckpt_dir):
                msg = f"ablation {label}: no checkpoint directory at {ckpt_dir!r}"
                if formal:
                    raise RuntimeError(
                        f"run_live_ablation_suite: formal=True requires every requested label's checkpoint to "
                        f"exist -- {msg}"
                    )
                logger(f"{msg} -- will be SKIPPED")
                continue
            map_channels = len(resolve_map_channel_names(profile.global_rl))
            max_nodes = profile.memory.max_nodes_in_observation if profile.global_rl.topology_feedback_enabled else 0
            agent = GlobalDQNAgent(profile.global_rl, map_channels, device="cpu", max_nodes=max_nodes)
            result = ckpt_manager.load_generation(
                ckpt_dir, global_checkpoint_name, agent.checkpoint_components(), map_location="cpu",
            )
            manifest = result["manifest"]
            # Defect-fix item 10: the SAME strict validator run_live_hierarchical_benchmark.py
            # uses (architecture/resolved-config/replay-schema/RNG-basis/Local
            # identity/total_optimizer_updates) -- this used to be a
            # materially weaker architecture-fingerprint-only check, letting
            # an A-G ablation run against a checkpoint the A/B gate would
            # have refused.
            try:
                validate_global_checkpoint_manifest(manifest, profile, live_executor.checkpoint_manifest)
            except RuntimeError as e:
                if formal:
                    raise RuntimeError(
                        f"run_live_ablation_suite: formal=True requires label {label!r}'s checkpoint to pass "
                        f"strict validation: {e}"
                    ) from e
                logger(f"ablation {label}: {e} -- SKIPPED, never run against an incompatible checkpoint")
                continue
            agents[label] = agent
            checkpoint_identity[label] = {
                "generation": result["generation"], "pt_sha256": manifest.get("pt_sha256"),
            }
            if label in ("E", "F", "G"):
                telemetry = FeasibilityEvaluatorTelemetry()
                evaluator_telemetry[label] = telemetry
                local_evaluators[label] = live_executor.build_local_feasibility_evaluator(
                    inference_timeout_sec=profile.feasibility.local_evaluator_inference_timeout_sec,
                    max_snapshot_age_sec=profile.feasibility.local_evaluator_max_snapshot_age_sec,
                    telemetry=telemetry,
                )

        # run_ablation_suite needs ONE shared config set for hierarchy/mapping/
        # long_horizon_world/robot (identical across every hierarchical_phase5_*
        # profile) plus each label's OWN global_rl/memory/feasibility (already
        # captured per-profile above and re-selected inside the loop via
        # effective_*_config in run_ablation_mission, keyed off the ablation
        # label itself -- so passing the base_profile's sections here is safe:
        # run_ablation_mission always overrides them to match `ablation`).
        result = run_ablation_suite(
            scenarios, labels=labels, agents=agents, local_evaluators=local_evaluators, seed=seed,
            benchmark_kind=("formal" if formal else "smoke"), min_valid_samples=FORMAL_MIN_SCENARIOS,
            global_cfg=base_profile.global_rl, hierarchy_cfg=base_profile.hierarchy,
            mapping_cfg=base_profile.mapping, memory_cfg=base_profile.memory,
            feasibility_cfg=base_profile.feasibility, long_horizon_cfg=base_profile.long_horizon_world,
            robot_radius_m=base_profile.robot.collision_radius_m, min_turning_radius_m=min_turning_radius_m,
            wheelbase_m=base_profile.robot.wheelbase_m,
            max_options=max_options or base_profile.hierarchical_training.max_global_options_per_mission,
            max_local_steps=max_local_steps or base_profile.hierarchical_training.max_local_steps_per_option,
            mission_timeout_steps=base_profile.hierarchy.mission_timeout_steps,
            local_executor_factory=live_executor.bind_mission,
        )
    finally:
        live_executor.close()

    # Defect-fix item 10: defensive re-affirmation -- a formal artifact can
    # never carry a nonempty labels_skipped (the per-label loop above and
    # ablation_suite.run_ablation_suite's own formal-mode check should
    # already have raised before this point; this catches any future
    # regression that reintroduces a silent skip path).
    if formal and result["labels_skipped"]:
        raise RuntimeError(
            f"run_live_ablation_suite: formal=True but labels_skipped={result['labels_skipped']!r} is "
            "non-empty -- refusing to write a formal artifact with missing labels"
        )

    run_dir = os.path.join(output_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    # item 3/10 pattern: hash the EXACT bytes written to manifest.json so a
    # later reader can independently re-verify this is the very scenario
    # set every label above ran against.
    scenario_manifest_json = json.dumps(
        [dataclasses.asdict(s) for s in scenarios], indent=2, default=str, sort_keys=True,
    )
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        f.write(scenario_manifest_json)
    scenario_manifest_hash = hashlib.sha256(scenario_manifest_json.encode("utf-8")).hexdigest()

    acceptance = result["acceptance_report"]
    out = {
        "benchmark_kind": result["benchmark_kind"], "num_scenarios": result["num_scenarios"],
        "seed": seed, "mode": mode, "scenario_manifest_hash": scenario_manifest_hash,
        "labels_run": result["labels_run"], "labels_skipped": result["labels_skipped"],
        "checkpoint_identity": checkpoint_identity, "summaries": result["summaries"],
        "episodes": result["episodes"],
        "evaluator_telemetry": {label: t.as_dict() for label, t in evaluator_telemetry.items()},
        "provenance": result.get("provenance"),
        "acceptance_report": {
            "schema_version": acceptance.schema_version, "benchmark_kind": acceptance.benchmark_kind,
            "labels_present": acceptance.labels_present,
            "comparisons": [dataclasses.asdict(c) for c in acceptance.comparisons],
        },
    }
    with open(os.path.join(run_dir, "ablation_suite_result.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    logger(f"results written to {run_dir}")
    return out


def main(args=None):
    import rclpy
    from rclpy.node import Node

    rclpy.init(args=args)
    node = Node("run_live_ablation_suite")
    node.declare_parameter("labels", "A,B,C,D,E,F,G")
    node.declare_parameter("num_scenarios", 20)
    node.declare_parameter("seed", 20000)
    node.declare_parameter("mode", "test")
    node.declare_parameter("global_checkpoint_root", "")
    node.declare_parameter("global_checkpoint_name", "final")
    node.declare_parameter("output_dir", "runtime/ablation_suite")
    node.declare_parameter("max_options", 0)
    node.declare_parameter("max_local_steps", 0)
    node.declare_parameter("formal", False)
    labels = tuple(s.strip() for s in node.get_parameter("labels").value.split(",") if s.strip())
    try:
        run_live_ablation_suite(
            labels=labels, num_scenarios=int(node.get_parameter("num_scenarios").value),
            seed=int(node.get_parameter("seed").value), mode=node.get_parameter("mode").value,
            global_checkpoint_root=node.get_parameter("global_checkpoint_root").value,
            global_checkpoint_name=node.get_parameter("global_checkpoint_name").value,
            output_dir=node.get_parameter("output_dir").value, logger=node.get_logger().info,
            max_options=int(node.get_parameter("max_options").value),
            max_local_steps=int(node.get_parameter("max_local_steps").value),
            formal=bool(node.get_parameter("formal").value),
        )
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
