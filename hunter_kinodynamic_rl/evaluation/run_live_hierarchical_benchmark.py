#!/usr/bin/env python3
"""``ros2 run hunter_kinodynamic_rl run_live_hierarchical_benchmark.py --ros-args -p profile:=hierarchical_phase4 -p num_scenarios:=20 -p global_checkpoint_dir:=... -p global_checkpoint_name:=final -p output_dir:=runtime/hierarchical_benchmark``

Detailed-spec "성능 검증": runs the SAME fixed unseen benchmark manifest
(``evaluation.long_horizon_benchmark.build_fixed_benchmark_manifest``,
mode defaults to "test" -- never the train pool) against BOTH:

  A. local-only/no-memory baseline (``ablation="A"`` -- the Local TQC
     chases the FINAL goal directly every option, no Global RL)
  B. Phase 4 partial-map+visited Global DQN + frozen Local TQC
     (``ablation="B"``)

over LIVE Gazebo with the REAL frozen Local TQC checkpoint
(``profile.hierarchical_training.local_checkpoint_dir``/
``local_checkpoint_name``) -- via
``navigation.local_rl.live_gazebo_executor.LiveGazeboLocalExecutor``, never
``SimplifiedKinematicLocalExecutor``.

Requires (in order, in separate terminals):

  1. ``ros2 launch hunter_se_gazebo simulate_hunter_se_ignition.launch.py rviz:=false``
  2. this script.

Writes ``<output_dir>/<timestamp>/{episodes_A.csv,episodes_B.csv,
metrics.json,manifest.json}`` -- ``metrics.json`` holds
``evaluation.global_metrics.aggregate``'s output for both ablations plus
the raw per-scenario episode dicts, so a later run can be compared without
re-executing Gazebo. ``global_checkpoint_dir`` empty (default) evaluates B
against the built-in goal-seeking heuristic (matches
``nodes/hierarchical_environment_node.py``'s own fallback) -- pass a real
Global checkpoint for the actual Phase 4 completion comparison."""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import os
from datetime import datetime
from typing import List

# Defect-fix item 10: the strict Global-checkpoint validator now lives in
# evaluation.global_checkpoint_validation, SHARED with
# run_live_ablation_suite.py (which previously ran a materially weaker
# architecture-fingerprint-only check) -- re-exported here under the old
# name so existing imports/tests keep working.
from hunter_kinodynamic_rl.evaluation.global_checkpoint_validation import (
    FORMAL_MIN_SCENARIOS, require_formal_budget_unmodified, require_formal_mode_is_test,
    require_promoted_local_checkpoint,
    validate_global_checkpoint_manifest as _validate_global_checkpoint_manifest,
)


def _min_turning_radius_m(robot) -> float:
    return 1.0 / robot.max_curvature


def run_live_hierarchical_benchmark(
    profile_name: str, num_scenarios: int = 20, seed: int = 20000, mode: str = "test",
    global_checkpoint_dir: str = "", global_checkpoint_name: str = "final",
    output_dir: str = "runtime/hierarchical_benchmark", logger=print,
    max_options: int = 0, max_local_steps: int = 0,
    package_source_root: str = "", formal: bool = False,
) -> dict:
    """``formal`` (requirement D, default False -- preserves this script's
    original "wiring smoke" behavior byte-for-byte): when True, this is a
    FORMAL A/B benchmark, enforced structurally rather than just labeled:

    - ``global_checkpoint_dir`` MUST be non-empty -- refuses to silently
      fall back to the built-in heuristic Global agent for ablation B (a
      formal B must be a REAL trained Global DQN, never a heuristic
      standing in for one).
    - ``num_scenarios`` MUST be ``>= FORMAL_MIN_SCENARIOS`` (20) -- raises
      before touching Gazebo otherwise.
    - the written artifact's ``benchmark_kind`` is always the literal
      string ``"formal"`` (never ``"trained_global_smoke"``/``"wiring_smoke"``,
      the two labels a non-formal run can produce) -- a caller (e.g.
      :mod:`evaluation.local_promotion`-style consumers) can gate on this
      exact string to refuse promoting/interpreting a smoke artifact as a
      formal result.

    Requires ``rclpy.init()`` to have already happened in the caller
    (mirrors ``nodes/hierarchical_train_node.py::train_hierarchical_dqn``'s
    identical contract) -- ``main()`` below does this and owns
    ``rclpy.shutdown()`` too; this function never calls either itself, so
    it composes cleanly with a caller that already has an rclpy context
    (e.g. a test, or a larger script running several live steps in one
    process).

    ``max_options``/``max_local_steps`` (0, default: use the profile's own
    ``hierarchical_training.max_global_options_per_mission``/
    ``max_local_steps_per_option``) let a caller bound per-mission
    wall-clock cost for a live-Gazebo run explicitly, without editing the
    profile -- live-Gazebo real-time execution makes each option cost real
    wall-clock seconds (confirmed live: tens of seconds per option), so
    the profile's training-time budgets can make a benchmark impractically
    slow; this does not change what a FULL live run would use, only what
    THIS invocation caps it at."""
    from hunter_kinodynamic_rl.config.loader import load_profile
    from hunter_kinodynamic_rl.evaluation.global_metrics import aggregate
    from hunter_kinodynamic_rl.evaluation.provenance import collect_package_provenance
    from hunter_kinodynamic_rl.evaluation.long_horizon_benchmark import (
        build_fixed_benchmark_manifest, run_ablation_benchmark,
    )
    from hunter_kinodynamic_rl.navigation.local_rl.live_gazebo_executor import LiveGazeboLocalExecutor
    from hunter_kinodynamic_rl.rl.checkpointing import manager as ckpt_manager

    provenance = collect_package_provenance(package_source_root, execution_file=__file__)
    if provenance.get("execution_matches_source_module") is False:
        raise RuntimeError(
            "installed run_live_hierarchical_benchmark.py does not match the package source; rebuild before "
            "recording benchmark provenance"
        )

    profile = load_profile(profile_name)
    if not profile.global_rl.enabled or not profile.hierarchical_training.enabled:
        raise SystemExit(
            f"profile {profile_name!r} must enable global_rl AND hierarchical_training -- use a Phase 4 "
            "profile (e.g. hierarchical_phase4)"
        )
    if formal:
        if not global_checkpoint_dir:
            raise RuntimeError(
                "run_live_hierarchical_benchmark: formal=True requires a real global_checkpoint_dir -- a "
                "formal A/B benchmark must compare against a TRAINED Global DQN, never the built-in heuristic "
                "fallback (requirement D: heuristic Global 대신 넣을 수 없다)"
            )
        if num_scenarios < FORMAL_MIN_SCENARIOS:
            raise RuntimeError(
                f"run_live_hierarchical_benchmark: formal=True requires num_scenarios >= "
                f"{FORMAL_MIN_SCENARIOS}, got {num_scenarios}"
            )
        require_formal_mode_is_test(formal, mode)
        require_formal_budget_unmodified(formal, max_options, max_local_steps)
        require_promoted_local_checkpoint(
            profile.hierarchical_training.local_checkpoint_dir, profile.hierarchical_training.local_checkpoint_name,
            context="run_live_hierarchical_benchmark",
        )
    min_turning_radius_m = _min_turning_radius_m(profile.robot)

    scenarios = build_fixed_benchmark_manifest(
        profile.long_horizon_world, profile.robot.collision_radius_m, min_turning_radius_m,
        profile.robot.wheelbase_m, num_scenarios=num_scenarios, seed=seed, mode=mode,
    )
    logger(f"built {len(scenarios)} fixed {mode!r}-pool scenarios (seed={seed})")

    live_executor = LiveGazeboLocalExecutor(
        profile, profile.long_horizon_world,
        profile.hierarchical_training.local_checkpoint_dir, profile.hierarchical_training.local_checkpoint_name,
    )
    try:
        mission_kwargs = dict(
            global_cfg=profile.global_rl, hierarchy_cfg=profile.hierarchy, mapping_cfg=profile.mapping,
            memory_cfg=profile.memory, feasibility_cfg=profile.feasibility,
            long_horizon_cfg=profile.long_horizon_world, robot_radius_m=profile.robot.collision_radius_m,
            min_turning_radius_m=min_turning_radius_m, wheelbase_m=profile.robot.wheelbase_m,
            max_options=max_options or profile.hierarchical_training.max_global_options_per_mission,
            max_local_steps=max_local_steps or profile.hierarchical_training.max_local_steps_per_option,
            mission_timeout_steps=profile.hierarchy.mission_timeout_steps,
            local_executor_factory=live_executor.bind_mission,
        )

        logger("running ablation A (local-only/no-memory baseline)...")
        episodes_a = run_ablation_benchmark("A", scenarios, agent=None, seed=seed, **mission_kwargs)

        # Bug fix (found via live verification): ablation B always requires
        # a real agent object (run_ablation_mission raises "requires a
        # Global agent" otherwise) -- this module's own docstring/the log
        # message below have always documented "no checkpoint -> fall back
        # to the SAME built-in goal-seeking heuristic
        # nodes/hierarchical_environment_node.py uses" as the intended
        # behaviour, but the code previously left `agent=None` in that
        # case, which crashed instead. Reused (never reimplemented) here --
        # see _HeuristicGlobalAgent's own docstring.
        if global_checkpoint_dir:
            import torch  # noqa: F401  -- local import, only needed on this path

            from hunter_kinodynamic_rl.navigation.global_rl.agent import GlobalDQNAgent
            from hunter_kinodynamic_rl.navigation.global_rl.observation import resolve_map_channel_names

            map_channels = len(resolve_map_channel_names(profile.global_rl))
            max_nodes = (
                profile.memory.max_nodes_in_observation if profile.global_rl.topology_feedback_enabled else 0
            )
            agent = GlobalDQNAgent(profile.global_rl, map_channels, device="cpu", max_nodes=max_nodes)
            result = ckpt_manager.load_generation(
                global_checkpoint_dir, global_checkpoint_name, agent.checkpoint_components(),
                map_location=str(agent.device),
            )
            global_checkpoint_manifest = result["manifest"]
            _validate_global_checkpoint_manifest(
                global_checkpoint_manifest, profile, live_executor.checkpoint_manifest,
            )
            global_checkpoint_generation = result["generation"]
            logger(f"loaded Global checkpoint {global_checkpoint_dir}/{global_checkpoint_name}: "
                   f"loaded={result.get('loaded')} skipped={result.get('skipped')} "
                   f"generation={global_checkpoint_generation} (fingerprint-verified against {profile_name!r})")
        else:
            global_checkpoint_manifest = None
            global_checkpoint_generation = None
            from hunter_kinodynamic_rl.navigation.global_rl.subgoal_sampler import build_candidate_set
            from hunter_kinodynamic_rl.nodes.hierarchical_environment_node import _heuristic_select_action

            class _HeuristicGlobalAgent:
                """Duck-typed ``agent.select_action(obs, epsilon, rng) -> int``
                wrapper around ``hierarchical_environment_node.py``'s own
                ``_heuristic_select_action`` (goal-bearing-aligned candidate
                choice) -- reused, never reimplemented, so this benchmark's
                "no Global checkpoint" fallback is IDENTICAL to the live
                deployment node's own fallback, exactly as this module's
                docstring documents."""

                def __init__(self, candidates):
                    self._candidates = candidates

                def select_action(self, obs, *, epsilon, rng):
                    return _heuristic_select_action(self._candidates, obs)

            agent = _HeuristicGlobalAgent(build_candidate_set(profile.global_rl))
            logger("no global_checkpoint_dir given -- ablation B evaluates the built-in goal-seeking "
                   "heuristic (hierarchical_environment_node._heuristic_select_action), NOT a trained "
                   "Global policy (fine for a wiring smoke test, not for the Phase 4 completion comparison)")

        logger("running ablation B (Phase 4 Global DQN + frozen Local TQC)...")
        episodes_b = run_ablation_benchmark("B", scenarios, agent=agent, seed=seed, **mission_kwargs)
    finally:
        live_executor.close()

    # Defect-fix item 10: "모든 scenario 결과가 존재해야 formal artifact
    # 작성 -- 중단/부분 결과는 formal이 아니라 incomplete 상태로 보존". A
    # crash mid-run_ablation_benchmark already prevents reaching this point
    # at all (no artifact is written); this is the defensive check for the
    # case where either call somehow returned FEWER episodes than
    # scenarios without raising.
    incomplete = len(episodes_a) != len(scenarios) or len(episodes_b) != len(scenarios)
    if formal and incomplete:
        logger(
            f"run_live_hierarchical_benchmark: incomplete results (A={len(episodes_a)}, B={len(episodes_b)}, "
            f"expected {len(scenarios)} each) -- writing benchmark_kind='incomplete', never 'formal'"
        )

    metrics_a = aggregate(episodes_a)
    metrics_b = aggregate(episodes_b)
    logger(f"A (baseline) metrics: {metrics_a}")
    logger(f"B (hierarchical)  metrics: {metrics_b}")

    run_dir = os.path.join(output_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    _write_episodes_csv(os.path.join(run_dir, "episodes_A.csv"), episodes_a)
    _write_episodes_csv(os.path.join(run_dir, "episodes_B.csv"), episodes_b)
    # item 5: hashed from the EXACT bytes written to manifest.json below --
    # a later reader can independently re-hash manifest.json and confirm it
    # is the very scenario set this metrics.json describes, without trusting
    # any other identifier.
    scenario_manifest_json = json.dumps(
        [dataclasses.asdict(s) for s in scenarios], indent=2, default=str, sort_keys=True,
    )
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        f.write(scenario_manifest_json)
    scenario_manifest_hash = hashlib.sha256(scenario_manifest_json.encode("utf-8")).hexdigest()

    from hunter_kinodynamic_rl.evaluation.fingerprint import hierarchical_architecture_fingerprint, sha256_of_obj

    local_manifest = live_executor.checkpoint_manifest
    result_summary = {
        "benchmark_schema_version": 2,
        "benchmark_kind": (
            "incomplete" if (formal and incomplete) else
            "formal" if formal else
            ("trained_global_smoke" if global_checkpoint_manifest is not None else "wiring_smoke")
        ),
        "trained_global_policy": global_checkpoint_manifest is not None,
        "ablation_B_label": "B_trained_global" if global_checkpoint_manifest is not None else "B_heuristic_smoke",
        "profile_name": profile_name, "num_scenarios": num_scenarios, "seed": seed, "mode": mode,
        # Mutable (informational only -- "latest"/"final" resolve
        # differently over time, see item 5's "이전 latest symlink가 바뀌면
        # 재구성할 수 없다" finding). Everything a later reader actually
        # needs to reconstruct/verify this EXACT run is in the immutable
        # fields below instead.
        "global_checkpoint_dir": global_checkpoint_dir, "global_checkpoint_name": global_checkpoint_name,
        # item 5: IMMUTABLE identity -- resolved once at load time (see
        # ckpt_manager.load_generation's own pin-by-realpath contract) and
        # never re-derived from the (possibly since-repointed) tag symlink.
        "global_checkpoint_generation": global_checkpoint_generation,
        "global_checkpoint_sha256": (global_checkpoint_manifest or {}).get("pt_sha256"),
        "global_checkpoint_training_steps": (global_checkpoint_manifest or {}).get("training_steps"),
        "global_checkpoint_global_step": (
            (global_checkpoint_manifest or {}).get("loop_state", {}) or {}
        ).get("global_step"),
        "local_checkpoint_generation": local_manifest.get("generation"),
        "local_checkpoint_sha256": local_manifest.get("pt_sha256"),
        "local_checkpoint_training_steps": local_manifest.get("training_steps"),
        "local_checkpoint_state_dim": local_manifest.get("state_dim"),
        "local_checkpoint_action_dim": local_manifest.get("action_dim"),
        "hierarchical_architecture_fingerprint": hierarchical_architecture_fingerprint(profile),
        "resolved_profile_hash": sha256_of_obj(dataclasses.asdict(profile)),
        "scenario_manifest_hash": scenario_manifest_hash,
        "max_options_used": mission_kwargs["max_options"], "max_local_steps_used": mission_kwargs["max_local_steps"],
        "mission_timeout_steps": mission_kwargs["mission_timeout_steps"],
        "local_executor_dt_sec": getattr(live_executor, "dt_sec", None),
        "provenance": provenance,
        "metrics_A": metrics_a, "metrics_B": metrics_b,
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(result_summary, f, indent=2, default=str)
    logger(f"results written to {run_dir}")
    return result_summary


def _write_episodes_csv(path: str, episodes: List[dict]) -> None:
    if not episodes:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(episodes[0].keys()))
        writer.writeheader()
        writer.writerows(episodes)


def main(args=None):
    import rclpy
    from rclpy.node import Node

    rclpy.init(args=args)
    node = Node("run_live_hierarchical_benchmark")
    node.declare_parameter("profile", "hierarchical_phase4")
    node.declare_parameter("num_scenarios", 20)
    node.declare_parameter("seed", 20000)
    node.declare_parameter("mode", "test")
    node.declare_parameter("global_checkpoint_dir", "")
    node.declare_parameter("global_checkpoint_name", "final")
    node.declare_parameter("output_dir", "runtime/hierarchical_benchmark")
    node.declare_parameter("max_options", 0)
    node.declare_parameter("max_local_steps", 0)
    node.declare_parameter("package_source_root", "")
    node.declare_parameter("formal", False)
    profile_name = node.get_parameter("profile").value
    num_scenarios = int(node.get_parameter("num_scenarios").value)
    seed = int(node.get_parameter("seed").value)
    mode = node.get_parameter("mode").value
    global_checkpoint_dir = node.get_parameter("global_checkpoint_dir").value
    global_checkpoint_name = node.get_parameter("global_checkpoint_name").value
    output_dir = node.get_parameter("output_dir").value
    max_options = int(node.get_parameter("max_options").value)
    max_local_steps = int(node.get_parameter("max_local_steps").value)
    package_source_root = node.get_parameter("package_source_root").value
    formal = bool(node.get_parameter("formal").value)
    try:
        run_live_hierarchical_benchmark(
            profile_name, num_scenarios=num_scenarios, seed=seed, mode=mode,
            global_checkpoint_dir=global_checkpoint_dir, global_checkpoint_name=global_checkpoint_name,
            output_dir=output_dir, logger=node.get_logger().info,
            max_options=max_options, max_local_steps=max_local_steps,
            package_source_root=package_source_root, formal=formal,
        )
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
