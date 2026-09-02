"""Fixed long-horizon benchmark + high/low-level ablation runner (plan
section 9.9/10.3/10.10) -- ROS-free by default, mirrors
``training.train_hierarchical_dqn.HierarchicalTrainingLoop``'s own
ROS-free-first convention so this benchmark is runnable in CI/Docker without
a live simulator (using
:class:`~hunter_kinodynamic_rl.training.train_hierarchical_dqn.SimplifiedKinematicLocalExecutor`
as the local-motion stand-in), while ``local_executor_factory``/
``local_evaluator`` let a live caller substitute
:class:`~hunter_kinodynamic_rl.navigation.local_rl.live_gazebo_executor.LiveGazeboLocalExecutor`
and a real
:class:`~hunter_kinodynamic_rl.navigation.hierarchy.local_feasibility_evaluator.FrozenLocalFeasibilityEvaluator`.

**Ablations (plan section 9.9)**: the 7 high-level ablations are

    A. Local only / no memory       -- no Global RL at all; the FINAL goal
                                        is repeatedly (re-)enqueued as the
                                        single subgoal every option
    B. Global partial-map RL        -- global_rl.enabled, include_visited_channel=False,
                                        every other Phase 5 ablation flag False
    C. + Visited map                -- B + include_visited_channel=True (requirement F:
                                        genuinely distinct map_tensor channel count/
                                        hierarchical_architecture_fingerprint from B --
                                        see config.schema.GlobalRLConfig.include_visited_channel)
    D. + Topological memory         -- + memory.enabled/topology_feedback_enabled
    E. + Local feasibility feedback -- + feasibility.enabled/feasibility_feedback_enabled
                                        (requires a real ``local_evaluator``, see
                                        :func:`run_ablation_mission`'s own docstring)
    F. + Global risk feedback       -- + global_risk_feedback_enabled (also requires
                                        ``local_evaluator``, reuses the same rollout)
    G. Full hierarchical system     -- D + E + F together
"""

from __future__ import annotations

import dataclasses
import hashlib
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from hunter_kinodynamic_rl.config.schema import (
    GlobalFeasibilityConfig, GlobalRLConfig, HierarchyConfig, LongHorizonWorldConfig, MappingConfig, MemoryConfig,
)
from hunter_kinodynamic_rl.env.scenarios.long_horizon_generator import (
    LongHorizonSeedScheduler, generate_long_horizon_world,
)
from hunter_kinodynamic_rl.env.scenarios.long_horizon_world import LongHorizonWorld
from hunter_kinodynamic_rl.navigation.global_rl.action_mask import compute_action_mask
from hunter_kinodynamic_rl.navigation.global_rl.subgoal_sampler import build_candidate_set, candidate_endpoint_mission
from hunter_kinodynamic_rl.navigation.global_rl.observation import build_global_observation
from hunter_kinodynamic_rl.navigation.hierarchy.coordinator import HierarchyCoordinator
from hunter_kinodynamic_rl.navigation.hierarchy.feasibility import compute_feasibility_features
from hunter_kinodynamic_rl.navigation.hierarchy.subgoal_manager import FAILURE_SUBGOAL_STATUSES, SubgoalStatus
from hunter_kinodynamic_rl.navigation.mapping.partial_map import PartialMap
from hunter_kinodynamic_rl.navigation.memory.dead_end_detector import DeadEndDetector
from hunter_kinodynamic_rl.navigation.memory.node_manager import TopologicalNodeManager, compute_candidate_topology_features
from hunter_kinodynamic_rl.navigation.memory.route_history import RouteEventType, RouteHistory
from hunter_kinodynamic_rl.navigation.memory.topological_graph import TopologicalGraph
from hunter_kinodynamic_rl.navigation.mission.mission_frame import MissionFrame, PoseXYYaw
from hunter_kinodynamic_rl.training.train_hierarchical_dqn import (
    SimplifiedKinematicLocalExecutor, feasibility_config_from, hierarchy_config_from, memory_config_from,
)

ABLATION_LABELS: Tuple[str, ...] = ("A", "B", "C", "D", "E", "F", "G")


@dataclass(frozen=True)
class BenchmarkScenarioSpec:
    """Immutable fixed-benchmark scenario metadata (plan 10.3). ``seed``
    ALONE regenerates the exact same :class:`LongHorizonWorld` (Phase 3's
    own determinism guarantee) -- every other field here is DERIVED, frozen
    metadata for reporting/split bookkeeping, never re-derived differently
    at eval time (a mismatch would mean the generator or config changed
    since this manifest was built, exactly what ``content_hash``/
    ``occupancy_hash`` exist to catch)."""

    scenario_id: str
    seed: int
    mode: str
    difficulty_class: str
    content_hash: str
    occupancy_hash: str
    start_pose: Tuple[float, float, float]
    relative_goal: Tuple[float, float]
    shortest_feasible_path_length_m: Optional[float]
    dead_end_count: int
    loop_count: int


def _occupancy_hash(world: LongHorizonWorld) -> str:
    return hashlib.sha256(np.ascontiguousarray(world.occupancy).tobytes()).hexdigest()


def _content_hash(world_seed: int, world: LongHorizonWorld) -> str:
    """SAME formula ``build_fixed_benchmark_manifest`` uses to compute the
    ``content_hash`` it freezes into each :class:`BenchmarkScenarioSpec`
    -- factored out so :func:`_verify_scenario_regeneration` can recompute
    it identically from a REGENERATED world."""
    content = f"{world_seed}|{world.start_pose}|{world.goal_pose}|{world.shortest_path_length_m}".encode()
    return hashlib.sha256(content).hexdigest()


def _verify_scenario_regeneration(scenario: "BenchmarkScenarioSpec", world: LongHorizonWorld) -> None:
    """Fail-fast reproducibility check (plan 10.3's manifest hash
    requirement, made REAL rather than just documented) -- every
    :func:`run_ablation_mission` call regenerates its world from
    ``scenario.seed`` alone, so a ``long_horizon_generator``/config change
    since the manifest was built could otherwise silently evaluate a
    DIFFERENT world under the same ``scenario_id``/seed, with the
    manifest's own frozen metadata (GT shortest path, difficulty class,
    dead-end/loop counts) then describing a world that isn't actually
    being run. Recomputes both hashes from the just-regenerated world and
    raises immediately on ANY mismatch -- never silently proceeds with a
    world that doesn't match its own manifest entry."""
    occupancy_hash = _occupancy_hash(world)
    content_hash = _content_hash(scenario.seed, world)
    if occupancy_hash == scenario.occupancy_hash and content_hash == scenario.content_hash:
        return
    raise ValueError(
        f"run_ablation_mission: regenerated world for scenario {scenario.scenario_id!r} (seed={scenario.seed}) "
        "does not match its manifest hash -- the long_horizon_world generator or its config changed since "
        "this manifest was built. occupancy_hash "
        f"{'matches' if occupancy_hash == scenario.occupancy_hash else f'MISMATCH (expected {scenario.occupancy_hash}, got {occupancy_hash})'}, "
        f"content_hash {'matches' if content_hash == scenario.content_hash else f'MISMATCH (expected {scenario.content_hash}, got {content_hash})'}"
        " -- regenerate the benchmark manifest (build_fixed_benchmark_manifest) before evaluating against it; "
        "never evaluate silently against a different world than the one this scenario_id was built from."
    )


def _difficulty_class(world: LongHorizonWorld, mode: str) -> str:
    meta = world.topology_metadata
    dead_end_count = int(meta.get("dead_end_count", 0))
    loop_count = int(meta.get("loop_count", 0))
    if mode != "train":
        base = "ood_geometry"
    else:
        base = "id_geometry"
    if dead_end_count >= 4:
        return f"{base}_long_dead_end"
    if loop_count >= 3:
        return f"{base}_multiple_loops"
    return base


def build_fixed_benchmark_manifest(
    cfg: LongHorizonWorldConfig, robot_radius_m: float, min_turning_radius_m: float, wheelbase_m: float, *,
    num_scenarios: int, seed: int = 0, mode: str = "test",
) -> Tuple[BenchmarkScenarioSpec, ...]:
    """Draws ``num_scenarios`` scenarios from the ``mode`` seed pool (test,
    by default -- never the train pool, plan's train/validation/test
    separation) via the SAME ``LongHorizonSeedScheduler`` Phase 4 training
    uses, and freezes each into a :class:`BenchmarkScenarioSpec`. Calling
    this twice with the same arguments produces byte-identical specs
    (deterministic seed draw + deterministic generation), which is exactly
    what makes ``content_hash``/``occupancy_hash`` meaningful as a
    reproducibility check rather than just documentation."""
    if num_scenarios <= 0:
        raise ValueError("build_fixed_benchmark_manifest: num_scenarios must be > 0")
    scheduler = LongHorizonSeedScheduler(seed, cfg, mode=mode)
    specs = []
    for i in range(num_scenarios):
        world_seed = scheduler.next_seed()
        world = generate_long_horizon_world(world_seed, cfg, robot_radius_m, min_turning_radius_m, wheelbase_m, mode=mode)
        # world.goal_pose is in the GENERATOR's own world/odom frame, not
        # relative-to-start -- BenchmarkScenarioSpec.relative_goal (the
        # field external tooling/real-mission replay actually reads) must
        # be the mission-frame-relative goal (plan section 3.1's contract),
        # via the SAME MissionFrame transform run_ablation_mission below
        # uses when it regenerates this exact world.
        mission_frame = MissionFrame()
        mission_frame.initialize(PoseXYYaw(*world.start_pose))
        relative_goal = mission_frame.odom_to_mission(world.goal_pose[0], world.goal_pose[1])
        specs.append(BenchmarkScenarioSpec(
            scenario_id=f"{mode}_{i:04d}_seed{world_seed}", seed=world_seed, mode=mode,
            difficulty_class=_difficulty_class(world, mode), content_hash=_content_hash(world_seed, world),
            occupancy_hash=_occupancy_hash(world), start_pose=world.start_pose, relative_goal=relative_goal,
            shortest_feasible_path_length_m=world.shortest_path_length_m, dead_end_count=int(
                world.topology_metadata.get("dead_end_count", 0)),
            loop_count=int(world.topology_metadata.get("loop_count", 0)),
        ))
    return tuple(specs)


def _ablation_flags(ablation: str) -> Tuple[bool, bool, bool, bool, bool]:
    """``(global_rl_enabled, include_visited, topology, feasibility,
    global_risk)`` for one ablation label. Requirement F: ``include_visited``
    is False for B and True for every other Global-RL-enabled ablation
    (C-G), making B and C genuinely distinct (see
    ``GlobalRLConfig.include_visited_channel``)."""
    if ablation == "A":
        return False, False, False, False, False
    if ablation == "B":
        return True, False, False, False, False
    if ablation == "C":
        return True, True, False, False, False
    if ablation == "D":
        return True, True, True, False, False
    if ablation == "E":
        return True, True, True, True, False
    if ablation == "F":
        return True, True, False, False, True
    if ablation == "G":
        return True, True, True, True, True
    raise ValueError(f"unknown ablation label {ablation!r}, expected one of {ABLATION_LABELS}")


def effective_global_rl_config(base: GlobalRLConfig, ablation: str) -> GlobalRLConfig:
    """Forces ``base``'s ablation-relevant flags to exactly what ``ablation``
    implies (see :func:`_ablation_flags`) -- the ablation LABEL is the
    single source of truth, never the caller-supplied config's own flags.
    Without this, a caller could accidentally run label "B" against a
    config with Phase 5 flags already on (``build_global_observation``
    would then require feature arrays this function's caller never
    computes for "B", raising) or run label "D" against a config with
    every flag off (silently producing a Phase-4-shaped observation with
    no topology features at all, defeating the ablation)."""
    global_rl_enabled, include_visited, topology_enabled, feasibility_enabled, global_risk_enabled = (
        _ablation_flags(ablation)
    )
    return dataclasses.replace(
        base, enabled=global_rl_enabled, include_visited_channel=include_visited,
        topology_feedback_enabled=topology_enabled,
        feasibility_feedback_enabled=feasibility_enabled, global_risk_feedback_enabled=global_risk_enabled,
    )


def effective_memory_config(base: MemoryConfig, ablation: str) -> MemoryConfig:
    """Companion to :func:`effective_global_rl_config` -- forces
    ``enabled`` to match the SAME topology flag, so ``global_rl.
    topology_feedback_enabled`` and ``memory.enabled`` can never disagree
    for a given ablation label (mirrors ``config.schema.Profile.validate``'s
    own cross-section requirement between these two sections)."""
    _, _, topology_enabled, _, _ = _ablation_flags(ablation)
    return dataclasses.replace(base, enabled=topology_enabled)


def effective_feasibility_config(base: GlobalFeasibilityConfig, ablation: str) -> GlobalFeasibilityConfig:
    """Companion to :func:`effective_global_rl_config` -- forces
    ``enabled`` on whenever the ablation needs EITHER the feasibility-
    feedback candidate columns OR the global-risk-feedback column (both
    reuse the SAME underlying feasibility computation, plan 9.6/9.9)."""
    _, _, _, feasibility_enabled, global_risk_enabled = _ablation_flags(ablation)
    return dataclasses.replace(base, enabled=(feasibility_enabled or global_risk_enabled))


def run_ablation_mission(
    ablation: str, scenario: BenchmarkScenarioSpec, *,
    global_cfg: GlobalRLConfig, hierarchy_cfg: HierarchyConfig, mapping_cfg: MappingConfig,
    memory_cfg: MemoryConfig, feasibility_cfg: GlobalFeasibilityConfig,
    long_horizon_cfg: LongHorizonWorldConfig, robot_radius_m: float, min_turning_radius_m: float,
    wheelbase_m: float, max_options: int, max_local_steps: int, mission_timeout_steps: Optional[int],
    agent=None, rng: Optional[np.random.RandomState] = None,
    local_executor_factory=None, local_evaluator=None,
) -> dict:
    """Runs ONE mission for ONE ablation against ``scenario`` (regenerated
    from its own ``seed``, never a cached world -- ``content_hash``/
    ``occupancy_hash`` let a caller verify this regeneration stayed
    faithful) and returns an episode dict consumable by
    ``evaluation.global_metrics.aggregate``. ``agent`` (a
    ``GlobalDQNAgent``-shaped object with ``select_action(obs, epsilon=0.0,
    rng=rng) -> int``) is REQUIRED for ablations B-G, ignored for A.

    ``global_cfg``/``memory_cfg``/``feasibility_cfg`` are ONLY a source of
    non-ablation-related settings (network dims, reward weights, node-
    creation thresholds, ...) -- their own ablation flags (``enabled``,
    ``topology_feedback_enabled``, ...) are OVERRIDDEN here to match
    ``ablation`` via :func:`effective_global_rl_config`/
    :func:`effective_memory_config`/:func:`effective_feasibility_config`,
    so the ablation label is always the single source of truth regardless
    of what flags the caller's base config happened to carry.

    ``local_evaluator`` (requirement E): a
    ``navigation.hierarchy.feasibility.LocalFeasibilityEvaluator`` (in
    practice a
    ``navigation.hierarchy.local_feasibility_evaluator.FrozenLocalFeasibilityEvaluator``
    bound to the SAME frozen Local policy/live sensor state as
    ``local_executor_factory``'s executor) -- REQUIRED for ablations E/F/G
    (whichever need ``feasibility_feedback_enabled``/
    ``global_risk_feedback_enabled``; raises immediately, before running
    any mission, if missing -- never silently zero-fills
    ``predicted_action_risk``/``progress_preserving`` for a tier that
    declared it needs them). Ignored (never even checked) for A/B/C/D."""
    global_cfg = effective_global_rl_config(global_cfg, ablation)
    memory_cfg = effective_memory_config(memory_cfg, ablation)
    feasibility_cfg = effective_feasibility_config(feasibility_cfg, ablation)
    global_rl_enabled, _, topology_enabled, feasibility_enabled, global_risk_enabled = _ablation_flags(ablation)
    if global_rl_enabled and agent is None:
        raise ValueError(f"run_ablation_mission: ablation {ablation!r} requires a Global agent")
    if (feasibility_enabled or global_risk_enabled) and local_evaluator is None:
        raise ValueError(
            f"run_ablation_mission: ablation {ablation!r} requires feasibility_feedback_enabled="
            f"{feasibility_enabled} / global_risk_feedback_enabled={global_risk_enabled}, both of which need a "
            "real local_evaluator (navigation.hierarchy.local_feasibility_evaluator."
            "FrozenLocalFeasibilityEvaluator) -- refusing to run this ablation with predicted_action_risk/"
            "progress_preserving silently zero-filled"
        )
    rng = rng if rng is not None else np.random.RandomState(0)

    world = generate_long_horizon_world(
        scenario.seed, long_horizon_cfg, robot_radius_m, min_turning_radius_m, wheelbase_m, mode=scenario.mode,
    )
    _verify_scenario_regeneration(scenario, world)
    mission_frame = MissionFrame()
    mission_frame.initialize(PoseXYYaw(*world.start_pose))
    partial_map = PartialMap(mapping_cfg)
    goal_mission = mission_frame.odom_to_mission(world.goal_pose[0], world.goal_pose[1])
    # local_executor_factory=None (default, every pre-existing caller):
    # byte-identical SimplifiedKinematicLocalExecutor construction. A live
    # baseline/hierarchical-benchmark run passes
    # navigation.local_rl.live_gazebo_executor.LiveGazeboLocalExecutor.bind_mission
    # instead, matching train_hierarchical_dqn.HierarchicalTrainingLoop's
    # identical injection point.
    executor = (
        local_executor_factory(world, mission_frame) if local_executor_factory is not None
        else SimplifiedKinematicLocalExecutor(world, mission_frame, robot_radius_m)
    )
    robot_max_curvature = 1.0 / min_turning_radius_m
    candidates = build_candidate_set(global_cfg)

    memory_active = topology_enabled and memory_cfg.enabled
    graph: Optional[TopologicalGraph] = None
    node_manager: Optional[TopologicalNodeManager] = None
    dead_end_detector: Optional[DeadEndDetector] = None
    if memory_active:
        graph = TopologicalGraph(max_nodes=memory_cfg.max_nodes)
        node_manager_cfg, dead_end_cfg = memory_config_from(memory_cfg)
        node_manager = TopologicalNodeManager(graph, RouteHistory(), node_manager_cfg)
        dead_end_detector = DeadEndDetector(dead_end_cfg)

    feasibility_active = (feasibility_enabled or global_risk_enabled) and feasibility_cfg.enabled
    feasibility_config = feasibility_config_from(feasibility_cfg) if feasibility_active else None
    max_nodes = memory_cfg.max_nodes_in_observation if memory_active else 0

    num_subgoals = 0
    subgoal_success_count = 0
    dead_end_entries = 0
    repeated_dead_end_entries = 0
    loop_count = 0
    local_planner_failures = 0
    global_replans = 0
    route_length_m = 0.0
    revisit_distance_m = 0.0
    backtracking_distance_m = 0.0
    navigation_time_sec = 0.0
    mission_wall_start = time.monotonic()
    cumulative_local_steps = 0
    mission_reached = False
    now_step = 0
    # item 8: Local safety telemetry, connected to the benchmark instead of
    # staying write-only (OptionTelemetry's own docstring). Optional
    # (``Optional[float]``) fields stay None when never provided by ANY
    # option this mission ran (SubgoalResult.minimum_clearance_m is None
    # when the executor never sampled clearance; OptionTelemetry's own
    # fields are None/0 by dataclass default for an executor that doesn't
    # compute them) -- never disguised as a fabricated 0.0.
    min_clearance_m: Optional[float] = None
    min_time_to_collision_sec: Optional[float] = None
    min_stopping_margin_m: Optional[float] = None
    steering_saturation_count = 0
    emergency_stop_count = 0
    inference_timeout_count = 0
    inference_error_count = 0
    command_timeout_count = 0
    collision_count = 0
    collision_signal_source = "unavailable"
    risk_weighted_sum = 0.0
    risk_weight_total = 0
    risk_integral = 0.0
    predicted_risk_max: Optional[float] = None
    local_failure_reason_codes: List[str] = []

    for option_index in range(max_options):
        pose_mission = mission_frame.odom_pose_to_mission(executor.pose_world)

        if not global_rl_enabled:
            subgoal_endpoint = goal_mission
        else:
            mask = compute_action_mask(candidates, partial_map, pose_mission, global_cfg, robot_max_curvature)
            feasibility_features = None
            if feasibility_active:
                feasibility_features = compute_feasibility_features(
                    candidates, partial_map, pose_mission, robot_max_curvature, feasibility_config,
                    local_evaluator=local_evaluator,
                )
            topology_features = None
            node_tensor = None
            node_validity_mask = None
            if memory_active and graph is not None:
                topology_features = compute_candidate_topology_features(
                    candidates, graph, pose_mission, node_manager.config.node_merge_radius_m,
                )
                node_tensor, node_validity_mask = graph.to_fixed_tensor(
                    max_nodes, pose_mission, goal_mission, goal_distance_norm_m=global_cfg.goal_distance_norm_m,
                    recency_norm_steps=memory_cfg.node_recency_norm_steps, now_step=now_step,
                )
            obs = build_global_observation(
                partial_map, pose_mission, goal_mission, candidates, mask, speed_mps=0.0,
                previous_action_index=global_cfg.fallback_index, elapsed_mission_ratio=option_index / max_options,
                config=global_cfg, topology_candidate_features=topology_features,
                feasibility_candidate_features=(feasibility_features if feasibility_enabled else None),
                global_risk_candidate_features=(
                    feasibility_features[:, 3:4] if (global_risk_enabled and feasibility_features is not None) else None
                ),
                node_tensor=node_tensor, node_validity_mask=node_validity_mask,
            )
            action = agent.select_action(obs, epsilon=0.0, rng=rng)
            subgoal_endpoint = candidate_endpoint_mission(candidates[action], pose_mission)

        coordinator = HierarchyCoordinator(hierarchy_config_from(hierarchy_cfg))
        # item 1: a FRESH coordinator exists for exactly ONE option, so it
        # must be seeded with an OPTION-LOCAL zero-based clock, never
        # `now_step` (a MISSION-cumulative local-step COUNT reused here as
        # a fake "seconds" value, subtracted below against whatever REAL
        # clock the executor's own run_option() reports -- for
        # LiveGazeboLocalExecutor that clock is real dt-paced seconds,
        # process-lifetime by construction, so this mismatch is exactly
        # what produced a 17.5s mission budget benchmarking at
        # 293-946.5s). `now_step` itself remains this function's own
        # mission-cumulative counter for topological-memory recency
        # bookkeeping further below -- only the coordinator's own clock
        # seeding changes here.
        coordinator.start_mission(goal_mission[0], goal_mission[1], now_step=0, now_time_sec=0.0)
        coordinator.enqueue_subgoal(*subgoal_endpoint)

        def _is_valid(x: float, y: float, _pm=partial_map) -> bool:
            cell = _pm.world_to_cell(x, y)
            return True if cell is None else not bool(_pm.channels().inflated[cell])

        activated = coordinator.activate_next_subgoal(
            pose_mission, now_step=0, now_time_sec=0.0, is_valid=_is_valid,
        )
        visited_before = partial_map.visited_cell_count()
        if activated:
            executor.run_option(coordinator, partial_map, rng, max_local_steps)
        visited_after = partial_map.visited_cell_count()

        result = coordinator.last_subgoal_result
        next_pose_mission = mission_frame.odom_pose_to_mission(executor.pose_world)
        local_steps = result.local_steps if result is not None else 1
        cumulative_local_steps += local_steps
        now_step += local_steps

        if result is not None:
            # item 1 (bullet 6): fail-fast if elapsed_time_sec falls
            # outside the actual control-time budget this option could
            # possibly have consumed -- a real, executable sanity check
            # (never just documentation) against exactly the class of bug
            # this coordinator-clock fix addresses (a 17.5s mission budget
            # silently benchmarking at 293-946.5s). A generous +5% +0.05s
            # tolerance absorbs float accumulation only, never a whole
            # extra domain mismatch. `executor.dt_sec` is optional (a
            # future LocalOptionExecutor implementation need not expose
            # it) -- the check is skipped, never fabricated, when absent.
            option_dt_sec = getattr(executor, "dt_sec", None)
            if option_dt_sec is not None:
                max_option_elapsed_sec = max_local_steps * option_dt_sec * 1.05 + 0.05
                if not (0.0 <= result.elapsed_time_sec <= max_option_elapsed_sec):
                    raise ValueError(
                        f"run_ablation_mission: scenario {scenario.scenario_id!r} option {option_index} "
                        f"produced SubgoalResult.elapsed_time_sec={result.elapsed_time_sec:.3f}s, outside "
                        f"the valid [0, {max_option_elapsed_sec:.3f}]s range for max_local_steps="
                        f"{max_local_steps} * dt_sec={option_dt_sec:.3f}s -- this indicates a coordinator/"
                        "executor clock-domain bug (see HierarchyCoordinator.force_terminate_active_subgoal "
                        "and LiveGazeboLocalExecutor.run_option's own option-local-clock contract), not a "
                        "plausible option duration; refusing to report this benchmark result as valid"
                    )
            num_subgoals += 1
            route_length_m += result.path_length_m
            navigation_time_sec += result.elapsed_time_sec
            # A cell that was ALREADY visited before this option but the
            # robot still passed through again contributes zero NEW visited
            # cells -- the gap between the option's own path length and the
            # newly-visited-cell count is a rough proxy for redundant travel.
            newly_visited_cells = visited_after - visited_before
            if newly_visited_cells <= 0 and result.path_length_m > 0.0:
                revisit_distance_m += result.path_length_m
            if result.status == SubgoalStatus.REACHED:
                subgoal_success_count += 1
            elif result.status in FAILURE_SUBGOAL_STATUSES:
                local_planner_failures += 1
                local_failure_reason_codes.append(result.reason)
            elif result.status == SubgoalStatus.CANCELLED_BY_REPLAN:
                global_replans += 1
                local_failure_reason_codes.append(result.reason)

            # item 8: connect SubgoalResult's own safety stats -- already
            # computed per option, but never surfaced past this function
            # before now.
            if result.minimum_clearance_m is not None:
                min_clearance_m = (
                    result.minimum_clearance_m if min_clearance_m is None
                    else min(min_clearance_m, result.minimum_clearance_m)
                )
            steering_saturation_count += result.steering_saturation_count
            emergency_stop_count += result.emergency_stop_count
            if result.mean_predicted_risk is not None:
                risk_weighted_sum += result.mean_predicted_risk * result.local_steps
                risk_weight_total += result.local_steps
                risk_integral += result.mean_predicted_risk * result.local_steps
                if result.max_predicted_risk is not None:
                    predicted_risk_max = (
                        result.max_predicted_risk if predicted_risk_max is None
                        else max(predicted_risk_max, result.max_predicted_risk)
                    )

        # item 8: OptionTelemetry (TTC/stopping-margin/inference-and-command-
        # timeout-counts/collision) -- only some LocalOptionExecutor
        # implementations provide this (see option_telemetry.py's own
        # docstring); getattr keeps this generic across executors that
        # don't.
        telemetry = getattr(executor, "last_option_telemetry", None)
        if telemetry is not None:
            if telemetry.min_time_to_collision_sec is not None:
                min_time_to_collision_sec = (
                    telemetry.min_time_to_collision_sec if min_time_to_collision_sec is None
                    else min(min_time_to_collision_sec, telemetry.min_time_to_collision_sec)
                )
            if telemetry.min_stopping_margin_m is not None:
                min_stopping_margin_m = (
                    telemetry.min_stopping_margin_m if min_stopping_margin_m is None
                    else min(min_stopping_margin_m, telemetry.min_stopping_margin_m)
                )
            inference_timeout_count += telemetry.inference_timeout_count
            inference_error_count += telemetry.inference_error_count
            command_timeout_count += telemetry.command_timeout_count
            collision_count += telemetry.collision_count
            collision_signal_source = telemetry.collision_signal_source

        if memory_active and node_manager is not None and dead_end_detector is not None:
            next_mask = compute_action_mask(candidates, partial_map, next_pose_mission, global_cfg, robot_max_curvature)
            free_count = int(next_mask.sum())
            channels = partial_map.channels()
            frontier_count = 0
            for c in candidates:
                if c.is_fallback or not next_mask[c.index]:
                    continue
                ep = candidate_endpoint_mission(c, next_pose_mission)
                cell = partial_map.world_to_cell(*ep)
                if cell is None or channels.unknown[cell]:
                    frontier_count += 1
            evidence = dead_end_detector.evaluate_evidence(
                known_free_direction_count=free_count, known_frontier_direction_count=frontier_count,
                recent_final_goal_distance_history=(), consecutive_emergency_stops=(
                    result.emergency_stop_count if result is not None else 0
                ),
                valid_candidate_count=free_count,
            )
            existing_node_id = graph.find_nearest_node(
                next_pose_mission.x, next_pose_mission.y, max_radius_m=node_manager.config.node_merge_radius_m,
            )
            verdict = dead_end_detector.verdict(evidence, graph=graph, node_id=existing_node_id)
            if verdict.is_dead_end:
                dead_end_entries += 1
                if verdict.is_repeated:
                    repeated_dead_end_entries += 1
            event = node_manager.maybe_create_or_update_node(
                next_pose_mission, now_step=now_step,
                subgoal_reached=(result is not None and result.status == SubgoalStatus.REACHED),
                subgoal_failed=(result is not None and result.status in FAILURE_SUBGOAL_STATUSES),
                dead_end_detected=verdict.is_dead_end, free_direction_count=free_count,
                path_length_since_last_m=(result.path_length_m if result is not None else 0.0),
                elapsed_time_since_last_sec=(result.elapsed_time_sec if result is not None else 0.0),
                risk=(result.mean_predicted_risk if result is not None else None),
            )
            if event is not None and event.route_event.event_type == RouteEventType.BACKTRACK:
                backtracking_distance_m += (result.path_length_m if result is not None else 0.0)
            elif event is not None and event.route_event.event_type == RouteEventType.REVISIT:
                # RouteEventType.REVISIT's own docstring: "re-crossing a loop
                # far from home" -- the "loop" metric plan 10.4/requirement D
                # asks for, distinct from a simple immediate BACKTRACK.
                loop_count += 1

        mission_reached = bool(coordinator.mission_reached)
        budget_exhausted = option_index == max_options - 1
        timed_out = bool(mission_timeout_steps is not None and cumulative_local_steps >= mission_timeout_steps)
        if mission_reached or budget_exhausted or timed_out:
            break

    final_snapshot = partial_map.snapshot()
    channels = final_snapshot.channels
    observed_area_m2 = float(np.count_nonzero(final_snapshot.observed)) * (partial_map.resolution_m ** 2)
    visited_area_m2 = float(np.count_nonzero(final_snapshot.visited_count)) * (partial_map.resolution_m ** 2)

    return {
        "ablation": ablation, "scenario_id": scenario.scenario_id, "success": mission_reached,
        "route_length_m": route_length_m,
        "shortest_feasible_path_length_m": scenario.shortest_feasible_path_length_m,
        # Compatibility alias: this is control time to termination, not
        # necessarily time-to-goal. Failed episodes expose no time-to-goal.
        "navigation_time_sec": navigation_time_sec,
        "control_elapsed_time_sec": navigation_time_sec,
        "time_to_goal_sec": navigation_time_sec if mission_reached else None,
        "time_to_termination_sec": navigation_time_sec,
        "wall_elapsed_time_sec": time.monotonic() - mission_wall_start,
        "num_global_subgoals": num_subgoals,
        "subgoal_success_count": subgoal_success_count, "subgoal_attempt_count": num_subgoals,
        "revisit_distance_m": revisit_distance_m, "dead_end_entries": dead_end_entries,
        "repeated_dead_end_entries": repeated_dead_end_entries, "loop_count": loop_count,
        "backtracking_distance_m": backtracking_distance_m,
        "cumulative_local_steps": cumulative_local_steps,
        "explored_area_m2": observed_area_m2, "visited_area_m2": visited_area_m2,
        "local_planner_failure_count": local_planner_failures, "global_replan_count": global_replans,
        "localization_drift_error_m": None,
        # item 8: Local safety telemetry, connected -- see the accumulation
        # comments above and OptionTelemetry's own docstring for what's a
        # ground-truth signal vs. a documented proxy vs. genuinely
        # unavailable (None/0, never disguised).
        "min_clearance_m": min_clearance_m,
        "min_time_to_collision_sec": min_time_to_collision_sec,
        "min_stopping_margin_m": min_stopping_margin_m,
        "steering_saturation_count": steering_saturation_count,
        "steering_saturation_rate": (
            steering_saturation_count / cumulative_local_steps if cumulative_local_steps > 0 else None
        ),
        "emergency_stop_count": emergency_stop_count,
        "predicted_risk_mean": (risk_weighted_sum / risk_weight_total) if risk_weight_total > 0 else None,
        "predicted_risk_max": predicted_risk_max,
        "risk_integral": risk_integral,
        "inference_timeout_count": inference_timeout_count,
        "inference_error_count": inference_error_count,
        "command_timeout_count": command_timeout_count,
        "collision_count": collision_count,
        "collision_signal_source": collision_signal_source,
        "local_failure_reason_codes": list(local_failure_reason_codes),
    }


def run_ablation_benchmark(
    ablation: str, scenarios: Sequence[BenchmarkScenarioSpec], *, agent=None, seed: int = 0, **mission_kwargs,
) -> List[dict]:
    """Runs every scenario in ``scenarios`` (the SAME manifest for every
    ablation -- plan 9.9/10.10's "동일 benchmark manifest에서 모든 baseline
    실행") for one ablation label, returning the list of episode dicts."""
    rng = np.random.RandomState(seed)
    return [
        run_ablation_mission(ablation, scenario, agent=agent, rng=rng, **mission_kwargs)
        for scenario in scenarios
    ]
