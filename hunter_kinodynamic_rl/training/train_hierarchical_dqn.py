#!/usr/bin/env python3
"""ROS-free Phase 4 hierarchical training core loop (plan section 8.9).

Ties together, WITHOUT any ROS/Gazebo/rclpy import:

- Phase 3's ``LongHorizonWorld`` (procedural room/corridor/junction/loop/
  dead-end world, generated per-mission via a mode-aware
  ``LongHorizonSeedScheduler`` -- train/validation/test pools stay separate)
- Phase 1/2's ``MissionFrame`` + ``PartialMap`` + ``HierarchyCoordinator``
  (mission/subgoal lifecycle, unchanged)
- Phase 4's Global RL (``observation``/``action_mask``/``subgoal_sampler``/
  ``reward``/``replay``/``agent`` -- this package)

Local-option execution is delegated to a ``LocalOptionExecutor`` -- this
loop never assumes HOW a subgoal is actually driven. The one shipped here,
:class:`SimplifiedKinematicLocalExecutor`, is a deliberately lightweight
point-robot motion model (steer-then-drive toward the active subgoal,
simulated LiDAR ray-cast into the SAME ``PartialMap``/raytracing code Phase 1
uses, termination via the SAME ``HierarchyCoordinator``/``SubgoalManager``/
``evaluate_replanning`` Phase 2 already implements) -- it stands in for the
real frozen local kinodynamic TQC so this loop's Global RL <-> map <->
hierarchy integration is trainable and testable without Gazebo (plan section
8: "실제 Gazebo live integration이 너무 크면 ROS-free core loop와 node
adapter skeleton을 분리할 것"). Driving the REAL frozen TQC checkpoint
through live Gazebo is ``nodes/hierarchical_environment_node.py``'s job --
this loop has no opinion on it beyond the ``LocalOptionExecutor`` protocol
and the local-checkpoint hash recorded into every Global checkpoint's
manifest (see :meth:`HierarchicalTrainingLoop.checkpoint_meta`).

Local policy is ALWAYS frozen here -- there is no code path in this module
that updates a local-policy checkpoint (plan section 8.9 item 7).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Tuple

import numpy as np

from hunter_kinodynamic_rl.common.geometry import wrap_to_pi
from hunter_kinodynamic_rl.config.schema import GlobalFeasibilityConfig, GlobalRLConfig, \
    HierarchicalTrainingConfig, HierarchyConfig, LongHorizonWorldConfig, MappingConfig, MemoryConfig
from hunter_kinodynamic_rl.env.scenarios.long_horizon_curriculum import apply_level
from hunter_kinodynamic_rl.env.scenarios.long_horizon_generator import (
    LongHorizonSeedScheduler, generate_long_horizon_world,
)
from hunter_kinodynamic_rl.env.scenarios.long_horizon_solvability import inflate_occupancy, world_to_cell
from hunter_kinodynamic_rl.env.scenarios.long_horizon_world import LongHorizonWorld
from hunter_kinodynamic_rl.navigation.global_rl.action_mask import compute_action_mask
from hunter_kinodynamic_rl.navigation.global_rl.observation import (
    GlobalObservation, N_SCALARS, resolve_candidate_feature_names, resolve_map_channel_names,
    build_global_observation,
)
from hunter_kinodynamic_rl.navigation.global_rl.reward import compute_global_reward
from hunter_kinodynamic_rl.navigation.global_rl.replay import GlobalReplayBuffer, GlobalTransition
from hunter_kinodynamic_rl.navigation.global_rl.subgoal_sampler import build_candidate_set, candidate_endpoint_mission
from hunter_kinodynamic_rl.navigation.hierarchy.coordinator import HierarchyCoordinator, HierarchyCoordinatorConfig
from hunter_kinodynamic_rl.navigation.hierarchy.failure_recovery import FailureRecoveryConfig
from hunter_kinodynamic_rl.navigation.hierarchy.feasibility import FeasibilityConfig, compute_feasibility_features
from hunter_kinodynamic_rl.navigation.hierarchy.replanning import ReplanningConfig
from hunter_kinodynamic_rl.navigation.hierarchy.subgoal_manager import (
    FAILURE_SUBGOAL_STATUSES, SubgoalManagerConfig, SubgoalStatus,
)
from hunter_kinodynamic_rl.navigation.local_rl.option_telemetry import OptionTelemetry
from hunter_kinodynamic_rl.navigation.mapping.partial_map import PartialMap
from hunter_kinodynamic_rl.navigation.memory.dead_end_detector import DeadEndDetector, DeadEndDetectorConfig
from hunter_kinodynamic_rl.navigation.memory.node_manager import (
    NodeManagerConfig, TopologicalNodeManager, compute_candidate_topology_features,
)
from hunter_kinodynamic_rl.navigation.memory.route_history import RouteHistory
from hunter_kinodynamic_rl.navigation.memory.topological_graph import (
    N_NODE_FEATURES, NODE_FEATURE_NAMES, TopologicalGraph,
)
from hunter_kinodynamic_rl.navigation.mission.goal_manager import GoalManagerConfig
from hunter_kinodynamic_rl.navigation.mission.mission_frame import MissionFrame, PoseXYYaw


def resolve_local_checkpoint_file(local_checkpoint_dir: str, local_checkpoint_name: str) -> Optional[str]:
    """``(directory, tag)`` -> the resolved ``model.pt`` path a
    generation-layout loader (``rl.checkpointing.manager.load_generation``)
    would actually read for that tag -- ``<local_checkpoint_dir>/<tag>`` is
    a symlink into ``.generations/<uuid>/``, so joining ``model.pt`` onto it
    transparently follows that symlink. Returns ``None`` when
    ``local_checkpoint_dir`` is unset (never a path built from an empty
    directory)."""
    if not local_checkpoint_dir:
        return None
    return os.path.join(local_checkpoint_dir, local_checkpoint_name, "model.pt")


def hierarchy_config_from(cfg: HierarchyConfig) -> HierarchyCoordinatorConfig:
    """Builds a Phase 2 ``HierarchyCoordinatorConfig`` from the flat Phase 2
    ``HierarchyConfig`` section -- shared verbatim between this trainer and
    any future ROS adapter so both sides of the loop use IDENTICAL subgoal
    tolerance/replanning/recovery thresholds."""
    return HierarchyCoordinatorConfig(
        goal=GoalManagerConfig(),
        subgoal=SubgoalManagerConfig(
            position_tolerance_m=cfg.subgoal_position_tolerance_m,
            heading_tolerance_rad=cfg.subgoal_heading_tolerance_rad,
            require_low_speed_on_reach=cfg.subgoal_require_low_speed_on_reach,
            goal_speed_threshold_mps=cfg.subgoal_goal_speed_threshold_mps,
        ),
        replanning=ReplanningConfig(
            local_option_timeout_steps=cfg.local_option_timeout_steps,
            no_progress_window_steps=cfg.no_progress_window_steps,
            no_progress_min_delta_m=cfg.no_progress_min_delta_m,
            consecutive_emergency_stop_limit=cfg.consecutive_emergency_stop_limit,
            local_risk_threshold=cfg.local_risk_threshold,
            localization_min_confidence=cfg.localization_min_confidence,
        ),
        recovery=FailureRecoveryConfig(max_retries_per_subgoal=cfg.max_retries_per_subgoal),
        mission_timeout_steps=cfg.mission_timeout_steps,
        mission_timeout_sec=cfg.mission_timeout_sec,
    )


def memory_config_from(cfg: MemoryConfig) -> "tuple[NodeManagerConfig, DeadEndDetectorConfig]":
    """Splits the flat YAML-facing ``MemoryConfig`` section (Phase 5, plan
    9.3/9.4) into ``navigation.memory``'s own pure-logic dataclasses --
    mirrors ``hierarchy_config_from``'s translation pattern exactly."""
    return (
        NodeManagerConfig(
            node_min_distance_m=cfg.node_min_distance_m, node_heading_change_rad=cfg.node_heading_change_rad,
            node_merge_radius_m=cfg.node_merge_radius_m,
        ),
        DeadEndDetectorConfig(
            free_direction_threshold=cfg.free_direction_threshold,
            progress_stall_window_steps=cfg.progress_stall_window_steps,
            progress_stall_min_delta_m=cfg.progress_stall_min_delta_m,
            repeated_estop_limit=cfg.repeated_estop_limit, evidence_vote_threshold=cfg.evidence_vote_threshold,
        ),
    )


def feasibility_config_from(cfg: GlobalFeasibilityConfig) -> FeasibilityConfig:
    """Translates the flat YAML-facing ``GlobalFeasibilityConfig`` section
    (Phase 5, plan 9.6) into ``navigation.hierarchy.feasibility``'s own
    pure-logic dataclass."""
    return FeasibilityConfig(
        rollout_sample_count=cfg.rollout_sample_count, clearance_search_radius_m=cfg.clearance_search_radius_m,
        clearance_norm_m=cfg.clearance_norm_m, historical_stats_radius_m=cfg.historical_stats_radius_m,
    )


class LocalOptionExecutor(Protocol):
    """Drives local control ticks (``coordinator.record_local_tick(...)``)
    against the CURRENT active subgoal until it terminates (reached/failed)
    or ``max_local_steps`` elapses, updating ``partial_map`` along the way.
    Never touches ``coordinator``'s mission-level state directly -- only
    ``record_local_tick``/reads of ``active_subgoal_mission``."""

    def run_option(
        self, coordinator: HierarchyCoordinator, partial_map: PartialMap, rng: np.random.RandomState,
        max_local_steps: int,
    ) -> None: ...


def _raycast_range(world: LongHorizonWorld, origin_xy: Tuple[float, float], angle_world: float,
                    max_range_m: float, step_m: float) -> float:
    """First occupied-cell hit distance along one beam, marched in fixed
    ``step_m`` increments from ``origin_xy`` -- ``math.inf`` (never
    ``max_range_m`` exactly) on no hit, matching
    ``PartialMap.integrate_beam``'s own "no return" convention."""
    h, w = world.occupancy.shape
    ox, oy = world.origin_xy
    n_steps = max(1, int(math.ceil(max_range_m / step_m)))
    for i in range(1, n_steps + 1):
        r = min(i * step_m, max_range_m)
        x = origin_xy[0] + math.cos(angle_world) * r
        y = origin_xy[1] + math.sin(angle_world) * r
        cell = world_to_cell(x, y, world.resolution_m, ox, oy, h, w)
        if cell is None:
            return math.inf
        if world.occupancy[cell]:
            return r
    return math.inf


@dataclass
class SimplifiedKinematicLocalExecutor:
    """Deliberately simplified stand-in for the frozen local kinodynamic TQC
    (see module docstring). ``world`` supplies collision/LiDAR ground truth
    ONLY -- it is never exposed to ``partial_map``/the Global observation
    directly, only via simulated ray-casts integrated the normal Phase 1 way."""

    world: LongHorizonWorld
    mission_frame: MissionFrame
    robot_radius_m: float
    lidar_range_max_m: float = 8.0
    lidar_beam_count: int = 36
    step_distance_m: float = 0.3
    max_turn_rate_rad: float = 0.6
    dt_sec: float = 0.3
    raycast_step_m: float = 0.1

    def __post_init__(self) -> None:
        self.pose_world = PoseXYYaw(*self.world.start_pose)
        self._inflated_occupancy = inflate_occupancy(self.world.occupancy, self.world.resolution_m,
                                                       self.robot_radius_m)
        # item 8: unlike LiveGazeboLocalExecutor's clearance-distance PROXY
        # (no real Gazebo contact sensor wired in there), this stand-in's
        # own _in_collision() check against the deterministic generated
        # world IS an authoritative ground-truth collision signal --
        # "ground_truth", never conflated with a proxy (see
        # option_telemetry.py's own docstring on the distinction).
        self.collision_signal_source = "ground_truth"
        self.last_option_telemetry: Optional[OptionTelemetry] = None

    def _in_collision(self, x: float, y: float) -> bool:
        h, w = self._inflated_occupancy.shape
        cell = world_to_cell(x, y, self.world.resolution_m, self.world.origin_xy[0], self.world.origin_xy[1], h, w)
        if cell is None:
            return True  # outside the generated world entirely -- treated as unsafe, never silently allowed
        return bool(self._inflated_occupancy[cell])

    def _integrate_lidar(self, partial_map: PartialMap, pose_mission: PoseXYYaw) -> Tuple[int, float]:
        before = partial_map.observed_count()
        angles_robot = np.linspace(-math.pi, math.pi, self.lidar_beam_count, endpoint=False)
        min_range = self.lidar_range_max_m
        ranges = np.empty(self.lidar_beam_count, dtype=np.float64)
        for i, a_robot in enumerate(angles_robot):
            a_world = wrap_to_pi(self.pose_world.yaw + a_robot)
            r = _raycast_range(self.world, (self.pose_world.x, self.pose_world.y), a_world,
                                self.lidar_range_max_m, self.raycast_step_m)
            ranges[i] = r
            if math.isfinite(r):
                min_range = min(min_range, r)
        angles_mission = pose_mission.yaw + angles_robot
        partial_map.integrate_scan((pose_mission.x, pose_mission.y), angles_mission, ranges, self.lidar_range_max_m)
        after = partial_map.observed_count()
        return after - before, min_range

    def run_option(
        self, coordinator: HierarchyCoordinator, partial_map: PartialMap, rng: np.random.RandomState,
        max_local_steps: int,
    ) -> None:
        # item 1: OPTION-LOCAL clock -- reset to zero on EVERY run_option()
        # call (a fresh coordinator is always seeded with now_step=0/
        # now_time_sec=0.0 too, see HierarchicalTrainingLoop.run_mission()
        # and long_horizon_benchmark.run_ablation_mission()) so this
        # matches LiveGazeboLocalExecutor's identical option-local
        # convention -- previously these were instance attributes reset
        # only once per MISSION (__post_init__), so a mission's second+
        # option kept accumulating the FIRST option's ticks/seconds into
        # its own elapsed_time_sec.
        now_step = 0
        now_time_sec = 0.0
        self.last_option_telemetry = OptionTelemetry(collision_signal_source=self.collision_signal_source)
        for _ in range(max_local_steps):
            if coordinator.stop_required:
                return
            subgoal_mission = coordinator.active_subgoal_mission
            sx_w, sy_w = self.mission_frame.mission_to_odom(*subgoal_mission)
            dx, dy = sx_w - self.pose_world.x, sy_w - self.pose_world.y
            desired_yaw = math.atan2(dy, dx)
            yaw_err = wrap_to_pi(desired_yaw - self.pose_world.yaw)
            turn = float(np.clip(yaw_err, -self.max_turn_rate_rad, self.max_turn_rate_rad))
            new_yaw = wrap_to_pi(self.pose_world.yaw + turn)
            steering_saturated = abs(yaw_err) > self.max_turn_rate_rad
            move_scale = max(0.0, math.cos(yaw_err))
            move_dist = self.step_distance_m * move_scale
            new_x = self.pose_world.x + math.cos(new_yaw) * move_dist
            new_y = self.pose_world.y + math.sin(new_yaw) * move_dist

            collided = self._in_collision(new_x, new_y)
            if collided:
                self.last_option_telemetry.collision_count += 1
            if not collided:
                self.pose_world = PoseXYYaw(new_x, new_y, new_yaw)
                actual_move = move_dist
            else:
                self.pose_world = PoseXYYaw(self.pose_world.x, self.pose_world.y, new_yaw)
                actual_move = 0.0

            pose_mission = self.mission_frame.odom_pose_to_mission(self.pose_world)
            newly_explored, min_range = self._integrate_lidar(partial_map, pose_mission)
            now_step += 1
            now_time_sec += self.dt_sec
            partial_map.record_visit(pose_mission.x, pose_mission.y, step=now_step)
            if collided:
                partial_map.record_failure(pose_mission.x, pose_mission.y)
            speed_mps = actual_move / self.dt_sec
            predicted_risk = float(np.clip(1.0 - min_range / self.lidar_range_max_m, 0.0, 1.0))
            subgoal_endpoint_blocked = self._in_collision(sx_w, sy_w)

            coordinator.record_local_tick(
                pose_mission, now_step=now_step, now_time_sec=now_time_sec,
                speed_mps=speed_mps, clearance_m=min_range, predicted_risk=predicted_risk,
                emergency_stop=collided, steering_saturated=steering_saturated,
                newly_explored_cells=newly_explored, subgoal_endpoint_blocked=subgoal_endpoint_blocked,
                localization_confidence=1.0,
            )
            if coordinator.subgoal_reached or coordinator.subgoal_failed or coordinator.mission_done:
                return
        # item 2: max_local_steps exhausted without any replanning trigger
        # firing (e.g. local_option_timeout_steps configured larger than
        # this call's own max_local_steps) -- never leave the subgoal
        # silently ACTIVE with no terminal SubgoalResult (mirrors
        # LiveGazeboLocalExecutor.run_option's identical fix).
        if coordinator.subgoal_manager.active:
            pose_mission = self.mission_frame.odom_pose_to_mission(self.pose_world)
            coordinator.force_terminate_active_subgoal(
                SubgoalStatus.FAILED_TIMEOUT, "max_local_steps_exhausted", pose_mission,
                now_step=now_step, now_time_sec=now_time_sec,
            )


@dataclass
class MissionOutcome:
    global_transitions: int
    mission_reached: bool
    mission_failed: bool
    mission_timed_out: bool


class HierarchicalTrainingLoop:
    """One instance per training run. ``run_mission()`` generates one
    Phase 3 long-horizon world (mode-scheduled), then drives repeated
    Global-decision -> local-option -> Global-reward cycles until the
    mission ends or ``hierarchical_training.max_global_options_per_mission``
    is hit. Every stored transition passes through
    ``GlobalReplayBuffer.add(..., mode="train")`` -- never validation/test
    (plan section 8.8's guard)."""

    def __init__(
        self, global_cfg: GlobalRLConfig, hierarchy_cfg: HierarchyConfig,
        long_horizon_cfg: LongHorizonWorldConfig, mapping_cfg: MappingConfig,
        training_cfg: HierarchicalTrainingConfig, robot_radius_m: float,
        min_turning_radius_m: float, wheelbase_m: float, seed: int, mode: str = "train",
        memory_cfg: Optional[MemoryConfig] = None, feasibility_cfg: Optional[GlobalFeasibilityConfig] = None,
        local_executor_factory: Optional[Callable[[LongHorizonWorld, MissionFrame], LocalOptionExecutor]] = None,
        local_evaluator=None,
    ):
        """``local_executor_factory`` (default ``None``): how ``run_mission``
        obtains the ``LocalOptionExecutor`` for each mission. ``None``
        (default, EVERY pre-existing caller) reproduces the original
        behaviour byte-identically -- constructs a fresh
        ``SimplifiedKinematicLocalExecutor(world, mission_frame,
        robot_radius_m)`` per mission, exactly as before this parameter
        existed. A caller wiring the real frozen Local TQC over live
        Gazebo (``navigation.local_rl.live_gazebo_executor.LiveGazeboLocalExecutor``)
        passes ``lambda world, mission_frame: live_executor.bind_mission(world,
        mission_frame)`` instead -- the world/seed are still generated HERE
        (this loop's own seed_scheduler), never re-drawn by the executor,
        so there is exactly one source of truth for which seed produced
        which mission.

        ``local_evaluator`` (requirement E): a
        ``navigation.hierarchy.feasibility.LocalFeasibilityEvaluator``
        (in practice a
        ``navigation.hierarchy.local_feasibility_evaluator.FrozenLocalFeasibilityEvaluator``).
        REQUIRED -- raises immediately, before any mission runs -- whenever
        ``global_cfg.feasibility_feedback_enabled`` or
        ``global_cfg.global_risk_feedback_enabled`` is True (ablation
        E/F/G-tier); this loop never silently zero-fills
        ``predicted_action_risk``/``progress_preserving`` for a config that
        declared it needs them."""
        self.global_cfg = global_cfg
        self.hierarchy_cfg = hierarchy_cfg
        self.mapping_cfg = mapping_cfg
        self.training_cfg = training_cfg
        self.memory_cfg = memory_cfg or MemoryConfig()
        self.feasibility_cfg = feasibility_cfg or GlobalFeasibilityConfig()
        self.robot_radius_m = robot_radius_m
        self.min_turning_radius_m = min_turning_radius_m
        self.wheelbase_m = wheelbase_m
        # kappa_max = 1 / R_min -- same relationship RobotConfig.max_curvature
        # itself uses (tan(steering_limit)/wheelbase == 1/turning_radius).
        # Threaded into every compute_action_mask() call below so the
        # Ackermann rollout check is constrained to what THIS vehicle can
        # actually turn, never an unconstrained straight-line assumption.
        self.robot_max_curvature = 1.0 / min_turning_radius_m
        self.mode = mode
        self.long_horizon_cfg = apply_level(long_horizon_cfg, training_cfg.long_horizon_level)
        self.seed_scheduler = LongHorizonSeedScheduler(seed, self.long_horizon_cfg, mode=mode)

        self.candidates = build_candidate_set(global_cfg)
        # Phase 5: ablation flags make these a function of global_cfg rather
        # than fixed module constants -- all default False, so a Phase-4-
        # only profile resolves to the exact same tuples/shape as before.
        self.map_channel_names = resolve_map_channel_names(global_cfg)
        self.candidate_feature_names = resolve_candidate_feature_names(global_cfg)
        self.memory_enabled = bool(self.memory_cfg.enabled and global_cfg.topology_feedback_enabled)
        self.feasibility_enabled = bool(self.feasibility_cfg.enabled and global_cfg.feasibility_feedback_enabled)
        self.global_risk_enabled = bool(self.feasibility_cfg.enabled and global_cfg.global_risk_feedback_enabled)
        self.local_evaluator = local_evaluator
        if (self.feasibility_enabled or self.global_risk_enabled) and self.local_evaluator is None:
            raise ValueError(
                "HierarchicalTrainingLoop: feasibility_feedback_enabled="
                f"{self.feasibility_enabled} / global_risk_feedback_enabled={self.global_risk_enabled} "
                "require a real local_evaluator (navigation.hierarchy.local_feasibility_evaluator."
                "FrozenLocalFeasibilityEvaluator) -- refusing to train this ablation with "
                "predicted_action_risk/progress_preserving silently zero-filled"
            )
        self.max_nodes = self.memory_cfg.max_nodes_in_observation if self.memory_enabled else 0
        self.node_manager_cfg, self.dead_end_cfg = memory_config_from(self.memory_cfg)
        self.feasibility_config = feasibility_config_from(self.feasibility_cfg)

        map_shape = (len(self.map_channel_names), global_cfg.map_crop_size_cells, global_cfg.map_crop_size_cells)
        self.replay = GlobalReplayBuffer(
            map_shape, N_SCALARS, global_cfg.n_candidates, len(self.candidate_feature_names),
            capacity=global_cfg.replay_capacity, seed=seed, node_shape=(self.max_nodes, N_NODE_FEATURES),
            map_channel_names=self.map_channel_names, candidate_feature_names=self.candidate_feature_names,
        )
        self.global_step = 0
        self.mission_index = 0
        self._local_executor_factory = local_executor_factory or (
            lambda world, mission_frame: SimplifiedKinematicLocalExecutor(world, mission_frame, robot_radius_m)
        )

    def _feasibility_features(self, partial_map: PartialMap, pose_mission: PoseXYYaw) -> Optional[np.ndarray]:
        """Computed ONCE per observation build and reused for BOTH the
        feasibility-feedback and global-risk-feedback candidate columns
        (plan 9.6/9.9: "global risk feedback는 feasibility의 rollout을
        재사용, 별도 risk 계산을 두 번 하지 않는다") -- ``None`` when neither
        ablation flag needs it, so a Phase-4-only profile never pays this
        computation's cost at all. The ROS-free training loop has no real
        local policy to query (see ``hierarchy.feasibility``'s module
        docstring), so ``local_evaluator=None`` -- ``predicted_action_risk``/
        ``progress_preserving`` come from ``self.local_evaluator`` --
        required (constructor fail-fast) whenever this branch is reached at
        all, so this is never a silent zero-fill for an ablation that
        declared it needs these features."""
        if not (self.feasibility_enabled or self.global_risk_enabled):
            return None
        return compute_feasibility_features(
            self.candidates, partial_map, pose_mission, self.robot_max_curvature, self.feasibility_config,
            local_evaluator=self.local_evaluator,
        )

    def _build_observation(
        self, partial_map: PartialMap, pose_mission: PoseXYYaw, final_goal_mission: Tuple[float, float],
        speed_mps: float, previous_action_index: int, elapsed_ratio: float,
        graph: Optional[TopologicalGraph] = None, feasibility_features: Optional[np.ndarray] = None,
    ) -> GlobalObservation:
        """``feasibility_features``, when the caller already computed it
        (e.g. to also read off the SELECTED candidate's predicted risk for
        the reward, see ``run_mission``), is reused as-is instead of being
        recomputed a second time -- plan 9.6/9.9's "재사용, 두 번 계산하지
        않는다" applies across observation AND reward, not just within one
        observation build."""
        mask = compute_action_mask(self.candidates, partial_map, pose_mission, self.global_cfg, self.robot_max_curvature)
        if feasibility_features is None:
            feasibility_features = self._feasibility_features(partial_map, pose_mission)
        topology_features = None
        node_tensor = None
        node_validity_mask = None
        if self.memory_enabled and graph is not None:
            topology_features = compute_candidate_topology_features(
                self.candidates, graph, pose_mission, self.node_manager_cfg.node_merge_radius_m,
            )
            node_tensor, node_validity_mask = graph.to_fixed_tensor(
                self.max_nodes, pose_mission, final_goal_mission,
                goal_distance_norm_m=self.global_cfg.goal_distance_norm_m,
                recency_norm_steps=self.memory_cfg.node_recency_norm_steps, now_step=self.global_step,
            )
        return build_global_observation(
            partial_map, pose_mission, final_goal_mission, self.candidates, mask,
            speed_mps=speed_mps, previous_action_index=previous_action_index, elapsed_mission_ratio=elapsed_ratio,
            config=self.global_cfg,
            topology_candidate_features=topology_features,
            feasibility_candidate_features=(feasibility_features if self.feasibility_enabled else None),
            global_risk_candidate_features=(feasibility_features[:, 3:4] if self.global_risk_enabled else None),
            node_tensor=node_tensor, node_validity_mask=node_validity_mask,
        )

    def run_mission(self, agent, epsilon: float, rng: np.random.RandomState) -> MissionOutcome:
        """Requires an object satisfying ``GlobalDQNAgent``'s
        ``select_action(obs, epsilon=..., rng=...) -> int`` contract (kept
        duck-typed here so this ROS-free loop stays importable without
        torch -- ``agent`` is only ever actually constructed by a
        torch-gated caller, e.g. ``main()`` below or a test).

        A FRESH ``HierarchyCoordinator`` is created for every Global option
        (never reused across options): Phase 2's own
        ``FailureRecoveryPolicy`` was designed around a heuristic, externally
        pre-populated subgoal QUEUE, where "no next candidate queued" after a
        failure genuinely means the sequence is exhausted. Phase 4 instead
        enqueues exactly ONE candidate per Global decision by design (the
        Global agent re-decides after every option, success or failure) --
        reusing one coordinator across options would make
        ``has_next_candidate`` False after every single non-REACHED outcome
        and abort the mission on the very first local failure. True mission
        termination is instead decided HERE, from
        ``coordinator.mission_reached`` (fires correctly even on a
        freshly-recreated coordinator, since ``GoalManager`` is re-armed with
        the SAME final goal every time) plus this loop's own option-budget
        and (optional) cumulative-local-step mission-timeout bookkeeping."""
        seed = self.seed_scheduler.next_seed()
        world = generate_long_horizon_world(
            seed, self.long_horizon_cfg, self.robot_radius_m, self.min_turning_radius_m, self.wheelbase_m,
            mode=self.mode,
        )
        self.mission_index += 1

        mission_frame = MissionFrame()
        mission_frame.initialize(PoseXYYaw(*world.start_pose))
        partial_map = PartialMap(self.mapping_cfg)
        goal_mission = mission_frame.odom_to_mission(world.goal_pose[0], world.goal_pose[1])
        executor = self._local_executor_factory(world, mission_frame)

        graph: Optional[TopologicalGraph] = None
        node_manager: Optional[TopologicalNodeManager] = None
        dead_end_detector: Optional[DeadEndDetector] = None
        if self.memory_enabled:
            graph = TopologicalGraph(max_nodes=self.memory_cfg.max_nodes)
            node_manager = TopologicalNodeManager(graph, RouteHistory(), self.node_manager_cfg)
            dead_end_detector = DeadEndDetector(self.dead_end_cfg)

        max_options = self.training_cfg.max_global_options_per_mission
        max_local_steps = self.training_cfg.max_local_steps_per_option
        mission_timeout_steps = self.hierarchy_cfg.mission_timeout_steps

        previous_action_index = self.global_cfg.fallback_index
        stored = 0
        mission_reached = False
        timed_out = False
        cumulative_local_steps = 0

        for option_index in range(max_options):
            pose_mission = mission_frame.odom_pose_to_mission(executor.pose_world)
            elapsed_ratio = option_index / max_options
            decision_feasibility_features = self._feasibility_features(partial_map, pose_mission)
            obs = self._build_observation(
                partial_map, pose_mission, goal_mission, speed_mps=0.0,
                previous_action_index=previous_action_index, elapsed_ratio=elapsed_ratio, graph=graph,
                feasibility_features=decision_feasibility_features,
            )
            action = agent.select_action(obs, epsilon=epsilon, rng=rng)
            candidate = self.candidates[action]
            endpoint_mission = candidate_endpoint_mission(candidate, pose_mission)
            predicted_risk_at_selection = (
                float(decision_feasibility_features[action, 3])
                if self.global_risk_enabled and decision_feasibility_features is not None else 0.0
            )

            coordinator = HierarchyCoordinator(hierarchy_config_from(self.hierarchy_cfg))
            # item 1: a FRESH coordinator exists for exactly ONE option
            # (see this method's own docstring), so it must be seeded with
            # an OPTION-LOCAL zero-based clock, never self.global_step
            # (a RUN-cumulative option-count index across every mission
            # this training loop instance ever runs) -- record_local_tick
            # below always reports the EXECUTOR's own option-local
            # now_step/now_time_sec (see SimplifiedKinematicLocalExecutor.
            # run_option/LiveGazeboLocalExecutor.run_option), so mixing a
            # run-cumulative baseline with an option-local tick clock
            # previously made SubgoalResult.elapsed_time_sec meaningless.
            # self.global_step remains this loop's own run-cumulative
            # counter for epsilon/replay/logging bookkeeping -- it is
            # simply never passed into the coordinator's clock anymore.
            coordinator.start_mission(goal_mission[0], goal_mission[1], now_step=0, now_time_sec=0.0)
            coordinator.enqueue_subgoal(*endpoint_mission)

            def _is_valid(x: float, y: float, _pm=partial_map) -> bool:
                cell = _pm.world_to_cell(x, y)
                return True if cell is None else not bool(_pm.channels().inflated[cell])

            start_final_goal_distance = coordinator.goal_manager.distance_and_bearing(pose_mission)[0]
            activated = coordinator.activate_next_subgoal(
                pose_mission, now_step=0, now_time_sec=0.0, is_valid=_is_valid,
            )
            observed_before = partial_map.observed_count()

            if activated:
                executor.run_option(coordinator, partial_map, rng, max_local_steps)

            next_pose_mission = mission_frame.odom_pose_to_mission(executor.pose_world)
            end_final_goal_distance = coordinator.goal_manager.distance_and_bearing(next_pose_mission)[0]
            newly_explored = partial_map.observed_count() - observed_before
            result = coordinator.last_subgoal_result
            local_steps = result.local_steps if result is not None else 1
            cumulative_local_steps += local_steps
            risk_integral = (result.mean_predicted_risk or 0.0) * local_steps if result is not None else 0.0
            subgoal_failed = bool(result is not None and result.status in FAILURE_SUBGOAL_STATUSES)

            if self.memory_enabled and node_manager is not None and dead_end_detector is not None:
                next_mask_for_topology = compute_action_mask(
                    self.candidates, partial_map, next_pose_mission, self.global_cfg, self.robot_max_curvature,
                )
                known_free_direction_count = int(next_mask_for_topology.sum())
                # No unknown-frontier estimate is computed at this point in
                # the loop (the unknown-gain candidate feature is part of
                # the NEXT observation, built further below) -- approximate
                # frontier presence from whether ANY valid candidate points
                # at unexplored (never-observed) territory; a value of 0
                # free directions with an unknown endpoint still counts as
                # "frontier present" (there's somewhere new to try), so this
                # under-counts conservatively rather than over-claiming a
                # dead end.
                frontier_count = 0
                channels_for_frontier = partial_map.channels()
                for c in self.candidates:
                    if c.is_fallback or not next_mask_for_topology[c.index]:
                        continue
                    ep = candidate_endpoint_mission(c, next_pose_mission)
                    cell = partial_map.world_to_cell(*ep)
                    if cell is None or channels_for_frontier.unknown[cell]:
                        frontier_count += 1
                evidence = dead_end_detector.evaluate_evidence(
                    known_free_direction_count=known_free_direction_count,
                    known_frontier_direction_count=frontier_count,
                    recent_final_goal_distance_history=(start_final_goal_distance, end_final_goal_distance),
                    consecutive_emergency_stops=(
                        result.emergency_stop_count if result is not None else 0
                    ),
                    valid_candidate_count=known_free_direction_count,
                )
                existing_node_id = graph.find_nearest_node(
                    next_pose_mission.x, next_pose_mission.y, max_radius_m=self.node_manager_cfg.node_merge_radius_m,
                )
                verdict = dead_end_detector.verdict(evidence, graph=graph, node_id=existing_node_id)
                repeated_deadend = bool(verdict.is_dead_end and verdict.is_repeated)
                node_manager.maybe_create_or_update_node(
                    next_pose_mission, now_step=self.global_step, subgoal_reached=(
                        result is not None and result.status == SubgoalStatus.REACHED
                    ),
                    subgoal_failed=subgoal_failed, dead_end_detected=verdict.is_dead_end,
                    free_direction_count=known_free_direction_count, final_goal_distance_m=end_final_goal_distance,
                    path_length_since_last_m=(result.path_length_m if result is not None else 0.0),
                    elapsed_time_since_last_sec=(result.elapsed_time_sec if result is not None else 0.0),
                    risk=(result.mean_predicted_risk if result is not None else None),
                )
            else:
                repeated_deadend = bool(
                    result is not None and result.status == SubgoalStatus.FAILED_BLOCKED
                    and _endpoint_failure_count(partial_map, endpoint_mission) > 1
                )

            mission_reached = bool(coordinator.mission_reached)
            budget_exhausted = option_index == max_options - 1
            timed_out = bool(mission_timeout_steps is not None and cumulative_local_steps >= mission_timeout_steps)
            mission_done_this_option = mission_reached or budget_exhausted or timed_out

            breakdown = compute_global_reward(
                self.global_cfg, mission_reached=mission_reached,
                start_final_goal_distance_m=start_final_goal_distance,
                end_final_goal_distance_m=end_final_goal_distance,
                newly_explored_cells=newly_explored, revisit_amount=0.0, repeated_deadend=repeated_deadend,
                subgoal_status=(result.status if result is not None else None),
                risk_integral=risk_integral, local_steps=local_steps,
                predicted_risk_at_selection=predicted_risk_at_selection,
            )

            next_elapsed_ratio = (option_index + 1) / max_options
            next_obs = self._build_observation(
                partial_map, next_pose_mission, goal_mission, speed_mps=0.0, previous_action_index=action,
                elapsed_ratio=next_elapsed_ratio, graph=graph,
            )
            subgoal_reached = result is not None and result.status == SubgoalStatus.REACHED
            transition = GlobalTransition.from_observations(
                obs, action, breakdown.total, next_obs, mission_done=mission_done_this_option,
                subgoal_success=subgoal_reached,
                failure_reason=(None if subgoal_reached else (result.status if result is not None else None)),
                local_steps=local_steps, exploration_gain=newly_explored, risk_integral=risk_integral,
            )
            # plan section 8.8's guard, enforced at BOTH layers: this loop
            # only ever calls add() for a mode="train" loop instance (a
            # validation/test-mode HierarchicalTrainingLoop -- e.g. periodic
            # eval rollouts -- never persists into the Global replay buffer
            # at all), and GlobalReplayBuffer.add() itself independently
            # rejects anything other than mode="train" as a second,
            # API-level backstop.
            if self.mode == "train":
                self.replay.add(transition, mode="train")
                stored += 1
            self.global_step += 1
            previous_action_index = action

            if mission_done_this_option:
                break

        return MissionOutcome(
            global_transitions=stored, mission_reached=mission_reached,
            mission_failed=(not mission_reached and not timed_out), mission_timed_out=timed_out,
        )

    def state_dict(self) -> dict:
        """Resumable loop-level state -- the Phase 3 seed-pool scheduler's
        own ``state_dict()`` (never a bare re-seed, mirrors
        ``TrainerBase``'s local ``seed_scheduler``/``validation_seed_scheduler``
        resume contract) plus this loop's own counters."""
        return {
            "seed_scheduler": self.seed_scheduler.state_dict(),
            "global_step": self.global_step,
            "mission_index": self.mission_index,
        }

    def load_state_dict(self, state: dict) -> None:
        self.seed_scheduler = LongHorizonSeedScheduler.from_state_dict(state["seed_scheduler"], self.long_horizon_cfg)
        self.global_step = state.get("global_step", 0)
        self.mission_index = state.get("mission_index", 0)

    def checkpoint_meta(self) -> dict:
        """Fields every Global checkpoint's manifest should embed -- notably
        the frozen local checkpoint's sha256 (plan requirement: "Local
        checkpoint hash가 Global manifest에 기록됨"). ``None`` when
        ``local_checkpoint_dir`` is unset or the resolved
        ``<local_checkpoint_dir>/<local_checkpoint_name>/model.pt`` does not
        exist (never a fabricated hash) -- this is the SAME (directory, tag)
        resolution ``rl.checkpointing.manager.load_generation`` uses (the
        tag is a symlink into ``.generations/<uuid>/model.pt``), so the
        hashed file is always the exact one a live loader would actually
        read."""
        local_checkpoint_sha256 = None
        checkpoint_file = resolve_local_checkpoint_file(
            self.training_cfg.local_checkpoint_dir, self.training_cfg.local_checkpoint_name,
        )
        if checkpoint_file and os.path.isfile(checkpoint_file):
            from hunter_kinodynamic_rl.rl.checkpointing.manager import sha256_of_file
            local_checkpoint_sha256 = sha256_of_file(checkpoint_file)
        return {
            "local_checkpoint_dir": self.training_cfg.local_checkpoint_dir,
            "local_checkpoint_name": self.training_cfg.local_checkpoint_name,
            "local_checkpoint_file": checkpoint_file,
            "local_checkpoint_sha256": local_checkpoint_sha256,
            # Phase 5: the ACTUAL resolved observation schema this
            # checkpoint's replay/network were built against -- lets a later
            # reader (evaluation, ablation comparison, checkpoint-
            # compatibility test) reconstruct exactly what shape this
            # Global checkpoint expects without re-deriving it from the
            # profile's raw ablation flags.
            "map_channel_names": list(self.map_channel_names),
            "candidate_feature_names": list(self.candidate_feature_names),
            "node_feature_names": list(NODE_FEATURE_NAMES) if self.memory_enabled else [],
            "max_nodes_in_observation": self.max_nodes,
            "memory_enabled": self.memory_enabled,
            "feasibility_enabled": self.feasibility_enabled,
            "global_risk_enabled": self.global_risk_enabled,
            "long_horizon_level": self.training_cfg.long_horizon_level,
            "mode": self.mode,
            "global_step": self.global_step,
            "mission_index": self.mission_index,
        }


def _endpoint_failure_count(partial_map: PartialMap, endpoint_mission: Tuple[float, float]) -> int:
    return partial_map.failure_count_at(*endpoint_mission)
