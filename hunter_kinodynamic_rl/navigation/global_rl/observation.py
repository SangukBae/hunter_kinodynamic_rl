"""Global observation assembly (plan section 8.5, extended by Phase 5's
9.5/9.6/9.9 topology/feasibility/global-risk ablation feedback).

Fixed-shape contract (never varies across calls for the SAME
:class:`~hunter_kinodynamic_rl.config.schema.GlobalRLConfig`):

- ``map_tensor``: ``(len(resolve_map_channel_names(config)), size_cells, size_cells)``
  float32 -- occupied/free/unknown/goal-direction (a rolling crop
  re-centered on the robot every call) always present, plus a ``visited``
  channel when ``config.include_visited_channel`` (default True --
  requirement F's ablation-B-vs-C toggle, see
  :func:`resolve_map_channel_names`), plus, when
  ``config.include_failure_channel``, a ``failure``-count raster channel
  (plan 9.5 step 1: "visited/failure raster channel만 사용").
- ``scalar_tensor``: ``(N_SCALARS,)`` float32 -- unchanged from Phase 4.
- ``candidate_tensor``: ``(n_candidates, len(resolve_candidate_feature_names(config)))``
  float32 -- the Phase 4 base 4 features, plus (each independently, per
  ablation flag) 2 topology columns, 6 feasibility columns, 1 global-risk
  column. A caller must supply the corresponding ``*_candidate_features``
  array whenever its flag is on (see :func:`build_global_observation`'s own
  docstring) -- this function never silently zero-fills a flag it was told
  is enabled.
- ``action_mask``: ``(n_candidates,)`` bool -- unchanged from Phase 4.
- ``node_tensor``/``node_validity_mask``: ``(max_nodes, N_NODE_FEATURES)``/
  ``(max_nodes,)`` -- the Phase 5 topological-memory node summary (plan 9.5
  steps 2-3: fixed-length node tensor + validity mask, no GNN yet). Present
  but EMPTY (``max_nodes=0``) whenever ``config.topology_feedback_enabled``
  is False -- every consumer (network/replay) can treat these fields
  uniformly instead of handling ``None``.

Every Phase 5 flag defaults to False, so a profile that sets none of them
reproduces Phase 4's exact ``map_tensor``/``candidate_tensor`` shape and
values byte-for-byte -- this module has exactly ONE observation-building
code path, ablations B-G differ only in which flags a profile sets and
which extra feature arrays its caller computes and passes in.

Deliberately reads ONLY the online :class:`PartialMap` + the mission-frame
final goal/pose (+ whatever pre-computed feature arrays the caller passes
in, themselves built from the SAME online, non-privileged sources) -- never
any Phase 3 :class:`LongHorizonWorld` ground truth, matching the plan's
information-boundary requirement (section 3.2/8.5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from hunter_kinodynamic_rl.common.geometry import goal_distance_and_heading
from hunter_kinodynamic_rl.config.schema import GlobalRLConfig
from hunter_kinodynamic_rl.navigation.global_rl.subgoal_sampler import SubgoalCandidate, candidate_endpoint_mission
from hunter_kinodynamic_rl.navigation.mapping.partial_map import PartialMap
from hunter_kinodynamic_rl.navigation.mapping.rolling_map import crop_rolling
from hunter_kinodynamic_rl.navigation.memory.topological_graph import N_NODE_FEATURES
from hunter_kinodynamic_rl.navigation.mission.mission_frame import PoseXYYaw

#: Default (``include_visited_channel=True``, every pre-existing profile's
#: implicit value) channel order -- kept as the module-level constant so
#: any caller that still imports it directly (fixed-shape assumptions
#: written before requirement F) keeps seeing the Phase-4-compatible
#: 5-channel tuple; `resolve_map_channel_names(config)` is the actual
#: source of truth for a GIVEN config's channel set/order.
MAP_CHANNEL_NAMES = ("occupied", "free", "unknown", "visited", "goal_direction")
N_MAP_CHANNELS = len(MAP_CHANNEL_NAMES)
VISITED_CHANNEL_NAME = "visited"
FAILURE_CHANNEL_NAME = "failure"

SCALAR_NAMES = (
    "final_goal_distance_norm", "final_goal_bearing_norm", "speed_norm",
    "previous_action_norm", "elapsed_mission_ratio",
)
N_SCALARS = len(SCALAR_NAMES)

CANDIDATE_FEATURE_NAMES = ("action_mask", "known_free_ratio", "unknown_gain_estimate", "curvature_difficulty")
N_CANDIDATE_FEATURES = len(CANDIDATE_FEATURE_NAMES)

#: Phase 5 candidate-feature extensions (plan 9.5/9.6/9.9) -- each block is
#: appended to the base ``CANDIDATE_FEATURE_NAMES`` only when its ablation
#: flag is set (see :func:`resolve_candidate_feature_names`).
TOPOLOGY_CANDIDATE_FEATURE_NAMES = ("repeated_deadend_flag", "branch_visit_count_norm")
N_TOPOLOGY_CANDIDATE_FEATURES = len(TOPOLOGY_CANDIDATE_FEATURE_NAMES)
GLOBAL_RISK_CANDIDATE_FEATURE_NAMES = ("global_risk_score",)
N_GLOBAL_RISK_CANDIDATE_FEATURES = len(GLOBAL_RISK_CANDIDATE_FEATURE_NAMES)


def resolve_map_channel_names(config: GlobalRLConfig) -> Tuple[str, ...]:
    """Requirement F: ``visited`` is now gated by
    ``config.include_visited_channel`` (default True == byte-identical to
    the old unconditional ``MAP_CHANNEL_NAMES`` order) instead of always
    present -- this is what lets ablation B (False) and C (True) differ."""
    names = ["occupied", "free", "unknown"]
    if config.include_visited_channel:
        names.append(VISITED_CHANNEL_NAME)
    names.append("goal_direction")
    if config.include_failure_channel:
        names.append(FAILURE_CHANNEL_NAME)
    return tuple(names)


def resolve_candidate_feature_names(config: GlobalRLConfig) -> Tuple[str, ...]:
    # Local import -- avoids a module-import-time cycle risk (this module
    # is imported very early by replay/networks/agent; hierarchy.feasibility
    # itself never imports observation.py, but keeping the import here
    # keeps the dependency direction obvious at the call site).
    from hunter_kinodynamic_rl.navigation.hierarchy.feasibility import FEASIBILITY_FEATURE_NAMES

    names = list(CANDIDATE_FEATURE_NAMES)
    if config.topology_feedback_enabled:
        names.extend(TOPOLOGY_CANDIDATE_FEATURE_NAMES)
    if config.feasibility_feedback_enabled:
        names.extend(FEASIBILITY_FEATURE_NAMES)
    if config.global_risk_feedback_enabled:
        names.extend(GLOBAL_RISK_CANDIDATE_FEATURE_NAMES)
    return tuple(names)


@dataclass(frozen=True)
class GlobalObservation:
    map_tensor: np.ndarray
    scalar_tensor: np.ndarray
    candidate_tensor: np.ndarray
    action_mask: np.ndarray
    node_tensor: np.ndarray = None  # type: ignore[assignment]
    node_validity_mask: np.ndarray = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        # Dataclass defaults can't reference N_NODE_FEATURES at class-body
        # evaluation time cleanly across both this module's own construction
        # sites and replay.py's -- normalize None -> the empty (0-node)
        # shape here, once, so every field is ALWAYS a real ndarray.
        if self.node_tensor is None:
            object.__setattr__(self, "node_tensor", np.zeros((0, N_NODE_FEATURES), dtype=np.float32))
        if self.node_validity_mask is None:
            object.__setattr__(self, "node_validity_mask", np.zeros((0,), dtype=bool))


def _goal_direction_channel(
    size_cells: int, resolution_m: float, crop_origin_x: float, crop_origin_y: float,
    robot_pose_mission: PoseXYYaw, final_goal_mission: Tuple[float, float],
) -> np.ndarray:
    """``(cos(angle_from_robot_to_cell - angle_from_robot_to_goal) + 1) / 2``
    -- cells roughly toward the final goal read near 1, cells away read near
    0, everywhere in the crop, independent of whether the goal itself lies
    inside the crop window."""
    gx, gy = final_goal_mission
    goal_angle = math.atan2(gy - robot_pose_mission.y, gx - robot_pose_mission.x)
    cols = np.arange(size_cells)
    rows = np.arange(size_cells)
    cell_x = crop_origin_x + (cols + 0.5) * resolution_m
    cell_y = crop_origin_y + (rows + 0.5) * resolution_m
    xx, yy = np.meshgrid(cell_x, cell_y)  # xx[row, col] = cell_x[col], yy[row, col] = cell_y[row]
    cell_angle = np.arctan2(yy - robot_pose_mission.y, xx - robot_pose_mission.x)
    aligned = np.cos(cell_angle - goal_angle)
    return ((aligned + 1.0) * 0.5).astype(np.float32)


def build_global_observation(
    partial_map: PartialMap, robot_pose_mission: PoseXYYaw, final_goal_mission: Tuple[float, float],
    candidates: Sequence[SubgoalCandidate], action_mask: np.ndarray, *,
    speed_mps: float, previous_action_index: int, elapsed_mission_ratio: float, config: GlobalRLConfig,
    topology_candidate_features: Optional[np.ndarray] = None,
    feasibility_candidate_features: Optional[np.ndarray] = None,
    global_risk_candidate_features: Optional[np.ndarray] = None,
    node_tensor: Optional[np.ndarray] = None,
    node_validity_mask: Optional[np.ndarray] = None,
) -> GlobalObservation:
    """The ``*_candidate_features``/``node_tensor``/``node_validity_mask``
    keyword arguments are REQUIRED (raise ``ValueError`` if omitted) exactly
    when the corresponding ``config`` flag is on, and IGNORED (never even
    read) when it is off -- this function fails fast rather than silently
    zero-filling a feature a profile explicitly asked for, but never
    demands an array a profile never asked for either."""
    crop = crop_rolling(partial_map, (robot_pose_mission.x, robot_pose_mission.y), config.map_crop_size_cells)
    goal_direction = _goal_direction_channel(
        crop.size_cells, crop.resolution_m, crop.origin_x, crop.origin_y, robot_pose_mission, final_goal_mission,
    )
    map_channels = [
        crop.occupied.astype(np.float32), crop.free.astype(np.float32), crop.unknown.astype(np.float32),
    ]
    if config.include_visited_channel:
        map_channels.append(crop.visited.astype(np.float32))
    map_channels.append(goal_direction)
    if config.include_failure_channel:
        # crop.failure is already normalize_counts()-mapped into [0, 1]
        # (same convention as crop.visited) -- see PartialMap.channels().
        map_channels.append(crop.failure.astype(np.float32))
    map_tensor = np.stack(map_channels, axis=0)

    dist_m, bearing_rad = goal_distance_and_heading(
        robot_pose_mission.x, robot_pose_mission.y, robot_pose_mission.yaw,
        final_goal_mission[0], final_goal_mission[1],
    )
    n_candidates = len(candidates)
    prev_action_norm = 0.0
    if n_candidates > 1:
        prev_action_norm = float(np.clip(2.0 * previous_action_index / (n_candidates - 1) - 1.0, -1.0, 1.0))
    scalar_tensor = np.array([
        float(np.clip(dist_m / config.goal_distance_norm_m, 0.0, 1.0)),
        float(np.clip(bearing_rad / math.pi, -1.0, 1.0)),
        float(np.clip(speed_mps / config.max_speed_norm_mps, -1.0, 1.0)),
        prev_action_norm,
        float(np.clip(elapsed_mission_ratio, 0.0, 1.0)),
    ], dtype=np.float32)

    channels = partial_map.channels()
    non_fallback_angles = [abs(c.angle_rad) for c in candidates if not c.is_fallback]
    max_abs_angle = max(non_fallback_angles) if non_fallback_angles else math.pi
    max_abs_angle = max_abs_angle or math.pi

    base_tensor = np.zeros((n_candidates, N_CANDIDATE_FEATURES), dtype=np.float32)
    for candidate in candidates:
        idx = candidate.index
        base_tensor[idx, 0] = 1.0 if action_mask[idx] else 0.0
        if candidate.is_fallback:
            base_tensor[idx, 1] = 0.0
            base_tensor[idx, 2] = 0.0
            base_tensor[idx, 3] = 1.0
            continue
        endpoint = candidate_endpoint_mission(candidate, robot_pose_mission)
        free_hits = unknown_hits = samples = 0
        for t in np.linspace(0.0, 1.0, config.rollout_sample_count):
            x = robot_pose_mission.x + t * (endpoint[0] - robot_pose_mission.x)
            y = robot_pose_mission.y + t * (endpoint[1] - robot_pose_mission.y)
            samples += 1
            cell = partial_map.world_to_cell(x, y)
            if cell is None:
                unknown_hits += 1
                continue
            if channels.free[cell]:
                free_hits += 1
            elif channels.unknown[cell]:
                unknown_hits += 1
        base_tensor[idx, 1] = (free_hits / samples) if samples else 0.0
        base_tensor[idx, 2] = (unknown_hits / samples) if samples else 0.0
        base_tensor[idx, 3] = abs(candidate.angle_rad) / max_abs_angle

    extra_blocks = [base_tensor]
    if config.topology_feedback_enabled:
        if topology_candidate_features is None:
            raise ValueError(
                "build_global_observation: config.topology_feedback_enabled=true requires "
                "topology_candidate_features"
            )
        block = np.asarray(topology_candidate_features, dtype=np.float32)
        if block.shape != (n_candidates, N_TOPOLOGY_CANDIDATE_FEATURES):
            raise ValueError(
                f"topology_candidate_features shape {block.shape} != "
                f"{(n_candidates, N_TOPOLOGY_CANDIDATE_FEATURES)}"
            )
        extra_blocks.append(block)
    if config.feasibility_feedback_enabled:
        if feasibility_candidate_features is None:
            raise ValueError(
                "build_global_observation: config.feasibility_feedback_enabled=true requires "
                "feasibility_candidate_features"
            )
        from hunter_kinodynamic_rl.navigation.hierarchy.feasibility import N_FEASIBILITY_FEATURES
        block = np.asarray(feasibility_candidate_features, dtype=np.float32)
        if block.shape != (n_candidates, N_FEASIBILITY_FEATURES):
            raise ValueError(
                f"feasibility_candidate_features shape {block.shape} != {(n_candidates, N_FEASIBILITY_FEATURES)}"
            )
        extra_blocks.append(block)
    if config.global_risk_feedback_enabled:
        if global_risk_candidate_features is None:
            raise ValueError(
                "build_global_observation: config.global_risk_feedback_enabled=true requires "
                "global_risk_candidate_features"
            )
        block = np.asarray(global_risk_candidate_features, dtype=np.float32)
        if block.shape != (n_candidates, N_GLOBAL_RISK_CANDIDATE_FEATURES):
            raise ValueError(
                f"global_risk_candidate_features shape {block.shape} != "
                f"{(n_candidates, N_GLOBAL_RISK_CANDIDATE_FEATURES)}"
            )
        extra_blocks.append(block)
    candidate_tensor = np.concatenate(extra_blocks, axis=1) if len(extra_blocks) > 1 else base_tensor

    if config.topology_feedback_enabled:
        if node_tensor is None or node_validity_mask is None:
            raise ValueError(
                "build_global_observation: config.topology_feedback_enabled=true requires "
                "node_tensor/node_validity_mask"
            )
        node_tensor_out = np.asarray(node_tensor, dtype=np.float32)
        node_validity_out = np.asarray(node_validity_mask, dtype=bool)
    else:
        node_tensor_out = np.zeros((0, N_NODE_FEATURES), dtype=np.float32)
        node_validity_out = np.zeros((0,), dtype=bool)

    return GlobalObservation(
        map_tensor=map_tensor, scalar_tensor=scalar_tensor, candidate_tensor=candidate_tensor,
        action_mask=np.asarray(action_mask, dtype=bool), node_tensor=node_tensor_out,
        node_validity_mask=node_validity_out,
    )
