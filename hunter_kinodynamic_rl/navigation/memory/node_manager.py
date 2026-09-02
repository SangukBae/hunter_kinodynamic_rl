"""Event-driven topological node creation (plan section 9.3) -- decides
WHEN a new :class:`~hunter_kinodynamic_rl.navigation.memory.topological_graph.TopoNode`
is worth creating and dedups against an existing one within
``node_merge_radius_m`` (plan requirement: "node/edge 중복 생성 방지") rather
than ever creating two nodes for the same physical spot.

Node-creation triggers (any one fires a node event):
  - distance since the last node-worthy pose >= ``node_min_distance_m``
  - heading change since then >= ``node_heading_change_rad``
  - caller-supplied ``junction_detected`` / ``subgoal_reached`` /
    ``subgoal_failed`` / ``dead_end_detected``
  - the very first call ever (always creates/attaches the START node)

Deliberately does NOT decide first-vs-repeated dead-end itself -- that
classification needs the graph's state BEFORE this call's own mutation
(see ``dead_end_detector.py``'s caller contract), so callers run
:meth:`~hunter_kinodynamic_rl.navigation.memory.dead_end_detector.DeadEndDetector.verdict`
first and pass its ``is_dead_end`` boolean in as ``dead_end_detected`` --
this manager unconditionally marks the resulting node dead-end when told,
nothing more.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from hunter_kinodynamic_rl.common.geometry import euclidean_distance, wrap_to_pi
from hunter_kinodynamic_rl.config.schema import ConfigError
from hunter_kinodynamic_rl.navigation.global_rl.subgoal_sampler import SubgoalCandidate, candidate_endpoint_mission
from hunter_kinodynamic_rl.navigation.memory.route_history import RouteEvent, RouteHistory
from hunter_kinodynamic_rl.navigation.memory.topological_graph import TopoNodeType, TopologicalGraph
from hunter_kinodynamic_rl.navigation.mission.mission_frame import PoseXYYaw

_VISIT_COUNT_NORM = 50.0


@dataclass(frozen=True)
class NodeManagerConfig:
    node_min_distance_m: float = 2.0
    node_heading_change_rad: float = math.pi / 3.0
    node_merge_radius_m: float = 1.0

    def validate(self) -> None:
        if self.node_min_distance_m <= 0.0:
            raise ConfigError("node_manager.node_min_distance_m must be > 0")
        if not (0.0 < self.node_heading_change_rad <= math.pi):
            raise ConfigError("node_manager.node_heading_change_rad must be in (0, pi]")
        if self.node_merge_radius_m <= 0.0:
            raise ConfigError("node_manager.node_merge_radius_m must be > 0")
        if self.node_merge_radius_m >= self.node_min_distance_m:
            raise ConfigError(
                "node_manager.node_merge_radius_m must be < node_min_distance_m, otherwise every "
                "distance-triggered node would immediately merge back into the previous one"
            )


@dataclass(frozen=True)
class NodeEvent:
    node_id: int
    created: bool
    route_event: RouteEvent
    edge_updated: bool


class TopologicalNodeManager:
    def __init__(self, graph: TopologicalGraph, route_history: RouteHistory, config: NodeManagerConfig) -> None:
        config.validate()
        self.graph = graph
        self.route_history = route_history
        self.config = config
        self._last_trigger_pose: Optional[PoseXYYaw] = None

    def reset(self, *, clear_graph: bool = False) -> None:
        """Re-arms for a fresh mission. ``clear_graph=False`` (default)
        keeps accumulated topology across missions (e.g. repeated episodes
        in the SAME unknown world within one training run); pass
        ``clear_graph=True`` when a brand-new/unrelated world starts (mirrors
        ``GoalManager``'s own ``reset_memory_on_goal_change`` opt-in)."""
        self._last_trigger_pose = None
        self.route_history.reset()
        if clear_graph:
            self.graph = TopologicalGraph(max_nodes=self.graph.max_nodes)

    def _should_trigger(
        self, robot_pose_mission: PoseXYYaw, *, junction_detected: bool, subgoal_reached: bool,
        subgoal_failed: bool, dead_end_detected: bool,
    ) -> bool:
        if junction_detected or subgoal_reached or subgoal_failed or dead_end_detected:
            return True
        if self._last_trigger_pose is None:
            return True
        dist = euclidean_distance(
            self._last_trigger_pose.x, self._last_trigger_pose.y, robot_pose_mission.x, robot_pose_mission.y,
        )
        if dist >= self.config.node_min_distance_m:
            return True
        heading_delta = abs(wrap_to_pi(robot_pose_mission.yaw - self._last_trigger_pose.yaw))
        return heading_delta >= self.config.node_heading_change_rad

    def maybe_create_or_update_node(
        self, robot_pose_mission: PoseXYYaw, *, now_step: int,
        node_type_hint: Optional[TopoNodeType] = None, junction_detected: bool = False,
        subgoal_reached: bool = False, subgoal_failed: bool = False, dead_end_detected: bool = False,
        free_direction_count: int = 0, final_goal_distance_m: float = 0.0, final_goal_bearing_rad: float = 0.0,
        path_length_since_last_m: float = 0.0, elapsed_time_since_last_sec: float = 0.0,
        risk: Optional[float] = None,
    ) -> Optional[NodeEvent]:
        """Returns ``None`` when no trigger fired this tick (too close, no
        event) -- the common case at 10-20 Hz local control rate, so most
        ticks are cheap no-ops. When a node event DOES fire, dedups against
        an existing node within ``node_merge_radius_m`` (updates its visit
        count / dead-end flag / risk sample instead of creating a
        duplicate); otherwise creates a new node. Either way records the
        arrival into ``route_history`` and updates the edge from whatever
        node preceded it (skipped on the very first node of a mission, when
        there is no predecessor yet)."""
        if not self._should_trigger(
            robot_pose_mission, junction_detected=junction_detected, subgoal_reached=subgoal_reached,
            subgoal_failed=subgoal_failed, dead_end_detected=dead_end_detected,
        ):
            return None

        existing = self.graph.find_nearest_node(
            robot_pose_mission.x, robot_pose_mission.y, max_radius_m=self.config.node_merge_radius_m,
        )
        created = existing is None
        if created:
            if node_type_hint is not None:
                node_type = node_type_hint
            elif dead_end_detected:
                node_type = TopoNodeType.DEAD_END
            elif junction_detected:
                node_type = TopoNodeType.JUNCTION
            else:
                node_type = TopoNodeType.WAYPOINT
            node_id = self.graph.add_node(
                node_type, robot_pose_mission.x, robot_pose_mission.y, now_step=now_step,
                free_direction_count=free_direction_count, dead_end=dead_end_detected,
                final_goal_distance_m=final_goal_distance_m, final_goal_bearing_rad=final_goal_bearing_rad,
            )
            # A newly-created node exists BECAUSE the robot just arrived
            # here -- record that arrival as its first visit (matching the
            # merge branch below, which always does) rather than leaving
            # visit_count at TopoNode's bare default of 0. Without this, a
            # brand-new node under-reports its own visit/branch statistics
            # (topology candidate features, dead-end evidence) relative to
            # an otherwise-identical merged node.
            self.graph.record_visit(node_id, now_step=now_step)
        else:
            node_id = existing
            self.graph.record_visit(node_id, now_step=now_step)
            if dead_end_detected:
                self.graph.mark_dead_end(node_id, True)

        if risk is not None and math.isfinite(risk):
            self.graph.get_node(node_id).risk_samples.append(float(risk))
        if subgoal_failed:
            self.graph.record_failure(node_id)

        previous_node_id = self.route_history.current_node_id
        route_event = self.route_history.record_arrival(node_id, now_step=now_step)
        edge_updated = False
        if previous_node_id is not None and previous_node_id != node_id:
            self.graph.add_or_update_edge(
                previous_node_id, node_id, path_length_m=path_length_since_last_m,
                elapsed_time_sec=elapsed_time_since_last_sec, risk=risk, success=not subgoal_failed,
                blocked=bool(subgoal_failed and dead_end_detected),
                direction_rad=math.atan2(
                    robot_pose_mission.y - self.graph.get_node(previous_node_id).y_mission,
                    robot_pose_mission.x - self.graph.get_node(previous_node_id).x_mission,
                ),
            )
            edge_updated = True

        self._last_trigger_pose = robot_pose_mission
        return NodeEvent(node_id=node_id, created=created, route_event=route_event, edge_updated=edge_updated)


def compute_candidate_topology_features(
    candidates: Sequence[SubgoalCandidate], graph: TopologicalGraph, robot_pose_mission: PoseXYYaw,
    merge_radius_m: float,
) -> np.ndarray:
    """``(n_candidates, 2)`` float32 -- ``(repeated_deadend_flag,
    branch_visit_count_norm)`` per candidate (plan 9.9's topology-feedback
    candidate columns, see ``global_rl.observation.TOPOLOGY_CANDIDATE_FEATURE_NAMES``).
    A candidate whose endpoint has no known node within ``merge_radius_m``
    (unexplored territory) reports ``(0.0, 0.0)`` -- exploring somewhere new
    is never penalized by this feature, only RE-entering a place already on
    the graph is."""
    n = len(candidates)
    out = np.zeros((n, 2), dtype=np.float32)
    for candidate in candidates:
        idx = candidate.index
        if candidate.is_fallback:
            continue
        endpoint = candidate_endpoint_mission(candidate, robot_pose_mission)
        node_id = graph.find_nearest_node(endpoint[0], endpoint[1], max_radius_m=merge_radius_m)
        if node_id is None:
            continue
        node = graph.get_node(node_id)
        out[idx, 0] = 1.0 if node.dead_end else 0.0
        out[idx, 1] = float(np.clip(node.visit_count / _VISIT_COUNT_NORM, 0.0, 1.0))
    return out
