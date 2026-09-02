"""Compact route-history graph (plan section 9.3) -- node/edge memory of
WHERE the robot has been and what happened there, kept small enough to
carry as a fixed-length observation tensor (plan 9.5) instead of the full
mission :class:`~hunter_kinodynamic_rl.navigation.mapping.partial_map.PartialMap`
raster.

Never reads simulator ground truth: every node/edge this module creates is
built exclusively from mission-frame poses and caller-supplied evidence
(visit/failure counts, risk samples) that themselves come from the online
:class:`PartialMap`/:class:`~hunter_kinodynamic_rl.navigation.hierarchy.subgoal_manager.SubgoalResult`
pipeline, not from :class:`~hunter_kinodynamic_rl.env.scenarios.long_horizon_world.LongHorizonWorld`.

Node IDs are stable, monotonically increasing integers -- :meth:`update_node_pose`
is a SEPARATE API from node creation specifically so a future loop-closure /
pose-graph correction can move an existing node without fabricating a new
id (plan 9.3: "loop closure 또는 localization pose correction을 고려해 node
ID와 pose를 분리"). Edges are stored UNDIRECTED (a corridor is traversable
both ways once discovered) but keep the most recent traversal DIRECTION as
a scalar feature.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


class TopoNodeType(enum.Enum):
    START = "start"
    JUNCTION = "junction"
    ROOM_ENTRANCE = "room_entrance"
    INTERSECTION = "intersection"
    DEAD_END = "dead_end"
    WAYPOINT = "waypoint"


#: Fixed column contract for :meth:`TopologicalGraph.to_fixed_tensor` -- the
#: SHAPE/ORDER every Global-observation consumer indexes into (mirrors
#: ``global_rl.observation.MAP_CHANNEL_NAMES``'s own contract role). All
#: node-pose-derived entries are computed RELATIVE to the robot's pose at
#: call time (never raw mission-frame coordinates) so the tensor stays
#: translation-invariant the same way the candidate/map tensors already are.
NODE_FEATURE_NAMES = (
    "relative_dx_norm", "relative_dy_norm", "visit_count_norm", "free_direction_count_norm",
    "dead_end_flag", "failure_count_norm", "mean_risk", "final_goal_distance_norm",
    "final_goal_bearing_norm", "recency_norm",
)
N_NODE_FEATURES = len(NODE_FEATURE_NAMES)

_VISIT_COUNT_NORM = 50.0
_FAILURE_COUNT_NORM = 20.0
_FREE_DIRECTION_NORM = 8.0


@dataclass
class TopoNode:
    node_id: int
    node_type: TopoNodeType
    x_mission: float
    y_mission: float
    visit_count: int = 0
    free_direction_count: int = 0
    dead_end: bool = False
    failure_count: int = 0
    risk_samples: List[float] = field(default_factory=list)
    final_goal_distance_m: float = 0.0
    final_goal_bearing_rad: float = 0.0
    last_visited_step: int = 0
    created_step: int = 0

    @property
    def mean_risk(self) -> float:
        return (sum(self.risk_samples) / len(self.risk_samples)) if self.risk_samples else 0.0

    @property
    def max_risk(self) -> float:
        return max(self.risk_samples) if self.risk_samples else 0.0


@dataclass
class TopoEdge:
    from_id: int
    to_id: int
    traversal_count: int = 0
    path_length_m: float = 0.0
    elapsed_time_sec: float = 0.0
    mean_risk: float = 0.0
    max_risk: float = 0.0
    success_count: int = 0
    failure_count: int = 0
    last_direction_rad: float = 0.0
    blocked: bool = False


class TopologicalGraph:
    """``max_nodes=None`` (default) never rejects a new node -- capacity
    enforcement (if any) is the caller's (``node_manager.py``'s) job, since
    only IT knows whether evicting the oldest/least-relevant node is safe
    for a given mission. Passing an explicit ``max_nodes`` here makes
    :meth:`add_node` raise once exceeded, for callers that want a hard
    guarantee instead."""

    def __init__(self, max_nodes: Optional[int] = None) -> None:
        self.max_nodes = max_nodes
        self._nodes: Dict[int, TopoNode] = {}
        self._edges: Dict[Tuple[int, int], TopoEdge] = {}
        self._adjacency: Dict[int, set] = {}
        self._next_node_id = 0

    # ------------------------------------------------------------ nodes
    def add_node(
        self, node_type: TopoNodeType, x_mission: float, y_mission: float, *,
        now_step: int = 0, free_direction_count: int = 0, dead_end: bool = False,
        final_goal_distance_m: float = 0.0, final_goal_bearing_rad: float = 0.0,
    ) -> int:
        if self.max_nodes is not None and len(self._nodes) >= self.max_nodes:
            raise RuntimeError(
                f"TopologicalGraph.add_node(): capacity max_nodes={self.max_nodes} already reached"
            )
        node_id = self._next_node_id
        self._next_node_id += 1
        self._nodes[node_id] = TopoNode(
            node_id=node_id, node_type=node_type, x_mission=float(x_mission), y_mission=float(y_mission),
            free_direction_count=int(free_direction_count), dead_end=bool(dead_end),
            final_goal_distance_m=float(final_goal_distance_m), final_goal_bearing_rad=float(final_goal_bearing_rad),
            last_visited_step=int(now_step), created_step=int(now_step),
        )
        self._adjacency[node_id] = set()
        return node_id

    def get_node(self, node_id: int) -> TopoNode:
        return self._nodes[node_id]

    def has_node(self, node_id: int) -> bool:
        return node_id in self._nodes

    def update_node_pose(self, node_id: int, x_mission: float, y_mission: float) -> None:
        """Moves an EXISTING node (loop-closure / pose-graph correction,
        plan 9.3) -- never allocates a new id, never touches any other
        field (visit/failure counts, edges survive untouched)."""
        node = self._nodes[node_id]
        node.x_mission = float(x_mission)
        node.y_mission = float(y_mission)

    def record_visit(self, node_id: int, *, now_step: int) -> None:
        node = self._nodes[node_id]
        node.visit_count += 1
        node.last_visited_step = int(now_step)

    def record_failure(self, node_id: int, risk: Optional[float] = None) -> None:
        node = self._nodes[node_id]
        node.failure_count += 1
        if risk is not None and math.isfinite(risk):
            node.risk_samples.append(float(risk))

    def mark_dead_end(self, node_id: int, dead_end: bool = True) -> None:
        self._nodes[node_id].dead_end = bool(dead_end)

    def find_nearest_node(
        self, x_mission: float, y_mission: float, max_radius_m: Optional[float] = None,
    ) -> Optional[int]:
        """Nearest-by-Euclidean-distance node within ``max_radius_m`` (or
        unconditionally if ``None``) -- the dedup lookup ``node_manager.py``
        uses to decide "merge into an existing node" vs. "create a new
        one" (plan requirement: "node/edge 중복 생성 방지")."""
        best_id: Optional[int] = None
        best_dist: Optional[float] = None
        for node_id, node in self._nodes.items():
            dist = math.hypot(node.x_mission - x_mission, node.y_mission - y_mission)
            if max_radius_m is not None and dist > max_radius_m:
                continue
            if best_dist is None or dist < best_dist:
                best_id, best_dist = node_id, dist
        return best_id

    @property
    def nodes(self) -> Dict[int, TopoNode]:
        return dict(self._nodes)

    def __len__(self) -> int:
        return len(self._nodes)

    # ------------------------------------------------------------ edges
    @staticmethod
    def _edge_key(a: int, b: int) -> Tuple[int, int]:
        return (a, b) if a <= b else (b, a)

    def add_or_update_edge(
        self, from_id: int, to_id: int, *, path_length_m: float = 0.0, elapsed_time_sec: float = 0.0,
        risk: Optional[float] = None, success: bool = True, blocked: bool = False, direction_rad: float = 0.0,
    ) -> None:
        """Undirected: traversing an existing edge in either direction
        updates the SAME edge record (running mean for length/time/risk
        over all traversals, running max for ``max_risk``, separate
        success/failure counters). A self-loop (``from_id == to_id``) is
        silently ignored -- it carries no connectivity information."""
        if from_id == to_id:
            return
        if from_id not in self._nodes or to_id not in self._nodes:
            raise KeyError(f"add_or_update_edge: both endpoints must already be nodes ({from_id}, {to_id})")
        key = self._edge_key(from_id, to_id)
        edge = self._edges.get(key)
        if edge is None:
            edge = TopoEdge(from_id=from_id, to_id=to_id)
            self._edges[key] = edge
            self._adjacency[from_id].add(to_id)
            self._adjacency[to_id].add(from_id)
        n = edge.traversal_count
        edge.path_length_m = (edge.path_length_m * n + float(path_length_m)) / (n + 1)
        edge.elapsed_time_sec = (edge.elapsed_time_sec * n + float(elapsed_time_sec)) / (n + 1)
        if risk is not None and math.isfinite(risk):
            edge.mean_risk = (edge.mean_risk * n + float(risk)) / (n + 1)
            edge.max_risk = max(edge.max_risk, float(risk))
        edge.traversal_count += 1
        if success:
            edge.success_count += 1
        else:
            edge.failure_count += 1
        edge.blocked = bool(blocked)
        edge.last_direction_rad = float(direction_rad)

    def get_edge(self, a: int, b: int) -> Optional[TopoEdge]:
        return self._edges.get(self._edge_key(a, b))

    def neighbors(self, node_id: int) -> Tuple[int, ...]:
        return tuple(sorted(self._adjacency.get(node_id, ())))

    @property
    def edges(self) -> Dict[Tuple[int, int], TopoEdge]:
        return dict(self._edges)

    def is_connected(self, a: int, b: int) -> bool:
        """BFS reachability -- used by tests to confirm a loop traversal
        never fragments the graph (plan requirement: "loop traversal 시
        graph connectivity 유지")."""
        if a not in self._nodes or b not in self._nodes:
            return False
        if a == b:
            return True
        seen = {a}
        frontier = [a]
        while frontier:
            nxt = []
            for node_id in frontier:
                for neighbor in self._adjacency.get(node_id, ()):
                    if neighbor == b:
                        return True
                    if neighbor not in seen:
                        seen.add(neighbor)
                        nxt.append(neighbor)
            frontier = nxt
        return False

    # ------------------------------------------------------------ observation tensor
    def to_fixed_tensor(
        self, max_nodes: int, robot_pose_mission, final_goal_mission: Tuple[float, float], *,
        goal_distance_norm_m: float, recency_norm_steps: float, now_step: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fixed-length ``(max_nodes, N_NODE_FEATURES)`` tensor + a
        ``(max_nodes,)`` validity mask (plan 9.5 steps 2-3: "fixed-length
        node tensor와 validity mask 우선 사용, GNN은 성능 이득 확인 후").
        The MOST-RECENTLY-VISITED ``max_nodes`` nodes are kept (ties broken
        by node id, oldest-created-first, for determinism); the rest are
        simply absent from this call's tensor (still in the graph, still
        available on the next call). ``max_nodes=0`` returns
        ``(0, N_NODE_FEATURES)``/``(0,)`` shaped empty arrays, never
        raises -- the caller-side "topology feedback disabled" no-op."""
        node_tensor = np.zeros((max_nodes, N_NODE_FEATURES), dtype=np.float32)
        validity = np.zeros((max_nodes,), dtype=bool)
        if max_nodes <= 0:
            return node_tensor, validity
        ordered = sorted(self._nodes.values(), key=lambda n: (-n.last_visited_step, n.node_id))[:max_nodes]
        norm = max(float(goal_distance_norm_m), 1e-6)
        recency_norm = max(float(recency_norm_steps), 1.0)
        for i, node in enumerate(ordered):
            dx = node.x_mission - robot_pose_mission.x
            dy = node.y_mission - robot_pose_mission.y
            node_tensor[i, 0] = float(np.clip(dx / norm, -1.0, 1.0))
            node_tensor[i, 1] = float(np.clip(dy / norm, -1.0, 1.0))
            node_tensor[i, 2] = float(np.clip(node.visit_count / _VISIT_COUNT_NORM, 0.0, 1.0))
            node_tensor[i, 3] = float(np.clip(node.free_direction_count / _FREE_DIRECTION_NORM, 0.0, 1.0))
            node_tensor[i, 4] = 1.0 if node.dead_end else 0.0
            node_tensor[i, 5] = float(np.clip(node.failure_count / _FAILURE_COUNT_NORM, 0.0, 1.0))
            node_tensor[i, 6] = float(np.clip(node.mean_risk, 0.0, 1.0))
            gdx = final_goal_mission[0] - node.x_mission
            gdy = final_goal_mission[1] - node.y_mission
            node_tensor[i, 7] = float(np.clip(math.hypot(gdx, gdy) / norm, 0.0, 1.0))
            node_tensor[i, 8] = float(np.clip(math.atan2(gdy, gdx) / math.pi, -1.0, 1.0))
            recency_steps = max(0, int(now_step) - node.last_visited_step)
            node_tensor[i, 9] = float(np.clip(1.0 - recency_steps / recency_norm, 0.0, 1.0))
            validity[i] = True
        return node_tensor, validity

    # ------------------------------------------------------------ serialization
    def to_dict(self) -> dict:
        """JSON-safe snapshot -- for embedding in a Global checkpoint
        manifest or a real-mission dry-run record (plan requirement:
        checkpoint manifest records the topology feature schema; this is
        the actual graph STATE, not just the schema)."""
        return {
            "max_nodes": self.max_nodes,
            "next_node_id": self._next_node_id,
            "nodes": [
                {
                    "node_id": n.node_id, "node_type": n.node_type.value, "x_mission": n.x_mission,
                    "y_mission": n.y_mission, "visit_count": n.visit_count,
                    "free_direction_count": n.free_direction_count, "dead_end": n.dead_end,
                    "failure_count": n.failure_count, "risk_samples": list(n.risk_samples),
                    "final_goal_distance_m": n.final_goal_distance_m,
                    "final_goal_bearing_rad": n.final_goal_bearing_rad,
                    "last_visited_step": n.last_visited_step, "created_step": n.created_step,
                }
                for n in self._nodes.values()
            ],
            "edges": [
                {
                    "from_id": e.from_id, "to_id": e.to_id, "traversal_count": e.traversal_count,
                    "path_length_m": e.path_length_m, "elapsed_time_sec": e.elapsed_time_sec,
                    "mean_risk": e.mean_risk, "max_risk": e.max_risk, "success_count": e.success_count,
                    "failure_count": e.failure_count, "last_direction_rad": e.last_direction_rad,
                    "blocked": e.blocked,
                }
                for e in self._edges.values()
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TopologicalGraph":
        graph = cls(max_nodes=data.get("max_nodes"))
        for row in data["nodes"]:
            node = TopoNode(
                node_id=row["node_id"], node_type=TopoNodeType(row["node_type"]), x_mission=row["x_mission"],
                y_mission=row["y_mission"], visit_count=row["visit_count"],
                free_direction_count=row["free_direction_count"], dead_end=row["dead_end"],
                failure_count=row["failure_count"], risk_samples=list(row["risk_samples"]),
                final_goal_distance_m=row["final_goal_distance_m"],
                final_goal_bearing_rad=row["final_goal_bearing_rad"],
                last_visited_step=row["last_visited_step"], created_step=row["created_step"],
            )
            graph._nodes[node.node_id] = node
            graph._adjacency[node.node_id] = set()
        for row in data["edges"]:
            edge = TopoEdge(
                from_id=row["from_id"], to_id=row["to_id"], traversal_count=row["traversal_count"],
                path_length_m=row["path_length_m"], elapsed_time_sec=row["elapsed_time_sec"],
                mean_risk=row["mean_risk"], max_risk=row["max_risk"], success_count=row["success_count"],
                failure_count=row["failure_count"], last_direction_rad=row["last_direction_rad"],
                blocked=row["blocked"],
            )
            key = cls._edge_key(edge.from_id, edge.to_id)
            graph._edges[key] = edge
            graph._adjacency[edge.from_id].add(edge.to_id)
            graph._adjacency[edge.to_id].add(edge.from_id)
        graph._next_node_id = data["next_node_id"]
        return graph
