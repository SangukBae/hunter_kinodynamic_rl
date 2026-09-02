"""Phase 5 topological memory: TopologicalNodeManager (plan section 9.3)."""

import math

import numpy as np
import pytest

from hunter_kinodynamic_rl.navigation.global_rl.subgoal_sampler import build_candidate_set
from hunter_kinodynamic_rl.config.schema import GlobalRLConfig
from hunter_kinodynamic_rl.navigation.memory.node_manager import (
    NodeManagerConfig, TopologicalNodeManager, compute_candidate_topology_features,
)
from hunter_kinodynamic_rl.navigation.memory.route_history import RouteHistory
from hunter_kinodynamic_rl.navigation.memory.topological_graph import TopoNodeType, TopologicalGraph
from hunter_kinodynamic_rl.navigation.mission.mission_frame import PoseXYYaw


def _manager(**overrides):
    graph = TopologicalGraph()
    route_history = RouteHistory()
    config = NodeManagerConfig(**overrides)
    return TopologicalNodeManager(graph, route_history, config), graph, route_history


def test_first_call_always_creates_start_node():
    mgr, graph, _ = _manager()
    event = mgr.maybe_create_or_update_node(PoseXYYaw(0.0, 0.0, 0.0), now_step=0)
    assert event is not None
    assert event.created
    assert len(graph) == 1


def test_small_movement_below_threshold_produces_no_event():
    mgr, graph, _ = _manager(node_min_distance_m=2.0, node_merge_radius_m=0.5)
    mgr.maybe_create_or_update_node(PoseXYYaw(0.0, 0.0, 0.0), now_step=0)
    event = mgr.maybe_create_or_update_node(PoseXYYaw(0.2, 0.0, 0.0), now_step=1)
    assert event is None
    assert len(graph) == 1


def test_distance_trigger_creates_new_node():
    mgr, graph, _ = _manager(node_min_distance_m=2.0, node_merge_radius_m=0.5)
    mgr.maybe_create_or_update_node(PoseXYYaw(0.0, 0.0, 0.0), now_step=0)
    event = mgr.maybe_create_or_update_node(PoseXYYaw(3.0, 0.0, 0.0), now_step=1)
    assert event is not None and event.created
    assert len(graph) == 2


def test_heading_change_trigger_creates_new_node_even_without_much_distance():
    mgr, graph, _ = _manager(node_min_distance_m=10.0, node_heading_change_rad=math.pi / 4, node_merge_radius_m=1.0)
    mgr.maybe_create_or_update_node(PoseXYYaw(0.0, 0.0, 0.0), now_step=0)
    event = mgr.maybe_create_or_update_node(PoseXYYaw(1.5, 0.0, math.pi / 2), now_step=1)
    assert event is not None and event.created


def test_duplicate_node_creation_is_prevented_by_merge_radius():
    """plan requirement: node/edge 중복 생성 방지."""
    mgr, graph, _ = _manager(node_min_distance_m=2.0, node_merge_radius_m=1.0)
    mgr.maybe_create_or_update_node(PoseXYYaw(0.0, 0.0, 0.0), now_step=0)
    # subgoal_reached forces a trigger even though the robot barely moved --
    # still within merge_radius_m of the first node, so it must MERGE, not
    # duplicate.
    event = mgr.maybe_create_or_update_node(PoseXYYaw(0.4, 0.0, 0.0), now_step=1, subgoal_reached=True)
    assert event is not None
    assert not event.created
    assert len(graph) == 1
    # 1 from the initial creation's own first-visit record + 1 from this merge.
    assert graph.get_node(event.node_id).visit_count == 2


def test_junction_detected_triggers_event_and_sets_type():
    mgr, graph, _ = _manager(node_min_distance_m=10.0, node_merge_radius_m=1.0)
    mgr.maybe_create_or_update_node(PoseXYYaw(0.0, 0.0, 0.0), now_step=0)
    event = mgr.maybe_create_or_update_node(PoseXYYaw(3.0, 0.0, 0.0), now_step=1, junction_detected=True)
    assert event is not None and event.created
    assert graph.get_node(event.node_id).node_type == TopoNodeType.JUNCTION


def test_edge_created_between_consecutive_distinct_nodes():
    mgr, graph, _ = _manager(node_min_distance_m=2.0, node_merge_radius_m=0.5)
    e1 = mgr.maybe_create_or_update_node(PoseXYYaw(0.0, 0.0, 0.0), now_step=0)
    e2 = mgr.maybe_create_or_update_node(PoseXYYaw(3.0, 0.0, 0.0), now_step=1, path_length_since_last_m=3.0)
    assert e2.edge_updated
    edge = graph.get_edge(e1.node_id, e2.node_id)
    assert edge is not None
    assert edge.path_length_m == pytest.approx(3.0)


def test_no_edge_on_very_first_node():
    mgr, graph, _ = _manager()
    event = mgr.maybe_create_or_update_node(PoseXYYaw(0.0, 0.0, 0.0), now_step=0)
    assert not event.edge_updated
    assert len(graph.edges) == 0


def test_dead_end_detected_marks_node_and_records_failure():
    mgr, graph, _ = _manager(node_min_distance_m=2.0, node_merge_radius_m=0.5)
    mgr.maybe_create_or_update_node(PoseXYYaw(0.0, 0.0, 0.0), now_step=0)
    event = mgr.maybe_create_or_update_node(
        PoseXYYaw(3.0, 0.0, 0.0), now_step=1, dead_end_detected=True, subgoal_failed=True,
    )
    node = graph.get_node(event.node_id)
    assert node.dead_end
    assert node.failure_count == 1


def test_reset_clears_trigger_state_and_optionally_the_graph():
    mgr, graph, route_history = _manager(node_min_distance_m=2.0, node_merge_radius_m=0.5)
    mgr.maybe_create_or_update_node(PoseXYYaw(0.0, 0.0, 0.0), now_step=0)
    mgr.reset()
    assert route_history.current_node_id is None
    assert len(graph) == 1  # graph preserved by default
    mgr.reset(clear_graph=True)
    assert len(mgr.graph) == 0


def test_config_rejects_merge_radius_not_smaller_than_min_distance():
    with pytest.raises(Exception):
        NodeManagerConfig(node_min_distance_m=1.0, node_merge_radius_m=1.0).validate()


def test_compute_candidate_topology_features_reports_repeated_deadend_and_visit_count():
    graph = TopologicalGraph()
    cfg = GlobalRLConfig(direction_degrees=[0.0], distances_m=[5.0])
    candidates = build_candidate_set(cfg)
    pose = PoseXYYaw(0.0, 0.0, 0.0)
    endpoint = candidates[0]
    ex = endpoint.radius_m * math.cos(endpoint.angle_rad)
    ey = endpoint.radius_m * math.sin(endpoint.angle_rad)
    node_id = graph.add_node(TopoNodeType.WAYPOINT, ex, ey, dead_end=True)
    graph.record_visit(node_id, now_step=1)
    graph.record_visit(node_id, now_step=2)

    features = compute_candidate_topology_features(candidates, graph, pose, merge_radius_m=1.0)
    assert features.shape == (len(candidates), 2)
    non_fallback = [c for c in candidates if not c.is_fallback][0]
    assert features[non_fallback.index, 0] == 1.0  # repeated_deadend_flag
    assert features[non_fallback.index, 1] == pytest.approx(2 / 50.0)


def test_compute_candidate_topology_features_zero_for_unexplored_candidate():
    graph = TopologicalGraph()
    cfg = GlobalRLConfig(direction_degrees=[0.0], distances_m=[5.0])
    candidates = build_candidate_set(cfg)
    pose = PoseXYYaw(0.0, 0.0, 0.0)
    features = compute_candidate_topology_features(candidates, graph, pose, merge_radius_m=1.0)
    assert np.all(features == 0.0)
