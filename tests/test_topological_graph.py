"""Phase 5 topological memory: TopologicalGraph (plan section 9.3)."""

import numpy as np
import pytest

from hunter_kinodynamic_rl.navigation.memory.topological_graph import (
    N_NODE_FEATURES, TopoNodeType, TopologicalGraph,
)
from hunter_kinodynamic_rl.navigation.mission.mission_frame import PoseXYYaw


def test_add_node_returns_stable_monotonic_ids():
    g = TopologicalGraph()
    a = g.add_node(TopoNodeType.START, 0.0, 0.0)
    b = g.add_node(TopoNodeType.JUNCTION, 5.0, 0.0)
    assert b == a + 1
    assert g.has_node(a) and g.has_node(b)
    assert len(g) == 2


def test_add_node_respects_max_nodes_capacity():
    g = TopologicalGraph(max_nodes=1)
    g.add_node(TopoNodeType.START, 0.0, 0.0)
    with pytest.raises(RuntimeError):
        g.add_node(TopoNodeType.WAYPOINT, 1.0, 0.0)


def test_find_nearest_node_dedup_within_radius():
    g = TopologicalGraph()
    a = g.add_node(TopoNodeType.WAYPOINT, 0.0, 0.0)
    g.add_node(TopoNodeType.WAYPOINT, 10.0, 0.0)
    found = g.find_nearest_node(0.4, 0.0, max_radius_m=1.0)
    assert found == a
    assert g.find_nearest_node(20.0, 0.0, max_radius_m=1.0) is None


def test_update_node_pose_moves_existing_node_never_allocates_new_id():
    g = TopologicalGraph()
    a = g.add_node(TopoNodeType.WAYPOINT, 0.0, 0.0)
    next_id_before = g._next_node_id
    g.update_node_pose(a, 3.0, 4.0)
    node = g.get_node(a)
    assert (node.x_mission, node.y_mission) == (3.0, 4.0)
    assert node.node_id == a
    assert g._next_node_id == next_id_before  # no new node allocated


def test_record_visit_and_failure_accumulate():
    g = TopologicalGraph()
    a = g.add_node(TopoNodeType.WAYPOINT, 0.0, 0.0)
    g.record_visit(a, now_step=1)
    g.record_visit(a, now_step=2)
    g.record_failure(a, risk=0.5)
    node = g.get_node(a)
    assert node.visit_count == 2
    assert node.failure_count == 1
    assert node.mean_risk == pytest.approx(0.5)
    assert node.last_visited_step == 2


def test_edge_is_undirected_and_averages_repeated_traversals():
    g = TopologicalGraph()
    a = g.add_node(TopoNodeType.START, 0.0, 0.0)
    b = g.add_node(TopoNodeType.JUNCTION, 5.0, 0.0)
    g.add_or_update_edge(a, b, path_length_m=5.0, elapsed_time_sec=10.0, risk=0.2, success=True)
    g.add_or_update_edge(b, a, path_length_m=7.0, elapsed_time_sec=14.0, risk=0.4, success=False)
    edge = g.get_edge(a, b)
    assert edge is not None
    assert edge.traversal_count == 2
    assert edge.path_length_m == pytest.approx(6.0)
    assert edge.mean_risk == pytest.approx(0.3)
    assert edge.max_risk == pytest.approx(0.4)
    assert edge.success_count == 1 and edge.failure_count == 1
    # symmetric lookup
    assert g.get_edge(b, a) is edge
    assert set(g.neighbors(a)) == {b}
    assert set(g.neighbors(b)) == {a}


def test_self_loop_edge_is_ignored():
    g = TopologicalGraph()
    a = g.add_node(TopoNodeType.START, 0.0, 0.0)
    g.add_or_update_edge(a, a, path_length_m=1.0, success=True)
    assert len(g.edges) == 0


def test_loop_traversal_keeps_graph_connected():
    g = TopologicalGraph()
    a = g.add_node(TopoNodeType.JUNCTION, 0.0, 0.0)
    b = g.add_node(TopoNodeType.JUNCTION, 5.0, 0.0)
    c = g.add_node(TopoNodeType.JUNCTION, 5.0, 5.0)
    d = g.add_node(TopoNodeType.JUNCTION, 0.0, 5.0)
    g.add_or_update_edge(a, b, success=True)
    g.add_or_update_edge(b, c, success=True)
    g.add_or_update_edge(c, d, success=True)
    g.add_or_update_edge(d, a, success=True)  # closes the loop
    assert g.is_connected(a, c)
    assert g.is_connected(b, d)


def test_to_fixed_tensor_shape_and_validity_mask():
    g = TopologicalGraph()
    g.add_node(TopoNodeType.START, 0.0, 0.0, now_step=0)
    g.add_node(TopoNodeType.JUNCTION, 5.0, 0.0, now_step=5)
    pose = PoseXYYaw(0.0, 0.0, 0.0)
    node_tensor, mask = g.to_fixed_tensor(
        max_nodes=4, robot_pose_mission=pose, final_goal_mission=(10.0, 0.0),
        goal_distance_norm_m=20.0, recency_norm_steps=10.0, now_step=6,
    )
    assert node_tensor.shape == (4, N_NODE_FEATURES)
    assert mask.shape == (4,)
    assert mask.dtype == bool
    assert mask.sum() == 2
    assert not mask[2] and not mask[3]
    assert node_tensor.dtype == np.float32


def test_to_fixed_tensor_zero_max_nodes_returns_empty_arrays_never_raises():
    g = TopologicalGraph()
    g.add_node(TopoNodeType.START, 0.0, 0.0)
    node_tensor, mask = g.to_fixed_tensor(
        max_nodes=0, robot_pose_mission=PoseXYYaw(0.0, 0.0, 0.0), final_goal_mission=(1.0, 0.0),
        goal_distance_norm_m=10.0, recency_norm_steps=10.0, now_step=0,
    )
    assert node_tensor.shape == (0, N_NODE_FEATURES)
    assert mask.shape == (0,)


def test_to_fixed_tensor_keeps_most_recently_visited_nodes_when_over_capacity():
    g = TopologicalGraph()
    ids = [g.add_node(TopoNodeType.WAYPOINT, float(i), 0.0, now_step=i) for i in range(5)]
    pose = PoseXYYaw(0.0, 0.0, 0.0)
    node_tensor, mask = g.to_fixed_tensor(
        max_nodes=2, robot_pose_mission=pose, final_goal_mission=(0.0, 0.0),
        goal_distance_norm_m=10.0, recency_norm_steps=10.0, now_step=10,
    )
    assert mask.sum() == 2
    # the two most recent nodes are ids[3] and ids[4] (last_visited_step 3, 4)
    # -- their relative_dx_norm should be 3/10 and 4/10 respectively.
    xs = sorted(node_tensor[mask, 0].tolist())
    assert xs == pytest.approx([0.3, 0.4])


def test_serialization_roundtrip_preserves_nodes_and_edges():
    g = TopologicalGraph(max_nodes=10)
    a = g.add_node(TopoNodeType.START, 0.0, 0.0, now_step=0, free_direction_count=3)
    b = g.add_node(TopoNodeType.DEAD_END, 5.0, 0.0, now_step=5, dead_end=True)
    g.record_failure(b, risk=0.7)
    g.add_or_update_edge(a, b, path_length_m=5.0, risk=0.1, success=True)

    data = g.to_dict()
    restored = TopologicalGraph.from_dict(data)
    assert restored.max_nodes == 10
    assert len(restored) == 2
    assert restored.get_node(b).dead_end is True
    assert restored.get_node(b).failure_count == 1
    assert restored.get_edge(a, b).traversal_count == 1
    # a new node after restore must not collide with an existing id
    c = restored.add_node(TopoNodeType.WAYPOINT, 1.0, 1.0)
    assert c not in (a, b)
