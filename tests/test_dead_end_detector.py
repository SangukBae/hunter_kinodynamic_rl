"""Phase 5 topological memory: DeadEndDetector (plan section 9.4)."""

import pytest

from hunter_kinodynamic_rl.navigation.memory.dead_end_detector import (
    DeadEndDetector, DeadEndDetectorConfig, DeadEndEvidence,
)
from hunter_kinodynamic_rl.navigation.memory.topological_graph import TopoNodeType, TopologicalGraph


def _detector(**overrides):
    return DeadEndDetector(DeadEndDetectorConfig(**overrides))


def test_evidence_vote_count():
    ev = DeadEndEvidence(
        low_free_direction_degree=True, frontier_absent=True, progress_stalled=False,
        repeated_emergency_stop=False, shrinking_valid_candidate_mask=True,
    )
    assert ev.vote_count == 3


def test_evaluate_evidence_all_signals_true_for_a_narrow_dead_pocket():
    d = _detector(free_direction_threshold=1, progress_stall_window_steps=3, progress_stall_min_delta_m=0.5,
                   repeated_estop_limit=2)
    ev = d.evaluate_evidence(
        known_free_direction_count=0, known_frontier_direction_count=0,
        recent_final_goal_distance_history=(10.0, 9.95, 9.9), consecutive_emergency_stops=3,
        valid_candidate_count=1,
    )
    assert ev.low_free_direction_degree
    assert ev.frontier_absent
    assert ev.progress_stalled
    assert ev.repeated_emergency_stop
    assert ev.shrinking_valid_candidate_mask
    assert ev.vote_count == 5


def test_evaluate_evidence_open_frontier_and_good_progress_is_not_dead_end_evidence():
    d = _detector()
    ev = d.evaluate_evidence(
        known_free_direction_count=4, known_frontier_direction_count=3,
        recent_final_goal_distance_history=(10.0, 5.0), consecutive_emergency_stops=0, valid_candidate_count=5,
    )
    assert ev.vote_count == 0


def test_verdict_requires_threshold_votes():
    d = _detector(evidence_vote_threshold=3)
    weak_evidence = DeadEndEvidence(True, True, False, False, False)  # 2 votes
    graph = TopologicalGraph()
    verdict = d.verdict(weak_evidence, graph=graph, node_id=None)
    assert not verdict.is_dead_end

    strong_evidence = DeadEndEvidence(True, True, True, False, False)  # 3 votes
    verdict2 = d.verdict(strong_evidence, graph=graph, node_id=None)
    assert verdict2.is_dead_end


def test_first_dead_end_is_not_repeated():
    """plan 9.4/15: the FIRST dead-end confirmation must be classified as a
    normal exploration event, not a repeated failure."""
    d = _detector(evidence_vote_threshold=2)
    graph = TopologicalGraph()
    node_id = graph.add_node(TopoNodeType.WAYPOINT, 5.0, 0.0)  # not yet flagged dead_end
    evidence = DeadEndEvidence(True, True, True, True, True)
    verdict = d.verdict(evidence, graph=graph, node_id=node_id)
    assert verdict.is_dead_end
    assert not verdict.is_repeated


def test_repeated_dead_end_only_after_node_already_flagged():
    """Caller contract: verdict() must be called BEFORE graph.mark_dead_end()
    -- calling it AFTER already marking the node dead_end classifies the
    SAME first visit as repeated, which is what this test also documents as
    the expected (caller-must-order-correctly) behavior."""
    d = _detector(evidence_vote_threshold=2)
    graph = TopologicalGraph()
    node_id = graph.add_node(TopoNodeType.WAYPOINT, 5.0, 0.0)
    evidence = DeadEndEvidence(True, True, True, True, True)

    first_verdict = d.verdict(evidence, graph=graph, node_id=node_id)
    assert not first_verdict.is_repeated
    graph.mark_dead_end(node_id, True)  # caller applies the verdict AFTER classifying

    # A LATER re-entry into the SAME already-flagged node/branch.
    second_verdict = d.verdict(evidence, graph=graph, node_id=node_id)
    assert second_verdict.is_dead_end
    assert second_verdict.is_repeated


def test_verdict_for_a_brand_new_node_is_never_repeated():
    d = _detector(evidence_vote_threshold=1)
    graph = TopologicalGraph()
    evidence = DeadEndEvidence(True, False, False, False, False)
    verdict = d.verdict(evidence, graph=graph, node_id=None)  # no existing node at all
    assert verdict.is_dead_end
    assert not verdict.is_repeated


def test_confidence_is_fraction_of_five_signals():
    d = _detector(evidence_vote_threshold=1)
    graph = TopologicalGraph()
    evidence = DeadEndEvidence(True, True, False, False, False)
    verdict = d.verdict(evidence, graph=graph, node_id=None)
    assert verdict.confidence == pytest.approx(0.4)


def test_config_validate_rejects_bad_threshold():
    with pytest.raises(Exception):
        DeadEndDetectorConfig(evidence_vote_threshold=0).validate()
    with pytest.raises(Exception):
        DeadEndDetectorConfig(evidence_vote_threshold=6).validate()
