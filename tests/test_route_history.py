"""Phase 5 topological memory: RouteHistory (plan section 9.3/15)."""

from hunter_kinodynamic_rl.navigation.memory.route_history import RouteEventType, RouteHistory


def test_first_arrival_is_enter():
    h = RouteHistory()
    event = h.record_arrival(1, now_step=0)
    assert event.event_type == RouteEventType.ENTER
    assert event.from_node_id is None
    assert h.current_node_id == 1


def test_immediate_reversal_is_backtrack_not_failure():
    h = RouteHistory()
    h.record_arrival(1, now_step=0)  # junction
    h.record_arrival(2, now_step=1)  # unknown branch
    event = h.record_arrival(1, now_step=2)  # back to junction
    assert event.event_type == RouteEventType.BACKTRACK


def test_revisit_far_from_immediate_predecessor_is_revisit():
    h = RouteHistory()
    h.record_arrival(1, now_step=0)
    h.record_arrival(2, now_step=1)
    h.record_arrival(3, now_step=2)
    event = h.record_arrival(1, now_step=3)  # revisits node 1, but predecessor was 3, not 1
    assert event.event_type == RouteEventType.REVISIT


def test_junction_dead_end_junction_new_branch_history_preserved():
    """plan section 15: Junction -> unknown branch -> dead end -> junction
    복귀 -> 다른 branch 선택 flow must be preserved in the sequence."""
    h = RouteHistory()
    h.record_arrival(0, now_step=0)   # Junction
    h.record_arrival(1, now_step=1)   # branch A
    h.record_arrival(2, now_step=2)   # dead end
    back = h.record_arrival(0, now_step=3)  # return to junction
    h.record_arrival(3, now_step=4)   # different branch B
    assert h.sequence == (0, 1, 2, 0, 3)
    assert back.event_type == RouteEventType.REVISIT  # predecessor was 2, not 0 -> not a pure backtrack step
    assert h.branch_since(0) == (3,)  # nodes visited after the SECOND (most recent) junction visit


def test_visit_count_and_has_visited():
    h = RouteHistory()
    h.record_arrival(1, now_step=0)
    h.record_arrival(2, now_step=1)
    h.record_arrival(1, now_step=2)
    assert h.visit_count(1) == 2
    assert h.has_visited(2)
    assert not h.has_visited(99)


def test_branch_since_empty_when_junction_never_visited():
    h = RouteHistory()
    h.record_arrival(1, now_step=0)
    assert h.branch_since(99) == ()


def test_reset_clears_state():
    h = RouteHistory()
    h.record_arrival(1, now_step=0)
    h.reset()
    assert h.current_node_id is None
    assert h.sequence == ()
    assert h.events == ()
