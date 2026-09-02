"""Chronological node-visit sequence (plan section 9.3/15) -- the trace a
dead-end detector and Global reward use to tell BACKTRACKING (returning to
a junction to try another branch -- normal, not a failure) apart from a
genuinely new node visit or a stale re-entry into an already-failed branch.

Deliberately holds no policy/reward opinion of its own: :class:`RouteHistory`
only classifies WHAT HAPPENED geometrically (enter / revisit / backtrack),
never whether it was good or bad -- that judgment is
``dead_end_detector.py``'s and ``global_rl/reward.py``'s job.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import List, Optional, Tuple


class RouteEventType(enum.Enum):
    #: First-ever arrival at this node id.
    ENTER = "enter"
    #: Arrival at a node visited before, but NOT the immediately preceding
    #: node in the sequence (e.g. re-crossing a loop far from home).
    REVISIT = "revisit"
    #: Arrival at the node that was visited immediately before the current
    #: one -- i.e. the robot just reversed back along the edge it came
    #: from. Plan 15: "backtracking 자체를 실패로 보면 안 된다".
    BACKTRACK = "backtrack"


@dataclass(frozen=True)
class RouteEvent:
    step: int
    node_id: int
    event_type: RouteEventType
    from_node_id: Optional[int]


class RouteHistory:
    def __init__(self) -> None:
        self._sequence: List[int] = []
        self._events: List[RouteEvent] = []
        self._visited_node_ids: set = set()

    def reset(self) -> None:
        self._sequence.clear()
        self._events.clear()
        self._visited_node_ids.clear()

    @property
    def current_node_id(self) -> Optional[int]:
        return self._sequence[-1] if self._sequence else None

    @property
    def sequence(self) -> Tuple[int, ...]:
        return tuple(self._sequence)

    @property
    def events(self) -> Tuple[RouteEvent, ...]:
        return tuple(self._events)

    def record_arrival(self, node_id: int, *, now_step: int) -> RouteEvent:
        """Appends one arrival. A repeated call with the SAME ``node_id``
        as the current one (the robot is still at/near the same node on
        consecutive ticks) is recorded as another ``REVISIT`` -- callers
        that only want one event per DISTINCT node transition should only
        call this when ``node_id != current_node_id``."""
        previous = self.current_node_id
        second_previous = self._sequence[-2] if len(self._sequence) >= 2 else None
        if previous is not None and node_id == second_previous:
            event_type = RouteEventType.BACKTRACK
        elif node_id in self._visited_node_ids:
            event_type = RouteEventType.REVISIT
        else:
            event_type = RouteEventType.ENTER
        event = RouteEvent(step=int(now_step), node_id=node_id, event_type=event_type, from_node_id=previous)
        self._sequence.append(node_id)
        self._visited_node_ids.add(node_id)
        self._events.append(event)
        return event

    def has_visited(self, node_id: int) -> bool:
        return node_id in self._visited_node_ids

    def branch_since(self, junction_node_id: int) -> Tuple[int, ...]:
        """Nodes visited strictly after the MOST RECENT occurrence of
        ``junction_node_id`` in the sequence -- the "branch explored since
        leaving this junction" trace (plan 15's "junction -> unknown branch
        -> dead end -> junction 복귀 -> 다른 branch" flow). Empty if the
        junction was never visited or is the current (last) node."""
        try:
            idx = len(self._sequence) - 1 - self._sequence[::-1].index(junction_node_id)
        except ValueError:
            return ()
        return tuple(self._sequence[idx + 1:])

    def visit_count(self, node_id: int) -> int:
        return sum(1 for n in self._sequence if n == node_id)
