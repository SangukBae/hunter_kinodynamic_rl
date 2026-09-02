"""Evidence-combining dead-end classifier (plan section 9.4) -- deliberately
never relies on a single heuristic. Each signal below is individually noisy
(a temporarily narrow action mask does not always mean a dead end; a single
emergency stop does not either) so :class:`DeadEndDetector` requires a
CONFIGURABLE NUMBER of signals to agree before calling something a dead end
at all, then separately classifies FIRST vs. REPEATED using the
:class:`~hunter_kinodynamic_rl.navigation.memory.topological_graph.TopologicalGraph`
node state that already existed BEFORE this arrival (plan 9.4/15: "첫
dead-end 확인은 정상 exploration event로 저장한다. 같은 branch 재진입만
repeated-dead-end로 계산한다").

**Caller contract**: call :meth:`DeadEndDetector.verdict` BEFORE marking the
node dead-end on the graph (``graph.mark_dead_end``/``record_failure``) --
the repeated/first classification reads the graph's PRE-arrival state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from hunter_kinodynamic_rl.config.schema import ConfigError
from hunter_kinodynamic_rl.navigation.memory.topological_graph import TopologicalGraph


@dataclass(frozen=True)
class DeadEndEvidence:
    low_free_direction_degree: bool
    frontier_absent: bool
    progress_stalled: bool
    repeated_emergency_stop: bool
    shrinking_valid_candidate_mask: bool

    @property
    def vote_count(self) -> int:
        return sum((
            self.low_free_direction_degree, self.frontier_absent, self.progress_stalled,
            self.repeated_emergency_stop, self.shrinking_valid_candidate_mask,
        ))


@dataclass(frozen=True)
class DeadEndVerdict:
    is_dead_end: bool
    confidence: float
    evidence: DeadEndEvidence
    is_repeated: bool


@dataclass(frozen=True)
class DeadEndDetectorConfig:
    free_direction_threshold: int = 1
    progress_stall_window_steps: int = 20
    progress_stall_min_delta_m: float = 0.2
    repeated_estop_limit: int = 3
    #: Number of the 5 boolean evidence signals (low free-direction degree,
    #: frontier absent, progress stalled, repeated e-stop, shrinking
    #: candidate mask) that must agree before this node counts as a dead
    #: end at all.
    evidence_vote_threshold: int = 2

    def validate(self) -> None:
        if self.free_direction_threshold < 0:
            raise ConfigError("dead_end_detector.free_direction_threshold must be >= 0")
        if self.progress_stall_window_steps <= 0:
            raise ConfigError("dead_end_detector.progress_stall_window_steps must be > 0")
        if self.progress_stall_min_delta_m < 0.0:
            raise ConfigError("dead_end_detector.progress_stall_min_delta_m must be >= 0")
        if self.repeated_estop_limit <= 0:
            raise ConfigError("dead_end_detector.repeated_estop_limit must be > 0")
        if not (1 <= self.evidence_vote_threshold <= 5):
            raise ConfigError("dead_end_detector.evidence_vote_threshold must be in [1, 5]")


class DeadEndDetector:
    def __init__(self, config: DeadEndDetectorConfig) -> None:
        config.validate()
        self.config = config

    def evaluate_evidence(
        self, *, known_free_direction_count: int, known_frontier_direction_count: int,
        recent_final_goal_distance_history: Sequence[float], consecutive_emergency_stops: int,
        valid_candidate_count: int,
    ) -> DeadEndEvidence:
        """``known_frontier_direction_count`` is the number of nearby
        UNKNOWN (unexplored, never occupied/free) directions -- a node with
        few known-free exits AND no unexplored frontier left to try really
        has nowhere new to go; one that still has an unexplored frontier is
        not yet provably a dead end, it just has not been fully explored."""
        cfg = self.config
        progress_stalled = False
        if len(recent_final_goal_distance_history) >= cfg.progress_stall_window_steps:
            window = recent_final_goal_distance_history[-cfg.progress_stall_window_steps:]
            progress_stalled = (window[0] - window[-1]) < cfg.progress_stall_min_delta_m
        return DeadEndEvidence(
            low_free_direction_degree=known_free_direction_count <= cfg.free_direction_threshold,
            frontier_absent=known_frontier_direction_count <= 0,
            progress_stalled=progress_stalled,
            repeated_emergency_stop=consecutive_emergency_stops >= cfg.repeated_estop_limit,
            shrinking_valid_candidate_mask=valid_candidate_count <= 1,
        )

    def verdict(
        self, evidence: DeadEndEvidence, *, graph: TopologicalGraph, node_id: Optional[int],
    ) -> DeadEndVerdict:
        votes = evidence.vote_count
        is_dead_end = votes >= self.config.evidence_vote_threshold
        confidence = votes / 5.0
        is_repeated = False
        if is_dead_end and node_id is not None and graph.has_node(node_id):
            is_repeated = graph.get_node(node_id).dead_end
        return DeadEndVerdict(is_dead_end=is_dead_end, confidence=confidence, evidence=evidence, is_repeated=is_repeated)
