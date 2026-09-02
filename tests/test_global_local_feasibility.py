"""Phase 5 Global-Local feasibility feedback (plan section 9.6/12)."""

import numpy as np
import pytest

from hunter_kinodynamic_rl.config.schema import GlobalRLConfig, MappingConfig
from hunter_kinodynamic_rl.navigation.global_rl.subgoal_sampler import build_candidate_set
from hunter_kinodynamic_rl.navigation.hierarchy.feasibility import (
    FEASIBILITY_FEATURE_NAMES, FeasibilityConfig, LocalActionEvaluation, compute_feasibility_features,
)
from hunter_kinodynamic_rl.navigation.mapping.partial_map import PartialMap
from hunter_kinodynamic_rl.navigation.mission.mission_frame import PoseXYYaw

ROBOT_MAX_CURVATURE = 1.0


def _setup(size_cells=128):
    cfg = GlobalRLConfig(direction_degrees=[0.0, 90.0], distances_m=[3.0])
    candidates = build_candidate_set(cfg)
    partial_map = PartialMap(MappingConfig(mission_size_cells=size_cells), size_cells=size_cells)
    return cfg, candidates, partial_map


def test_feature_array_shape_and_order_matches_candidates():
    cfg, candidates, partial_map = _setup()
    features = compute_feasibility_features(
        candidates, partial_map, PoseXYYaw(0.0, 0.0, 0.0), ROBOT_MAX_CURVATURE, FeasibilityConfig(),
    )
    assert features.shape == (len(candidates), len(FEASIBILITY_FEATURE_NAMES))
    assert features.dtype == np.float32


def test_fallback_candidate_reports_honest_geometry_never_fabricated_progress():
    """Defect-fix item 7: the BACKTRACK/STOP_RECOVERY fallback candidate is
    not reachable via the same single constant-curvature-arc fit as a
    forward candidate (radius=0 divides by zero; angle=pi is outside the
    forward-hemisphere assumption), so its geometry comes from a direct
    endpoint clearance/occupancy lookup (honestly reporting "clear" on an
    unobserved map, exactly like every other candidate would) -- but its
    policy-conditioned progress_preserving is NEVER hardcoded to
    fabricate perfect progress; with no local_evaluator wired it reports
    the same zero-filled "unknown" default every other candidate gets."""
    cfg, candidates, partial_map = _setup()
    features = compute_feasibility_features(
        candidates, partial_map, PoseXYYaw(0.0, 0.0, 0.0), ROBOT_MAX_CURVATURE, FeasibilityConfig(),
    )
    fallback = candidates[-1]
    row = features[fallback.index]
    assert row[0] == 0.0  # rollout_collision -- unobserved map reports "clear", not "definitely blocked"
    assert row[1] == 1.0  # steering_saturation_ratio -- worst-case difficulty, never reported as "easy"
    assert row[2] == 1.0  # min_clearance_norm (max, on an unobserved map)
    assert row[3] == 0.0  # predicted_action_risk -- zero-filled (no evaluator), never fabricated as "confirmed safe"
    assert row[4] == 0.0  # progress_preserving -- zero-filled, never hardcoded True


def test_rollout_collision_detected_for_occupied_endpoint():
    cfg, candidates, partial_map = _setup()
    pose = PoseXYYaw(0.0, 0.0, 0.0)
    straight = next(c for c in candidates if not c.is_fallback and abs(c.angle_rad) < 1e-6)
    # Mark cells along the straight-ahead candidate's path as occupied.
    for t in np.linspace(0.5, 1.0, 5):
        x = t * straight.radius_m
        cell = partial_map.world_to_cell(x, 0.0)
        if cell is not None:
            partial_map.log_odds[cell] = partial_map.config.log_odds_max
            partial_map.observed[cell] = True

    features = compute_feasibility_features(candidates, partial_map, pose, ROBOT_MAX_CURVATURE, FeasibilityConfig())
    assert features[straight.index, 0] == 1.0  # rollout_collision


def test_clear_path_reports_no_collision_and_full_clearance():
    cfg, candidates, partial_map = _setup()
    pose = PoseXYYaw(0.0, 0.0, 0.0)
    straight = next(c for c in candidates if not c.is_fallback and abs(c.angle_rad) < 1e-6)
    features = compute_feasibility_features(
        candidates, partial_map, pose, ROBOT_MAX_CURVATURE,
        FeasibilityConfig(clearance_search_radius_m=1.0, clearance_norm_m=1.0),
    )
    assert features[straight.index, 0] == 0.0
    assert features[straight.index, 2] == pytest.approx(1.0)


def test_historical_success_rate_reflects_recorded_failures():
    cfg, candidates, partial_map = _setup()
    pose = PoseXYYaw(0.0, 0.0, 0.0)
    straight = next(c for c in candidates if not c.is_fallback and abs(c.angle_rad) < 1e-6)
    x = straight.radius_m
    partial_map.record_visit(x, 0.0, step=1)
    partial_map.record_failure(x, 0.0)
    partial_map.record_failure(x, 0.0)
    features = compute_feasibility_features(
        candidates, partial_map, pose, ROBOT_MAX_CURVATURE,
        FeasibilityConfig(historical_stats_radius_m=0.5),
    )
    assert features[straight.index, 5] < 0.5  # mostly failures -> low success rate


def test_historical_success_rate_is_neutral_when_unexplored():
    cfg, candidates, partial_map = _setup()
    pose = PoseXYYaw(0.0, 0.0, 0.0)
    features = compute_feasibility_features(candidates, partial_map, pose, ROBOT_MAX_CURVATURE, FeasibilityConfig())
    non_fallback = [c for c in candidates if not c.is_fallback]
    for c in non_fallback:
        assert features[c.index, 5] == pytest.approx(0.5)


def test_no_local_evaluator_zero_fills_predicted_risk_and_progress_preserving():
    cfg, candidates, partial_map = _setup()
    pose = PoseXYYaw(0.0, 0.0, 0.0)
    features = compute_feasibility_features(
        candidates, partial_map, pose, ROBOT_MAX_CURVATURE, FeasibilityConfig(), local_evaluator=None,
    )
    non_fallback = [c for c in candidates if not c.is_fallback]
    for c in non_fallback:
        assert features[c.index, 3] == 0.0  # predicted_action_risk
        assert features[c.index, 4] == 0.0  # progress_preserving


class _FakeEvaluator:
    """Mimics a live LocalPolicyController+local-agent composition -- the
    class exists ONLY to prove the caller-conditioned-action contract
    (candidate first, THEN evaluate an action for it), never fed a bare
    [radius, angle]. Two-phase (defect-fix item 6): ``capture_context``
    once per decision, ``evaluate`` once per candidate, sharing that one
    context object -- this fake asserts exactly that usage pattern."""

    _SENTINEL_CONTEXT = object()

    def __init__(self, risk: float, progress_preserving: bool):
        self.risk = risk
        self.progress_preserving = progress_preserving
        self.seen_candidates = []
        self.capture_context_calls = 0

    def capture_context(self):
        self.capture_context_calls += 1
        return self._SENTINEL_CONTEXT

    def evaluate(self, candidate, robot_pose_mission, context):
        assert context is self._SENTINEL_CONTEXT
        self.seen_candidates.append(candidate)
        return LocalActionEvaluation(
            action=np.array([0.1, 0.2, 1.0]), predicted_risk=self.risk, progress_preserving=self.progress_preserving,
        )


def test_local_evaluator_supplies_predicted_risk_and_progress_preserving_per_candidate():
    cfg, candidates, partial_map = _setup()
    pose = PoseXYYaw(0.0, 0.0, 0.0)
    evaluator = _FakeEvaluator(risk=0.6, progress_preserving=True)
    features = compute_feasibility_features(
        candidates, partial_map, pose, ROBOT_MAX_CURVATURE, FeasibilityConfig(), local_evaluator=evaluator,
    )
    # defect-fix item 6: exactly ONE context captured for the whole
    # decision, shared by every candidate (fallback included below).
    assert evaluator.capture_context_calls == 1
    for c in candidates:
        assert features[c.index, 3] == pytest.approx(0.6)
        assert features[c.index, 4] == 1.0
    # Evaluated once per candidate -- INCLUDING the fallback (defect-fix
    # item 7: no more hardcoded shortcut for it) -- each with the
    # CANDIDATE object itself (never a bare radius/angle tuple).
    assert len(evaluator.seen_candidates) == len(candidates)
    for c in evaluator.seen_candidates:
        assert hasattr(c, "radius_m") and hasattr(c, "angle_rad")
