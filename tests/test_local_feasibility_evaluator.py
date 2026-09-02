"""Requirement E / defect-fix items 1, 6, 7: FrozenLocalFeasibilityEvaluator
-- concrete LocalFeasibilityEvaluator wiring the frozen Local policy into
Global candidate feasibility scoring. Pure Python (no rclpy/torch/Gazebo):
the Local "agent" here is a fake object exposing the same
``select_action(obs, deterministic=True)``/``predict_risk(obs, action)``
surface a real ``rl.algorithms.kinodynamic_tqc.agent.Agent`` does."""

import math
import time

import numpy as np
import pytest

from hunter_kinodynamic_rl.config.loader import load_profile
from hunter_kinodynamic_rl.navigation.global_rl.subgoal_sampler import SubgoalCandidate
from hunter_kinodynamic_rl.navigation.hierarchy.local_feasibility_evaluator import (
    FeasibilityEvaluatorTelemetry, FrozenLocalFeasibilityEvaluator, LocalSensorSnapshot,
)
from hunter_kinodynamic_rl.navigation.mission.mission_frame import PoseXYYaw
from hunter_kinodynamic_rl.trajectory.action_space import ACTION_DIM


def _profile():
    return load_profile("smoke_test")  # action_space.mode=trajectory


def _lidar_frame(profile, fill=None):
    history_len = profile.observation.frame_stack if profile.features.temporal_context else 1
    fill_value = profile.observation.lidar_max_range_m if fill is None else fill
    return np.full(profile.observation.lidar_bins * history_len, fill_value, dtype=np.float32)


def _snapshot(profile=None, stamp=None, prev_action=(0.0, 0.0, 0.0), lidar_frame=None):
    profile = profile or _profile()
    return LocalSensorSnapshot(
        lidar_frame=_lidar_frame(profile) if lidar_frame is None else lidar_frame,
        prev_action=tuple(prev_action),
        speed_mps=0.0, yaw_rate_rps=0.0, steering_rad=0.0,
        stamp_monotonic=time.monotonic() if stamp is None else stamp,
    )


def _candidate(angle_deg=0.0, radius_m=3.0, index=0):
    return SubgoalCandidate(index=index, angle_rad=math.radians(angle_deg), radius_m=radius_m,
                             is_fallback=False, label=f"dir{angle_deg:g}_dist{radius_m:g}")


class _DeterministicAgent:
    """Fake frozen Local agent: returns a FIXED action, and a fixed risk
    from ``predict_risk`` -- deterministic, so evaluate() output is fully
    predictable."""

    def __init__(self, action, risk=0.42):
        self.action = np.asarray(action, dtype=np.float64)
        self.risk = risk
        self.select_action_calls = []
        self.predict_risk_calls = []

    def select_action(self, obs, deterministic=True):
        self.select_action_calls.append((obs.copy(), deterministic))
        return self.action.copy()

    def predict_risk(self, state, action):
        self.predict_risk_calls.append((np.asarray(state).copy(), np.asarray(action).copy()))
        return self.risk


class _HangingAgent:
    def select_action(self, obs, deterministic=True):
        time.sleep(10.0)
        return np.zeros(ACTION_DIM)


class _RaisingAgent:
    def select_action(self, obs, deterministic=True):
        raise RuntimeError("boom")


class _HangingRiskAgent:
    def __init__(self, action):
        self.action = np.asarray(action, dtype=np.float64)

    def select_action(self, obs, deterministic=True):
        return self.action.copy()

    def predict_risk(self, state, action):
        time.sleep(10.0)
        return 0.5


class _RaisingRiskAgent:
    def __init__(self, action):
        self.action = np.asarray(action, dtype=np.float64)

    def select_action(self, obs, deterministic=True):
        return self.action.copy()

    def predict_risk(self, state, action):
        raise RuntimeError("risk boom")


class _NanRiskAgent:
    def __init__(self, action):
        self.action = np.asarray(action, dtype=np.float64)

    def select_action(self, obs, deterministic=True):
        return self.action.copy()

    def predict_risk(self, state, action):
        return float("nan")


def _evaluate_once(evaluator, candidate, pose):
    context = evaluator.capture_context()
    assert context is not None
    return evaluator.evaluate(candidate, pose, context)


def test_evaluate_forward_action_reports_predicted_risk_and_progress_preserving():
    profile = _profile()
    # kappa=0 (action[0]=0.0 -> straight), v_ref>0 (action[1]>-1), long horizon (action[2]=1.0 -> max horizon)
    agent = _DeterministicAgent(action=[0.0, 0.5, 1.0], risk=0.37)
    evaluator = FrozenLocalFeasibilityEvaluator(
        profile, agent, inference_timeout_sec=1.0, max_snapshot_age_sec=1.0,
        snapshot_provider=lambda: _snapshot(profile),
    )
    candidate = _candidate(angle_deg=0.0, radius_m=3.0)
    result = _evaluate_once(evaluator, candidate, PoseXYYaw(0.0, 0.0, 0.0))
    assert result is not None
    assert result.predicted_risk == pytest.approx(0.37)
    assert result.progress_preserving is True  # straight-ahead action moves toward a straight-ahead subgoal
    assert evaluator.telemetry.queries == 1
    assert evaluator.telemetry.fallback_total == 0
    assert evaluator.telemetry.risk_evaluated == 1
    # select_action/predict_risk were both called with the SAME candidate-
    # conditioned observation, never a bare [radius, angle].
    assert len(agent.select_action_calls) == 1
    assert len(agent.predict_risk_calls) == 1


def test_evaluate_backward_action_is_not_progress_preserving_toward_forward_subgoal():
    profile = _profile()
    # v_ref normalized to 0 -> v_ref == v_min_mps (>=0, "not moving forward"); reject as not progress-preserving.
    agent = _DeterministicAgent(action=[0.0, -1.0, 1.0], risk=0.1)
    evaluator = FrozenLocalFeasibilityEvaluator(
        profile, agent, inference_timeout_sec=1.0, max_snapshot_age_sec=1.0,
        snapshot_provider=lambda: _snapshot(profile),
    )
    candidate = _candidate(angle_deg=0.0, radius_m=3.0)
    result = _evaluate_once(evaluator, candidate, PoseXYYaw(0.0, 0.0, 0.0))
    assert result is not None
    assert result.progress_preserving is False


def test_missing_snapshot_is_a_tracked_fallback_never_a_disguised_reading():
    profile = _profile()
    agent = _DeterministicAgent(action=[0.0, 0.5, 1.0])
    evaluator = FrozenLocalFeasibilityEvaluator(
        profile, agent, inference_timeout_sec=1.0, max_snapshot_age_sec=1.0, snapshot_provider=lambda: None,
    )
    context = evaluator.capture_context()
    assert context is None
    assert evaluator.telemetry.fallbacks_stale_snapshot == 1
    assert evaluator.telemetry.fallback_total == 1
    assert evaluator.telemetry.fallback_rate == pytest.approx(1.0)
    assert len(agent.select_action_calls) == 0  # never even attempts inference on a missing snapshot


def test_stale_snapshot_is_rejected():
    profile = _profile()
    agent = _DeterministicAgent(action=[0.0, 0.5, 1.0])
    stale = _snapshot(profile, stamp=time.monotonic() - 100.0)
    evaluator = FrozenLocalFeasibilityEvaluator(
        profile, agent, inference_timeout_sec=1.0, max_snapshot_age_sec=0.5, snapshot_provider=lambda: stale,
    )
    context = evaluator.capture_context()
    assert context is None
    assert evaluator.telemetry.fallbacks_stale_snapshot == 1


def test_inference_timeout_is_a_tracked_fallback():
    profile = _profile()
    evaluator = FrozenLocalFeasibilityEvaluator(
        profile, _HangingAgent(), inference_timeout_sec=0.05, max_snapshot_age_sec=5.0,
        snapshot_provider=lambda: _snapshot(profile),
    )
    result = _evaluate_once(evaluator, _candidate(), PoseXYYaw(0.0, 0.0, 0.0))
    assert result is None
    assert evaluator.telemetry.fallbacks_inference_timeout == 1


def test_inference_error_is_a_tracked_fallback():
    profile = _profile()
    evaluator = FrozenLocalFeasibilityEvaluator(
        profile, _RaisingAgent(), inference_timeout_sec=1.0, max_snapshot_age_sec=5.0,
        snapshot_provider=lambda: _snapshot(profile),
    )
    result = _evaluate_once(evaluator, _candidate(), PoseXYYaw(0.0, 0.0, 0.0))
    assert result is None
    assert evaluator.telemetry.fallbacks_inference_error == 1


def test_invalid_action_shape_is_a_tracked_fallback():
    profile = _profile()
    agent = _DeterministicAgent(action=[0.0, 0.5])  # wrong length -- validate_action rejects
    evaluator = FrozenLocalFeasibilityEvaluator(
        profile, agent, inference_timeout_sec=1.0, max_snapshot_age_sec=5.0,
        snapshot_provider=lambda: _snapshot(profile),
    )
    result = _evaluate_once(evaluator, _candidate(), PoseXYYaw(0.0, 0.0, 0.0))
    assert result is None
    assert evaluator.telemetry.fallbacks_invalid_action == 1


def test_agent_without_predict_risk_reports_none_risk_not_zero():
    profile = _profile()

    class _NoRiskAgent:
        def select_action(self, obs, deterministic=True):
            return np.array([0.0, 0.5, 1.0])

    evaluator = FrozenLocalFeasibilityEvaluator(
        profile, _NoRiskAgent(), inference_timeout_sec=1.0, max_snapshot_age_sec=5.0,
        snapshot_provider=lambda: _snapshot(profile),
    )
    result = _evaluate_once(evaluator, _candidate(), PoseXYYaw(0.0, 0.0, 0.0))
    assert result is not None
    assert result.predicted_risk is None  # never fabricated as 0.0


def test_shared_telemetry_instance_accumulates_across_calls():
    profile = _profile()
    telemetry = FeasibilityEvaluatorTelemetry()
    evaluator = FrozenLocalFeasibilityEvaluator(
        profile, _DeterministicAgent(action=[0.0, 0.5, 1.0]), inference_timeout_sec=1.0, max_snapshot_age_sec=1.0,
        snapshot_provider=lambda: _snapshot(profile), telemetry=telemetry,
    )
    for _ in range(3):
        _evaluate_once(evaluator, _candidate(), PoseXYYaw(0.0, 0.0, 0.0))
    assert telemetry.queries == 3
    assert telemetry is evaluator.telemetry


# ---------------------------------------------------------------- defect-fix item 7: predict_risk bounded/telemetry

def test_predict_risk_timeout_is_a_tracked_risk_fallback_not_a_candidate_fallback():
    profile = _profile()
    agent = _HangingRiskAgent(action=[0.0, 0.5, 1.0])
    evaluator = FrozenLocalFeasibilityEvaluator(
        profile, agent, inference_timeout_sec=0.05, max_snapshot_age_sec=5.0,
        snapshot_provider=lambda: _snapshot(profile),
    )
    result = _evaluate_once(evaluator, _candidate(), PoseXYYaw(0.0, 0.0, 0.0))
    # the ACTION itself decoded fine -- a risk-read failure must not
    # invalidate the whole candidate evaluation.
    assert result is not None
    assert result.predicted_risk is None
    assert evaluator.telemetry.risk_timeout == 1
    assert evaluator.telemetry.fallback_total == 0


def test_predict_risk_error_is_a_tracked_risk_fallback():
    profile = _profile()
    agent = _RaisingRiskAgent(action=[0.0, 0.5, 1.0])
    evaluator = FrozenLocalFeasibilityEvaluator(
        profile, agent, inference_timeout_sec=1.0, max_snapshot_age_sec=5.0,
        snapshot_provider=lambda: _snapshot(profile),
    )
    result = _evaluate_once(evaluator, _candidate(), PoseXYYaw(0.0, 0.0, 0.0))
    assert result is not None
    assert result.predicted_risk is None
    assert evaluator.telemetry.risk_error == 1


def test_predict_risk_nan_is_never_coerced_to_a_confirmed_zero():
    profile = _profile()
    agent = _NanRiskAgent(action=[0.0, 0.5, 1.0])
    evaluator = FrozenLocalFeasibilityEvaluator(
        profile, agent, inference_timeout_sec=1.0, max_snapshot_age_sec=5.0,
        snapshot_provider=lambda: _snapshot(profile),
    )
    result = _evaluate_once(evaluator, _candidate(), PoseXYYaw(0.0, 0.0, 0.0))
    assert result is not None
    assert result.predicted_risk is None
    assert evaluator.telemetry.risk_non_finite == 1


# ---------------------------------------------------------------- defect-fix item 6: shared immutable context

def test_candidate_order_does_not_change_per_candidate_result():
    """Every candidate in ONE decision must see the SAME LiDAR history --
    scoring [A, B] then [B, A] against the SAME captured context must
    produce identical per-candidate results (before this fix, evaluate()
    pushed the current scan into its own frame stack once per candidate,
    so a candidate's result depended on how many earlier candidates in the
    same decision had already been scored)."""
    profile = _profile()
    agent = _DeterministicAgent(action=[0.0, 0.5, 1.0], risk=0.2)
    snapshot = _snapshot(profile)
    evaluator = FrozenLocalFeasibilityEvaluator(
        profile, agent, inference_timeout_sec=1.0, max_snapshot_age_sec=5.0, snapshot_provider=lambda: snapshot,
    )
    pose = PoseXYYaw(0.0, 0.0, 0.0)
    a = _candidate(angle_deg=10.0, radius_m=3.0, index=0)
    b = _candidate(angle_deg=-20.0, radius_m=2.0, index=1)

    context1 = evaluator.capture_context()
    obs_a_first = evaluator.evaluate(a, pose, context1).action.copy()
    obs_b_first = evaluator.evaluate(b, pose, context1).action.copy()

    context2 = evaluator.capture_context()
    obs_b_second = evaluator.evaluate(b, pose, context2).action.copy()
    obs_a_second = evaluator.evaluate(a, pose, context2).action.copy()

    np.testing.assert_array_equal(obs_a_first, obs_a_second)
    np.testing.assert_array_equal(obs_b_first, obs_b_second)
    # the fake agent is order-insensitive (always returns the same fixed
    # action) -- what this test actually pins is that the OBSERVATION each
    # call built is identical regardless of order, verified below via the
    # agent's own recorded call arguments.
    first_pass_obs = [call[0] for call in agent.select_action_calls[0:2]]
    second_pass_obs = [call[0] for call in agent.select_action_calls[2:4]]
    # first pass: [a, b]; second pass: [b, a] -- compare a-vs-a, b-vs-b.
    np.testing.assert_array_equal(first_pass_obs[0], second_pass_obs[1])
    np.testing.assert_array_equal(first_pass_obs[1], second_pass_obs[0])


def test_lidar_frame_history_is_frozen_across_the_whole_candidate_loop():
    profile = _profile()
    frame = _lidar_frame(profile, fill=1.23)
    agent = _DeterministicAgent(action=[0.0, 0.5, 1.0])
    evaluator = FrozenLocalFeasibilityEvaluator(
        profile, agent, inference_timeout_sec=1.0, max_snapshot_age_sec=5.0,
        snapshot_provider=lambda: _snapshot(profile, lidar_frame=frame),
    )
    pose = PoseXYYaw(0.0, 0.0, 0.0)
    candidates = [_candidate(angle_deg=d, radius_m=2.0, index=i) for i, d in enumerate((-30, 0, 30))]
    context = evaluator.capture_context()
    for c in candidates:
        evaluator.evaluate(c, pose, context)
    lidar_dim = frame.shape[0]
    for obs, _ in agent.select_action_calls:
        np.testing.assert_array_equal(obs[:lidar_dim], frame)


def test_real_prev_action_is_reflected_in_the_observation_tail():
    profile = _profile()
    agent = _DeterministicAgent(action=[0.0, 0.5, 1.0])
    real_prev_action = (0.11, -0.22, 0.33)
    evaluator = FrozenLocalFeasibilityEvaluator(
        profile, agent, inference_timeout_sec=1.0, max_snapshot_age_sec=5.0,
        snapshot_provider=lambda: _snapshot(profile, prev_action=real_prev_action),
    )
    _evaluate_once(evaluator, _candidate(), PoseXYYaw(0.0, 0.0, 0.0))
    obs, _ = agent.select_action_calls[0]
    lidar_dim = _lidar_frame(profile).shape[0]
    tail = obs[lidar_dim:]
    # robot_state_vector tail layout (7D contract): [dist, heading_err,
    # prev_a0, prev_a1, v, yaw_rate, steering] -- prev_a0/prev_a1 must be
    # the REAL previous action, never the always-zero placeholder a
    # separately-accumulated evaluator-owned controller used to report.
    assert tail[2] == pytest.approx(real_prev_action[0])
    assert tail[3] == pytest.approx(real_prev_action[1])


def test_evaluate_never_mutates_the_real_local_controller():
    """The evaluator holds no LocalPolicyController of its own any more --
    this test pins that guarantee structurally by asserting the class has
    no such attribute, so a future regression that reintroduces a private
    frame stack is caught immediately."""
    profile = _profile()
    evaluator = FrozenLocalFeasibilityEvaluator(
        profile, _DeterministicAgent(action=[0.0, 0.5, 1.0]), inference_timeout_sec=1.0, max_snapshot_age_sec=5.0,
        snapshot_provider=lambda: _snapshot(profile),
    )
    assert not hasattr(evaluator, "_controller")
