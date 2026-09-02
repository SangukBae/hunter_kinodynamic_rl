"""Concrete :class:`~hunter_kinodynamic_rl.navigation.hierarchy.feasibility.LocalFeasibilityEvaluator`
(plan 9.6/9.7/12) -- wires the frozen Local kinodynamic TQC into Global
candidate feasibility scoring.

Requirement E's exact gap: ``feasibility.py`` only ever defined the
Protocol; every production call site (``training/train_hierarchical_dqn.py``,
``evaluation/long_horizon_benchmark.py``, ``nodes/hierarchical_navigation_node.py``)
hardcoded ``local_evaluator=None``, so ``predicted_action_risk``/
``progress_preserving`` were always zero-filled even for ablations E/F/G that
declare those columns part of their observation. This module is the
implementation those call sites now construct and pass in.

This is the ONE production ``LocalFeasibilityEvaluator`` implementation --
a prior, never-instantiated duplicate
(``navigation.local_rl.feasibility_evaluator.LiveLocalFeasibilityEvaluator``)
shared this module's exact coordinate-frame defect (item 1) and has been
removed rather than fixed in parallel (defect-fix item 7: "중복된
feasibility evaluator 구현을 production 정본 하나로 통합").

Two-phase call contract (defect-fix item 6 -- candidate-order-dependent
temporal contamination): a caller scoring N candidates for ONE Global
decision must call :meth:`capture_context` EXACTLY ONCE, then
:meth:`evaluate` once per candidate, all sharing that SAME
:class:`LocalSensorSnapshot`. This guarantees every candidate in the
decision sees an IDENTICAL LiDAR frame history and previous action -- only
``candidate`` (and therefore the subgoal) varies across calls. The snapshot
itself is a frozen copy of the REAL control loop's current temporal state
(``LocalPolicyController.snapshot_temporal_context``), never a second,
independently-accumulated frame stack that this evaluator would otherwise
have to feed itself one scan at a time (the OLD design: see git history --
that fed the SAME current-tick scan into its own private FrameStack once
per candidate, shifting the temporal window by one extra repeated frame
per candidate and never reflecting the real controller's actual recent
history or its real ``prev_action``).

Coordinate-frame contract (defect-fix item 1): the candidate endpoint is
converted mission-frame -> robot-frame via ``MissionFrame.mission_to_robot``,
and the ``RobotState`` passed alongside it is pinned to the origin via
``LocalPolicyController.robot_relative_state`` -- see that method's and
``LocalPolicyController.build_observation``'s docstrings for the full
two-contract explanation this evaluator must never violate.

For one Global candidate, :meth:`FrozenLocalFeasibilityEvaluator.evaluate`:

1. Converts the candidate's endpoint (mission frame) to the active-subgoal
   convention the Local policy already expects (robot frame) via
   ``MissionFrame.mission_to_robot`` -- the SAME transform every other
   production Local-inference path uses (``LiveGazeboLocalExecutor.run_option``,
   ``real_policy_node.py``).
2. Builds a Local observation with THAT subgoal from the shared
   :class:`LocalSensorSnapshot` context (never a second, per-candidate
   frame-stack push).
3. Runs the frozen Local policy, bounded by a single-flight timeout
   (``navigation.local_rl.single_flight_worker.SingleFlightThreadWorker``,
   the same discipline ``live_gazebo_executor.py`` already uses for its own
   inference calls) -- a genuinely hung/slow policy can never block Global
   decision-making indefinitely.
4. Reads the Local risk critic's ``predict_risk(obs, action)`` (when the
   agent has one -- ``getattr(..., "predict_risk", None)``, mirrors
   ``live_gazebo_executor.py``'s own optional-attribute check) -- through
   THE SAME bounded single-flight timeout as ``select_action`` (defect-fix
   item 7: a hung/erroring/NaN-producing risk critic must never block
   Global decision-making, and must never be silently reported as
   "risk=0").
5. Decodes the action into a constant-curvature arc endpoint (trajectory
   mode only -- ``[kappa, v_ref, horizon_m]``) and reports
   ``progress_preserving`` as "does this arc's endpoint move strictly closer
   to the candidate subgoal than the robot's current position".

``evaluate()`` returns ``None`` (a genuine fallback, tracked in
``telemetry``, never silently reported as ``predicted_risk=0``/
``progress_preserving=True``) whenever: the caller-supplied sensor snapshot
is missing or stale, action inference times out, action inference raises,
or the raw policy output fails ``LocalPolicyController.validate_action``.
When only the RISK read fails (timeout/error/non-finite) but the action
itself decoded successfully, ``evaluate()`` still returns a real
:class:`LocalActionEvaluation` with ``predicted_risk=None`` -- a risk-read
failure never invalidates an otherwise-valid action evaluation.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from hunter_kinodynamic_rl.config.schema import Profile
from hunter_kinodynamic_rl.env.observation.observation_builder import build_observation, build_robot_state_vector
from hunter_kinodynamic_rl.navigation.global_rl.subgoal_sampler import SubgoalCandidate, candidate_endpoint_mission
from hunter_kinodynamic_rl.navigation.hierarchy.feasibility import LocalActionEvaluation
from hunter_kinodynamic_rl.navigation.local_rl.controller import LocalPolicyController
from hunter_kinodynamic_rl.navigation.local_rl.single_flight_worker import SingleFlightThreadWorker
from hunter_kinodynamic_rl.navigation.mission.mission_frame import MissionFrame, PoseXYYaw
from hunter_kinodynamic_rl.trajectory.action_space import normalized_to_trajectory_command


@dataclass(frozen=True)
class LocalSensorSnapshot:
    """Everything :class:`FrozenLocalFeasibilityEvaluator` needs to build a
    Local observation for a candidate-conditioned subgoal, captured ONCE
    per Global decision by :meth:`FrozenLocalFeasibilityEvaluator.capture_context`
    and shared, unchanged, across every candidate that decision scores
    (defect-fix item 6). ``lidar_frame`` and ``prev_action`` are frozen
    copies of the REAL control loop's own temporal state
    (``LocalPolicyController.snapshot_temporal_context``) -- this evaluator
    never re-derives or re-accumulates LiDAR history of its own."""

    lidar_frame: np.ndarray
    prev_action: Tuple[float, float, float]
    speed_mps: float
    yaw_rate_rps: float
    steering_rad: float
    stamp_monotonic: float


@dataclass
class FeasibilityEvaluatorTelemetry:
    """Diagnostic counters for :class:`FrozenLocalFeasibilityEvaluator` --
    deliberately NEVER folded back into ``predicted_action_risk`` itself
    (requirement E: "fallback을 정상 risk=0으로 기록하지 않는다"). A caller
    surfaces these in its own per-mission/episode telemetry so a run with
    many fallbacks is visibly distinguishable from one where the evaluator
    genuinely always agreed risk was near zero. ``queries`` counts
    :meth:`FrozenLocalFeasibilityEvaluator.capture_context` calls (one per
    Global decision), never per-candidate -- the ``risk_*`` fields count
    per-candidate risk-read outcomes independently of whether the action
    itself evaluated successfully."""

    queries: int = 0
    fallbacks_stale_snapshot: int = 0
    fallbacks_inference_timeout: int = 0
    fallbacks_inference_error: int = 0
    fallbacks_invalid_action: int = 0
    risk_evaluated: int = 0
    risk_timeout: int = 0
    risk_error: int = 0
    risk_non_finite: int = 0

    @property
    def fallback_total(self) -> int:
        return (
            self.fallbacks_stale_snapshot + self.fallbacks_inference_timeout
            + self.fallbacks_inference_error + self.fallbacks_invalid_action
        )

    @property
    def fallback_rate(self) -> float:
        return (self.fallback_total / self.queries) if self.queries > 0 else 0.0

    @property
    def risk_fallback_total(self) -> int:
        return self.risk_timeout + self.risk_error + self.risk_non_finite

    @property
    def risk_fallback_rate(self) -> float:
        denom = self.risk_evaluated + self.risk_fallback_total
        return (self.risk_fallback_total / denom) if denom > 0 else 0.0

    def as_dict(self) -> dict:
        return {
            "queries": self.queries,
            "fallbacks_stale_snapshot": self.fallbacks_stale_snapshot,
            "fallbacks_inference_timeout": self.fallbacks_inference_timeout,
            "fallbacks_inference_error": self.fallbacks_inference_error,
            "fallbacks_invalid_action": self.fallbacks_invalid_action,
            "fallback_total": self.fallback_total,
            "fallback_rate": self.fallback_rate,
            "risk_evaluated": self.risk_evaluated,
            "risk_timeout": self.risk_timeout,
            "risk_error": self.risk_error,
            "risk_non_finite": self.risk_non_finite,
            "risk_fallback_total": self.risk_fallback_total,
            "risk_fallback_rate": self.risk_fallback_rate,
        }


def _progress_preserving(action_arr: np.ndarray, subgoal_robot, profile: Profile) -> bool:
    """``True`` iff the decoded Local action's constant-curvature-arc
    endpoint is strictly closer to ``subgoal_robot`` (robot frame) than the
    robot's current position (the origin, in robot frame). Only defined for
    ``action_space.mode == "trajectory"`` -- the ``legacy_waypoint`` ablation
    has no kappa/v_ref/horizon decode to roll out, so this reports ``False``
    (treated as "not confirmed progress-preserving", never fabricated)."""
    if profile.action_space.mode != "trajectory":
        return False
    cmd = normalized_to_trajectory_command(action_arr, profile.action_space, profile.robot)
    if cmd.v_ref <= 1e-3:
        return False
    if abs(cmd.kappa) < 1e-6:
        dx, dy = cmd.horizon_m, 0.0
    else:
        radius = 1.0 / cmd.kappa
        dtheta = cmd.kappa * cmd.horizon_m
        dx = radius * math.sin(dtheta)
        dy = radius * (1.0 - math.cos(dtheta))
    dist_before = math.hypot(subgoal_robot[0], subgoal_robot[1])
    dist_after = math.hypot(subgoal_robot[0] - dx, subgoal_robot[1] - dy)
    return dist_after < dist_before


class FrozenLocalFeasibilityEvaluator:
    """One instance per running Global decision loop (mirrors
    ``LocalPolicyController``/``LiveGazeboLocalExecutor``'s own "one
    long-lived instance" convention) -- NOT thread-safe for concurrent
    ``evaluate()`` calls; callers must serialize candidate evaluation on
    one thread, exactly like every other Global-decision code path already
    does (one candidate loop, no concurrent evaluation). Holds no
    ``LocalPolicyController`` of its own -- ``capture_context`` reads a
    frozen snapshot supplied by the caller's ``snapshot_provider`` closure
    (which itself reads the REAL control-loop controller's
    ``snapshot_temporal_context()``), so this evaluator can neither mutate
    nor diverge from that controller's actual temporal state."""

    def __init__(
        self, profile: Profile, local_agent, *, inference_timeout_sec: float,
        max_snapshot_age_sec: float,
        snapshot_provider: Callable[[], Optional[LocalSensorSnapshot]],
        telemetry: Optional[FeasibilityEvaluatorTelemetry] = None,
    ) -> None:
        self.profile = profile
        self.local_agent = local_agent
        self.inference_timeout_sec = inference_timeout_sec
        self.max_snapshot_age_sec = max_snapshot_age_sec
        self.snapshot_provider = snapshot_provider
        self.telemetry = telemetry if telemetry is not None else FeasibilityEvaluatorTelemetry()
        self._worker: SingleFlightThreadWorker = SingleFlightThreadWorker()

    def capture_context(self) -> Optional[LocalSensorSnapshot]:
        """Call EXACTLY ONCE per Global decision, before scoring any of its
        candidates -- never per-candidate (that per-candidate re-fetch,
        combined with the old design's per-candidate frame-stack push, was
        the defect-fix item 6 contamination bug). Returns ``None`` (and
        records ``fallbacks_stale_snapshot``, the same counter a per-call
        rejection used to use) when no fresh snapshot is available; the
        caller must then treat EVERY candidate this decision as a
        fallback, never retry per candidate."""
        self.telemetry.queries += 1
        snapshot = self.snapshot_provider()
        if snapshot is None:
            self.telemetry.fallbacks_stale_snapshot += 1
            return None
        age_sec = time.monotonic() - snapshot.stamp_monotonic
        if age_sec > self.max_snapshot_age_sec:
            self.telemetry.fallbacks_stale_snapshot += 1
            return None
        return snapshot

    def evaluate(
        self, candidate: SubgoalCandidate, robot_pose_mission: PoseXYYaw, context: LocalSensorSnapshot,
    ) -> Optional[LocalActionEvaluation]:
        """``context`` MUST be the SAME object returned by a single
        preceding :meth:`capture_context` call, shared by every candidate
        in this Global decision -- only ``candidate`` (and therefore the
        subgoal) may vary between calls that share one ``context``."""
        endpoint_mission = candidate_endpoint_mission(candidate, robot_pose_mission)
        subgoal_robot = MissionFrame.mission_to_robot(endpoint_mission, robot_pose_mission)
        robot_state = LocalPolicyController.robot_relative_state(
            v=context.speed_mps, yaw_rate=context.yaw_rate_rps, steering=context.steering_rad,
        )
        robot_state_vector = build_robot_state_vector(
            robot_state, subgoal_robot[0], subgoal_robot[1], context.prev_action,
            robot_state_dim=self.profile.observation.robot_state_dim,
        )
        observation = build_observation(context.lidar_frame, robot_state_vector)

        raw_action, infer_error, timed_out = self._worker.call(
            lambda obs=observation: self.local_agent.select_action(obs, deterministic=True),
            self.inference_timeout_sec,
        )
        if timed_out:
            self.telemetry.fallbacks_inference_timeout += 1
            return None
        if infer_error is not None:
            self.telemetry.fallbacks_inference_error += 1
            return None
        action_arr = LocalPolicyController.validate_action(raw_action)
        if action_arr is None:
            self.telemetry.fallbacks_invalid_action += 1
            return None

        predicted_risk = self._predict_risk_bounded(observation, action_arr)

        progress_preserving = _progress_preserving(action_arr, subgoal_robot, self.profile)
        return LocalActionEvaluation(
            action=action_arr, predicted_risk=predicted_risk, progress_preserving=progress_preserving,
        )

    def _predict_risk_bounded(self, observation: np.ndarray, action_arr: np.ndarray) -> Optional[float]:
        """Defect-fix item 7: ``predict_risk`` gets the SAME bounded
        single-flight timeout/error protection as ``select_action`` --
        never a direct unbounded call -- and a NaN/Inf result is treated
        exactly like an error (a broken risk read), never coerced into a
        numeric ``0.0`` that would silently read as "confirmed safe"."""
        predict_risk_fn = getattr(self.local_agent, "predict_risk", None)
        if predict_risk_fn is None:
            return None
        risk, error, timed_out = self._worker.call(
            lambda: predict_risk_fn(observation, action_arr), self.inference_timeout_sec,
        )
        if timed_out:
            self.telemetry.risk_timeout += 1
            return None
        if error is not None:
            self.telemetry.risk_error += 1
            return None
        try:
            risk_value = float(risk)
        except (TypeError, ValueError):
            self.telemetry.risk_non_finite += 1
            return None
        if not math.isfinite(risk_value):
            self.telemetry.risk_non_finite += 1
            return None
        self.telemetry.risk_evaluated += 1
        return risk_value
