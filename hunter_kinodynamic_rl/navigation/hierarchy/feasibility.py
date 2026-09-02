"""Global-Local feasibility feedback (plan section 9.6/12) -- per-candidate
features describing whether Hunter SE can PHYSICALLY reach a Global
candidate subgoal, not just whether its endpoint is geometrically free.

Two feature groups, computed differently:

- **Geometry-only** (``rollout_collision``, ``steering_saturation_ratio``,
  ``min_clearance_norm``, ``historical_success_rate``): derived purely from
  ``partial_map`` + the candidate's Ackermann arc, reusing
  ``action_mask.py``'s own arc-sweep geometry (never re-implemented here) --
  always available, no local policy needed.
- **Policy-conditioned** (``predicted_action_risk``, ``progress_preserving``):
  require actually asking the LOCAL policy what it would do if this
  candidate were its active subgoal, then evaluating THAT action -- plan
  9.6's hard requirement: "Local risk critic는 [local state, local action]의
  위험 모델이므로 Global [radius, angle]에 직접 적용하지 않는다. candidate
  subgoal로 conditioned된 local action을 먼저 생성한 후 그 action을 평가한다."
  Supplied via the caller-provided :class:`LocalFeasibilityEvaluator`
  Protocol (a live node composes
  ``LocalPolicyController.build_observation`` + the frozen local agent's
  ``select_action``/``predict_risk``); when ``local_evaluator=None`` (the
  ROS-free training loop's ``SimplifiedKinematicLocalExecutor`` stand-in has
  no real local policy to query -- same documented limitation Phase 4 left
  for live Gazebo integration) these two features are zero-filled, never
  fabricated.

Two-phase protocol (defect-fix item 6): :func:`compute_feasibility_features`
calls :meth:`LocalFeasibilityEvaluator.capture_context` EXACTLY ONCE per
call (i.e. once per Global decision), then :meth:`LocalFeasibilityEvaluator.evaluate`
once per candidate, all sharing that ONE context object -- this guarantees
every candidate sees an identical LiDAR frame history/previous action, with
only the candidate's subgoal varying. A ``None`` context (no fresh
snapshot available) makes every candidate this call a policy-conditioned
fallback -- ``evaluate`` is never even attempted per-candidate in that case.

The fallback candidate (``candidate.is_fallback`` -- ``BACKTRACK``/
``STOP_RECOVERY``, ``navigation.global_rl.subgoal_sampler``) is NOT
reachable via the same single constant-curvature-arc fit used for a
forward candidate (``STOP_RECOVERY``'s ``radius_m=0`` divides by zero in
the chord-curvature formula; ``BACKTRACK``'s ``angle_rad=pi`` is outside
the forward hemisphere that formula assumes), so its geometry-only
features come from a direct endpoint clearance/occupancy lookup instead of
an arc sweep. Its policy-conditioned features (``predicted_action_risk``/
``progress_preserving``) go through the EXACT SAME ``local_evaluator``
call as every other candidate -- they used to be hardcoded to
``risk=0.0``/``progress_preserving=True`` regardless of whether the local
action space (forward-only, ``action_space.v_min_mps >= 0``) can actually
make one-tick progress toward a candidate directly behind the robot,
which fabricated a spuriously perfect safety/progress signal for
precisely the "everything else was infeasible" recovery candidate
(defect-fix item 7). Reporting the real (typically ``progress_preserving=False``
for a single-tick check against a rear target) values is the honest
answer given this package's forward-only kinematics; a future multi-tick
U-turn-aware evaluation would be a deliberate, separately-designed
extension, never a silent free upgrade smuggled into this hardcoded path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence, Tuple

import numpy as np

from hunter_kinodynamic_rl.config.schema import ConfigError
from hunter_kinodynamic_rl.navigation.global_rl.action_mask import (
    _ackermann_feasible_arc_points_robot_frame, _rollout_hits_inflated,
)
from hunter_kinodynamic_rl.navigation.global_rl.subgoal_sampler import SubgoalCandidate, candidate_endpoint_mission
from hunter_kinodynamic_rl.navigation.mapping.partial_map import PartialMap
from hunter_kinodynamic_rl.navigation.mapping.visited_map import rasterize_circle_cells
from hunter_kinodynamic_rl.navigation.mission.mission_frame import MissionFrame, PoseXYYaw

FEASIBILITY_FEATURE_NAMES = (
    "rollout_collision", "steering_saturation_ratio", "min_clearance_norm", "predicted_action_risk",
    "progress_preserving", "historical_success_rate",
)
N_FEASIBILITY_FEATURES = len(FEASIBILITY_FEATURE_NAMES)


@dataclass(frozen=True)
class FeasibilityConfig:
    #: Number of interpolated points along a candidate's Ackermann arc used
    #: for the clearance/steering-saturation sweep (mirrors
    #: ``global_rl.rollout_sample_count``'s own adaptive-subdivision
    #: contract -- kept a SEPARATE knob since feasibility scoring can afford
    #: a different sample density than the pass/fail action mask).
    rollout_sample_count: int = 8
    #: Search radius (m) for the min-clearance ring search around each arc
    #: sample -- "no known-occupied cell found within this radius" reports
    #: clearance == this radius (a bounded lower-confidence answer, never a
    #: fabricated infinity).
    clearance_search_radius_m: float = 2.0
    #: Normalizes ``min_clearance_norm`` into ``[0, 1]``.
    clearance_norm_m: float = 2.0
    #: Radius (m) around a candidate's endpoint used to look up historical
    #: local success/failure statistics from ``PartialMap.failure_count``/
    #: ``visited_count`` (plan 9.6: "과거 동일 영역의 local success/failure
    #: 통계").
    historical_stats_radius_m: float = 1.0

    def validate(self) -> None:
        if self.rollout_sample_count < 2:
            raise ConfigError("feasibility.rollout_sample_count must be >= 2")
        if self.clearance_search_radius_m <= 0.0:
            raise ConfigError("feasibility.clearance_search_radius_m must be > 0")
        if self.clearance_norm_m <= 0.0:
            raise ConfigError("feasibility.clearance_norm_m must be > 0")
        if self.historical_stats_radius_m <= 0.0:
            raise ConfigError("feasibility.historical_stats_radius_m must be > 0")


@dataclass(frozen=True)
class LocalActionEvaluation:
    action: np.ndarray
    predicted_risk: Optional[float]
    progress_preserving: bool


@dataclass
class FeasibilityEvaluatorTelemetry:
    """Mutable counters distinguishing a REAL local-policy evaluation from
    every reason a candidate's policy-conditioned features
    (``predicted_action_risk``/``progress_preserving``) fell back to the
    neutral default -- requirement E: "fallback을 정상 risk=0으로 기록하지
    않는다". ``fallback_no_evaluator_count`` is incremented by
    :func:`compute_feasibility_features` itself (no
    :class:`LocalFeasibilityEvaluator` was wired at all -- e.g. the
    ROS-free ``SimplifiedKinematicLocalExecutor`` stand-in, a documented,
    non-fatal limitation); every other field is owned by the concrete
    evaluator implementation (see
    ``navigation.hierarchy.local_feasibility_evaluator.FrozenLocalFeasibilityEvaluator``,
    the ONE production implementation of this Protocol).
    A caller that wires a real evaluator should pass the SAME telemetry
    instance to both the evaluator's own constructor and
    :func:`compute_feasibility_features`'s ``telemetry`` kwarg so every
    fallback reason lands in one place. Never reset automatically --
    callers own the reset cadence (per-mission, per-benchmark-run, ...)."""

    evaluated_count: int = 0
    fallback_no_evaluator_count: int = 0
    skipped_no_snapshot_count: int = 0
    skipped_stale_snapshot_count: int = 0
    skipped_inference_failed_count: int = 0
    error_count: int = 0

    def reset(self) -> None:
        for f in (
            "evaluated_count", "fallback_no_evaluator_count", "skipped_no_snapshot_count",
            "skipped_stale_snapshot_count", "skipped_inference_failed_count", "error_count",
        ):
            setattr(self, f, 0)

    def as_dict(self) -> dict:
        return {
            "evaluated_count": self.evaluated_count,
            "fallback_no_evaluator_count": self.fallback_no_evaluator_count,
            "skipped_no_snapshot_count": self.skipped_no_snapshot_count,
            "skipped_stale_snapshot_count": self.skipped_stale_snapshot_count,
            "skipped_inference_failed_count": self.skipped_inference_failed_count,
            "error_count": self.error_count,
        }


class LocalFeasibilityEvaluator(Protocol):
    """Builds a LOCAL observation with ``candidate`` as the active subgoal,
    runs the (frozen) local policy, and evaluates the resulting action --
    NEVER receives a bare ``[radius, angle]`` polar coordinate as if it were
    a trajectory action. Two-phase per Global decision (defect-fix item 6):
    :meth:`capture_context` once, then :meth:`evaluate` once per candidate,
    all sharing that one context -- never re-derive/re-fetch sensor state
    inside a per-candidate loop."""

    def capture_context(self) -> Optional[object]:
        """Call exactly once per Global decision, before any ``evaluate``
        call for that decision's candidates. Returns ``None`` when no
        evaluation is possible this decision (e.g. sensor not ready) -- in
        that case every candidate must be treated as a fallback without
        ``evaluate`` being called at all."""
        ...

    def evaluate(
        self, candidate: SubgoalCandidate, robot_pose_mission: PoseXYYaw, context: object,
    ) -> Optional[LocalActionEvaluation]:
        """``context`` MUST be the object returned by the ONE preceding
        ``capture_context`` call for this decision."""
        ...


def _clearance_m(partial_map: PartialMap, inflated: np.ndarray, x: float, y: float, search_radius_m: float) -> float:
    cell = partial_map.world_to_cell(x, y)
    if cell is None:
        return search_radius_m
    max_radius_cells = int(math.ceil(search_radius_m / partial_map.resolution_m))
    for radius_cells in range(0, max_radius_cells + 1):
        ring_cells = rasterize_circle_cells(cell[0], cell[1], radius_cells)
        for r, c in ring_cells:
            if partial_map.in_bounds(r, c) and inflated[r, c]:
                return radius_cells * partial_map.resolution_m
    return search_radius_m


def _historical_success_rate(
    partial_map: PartialMap, visited_count: np.ndarray, failure_count: np.ndarray,
    x: float, y: float, radius_m: float,
) -> float:
    cell = partial_map.world_to_cell(x, y)
    if cell is None:
        return 0.5
    radius_cells = max(1, int(math.ceil(radius_m / partial_map.resolution_m)))
    total_visits = 0
    total_failures = 0
    for r, c in rasterize_circle_cells(cell[0], cell[1], radius_cells):
        if not partial_map.in_bounds(r, c):
            continue
        total_visits += int(visited_count[r, c])
        total_failures += int(failure_count[r, c])
    denom = total_visits + total_failures
    if denom <= 0:
        return 0.5
    return float(np.clip(1.0 - total_failures / denom, 0.0, 1.0))


def compute_feasibility_features(
    candidates: Sequence[SubgoalCandidate], partial_map: PartialMap, robot_pose_mission: PoseXYYaw,
    robot_max_curvature: float, config: FeasibilityConfig, *,
    local_evaluator: Optional[LocalFeasibilityEvaluator] = None,
    telemetry: Optional[FeasibilityEvaluatorTelemetry] = None,
) -> np.ndarray:
    """``(n_candidates, N_FEASIBILITY_FEATURES)`` float32 array, same index
    order as ``candidates`` (mirrors every other Global candidate-tensor
    contract). The fallback candidate always reports the benign/neutral
    values (never blocked, never saturated, unknown historical rate)."""
    n = len(candidates)
    out = np.zeros((n, N_FEASIBILITY_FEATURES), dtype=np.float32)
    snapshot = partial_map.snapshot()
    channels = snapshot.channels
    inflated = channels.inflated

    # defect-fix item 6: ONE context for the whole decision (every
    # candidate below shares it), never re-fetched per candidate.
    context = None
    if local_evaluator is not None:
        context = local_evaluator.capture_context()

    def _policy_conditioned(candidate: SubgoalCandidate) -> Tuple[float, float]:
        predicted_action_risk = 0.0
        progress_preserving = 0.0
        if local_evaluator is None:
            if telemetry is not None:
                telemetry.fallback_no_evaluator_count += 1
            return predicted_action_risk, progress_preserving
        if context is None:
            # capture_context() already recorded the specific fallback
            # reason for this decision -- do not attempt evaluate() at all
            # and do not double-count telemetry per candidate.
            return predicted_action_risk, progress_preserving
        try:
            evaluation = local_evaluator.evaluate(candidate, robot_pose_mission, context)
        except Exception:  # noqa: BLE001 -- a misbehaving evaluator must never abort Global decision-making
            evaluation = None
            if telemetry is not None:
                telemetry.error_count += 1
        if evaluation is not None:
            if evaluation.predicted_risk is not None and math.isfinite(evaluation.predicted_risk):
                predicted_action_risk = float(np.clip(evaluation.predicted_risk, 0.0, 1.0))
            progress_preserving = 1.0 if evaluation.progress_preserving else 0.0
        return predicted_action_risk, progress_preserving

    for candidate in candidates:
        idx = candidate.index
        if candidate.is_fallback:
            # BACKTRACK (angle=pi)/STOP_RECOVERY (radius=0) are not
            # reachable via the same single constant-curvature-arc fit as
            # a forward candidate (radius=0 divides by zero; angle=pi is
            # outside the forward-hemisphere assumption) -- use a direct
            # endpoint clearance/occupancy lookup instead of an arc sweep,
            # and never fabricate "always clear, zero risk, guaranteed
            # progress" (module docstring, defect-fix item 7).
            endpoint_mission = candidate_endpoint_mission(candidate, robot_pose_mission)
            clearance_m = _clearance_m(
                partial_map, inflated, endpoint_mission[0], endpoint_mission[1], config.clearance_search_radius_m,
            )
            min_clearance_norm = float(np.clip(clearance_m / config.clearance_norm_m, 0.0, 1.0))
            rollout_collision = 1.0 if clearance_m <= 0.0 else 0.0
            # Worst-case difficulty: reaching this endpoint needs a
            # turning maneuver no single forward arc achieves -- never
            # reported as "easy" (steering_saturation_ratio=0.0).
            steering_saturation_ratio = 1.0
            historical_success_rate = _historical_success_rate(
                partial_map, snapshot.visited_count, snapshot.failure_count,
                endpoint_mission[0], endpoint_mission[1], config.historical_stats_radius_m,
            )
            predicted_action_risk, progress_preserving = _policy_conditioned(candidate)
            out[idx] = (
                rollout_collision, steering_saturation_ratio, min_clearance_norm, predicted_action_risk,
                progress_preserving, historical_success_rate,
            )
            continue

        arc_points = _ackermann_feasible_arc_points_robot_frame(
            candidate, robot_max_curvature, config.rollout_sample_count, partial_map.resolution_m * 0.5,
        )
        phi, r = candidate.angle_rad, candidate.radius_m
        sin_phi = math.sin(phi)
        kappa = 0.0 if abs(sin_phi) < 1e-9 else (2.0 / r) * sin_phi
        steering_saturation_ratio = float(np.clip(abs(kappa) / robot_max_curvature, 0.0, 1.0)) \
            if robot_max_curvature > 0.0 else 1.0

        if arc_points is None:
            rollout_collision = 1.0
            min_clearance_norm = 0.0
            steering_saturation_ratio = 1.0
        else:
            rollout_collision = 1.0 if _rollout_hits_inflated(
                partial_map, inflated, robot_pose_mission, arc_points,
            ) else 0.0
            clearances = [
                _clearance_m(
                    partial_map, inflated, *MissionFrame.robot_to_mission(p, robot_pose_mission),
                    config.clearance_search_radius_m,
                )
                for p in arc_points
            ]
            min_clearance_norm = float(np.clip(min(clearances) / config.clearance_norm_m, 0.0, 1.0))

        endpoint_mission = candidate_endpoint_mission(candidate, robot_pose_mission)
        historical_success_rate = _historical_success_rate(
            partial_map, snapshot.visited_count, snapshot.failure_count,
            endpoint_mission[0], endpoint_mission[1], config.historical_stats_radius_m,
        )

        predicted_action_risk, progress_preserving = _policy_conditioned(candidate)

        out[idx] = (
            rollout_collision, steering_saturation_ratio, min_clearance_norm, predicted_action_risk,
            progress_preserving, historical_success_rate,
        )
    return out
