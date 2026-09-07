"""Live Environment-v2 transition recorder for ``tractor_sequence_v2``.

The ROS client remains in :mod:`trainer_base`; this module converts its
step-synchronised observation, command, risk and sensor diagnostics into an
immutable episode chunk.  Privileged candidate summaries are persisted as
training labels and never enter :class:`TractorInputs`.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Mapping

import numpy as np

from hunter_kinodynamic_rl.env.simulation import risk_telemetry as rt
from hunter_kinodynamic_rl.env.simulation import sensor_diagnostics as sd
from hunter_kinodynamic_rl.rl.networks.tractor import TractorConfig
from hunter_kinodynamic_rl.rl.replay.sequence_schema import (
    TerminationReason, discount_from_dt,
)
from hunter_kinodynamic_rl.trajectory.action_space import (
    TrajectoryCommand, decode_action, trajectory_command_to_normalized,
)


LABEL_SOURCE = "nominal_preaction_rollout_summary_v1"


def _wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def _older_to_current(older, current) -> np.ndarray:
    dx, dy = older[0] - current[0], older[1] - current[1]
    c, s = math.cos(current[2]), math.sin(current[2])
    return np.asarray([
        c * dx + s * dy,
        -s * dx + c * dy,
        _wrap_angle(older[2] - current[2]),
    ], dtype=np.float32)


@dataclass(frozen=True)
class TractorObservationSnapshot:
    observation: np.ndarray
    scan_valid: np.ndarray
    motion_delta_from_previous: np.ndarray
    motion_valid: np.ndarray
    decision_timestamp_ns: int
    previous_intent_valid: bool
    previous_command_published: np.ndarray
    previous_command_valid: bool
    vehicle_response_valid: np.ndarray
    pose_covariance: np.ndarray
    localization_valid: bool
    localization_confidence: float
    localization_confidence_valid: bool
    sensor_freshness_sec: float
    sensor_freshness_valid: bool
    scene_reset: bool
    response_reset: bool
    pose_history: tuple[tuple[float, float, float] | None, ...]

    def input_columns(self, prefix: str = "") -> dict[str, np.ndarray | int | float | bool]:
        return {
            f"{prefix}observation": self.observation,
            f"{prefix}scan_valid": self.scan_valid,
            f"{prefix}motion_delta_from_previous": self.motion_delta_from_previous,
            f"{prefix}motion_valid": self.motion_valid,
            f"{prefix}decision_timestamp_ns": self.decision_timestamp_ns,
            f"{prefix}previous_intent_valid": self.previous_intent_valid,
            f"{prefix}previous_command_published": self.previous_command_published,
            f"{prefix}previous_command_valid": self.previous_command_valid,
            f"{prefix}vehicle_response_valid": self.vehicle_response_valid,
            f"{prefix}pose_covariance": self.pose_covariance,
            f"{prefix}localization_valid": self.localization_valid,
            f"{prefix}localization_confidence": self.localization_confidence,
            f"{prefix}localization_confidence_valid": self.localization_confidence_valid,
            f"{prefix}sensor_freshness_sec": self.sensor_freshness_sec,
            f"{prefix}sensor_freshness_valid": self.sensor_freshness_valid,
            f"{prefix}scene_reset": self.scene_reset,
            f"{prefix}response_reset": self.response_reset,
        }


def make_snapshot(
    observation,
    config: TractorConfig,
    *,
    diagnostics: sd.SensorDiagnostics | None,
    pose_covariance,
    previous: TractorObservationSnapshot | None,
    previous_action=None,
    previous_command=None,
    nominal_dt_sec: float = 0.1,
) -> TractorObservationSnapshot:
    state = np.asarray(observation, dtype=np.float32).reshape(-1)
    if state.shape != (config.observation_dim,) or not np.isfinite(state).all():
        raise ValueError(f"Environment v2 observation must be finite ({config.observation_dim},)")
    diagnostic_valid = diagnostics is not None and diagnostics.valid
    timestamp_sec = float(diagnostics.sim_timestamp_sec) if diagnostic_valid else float("nan")
    if not math.isfinite(timestamp_sec):
        timestamp_ns = 0 if previous is None else previous.decision_timestamp_ns + int(nominal_dt_sec * 1e9)
    else:
        timestamp_ns = int(round(timestamp_sec * 1e9))
        if previous is not None and timestamp_ns <= previous.decision_timestamp_ns:
            timestamp_ns = previous.decision_timestamp_ns + int(nominal_dt_sec * 1e9)

    current_scan_valid = np.full(
        config.n_scan,
        bool(diagnostic_valid and diagnostics.lidar_dropout_count == 0),
        dtype=bool,
    )
    if previous is None:
        scan_valid = np.zeros((config.t_obs, config.n_scan), dtype=bool)
        scan_valid[0] = current_scan_valid
        pose_history = (None,) * config.t_obs
    else:
        scan_valid = np.concatenate((current_scan_valid[None], previous.scan_valid[:-1]), axis=0)
        pose = None
        if diagnostic_valid and all(math.isfinite(v) for v in (
            diagnostics.noisy_x, diagnostics.noisy_y, diagnostics.noisy_yaw,
        )):
            pose = (diagnostics.noisy_x, diagnostics.noisy_y, diagnostics.noisy_yaw)
        pose_history = (pose, *previous.pose_history[: config.t_obs - 1])

    motion = np.zeros((config.t_obs - 1, 4), dtype=np.float32)
    motion_valid = np.zeros(config.t_obs - 1, dtype=bool)
    if previous is not None:
        for edge in range(config.t_obs - 1):
            current_pose, older_pose = pose_history[edge], pose_history[edge + 1]
            if current_pose is None or older_pose is None:
                continue
            transform = _older_to_current(older_pose, current_pose)
            if edge == 0:
                dt_sec = (timestamp_ns - previous.decision_timestamp_ns) * 1e-9
            else:
                # The per-frame observation stack is sampled at the control
                # cadence.  Older exact timestamps are unavailable on the
                # legacy observation service and therefore this field is
                # explicitly tied to the measured current cadence.
                dt_sec = (timestamp_ns - previous.decision_timestamp_ns) * 1e-9
            if math.isfinite(dt_sec) and dt_sec > 0.0:
                motion[edge, :3] = transform
                motion[edge, 3] = dt_sec
                motion_valid[edge] = True

    covariance = np.asarray(pose_covariance, dtype=np.float32).reshape(3, 3)
    covariance = 0.5 * (covariance + covariance.T)
    if not np.isfinite(covariance).all() or np.linalg.eigvalsh(covariance).min() < -1e-6:
        covariance = np.zeros((3, 3), dtype=np.float32)
        localization_valid = False
    else:
        localization_valid = diagnostic_valid
    confidence = float(math.exp(-max(0.0, float(np.trace(covariance)))))
    previous_intent_valid = previous_action is not None
    if previous_intent_valid:
        previous_intent = np.asarray(previous_action, dtype=np.float32).reshape(3)
        if not np.isfinite(previous_intent).all():
            raise ValueError("previous action must be finite")
    command = np.zeros(2, dtype=np.float32) if previous_command is None else np.asarray(
        previous_command, dtype=np.float32,
    ).reshape(2)
    return TractorObservationSnapshot(
        observation=state, scan_valid=scan_valid,
        motion_delta_from_previous=motion, motion_valid=motion_valid,
        decision_timestamp_ns=timestamp_ns,
        previous_intent_valid=previous_intent_valid,
        previous_command_published=command,
        previous_command_valid=previous_command is not None,
        vehicle_response_valid=np.full(3, diagnostic_valid, dtype=bool),
        pose_covariance=covariance, localization_valid=localization_valid,
        localization_confidence=confidence,
        localization_confidence_valid=localization_valid,
        sensor_freshness_sec=0.0, sensor_freshness_valid=diagnostic_valid,
        scene_reset=previous is None, response_reset=previous is None,
        pose_history=pose_history,
    )


def termination_semantics(done: bool, target: bool, collision: bool, telemetry: rt.RiskTelemetry):
    if target:
        return True, False, TerminationReason.GOAL.value, True, True
    if collision:
        return True, False, TerminationReason.COLLISION.value, True, True
    if telemetry.sensor_stale:
        return False, True, TerminationReason.SENSOR_FAILURE.value, False, False
    if done:
        return False, True, TerminationReason.TIME_LIMIT.value, True, True
    return False, False, TerminationReason.NONE.value, True, True


def _candidate_columns(
    telemetry: rt.RiskTelemetry,
    requested_action,
    profile,
    config: TractorConfig,
) -> dict[str, np.ndarray]:
    k, h = config.num_candidates, config.horizon_steps
    actions = np.zeros((k, 3), dtype=np.float32)
    physical = np.zeros((k, 3), dtype=np.float32)
    present = np.zeros(k, dtype=bool)
    is_stop = np.zeros(k, dtype=bool)
    model_valid = np.zeros(k, dtype=bool)
    event_observed = np.zeros(k, dtype=bool)
    event_step = np.full(k, -2, dtype=np.int16)
    event_cause = np.full(k, -2, dtype=np.int8)
    censor_step = np.full(k, -2, dtype=np.int16)
    event_valid = np.zeros(k, dtype=bool)
    horizon_mask = np.zeros((k, h), dtype=bool)
    clearance = np.full((k, h), np.nan, dtype=np.float32)
    clearance_valid = np.zeros((k, h), dtype=bool)
    stopping = np.full((k, h), np.nan, dtype=np.float32)
    stopping_valid = np.zeros((k, h), dtype=bool)
    progress = np.full(k, np.nan, dtype=np.float32)
    progress_valid = np.zeros(k, dtype=bool)

    candidates = list(telemetry.candidates[:k])
    if not candidates and telemetry.valid:
        command = decode_action(requested_action, profile.action_space, profile.robot)
        if isinstance(command, TrajectoryCommand):
            candidates = [rt.CandidateTelemetry(
                command.kappa, command.v_ref, command.horizon_m, telemetry.risk_target,
                telemetry.goal_progress_m, telemetry.min_clearance_m, telemetry.ttc_sec,
                telemetry.collision_within_horizon, telemetry.stopping_margin_m, -1,
            )]
    stop_identity_assigned = False
    for index, candidate in enumerate(candidates):
        command = TrajectoryCommand(candidate.kappa, candidate.v_ref, candidate.horizon_m)
        actions[index] = trajectory_command_to_normalized(command, profile.action_space, profile.robot)
        physical[index] = (candidate.kappa, candidate.v_ref, candidate.horizon_m)
        present[index] = True
        candidate_is_stop = candidate.v_ref <= config.min_speed_mps + 1e-6
        is_stop[index] = candidate_is_stop and not stop_identity_assigned
        stop_identity_assigned = stop_identity_assigned or candidate_is_stop
        model_valid[index] = telemetry.valid
        if candidate.v_ref <= 0.05:
            horizon_sec = config.max_horizon_sec
        else:
            horizon_sec = candidate.horizon_m / candidate.v_ref
            horizon_sec = min(config.max_horizon_sec, max(config.min_horizon_sec, horizon_sec))
        horizon_sec = max(horizon_sec, config.min_safety_horizon_sec)
        last_bin = min(h - 1, max(0, int(math.ceil(horizon_sec / config.dt_out_sec) - 1)))
        horizon_mask[index, : last_bin + 1] = True
        valid_summary = telemetry.valid and math.isfinite(candidate.min_clearance_m)
        if not valid_summary:
            continue
        if candidate.collision_within_horizon:
            valid_summary = math.isfinite(candidate.ttc_sec) and candidate.event_cause in (0, 1, 2)
            if not valid_summary:
                continue
            q = min(last_bin, max(0, int(math.ceil(candidate.ttc_sec / config.dt_out_sec) - 1)))
            event_observed[index] = True
            event_step[index] = censor_step[index] = q
            event_cause[index] = candidate.event_cause
        else:
            event_step[index] = event_cause[index] = -1
            censor_step[index] = last_bin
        event_valid[index] = True
        severity_bin = int(censor_step[index])
        clearance[index, severity_bin] = candidate.min_clearance_m
        clearance_valid[index, severity_bin] = True
        if math.isfinite(candidate.stopping_margin_m):
            stopping[index, severity_bin] = candidate.stopping_margin_m
            stopping_valid[index, severity_bin] = True
        if math.isfinite(candidate.goal_progress_m):
            progress[index] = candidate.goal_progress_m
            progress_valid[index] = True

    hashes = np.ascontiguousarray(actions).tobytes() + np.ascontiguousarray(present).tobytes()
    candidate_hash = hashlib.sha256(hashes).hexdigest()
    return {
        "candidate_actions_normalized": actions,
        "candidate_trajectories_physical": physical,
        "candidate_present": present, "candidate_is_stop": is_stop,
        "candidate_model_valid": model_valid, "candidate_horizon_mask": horizon_mask,
        "candidate_event_observed": event_observed, "candidate_event_step": event_step,
        "candidate_event_cause": event_cause, "candidate_censor_step": censor_step,
        "candidate_event_label_valid": event_valid,
        "candidate_clearance_m": clearance, "candidate_clearance_valid": clearance_valid,
        "candidate_stopping_margin_m": stopping,
        "candidate_stopping_margin_valid": stopping_valid,
        "candidate_progress_m": progress, "candidate_progress_valid": progress_valid,
        "candidate_set_sha256": np.asarray(candidate_hash),
        "candidate_label_source": np.asarray(LABEL_SOURCE),
    }


class TractorEpisodeRecorder:
    def __init__(self, profile, model_config: TractorConfig, reference_gamma: float = 0.99):
        self.profile = profile
        self.model_config = model_config
        self.reference_gamma = float(reference_gamma)
        self.rows: list[dict] = []

    def append(
        self,
        current: TractorObservationSnapshot,
        action,
        next_snapshot: TractorObservationSnapshot,
        reward: float,
        done: bool,
        target: bool,
        collision: bool,
        telemetry: rt.RiskTelemetry,
    ) -> None:
        terminated, truncated, reason, next_valid, bellman_valid = termination_semantics(
            done, target, collision, telemetry,
        )
        dt_sec = (next_snapshot.decision_timestamp_ns - current.decision_timestamp_ns) * 1e-9
        if not math.isfinite(dt_sec) or dt_sec <= 0.0:
            raise ValueError("Environment v2 transition timestamp did not advance")
        row = current.input_columns()
        row.update(next_snapshot.input_columns("next_"))
        row.update({
            "reset_epoch": 0,
            "action_normalized_requested": np.asarray(action, dtype=np.float32).reshape(3),
            "reward": np.float32(reward), "transition_dt_sec": np.float32(dt_sec),
            "discount_factor": np.float32(discount_from_dt(
                np.asarray([dt_sec]), self.reference_gamma, self.profile.runtime.time_delta_sec,
            )[0]),
            "next_observation_valid": next_valid, "terminated": terminated,
            "truncated": truncated, "termination_reason": reason,
            "bellman_sample_valid": bellman_valid,
        })
        row.update(_candidate_columns(telemetry, action, self.profile, self.model_config))
        self.rows.append(row)

    def columns(self) -> Mapping[str, np.ndarray]:
        if not self.rows:
            raise ValueError("cannot finalize an empty episode")
        names = set(self.rows[0])
        if any(set(row) != names for row in self.rows):
            raise RuntimeError("episode rows have inconsistent columns")
        return {name: np.asarray([row[name] for row in self.rows]) for name in sorted(names)}
