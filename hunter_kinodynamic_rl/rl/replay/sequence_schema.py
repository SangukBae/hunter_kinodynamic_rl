"""Versioned, fail-closed episode schema for TRACTOR sequence replay."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
from typing import Mapping

import numpy as np


SCHEMA_ID = "tractor_sequence_v2"
SEQUENCE_CONTRACT = "reset_prefix_exact_v1"
CAUSE_STATIC = 0
CAUSE_DYNAMIC = 1
CAUSE_BOUNDARY_UNKNOWN = 2


class TerminationReason(str, Enum):
    NONE = "none"
    GOAL = "goal"
    COLLISION = "collision"
    TIME_LIMIT = "time_limit"
    OPERATOR_STOP = "operator_stop"
    GUARD_STOP = "guard_stop"
    RESET = "reset"
    SENSOR_FAILURE = "sensor_failure"
    LOCALIZATION_FAILURE = "localization_failure"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"


BOOTSTRAP_TRUNCATIONS = frozenset({TerminationReason.TIME_LIMIT.value})
INVALID_TRUNCATIONS = frozenset({
    TerminationReason.OPERATOR_STOP.value,
    TerminationReason.GUARD_STOP.value,
    TerminationReason.RESET.value,
    TerminationReason.SENSOR_FAILURE.value,
    TerminationReason.LOCALIZATION_FAILURE.value,
    TerminationReason.INFRASTRUCTURE_FAILURE.value,
})


@dataclass(frozen=True)
class EpisodeHeader:
    episode_id: str
    scenario_id: str
    split_id: str
    seed: int
    software_commit: str
    dirty_state_digest: str
    container_image_digest: str
    resolved_config_hash: str
    protocol_version: str
    environment_attestation_hash: str
    robot_attestation_hash: str
    observation_contract_hash: str
    action_contract_hash: str
    trajectory_contract_hash: str
    start_utc: str
    end_utc: str
    termination_reason: str
    step_count: int
    sensor_source: str
    localization_source: str
    controller_source: str
    clock_domain: str
    group_id: str = ""
    scenario_family: str = ""
    obstacle_contract: str = ""
    scenario_geometry_sha256: str = ""
    schema_id: str = SCHEMA_ID

    def validate(self) -> None:
        if self.schema_id != SCHEMA_ID:
            raise ValueError(f"expected schema_id={SCHEMA_ID}, got {self.schema_id}")
        if self.split_id not in {"development", "calibration", "locked_test"}:
            raise ValueError("split_id must be development, calibration, or locked_test")
        if not self.episode_id or "/" in self.episode_id or ".." in self.episode_id:
            raise ValueError("episode_id must be a non-empty path-safe identifier")
        if not self.group_id or "/" in self.group_id or ".." in self.group_id:
            raise ValueError("group_id must be a non-empty path-safe split-group identifier")
        if self.obstacle_contract not in {"static_only", "dynamic"}:
            raise ValueError("obstacle_contract must be static_only or dynamic")
        if not self.scenario_family.strip():
            raise ValueError("scenario_family must be non-empty")
        if len(self.scenario_geometry_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.scenario_geometry_sha256
        ):
            raise ValueError("scenario_geometry_sha256 must be a lowercase SHA-256 digest")
        if self.group_id != self.scenario_geometry_sha256:
            raise ValueError("group_id must equal scenario_geometry_sha256 in sequence v2")
        if self.step_count <= 0:
            raise ValueError("step_count must be positive")
        required_strings = (
            "scenario_id", "scenario_family", "software_commit", "resolved_config_hash", "protocol_version",
            "environment_attestation_hash", "robot_attestation_hash",
            "observation_contract_hash", "action_contract_hash", "trajectory_contract_hash",
            "start_utc", "end_utc", "sensor_source", "localization_source",
            "controller_source", "clock_domain",
        )
        for name in required_strings:
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty")
        if self.termination_reason not in {reason.value for reason in TerminationReason}:
            raise ValueError(f"unknown termination_reason {self.termination_reason!r}")

    def canonical_json(self) -> str:
        self.validate()
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


CORE_STEP_FIELDS = (
    "decision_timestamp_ns", "observation", "scan_valid", "motion_delta_from_previous",
    "motion_valid", "previous_intent_valid", "previous_command_published",
    "previous_command_valid", "vehicle_response_valid", "pose_covariance",
    "localization_valid", "localization_confidence", "localization_confidence_valid",
    "sensor_freshness_sec",
    "sensor_freshness_valid", "reset_epoch", "scene_reset", "response_reset",
    "action_normalized_requested", "reward", "transition_dt_sec", "discount_factor",
    "next_observation_valid", "terminated", "truncated", "termination_reason",
    "bellman_sample_valid",
)

CANDIDATE_FIELDS = (
    "candidate_actions_normalized", "candidate_trajectories_physical",
    "candidate_present", "candidate_is_stop", "candidate_model_valid",
    "candidate_horizon_mask", "candidate_event_observed", "candidate_event_step",
    "candidate_event_cause", "candidate_censor_step", "candidate_event_label_valid",
    "candidate_clearance_m", "candidate_clearance_valid", "candidate_stopping_margin_m",
    "candidate_stopping_margin_valid", "candidate_progress_m", "candidate_progress_valid",
    "candidate_set_sha256",
)

PRIVILEGED_LABEL_FIELDS = (
    "privileged_snapshot_valid", "privileged_snapshot_timestamp_sec",
    "privileged_ego_pose_world", "privileged_world_half_extent_m",
    "privileged_obstacles_world",
)


def discount_from_dt(dt_sec: np.ndarray, reference_gamma: float, reference_dt_sec: float) -> np.ndarray:
    if not (0.0 < reference_gamma <= 1.0) or reference_dt_sec <= 0.0:
        raise ValueError("invalid reference discount rule")
    dt = np.asarray(dt_sec, dtype=np.float64)
    if np.any(~np.isfinite(dt)) or np.any(dt <= 0.0):
        raise ValueError("transition dt must be finite and positive")
    return np.power(reference_gamma, dt / reference_dt_sec)


def validate_terminal_semantics(
    terminated: bool,
    truncated: bool,
    next_observation_valid: bool,
    reason: str,
    bellman_sample_valid: bool,
) -> None:
    if reason not in {item.value for item in TerminationReason}:
        raise ValueError(f"unknown termination reason {reason!r}")
    if terminated and truncated:
        raise ValueError("terminated and truncated cannot both be true in v1")
    if terminated:
        if reason not in {TerminationReason.GOAL.value, TerminationReason.COLLISION.value}:
            raise ValueError("true terminals require goal or collision reason")
        if not bellman_sample_valid:
            raise ValueError("true terminal samples must remain valid reward-only Bellman samples")
        return
    if truncated:
        if reason in BOOTSTRAP_TRUNCATIONS:
            if not next_observation_valid or not bellman_sample_valid:
                raise ValueError("time-limit truncation must bootstrap from a valid next observation")
            return
        if reason in INVALID_TRUNCATIONS:
            if next_observation_valid or bellman_sample_valid:
                raise ValueError("invalid/infrastructure truncation must be excluded from Bellman loss")
            return
        raise ValueError("truncated row has a reason with no declared handling")
    if reason != TerminationReason.NONE.value:
        raise ValueError("non-terminal row must use termination_reason='none'")
    if not next_observation_valid or not bellman_sample_valid:
        raise ValueError("continuing row requires a valid next observation and Bellman sample")


def validate_candidate_event_tuple(
    *, valid: bool, observed: bool, event_step: int, cause: int, censor_step: int, horizon: int
) -> None:
    if not valid:
        if observed or (event_step, cause, censor_step) != (-2, -2, -2):
            raise ValueError("invalid label must use the canonical (-2,-2,-2) tuple")
        return
    if not 0 <= censor_step < horizon:
        raise ValueError("valid label censor_step is outside candidate horizon")
    if observed:
        if event_step != censor_step or not 0 <= cause <= 2:
            raise ValueError("observed event requires event_step=censor_step and cause 0..2")
    elif event_step != -1 or cause != -1:
        raise ValueError("censored no-event label must use event_step=cause=-1")


def validate_episode_columns(header: EpisodeHeader, columns: Mapping[str, np.ndarray]) -> None:
    header.validate()
    missing = sorted(set(CORE_STEP_FIELDS) - set(columns))
    if missing:
        raise ValueError(f"episode is missing required fields: {missing}")
    arrays = {name: np.asarray(value) for name, value in columns.items()}
    bad_lengths = {name: value.shape for name, value in arrays.items() if value.ndim == 0 or value.shape[0] != header.step_count}
    if bad_lengths:
        raise ValueError(f"all episode arrays must have leading size {header.step_count}: {bad_lengths}")
    timestamps = arrays["decision_timestamp_ns"].astype(np.int64)
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("decision timestamps must be strictly increasing")
    dt = arrays["transition_dt_sec"].astype(np.float64)
    discount = arrays["discount_factor"].astype(np.float64)
    if np.any(~np.isfinite(dt)) or np.any(dt <= 0.0):
        raise ValueError("transition_dt_sec must be finite and positive")
    if np.any(~np.isfinite(discount)) or np.any((discount < 0.0) | (discount > 1.0)):
        raise ValueError("discount_factor must be finite and in [0,1]")
    motion = arrays["motion_delta_from_previous"].astype(np.float64)
    motion_valid = arrays["motion_valid"].astype(bool)
    if motion.shape[:-1] != motion_valid.shape or motion.shape[-1] != 4:
        raise ValueError("motion delta/mask shapes must be (...,4) and matching (...)")
    if np.any(motion[..., 3][motion_valid] <= 0.0):
        raise ValueError("valid motion edges require positive timestamp-derived dt")
    confidence = arrays["localization_confidence"].astype(np.float64)
    confidence_valid = arrays["localization_confidence_valid"].astype(bool)
    if np.any((confidence[confidence_valid] < 0.0) | (confidence[confidence_valid] > 1.0)):
        raise ValueError("valid localization confidence must be in [0,1]")
    reasons = arrays["termination_reason"].astype(str)
    for row in range(header.step_count):
        validate_terminal_semantics(
            bool(arrays["terminated"][row]), bool(arrays["truncated"][row]),
            bool(arrays["next_observation_valid"][row]), str(reasons[row]),
            bool(arrays["bellman_sample_valid"][row]),
        )
    resets = arrays["reset_epoch"].astype(np.int64)
    if np.any(np.diff(resets) < 0):
        raise ValueError("reset_epoch must be monotonically non-decreasing")
    scene_reset = arrays["scene_reset"].astype(bool)
    response_reset = arrays["response_reset"].astype(bool)
    epoch_change = np.r_[True, np.diff(resets) != 0]
    if np.any(epoch_change & ~(scene_reset & response_reset)):
        raise ValueError("each reset epoch must start with both recurrent reset flags")
    candidate_present_fields = set(CANDIDATE_FIELDS) & set(arrays)
    if candidate_present_fields:
        missing_candidate = sorted(set(CANDIDATE_FIELDS) - set(arrays))
        if missing_candidate:
            raise ValueError(f"partial candidate sidecar fields are forbidden: missing {missing_candidate}")
        actions = arrays["candidate_actions_normalized"]
        trajectories = arrays["candidate_trajectories_physical"]
        present = arrays["candidate_present"].astype(bool)
        horizon_mask = arrays["candidate_horizon_mask"].astype(bool)
        if actions.ndim != 3 or actions.shape[-1] != 3 or trajectories.shape != actions.shape:
            raise ValueError("candidate action/trajectory tensors must be matching (N,K,3)")
        n, k = present.shape
        if actions.shape[:2] != (n, k) or horizon_mask.shape[:2] != (n, k):
            raise ValueError("candidate sidecar leading axes do not match")
        horizon = horizon_mask.shape[-1]
        for row in range(n):
            stop = arrays["candidate_is_stop"][row].astype(bool) & present[row]
            if stop.sum() > 1:
                raise ValueError("at most one present candidate may carry stop identity")
            for candidate in range(k):
                validate_candidate_event_tuple(
                    valid=bool(arrays["candidate_event_label_valid"][row, candidate]),
                    observed=bool(arrays["candidate_event_observed"][row, candidate]),
                    event_step=int(arrays["candidate_event_step"][row, candidate]),
                    cause=int(arrays["candidate_event_cause"][row, candidate]),
                    censor_step=int(arrays["candidate_censor_step"][row, candidate]),
                    horizon=horizon,
                )
                if bool(arrays["candidate_event_label_valid"][row, candidate]) and not (
                    present[row, candidate] and bool(arrays["candidate_model_valid"][row, candidate])
                ):
                    raise ValueError("valid candidate label requires present and model-valid candidate")
    privileged_present = set(PRIVILEGED_LABEL_FIELDS) & set(arrays)
    if privileged_present:
        missing_privileged = sorted(set(PRIVILEGED_LABEL_FIELDS) - set(arrays))
        if missing_privileged:
            raise ValueError(
                f"partial privileged label sidecars are forbidden: missing {missing_privileged}"
            )
        valid = arrays["privileged_snapshot_valid"].astype(bool)
        timestamps = arrays["privileged_snapshot_timestamp_sec"].astype(np.float64)
        poses = arrays["privileged_ego_pose_world"].astype(np.float64)
        half_extent = arrays["privileged_world_half_extent_m"].astype(np.float64)
        obstacles = arrays["privileged_obstacles_world"].astype(np.float64)
        if poses.shape != (header.step_count, 3):
            raise ValueError("privileged ego poses must be (N,3)")
        if obstacles.ndim != 3 or obstacles.shape[0] != header.step_count or obstacles.shape[2] != 4:
            raise ValueError("privileged obstacle states must be (N,M,4)")
        if np.any(~np.isfinite(timestamps[valid])) or np.any(~np.isfinite(poses[valid])):
            raise ValueError("valid privileged snapshots require finite timestamp and ego pose")
        if np.any(~np.isfinite(half_extent[valid])) or np.any(half_extent[valid] <= 0.0):
            raise ValueError("valid privileged snapshots require positive finite world extent")
        if np.any(~np.isfinite(obstacles[valid])):
            raise ValueError("valid privileged obstacle states must be finite")
        if obstacles.shape[1] and (
            np.any(obstacles[valid, :, 2] <= 0.0)
            or np.any(~np.isin(obstacles[valid, :, 3].astype(np.int64), (0, 1)))
        ):
            raise ValueError("privileged obstacles require positive radii and static/dynamic cause")
