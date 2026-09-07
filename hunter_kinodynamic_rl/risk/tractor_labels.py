"""Timestamp-aligned cause/time/severity labels for TRACTOR candidates."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from hunter_kinodynamic_rl.rl.replay.sequence_schema import (
    CAUSE_BOUNDARY_UNKNOWN, CAUSE_DYNAMIC, CAUSE_STATIC,
)


@dataclass(frozen=True)
class ObstacleTrackBatch:
    xy_m: np.ndarray  # (O,S,2), realized positions at timestamps
    radius_m: np.ndarray  # (O,)
    cause: np.ndarray  # (O,), static or dynamic
    valid: np.ndarray  # (O,S)


@dataclass(frozen=True)
class CandidateLabels:
    event_observed: np.ndarray
    event_step: np.ndarray
    event_cause: np.ndarray
    censor_step: np.ndarray
    event_label_valid: np.ndarray
    clearance_m: np.ndarray
    clearance_valid: np.ndarray
    stopping_margin_m: np.ndarray
    stopping_margin_valid: np.ndarray
    progress_m: np.ndarray
    progress_valid: np.ndarray
    horizon_mask: np.ndarray


_CAUSE_PRIORITY = {
    CAUSE_STATIC: 0,
    CAUSE_DYNAMIC: 1,
    CAUSE_BOUNDARY_UNKNOWN: 2,
}


def _time_bin(timestamp_sec: float, dt_out_sec: float) -> int:
    return max(0, int(math.ceil(timestamp_sec / dt_out_sec) - 1))


def generate_candidate_labels(
    *,
    timestamps_sec: np.ndarray,
    candidate_xy_m: np.ndarray,
    candidate_speed_mps: np.ndarray,
    candidate_valid: np.ndarray,
    candidate_horizon_sec: np.ndarray,
    obstacle_tracks: ObstacleTrackBatch,
    ego_radius_m: float,
    world_half_extent_m: float,
    goal_xy_m: np.ndarray,
    dt_out_sec: float,
    horizon_steps: int,
    brake_decel_mps2: float,
) -> CandidateLabels:
    """Generate labels from realized fine-substep trajectories.

    Obstacle velocity commands are deliberately absent from this API. The
    only dynamic truth accepted is timestamp-aligned realized position.
    """
    times = np.asarray(timestamps_sec, dtype=np.float64)
    xy = np.asarray(candidate_xy_m, dtype=np.float64)
    speed = np.asarray(candidate_speed_mps, dtype=np.float64)
    valid = np.asarray(candidate_valid, dtype=bool)
    horizon_sec = np.asarray(candidate_horizon_sec, dtype=np.float64)
    goal = np.asarray(goal_xy_m, dtype=np.float64)
    tracks = np.asarray(obstacle_tracks.xy_m, dtype=np.float64)
    radii = np.asarray(obstacle_tracks.radius_m, dtype=np.float64)
    causes = np.asarray(obstacle_tracks.cause, dtype=np.int64)
    track_valid = np.asarray(obstacle_tracks.valid, dtype=bool)
    if times.ndim != 1 or times.size == 0 or np.any(~np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
        raise ValueError("timestamps must be a non-empty strictly increasing finite vector")
    if xy.ndim != 3 or xy.shape[1:] != (times.size, 2):
        raise ValueError("candidate_xy_m must be (K,S,2)")
    k, samples = xy.shape[:2]
    if speed.shape != (k, samples) or valid.shape != (k, samples) or horizon_sec.shape != (k,):
        raise ValueError("candidate speed/valid/horizon shapes do not match")
    if tracks.ndim != 3 or tracks.shape[1:] != (samples, 2):
        raise ValueError("obstacle realized tracks must be (O,S,2)")
    if radii.shape != (tracks.shape[0],) or causes.shape != radii.shape or track_valid.shape != tracks.shape[:2]:
        raise ValueError("obstacle radius/cause/valid shapes do not match")
    if np.any(~np.isin(causes, [CAUSE_STATIC, CAUSE_DYNAMIC])):
        raise ValueError("obstacle causes must be static or dynamic")
    if goal.shape != (2,) or ego_radius_m <= 0.0 or world_half_extent_m <= ego_radius_m:
        raise ValueError("invalid goal/ego/world geometry")
    if dt_out_sec <= 0.0 or horizon_steps <= 0 or brake_decel_mps2 <= 0.0:
        raise ValueError("invalid output timing or braking contract")

    event_observed = np.zeros(k, dtype=bool)
    event_step = np.full(k, -2, dtype=np.int16)
    event_cause = np.full(k, -2, dtype=np.int8)
    censor_step = np.full(k, -2, dtype=np.int16)
    event_label_valid = np.zeros(k, dtype=bool)
    clearance = np.full((k, horizon_steps), np.nan, dtype=np.float32)
    clearance_valid = np.zeros((k, horizon_steps), dtype=bool)
    stopping = np.full((k, horizon_steps), np.nan, dtype=np.float32)
    stopping_valid = np.zeros((k, horizon_steps), dtype=bool)
    progress = np.full(k, np.nan, dtype=np.float32)
    progress_valid = np.zeros(k, dtype=bool)
    horizon_mask = np.zeros((k, horizon_steps), dtype=bool)

    for candidate in range(k):
        last_bin = min(horizon_steps - 1, _time_bin(float(horizon_sec[candidate]), dt_out_sec))
        horizon_mask[candidate, :last_bin + 1] = True
        fine_valid = valid[candidate] & (times <= horizon_sec[candidate] + 1e-9)
        if not fine_valid.any():
            continue
        per_sample = []
        collision_candidates = []
        for sample in np.flatnonzero(fine_valid):
            x, y = xy[candidate, sample]
            cause_clearances = []
            boundary = world_half_extent_m - max(abs(x), abs(y)) - ego_radius_m
            cause_clearances.append((boundary, CAUSE_BOUNDARY_UNKNOWN))
            for obstacle in range(tracks.shape[0]):
                if not track_valid[obstacle, sample]:
                    continue
                distance = np.linalg.norm(xy[candidate, sample] - tracks[obstacle, sample])
                margin = distance - ego_radius_m - radii[obstacle]
                cause_clearances.append((float(margin), int(causes[obstacle])))
            minimum = min(value for value, _ in cause_clearances)
            tied = [cause for value, cause in cause_clearances if abs(value - minimum) <= 1e-9]
            winning_cause = max(tied, key=lambda cause: _CAUSE_PRIORITY[cause])
            per_sample.append((sample, minimum, winning_cause))
            if minimum <= 0.0:
                collision_candidates.append((float(times[sample]), winning_cause, sample))

        for output_bin in range(last_bin + 1):
            in_bin = [
                item for item in per_sample
                if _time_bin(float(times[item[0]]), dt_out_sec) == output_bin
            ]
            if not in_bin:
                continue
            min_clearance = min(item[1] for item in in_bin)
            closest_samples = [item[0] for item in in_bin if abs(item[1] - min_clearance) <= 1e-9]
            matching_speed = max(float(speed[candidate, sample]) for sample in closest_samples)
            clearance[candidate, output_bin] = min_clearance
            clearance_valid[candidate, output_bin] = True
            stopping[candidate, output_bin] = (
                min_clearance - matching_speed * matching_speed / (2.0 * brake_decel_mps2)
            )
            stopping_valid[candidate, output_bin] = True

        final_sample = max(item[0] for item in per_sample)
        initial_distance = float(np.linalg.norm(goal - xy[candidate, 0]))
        final_distance = float(np.linalg.norm(goal - xy[candidate, final_sample]))
        progress[candidate] = initial_distance - final_distance
        progress_valid[candidate] = bool(valid[candidate, 0])
        event_label_valid[candidate] = True
        if collision_candidates:
            earliest_time = min(item[0] for item in collision_candidates)
            tied = [item for item in collision_candidates if abs(item[0] - earliest_time) <= 1e-9]
            winner = max(tied, key=lambda item: _CAUSE_PRIORITY[item[1]])
            event_bin = min(last_bin, _time_bin(winner[0], dt_out_sec))
            event_observed[candidate] = True
            event_step[candidate] = event_bin
            event_cause[candidate] = winner[1]
            censor_step[candidate] = event_bin
        else:
            event_step[candidate] = -1
            event_cause[candidate] = -1
            censor_step[candidate] = last_bin

    return CandidateLabels(
        event_observed, event_step, event_cause, censor_step, event_label_valid,
        clearance, clearance_valid, stopping, stopping_valid,
        progress, progress_valid, horizon_mask,
    )
