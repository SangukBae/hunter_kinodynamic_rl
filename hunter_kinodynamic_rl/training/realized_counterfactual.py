"""Post-episode counterfactual labels against timestamp-aligned realized tracks.

The ego future remains a counterfactual Ackermann rollout.  Obstacles are not
extrapolated from a velocity estimate: every horizon bin is compared against
the privileged world-frame obstacle snapshot actually observed at that future
simulator timestamp.  These columns are labels only and never policy inputs.
"""

from __future__ import annotations

import math
from typing import MutableMapping, Sequence

import numpy as np

from hunter_kinodynamic_rl.dynamics.ackermann_rollout import rollout_trajectory_command
from hunter_kinodynamic_rl.rl.networks.tractor import TractorConfig
from hunter_kinodynamic_rl.risk.stopping_margin import stopping_margin_m
from hunter_kinodynamic_rl.robot.interface import VehicleState


LABEL_SOURCE = "realized_timestamp_aligned_counterfactual_v1"


def _interpolate_state(rollout, time_sec: float, initial: VehicleState) -> VehicleState | None:
    if time_sec < 0.0 or not rollout.points:
        return None
    prior_time, prior = 0.0, initial
    for point in rollout.points:
        if abs(point.t_sec - time_sec) <= 1e-8:
            return point.state
        if point.t_sec > time_sec:
            span = point.t_sec - prior_time
            fraction = 0.0 if span <= 0.0 else (time_sec - prior_time) / span
            left, right = prior, point.state
            return VehicleState(
                x=left.x + fraction * (right.x - left.x),
                y=left.y + fraction * (right.y - left.y),
                yaw=left.yaw + fraction * (right.yaw - left.yaw),
                v=left.v + fraction * (right.v - left.v),
                steering=left.steering + fraction * (right.steering - left.steering),
            )
        prior_time, prior = point.t_sec, point.state
    return prior if abs(prior_time - time_sec) <= 1e-8 else None


def _future_snapshot_index(
    rows: Sequence[MutableMapping], start: int, target_timestamp: float, tolerance_sec: float,
) -> int | None:
    best = None
    best_error = math.inf
    for index in range(start + 1, len(rows)):
        row = rows[index]
        if not bool(row.get("privileged_snapshot_valid", False)):
            continue
        timestamp = float(row["privileged_snapshot_timestamp_sec"])
        if timestamp > target_timestamp + tolerance_sec:
            break
        error = abs(timestamp - target_timestamp)
        if error < best_error:
            best, best_error = index, error
    return best if best is not None and best_error <= tolerance_sec else None


def _clearance_and_cause(
    ego_x: float, ego_y: float, ego_radius: float,
    obstacle_rows: np.ndarray, world_half_extent_m: float,
) -> tuple[float, int]:
    entries: list[tuple[float, int]] = []
    for obstacle in np.asarray(obstacle_rows, dtype=np.float64).reshape(-1, 4):
        x, y, radius, cause = obstacle
        entries.append((math.hypot(ego_x - x, ego_y - y) - ego_radius - radius, int(cause)))
    boundary = min(world_half_extent_m - abs(ego_x), world_half_extent_m - abs(ego_y)) - ego_radius
    entries.append((boundary, 2))
    minimum = min(value for value, _cause in entries)
    # Frozen exact-tie priority: boundary > dynamic > static.
    cause = max(cause for value, cause in entries if abs(value - minimum) <= 1e-9)
    return minimum, cause


def relabel_rows(
    rows: Sequence[MutableMapping], profile, config: TractorConfig,
) -> dict[str, int | float | str]:
    if not rows:
        raise ValueError("cannot realize labels for an empty episode")
    config.validate()
    tolerance = max(1e-4, float(profile.runtime.time_delta_sec) * 0.55)
    valid_candidate_labels = observed_events = valid_severity_bins = 0
    for row_index, row in enumerate(rows):
        k, h = config.num_candidates, config.horizon_steps
        event_observed = np.zeros(k, dtype=bool)
        event_step = np.full(k, -2, dtype=np.int16)
        event_cause = np.full(k, -2, dtype=np.int8)
        censor_step = np.full(k, -2, dtype=np.int16)
        event_valid = np.zeros(k, dtype=bool)
        clearance = np.full((k, h), np.nan, dtype=np.float32)
        clearance_valid = np.zeros((k, h), dtype=bool)
        stopping = np.full((k, h), np.nan, dtype=np.float32)
        stopping_valid = np.zeros((k, h), dtype=bool)
        present = np.asarray(row["candidate_present"], dtype=bool)
        model_valid = np.asarray(row["candidate_model_valid"], dtype=bool)
        horizon_mask = np.asarray(row["candidate_horizon_mask"], dtype=bool)
        physical = np.asarray(row["candidate_trajectories_physical"], dtype=np.float64)
        snapshot_valid = bool(row.get("privileged_snapshot_valid", False))
        start_timestamp = float(row.get("privileged_snapshot_timestamp_sec", float("nan")))
        ego_pose = np.asarray(row.get("privileged_ego_pose_world", []), dtype=np.float64)
        observation = np.asarray(row["observation"], dtype=np.float64)
        tail = observation[config.t_obs * config.n_scan :]
        if snapshot_valid and math.isfinite(start_timestamp) and ego_pose.shape == (3,):
            initial = VehicleState(v=float(tail[5]), steering=float(tail[7]))
            cosine, sine = math.cos(float(ego_pose[2])), math.sin(float(ego_pose[2]))
            for candidate in range(k):
                if not (present[candidate] and model_valid[candidate]):
                    continue
                kappa, v_ref, horizon_m = physical[candidate]
                rollout = rollout_trajectory_command(
                    initial, float(kappa), float(v_ref), float(horizon_m),
                    profile.robot, profile.dynamics,
                    commit_blend=profile.features.trajectory_l_preview_blend,
                )
                last_valid = -1
                for horizon_bin in range(h):
                    if not horizon_mask[candidate, horizon_bin]:
                        break
                    relative_time = (horizon_bin + 1) * config.dt_out_sec
                    state = _interpolate_state(rollout, relative_time, initial)
                    future_index = _future_snapshot_index(
                        rows, row_index, start_timestamp + relative_time, tolerance,
                    )
                    if state is None or future_index is None:
                        break
                    future = rows[future_index]
                    world_x = float(ego_pose[0]) + cosine * state.x - sine * state.y
                    world_y = float(ego_pose[1]) + sine * state.x + cosine * state.y
                    value, cause = _clearance_and_cause(
                        world_x, world_y, config.footprint_radius_m,
                        np.asarray(future["privileged_obstacles_world"]),
                        float(future["privileged_world_half_extent_m"]),
                    )
                    clearance[candidate, horizon_bin] = value
                    clearance_valid[candidate, horizon_bin] = True
                    stopping[candidate, horizon_bin] = stopping_margin_m(
                        abs(float(state.v)), value, profile.robot,
                    )
                    stopping_valid[candidate, horizon_bin] = True
                    valid_severity_bins += 1
                    last_valid = horizon_bin
                    if value <= 0.0:
                        event_observed[candidate] = True
                        event_step[candidate] = horizon_bin
                        event_cause[candidate] = cause
                        censor_step[candidate] = horizon_bin
                        break
                if last_valid >= 0:
                    event_valid[candidate] = True
                    valid_candidate_labels += 1
                    if event_observed[candidate]:
                        observed_events += 1
                    else:
                        event_step[candidate] = -1
                        event_cause[candidate] = -1
                        censor_step[candidate] = last_valid
        row.update({
            "candidate_event_observed": event_observed,
            "candidate_event_step": event_step,
            "candidate_event_cause": event_cause,
            "candidate_censor_step": censor_step,
            "candidate_event_label_valid": event_valid,
            "candidate_clearance_m": clearance,
            "candidate_clearance_valid": clearance_valid,
            "candidate_stopping_margin_m": stopping,
            "candidate_stopping_margin_valid": stopping_valid,
            "candidate_label_source": np.asarray(LABEL_SOURCE),
        })
    if valid_candidate_labels == 0:
        raise RuntimeError("realized-track relabeling produced zero valid candidate labels")
    return {
        "label_source": LABEL_SOURCE,
        "rows": len(rows),
        "valid_candidate_labels": valid_candidate_labels,
        "observed_events": observed_events,
        "valid_severity_bins": valid_severity_bins,
        "timestamp_tolerance_sec": tolerance,
    }
