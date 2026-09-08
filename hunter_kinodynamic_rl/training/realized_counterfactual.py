"""Post-episode counterfactual labels against timestamp-aligned realized tracks.

The ego future remains a counterfactual Ackermann rollout.  Obstacles are not
extrapolated from a velocity estimate: every horizon bin is compared against
the privileged world-frame obstacle snapshot actually observed at that future
simulator timestamp.  These columns are labels only and never policy inputs.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import MutableMapping, Sequence

import numpy as np

from hunter_kinodynamic_rl.dynamics.ackermann_rollout import (
    rollout_tractor_output_grid, rollout_trajectory_command,
)
from hunter_kinodynamic_rl.rl.networks.tractor import TractorConfig
from hunter_kinodynamic_rl.rl.networks.tractor.nominal_rollout_adapter import (
    nominal_rollout_source_fingerprint,
)
from hunter_kinodynamic_rl.rl.replay.sequence_schema import CORE_STEP_FIELDS
from hunter_kinodynamic_rl.risk.stopping_margin import stopping_margin_m
from hunter_kinodynamic_rl.robot.interface import VehicleState


LABEL_SOURCE = "realized_timestamp_aligned_counterfactual_v1"
DENSE_LABEL_GENERATOR = "tractor_privileged_dense_bev_v1"


def _canonical_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _array_sha256(items: Sequence[object]) -> str:
    digest = hashlib.sha256()
    for item in items:
        array = np.ascontiguousarray(np.asarray(item))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def candidate_decoder_fingerprint(action_contract_sha256: str, robot_sha256: str) -> str:
    """Bind candidate decoding to both the action and effective robot contracts."""
    for name, value in (
        ("action contract", action_contract_sha256), ("robot", robot_sha256),
    ):
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"{name} fingerprint must be a lowercase SHA-256 digest")
    return _canonical_sha256({
        "schema": "normalized_trajectory_to_kappa_vref_L_v1",
        "action_contract_sha256": action_contract_sha256,
        "effective_robot_sha256": robot_sha256,
    })


def realized_label_generator_fingerprint() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def base_transition_fingerprint(row: MutableMapping) -> str:
    """Fingerprint the ordered base transition without making relabeling brittle.

    Development callers may use the relabeler with the minimal fields needed to
    exercise the geometric label logic.  Formal episode validation independently
    requires every ``CORE_STEP_FIELDS`` column, so recording explicit missing-field
    markers here preserves that useful API while keeping the digest unambiguous.
    """
    digest = hashlib.sha256(b"tractor-base-transition-row-v1")
    for name in CORE_STEP_FIELDS:
        digest.update(name.encode("utf-8"))
        if name not in row:
            digest.update(b"\x00missing")
            continue
        digest.update(b"\x01present")
        array = np.ascontiguousarray(np.asarray(row[name]))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _default_lineage(profile, config: TractorConfig) -> dict[str, str]:
    source_hash = realized_label_generator_fingerprint()
    action_hash = _canonical_sha256({
        "schema": "trajectory_kappa_vref_L_v1", "min_speed_mps": config.min_speed_mps,
        "max_speed_mps": config.max_speed_mps, "min_arc_m": config.min_arc_m,
        "max_arc_m": config.max_arc_m, "steering_limit_rad": config.steering_limit_rad,
    })
    robot_hash = _canonical_sha256(dataclasses.asdict(profile.robot))
    return {
        "candidate_action_contract_sha256": action_hash,
        "candidate_trajectory_contract_sha256": _canonical_sha256(
            dataclasses.asdict(profile.trajectory)
        ),
        "candidate_decoder_sha256": candidate_decoder_fingerprint(action_hash, robot_hash),
        "candidate_execution_sha256": nominal_rollout_source_fingerprint(),
        "candidate_robot_sha256": robot_hash,
        "candidate_scenario_sha256": "0" * 64,
        "candidate_label_generator_sha256": source_hash,
        "candidate_source_artifact_sha256": "0" * 64,
        "rollout_source_sha256": nominal_rollout_source_fingerprint(),
    }


def _grid_world_centers(decision_pose: np.ndarray, config: TractorConfig) -> tuple[np.ndarray, np.ndarray]:
    x = config.x_min_m + (np.arange(config.grid_width) + 0.5) * config.resolution_m
    y = config.y_min_m + (np.arange(config.grid_height) + 0.5) * config.resolution_m
    ego_x, ego_y = np.meshgrid(x, y)
    c, s = math.cos(float(decision_pose[2])), math.sin(float(decision_pose[2]))
    return (
        float(decision_pose[0]) + c * ego_x - s * ego_y,
        float(decision_pose[1]) + s * ego_x + c * ego_y,
    )


def _rasterize_snapshot(
    snapshot: MutableMapping, decision_pose: np.ndarray, config: TractorConfig,
) -> tuple[np.ndarray, np.ndarray]:
    shape = (config.grid_height, config.grid_width)
    target = np.full(shape, 3, dtype=np.int8)
    valid = np.zeros(shape, dtype=bool)
    if (
        not bool(snapshot.get("privileged_snapshot_valid", False))
        or np.asarray(decision_pose).shape != (3,)
    ):
        return target, valid
    world_x, world_y = _grid_world_centers(decision_pose, config)
    half_extent = float(snapshot["privileged_world_half_extent_m"])
    inside = (np.abs(world_x) < half_extent) & (np.abs(world_y) < half_extent)
    target[inside] = 0
    valid[inside] = True
    obstacles = np.asarray(snapshot["privileged_obstacles_world"], dtype=np.float64).reshape(-1, 4)
    # Static first, dynamic second: dynamic has deterministic tie priority.
    for cause in (0, 1):
        for obstacle_x, obstacle_y, radius, obstacle_cause in obstacles:
            if int(obstacle_cause) != cause:
                continue
            occupied = inside & (
                (world_x - obstacle_x) ** 2 + (world_y - obstacle_y) ** 2 <= radius ** 2
            )
            target[occupied] = 1 + cause
    return target, valid


def _rasterize_dynamic_flow(
    source: MutableMapping, following: MutableMapping | None,
    decision_pose: np.ndarray, config: TractorConfig,
) -> tuple[np.ndarray, np.ndarray]:
    flow = np.full((2, config.grid_height, config.grid_width), np.nan, dtype=np.float32)
    valid = np.zeros((config.grid_height, config.grid_width), dtype=bool)
    if following is None or not (
        bool(source.get("privileged_snapshot_valid", False))
        and bool(following.get("privileged_snapshot_valid", False))
    ):
        return flow, valid
    dt = float(following["privileged_snapshot_timestamp_sec"]) - float(
        source["privileged_snapshot_timestamp_sec"]
    )
    before = np.asarray(source["privileged_obstacles_world"], dtype=np.float64).reshape(-1, 4)
    after = np.asarray(following["privileged_obstacles_world"], dtype=np.float64).reshape(-1, 4)
    if not math.isfinite(dt) or dt <= 0.0 or before.shape != after.shape:
        return flow, valid
    world_x, world_y = _grid_world_centers(decision_pose, config)
    c, s = math.cos(float(decision_pose[2])), math.sin(float(decision_pose[2]))
    for left, right in zip(before, after):
        if int(left[3]) != 1 or int(right[3]) != 1 or not math.isclose(left[2], right[2], abs_tol=1e-5):
            continue
        dx, dy = (right[0] - left[0]) / dt, (right[1] - left[1]) / dt
        occupied = (world_x - left[0]) ** 2 + (world_y - left[1]) ** 2 <= left[2] ** 2
        flow[0, occupied] = c * dx + s * dy
        flow[1, occupied] = -s * dx + c * dy
        valid[occupied] = True
    return flow, valid


def _tube_oob_targets(row: MutableMapping, profile, config: TractorConfig) -> tuple[np.ndarray, np.ndarray]:
    k, h = config.num_candidates, config.horizon_steps
    oob = np.zeros((k, h), dtype=np.float32)
    valid = np.zeros((k, h), dtype=bool)
    observation = np.asarray(row["observation"], dtype=np.float64)
    tail = observation[config.t_obs * config.n_scan :]
    initial = VehicleState(v=float(tail[5]), steering=float(tail[7]))
    physical = np.asarray(row["candidate_trajectories_physical"], dtype=np.float64)
    present = np.asarray(row["candidate_present"], dtype=bool)
    model_valid = np.asarray(row["candidate_model_valid"], dtype=bool)
    samples = config.sparse_tube_samples
    ids = np.arange(samples, dtype=np.float64)
    angle = ids * (math.pi * (3.0 - math.sqrt(5.0)))
    radius = np.sqrt((ids + 0.5) / samples) * config.footprint_radius_m
    offsets = np.stack((radius * np.cos(angle), radius * np.sin(angle)), axis=-1)
    x_max = config.x_min_m + config.grid_width * config.resolution_m
    y_max = config.y_min_m + config.grid_height * config.resolution_m
    dynamics = dataclasses.replace(
        profile.dynamics, dt_sec=config.dt_dyn_sec,
        horizon_min_sec=config.min_horizon_sec,
        horizon_max_sec=config.max_horizon_sec,
        min_safety_horizon_sec=config.min_safety_horizon_sec,
    )
    for candidate in range(k):
        if not (present[candidate] and model_valid[candidate]):
            continue
        kappa, speed, arc = physical[candidate]
        rollout, mask = rollout_tractor_output_grid(
            initial, float(kappa), float(speed), float(arc), profile.robot, dynamics,
            horizon_steps=h, dt_out_sec=config.dt_out_sec,
        )
        for step, (point, active) in enumerate(zip(rollout.points, mask)):
            if not active:
                continue
            c, s = math.cos(point.state.yaw), math.sin(point.state.yaw)
            sample_x = point.state.x + c * offsets[:, 0] - s * offsets[:, 1]
            sample_y = point.state.y + s * offsets[:, 0] + c * offsets[:, 1]
            inside = (
                (sample_x >= config.x_min_m) & (sample_x < x_max)
                & (sample_y >= config.y_min_m) & (sample_y < y_max)
            )
            oob[candidate, step] = 1.0 - float(inside.mean())
            valid[candidate, step] = True
    return oob, valid


def _attach_dense_supervision(
    rows: Sequence[MutableMapping], row_index: int, profile, config: TractorConfig,
    tolerance: float, lineage: dict[str, str],
) -> None:
    row = rows[row_index]
    h = config.horizon_steps
    decision_pose = np.asarray(row.get("privileged_ego_pose_world", []), dtype=np.float64)
    current_class, current_valid = _rasterize_snapshot(row, decision_pose, config)
    following = rows[row_index + 1] if row_index + 1 < len(rows) else None
    current_flow, current_flow_valid = _rasterize_dynamic_flow(
        row, following, decision_pose, config,
    )
    future_class = np.full(
        (h, config.grid_height, config.grid_width), 3, dtype=np.int8,
    )
    future_valid = np.zeros_like(future_class, dtype=bool)
    future_flow = np.full(
        (h, 2, config.grid_height, config.grid_width), np.nan, dtype=np.float32,
    )
    future_flow_valid = np.zeros(
        (h, config.grid_height, config.grid_width), dtype=bool,
    )
    source_items: list[object] = [
        row.get("privileged_snapshot_timestamp_sec"), decision_pose,
        row.get("privileged_obstacles_world"),
    ]
    start_timestamp = float(row.get("privileged_snapshot_timestamp_sec", float("nan")))
    if decision_pose.shape == (3,) and math.isfinite(start_timestamp):
        for step in range(h):
            index = _future_snapshot_index(
                rows, row_index, start_timestamp + (step + 1) * config.dt_out_sec, tolerance,
            )
            if index is None:
                continue
            future = rows[index]
            future_class[step], future_valid[step] = _rasterize_snapshot(
                future, decision_pose, config,
            )
            next_future = rows[index + 1] if index + 1 < len(rows) else None
            future_flow[step], future_flow_valid[step] = _rasterize_dynamic_flow(
                future, next_future, decision_pose, config,
            )
            source_items.extend((
                future["privileged_snapshot_timestamp_sec"],
                future["privileged_obstacles_world"],
            ))
    tube_oob, tube_valid = _tube_oob_targets(row, profile, config)
    response_target = np.full(3, np.nan, dtype=np.float32)
    response_valid = np.zeros(3, dtype=bool)
    if "next_observation" in row and "next_vehicle_response_valid" in row:
        next_observation = np.asarray(row["next_observation"], dtype=np.float32)
        response_target = next_observation[config.t_obs * config.n_scan + 5 :][:3].copy()
        response_valid = np.asarray(row["next_vehicle_response_valid"], dtype=bool).reshape(3)
        response_valid &= np.isfinite(response_target)
    row.update({
        "current_bev_class_target": current_class,
        "current_bev_class_valid": current_valid,
        "current_dynamic_flow_target": current_flow,
        "current_dynamic_flow_valid": current_flow_valid,
        "future_bev_class_target": future_class,
        "future_bev_class_valid": future_valid,
        "future_dynamic_flow_target": future_flow,
        "future_dynamic_flow_valid": future_flow_valid,
        "vehicle_response_target": response_target,
        "vehicle_response_target_valid": response_valid,
        "tube_oob_mass_target": tube_oob,
        "tube_coverage_valid": tube_valid,
        "base_transition_row_sha256": np.asarray(base_transition_fingerprint(row)),
        "label_frame_id": np.asarray("decision_ego_se2_v1"),
        "label_generator_version": np.asarray(DENSE_LABEL_GENERATOR),
        "label_source_timestamp_ns": np.int64(row.get(
            "decision_timestamp_ns", round(start_timestamp * 1e9),
        )),
        "label_source_sha256": np.asarray(_array_sha256(source_items)),
        **{name: np.asarray(value) for name, value in lineage.items()},
    })


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
    *, lineage: dict[str, str] | None = None,
) -> dict[str, int | float | str]:
    if not rows:
        raise ValueError("cannot realize labels for an empty episode")
    config.validate()
    resolved_lineage = _default_lineage(profile, config)
    resolved_lineage.update(lineage or {})
    if any(
        len(str(value)) != 64 or any(character not in "0123456789abcdef" for character in str(value))
        for value in resolved_lineage.values()
    ):
        raise ValueError("formal label lineage values must be lowercase SHA-256 digests")
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
        _attach_dense_supervision(
            rows, row_index, profile, config, tolerance, resolved_lineage,
        )
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
