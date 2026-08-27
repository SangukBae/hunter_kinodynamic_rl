#!/usr/bin/env python3
"""Research-grade run logging (section 10): per-step JSONL, per-episode CSV
summary, TensorBoard scalars, and a metadata.json snapshot -- run directory
layout:

    runs/<run_name>/
      configs/profile_snapshot.json
      checkpoints/{latest,best,final}.{pt,json}, {latest,best,final}_replay.npz
      logs/steps.jsonl, episodes.csv
      evaluation/
      benchmark/
      tensorboard/
      metadata.json

DISCLOSED TRADE-OFF: the per-step JSONL record does NOT include the raw
(88-328-D) state vector by default (``log_full_state=False``) -- for a
multi-hour training run that would dominate disk usage without adding
information beyond what's reconstructible from the episode's action/reward
sequence. Set ``log_full_state=True`` to include it (e.g. for a short
debugging run) -- see docs/RESEARCH_PROTOCOL.md.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
import os
import time
from datetime import datetime
from typing import Dict, Optional

from hunter_kinodynamic_rl.config.schema import Profile
from hunter_kinodynamic_rl.env.simulation import risk_telemetry as rt
from hunter_kinodynamic_rl.env.simulation import sensor_diagnostics as sd
from hunter_kinodynamic_rl.trajectory.action_space import TrajectoryCommand
from hunter_kinodynamic_rl.trajectory.trajectory_primitive import make_primitive


def _nan_to_none(value):
    """section P1-11: JSON has no NaN literal (RFC 8259) -- Python's json
    module happily emits the non-standard token anyway, which most OTHER
    JSON parsers reject outright. Every risk-telemetry field that can be
    NaN (rt.invalid()'s convention for "no label this step") is converted
    to ``null`` here instead -- ``risk_valid``/an explicit ``*_valid``
    companion flag is what distinguishes "no data" from "data happens to
    be exactly this number", never a bare NaN doing double duty."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _sensor_diagnostics_dict(d: sd.SensorDiagnostics) -> dict:
    """requirement 5 fix (invalid-diagnostics fields must be null, never
    0/0.0): ``sensor_diagnostics.invalid()`` only sets the pose/velocity/
    steering fields to NaN (already caught by ``_nan_to_none`` below) --
    ``drift_*``/``localization_latency_steps``/``lidar_beam_count``/
    ``lidar_dropout_count``/``lidar_perturbation_{mean,max}_m`` all default
    to a concrete ``0``/``0.0`` on the dataclass, which is
    INDISTINGUISHABLE, once serialized, from "the measurement really was
    zero this step". So every measurement field is forced to an explicit
    ``None`` here whenever ``d.valid`` is False, regardless of what those
    defaults happen to hold -- a reader must trust ``valid``/
    ``invalid_reason`` alone to know whether ANY of these numbers mean
    anything, never infer it from a field happening to read 0.

    ``schema_version``/``episode_id``/``sim_timestamp_sec`` are always
    included (even when invalid) for traceability -- pairing a JSONL record
    back to the exact wire-format version and episode/sim-time it came from
    without needing to cross-reference a separate ``episode_start`` record."""
    base = {
        "schema_version": sd.SCHEMA_VERSION, "valid": d.valid, "invalid_reason": d.invalid_reason,
        "step_id": d.step_id, "reset_generation": d.reset_generation, "episode_id": d.episode_id,
        "sim_timestamp_sec": _nan_to_none(d.sim_timestamp_sec),
    }
    if not d.valid:
        base.update({
            "gt_pose": {"x": None, "y": None, "yaw": None},
            "noisy_pose": {"x": None, "y": None, "yaw": None},
            "gt_velocity_mps": None, "noisy_velocity_mps": None,
            "gt_yaw_rate_rad_s": None, "noisy_yaw_rate_rad_s": None,
            "gt_steering_rad": None, "noisy_steering_rad": None,
            "localization_drift": {"x_m": None, "y_m": None, "yaw_rad": None},
            "localization_latency_steps": None,
            "lidar_beam_count": None, "lidar_dropout_count": None,
            "lidar_perturbation_mean_m": None, "lidar_perturbation_max_m": None,
        })
        return base
    base.update({
        "gt_pose": {"x": _nan_to_none(d.gt_x), "y": _nan_to_none(d.gt_y), "yaw": _nan_to_none(d.gt_yaw)},
        "noisy_pose": {"x": _nan_to_none(d.noisy_x), "y": _nan_to_none(d.noisy_y),
                       "yaw": _nan_to_none(d.noisy_yaw)},
        "gt_velocity_mps": _nan_to_none(d.gt_v_mps),
        "noisy_velocity_mps": _nan_to_none(d.noisy_v_mps),
        "gt_yaw_rate_rad_s": _nan_to_none(d.gt_yaw_rate_radps),
        "noisy_yaw_rate_rad_s": _nan_to_none(d.noisy_yaw_rate_radps),
        "gt_steering_rad": _nan_to_none(d.gt_steering_rad),
        "noisy_steering_rad": _nan_to_none(d.noisy_steering_rad),
        "localization_drift": {"x_m": d.drift_x_m, "y_m": d.drift_y_m, "yaw_rad": d.drift_yaw_rad},
        "localization_latency_steps": d.localization_latency_steps,
        "lidar_beam_count": d.lidar_beam_count, "lidar_dropout_count": d.lidar_dropout_count,
        "lidar_perturbation_mean_m": d.lidar_perturbation_mean_m,
        "lidar_perturbation_max_m": d.lidar_perturbation_max_m,
    })
    return base


def run_directory(profile: Profile, base_root: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(base_root, f"{stamp}_{profile.name}_seed{profile.training.seed}")
    for sub in ("configs", "checkpoints", "logs", "evaluation", "benchmark", "tensorboard"):
        os.makedirs(os.path.join(path, sub), exist_ok=True)
    return path


class RunLogger:
    def __init__(self, run_dir: str, profile: Profile, log_full_state: bool = False):
        self.run_dir = run_dir
        self.profile = profile
        self.log_full_state = log_full_state
        self._start_time = time.time()

        with open(os.path.join(run_dir, "configs", "profile_snapshot.json"), "w") as f:
            json.dump(_profile_to_dict(profile), f, indent=2, sort_keys=True)

        self._metadata_path = os.path.join(run_dir, "metadata.json")
        self._write_metadata(status="running")

        self._steps_path = os.path.join(run_dir, "logs", "steps.jsonl")
        self._steps_file = open(self._steps_path, "a")

        episodes_path = os.path.join(run_dir, "logs", "episodes.csv")
        episodes_is_new = not os.path.isfile(episodes_path) or os.path.getsize(episodes_path) == 0
        self._episodes_file = open(episodes_path, "a", newline="")
        self._episodes_writer = csv.writer(self._episodes_file)
        if episodes_is_new:
            self._episodes_writer.writerow(
                ["episode_index", "seed", "global_step", "reward", "length", "reached_goal", "collision",
                 "timeout", "unrecoverable", "sensor_stale"])

        self._tb = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._tb = SummaryWriter(os.path.join(run_dir, "tensorboard"))
        except Exception:
            self._tb = None

    def _write_metadata(self, status: str) -> None:
        meta = {
            "profile_name": self.profile.name,
            "run_dir": self.run_dir,
            "status": status,
            "start_time": self._start_time,
            "last_update_time": time.time(),
        }
        tmp_path = self._metadata_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True)
        os.replace(tmp_path, self._metadata_path)  # atomic

    def log_episode_start(self, episode_index: int, seed: int, global_step: int,
                           domain_rand_draw=None) -> None:
        """section P1-8: "log sampled/applied [domain-randomization] values
        per episode" -- ``domain_rand_draw`` is the caller's OWN
        ``sample_draw(seed, profile.domain_randomization)`` result (a pure
        function of the same seed/config the environment node independently
        applies server-side -- see trainer_base.py's ``_new_episode``), None
        when domain randomization is disabled for this profile."""
        record = {
            "event": "episode_start", "episode_index": episode_index, "seed": seed,
            "global_step": global_step, "timestamp": time.time(),
            "domain_rand_draw": dataclasses.asdict(domain_rand_draw) if domain_rand_draw is not None else None,
        }
        self._steps_file.write(json.dumps(record) + "\n")
        self._steps_file.flush()

    def log_step(self, global_step: int, episode_index: int, action, reward: float,
                 telemetry: rt.RiskTelemetry, collision: bool, target: bool,
                 pose=None, trajectory_command=None, actual_dt_sec: Optional[float] = None,
                 state=None, next_state=None, measured_velocity_mps: Optional[float] = None,
                 measured_yaw_rate_rad_s: Optional[float] = None,
                 measured_steering_rad: Optional[float] = None,
                 predicted_risk: Optional[float] = None,
                 sensor_diagnostics: Optional[sd.SensorDiagnostics] = None) -> None:
        """section P1-11: ``pose`` is an optional ``(x, y, yaw)`` tuple (real
        odometry, from the trainer's own EnvironmentClient -- None if not
        yet received); ``trajectory_command`` is the DECODED
        TrajectoryCommand/LegacyWaypointCommand for this step's raw action
        (``trajectory/action_space.decode_action``'s output -- None if the
        caller didn't decode one); ``actual_dt_sec`` is the REAL elapsed
        sim-time this step took (via /clock), not the nominal
        ``runtime.time_delta_sec`` -- None if /clock data wasn't available.

        Every risk-telemetry field that can be NaN (section 5's "no label
        this step" convention) is emitted as JSON ``null`` here, never a
        raw NaN token (section P1-11 -- see ``_nan_to_none``).

        ``sensor_diagnostics`` (requirement 5) is the GT-vs-noisy-observation
        side-channel sample for THIS exact step (``env/simulation/
        sensor_diagnostics.py``'s ``SensorDiagnostics``, as fetched by
        ``EnvironmentClient._await_matching_sensor_diagnostics`` -- already
        paired to this step's own ``(reset_generation, step_id)``, so it is
        NEVER a stale sample from a different step/episode by construction).
        ``None`` (the default) omits the ``sensor_diagnostics`` key entirely
        -- for callers that never fetch it; a caller that DID attempt the
        fetch always passes a real (possibly ``valid=False``, e.g. on a poll
        timeout) object, never silently drops it, so ``valid``/
        ``invalid_reason`` are always there to tell a reader whether the
        embedded GT/noisy values are trustworthy for this record."""
        trajectory_points = None
        if isinstance(trajectory_command, TrajectoryCommand):
            primitive = make_primitive(
                self.profile.trajectory.primitive, trajectory_command.kappa,
                trajectory_command.v_ref, trajectory_command.horizon_m,
            )
            trajectory_points = [dataclasses.asdict(p) for p in primitive.sample(self.profile.trajectory.num_samples)]

        record = {
            "timestamp": time.time(), "global_step": global_step, "episode_index": episode_index,
            "normalized_action": [float(a) for a in action], "reward": float(reward),
            "collision": bool(collision), "goal_reached": bool(target),
            "pose": {"x": pose[0], "y": pose[1], "yaw": pose[2]} if pose is not None else None,
            "goal": ({"x": telemetry.goal_x, "y": telemetry.goal_y}
                     if math.isfinite(telemetry.goal_x) and math.isfinite(telemetry.goal_y) else None),
            "measured_velocity_mps": measured_velocity_mps,
            "measured_yaw_rate_rad_s": measured_yaw_rate_rad_s,
            "measured_steering_rad": measured_steering_rad,
            "actual_dt_sec": actual_dt_sec,
            "trajectory": dataclasses.asdict(trajectory_command) if trajectory_command is not None else None,
            "trajectory_points": trajectory_points,
            "state": ([float(v) for v in state] if self.log_full_state and state is not None else None),
            "next_state": ([float(v) for v in next_state]
                           if self.log_full_state and next_state is not None else None),
            "emergency_stop": telemetry.emergency_stop,
            "plant_limited": telemetry.plant_limited,
            "guard_intervened": telemetry.guard_intervened,
            "sensor_stale": telemetry.sensor_stale,
            "nominal_speed_mps": telemetry.nominal_speed_mps,
            "nominal_steering_rad": telemetry.nominal_steering_rad,
            "guarded_speed_mps": telemetry.guarded_speed_mps,
            "guarded_steering_rad": telemetry.guarded_steering_rad,
            "published_speed_mps": telemetry.published_speed_mps,
            "published_steering_rad": telemetry.published_steering_rad,
            "risk_valid": telemetry.valid,
            "risk_target": _nan_to_none(telemetry.risk_target),
            "risk_ground_truth": _nan_to_none(telemetry.risk_target),
            "risk_predicted": _nan_to_none(predicted_risk),
            "min_clearance_m": _nan_to_none(telemetry.min_clearance_m),
            "ttc_sec": _nan_to_none(telemetry.ttc_sec),
            "collision_within_horizon": telemetry.collision_within_horizon,
            "stopping_margin_m": _nan_to_none(telemetry.stopping_margin_m),
            "steering_saturation": telemetry.steering_saturation,
            "goal_progress_m": _nan_to_none(telemetry.goal_progress_m),
            "reward_components": {
                "goal": telemetry.reward_goal,
                "collision": telemetry.reward_collision,
                "progress": telemetry.reward_progress,
                "step": telemetry.reward_step,
                "control_smoothness": telemetry.reward_control_smoothness,
                "trajectory_smoothness": telemetry.reward_trajectory_smoothness,
            },
            "unrecoverable": telemetry.unrecoverable,
            "safer_alternative_margin": _nan_to_none(telemetry.safer_alternative_margin),
            "num_candidates": len(telemetry.candidates),
            "candidates": [
                {"kappa": c.kappa, "v_ref": c.v_ref, "horizon_m": c.horizon_m,
                 "risk_score": c.risk_score, "goal_progress_m": c.goal_progress_m}
                for c in telemetry.candidates
            ],
        }
        if sensor_diagnostics is not None:
            record["sensor_diagnostics"] = _sensor_diagnostics_dict(sensor_diagnostics)
        self._steps_file.write(json.dumps(record) + "\n")
        self._steps_file.flush()

    def log_episode_end(self, episode_index: int, seed: int, global_step: int, episode_reward: float,
                         episode_len: int, reached_goal: bool, collision: bool,
                         timeout: bool = False, unrecoverable: bool = False,
                         sensor_stale: bool = False) -> None:
        self._episodes_writer.writerow(
            [episode_index, seed, global_step, episode_reward, episode_len, reached_goal, collision,
             timeout, unrecoverable, sensor_stale])
        self._episodes_file.flush()
        if self._tb is not None:
            self._tb.add_scalar("episode/reward", episode_reward, episode_index)
            self._tb.add_scalar("episode/length", episode_len, episode_index)
            self._tb.add_scalar("episode/success", float(reached_goal), episode_index)
            self._tb.add_scalar("episode/collision", float(collision), episode_index)

    def log_validation(self, global_step: int, result: Dict, eval_metric: float, is_new_best: bool) -> None:
        """section P1-13: periodic held-out validation-set evaluation --
        ``result`` is ``TrainerBase._run_validation_episodes()``'s dict
        (success_rate/collision_rate/mean_reward/num_episodes),
        ``eval_metric`` is the documented best-checkpoint criterion
        (success_rate - collision_rate), ``is_new_best`` records whether
        THIS validation pass actually triggered a new "best" checkpoint
        save (so the JSONL/TensorBoard history shows exactly which
        validation passes did, not just the metric trend)."""
        record = {
            "event": "validation", "global_step": global_step, "timestamp": time.time(),
            "eval_metric": eval_metric, "is_new_best": bool(is_new_best), **result,
        }
        self._steps_file.write(json.dumps(record) + "\n")
        self._steps_file.flush()
        if self._tb is not None:
            self._tb.add_scalar("validation/success_rate", result["success_rate"], global_step)
            self._tb.add_scalar("validation/collision_rate", result["collision_rate"], global_step)
            self._tb.add_scalar("validation/mean_reward", result["mean_reward"], global_step)
            self._tb.add_scalar("validation/eval_metric", eval_metric, global_step)

    def log_telemetry_health(self, global_step: int, valid_ratio: float, telemetry_matched: int,
                              telemetry_timeouts: int, reset_marker_timeouts: int,
                              risk_supervised_updates: Optional[int]) -> None:
        """code review (risk-telemetry correctness bug): a directly
        greppable/parseable JSONL record (never just a TensorBoard scalar,
        which needs a protobuf reader to verify programmatically) proving
        the risk-telemetry side channel's health at THIS point in the run
        -- see TrainerBase._check_risk_telemetry_health, the enforcement
        this logs alongside."""
        record = {
            "event": "telemetry_health", "global_step": global_step, "timestamp": time.time(),
            "valid_ratio": valid_ratio, "telemetry_matched": telemetry_matched,
            "telemetry_timeouts": telemetry_timeouts, "reset_marker_timeouts": reset_marker_timeouts,
            "risk_supervised_updates": risk_supervised_updates,
        }
        self._steps_file.write(json.dumps(record) + "\n")
        self._steps_file.flush()
        if self._tb is not None:
            self._tb.add_scalar("telemetry/valid_ratio", valid_ratio, global_step)
            self._tb.add_scalar("telemetry/timeouts", telemetry_timeouts, global_step)
            self._tb.add_scalar("telemetry/reset_marker_timeouts", reset_marker_timeouts, global_step)
            if risk_supervised_updates is not None:
                self._tb.add_scalar("telemetry/risk_supervised_updates", risk_supervised_updates, global_step)

    def log_train_metrics(self, global_step: int, metrics: Dict, valid_ratio: float) -> None:
        if self._tb is None:
            return
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self._tb.add_scalar(f"train/{k}", v, global_step)
        self._tb.add_scalar("replay/valid_ratio", valid_ratio, global_step)

    def log_resume(self, global_step: int, episode_index: int, checkpoint_tag: str, checkpoint_dir: str) -> None:
        self._steps_file.write(json.dumps({
            "event": "resume", "global_step": global_step, "episode_index": episode_index,
            "checkpoint_tag": checkpoint_tag, "checkpoint_dir": checkpoint_dir,
            "timestamp": time.time(),
        }) + "\n")
        self._steps_file.flush()

    def log_resume_noop(self, global_step: int, max_timesteps: int) -> None:
        """Resumed run had already reached (or exceeded) its target step
        count -- a NORMAL, successful early exit, not an error (section
        P1-1: "이미 목표 step 이상이면 추가 학습 없이 정상 종료한다")."""
        self._steps_file.write(json.dumps({
            "event": "resume_noop", "global_step": global_step, "max_timesteps": max_timesteps,
            "timestamp": time.time(),
        }) + "\n")
        self._steps_file.flush()

    def log_eval(self, global_step: int, summary: Dict) -> None:
        if self._tb is not None:
            for k, v in summary.items():
                if isinstance(v, (int, float)):
                    self._tb.add_scalar(f"eval/{k}", v, global_step)

    def close(self, failed: bool = False) -> None:
        self._write_metadata(status="failed" if failed else "finished")
        self._steps_file.close()
        self._episodes_file.close()
        if self._tb is not None:
            self._tb.close()


def _profile_to_dict(profile: Profile) -> dict:
    import dataclasses
    return dataclasses.asdict(profile)
