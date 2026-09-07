"""End-to-end unit coverage for the new sequence training bridge."""

import hashlib
from pathlib import Path

import numpy as np
import torch

from hunter_kinodynamic_rl.env.simulation.risk_telemetry import CandidateTelemetry, RiskTelemetry
from hunter_kinodynamic_rl.env.simulation.sensor_diagnostics import SensorDiagnostics
from hunter_kinodynamic_rl.rl.algorithms.tractor_tqc import TractorAgent, TractorAgentConfig
from hunter_kinodynamic_rl.rl.networks.tractor import TractorConfig
from hunter_kinodynamic_rl.rl.replay import EpisodeHeader, EpisodeStore, SequenceBuffer, SequenceIndex
from hunter_kinodynamic_rl.rl.replay.sequence_schema import validate_episode_columns
from hunter_kinodynamic_rl.training.tractor_episode_collector import (
    TractorEpisodeRecorder, make_snapshot,
)
from hunter_kinodynamic_rl.training.tractor_sequence_training import TractorSequenceTrainer


def _config():
    return TractorConfig(
        t_obs=2, n_scan=8, grid_height=8, grid_width=8, resolution_m=1.0,
        x_min_m=-4.0, y_min_m=-4.0, ray_free_samples=2, scene_channels=8,
        ego_dim=8, plant_dim=8, interaction_dim=8, num_candidates=2,
        horizon_steps=2, dt_out_sec=0.2, dt_dyn_sec=0.1,
        sparse_tube_samples=4, n_quantiles=3, n_severity_quantiles=2,
        severity_quantile_levels=(0.1, 0.9), max_speed_mps=1.0,
        min_arc_m=0.1, max_arc_m=0.4, min_horizon_sec=0.2,
        min_safety_horizon_sec=0.2, max_horizon_sec=0.4,
    )


def _header(steps):
    identity = hashlib.sha256(b"training-path").hexdigest()
    return EpisodeHeader(
        episode_id="training-path", scenario_id="fixture", split_id="development", seed=3,
        software_commit="commit", dirty_state_digest="dirty", container_image_digest="container",
        resolved_config_hash="config", protocol_version="protocol",
        environment_attestation_hash="env", robot_attestation_hash="robot",
        observation_contract_hash="obs", action_contract_hash="action",
        trajectory_contract_hash="trajectory", start_utc="start", end_utc="end",
        termination_reason="goal", step_count=steps, sensor_source="fixture",
        localization_source="fixture", controller_source="fixture", clock_domain="sim",
        group_id=identity, scenario_family="fixture", obstacle_contract="dynamic",
        scenario_geometry_sha256=identity,
    )


def _columns(config, steps=3):
    observation = np.zeros((steps, config.observation_dim), np.float32)
    observation[:, : config.t_obs * config.n_scan] = 2.0
    observation[:, -8:] = np.asarray([2.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0])
    columns = {
        "decision_timestamp_ns": np.arange(1, steps + 1, dtype=np.int64) * 100_000_000,
        "observation": observation,
        "scan_valid": np.ones((steps, config.t_obs, config.n_scan), bool),
        "motion_delta_from_previous": np.tile(
            np.asarray([0.0, 0.0, 0.0, 0.1], np.float32), (steps, config.t_obs - 1, 1)
        ),
        "motion_valid": np.ones((steps, config.t_obs - 1), bool),
        "previous_intent_valid": np.ones(steps, bool),
        "previous_command_published": np.zeros((steps, 2), np.float32),
        "previous_command_valid": np.ones(steps, bool),
        "vehicle_response_valid": np.ones((steps, 3), bool),
        "pose_covariance": np.zeros((steps, 3, 3), np.float32),
        "localization_valid": np.ones(steps, bool),
        "localization_confidence": np.ones(steps, np.float32),
        "localization_confidence_valid": np.ones(steps, bool),
        "sensor_freshness_sec": np.zeros(steps, np.float32),
        "sensor_freshness_valid": np.ones(steps, bool),
        "reset_epoch": np.zeros(steps, np.int64),
        "scene_reset": np.asarray([True] + [False] * (steps - 1)),
        "response_reset": np.asarray([True] + [False] * (steps - 1)),
        "action_normalized_requested": np.zeros((steps, 3), np.float32),
        "reward": np.ones(steps, np.float32),
        "transition_dt_sec": np.full(steps, 0.1, np.float32),
        "discount_factor": np.full(steps, 0.99, np.float32),
        "next_observation_valid": np.ones(steps, bool),
        "terminated": np.asarray([False] * (steps - 1) + [True]),
        "truncated": np.zeros(steps, bool),
        "termination_reason": np.asarray(["none"] * (steps - 1) + ["goal"]),
        "bellman_sample_valid": np.ones(steps, bool),
        "candidate_actions_normalized": np.zeros((steps, config.num_candidates, 3), np.float32),
        "candidate_trajectories_physical": np.zeros((steps, config.num_candidates, 3), np.float32),
        "candidate_present": np.ones((steps, config.num_candidates), bool),
        "candidate_is_stop": np.tile(np.asarray([False, True]), (steps, 1)),
        "candidate_model_valid": np.ones((steps, config.num_candidates), bool),
        "candidate_horizon_mask": np.ones((steps, config.num_candidates, config.horizon_steps), bool),
        "candidate_event_observed": np.zeros((steps, config.num_candidates), bool),
        "candidate_event_step": np.full((steps, config.num_candidates), -1, np.int16),
        "candidate_event_cause": np.full((steps, config.num_candidates), -1, np.int8),
        "candidate_censor_step": np.full((steps, config.num_candidates), 1, np.int16),
        "candidate_event_label_valid": np.ones((steps, config.num_candidates), bool),
        "candidate_clearance_m": np.ones((steps, config.num_candidates, config.horizon_steps), np.float32),
        "candidate_clearance_valid": np.ones((steps, config.num_candidates, config.horizon_steps), bool),
        "candidate_stopping_margin_m": np.ones((steps, config.num_candidates, config.horizon_steps), np.float32),
        "candidate_stopping_margin_valid": np.ones((steps, config.num_candidates, config.horizon_steps), bool),
        "candidate_progress_m": np.ones((steps, config.num_candidates), np.float32),
        "candidate_progress_valid": np.ones((steps, config.num_candidates), bool),
        "candidate_set_sha256": np.full(steps, "0" * 64),
    }
    return columns


def test_sequence_replay_reaches_all_agent_transactions(tmp_path: Path):
    torch.manual_seed(4)
    config = _config()
    store = EpisodeStore(tmp_path)
    store.append(_header(3), _columns(config))
    index = SequenceIndex.build(store, loss_window=2)
    sampler = SequenceBuffer(store, index, seed=9)
    agent = TractorAgent(config, TractorAgentConfig(top_quantiles_to_drop_per_net=1), target_seed=2)
    trainer = TractorSequenceTrainer(agent, sampler, {"clearance": 0.25, "stopping_margin": 0.25})
    metrics = trainer.update(batch_size=1)
    assert metrics["update/transaction_applied"] == 1.0
    assert metrics["risk/update_applied"] == 1.0
    assert agent.update_step == 1
    assert sampler.draw_ordinal == 1


def test_live_snapshot_and_candidate_telemetry_form_valid_episode_columns():
    config = _config()
    # Only the collector's profile fields are needed by this pure unit test.
    from hunter_kinodynamic_rl.config.loader import load_profile
    profile = load_profile(
        "tractor_local_dynamic_v2", str(Path(__file__).resolve().parents[1] / "config"),
    )
    diagnostics = SensorDiagnostics(
        step_id=1, valid=True, sim_timestamp_sec=0.1,
        noisy_x=0.0, noisy_y=0.0, noisy_yaw=0.0,
        lidar_beam_count=8, lidar_dropout_count=0,
    )
    state = np.r_[np.full(config.t_obs * config.n_scan, 2.0), np.zeros(8)].astype(np.float32)
    first = make_snapshot(
        state, config, diagnostics=None, pose_covariance=np.zeros((3, 3)), previous=None,
    )
    second = make_snapshot(
        state, config, diagnostics=diagnostics, pose_covariance=np.zeros((3, 3)), previous=first,
        previous_action=np.zeros(3), previous_command=np.zeros(2),
    )
    telemetry = RiskTelemetry(
        step_id=1, valid=True, risk_target=0.2, min_clearance_m=1.0, ttc_sec=3.0,
        collision_within_horizon=False, stopping_margin_m=0.5, unrecoverable=False,
        safer_alternative_margin=0.0, actor_candidate_index=0,
        candidates=[
            CandidateTelemetry(0.0, 0.5, 0.3, 0.2, 0.1, 1.0, 3.0, False, 0.5, -1),
            CandidateTelemetry(0.0, 0.0, 0.3, 0.0, 0.0, 1.2, 3.0, False, 1.2, -1),
        ],
    )
    recorder = TractorEpisodeRecorder(profile, config)
    recorder.append(first, np.zeros(3), second, 1.0, True, True, False, telemetry)
    columns = recorder.columns()
    validate_episode_columns(_header(1), columns)
    assert columns["candidate_event_label_valid"].all()
    assert columns["candidate_label_source"].item() == "nominal_preaction_rollout_summary_v1"
    assert columns["next_observation"].shape == (1, config.observation_dim)
