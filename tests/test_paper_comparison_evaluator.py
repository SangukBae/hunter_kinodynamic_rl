import dataclasses

import numpy as np

from hunter_kinodynamic_rl.config.loader import load_profile
from hunter_kinodynamic_rl.env.scenarios.benchmark_loader import BenchmarkScenario
from hunter_kinodynamic_rl.env.scenarios.procedural_generator import ScenarioSpec
from hunter_kinodynamic_rl.env.simulation.risk_telemetry import RiskTelemetry
from hunter_kinodynamic_rl.env.simulation.sensor_diagnostics import SensorDiagnostics
from hunter_kinodynamic_rl.evaluation.evaluate_paper_comparison import run_locked_episode
from hunter_kinodynamic_rl.rl.networks.tractor import TractorConfig


def _config():
    return TractorConfig(
        t_obs=2, n_scan=8, grid_height=8, grid_width=8, resolution_m=1.0,
        x_min_m=-4.0, y_min_m=-4.0, ray_free_samples=2, scene_channels=8,
        ego_dim=8, plant_dim=8, interaction_dim=8, num_candidates=2,
        horizon_steps=2, dt_out_sec=0.2, dt_dyn_sec=0.1, sparse_tube_samples=4,
        n_quantiles=3, n_severity_quantiles=2, max_speed_mps=1.0,
        severity_quantile_levels=(0.1, 0.9), min_arc_m=0.1, max_arc_m=0.4,
        min_horizon_sec=0.2, min_safety_horizon_sec=0.2, max_horizon_sec=0.4,
    )


def _diagnostics(step):
    return SensorDiagnostics(
        step_id=step, valid=True, sim_timestamp_sec=0.1 * step,
        noisy_x=0.1 * step, noisy_y=0.0, noisy_yaw=0.0,
        gt_x=0.1 * step, gt_y=0.0, gt_yaw=0.0,
        gt_v_mps=1.0, gt_yaw_rate_radps=0.0, gt_steering_rad=0.0,
        noisy_v_mps=1.0, noisy_yaw_rate_radps=0.0, noisy_steering_rad=0.0,
        lidar_beam_count=8,
    )


class _Policy:
    def reset(self):
        self.resets = getattr(self, "resets", 0) + 1

    def act(self, snapshot, snapshot_id):
        assert snapshot_id == 1
        assert snapshot.sensor_freshness_valid
        return np.asarray([0.0, -1.0, 0.0], np.float32)


class _Env:
    def __init__(self, observation_dim):
        self.observation_dim = observation_dim
        self.latest_pose_covariance = np.eye(3, dtype=np.float32) * 0.001
        self.episode_elapsed_sim_time_sec = 0.1
        self.episode_path_length_m = 0.1

    def reset_with_diagnostics(self):
        return np.zeros(self.observation_dim, np.float32), _diagnostics(0)

    def step(self, action):
        telemetry = RiskTelemetry(
            step_id=1, valid=True, risk_target=0.0, min_clearance_m=0.7,
            ttc_sec=3.0, collision_within_horizon=False, stopping_margin_m=0.5,
            unrecoverable=False, safer_alternative_margin=0.0, actor_candidate_index=0,
        )
        return (
            np.zeros(self.observation_dim, np.float32), 1.0, True, True, False,
            0.7, telemetry, _diagnostics(1),
        )


def test_locked_episode_produces_the_four_headline_metrics_and_latency():
    cfg = _config()
    profile = load_profile("smoke_test")
    profile = dataclasses.replace(
        profile, evaluation=dataclasses.replace(profile.evaluation, max_episode_steps=2),
    )
    scenario = BenchmarkScenario(
        "fixture", ScenarioSpec(
            seed=1, start_x=0.0, start_y=0.0, start_yaw=0.0,
            goal_x=1.0, goal_y=0.0,
        ),
    )
    result = run_locked_episode(_Env(cfg.observation_dim), _Policy(), profile, scenario, cfg, 100.0)
    assert result["complete"]
    assert result["metrics"]["success_rate"] == 1.0
    assert result["metrics"]["collision_rate"] == 0.0
    assert result["metrics"]["time_to_goal_sec_mean"] == 0.1
    assert result["metrics"]["min_clearance_m_mean"] == 0.7
    assert result["metrics"]["decision_count"] == 1


def test_failed_episode_uses_the_common_capped_time_budget():
    cfg = _config()
    profile = load_profile("smoke_test")
    profile = dataclasses.replace(
        profile, evaluation=dataclasses.replace(profile.evaluation, max_episode_steps=2),
    )
    env = _Env(cfg.observation_dim)
    original = env.step

    def collision_step(action):
        state, reward, _done, _target, _collision, distance, telemetry, diagnostics = original(action)
        return state, reward, True, False, True, distance, telemetry, diagnostics

    env.step = collision_step
    scenario = BenchmarkScenario(
        "fixture", ScenarioSpec(
            seed=1, start_x=0.0, start_y=0.0, start_yaw=0.0,
            goal_x=1.0, goal_y=0.0,
        ),
    )
    result = run_locked_episode(env, _Policy(), profile, scenario, cfg, 100.0)
    assert result["metrics"]["collision_rate"] == 1.0
    assert result["metrics"]["time_to_goal_sec_mean"] == (
        profile.evaluation.max_episode_steps * profile.runtime.time_delta_sec
    )
