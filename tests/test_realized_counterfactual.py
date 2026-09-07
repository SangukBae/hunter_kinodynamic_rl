import numpy as np

from hunter_kinodynamic_rl.config.loader import load_profile
from hunter_kinodynamic_rl.rl.networks.tractor import TractorConfig
from hunter_kinodynamic_rl.training.realized_counterfactual import LABEL_SOURCE, relabel_rows


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


def _row(timestamp, obstacle_x):
    cfg = _config()
    observation = np.zeros(cfg.observation_dim, np.float32)
    observation[-3] = 0.0  # current speed
    observation[-1] = 0.0  # steering
    return {
        "observation": observation,
        "candidate_present": np.asarray([True, True]),
        "candidate_model_valid": np.asarray([True, True]),
        "candidate_horizon_mask": np.ones((2, 2), bool),
        "candidate_trajectories_physical": np.asarray([
            [0.0, 1.0, 1.0], [0.0, 0.0, 1.0],
        ], np.float32),
        "privileged_snapshot_valid": True,
        "privileged_snapshot_timestamp_sec": timestamp,
        "privileged_ego_pose_world": np.asarray([0.0, 0.0, 0.0]),
        "privileged_world_half_extent_m": 8.0,
        "privileged_obstacles_world": np.asarray([[obstacle_x, 0.0, 0.3, 1.0]]),
    }


def test_relabel_uses_future_realized_obstacle_position_not_current_velocity_extrapolation():
    # The obstacle is far away at t=0 but really occupies the candidate path at t=0.2.
    rows = [_row(0.1 * index, 5.0 if index != 2 else 0.8) for index in range(5)]
    report = relabel_rows(rows, load_profile("tractor_local_dynamic"), _config())
    assert report["valid_candidate_labels"] > 0
    assert report["observed_events"] > 0
    assert rows[0]["candidate_event_observed"][0]
    assert rows[0]["candidate_event_step"][0] == 0
    assert rows[0]["candidate_event_cause"][0] == 1
    assert str(rows[0]["candidate_label_source"]) == LABEL_SOURCE


def test_relabel_right_censors_when_episode_ends_before_full_horizon():
    rows = [_row(0.1 * index, 5.0) for index in range(3)]
    relabel_rows(rows, load_profile("tractor_local_dynamic"), _config())
    assert rows[0]["candidate_event_label_valid"][0]
    assert rows[0]["candidate_event_observed"][0] == 0
    assert rows[0]["candidate_censor_step"][0] == 0
    assert not rows[-1]["candidate_event_label_valid"].any()
