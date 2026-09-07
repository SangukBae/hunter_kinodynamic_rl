"""Runtime one-snapshot and fail-safe selection contracts."""

from dataclasses import replace

import pytest
import torch

from hunter_kinodynamic_rl.navigation.local_rl.tractor_policy import TractorPolicy
from hunter_kinodynamic_rl.rl.networks.tractor import TractorConfig, TractorInputs, TractorTQC
from hunter_kinodynamic_rl.rl.networks.tractor.calibration import PlattCalibration
from hunter_kinodynamic_rl.rl.networks.tractor.selector import SelectorConfig


def _config():
    return TractorConfig(
        t_obs=2, n_scan=8, grid_height=8, grid_width=8, resolution_m=1.0,
        x_min_m=-4.0, y_min_m=-4.0, ray_free_samples=2, scene_channels=8,
        ego_dim=8, plant_dim=8, interaction_dim=8, num_candidates=2,
        horizon_steps=2, dt_out_sec=0.2, dt_dyn_sec=0.1, sparse_tube_samples=4,
        n_quantiles=3, n_severity_quantiles=2, max_speed_mps=1.0,
        severity_quantile_levels=(0.1, 0.9),
        min_arc_m=0.1, max_arc_m=0.4, min_horizon_sec=0.2,
        min_safety_horizon_sec=0.2, max_horizon_sec=0.4,
    )


def _input(config):
    scans = torch.full((1, config.t_obs, config.n_scan), 2.0)
    tail = torch.tensor([[2.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0]])
    return TractorInputs(
        observation=torch.cat((scans.reshape(1, -1), tail), -1),
        scan_valid=torch.ones_like(scans, dtype=torch.bool),
        motion_delta=torch.tensor([[[0.0, 0.0, 0.0, 0.1]]]),
        motion_valid=torch.ones(1, 1, dtype=torch.bool),
        decision_timestamp_sec=torch.ones(1, 1),
        previous_intent_valid=torch.ones(1, 1, dtype=torch.bool),
        previous_command_published=torch.zeros(1, 2),
        previous_command_valid=torch.ones(1, 1, dtype=torch.bool),
        vehicle_response_valid=torch.ones(1, 3, dtype=torch.bool),
        localization_covariance=torch.eye(3).mul(1e-3).unsqueeze(0),
        localization_valid=torch.ones(1, 1, dtype=torch.bool),
        localization_confidence=torch.ones(1, 1),
        localization_confidence_valid=torch.ones(1, 1, dtype=torch.bool),
        sensor_freshness_sec=torch.zeros(1, 1),
        sensor_freshness_valid=torch.ones(1, 1, dtype=torch.bool),
        scene_reset=torch.ones(1, 1, dtype=torch.bool),
        response_reset=torch.ones(1, 1, dtype=torch.bool),
    )


def _calibration():
    return PlattCalibration(1.0, 0.0, "checkpoint", "calibration-split")


def test_deployment_requires_calibrator():
    config = _config()
    with pytest.raises(ValueError, match="calibration"):
        TractorPolicy(TractorTQC(config), SelectorConfig(), None, deployment=True)


def test_same_snapshot_cannot_advance_recurrence_twice():
    config = _config()
    policy = TractorPolicy(TractorTQC(config), SelectorConfig(), _calibration())
    policy.step(10, _input(config))
    hidden = policy._scene_hidden.clone()
    with pytest.raises(RuntimeError, match="strictly"):
        policy.step(10, _input(config))
    assert torch.equal(policy._scene_hidden, hidden)


def test_invalid_sensor_snapshot_forces_no_publish_fallback():
    config = _config()
    policy = TractorPolicy(TractorTQC(config), SelectorConfig(), _calibration())
    invalid = replace(
        _input(config),
        scan_valid=torch.zeros((1, config.t_obs, config.n_scan), dtype=torch.bool),
        sensor_freshness_valid=torch.zeros((1, 1), dtype=torch.bool),
    )
    decision = policy.step(1, invalid)
    assert not decision.publish_allowed
    assert decision.fallback_reason == "invalid_belief"
    assert int(decision.selection.selected_index[0]) == -1
