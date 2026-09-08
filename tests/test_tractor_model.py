"""Executable structural contracts for the TRACTOR-TQC model."""

from dataclasses import replace
import math

import pytest
import torch

from hunter_kinodynamic_rl.config.loader import load_profile
from hunter_kinodynamic_rl.dynamics.ackermann_rollout import rollout_tractor_output_grid
from hunter_kinodynamic_rl.rl.networks.tractor import TractorConfig, TractorInputs, TractorTQC
from hunter_kinodynamic_rl.rl.networks.tractor.contracts import CandidateSet
from hunter_kinodynamic_rl.rl.networks.tractor.nominal_rollout_adapter import (
    nominal_rollout_source_fingerprint,
)
from hunter_kinodynamic_rl.rl.networks.tractor.se2_warp import SE2HistoryWarp
from hunter_kinodynamic_rl.robot.interface import VehicleState


def _small_config(**overrides):
    config = TractorConfig(
        t_obs=2,
        n_scan=16,
        grid_height=16,
        grid_width=16,
        resolution_m=0.5,
        x_min_m=-4.0,
        y_min_m=-4.0,
        ray_free_samples=4,
        scene_channels=8,
        ego_dim=16,
        plant_dim=8,
        interaction_dim=16,
        num_candidates=4,
        horizon_steps=3,
        dt_out_sec=0.2,
        dt_dyn_sec=0.1,
        sparse_tube_samples=8,
        n_quantiles=5,
        n_severity_quantiles=3,
        severity_quantile_levels=(0.1, 0.5, 0.9),
        max_speed_mps=1.0,
        min_arc_m=0.1,
        max_arc_m=0.6,
        min_horizon_sec=0.2,
        min_safety_horizon_sec=0.4,
        max_horizon_sec=0.6,
    )
    config = replace(config, **overrides)
    config.validate()
    return config


def _inputs(config, batch_size=2):
    scans = torch.full((batch_size, config.t_obs, config.n_scan), 3.0)
    tail = torch.tensor(
        [2.0, 0.1, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0], dtype=torch.float32
    ).expand(batch_size, -1)
    observation = torch.cat((scans.reshape(batch_size, -1), tail), dim=-1)
    covariance = torch.diag(torch.tensor([0.01, 0.01, 0.002])).expand(batch_size, -1, -1).clone()
    return TractorInputs(
        observation=observation,
        scan_valid=torch.ones_like(scans, dtype=torch.bool),
        motion_delta=torch.cat((
            torch.zeros((batch_size, config.t_obs - 1, 3)),
            torch.full((batch_size, config.t_obs - 1, 1), 0.1),
        ), dim=-1),
        motion_valid=torch.ones((batch_size, config.t_obs - 1), dtype=torch.bool),
        decision_timestamp_sec=torch.ones((batch_size, 1)),
        previous_intent_valid=torch.ones((batch_size, 1), dtype=torch.bool),
        previous_command_published=torch.zeros((batch_size, 2)),
        previous_command_valid=torch.ones((batch_size, 1), dtype=torch.bool),
        vehicle_response_valid=torch.ones((batch_size, 3), dtype=torch.bool),
        localization_covariance=covariance,
        localization_valid=torch.ones((batch_size, 1), dtype=torch.bool),
        localization_confidence=torch.ones((batch_size, 1)),
        localization_confidence_valid=torch.ones((batch_size, 1), dtype=torch.bool),
        sensor_freshness_sec=torch.full((batch_size, 1), 0.01),
        sensor_freshness_valid=torch.ones((batch_size, 1), dtype=torch.bool),
        scene_reset=torch.ones((batch_size, 1), dtype=torch.bool),
        response_reset=torch.ones((batch_size, 1), dtype=torch.bool),
    )


def test_config_fingerprint_covers_structural_fields():
    config = _small_config()
    assert config.fingerprint() != replace(config, num_candidates=5).fingerprint()
    assert config.fingerprint() != replace(config, footprint_radius_m=0.6).fingerprint()


def test_forward_shapes_and_risk_invariants():
    torch.manual_seed(7)
    config = _small_config()
    model = TractorTQC(config).eval()
    candidates, output, belief, _ = model(_inputs(config))

    assert candidates.normalized_actions.shape == (2, config.num_candidates, 3)
    assert candidates.is_stop.sum(dim=1).tolist() == [1, 1]
    assert output.return_quantiles.shape == (
        2, config.num_candidates, config.n_critics, config.n_quantiles
    )
    assert output.hazard.shape == (
        2, config.q_m_eff, config.risk_members, config.num_candidates,
        config.horizon_steps, config.n_causes,
    )
    assert output.candidate_valid.all()
    assert torch.all(output.hazard >= 0.0)
    assert torch.all(output.hazard.sum(dim=-1) < 1.0)
    assert torch.all(torch.diff(output.clearance_quantiles, dim=-1) >= 0.0)
    assert torch.all(torch.diff(output.stopping_quantiles, dim=-1) >= 0.0)
    assert belief.future_occupancy.shape == (
        2, config.horizon_steps, 4, config.grid_height, config.grid_width
    )
    assert torch.allclose(
        belief.future_occupancy.sum(dim=2),
        torch.ones_like(belief.future_occupancy[:, :, 0]),
        atol=1e-6,
    )
    rear_column = 0
    center_row = config.grid_height // 2
    assert torch.equal(belief.occupancy[:, 3, center_row, rear_column], torch.ones(2))
    assert torch.equal(
        belief.future_occupancy[:, :, 3, center_row, rear_column],
        torch.ones(2, config.horizon_steps),
    )


@pytest.mark.parametrize(
    "variant,residual_members,risk_members",
    [("tractor_residual_v1", 1, 1), ("tractor_ensemble_v1", 2, 2)],
)
def test_residual_and_ensemble_variants_execute_registered_axes(
    variant, residual_members, risk_members
):
    config = _small_config(
        variant_id=variant, residual_members=residual_members, risk_members=risk_members
    )
    model = TractorTQC(config).eval()
    candidates, output, _, _ = model(_inputs(config, batch_size=1))
    assert output.hazard.shape[:3] == (1, residual_members, risk_members)
    assert output.return_quantiles.shape[1] == candidates.normalized_actions.shape[1]


def test_tube_probability_mass_is_conserved_with_explicit_oob():
    config = _small_config()
    model = TractorTQC(config).eval()
    belief, context = model.encode(_inputs(config, batch_size=1))
    candidates = model.propose(belief)
    rollout = model.rollout(candidates, context, belief.plant_latent)
    tube = model.tube_rasterizer(rollout)
    inside = (tube.sample_weight * tube.sample_valid).sum(dim=-1)
    expected = tube.horizon_mask.to(inside.dtype)
    assert torch.allclose(inside + tube.oob_mass, expected, atol=1e-6)


def test_normalized_physical_action_round_trip():
    config = _small_config()
    adapter = TractorTQC(config).rollout
    normalized = torch.tensor([[[-1.0, -1.0, -1.0], [0.2, 0.4, 0.8], [1.0, 1.0, 1.0]]])
    assert torch.allclose(adapter.encode(adapter.decode(normalized)), normalized, atol=1e-6)


def test_nominal_rollout_matches_independent_cpu_output_grid_and_has_source_fingerprint():
    config = _small_config()
    model = TractorTQC(config).eval()
    belief, context = model.encode(_inputs(config, batch_size=1))
    normalized = torch.tensor([[[-0.7, -0.4, -0.8], [0.6, 0.5, 0.9]]])
    present = torch.ones(1, 2, dtype=torch.bool)
    candidates = CandidateSet(normalized, present, torch.zeros_like(present), "parity")
    with torch.inference_mode():
        model_rollout = model.rollout(candidates, context, belief.plant_latent)
    profile = load_profile("tractor_local_dynamic")
    dynamics = replace(
        profile.dynamics, dt_sec=config.dt_dyn_sec,
        horizon_min_sec=config.min_horizon_sec,
        horizon_max_sec=config.max_horizon_sec,
        min_safety_horizon_sec=config.min_safety_horizon_sec,
    )
    robot = replace(
        profile.robot, wheelbase_m=config.wheelbase_m,
        steering_limit_deg=math.degrees(config.steering_limit_rad),
        max_forward_speed_mps=config.max_speed_mps,
        accel_limit_mps2=config.accel_limit_mps2,
        brake_decel_mps2=config.brake_decel_mps2,
        steering_rate_deg_s=math.degrees(config.steering_rate_rad_s),
        speed_lag_tau_sec=config.speed_lag_tau_sec,
        collision_radius_m=config.footprint_radius_m,
    )
    physical = model.rollout.decode(normalized)[0]
    for candidate in range(2):
        kappa, speed, arc = physical[candidate].tolist()
        reference, mask = rollout_tractor_output_grid(
            VehicleState(v=0.2, steering=0.0), kappa, speed, arc,
            robot, dynamics, horizon_steps=config.horizon_steps,
            dt_out_sec=config.dt_out_sec,
        )
        expected = torch.tensor([
            [point.state.x, point.state.y, point.state.yaw] for point in reference.points
        ])
        assert torch.allclose(model_rollout.poses[0, 0, candidate], expected, atol=2e-6)
        assert model_rollout.horizon_mask[0, 0, candidate].tolist() == list(mask)
    fingerprint = nominal_rollout_source_fingerprint()
    assert len(fingerprint) == 64 and set(fingerprint) <= set("0123456789abcdef")


def test_candidate_permutation_only_permutes_scores():
    torch.manual_seed(11)
    config = _small_config()
    model = TractorTQC(config).eval()
    belief, context = model.encode(_inputs(config, batch_size=1))
    candidates = model.propose(belief)
    baseline = model.score(belief, context, candidates)
    order = torch.tensor([2, 0, 3, 1])
    permuted = model.score(belief, context, candidates.permute(order))

    assert torch.allclose(permuted.return_quantiles, baseline.return_quantiles[:, order], atol=1e-6)
    assert torch.allclose(permuted.hazard, baseline.hazard[:, :, :, order], atol=1e-6)
    assert torch.allclose(
        permuted.clearance_quantiles, baseline.clearance_quantiles[:, :, :, order], atol=1e-6
    )
    assert torch.equal(permuted.candidate_valid, baseline.candidate_valid[:, order])


def test_single_candidate_matches_its_batched_score_and_other_candidates_are_isolated():
    torch.manual_seed(17)
    config = _small_config()
    model = TractorTQC(config).eval()
    belief, context = model.encode(_inputs(config, batch_size=1))
    candidates = model.propose(belief)
    baseline = model.score(belief, context, candidates)
    index = 2
    single = CandidateSet(
        candidates.normalized_actions[:, index:index + 1],
        candidates.present[:, index:index + 1], candidates.is_stop[:, index:index + 1], "single",
    )
    single_output = model.score(belief, context, single)
    assert torch.allclose(single_output.return_quantiles[:, 0], baseline.return_quantiles[:, index], atol=1e-6)
    assert torch.allclose(single_output.hazard[:, :, :, 0], baseline.hazard[:, :, :, index], atol=1e-6)

    changed_actions = candidates.normalized_actions.clone()
    changed_actions[:, 0] = torch.tensor([1.0, -1.0, 1.0])
    changed = CandidateSet(changed_actions, candidates.present, candidates.is_stop, "perturbed")
    changed_output = model.score(belief, context, changed)
    assert torch.allclose(
        changed_output.return_quantiles[:, 1:], baseline.return_quantiles[:, 1:], atol=1e-6
    )
    assert torch.allclose(changed_output.hazard[:, :, :, 1:], baseline.hazard[:, :, :, 1:], atol=1e-6)


def test_scoring_does_not_mutate_shared_future():
    config = _small_config()
    model = TractorTQC(config).eval()
    belief, context = model.encode(_inputs(config, batch_size=1))
    before_feature = belief.future_feature.clone()
    before_occupancy = belief.future_occupancy.clone()
    model.score(belief, context, model.propose(belief))
    assert torch.equal(belief.future_feature, before_feature)
    assert torch.equal(belief.future_occupancy, before_occupancy)


def test_critic_signal_reaches_actor_candidate_mean():
    torch.manual_seed(13)
    config = _small_config()
    model = TractorTQC(config).train()
    belief, context = model.encode(_inputs(config, batch_size=1))
    output = model.score(belief, context, model.propose(belief))
    loss = output.return_quantiles.mean() + output.hazard.mean()
    loss.backward()
    gradient = model.actor.mean.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0.0


def test_identity_history_warp_preserves_evidence():
    config = _small_config()
    warp = SE2HistoryWarp(config)
    evidence = torch.randn(2, config.t_obs, 4, config.grid_height, config.grid_width)
    aligned, valid = warp(
        evidence,
        torch.zeros((2, config.t_obs - 1, 4)),
        torch.ones((2, config.t_obs - 1), dtype=torch.bool),
    )
    assert torch.allclose(aligned, evidence, atol=1e-6)
    assert valid.all()


def test_invalid_observation_holds_valid_prior_hidden_but_blocks_policy_health():
    config = _small_config()
    model = TractorTQC(config).eval()
    inputs = _inputs(config, batch_size=1)
    scene_prior = torch.randn(1, config.scene_channels, config.grid_height, config.grid_width)
    response_prior = torch.randn(1, config.plant_dim)
    invalid = replace(
        inputs,
        scan_valid=torch.zeros_like(inputs.scan_valid),
        sensor_freshness_valid=torch.zeros_like(inputs.sensor_freshness_valid),
        scene_reset=torch.zeros_like(inputs.scene_reset),
        response_reset=torch.zeros_like(inputs.response_reset),
        previous_scene_hidden=scene_prior,
        previous_scene_hidden_valid=torch.ones(1, 1, dtype=torch.bool),
        previous_response_hidden=response_prior,
        previous_response_hidden_valid=torch.ones(1, 1, dtype=torch.bool),
    )
    belief, _ = model.encode(invalid)
    assert torch.equal(belief.next_scene_hidden, scene_prior)
    assert not belief.health.model_valid.any()


def test_non_psd_localization_covariance_is_rejected():
    config = _small_config()
    inputs = _inputs(config, batch_size=1)
    covariance = torch.diag(torch.tensor([0.01, -0.01, 0.01])).unsqueeze(0)
    with pytest.raises(ValueError, match="positive semidefinite"):
        replace(inputs, localization_covariance=covariance).validate(config)
