"""Executable fairness and learning-path tests for B1--B8."""

import torch

from hunter_kinodynamic_rl.config.comparison import (
    BASELINE_METHODS, ComparisonModelConfig, load_comparison_contract,
)
from hunter_kinodynamic_rl.rl.algorithms.comparison_baselines import ComparisonAgent
from hunter_kinodynamic_rl.rl.algorithms.comparison_baselines.model import (
    CurrentTQCActorAdapter, CurrentTQCObservation,
)
from hunter_kinodynamic_rl.rl.networks.tqc import Critic as CurrentTQCCritic
from hunter_kinodynamic_rl.rl.algorithms.tractor_tqc import (
    TractorAgent, TractorAgentConfig, TractorRiskBatch, TractorTrainingBatch,
)
from hunter_kinodynamic_rl.rl.networks.tractor import TractorConfig, TractorInputs
from hunter_kinodynamic_rl.rl.networks.tractor.contracts import CandidateSet


AXES = {
    "B1": ("current_tqc_328d", "none", "none"),
    "B2": ("flat_parameter_matched", "none", "none"),
    "B3": ("recurrent_vector", "none", "none"),
    "B4": ("factorized_ego_warped_bev", "implicit_concat", "none"),
    "B5": ("factorized_ego_warped_bev", "cross_attention", "none"),
    "B6": ("compact_latent_world_model", "latent_rollout", "none"),
    "B7": ("flat_328d", "none", "cvar_return"),
    "B8": ("flat_328d", "none", "scalar_endpoint"),
}


def _input_config():
    return TractorConfig(
        t_obs=2, n_scan=8, grid_height=8, grid_width=8, resolution_m=1.0,
        x_min_m=-4.0, y_min_m=-4.0, ray_free_samples=2, scene_channels=8,
        ego_dim=8, plant_dim=8, interaction_dim=8, num_candidates=2,
        horizon_steps=2, dt_out_sec=0.2, dt_dyn_sec=0.1, sparse_tube_samples=4,
        n_quantiles=3, n_severity_quantiles=2, max_speed_mps=1.0,
        severity_quantile_levels=(0.1, 0.9), min_arc_m=0.1, max_arc_m=0.4,
        min_horizon_sec=0.2, min_safety_horizon_sec=0.2, max_horizon_sec=0.4,
    )


def _model_config(method):
    input_cfg = _input_config()
    representation, interaction, risk = AXES[method]
    return ComparisonModelConfig(
        method_id=method, representation=representation, interaction=interaction, risk=risk,
        observation_dim=input_cfg.observation_dim, t_obs=input_cfg.t_obs,
        n_scan=input_cfg.n_scan, tail_dim=input_cfg.tail_dim,
        n_critics=input_cfg.n_critics, n_quantiles=input_cfg.n_quantiles,
        log_std_min=input_cfg.log_std_min, log_std_max=input_cfg.log_std_max,
        encoder_hidden=32,
        latent_dim=input_cfg.observation_dim if method == "B1" else 16,
        recurrent_hidden=16, attention_heads=2,
        cvar_fraction=0.25 if method == "B7" else 1.0,
        dynamics_loss_weight=0.25 if method == "B6" else 0.0,
        actor_risk_weight=1.0 if method == "B8" else 0.0,
    )


def _inputs(batch=2):
    cfg = _input_config()
    scans = torch.full((batch, cfg.t_obs, cfg.n_scan), 2.0)
    tail = torch.tensor([2.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0]).expand(batch, -1)
    return TractorInputs(
        observation=torch.cat((scans.reshape(batch, -1), tail), -1),
        scan_valid=torch.ones_like(scans, dtype=torch.bool),
        motion_delta=torch.cat((
            torch.zeros(batch, cfg.t_obs - 1, 3),
            torch.full((batch, cfg.t_obs - 1, 1), 0.1),
        ), -1),
        motion_valid=torch.ones(batch, cfg.t_obs - 1, dtype=torch.bool),
        decision_timestamp_sec=torch.ones(batch, 1),
        previous_intent_valid=torch.ones(batch, 1, dtype=torch.bool),
        previous_command_published=torch.zeros(batch, 2),
        previous_command_valid=torch.ones(batch, 1, dtype=torch.bool),
        vehicle_response_valid=torch.ones(batch, 3, dtype=torch.bool),
        localization_covariance=torch.eye(3).mul(0.001).expand(batch, -1, -1).clone(),
        localization_valid=torch.ones(batch, 1, dtype=torch.bool),
        localization_confidence=torch.ones(batch, 1),
        localization_confidence_valid=torch.ones(batch, 1, dtype=torch.bool),
        sensor_freshness_sec=torch.zeros(batch, 1),
        sensor_freshness_valid=torch.ones(batch, 1, dtype=torch.bool),
        scene_reset=torch.ones(batch, 1, dtype=torch.bool),
        response_reset=torch.ones(batch, 1, dtype=torch.bool),
    )


def test_all_registered_baselines_execute_the_shared_action_contract():
    inputs = _inputs()
    for method in BASELINE_METHODS:
        agent = ComparisonAgent(
            _model_config(method), _input_config(),
            TractorAgentConfig(top_quantiles_to_drop_per_net=1),
        )
        state = agent.online.encode(inputs)
        action, log_probability = agent.online.sample_action(state)
        quantiles = agent.online.quantiles_from_state(state, action)
        assert action.shape == (2, 3)
        assert log_probability.shape == (2, 1)
        assert quantiles.shape == (2, 2, 3)
        assert torch.isfinite(quantiles).all()
        if method == "B8":
            assert agent.online.risk_probability_from_state(state, action).shape == (2, 1)


def test_b1_is_the_current_direct_328d_tqc_network_not_a_new_latent_mlp():
    config = _model_config("B1")
    agent = ComparisonAgent(
        config, _input_config(), TractorAgentConfig(top_quantiles_to_drop_per_net=1),
    )
    assert isinstance(agent.online.encoder, CurrentTQCObservation)
    assert isinstance(agent.online.actor, CurrentTQCActorAdapter)
    assert isinstance(agent.online.critics, CurrentTQCCritic)
    assert agent.representation_optimizer is None
    state = agent.online.encode(_inputs())
    assert torch.equal(state.vector, _inputs().observation)
    assert agent.online.actor.actor.log_std_min == -20.0
    assert agent.online.actor.actor.log_std_max == 2.0


def test_b2_is_parameter_matched_to_a7_under_frozen_full_config():
    contract = load_comparison_contract(None, "B2")
    baseline = ComparisonAgent(contract["model"], contract["input_model"], contract["agent"])
    tractor = TractorAgent(contract["input_model"], contract["agent"])
    baseline_count = sum(parameter.numel() for parameter in baseline.online.parameters())
    tractor_count = sum(parameter.numel() for parameter in tractor.online.parameters())
    relative_error = abs(baseline_count - tractor_count) / tractor_count
    assert relative_error <= contract["parameter_match"]["relative_tolerance"]


def test_b6_value_update_and_b8_risk_update_are_real_gradient_paths():
    inputs = _inputs()
    batch = TractorTrainingBatch(
        current=inputs, next=inputs,
        action_normalized_requested=torch.zeros(2, 3), reward=torch.ones(2, 1),
        discount_factor=torch.full((2, 1), 0.99),
        terminated=torch.zeros(2, 1, dtype=torch.bool),
        truncated=torch.zeros(2, 1, dtype=torch.bool),
        bellman_sample_valid=torch.ones(2, 1, dtype=torch.bool),
    )
    b1 = ComparisonAgent(
        _model_config("B1"), _input_config(),
        TractorAgentConfig(top_quantiles_to_drop_per_net=1),
    )
    b1_critic_before = next(b1.online.critics.parameters()).detach().clone()
    b1_metrics = b1.critic_step(batch)
    assert b1_metrics["update/applied"] == 1.0
    assert not torch.equal(b1_critic_before, next(b1.online.critics.parameters()))

    b6 = ComparisonAgent(
        _model_config("B6"), _input_config(),
        TractorAgentConfig(top_quantiles_to_drop_per_net=1),
    )
    transition_before = next(b6.online.latent_transition.parameters()).detach().clone()
    metrics = b6.critic_step(batch)
    assert metrics["update/applied"] == 1.0
    assert not torch.equal(transition_before, next(b6.online.latent_transition.parameters()))

    b8 = ComparisonAgent(
        _model_config("B8"), _input_config(),
        TractorAgentConfig(top_quantiles_to_drop_per_net=1),
    )
    candidates = CandidateSet(
        torch.zeros(2, 2, 3), torch.ones(2, 2, dtype=torch.bool),
        torch.zeros(2, 2, dtype=torch.bool), "fixture",
    )
    risk_batch = TractorRiskBatch(
        current=inputs, candidates=candidates,
        event_observed=torch.tensor([[True, False], [False, True]]),
        event_step=torch.tensor([[0, -1], [-1, 0]]),
        event_cause=torch.tensor([[0, -1], [-1, 1]]),
        censor_step=torch.tensor([[0, 1], [1, 0]]),
        event_label_valid=torch.ones(2, 2, dtype=torch.bool),
        clearance_m=torch.zeros(2, 2, 2),
        clearance_valid=torch.ones(2, 2, 2, dtype=torch.bool),
        stopping_margin_m=torch.zeros(2, 2, 2),
        stopping_margin_valid=torch.ones(2, 2, 2, dtype=torch.bool),
    )
    risk_before = next(b8.online.risk_head.parameters()).detach().clone()
    risk_metrics = b8.risk_step(risk_batch)
    assert risk_metrics["risk/update_applied"] == 1.0
    assert not torch.equal(risk_before, next(b8.online.risk_head.parameters()))
