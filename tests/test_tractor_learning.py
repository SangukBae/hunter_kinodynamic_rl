"""Loss, optimizer ownership, target and gradient-routing tests."""

import copy

import pytest
import torch

from hunter_kinodynamic_rl.rl.algorithms.tractor_tqc.agent import (
    TractorAgent, TractorAgentConfig, TractorRepresentationBatch, TractorRiskBatch,
    TractorTrainingBatch,
)
from hunter_kinodynamic_rl.rl.algorithms.tractor_tqc.losses import (
    competing_risk_nll, masked_pinball_loss, truncated_bellman_target,
)
from hunter_kinodynamic_rl.rl.algorithms.tractor_tqc.optimizer_groups import assert_disjoint_complete
from hunter_kinodynamic_rl.rl.networks.tractor import TractorConfig, TractorInputs
from hunter_kinodynamic_rl.rl.networks.tractor.contracts import CandidateSet


def _config():
    return TractorConfig(
        t_obs=2, n_scan=16, grid_height=16, grid_width=16, resolution_m=0.5,
        x_min_m=-4.0, y_min_m=-4.0, ray_free_samples=4, scene_channels=8,
        ego_dim=16, plant_dim=8, interaction_dim=16, num_candidates=4,
        horizon_steps=3, dt_out_sec=0.2, dt_dyn_sec=0.1,
        sparse_tube_samples=8, n_quantiles=5, n_severity_quantiles=3,
        severity_quantile_levels=(0.1, 0.5, 0.9),
        max_speed_mps=1.0, min_arc_m=0.1, max_arc_m=0.6,
        min_horizon_sec=0.2, min_safety_horizon_sec=0.4, max_horizon_sec=0.6,
    )


def _inputs(config, batch_size=1):
    scans = torch.full((batch_size, config.t_obs, config.n_scan), 2.0)
    tail = torch.tensor([2.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0]).expand(batch_size, -1)
    return TractorInputs(
        observation=torch.cat((scans.reshape(batch_size, -1), tail), -1),
        scan_valid=torch.ones_like(scans, dtype=torch.bool),
        motion_delta=torch.cat((
            torch.zeros(batch_size, config.t_obs - 1, 3),
            torch.full((batch_size, config.t_obs - 1, 1), 0.1),
        ), dim=-1),
        motion_valid=torch.ones(batch_size, config.t_obs - 1, dtype=torch.bool),
        decision_timestamp_sec=torch.ones(batch_size, 1),
        previous_intent_valid=torch.ones(batch_size, 1, dtype=torch.bool),
        previous_command_published=torch.zeros(batch_size, 2),
        previous_command_valid=torch.ones(batch_size, 1, dtype=torch.bool),
        vehicle_response_valid=torch.ones(batch_size, 3, dtype=torch.bool),
        localization_covariance=torch.eye(3).mul(0.001).expand(batch_size, -1, -1).clone(),
        localization_valid=torch.ones(batch_size, 1, dtype=torch.bool),
        localization_confidence=torch.ones(batch_size, 1),
        localization_confidence_valid=torch.ones(batch_size, 1, dtype=torch.bool),
        sensor_freshness_sec=torch.zeros(batch_size, 1),
        sensor_freshness_valid=torch.ones(batch_size, 1, dtype=torch.bool),
        scene_reset=torch.ones(batch_size, 1, dtype=torch.bool),
        response_reset=torch.ones(batch_size, 1, dtype=torch.bool),
    )


def test_bellman_bootstraps_time_limit_but_not_true_terminal():
    next_quantiles = torch.tensor([[[1.0, 2.0]], [[1.0, 2.0]]])
    target = truncated_bellman_target(
        reward=torch.tensor([[3.0], [3.0]]),
        discount_factor=torch.tensor([[0.5], [0.5]]),
        terminated=torch.tensor([[True], [False]]),
        next_quantiles=next_quantiles,
        next_log_probability=torch.zeros(2, 1),
        entropy_temperature=torch.tensor(0.2),
        top_quantiles_to_drop=0,
    )
    assert torch.equal(target[0], torch.tensor([3.0, 3.0]))
    assert torch.equal(target[1], torch.tensor([3.5, 4.0]))


def test_competing_event_and_censor_likelihood_matches_hand_value():
    hazard = torch.tensor([
        [[0.1, 0.2], [0.2, 0.1]],
        [[0.1, 0.2], [0.2, 0.1]],
    ])
    loss = competing_risk_nll(
        hazard,
        event_observed=torch.tensor([True, False]),
        event_step=torch.tensor([1, -1]),
        event_cause=torch.tensor([0, -1]),
        censor_step=torch.tensor([1, 1]),
        valid=torch.tensor([True, True]),
    )
    event = -(torch.log(torch.tensor(0.7)) + torch.log(torch.tensor(0.2)))
    censored = -2.0 * torch.log(torch.tensor(0.7))
    assert torch.allclose(loss, (event + censored) / 2.0)


def test_pinball_masks_missing_labels():
    prediction = torch.tensor([[0.0, 1.0, 2.0], [100.0, 101.0, 102.0]])
    target = torch.tensor([1.0, float("nan")])
    loss = masked_pinball_loss(prediction, target, torch.tensor([True, False]))
    assert torch.isfinite(loss)


def test_optimizer_groups_are_disjoint_and_complete():
    agent = TractorAgent(_config(), TractorAgentConfig(top_quantiles_to_drop_per_net=1))
    assert_disjoint_complete(agent.online, agent.parameter_groups())
    assert agent.representation_optimizer is agent.value_optimizer


def test_stage3_representation_objectives_update_only_the_registered_belief_path():
    config = _config()
    agent = TractorAgent(config, TractorAgentConfig(top_quantiles_to_drop_per_net=1))
    inputs = _inputs(config)
    grid = (1, config.grid_height, config.grid_width)
    batch = TractorRepresentationBatch(
        current=inputs,
        current_bev_class_target=torch.zeros(grid, dtype=torch.long),
        current_bev_class_valid=torch.ones(grid, dtype=torch.bool),
        current_dynamic_flow_target=torch.zeros(1, 2, *grid[1:]),
        current_dynamic_flow_valid=torch.ones(grid, dtype=torch.bool),
        future_bev_class_target=torch.zeros(
            1, config.horizon_steps, *grid[1:], dtype=torch.long,
        ),
        future_bev_class_valid=torch.ones(
            1, config.horizon_steps, *grid[1:], dtype=torch.bool,
        ),
        future_dynamic_flow_target=torch.zeros(
            1, config.horizon_steps, 2, *grid[1:],
        ),
        future_dynamic_flow_valid=torch.ones(
            1, config.horizon_steps, *grid[1:], dtype=torch.bool,
        ),
        vehicle_response_target=torch.tensor([[0.25, 0.0, 0.01]]),
        vehicle_response_target_valid=torch.ones(1, 3, dtype=torch.bool),
        action_normalized_requested=torch.zeros(1, 3),
        transition_dt_sec=torch.full((1, 1), 0.1),
    )
    before = {name: value.detach().clone() for name, value in agent.online.named_parameters()}
    metrics = agent.representation_step(batch)
    assert metrics["representation/update_applied"] == 1.0
    changed = [
        name for name, value in agent.online.named_parameters()
        if not torch.equal(value, before[name])
    ]
    assert changed
    assert all(name.startswith("belief_encoder.") for name in changed)


def test_critic_transaction_and_actor_gradient_routing():
    torch.manual_seed(5)
    config = _config()
    agent = TractorAgent(
        config,
        TractorAgentConfig(top_quantiles_to_drop_per_net=1, target_tau=0.1),
        target_seed=17,
    )
    inputs = _inputs(config)
    batch = TractorTrainingBatch(
        current=inputs,
        next=inputs,
        action_normalized_requested=torch.zeros(1, 3),
        reward=torch.ones(1, 1),
        discount_factor=torch.full((1, 1), 0.99),
        terminated=torch.zeros(1, 1, dtype=torch.bool),
        truncated=torch.ones(1, 1, dtype=torch.bool),
        bellman_sample_valid=torch.ones(1, 1, dtype=torch.bool),
    )
    critic_metrics = agent.critic_step(batch)
    assert critic_metrics["update/applied"] == 1.0
    assert agent.update_step == 1

    non_actor_before = {
        name: parameter.detach().clone()
        for name, parameter in agent.online.named_parameters()
        if not name.startswith("actor.")
    }
    actor_before = agent.online.actor.mean.weight.detach().clone()
    metrics = agent.actor_step(inputs)
    assert torch.isfinite(torch.tensor(list(metrics.values()))).all()
    assert not torch.equal(agent.online.actor.mean.weight, actor_before)
    for name, before in non_actor_before.items():
        assert torch.equal(dict(agent.online.named_parameters())[name], before), name


def test_target_policy_rng_state_round_trips():
    config = _config()
    first = TractorAgent(config, TractorAgentConfig(top_quantiles_to_drop_per_net=1), target_seed=3)
    state = first.extra_state_dict()
    expected = torch.randn(5, generator=first.target_generator)
    resumed = TractorAgent(config, TractorAgentConfig(top_quantiles_to_drop_per_net=1), target_seed=999)
    resumed.load_extra_state_dict(state)
    actual = torch.randn(5, generator=resumed.target_generator)
    assert torch.equal(actual, expected)


def test_actor_entropy_transaction_rolls_back_on_late_failure(monkeypatch):
    torch.manual_seed(31)
    config = _config()
    agent = TractorAgent(config, TractorAgentConfig(top_quantiles_to_drop_per_net=1))
    actor_before = copy.deepcopy(agent.online.actor.state_dict())
    temperature_before = copy.deepcopy(agent.temperature.state_dict())
    target_before = copy.deepcopy(agent.target.state_dict())
    rng_before = torch.get_rng_state().clone()

    def fail_entropy_step(*_args, **_kwargs):
        raise FloatingPointError("injected entropy optimizer failure")

    monkeypatch.setattr(agent.entropy_optimizer, "step", fail_entropy_step)
    with pytest.raises(FloatingPointError, match="injected entropy"):
        agent.actor_step(_inputs(config), update_target=True)
    for name, value in agent.online.actor.state_dict().items():
        assert torch.equal(value, actor_before[name]), name
    for name, value in agent.temperature.state_dict().items():
        assert torch.equal(value, temperature_before[name]), name
    for name, value in agent.target.state_dict().items():
        assert torch.equal(value, target_before[name]), name
    assert agent.actor_optimizer.state_dict()["state"] == {}
    assert agent.entropy_optimizer.state_dict()["state"] == {}
    assert torch.equal(torch.get_rng_state(), rng_before)


def test_risk_transaction_updates_only_risk_head():
    torch.manual_seed(8)
    config = _config()
    agent = TractorAgent(config, TractorAgentConfig(top_quantiles_to_drop_per_net=1))
    inputs = _inputs(config)
    actions = torch.zeros(1, config.num_candidates, 3)
    present = torch.ones(1, config.num_candidates, dtype=torch.bool)
    batch = TractorRiskBatch(
        current=inputs,
        candidates=CandidateSet(actions, present, torch.zeros_like(present), "fixture"),
        event_observed=torch.tensor([[True, False, False, False]]),
        event_step=torch.tensor([[1, -1, -1, -1]]),
        event_cause=torch.tensor([[0, -1, -1, -1]]),
        censor_step=torch.tensor([[1, 2, 2, 2]]),
        event_label_valid=present.clone(),
        clearance_m=torch.ones(1, config.num_candidates, config.horizon_steps),
        clearance_valid=torch.ones(1, config.num_candidates, config.horizon_steps, dtype=torch.bool),
        stopping_margin_m=torch.ones(1, config.num_candidates, config.horizon_steps),
        stopping_margin_valid=torch.ones(1, config.num_candidates, config.horizon_steps, dtype=torch.bool),
    )
    before = {name: value.detach().clone() for name, value in agent.online.named_parameters()}
    metrics = agent.risk_step(batch)
    assert metrics["risk/update_applied"] == 1.0
    changed = []
    for name, value in agent.online.named_parameters():
        if not torch.equal(value, before[name]):
            changed.append(name)
    assert changed
    assert all(name.startswith("risk_head.") for name in changed)


def test_stage4_atomic_feature_and_head_transaction_updates_only_registered_modules():
    torch.manual_seed(28)
    config = _config()
    agent = TractorAgent(config, TractorAgentConfig(top_quantiles_to_drop_per_net=1))
    inputs = _inputs(config)
    present = torch.ones(1, config.num_candidates, dtype=torch.bool)
    batch = TractorRiskBatch(
        current=inputs,
        candidates=CandidateSet(
            torch.zeros(1, config.num_candidates, 3), present,
            torch.zeros_like(present), "stage4-fixture",
        ),
        event_observed=torch.tensor([[True, False, False, False]]),
        event_step=torch.tensor([[1, -1, -1, -1]]),
        event_cause=torch.tensor([[0, -1, -1, -1]]),
        censor_step=torch.tensor([[1, 2, 2, 2]]),
        event_label_valid=present.clone(),
        clearance_m=torch.ones(1, config.num_candidates, config.horizon_steps),
        clearance_valid=torch.ones(
            1, config.num_candidates, config.horizon_steps, dtype=torch.bool,
        ),
        stopping_margin_m=torch.ones(1, config.num_candidates, config.horizon_steps),
        stopping_margin_valid=torch.ones(
            1, config.num_candidates, config.horizon_steps, dtype=torch.bool,
        ),
    )
    before = {name: value.detach().clone() for name, value in agent.online.named_parameters()}
    metrics = agent.atomic_stage4_risk_step(batch)
    assert metrics["stage4/update_applied"] == 1.0
    assert agent.update_step == 1
    changed = [
        name for name, value in agent.online.named_parameters()
        if not torch.equal(value, before[name])
    ]
    assert changed
    assert all(name.startswith(("interaction.", "temporal_aggregator.", "risk_head.")) for name in changed)


def test_atomic_value_risk_transaction_rolls_back_weights_optimizer_step_and_rng(monkeypatch):
    torch.manual_seed(18)
    config = _config()
    agent = TractorAgent(
        config, TractorAgentConfig(top_quantiles_to_drop_per_net=1), target_seed=29,
    )
    inputs = _inputs(config)
    value_batch = TractorTrainingBatch(
        current=inputs, next=inputs,
        action_normalized_requested=torch.zeros(1, 3), reward=torch.ones(1, 1),
        discount_factor=torch.full((1, 1), 0.99),
        terminated=torch.zeros(1, 1, dtype=torch.bool),
        truncated=torch.zeros(1, 1, dtype=torch.bool),
        bellman_sample_valid=torch.ones(1, 1, dtype=torch.bool),
    )
    present = torch.ones(1, config.num_candidates, dtype=torch.bool)
    risk_batch = TractorRiskBatch(
        current=inputs,
        candidates=CandidateSet(
            torch.zeros(1, config.num_candidates, 3), present,
            torch.zeros_like(present), "fixture",
        ),
        event_observed=torch.zeros_like(present),
        event_step=torch.full((1, config.num_candidates), -1, dtype=torch.long),
        event_cause=torch.full((1, config.num_candidates), -1, dtype=torch.long),
        censor_step=torch.full(
            (1, config.num_candidates), config.horizon_steps - 1, dtype=torch.long,
        ),
        event_label_valid=present.clone(),
        clearance_m=torch.ones(1, config.num_candidates, config.horizon_steps),
        clearance_valid=torch.ones(
            1, config.num_candidates, config.horizon_steps, dtype=torch.bool,
        ),
        stopping_margin_m=torch.ones(1, config.num_candidates, config.horizon_steps),
        stopping_margin_valid=torch.ones(
            1, config.num_candidates, config.horizon_steps, dtype=torch.bool,
        ),
    )
    weights_before = {
        name: value.detach().clone() for name, value in agent.online.state_dict().items()
    }
    rng_before = agent.target_generator.get_state().clone()

    real_objective = agent._risk_objective
    calls = {"count": 0}

    def _reject_head_side(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise FloatingPointError("injected risk-head failure")
        return real_objective(*args, **kwargs)

    monkeypatch.setattr(agent, "_risk_objective", _reject_head_side)
    with pytest.raises(FloatingPointError, match="injected"):
        agent.atomic_value_risk_step(value_batch, risk_batch)
    assert agent.update_step == 0
    assert torch.equal(agent.target_generator.get_state(), rng_before)
    for name, value in agent.online.state_dict().items():
        assert torch.equal(value, weights_before[name]), name
