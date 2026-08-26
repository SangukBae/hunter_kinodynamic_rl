"""TQC network + vanilla agent tests. Torch-gated (skips cleanly when torch
is not installed -- e.g. on the host outside the Docker container), mirroring
drl_agent's own gating convention."""

import pytest

torch = pytest.importorskip("torch")

from hunter_kinodynamic_rl.config.schema import TQCHyperparameters  # noqa: E402
from hunter_kinodynamic_rl.rl.algorithms.tqc.agent import Agent  # noqa: E402
from hunter_kinodynamic_rl.rl.networks.tqc import Actor, Critic, quantile_huber_loss  # noqa: E402


def test_actor_forward_shape():
    actor = Actor(state_dim=10, action_dim=3)
    state = torch.randn(4, 10)
    action = actor.forward(state, deterministic=True)
    assert action.shape == (4, 3)
    assert torch.all(action >= -1.0) and torch.all(action <= 1.0)


def test_actor_action_log_prob_shapes():
    actor = Actor(state_dim=10, action_dim=3)
    state = torch.randn(4, 10)
    action, log_prob = actor.action_log_prob(state)
    assert action.shape == (4, 3)
    assert log_prob.shape == (4, 1)


def test_critic_forward_shape():
    critic = Critic(state_dim=10, action_dim=3, n_quantiles=25, n_critics=2)
    state = torch.randn(4, 10)
    action = torch.randn(4, 3)
    q = critic(state, action)
    assert q.shape == (4, 2, 25)


def test_quantile_huber_loss_is_finite_and_scalar():
    current = torch.randn(8, 2, 25)
    target = torch.randn(8, 1, 46)
    loss = quantile_huber_loss(current, target)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def _make_hp(**overrides) -> TQCHyperparameters:
    hp = TQCHyperparameters(n_critics=2, n_quantiles=25, top_quantiles_to_drop_per_net=2,
                             batch_size=8, buffer_size=100)
    for k, v in overrides.items():
        setattr(hp, k, v)
    return hp


def test_agent_select_action_bounded():
    torch.manual_seed(0)
    agent = Agent(state_dim=6, action_dim=3, max_action=1.0, hyperparameters=_make_hp(), device="cpu")
    import numpy as np
    state = np.random.randn(6).astype(np.float32)
    action = agent.select_action(state, deterministic=True)
    assert action.shape == (3,)
    assert (abs(action) <= 1.0 + 1e-5).all()


def test_agent_train_step_produces_finite_losses_and_no_nan_params():
    torch.manual_seed(0)
    agent = Agent(state_dim=6, action_dim=3, max_action=1.0, hyperparameters=_make_hp(), device="cpu")
    batch = 8
    state = torch.randn(batch, 6)
    action = torch.randn(batch, 3).clamp(-1, 1)
    next_state = torch.randn(batch, 6)
    reward = torch.randn(batch, 1)
    not_done = torch.ones(batch, 1)

    for _ in range(5):
        metrics = agent.train_step(state, action, next_state, reward, not_done)
        assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values() if isinstance(v, float))

    for p in list(agent.actor.parameters()) + list(agent.critic.parameters()):
        assert torch.isfinite(p).all(), "NaN/Inf detected in agent parameters after training"


def test_agent_critic_loss_decreases_on_repeated_batch():
    """Not a convergence proof -- just confirms the update actually reduces
    loss on a FIXED batch (a broken sign/target formula would make it grow)."""
    torch.manual_seed(1)
    agent = Agent(state_dim=6, action_dim=3, max_action=1.0, hyperparameters=_make_hp(), device="cpu")
    batch = 16
    torch.manual_seed(2)
    state = torch.randn(batch, 6)
    action = torch.randn(batch, 3).clamp(-1, 1)
    next_state = torch.randn(batch, 6)
    reward = torch.ones(batch, 1)
    not_done = torch.ones(batch, 1)

    losses = [agent.train_step(state, action, next_state, reward, not_done)["loss/critic"] for _ in range(50)]
    assert losses[-1] < losses[0]


# ---------------------------------------------------- section P0-4: update mechanics
def _batch(agent, batch=8):
    return (
        torch.randn(batch, agent.state_dim), torch.randn(batch, agent.action_dim).clamp(-1, 1),
        torch.randn(batch, agent.state_dim), torch.randn(batch, 1), torch.ones(batch, 1),
    )


def test_target_network_polyak_update_matches_tau_exactly():
    """section P0-4: the standard TQC/SAC target-update rule --
    target <- (1-tau)*target + tau*critic -- verified NUMERICALLY (not just
    "it changed"), with target_update_interval=1 so it fires every step."""
    torch.manual_seed(0)
    tau = 0.37  # deliberately far from any "looks right by luck" default
    agent = Agent(state_dim=6, action_dim=3, max_action=1.0,
                  hyperparameters=_make_hp(tau=tau, target_update_interval=1), device="cpu")
    pre_target = [p.clone() for p in agent.critic_target.parameters()]

    agent.train_step(*_batch(agent))  # this step's critic update happens BEFORE the target polyak blend

    post_critic = list(agent.critic.parameters())
    post_target = list(agent.critic_target.parameters())
    for before, critic_p, target_p in zip(pre_target, post_critic, post_target):
        expected = (1.0 - tau) * before + tau * critic_p.detach()
        assert torch.allclose(target_p.detach(), expected, atol=1e-6)


def test_target_network_frozen_between_update_intervals():
    """target_update_interval=5 -- the target must stay COMPLETELY frozen
    for steps 1-4, then update exactly on step 5."""
    torch.manual_seed(0)
    agent = Agent(state_dim=6, action_dim=3, max_action=1.0,
                  hyperparameters=_make_hp(tau=0.5, target_update_interval=5), device="cpu")
    pre_target = [p.clone() for p in agent.critic_target.parameters()]

    for _ in range(4):
        agent.train_step(*_batch(agent))
        for before, after in zip(pre_target, agent.critic_target.parameters()):
            assert torch.equal(before, after.detach())  # bit-for-bit frozen, not just "close"

    agent.train_step(*_batch(agent))  # the 5th step
    changed = any(not torch.equal(before, after.detach())
                  for before, after in zip(pre_target, agent.critic_target.parameters()))
    assert changed


def test_fixed_ent_coef_never_changes():
    """A non-'auto' ent_coef (a plain float) must stay EXACTLY constant --
    no ent_coef_optimizer/log_ent_coef exists in this mode at all."""
    torch.manual_seed(0)
    agent = Agent(state_dim=6, action_dim=3, max_action=1.0,
                  hyperparameters=_make_hp(ent_coef=0.2), device="cpu")
    assert agent.ent_coef_auto is False
    assert agent.ent_coef_optimizer is None
    for _ in range(5):
        metrics = agent.train_step(*_batch(agent))
        assert metrics["ent_coef"] == pytest.approx(0.2)


def test_auto_ent_coef_target_entropy_is_negative_action_dim():
    """The standard SAC/TQC convention this package's Agent claims to
    follow (agent.py's module docstring): target_entropy = -action_dim."""
    torch.manual_seed(0)
    agent = Agent(state_dim=6, action_dim=3, max_action=1.0,
                  hyperparameters=_make_hp(ent_coef="auto"), device="cpu")
    assert agent.target_entropy == pytest.approx(-3.0)


def test_auto_ent_coef_updates_and_stays_positive():
    """ent_coef = exp(log_ent_coef) is a variance-like quantity that must
    NEVER go negative or zero regardless of how the auto-tuning gradient
    pushes log_ent_coef -- and it must actually move (not silently freeze)
    when ent_coef_auto is on."""
    torch.manual_seed(0)
    agent = Agent(state_dim=6, action_dim=3, max_action=1.0,
                  hyperparameters=_make_hp(ent_coef="auto_1.0", ent_coef_lr=0.1), device="cpu")
    values = []
    for _ in range(10):
        metrics = agent.train_step(*_batch(agent))
        assert metrics["ent_coef"] > 0.0
        values.append(metrics["ent_coef"])
    assert len(set(values)) > 1  # actually changed across steps, not frozen


def test_ent_coef_state_roundtrip_restores_auto_tuned_value():
    """section P0-4: checkpoint roundtrip for the ONE piece of agent state
    that lives outside any nn.Module's state_dict (log_ent_coef is a bare
    tensor) -- ent_coef_state()/load_ent_coef_state() must round-trip it
    exactly, restoring both the SCALAR value used in metrics and its effect
    on the next train_step's target computation."""
    torch.manual_seed(0)
    agent = Agent(state_dim=6, action_dim=3, max_action=1.0,
                  hyperparameters=_make_hp(ent_coef="auto_1.0", ent_coef_lr=0.1), device="cpu")
    for _ in range(5):
        agent.train_step(*_batch(agent))
    saved = agent.ent_coef_state()

    fresh = Agent(state_dim=6, action_dim=3, max_action=1.0,
                  hyperparameters=_make_hp(ent_coef="auto_1.0", ent_coef_lr=0.1), device="cpu")
    assert fresh.ent_coef_state() != pytest.approx(saved)  # sanity: actually different before restore
    fresh.load_ent_coef_state(saved)
    assert fresh.ent_coef_state() == pytest.approx(saved)
