"""Regression tests for the "Vanilla SAC" algorithm-choice baseline
(docs/RESEARCH_PROTOCOL.md section 34/35): rl/algorithms/sac/agent.py
reuses rl/networks/tqc.py's Actor/Critic verbatim (Critic(n_quantiles=1)
collapsing to a plain twin-Q critic pair) rather than a second from-scratch
network definition -- see that module's docstring for the full rationale.
"""

import pytest

torch = pytest.importorskip("torch")

import numpy as np  # noqa: E402

from hunter_kinodynamic_rl.config.schema import SACHyperparameters  # noqa: E402
from hunter_kinodynamic_rl.rl.algorithms.sac.agent import Agent  # noqa: E402
from hunter_kinodynamic_rl.rl.networks.tqc import Actor  # noqa: E402


def _make_hp(**overrides) -> SACHyperparameters:
    kwargs = dict(batch_size=8, buffer_size=100, n_critics=2)
    kwargs.update(overrides)
    return SACHyperparameters(**kwargs)


def _make_batch(batch_size=8, state_dim=8, action_dim=3):
    return dict(
        state=torch.randn(batch_size, state_dim),
        action=torch.randn(batch_size, action_dim).clamp(-1, 1),
        next_state=torch.randn(batch_size, state_dim),
        reward=torch.randn(batch_size, 1),
        not_done=torch.ones(batch_size, 1),
    )


def test_actor_reuses_the_verbatim_tqc_actor_class():
    """SAC's actor must be the SAME Gaussian-policy class TQC uses -- a
    squashed-Gaussian policy has no algorithm-specific logic, and this
    package deliberately does not duplicate a second copy of it."""
    torch.manual_seed(3)
    agent = Agent(state_dim=8, action_dim=3, max_action=1.0, hyperparameters=_make_hp(), device="cpu")
    assert isinstance(agent.actor, Actor)


def test_critic_collapses_to_scalar_twin_q():
    agent = Agent(state_dim=8, action_dim=3, max_action=1.0, hyperparameters=_make_hp(n_critics=2), device="cpu")
    assert agent.critic.n_quantiles == 1
    assert agent.critic.n_critics == 2
    state = torch.randn(4, 8)
    action = torch.randn(4, 3).clamp(-1, 1)
    out = agent.critic(state, action)
    assert out.shape == (4, 2, 1)  # (batch, n_critics, n_quantiles=1)


def test_train_step_runs_without_nan_and_reports_expected_metrics():
    torch.manual_seed(7)
    agent = Agent(state_dim=8, action_dim=3, max_action=1.0, hyperparameters=_make_hp(), device="cpu")
    for _ in range(10):
        metrics = agent.train_step(**_make_batch())
        assert set(metrics.keys()) == {"loss/critic", "loss/actor", "ent_coef", "training_steps"}
        for v in metrics.values():
            assert v == v  # NaN check (NaN != NaN)
    for p in list(agent.actor.parameters()) + list(agent.critic.parameters()):
        assert torch.isfinite(p).all()


def test_select_action_is_deterministic_when_requested():
    torch.manual_seed(1)
    agent = Agent(state_dim=8, action_dim=3, max_action=1.0, hyperparameters=_make_hp(), device="cpu")
    state = np.random.RandomState(0).randn(8).astype(np.float32)
    a1 = agent.select_action(state, deterministic=True)
    a2 = agent.select_action(state, deterministic=True)
    assert (a1 == a2).all()


def test_select_action_respects_max_action_scale():
    torch.manual_seed(2)
    agent = Agent(state_dim=8, action_dim=3, max_action=2.5, hyperparameters=_make_hp(), device="cpu")
    state = np.random.RandomState(1).randn(8).astype(np.float32)
    action = agent.select_action(state, deterministic=True)
    assert np.all(np.abs(action) <= 2.5 + 1e-5)  # tanh output in [-1,1] * max_action


def test_target_critic_tracks_online_critic_via_polyak_update():
    torch.manual_seed(4)
    agent = Agent(state_dim=8, action_dim=3, max_action=1.0,
                  hyperparameters=_make_hp(tau=1.0, target_update_interval=1), device="cpu")
    agent.train_step(**_make_batch())
    for p_online, p_target in zip(agent.critic.parameters(), agent.critic_target.parameters()):
        assert torch.allclose(p_online, p_target)  # tau=1.0 -> full hard copy each step


def test_fixed_ent_coef_never_constructs_an_optimizer():
    agent = Agent(state_dim=8, action_dim=3, max_action=1.0,
                  hyperparameters=_make_hp(ent_coef="0.2"), device="cpu")
    assert agent.ent_coef_auto is False
    assert agent.ent_coef_optimizer is None
    assert agent.ent_coef_state() is None
    metrics = agent.train_step(**_make_batch())
    assert metrics["ent_coef"] == pytest.approx(0.2)


def test_checkpoint_components_includes_ent_coef_optimizer_only_when_auto():
    auto_agent = Agent(state_dim=8, action_dim=3, max_action=1.0,
                        hyperparameters=_make_hp(ent_coef="auto_1.0"), device="cpu")
    assert "ent_coef_optimizer" in auto_agent.checkpoint_components()

    fixed_agent = Agent(state_dim=8, action_dim=3, max_action=1.0,
                         hyperparameters=_make_hp(ent_coef="0.2"), device="cpu")
    assert "ent_coef_optimizer" not in fixed_agent.checkpoint_components()
