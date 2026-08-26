import pytest

torch = pytest.importorskip("torch")

from hunter_kinodynamic_rl.config.schema import CounterfactualConfig, RiskConfig, TQCHyperparameters  # noqa: E402
from hunter_kinodynamic_rl.rl.algorithms.kinodynamic_tqc.agent import Agent as RiskAgent  # noqa: E402
from hunter_kinodynamic_rl.rl.algorithms.tqc.agent import Agent as VanillaAgent  # noqa: E402


def _make_hp() -> TQCHyperparameters:
    return TQCHyperparameters(n_critics=2, n_quantiles=25, top_quantiles_to_drop_per_net=2,
                               batch_size=8, buffer_size=100)


def _make_batch(batch_size, state_dim, action_dim, risk_target=None, valid=None,
                 candidate_actions=None, candidate_risk=None, candidate_valid_mask=None,
                 safer_alternative_margin=None):
    b = {
        "state": torch.randn(batch_size, state_dim),
        "action": torch.randn(batch_size, action_dim).clamp(-1, 1),
        "next_state": torch.randn(batch_size, state_dim),
        "reward": torch.randn(batch_size, 1),
        "not_done": torch.ones(batch_size, 1),
        "risk_target": risk_target if risk_target is not None else torch.rand(batch_size, 1),
        "valid": valid if valid is not None else torch.ones(batch_size, 1),
    }
    if candidate_actions is not None:
        b["candidate_actions_normalized"] = candidate_actions
        b["candidate_risk"] = candidate_risk
        b["candidate_valid_mask"] = candidate_valid_mask
    if safer_alternative_margin is not None:
        b["safer_alternative_margin"] = safer_alternative_margin
    return b


def test_risk_disabled_is_bytewise_identical_to_vanilla():
    """risk.enabled=False -> no risk critic constructed, actor/critic
    construction consumes identical RNG draws to the vanilla agent, so with
    the same seed the weights (and therefore selected actions) must match
    exactly (section 6: "임의로 바꾸지 않는다")."""
    torch.manual_seed(11)
    vanilla = VanillaAgent(state_dim=8, action_dim=3, max_action=1.0, hyperparameters=_make_hp(), device="cpu")
    torch.manual_seed(11)
    risk_off = RiskAgent(state_dim=8, action_dim=3, max_action=1.0, hyperparameters=_make_hp(),
                          risk_config=RiskConfig(enabled=False), device="cpu")

    assert risk_off.risk_critic is None
    for (n1, p1), (n2, p2) in zip(vanilla.actor.named_parameters(), risk_off.actor.named_parameters()):
        assert torch.equal(p1, p2), f"actor param {n1} diverged with risk disabled"

    import numpy as np
    state = np.random.RandomState(0).randn(8).astype(np.float32)
    a1 = vanilla.select_action(state, deterministic=True)
    a2 = risk_off.select_action(state, deterministic=True)
    assert (a1 == a2).all()


def test_risk_enabled_constructs_risk_critic_and_trains_without_nan():
    torch.manual_seed(5)
    agent = RiskAgent(
        state_dim=8, action_dim=3, max_action=1.0, hyperparameters=_make_hp(),
        risk_config=RiskConfig(enabled=True, actor_lambda=0.1, actor_penalty_warmup_updates=0,
                                min_valid_labels_per_batch=1),
        device="cpu",
    )
    assert agent.risk_critic is not None

    for _ in range(5):
        batch = _make_batch(8, 8, 3)
        metrics = agent.train_step(batch)
        assert "loss/risk_critic" in metrics and "loss/risk_actor_penalty" in metrics
        for v in metrics.values():
            if isinstance(v, float):
                assert v == v  # NaN check (NaN != NaN)

    for p in list(agent.actor.parameters()) + list(agent.risk_critic.parameters()):
        assert torch.isfinite(p).all()


def test_actor_penalty_gated_by_warmup():
    torch.manual_seed(9)
    agent = RiskAgent(
        state_dim=6, action_dim=3, max_action=1.0, hyperparameters=_make_hp(),
        risk_config=RiskConfig(enabled=True, actor_lambda=0.1, actor_penalty_warmup_updates=3,
                                min_valid_labels_per_batch=1),
        device="cpu",
    )
    for i in range(2):
        metrics = agent.train_step(_make_batch(8, 6, 3))
        assert metrics["risk/actor_penalty_active"] == 0.0
        assert "loss/risk_actor_penalty" not in metrics
    # By now risk_supervised_updates should have reached 3 (2 updates above + this one triggers check
    # AFTER incrementing, so the penalty activates once supervised_updates >= warmup).
    for _ in range(3):
        metrics = agent.train_step(_make_batch(8, 6, 3))
    assert metrics["risk/actor_penalty_active"] == 1.0
    assert "loss/risk_actor_penalty" in metrics


def test_risk_critic_handles_no_valid_labels_gracefully():
    """A batch with NO privileged risk labels recorded (valid=all-zero) must
    not crash or NaN out the risk critic, and must skip the supervised
    update entirely (not train on garbage)."""
    torch.manual_seed(6)
    agent = RiskAgent(state_dim=5, action_dim=2, max_action=1.0, hyperparameters=_make_hp(),
                       risk_config=RiskConfig(enabled=True, actor_lambda=0.1, min_valid_labels_per_batch=4),
                       device="cpu")
    batch = _make_batch(4, 5, 2, valid=torch.zeros(4, 1))
    metrics = agent.train_step(batch)
    assert metrics["loss/risk_critic"] == 0.0
    assert metrics["risk/valid_labels_in_batch"] == 0
    for p in agent.risk_critic.parameters():
        assert torch.isfinite(p).all()


def test_counterfactual_candidate_supervision_engages_when_enabled():
    torch.manual_seed(3)
    cf_cfg = CounterfactualConfig(enabled=True, num_candidates=4, candidate_supervision_weight=1.0)
    agent = RiskAgent(state_dim=6, action_dim=3, max_action=1.0, hyperparameters=_make_hp(),
                       risk_config=RiskConfig(enabled=True, min_valid_labels_per_batch=1),
                       counterfactual_config=cf_cfg, device="cpu")
    k = 4
    batch = _make_batch(
        8, 6, 3,
        candidate_actions=torch.randn(8, k, 3).clamp(-1, 1),
        candidate_risk=torch.rand(8, k),
        candidate_valid_mask=torch.ones(8, k),
    )
    metrics = agent.train_step(batch)
    assert metrics["loss/risk_candidate_supervision"] > 0.0
    for p in agent.risk_critic.parameters():
        assert torch.isfinite(p).all()


def test_counterfactual_margin_reweights_actor_penalty():
    """A batch where every transition has a LARGE positive
    safer_alternative_margin should produce a larger risk_actor_penalty
    magnitude than one with zero margin (all else equal, same seed/state)."""
    torch.manual_seed(4)
    cf_cfg = CounterfactualConfig(enabled=True, counterfactual_weight_scale=5.0)
    risk_cfg = RiskConfig(enabled=True, actor_lambda=0.1, actor_penalty_warmup_updates=0,
                           min_valid_labels_per_batch=1)

    torch.manual_seed(100)
    agent_a = RiskAgent(state_dim=6, action_dim=3, max_action=1.0, hyperparameters=_make_hp(),
                         risk_config=risk_cfg, counterfactual_config=cf_cfg, device="cpu")
    torch.manual_seed(100)
    agent_b = RiskAgent(state_dim=6, action_dim=3, max_action=1.0, hyperparameters=_make_hp(),
                         risk_config=risk_cfg, counterfactual_config=cf_cfg, device="cpu")

    torch.manual_seed(200)
    shared_batch_kwargs = dict(state=torch.randn(8, 6), action=torch.randn(8, 3).clamp(-1, 1))
    batch_zero_margin = {
        **shared_batch_kwargs, "next_state": torch.randn(8, 6), "reward": torch.randn(8, 1),
        "not_done": torch.ones(8, 1), "risk_target": torch.rand(8, 1), "valid": torch.ones(8, 1),
        "safer_alternative_margin": torch.zeros(8, 1),
    }
    batch_large_margin = dict(batch_zero_margin)
    batch_large_margin["safer_alternative_margin"] = torch.full((8, 1), 5.0)

    metrics_a = agent_a.train_step(batch_zero_margin)
    metrics_b = agent_b.train_step(batch_large_margin)
    assert abs(metrics_b["loss/risk_actor_penalty"]) >= abs(metrics_a["loss/risk_actor_penalty"])


def test_risk_enabled_vs_disabled_produce_different_actor_updates_on_the_same_batch():
    """Section 7's explicit requirement: risk-enabled and risk-disabled
    profiles must produce a MEASURABLY different actor loss/update on the
    identical batch once the actor penalty is active -- not just "the risk
    critic exists but never influences anything"."""
    hp = _make_hp()
    torch.manual_seed(42)
    disabled = RiskAgent(state_dim=6, action_dim=3, max_action=1.0, hyperparameters=hp,
                          risk_config=RiskConfig(enabled=False), device="cpu")
    torch.manual_seed(42)
    enabled = RiskAgent(state_dim=6, action_dim=3, max_action=1.0, hyperparameters=hp,
                         risk_config=RiskConfig(enabled=True, actor_lambda=0.5,
                                                 actor_penalty_warmup_updates=0, min_valid_labels_per_batch=1),
                         device="cpu")
    # Force a strong, non-degenerate risk critic so its penalty is not
    # numerically negligible: hand-set its output layer to a sizeable
    # NONZERO weight (never .zero_() the weight -- a zero weight makes
    # d(risk)/d(action) IDENTICALLY zero regardless of the hidden
    # activations, which would make the risk penalty a constant w.r.t. the
    # actor and silently defeat the very effect this test checks for).
    with torch.no_grad():
        enabled.risk_critic.net[-1].weight.fill_(0.5)
        enabled.risk_critic.net[-1].bias.fill_(-1.0)

    torch.manual_seed(7)
    batch = _make_batch(8, 6, 3)
    # Reseed to the SAME value immediately before EACH train_step call --
    # the actor's Gaussian policy uses reparameterized rsample() (a global
    # RNG draw), so calling disabled.train_step() first would otherwise
    # leave enabled.train_step() sampling different noise than disabled saw,
    # making the two actor updates diverge for a reason having nothing to
    # do with the risk penalty being tested here.
    torch.manual_seed(123)
    metrics_disabled = disabled.train_step(dict(batch))
    torch.manual_seed(123)
    metrics_enabled = enabled.train_step(batch)

    assert metrics_enabled["risk/actor_penalty_active"] == 1.0
    assert metrics_enabled["loss/risk_actor_penalty"] != 0.0
    # Aggregate over ALL actor parameters rather than requiring every
    # individual tensor to differ: Adam's bias-corrected first step is
    # roughly lr*sign(gradient), so a single low-dimensional bias vector
    # (e.g. mean.bias, only action_dim=3 elements) CAN coincidentally take
    # an identical step even with a genuinely nonzero, different gradient,
    # if the sign happens to match -- that is a property of Adam's step
    # geometry, not evidence the risk penalty had no effect.
    total_diff = sum(
        (p_dis - p_en).abs().sum().item()
        for (_, p_dis), (_, p_en) in zip(disabled.actor.named_parameters(), enabled.actor.named_parameters())
    )
    assert total_diff > 0.0, (
        "actor is byte-identical between risk-enabled and risk-disabled agents "
        "after one update -- the risk penalty had NO measurable effect on the actor"
    )


def test_actor_penalty_freezes_risk_critic_weights_but_not_the_action_gradient():
    """The core P0-6 gradient-rule requirement: the risk critic's OWN
    parameters must be frozen w.r.t. the actor's loss (only its own
    supervised loss may move them), while the gradient of the penalty
    w.r.t. the ACTION (and therefore back into the actor's own weights)
    must NOT be cut off. This directly exercises _actor_extra_loss's
    requires_grad_(False)/True bracketing (not just detach()), which a
    naive .detach() on the risk-critic output would also pass the
    "actor weights change" half of this test but would make d(risk)/d(action)
    identically zero -- so this test checks BOTH halves."""
    torch.manual_seed(21)
    agent = RiskAgent(
        state_dim=6, action_dim=3, max_action=1.0, hyperparameters=_make_hp(),
        risk_config=RiskConfig(enabled=True, actor_lambda=1.0, actor_penalty_warmup_updates=0,
                                min_valid_labels_per_batch=1),
        device="cpu",
    )
    # Prime the risk critic once so actor_penalty_warmup_updates is satisfied
    # going into the actual measurement step below.
    agent.train_step(_make_batch(8, 6, 3))
    risk_critic_before = [p.clone() for p in agent.risk_critic.parameters()]
    # The priming step's OWN supervised backward left nonzero .grad on the
    # risk critic's parameters -- clear it so the assertions below measure
    # ONLY what this isolated actor-penalty backward contributes.
    agent.risk_critic_optimizer.zero_grad()

    # Directly probe _actor_extra_loss's gradient rule in isolation, without
    # the confound of the risk critic's OWN supervised backward (which also
    # runs inside train_step and legitimately moves its weights).
    state = torch.randn(8, 6)
    actions_pi = torch.randn(8, 3, requires_grad=True)
    penalty, info = agent._actor_extra_loss(state, actions_pi)
    assert info["risk/actor_penalty_active"] == 1.0
    penalty.backward()

    assert actions_pi.grad is not None and torch.isfinite(actions_pi.grad).all() and \
        actions_pi.grad.abs().sum() > 0.0, (
        "d(risk_actor_penalty)/d(action) is zero/missing -- the risk critic's "
        "output was detached from the action instead of only having its "
        "OWN parameters frozen"
    )
    for p in agent.risk_critic.parameters():
        assert p.grad is None or torch.equal(p.grad, torch.zeros_like(p.grad)), (
            "risk critic parameter received a nonzero gradient from the actor "
            "penalty backward -- its weights were not frozen during the actor's "
            "forward pass through it"
        )
    for p_before, p_after in zip(risk_critic_before, agent.risk_critic.parameters()):
        assert torch.equal(p_before, p_after), (
            "risk critic weights changed after an isolated actor-penalty "
            "backward with no risk_critic_optimizer.step() call -- they must "
            "only move via their own supervised loss"
        )


def test_risk_critic_learns_to_discriminate_high_vs_low_risk_states():
    """Guards against reporting a shrinking risk-critic loss as "risk
    learning succeeded" when it is actually just collapsing to predict a
    constant near-zero output for an all-(or mostly)-zero-labeled batch
    (section 7: "단순 all-zero label로 risk critic MSE가 작아지는 것을 성공
    으로 판정하지 않는다"). Trains on a batch with CLEARLY separated
    high-risk (label≈1, state marked with a distinct feature) and low-risk
    (label≈0) transitions, then checks the trained critic actually predicts
    higher risk for held-out high-risk-marked states than low-risk ones."""
    torch.manual_seed(11)
    agent = RiskAgent(state_dim=4, action_dim=2, max_action=1.0, hyperparameters=_make_hp(),
                       risk_config=RiskConfig(enabled=True, min_valid_labels_per_batch=1), device="cpu")

    def make_state(is_risky: bool, n: int) -> torch.Tensor:
        base = torch.zeros(n, 4)
        base[:, 0] = 1.0 if is_risky else -1.0  # the one feature that predicts risk
        return base + 0.05 * torch.randn(n, 4)

    for _ in range(300):
        n_each = 8
        state = torch.cat([make_state(True, n_each), make_state(False, n_each)], dim=0)
        action = torch.zeros(2 * n_each, 2)
        risk_target = torch.cat([torch.ones(n_each, 1), torch.zeros(n_each, 1)], dim=0)
        batch = {
            "state": state, "action": action, "next_state": state.clone(),
            "reward": torch.zeros(2 * n_each, 1), "not_done": torch.ones(2 * n_each, 1),
            "risk_target": risk_target, "valid": torch.ones(2 * n_each, 1),
        }
        agent.train_step(batch)

    with torch.no_grad():
        held_out_risky = make_state(True, 32)
        held_out_safe = make_state(False, 32)
        pred_risky = agent.risk_critic(held_out_risky, torch.zeros(32, 2)).mean().item()
        pred_safe = agent.risk_critic(held_out_safe, torch.zeros(32, 2)).mean().item()

    assert pred_risky > pred_safe + 0.2, (
        f"risk critic did not learn to discriminate: pred_risky={pred_risky:.3f}, "
        f"pred_safe={pred_safe:.3f} -- loss may be shrinking via degenerate constant output, "
        "not genuine discrimination"
    )
