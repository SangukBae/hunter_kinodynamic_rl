"""Section P0-7: the A-F ablation matrix's network/replay/loss differences
must be proven by REAL integration tests against the actual shipped profile
YAML files (config/profiles/*.yaml, loaded through load_profile() exactly as
train_node.py does) -- not just hand-built dataclasses in test code, and not
just "the profile loads without error"."""

import pytest

torch = pytest.importorskip("torch")

from hunter_kinodynamic_rl.config.loader import load_profile  # noqa: E402
from hunter_kinodynamic_rl.rl.algorithms.kinodynamic_tqc.agent import Agent as RiskAgent  # noqa: E402
from hunter_kinodynamic_rl.rl.algorithms.tqc.agent import Agent as VanillaAgent  # noqa: E402

STATE_DIM = 8
ACTION_DIM = 3


def _make_batch(batch_size, candidates_k=None, margin=None):
    b = {
        "state": torch.randn(batch_size, STATE_DIM),
        "action": torch.randn(batch_size, ACTION_DIM).clamp(-1, 1),
        "next_state": torch.randn(batch_size, STATE_DIM),
        "reward": torch.randn(batch_size, 1),
        "not_done": torch.ones(batch_size, 1),
        "risk_target": torch.rand(batch_size, 1),
        "valid": torch.ones(batch_size, 1),
    }
    if candidates_k is not None:
        b["candidate_actions_normalized"] = torch.randn(batch_size, candidates_k, ACTION_DIM).clamp(-1, 1)
        b["candidate_risk"] = torch.rand(batch_size, candidates_k)
        b["candidate_valid_mask"] = torch.ones(batch_size, candidates_k)
    if margin is not None:
        b["safer_alternative_margin"] = torch.full((batch_size, 1), margin)
    return b


# --------------------------------------------------------------- A: vanilla parity
def test_ablation_a_baseline_profile_is_byte_identical_to_vanilla_tqc():
    """`baseline_tqc.yaml` must drive train_node.py to the PLAIN VanillaAgent
    code path (features.risk_critic=False), and its hyperparameters must
    produce the exact same network as constructing VanillaAgent directly --
    ablation A is the "fair apples-to-apples vanilla TQC" comparison point
    and must not be silently perturbed by any risk/counterfactual wiring."""
    profile = load_profile("baseline_tqc")
    assert profile.action_space.mode == "legacy_waypoint"
    assert profile.features.risk_critic is False
    assert profile.features.counterfactual_risk is False
    assert profile.risk.enabled is False

    torch.manual_seed(3)
    direct = VanillaAgent(STATE_DIM, ACTION_DIM, max_action=1.0,
                           hyperparameters=profile.hyperparameters, device="cpu")
    torch.manual_seed(3)
    via_risk_agent_disabled = RiskAgent(STATE_DIM, ACTION_DIM, max_action=1.0,
                                         hyperparameters=profile.hyperparameters,
                                         risk_config=profile.risk,
                                         counterfactual_config=profile.counterfactual, device="cpu")
    assert via_risk_agent_disabled.risk_critic is None
    for (n, p1), (_, p2) in zip(direct.actor.named_parameters(), via_risk_agent_disabled.actor.named_parameters()):
        assert torch.equal(p1, p2), f"actor param {n} diverged for the vanilla-equivalent ablation A profile"


# ------------------------------------------------------------- B/C: structural flags
def test_ablation_b_profile_enables_trajectory_action_only():
    profile = load_profile("kinodynamic_tqc")
    assert profile.action_space.mode == "trajectory"
    assert profile.features.ackermann_rollout is True
    assert profile.features.temporal_context is False
    assert profile.features.risk_critic is False
    assert profile.features.counterfactual_risk is False


def test_ablation_c_profile_adds_temporal_context_only():
    profile = load_profile("kinodynamic_tqc_temporal")
    assert profile.features.temporal_context is True
    assert profile.observation.frame_stack == 4
    assert profile.features.risk_critic is False
    assert profile.features.counterfactual_risk is False


# ----------------------------------------------------------- D vs E: actor penalty
def test_ablation_d_trains_only_the_risk_critic_not_the_actor():
    """D (`kinodynamic_tqc_risk_supervised_only.yaml`): risk.actor_lambda=0.0
    must make the risk critic train (nonzero supervised loss, moving
    weights) while leaving NO trace of an actor penalty in the metrics and
    producing an actor update IDENTICAL to a risk-disabled agent fed the
    same batch/seed."""
    profile = load_profile("kinodynamic_tqc_risk_supervised_only")
    assert profile.risk.enabled is True
    assert profile.risk.actor_lambda == pytest.approx(0.0)
    assert profile.features.risk_critic is True
    assert profile.features.counterfactual_risk is False

    torch.manual_seed(17)
    d_agent = RiskAgent(STATE_DIM, ACTION_DIM, max_action=1.0, hyperparameters=profile.hyperparameters,
                         risk_config=profile.risk, counterfactual_config=profile.counterfactual, device="cpu")
    torch.manual_seed(17)
    vanilla = VanillaAgent(STATE_DIM, ACTION_DIM, max_action=1.0,
                            hyperparameters=profile.hyperparameters, device="cpu")

    torch.manual_seed(200)
    batch_for_d = _make_batch(8)
    torch.manual_seed(200)
    batch_for_vanilla = _make_batch(8)

    # Reseed to the SAME value immediately before EACH train_step call: the
    # actor's Gaussian policy uses reparameterized rsample() (a global RNG
    # draw), so calling d_agent's train_step first would otherwise leave
    # vanilla's train_step sampling different noise, making the two updates
    # diverge for a reason unrelated to the thing this test checks.
    torch.manual_seed(321)
    metrics = d_agent.train_step(batch_for_d)
    torch.manual_seed(321)
    vanilla.train_step(batch_for_vanilla["state"], batch_for_vanilla["action"], batch_for_vanilla["next_state"],
                        batch_for_vanilla["reward"], batch_for_vanilla["not_done"])

    assert metrics.get("risk/actor_penalty_active", 0.0) == 0.0
    assert "loss/risk_actor_penalty" not in metrics
    assert metrics["loss/risk_critic"] > 0.0, "risk critic must still train under D -- only the actor is unaffected"
    for (n, p_d), (_, p_v) in zip(d_agent.actor.named_parameters(), vanilla.actor.named_parameters()):
        assert torch.equal(p_d, p_v), (
            f"actor param {n} diverged between D (actor_lambda=0) and a plain vanilla update on the "
            "identical batch/seed -- D must produce a byte-identical actor update to vanilla TQC"
        )


def test_ablation_e_actor_loss_differs_from_d_on_the_same_batch():
    """E (`kinodynamic_tqc_risk.yaml`) is D + a nonzero risk-aware actor
    penalty -- section P0-7's explicit requirement that E's actor loss must
    differ from D's, proven here by loading BOTH real profiles and feeding
    them the identical batch/seed."""
    profile_d = load_profile("kinodynamic_tqc_risk_supervised_only")
    profile_e = load_profile("kinodynamic_tqc_risk")
    assert profile_e.risk.actor_lambda > profile_d.risk.actor_lambda == 0.0

    torch.manual_seed(9)
    d_agent = RiskAgent(STATE_DIM, ACTION_DIM, max_action=1.0, hyperparameters=profile_d.hyperparameters,
                         risk_config=profile_d.risk, counterfactual_config=profile_d.counterfactual, device="cpu")
    torch.manual_seed(9)
    e_agent = RiskAgent(STATE_DIM, ACTION_DIM, max_action=1.0, hyperparameters=profile_e.hyperparameters,
                         risk_config=profile_e.risk, counterfactual_config=profile_e.counterfactual, device="cpu")
    # Force a strong, non-degenerate risk critic on BOTH so E's penalty is
    # not numerically negligible by random-init luck (mirrors
    # test_kinodynamic_tqc.py's identical trick) -- a NONZERO weight, never
    # .zero_() (a zero weight makes d(risk)/d(action) identically zero,
    # silently defeating the effect this test checks for).
    for agent in (d_agent, e_agent):
        with torch.no_grad():
            agent.risk_critic.net[-1].weight.fill_(0.5)
            agent.risk_critic.net[-1].bias.fill_(-1.0)
    # D's actor_penalty_warmup_updates default (500) would otherwise gate E's
    # penalty off for this single-step test.
    d_agent.risk_cfg.actor_penalty_warmup_updates = 0
    e_agent.risk_cfg.actor_penalty_warmup_updates = 0

    torch.manual_seed(500)
    batch_d = _make_batch(8)
    torch.manual_seed(500)
    batch_e = _make_batch(8)

    # Reseed to the SAME value immediately before EACH train_step call (see
    # the identical comment in test_ablation_d_... above -- rsample() draws
    # from the global RNG, so D's train_step would otherwise leave E's call
    # sampling different noise, contaminating the comparison).
    torch.manual_seed(600)
    metrics_d = d_agent.train_step(batch_d)
    torch.manual_seed(600)
    metrics_e = e_agent.train_step(batch_e)

    assert metrics_d.get("risk/actor_penalty_active", 0.0) == 0.0
    assert metrics_e["risk/actor_penalty_active"] == 1.0
    assert metrics_e["loss/risk_actor_penalty"] != 0.0
    # Aggregate over ALL actor parameters (see the identical comment in
    # test_kinodynamic_tqc.py's analogous test for why per-tensor equality
    # is the wrong check under Adam's bias-corrected first-step geometry).
    total_diff = sum(
        (p_d - p_e).abs().sum().item()
        for (_, p_d), (_, p_e) in zip(d_agent.actor.named_parameters(), e_agent.actor.named_parameters())
    )
    assert total_diff > 0.0, (
        "actor is byte-identical between D and E after one update -- E's risk-aware actor "
        "penalty had no measurable effect relative to D"
    )


# --------------------------------------------------------------- F: counterfactual
def test_ablation_f_activates_candidate_replay_and_loss_that_e_does_not_use():
    """F (`kinodynamic_tqc_counterfactual.yaml`) must be the only profile in
    D/E/F that consumes candidate_actions_normalized/candidate_risk/
    candidate_valid_mask -- E's counterfactual.enabled=False must ignore
    those same batch keys even when present (so a partially-populated batch
    from a mixed-profile buffer can never silently leak candidate
    supervision into a non-F run)."""
    profile_f = load_profile("kinodynamic_tqc_counterfactual")
    profile_e = load_profile("kinodynamic_tqc_risk")
    assert profile_f.counterfactual.enabled is True
    assert profile_f.counterfactual.num_candidates == 8
    assert profile_e.counterfactual.enabled is False

    torch.manual_seed(4)
    f_agent = RiskAgent(STATE_DIM, ACTION_DIM, max_action=1.0, hyperparameters=profile_f.hyperparameters,
                         risk_config=profile_f.risk, counterfactual_config=profile_f.counterfactual, device="cpu")
    torch.manual_seed(4)
    e_agent = RiskAgent(STATE_DIM, ACTION_DIM, max_action=1.0, hyperparameters=profile_e.hyperparameters,
                         risk_config=profile_e.risk, counterfactual_config=profile_e.counterfactual, device="cpu")

    torch.manual_seed(700)
    batch = _make_batch(8, candidates_k=profile_f.counterfactual.num_candidates, margin=2.0)

    metrics_f = f_agent.train_step(dict(batch))
    metrics_e = e_agent.train_step(dict(batch))  # same batch, including candidate keys E must ignore

    assert metrics_f["loss/risk_candidate_supervision"] > 0.0
    assert metrics_e["loss/risk_candidate_supervision"] == 0.0, (
        "E (counterfactual.enabled=False) must never engage the candidate supervision loss "
        "even when candidate keys are present in the batch"
    )


def test_no_ablation_profile_leaves_features_and_risk_counterfactual_sections_inconsistent():
    """Cross-check every A-F profile's features.* flags agree with the
    risk/counterfactual sections' own .enabled -- catches a profile YAML
    edit that flips one but forgets the other (Profile.validate() already
    enforces this at load time; this test just makes the coverage explicit
    per profile so a future profile addition to this matrix is caught)."""
    for name in (
        "baseline_tqc", "kinodynamic_tqc", "kinodynamic_tqc_temporal",
        "kinodynamic_tqc_risk_supervised_only", "kinodynamic_tqc_risk", "kinodynamic_tqc_counterfactual",
    ):
        profile = load_profile(name)
        assert profile.features.risk_critic == profile.risk.enabled
        assert profile.features.counterfactual_risk == profile.counterfactual.enabled
