"""Parity test vs. drl_agent's TQC network definitions (section 48).

``rl/networks/tqc.py`` in this package is a VERBATIM copy of
``drl_agent/drl_agent/rl/networks/tqc.py`` (hash-checked in
docs/SOURCE_MAP.md). This test proves that claim operationally: given the
same seed and the same constructor arguments, this package's ``Actor``/
``Critic`` produce BIT-IDENTICAL weights and forward output to drl_agent's.
If anyone edits either copy without the other, this test starts failing --
that is the point (guards against silent drift between the two).

drl_agent is loaded by inserting its source path directly (NOT declared as a
runtime dependency of this package -- see package.xml / docs/SOURCE_MAP.md):
this is dev-time-only comparison tooling. Skips cleanly if the sibling
checkout is not where expected (e.g. this package copied out of the
monorepo) or if torch is unavailable.
"""

import importlib.util
import os
import sys

import pytest

torch = pytest.importorskip("torch")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DRL_AGENT_TQC_NET = os.path.normpath(
    os.path.join(_THIS_DIR, "..", "..", "drl_agent", "drl_agent", "rl", "networks", "tqc.py")
)

if not os.path.isfile(_DRL_AGENT_TQC_NET):
    pytest.skip(f"drl_agent reference not found at {_DRL_AGENT_TQC_NET} -- skipping parity check",
                allow_module_level=True)


def _load_drl_agent_tqc_networks():
    spec = importlib.util.spec_from_file_location("drl_agent_reference_tqc_networks", _DRL_AGENT_TQC_NET)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


from hunter_kinodynamic_rl.rl.networks.tqc import Actor as NewActor, Critic as NewCritic  # noqa: E402


def test_actor_weights_identical_given_same_seed():
    ref = _load_drl_agent_tqc_networks()

    torch.manual_seed(42)
    ref_actor = ref.Actor(state_dim=87, action_dim=3)
    torch.manual_seed(42)
    new_actor = NewActor(state_dim=87, action_dim=3)

    for (n1, p1), (n2, p2) in zip(ref_actor.named_parameters(), new_actor.named_parameters()):
        assert n1 == n2
        assert torch.equal(p1, p2), f"parameter {n1} diverged between drl_agent and hunter_kinodynamic_rl"


def test_actor_forward_output_identical():
    ref = _load_drl_agent_tqc_networks()

    torch.manual_seed(7)
    ref_actor = ref.Actor(state_dim=20, action_dim=3)
    torch.manual_seed(7)
    new_actor = NewActor(state_dim=20, action_dim=3)

    state = torch.randn(5, 20)
    ref_out = ref_actor.forward(state, deterministic=True)
    new_out = new_actor.forward(state, deterministic=True)
    assert torch.equal(ref_out, new_out)


def test_critic_forward_output_identical():
    ref = _load_drl_agent_tqc_networks()

    torch.manual_seed(3)
    ref_critic = ref.Critic(state_dim=20, action_dim=3, n_quantiles=25, n_critics=2)
    torch.manual_seed(3)
    new_critic = NewCritic(state_dim=20, action_dim=3, n_quantiles=25, n_critics=2)

    state = torch.randn(5, 20)
    action = torch.randn(5, 3)
    ref_q = ref_critic(state, action)
    new_q = new_critic(state, action)
    assert torch.equal(ref_q, new_q)


def test_quantile_huber_loss_identical():
    ref = _load_drl_agent_tqc_networks()
    current = torch.randn(8, 2, 25)
    target = torch.randn(8, 1, 46)
    ref_loss = ref.quantile_huber_loss(current, target)
    from hunter_kinodynamic_rl.rl.networks.tqc import quantile_huber_loss as new_loss_fn
    new_loss = new_loss_fn(current, target)
    assert torch.equal(ref_loss, new_loss)
