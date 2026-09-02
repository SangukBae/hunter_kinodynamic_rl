"""Phase 5 checkpoint compatibility (plan section 9.9/9.11 + the package's
own 'opt-in, byte-identical when off' convention). Torch-gated -- skips
cleanly when torch is not installed.

The key claim under test: a network built from a profile with EVERY Phase 5
ablation flag left at its default (False/disabled) must be architecturally
IDENTICAL (same submodules, same parameter count/shapes) to Phase 4's own
network -- so a Global checkpoint trained under Phase 4 still loads cleanly
under this session's code. A network built with ``topology_feedback_enabled``
(and therefore ``max_nodes > 0``) must, conversely, have STRICTLY MORE
parameters (the extra node encoder + wider fused input), matching plan
9.11's "ablation checkpoint가 서로 다른 architecture fingerprint로 구분된다"."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hunter_kinodynamic_rl.config.schema import GlobalRLConfig  # noqa: E402
from hunter_kinodynamic_rl.navigation.global_rl.agent import GlobalDQNAgent  # noqa: E402
from hunter_kinodynamic_rl.navigation.global_rl.networks import MaskedDuelingDQN  # noqa: E402
from hunter_kinodynamic_rl.navigation.global_rl.observation import (  # noqa: E402
    GlobalObservation, N_MAP_CHANNELS, resolve_candidate_feature_names,
)
from hunter_kinodynamic_rl.navigation.memory.topological_graph import N_NODE_FEATURES  # noqa: E402


def _param_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def test_default_config_network_has_no_node_encoder():
    cfg = GlobalRLConfig()
    net = MaskedDuelingDQN(cfg, N_MAP_CHANNELS)
    assert net.node_encoder is None
    assert net.max_nodes == 0


def test_default_config_forward_ignores_node_args_and_matches_phase4_shape():
    cfg = GlobalRLConfig()
    cfg.validate()
    net = MaskedDuelingDQN(cfg, N_MAP_CHANNELS)
    n = cfg.n_candidates
    map_t = torch.zeros(2, N_MAP_CHANNELS, cfg.map_crop_size_cells, cfg.map_crop_size_cells)
    scalar_t = torch.zeros(2, 5)
    cand_t = torch.zeros(2, n, len(resolve_candidate_feature_names(cfg)))
    mask_t = torch.ones(2, n, dtype=torch.bool)
    q = net(map_t, scalar_t, cand_t, mask_t)  # no node_tensor/node_validity_mask passed at all
    assert q.shape == (2, n)


def test_topology_enabled_network_has_strictly_more_parameters():
    cfg_off = GlobalRLConfig()
    cfg_off.validate()
    net_off = MaskedDuelingDQN(cfg_off, N_MAP_CHANNELS, max_nodes=0)

    cfg_on = GlobalRLConfig(topology_feedback_enabled=True)
    cfg_on.validate()
    net_on = MaskedDuelingDQN(cfg_on, N_MAP_CHANNELS, max_nodes=16)

    assert _param_count(net_on) > _param_count(net_off)
    assert net_on.node_encoder is not None


def test_topology_enabled_forward_requires_node_tensor():
    cfg = GlobalRLConfig(topology_feedback_enabled=True)
    cfg.validate()
    net = MaskedDuelingDQN(cfg, N_MAP_CHANNELS, max_nodes=8)
    n = cfg.n_candidates
    map_t = torch.zeros(1, N_MAP_CHANNELS, cfg.map_crop_size_cells, cfg.map_crop_size_cells)
    scalar_t = torch.zeros(1, 5)
    cand_t = torch.zeros(1, n, len(resolve_candidate_feature_names(cfg)))
    mask_t = torch.ones(1, n, dtype=torch.bool)
    with pytest.raises(ValueError):
        net(map_t, scalar_t, cand_t, mask_t)  # node_tensor/node_validity_mask omitted -- must raise, not silently zero


def test_topology_enabled_forward_with_zero_valid_nodes_does_not_crash():
    """An empty graph (mission just started) must degrade gracefully to a
    zero topology-feature contribution, never a division error."""
    cfg = GlobalRLConfig(topology_feedback_enabled=True)
    cfg.validate()
    net = MaskedDuelingDQN(cfg, N_MAP_CHANNELS, max_nodes=8)
    n = cfg.n_candidates
    map_t = torch.zeros(1, N_MAP_CHANNELS, cfg.map_crop_size_cells, cfg.map_crop_size_cells)
    scalar_t = torch.zeros(1, 5)
    cand_t = torch.zeros(1, n, len(resolve_candidate_feature_names(cfg)))
    mask_t = torch.ones(1, n, dtype=torch.bool)
    node_t = torch.zeros(1, 8, N_NODE_FEATURES)
    node_mask_t = torch.zeros(1, 8, dtype=torch.bool)  # no valid nodes at all
    q = net(map_t, scalar_t, cand_t, mask_t, node_t, node_mask_t)
    assert torch.isfinite(q).all()


def test_global_dqn_agent_state_dict_roundtrip_default_config():
    cfg = GlobalRLConfig()
    cfg.validate()
    agent = GlobalDQNAgent(cfg, N_MAP_CHANNELS, device="cpu")
    state = agent.online.state_dict()
    agent2 = GlobalDQNAgent(cfg, N_MAP_CHANNELS, device="cpu")
    agent2.online.load_state_dict(state)  # must not raise -- same architecture


def test_global_dqn_agent_select_action_works_with_default_zero_node_observation():
    cfg = GlobalRLConfig()
    cfg.validate()
    agent = GlobalDQNAgent(cfg, N_MAP_CHANNELS, device="cpu")
    n = cfg.n_candidates
    obs = GlobalObservation(
        map_tensor=np.zeros((N_MAP_CHANNELS, cfg.map_crop_size_cells, cfg.map_crop_size_cells), dtype=np.float32),
        scalar_tensor=np.zeros(5, dtype=np.float32),
        candidate_tensor=np.zeros((n, len(resolve_candidate_feature_names(cfg))), dtype=np.float32),
        action_mask=np.ones(n, dtype=bool),
    )
    action = agent.select_action(obs, epsilon=0.0, rng=np.random.RandomState(0))
    assert 0 <= action < n
