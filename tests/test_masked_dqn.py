"""Phase 4 Global RL: masked Dueling Double DQN network + agent (plan
section 8.6). Torch-gated -- skips cleanly when torch is not installed."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hunter_kinodynamic_rl.config.schema import GlobalRLConfig  # noqa: E402
from hunter_kinodynamic_rl.navigation.global_rl.agent import GlobalDQNAgent  # noqa: E402
from hunter_kinodynamic_rl.navigation.global_rl.networks import (  # noqa: E402
    MaskedDuelingDQN, masked_argmax, masked_mean,
)
from hunter_kinodynamic_rl.navigation.global_rl.observation import (  # noqa: E402
    GlobalObservation, N_CANDIDATE_FEATURES, N_MAP_CHANNELS, N_SCALARS,
)


def _cfg(**overrides):
    cfg = GlobalRLConfig(**overrides)
    cfg.validate()
    return cfg


def _random_observation(cfg: GlobalRLConfig, mask=None, rng=None) -> GlobalObservation:
    rng = rng or np.random.RandomState(0)
    n = cfg.n_candidates
    if mask is None:
        mask = np.ones(n, dtype=bool)
    return GlobalObservation(
        map_tensor=rng.randn(N_MAP_CHANNELS, cfg.map_crop_size_cells, cfg.map_crop_size_cells).astype(np.float32),
        scalar_tensor=rng.randn(N_SCALARS).astype(np.float32),
        candidate_tensor=rng.randn(n, N_CANDIDATE_FEATURES).astype(np.float32),
        action_mask=mask,
    )


def test_forward_output_shape_and_masked_entries_are_negative_infinity():
    cfg = _cfg()
    net = MaskedDuelingDQN(cfg, N_MAP_CHANNELS)
    b, n = 4, cfg.n_candidates
    map_t = torch.randn(b, N_MAP_CHANNELS, cfg.map_crop_size_cells, cfg.map_crop_size_cells)
    scalar_t = torch.randn(b, N_SCALARS)
    cand_t = torch.randn(b, n, N_CANDIDATE_FEATURES)
    mask = torch.ones(b, n, dtype=torch.bool)
    mask[:, 0] = False
    mask[:, 3] = False
    q = net(map_t, scalar_t, cand_t, mask)
    assert q.shape == (b, n)
    assert torch.isneginf(q[:, 0]).all()
    assert torch.isneginf(q[:, 3]).all()
    assert torch.isfinite(q[mask]).all()


def test_masked_argmax_never_returns_an_invalid_index():
    q = torch.tensor([[5.0, 10.0, 1.0], [3.0, 2.0, 9.0]])
    mask = torch.tensor([[True, False, True], [False, False, True]])
    idx = masked_argmax(q, mask)
    assert idx.tolist() == [0, 2]


def test_masked_mean_ignores_invalid_entries():
    values = torch.tensor([[1.0, 100.0, 3.0]])
    mask = torch.tensor([[True, False, True]])
    mean = masked_mean(values, mask)
    assert torch.isclose(mean, torch.tensor([[2.0]]))


def test_all_but_fallback_invalid_still_selects_a_valid_index():
    cfg = _cfg()
    net = MaskedDuelingDQN(cfg, N_MAP_CHANNELS)
    n = cfg.n_candidates
    map_t = torch.randn(1, N_MAP_CHANNELS, cfg.map_crop_size_cells, cfg.map_crop_size_cells)
    scalar_t = torch.randn(1, N_SCALARS)
    cand_t = torch.randn(1, n, N_CANDIDATE_FEATURES)
    mask = torch.zeros(1, n, dtype=torch.bool)
    mask[0, cfg.fallback_index] = True
    q = net(map_t, scalar_t, cand_t, mask)
    assert torch.argmax(q, dim=-1).item() == cfg.fallback_index


def test_agent_select_action_greedy_respects_mask():
    cfg = _cfg(epsilon_decay_steps=100)
    agent = GlobalDQNAgent(cfg, N_MAP_CHANNELS, device="cpu")
    mask = np.ones(cfg.n_candidates, dtype=bool)
    mask[0] = False
    mask[1] = False
    obs = _random_observation(cfg, mask=mask)
    rng = np.random.RandomState(0)
    for _ in range(20):
        action = agent.select_action(obs, epsilon=0.0, rng=rng)
        assert mask[action]


def test_agent_select_action_random_respects_mask():
    cfg = _cfg()
    agent = GlobalDQNAgent(cfg, N_MAP_CHANNELS, device="cpu")
    mask = np.zeros(cfg.n_candidates, dtype=bool)
    mask[2] = True
    mask[cfg.fallback_index] = True
    obs = _random_observation(cfg, mask=mask)
    rng = np.random.RandomState(1)
    seen = set()
    for _ in range(50):
        action = agent.select_action(obs, epsilon=1.0, rng=rng)
        assert mask[action]
        seen.add(action)
    assert seen <= {2, cfg.fallback_index}


def test_agent_select_action_raises_if_no_valid_candidate_at_all():
    cfg = _cfg()
    agent = GlobalDQNAgent(cfg, N_MAP_CHANNELS, device="cpu")
    mask = np.zeros(cfg.n_candidates, dtype=bool)  # pathological: not even the fallback is valid
    obs = _random_observation(cfg, mask=mask)
    rng = np.random.RandomState(0)
    with pytest.raises(RuntimeError):
        agent.select_action(obs, epsilon=0.0, rng=rng)


def _random_batch(cfg: GlobalRLConfig, batch_size=8, local_steps=None, mission_done=None):
    n = cfg.n_candidates
    mask = torch.ones(batch_size, n, dtype=torch.bool)
    return {
        "map_state": torch.randn(batch_size, N_MAP_CHANNELS, cfg.map_crop_size_cells, cfg.map_crop_size_cells),
        "scalar_state": torch.randn(batch_size, N_SCALARS),
        "candidate_features": torch.randn(batch_size, n, N_CANDIDATE_FEATURES),
        "action_mask": mask,
        "next_map_state": torch.randn(batch_size, N_MAP_CHANNELS, cfg.map_crop_size_cells, cfg.map_crop_size_cells),
        "next_scalar_state": torch.randn(batch_size, N_SCALARS),
        "next_candidate_features": torch.randn(batch_size, n, N_CANDIDATE_FEATURES),
        "next_action_mask": mask,
        "action": torch.randint(0, n, (batch_size, 1), dtype=torch.int64),
        "option_reward": torch.ones(batch_size, 1),
        "mission_done": torch.zeros(batch_size, 1) if mission_done is None else mission_done,
        "local_steps": torch.ones(batch_size, 1) if local_steps is None else local_steps,
    }


def test_train_step_runs_and_updates_target_network_on_schedule():
    cfg = _cfg(target_update_interval_steps=2)
    agent = GlobalDQNAgent(cfg, N_MAP_CHANNELS, device="cpu")
    before = {k: v.clone() for k, v in agent.target.state_dict().items()}
    batch = _random_batch(cfg)
    agent.train_step(batch)
    after_one = {k: v.clone() for k, v in agent.target.state_dict().items()}
    assert all(torch.equal(before[k], after_one[k]) for k in before)  # not yet due
    agent.train_step(batch)
    after_two = agent.target.state_dict()
    assert any(not torch.equal(before[k], after_two[k]) for k in before)  # updated on the 2nd step
