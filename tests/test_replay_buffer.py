"""Pure-numpy replay buffer tests -- no torch required."""

import math
import os

import numpy as np
import pytest

from hunter_kinodynamic_rl.rl.replay.buffer import ReplayBuffer, RiskTransition
from hunter_kinodynamic_rl.rl.replay.schema import SCHEMA_VERSION


def _fill(buf: ReplayBuffer, n: int, state_dim: int, action_dim: int, with_risk=True):
    rng = np.random.RandomState(0)
    for i in range(n):
        s = rng.randn(state_dim).astype(np.float32)
        a = rng.randn(action_dim).astype(np.float32)
        ns = rng.randn(state_dim).astype(np.float32)
        r = float(rng.randn())
        done = bool(i % 7 == 0)
        risk = RiskTransition(risk_target=float(rng.uniform(0, 1))) if with_risk else None
        buf.add(s, a, ns, r, done, risk=risk)


def test_add_and_sample_shapes():
    buf = ReplayBuffer(state_dim=5, action_dim=3, capacity=100, seed=1, max_candidates=4)
    _fill(buf, 50, 5, 3)
    assert len(buf) == 50
    batch = buf.sample(16)
    assert batch["state"].shape == (16, 5)
    assert batch["action"].shape == (16, 3)
    assert batch["next_state"].shape == (16, 5)
    assert batch["reward"].shape == (16, 1)
    assert batch["not_done"].shape == (16, 1)
    assert batch["risk_target"].shape == (16, 1)
    assert batch["candidate_kappa"].shape == (16, 4)


def test_sample_from_empty_buffer_raises():
    buf = ReplayBuffer(state_dim=4, action_dim=2, capacity=10)
    with pytest.raises(RuntimeError):
        buf.sample(4)


def test_capacity_wraps_around():
    buf = ReplayBuffer(state_dim=2, action_dim=1, capacity=10, seed=0)
    _fill(buf, 25, 2, 1)
    assert len(buf) == 10
    assert buf.ptr == 5  # 25 % 10


def test_no_risk_transition_is_explicitly_invalid_not_just_nan():
    buf = ReplayBuffer(state_dim=2, action_dim=1, capacity=10)
    buf.add([0, 0], [0], [0, 0], 1.0, False, risk=None)
    batch = buf.sample(1)
    assert math.isnan(batch["risk_target"][0, 0])
    assert batch["valid"][0, 0] == 0.0  # explicit mask, not inferred from NaN


def test_valid_ratio_reflects_stored_labels():
    buf = ReplayBuffer(state_dim=2, action_dim=1, capacity=10)
    buf.add([0, 0], [0], [0, 0], 1.0, False, risk=RiskTransition(risk_target=0.5))
    buf.add([0, 0], [0], [0, 0], 1.0, False, risk=None)
    assert buf.valid_ratio() == pytest.approx(0.5)


def test_not_done_is_inverse_of_done():
    buf = ReplayBuffer(state_dim=2, action_dim=1, capacity=10)
    buf.add([0, 0], [0], [0, 0], 1.0, True, risk=RiskTransition(risk_target=0.5))
    batch = buf.sample(1)
    assert batch["not_done"][0, 0] == 0.0


def test_candidate_fields_stored_and_valid_mask_matches_count():
    buf = ReplayBuffer(state_dim=2, action_dim=1, capacity=10, max_candidates=5)
    risk = RiskTransition(
        risk_target=0.2, candidate_kappa=[0.1, -0.1, 0.0], candidate_v_ref=[1.0, 1.0, 0.5],
        candidate_horizon=[1.0, 1.0, 1.0], candidate_risk=[0.2, 0.1, 0.05],
        candidate_goal_progress=[0.8, 0.75, 0.2], steering_saturation=True,
        goal_progress_m=0.8,
    )
    buf.add([0, 0], [0], [0, 0], 1.0, False, risk=risk)
    batch = buf.sample(1)
    assert batch["candidate_valid_mask"][0].sum() == 3
    np.testing.assert_allclose(batch["candidate_kappa"][0, :3], [0.1, -0.1, 0.0])
    np.testing.assert_allclose(batch["candidate_valid_mask"][0, 3:], [0.0, 0.0])
    np.testing.assert_allclose(batch["candidate_goal_progress"][0, :3], [0.8, 0.75, 0.2])
    assert batch["steering_saturation"][0, 0] == 1.0
    assert batch["goal_progress_m"][0, 0] == pytest.approx(0.8)


def test_actor_candidate_index_stored_and_round_trips():
    """section P0-6: the actor's own action's index within the stored
    candidate arrays must be recorded per-transition, not just assumed to
    always be 0 by trajectory_sampler's ordering convention."""
    buf = ReplayBuffer(state_dim=2, action_dim=1, capacity=10, max_candidates=4)
    buf.add([0, 0], [0], [0, 0], 1.0, False,
            risk=RiskTransition(risk_target=0.3, actor_candidate_index=0))
    batch = buf.sample(1)
    assert batch["actor_candidate_index"][0, 0] == 0.0


def test_save_load_roundtrip(tmp_path):
    buf = ReplayBuffer(state_dim=4, action_dim=2, capacity=64, seed=2, max_candidates=3)
    _fill(buf, 30, 4, 2)
    path = os.path.join(str(tmp_path), "buf")
    buf.save(path)
    restored = ReplayBuffer.load(path + ".npz")
    assert len(restored) == len(buf)
    np.testing.assert_array_equal(restored.state[: len(buf)], buf.state[: len(buf)])
    np.testing.assert_array_equal(restored.action[: len(buf)], buf.action[: len(buf)])
    np.testing.assert_array_equal(restored.valid[: len(buf)], buf.valid[: len(buf)])
    np.testing.assert_array_equal(
        restored.actor_candidate_index[: len(buf)], buf.actor_candidate_index[: len(buf)]
    )


def test_load_rejects_mismatched_schema_version(tmp_path):
    buf = ReplayBuffer(state_dim=3, action_dim=1, capacity=8)
    _fill(buf, 5, 3, 1)
    path = os.path.join(str(tmp_path), "buf")
    buf.save(path)
    npz_path = path + ".npz"
    data = dict(np.load(npz_path))
    data["schema_version"] = SCHEMA_VERSION + 999
    np.savez_compressed(npz_path, **data)
    with pytest.raises(ValueError):
        ReplayBuffer.load(npz_path)


def test_load_explicitly_migrates_v3_missing_new_risk_fields(tmp_path):
    buf = ReplayBuffer(state_dim=3, action_dim=1, capacity=8, max_candidates=2)
    _fill(buf, 5, 3, 1)
    path = os.path.join(str(tmp_path), "buf")
    buf.save(path)
    npz_path = path + ".npz"
    data = dict(np.load(npz_path))
    data["schema_version"] = 3
    for key in ("steering_saturation", "goal_progress_m", "candidate_goal_progress"):
        data.pop(key)
    np.savez_compressed(npz_path, **data)

    restored = ReplayBuffer.load(npz_path)
    assert np.all(restored.steering_saturation[: len(restored)] == 0.0)
    assert np.all(np.isnan(restored.goal_progress_m[: len(restored)]))
    assert np.all(restored.candidate_goal_progress[: len(restored)] == 0.0)


def test_save_load_restores_sampling_rng_state_not_just_a_reseed(tmp_path):
    """P1-1: resume must continue the EXACT sample_indices() draw sequence,
    not restart from a fresh reseed of the same base seed (which would
    silently replay the same samples that were already drawn before the
    checkpoint, or diverge from what training would have done had it not
    stopped)."""
    buf = ReplayBuffer(state_dim=4, action_dim=2, capacity=64, seed=99)
    _fill(buf, 40, 4, 2)
    buf.sample_indices(8)  # advance the RNG past its initial seed state
    buf.sample_indices(8)
    path = os.path.join(str(tmp_path), "buf")
    buf.save(path)

    # A FRESH buffer re-seeded with the SAME base seed would draw a
    # DIFFERENT next sample than the original (which has already advanced
    # twice) -- this is the failure mode being guarded against.
    fresh_reseed = ReplayBuffer(state_dim=4, action_dim=2, capacity=64, seed=99)
    _fill(fresh_reseed, 40, 4, 2)

    restored = ReplayBuffer.load(path + ".npz", seed=99)
    next_from_original = buf.sample_indices(8)
    next_from_restored = restored.sample_indices(8)
    next_from_fresh_reseed = fresh_reseed.sample_indices(8)

    np.testing.assert_array_equal(next_from_restored, next_from_original)
    assert not np.array_equal(next_from_fresh_reseed, next_from_original), (
        "test setup invalid: a fresh reseed happened to match anyway"
    )


def test_constructor_rejects_invalid_dims():
    with pytest.raises(ValueError):
        ReplayBuffer(state_dim=0, action_dim=2)
    with pytest.raises(ValueError):
        ReplayBuffer(state_dim=2, action_dim=2, capacity=0)
