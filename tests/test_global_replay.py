"""Phase 4 Global RL: Global (option-level) replay buffer (plan section 8.8)."""

import os

import numpy as np
import pytest

from hunter_kinodynamic_rl.navigation.global_rl.replay import GlobalReplayBuffer, GlobalTransition
from hunter_kinodynamic_rl.navigation.global_rl.replay_schema import SCHEMA_VERSION
from hunter_kinodynamic_rl.navigation.hierarchy.subgoal_manager import SubgoalStatus

MAP_SHAPE = (5, 8, 8)
N_SCALARS = 5
N_CANDIDATES = 17
N_CAND_FEATURES = 4


def _transition(action=0, reward=1.0, mission_done=False, failure_reason=None, local_steps=5,
                 exploration_gain=3.0, risk_integral=0.2, subgoal_success=True):
    map_state = np.random.RandomState(0).rand(*MAP_SHAPE).astype(np.float32)
    scalar_state = np.random.RandomState(1).rand(N_SCALARS).astype(np.float32)
    candidate_features = np.random.RandomState(2).rand(N_CANDIDATES, N_CAND_FEATURES).astype(np.float32)
    action_mask = np.ones(N_CANDIDATES, dtype=bool)
    return GlobalTransition(
        map_state=map_state, scalar_state=scalar_state, candidate_features=candidate_features,
        action_mask=action_mask, action=action, option_reward=reward,
        next_map_state=map_state, next_scalar_state=scalar_state, next_candidate_features=candidate_features,
        next_action_mask=action_mask, mission_done=mission_done, subgoal_success=subgoal_success,
        failure_reason=failure_reason, local_steps=local_steps, exploration_gain=exploration_gain,
        risk_integral=risk_integral,
    )


def _buffer(capacity=16, seed=0):
    return GlobalReplayBuffer(MAP_SHAPE, N_SCALARS, N_CANDIDATES, N_CAND_FEATURES, capacity=capacity, seed=seed)


def test_add_and_sample_roundtrip_shapes():
    buf = _buffer()
    buf.add(_transition(), mode="train")
    assert len(buf) == 1
    batch = buf.sample(1)
    assert batch["map_state"].shape == (1, *MAP_SHAPE)
    assert batch["scalar_state"].shape == (1, N_SCALARS)
    assert batch["candidate_features"].shape == (1, N_CANDIDATES, N_CAND_FEATURES)
    assert batch["action_mask"].shape == (1, N_CANDIDATES)
    assert batch["action"].shape == (1, 1)


def test_map_quantization_is_near_lossless():
    buf = _buffer()
    transition = _transition()
    buf.add(transition, mode="train")
    batch = buf.sample(1)
    np.testing.assert_allclose(batch["map_state"][0], transition.map_state, atol=1.0 / 255.0)


def test_add_rejects_non_train_mode():
    buf = _buffer()
    with pytest.raises(ValueError):
        buf.add(_transition(), mode="validation")
    with pytest.raises(ValueError):
        buf.add(_transition(), mode="test")
    assert len(buf) == 0


def test_failure_reason_encoding_roundtrips():
    buf = _buffer()
    buf.add(_transition(failure_reason=SubgoalStatus.FAILED_BLOCKED), mode="train")
    buf.add(_transition(failure_reason=None), mode="train")
    assert buf.failure_reason_code[0, 0] != buf.failure_reason_code[1, 0]
    assert buf.failure_reason_code[1, 0] == -1


def test_save_load_preserves_schema_generation_and_contents(tmp_path):
    buf = _buffer(capacity=32, seed=42)
    for i in range(10):
        buf.add(_transition(action=i % N_CANDIDATES, reward=float(i)), mode="train")

    path = os.path.join(str(tmp_path), "test_global_replay_buffer")
    buf.save(path, generation="gen-abc")
    assert os.path.isfile(path + ".npz")

    loaded = GlobalReplayBuffer.load(path + ".npz", seed=0)
    assert loaded.generation == "gen-abc"
    assert len(loaded) == len(buf)
    np.testing.assert_array_equal(loaded.action[: len(buf)], buf.action[: len(buf)])
    np.testing.assert_array_equal(loaded.option_reward[: len(buf)], buf.option_reward[: len(buf)])


def test_save_load_restores_sampling_rng_state_not_just_a_reseed(tmp_path):
    """Resume must continue the EXACT sample_indices() draw sequence, not
    restart from a fresh reseed of the same base seed (mirrors
    tests/test_replay_buffer.py's own local-buffer regression test)."""
    buf = _buffer(capacity=64, seed=99)
    for i in range(40):
        buf.add(_transition(action=i % N_CANDIDATES, reward=float(i)), mode="train")
    buf.sample_indices(8)
    buf.sample_indices(8)
    path = os.path.join(str(tmp_path), "buf")
    buf.save(path)

    fresh_reseed = _buffer(capacity=64, seed=99)
    for i in range(40):
        fresh_reseed.add(_transition(action=i % N_CANDIDATES, reward=float(i)), mode="train")

    restored = GlobalReplayBuffer.load(path + ".npz", seed=99)
    next_from_original = buf.sample_indices(8)
    next_from_restored = restored.sample_indices(8)
    next_from_fresh_reseed = fresh_reseed.sample_indices(8)

    np.testing.assert_array_equal(next_from_restored, next_from_original)
    assert not np.array_equal(next_from_fresh_reseed, next_from_original), (
        "test setup invalid: a fresh reseed happened to match anyway"
    )


def test_load_rejects_unknown_schema_version(tmp_path):
    buf = _buffer()
    buf.add(_transition(), mode="train")
    path = str(tmp_path / "buf")
    buf.save(path)
    data = dict(np.load(path + ".npz"))
    data["schema_version"] = np.array(SCHEMA_VERSION + 1)
    np.savez_compressed(path + "_bad.npz", **data)
    with pytest.raises(ValueError):
        GlobalReplayBuffer.load(path + "_bad.npz")


def test_save_records_current_channel_metadata():
    from hunter_kinodynamic_rl.navigation.global_rl.observation import (
        CANDIDATE_FEATURE_NAMES, MAP_CHANNEL_NAMES, SCALAR_NAMES,
    )
    buf = _buffer()
    buf.add(_transition(), mode="train")
    path = os.path.join("/tmp", "test_global_replay_metadata")
    buf.save(path)
    try:
        data = np.load(path + ".npz")
        assert tuple(str(x) for x in data["map_channel_names"]) == MAP_CHANNEL_NAMES
        assert tuple(str(x) for x in data["scalar_names"]) == SCALAR_NAMES
        assert tuple(str(x) for x in data["candidate_feature_names"]) == CANDIDATE_FEATURE_NAMES
    finally:
        os.remove(path + ".npz")


def test_load_rejects_a_renamed_or_reordered_map_channel(tmp_path):
    """Regression guard: a file whose SHAPE still matches but whose channel
    SEMANTICS have drifted (a channel renamed/reordered/inserted since the
    file was saved) must be rejected, not silently sampled as if nothing
    changed."""
    buf = _buffer()
    buf.add(_transition(), mode="train")
    path = str(tmp_path / "buf")
    buf.save(path)
    data = dict(np.load(path + ".npz"))
    data["map_channel_names"] = np.array(["occupied", "free", "unknown", "visited", "TOTALLY_DIFFERENT"])
    np.savez_compressed(path + "_renamed.npz", **data)
    with pytest.raises(ValueError):
        GlobalReplayBuffer.load(path + "_renamed.npz")


def test_capacity_wraps_and_size_saturates():
    buf = _buffer(capacity=3)
    for i in range(5):
        buf.add(_transition(action=i), mode="train")
    assert len(buf) == 3
    assert buf.ptr == 2
