"""``rl.checkpointing.rng_state`` (item 4): full RNG-state bundle capture/
restore/file round-trip for deterministic Global-training resume. Pure
Python + numpy (+torch, gated) -- no ROS/Gazebo needed."""

from __future__ import annotations

import random

import numpy as np
import pytest

from hunter_kinodynamic_rl.rl.checkpointing.rng_state import (
    capture_rng_state, load_rng_state_file, restore_rng_state, save_rng_state_file,
)

pytest.importorskip("torch")


def test_capture_restore_round_trip_reproduces_python_random_stream():
    random.seed(123)
    for _ in range(5):
        random.random()  # advance the stream away from its freshly-seeded state
    state = capture_rng_state()
    expected = [random.random() for _ in range(10)]

    random.seed(999)  # perturb
    restore_rng_state(state)
    actual = [random.random() for _ in range(10)]
    assert actual == expected


def test_capture_restore_round_trip_reproduces_numpy_global_stream():
    np.random.seed(7)
    np.random.rand(5)
    state = capture_rng_state()
    expected = np.random.rand(10).tolist()

    np.random.seed(0)
    restore_rng_state(state)
    actual = np.random.rand(10).tolist()
    assert actual == expected


def test_capture_restore_round_trip_reproduces_dedicated_action_rng():
    action_rng = np.random.RandomState(42)
    action_rng.rand(3)
    state = capture_rng_state(action_rng=action_rng)
    expected = action_rng.rand(10).tolist()

    action_rng.seed(0)  # perturb the SAME object
    restore_rng_state(state, action_rng=action_rng)
    actual = action_rng.rand(10).tolist()
    assert actual == expected


def test_restoring_action_rng_does_not_perturb_numpy_global_stream():
    """action_rng is a DEDICATED stream -- restoring it must have zero
    effect on numpy's own global RNG (a caller relying on the global
    stream elsewhere must never see it silently reseeded by this call)."""
    np.random.seed(11)
    global_before = np.random.rand(3).tolist()

    action_rng = np.random.RandomState(1)
    state = capture_rng_state(action_rng=action_rng)
    restore_rng_state(state, action_rng=action_rng)

    global_after = np.random.rand(3).tolist()
    # global stream continued naturally from where it was, unaffected by
    # the action_rng capture/restore calls in between.
    np.random.seed(11)
    np.random.rand(3)
    expected_continuation = np.random.rand(3).tolist()
    assert global_after == expected_continuation
    assert global_before != global_after  # sanity: these ARE different draws


def test_capture_restore_round_trip_reproduces_torch_cpu_stream():
    import torch

    torch.manual_seed(5)
    torch.rand(3)
    state = capture_rng_state()
    expected = torch.rand(4).tolist()

    torch.manual_seed(0)
    restore_rng_state(state)
    actual = torch.rand(4).tolist()
    assert actual == expected


def test_save_and_load_rng_state_file_round_trips(tmp_path):
    from hunter_kinodynamic_rl.rl.checkpointing.manager import sha256_of_file

    random.seed(3)
    np.random.seed(3)
    action_rng = np.random.RandomState(3)
    state = capture_rng_state(action_rng=action_rng)
    expected = action_rng.rand(5).tolist()  # advance past the captured point
    path = str(tmp_path / "rng_state.pt")
    returned_sha256 = save_rng_state_file(path, state)
    assert returned_sha256 == sha256_of_file(path)

    loaded = load_rng_state_file(path)
    fresh_action_rng = np.random.RandomState(999)
    restore_rng_state(loaded, action_rng=fresh_action_rng)
    assert fresh_action_rng.rand(5).tolist() == expected


def test_full_bundle_round_trip_reproduces_every_stream_simultaneously():
    """All streams captured/restored together, exactly how
    nodes/hierarchical_train_node.py actually uses this module -- a
    regression guard against one stream's restore accidentally disturbing
    another's."""
    import torch

    random.seed(1)
    np.random.seed(2)
    torch.manual_seed(3)
    action_rng = np.random.RandomState(4)

    state = capture_rng_state(action_rng=action_rng)
    expected_random = [random.random() for _ in range(3)]
    expected_numpy = np.random.rand(3).tolist()
    expected_action = action_rng.rand(3).tolist()
    expected_torch = torch.rand(3).tolist()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    action_rng.seed(0)
    restore_rng_state(state, action_rng=action_rng)

    assert [random.random() for _ in range(3)] == expected_random
    assert np.random.rand(3).tolist() == expected_numpy
    assert action_rng.rand(3).tolist() == expected_action
    assert torch.rand(3).tolist() == expected_torch
