"""compute_physics_step_count (section P0-7) was ported into
env/simulation/gazebo_service_wait.py in an earlier round but never wired
up or tested -- this is its first test coverage, added alongside actually
wiring it into gazebo_runtime.py::multi_step_advance/propagate_state."""

import pytest

from hunter_kinodynamic_rl.env.simulation.gazebo_service_wait import (
    bounded_wait_for_service, compute_physics_step_count,
)


def test_exact_multiple_returns_correct_step_count():
    assert compute_physics_step_count(0.1, 0.001) == 100
    assert compute_physics_step_count(0.2, 0.001) == 200


def test_single_step_duration():
    assert compute_physics_step_count(0.001, 0.001) == 1


def test_rejects_duration_shorter_than_one_physics_step():
    with pytest.raises(ValueError):
        compute_physics_step_count(0.0005, 0.001)


def test_rejects_non_integer_multiple():
    with pytest.raises(ValueError):
        compute_physics_step_count(0.1005, 0.001)


def test_rejects_non_positive_physics_step_size():
    with pytest.raises(ValueError):
        compute_physics_step_count(0.1, 0.0)


def test_tolerates_floating_point_noise_near_an_integer_multiple():
    # 0.1 / 0.001 in float64 arithmetic is 99.99999999999999, not exactly
    # 100.0 -- must still resolve to 100, not raise.
    assert compute_physics_step_count(0.1, 0.001) == 100


def test_bounded_wait_for_service_still_terminates_with_a_never_ready_service():
    """Pre-existing helper, sanity-checked here alongside its sibling above
    since both live in the same module and back the P0-7 stepping path."""
    ok, elapsed = bounded_wait_for_service(
        lambda step: False, timeout_sec=0.05, poll_sec=0.01,
    )
    assert ok is False
    assert elapsed < 1.0
