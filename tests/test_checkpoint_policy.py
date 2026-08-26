"""Regression tests for the P0-4 checkpoint-cadence gate
(training/checkpoint_policy.py::checkpoint_due) -- extracted as a pure,
ROS-free function specifically so this invariant (checkpoints only ever
fire at episode boundaries, on a bounded cadence) is testable without a
live ROS/Gazebo stack. See that module's docstring for the full
determinism rationale."""

import pytest

from hunter_kinodynamic_rl.training.checkpoint_policy import checkpoint_due


def test_not_due_before_eval_freq_elapsed():
    assert checkpoint_due(step=50, last_checkpoint_step=0, eval_freq=100) is False


def test_due_once_eval_freq_elapsed():
    assert checkpoint_due(step=100, last_checkpoint_step=0, eval_freq=100) is True


def test_due_when_elapsed_overshoots_eval_freq():
    """An episode boundary can land past the exact eval_freq multiple
    (episodes rarely align to it) -- still due, not skipped entirely."""
    assert checkpoint_due(step=137, last_checkpoint_step=0, eval_freq=100) is True


def test_not_due_again_immediately_after_a_save():
    assert checkpoint_due(step=101, last_checkpoint_step=100, eval_freq=100) is False


@pytest.mark.parametrize("eval_freq", [1, 5, 12000, 50000])
def test_cadence_scales_with_eval_freq(eval_freq):
    assert checkpoint_due(step=eval_freq - 1, last_checkpoint_step=0, eval_freq=eval_freq) is False
    assert checkpoint_due(step=eval_freq, last_checkpoint_step=0, eval_freq=eval_freq) is True
