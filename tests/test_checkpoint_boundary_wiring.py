"""Regression test for P0-4's run()-level wiring: TrainerBase.run() must
call _save_checkpoint() ONLY at true episode boundaries (right after
done=True), never mid-episode -- see
training/checkpoint_policy.py::checkpoint_due's docstring for the full
determinism rationale.

trainer_base.py imports rclpy at module scope, so this self-skips cleanly
on a bare host checkout (mirrors test_environment_node.py's pattern). No
live ROS graph or Gazebo is needed, though -- TrainerBase is constructed
via __new__ (bypassing __init__, which would need a real EnvironmentClient
service connection) and wired to fully fake env/agent/logger objects that
satisfy the same duck-typed interface run() actually calls.
"""

import dataclasses

import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("drl_agent_interfaces")  # training.trainer_base imports it at module scope

import numpy as np  # noqa: E402

from hunter_kinodynamic_rl.config.loader import load_profile  # noqa: E402
from hunter_kinodynamic_rl.env.scenarios.seed_scheduler import SeedScheduler  # noqa: E402
from hunter_kinodynamic_rl.env.simulation import risk_telemetry as rt  # noqa: E402
from hunter_kinodynamic_rl.rl.replay.buffer import ReplayBuffer  # noqa: E402
from hunter_kinodynamic_rl.training.trainer_base import TrainerBase  # noqa: E402

_EPISODE_LEN = 5


class _FakeEnv:
    """Every episode is EXACTLY _EPISODE_LEN steps -- done fires on steps
    5, 10, 15, 20, ... making it trivial to check every recorded checkpoint
    step against "was this actually an episode-ending step"."""

    latest_sim_time_sec = None  # section P1-11: run()'s log_step call site reads this
    latest_pose = None

    def __init__(self, state_dim: int):
        self._state_dim = state_dim
        self._episode_step = 0

    def seed(self, seed):
        pass

    def reset(self):
        self._episode_step = 0
        return np.zeros(self._state_dim, dtype=np.float32)

    def step(self, action):
        self._episode_step += 1
        done = self._episode_step >= _EPISODE_LEN
        next_state = np.zeros(self._state_dim, dtype=np.float32)
        telemetry = rt.invalid(step_id=self._episode_step)
        return next_state, 0.0, done, False, False, 10.0, telemetry


class _FakeAgent:
    device = "cpu"

    def select_action(self, state, deterministic=False):
        return np.zeros(3, dtype=np.float32)


class _FakeLogger:
    def log_episode_start(self, *a, **kw):
        pass

    def log_step(self, *a, **kw):
        pass

    def log_episode_end(self, *a, **kw):
        pass

    def log_validation(self, *a, **kw):
        pass

    def log_train_metrics(self, *a, **kw):
        pass


def _make_trainer(eval_freq: int, max_timesteps: int) -> TrainerBase:
    profile = load_profile("smoke_test")
    profile = dataclasses.replace(
        profile,
        training=dataclasses.replace(
            profile.training, eval_freq=eval_freq, max_timesteps=max_timesteps,
            timesteps_before_training=0,
        ),
        counterfactual=dataclasses.replace(profile.counterfactual, enabled=False),
        risk=dataclasses.replace(profile.risk, enabled=False),
    )

    trainer = TrainerBase.__new__(TrainerBase)
    trainer.profile = profile
    trainer.run_dir = "/tmp/hkrl_test_checkpoint_boundary_wiring"
    trainer.state_dim = 4
    trainer.action_dim = 3
    trainer.max_action = 1.0
    trainer.max_candidates = 0
    trainer.env = _FakeEnv(trainer.state_dim)
    trainer.replay_buffer = ReplayBuffer(
        trainer.state_dim, trainer.action_dim, capacity=1000, seed=0, max_candidates=1,
    )
    trainer.agent = _FakeAgent()
    trainer.agent_train_step = lambda batch: {}
    trainer.seed_scheduler = SeedScheduler(profile.training.seed, profile.scenario, mode="train")
    trainer.validation_seed_scheduler = SeedScheduler(profile.training.seed, profile.scenario, mode="validation")
    trainer.global_step = 0
    trainer.episode_index = 0
    trainer.episode_reward = 0.0
    trainer.episode_len = 0
    trainer.best_eval_metric = -float("inf")
    trainer._last_checkpoint_step = 0
    trainer.logger = _FakeLogger()
    # section P1-13: this test suite is about CHECKPOINT-BOUNDARY WIRING,
    # not validation itself (see test_trainer_validation.py for that) --
    # stubbed out so it never calls self.env.step() and disrupts this
    # FakeEnv's own episode-length bookkeeping, which has no notion of
    # "validation episode" vs "training episode" (both just advance the
    # same internal step counter).
    trainer._run_validation_episodes = lambda: {
        "success_rate": 0.0, "collision_rate": 0.0, "mean_reward": 0.0, "num_episodes": 0,
    }
    return trainer


def test_periodic_checkpoints_only_fire_on_true_episode_boundary_steps():
    # eval_freq=3 is DELIBERATELY not a multiple of _EPISODE_LEN=5 --
    # without the P0-4 fix (a bare `step % eval_freq == 0` gate), this
    # would trip mid-episode almost every time (steps 3, 6, 9, 12, ... vs
    # episode boundaries at 5, 10, 15, 20).
    trainer = _make_trainer(eval_freq=3, max_timesteps=23)
    recorded_steps = []
    trainer._save_checkpoint = lambda tag="periodic": recorded_steps.append((trainer.global_step, tag))

    trainer.run()

    periodic_steps = [s for s, tag in recorded_steps if tag == "periodic"]
    assert len(periodic_steps) > 0, "test setup sanity: eval_freq should have triggered at least one save"
    for s in periodic_steps:
        assert s % _EPISODE_LEN == 0, (
            f"checkpoint saved at step {s}, which is NOT an episode boundary "
            f"(episodes are {_EPISODE_LEN} steps long) -- this is exactly the P0-4 regression"
        )

    # The unconditional end-of-run "final" save is a separate, documented
    # exception (see run()'s comment) -- confirm it's the only non-periodic
    # entry, so this test isn't accidentally masking a periodic bug.
    final_entries = [(s, tag) for s, tag in recorded_steps if tag == "final"]
    assert len(final_entries) == 1
    assert final_entries[0][0] == 23
