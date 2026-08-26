"""Regression tests for P1-13's periodic held-out validation + best-
checkpoint selection: TrainerBase._run_validation_episodes() draws from the
VALIDATION seed pool (never train/test), never stores transitions in the
replay buffer, and run()'s checkpoint_due block writes a "best" checkpoint
exactly when a validation pass improves on the documented criterion
(success_rate - collision_rate).

trainer_base.py imports rclpy at module scope -- self-skips cleanly on a
bare host checkout, mirroring this package's other TrainerBase test files.
"""

import dataclasses
import os

import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("drl_agent_interfaces")  # training.trainer_base imports it at module scope

import numpy as np  # noqa: E402

from hunter_kinodynamic_rl.config.loader import load_profile  # noqa: E402
from hunter_kinodynamic_rl.env.scenarios.seed_scheduler import SeedScheduler  # noqa: E402
from hunter_kinodynamic_rl.env.simulation import risk_telemetry as rt  # noqa: E402
from hunter_kinodynamic_rl.rl.replay.buffer import ReplayBuffer  # noqa: E402
from hunter_kinodynamic_rl.training.trainer_base import TrainerBase  # noqa: E402


class _AlwaysSucceedsEnv:
    """Every episode is exactly 3 steps and ends in success (target=True,
    collision=False) -- lets validation's success_rate/collision_rate be
    asserted exactly, and proves transitions from these calls never reach
    the replay buffer regardless of who calls .step() (training vs
    validation)."""

    latest_sim_time_sec = None
    latest_pose = None

    def __init__(self, state_dim: int):
        self._state_dim = state_dim
        self._episode_step = 0
        self.step_calls = 0
        self.explicit_seed_modes = []

    def set_explicit_seed_mode(self, mode):
        self.explicit_seed_modes.append(mode)
        return True

    def seed(self, seed):
        pass

    def reset(self):
        self._episode_step = 0
        return np.zeros(self._state_dim, dtype=np.float32)

    def step(self, action):
        self._episode_step += 1
        self.step_calls += 1
        done = self._episode_step >= 3
        target = done
        telemetry = rt.invalid(step_id=self._episode_step)
        return np.zeros(self._state_dim, dtype=np.float32), 1.0, done, target, False, 10.0, telemetry


class _AlwaysCollidesEnv(_AlwaysSucceedsEnv):
    def step(self, action):
        self._episode_step += 1
        self.step_calls += 1
        done = self._episode_step >= 3
        collision = done
        telemetry = rt.invalid(step_id=self._episode_step)
        return np.zeros(self._state_dim, dtype=np.float32), -1.0, done, False, collision, 10.0, telemetry


class _FakeAgent:
    device = "cpu"

    def __init__(self):
        self.training_steps = 0

    def select_action(self, state, deterministic=False):
        return np.zeros(3, dtype=np.float32)

    def ent_coef_state(self):
        return {}

    def load_ent_coef_state(self, state):
        pass

    def checkpoint_components(self):
        return {}


class _FakeLogger:
    def log_episode_start(self, *a, **kw):
        pass

    def log_step(self, *a, **kw):
        pass

    def log_episode_end(self, *a, **kw):
        pass

    def log_train_metrics(self, *a, **kw):
        pass

    def log_validation(self, *a, **kw):
        pass


def _make_trainer(env, run_dir: str, eval_episodes: int = 4) -> TrainerBase:
    profile = load_profile("smoke_test")
    profile = dataclasses.replace(
        profile,
        training=dataclasses.replace(profile.training, eval_episodes=eval_episodes, episode_length_steps=10),
        counterfactual=dataclasses.replace(profile.counterfactual, enabled=False),
        risk=dataclasses.replace(profile.risk, enabled=False),
    )
    trainer = TrainerBase.__new__(TrainerBase)
    trainer.profile = profile
    trainer.run_dir = run_dir
    trainer.state_dim = 4
    trainer.action_dim = 3
    trainer.max_action = 1.0
    trainer.max_candidates = 0
    trainer.env = env
    trainer.replay_buffer = ReplayBuffer(trainer.state_dim, trainer.action_dim, capacity=1000, seed=0, max_candidates=1)
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
    return trainer


def test_validation_reports_success_rate_one_when_every_episode_succeeds(tmp_path):
    env = _AlwaysSucceedsEnv(4)
    trainer = _make_trainer(env, str(tmp_path / "run"), eval_episodes=4)
    result = trainer._run_validation_episodes()
    assert result["num_episodes"] == 4
    assert result["success_rate"] == pytest.approx(1.0)
    assert result["collision_rate"] == pytest.approx(0.0)


def test_validation_reports_collision_rate_one_when_every_episode_collides(tmp_path):
    env = _AlwaysCollidesEnv(4)
    trainer = _make_trainer(env, str(tmp_path / "run"), eval_episodes=4)
    result = trainer._run_validation_episodes()
    assert result["collision_rate"] == pytest.approx(1.0)
    assert result["success_rate"] == pytest.approx(0.0)


def test_validation_transitions_never_reach_the_replay_buffer(tmp_path):
    env = _AlwaysSucceedsEnv(4)
    trainer = _make_trainer(env, str(tmp_path / "run"), eval_episodes=5)
    assert len(trainer.replay_buffer) == 0
    trainer._run_validation_episodes()
    # 5 validation episodes x 3 steps each = 15 real env.step() calls...
    assert env.step_calls == 15
    # ...but NONE stored -- _run_validation_episodes never calls
    # _store_transition at all, by construction.
    assert len(trainer.replay_buffer) == 0


def test_validation_draws_from_the_validation_pool_never_train_or_test(tmp_path):
    env = _AlwaysSucceedsEnv(4)
    trainer = _make_trainer(env, str(tmp_path / "run"), eval_episodes=3)
    seeds_seen = []
    real_seed = env.seed

    def _spy_seed(seed):
        seeds_seen.append(seed)
        return real_seed(seed)

    env.seed = _spy_seed
    trainer._run_validation_episodes()

    val_lo, val_hi = trainer.profile.scenario.validation_seed_range
    for s in seeds_seen:
        assert val_lo <= s <= val_hi
    assert env.explicit_seed_modes == ["validation", "train"]


def test_run_writes_a_best_checkpoint_when_validation_improves_on_the_metric(tmp_path):
    """End-to-end: a fresh trainer's first periodic checkpoint (eval_freq
    reached at an episode boundary) with an always-succeeding validation
    env must produce a REAL "best" checkpoint on disk (eval_metric=1.0 >
    the initial -inf), alongside "latest"."""
    env = _AlwaysSucceedsEnv(4)
    run_dir = str(tmp_path / "run")
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    trainer = _make_trainer(env, run_dir, eval_episodes=2)
    trainer.profile = dataclasses.replace(
        trainer.profile,
        training=dataclasses.replace(trainer.profile.training, eval_freq=3, max_timesteps=3,
                                      timesteps_before_training=0, eval_episodes=2),
    )

    trainer.run()

    assert os.path.isfile(os.path.join(run_dir, "checkpoints", "best", "manifest.json"))
    assert trainer.best_eval_metric == pytest.approx(1.0)


def test_run_first_ever_validation_always_counts_as_a_new_best(tmp_path):
    """-inf is the initial floor -- ANY finite metric (even a bad one,
    e.g. every validation episode colliding) counts as an improvement on
    the very first validation pass. Documents this explicitly rather than
    assuming "bad outcome -> no best checkpoint"."""
    env = _AlwaysCollidesEnv(4)
    run_dir = str(tmp_path / "run")
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    trainer = _make_trainer(env, run_dir, eval_episodes=2)
    trainer.profile = dataclasses.replace(
        trainer.profile,
        training=dataclasses.replace(trainer.profile.training, eval_freq=3, max_timesteps=3,
                                      timesteps_before_training=0, eval_episodes=2),
    )

    trainer.run()

    assert os.path.isfile(os.path.join(run_dir, "checkpoints", "best", "manifest.json"))
    assert trainer.best_eval_metric == pytest.approx(-1.0)  # collision_rate=1.0 -> 0 - 1 = -1.0


def test_run_does_not_regress_best_metric_on_a_worse_validation(tmp_path):
    """A validation pass that's WORSE than the CURRENT best_eval_metric
    must NOT overwrite it -- simulated via a trainer whose
    best_eval_metric already reflects a strong prior result (e.g. resumed
    from a checkpoint), then run with an always-COLLIDING validation env."""
    env = _AlwaysCollidesEnv(4)
    run_dir = str(tmp_path / "run")
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    trainer = _make_trainer(env, run_dir, eval_episodes=2)
    trainer.best_eval_metric = 1.0  # a strong prior result (e.g. from a resumed checkpoint)
    trainer.profile = dataclasses.replace(
        trainer.profile,
        training=dataclasses.replace(trainer.profile.training, eval_freq=3, max_timesteps=3,
                                      timesteps_before_training=0, eval_episodes=2),
    )

    trainer.run()

    # collision_rate=1.0 -> eval_metric = 0 - 1 = -1.0, strictly WORSE than
    # the pre-seeded best_eval_metric=1.0 -> must not regress.
    assert trainer.best_eval_metric == pytest.approx(1.0)
    assert not os.path.lexists(os.path.join(run_dir, "checkpoints", "best"))
