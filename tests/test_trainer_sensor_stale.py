"""Regression test for P0-9's trainer-side half: TrainerBase.run() must
NEVER call _store_transition() for a step whose telemetry.sensor_stale is
True -- environment_node.py already truncates the episode (done=True) on
its own end, but the trainer's OWN responsibility is to keep that one
uncertain transition out of the replay buffer entirely, not just rely on
`done` marking it terminal.

trainer_base.py imports rclpy at module scope -- self-skips cleanly on a
bare host checkout, mirroring test_checkpoint_boundary_wiring.py's pattern
(which this borrows its fake-object harness from).
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


class _FakeEnv:
    """step_id 3 (of 5) reports sensor_stale=True; every other step is
    normal -- isolates exactly one contaminated transition to check for."""

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
        stale = self._episode_step == 3
        done = stale or self._episode_step >= 5
        next_state = np.zeros(self._state_dim, dtype=np.float32)
        telemetry = rt.invalid(step_id=self._episode_step, sensor_stale=stale)
        return next_state, 0.0, done, False, False, 10.0, telemetry, None


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


def test_sensor_stale_transition_is_never_stored_in_replay_buffer(tmp_path):
    profile = load_profile("smoke_test")
    profile = dataclasses.replace(
        profile,
        training=dataclasses.replace(
            profile.training, eval_freq=1000, max_timesteps=5, timesteps_before_training=0,
        ),
        counterfactual=dataclasses.replace(profile.counterfactual, enabled=False),
        risk=dataclasses.replace(profile.risk, enabled=False),
    )

    trainer = TrainerBase.__new__(TrainerBase)
    trainer.profile = profile
    trainer.run_dir = str(tmp_path / "run")
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

    trainer.run()

    # 5 steps run, step 3 is sensor_stale AND ends its episode (done=True) --
    # only steps 1, 2 should have been stored (step 3 skipped; a fresh
    # episode then starts and runs 2 more steps, 4 and 5, both stored).
    assert len(trainer.replay_buffer) == 4
