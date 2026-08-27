"""The strongest form of the P0-4 regression: prove that an uninterrupted
N-step run and a "run K steps -> save -> kill process -> construct a BRAND
NEW trainer object -> resume -> run the remaining N-K steps" run produce
the EXACT SAME action sequence end-to-end.

This isolates the TRAINER's own RNG-driven decisions (stochastic action
sampling via python random / numpy / torch, which any real policy's
exploration noise depends on) from Gazebo's own physics, which is not
covered by this fix and is not claimed to be bit-reproducible across
separate process runs. Uses a deliberately RNG-hungry fake agent so any
one of python-random / numpy / torch state NOT being restored correctly
would make this test fail -- see _StochasticFakeAgent.select_action.

Goes through the REAL _save_checkpoint()/_resume_from() disk round-trip
(unlike test_checkpoint_boundary_wiring.py, which stubs _save_checkpoint
out entirely to isolate the episode-boundary WIRING instead) -- this test
is about RNG-restoration correctness, that one is about WHEN saves happen.
"""

import dataclasses
import json
import os
import random

import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("drl_agent_interfaces")  # training.trainer_base imports it at module scope
torch = pytest.importorskip("torch")

import numpy as np  # noqa: E402

from hunter_kinodynamic_rl.config.loader import load_profile  # noqa: E402
from hunter_kinodynamic_rl.env.scenarios.seed_scheduler import SeedScheduler  # noqa: E402
from hunter_kinodynamic_rl.env.simulation import risk_telemetry as rt  # noqa: E402
from hunter_kinodynamic_rl.rl.replay.buffer import ReplayBuffer  # noqa: E402
from hunter_kinodynamic_rl.training.trainer_base import TrainerBase  # noqa: E402

_EPISODE_LEN = 4


class _FakeEnv:
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
        return next_state, 0.0, done, False, False, 10.0, telemetry, None


class _StochasticFakeAgent:
    """Draws from EVERY RNG stream _save_checkpoint/_resume_from claims to
    restore (python random, numpy, torch CPU) -- a real stochastic policy's
    forward pass + exploration noise sampling does the equivalent. If even
    one stream is not correctly saved/restored across a checkpoint
    boundary, the resumed run's action sequence diverges from mile K
    onward and this test catches it."""

    device = "cpu"

    def __init__(self):
        self.training_steps = 0

    def select_action(self, state, deterministic=False):
        a = float(np.random.uniform(-1.0, 1.0))
        b = random.uniform(-1.0, 1.0)
        c = float(torch.rand(1).item())
        self.training_steps += 1
        return np.array([a, b, c], dtype=np.float32)

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

    def log_resume(self, *a, **kw):
        pass

    def log_validation(self, *a, **kw):
        pass


def _make_trainer(run_dir: str, eval_freq: int, max_timesteps: int, seed: int = 0) -> TrainerBase:
    profile = load_profile("smoke_test")
    profile = dataclasses.replace(
        profile,
        training=dataclasses.replace(
            profile.training, eval_freq=eval_freq, max_timesteps=max_timesteps,
            timesteps_before_training=0, seed=seed,
        ),
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
    trainer.env = _FakeEnv(trainer.state_dim)
    trainer.replay_buffer = ReplayBuffer(
        trainer.state_dim, trainer.action_dim, capacity=1000, seed=seed, max_candidates=1,
    )
    trainer.agent = _StochasticFakeAgent()
    trainer.agent_train_step = lambda batch: {}
    trainer.seed_scheduler = SeedScheduler(seed, profile.scenario, mode="train")
    trainer.validation_seed_scheduler = SeedScheduler(seed, profile.scenario, mode="validation")
    trainer.global_step = 0
    trainer.episode_index = 0
    trainer.episode_reward = 0.0
    trainer.episode_len = 0
    trainer.best_eval_metric = -float("inf")
    trainer._last_checkpoint_step = 0
    trainer.logger = _FakeLogger()
    # section P1-13: this test suite is specifically about TRAINING-loop
    # RNG-stream determinism (agent.select_action's action sequence across
    # a checkpoint/resume boundary) -- stubbed out so validation's OWN
    # agent.select_action(deterministic=True) calls never mix into the
    # recorded action sequence this test compares (see
    # test_trainer_validation.py for validation's own dedicated coverage).
    trainer._run_validation_episodes = lambda: {
        "success_rate": 0.0, "collision_rate": 0.0, "mean_reward": 0.0, "num_episodes": 0,
    }
    return trainer


def _run_and_record_actions(trainer: TrainerBase) -> list:
    """Also records every replay-buffer SAMPLE draw (the indices
    ``sample_indices`` returns) alongside actions -- returned as
    ``trainer._recorded_replay_samples`` so callers can compare both the
    policy's action sequence AND the replay buffer's own sampling RNG
    sequence across an interrupted-and-resumed run (section P0-1: "다음
    seed, replay sample, action/loss가 허용 오차 내에서 일치해야 한다")."""
    recorded = []
    real_step = trainer.env.step

    def _recording_step(action):
        recorded.append(np.array(action, dtype=np.float32).copy())
        return real_step(action)

    trainer.env.step = _recording_step

    recorded_samples = []
    real_sample_indices = trainer.replay_buffer.sample_indices

    def _recording_sample_indices(batch_size):
        indices = real_sample_indices(batch_size)
        recorded_samples.append(np.array(indices, dtype=np.int64).copy())
        return indices

    trainer.replay_buffer.sample_indices = _recording_sample_indices

    trainer.run()
    trainer._recorded_replay_samples = recorded_samples
    return recorded


def test_action_sequence_identical_across_an_interrupted_and_resumed_run(tmp_path):
    seed_all_streams = 12345

    # ---- Trial A: uninterrupted, straight through N steps ----
    import hunter_kinodynamic_rl.common.seed as seed_mod
    seed_mod.seed_all(seed_all_streams)
    trainer_a = _make_trainer(str(tmp_path / "trial_a"), eval_freq=4, max_timesteps=14, seed=7)
    actions_a = _run_and_record_actions(trainer_a)
    assert len(actions_a) == 14

    # ---- Trial B: same seeds, but killed after 8 steps and resumed ----
    seed_mod.seed_all(seed_all_streams)
    trainer_b1 = _make_trainer(str(tmp_path / "trial_b"), eval_freq=4, max_timesteps=8, seed=7)
    actions_b_part1 = _run_and_record_actions(trainer_b1)
    assert len(actions_b_part1) == 8

    # Simulate the process actually dying and a BRAND NEW TrainerBase being
    # constructed later to resume -- reusing trainer_b1's live Python
    # objects would trivially "work" even if _resume_from() were broken,
    # since in-memory RNG state would still be correctly positioned by
    # sheer continuity. A fresh object with globally RE-RANDOMIZED RNG
    # streams (seeded differently here on purpose) proves _resume_from()
    # itself, not object reuse, is what makes this deterministic.
    seed_mod.seed_all(seed_all_streams + 999)
    trainer_b2 = _make_trainer(str(tmp_path / "trial_b"), eval_freq=4, max_timesteps=14, seed=7)
    trainer_b2._resume_from(str(tmp_path / "trial_b"), checkpoint_tag="latest")
    assert trainer_b2.global_step == trainer_b1._last_checkpoint_step
    actions_b_part2 = _run_and_record_actions(trainer_b2)

    actions_b_full = actions_b_part1[:trainer_b1._last_checkpoint_step] + actions_b_part2
    assert len(actions_b_full) == 14

    for i, (a, b) in enumerate(zip(actions_a, actions_b_full)):
        np.testing.assert_array_equal(a, b, err_msg=f"action sequence diverged at step {i + 1}")

    # ---- The replay buffer's OWN sampling RNG sequence must ALSO survive
    # the interrupt/resume boundary -- a SEPARATE np.random.RandomState
    # instance from python-random/numpy-global/torch (rl/replay/buffer.py's
    # self._rng), restored via its own explicit rng_state_* fields in the
    # saved .npz (see ReplayBuffer.save/load). Compare the FULL sequence of
    # sampled batch indices, not just actions, closing the exact gap named
    # in section P0-1.
    samples_a = trainer_a._recorded_replay_samples
    # Mirrors actions_b_full's own truncation above: only sample() calls up
    # to and including the checkpoint boundary count from trial B's first
    # (killed) run -- timesteps_before_training=0 means one sample() call
    # per env.step() from step 1 onward, so the same _last_checkpoint_step
    # index truncates both sequences consistently.
    samples_b_full = trainer_b1._recorded_replay_samples[:trainer_b1._last_checkpoint_step] + \
        trainer_b2._recorded_replay_samples
    assert len(samples_a) == len(samples_b_full)
    for i, (idx_a, idx_b) in enumerate(zip(samples_a, samples_b_full)):
        np.testing.assert_array_equal(
            idx_a, idx_b, err_msg=f"replay-buffer sampled indices diverged at training step {i + 1}")


def test_saved_last_checkpoint_step_matches_its_own_global_step(tmp_path):
    """A checkpoint's own meta must describe ITSELF, not the PREVIOUS
    checkpoint -- found live (P0-4 follow-up): _last_checkpoint_step was
    being written into meta BEFORE being advanced to the current step, so
    every saved checkpoint's `last_checkpoint_step` field pointed at the
    PRIOR save instead of its own. Harmless for the core seed/RNG-sequence
    determinism (still fixed by the episode-boundary gate alone), but it
    corrupts the periodic-save CADENCE after a resume (the next save fires
    far too soon, thinking less time has elapsed since the last one than
    actually has)."""
    trainer = _make_trainer(str(tmp_path / "trial_c"), eval_freq=4, max_timesteps=8, seed=3)
    trainer.run()

    manifest_path = os.path.join(str(tmp_path / "trial_c"), "checkpoints", "latest", "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    assert manifest["last_checkpoint_step"] == manifest["global_step"] == 8


def test_checkpoint_manifest_embeds_the_full_resolved_config_not_just_profile_name(tmp_path):
    """A checkpoint must be self-describing regardless of what the
    profile's YAML file on disk looks like LATER -- configs/
    profile_snapshot.json (RunLogger) is a separate, run-level file that
    gets OVERWRITTEN on every process start (including a later resume), so
    it cannot preserve what config was actually in effect for an EARLIER
    checkpoint if the same-named profile is edited between runs. Every
    checkpoint's own manifest must carry its own resolved config."""
    trainer = _make_trainer(str(tmp_path / "trial_d"), eval_freq=4, max_timesteps=4, seed=1)
    trainer.run()

    manifest_path = os.path.join(str(tmp_path / "trial_d"), "checkpoints", "latest", "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    assert manifest["schema_version"] == 2
    assert "resolved_config" in manifest
    resolved = manifest["resolved_config"]
    # Spot-check a real, non-default value survives the round trip (proves
    # this is the ACTUAL resolved profile, not an empty/default stand-in).
    assert resolved["training"]["seed"] == 1
    assert resolved["name"] == trainer.profile.name
    assert "normalization_state" in manifest and manifest["normalization_state"] is None


def test_resume_raises_on_an_incomplete_checkpoint_missing_its_replay_buffer(tmp_path):
    """An incomplete checkpoint (the .pt/.json manifest exists but its
    companion _replay.npz is missing -- e.g. a disk-full mid-save, or a
    partial copy between machines) must FAIL LOUDLY on resume, never
    silently continue with a fresh, EMPTY replay buffer (which would
    silently corrupt training -- the agent would start sampling as if it
    had no prior experience despite global_step/model weights claiming
    otherwise)."""
    trainer = _make_trainer(str(tmp_path / "trial_e"), eval_freq=4, max_timesteps=4, seed=2)
    trainer.run()

    replay_path = os.path.join(str(tmp_path / "trial_e"), "checkpoints", "latest", "replay.npz")
    assert os.path.isfile(replay_path)  # test setup sanity
    os.remove(replay_path)

    fresh_trainer = _make_trainer(str(tmp_path / "trial_e"), eval_freq=4, max_timesteps=8, seed=2)
    with pytest.raises(FileNotFoundError, match="incomplete checkpoint"):
        fresh_trainer._resume_from(str(tmp_path / "trial_e"), checkpoint_tag="latest")


# --------------------------------------------------------------- section P1-4 / item-3
def test_checkpoint_manifest_records_replay_hash_and_generation(tmp_path):
    trainer = _make_trainer(str(tmp_path / "trial_f"), eval_freq=4, max_timesteps=4, seed=4)
    trainer.run()

    manifest_path = os.path.join(str(tmp_path / "trial_f"), "checkpoints", "latest", "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    assert "generation" in manifest and manifest["generation"]
    replay_path = os.path.join(str(tmp_path / "trial_f"), "checkpoints", "latest", "replay.npz")
    from hunter_kinodynamic_rl.rl.checkpointing import manager as ckpt_manager
    assert manifest["replay_sha256"] == ckpt_manager.sha256_of_file(replay_path)
    assert manifest["replay_size_bytes"] == os.path.getsize(replay_path)


def test_resume_raises_on_a_replay_file_from_a_different_generation(tmp_path):
    """section P1-4 / item-3: the core regression -- a replay file that
    EXISTS at the expected path but belongs to a DIFFERENT generation than
    the paired model.pt/manifest.json (e.g. left over from an interrupted
    save, or a manual file swap) must be rejected, not silently trusted
    just because the path resolves. Simulated here by swapping in the
    checkpoint's OWN "final" replay file (a genuinely different generation,
    same trainer/profile) at the "latest"-equivalent "final" slot -- since
    "final" is a symlink to a real ``.generations/<gen>/`` directory,
    writing through the symlink's own path mutates that directory's
    replay.npz directly, exactly like an out-of-band file copy would."""
    trainer = _make_trainer(str(tmp_path / "trial_g"), eval_freq=100, max_timesteps=4, seed=5)
    trainer.run()  # eval_freq > max_timesteps -- only the unconditional "final" save fires

    directory = os.path.join(str(tmp_path / "trial_g"), "checkpoints")
    # Run a SECOND, genuinely different trial so its final replay.npz has
    # different content, then swap it into trial_g's "final" slot.
    trainer2 = _make_trainer(str(tmp_path / "trial_g2"), eval_freq=100, max_timesteps=6, seed=6)
    trainer2.run()
    other_replay = os.path.join(str(tmp_path / "trial_g2"), "checkpoints", "final", "replay.npz")
    assert os.path.isfile(other_replay)  # test setup sanity

    import shutil
    shutil.copyfile(other_replay, os.path.join(directory, "final", "replay.npz"))

    fresh_trainer = _make_trainer(str(tmp_path / "trial_g"), eval_freq=100, max_timesteps=8, seed=5)
    with pytest.raises(RuntimeError, match="generation mismatch"):
        fresh_trainer._resume_from(str(tmp_path / "trial_g"), checkpoint_tag="final")
