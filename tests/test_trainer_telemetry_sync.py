"""Regression coverage for the risk-telemetry correctness bug (code
review): environment_node.py's ``reset_generation`` is a PROCESS-LIFETIME
monotonic counter, never reset per-client -- a fresh EnvironmentClient
that just assumes it starts at 0 permanently desyncs from the server's
real value the moment it isn't the FIRST client to ever talk to that
environment_node process (a restarted/resumed trainer reconnecting to an
already-running node, or a second concurrent EnvironmentClient). Confirmed
LIVE against real Gazebo/environment_node: this silently produced
risk_valid=False on 100% of an entire smoke-test training run's steps,
with the server's OWN published telemetry being genuinely valid the whole
time (verified directly via `ros2 topic echo`) -- a pure client-side
bookkeeping bug, not a communication failure. A SEPARATE, ALSO-real bug
(Gazebo's extreme-rate ``/clock`` topic starving ``rclpy.spin_once()`` from
ever reaching a lower-traffic subscription sharing its node/executor) was
found and fixed alongside it (see trainer_base.py's
``_RiskTelemetryListener``) -- not independently unit-testable without a
live rclpy graph with real topic traffic, so it's covered by the live
Gazebo verification documented in this task's final report instead.

P2 follow-up (code review): a SECOND version of this fix tried adding a
``reset_generation`` field to ``drl_agent_interfaces/srv/Reset.srv``'s
response, learned race-free by ``EnvironmentClient.reset()`` via the
RMW/DDS layer's atomic per-call request/response correlation -- closing a
residual multi-client-marker-misattribution gap the first (telemetry-only)
fix could not. Section P1-5 (shared-interface preservation) review found
that change violated this project's "reuse shared infrastructure
unmodified" requirement and reverted it: ``drl_agent_interfaces`` is
byte-identical to ``drl_agent``'s own again (see docs/SOURCE_MAP.md).
``reset_generation`` is learned PURELY from this package's OWN
risk-telemetry broadcast topic once more, via
``telemetry_is_new_reset_marker``/``_await_new_reset_marker`` below --
narrower than the reverted field's guarantee (documented: exactly one
client may call ``/reset`` on a given environment_node instance at a
time), not a regression back to the FIRST fix's fully-heuristic scheme
(this one relies on the server's own ``/reset``-callback serialization,
not observational freshness heuristics).

trainer_base.py imports rclpy/drl_agent_interfaces at module scope --
self-skips cleanly on a bare host checkout (mirrors every other
trainer_base.py test in this suite).
"""

import dataclasses

import pytest

pytest.importorskip("rclpy")
pytest.importorskip("drl_agent_interfaces")  # training.trainer_base imports it at module scope

from hunter_kinodynamic_rl.config.loader import load_profile  # noqa: E402
from hunter_kinodynamic_rl.env.simulation import risk_telemetry as rt  # noqa: E402
from hunter_kinodynamic_rl.env.simulation import sensor_diagnostics as sd  # noqa: E402
from hunter_kinodynamic_rl.env.scenarios.seed_scheduler import SeedScheduler  # noqa: E402
from hunter_kinodynamic_rl.rl.replay.buffer import ReplayBuffer  # noqa: E402
from hunter_kinodynamic_rl.training.trainer_base import (  # noqa: E402
    EnvironmentClient, EnvServiceError, TrainerBase, sensor_diagnostics_matches_step,
    telemetry_is_new_reset_marker, telemetry_matches_step,
)

# --------------------------------------------------------- pure predicates


def test_telemetry_matches_step_requires_exact_step_id_and_generation():
    t = rt.RiskTelemetry(
        step_id=5, valid=True, risk_target=0.1, min_clearance_m=1.0, ttc_sec=1.0,
        collision_within_horizon=False, stopping_margin_m=0.5, unrecoverable=False,
        safer_alternative_margin=0.0, actor_candidate_index=0, reset_generation=3,
    )
    assert telemetry_matches_step(t, expected_step_id=5, expected_reset_generation=3) is True
    assert telemetry_matches_step(t, expected_step_id=6, expected_reset_generation=3) is False
    assert telemetry_matches_step(t, expected_step_id=5, expected_reset_generation=4) is False


def test_telemetry_matches_step_never_matches_when_generation_unknown():
    """The core fix's fallback: while a fresh episode's reset_generation
    hasn't been learned yet (None), NOTHING can match -- this is what makes
    _await_new_reset_marker's timeout path degrade safely (every step
    cleanly times out) instead of guessing."""
    t = rt.RiskTelemetry(
        step_id=1, valid=True, risk_target=0.1, min_clearance_m=1.0, ttc_sec=1.0,
        collision_within_horizon=False, stopping_margin_m=0.5, unrecoverable=False,
        safer_alternative_margin=0.0, actor_candidate_index=0, reset_generation=0,
    )
    assert telemetry_matches_step(t, expected_step_id=1, expected_reset_generation=None) is False


def test_telemetry_matches_step_none_telemetry_never_matches():
    assert telemetry_matches_step(None, expected_step_id=1, expected_reset_generation=1) is False


# -------------------------------------------- requirement 5: sensor_diagnostics
def test_sensor_diagnostics_matches_step_requires_exact_step_id_and_generation():
    d = sd.SensorDiagnostics(step_id=5, valid=True, reset_generation=3)
    assert sensor_diagnostics_matches_step(d, expected_step_id=5, expected_reset_generation=3) is True
    assert sensor_diagnostics_matches_step(d, expected_step_id=6, expected_reset_generation=3) is False
    assert sensor_diagnostics_matches_step(d, expected_step_id=5, expected_reset_generation=4) is False


def test_sensor_diagnostics_matches_step_never_matches_when_generation_unknown():
    d = sd.SensorDiagnostics(step_id=1, valid=True, reset_generation=0)
    assert sensor_diagnostics_matches_step(d, expected_step_id=1, expected_reset_generation=None) is False


def test_sensor_diagnostics_matches_step_none_never_matches():
    assert sensor_diagnostics_matches_step(None, expected_step_id=1, expected_reset_generation=1) is False


def test_new_reset_marker_requires_step_id_zero_and_reset_marker_reason():
    marker = rt.invalid(step_id=0, reset_generation=8, reason=rt.InvalidReason.RESET_MARKER)
    assert telemetry_is_new_reset_marker(marker, known_generation=7) is True

    wrong_step = rt.invalid(step_id=1, reset_generation=8, reason=rt.InvalidReason.RESET_MARKER)
    assert telemetry_is_new_reset_marker(wrong_step, known_generation=7) is False

    wrong_reason = rt.invalid(step_id=0, reset_generation=8, reason=rt.InvalidReason.NONE)
    assert telemetry_is_new_reset_marker(wrong_reason, known_generation=7) is False


def test_new_reset_marker_requires_strictly_greater_generation():
    marker = rt.invalid(step_id=0, reset_generation=8, reason=rt.InvalidReason.RESET_MARKER)
    assert telemetry_is_new_reset_marker(marker, known_generation=8) is False  # already-seen generation
    assert telemetry_is_new_reset_marker(marker, known_generation=9) is False  # somehow ahead -- never "new"
    assert telemetry_is_new_reset_marker(marker, known_generation=7) is True


def test_none_telemetry_never_matches_new_reset_marker():
    assert telemetry_is_new_reset_marker(None, known_generation=0) is False


# --------------------------- single-owner-client reset correlation (P1-5)
#
# section P1-5 (shared-interface preservation): a later revision of this
# module ADDED a ``reset_generation`` field to
# ``drl_agent_interfaces/srv/Reset.srv``'s response specifically to close a
# residual multi-client marker-misattribution gap the ORIGINAL
# (telemetry-only, generation-monotonic) fix could not -- a DIFFERENT
# client's marker landing strictly AFTER this client's own request was sent
# but BEFORE this client's own marker arrives was indistinguishable from
# "my own marker, just slow" under any purely observational heuristic. That
# field addition was reverted (docs/SOURCE_MAP.md) because it touched a
# SHARED interface this project requires stay unmodified. The tests below
# cover the reinstated telemetry-only design's actual guarantee: exactly
# one client calling ``/reset`` on a given environment_node instance at a
# time (true of every shipped profile/launch path) -- NOT the fully
# general concurrent-multi-client case the reverted field covered.


class _FakeTelemetryListener:
    """Stands in for _RiskTelemetryListener without needing a live rclpy
    subscription -- feed() simulates a message arriving."""

    def __init__(self):
        self.latest_telemetry = None

    def feed(self, telemetry):
        self.latest_telemetry = telemetry


class _FakeLoggerForReset:
    def warn(self, msg):
        pass


class _FakeEnvironmentClientSelf:
    """A minimal stand-in exposing exactly the attributes/methods
    EnvironmentClient._await_new_reset_marker (the REAL, unmodified
    production method, called bound to this fake) reads/writes -- lets
    these tests exercise the actual polling loop/logic without a live
    rclpy graph or a running environment_node."""

    def __init__(self, listener, wait_timeout_sec=0.3):
        self._telemetry_listener = listener
        self._reset_marker_wait_timeout_sec = wait_timeout_sec
        self._reset_generation = None
        self.reset_marker_timeouts = 0

    def _spin_telemetry_once(self, timeout_sec):
        pass  # no-op: this test drives the listener's state directly

    def get_logger(self):
        return _FakeLoggerForReset()


def test_await_new_reset_marker_ignores_a_stale_cached_marker_and_waits_for_a_new_one():
    """Regression: an OLD marker (generation 6, this client's own
    previously-known value) is already cached -- must not be mistaken for
    a fresh one, and must keep waiting until a genuinely NEW marker (fed
    mid-poll) arrives."""
    listener = _FakeTelemetryListener()
    listener.feed(rt.invalid(step_id=0, reset_generation=6, reason=rt.InvalidReason.RESET_MARKER))

    calls = {"n": 0}
    real_spin = _FakeEnvironmentClientSelf._spin_telemetry_once

    def spin_and_deliver(fake_self, timeout_sec):
        calls["n"] += 1
        if calls["n"] == 2:  # deliver the real marker a couple of polls in
            listener.feed(rt.invalid(step_id=0, reset_generation=7, reason=rt.InvalidReason.RESET_MARKER))
        return real_spin(fake_self, timeout_sec)

    fake_self = _FakeEnvironmentClientSelf(listener, wait_timeout_sec=2.0)
    fake_self._spin_telemetry_once = spin_and_deliver.__get__(fake_self)
    generation = EnvironmentClient._await_new_reset_marker(fake_self, known_generation=6)

    assert fake_self.reset_marker_timeouts == 0
    assert generation == 7


def test_await_new_reset_marker_raises_on_timeout():
    """section P1-5: unlike the (reverted) response-field design's purely
    diagnostic marker wait, a timeout HERE means reset_generation is
    genuinely unknown -- must raise, never silently continue with a
    guessed/stale value."""
    listener = _FakeTelemetryListener()  # nothing ever arrives

    fake_self = _FakeEnvironmentClientSelf(listener, wait_timeout_sec=0.1)
    with pytest.raises(EnvServiceError, match="no NEW reset-marker"):
        EnvironmentClient._await_new_reset_marker(fake_self, known_generation=42)
    assert fake_self.reset_marker_timeouts == 1


def test_await_new_reset_marker_accepts_a_normal_marker_without_timeout():
    """The common case: a marker newer than the known generation arrives
    promptly, well within the timeout budget."""
    listener = _FakeTelemetryListener()
    listener.feed(rt.invalid(step_id=0, reset_generation=1, reason=rt.InvalidReason.RESET_MARKER))

    fake_self = _FakeEnvironmentClientSelf(listener, wait_timeout_sec=2.0)
    start = __import__("time").monotonic()
    generation = EnvironmentClient._await_new_reset_marker(fake_self, known_generation=0)
    elapsed = __import__("time").monotonic() - start

    assert fake_self.reset_marker_timeouts == 0
    assert generation == 1
    assert elapsed < 1.0  # accepted promptly, not stalled to the full 2.0s budget


# ------------------------------------------------------------ reset() (P1-5)


class _FakeResetResult:
    def __init__(self, state=None):
        self.state = state if state is not None else [0.0]


class _FakeEnvironmentClientSelfForReset(_FakeEnvironmentClientSelf):
    """Extends the _await_new_reset_marker stand-in with everything
    EnvironmentClient.reset() (the REAL, unmodified production method,
    called bound to this fake) additionally reads/writes. ``reset()``
    calls ``self._await_new_reset_marker(...)`` (an attribute lookup on
    ``type(self)``, i.e. THIS class, not ``EnvironmentClient`` -- unlike
    the standalone tests above, which invoke
    ``EnvironmentClient._await_new_reset_marker(fake_self, ...)``
    directly), so the real implementation must be bound here too."""

    _await_new_reset_marker = EnvironmentClient._await_new_reset_marker

    def __init__(self, listener, wait_timeout_sec=0.3):
        super().__init__(listener, wait_timeout_sec)
        self._reset_result = _FakeResetResult()
        self._reset_client = object()
        self.episode_path_length_m = 1.23  # deliberately non-default
        self._latest_xy = (1.0, 2.0)
        self._latest_sim_time_sec = 5.0
        self._episode_start_sim_time_sec = None

    def _call(self, client, request):
        return self._reset_result


def test_reset_fails_fast_when_no_reset_marker_ever_arrives():
    """section P1-5: Reset.srv's response no longer carries
    reset_generation at all -- if the risk-telemetry broadcast never
    delivers a new marker either, reset() has no way to learn this
    episode's generation and must raise rather than silently proceed."""
    listener = _FakeTelemetryListener()
    fake_self = _FakeEnvironmentClientSelfForReset(listener, wait_timeout_sec=0.1)
    with pytest.raises(EnvServiceError, match="no NEW reset-marker"):
        EnvironmentClient.reset(fake_self)
    # Must raise BEFORE mutating any episode-tracking state.
    assert fake_self.episode_path_length_m == 1.23
    assert fake_self._latest_xy == (1.0, 2.0)
    assert fake_self._reset_generation is None


def test_reset_learns_generation_from_the_telemetry_marker():
    """The normal case: reset() sends the request, then learns
    reset_generation purely from the risk-telemetry RESET_MARKER (never
    from a Reset.srv response field, which no longer carries it)."""
    import numpy as np

    listener = _FakeTelemetryListener()
    listener.feed(rt.invalid(step_id=0, reset_generation=3, reason=rt.InvalidReason.RESET_MARKER))
    fake_self = _FakeEnvironmentClientSelfForReset(listener)

    state = EnvironmentClient.reset(fake_self)

    assert fake_self._reset_generation == 3
    assert fake_self._step_id == 0
    assert fake_self.episode_path_length_m == 0.0
    assert fake_self._latest_xy is None
    assert isinstance(state, np.ndarray)


def test_reset_requires_a_strictly_newer_generation_than_the_previous_episode():
    """A second reset() call must not accept a marker whose generation is
    <= what the FIRST reset() already learned (e.g. a stale cached value
    left over from the previous episode)."""
    listener = _FakeTelemetryListener()
    listener.feed(rt.invalid(step_id=0, reset_generation=3, reason=rt.InvalidReason.RESET_MARKER))
    fake_self = _FakeEnvironmentClientSelfForReset(listener)
    EnvironmentClient.reset(fake_self)
    assert fake_self._reset_generation == 3

    fake_self._reset_marker_wait_timeout_sec = 0.1
    # Still only the OLD (generation=3) marker cached -- no new one arrives.
    with pytest.raises(EnvServiceError, match="no NEW reset-marker"):
        EnvironmentClient.reset(fake_self)
    assert fake_self._reset_generation == 3  # left at the last GOOD value, not clobbered


# --------------------------------------------------- run()-level health check


class _FakeAgent:
    device = "cpu"

    def __init__(self, risk_supervised_updates=0):
        self.training_steps = 0
        self.risk_supervised_updates = risk_supervised_updates

    def select_action(self, state, deterministic=False):
        import numpy as np
        return np.zeros(3, dtype=np.float32)

    def ent_coef_state(self):
        return {}

    def load_ent_coef_state(self, state):
        pass

    def checkpoint_components(self):
        return {}


class _FakeEnv:
    """Every step reports telemetry.valid according to a scripted
    per-step sequence, and tracks the SAME telemetry_matched_count/
    telemetry_timeouts/reset_marker_timeouts attributes the real
    EnvironmentClient exposes -- exactly what
    TrainerBase._check_risk_telemetry_health reads."""

    latest_sim_time_sec = None
    latest_pose = None

    def __init__(self, state_dim: int, valid_sequence, episode_len: int = 1000):
        import numpy as np
        self._np = np
        self._state_dim = state_dim
        self._valid_sequence = valid_sequence
        self._i = 0
        self._episode_step = 0
        self._episode_len = episode_len
        self.telemetry_matched_count = 0
        self.telemetry_timeouts = 0
        self.reset_marker_timeouts = 0

    def seed(self, seed):
        pass

    def reset(self):
        self._episode_step = 0
        return self._np.zeros(self._state_dim, dtype=self._np.float32)

    def step(self, action):
        self._episode_step += 1
        valid = self._valid_sequence[self._i % len(self._valid_sequence)]
        self._i += 1
        if valid:
            self.telemetry_matched_count += 1
            telemetry = rt.RiskTelemetry(
                step_id=self._episode_step, valid=True, risk_target=0.1, min_clearance_m=1.0, ttc_sec=1.0,
                collision_within_horizon=False, stopping_margin_m=0.5, unrecoverable=False,
                safer_alternative_margin=0.0, actor_candidate_index=0,
            )
        else:
            self.telemetry_timeouts += 1
            telemetry = rt.invalid(step_id=self._episode_step, reason=rt.InvalidReason.POLL_TIMEOUT)
        done = self._episode_step >= self._episode_len
        next_state = self._np.zeros(self._state_dim, dtype=self._np.float32)
        diagnostics = sd.invalid(step_id=self._episode_step)
        return next_state, 0.0, done, False, False, 10.0, telemetry, diagnostics


class _FakeLogger:
    def __init__(self):
        self.telemetry_health_calls = []

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

    def log_telemetry_health(self, step, **kw):
        self.telemetry_health_calls.append((step, kw))


def _make_trainer(tmp_path, *, valid_sequence, max_timesteps, risk_aware, risk_overrides,
                   risk_supervised_updates=0, episode_len=1000):
    profile = load_profile("smoke_test")
    profile = dataclasses.replace(
        profile,
        training=dataclasses.replace(
            profile.training, eval_freq=100_000, max_timesteps=max_timesteps, timesteps_before_training=0,
        ),
        counterfactual=dataclasses.replace(profile.counterfactual, enabled=False),
        risk=dataclasses.replace(profile.risk, **risk_overrides),
    )

    trainer = TrainerBase.__new__(TrainerBase)
    trainer.risk_aware = risk_aware
    trainer.profile = profile
    trainer.run_dir = str(tmp_path / "run")
    trainer.state_dim = 4
    trainer.action_dim = 3
    trainer.max_action = 1.0
    trainer.max_candidates = 0
    trainer.env = _FakeEnv(trainer.state_dim, valid_sequence, episode_len=episode_len)
    trainer.replay_buffer = ReplayBuffer(trainer.state_dim, trainer.action_dim, capacity=10_000, seed=0,
                                          max_candidates=1)
    trainer.agent = _FakeAgent(risk_supervised_updates=risk_supervised_updates)
    trainer.agent_train_step = lambda batch: {}
    trainer.seed_scheduler = SeedScheduler(profile.training.seed, profile.scenario, mode="train")
    trainer.validation_seed_scheduler = SeedScheduler(profile.training.seed, profile.scenario, mode="validation")
    trainer.global_step = 0
    trainer.episode_index = 0
    trainer.episode_reward = 0.0
    trainer.episode_len = 0
    trainer.best_eval_metric = -float("inf")
    trainer._last_checkpoint_step = 0
    trainer._prev_step_sim_time_sec = None
    trainer.logger = _FakeLogger()
    return trainer


def test_vanilla_trainer_never_health_checks_regardless_of_telemetry_quality(tmp_path):
    """risk_aware=False (vanilla TQC/SAC): telemetry validity is
    diagnostic-only, never a training-correctness gate -- must run to
    completion (no raise) even with ZERO valid telemetry the whole run."""
    trainer = _make_trainer(
        tmp_path, valid_sequence=[False], max_timesteps=50, risk_aware=False,
        risk_overrides=dict(enabled=True, min_telemetry_valid_ratio=0.9, telemetry_valid_ratio_grace_steps=1,
                             telemetry_health_check_interval_steps=10),
    )
    result = trainer.run()
    assert result["final_step"] == 50


def test_risk_disabled_trainer_never_health_checks(tmp_path):
    """risk_aware=True but risk.enabled=False (an ablation profile that
    happens to use the risk-aware trainer class with the feature off) --
    also never gates."""
    trainer = _make_trainer(
        tmp_path, valid_sequence=[False], max_timesteps=50, risk_aware=True,
        risk_overrides=dict(enabled=False, min_telemetry_valid_ratio=0.9, telemetry_valid_ratio_grace_steps=1,
                             telemetry_health_check_interval_steps=10),
    )
    result = trainer.run()
    assert result["final_step"] == 50


def test_risk_aware_trainer_with_healthy_telemetry_completes_normally(tmp_path):
    trainer = _make_trainer(
        tmp_path, valid_sequence=[True], max_timesteps=50, risk_aware=True,
        risk_overrides=dict(enabled=True, min_telemetry_valid_ratio=0.5, telemetry_valid_ratio_grace_steps=10,
                             telemetry_health_check_interval_steps=10),
        risk_supervised_updates=5,
    )
    result = trainer.run()
    assert result["final_step"] == 50
    # health check actually ran and logged (not just silently skipped).
    assert len(trainer.logger.telemetry_health_calls) > 0
    last_step, last_kw = trainer.logger.telemetry_health_calls[-1]
    assert last_kw["valid_ratio"] == pytest.approx(1.0)


def test_risk_aware_trainer_with_systematically_broken_telemetry_raises(tmp_path):
    """The core regression: a risk-enabled run whose telemetry NEVER
    matches (the exact live-confirmed bug -- 100% risk_valid=False) must
    raise, not finish looking successful."""
    trainer = _make_trainer(
        tmp_path, valid_sequence=[False], max_timesteps=1000, risk_aware=True,
        risk_overrides=dict(enabled=True, min_telemetry_valid_ratio=0.3, telemetry_valid_ratio_grace_steps=20,
                             telemetry_health_check_interval_steps=10),
        episode_len=1000,
    )
    with pytest.raises(RuntimeError, match="risk-telemetry health check FAILED"):
        trainer.run()
    # Must have run PAST the grace period before raising, not immediately
    # on step 1 (would be a false alarm on an empty buffer).
    assert trainer.global_step >= 20


def test_risk_aware_trainer_within_grace_period_does_not_raise_early(tmp_path):
    """A short run that never gets past telemetry_valid_ratio_grace_steps
    must not raise mid-run purely from a noisy early sample -- but see the
    NEXT test for the final-step check that still applies."""
    trainer = _make_trainer(
        tmp_path, valid_sequence=[True], max_timesteps=15, risk_aware=True,
        risk_overrides=dict(enabled=True, min_telemetry_valid_ratio=0.99,
                             telemetry_valid_ratio_grace_steps=1000, telemetry_health_check_interval_steps=5,
                             actor_penalty_warmup_updates=0, min_valid_labels_per_batch=1),
        risk_supervised_updates=1,
    )
    result = trainer.run()  # would raise on valid_ratio if the grace period weren't honored
    assert result["final_step"] == 15


def test_risk_aware_trainer_final_check_catches_zero_supervised_updates_even_with_perfect_valid_ratio(tmp_path):
    """The second half of the review's requirement: even a PERFECT
    telemetry valid_ratio doesn't excuse risk_supervised_updates staying
    at 0 for the entire run (e.g. every batch had too few valid labels) --
    must still raise at the final check."""
    trainer = _make_trainer(
        tmp_path, valid_sequence=[True], max_timesteps=50, risk_aware=True,
        risk_overrides=dict(enabled=True, min_telemetry_valid_ratio=0.0, telemetry_valid_ratio_grace_steps=5,
                             telemetry_health_check_interval_steps=10),
        risk_supervised_updates=0,
    )
    with pytest.raises(RuntimeError, match="risk_supervised_updates==0"):
        trainer.run()


def test_checkpoint_meta_carries_telemetry_health_fields(tmp_path):
    """code review requirement: telemetry valid ratio / timeout count /
    risk_supervised_updates must be verifiable IN THE CHECKPOINT, not just
    in transient run-time state."""
    trainer = _make_trainer(
        tmp_path, valid_sequence=[True, True, False], max_timesteps=30, risk_aware=True,
        risk_overrides=dict(enabled=True, min_telemetry_valid_ratio=0.0, telemetry_valid_ratio_grace_steps=1000,
                             telemetry_health_check_interval_steps=1000),
        risk_supervised_updates=3,
    )
    trainer.run()
    import json
    import os
    manifest_path = os.path.join(trainer.run_dir, "checkpoints", "final", "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    assert manifest["telemetry_matched_count"] == trainer.env.telemetry_matched_count
    assert manifest["telemetry_timeouts"] == trainer.env.telemetry_timeouts
    assert manifest["reset_marker_timeouts"] == 0
    assert manifest["risk_supervised_updates"] == 3
    assert 0.0 < manifest["telemetry_valid_ratio"] < 1.0
