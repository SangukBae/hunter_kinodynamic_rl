"""section item-5: real (never mocked) coverage for the opt-in
``runtime.inference_worker_mode='process'`` path -- ``InferenceWorkerProcess``
spawns a GENUINE OS subprocess (multiprocessing, 'spawn'), loads a REAL
checkpoint into it, and communicates over REAL Queue-based IPC. Unlike
tests/test_real_policy_node.py's duck-typed fake-agent harness (which
injects a locally-defined class as ``node.agent`` -- impossible to hand to a
'spawn'-started child process, since it must be importable/picklable by
reference), these tests build an actual tiny checkpoint on disk and run the
actual worker-process entry point end-to-end.

``HKRL_TEST_FORCE_INFERENCE_HANG=1`` is a narrow, explicit, documented
TEST-ONLY hook inside ``_inference_worker_process_main`` itself (checked via
``os.environ``, off unless a test sets it) -- the only practical way to make
a genuinely SEPARATE, freshly-spawned interpreter hang predictably: a
`monkeypatch.setattr` on the agent class in THIS (parent) process has no
effect on a 'spawn'-started child, which re-imports every module fresh.
"""

import dataclasses

import numpy as np
import pytest

pytest.importorskip("rclpy")
torch = pytest.importorskip("torch")

import time  # noqa: E402

from hunter_kinodynamic_rl.config.loader import load_profile  # noqa: E402
from hunter_kinodynamic_rl.nodes.real_policy_node import (  # noqa: E402
    InferenceWorkerProcess, RealPolicyNode, _safety_limits_from_profile,
)
from hunter_kinodynamic_rl.rl.algorithms.tqc.agent import Agent as VanillaAgent  # noqa: E402
from hunter_kinodynamic_rl.rl.checkpointing import manager as ckpt_manager  # noqa: E402
from hunter_kinodynamic_rl.rl.replay.buffer import ReplayBuffer  # noqa: E402
from hunter_kinodynamic_rl.sensing.temporal_stack import FrameStack  # noqa: E402
from hunter_kinodynamic_rl.trajectory.action_space import ACTION_DIM  # noqa: E402


def _tiny_profile():
    # kinodynamic_tqc.yaml: features.risk_critic=false -- VanillaAgent,
    # the cheapest real agent this package has to build/checkpoint/load.
    return load_profile("kinodynamic_tqc")


def _state_dim(profile) -> int:
    history_len = profile.observation.frame_stack if profile.features.temporal_context else 1
    return profile.observation.lidar_bins * history_len + profile.observation.robot_state_dim


def _make_checkpoint(tmp_path, profile) -> tuple:
    state_dim = _state_dim(profile)
    agent = VanillaAgent(state_dim, ACTION_DIM, 1.0, profile.hyperparameters)
    replay = ReplayBuffer(state_dim, ACTION_DIM, capacity=10, seed=0, max_candidates=1)
    replay.add(state=[0.0] * state_dim, action=[0.0] * ACTION_DIM, next_state=[0.0] * state_dim,
               reward=0.0, done=False)
    ckpt_manager.save_generation(str(tmp_path), "ckpt", agent.checkpoint_components(), meta={}, replay_buffer=replay)
    return state_dim, agent


def test_inference_worker_process_roundtrip_returns_a_real_action(tmp_path):
    """A genuine spawn -> checkpoint-load -> inference -> IPC-response
    round trip, with no fake/mocked agent anywhere in the chain."""
    profile = _tiny_profile()
    state_dim, _agent = _make_checkpoint(tmp_path, profile)
    worker = InferenceWorkerProcess(profile, str(tmp_path), "ckpt", ready_timeout_sec=60.0)
    try:
        assert not worker.busy
        observation = np.zeros(state_dim, dtype=np.float32)
        worker.submit(observation)
        assert worker.busy
        action, error = worker.poll(10.0)
        assert error is None, error
        assert action is not None
        assert action.shape == (ACTION_DIM,)
        assert np.all(np.isfinite(action))
        assert not worker.busy
    finally:
        worker.shutdown()


def test_inference_worker_process_submit_while_busy_raises_single_flight_violation(tmp_path):
    profile = _tiny_profile()
    state_dim, _agent = _make_checkpoint(tmp_path, profile)
    worker = InferenceWorkerProcess(profile, str(tmp_path), "ckpt", ready_timeout_sec=60.0)
    try:
        worker.submit(np.zeros(state_dim, dtype=np.float32))
        with pytest.raises(RuntimeError, match="single-flight"):
            worker.submit(np.zeros(state_dim, dtype=np.float32))
        worker.poll(10.0)  # drain the first request so shutdown doesn't need to wait on it
    finally:
        worker.shutdown()


def test_inference_worker_process_reports_a_construction_failure_instead_of_hanging(tmp_path):
    """A checkpoint that does not match the requested profile's architecture
    (e.g. missing a component the agent declares) must surface as a raised
    error from InferenceWorkerProcess's own constructor -- never leave the
    caller blocked forever waiting for a worker that can never become
    ready."""
    profile = _tiny_profile()
    # An EMPTY directory -- no checkpoint at all under "ckpt".
    with pytest.raises(RuntimeError, match="failed to initialize"):
        InferenceWorkerProcess(profile, str(tmp_path), "ckpt", ready_timeout_sec=60.0)


def test_inference_worker_process_restart_terminates_a_genuinely_hung_worker_and_recovers(tmp_path, monkeypatch):
    """section item-5, the CORE new regression: a genuinely (not just
    slow) wedged inference call inside the worker SUBPROCESS -- something a
    thread-mode worker structurally cannot recover from -- gets forcibly
    terminated by restart(), and the FRESH replacement process serves a
    real, correct inference afterward."""
    profile = _tiny_profile()
    state_dim, _agent = _make_checkpoint(tmp_path, profile)
    monkeypatch.setenv("HKRL_TEST_FORCE_INFERENCE_HANG", "1")
    worker = InferenceWorkerProcess(profile, str(tmp_path), "ckpt", ready_timeout_sec=60.0)
    try:
        pid_before = worker._process.pid
        worker.submit(np.zeros(state_dim, dtype=np.float32))
        action, error = worker.poll(1.5)  # far shorter than the 3600s test-hook sleep
        assert action is None and error is None
        assert worker.busy, "the worker must still be genuinely wedged, not have somehow returned"

        monkeypatch.setenv("HKRL_TEST_FORCE_INFERENCE_HANG", "0")  # the RESPAWNED child must not hang
        worker.restart()
        assert worker._process.pid != pid_before
        assert not worker.busy

        observation = np.zeros(state_dim, dtype=np.float32)
        worker.submit(observation)
        action2, error2 = worker.poll(10.0)
        assert error2 is None, error2
        assert action2 is not None and action2.shape == (ACTION_DIM,)
        assert np.all(np.isfinite(action2))
    finally:
        monkeypatch.setenv("HKRL_TEST_FORCE_INFERENCE_HANG", "0")
        worker.shutdown()


def test_inference_worker_process_shutdown_is_idempotent(tmp_path):
    profile = _tiny_profile()
    _state_dim_val, _agent = _make_checkpoint(tmp_path, profile)
    worker = InferenceWorkerProcess(profile, str(tmp_path), "ckpt", ready_timeout_sec=60.0)
    worker.shutdown()
    worker.shutdown()  # must not raise on a second call


class _FakePublisher:
    def __init__(self):
        self.published = []

    def publish(self, msg):
        self.published.append(msg)


class _NullLogger:
    def error(self, *a, **kw):
        pass

    def warn(self, *a, **kw):
        pass

    def info(self, *a, **kw):
        pass


def test_on_control_tick_dispatches_to_a_real_process_mode_worker_end_to_end(tmp_path):
    """Full integration: RealPolicyNode._on_control_tick, dispatching
    through _infer_with_timeout -> _infer_with_timeout_process, driving a
    REAL InferenceWorkerProcess to publish a real command -- proves the
    dispatch wiring itself, not just InferenceWorkerProcess in isolation."""
    profile = _tiny_profile()
    profile = dataclasses.replace(
        profile, runtime=dataclasses.replace(profile.runtime, inference_worker_mode="process"))
    state_dim, _agent = _make_checkpoint(tmp_path, profile)

    worker = InferenceWorkerProcess(profile, str(tmp_path), "ckpt", ready_timeout_sec=60.0)
    try:
        node = RealPolicyNode.__new__(RealPolicyNode)
        node.profile = profile
        node.dry_run = False
        node.replay_mode = False
        node._estopped = False
        node._cmd_pub = _FakePublisher()
        node._diag_pub = _FakePublisher()
        node.get_logger = lambda: _NullLogger()
        node.goal_x = 1.0
        node.goal_y = 0.5
        node._latest_steering_rad = 0.0
        node._prev_action_01 = [0.0, 0.0, 0.0]
        history_len = profile.observation.frame_stack if profile.features.temporal_context else 1
        node._frame_stack = FrameStack(profile.observation.lidar_bins, history_len)
        node._frame_stack_ready = False
        node._safety_limits = _safety_limits_from_profile(profile)
        node._inference_timeouts = 0
        node._inference_errors = 0
        node._last_successful_inference_time = None
        node._last_command_time = None
        node.agent = None
        node._inference_worker = worker
        node._inference_thread = None
        node._inference_consecutive_timeouts = 0

        n_ranges = 360
        ranges = np.full(n_ranges, profile.observation.lidar_max_range_m, dtype=np.float32)
        now = time.monotonic()
        node._latest_scan = (ranges, -np.pi, 2 * np.pi / n_ranges)
        node._latest_scan_time = now
        node._latest_odom = (0.0, 0.0, 0.0, 0.0, 0.0)
        node._latest_odom_time = now

        node._on_control_tick()

        assert node._inference_timeouts == 0
        assert node._inference_errors == 0
        assert node._last_successful_inference_time is not None
        assert len(node._cmd_pub.published) == 1
    finally:
        worker.shutdown()
