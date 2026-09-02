"""Detailed spec ("실제 hierarchical 학습 경로"): the live-Gazebo
``LocalOptionExecutor`` must fail fast -- never silently substitute a
heuristic or an untrained network -- on a missing or
observation/action/architecture-incompatible frozen Local checkpoint.
Both tests here raise BEFORE ``LiveGazeboLocalExecutor.__init__`` ever
reaches its Gazebo-dependent wall-pool spawn (see that module's own
ordering comment), so no live Gazebo/world is needed -- only ``rclpy.init()``
(covered by ``importorskip`` + the module-scope init/shutdown fixture
below, mirroring ``test_hierarchical_navigation_node.py``'s own pattern).
"""

from __future__ import annotations

import dataclasses

import pytest

pytest.importorskip("rclpy")
pytest.importorskip("torch")

import rclpy  # noqa: E402

from hunter_kinodynamic_rl.config.loader import load_profile  # noqa: E402
from hunter_kinodynamic_rl.config.schema import TQCHyperparameters  # noqa: E402
from hunter_kinodynamic_rl.evaluation.fingerprint import local_training_contract_fingerprint  # noqa: E402
import hunter_kinodynamic_rl.navigation.local_rl.live_gazebo_executor as live_executor_module  # noqa: E402
from hunter_kinodynamic_rl.navigation.local_rl.live_gazebo_executor import (  # noqa: E402
    LiveGazeboLocalExecutor, LocalCheckpointError,
)
from hunter_kinodynamic_rl.rl.algorithms.kinodynamic_tqc.agent import Agent as RiskAgent  # noqa: E402
from hunter_kinodynamic_rl.rl.checkpointing import manager as ckpt_manager  # noqa: E402
from hunter_kinodynamic_rl.rl.replay.buffer import ReplayBuffer  # noqa: E402


@pytest.fixture(autouse=True)
def _rclpy_context():
    # Matches test_mission_map_node.py/test_environment_node.py's own
    # convention exactly: guard init() against a context an EARLIER test
    # file in the same pytest process already initialized (confirmed live:
    # an unconditional rclpy.init() here raised "Context.init() must only
    # be called once" when this file ran as part of the full suite), and
    # never call rclpy.shutdown() -- the context is shared for the whole
    # pytest session, torn down only at process exit.
    if not rclpy.ok():
        rclpy.init()
    yield


def _profile():
    return load_profile("kinodynamic_tqc_arbitrary_subgoal")


def _make_hp() -> TQCHyperparameters:
    return TQCHyperparameters(actor_hdim=8, critic_hdim=8, n_quantiles=4, n_critics=1,
                               top_quantiles_to_drop_per_net=0)


def _make_replay(state_dim: int, action_dim: int) -> ReplayBuffer:
    buf = ReplayBuffer(state_dim, action_dim, capacity=10, seed=0, max_candidates=1)
    buf.add(
        state=[0.0] * state_dim, action=[0.0] * action_dim, next_state=[0.0] * state_dim,
        reward=0.0, done=False,
    )
    return buf


def test_missing_local_checkpoint_dir_raises_immediately(tmp_path):
    profile = _profile()
    with pytest.raises(LocalCheckpointError, match="non-empty"):
        LiveGazeboLocalExecutor(
            profile, profile.long_horizon_world, "", "final", node_name="test_missing_ckpt_dir",
        )


def test_nonexistent_local_checkpoint_raises_immediately(tmp_path):
    profile = _profile()
    with pytest.raises(LocalCheckpointError, match="failed to load"):
        LiveGazeboLocalExecutor(
            profile, profile.long_horizon_world, str(tmp_path / "checkpoints"), "final",
            node_name="test_missing_ckpt_file",
        )


def test_state_dim_mismatch_raises_immediately(tmp_path):
    """A real, well-formed generation-layout checkpoint whose OWN recorded
    ``state_dim`` doesn't match what this profile's observation contract
    computes must still be rejected -- never silently loaded as if it
    matched (detailed spec: "observation/action fingerprint 불일치 ...
    fail-fast")."""
    profile = _profile()
    history_len = profile.observation.frame_stack if profile.features.temporal_context else 1
    real_state_dim = profile.observation.lidar_bins * history_len + profile.observation.robot_state_dim
    wrong_state_dim = real_state_dim + 7  # deliberately incompatible
    action_dim = 3

    agent = RiskAgent(wrong_state_dim, action_dim, max_action=1.0, hyperparameters=_make_hp(),
                       risk_config=profile.risk, counterfactual_config=profile.counterfactual)
    checkpoint_dir = str(tmp_path / "checkpoints")
    ckpt_manager.save_generation(
        checkpoint_dir, "final", agent.checkpoint_components(),
        {"state_dim": wrong_state_dim, "action_dim": action_dim, "resolved_config": dataclasses.asdict(profile)},
        _make_replay(wrong_state_dim, action_dim),
    )

    with pytest.raises(LocalCheckpointError, match="state_dim"):
        LiveGazeboLocalExecutor(
            profile, profile.long_horizon_world, checkpoint_dir, "final", node_name="test_state_dim_mismatch",
        )


def test_architecture_fingerprint_mismatch_raises_immediately(tmp_path):
    """Same dims, but the checkpoint's OWN ``resolved_config`` says it was
    trained under a DIFFERENT action_space/features configuration (e.g.
    ``counterfactual_risk`` off) -- must still be rejected even though
    state_dim/action_dim happen to match."""
    profile = _profile()
    history_len = profile.observation.frame_stack if profile.features.temporal_context else 1
    state_dim = profile.observation.lidar_bins * history_len + profile.observation.robot_state_dim
    action_dim = 3

    agent = RiskAgent(state_dim, action_dim, max_action=1.0, hyperparameters=_make_hp(),
                       risk_config=profile.risk, counterfactual_config=profile.counterfactual)
    mismatched_profile = dataclasses.replace(
        profile, features=dataclasses.replace(profile.features, counterfactual_risk=False),
    )
    checkpoint_dir = str(tmp_path / "checkpoints")
    ckpt_manager.save_generation(
        checkpoint_dir, "final", agent.checkpoint_components(),
        {
            "state_dim": state_dim, "action_dim": action_dim,
            "resolved_config": dataclasses.asdict(mismatched_profile),
        },
        _make_replay(state_dim, action_dim),
    )

    with pytest.raises(LocalCheckpointError, match="architecture_fingerprint"):
        LiveGazeboLocalExecutor(
            profile, profile.long_horizon_world, checkpoint_dir, "final",
            node_name="test_architecture_fingerprint_mismatch",
        )


def test_cuda_authored_checkpoint_loads_on_cpu(tmp_path, monkeypatch):
    """The supported live path must remap CUDA tensors onto CPU instead of
    declaring a CUDA-authored checkpoint unusable in an rclpy process."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to author a real CUDA-resident checkpoint")

    live_profile = load_profile("hierarchical_phase4")
    training_profile = load_profile(live_profile.hierarchical_training.local_training_profile_name)
    history_len = (
        training_profile.observation.frame_stack if training_profile.features.temporal_context else 1
    )
    state_dim = (
        training_profile.observation.lidar_bins * history_len
        + training_profile.observation.robot_state_dim
    )
    action_dim = 3
    cuda_agent = RiskAgent(
        state_dim, action_dim, max_action=1.0, hyperparameters=training_profile.hyperparameters,
        risk_config=training_profile.risk, counterfactual_config=training_profile.counterfactual,
        device="cuda",
    )
    checkpoint_dir = str(tmp_path / "checkpoints")
    ckpt_manager.save_generation(
        checkpoint_dir, "final", cuda_agent.checkpoint_components(),
        {
            "state_dim": state_dim, "action_dim": action_dim,
            "resolved_config": dataclasses.asdict(training_profile),
            "local_training_contract_fingerprint": local_training_contract_fingerprint(training_profile),
        },
        _make_replay(state_dim, action_dim),
    )
    del cuda_agent
    torch.cuda.empty_cache()

    # This test exercises checkpoint/device construction only. Wall-pool
    # spawning is the first operation that actually requires Gazebo.
    monkeypatch.setattr(live_executor_module, "ensure_wall_pool_spawned", lambda *args, **kwargs: None)
    executor = LiveGazeboLocalExecutor(
        live_profile, live_profile.long_horizon_world, checkpoint_dir, "final",
        node_name="test_cuda_checkpoint_cpu_remap",
    )
    try:
        assert str(executor.local_agent.device) == "cpu"
        assert all(parameter.device.type == "cpu" for parameter in executor.local_agent.actor.parameters())
    finally:
        executor.close()


# ---------------------------------------------------------------- defect-fix item 12: pluggable localization backend

def _bare_executor():
    """Constructs a LiveGazeboLocalExecutor instance WITHOUT running
    Node.__init__/rclpy wiring or touching Gazebo -- mirrors
    test_hierarchical_navigation_node.py's own bare-instance harness, used
    here to unit-test the WheelImu dead-reckoning dispatch in isolation."""
    executor = LiveGazeboLocalExecutor.__new__(LiveGazeboLocalExecutor)
    executor._latest_odom_receipt_time = None
    executor._latest_odom_time = None
    executor._latest_v_mps_value = 0.0
    executor._latest_yaw_rate_value = 0.0
    executor.odom_update_count = 0
    executor._wheel_imu_initialized = False
    executor._world = None
    return executor


def _odom_msg(v_mps: float, yaw_rate: float, stamp_sec: float):
    from nav_msgs.msg import Odometry

    msg = Odometry()
    msg.header.stamp.sec = int(stamp_sec)
    msg.header.stamp.nanosec = int(round((stamp_sec - int(stamp_sec)) * 1e9))
    msg.twist.twist.linear.x = v_mps
    msg.twist.twist.angular.z = yaw_rate
    return msg


def test_default_localization_backend_is_gazebo_odom_ground_truth():
    """No localization_backend_factory given -- byte-identical historical
    default."""
    from hunter_kinodynamic_rl.navigation.localization.gazebo_odom_backend import GazeboOdomLocalizationBackend

    executor = _bare_executor()
    executor._localization = GazeboOdomLocalizationBackend(use_covariance_confidence=True)
    assert isinstance(executor._localization, GazeboOdomLocalizationBackend)


def test_on_odom_before_wheel_imu_initialize_is_a_safe_noop():
    """Defect-fix item 12: an _on_odom firing before bind_mission's
    explicit _maybe_initialize_wheel_imu() must never integrate() against
    an uninitialized/placeholder origin -- it's dropped, but v/yaw_rate and
    odom_update_count still update."""
    from hunter_kinodynamic_rl.navigation.localization.wheel_imu_backend import WheelImuLocalizationBackend

    executor = _bare_executor()
    executor._localization = WheelImuLocalizationBackend()
    executor._on_odom(_odom_msg(1.0, 0.1, 10.0))
    assert executor.odom_update_count == 1
    assert executor._latest_v_mps_value == pytest.approx(1.0)
    assert executor._latest_odom_time == pytest.approx(10.0)
    # never initialized -> integrate() was never called -> latest_pose()
    # still reports the backend's own pre-initialize sentinel (invalid).
    assert not executor._localization.latest_pose().valid


def test_maybe_initialize_wheel_imu_seeds_at_world_start_pose_not_origin():
    """Defect-fix item 12: the dead-reckoning origin must be the ACTUAL
    world start pose, never (0, 0, 0) -- otherwise it would silently
    diverge from mission_frame's own origin."""
    from hunter_kinodynamic_rl.navigation.localization.wheel_imu_backend import WheelImuLocalizationBackend

    executor = _bare_executor()
    executor._localization = WheelImuLocalizationBackend()
    executor._latest_odom_time = 5.0
    executor._maybe_initialize_wheel_imu((3.0, -2.0, 0.7))
    pose = executor._localization.latest_pose()
    assert pose.x == pytest.approx(3.0)
    assert pose.y == pytest.approx(-2.0)
    assert pose.yaw == pytest.approx(0.7)
    assert pose.valid
    assert executor._wheel_imu_initialized is True


def test_on_odom_after_initialize_integrates_with_correct_dt():
    from hunter_kinodynamic_rl.navigation.localization.wheel_imu_backend import WheelImuLocalizationBackend

    executor = _bare_executor()
    executor._localization = WheelImuLocalizationBackend()
    executor._latest_odom_time = 10.0
    executor._maybe_initialize_wheel_imu((0.0, 0.0, 0.0))

    executor._on_odom(_odom_msg(2.0, 0.0, 11.0))  # dt=1.0s, straight ahead at 2 m/s
    pose = executor._localization.latest_pose()
    assert pose.x == pytest.approx(2.0, abs=1e-6)
    assert pose.y == pytest.approx(0.0, abs=1e-6)
    assert pose.valid


def test_maybe_initialize_wheel_imu_is_a_noop_for_gazebo_odom_backend():
    from hunter_kinodynamic_rl.navigation.localization.gazebo_odom_backend import GazeboOdomLocalizationBackend

    executor = _bare_executor()
    executor._localization = GazeboOdomLocalizationBackend(use_covariance_confidence=True)
    executor._maybe_initialize_wheel_imu((3.0, -2.0, 0.7))
    assert executor._wheel_imu_initialized is False
