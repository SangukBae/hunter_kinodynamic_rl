"""Phase 5/6 hierarchical_navigation_node.py real-hardware safety layer
(plan section 10.7/25): dry-run/replay actuation blocking, E-stop latch,
watchdog command timeout, and policy inference timeout -- mirrors
tests/test_real_policy_node.py's own coverage of the identical pattern in
real_policy_node.py, applied to this node's Global+Local pipeline."""

import time

import numpy as np
import pytest

pytest.importorskip("rclpy")
pytest.importorskip("torch")

from hunter_kinodynamic_rl.config.loader import load_profile  # noqa: E402
from hunter_kinodynamic_rl.env.safety.action_guard import STOP_COMMAND, SafetyLimits  # noqa: E402
from hunter_kinodynamic_rl.nodes.hierarchical_navigation_node import HierarchicalNavigationNode  # noqa: E402
from hunter_kinodynamic_rl.trajectory.pure_pursuit_adapter import VehicleCommand  # noqa: E402


class _NullLogger:
    def info(self, *a, **k): pass
    def warn(self, *a, **k): pass
    def error(self, *a, **k): pass


class _FakePublisher:
    def __init__(self):
        self.published = []

    def publish(self, msg):
        self.published.append(msg)


def _bare_node(dry_run: bool = False) -> HierarchicalNavigationNode:
    profile = load_profile("hierarchical_phase5_b")
    node = HierarchicalNavigationNode.__new__(HierarchicalNavigationNode)
    node.profile = profile
    node.dry_run = dry_run
    node.replay_mode = False
    node._estopped = False
    node.get_logger = lambda: _NullLogger()
    node._cmd_pub = _FakePublisher()
    node._latest_scan = None
    node._latest_odom = None
    node._last_command_time = None
    node._inference_thread = None
    node._policy_inference_timeout_sec = 0.2
    node._safety_limits = SafetyLimits(
        max_sensor_age_sec=profile.runtime.sensor_freshness_timeout_sec,
        max_odom_age_sec=profile.runtime.sensor_freshness_timeout_sec,
        max_command_age_sec=profile.runtime.watchdog_command_timeout_sec,
        min_obstacle_stop_distance_m=profile.risk.min_safe_clearance_m,
    )
    return node


def test_publish_reaches_cmd_vel_when_not_dry_run():
    node = _bare_node(dry_run=False)
    node._publish(VehicleCommand(speed_mps=1.0, steering_rad=0.1))
    assert len(node._cmd_pub.published) == 1


def test_publish_never_reaches_cmd_vel_when_dry_run():
    node = _bare_node(dry_run=True)
    node._publish(VehicleCommand(speed_mps=2.0, steering_rad=-0.2))
    assert node._cmd_pub.published == []


def test_replay_mode_without_dry_run_is_rejected_by_construction_contract():
    """This module's __init__ raises SystemExit when replay_mode=true and
    dry_run=false BEFORE any ROS wiring happens -- verified here at the
    source level (the exact same guard real_policy_node.py's own
    constructor uses) since exercising __init__ itself needs a live rclpy
    node/parameter context this test suite does not stand up."""
    import inspect

    from hunter_kinodynamic_rl.nodes import hierarchical_navigation_node as module
    source = inspect.getsource(module.HierarchicalNavigationNode.__init__)
    assert "replay_mode and not self.dry_run" in source
    assert "SystemExit" in source


def test_on_estop_latches_true_and_clears_on_false():
    node = _bare_node(dry_run=False)

    class _Msg:
        data = True

    node._on_estop(_Msg())
    assert node._estopped is True
    _Msg.data = False
    node._on_estop(_Msg())
    assert node._estopped is False


def test_estopped_control_tick_publishes_stop_before_touching_sensors():
    node = _bare_node(dry_run=False)
    node._estopped = True
    node.mission_frame = None  # would raise .initialized if ever touched
    node._on_control_tick()
    assert node._cmd_pub.published[0].linear.x == 0.0
    assert node._cmd_pub.published[0].angular.z == 0.0


def test_infer_local_action_with_timeout_returns_none_on_hang():
    node = _bare_node(dry_run=False)

    class _HangingAgent:
        @staticmethod
        def select_action(observation, deterministic=True):
            time.sleep(2.0)
            return np.zeros(3, dtype=np.float32)

    node.local_agent = _HangingAgent()
    result = node._infer_local_action_with_timeout(np.zeros(4, dtype=np.float32))
    assert result is None


def test_infer_local_action_with_timeout_returns_action_on_success():
    node = _bare_node(dry_run=False)

    class _FastAgent:
        @staticmethod
        def select_action(observation, deterministic=True):
            return np.array([0.1, 0.2, 0.3], dtype=np.float32)

    node.local_agent = _FastAgent()
    result = node._infer_local_action_with_timeout(np.zeros(4, dtype=np.float32))
    assert result is not None
    np.testing.assert_allclose(result, [0.1, 0.2, 0.3])


def test_watchdog_thread_respects_dry_run():
    node = _bare_node(dry_run=True)
    import dataclasses
    node.profile = dataclasses.replace(
        node.profile,
        runtime=dataclasses.replace(
            node.profile.runtime, watchdog_command_timeout_sec=0.05, real_policy_watchdog_period_sec=0.02,
        ),
    )
    node._last_command_time = time.monotonic()
    node._start_watchdog_thread()
    time.sleep(0.3)
    node._watchdog_stop_event.set()
    node._watchdog_thread.join(timeout=1.0)
    assert node._cmd_pub.published == []


def test_watchdog_thread_publishes_stop_after_command_staleness_when_not_dry_run():
    node = _bare_node(dry_run=False)
    import dataclasses
    node.profile = dataclasses.replace(
        node.profile,
        runtime=dataclasses.replace(
            node.profile.runtime, watchdog_command_timeout_sec=0.05, real_policy_watchdog_period_sec=0.02,
        ),
    )
    node._last_command_time = time.monotonic() - 1.0  # already stale
    node._start_watchdog_thread()
    time.sleep(0.15)
    node._watchdog_stop_event.set()
    node._watchdog_thread.join(timeout=1.0)
    assert len(node._cmd_pub.published) >= 1
    assert node._cmd_pub.published[0].linear.x == 0.0
