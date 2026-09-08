"""Regression tests for env/spawning/obstacle_spawner.py's P0-3 fail-fast
fix: spawn/delete failures (timeout, success=false, or the service itself
being unavailable) must RAISE GazeboServiceError, never silently continue
with an incomplete/contaminated obstacle set -- see that module's docstring
("obstacle placement must be CONFIRMED before the episode is considered
ready") and environment_node.py's _on_reset, which now wraps
_clear_previous_obstacles() in its own try/except to convert this into the
same RuntimeError("reset failed: ...") every other Gazebo-service failure
in that method already produces.

Requires ros_gz_interfaces (only resolvable after colcon build) -- self-skips
cleanly on a bare host checkout, mirroring this package's other
ROS-dependent test files.
"""

import xml.etree.ElementTree as ET

import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("ros_gz_interfaces")

from hunter_kinodynamic_rl.env.scenarios.procedural_generator import StaticObstacle  # noqa: E402
from hunter_kinodynamic_rl.env.simulation.gazebo_service_wait import GazeboServiceError  # noqa: E402
from hunter_kinodynamic_rl.env.spawning import obstacle_spawner  # noqa: E402


class _FakeLogger:
    def warn(self, msg):
        pass

    def error(self, msg):
        pass


class _FakeFuture:
    def __init__(self, result):
        self._result = result


class _FakeClient:
    def __init__(self, result):
        self._result = result
        self.requests = []

    def call_async(self, req):
        self.requests.append(req)
        return _FakeFuture(self._result)


class _Result:
    def __init__(self, success: bool):
        self.success = success


class _FakeNode:
    """Duck-types the subset of GazeboRuntimeMixin's interface
    obstacle_spawner.py actually calls: _wait_for_srv, _await_future,
    get_logger, _gz_call_timeout_sec."""

    def __init__(self, service_available: bool = True, await_result=None):
        self._service_available = service_available
        self._await_result = await_result
        self._gz_call_timeout_sec = 1.0

    def _wait_for_srv(self, client, name, op):
        return self._service_available

    def _await_future(self, future, timeout, op):
        return self._await_result

    def get_logger(self):
        return _FakeLogger()


def test_spawn_one_raises_when_service_never_available():
    node = _FakeNode(service_available=False)
    client = _FakeClient(_Result(True))
    with pytest.raises(GazeboServiceError, match="unavailable"):
        obstacle_spawner._spawn_one(node, client, "m", "<sdf/>", 0.0, 0.0, 0.0, 0.0)


def test_spawn_one_raises_on_future_timeout():
    node = _FakeNode(service_available=True, await_result=None)
    client = _FakeClient(_Result(True))
    with pytest.raises(GazeboServiceError, match="no response"):
        obstacle_spawner._spawn_one(node, client, "m", "<sdf/>", 0.0, 0.0, 0.0, 0.0)


def test_spawn_one_raises_on_success_false():
    node = _FakeNode(service_available=True, await_result=_Result(False))
    client = _FakeClient(_Result(False))
    with pytest.raises(GazeboServiceError, match="success=false"):
        obstacle_spawner._spawn_one(node, client, "m", "<sdf/>", 0.0, 0.0, 0.0, 0.0)


def test_spawn_one_returns_true_on_genuine_success():
    node = _FakeNode(service_available=True, await_result=_Result(True))
    client = _FakeClient(_Result(True))
    assert obstacle_spawner._spawn_one(node, client, "m", "<sdf/>", 0.0, 0.0, 0.0, 0.0) is True


def test_l_shape_sdf_has_valid_six_component_poses():
    """Every SDF pose is ``x y z roll pitch yaw``; Gazebo rejects five values."""
    root = ET.fromstring(obstacle_spawner._l_shape_sdf("l-shape", 0.8, 0.5))
    poses = root.findall(".//pose")
    assert len(poses) == 4
    assert all(len((pose.text or "").split()) == 6 for pose in poses)


def test_spawn_static_obstacles_raises_on_the_first_failure_not_just_undercounting():
    """The core P0-3 regression: a partial spawn failure must propagate as
    an exception -- previously it just silently returned a smaller
    "spawned" count with no error anywhere, leaving the episode running
    with FEWER obstacles than the seed/scenario actually specifies."""
    node = _FakeNode(service_available=True, await_result=_Result(False))
    client = _FakeClient(_Result(False))
    obstacles = [StaticObstacle(x=1.0, y=0.0, radius=0.3), StaticObstacle(x=2.0, y=0.0, radius=0.3)]
    with pytest.raises(GazeboServiceError):
        obstacle_spawner.spawn_static_obstacles(node, client, obstacles, catalog=[], rng=None)


def test_delete_entities_raises_when_service_unavailable():
    node = _FakeNode(service_available=False)
    client = _FakeClient(_Result(True))
    with pytest.raises(GazeboServiceError, match="unavailable"):
        obstacle_spawner.delete_entities(node, client, ["hkrl_static_0"])


def test_delete_entities_raises_on_a_failed_individual_delete():
    node = _FakeNode(service_available=True, await_result=_Result(False))
    client = _FakeClient(_Result(False))
    with pytest.raises(GazeboServiceError, match="success=false"):
        obstacle_spawner.delete_entities(node, client, ["hkrl_static_0"])


def test_delete_entities_no_op_on_an_empty_name_list():
    node = _FakeNode(service_available=False)  # would raise if it tried to use the service at all
    client = _FakeClient(_Result(True))
    obstacle_spawner.delete_entities(node, client, [])  # must not raise -- nothing to delete


def test_delete_entities_succeeds_with_genuine_confirmations():
    from ros_gz_interfaces.msg import Entity as GzEntity

    node = _FakeNode(service_available=True, await_result=_Result(True))
    client = _FakeClient(_Result(True))
    obstacle_spawner.delete_entities(node, client, ["hkrl_static_0", "hkrl_static_1"])  # must not raise
    assert [request.entity.type for request in client.requests] == [GzEntity.MODEL, GzEntity.MODEL]
