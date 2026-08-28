"""Regression test for the ROS import-boundary bug found in code review:
``env/spawning/wall_segment_spawner.py`` previously imported
``obstacle_pool._ensure_slot_spawned``/``snap_up_to_class`` at MODULE
SCOPE, which transitively pulled in
``obstacle_pool -> obstacle_spawner -> ros_gz_interfaces.srv`` -- making
``import hunter_kinodynamic_rl.env.spawning.wall_segment_spawner`` itself
fail with ``ModuleNotFoundError`` on a bare, ROS-free host, contradicting
this package's "pure-Python/no ROS import at module scope" convention for
every other Phase 3 module.

Deliberately a SEPARATE file from ``test_wall_segment_spawner.py`` -- that
file's own module-level ``pytest.importorskip("ros_gz_interfaces")`` would
skip this whole check on a bare host (exactly the environment this test
needs to exercise) before it ever ran. This file has no such guard: it
must run (and pass) regardless of whether ``ros_gz_interfaces`` happens to
be installed, by simulating "not installed" via ``sys.modules`` regardless
of the real host.
"""

import importlib
import sys


def test_wall_segment_spawner_imports_without_ros_gz_interfaces(monkeypatch):
    # sys.modules[name] = None is the standard trick to force ANY import of
    # `name` to raise ImportError, regardless of whether it is actually
    # installed on this host -- so this test's guarantee holds even when
    # run inside the Docker image where ros_gz_interfaces IS resolvable.
    monkeypatch.setitem(sys.modules, "ros_gz_interfaces", None)
    monkeypatch.setitem(sys.modules, "ros_gz_interfaces.srv", None)
    # Drop any already-imported copy so Python re-executes the module body
    # under the simulated environment rather than returning a cached
    # module that was imported earlier (e.g. by test_wall_segment_spawner.py
    # in the same pytest session).
    for name in list(sys.modules):
        if name == "hunter_kinodynamic_rl.env.spawning.wall_segment_spawner" or \
                name.startswith("hunter_kinodynamic_rl.env.spawning.wall_segment_spawner."):
            monkeypatch.delitem(sys.modules, name, raising=False)

    module = importlib.import_module("hunter_kinodynamic_rl.env.spawning.wall_segment_spawner")

    assert hasattr(module, "build_wall_pool")
    assert hasattr(module, "WallSegmentPool")
    # And the pure functions are actually CALLABLE under this simulated
    # environment, not merely importable -- ensure_spawned/activate_walls
    # are the ONLY functions allowed to need ros_gz_interfaces (lazily).
    assert module._snap_length_up_to_class(1.3, [1.0, 2.0, 4.0]) == 2.0


def test_ensure_spawned_still_needs_ros_gz_interfaces_when_actually_called(monkeypatch):
    """The flip side of the fix: ensure_spawned's LAZY import must still
    fail loudly (never silently no-op) when ros_gz_interfaces is genuinely
    unavailable and the function is actually invoked -- this is a targeted
    fallback, not a way to make the Gazebo-dependent half of this module
    secretly optional."""
    monkeypatch.setitem(sys.modules, "ros_gz_interfaces", None)
    monkeypatch.setitem(sys.modules, "ros_gz_interfaces.srv", None)
    for name in list(sys.modules):
        if name.startswith("hunter_kinodynamic_rl.env.spawning"):
            monkeypatch.delitem(sys.modules, name, raising=False)

    module = importlib.import_module("hunter_kinodynamic_rl.env.spawning.wall_segment_spawner")
    from hunter_kinodynamic_rl.config.schema import WallSegmentPoolConfig

    cfg = WallSegmentPoolConfig(enabled=True, max_segments=2, length_classes_m=[1.0])
    pool = module.build_wall_pool(cfg, wall_thickness_m=0.2, wall_height_m=1.0,
                                   world_size_m=16.0, lidar_max_range_m=10.0)
    try:
        module.ensure_spawned(node=None, spawn_client=None, pool=pool)
        raise AssertionError("expected ensure_spawned to fail without ros_gz_interfaces")
    except (ImportError, ModuleNotFoundError):
        pass
