"""Coverage for the Phase 3 wall-segment Gazebo entity pool
(``config/schema.py``'s ``WallSegmentPoolConfig`` and
``env/spawning/wall_segment_spawner.py``) -- mirrors
``test_obstacle_pool.py``'s fake-node approach exactly (same duck-typed
``_FakeNode``/``_FakeClient`` contract, since ``wall_segment_spawner.py``
reuses ``obstacle_pool``'s own ``_ensure_slot_spawned``).

Requires ``ros_gz_interfaces`` (only resolvable after ``colcon build``) --
self-skips cleanly on a bare host checkout, exactly like
``test_obstacle_pool.py``/``test_obstacle_spawner.py`` (the transitive
import chain: ``wall_segment_spawner`` -> ``obstacle_pool`` ->
``obstacle_spawner`` -> ``ros_gz_interfaces.srv``)."""

import math

import pytest

pytest.importorskip("ros_gz_interfaces")

from hunter_kinodynamic_rl.config.schema import ConfigError, WallSegmentPoolConfig  # noqa: E402
from hunter_kinodynamic_rl.env.scenarios.long_horizon_world import WallSegment  # noqa: E402
from hunter_kinodynamic_rl.env.simulation.gazebo_service_wait import GazeboServiceError  # noqa: E402
from hunter_kinodynamic_rl.env.spawning import wall_segment_spawner  # noqa: E402


class _FakeLogger:
    def info(self, msg):
        pass

    def warn(self, msg):
        pass

    def error(self, msg):
        pass


class _Result:
    def __init__(self, success: bool):
        self.success = success


class _FakeFuture:
    def __init__(self, result):
        self._result = result


class _FakeClient:
    def __init__(self, result):
        self._result = result

    def call_async(self, req):
        return _FakeFuture(self._result)


class _FakeNode:
    def __init__(self, service_available: bool = True, await_result=None, pose_ignition_fails: bool = False):
        self._service_available = service_available
        self._await_result = _Result(True) if await_result is None else await_result
        self._gz_call_timeout_sec = 1.0
        self.spawn_calls = []
        self.pose_calls = []
        self._pose_ignition_fails = pose_ignition_fails

    def _wait_for_srv(self, client, name, op):
        return self._service_available

    def _await_future(self, future, timeout, op):
        self.spawn_calls.append(future)
        return self._await_result

    def get_logger(self):
        return _FakeLogger()

    def set_entity_pose_ignition(self, name, x, y, z, qx, qy, qz, qw):
        self.pose_calls.append((name, x, y, z, qz, qw))
        if self._pose_ignition_fails:
            raise GazeboServiceError(f"set_pose[{name}]: entity not found")


def _pool(max_segments=6, classes=(1.0, 2.0, 4.0)):
    cfg = WallSegmentPoolConfig(enabled=True, max_segments=max_segments, length_classes_m=list(classes))
    return wall_segment_spawner.build_wall_pool(cfg, wall_thickness_m=0.2, wall_height_m=1.0,
                                                 world_size_m=16.0, lidar_max_range_m=10.0)


def _segment(x=0.0, y=0.0, length=1.5, yaw=0.0, entity_id="wseg_0", semantic="corridor"):
    return WallSegment(center_x_m=x, center_y_m=y, length_m=length, thickness_m=0.2, height_m=1.0,
                        yaw_rad=yaw, semantic=semantic, entity_id=entity_id)


# --------------------------------------------------------------- schema
def test_wall_segment_pool_config_defaults_are_disabled_and_validate():
    cfg = WallSegmentPoolConfig()
    cfg.validate()
    assert cfg.enabled is False


@pytest.mark.parametrize("kwargs", [
    {"max_segments": -1},
    {"enabled": True, "length_classes_m": []},
    {"length_classes_m": [2.0, 1.0]},  # not ascending
    {"length_classes_m": [0.0]},
    {"parking_margin_m": 0.0},
])
def test_wall_segment_pool_config_rejects_invalid_values(kwargs):
    with pytest.raises(ConfigError):
        WallSegmentPoolConfig(**kwargs).validate()


def test_profile_rejects_wall_pool_enabled_without_long_horizon_world():
    from hunter_kinodynamic_rl.config.schema import Profile, RobotConfig

    profile = Profile(
        name="test",
        robot=RobotConfig(wheelbase_m=0.5, steering_limit_deg=20.0, max_forward_speed_mps=2.0,
                           accel_limit_mps2=1.0, brake_decel_mps2=1.0),
        wall_segment_pool=WallSegmentPoolConfig(enabled=True, length_classes_m=[20.0]),
    )
    with pytest.raises(ConfigError, match="wall_segment_pool.enabled"):
        profile.validate()


def test_profile_rejects_wall_pool_length_classes_too_small_for_worst_case_pitch():
    from hunter_kinodynamic_rl.config.schema import LongHorizonWorldConfig, Profile, RobotConfig

    profile = Profile(
        name="test",
        robot=RobotConfig(wheelbase_m=0.5, steering_limit_deg=20.0, max_forward_speed_mps=2.0,
                           accel_limit_mps2=1.0, brake_decel_mps2=1.0),
        long_horizon_world=LongHorizonWorldConfig(enabled=True, size_m=30.0),
        wall_segment_pool=WallSegmentPoolConfig(enabled=True, length_classes_m=[1.0]),
    )
    with pytest.raises(ConfigError, match="length_classes_m"):
        profile.validate()


def test_wall_segment_pool_config_rejects_zero_max_segments_when_enabled():
    """Code review regression: enabled=True, max_segments=0 previously
    validated cleanly (WallSegmentPoolConfig.validate() only checked
    max_segments >= 0) and only failed loudly much later, at runtime,
    inside activate_walls -- caught here, at config-load time, instead."""
    with pytest.raises(ConfigError, match="max_segments"):
        WallSegmentPoolConfig(enabled=True, max_segments=0, length_classes_m=[20.0]).validate()


def test_wall_segment_pool_config_allows_zero_max_segments_when_disabled():
    WallSegmentPoolConfig(enabled=False, max_segments=0).validate()  # must not raise


def test_profile_rejects_the_exact_zero_max_segments_repro_from_code_review():
    """The exact profile shape code review reported as incorrectly passing
    validate(): wall_segment_pool.enabled=True, max_segments=0,
    length_classes_m=[20.0] against a long_horizon_world large enough that
    length_classes_m alone would have passed the pitch check."""
    from hunter_kinodynamic_rl.config.schema import LongHorizonWorldConfig, Profile, RobotConfig

    profile = Profile(
        name="test",
        robot=RobotConfig(wheelbase_m=0.5, steering_limit_deg=20.0, max_forward_speed_mps=2.0,
                           accel_limit_mps2=1.0, brake_decel_mps2=1.0),
        long_horizon_world=LongHorizonWorldConfig(enabled=True, size_m=30.0),
        wall_segment_pool=WallSegmentPoolConfig(enabled=True, max_segments=0, length_classes_m=[20.0]),
    )
    with pytest.raises(ConfigError, match="max_segments"):
        profile.validate()


def test_profile_rejects_wall_pool_max_segments_too_small_for_worst_case_segment_count():
    """length_classes_m alone covers the worst-case PITCH, but max_segments
    is too small for the worst-case wall segment COUNT this lattice size
    can produce -- must still be rejected."""
    from hunter_kinodynamic_rl.config.schema import LongHorizonWorldConfig, Profile, RobotConfig

    profile = Profile(
        name="test",
        robot=RobotConfig(wheelbase_m=0.5, steering_limit_deg=20.0, max_forward_speed_mps=2.0,
                           accel_limit_mps2=1.0, brake_decel_mps2=1.0),
        long_horizon_world=LongHorizonWorldConfig(enabled=True, size_m=30.0),
        wall_segment_pool=WallSegmentPoolConfig(enabled=True, max_segments=5, length_classes_m=[15.0]),
    )
    with pytest.raises(ConfigError, match="max_segments"):
        profile.validate()


def test_profile_accepts_wall_pool_length_classes_covering_worst_case_pitch():
    from hunter_kinodynamic_rl.config.schema import LongHorizonWorldConfig, Profile, RobotConfig

    profile = Profile(
        name="test",
        robot=RobotConfig(wheelbase_m=0.5, steering_limit_deg=20.0, max_forward_speed_mps=2.0,
                           accel_limit_mps2=1.0, brake_decel_mps2=1.0),
        long_horizon_world=LongHorizonWorldConfig(enabled=True, size_m=30.0),
        wall_segment_pool=WallSegmentPoolConfig(enabled=True, length_classes_m=[15.0]),
    )
    profile.validate()  # must not raise (30/3 = 10 <= 15)


def test_profile_rejects_the_exact_per_class_exhaustion_repro_from_code_review():
    """Code review, round 2: max_segments=100 split evenly across 10
    length classes gives only 10 slots/class -- nowhere near enough for
    one specific class's worst-case need (80, see
    test_long_horizon_generator.py's own per-class bound test) even
    though the grand TOTAL (100) covers every world's actual segment
    count. Round 1's simple total-only check passed this profile;
    activate_walls() then failed at runtime with "no free slot in the
    length class 1.0" for a real generated world."""
    from hunter_kinodynamic_rl.config.schema import LongHorizonWorldConfig, Profile, RobotConfig

    profile = Profile(
        name="test",
        robot=RobotConfig(wheelbase_m=0.65, steering_limit_deg=25.0, max_forward_speed_mps=2.0,
                           accel_limit_mps2=1.0, brake_decel_mps2=1.0),
        long_horizon_world=LongHorizonWorldConfig(enabled=True, size_m=40.0),
        wall_segment_pool=WallSegmentPoolConfig(
            enabled=True, max_segments=110,
            length_classes_m=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 13.4],
        ),
    )
    with pytest.raises(ConfigError, match="max_segments"):
        profile.validate()


def test_a_validated_wall_pool_profile_never_fails_activate_walls_across_many_seeds():
    """End-to-end confirmation that a profile passing Profile.validate()
    (per-class capacity included) genuinely never hits activate_walls'
    "no free slot in the length class" RuntimeError for any generated
    world -- the exact runtime failure code review reproduced."""
    from hunter_kinodynamic_rl.config.schema import LongHorizonWorldConfig, Profile, RobotConfig
    from hunter_kinodynamic_rl.env.scenarios.long_horizon_generator import generate_long_horizon_world

    profile = Profile(
        name="test",
        robot=RobotConfig(wheelbase_m=0.65, steering_limit_deg=25.0, max_forward_speed_mps=2.0,
                           accel_limit_mps2=1.0, brake_decel_mps2=1.0),
        long_horizon_world=LongHorizonWorldConfig(enabled=True, size_m=40.0),
        wall_segment_pool=WallSegmentPoolConfig(
            enabled=True, max_segments=880,
            length_classes_m=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 13.4],
        ),
    )
    profile.validate()  # must not raise

    pool = wall_segment_spawner.build_wall_pool(
        profile.wall_segment_pool, wall_thickness_m=profile.long_horizon_world.wall_thickness_m,
        wall_height_m=profile.long_horizon_world.wall_height_m,
        world_size_m=profile.long_horizon_world.size_m, lidar_max_range_m=10.0,
    )
    for slot in pool.slots:  # skip real ensure_spawned -- ROS/Gazebo unrelated to this check
        slot.spawned = True
    pool.ready = True
    node = _FakeNode()

    for seed in range(20):
        world = generate_long_horizon_world(
            seed=seed, cfg=profile.long_horizon_world, robot_radius_m=0.45,
            min_turning_radius_m=1.2, wheelbase_m=0.65,
        )
        wall_segment_spawner.activate_walls(node, pool, world.wall_segments)  # must not raise


# --------------------------------------------------------------- build_wall_pool
def test_build_wall_pool_parking_distance_clears_lidar_range():
    pool = _pool()
    assert pool.parking_distance_m >= 16.0 / 2.0 + 10.0


def test_pool_slots_are_allocated_across_configured_length_classes():
    pool = _pool(max_segments=6, classes=(1.0, 2.0, 4.0))
    seen = {slot.length_class_m for slot in pool.slots}
    assert seen == {1.0, 2.0, 4.0}
    assert len(pool.slots) == 6


# --------------------------------------------------------------- ensure_spawned
def test_ensure_spawned_spawns_every_slot_exactly_once_across_repeated_calls():
    pool = _pool(max_segments=4)
    node = _FakeNode()
    wall_segment_spawner.ensure_spawned(node, client := _FakeClient(_Result(True)), pool)
    assert len(node.spawn_calls) == 4
    assert pool.ready is True

    wall_segment_spawner.ensure_spawned(node, client, pool)  # second call: no-op
    assert len(node.spawn_calls) == 4


def test_ensure_spawned_raises_on_genuine_spawn_failure():
    pool = _pool(max_segments=2)
    node = _FakeNode(await_result=_Result(False), pose_ignition_fails=True)
    with pytest.raises(GazeboServiceError):
        wall_segment_spawner.ensure_spawned(node, _FakeClient(_Result(False)), pool)
    assert pool.ready is False


# --------------------------------------------------------------- activate_walls
def test_activate_walls_teleports_active_slots_and_parks_the_rest():
    pool = _pool(max_segments=6, classes=(1.0, 2.0, 4.0))
    node = _FakeNode()
    wall_segment_spawner.ensure_spawned(node, _FakeClient(_Result(True)), pool)
    segments = [_segment(x=1.0, y=2.0, length=0.9, entity_id="a"), _segment(x=-1.0, y=0.5, length=1.8, entity_id="b")]
    names = wall_segment_spawner.activate_walls(node, pool, segments)
    assert len(names) == 2
    active = [s for s in pool.slots if s.active]
    assert len(active) == 2
    parked = [s for s in pool.slots if not s.active]
    assert len(parked) == 4
    # every slot (active or parked) had a pose call this activation
    assert len(node.pose_calls) >= 6


def test_activate_walls_snaps_length_up_to_nearest_class():
    pool = _pool(max_segments=3, classes=(1.0, 2.0, 4.0))
    node = _FakeNode()
    wall_segment_spawner.ensure_spawned(node, _FakeClient(_Result(True)), pool)
    seg = _segment(length=1.3)  # snaps up to 2.0
    wall_segment_spawner.activate_walls(node, pool, [seg])
    active = [s for s in pool.slots if s.active]
    assert len(active) == 1
    assert active[0].length_class_m == 2.0


def test_activate_walls_orientation_quaternion_matches_yaw():
    pool = _pool(max_segments=2, classes=(1.0, 2.0))
    node = _FakeNode()
    wall_segment_spawner.ensure_spawned(node, _FakeClient(_Result(True)), pool)
    yaw = math.pi / 2.0
    seg = _segment(length=0.8, yaw=yaw)
    wall_segment_spawner.activate_walls(node, pool, [seg])
    # activate_walls teleports every ACTIVE slot before parking the unused
    # ones -- with a single segment given, its own pose call is always the
    # FIRST recorded call (the later ones are the identity-quaternion park
    # calls for the remaining, unused slots).
    name, x, y, z, qz, qw = node.pose_calls[0]
    assert abs(qz - math.sin(yaw / 2.0)) < 1e-9
    assert abs(qw - math.cos(yaw / 2.0)) < 1e-9


def test_activate_walls_raises_when_more_segments_than_slots():
    pool = _pool(max_segments=2, classes=(2.0,))
    node = _FakeNode()
    wall_segment_spawner.ensure_spawned(node, _FakeClient(_Result(True)), pool)
    segments = [_segment(entity_id=f"s{i}") for i in range(3)]
    with pytest.raises(RuntimeError, match="only"):
        wall_segment_spawner.activate_walls(node, pool, segments)


def test_activate_walls_raises_when_class_exhausted_never_escalates_silently():
    pool = _pool(max_segments=2, classes=(1.0, 2.0))  # one slot per class
    node = _FakeNode()
    wall_segment_spawner.ensure_spawned(node, _FakeClient(_Result(True)), pool)
    # Two segments both needing the SAME (smaller) class -- only one slot
    # of that class exists, so this must raise rather than silently placing
    # the second one in the larger class.
    segments = [_segment(length=0.5, entity_id="a"), _segment(length=0.5, entity_id="b")]
    with pytest.raises(RuntimeError, match="no free slot"):
        wall_segment_spawner.activate_walls(node, pool, segments)


def test_activate_walls_raises_when_length_exceeds_every_class():
    pool = _pool(max_segments=2, classes=(1.0, 2.0))
    node = _FakeNode()
    wall_segment_spawner.ensure_spawned(node, _FakeClient(_Result(True)), pool)
    with pytest.raises(RuntimeError, match="exceeds every configured"):
        wall_segment_spawner.activate_walls(node, pool, [_segment(length=10.0)])


def test_activate_walls_is_idempotent_across_episodes_reusing_the_same_slots():
    pool = _pool(max_segments=4, classes=(1.0, 2.0))
    node = _FakeNode()
    wall_segment_spawner.ensure_spawned(node, _FakeClient(_Result(True)), pool)
    spawn_calls_after_setup = len(node.spawn_calls)

    wall_segment_spawner.activate_walls(node, pool, [_segment(length=0.9, entity_id="ep1_a")])
    wall_segment_spawner.activate_walls(node, pool, [_segment(length=1.9, entity_id="ep2_a")])
    # No NEW SpawnEntity calls happened across episode re-activation --
    # only SetEntityPose (teleport/park) calls, the whole point of the pool.
    assert len(node.spawn_calls) == spawn_calls_after_setup
