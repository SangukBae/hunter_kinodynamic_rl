"""Coverage for the deterministic Gazebo obstacle entity pool (drl_agent ->
hunter_kinodynamic_rl requirement 2): ``config/schema.py``'s
``ObstaclePoolConfig`` and ``env/spawning/obstacle_pool.py``.

Requires ros_gz_interfaces (only resolvable after colcon build) -- self-skips
cleanly on a bare host checkout, mirroring test_obstacle_spawner.py.
"""

import pytest

pytest.importorskip("ros_gz_interfaces")

from hunter_kinodynamic_rl.config.schema import ConfigError, ObstaclePoolConfig  # noqa: E402
from hunter_kinodynamic_rl.env.scenarios.procedural_generator import (  # noqa: E402
    DynamicObstacleSpec, StaticObstacle,
)
from hunter_kinodynamic_rl.env.simulation.gazebo_service_wait import GazeboServiceError  # noqa: E402
from hunter_kinodynamic_rl.env.spawning import obstacle_pool  # noqa: E402


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
    """Duck-types the subset of GazeboRuntimeMixin's interface
    obstacle_pool.py actually calls: _wait_for_srv, _await_future,
    get_logger, _gz_call_timeout_sec, set_entity_pose_ignition."""

    def __init__(self, service_available: bool = True, await_result=None, pose_ignition_fails: bool = False):
        self._service_available = service_available
        self._await_result = _Result(True) if await_result is None else await_result
        self._gz_call_timeout_sec = 1.0
        self.spawn_calls = []
        self.pose_calls = []
        # requirement 4, round 2: models a GENUINELY absent entity -- when
        # True, set_entity_pose_ignition (the recovery probe
        # _ensure_slot_spawned falls back to on a spawn failure) also
        # fails, exactly like real Gazebo's SetEntityPose would for a name
        # it has never actually created. False (the default) matches every
        # OTHER test in this module, where every slot really does exist.
        self._pose_ignition_fails = pose_ignition_fails

    def _wait_for_srv(self, client, name, op):
        return self._service_available

    def _await_future(self, future, timeout, op):
        self.spawn_calls.append(future)
        return self._await_result

    def get_logger(self):
        return _FakeLogger()

    def set_entity_pose_ignition(self, name, x, y, z, qx, qy, qz, qw):
        self.pose_calls.append((name, x, y, z))
        if self._pose_ignition_fails:
            raise GazeboServiceError(f"set_pose[{name}]: entity not found")


def _pool(max_static=4, max_dynamic=2, classes=(0.2, 0.35, 0.5)):
    cfg = ObstaclePoolConfig(enabled=True, max_static=max_static, max_dynamic=max_dynamic,
                              static_size_classes_m=list(classes))
    return obstacle_pool.build_pool(cfg, world_size_m=12.0, lidar_max_range_m=10.0)


# --------------------------------------------------------------- schema
def test_obstacle_pool_config_defaults_are_disabled_and_validate():
    cfg = ObstaclePoolConfig()
    cfg.validate()
    assert cfg.enabled is False


@pytest.mark.parametrize("kwargs", [
    {"max_static": -1},
    {"max_dynamic": -1},
    {"enabled": True, "static_size_classes_m": []},
    {"static_size_classes_m": [0.5, 0.2]},  # not ascending
    {"static_size_classes_m": [0.0]},
    {"parking_margin_m": 0.0},
])
def test_obstacle_pool_config_rejects_invalid_values(kwargs):
    with pytest.raises(ConfigError):
        ObstaclePoolConfig(**kwargs).validate()


# ------------------------------------------------ cross-section (Profile) validation
def test_profile_rejects_pool_capacity_that_can_exhaust_the_largest_size_class():
    """Live-Docker-verification regression: max_static >= scenario.max_obstacles
    alone is NOT sufficient -- a profile whose 4 static obstacles could all
    happen to draw a radius needing the SAME (largest) class must have
    enough slots in EVERY class, not just in total, or activate_static can
    raise mid-run on a perfectly ordinary episode (see this module's
    test_activate_static_raises_when_pool_capacity_is_insufficient for the
    unit-level version of the same failure)."""
    from hunter_kinodynamic_rl.config.schema import Profile, RobotConfig, ScenarioConfig

    profile = Profile(
        name="test",
        robot=RobotConfig(wheelbase_m=0.5, steering_limit_deg=20.0, max_forward_speed_mps=2.0,
                           accel_limit_mps2=1.0, brake_decel_mps2=1.0),
        scenario=ScenarioConfig(max_obstacles=4),
        obstacle_pool=ObstaclePoolConfig(enabled=True, max_static=8, static_size_classes_m=[0.2, 0.35, 0.5]),
    )
    with pytest.raises(ConfigError, match="obstacle_pool.max_static"):
        profile.validate()


def test_profile_accepts_pool_capacity_that_covers_every_class_worst_case():
    from hunter_kinodynamic_rl.config.schema import Profile, RobotConfig, ScenarioConfig

    profile = Profile(
        name="test",
        robot=RobotConfig(wheelbase_m=0.5, steering_limit_deg=20.0, max_forward_speed_mps=2.0,
                           accel_limit_mps2=1.0, brake_decel_mps2=1.0),
        scenario=ScenarioConfig(max_obstacles=4),
        obstacle_pool=ObstaclePoolConfig(enabled=True, max_static=12, static_size_classes_m=[0.2, 0.35, 0.5]),
    )
    profile.validate()  # must not raise


def test_profile_skips_pool_capacity_check_for_a_fixed_benchmark_profile():
    """A fixed benchmark (evaluation.benchmark set) NEVER draws from
    scenario.max_obstacles -- environment_node.py's _spawn_scenario_obstacles
    always parks every pool slot (activate_static([])/activate_dynamic([]))
    for a fixed-benchmark episode instead (see this module's own module
    docstring). So checking pool capacity against scenario.max_obstacles for
    a benchmark profile is meaningless, and previously broke evaluating ANY
    pool-enabled checkpoint (whose own training-time obstacle_pool sizing
    has nothing to do with the requested evaluation profile's
    scenario.max_obstacles) through nodes/evaluation_node.py's
    build_effective_profile, which layers the checkpoint's obstacle_pool
    together with the eval profile's scenario section into one Profile
    before calling validate()."""
    from hunter_kinodynamic_rl.config.schema import EvaluationConfig, Profile, RobotConfig, ScenarioConfig

    profile = Profile(
        name="test",
        robot=RobotConfig(wheelbase_m=0.5, steering_limit_deg=20.0, max_forward_speed_mps=2.0,
                           accel_limit_mps2=1.0, brake_decel_mps2=1.0),
        scenario=ScenarioConfig(max_obstacles=8),
        obstacle_pool=ObstaclePoolConfig(enabled=True, max_static=6, static_size_classes_m=[0.2, 0.35, 0.5]),
        evaluation=EvaluationConfig(benchmark="id"),
    )
    profile.validate()  # must not raise despite max_static (6) < max_obstacles (8)


def test_profile_still_rejects_insufficient_pool_capacity_for_a_procedural_profile():
    """The benchmark bypass above must NOT weaken the check for an ordinary
    (non-benchmark) procedural training/evaluation profile -- only
    evaluation.benchmark being set skips it."""
    from hunter_kinodynamic_rl.config.schema import Profile, RobotConfig, ScenarioConfig

    profile = Profile(
        name="test",
        robot=RobotConfig(wheelbase_m=0.5, steering_limit_deg=20.0, max_forward_speed_mps=2.0,
                           accel_limit_mps2=1.0, brake_decel_mps2=1.0),
        scenario=ScenarioConfig(max_obstacles=8),
        obstacle_pool=ObstaclePoolConfig(enabled=True, max_static=6, static_size_classes_m=[0.2, 0.35, 0.5]),
    )
    with pytest.raises(ConfigError, match="obstacle_pool.max_static"):
        profile.validate()


def test_activate_static_never_raises_when_every_obstacle_needs_the_largest_class():
    """The worst case Profile.validate() above now guarantees can't happen
    at runtime for a validated profile: every one of max_obstacles static
    obstacles draws a radius needing the LARGEST class."""
    pool = _pool(max_static=12, classes=(0.2, 0.35, 0.5))
    node = _FakeNode()
    obstacle_pool.ensure_spawned(node, _FakeClient(_Result(True)), pool)
    obstacles = [StaticObstacle(x=float(i), y=0.0, radius=0.5) for i in range(4)]
    names = obstacle_pool.activate_static(node, pool, obstacles)  # must not raise
    assert len(names) == 4


# --------------------------------------------------------------- build_pool
def test_build_pool_parking_distance_clears_lidar_range_from_anywhere_in_the_arena():
    pool = _pool()
    # Worst case: robot at one edge of a 12m world (half=6), marker parked
    # just past the OPPOSITE edge -- straight-line distance must exceed
    # lidar_max_range_m (10.0) with the configured margin (5.0 default) to
    # spare, i.e. >= 6 + 6 + 10 - (already inside the world) ... concretely
    # parking_distance_m itself (measured from the origin) must clear
    # half-extent + lidar_max_range_m.
    assert pool.parking_distance_m >= 12.0 / 2.0 + 10.0


def test_pool_slots_are_allocated_across_configured_size_classes():
    pool = _pool(max_static=6, classes=(0.2, 0.35, 0.5))
    seen_classes = {slot.size_class_m for slot in pool.static_slots}
    assert seen_classes == {0.2, 0.35, 0.5}
    assert len(pool.static_slots) == 6
    assert len(pool.dynamic_slots) == 2


# --------------------------------------------------------------- ensure_spawned
def test_ensure_spawned_spawns_every_slot_exactly_once_across_repeated_calls():
    """The core requirement 2 regression: repeated /reset cycles must NOT
    re-spawn pool entities -- ensure_spawned is idempotent."""
    pool = _pool(max_static=3, max_dynamic=2)
    node = _FakeNode()
    obstacle_pool.ensure_spawned(node, client := _FakeClient(_Result(True)), pool)
    assert len(node.spawn_calls) == 5  # 3 static + 2 dynamic
    assert pool.ready is True

    obstacle_pool.ensure_spawned(node, client, pool)  # second call: pure no-op
    assert len(node.spawn_calls) == 5  # unchanged -- nothing re-spawned


def test_ensure_spawned_raises_on_spawn_failure_never_silently_continues():
    """pose_ignition_fails=True: the entity genuinely never got created, so
    _ensure_slot_spawned's recovery probe (attempted after the initial
    spawn failure) must ALSO fail here, exactly like real Gazebo's
    SetEntityPose would for a name it never actually spawned -- otherwise
    this would incorrectly look like the ambiguous "response lost" case
    and get silently marked spawned."""
    pool = _pool(max_static=2, max_dynamic=0)
    node = _FakeNode(await_result=_Result(False), pose_ignition_fails=True)
    with pytest.raises(GazeboServiceError):
        obstacle_pool.ensure_spawned(node, _FakeClient(_Result(False)), pool)
    assert pool.ready is False


class _FlakyFuture:
    def __init__(self, result):
        self._result = result


class _FlakyClient:
    """Fails every SpawnEntity call whose model_name is in `fail_names`,
    succeeds otherwise -- lets a test simulate a partial ensure_spawned
    failure (some slots really did spawn in Gazebo, one call failed) and
    then a recovering retry."""

    def __init__(self, fail_names):
        self.fail_names = set(fail_names)
        self.calls = []

    def call_async(self, req):
        name = req.entity_factory.name
        self.calls.append(name)
        return _FlakyFuture(_Result(name not in self.fail_names))


class _FlakyNode(_FakeNode):
    def _await_future(self, future, timeout, op):
        self.spawn_calls.append(future)
        if not future._result.success:
            raise GazeboServiceError(op)
        return future._result


def test_ensure_spawned_partial_failure_then_retry_never_respawns_already_spawned_slots():
    """requirement 4: a slot that already spawned successfully before a
    LATER slot's SpawnEntity call failed must NEVER be re-spawned on a
    retry -- only the slot(s) that never got a confirmed success are
    (re)attempted."""
    pool = _pool(max_static=3, max_dynamic=0)
    failing_name = pool.static_slots[2].name  # the 3rd (last) static slot
    client = _FlakyClient(fail_names={failing_name})
    # pose_ignition_fails=True: the failing slot genuinely never got
    # created, so the recovery probe must also fail for it -- see
    # test_ensure_spawned_raises_on_spawn_failure_never_silently_continues's
    # own comment.
    node = _FlakyNode(pose_ignition_fails=True)

    with pytest.raises(GazeboServiceError):
        obstacle_pool.ensure_spawned(node, client, pool)
    assert pool.ready is False
    assert pool.static_slots[0].spawned is True
    assert pool.static_slots[1].spawned is True
    assert pool.static_slots[2].spawned is False
    assert client.calls == [s.name for s in pool.static_slots]  # all 3 attempted once

    # Fix the flake (the 3rd slot now spawns fine) and retry.
    client.fail_names.clear()
    obstacle_pool.ensure_spawned(node, client, pool)
    assert pool.ready is True
    assert all(s.spawned for s in pool.static_slots)
    # The retry must have (re)issued SpawnEntity ONLY for the still-unspawned
    # 3rd slot -- never re-spawning the first two, which already succeeded.
    assert client.calls == [s.name for s in pool.static_slots] + [failing_name]


class _TimeoutThenExistsNode(_FakeNode):
    """Simulates a SpawnEntity call that times out client-side (no response
    ever arrives) even though Gazebo actually created the entity anyway --
    the AMBIGUOUS case a bare _spawn_one failure can't distinguish from a
    genuine absence."""

    def _await_future(self, future, timeout, op):
        self.spawn_calls.append(future)
        return None  # never resolves within budget -> _spawn_one raises


def test_ensure_spawned_recovers_when_spawn_response_is_lost_but_entity_already_exists():
    """requirement 4, round 2: a SpawnEntity call that times out (no
    response) does NOT prove the entity is absent -- Gazebo may have
    created it anyway. ensure_spawned must probe via SetEntityPose (which
    reports success=false for a name Gazebo doesn't recognize) and mark
    the slot spawned=True on a successful probe, rather than leaving it
    stuck retrying SpawnEntity against an entity that already exists
    forever."""
    pool = _pool(max_static=2, max_dynamic=1)
    node = _TimeoutThenExistsNode()
    obstacle_pool.ensure_spawned(node, _FakeClient(_Result(True)), pool)
    assert pool.ready is True
    assert all(s.spawned for s in pool.static_slots)
    assert all(s.spawned for s in pool.dynamic_slots)
    # Each slot's existence was confirmed via a SetEntityPose probe at its
    # own parking position -- never a second SpawnEntity call for it.
    probed_names = {name for name, x, y, z in node.pose_calls}
    all_names = {s.name for s in pool.static_slots} | {s.name for s in pool.dynamic_slots}
    assert probed_names == all_names


class _AlwaysAbsentNode(_FakeNode):
    """Both SpawnEntity and the recovery-probe SetEntityPose fail --
    simulates a genuinely absent entity (the common case: the entity
    really was never created)."""

    def _await_future(self, future, timeout, op):
        self.spawn_calls.append(future)
        return None

    def set_entity_pose_ignition(self, name, x, y, z, qx, qy, qz, qw):
        self.pose_calls.append((name, x, y, z))
        raise GazeboServiceError(f"set_pose[{name}]: entity not found")


def test_ensure_spawned_reraises_original_error_when_entity_genuinely_absent():
    """The recovery probe must never MASK a real failure -- if the entity
    truly doesn't exist (probe also fails), the original spawn error
    propagates and the slot stays unspawned for a future retry, exactly
    like before this recovery path existed."""
    pool = _pool(max_static=1, max_dynamic=0)
    node = _AlwaysAbsentNode()
    with pytest.raises(GazeboServiceError, match="no response"):
        obstacle_pool.ensure_spawned(node, _FakeClient(_Result(True)), pool)
    assert pool.ready is False
    assert pool.static_slots[0].spawned is False


# --------------------------------------------------------------- activate_static
def test_activate_static_teleports_needed_slots_and_parks_the_rest():
    pool = _pool(max_static=4, classes=(0.5,))
    node = _FakeNode()
    obstacle_pool.ensure_spawned(node, _FakeClient(_Result(True)), pool)
    node.pose_calls.clear()

    obstacles = [StaticObstacle(x=1.0, y=2.0, radius=0.5), StaticObstacle(x=-1.0, y=-2.0, radius=0.5)]
    names = obstacle_pool.activate_static(node, pool, obstacles)
    assert len(names) == 2
    assert len(set(names)) == 2  # distinct slots

    active_positions = {(n, (x, y)) for (n, x, y, z) in node.pose_calls if n in names}
    assert (names[0], (1.0, 2.0)) in active_positions
    assert (names[1], (-1.0, -2.0)) in active_positions

    parked_calls = [c for c in node.pose_calls if c[0] not in names]
    assert len(parked_calls) == 2  # the other 2 (of 4) slots parked
    for _name, x, y, z in parked_calls:
        assert z == obstacle_pool._PARKING_Z_M
        assert abs(x) >= pool.parking_distance_m - 1e-6 or abs(y) >= pool.parking_distance_m - 1e-6


def test_activate_static_reuses_slots_across_varying_obstacle_counts():
    """Static/dynamic obstacle counts vary episode-to-episode -- activation
    must correctly reflect each call's own demand without leaking state
    from a previous call with a different count."""
    pool = _pool(max_static=5, classes=(0.5,))
    node = _FakeNode()
    obstacle_pool.ensure_spawned(node, _FakeClient(_Result(True)), pool)

    for n_obstacles in (0, 3, 1, 5, 2):
        node.pose_calls.clear()
        obstacles = [StaticObstacle(x=float(i), y=0.0, radius=0.5) for i in range(n_obstacles)]
        names = obstacle_pool.activate_static(node, pool, obstacles)
        assert len(names) == n_obstacles
        active_slots = [s for s in pool.static_slots if s.active]
        assert len(active_slots) == n_obstacles
        # every pose call is either one of the active names or a park call
        assert len(node.pose_calls) == len(pool.static_slots)


def test_activate_static_never_escalates_to_a_larger_class_and_fails_fast_instead():
    """requirement 4 (deliberate policy change): a spawned slot's geometry
    must match ScenarioSpec.radius EXACTLY -- silently borrowing a larger
    class's slot when the exact class runs out would spawn Gazebo collision
    geometry bigger than the radius every feasibility/risk computation
    assumed. So exhausting the exact class must raise, never spill over to
    a larger one. Radii are ALREADY-QUANTIZED (0.2, exactly a configured
    class) here -- see test_activate_static_raises_when_radius_was_not_pre_quantized
    for the separate "radius doesn't exactly match any class at all" case."""
    pool = _pool(max_static=4, classes=(0.2, 0.5))  # 2 slots of each class
    node = _FakeNode()
    obstacle_pool.ensure_spawned(node, _FakeClient(_Result(True)), pool)

    # 3 obstacles that all need the smaller (0.2) class -- only 2 slots of
    # that class exist; the 3rd must raise rather than silently use one of
    # the (otherwise free) 0.5 slots.
    obstacles = [StaticObstacle(x=float(i), y=0.0, radius=0.2) for i in range(3)]
    with pytest.raises(RuntimeError, match="EXACT size_class"):
        obstacle_pool.activate_static(node, pool, obstacles)


def test_activate_static_uses_the_exact_class_slot_and_never_a_bigger_free_one():
    pool = _pool(max_static=4, classes=(0.2, 0.5))  # 2 slots of each class
    node = _FakeNode()
    obstacle_pool.ensure_spawned(node, _FakeClient(_Result(True)), pool)

    obstacles = [StaticObstacle(x=float(i), y=0.0, radius=0.2) for i in range(2)]
    names = obstacle_pool.activate_static(node, pool, obstacles)
    assert len(names) == 2
    used_classes = sorted(s.size_class_m for s in pool.static_slots if s.active)
    assert used_classes == [0.2, 0.2]


def test_activate_static_raises_when_radius_was_not_pre_quantized():
    """requirement 4, round 2: activate_static must never ROUND UP a radius
    that doesn't already exactly match a configured class -- that would
    silently spawn geometry bigger than ScenarioSpec.radius and misreport
    it as an "exact class" match. A procedurally-generated scenario's radii
    are always pre-quantized by generate_scenario's own
    static_radius_quantizer before reaching here (see this module's
    docstring); a non-exact radius reaching activate_static is a bug
    upstream, surfaced immediately."""
    pool = _pool(max_static=4, classes=(0.2, 0.5))
    node = _FakeNode()
    obstacle_pool.ensure_spawned(node, _FakeClient(_Result(True)), pool)

    obstacles = [StaticObstacle(x=0.0, y=0.0, radius=0.15)]  # not pre-quantized
    with pytest.raises(RuntimeError, match="does not EXACTLY match"):
        obstacle_pool.activate_static(node, pool, obstacles)


def test_activate_static_raises_when_pool_capacity_is_insufficient():
    pool = _pool(max_static=2, classes=(0.5,))
    node = _FakeNode()
    obstacle_pool.ensure_spawned(node, _FakeClient(_Result(True)), pool)
    obstacles = [StaticObstacle(x=float(i), y=0.0, radius=0.5) for i in range(3)]
    with pytest.raises(RuntimeError, match="obstacle_pool"):
        obstacle_pool.activate_static(node, pool, obstacles)


def test_snap_up_to_class_rounds_up_and_raises_when_out_of_range():
    assert obstacle_pool.snap_up_to_class(0.18, [0.2, 0.35, 0.5]) == 0.2
    assert obstacle_pool.snap_up_to_class(0.5, [0.2, 0.35, 0.5]) == 0.5
    with pytest.raises(RuntimeError):
        obstacle_pool.snap_up_to_class(0.6, [0.2, 0.35, 0.5])


# --------------------------------------------------------------- activate_dynamic
def test_activate_dynamic_teleports_needed_slots_and_parks_the_rest():
    pool = _pool(max_dynamic=3)
    node = _FakeNode()
    obstacle_pool.ensure_spawned(node, _FakeClient(_Result(True)), pool)
    node.pose_calls.clear()

    specs = [DynamicObstacleSpec(x0=1.0, y0=1.0, vx=0.1, vy=0.0, radius=0.3)]
    names = obstacle_pool.activate_dynamic(node, pool, specs)
    assert len(names) == 1
    assert (names[0], 1.0, 1.0) in {(n, x, y) for n, x, y, z in node.pose_calls}
    parked = [c for c in node.pose_calls if c[0] != names[0]]
    assert len(parked) == 2  # the other 2 (of 3) dynamic slots parked


def test_activate_dynamic_raises_when_pool_capacity_is_insufficient():
    pool = _pool(max_dynamic=1)
    node = _FakeNode()
    obstacle_pool.ensure_spawned(node, _FakeClient(_Result(True)), pool)
    specs = [DynamicObstacleSpec(x0=0.0, y0=0.0, vx=0.0, vy=0.0, radius=0.3) for _ in range(2)]
    with pytest.raises(RuntimeError, match="obstacle_pool"):
        obstacle_pool.activate_dynamic(node, pool, specs)


def test_activate_dynamic_sets_active_true_for_teleported_and_false_for_parked_slots():
    """requirement 4: activate_dynamic previously never touched `.active`
    at all -- it must now correctly mark every teleported slot active=True
    and every parked slot active=False, on every call (not just the first),
    so callers (e.g. environment_node.py's own active/parked count logging)
    can trust it."""
    pool = _pool(max_dynamic=3)
    node = _FakeNode()
    obstacle_pool.ensure_spawned(node, _FakeClient(_Result(True)), pool)

    specs = [DynamicObstacleSpec(x0=1.0, y0=1.0, vx=0.1, vy=0.0, radius=0.3),
             DynamicObstacleSpec(x0=2.0, y0=2.0, vx=0.0, vy=0.1, radius=0.3)]
    obstacle_pool.activate_dynamic(node, pool, specs)
    active_names = {s.name for s in pool.dynamic_slots if s.active}
    parked_names = {s.name for s in pool.dynamic_slots if not s.active}
    assert len(active_names) == 2
    assert len(parked_names) == 1

    # A later call with FEWER specs must flip the previously-active slots
    # back to active=False (never leave stale True state behind).
    obstacle_pool.activate_dynamic(node, pool, specs[:1])
    assert sum(1 for s in pool.dynamic_slots if s.active) == 1
    assert sum(1 for s in pool.dynamic_slots if not s.active) == 2


def test_activate_dynamic_with_zero_specs_parks_every_slot():
    """A fixed-benchmark episode (never using the pool) must still fully
    park it if the pool was previously left with active slots."""
    pool = _pool(max_dynamic=2)
    node = _FakeNode()
    obstacle_pool.ensure_spawned(node, _FakeClient(_Result(True)), pool)
    obstacle_pool.activate_dynamic(node, pool, [DynamicObstacleSpec(x0=0, y0=0, vx=0, vy=0, radius=0.3)])
    node.pose_calls.clear()

    names = obstacle_pool.activate_dynamic(node, pool, [])
    assert names == []
    assert len(node.pose_calls) == 2  # both slots parked
