#!/usr/bin/env python3
"""Deterministic Gazebo obstacle ENTITY POOL (requirement 2 of the
drl_agent -> hunter_kinodynamic_rl port -- see ``config/schema.py``'s
``ObstaclePoolConfig``).

Problem this solves: ``environment_node.py``'s legacy path (see
``env/spawning/obstacle_spawner.py``) deletes every obstacle entity and
re-spawns a fresh set on EVERY ``/reset`` -- at a high-frequency training
reset rate this is a lot of avoidable ``SpawnEntity``/``DeleteEntity``
service-call churn, and any bookkeeping drift between what this node
THINKS is spawned and what Gazebo actually has produces Ignition's own
"Entity named [X] ... not found, so not removed" log line.

Design: pre-spawn a FIXED set of plain-cylinder markers once (`ensure_spawned`),
then every episode TELEPORT (``SetEntityPose``) exactly the slots this
episode needs to their scenario positions and PARK every other slot at a
fixed, far-outside-the-arena position (`activate_static`/`activate_dynamic`).
Entity names and count never change again after `ensure_spawned` -- no
further Spawn/Delete calls happen for pooled entities for the rest of the
process's lifetime.

Static obstacles only: procedurally generate_scenario draws a CONTINUOUS
radius per obstacle, but a spawned SDF's collision geometry is fixed at
spawn time -- a pool slot can be TELEPORTED but never RESIZED. So a pool
slot's radius is one of ``ObstaclePoolConfig.static_size_classes_m`` (a
small, config-authored set of discrete sizes), and
``procedural_generator.generate_scenario`` is called with a
``static_radius_quantizer`` (see :func:`snap_up_to_class`) that snaps every
drawn radius UP to the nearest class BEFORE any clearance/feasibility check
runs -- so the feasibility math and the actual spawned geometry are always
for the exact same radius, never a silent mismatch. Dynamic obstacles need
no such quantization: ``procedural_generator.DYNAMIC_OBSTACLE_RADIUS_M`` is
already a single fixed constant.

Scope: only PROCEDURALLY generated (training) scenarios ever use this pool
-- a fixed benchmark scenario (``evaluation_node.py``) always uses the
legacy spawn/delete path (``obstacle_spawner.py``) regardless of
``obstacle_pool.enabled``, so evaluation geometry is never approximated by
the pool's quantized sizes. ``environment_node.py`` is responsible for this
branch (see its ``_spawn_scenario_obstacles``); every function in this
module simply does whatever it's told for however many obstacles it's
given (0 obstacles is a completely valid "this episode uses none of the
pool" call, which is exactly how the fixed-benchmark case is handled: it
requests zero active slots, parking every one of them).

Reuses the calling node's already-hang-safe ``_wait_for_srv``/
``_await_future``/``set_entity_pose_ignition`` (see gazebo_runtime.py) --
every spawn/teleport call here is bounded and RAISES
``GazeboServiceError`` on timeout or ``success=false``, exactly like the
legacy spawner (never a silent partial failure).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, TYPE_CHECKING

from hunter_kinodynamic_rl.env.scenarios.procedural_generator import DYNAMIC_OBSTACLE_RADIUS_M
from hunter_kinodynamic_rl.env.simulation.gazebo_service_wait import GazeboServiceError
from hunter_kinodynamic_rl.env.spawning.obstacle_spawner import _cylinder_sdf, _spawn_one

if TYPE_CHECKING:
    from hunter_kinodynamic_rl.config.schema import ObstaclePoolConfig
    from hunter_kinodynamic_rl.env.scenarios.procedural_generator import DynamicObstacleSpec, StaticObstacle

STATIC_POOL_PREFIX = "hkrl_pool_static_"
DYNAMIC_POOL_PREFIX = "hkrl_pool_dynamic_"

# Marker markers are parked deep below ground on top of being pushed far
# outside the arena -- a second, independent margin of safety against ever
# being mistaken for a real obstacle regardless of any XY margin mistake.
_PARKING_Z_M = -50.0
_PARKING_SPACING_M = 2.0
_PARKING_ROW_LEN = 20


@dataclass
class _StaticSlot:
    name: str
    size_class_m: float
    active: bool = False
    # requirement 4: tracked PER SLOT (not just the pool-level `ready`
    # flag) so a partial ensure_spawned failure (some slots spawned, then a
    # SpawnEntity call fails) can RESUME on the next call/retry by spawning
    # only the slots still False here -- never blindly re-spawns a slot
    # that already exists in Gazebo (which would itself fail/duplicate).
    spawned: bool = False


@dataclass
class _DynamicSlot:
    name: str
    active: bool = False
    spawned: bool = False


@dataclass
class ObstaclePool:
    """Constructed ONCE per environment_node process, from the LAUNCH-time
    profile (never re-derived from a later evaluation-contract override --
    see environment_node.py's own RUNTIME_FIELDS_FIXED_AT_LAUNCH for the
    same launch-time-fixed pattern). Holds every pool slot's stable
    identity; ``ready`` flips True after :func:`ensure_spawned` has
    actually spawned them."""

    cfg: "ObstaclePoolConfig"
    parking_distance_m: float
    static_slots: List[_StaticSlot] = field(default_factory=list)
    dynamic_slots: List[_DynamicSlot] = field(default_factory=list)
    ready: bool = False

    def __post_init__(self) -> None:
        if self.static_slots or self.dynamic_slots:
            return
        classes = list(self.cfg.static_size_classes_m)
        for i in range(self.cfg.max_static):
            size_class = classes[i % len(classes)]
            self.static_slots.append(_StaticSlot(name=f"{STATIC_POOL_PREFIX}{i}", size_class_m=size_class))
        for i in range(self.cfg.max_dynamic):
            self.dynamic_slots.append(_DynamicSlot(name=f"{DYNAMIC_POOL_PREFIX}{i}"))

    def parking_position(self, slot_index: int) -> Tuple[float, float]:
        row, col = divmod(slot_index, _PARKING_ROW_LEN)
        x = self.parking_distance_m + col * _PARKING_SPACING_M
        y = -self.parking_distance_m - row * _PARKING_SPACING_M
        return x, y


def build_pool(cfg: "ObstaclePoolConfig", world_size_m: float, lidar_max_range_m: float) -> ObstaclePool:
    """``parking_distance_m`` guarantees a parked marker is unreachable by
    ANY LiDAR ray cast from anywhere inside the arena, regardless of where
    in the arena the robot is standing: the worst case is the robot at one
    edge, looking at a marker parked just past the OPPOSITE edge, a
    straight-line distance of (world_size_m + parking_margin_m) -- so this
    only needs to exceed lidar_max_range_m by parking_margin_m from the
    world's OWN half-extent, not by a further half-extent on top."""
    parking_distance_m = world_size_m / 2.0 + lidar_max_range_m + cfg.parking_margin_m
    return ObstaclePool(cfg=cfg, parking_distance_m=parking_distance_m)


def snap_up_to_class(radius: float, size_classes: Sequence[float]) -> float:
    """Smallest class >= radius -- raises if the radius exceeds every
    configured class (a config bug: ``Profile.validate()`` already checks
    ``static_size_classes_m``'s largest entry covers
    ``STATIC_OBSTACLE_RADIUS_RANGE_M``'s max, so this should be
    unreachable in practice; kept as a real, loud runtime backstop rather
    than a silent clamp). This is the QUANTIZATION step -- only ever called
    from ``generate_scenario``'s own ``static_radius_quantizer`` (see this
    module's docstring), BEFORE any feasibility/clearance check runs.
    :func:`activate_static` deliberately does NOT call this (see
    :func:`matching_exact_class` instead) -- by the time a radius reaches
    activation it must ALREADY be exactly one of these classes."""
    for size_class in size_classes:
        if size_class >= radius - 1e-9:
            return size_class
    raise RuntimeError(
        f"obstacle_pool: radius {radius} exceeds every configured static_size_classes_m "
        f"{list(size_classes)} -- increase the largest class"
    )


def matching_exact_class(radius: float, size_classes: Sequence[float], tolerance: float = 1e-9) -> Optional[float]:
    """The class ``radius`` already exactly equals (within ``tolerance``),
    or ``None`` if it doesn't exactly match ANY configured class.

    Unlike :func:`snap_up_to_class` (which ROUNDS UP -- the correct
    behaviour for quantizing a freshly-drawn continuous radius at
    generation time), :func:`activate_static` must never silently widen an
    obstacle's declared radius to fit a slot: a procedurally-generated
    scenario's radii are already pre-quantized by
    ``generate_scenario``'s ``static_radius_quantizer`` (== this same
    ``snap_up_to_class``) before any feasibility/clearance math runs, so by
    the time a radius reaches activation it should already be exactly one
    of these classes. A caller that bypasses that quantization step (e.g.
    a runtime backstop path, or a test) supplying a genuinely
    non-quantized radius is a bug at the CALLER, not something
    ``activate_static`` should paper over by rounding -- see that
    function's own docstring."""
    for size_class in size_classes:
        if abs(size_class - radius) <= tolerance:
            return size_class
    return None


def _ensure_slot_spawned(node, spawn_client, slot, sdf: str, x: float, y: float) -> None:
    """Spawns ``slot`` at its own parking position, but first handles the
    AMBIGUOUS case a plain ``_spawn_one`` failure can't distinguish
    (requirement 4, round 2): Gazebo may have actually CREATED the entity
    server-side even though the ``SpawnEntity`` RESPONSE never reached this
    node within ``_gz_call_timeout_sec`` (or some other exception fired
    after the request was already accepted) -- ``_spawn_one`` raising
    ``GazeboServiceError`` does NOT prove the entity is absent. Blindly
    leaving ``slot.spawned=False`` in that case makes the NEXT
    ``ensure_spawned`` call re-issue ``SpawnEntity`` for a name Gazebo
    already has, which itself fails (or, worse, could duplicate the
    entity).

    So on a spawn failure, this probes the entity's REAL existence by
    attempting to teleport it to its own parking pose via
    ``set_entity_pose_ignition``: ``SetEntityPose`` reports
    ``success=false`` for an entity name Gazebo doesn't recognize (see
    ``gazebo_runtime.py``'s own P0-2 comment), so a SUCCESSFUL probe proves
    the entity really is there (and leaves it correctly parked, for free)
    while a FAILED probe proves it genuinely still needs spawning. Either
    way this re-raises the ORIGINAL spawn error when the entity turns out
    to be genuinely absent, so ``ensure_spawned``'s existing "leave
    ``spawned=False``, retry on the next call" contract is unchanged for a
    real failure."""
    try:
        _spawn_one(node, spawn_client, slot.name, sdf, x, y, _PARKING_Z_M, 0.0)
        slot.spawned = True
        return
    except GazeboServiceError as e:
        # `except ... as e` unbinds `e` once this block exits (Python 3
        # semantics) -- copy it to a plain variable so it survives to the
        # probe below and can be re-raised after it.
        spawn_error = e
    try:
        node.set_entity_pose_ignition(slot.name, x, y, _PARKING_Z_M, 0.0, 0.0, 0.0, 1.0)
    except GazeboServiceError:
        raise spawn_error
    slot.spawned = True
    node.get_logger().warn(
        f"[obstacle_pool] spawn[{slot.name}] response was lost/failed ({spawn_error}) but the entity "
        "already exists in Gazebo (confirmed via SetEntityPose) -- treating as spawned rather than "
        "retrying SpawnEntity against an existing name"
    )


def ensure_spawned(node, spawn_client, pool: ObstaclePool) -> None:
    """Idempotent: spawns every pool slot EXACTLY ONCE, at its own parking
    position, the first time this is called for a given ``pool`` instance.
    A later call with ``pool.ready`` already True is a complete no-op --
    never re-spawns, never re-checks existence proactively (ongoing
    existence is verified implicitly every episode by the strict, raising
    ``set_entity_pose_ignition`` calls in :func:`activate_static`/
    :func:`activate_dynamic`, see this module's docstring).

    requirement 4 (partial-failure recovery): if a PREVIOUS call raised
    partway through (e.g. slots 0..k spawned, then slot k+1's SpawnEntity
    call failed/timed out), each already-spawned slot's own ``spawned``
    flag stays True across calls -- a RETRY call below only (re-)attempts
    slots still marked ``spawned=False``, so it never re-issues
    ``SpawnEntity`` for an entity that already exists in Gazebo (which
    would itself fail or, worse, duplicate it). Each attempt goes through
    :func:`_ensure_slot_spawned`, which additionally disambiguates "the
    call failed/timed out but Gazebo actually created it anyway" from a
    genuine absence (round 2 of requirement 4 -- see that function's own
    docstring). ``pool.ready`` is set True ONLY once every single slot
    (static AND dynamic) has been individually CONFIRMED spawned this way
    -- never optimistically before that."""
    if pool.ready:
        return
    slot_index = 0
    for slot in pool.static_slots:
        x, y = pool.parking_position(slot_index)
        slot_index += 1
        if slot.spawned:
            continue
        _ensure_slot_spawned(node, spawn_client, slot, _cylinder_sdf(slot.name, slot.size_class_m), x, y)
    for slot in pool.dynamic_slots:
        x, y = pool.parking_position(slot_index)
        slot_index += 1
        if slot.spawned:
            continue
        _ensure_slot_spawned(node, spawn_client, slot, _cylinder_sdf(slot.name, DYNAMIC_OBSTACLE_RADIUS_M), x, y)
    pool.ready = all(s.spawned for s in pool.static_slots) and all(s.spawned for s in pool.dynamic_slots)
    if pool.ready:
        node.get_logger().info(
            f"[obstacle_pool] spawned {len(pool.static_slots)} static + {len(pool.dynamic_slots)} dynamic "
            f"pool slots (parking_distance_m={pool.parking_distance_m:.1f})"
        )


def _park(node, pool: ObstaclePool, slot_index: int, name: str) -> None:
    x, y = pool.parking_position(slot_index)
    node.set_entity_pose_ignition(name, x, y, _PARKING_Z_M, 0.0, 0.0, 0.0, 1.0)


def activate_static(node, pool: ObstaclePool, obstacles: "List[StaticObstacle]") -> List[str]:
    """Assigns each of ``obstacles`` (already radius-snapped by
    :func:`snap_up_to_class` via generate_scenario's own
    ``static_radius_quantizer`` -- see this module's docstring) to a free
    slot whose ``size_class_m`` EXACTLY equals its (already-quantized)
    radius, teleports it there, and parks every slot NOT used this
    episode. requirement 4: never escalates to a LARGER class when its own
    exact class is momentarily exhausted -- doing so would spawn Gazebo
    collision geometry strictly bigger than ``ScenarioSpec.radius``, a
    silent mismatch between what feasibility/risk computations assumed and
    what is physically in the world (e.g. a gap sized for the smaller
    declared radius that the robot could actually clip on the larger
    spawned one). Raises ``RuntimeError`` immediately instead whenever the
    exact class has no free slot (``Profile.validate()``'s own per-class
    worst-case capacity check should already have made this unreachable
    for a procedurally-generated scenario; this is the runtime backstop
    for any other caller). Returns the activated entity names in
    ``obstacles`` order, exactly like the legacy path's
    ``_spawned_static_names``.

    requirement 4, round 2: looks up each obstacle's class via
    :func:`matching_exact_class`, NOT :func:`snap_up_to_class` -- the
    latter ROUNDS UP an arbitrary radius to the nearest class, which would
    let an obstacle whose radius was never actually pre-quantized (e.g.
    0.15 with classes [0.2, 0.5]) silently activate in a slot bigger than
    its own declared ``ScenarioSpec.radius`` and be misreported as an
    "exact class" match. A procedurally-generated scenario's radii are
    ALWAYS already exactly one of these classes by the time they reach
    here (see this function's own docstring above), so a non-exact radius
    reaching this point is itself a bug upstream -- surfaced immediately
    rather than papered over."""
    if len(obstacles) > len(pool.static_slots):
        raise RuntimeError(
            f"obstacle_pool: scenario needs {len(obstacles)} static obstacles but only "
            f"{len(pool.static_slots)} pool slots exist -- increase obstacle_pool.max_static"
        )
    for slot in pool.static_slots:
        slot.active = False
    names: List[str] = []
    classes_ascending = sorted({slot.size_class_m for slot in pool.static_slots})
    for obstacle in obstacles:
        needed = matching_exact_class(obstacle.radius, classes_ascending)
        if needed is None:
            raise RuntimeError(
                f"obstacle_pool: obstacle radius={obstacle.radius} does not EXACTLY match any "
                f"configured static_size_classes_m {classes_ascending} -- generate_scenario's "
                "static_radius_quantizer should have pre-quantized this radius to one of these "
                "classes before it ever reached activate_static; the scenario was not properly "
                "quantized upstream"
            )
        chosen = next((s for s in pool.static_slots if s.size_class_m == needed and not s.active), None)
        if chosen is None:
            raise RuntimeError(
                f"obstacle_pool: no free static slot in the EXACT size_class {needed} needed for "
                f"obstacle radius={obstacle.radius} (classes={classes_ascending}) -- refusing to "
                "escalate to a larger class (the spawned geometry must match ScenarioSpec.radius "
                "exactly); increase obstacle_pool.max_static for that class, or fewer "
                "static_size_classes_m"
            )
        chosen.active = True
        node.set_entity_pose_ignition(chosen.name, obstacle.x, obstacle.y, 0.0, 0.0, 0.0, 0.0, 1.0)
        names.append(chosen.name)
    for slot_index, slot in enumerate(pool.static_slots):
        if not slot.active:
            _park(node, pool, slot_index, slot.name)
    return names


def activate_dynamic(node, pool: ObstaclePool, specs: "List[DynamicObstacleSpec]") -> List[str]:
    """Same contract as :func:`activate_static` but for the (single,
    fixed-radius) dynamic-obstacle pool -- teleports the first
    ``len(specs)`` slots to their t=0 position and parks the rest.

    requirement 4: every slot's own ``.active`` is explicitly set (True for
    a teleported slot, False for a parked one) on EVERY call -- previously
    this never touched ``.active`` at all, so it silently stayed at
    whatever it was last set to (its dataclass default, since nothing else
    ever wrote it), making ``.active`` an unreliable signal of a dynamic
    slot's real state (e.g. for logging/diagnostics) regardless of what was
    actually teleported/parked."""
    if len(specs) > len(pool.dynamic_slots):
        raise RuntimeError(
            f"obstacle_pool: scenario needs {len(specs)} dynamic obstacles but only "
            f"{len(pool.dynamic_slots)} pool slots exist -- increase obstacle_pool.max_dynamic"
        )
    names: List[str] = []
    static_slot_count = len(pool.static_slots)
    for i, slot in enumerate(pool.dynamic_slots):
        if i < len(specs):
            spec = specs[i]
            slot.active = True
            node.set_entity_pose_ignition(slot.name, spec.x0, spec.y0, 0.0, 0.0, 0.0, 0.0, 1.0)
            names.append(slot.name)
        else:
            slot.active = False
            _park(node, pool, static_slot_count + i, slot.name)
    return names
