#!/usr/bin/env python3
"""Deterministic Gazebo wall-segment ENTITY POOL (plan section 7.4, opt-in
via ``config/schema.py``'s ``WallSegmentPoolConfig``) -- mirrors
``env/spawning/obstacle_pool.py``'s "pre-spawn a fixed set once, TELEPORT
the active slots to each new episode's positions, park the rest off-arena"
design (see that module's docstring for the full rationale: avoiding
per-``/reset`` Spawn/Delete service-call churn for procedurally-generated
geometry that changes every episode). Reuses ``obstacle_pool``'s own
``_ensure_slot_spawned`` helper for partial-spawn-failure recovery (lazily
imported -- see the import-boundary note below); its size-class-
quantization helper is reimplemented locally instead of imported, for the
same reason.

Only LENGTH is pool-quantized (to the nearest ``length_classes_m`` class,
rounded UP, at ACTIVATION time): thickness/height are a single fixed value
per world (``LongHorizonWorldConfig.wall_thickness_m``/``wall_height_m``),
passed into :func:`build_wall_pool` once. This is a DIFFERENT quantization
point than ``obstacle_pool``'s own static-obstacle radius handling (which
quantizes at GENERATION time, before any feasibility check runs, so the
declared and spawned radius always match EXACTLY) -- seeing walls
quantized only at activation is a deliberate, documented approximation: see
:func:`activate_walls`'s own docstring for the exact trade-off and why it
was chosen (this Phase 3 delivery has no live Gazebo verification session
available to it; see ``docs/verification/``).

Module-scope import boundary (code review): unlike ``obstacle_pool.py``
itself (which legitimately needs ``ros_gz_interfaces`` at module scope for
its ``SpawnEntity``/``DeleteEntity`` request types), THIS module's own
public API (``WallSegmentPool``, ``build_wall_pool``, the SDF-string
builder) has no such need -- ``ensure_spawned``/``activate_walls`` only
ever receive an already-connected ``node``/``spawn_client`` from their
caller and never construct a ROS message type themselves. A previous
version imported ``obstacle_pool._ensure_slot_spawned``/``snap_up_to_class``
at MODULE scope, which transitively pulled in
``obstacle_pool -> obstacle_spawner -> ros_gz_interfaces.srv`` -- making
``import hunter_kinodynamic_rl.env.spawning.wall_segment_spawner`` itself
fail with ``ModuleNotFoundError: ros_gz_interfaces`` on a bare, ROS-free
host (confirmed live). Fixed: :func:`_snap_length_up_to_class` is a small,
local, genuinely pure reimplementation of ``obstacle_pool.snap_up_to_class``
(radius-vs-length is just naming -- the algorithm is identical, and a
length-specific error message is clearer here than importing the
radius-worded original would be); ``obstacle_pool._ensure_slot_spawned``
is imported LAZILY, inside :func:`ensure_spawned` itself, the one function
that actually needs it -- so this module now stays importable (and every
non-Gazebo function in it callable) on a bare host with no ROS install at
all, matching every other pure-Python Phase 3 module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from hunter_kinodynamic_rl.config.schema import WallSegmentPoolConfig
    from hunter_kinodynamic_rl.env.scenarios.long_horizon_world import WallSegment


def _snap_length_up_to_class(length_m: float, length_classes_m: Sequence[float]) -> float:
    """Smallest configured class >= ``length_m`` -- a local, pure
    reimplementation of ``obstacle_pool.snap_up_to_class`` (see this
    module's docstring for why it is not imported instead). Raises if
    ``length_m`` exceeds every class -- ``config/schema.py``'s
    ``validate_wall_pool_capacity`` already checks the largest class covers
    the worst-case wall length, so this should be unreachable for a
    validated profile; kept as a real, loud runtime backstop rather than a
    silent clamp, exactly like the function it mirrors."""
    for length_class in length_classes_m:
        if length_class >= length_m - 1e-9:
            return length_class
    raise RuntimeError(
        f"wall_segment_spawner: length {length_m} exceeds every configured length_classes_m "
        f"{list(length_classes_m)} -- increase the largest class"
    )

WALL_POOL_PREFIX = "hkrl_wall_"

# Same deep-parking convention as obstacle_pool.py -- pushed far outside the
# arena AND deep below ground, an independent second margin of safety.
_PARKING_Z_M = -50.0
_PARKING_SPACING_M = 2.0
_PARKING_ROW_LEN = 20


def _box_sdf(model_name: str, length_m: float, thickness_m: float, height_m: float) -> str:
    """Local X == length (the segment's own long axis before yaw rotation),
    local Y == thickness, local Z == height -- matches
    ``long_horizon_world.WallSegment``'s ``yaw_rad=0`` convention (long axis
    along world X) exactly, so the spawn orientation quaternion below is a
    pure yaw rotation with no axis remapping needed."""
    return (
        '<sdf version="1.8">'
        f'<model name="{model_name}">'
        "<static>true</static>"
        '<link name="link">'
        '<collision name="collision">'
        f"<geometry><box><size>{length_m:.3f} {thickness_m:.3f} {height_m:.3f}</size></box></geometry>"
        "</collision>"
        '<visual name="visual">'
        f"<geometry><box><size>{length_m:.3f} {thickness_m:.3f} {height_m:.3f}</size></box></geometry>"
        "<material><ambient>0.55 0.55 0.6 1</ambient><diffuse>0.55 0.55 0.6 1</diffuse></material>"
        "</visual>"
        "</link>"
        "</model>"
        "</sdf>"
    )


@dataclass
class _WallSlot:
    name: str
    length_class_m: float
    active: bool = False
    spawned: bool = False


@dataclass
class WallSegmentPool:
    """Constructed ONCE per environment process, from the launch-time
    profile -- mirrors ``ObstaclePool``'s own lifecycle contract exactly."""

    cfg: "WallSegmentPoolConfig"
    wall_thickness_m: float
    wall_height_m: float
    parking_distance_m: float
    slots: List[_WallSlot] = field(default_factory=list)
    ready: bool = False

    def __post_init__(self) -> None:
        if self.slots:
            return
        classes = list(self.cfg.length_classes_m)
        for i in range(self.cfg.max_segments):
            length_class = classes[i % len(classes)]
            self.slots.append(_WallSlot(name=f"{WALL_POOL_PREFIX}{i}", length_class_m=length_class))

    def parking_position(self, slot_index: int) -> Tuple[float, float]:
        row, col = divmod(slot_index, _PARKING_ROW_LEN)
        x = self.parking_distance_m + col * _PARKING_SPACING_M
        y = -self.parking_distance_m - row * _PARKING_SPACING_M
        return x, y


def build_wall_pool(
    cfg: "WallSegmentPoolConfig", wall_thickness_m: float, wall_height_m: float,
    world_size_m: float, lidar_max_range_m: float,
) -> WallSegmentPool:
    """Same "unreachable by any LiDAR ray from anywhere inside the arena"
    parking-distance derivation as ``obstacle_pool.build_pool`` -- see that
    function's docstring for the worst-case geometry argument."""
    parking_distance_m = world_size_m / 2.0 + lidar_max_range_m + cfg.parking_margin_m
    return WallSegmentPool(
        cfg=cfg, wall_thickness_m=wall_thickness_m, wall_height_m=wall_height_m,
        parking_distance_m=parking_distance_m,
    )


def _park(node, pool: WallSegmentPool, slot_index: int, name: str) -> None:
    x, y = pool.parking_position(slot_index)
    node.set_entity_pose_ignition(name, x, y, _PARKING_Z_M, 0.0, 0.0, 0.0, 1.0)


def ensure_spawned(node, spawn_client, pool: WallSegmentPool) -> None:
    """Idempotent, resumable partial-failure recovery -- identical contract
    to ``obstacle_pool.ensure_spawned`` (reuses its own
    ``_ensure_slot_spawned`` helper directly, see this module's docstring).

    ``_ensure_slot_spawned`` is imported LAZILY here (not at module scope --
    see this module's docstring) so importing ``wall_segment_spawner`` and
    calling every OTHER function in it (``build_wall_pool``,
    ``activate_walls``, the SDF builder) never requires ``ros_gz_interfaces``
    to be resolvable; only actually calling this specific function does."""
    from hunter_kinodynamic_rl.env.spawning.obstacle_pool import _ensure_slot_spawned

    if pool.ready:
        return
    for slot_index, slot in enumerate(pool.slots):
        if slot.spawned:
            continue
        x, y = pool.parking_position(slot_index)
        sdf = _box_sdf(slot.name, slot.length_class_m, pool.wall_thickness_m, pool.wall_height_m)
        _ensure_slot_spawned(node, spawn_client, slot, sdf, x, y)
    pool.ready = all(s.spawned for s in pool.slots)
    if pool.ready:
        node.get_logger().info(
            f"[wall_segment_pool] spawned {len(pool.slots)} wall pool slots "
            f"(parking_distance_m={pool.parking_distance_m:.1f})"
        )


def activate_walls(node, pool: WallSegmentPool, wall_segments: "List[WallSegment]") -> List[str]:
    """Assigns each of ``wall_segments`` to a free slot whose
    ``length_class_m`` is the SMALLEST configured class >= its own
    ``length_m`` (:func:`_snap_length_up_to_class`), teleports+yaw-orients
    it there, and parks every slot NOT used this
    world. Raises ``RuntimeError`` (never silently escalates to a larger
    class beyond the exact quantized one, or drops a segment) whenever a
    segment's own class has no free slot -- the same "loud, immediate
    capacity failure, never a silent geometry mismatch" discipline
    ``obstacle_pool.activate_static`` uses.

    Approximation (documented, see module docstring): quantization happens
    HERE, at activation time, not during ``long_horizon_generator``'s own
    world generation -- so the physically-spawned Gazebo box can be up to
    one length-class step LONGER than ``seg.length_m``, growing
    symmetrically about the segment's own declared center
    (``seg.center_x_m``, ``seg.center_y_m``). For one of a passage-flanking
    remainder-segment pair, this can narrow the PHYSICAL passage by up to
    half that overflow versus what ``LongHorizonWorld.occupancy`` (the pure-
    Python contract every test/solvability check reasons about) declares --
    never affects the returned world's own occupancy/connectivity/shortest-
    path guarantees, only the live Gazebo geometry. Acceptable for a
    training-time approximation; a live Gazebo run should keep
    ``wall_segment_pool.length_classes_m``'s step size small relative to the
    smallest configured ``corridor_width_min_m`` to keep this negligible."""
    if len(wall_segments) > len(pool.slots):
        raise RuntimeError(
            f"wall_segment_pool: world needs {len(wall_segments)} wall segments but only "
            f"{len(pool.slots)} pool slots exist -- increase wall_segment_pool.max_segments"
        )
    for slot in pool.slots:
        slot.active = False
    names: List[str] = []
    classes_ascending = sorted({s.length_class_m for s in pool.slots})
    for seg in wall_segments:
        try:
            needed = _snap_length_up_to_class(seg.length_m, classes_ascending)
        except RuntimeError as e:
            raise RuntimeError(
                f"wall_segment_pool: wall segment {seg.entity_id} length={seg.length_m} exceeds every "
                f"configured length_classes_m {classes_ascending} -- increase the largest length class"
            ) from e
        chosen = next((s for s in pool.slots if s.length_class_m == needed and not s.active), None)
        if chosen is None:
            raise RuntimeError(
                f"wall_segment_pool: no free slot in the length class {needed} needed for wall segment "
                f"{seg.entity_id} (length={seg.length_m}, classes={classes_ascending}) -- increase "
                "wall_segment_pool.max_segments for that class, or use fewer length_classes_m"
            )
        chosen.active = True
        qz, qw = math.sin(seg.yaw_rad / 2.0), math.cos(seg.yaw_rad / 2.0)
        node.set_entity_pose_ignition(chosen.name, seg.center_x_m, seg.center_y_m, 0.0, 0.0, 0.0, qz, qw)
        names.append(chosen.name)
    for slot_index, slot in enumerate(pool.slots):
        if not slot.active:
            _park(node, pool, slot_index, slot.name)
    return names
