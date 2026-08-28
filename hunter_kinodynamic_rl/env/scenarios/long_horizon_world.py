"""Long-horizon procedural world data contracts (plan section 7.2).

Pure dataclasses only -- no generation logic here (see
``long_horizon_generator.py``), no Gazebo (see ``env/spawning/wall_segment_spawner.py``).

``WallSegment`` is an axis-or-rotated oriented box: ``yaw_rad=0`` means the
segment's LONG axis (``length_m``) runs along world X and its SHORT axis
(``thickness_m``) along world Y; ``yaw_rad`` rotates that box CCW about its
own center exactly like every other yaw convention in this package
(``dynamics/bicycle_model.py``'s module docstring). ``long_horizon_generator``
only ever emits axis-aligned segments (``yaw_rad in {0.0, pi/2}``) -- the
field itself is general so the type can outlive that implementation detail.

``LongHorizonWorld.occupancy`` is ALWAYS the exact rasterization of
``wall_segments`` (see ``long_horizon_generator.build_occupancy_from_segments``) --
the two are never allowed to drift apart, which is what
``tests/test_long_horizon_generator.py``'s consistency tests check.

``topology_metadata`` is PRIVILEGED (generator/evaluation/teacher use only --
plan section 3.2's information boundary): it must never be threaded into a
policy observation. See ``tests/test_world_information_boundary.py``.

Immutability (code review): ``frozen=True`` alone only blocks reassigning a
FIELD (``world.occupancy = ...`` raises) -- it does nothing to stop
mutating an already-assigned MUTABLE field's own contents in place
(``world.occupancy[:] = False``, ``world.wall_segments.append(...)``,
``world.topology_metadata["x"] = 1`` were all previously possible despite
``frozen=True``). ``LongHorizonWorld.__post_init__`` below normalizes every
such field into a genuinely immutable value at construction time
(read-only ``ndarray`` copy, ``tuple``, ``MappingProxyType``) regardless of
what the caller passed in -- see ``tests/test_world_information_boundary.py``'s
mutation-attempt tests.

Round-2-of-review follow-up: the FIRST version of this freeze only wrapped
``topology_metadata``'s TOP-LEVEL dict in ``MappingProxyType`` -- a nested
mutable value (e.g. an external caller passing
``topology_metadata={"room_cells": [1, 2]}``, a plain ``list``, rather than
the generator's own ``tuple``) was still mutable in place
(``world.topology_metadata["room_cells"].append(3)``), even though the
generator itself happened to always pass a tuple. :func:`_deep_freeze`
below recursively freezes every ``Mapping``/``list``/``tuple``/``ndarray``
value found ANYWHERE inside ``topology_metadata``, regardless of what the
caller supplied -- the generator no longer needs (and no longer bothers)
to pre-convert ``room_cells`` to a tuple itself; this is now the ONE place
that guarantee is enforced.
"""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence, Tuple

import numpy as np


def _deep_freeze(value: Any) -> Any:
    """Recursively converts ``value`` into a genuinely immutable
    equivalent: any ``Mapping`` -> a ``MappingProxyType`` over a freshly
    built dict whose OWN values are themselves recursively frozen; any
    ``list``/``tuple`` -> a ``tuple`` of recursively frozen elements; any
    ``numpy.ndarray`` -> a read-only COPY (never aliasing the input); every
    other type (``int``/``float``/``str``/``bool``/``None``/already-frozen
    dataclasses like ``WallSegment``) is returned as-is, since it is
    already immutable or this module has no way to know how to freeze it
    further. Never mutates ``value`` itself."""
    if isinstance(value, MappingABC):
        return MappingProxyType({k: _deep_freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(v) for v in value)
    if isinstance(value, np.ndarray):
        arr = np.array(value, copy=True)
        arr.setflags(write=False)
        return arr
    return value


@dataclass(frozen=True)
class WallSegment:
    center_x_m: float
    center_y_m: float
    length_m: float
    thickness_m: float
    height_m: float
    yaw_rad: float
    # "corridor" | "room_boundary" | "boundary" -- see long_horizon_generator's
    # module docstring for the exact assignment rule. Free-form beyond that
    # (a future generator variant may add more semantic classes), so this is
    # a plain str, not an enum.
    semantic: str
    # Stable index into LongHorizonWorld.wall_segments -- also the ONLY
    # identity a Gazebo spawner needs (wall_segment_spawner.py derives its
    # own spawn/pool slot naming from this, never from list position, which
    # would silently reassign identities across two worlds with different
    # segment counts).
    entity_id: str


@dataclass(frozen=True)
class LongHorizonWorld:
    seed: int
    occupancy: np.ndarray
    resolution_m: float
    origin_xy: Tuple[float, float]
    wall_segments: Sequence[WallSegment] = field(default_factory=tuple)
    start_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    goal_pose: Tuple[float, float] = (0.0, 0.0)
    shortest_path_length_m: float = 0.0
    # PRIVILEGED (generator/evaluation/teacher-only, plan section 3.2) --
    # room_count, dead_end_count, loop_count, alternative_route_count,
    # grid_n, pitch_m, plus diagnostic node/edge lists. Never expose this
    # dict (or any array derived from it) to a policy observation builder.
    topology_metadata: Mapping = field(default_factory=dict)

    def __post_init__(self) -> None:
        # np.array(..., copy=True) decouples from whatever mutable array
        # object the caller passed in (never aliasing it), THEN
        # setflags(write=False) makes the returned array itself reject an
        # in-place write (world.occupancy[:] = False now raises
        # ValueError: assignment destination is read-only).
        occupancy = np.array(self.occupancy, copy=True)
        occupancy.setflags(write=False)
        object.__setattr__(self, "occupancy", occupancy)
        # tuple(...) of already-frozen WallSegment dataclasses -- the outer
        # container can no longer be appended/cleared, and each element was
        # already immutable on its own.
        object.__setattr__(self, "wall_segments", tuple(self.wall_segments))
        # _deep_freeze recursively copies+freezes topology_metadata AND
        # every nested Mapping/list/tuple/ndarray inside it (never aliases
        # anything the caller passed in) -- world.topology_metadata["x"] = 1
        # raises TypeError, and world.topology_metadata["room_cells"].append(...)
        # raises AttributeError (it's a tuple, whether the caller passed a
        # tuple or a plain list).
        object.__setattr__(self, "topology_metadata", _deep_freeze(self.topology_metadata))
