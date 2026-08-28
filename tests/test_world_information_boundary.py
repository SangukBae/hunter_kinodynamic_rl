"""Plan section 3.2's information boundary, Phase 3 slice: a
``LongHorizonWorld``'s privileged fields (ground-truth ``occupancy``,
``topology_metadata``, ``shortest_path_length_m``) must never be reachable
from anything this package hands to a POLICY observation builder.

This package has no learned Global RL yet (Phase 3 explicitly excludes it --
see the plan doc), so there is no policy-observation function to call
directly; instead this module asserts the boundary the way it is actually
enforceable today:

1. The only Phase 1/2 policy-facing map/observation surfaces
   (``navigation.mapping.partial_map.PartialMap.channels()``,
   ``navigation.local_rl.controller.LocalPolicyController``) take no
   ``LongHorizonWorld``/``WallSegment`` argument anywhere in their public
   API -- there is no code path by which Phase 3 privileged data could even
   be threaded into them.
2. ``LongHorizonWorld``/``WallSegment`` themselves carry no method that
   returns a policy-shaped observation -- they are inert data, and their
   privileged fields are named/typed distinctly from every Phase 1/2
   observation contract (``ROBOT_STATE_DIM_*``, ``PartialMap.channels()``'s
   ``MapChannels``).
3. ``topology_metadata`` is JSON/dict-shaped free-form privileged data --
   asserted to never collide with (be indistinguishable from) an
   observation array, and never accidentally embedded inside
   ``occupancy``/``wall_segments`` (the two fields a spawner/mapping
   integration WOULD plausibly touch).
"""

import inspect
from types import MappingProxyType

import numpy as np
import pytest

from hunter_kinodynamic_rl.config.schema import LongHorizonWorldConfig
from hunter_kinodynamic_rl.env.scenarios.long_horizon_generator import generate_long_horizon_world
from hunter_kinodynamic_rl.env.scenarios.long_horizon_world import LongHorizonWorld, WallSegment
from hunter_kinodynamic_rl.navigation.local_rl.controller import LocalPolicyController
from hunter_kinodynamic_rl.navigation.mapping.partial_map import MapChannels, PartialMap


def _world():
    cfg = LongHorizonWorldConfig(
        enabled=True, size_m=16.0, resolution_m=0.25,
        corridor_width_min_m=1.5, corridor_width_max_m=2.5, wall_thickness_m=0.15,
        room_count_range=[1, 3], dead_end_count_range=[1, 3], loop_count_range=[1, 2],
        alternative_route_min_count=1, start_goal_geodesic_min_m=6.0,
        require_ackermann_feasibility=True, generation_attempt_limit=100,
    )
    return generate_long_horizon_world(seed=0, cfg=cfg, robot_radius_m=0.3, min_turning_radius_m=0.8, wheelbase_m=0.65)


def test_topology_metadata_is_privileged_and_structurally_separate_from_occupancy():
    import collections.abc

    world = _world()
    # topology_metadata is a read-only Mapping (types.MappingProxyType, see
    # test_world_information_boundary.py's own mutation-attempt tests
    # below), never an ndarray -- it cannot be silently concatenated into a
    # numeric observation array the way a stray extra channel could.
    assert isinstance(world.topology_metadata, collections.abc.Mapping)
    assert not isinstance(world.topology_metadata, np.ndarray)
    # None of the privileged keys leak into the occupancy array's own
    # dtype/shape metadata, and occupancy carries no side-channel attribute
    # smuggling topology data onto the array itself.
    assert world.occupancy.dtype == np.bool_
    assert not hasattr(world.occupancy, "topology_metadata")


def test_wall_segment_has_no_privileged_topology_fields():
    world = _world()
    assert world.wall_segments, "expected at least one wall segment"
    seg_fields = set(vars(world.wall_segments[0]).keys()) if not hasattr(world.wall_segments[0], "__dataclass_fields__") \
        else set(world.wall_segments[0].__dataclass_fields__.keys())
    privileged_only = {"dead_end_count", "loop_count", "alternative_route_count", "room_cells", "tree_edges"}
    assert seg_fields.isdisjoint(privileged_only)


def test_longhorizonworld_is_a_frozen_dataclass_with_no_observation_method():
    # No method on LongHorizonWorld/WallSegment returns anything
    # observation-shaped (e.g. "to_observation", "policy_state") -- a Phase
    # 4 Global RL integration must build its own explicit, reviewed
    # observation extractor rather than finding a convenience method here
    # that silently forwards privileged fields.
    for cls in (LongHorizonWorld, WallSegment):
        assert cls.__dataclass_params__.frozen is True
        forbidden_prefixes = ("to_observation", "as_observation", "policy_state", "to_policy")
        for name, _ in inspect.getmembers(cls, predicate=callable):
            assert not name.startswith(forbidden_prefixes)


def test_partial_map_channels_signature_takes_no_long_horizon_world_argument():
    """The only Phase 1/2 policy-facing map surface -- PartialMap.channels()
    -- has no parameter through which a LongHorizonWorld/its privileged
    topology_metadata could be passed in."""
    sig = inspect.signature(PartialMap.channels)
    assert list(sig.parameters.keys()) == ["self"]
    channel_fields = set(MapChannels.__dataclass_fields__.keys())
    privileged_fields = set(LongHorizonWorld.__dataclass_fields__.keys()) - {"occupancy", "resolution_m"}
    assert channel_fields.isdisjoint(privileged_fields)


def test_local_policy_controller_build_observation_has_no_world_or_topology_parameter():
    """LocalPolicyController.build_observation is the only Phase 2
    observation-assembly entrypoint in this package -- confirm its
    parameter list has no ``world``/``topology``/``occupancy`` slot a
    caller could (even by mistake) route privileged Phase 3 data through."""
    sig = inspect.signature(LocalPolicyController.build_observation)
    forbidden_substrings = ("world", "topology", "shortest_path", "occupancy", "wall_segment")
    for name in sig.parameters:
        lowered = name.lower()
        assert not any(f in lowered for f in forbidden_substrings), (
            f"LocalPolicyController.build_observation has a suspicious parameter {name!r} -- "
            "verify it cannot carry LongHorizonWorld privileged data"
        )


# --------------------------------------------------------------- real mutability (code review)
# frozen=True alone only blocks REASSIGNING a field (world.occupancy = ...
# raises) -- it says nothing about mutating an already-assigned MUTABLE
# field's own contents in place. These tests catch exactly that regression
# class, which test_longhorizonworld_is_a_frozen_dataclass_with_no_observation_method
# above (checking __dataclass_params__.frozen only) cannot.
def test_occupancy_array_rejects_in_place_mutation():
    world = _world()
    assert world.occupancy.flags.writeable is False
    with pytest.raises(ValueError):
        world.occupancy[0, 0] = True


def test_occupancy_array_is_a_copy_not_an_alias_of_the_caller_supplied_array():
    from hunter_kinodynamic_rl.env.scenarios.long_horizon_world import LongHorizonWorld

    caller_array = np.zeros((4, 4), dtype=bool)
    world = LongHorizonWorld(seed=0, occupancy=caller_array, resolution_m=0.25, origin_xy=(0.0, 0.0))
    caller_array[0, 0] = True  # mutate the ORIGINAL after construction
    assert not world.occupancy[0, 0]  # world's own copy is unaffected


def test_wall_segments_is_a_tuple_with_no_append_or_clear():
    world = _world()
    assert isinstance(world.wall_segments, tuple)
    assert not hasattr(world.wall_segments, "append")
    assert not hasattr(world.wall_segments, "clear")


def test_topology_metadata_rejects_item_assignment():
    world = _world()
    assert isinstance(world.topology_metadata, MappingProxyType)
    with pytest.raises(TypeError):
        world.topology_metadata["injected"] = 1


def test_topology_metadata_is_a_copy_not_an_alias_of_the_caller_supplied_dict():
    from hunter_kinodynamic_rl.env.scenarios.long_horizon_world import LongHorizonWorld

    caller_meta = {"room_count": 3}
    world = LongHorizonWorld(seed=0, occupancy=np.zeros((2, 2), dtype=bool), resolution_m=0.25,
                              origin_xy=(0.0, 0.0), topology_metadata=caller_meta)
    caller_meta["room_count"] = 999  # mutate the ORIGINAL dict after construction
    assert world.topology_metadata["room_count"] == 3  # world's own copy is unaffected


def test_topology_metadata_room_cells_is_a_tuple_not_a_list():
    world = _world()
    assert isinstance(world.topology_metadata["room_cells"], tuple)


# --------------------------------------------------------------- nested immutability (code review, round 2)
# Round 1's __post_init__ only wrapped topology_metadata's TOP-LEVEL dict in
# MappingProxyType -- a NESTED mutable value supplied by an external caller
# (not the generator, which happened to always pass a tuple already) was
# still mutable in place. These tests construct LongHorizonWorld directly
# with deliberately mutable nested containers to confirm _deep_freeze
# catches them regardless of what the caller passes in.
def test_topology_metadata_nested_list_from_an_external_caller_is_frozen_to_a_tuple():
    from hunter_kinodynamic_rl.env.scenarios.long_horizon_world import LongHorizonWorld

    world = LongHorizonWorld(
        seed=0, occupancy=np.zeros((2, 2), dtype=bool), resolution_m=0.25, origin_xy=(0.0, 0.0),
        topology_metadata={"room_cells": [1, 2]},  # a plain list, not the generator's own tuple
    )
    assert isinstance(world.topology_metadata["room_cells"], tuple)
    with pytest.raises(AttributeError):
        world.topology_metadata["room_cells"].append(3)


def test_topology_metadata_nested_dict_from_an_external_caller_rejects_item_assignment():
    from types import MappingProxyType

    from hunter_kinodynamic_rl.env.scenarios.long_horizon_world import LongHorizonWorld

    world = LongHorizonWorld(
        seed=0, occupancy=np.zeros((2, 2), dtype=bool), resolution_m=0.25, origin_xy=(0.0, 0.0),
        topology_metadata={"nested": {"a": [1, 2, 3]}},
    )
    assert isinstance(world.topology_metadata["nested"], MappingProxyType)
    with pytest.raises(TypeError):
        world.topology_metadata["nested"]["a"] = 99
    # And the doubly-nested list is ALSO frozen, not just the dict wrapping it.
    assert isinstance(world.topology_metadata["nested"]["a"], tuple)


def test_topology_metadata_nested_list_mutation_by_the_caller_after_construction_does_not_leak():
    from hunter_kinodynamic_rl.env.scenarios.long_horizon_world import LongHorizonWorld

    caller_list = [1, 2]
    world = LongHorizonWorld(
        seed=0, occupancy=np.zeros((2, 2), dtype=bool), resolution_m=0.25, origin_xy=(0.0, 0.0),
        topology_metadata={"room_cells": caller_list},
    )
    caller_list.append(999)  # mutate the ORIGINAL list after construction
    assert world.topology_metadata["room_cells"] == (1, 2)  # world's own frozen copy is unaffected


def test_generated_world_privileged_fields_are_not_accidentally_equal_to_public_fields():
    """Sanity check that the privileged/public split is real data, not two
    names for the same object (which would make the split meaningless)."""
    world = _world()
    assert world.topology_metadata is not world.occupancy
    assert "occupancy" not in world.topology_metadata
    assert "wall_segments" not in world.topology_metadata
