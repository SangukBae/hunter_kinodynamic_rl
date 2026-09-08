#!/usr/bin/env python3
"""Places a scenario's static/dynamic obstacles into the running Gazebo
world via ``ros_gz_interfaces/SpawnEntity`` + ``DeleteEntity``, reusing
``drl_obstacle_assets`` catalog models for static obstacles (dependency,
not copied) with a plain-cylinder SDF fallback when the catalog is
unavailable or has no close-radius match.

Every spawn/delete call is BOUNDED (uses the calling node's
``GazeboRuntimeMixin._await_future``/``_wait_for_srv`` -- never a bare
fire-and-forget ``call_async`` with no result check, which was the bug
identified in section 3.3: obstacle placement must be CONFIRMED before the
episode is considered ready, exactly like the robot's own
``set_entity_pose_ignition``).
"""

from __future__ import annotations

from typing import List

from ros_gz_interfaces.msg import Entity as GzEntity
from ros_gz_interfaces.srv import DeleteEntity, SpawnEntity

from hunter_kinodynamic_rl.env.simulation.gazebo_service_wait import GazeboServiceError
from hunter_kinodynamic_rl.env.scenarios.procedural_generator import StaticObstacle
from hunter_kinodynamic_rl.env.spawning.obstacle_catalog import CatalogEntry, closest_entry

STATIC_ENTITY_PREFIX = "hkrl_static_"
DYNAMIC_ENTITY_PREFIX = "hkrl_dynamic_"


def _cylinder_sdf(model_name: str, radius: float, height: float = 1.0) -> str:
    return (
        '<sdf version="1.8">'
        f'<model name="{model_name}">'
        "<static>true</static>"
        '<link name="link">'
        '<collision name="collision">'
        f"<geometry><cylinder><radius>{radius:.3f}</radius><length>{height:.3f}</length></cylinder></geometry>"
        "</collision>"
        '<visual name="visual">'
        f"<geometry><cylinder><radius>{radius:.3f}</radius><length>{height:.3f}</length></cylinder></geometry>"
        "<material><ambient>0.7 0.25 0.2 1</ambient><diffuse>0.7 0.25 0.2 1</diffuse></material>"
        "</visual>"
        "</link>"
        "</model>"
        "</sdf>"
    )


def _box_sdf(model_name: str, length_m: float, width_m: float, height: float = 0.8) -> str:
    size = f"{length_m:.3f} {width_m:.3f} {height:.3f}"
    return (
        '<sdf version="1.8">'
        f'<model name="{model_name}"><static>true</static><link name="link">'
        f'<collision name="collision"><geometry><box><size>{size}</size></box></geometry></collision>'
        f'<visual name="visual"><geometry><box><size>{size}</size></box></geometry>'
        '<material><ambient>0.25 0.45 0.75 1</ambient><diffuse>0.25 0.45 0.75 1</diffuse></material>'
        '</visual></link></model></sdf>'
    )


def _l_shape_sdf(model_name: str, length_m: float, width_m: float, height: float = 0.8) -> str:
    arm = max(0.12, min(length_m, width_m) * 0.32)
    x_offset = -0.5 * length_m + 0.5 * arm
    y_offset = -0.5 * width_m + 0.5 * arm
    return (
        '<sdf version="1.8">'
        f'<model name="{model_name}"><static>true</static><link name="link">'
        f'<collision name="collision_x"><pose>0 {y_offset:.3f} 0 0 0 0</pose><geometry><box>'
        f'<size>{length_m:.3f} {arm:.3f} {height:.3f}</size></box></geometry></collision>'
        f'<visual name="visual_x"><pose>0 {y_offset:.3f} 0 0 0 0</pose><geometry><box>'
        f'<size>{length_m:.3f} {arm:.3f} {height:.3f}</size></box></geometry></visual>'
        f'<collision name="collision_y"><pose>{x_offset:.3f} 0 0 0 0 0</pose><geometry><box>'
        f'<size>{arm:.3f} {width_m:.3f} {height:.3f}</size></box></geometry></collision>'
        f'<visual name="visual_y"><pose>{x_offset:.3f} 0 0 0 0 0</pose><geometry><box>'
        f'<size>{arm:.3f} {width_m:.3f} {height:.3f}</size></box></geometry></visual>'
        '</link></model></sdf>'
    )


def _primitive_sdf(model_name: str, shape: str, radius: float, length_m: float, width_m: float) -> str:
    if shape == "l_shape" and length_m > 0.0 and width_m > 0.0:
        return _l_shape_sdf(model_name, length_m, width_m)
    if shape in ("box", "cart") and length_m > 0.0 and width_m > 0.0:
        return _box_sdf(model_name, length_m, width_m, height=0.65 if shape == "cart" else 0.8)
    return _cylinder_sdf(model_name, radius)


def _catalog_include_sdf(model_name: str, uri: str) -> str:
    return (
        '<sdf version="1.8">'
        f'<model name="{model_name}">'
        "<static>true</static>"
        f"<include><uri>{uri}</uri></include>"
        "</model>"
        "</sdf>"
    )


def _spawn_one(node, spawn_client, model_name: str, sdf: str, x: float, y: float, z: float, yaw: float) -> bool:
    srv_name = "spawn_entity"
    if not node._wait_for_srv(spawn_client, srv_name, f"spawn[{model_name}]"):
        raise GazeboServiceError(f"SpawnEntity service unavailable for {model_name}")
    req = SpawnEntity.Request()
    req.entity_factory.name = model_name
    req.entity_factory.allow_renaming = False
    req.entity_factory.sdf = sdf
    req.entity_factory.pose.position.x, req.entity_factory.pose.position.y = float(x), float(y)
    req.entity_factory.pose.position.z = float(z)
    import math
    req.entity_factory.pose.orientation.z = math.sin(yaw / 2.0)
    req.entity_factory.pose.orientation.w = math.cos(yaw / 2.0)
    future = spawn_client.call_async(req)
    result = node._await_future(future, timeout=node._gz_call_timeout_sec, op=f"spawn[{model_name}]")
    # section P0-3: a PARTIAL obstacle-spawn failure (this docstring's own
    # stated intent: "obstacle placement must be CONFIRMED before the
    # episode is considered ready") must FAIL the reset, never silently
    # continue with fewer obstacles than the scenario/seed actually
    # specifies -- that would silently corrupt the seeded scenario's
    # reproducibility guarantee (a "seed 1710" episode would no longer
    # deterministically mean the same obstacle layout every time it's
    # replayed).
    if result is None:
        raise GazeboServiceError(f"spawn[{model_name}]: no response within {node._gz_call_timeout_sec:.1f}s")
    if not result.success:
        raise GazeboServiceError(f"spawn[{model_name}]: request completed but reported success=false")
    return True


def spawn_static_obstacles(
    node, spawn_client, obstacles: List[StaticObstacle], catalog: List[CatalogEntry], rng,
) -> int:
    """Returns the number spawned (== len(obstacles) always -- _spawn_one
    now RAISES GazeboServiceError on any failure instead of returning
    False, so a partial failure propagates out of this loop immediately
    rather than silently under-spawning the scenario). Each obstacle
    prefers a catalog entry whose radius is closest to its own; falls back
    to a plain cylinder when no catalog entry is available."""
    spawned = 0
    for i, obstacle in enumerate(obstacles):
        model_name = f"{STATIC_ENTITY_PREFIX}{i}"
        entry = closest_entry(catalog, obstacle.radius) if obstacle.shape == "cylinder" else None
        yaw = obstacle.yaw_rad
        if entry is not None and entry.yaw_random:
            yaw = float(rng.uniform(-3.14159, 3.14159))
        sdf = (
            _catalog_include_sdf(model_name, entry.uri) if entry is not None
            else _primitive_sdf(
                model_name, obstacle.shape, obstacle.radius, obstacle.length_m, obstacle.width_m,
            )
        )
        _spawn_one(node, spawn_client, model_name, sdf, obstacle.x, obstacle.y, 0.0, yaw)
        spawned += 1
    return spawned


def spawn_dynamic_obstacle_marker(
    node, spawn_client, index: int, radius: float, x: float, y: float,
    shape: str = "cylinder", length_m: float = 0.0, width_m: float = 0.0, yaw_rad: float = 0.0,
) -> bool:
    """Dynamic obstacles use a plain cylinder marker (moved every tick via
    set_entity_pose_ignition, not re-spawned) -- catalog meshes are static-
    only in this package (section 26: box/cylinder preferred for dynamic
    obstacles regardless)."""
    model_name = f"{DYNAMIC_ENTITY_PREFIX}{index}"
    sdf = _primitive_sdf(model_name, shape, radius, length_m, width_m)
    return _spawn_one(node, spawn_client, model_name, sdf, x, y, 0.0, yaw_rad)


def delete_entities(node, delete_client, names: List[str]) -> None:
    """section P0-3: a failed deletion must FAIL the reset, never silently
    leave a PREVIOUS episode's obstacle in the world -- an un-cleared
    entity would contaminate the NEW episode's scenario (an extra,
    unaccounted-for obstacle the seed/scenario spec never placed there),
    corrupting both the collision/risk computation and the reproducibility
    guarantee the same way an incomplete spawn would."""
    srv_name = "delete_entity"
    if not names:
        return
    if not node._wait_for_srv(delete_client, srv_name, "delete"):
        raise GazeboServiceError(
            f"{srv_name}: service unavailable -- cannot clear {len(names)} previous-episode obstacle(s)"
        )
    for name in names:
        req = DeleteEntity.Request()
        req.entity.name = name
        # Ignition resolves (name, type), not name alone.  The default type
        # is NONE (0), which produced `Entity ... type [0] not found` while
        # the MODEL remained in the world; the service could still report
        # success, making the old confirmation check a false positive.
        req.entity.type = GzEntity.MODEL
        future = delete_client.call_async(req)
        result = node._await_future(future, timeout=node._gz_call_timeout_sec, op=f"delete[{name}]")
        if result is None:
            raise GazeboServiceError(f"delete[{name}]: no response within {node._gz_call_timeout_sec:.1f}s")
        if not result.success:
            raise GazeboServiceError(f"delete[{name}]: request completed but reported success=false")
