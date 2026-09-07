#!/usr/bin/env python3
"""Build and materialize the frozen static/dynamic TRACTOR scenario plan."""

from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable

import numpy as np
import yaml

from hunter_kinodynamic_rl.config.loader import default_config_root, load_profile
from hunter_kinodynamic_rl.env.scenarios.ackermann_feasibility import is_ackermann_feasible
from hunter_kinodynamic_rl.env.scenarios.procedural_generator import StaticObstacle


PLAN_SCHEMA_ID = "tractor_scenario_plan_v1"
MATERIALIZED_SCHEMA_ID = "tractor_materialized_scenarios_v1"


def _canonical_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _read_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a top-level mapping")
    return value


def _round(value: float) -> float:
    return round(float(value), 6)


def _obstacles(points: Iterable[tuple[float, float, float]], dx: float, dy: float, scale: float):
    return [
        {"x": _round(x + dx), "y": _round(y + dy), "radius": _round(radius * scale)}
        for x, y, radius in points
    ]


def _template_geometry(family_id: str, seed: int, obstacle_contract: str) -> dict:
    family_seed = int(hashlib.sha256(family_id.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng((int(seed) << 32) ^ family_seed)
    dx, dy = rng.uniform(-0.15, 0.15, size=2)
    scale = float(rng.uniform(0.94, 1.06))
    speed_scale = float(rng.uniform(0.9, 1.1))
    straight_start = (-5.5 + dx, 0.0 + dy)
    straight_goal = (5.5 + dx, 0.0 + dy)
    static_points: list[tuple[float, float, float]] = []
    moving = []
    start, goal, yaw = straight_start, straight_goal, 0.0

    if family_id == "open_static":
        static_points = [(-1.5, 3.5, 0.35), (1.5, -3.5, 0.35)]
    elif family_id == "corridor_static":
        static_points = [(x, y, 0.42) for x in (-4.5, -2.25, 0.0, 2.25, 4.5) for y in (-2.2, 2.2)]
    elif family_id == "corner_static":
        start, goal, yaw = (-5.2 + dx, -4.5 + dy), (4.8 + dx, 4.8 + dy), 0.25
        static_points = [
            (0.0, -2.0, 0.38), (0.0, 0.0, 0.38),
            (0.0, 1.8, 0.38), (2.0, 1.8, 0.38),
        ]
    elif family_id == "choke_static":
        static_points = [(0.0, y, 0.46) for y in (-5.2, -3.1, 3.1, 5.2)]
    elif family_id == "clutter_static":
        static_points = [
            (-3.8, 2.2, 0.38), (-2.4, -2.0, 0.42), (-0.8, 2.7, 0.35),
            (0.9, -2.6, 0.43), (2.5, 2.0, 0.40), (4.0, -2.1, 0.36),
            (0.0, 4.8, 0.34), (0.0, -4.8, 0.34),
        ]
    elif family_id == "dead_end_static":
        static_points = [
            (6.2, -2.4, 0.36), (6.2, 0.0, 0.36), (6.2, 2.4, 0.36),
            (3.8, -2.4, 0.36), (3.8, 2.4, 0.36),
        ]
        goal = (4.8 + dx, 0.0 + dy)
    elif family_id == "crossing_dynamic":
        moving = [{"x0": _round(dx), "y0": _round(-3.6 + dy), "vx": 0.0,
                   "vy": _round(0.65 * speed_scale), "radius": 0.3}]
    elif family_id == "head_on_dynamic":
        moving = [{"x0": _round(3.2 + dx), "y0": _round(0.35 + dy),
                   "vx": _round(-0.6 * speed_scale), "vy": 0.0, "radius": 0.3}]
    elif family_id == "overtaking_dynamic":
        moving = [{"x0": _round(-2.4 + dx), "y0": _round(0.25 + dy),
                   "vx": _round(0.25 * speed_scale), "vy": 0.0, "radius": 0.3}]
    elif family_id == "occlusion_dynamic":
        static_points = [(-0.8, -1.3, 0.62), (-0.8, 1.3, 0.62)]
        moving = [{"x0": _round(0.45 + dx), "y0": _round(-3.8 + dy), "vx": 0.0,
                   "vy": _round(0.7 * speed_scale), "radius": 0.3}]
    elif family_id == "mixed_dynamic":
        static_points = [(-1.8, 2.7, 0.38), (1.8, -2.7, 0.38)]
        moving = [
            {"x0": _round(-0.8 + dx), "y0": _round(-3.8 + dy), "vx": 0.0,
             "vy": _round(0.65 * speed_scale), "radius": 0.3},
            {"x0": _round(3.6 + dx), "y0": _round(0.65 + dy),
             "vx": _round(-0.55 * speed_scale), "vy": 0.0, "radius": 0.3},
            {"x0": _round(-3.0 + dx), "y0": _round(-0.9 + dy),
             "vx": _round(0.35 * speed_scale), "vy": 0.0, "radius": 0.3},
        ]
    else:
        raise ValueError(f"no executable geometry template for family {family_id!r}")
    if obstacle_contract == "static_only" and moving:
        raise ValueError(f"static family {family_id!r} generated moving obstacles")
    if obstacle_contract == "dynamic" and not moving:
        raise ValueError(f"dynamic family {family_id!r} generated no moving obstacles")
    static = _obstacles(static_points, dx, dy, scale)
    return {
        "start": {"x": _round(start[0]), "y": _round(start[1]), "yaw": _round(yaw)},
        "goal": {"x": _round(goal[0]), "y": _round(goal[1])},
        "static_obstacles": static,
        "moving_obstacles": moving,
        "dynamics": {}, "sensor": {},
    }


def _geometry_sha256(geometry: dict) -> str:
    return _canonical_sha256({
        key: geometry[key]
        for key in ("start", "goal", "static_obstacles", "moving_obstacles", "dynamics", "sensor")
    })


def _validate_geometry(geometry: dict, *, world_size_m: float, profile) -> None:
    robot = profile.robot
    start, goal = geometry["start"], geometry["goal"]
    half = world_size_m / 2.0
    for point_name, point in (("start", start), ("goal", goal)):
        if max(abs(float(point["x"])), abs(float(point["y"]))) + robot.collision_radius_m >= half:
            raise ValueError(f"{point_name} violates the robot footprint boundary inset")
    static = [StaticObstacle(**item) for item in geometry["static_obstacles"]]
    initial_dynamic = [
        StaticObstacle(x=float(item["x0"]), y=float(item["y0"]), radius=float(item["radius"]))
        for item in geometry["moving_obstacles"]
    ]
    obstacles = static + initial_dynamic
    for obstacle in obstacles:
        if max(abs(obstacle.x), abs(obstacle.y)) + obstacle.radius > half:
            raise ValueError("obstacle footprint leaves the declared world")
    if not is_ackermann_feasible(
        float(start["x"]), float(start["y"]), float(start["yaw"]),
        float(goal["x"]), float(goal["y"]), profile.reward.goal_threshold_m,
        obstacles, world_size_m, robot.collision_radius_m,
        1.0 / robot.max_curvature, robot.wheelbase_m,
        max_expansions=50000,
    ):
        raise ValueError("scenario failed the registered bounded Ackermann feasibility filter")


def _load_plan_inputs(config_root: str | None = None):
    config_root_path = Path(config_root or default_config_root())
    tractor_root = config_root_path / "tractor"
    plan = _read_yaml(tractor_root / "scenario_plan.yaml")
    required = {
        "schema_id", "plan_version", "protocol_version", "generator_version", "world_size_m",
        "expected_manifest_sha256", "splits", "static_manifest", "dynamic_manifest",
        "split_group_key", "require_unique_seed_across_splits",
        "require_unique_geometry_across_splits", "require_ackermann_feasibility",
    }
    if set(plan) != required or plan["schema_id"] != PLAN_SCHEMA_ID:
        raise ValueError("scenario_plan.yaml fields or schema are incompatible")
    static_manifest = _read_yaml(tractor_root / str(plan["static_manifest"]))
    dynamic_manifest = _read_yaml(tractor_root / str(plan["dynamic_manifest"]))
    for manifest, expected_contract in ((static_manifest, "static_only"), (dynamic_manifest, "dynamic")):
        if manifest.get("schema_id") != "tractor_scenario_manifest_v1":
            raise ValueError("scenario family manifest schema is incompatible")
        if not manifest.get("scenarios"):
            raise ValueError("scenario family manifest is empty")
        for family in manifest["scenarios"]:
            if expected_contract == "static_only" and family.get("obstacle_motion") != "none":
                raise ValueError("static manifest contains a moving family")
            if expected_contract == "dynamic" and family.get("obstacle_motion") == "none":
                raise ValueError("dynamic manifest contains a static family")
    return config_root_path, plan, static_manifest, dynamic_manifest


def _build_plan_payload(config_root: str | None = None, *, validate_feasibility: bool = True) -> dict:
    config_root_path, plan, static_manifest, dynamic_manifest = _load_plan_inputs(config_root)
    if plan["split_group_key"] != "scenario_geometry_sha256":
        raise ValueError("split group key must be the exact scenario geometry hash")
    profile = load_profile("tractor_local_dynamic", str(config_root_path))
    families = [
        (item, "static_only") for item in static_manifest["scenarios"]
    ] + [
        (item, "dynamic") for item in dynamic_manifest["scenarios"]
    ]
    expected_family_ids = {item["id"] for item, _contract in families}
    if len(expected_family_ids) != len(families):
        raise ValueError("scenario family ids must be globally unique")
    split_intervals = []
    entries = []
    used_seeds = set()
    geometries_by_split = {}
    for split_id in ("development", "calibration", "locked_test"):
        split = plan["splits"].get(split_id)
        if not isinstance(split, dict) or set(split) != {
            "seed_start", "seed_end", "instances_per_family", "permitted_use",
        }:
            raise ValueError(f"invalid split plan for {split_id}")
        start_seed, end_seed = int(split["seed_start"]), int(split["seed_end"])
        count = int(split["instances_per_family"])
        if start_seed > end_seed or count <= 0:
            raise ValueError(f"invalid seed/count contract for {split_id}")
        split_intervals.append((start_seed, end_seed, split_id))
        required_seeds = len(families) * count
        if start_seed + required_seeds - 1 > end_seed:
            raise ValueError(f"seed interval for {split_id} cannot hold every family instance")
        for family_index, (family, obstacle_contract) in enumerate(families):
            for instance in range(count):
                seed = start_seed + family_index * count + instance
                if seed in used_seeds:
                    raise ValueError(f"seed {seed} is reused across splits/families")
                used_seeds.add(seed)
                geometry = _template_geometry(str(family["id"]), seed, obstacle_contract)
                if validate_feasibility and plan["require_ackermann_feasibility"]:
                    try:
                        _validate_geometry(
                            geometry, world_size_m=float(plan["world_size_m"]), profile=profile,
                        )
                    except ValueError as error:
                        raise ValueError(
                            f"{split_id}/{family['id']}/seed={seed}: {error}"
                        ) from error
                geometry_sha = _geometry_sha256(geometry)
                previous_split = geometries_by_split.setdefault(geometry_sha, split_id)
                if previous_split != split_id:
                    raise ValueError(
                        f"geometry {geometry_sha} leaks across {previous_split} and {split_id}"
                    )
                scenario_id = f"{split_id}-{family['id']}-{instance:03d}"
                relative_path = f"{split_id}/{obstacle_contract}/{scenario_id}.yaml"
                entries.append({
                    "scenario_id": scenario_id, "family_id": str(family["id"]),
                    "topology": str(family["topology"]), "obstacle_motion": str(family["obstacle_motion"]),
                    "obstacle_contract": obstacle_contract, "split_id": split_id,
                    "permitted_use": str(split["permitted_use"]), "seed": seed,
                    "scenario_geometry_sha256": geometry_sha, "group_id": geometry_sha,
                    "relative_path": relative_path, "geometry": geometry,
                })
    intervals = sorted(split_intervals)
    for left, right in zip(intervals, intervals[1:]):
        if left[1] >= right[0]:
            raise ValueError(f"split seed intervals overlap: {left} and {right}")
    payload = {
        "schema_id": MATERIALIZED_SCHEMA_ID,
        "plan_version": plan["plan_version"],
        "protocol_version": plan["protocol_version"],
        "generator_version": plan["generator_version"],
        "world_size_m": float(plan["world_size_m"]),
        "entries": entries,
    }
    payload["manifest_sha256"] = _canonical_sha256(payload)
    return payload


@lru_cache(maxsize=8)
def build_scenario_plan(config_root: str | None = None, *, validate_feasibility: bool = True) -> dict:
    _root, plan, _static, _dynamic = _load_plan_inputs(config_root)
    payload = _build_plan_payload(config_root, validate_feasibility=validate_feasibility)
    if payload["manifest_sha256"] != plan["expected_manifest_sha256"]:
        raise ValueError(
            "materialized scenario manifest changed; create a new plan/protocol version "
            "instead of reusing locked-test identity"
        )
    return payload


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def materialize_scenario_plan(output_root: str | Path, config_root: str | None = None) -> dict:
    root = Path(output_root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty scenario root: {root}")
    root.mkdir(parents=True, exist_ok=True)
    payload = build_scenario_plan(config_root)
    manifest_entries = []
    for entry in payload["entries"]:
        scenario = {
            "schema_id": "tractor_fixed_scenario_v1",
            "protocol_version": payload["protocol_version"],
            "plan_version": payload["plan_version"],
            **{key: entry[key] for key in (
                "scenario_id", "family_id", "topology", "obstacle_motion", "obstacle_contract",
                "split_id", "permitted_use", "seed", "scenario_geometry_sha256", "group_id",
            )},
            **entry["geometry"],
        }
        serialized = yaml.safe_dump(scenario, sort_keys=False)
        destination = root / entry["relative_path"]
        _atomic_write(destination, serialized)
        artifact_sha = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        manifest_entries.append({
            key: entry[key] for key in entry if key not in {"geometry"}
        } | {"artifact_sha256": artifact_sha})
    artifact_manifest = {
        key: payload[key] for key in payload if key not in {"entries", "manifest_sha256"}
    }
    artifact_manifest["plan_manifest_sha256"] = payload["manifest_sha256"]
    artifact_manifest["entries"] = manifest_entries
    artifact_manifest["artifact_manifest_sha256"] = _canonical_sha256(artifact_manifest)
    _atomic_write(root / "manifest.json", json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n")
    return artifact_manifest


def validate_materialized_scenario_manifest(
    manifest_path: str | Path, config_root: str | None = None,
) -> dict:
    """Validate identity, confinement and byte hashes before data collection."""
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    stored_digest = manifest.get("artifact_manifest_sha256")
    digest_payload = dict(manifest)
    digest_payload.pop("artifact_manifest_sha256", None)
    if stored_digest != _canonical_sha256(digest_payload):
        raise ValueError("materialized scenario artifact manifest checksum mismatch")
    expected = build_scenario_plan(config_root, validate_feasibility=False)
    if manifest.get("plan_manifest_sha256") != expected["manifest_sha256"]:
        raise ValueError("materialized scenarios do not match the frozen scenario plan")
    expected_entries = {entry["scenario_id"]: entry for entry in expected["entries"]}
    observed_entries = manifest.get("entries")
    if not isinstance(observed_entries, list) or len(observed_entries) != len(expected_entries):
        raise ValueError("materialized scenario manifest has missing or extra entries")
    root = path.parent
    seen = set()
    for entry in observed_entries:
        scenario_id = str(entry.get("scenario_id", ""))
        if scenario_id in seen or scenario_id not in expected_entries:
            raise ValueError(f"duplicate or unregistered scenario {scenario_id!r}")
        seen.add(scenario_id)
        expected_entry = expected_entries[scenario_id]
        for field in (
            "family_id", "obstacle_contract", "split_id", "seed",
            "scenario_geometry_sha256", "group_id", "relative_path",
        ):
            if entry.get(field) != expected_entry[field]:
                raise ValueError(f"scenario {scenario_id!r} changed frozen field {field!r}")
        candidate = (root / str(entry["relative_path"])).resolve()
        if root != candidate and root not in candidate.parents:
            raise ValueError(f"scenario path escapes the materialized root: {candidate}")
        if not candidate.is_file():
            raise ValueError(f"missing materialized scenario file: {candidate}")
        artifact_digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if artifact_digest != entry.get("artifact_sha256"):
            raise ValueError(f"scenario artifact checksum mismatch: {scenario_id}")
        scenario = _read_yaml(candidate)
        if scenario.get("scenario_geometry_sha256") != expected_entry["scenario_geometry_sha256"]:
            raise ValueError(f"scenario file geometry identity mismatch: {scenario_id}")
        if _geometry_sha256(scenario) != expected_entry["scenario_geometry_sha256"]:
            raise ValueError(f"scenario file geometry content mismatch: {scenario_id}")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-root")
    parser.add_argument("--output-root")
    parser.add_argument("--skip-feasibility", action="store_true", help="development diagnostics only")
    args = parser.parse_args(argv)
    if args.output_root:
        result = materialize_scenario_plan(args.output_root, args.config_root)
        summary = {
            "ok": True, "materialized": True, "episode_count": len(result["entries"]),
            "plan_manifest_sha256": result["plan_manifest_sha256"],
            "artifact_manifest_sha256": result["artifact_manifest_sha256"],
            "output_root": str(Path(args.output_root).resolve()),
        }
    else:
        result = build_scenario_plan(
            args.config_root, validate_feasibility=not args.skip_feasibility,
        )
        split_counts = {}
        family_counts = {}
        for entry in result["entries"]:
            split_counts[entry["split_id"]] = split_counts.get(entry["split_id"], 0) + 1
            family_counts[entry["family_id"]] = family_counts.get(entry["family_id"], 0) + 1
        summary = {
            "ok": True, "materialized": False, "episode_count": len(result["entries"]),
            "split_counts": split_counts, "family_counts": family_counts,
            "plan_manifest_sha256": result["manifest_sha256"],
        }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
