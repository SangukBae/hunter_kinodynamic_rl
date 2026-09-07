#!/usr/bin/env python3
"""Materialize deterministic, disjoint fixed YAML suites for env v2."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

import yaml

from hunter_kinodynamic_rl.config.loader import default_config_root, load_profile
from hunter_kinodynamic_rl.env.scenarios.tractor_environment_v2 import generate_v2_scenario


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scenario_payload(scenario, scenario_id: str, dynamics: dict) -> dict:
    return {
        "scenario_id": scenario_id,
        "seed": scenario.seed,
        "start": {"x": scenario.start_x, "y": scenario.start_y, "yaw": scenario.start_yaw},
        "goal": {"x": scenario.goal_x, "y": scenario.goal_y},
        "static_obstacles": [dataclasses.asdict(item) for item in scenario.static_obstacles],
        "moving_obstacles": [dataclasses.asdict(item) for item in scenario.dynamic_obstacles],
        "dynamics": dynamics,
        "environment_metadata": {
            "environment_version": scenario.environment_version,
            "curriculum_level": scenario.curriculum_level,
            "topology": scenario.topology,
            "conflict_obstacle_count": scenario.conflict_obstacle_count,
        },
    }


def materialize(config_root: str | None = None, output_root: str | None = None) -> dict:
    root = Path(config_root or default_config_root())
    plan_path = root / "environment_v2" / "evaluation_suites.yaml"
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8")) or {}
    if plan.get("schema_id") != "tractor_environment_evaluation_suites_v2":
        raise ValueError("invalid environment-v2 evaluation suite schema")
    profile = load_profile(str(plan.get("base_profile", "tractor_local_dynamic_v2")), str(root))
    destination = Path(output_root) if output_root else root / "benchmarks"
    manifest = {
        "schema_id": "tractor_environment_benchmark_manifest_v2",
        "plan_sha256": _sha256(plan_path),
        "suites": {},
    }
    for suite_name, suite in plan["suites"].items():
        lo, hi = [int(value) for value in suite["seeds"]]
        if hi - lo + 1 != int(plan["scenarios_per_suite"]):
            raise ValueError(f"{suite_name}: seed range does not match scenarios_per_suite")
        scenario_cfg = dataclasses.replace(profile.scenario, world_size_m=float(suite["world_size_m"]))
        env_cfg = dataclasses.replace(
            profile.environment_v2,
            curriculum_static_limits=list(profile.environment_v2.curriculum_static_limits[:-1])
            + [int(suite["max_static_obstacles"])],
            curriculum_dynamic_counts=list(profile.environment_v2.curriculum_dynamic_counts[:-1])
            + [int(suite["dynamic_obstacle_count"])],
            curriculum_speed_ratio_caps=list(profile.environment_v2.curriculum_speed_ratio_caps[:-1])
            + [float(suite["speed_ratio_range"][1])],
            dynamic_speed_ratio_range=[float(v) for v in suite["speed_ratio_range"]],
            topology_families=list(suite.get("topology_families", profile.environment_v2.topology_families)),
            motion_patterns=list(suite.get("motion_patterns", profile.environment_v2.motion_patterns)),
        )
        env_cfg.validate()
        suite_dir = destination / suite_name
        suite_dir.mkdir(parents=True, exist_ok=True)
        files = []
        dynamics_cycle = suite.get("dynamics_cycle") or [{}]
        for index, seed in enumerate(range(lo, hi + 1), start=1):
            scenario = generate_v2_scenario(
                seed, scenario_cfg, env_cfg, profile.robot, profile.start_pose,
                profile.reward.goal_threshold_m, episode_index=0, mode="test",
            )
            scenario_id = f"{suite_name}_{index:03d}"
            path = suite_dir / f"scenario_{index:03d}.yaml"
            payload = _scenario_payload(scenario, scenario_id, dict(dynamics_cycle[(index - 1) % len(dynamics_cycle)]))
            path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
            files.append({"file": path.name, "seed": seed, "sha256": _sha256(path)})
        manifest["suites"][suite_name] = {
            "world_size_m": float(suite["world_size_m"]),
            "files": files,
        }
    manifest_path = destination / "environment_v2_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-root")
    parser.add_argument("--output-root")
    args = parser.parse_args(argv)
    manifest = materialize(args.config_root, args.output_root)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
