"""Load FIXED benchmark scenarios (section 32/33) from
``config/benchmarks/<id|ood_geometry|ood_dynamics|dynamic>/*.yaml`` -- every
baseline is evaluated against the EXACT same scenario files, never a
freshly-regenerated random draw."""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import Dict, List

import yaml

from hunter_kinodynamic_rl.config.loader import default_config_root
from hunter_kinodynamic_rl.env.scenarios.procedural_generator import (
    DynamicObstacleSpec, ScenarioSpec, StaticObstacle,
)


@dataclass(frozen=True)
class BenchmarkScenario:
    scenario_id: str
    spec: ScenarioSpec
    dynamics_overrides: Dict = field(default_factory=dict)
    sensor_overrides: Dict = field(default_factory=dict)


def _scenario_from_dict(scenario_id: str, data: dict) -> BenchmarkScenario:
    start = data["start"]
    goal = data["goal"]
    static_obstacles = [StaticObstacle(**o) for o in data.get("static_obstacles", [])]
    dynamic_obstacles = [DynamicObstacleSpec(**o) for o in data.get("moving_obstacles", [])]
    spec = ScenarioSpec(
        seed=int(data["seed"]),
        start_x=float(start["x"]), start_y=float(start["y"]), start_yaw=float(start.get("yaw", 0.0)),
        goal_x=float(goal["x"]), goal_y=float(goal["y"]),
        static_obstacles=static_obstacles, dynamic_obstacles=dynamic_obstacles,
    )
    return BenchmarkScenario(
        scenario_id=scenario_id, spec=spec,
        dynamics_overrides=data.get("dynamics", {}), sensor_overrides=data.get("sensor", {}),
    )


def load_scenario_file(path: str) -> BenchmarkScenario:
    """Load ONE fixed scenario YAML by explicit path -- used by
    ``environment_node.py``'s ``scenario_override_path`` parameter (section
    6/11: an evaluation run places the EXACT benchmark obstacle layout in
    Gazebo instead of procedurally regenerating a scenario from a seed)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no such scenario file: {path}")
    with open(path) as f:
        data = yaml.safe_load(f)
    scenario_id = data.get("scenario_id", os.path.splitext(os.path.basename(path))[0])
    return _scenario_from_dict(scenario_id, data)


def load_benchmark(benchmark_name: str, config_root: str = None) -> List[BenchmarkScenario]:
    root = config_root or default_config_root()
    directory = os.path.join(root, "benchmarks", benchmark_name)
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"no such benchmark directory: {directory}")
    scenarios = []
    for path in sorted(glob.glob(os.path.join(directory, "*.yaml"))):
        with open(path) as f:
            data = yaml.safe_load(f)
        scenario_id = data.get("scenario_id", os.path.splitext(os.path.basename(path))[0])
        scenarios.append(_scenario_from_dict(scenario_id, data))
    return scenarios
