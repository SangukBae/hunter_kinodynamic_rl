"""Research benchmark-set coverage and basic fixed-layout validity."""

import math

from hunter_kinodynamic_rl.env.scenarios.benchmark_loader import load_benchmark
from hunter_kinodynamic_rl.env.scenarios.procedural_generator import is_reachable


def test_current_simulation_benchmarks_have_multiple_distinct_scenarios():
    minimum_counts = {"id": 5, "ood_geometry": 4, "dynamic": 4}
    for name, minimum in minimum_counts.items():
        scenarios = load_benchmark(name)
        assert len(scenarios) >= minimum
        assert len({s.scenario_id for s in scenarios}) == len(scenarios)
        assert len({s.spec.seed for s in scenarios}) == len(scenarios)


def test_all_current_fixed_layouts_are_reachable_and_start_goal_clear():
    world_size_m = 12.0
    robot_radius_m = 0.45
    for name in ("id", "ood_geometry", "dynamic"):
        for scenario in load_benchmark(name):
            spec = scenario.spec
            assert is_reachable(
                (spec.start_x, spec.start_y), (spec.goal_x, spec.goal_y),
                spec.static_obstacles, world_size_m, robot_radius_m,
            ), scenario.scenario_id
            for obs in spec.static_obstacles:
                assert math.hypot(spec.start_x - obs.x, spec.start_y - obs.y) > robot_radius_m + obs.radius
                assert math.hypot(spec.goal_x - obs.x, spec.goal_y - obs.y) > robot_radius_m + obs.radius
            for obs in spec.dynamic_obstacles:
                assert math.hypot(spec.start_x - obs.x0, spec.start_y - obs.y0) > robot_radius_m + obs.radius
                assert math.hypot(spec.goal_x - obs.x0, spec.goal_y - obs.y0) > robot_radius_m + obs.radius
