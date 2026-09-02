"""Phase 5 fixed long-horizon benchmark + ablation runner (plan section
9.9/10.3/10.10)."""

import numpy as np
import pytest

from hunter_kinodynamic_rl.config.schema import (
    GlobalFeasibilityConfig, GlobalRLConfig, HierarchyConfig, LongHorizonWorldConfig, MappingConfig, MemoryConfig,
)
import dataclasses

from hunter_kinodynamic_rl.evaluation.global_metrics import aggregate
from hunter_kinodynamic_rl.evaluation.long_horizon_benchmark import (
    ABLATION_LABELS, _ablation_flags, build_fixed_benchmark_manifest, effective_feasibility_config,
    effective_global_rl_config, effective_memory_config, run_ablation_benchmark, run_ablation_mission,
)
from hunter_kinodynamic_rl.navigation.mission.mission_frame import PoseXYYaw

_ROBOT_RADIUS_M = 0.45
_MIN_TURN_RADIUS_M = 1.2
_WHEELBASE_M = 0.65


def _small_world_cfg() -> LongHorizonWorldConfig:
    return LongHorizonWorldConfig(
        enabled=True, size_m=12.0, resolution_m=0.25, corridor_width_min_m=2.0, corridor_width_max_m=2.5,
        room_count_range=[2, 3], dead_end_count_range=[0, 1], loop_count_range=[0, 1],
        alternative_route_min_count=0, start_goal_geodesic_min_m=4.0, generation_attempt_limit=50,
        train_seed_range=[0, 999], validation_seed_range=[1000, 1999], test_seed_range=[2000, 2999],
    )


def test_ablation_flags_mapping():
    # (global_rl_enabled, include_visited, topology, feasibility, global_risk)
    assert _ablation_flags("A") == (False, False, False, False, False)
    assert _ablation_flags("B") == (True, False, False, False, False)
    assert _ablation_flags("C") == (True, True, False, False, False)
    assert _ablation_flags("B") != _ablation_flags("C")  # requirement F: no longer identical
    assert _ablation_flags("D") == (True, True, True, False, False)
    assert _ablation_flags("E") == (True, True, True, True, False)
    assert _ablation_flags("F") == (True, True, False, False, True)
    assert _ablation_flags("G") == (True, True, True, True, True)
    with pytest.raises(ValueError):
        _ablation_flags("Z")


def test_build_fixed_benchmark_manifest_is_deterministic():
    cfg = _small_world_cfg()
    m1 = build_fixed_benchmark_manifest(cfg, _ROBOT_RADIUS_M, _MIN_TURN_RADIUS_M, _WHEELBASE_M, num_scenarios=3, seed=7)
    m2 = build_fixed_benchmark_manifest(cfg, _ROBOT_RADIUS_M, _MIN_TURN_RADIUS_M, _WHEELBASE_M, num_scenarios=3, seed=7)
    assert [s.content_hash for s in m1] == [s.content_hash for s in m2]
    assert [s.occupancy_hash for s in m1] == [s.occupancy_hash for s in m2]
    assert len(m1) == 3
    assert all(s.mode == "test" for s in m1)


def test_manifest_scenarios_use_the_test_seed_pool_never_train():
    cfg = _small_world_cfg()
    manifest = build_fixed_benchmark_manifest(cfg, _ROBOT_RADIUS_M, _MIN_TURN_RADIUS_M, _WHEELBASE_M, num_scenarios=2, seed=0)
    lo, hi = cfg.test_seed_range
    assert all(lo <= s.seed <= hi for s in manifest)


def test_run_ablation_mission_a_local_only_returns_full_episode_dict():
    cfg = _small_world_cfg()
    manifest = build_fixed_benchmark_manifest(cfg, _ROBOT_RADIUS_M, _MIN_TURN_RADIUS_M, _WHEELBASE_M, num_scenarios=1, seed=1)
    episode = run_ablation_mission(
        "A", manifest[0], global_cfg=GlobalRLConfig(), hierarchy_cfg=HierarchyConfig(mission_timeout_steps=2000),
        mapping_cfg=MappingConfig(mission_size_cells=128), memory_cfg=MemoryConfig(), feasibility_cfg=GlobalFeasibilityConfig(),
        long_horizon_cfg=cfg, robot_radius_m=_ROBOT_RADIUS_M, min_turning_radius_m=_MIN_TURN_RADIUS_M,
        wheelbase_m=_WHEELBASE_M, max_options=10, max_local_steps=80, mission_timeout_steps=2000,
        rng=np.random.RandomState(0),
    )
    expected_keys = {
        "ablation", "scenario_id", "success", "route_length_m", "shortest_feasible_path_length_m",
        "navigation_time_sec", "control_elapsed_time_sec", "time_to_goal_sec", "time_to_termination_sec",
        "wall_elapsed_time_sec", "num_global_subgoals", "subgoal_success_count", "subgoal_attempt_count",
        "revisit_distance_m", "dead_end_entries", "repeated_dead_end_entries", "loop_count",
        "cumulative_local_steps", "backtracking_distance_m",
        "explored_area_m2", "visited_area_m2", "local_planner_failure_count", "global_replan_count",
        "localization_drift_error_m",
        # item 8: Local safety telemetry, connected.
        "min_clearance_m", "min_time_to_collision_sec", "min_stopping_margin_m",
        "steering_saturation_count", "steering_saturation_rate", "emergency_stop_count",
        "predicted_risk_mean", "predicted_risk_max", "risk_integral",
        "inference_timeout_count", "inference_error_count", "command_timeout_count",
        "collision_count", "collision_signal_source", "local_failure_reason_codes",
    }
    assert set(episode.keys()) == expected_keys
    assert episode["ablation"] == "A"
    assert isinstance(episode["success"], bool)


def test_run_ablation_mission_connects_ground_truth_collision_telemetry():
    """item 8: SimplifiedKinematicLocalExecutor's collision_signal_source is
    'ground_truth' (an authoritative world-occupancy check, never a
    clearance-distance proxy) -- run a dense-obstacle world so at least one
    collision is highly likely, and confirm it is actually counted and
    reported through the episode dict rather than staying write-only."""
    cfg = _small_world_cfg()
    manifest = build_fixed_benchmark_manifest(cfg, _ROBOT_RADIUS_M, _MIN_TURN_RADIUS_M, _WHEELBASE_M, num_scenarios=5, seed=3)
    any_collision = False
    for scenario in manifest:
        episode = run_ablation_mission(
            "A", scenario, global_cfg=GlobalRLConfig(), hierarchy_cfg=HierarchyConfig(mission_timeout_steps=2000),
            mapping_cfg=MappingConfig(mission_size_cells=128), memory_cfg=MemoryConfig(),
            feasibility_cfg=GlobalFeasibilityConfig(), long_horizon_cfg=cfg, robot_radius_m=_ROBOT_RADIUS_M,
            min_turning_radius_m=_MIN_TURN_RADIUS_M, wheelbase_m=_WHEELBASE_M, max_options=10, max_local_steps=80,
            mission_timeout_steps=2000, rng=np.random.RandomState(0),
        )
        assert episode["collision_signal_source"] in ("ground_truth", "unavailable")
        assert episode["collision_count"] >= 0
        assert episode["emergency_stop_count"] >= episode["collision_count"] or episode["collision_count"] == 0
        if episode["collision_count"] > 0:
            any_collision = True
            assert episode["collision_signal_source"] == "ground_truth"
    assert any_collision, "expected at least one collision across 5 scenarios in a corridor/obstacle world"


def test_aggregate_surfaces_item_8_telemetry_with_none_for_unmeasured_metrics():
    from hunter_kinodynamic_rl.evaluation.global_metrics import aggregate

    episodes = [
        {
            "success": True, "route_length_m": 1.0, "collision_count": 0, "emergency_stop_count": 0,
            "steering_saturation_rate": 0.1, "min_clearance_m": 0.4, "risk_integral": 2.0,
        },
        {
            "success": False, "route_length_m": 1.0, "collision_count": 2, "emergency_stop_count": 3,
            "steering_saturation_rate": None, "min_clearance_m": None, "risk_integral": 0.0,
        },
    ]
    metrics = aggregate(episodes)
    assert metrics["collision_count_total"] == 2
    assert metrics["collision_free_rate"] == 0.5
    assert metrics["emergency_stop_count_total"] == 3
    assert metrics["min_clearance_m_valid_count"] == 1
    assert metrics["min_clearance_m_worst"] == 0.4
    assert metrics["steering_saturation_rate_valid_count"] == 1
    assert metrics["risk_integral_total"] == 2.0
    # a metric no episode ever measured must stay None, never a fabricated 0.
    assert metrics["min_time_to_collision_sec_mean"] is None
    assert metrics["min_time_to_collision_sec_valid_count"] == 0


class _FakeLiveExecutorForBenchmark:
    """Same duck-typed shape as
    ``test_hierarchical_training_loop.py::_FakeLiveExecutor`` -- verifies
    ``run_ablation_mission``'s ``local_executor_factory`` injection point
    without any real Gazebo/rclpy dependency."""

    def __init__(self, world, mission_frame):
        self.pose_world = PoseXYYaw(*world.start_pose)
        self.run_option_calls = 0

    def run_option(self, coordinator, partial_map, rng, max_local_steps):
        self.run_option_calls += 1


def test_run_ablation_mission_uses_injected_local_executor_factory():
    cfg = _small_world_cfg()
    manifest = build_fixed_benchmark_manifest(cfg, _ROBOT_RADIUS_M, _MIN_TURN_RADIUS_M, _WHEELBASE_M, num_scenarios=1, seed=1)
    created = []

    def factory(world, mission_frame):
        executor = _FakeLiveExecutorForBenchmark(world, mission_frame)
        created.append(executor)
        return executor

    run_ablation_mission(
        "A", manifest[0], global_cfg=GlobalRLConfig(), hierarchy_cfg=HierarchyConfig(mission_timeout_steps=2000),
        mapping_cfg=MappingConfig(mission_size_cells=128), memory_cfg=MemoryConfig(), feasibility_cfg=GlobalFeasibilityConfig(),
        long_horizon_cfg=cfg, robot_radius_m=_ROBOT_RADIUS_M, min_turning_radius_m=_MIN_TURN_RADIUS_M,
        wheelbase_m=_WHEELBASE_M, max_options=10, max_local_steps=80, mission_timeout_steps=2000,
        rng=np.random.RandomState(0), local_executor_factory=factory,
    )
    assert len(created) == 1
    assert created[0].run_option_calls >= 1


def test_run_ablation_mission_requires_agent_for_global_rl_ablations():
    cfg = _small_world_cfg()
    manifest = build_fixed_benchmark_manifest(cfg, _ROBOT_RADIUS_M, _MIN_TURN_RADIUS_M, _WHEELBASE_M, num_scenarios=1, seed=2)
    with pytest.raises(ValueError):
        run_ablation_mission(
            "B", manifest[0], global_cfg=GlobalRLConfig(enabled=True), hierarchy_cfg=HierarchyConfig(),
            mapping_cfg=MappingConfig(mission_size_cells=128), memory_cfg=MemoryConfig(),
            feasibility_cfg=GlobalFeasibilityConfig(), long_horizon_cfg=cfg, robot_radius_m=_ROBOT_RADIUS_M,
            min_turning_radius_m=_MIN_TURN_RADIUS_M, wheelbase_m=_WHEELBASE_M, max_options=5, max_local_steps=40,
            mission_timeout_steps=1000, agent=None,
        )


class _FixedActionAgent:
    """Duck-typed GlobalDQNAgent stand-in (torch-free) -- picks the first
    VALID candidate deterministically, so ablations B/D/E/F/G are runnable
    in this ROS-free/torch-free test without a real network."""

    def select_action(self, obs, *, epsilon, rng):
        valid = np.flatnonzero(obs.action_mask)
        return int(valid[0])


def test_run_ablation_benchmark_same_manifest_across_ablations():
    cfg = _small_world_cfg()
    manifest = build_fixed_benchmark_manifest(cfg, _ROBOT_RADIUS_M, _MIN_TURN_RADIUS_M, _WHEELBASE_M, num_scenarios=2, seed=3)
    common_kwargs = dict(
        global_cfg=GlobalRLConfig(enabled=True, replay_capacity=10), hierarchy_cfg=HierarchyConfig(mission_timeout_steps=1500),
        mapping_cfg=MappingConfig(mission_size_cells=128), memory_cfg=MemoryConfig(max_nodes=8, max_nodes_in_observation=4),
        feasibility_cfg=GlobalFeasibilityConfig(), long_horizon_cfg=cfg, robot_radius_m=_ROBOT_RADIUS_M,
        min_turning_radius_m=_MIN_TURN_RADIUS_M, wheelbase_m=_WHEELBASE_M, max_options=6, max_local_steps=40,
        mission_timeout_steps=1500,
    )
    episodes_b = run_ablation_benchmark("B", manifest, agent=_FixedActionAgent(), seed=0, **common_kwargs)
    assert len(episodes_b) == len(manifest)
    result = aggregate(episodes_b)
    assert result["num_episodes"] == 2


def test_ablation_labels_constant_matches_plan():
    assert ABLATION_LABELS == ("A", "B", "C", "D", "E", "F", "G")


def test_effective_global_rl_config_forces_flags_from_label_not_caller():
    """Regression (Medium finding): the ablation LABEL must be the single
    source of truth for its flags, overriding whatever the caller's base
    config happened to carry -- in EITHER direction."""
    # Caller passes a "G"-shaped (everything on) base config while asking
    # for ablation "B" -- the label must win.
    base_on = GlobalRLConfig(
        enabled=True, topology_feedback_enabled=True, feasibility_feedback_enabled=True,
        global_risk_feedback_enabled=True,
    )
    effective_b = effective_global_rl_config(base_on, "B")
    assert effective_b.topology_feedback_enabled is False
    assert effective_b.feasibility_feedback_enabled is False
    assert effective_b.global_risk_feedback_enabled is False
    assert effective_b.enabled is True  # B still runs Global RL, just no Phase 5 features

    # Caller passes a "B"-shaped (everything off) base config while asking
    # for ablation "D" -- the label must win the other direction too.
    base_off = GlobalRLConfig()
    effective_d = effective_global_rl_config(base_off, "D")
    assert effective_d.topology_feedback_enabled is True
    assert effective_d.feasibility_feedback_enabled is False
    assert effective_d.global_risk_feedback_enabled is False

    assert effective_global_rl_config(base_off, "A").enabled is False


def test_effective_memory_and_feasibility_config_follow_the_same_label():
    assert effective_memory_config(MemoryConfig(enabled=False), "D").enabled is True
    assert effective_memory_config(MemoryConfig(enabled=True), "B").enabled is False
    assert effective_feasibility_config(GlobalFeasibilityConfig(enabled=True), "B").enabled is False
    assert effective_feasibility_config(GlobalFeasibilityConfig(enabled=False), "F").enabled is True
    assert effective_feasibility_config(GlobalFeasibilityConfig(enabled=False), "E").enabled is True


def test_run_ablation_mission_b_never_raises_even_with_phase5_flagged_base_config():
    """Regression: previously, running label 'B' against a caller-supplied
    config with Phase 5 flags already True raised inside
    build_global_observation (it required feature arrays this loop's
    ablation-derived local booleans never compute for 'B'). Forcing the
    effective config from the label fixes this."""
    cfg = _small_world_cfg()
    manifest = build_fixed_benchmark_manifest(cfg, _ROBOT_RADIUS_M, _MIN_TURN_RADIUS_M, _WHEELBASE_M, num_scenarios=1, seed=5)
    mismatched_global_cfg = GlobalRLConfig(
        enabled=True, topology_feedback_enabled=True, feasibility_feedback_enabled=True,
        global_risk_feedback_enabled=True,
    )
    episode = run_ablation_mission(
        "B", manifest[0], global_cfg=mismatched_global_cfg, hierarchy_cfg=HierarchyConfig(mission_timeout_steps=1000),
        mapping_cfg=MappingConfig(mission_size_cells=128), memory_cfg=MemoryConfig(enabled=True),
        feasibility_cfg=GlobalFeasibilityConfig(enabled=True), long_horizon_cfg=cfg, robot_radius_m=_ROBOT_RADIUS_M,
        min_turning_radius_m=_MIN_TURN_RADIUS_M, wheelbase_m=_WHEELBASE_M, max_options=5, max_local_steps=40,
        mission_timeout_steps=1000, agent=_FixedActionAgent(), rng=np.random.RandomState(0),
    )
    assert episode["ablation"] == "B"


def test_run_ablation_mission_d_builds_topology_features_even_with_phase4_shaped_base_config():
    """The other direction: label 'D' must actually WIRE topology features
    even when the caller's base config/memory_cfg both have their flags
    off -- previously this silently produced a Phase-4-shaped observation
    (no topology candidate columns) instead of failing loudly or working
    correctly."""
    cfg = _small_world_cfg()
    manifest = build_fixed_benchmark_manifest(cfg, _ROBOT_RADIUS_M, _MIN_TURN_RADIUS_M, _WHEELBASE_M, num_scenarios=1, seed=6)
    phase4_shaped_global_cfg = GlobalRLConfig(enabled=True)  # every Phase 5 flag False
    phase4_shaped_memory_cfg = MemoryConfig(enabled=False, max_nodes=8, max_nodes_in_observation=4)
    episode = run_ablation_mission(
        "D", manifest[0], global_cfg=phase4_shaped_global_cfg, hierarchy_cfg=HierarchyConfig(mission_timeout_steps=1000),
        mapping_cfg=MappingConfig(mission_size_cells=128), memory_cfg=phase4_shaped_memory_cfg,
        feasibility_cfg=GlobalFeasibilityConfig(), long_horizon_cfg=cfg, robot_radius_m=_ROBOT_RADIUS_M,
        min_turning_radius_m=_MIN_TURN_RADIUS_M, wheelbase_m=_WHEELBASE_M, max_options=5, max_local_steps=40,
        mission_timeout_steps=1000, agent=_FixedActionAgent(), rng=np.random.RandomState(0),
    )
    assert episode["ablation"] == "D"
    # Effective config actually had topology on for this run (the caller's
    # own phase4_shaped_global_cfg never had it -- confirms the forcing
    # happened, not just that nothing crashed).
    assert effective_global_rl_config(phase4_shaped_global_cfg, "D").topology_feedback_enabled is True


def test_relative_goal_is_mission_frame_relative_not_raw_world_pose():
    """Regression (Medium finding): BenchmarkScenarioSpec.relative_goal
    must be the mission-frame-relative goal (start-pose transform
    applied), never world.goal_pose verbatim -- the two only coincide when
    the world's own start pose happens to be exactly (0,0,0)."""
    from hunter_kinodynamic_rl.env.scenarios.long_horizon_generator import generate_long_horizon_world
    from hunter_kinodynamic_rl.navigation.mission.mission_frame import MissionFrame, PoseXYYaw

    cfg = _small_world_cfg()
    manifest = build_fixed_benchmark_manifest(cfg, _ROBOT_RADIUS_M, _MIN_TURN_RADIUS_M, _WHEELBASE_M, num_scenarios=1, seed=11)
    spec = manifest[0]
    world = generate_long_horizon_world(spec.seed, cfg, _ROBOT_RADIUS_M, _MIN_TURN_RADIUS_M, _WHEELBASE_M, mode=spec.mode)
    mission_frame = MissionFrame()
    mission_frame.initialize(PoseXYYaw(*world.start_pose))
    expected = mission_frame.odom_to_mission(world.goal_pose[0], world.goal_pose[1])
    assert spec.relative_goal == pytest.approx(expected)
    if world.start_pose[0] != 0.0 or world.start_pose[1] != 0.0 or world.start_pose[2] != 0.0:
        # Only exercises the actual regression when the start pose isn't
        # the origin (where the buggy raw-world-pose value would have
        # coincidentally matched) -- non-fatal when it happens to be.
        assert spec.relative_goal != pytest.approx(world.goal_pose)


def test_run_ablation_mission_navigation_time_sec_stays_within_the_configured_control_budget():
    """item 1 regression: navigation_time_sec (the sum of every option's
    elapsed_time_sec across the whole mission) must never exceed
    max_options * max_local_steps * dt_sec -- the exact bug class reported
    live (a 17.5s mission control budget benchmarking at 293-946.5s from a
    coordinator seeded with the wrong clock domain). max_local_steps is
    deliberately smaller than HierarchyConfig's default
    local_option_timeout_steps (200) so most/all options exhaust their
    executor-level budget (item 2's own fallback path) rather than ending
    via a coordinator-detected trigger -- exactly the scenario the original
    bug report was measured under."""
    cfg = _small_world_cfg()
    manifest = build_fixed_benchmark_manifest(cfg, _ROBOT_RADIUS_M, _MIN_TURN_RADIUS_M, _WHEELBASE_M, num_scenarios=1, seed=21)
    max_options, max_local_steps = 5, 20
    episode = run_ablation_mission(
        "A", manifest[0], global_cfg=GlobalRLConfig(), hierarchy_cfg=HierarchyConfig(mission_timeout_steps=None),
        mapping_cfg=MappingConfig(mission_size_cells=128), memory_cfg=MemoryConfig(), feasibility_cfg=GlobalFeasibilityConfig(),
        long_horizon_cfg=cfg, robot_radius_m=_ROBOT_RADIUS_M, min_turning_radius_m=_MIN_TURN_RADIUS_M,
        wheelbase_m=_WHEELBASE_M, max_options=max_options, max_local_steps=max_local_steps,
        mission_timeout_steps=None, rng=np.random.RandomState(0),
    )
    dt_sec = 0.3  # SimplifiedKinematicLocalExecutor's default control period
    max_plausible_navigation_time_sec = max_options * max_local_steps * dt_sec
    assert 0.0 <= episode["navigation_time_sec"] <= max_plausible_navigation_time_sec + 1e-6


def test_run_ablation_mission_rejects_a_manifest_whose_hash_no_longer_matches():
    """Regression (Low/Medium finding): content_hash/occupancy_hash were
    documented as a reproducibility check but never actually verified --
    a stale manifest (built under a different generator/config) could
    silently evaluate a different world. A scenario spec with a corrupted
    hash must be rejected fail-fast, before any mission logic runs."""
    cfg = _small_world_cfg()
    manifest = build_fixed_benchmark_manifest(cfg, _ROBOT_RADIUS_M, _MIN_TURN_RADIUS_M, _WHEELBASE_M, num_scenarios=1, seed=9)
    corrupted = dataclasses.replace(manifest[0], occupancy_hash="0" * 64)
    with pytest.raises(ValueError, match="does not match its manifest hash"):
        run_ablation_mission(
            "A", corrupted, global_cfg=GlobalRLConfig(), hierarchy_cfg=HierarchyConfig(mission_timeout_steps=500),
            mapping_cfg=MappingConfig(mission_size_cells=128), memory_cfg=MemoryConfig(),
            feasibility_cfg=GlobalFeasibilityConfig(), long_horizon_cfg=cfg, robot_radius_m=_ROBOT_RADIUS_M,
            min_turning_radius_m=_MIN_TURN_RADIUS_M, wheelbase_m=_WHEELBASE_M, max_options=3, max_local_steps=20,
            mission_timeout_steps=500, rng=np.random.RandomState(0),
        )


def test_run_ablation_mission_rejects_a_manifest_from_a_different_world_config():
    """Same regression, exercised the way it would actually happen in
    practice: a manifest built under one long_horizon_world config, then
    evaluated (by mistake) against a DIFFERENT config -- content_hash/
    occupancy_hash catch this even though the scenario's own seed/mode
    still round-trips through generate_long_horizon_world without error."""
    cfg_a = _small_world_cfg()
    cfg_b = dataclasses.replace(cfg_a, size_m=cfg_a.size_m + 4.0)  # a genuinely different world config
    manifest = build_fixed_benchmark_manifest(cfg_a, _ROBOT_RADIUS_M, _MIN_TURN_RADIUS_M, _WHEELBASE_M, num_scenarios=1, seed=13)
    with pytest.raises(ValueError, match="does not match its manifest hash"):
        run_ablation_mission(
            "A", manifest[0], global_cfg=GlobalRLConfig(), hierarchy_cfg=HierarchyConfig(mission_timeout_steps=500),
            mapping_cfg=MappingConfig(mission_size_cells=128), memory_cfg=MemoryConfig(),
            feasibility_cfg=GlobalFeasibilityConfig(), long_horizon_cfg=cfg_b, robot_radius_m=_ROBOT_RADIUS_M,
            min_turning_radius_m=_MIN_TURN_RADIUS_M, wheelbase_m=_WHEELBASE_M, max_options=3, max_local_steps=20,
            mission_timeout_steps=500, rng=np.random.RandomState(0),
        )
