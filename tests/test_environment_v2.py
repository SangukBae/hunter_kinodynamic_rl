import dataclasses
import hashlib
import json
import math
from pathlib import Path

import pytest

from hunter_kinodynamic_rl.config.loader import default_config_root, load_profile
from hunter_kinodynamic_rl.config.schema import ConfigError
from hunter_kinodynamic_rl.env.humans.dynamic_obstacle_motion import (
    KinematicMotionState, MotionPattern, assign_pattern, motion_substep_durations,
)
from hunter_kinodynamic_rl.env.randomization.calibration_manifest import validate_calibration_manifest
from hunter_kinodynamic_rl.env.scenarios.benchmark_loader import load_benchmark
from hunter_kinodynamic_rl.env.scenarios.footprint_geometry import (
    circle_to_oriented_rectangle_clearance, footprint_overlaps_obstacle,
    oriented_rectangle_boundary_clearance,
)
from hunter_kinodynamic_rl.env.scenarios.procedural_generator import DynamicObstacleSpec, StaticObstacle
from hunter_kinodynamic_rl.env.scenarios.tractor_environment_v2 import (
    closest_approach_metrics, curriculum_stage, generate_v2_scenario,
)
from hunter_kinodynamic_rl.evaluation.fingerprint import (
    EVALUATION_CONTRACT_SECTIONS, evaluation_contract_fingerprint, sha256_of_obj,
    training_profile_fingerprint, training_profile_fingerprint_from_resolved_config,
)


def _v2_profile():
    return load_profile("tractor_local_dynamic_v2")


def test_disabled_v2_preserves_legacy_training_and_evaluation_fingerprints():
    profile = load_profile("tractor_local_dynamic")
    resolved = dataclasses.asdict(profile)
    legacy_resolved = dict(resolved)
    legacy_resolved.pop("environment_v2")
    assert training_profile_fingerprint(profile) == training_profile_fingerprint_from_resolved_config(legacy_resolved)

    legacy_evaluation_payload = {
        name: resolved[name] for name in EVALUATION_CONTRACT_SECTIONS if name != "environment_v2"
    }
    assert evaluation_contract_fingerprint(profile) == sha256_of_obj(legacy_evaluation_payload)


def test_v2_schema_rejects_legacy_cylinder_pool_and_duplicate_taxonomy():
    profile = _v2_profile()
    with pytest.raises(ConfigError, match="legacy pool"):
        dataclasses.replace(
            profile, obstacle_pool=dataclasses.replace(profile.obstacle_pool, enabled=True, max_dynamic=8),
        ).validate()
    with pytest.raises(ConfigError, match="duplicates"):
        dataclasses.replace(
            profile.environment_v2, topology_families=["open", "open"],
        ).validate()


def test_curriculum_has_expected_density_and_speed_stages():
    cfg = _v2_profile().environment_v2
    expected = [(0, 0, 0.0), (1, 1, 0.5), (2, 2, 0.75), (3, 4, 1.0), (4, 8, 1.0)]
    for episode, values in zip(cfg.curriculum_episode_boundaries, expected):
        stage = curriculum_stage(cfg, episode, "train")
        assert (stage.level, stage.dynamic_obstacle_count, stage.dynamic_speed_ratio_cap) == values
    assert curriculum_stage(cfg, 0, "test").level == cfg.evaluation_level


def test_v2_generation_is_deterministic_and_conflict_conditioned():
    profile = _v2_profile()
    args = (
        20000, profile.scenario, profile.environment_v2, profile.robot,
        profile.start_pose, profile.reward.goal_threshold_m, 6000, "test",
    )
    first = generate_v2_scenario(*args)
    second = generate_v2_scenario(*args)
    assert dataclasses.asdict(first) == dataclasses.asdict(second)
    assert first.environment_version == "tractor_env_v2"
    assert len(first.dynamic_obstacles) == 8
    assert first.conflict_obstacle_count >= 3
    assert {item.shape for item in first.static_obstacles + first.dynamic_obstacles}.issubset(
        set(profile.environment_v2.obstacle_shapes)
    )
    for item in first.dynamic_obstacles:
        if item.target_ttc_sec is not None:
            assert profile.environment_v2.conflict_ttc_range_sec[0] <= item.target_ttc_sec
            assert item.target_ttc_sec <= profile.environment_v2.conflict_ttc_range_sec[1]
            assert item.target_dcpa_m <= profile.environment_v2.conflict_dcpa_range_m[1] + 1e-9


def test_closest_approach_metrics_recover_constructed_crossing():
    ttc, dcpa = closest_approach_metrics((0.0, 0.0), (1.0, 0.0), (2.0, -2.0), (0.0, 1.0))
    assert ttc == pytest.approx(2.0)
    assert dcpa == pytest.approx(0.0)


def test_v2_kinematic_motion_obeys_acceleration_and_world_boundary():
    spec = DynamicObstacleSpec(
        x0=0.0, y0=0.0, vx=1.0, vy=0.0, radius=0.3,
        motion_pattern="stop_go", accel_limit_mps2=0.5, turn_rate_rad_s=1.0,
    )
    state = KinematicMotionState.from_spec(spec, half_extent_m=5.0, seed=7)
    for _ in range(30):
        before = math.hypot(state.vx, state.vy)
        state.tick(0.1)
        after = math.hypot(state.vx, state.vy)
        assert abs(after - before) <= 0.05 + 1e-12
    assert math.hypot(state.vx, state.vy) < 1.0

    bounce = KinematicMotionState.from_spec(
        dataclasses.replace(spec, x0=0.69, vx=1.0, motion_pattern="bounce"),
        half_extent_m=1.0, seed=8,
    )
    x, _, vx, _, _ = bounce.tick(0.1)
    assert x <= 0.7 + 1e-12
    assert vx < 0.0


def test_substeps_preserve_control_interval():
    durations = motion_substep_durations(0.1, 5)
    assert durations == pytest.approx((0.02,) * 5)
    assert sum(durations) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        motion_substep_durations(0.1, 0)


def test_oriented_hunter_footprint_geometry_handles_side_clearance_and_shapes():
    assert circle_to_oriented_rectangle_clearance(
        (0.0, 0.40), 0.05, (0.0, 0.0), 0.0, 0.96, 0.64,
    ) > 0.0
    assert footprint_overlaps_obstacle(
        (0.0, 0.0), 0.0, 0.96, 0.64,
        StaticObstacle(0.45, 0.0, 0.20), padding_m=0.0,
    )
    assert footprint_overlaps_obstacle(
        (0.0, 0.0), 0.0, 0.96, 0.64,
        StaticObstacle(0.55, 0.0, 0.5, shape="box", length_m=0.4, width_m=0.4),
    )
    assert oriented_rectangle_boundary_clearance((0.0, 0.0), 0.0, 0.96, 0.64, 1.0) > 0.0
    assert oriented_rectangle_boundary_clearance((0.7, 0.0), 0.0, 0.96, 0.64, 1.0) < 0.0


def test_calibration_manifest_exposes_engineering_prior_and_model_only_axes():
    profile = _v2_profile()
    report = validate_calibration_manifest(profile, default_config_root())
    assert report.ok
    assert report.status == "engineering_prior"
    assert any("model-only" in warning for warning in report.warnings)


def test_materialized_v2_suites_are_disjoint_and_checksum_verified():
    root = Path(default_config_root())
    manifest = json.loads((root / "benchmarks" / "environment_v2_manifest.json").read_text())
    seen = set()
    for suite_name, suite in manifest["suites"].items():
        assert len(suite["files"]) == 8
        for entry in suite["files"]:
            assert entry["seed"] not in seen
            seen.add(entry["seed"])
            path = root / "benchmarks" / suite_name / entry["file"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
    loaded = load_benchmark("v2_id")
    assert len(loaded) == 8
    assert all(item.spec.environment_version == "tractor_env_v2" for item in loaded)


def test_all_v2_evaluation_profiles_load_with_fixed_benchmarks():
    names = (
        "evaluation_v2_id", "evaluation_v2_ood_motion", "evaluation_v2_ood_density",
        "evaluation_v2_ood_geometry_12m", "evaluation_v2_ood_geometry_24m",
        "evaluation_v2_ood_system",
    )
    for name in names:
        profile = load_profile(name)
        assert profile.environment_v2.enabled
        assert profile.runtime.deterministic_stepping
        assert len(load_benchmark(profile.evaluation.benchmark)) == 8


def test_legacy_pattern_assignment_never_selects_v2_only_patterns():
    legacy = {
        MotionPattern.CROSSING, MotionPattern.HEAD_ON, MotionPattern.CUT_IN,
        MotionPattern.PARALLEL, MotionPattern.RANDOM_WAYPOINT, MotionPattern.CONSTANT_VELOCITY,
    }
    assert {assign_pattern(seed, index) for seed in range(20) for index in range(5)} <= legacy
