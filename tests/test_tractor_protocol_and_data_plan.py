"""Frozen acceptance criteria and split-safe scenario data-plan tests."""

from pathlib import Path

import pytest

from hunter_kinodynamic_rl.config.loader import load_profile
from hunter_kinodynamic_rl.config.tractor import load_tractor_contract, load_tractor_protocol
from hunter_kinodynamic_rl.env.randomization.domain_randomizer import apply_dynamics_overrides
from hunter_kinodynamic_rl.evaluation.tractor_acceptance import evaluate_acceptance
from hunter_kinodynamic_rl.training.collect_formal_comparison_data import _header
from hunter_kinodynamic_rl.training.tractor_scenario_plan import (
    build_scenario_plan, materialize_scenario_plan, validate_materialized_scenario_manifest,
)


CONFIG_ROOT = str(Path(__file__).resolve().parents[1] / "config")


def test_protocol_is_digest_locked_and_freezes_requested_metrics():
    protocol = load_tractor_protocol(CONFIG_ROOT)
    assert protocol["status"] == "frozen"
    assert protocol["protocol_version"] == "tractor_protocol_v2"
    assert len(protocol["frozen_payload_sha256"]) == 64
    assert {gate["metric"] for gate in protocol["headline_gates"]} == {
        "success_rate", "collision_rate", "time_to_goal_sec_mean", "min_clearance_m_mean",
    }
    assert protocol["runtime_gates"]["p99_latency_ms_max"] == 100.0
    assert protocol["runtime_gates"]["deadline_miss_rate_max"] == 0.01


def test_plan_has_disjoint_seeds_geometry_and_complete_static_dynamic_splits():
    plan = build_scenario_plan(CONFIG_ROOT, validate_feasibility=True)
    assert plan["manifest_sha256"] == "64452342d1baebc16fc8cead0ccca77574e19e055dc1a4639300975e5c86bda2"
    assert len(plan["entries"]) == 616
    seeds_by_split, geometry_by_split = {}, {}
    for split in ("development", "calibration", "locked_test"):
        selected = [entry for entry in plan["entries"] if entry["split_id"] == split]
        seeds_by_split[split] = {entry["seed"] for entry in selected}
        geometry_by_split[split] = {entry["scenario_geometry_sha256"] for entry in selected}
        assert {entry["obstacle_contract"] for entry in selected} == {"static_only", "dynamic"}
        assert len({entry["family_id"] for entry in selected}) == 11
        assert {entry["system_domain"] for entry in selected} == {"id", "ood"}
        assert {entry["vehicle_axis"] for entry in selected} == {
            "vehicle_nominal", "low_friction", "steering_response", "command_latency",
        }
        assert {entry["sensor_axis"] for entry in selected} == {
            "sensor_nominal", "lidar_range_noise", "lidar_dropout", "frame_dropout",
        }
        assert {entry["localization_axis"] for entry in selected} == {
            "localization_nominal", "odometry_noise",
        }
        assert all(len(entry["effective_robot_sha256"]) == 64 for entry in selected)
    assert [len(seeds_by_split[name]) for name in seeds_by_split] == [352, 88, 176]
    for left, right in (("development", "calibration"), ("development", "locked_test"),
                        ("calibration", "locked_test")):
        assert seeds_by_split[left].isdisjoint(seeds_by_split[right])
        assert geometry_by_split[left].isdisjoint(geometry_by_split[right])


def test_formal_episode_header_binds_the_scenario_effective_robot():
    plan = build_scenario_plan(CONFIG_ROOT, validate_feasibility=False)
    entry = next(item for item in plan["entries"] if item["vehicle_axis"] == "low_friction")
    profile = load_profile("tractor_local_dynamic", CONFIG_ROOT)
    effective_robot = apply_dynamics_overrides(profile.robot, entry["geometry"]["dynamics"])
    header = _header(
        entry=entry, profile=profile, contract=load_tractor_contract(CONFIG_ROOT, "a7"),
        attestation={"fixture": True},
        provenance={
            "package_git_commit_sha": "fixture-commit",
            "tracked_diff_sha256": "0" * 64,
            "untracked_source_manifest_sha256": "0" * 64,
        },
        effective_robot=effective_robot, step_count=1, termination_reason="goal",
        start_utc="2026-09-08T00:00:00Z", end_utc="2026-09-08T00:00:01Z",
    )
    assert header.robot_attestation_hash == entry["effective_robot_sha256"]


def test_materialized_plan_is_immutable_and_tampering_is_detected(tmp_path: Path):
    root = tmp_path / "scenarios"
    manifest = materialize_scenario_plan(root, CONFIG_ROOT)
    loaded = validate_materialized_scenario_manifest(root / "manifest.json", CONFIG_ROOT)
    assert loaded["artifact_manifest_sha256"] == manifest["artifact_manifest_sha256"]
    first = root / manifest["entries"][0]["relative_path"]
    first.write_text(first.read_text(encoding="utf-8") + "# tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_materialized_scenario_manifest(root / "manifest.json", CONFIG_ROOT)


def _acceptance_records():
    locked = [
        entry for entry in build_scenario_plan(CONFIG_ROOT, validate_feasibility=False)["entries"]
        if entry["split_id"] == "locked_test"
    ]
    records = []
    for method in ("A7", "B1"):
        for seed in (11, 23, 37, 53, 71):
            for entry in locked:
                candidate = method == "A7"
                records.append({
                    "experiment_id": f"{method}-{seed}-{entry['scenario_id']}",
                    "method_id": method, "seed": seed,
                    "scenario_id": entry["scenario_id"], "split_id": "locked_test",
                    "scenario_family": entry["family_id"],
                    "obstacle_contract": entry["obstacle_contract"], "complete": True,
                    "metrics": {
                        "success_rate": 0.90 if candidate else 0.80,
                        "collision_rate": 0.02 if candidate else 0.03,
                        "time_to_goal_sec_mean": 10.0,
                        "min_clearance_m_mean": 0.30,
                    },
                })
    return records


def test_acceptance_gate_uses_seed_level_ci_and_target_hardware_runtime():
    runtime = {
        "target_hardware": True, "timed_decisions": 10000,
        "p99_latency_ms": 90.0, "deadline_miss_rate": 0.005,
    }
    report = evaluate_acceptance(_acceptance_records(), runtime, config_root=CONFIG_ROOT)
    assert report.ok
    assert report.seed_count == 5
    assert report.scenario_count_per_seed == 176
    assert all(item["absolute_interval"]["count"] == 5 for item in report.gate_results)
    runtime["target_hardware"] = False
    failed = evaluate_acceptance(_acceptance_records(), runtime, config_root=CONFIG_ROOT)
    assert not failed.ok
    assert not next(item for item in failed.runtime_results if item["id"] == "target_hardware")["passed"]
