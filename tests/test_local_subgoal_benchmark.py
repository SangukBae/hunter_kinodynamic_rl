"""Requirement B / defect-fix items 2, 3: local subgoal benchmark
manifest/metrics -- pure/synthetic, no live Gazebo execution."""

import pytest

from hunter_kinodynamic_rl.config.loader import load_profile
from hunter_kinodynamic_rl.evaluation.local_subgoal_benchmark import (
    BENCHMARK_SCHEMA_VERSION, BenchmarkArtifactError, LocalBenchmarkScenarioSpec,
    aggregate_local_benchmark_episodes, build_local_benchmark_artifact, build_local_benchmark_manifest,
    load_local_benchmark_manifest, save_local_benchmark_manifest,
)


def _profile():
    return load_profile("kinodynamic_tqc_arbitrary_subgoal")


def _episode(scenario_id="s0", seed=1, mode="test", success=False, collision=False, timeout=False,
             intended_infeasible=False, high_risk=False, reason="unknown", tgoal=None, tterm=10.0):
    return {
        "scenario_id": scenario_id, "seed": seed, "mode": mode, "subgoal_success": success, "collision": collision,
        "timeout": timeout, "intended_infeasible": intended_infeasible, "high_risk_failure": high_risk,
        "termination_reason": reason, "time_to_goal_sec": tgoal, "time_to_termination_sec": tterm,
    }


def _manifest_for(episodes):
    """A manifest whose scenario_id/seed/mode agree 1:1 with `episodes`,
    for tests that need build_local_benchmark_artifact's pairing check to
    pass without exercising build_local_benchmark_manifest's own seeds."""
    return [LocalBenchmarkScenarioSpec(scenario_id=e["scenario_id"], seed=e["seed"], mode=e["mode"]) for e in episodes]


def test_manifest_is_deterministic_and_test_pool_only():
    profile = _profile()
    m1 = build_local_benchmark_manifest(profile, num_scenarios=10, run_seed=3)
    m2 = build_local_benchmark_manifest(profile, num_scenarios=10, run_seed=3)
    assert [s.seed for s in m1] == [s.seed for s in m2]
    lo, hi = profile.scenario.test_seed_range
    assert all(lo <= s.seed <= hi for s in m1)
    assert len(m1) == 10


def test_manifest_different_run_seed_differs():
    profile = _profile()
    m1 = build_local_benchmark_manifest(profile, num_scenarios=10, run_seed=1)
    m2 = build_local_benchmark_manifest(profile, num_scenarios=10, run_seed=2)
    assert [s.seed for s in m1] != [s.seed for s in m2]


def test_manifest_save_and_load_roundtrip(tmp_path):
    profile = _profile()
    manifest = build_local_benchmark_manifest(profile, num_scenarios=5, run_seed=0)
    path = str(tmp_path / "manifest.json")
    save_local_benchmark_manifest(manifest, path)
    loaded = load_local_benchmark_manifest(path)
    assert [(s.scenario_id, s.seed, s.mode) for s in manifest] == [(s.scenario_id, s.seed, s.mode) for s in loaded]


def test_zero_scenarios_rejected():
    profile = _profile()
    with pytest.raises(ValueError):
        build_local_benchmark_manifest(profile, num_scenarios=0)


def test_manifest_rejects_duplicate_seeds():
    """Defect-fix item 3: a formal manifest must have unique seeds -- this
    pins the defensive check directly (SeedScheduler duplicating within a
    single small draw is not expected, but a caller must never silently
    get a manifest that would double-count one scenario)."""
    from unittest.mock import patch

    profile = _profile()
    with patch(
        "hunter_kinodynamic_rl.evaluation.local_subgoal_benchmark.SeedScheduler.next_seed",
        side_effect=[10, 10, 11],
    ):
        with pytest.raises(BenchmarkArtifactError, match="duplicate seed"):
            build_local_benchmark_manifest(profile, num_scenarios=3, run_seed=0)


def test_aggregate_computes_rates_and_reason_counts():
    episodes = [
        _episode("s0", success=True, reason="goal_reached", tgoal=5.0, tterm=5.0),
        _episode("s1", collision=True, reason="collision", tterm=3.0),
        _episode("s2", timeout=True, reason="timeout", tterm=20.0),
        _episode("s3", intended_infeasible=True, timeout=True, reason="timeout", tterm=20.0),
    ]
    summary = aggregate_local_benchmark_episodes(episodes)
    assert summary["num_episodes"] == 4
    assert summary["collision_rate"] == pytest.approx(0.25)
    assert summary["timeout_rate"] == pytest.approx(0.5)
    # 3 feasible episodes (s0/s1/s2), 1 success among them.
    assert summary["feasible_valid_count"] == 3
    assert summary["feasible_subgoal_success_rate"] == pytest.approx(1.0 / 3.0)
    # 1 infeasible episode (s3), timed out without success/collision/high-risk
    # -> counted as a safe termination, never as a "rejection".
    assert summary["infeasible_scenario_count"] == 1
    assert summary["infeasible_valid_count"] == 1
    assert summary["infeasible_false_success_rate"] == pytest.approx(0.0)
    assert summary["infeasible_collision_rate"] == pytest.approx(0.0)
    assert summary["infeasible_safe_termination_rate"] == pytest.approx(1.0)
    assert summary["time_to_goal_sec_mean"] == pytest.approx(5.0)
    assert summary["time_to_goal_sec_valid_count"] == 1  # success-only
    assert summary["time_to_termination_sec_valid_count"] == 4  # all episodes
    assert summary["termination_reason_counts"]["collision"] == 1
    assert "infeasible_goal_rejection_rate" not in summary
    assert "subgoal_success_rate" not in summary


def test_infeasible_metrics_are_none_never_zero_when_no_infeasible_sample():
    """Defect-fix item 2: a 0-sample infeasible rate must be None, not a
    fabricated 0.0 that would misleadingly read as "perfect"."""
    episodes = [_episode("s0", success=True, reason="goal_reached", tgoal=1.0, tterm=1.0)]
    summary = aggregate_local_benchmark_episodes(episodes)
    assert summary["infeasible_valid_count"] == 0
    assert summary["infeasible_false_success_rate"] is None
    assert summary["infeasible_collision_rate"] is None
    assert summary["infeasible_high_risk_rate"] is None
    assert summary["infeasible_safe_termination_rate"] is None


def test_infeasible_false_success_is_never_conflated_with_collision_or_timeout():
    """Defect-fix item 2's core distinction: a collision and a timeout on
    an infeasible scenario must never both land in one "rejection" bucket
    -- each is its own independently-readable rate."""
    episodes = [
        _episode("s0", intended_infeasible=True, collision=True, reason="collision", tterm=2.0),
        _episode("s1", intended_infeasible=True, success=True, reason="goal_reached", tgoal=3.0, tterm=3.0),
        _episode("s2", intended_infeasible=True, timeout=True, reason="timeout", tterm=20.0),
        _episode("s3", intended_infeasible=True, high_risk=True, timeout=True, reason="timeout", tterm=20.0),
    ]
    summary = aggregate_local_benchmark_episodes(episodes)
    assert summary["infeasible_scenario_count"] == 4
    assert summary["infeasible_collision_rate"] == pytest.approx(0.25)
    assert summary["infeasible_false_success_rate"] == pytest.approx(0.25)
    assert summary["infeasible_high_risk_rate"] == pytest.approx(0.25)
    # only s2 is collision-free, success-free, and high-risk-free.
    assert summary["infeasible_safe_termination_rate"] == pytest.approx(0.25)


def test_time_to_goal_excludes_failed_episodes_time_to_termination_includes_all():
    episodes = [
        _episode("s0", success=True, tgoal=5.0, tterm=5.0, reason="goal_reached"),
        _episode("s1", collision=True, tgoal=None, tterm=99.0, reason="collision"),
    ]
    summary = aggregate_local_benchmark_episodes(episodes)
    assert summary["time_to_goal_sec_valid_count"] == 1
    assert summary["time_to_termination_sec_valid_count"] == 2
    assert summary["time_to_termination_sec_mean"] == pytest.approx(52.0)


def test_aggregate_empty_raises():
    with pytest.raises(ValueError):
        aggregate_local_benchmark_episodes([])


def test_aggregate_missing_field_raises():
    bad = {"scenario_id": "s0"}
    with pytest.raises(KeyError):
        aggregate_local_benchmark_episodes([bad])


def test_build_artifact_includes_identity_and_provenance():
    profile = _profile()
    episodes = [_episode(f"local_test_{i:04d}", seed=i, success=True, reason="goal_reached", tgoal=1.0, tterm=1.0)
                for i in range(3)]
    manifest = _manifest_for(episodes)
    artifact = build_local_benchmark_artifact(
        profile, episodes, manifest=manifest, checkpoint_generation="gen-abc", checkpoint_sha256="deadbeef",
        benchmark_kind="formal",
    )
    assert artifact["schema_version"] == BENCHMARK_SCHEMA_VERSION
    assert artifact["benchmark_kind"] == "formal"
    assert artifact["checkpoint_generation"] == "gen-abc"
    assert artifact["checkpoint_sha256"] == "deadbeef"
    assert artifact["local_training_contract_fingerprint"]
    assert artifact["architecture_fingerprint"]
    assert artifact["scenario_manifest_sha256"]
    assert artifact["num_scenarios"] == 3
    assert "provenance" in artifact
    assert artifact["summary"]["num_episodes"] == artifact["num_scenarios"]
    assert artifact["summary"]["feasible_subgoal_success_rate"] == pytest.approx(1.0)


def test_build_artifact_rejects_invalid_benchmark_kind():
    profile = _profile()
    episodes = [_episode(f"s{i}", seed=i) for i in range(2)]
    manifest = _manifest_for(episodes)
    with pytest.raises(ValueError):
        build_local_benchmark_artifact(
            profile, episodes, manifest=manifest, checkpoint_generation="g", checkpoint_sha256="s",
            benchmark_kind="bogus",
        )


def test_build_artifact_rejects_manifest_episode_length_mismatch():
    profile = _profile()
    episodes = [_episode("local_test_0000", seed=0), _episode("local_test_0001", seed=1)]
    manifest = [LocalBenchmarkScenarioSpec(scenario_id="local_test_0000", seed=0, mode="test")]
    with pytest.raises(BenchmarkArtifactError, match="length mismatch"):
        build_local_benchmark_artifact(
            profile, episodes, manifest=manifest, checkpoint_generation="g", checkpoint_sha256="s",
        )


def test_build_artifact_rejects_scenario_id_set_mismatch():
    profile = _profile()
    episodes = [_episode("local_test_0000", seed=0), _episode("WRONG_ID", seed=1)]
    manifest = [
        LocalBenchmarkScenarioSpec(scenario_id="local_test_0000", seed=0, mode="test"),
        LocalBenchmarkScenarioSpec(scenario_id="local_test_0001", seed=1, mode="test"),
    ]
    with pytest.raises(BenchmarkArtifactError, match="scenario_id sets differ"):
        build_local_benchmark_artifact(
            profile, episodes, manifest=manifest, checkpoint_generation="g", checkpoint_sha256="s",
        )


def test_build_artifact_rejects_duplicate_scenario_id_in_manifest():
    profile = _profile()
    episodes = [_episode("local_test_0000", seed=0), _episode("local_test_0000", seed=1)]
    manifest = [
        LocalBenchmarkScenarioSpec(scenario_id="local_test_0000", seed=0, mode="test"),
        LocalBenchmarkScenarioSpec(scenario_id="local_test_0000", seed=1, mode="test"),
    ]
    with pytest.raises(BenchmarkArtifactError, match="duplicate scenario_id"):
        build_local_benchmark_artifact(
            profile, episodes, manifest=manifest, checkpoint_generation="g", checkpoint_sha256="s",
        )


def test_build_artifact_rejects_seed_mismatch_for_same_scenario_id():
    profile = _profile()
    episodes = [_episode("local_test_0000", seed=999)]
    manifest = [LocalBenchmarkScenarioSpec(scenario_id="local_test_0000", seed=0, mode="test")]
    with pytest.raises(BenchmarkArtifactError, match="seed mismatch"):
        build_local_benchmark_artifact(
            profile, episodes, manifest=manifest, checkpoint_generation="g", checkpoint_sha256="s",
        )


def test_build_artifact_rejects_non_test_mode_manifest_entry():
    profile = _profile()
    episodes = [_episode("local_test_0000", seed=0, mode="train")]
    manifest = [LocalBenchmarkScenarioSpec(scenario_id="local_test_0000", seed=0, mode="train")]
    with pytest.raises(BenchmarkArtifactError, match="non-'test'-mode"):
        build_local_benchmark_artifact(
            profile, episodes, manifest=manifest, checkpoint_generation="g", checkpoint_sha256="s",
        )


def test_run_local_benchmark_episode_rejects_non_test_scenario():
    """Defect-fix item 3: a formal benchmark must never run a train/
    validation-pool seed, even if one somehow ended up in a manifest."""
    from hunter_kinodynamic_rl.evaluation.local_subgoal_benchmark import run_local_benchmark_episode

    scenario = LocalBenchmarkScenarioSpec(scenario_id="x", seed=0, mode="train")
    with pytest.raises(BenchmarkArtifactError, match="expected 'test'"):
        run_local_benchmark_episode(env=None, agent=None, profile=_profile(), scenario=scenario)
