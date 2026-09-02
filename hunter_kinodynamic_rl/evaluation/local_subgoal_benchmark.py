#!/usr/bin/env python3
"""Requirement B: reproducible unseen short-range Local subgoal benchmark.

Drives the SAME live ``EnvironmentClient``/environment_node service contract
``training/trainer_base.py`` and ``evaluation/benchmark_runner.py`` already
use (``env.set_explicit_seed_mode``/``env.seed``/``env.reset``/``env.step``)
-- the live environment_node must already be running with a
``robot_relative_band`` (arbitrary-subgoal) profile loaded (e.g.
``kinodynamic_tqc_arbitrary_subgoal``), so its OWN server-side
``ScenarioConfig`` (goal distance/direction/infeasible-fraction/feasibility
mode) is what actually generates each seed's scenario -- this module never
re-implements or second-guesses that generation, only supplies TEST-pool
seeds (``SeedScheduler``, ``mode="test"`` -- never train/validation) and
classifies+aggregates the outcome.

Per-episode ``termination_reason`` (mutually exclusive, in priority order):

  collision > goal_reached > timeout > unknown

Schema v2 / defect-fix item 2 (metric redesign): there is NO policy- or
environment-level signal that the agent "recognized and explicitly
rejected" an infeasible goal -- the v1 ``infeasible_goal_rejection``/
``infeasible_goal_rejection_rate`` fields fabricated that claim by folding
collision AND timeout outcomes on an infeasible scenario into one bucket
labeled "rejection", then divided by the TOTAL episode count (diluted by
every feasible scenario too). Both are gone. In their place, every
infeasible-conditional metric below is conditioned ONLY on episodes where
``ScenarioSpec.realized_infeasible`` was True, with an explicit
``infeasible_valid_count`` denominator (``None``, never a fabricated
``0.0``, when that count is 0):

- ``infeasible_scenario_count`` -- how many of this benchmark's scenarios
  were actually infeasible (may differ run-to-run at fixed
  ``goal_infeasible_fraction`` -- it's a per-scenario coin flip).
- ``infeasible_false_success_rate`` -- the policy reported ``target=True``
  on a scenario verified infeasible up front. This is a genuine defect
  signal (a false positive), never conflated with a timeout.
- ``infeasible_collision_rate`` -- collided while attempting an infeasible
  scenario. A real safety failure, counted independently of whether the
  scenario was ever reachable.
- ``infeasible_high_risk_rate`` -- the environment's own real risk
  telemetry ``unrecoverable`` flag fired on an infeasible scenario.
- ``infeasible_safe_termination_rate`` -- the episode ended WITHOUT a false
  success, a collision, or a high-risk flag. This is the honest, weaker
  claim the old field overstated: "nothing bad happened", not "the policy
  proved it understood the goal was unreachable" (this benchmark has no
  instrument for that stronger claim).

``high_risk_failure`` is the environment's own real risk-telemetry
``unrecoverable`` flag (never re-derived from the normalized action) --
independent of ``termination_reason`` (can co-occur with ``collision``).

Overall (unconditional) ``collision_rate``/``timeout_rate``/
``high_risk_failure_rate`` still cover ALL episodes (safety is not
conditioned on feasibility). ``subgoal_success_rate`` was renamed
``feasible_subgoal_success_rate`` and is now conditioned on FEASIBLE
scenarios only -- an infeasible scenario cannot be "solved" by
construction, so including it in the denominator only dilutes what the
number means and, at a nonzero ``goal_infeasible_fraction``, silently caps
the achievable rate below 1.0 in a way an acceptance threshold reader would
not expect. See :data:`config/local_acceptance.yaml`'s comments for the
`min_infeasible_scenarios` gate this feeds and the acceptance-contract
rationale at ``goal_infeasible_fraction=0.15``/``num_scenarios=20``.

Real live execution is intentionally NOT run by this session (see the
formal-training/benchmark execution restriction in the task that produced
this module) -- :func:`build_local_benchmark_manifest`/
:func:`aggregate_local_benchmark_episodes` are pure and unit-tested with
synthetic episode dicts; :func:`run_local_benchmark_episode`/
:func:`run_local_subgoal_benchmark` require a live Gazebo + environment_node
and are exercised only by import/signature checks in this session.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional

from hunter_kinodynamic_rl.config.schema import Profile
from hunter_kinodynamic_rl.env.scenarios.procedural_generator import ScenarioSpec, generate_scenario
from hunter_kinodynamic_rl.env.scenarios.seed_scheduler import SeedScheduler
from hunter_kinodynamic_rl.evaluation.fingerprint import architecture_fingerprint, local_training_contract_fingerprint
from hunter_kinodynamic_rl.evaluation.provenance import collect_package_provenance

BENCHMARK_SCHEMA_VERSION = 2
MIN_SUPPORTED_BENCHMARK_SCHEMA_VERSION = 2
TERMINATION_REASONS = ("collision", "goal_reached", "timeout", "unknown")


class BenchmarkArtifactError(ValueError):
    """A benchmark artifact/manifest failed a structural-integrity check
    (defect-fix item 3) -- never silently coerced or partially trusted."""


@dataclass(frozen=True)
class LocalBenchmarkScenarioSpec:
    """One fixed, reproducible benchmark scenario -- ``seed`` alone
    reproduces the identical procedural scenario (server-side, via the live
    environment_node's own generator), so this is a thin, hashable manifest
    entry, not a duplicate of the actual generated geometry."""

    scenario_id: str
    seed: int
    mode: str  # always "test" for a formal benchmark manifest


def build_local_benchmark_manifest(
    profile: Profile, *, num_scenarios: int, run_seed: int = 0,
) -> List[LocalBenchmarkScenarioSpec]:
    """Deterministic given ``(profile.scenario test_seed_range, run_seed,
    num_scenarios)`` -- calling this twice with the same arguments produces
    byte-identical seed sequences (``SeedScheduler``'s own determinism
    guarantee), which is what makes this manifest reusable/reproducible
    rather than redrawn per run. Defect-fix item 3: a formal manifest's
    seeds MUST be unique -- a duplicate seed would silently score the SAME
    scenario twice under two different ``scenario_id``s, double-counting it
    in every aggregate rate. ``SeedScheduler`` is not expected to ever
    produce a duplicate within one call; this is a defensive, fail-loud
    check on that assumption, not a workaround for a known collision."""
    if num_scenarios <= 0:
        raise ValueError("build_local_benchmark_manifest: num_scenarios must be > 0")
    scheduler = SeedScheduler(run_seed, profile.scenario, mode="test")
    manifest = [
        LocalBenchmarkScenarioSpec(scenario_id=f"local_test_{i:04d}", seed=scheduler.next_seed(), mode="test")
        for i in range(num_scenarios)
    ]
    seed_counts = Counter(s.seed for s in manifest)
    duplicates = {seed: count for seed, count in seed_counts.items() if count > 1}
    if duplicates:
        raise BenchmarkArtifactError(
            f"build_local_benchmark_manifest: SeedScheduler produced duplicate seed(s) {duplicates} within "
            "one formal manifest -- a formal benchmark requires unique test seeds per scenario"
        )
    return manifest


def save_local_benchmark_manifest(manifest: List[LocalBenchmarkScenarioSpec], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump([{"scenario_id": s.scenario_id, "seed": s.seed, "mode": s.mode} for s in manifest], f, indent=2)


def load_local_benchmark_manifest(path: str) -> List[LocalBenchmarkScenarioSpec]:
    with open(path) as f:
        rows = json.load(f)
    return [LocalBenchmarkScenarioSpec(scenario_id=r["scenario_id"], seed=r["seed"], mode=r["mode"]) for r in rows]


def _synthetic_scenario_spec(profile: Profile, seed: int) -> ScenarioSpec:
    """Recomputes the SAME procedural scenario a live environment_node
    would generate server-side for ``seed`` -- used ONLY to read
    ``realized_infeasible`` for termination classification (never to
    re-derive start/goal/obstacle placement the live episode itself
    actually ran against)."""
    min_turning_radius_m = 1.0 / max(1e-9, profile.robot.max_curvature)
    return generate_scenario(
        seed, profile.scenario, robot_radius=profile.robot.collision_radius_m,
        min_turning_radius_m=min_turning_radius_m, wheelbase_m=profile.robot.wheelbase_m,
        start_pose_cfg=profile.start_pose,
    )


def verify_environment_server_identity(env, profile: Profile) -> None:
    """Defect-fix item 3: refuse to start (or continue) a Local benchmark
    against a live environment_node whose ACTUALLY-loaded profile doesn't
    match the profile this benchmark script itself loaded -- queries the
    server's own launch-time-resolved fingerprints
    (``EnvironmentClient.get_remote_parameter``, the SAME mechanism
    ``evaluation_node.py`` already uses to verify a live environment
    against a checkpoint) rather than trusting a ``--profile`` NAME alone,
    which cannot detect the server having been started from an
    edited/older copy of a same-named profile's YAML. Raises
    :class:`BenchmarkArtifactError` immediately -- before running a single
    scenario -- on any mismatch; never proceeds "with a warning"."""
    remote_arch = env.get_remote_parameter("architecture_fingerprint_sha256")
    remote_contract = env.get_remote_parameter("local_training_contract_fingerprint_sha256")
    local_arch = architecture_fingerprint(profile)
    local_contract = local_training_contract_fingerprint(profile)
    mismatches = []
    if remote_arch != local_arch:
        mismatches.append(
            f"architecture_fingerprint_sha256: server={remote_arch!r} != benchmark profile={local_arch!r}"
        )
    if remote_contract != local_contract:
        mismatches.append(
            "local_training_contract_fingerprint_sha256: server="
            f"{remote_contract!r} != benchmark profile={local_contract!r}"
        )
    if mismatches:
        raise BenchmarkArtifactError(
            "Local benchmark refused: the live environment_node's actually-loaded profile does not match "
            "the profile this benchmark run requested -- " + "; ".join(mismatches)
        )


def run_local_benchmark_episode(env, agent, profile: Profile, scenario: LocalBenchmarkScenarioSpec) -> dict:
    """Live episode runner -- requires an already-constructed
    ``training.trainer_base.EnvironmentClient`` against a running
    environment_node loaded with a ``robot_relative_band`` profile. NOT
    executed in this session (see module docstring)."""
    if scenario.mode != "test":
        raise BenchmarkArtifactError(
            f"run_local_benchmark_episode: scenario {scenario.scenario_id!r} has mode={scenario.mode!r}, "
            "expected 'test' -- a formal benchmark must never run a train/validation-pool seed"
        )
    set_mode = env.set_explicit_seed_mode
    if not set_mode("test"):
        raise RuntimeError("environment rejected explicit_seed_mode='test'")
    if env.seed(scenario.seed) is False:
        raise RuntimeError(f"environment rejected benchmark seed {scenario.seed}")
    state = env.reset()

    collided = success = False
    unrecoverable = False
    total_reward = 0.0
    steps_run = 0
    episode_length_steps = profile.evaluation.max_episode_steps
    for step in range(episode_length_steps):
        action = agent.select_action(state, deterministic=True)
        state, reward, done, target, collision, _min_dist, telemetry, _diagnostics = env.step(action)
        total_reward += reward
        steps_run = step + 1
        if telemetry.valid:
            unrecoverable = unrecoverable or telemetry.unrecoverable
        if collision:
            collided = True
        if target:
            success = True
        if done:
            break

    scenario_spec = _synthetic_scenario_spec(profile, scenario.seed)
    timeout = not (collided or success)
    if collided:
        termination_reason = "collision"
    elif success:
        termination_reason = "goal_reached"
    elif timeout:
        termination_reason = "timeout"
    else:
        termination_reason = "unknown"

    elapsed_sim_time = env.episode_elapsed_sim_time_sec
    return {
        "scenario_id": scenario.scenario_id, "seed": scenario.seed, "mode": scenario.mode,
        "subgoal_success": success, "collision": collided, "timeout": timeout,
        "intended_infeasible": bool(scenario_spec.realized_infeasible),
        "high_risk_failure": bool(unrecoverable),
        "termination_reason": termination_reason,
        "time_to_goal_sec": elapsed_sim_time if success else None,
        "time_to_termination_sec": elapsed_sim_time,
        "steps": steps_run, "total_reward": total_reward,
    }


def _validate_manifest_episode_pairing(manifest: List[LocalBenchmarkScenarioSpec], episodes: List[dict]) -> None:
    """Defect-fix item 3: the manifest and the episode list must agree
    1:1 -- same length, same scenario_id set, each scenario_id mapping to
    the SAME seed on both sides, no duplicate scenario_id on either side,
    every manifest entry's mode is "test". Raises :class:`BenchmarkArtifactError`
    (never silently truncates/reorders/ignores a mismatch) the first time
    any of these is violated."""
    if len(manifest) != len(episodes):
        raise BenchmarkArtifactError(
            f"manifest/episode length mismatch: manifest has {len(manifest)} scenarios, "
            f"episodes has {len(episodes)}"
        )
    manifest_ids = [m.scenario_id for m in manifest]
    episode_ids = [e["scenario_id"] for e in episodes]
    dup_manifest_ids = {sid for sid, count in Counter(manifest_ids).items() if count > 1}
    if dup_manifest_ids:
        raise BenchmarkArtifactError(f"manifest contains duplicate scenario_id(s): {sorted(dup_manifest_ids)}")
    dup_episode_ids = {sid for sid, count in Counter(episode_ids).items() if count > 1}
    if dup_episode_ids:
        raise BenchmarkArtifactError(f"episodes contain duplicate scenario_id(s): {sorted(dup_episode_ids)}")
    if set(manifest_ids) != set(episode_ids):
        raise BenchmarkArtifactError(
            f"manifest/episode scenario_id sets differ: manifest_only={sorted(set(manifest_ids) - set(episode_ids))}, "
            f"episode_only={sorted(set(episode_ids) - set(manifest_ids))}"
        )
    seed_by_id = {m.scenario_id: m.seed for m in manifest}
    seed_mismatches = [
        (e["scenario_id"], seed_by_id[e["scenario_id"]], e["seed"])
        for e in episodes if e.get("seed") is not None and e["seed"] != seed_by_id[e["scenario_id"]]
    ]
    if seed_mismatches:
        raise BenchmarkArtifactError(f"manifest/episode seed mismatch for scenario_id(s): {seed_mismatches}")
    non_test_manifest = [m.scenario_id for m in manifest if m.mode != "test"]
    if non_test_manifest:
        raise BenchmarkArtifactError(
            f"formal benchmark manifest contains non-'test'-mode scenario(s): {non_test_manifest}"
        )
    non_test_episodes = [e["scenario_id"] for e in episodes if e.get("mode", "test") != "test"]
    if non_test_episodes:
        raise BenchmarkArtifactError(f"episode(s) recorded a non-'test' mode: {non_test_episodes}")


def aggregate_local_benchmark_episodes(episodes: List[dict]) -> dict:
    """Pure aggregation -- no live execution. Fails fast (KeyError) on a
    malformed episode dict missing a required field, rather than silently
    treating it as 0/False. See module docstring for the item-2 metric
    redesign (feasible-conditional success rate, infeasible-conditional
    metrics with an explicit valid-count denominator)."""
    if not episodes:
        raise ValueError("aggregate_local_benchmark_episodes: episodes must be non-empty")
    required = (
        "scenario_id", "seed", "subgoal_success", "collision", "timeout", "intended_infeasible",
        "high_risk_failure", "termination_reason", "time_to_goal_sec", "time_to_termination_sec",
    )
    for ep in episodes:
        missing = [k for k in required if k not in ep]
        if missing:
            raise KeyError(f"aggregate_local_benchmark_episodes: episode {ep.get('scenario_id')!r} missing {missing}")

    n = len(episodes)

    def _rate(subset: List[dict], key: str) -> Optional[float]:
        return (sum(1 for e in subset if e[key]) / len(subset)) if subset else None

    feasible = [e for e in episodes if not e["intended_infeasible"]]
    infeasible = [e for e in episodes if e["intended_infeasible"]]

    goal_times = [e["time_to_goal_sec"] for e in episodes if e["subgoal_success"] and e["time_to_goal_sec"] is not None]
    term_times = [e["time_to_termination_sec"] for e in episodes if e["time_to_termination_sec"] is not None]
    reason_counts = {r: 0 for r in TERMINATION_REASONS}
    for e in episodes:
        reason_counts[e["termination_reason"]] = reason_counts.get(e["termination_reason"], 0) + 1

    infeasible_safe_termination = [
        e for e in infeasible if not e["subgoal_success"] and not e["collision"] and not e["high_risk_failure"]
    ]

    return {
        "num_episodes": n,
        "collision_rate": _rate(episodes, "collision"),
        "timeout_rate": _rate(episodes, "timeout"),
        "high_risk_failure_rate": _rate(episodes, "high_risk_failure"),
        "termination_reason_counts": reason_counts,
        "time_to_goal_sec_mean": (sum(goal_times) / len(goal_times)) if goal_times else None,
        "time_to_goal_sec_valid_count": len(goal_times),
        "time_to_termination_sec_mean": (sum(term_times) / len(term_times)) if term_times else None,
        "time_to_termination_sec_valid_count": len(term_times),
        # feasible-conditional (item 2): an infeasible scenario cannot be
        # "solved" -- diluting this rate with unsolvable scenarios would
        # silently cap it below 1.0 at any nonzero goal_infeasible_fraction.
        "feasible_valid_count": len(feasible),
        "feasible_subgoal_success_rate": _rate(feasible, "subgoal_success"),
        # infeasible-conditional (item 2): every rate below is None (never
        # a fabricated 0.0) when infeasible_valid_count == 0.
        "infeasible_scenario_count": len(infeasible),
        "infeasible_valid_count": len(infeasible),
        "infeasible_false_success_rate": _rate(infeasible, "subgoal_success"),
        "infeasible_collision_rate": _rate(infeasible, "collision"),
        "infeasible_high_risk_rate": _rate(infeasible, "high_risk_failure"),
        "infeasible_safe_termination_rate": (
            (len(infeasible_safe_termination) / len(infeasible)) if infeasible else None
        ),
    }


def build_local_benchmark_artifact(
    profile: Profile, episodes: List[dict], *, manifest: List[LocalBenchmarkScenarioSpec],
    checkpoint_generation: str, checkpoint_sha256: str, benchmark_kind: str = "formal",
) -> dict:
    """Assembles the full artifact dict (schema/provenance/identity + the
    aggregated summary) that :mod:`evaluation.local_promotion` consumes and
    a caller writes to disk (e.g. ``result_writer``-style
    ``json.dump(artifact, f, indent=2)``). ``benchmark_kind`` must be
    ``"formal"`` or ``"smoke"`` -- never used to disguise one as the other.
    Defect-fix item 3: validates manifest/episode pairing BEFORE
    aggregating (:func:`_validate_manifest_episode_pairing`) and asserts
    ``summary.num_episodes == num_scenarios`` afterward -- both would
    already be implied by a correct caller, but this function never trusts
    that and fails loudly instead of producing a self-inconsistent
    artifact."""
    if benchmark_kind not in ("formal", "smoke"):
        raise ValueError(f"benchmark_kind must be 'formal' or 'smoke', got {benchmark_kind!r}")
    _validate_manifest_episode_pairing(manifest, episodes)
    summary = aggregate_local_benchmark_episodes(episodes)
    if summary["num_episodes"] != len(manifest):
        raise BenchmarkArtifactError(
            f"summary.num_episodes={summary['num_episodes']} != len(manifest)={len(manifest)}"
        )
    manifest_sha256 = _sha256_of_manifest(manifest)
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "benchmark_kind": benchmark_kind,
        "profile_name": profile.name,
        "checkpoint_generation": checkpoint_generation,
        "checkpoint_sha256": checkpoint_sha256,
        "architecture_fingerprint": architecture_fingerprint(profile),
        "local_training_contract_fingerprint": local_training_contract_fingerprint(profile),
        "scenario_manifest_sha256": manifest_sha256,
        "num_scenarios": len(manifest),
        "provenance": collect_package_provenance(),
        "created_at_unix": time.time(),
        "summary": summary,
        "episodes": episodes,
    }


def _sha256_of_manifest(manifest: List[LocalBenchmarkScenarioSpec]) -> str:
    payload = json.dumps(
        [{"scenario_id": s.scenario_id, "seed": s.seed, "mode": s.mode} for s in manifest],
        sort_keys=True, separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def run_local_subgoal_benchmark(
    profile: Profile, agent, *, manifest: List[LocalBenchmarkScenarioSpec],
    checkpoint_generation: str, checkpoint_sha256: str, env=None, benchmark_kind: str = "formal",
) -> dict:
    """Live end-to-end driver -- constructs/owns an ``EnvironmentClient`` if
    ``env`` is not supplied, verifies the live server's actually-loaded
    profile matches ``profile`` BEFORE running any scenario
    (:func:`verify_environment_server_identity`, item 3), runs every
    scenario in ``manifest`` in order, and returns
    :func:`build_local_benchmark_artifact`'s result. NOT executed in this
    session; see module docstring."""
    from hunter_kinodynamic_rl.training.trainer_base import EnvironmentClient

    owns_env = env is None
    if owns_env:
        env = EnvironmentClient()
    try:
        verify_environment_server_identity(env, profile)
        episodes: List[dict] = []
        for scenario in manifest:
            episodes.append(run_local_benchmark_episode(env, agent, profile, scenario))
    finally:
        if owns_env:
            env.destroy_node()
    return build_local_benchmark_artifact(
        profile, episodes, manifest=manifest, checkpoint_generation=checkpoint_generation,
        checkpoint_sha256=checkpoint_sha256, benchmark_kind=benchmark_kind,
    )
