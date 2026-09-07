"""Reproducible Stage-2 Local L0-L5 multi-seed experiment contract.

This module owns the parts that must remain stable across a multi-day live
Gazebo campaign: the label/profile mapping, five training seeds, one shared
test manifest, deterministic artifact locations, cross-run integrity checks,
seed-level aggregation and the pre-registered promotion choice.  It never
silently substitutes a missing run or chooses the best test-set seed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import yaml

from hunter_kinodynamic_rl.config.loader import default_config_root, load_profile
from hunter_kinodynamic_rl.evaluation.fingerprint import (
    training_profile_fingerprint,
    training_profile_fingerprint_from_resolved_config,
)
from hunter_kinodynamic_rl.evaluation.local_promotion import (
    DEFAULT_PROMOTION_TAG,
    PromotionResult,
    promote_local_checkpoint,
)
from hunter_kinodynamic_rl.evaluation.local_subgoal_benchmark import (
    LocalBenchmarkScenarioSpec,
    build_local_benchmark_manifest,
    load_local_benchmark_manifest,
    save_local_benchmark_manifest,
)


STAGE2_SCHEMA_VERSION = 1
STAGE2_PROFILE_BY_LABEL = {
    "L0": "local_l0_direct_control",
    "L1": "local_l1_trajectory",
    "L2": "local_l2_temporal",
    "L3": "local_l3_supervised_risk",
    "L4": "local_l4_risk_actor",
    "L5": "local_l5_counterfactual",
}
DEFAULT_TRAINING_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_BENCHMARK_SCENARIOS = 20
# run_seed=0 deterministically duplicates test seed 20543 in its first 20
# draws. Seed 1 is pre-registered because its first 20 test seeds are unique.
DEFAULT_BENCHMARK_RUN_SEED = 1
# Pre-registered before results exist. Never replace this with the best test
# result: doing so would use the formal test set for model selection.
PROMOTION_LABEL = "L5"
PROMOTION_SEED = 0
SUMMARY_METRICS = (
    "feasible_subgoal_success_rate",
    "collision_rate",
    "timeout_rate",
    "high_risk_failure_rate",
    "infeasible_false_success_rate",
    "infeasible_collision_rate",
    "infeasible_high_risk_rate",
    "infeasible_safe_termination_rate",
    "time_to_goal_sec_mean",
    "time_to_termination_sec_mean",
)


class Stage2ContractError(RuntimeError):
    """A Stage-2 experiment artifact violates the pre-registered contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scenario_manifest_sha256(manifest: Sequence[LocalBenchmarkScenarioSpec]) -> str:
    """Match the benchmark artifact's canonical scenario-manifest digest."""
    payload = json.dumps(
        [{"scenario_id": row.scenario_id, "seed": row.seed, "mode": row.mode} for row in manifest],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def _validate_seeds(seeds: Sequence[int]) -> Tuple[int, ...]:
    normalized = tuple(int(seed) for seed in seeds)
    if len(normalized) < 5:
        raise Stage2ContractError(f"Stage 2 requires at least 5 training seeds, got {normalized}")
    if len(set(normalized)) != len(normalized):
        raise Stage2ContractError(f"training seeds must be unique, got {normalized}")
    if any(seed < 0 for seed in normalized):
        raise Stage2ContractError(f"training seeds must be non-negative, got {normalized}")
    return normalized


def seeded_profile_path(stage_root: os.PathLike, label: str, seed: int) -> Path:
    return Path(stage_root) / "profiles" / f"{label.lower()}_seed{seed}.yaml"


def training_root(stage_root: os.PathLike, label: str, seed: int) -> Path:
    return Path(stage_root) / "runs" / label / f"seed_{seed}"


def benchmark_artifact_path(stage_root: os.PathLike, label: str, seed: int) -> Path:
    return Path(stage_root) / "benchmarks" / label / f"seed_{seed}" / "artifact.json"


def shared_benchmark_manifest_path(stage_root: os.PathLike) -> Path:
    return Path(stage_root) / "benchmark_manifest.json"


def experiment_manifest_path(stage_root: os.PathLike) -> Path:
    return Path(stage_root) / "experiment_manifest.json"


def prepare_stage2(
    stage_root: os.PathLike,
    *,
    seeds: Sequence[int] = DEFAULT_TRAINING_SEEDS,
    num_scenarios: int = DEFAULT_BENCHMARK_SCENARIOS,
    benchmark_run_seed: int = DEFAULT_BENCHMARK_RUN_SEED,
) -> dict:
    """Create immutable seed profiles and one shared formal-test manifest.

    Existing files are accepted only when byte-identical to what this call
    would create. This prevents an interrupted campaign from silently changing
    a seed, profile or benchmark on resume.
    """
    root = Path(stage_root).resolve()
    normalized_seeds = _validate_seeds(seeds)
    if num_scenarios < DEFAULT_BENCHMARK_SCENARIOS:
        raise Stage2ContractError(
            f"formal Stage 2 requires at least {DEFAULT_BENCHMARK_SCENARIOS} scenarios, got {num_scenarios}"
        )
    config_root = Path(default_config_root())
    root.mkdir(parents=True, exist_ok=True)

    profile_records = {}
    for label, base_name in STAGE2_PROFILE_BY_LABEL.items():
        base_path = config_root / "profiles" / f"{base_name}.yaml"
        with base_path.open() as handle:
            base_yaml = yaml.safe_load(handle) or {}
        for seed in normalized_seeds:
            payload = json.loads(json.dumps(base_yaml))
            payload.setdefault("training", {})["seed"] = seed
            path = seeded_profile_path(root, label, seed)
            rendered = yaml.safe_dump(payload, sort_keys=False)
            if path.exists() and path.read_text() != rendered:
                raise Stage2ContractError(f"refusing to rewrite changed Stage-2 profile {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_text(rendered)
            profile = load_profile(str(path))
            if profile.training.seed != seed:
                raise Stage2ContractError(f"seeded profile {path} resolved seed {profile.training.seed}, expected {seed}")
            profile_records[f"{label}/seed_{seed}"] = {
                "base_profile": base_name,
                "path": str(path),
                "sha256": _sha256(path),
                "profile_name": profile.name,
                "max_timesteps": profile.training.max_timesteps,
            }

    benchmark_path = shared_benchmark_manifest_path(root)
    reference = load_profile(STAGE2_PROFILE_BY_LABEL["L0"])
    expected_manifest = build_local_benchmark_manifest(
        reference, num_scenarios=num_scenarios, run_seed=benchmark_run_seed,
    )
    if benchmark_path.exists():
        existing = load_local_benchmark_manifest(str(benchmark_path))
        if existing != expected_manifest:
            raise Stage2ContractError(f"existing shared benchmark manifest differs from the contract: {benchmark_path}")
    else:
        save_local_benchmark_manifest(expected_manifest, str(benchmark_path))

    payload = {
        "schema_version": STAGE2_SCHEMA_VERSION,
        "labels": STAGE2_PROFILE_BY_LABEL,
        "training_seeds": list(normalized_seeds),
        "minimum_training_seeds": 5,
        "benchmark": {
            "kind": "formal",
            "num_scenarios": num_scenarios,
            "run_seed": benchmark_run_seed,
            "manifest_path": str(benchmark_path),
            "file_sha256": _sha256(benchmark_path),
            "scenario_manifest_sha256": _scenario_manifest_sha256(expected_manifest),
        },
        "promotion_policy": {
            "label": PROMOTION_LABEL,
            "seed": PROMOTION_SEED,
            "checkpoint_tag": "final",
            "selection_uses_test_metrics": False,
        },
        "profiles": profile_records,
    }
    path = experiment_manifest_path(root)
    if path.exists():
        with path.open() as handle:
            existing = json.load(handle)
        if existing != payload:
            raise Stage2ContractError(f"existing experiment manifest differs from requested contract: {path}")
    else:
        _atomic_json(path, payload)
    return payload


def discover_run_dir(stage_root: os.PathLike, label: str, seed: int) -> Path | None:
    root = training_root(stage_root, label, seed)
    if not root.exists():
        return None
    runs = sorted(path for path in root.iterdir() if path.is_dir() and (path / "configs").is_dir())
    if len(runs) > 1:
        raise Stage2ContractError(
            f"{label}/seed_{seed} has {len(runs)} run directories under {root}; refuse ambiguous resume"
        )
    return runs[0] if runs else None


def checkpoint_manifest(run_dir: os.PathLike, tag: str = "final") -> dict | None:
    path = Path(run_dir) / "checkpoints" / tag / "manifest.json"
    if not path.is_file():
        return None
    with path.open() as handle:
        return json.load(handle)


def training_complete(run_dir: os.PathLike, profile_path: os.PathLike) -> bool:
    manifest = checkpoint_manifest(run_dir, "final")
    if manifest is None:
        return False
    profile = load_profile(str(profile_path))
    if int(manifest.get("global_step", -1)) < profile.training.max_timesteps:
        return False
    if manifest.get("profile_name") != profile.name:
        return False
    attestation = manifest.get("training_environment_attestation")
    if not isinstance(attestation, dict):
        return False
    if attestation.get("profile_name") != profile.name:
        return False
    if attestation.get("urdf_robot_name") != profile.robot.name:
        return False
    resolved_config = manifest.get("resolved_config")
    if not isinstance(resolved_config, dict):
        return False
    try:
        expected_fingerprint = training_profile_fingerprint(profile)
        return (
            training_profile_fingerprint_from_resolved_config(resolved_config) == expected_fingerprint
            and attestation.get("training_profile_fingerprint_sha256") == expected_fingerprint
        )
    except (KeyError, TypeError, ValueError):
        return False


def resume_checkpoint_tag(run_dir: os.PathLike) -> str | None:
    for tag in ("latest", "best"):
        if checkpoint_manifest(run_dir, tag) is not None:
            return tag
    return None


def stage2_status(stage_root: os.PathLike) -> dict:
    root = Path(stage_root).resolve()
    with experiment_manifest_path(root).open() as handle:
        contract = json.load(handle)
    rows = {}
    for label in STAGE2_PROFILE_BY_LABEL:
        for seed in contract["training_seeds"]:
            key = f"{label}/seed_{seed}"
            profile_path = Path(contract["profiles"][key]["path"])
            run_dir = discover_run_dir(root, label, seed)
            artifact_path = benchmark_artifact_path(root, label, seed)
            rows[key] = {
                "profile_path": str(profile_path),
                "run_dir": str(run_dir) if run_dir else None,
                "training_complete": bool(run_dir and training_complete(run_dir, profile_path)),
                "benchmark_complete": artifact_path.is_file(),
                "benchmark_artifact": str(artifact_path),
            }
    return {"stage_root": str(root), "rows": rows}


def _bootstrap_interval(values: Sequence[float], *, seed: int = 0, draws: int = 10_000) -> List[float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 1:
        value = float(array[0])
        return [value, value]
    rng = np.random.RandomState(seed)
    indices = rng.randint(0, array.size, size=(draws, array.size))
    means = array[indices].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def aggregate_stage2(stage_root: os.PathLike) -> dict:
    """Validate the complete 6xN matrix and aggregate seed-level metrics."""
    root = Path(stage_root).resolve()
    with experiment_manifest_path(root).open() as handle:
        contract = json.load(handle)
    seeds = tuple(contract["training_seeds"])
    _validate_seeds(seeds)
    expected_manifest_sha = contract["benchmark"]["file_sha256"]
    expected_scenario_hash = contract["benchmark"]["scenario_manifest_sha256"]
    rows = {}
    aggregates = {}
    missing = []
    incomplete_training = []

    for label in STAGE2_PROFILE_BY_LABEL:
        label_artifacts = []
        for seed in seeds:
            key = f"{label}/seed_{seed}"
            profile_path = Path(contract["profiles"][key]["path"])
            run_dir = discover_run_dir(root, label, seed)
            if run_dir is None or not training_complete(run_dir, profile_path):
                incomplete_training.append(key)
                continue
            final_manifest = checkpoint_manifest(run_dir, "final")
            path = benchmark_artifact_path(root, label, seed)
            if not path.is_file():
                missing.append(str(path))
                continue
            with path.open() as handle:
                artifact = json.load(handle)
            if artifact.get("benchmark_kind") != "formal":
                raise Stage2ContractError(f"{path}: benchmark_kind must be 'formal'")
            if artifact.get("num_scenarios") != contract["benchmark"]["num_scenarios"]:
                raise Stage2ContractError(f"{path}: num_scenarios differs from experiment manifest")
            scenario_hash = artifact.get("scenario_manifest_sha256")
            if scenario_hash != expected_scenario_hash:
                raise Stage2ContractError(f"{path}: not evaluated on the shared scenario manifest")
            expected_profile_name = contract["profiles"][key]["profile_name"]
            if artifact.get("profile_name") != expected_profile_name:
                raise Stage2ContractError(
                    f"{path}: profile_name={artifact.get('profile_name')!r}, expected {expected_profile_name!r}"
                )
            if artifact.get("checkpoint_generation") != final_manifest.get("generation"):
                raise Stage2ContractError(f"{path}: artifact checkpoint generation does not match final checkpoint")
            if artifact.get("checkpoint_sha256") != final_manifest.get("pt_sha256"):
                raise Stage2ContractError(f"{path}: artifact checkpoint SHA does not match final checkpoint")
            rows[key] = {
                "artifact_path": str(path),
                "checkpoint_generation": artifact.get("checkpoint_generation"),
                "checkpoint_sha256": artifact.get("checkpoint_sha256"),
                "summary": artifact["summary"],
            }
            label_artifacts.append(artifact)
        if len(label_artifacts) == len(seeds):
            metric_aggregates = {}
            for metric in SUMMARY_METRICS:
                values = [a["summary"].get(metric) for a in label_artifacts]
                finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
                metric_aggregates[metric] = {
                    "valid_seed_count": len(finite),
                    "mean": statistics.fmean(finite) if finite else None,
                    "std_population": statistics.pstdev(finite) if len(finite) > 1 else (0.0 if finite else None),
                    "bootstrap_95_ci": _bootstrap_interval(finite) if finite else None,
                }
            aggregates[label] = metric_aggregates

    if incomplete_training:
        raise Stage2ContractError(
            "cannot aggregate/promote before every Stage-2 final checkpoint reaches max_timesteps; "
            f"incomplete rows: {incomplete_training}"
        )
    if missing:
        raise Stage2ContractError(
            f"cannot aggregate/promote an incomplete Stage-2 matrix; missing {len(missing)} artifact(s): {missing}"
        )
    payload = {
        "schema_version": STAGE2_SCHEMA_VERSION,
        "experiment_manifest_sha256": _sha256(experiment_manifest_path(root)),
        "shared_benchmark_file_sha256": expected_manifest_sha,
        "scenario_manifest_sha256": expected_scenario_hash,
        "training_seed_count": len(seeds),
        "rows": rows,
        "aggregates": aggregates,
    }
    _atomic_json(root / "aggregate.json", payload)
    return payload


def promote_preregistered_local(
    stage_root: os.PathLike,
    *,
    target_dir: os.PathLike = "runtime/experiments/local_frozen/checkpoints",
    target_tag: str = DEFAULT_PROMOTION_TAG,
) -> PromotionResult:
    """Promote only the pre-registered L5/seed-0 final after a full matrix."""
    root = Path(stage_root).resolve()
    aggregate_stage2(root)  # completeness/integrity gate
    artifact_path = benchmark_artifact_path(root, PROMOTION_LABEL, PROMOTION_SEED)
    with artifact_path.open() as handle:
        artifact = json.load(handle)
    run_dir = discover_run_dir(root, PROMOTION_LABEL, PROMOTION_SEED)
    if run_dir is None:
        raise Stage2ContractError("pre-registered promotion run directory is missing")
    return promote_local_checkpoint(
        artifact,
        None,
        source_checkpoint_dir=str(run_dir / "checkpoints"),
        source_checkpoint_tag="final",
        target_dir=str(target_dir),
        target_tag=target_tag,
    )
