#!/usr/bin/env python3
"""Defect-fix item 10: strict Global-checkpoint validation shared by BOTH
the A/B live benchmark (``run_live_hierarchical_benchmark.py``) and the A-G
live ablation suite (``run_live_ablation_suite.py``) -- extracted from the
former (which already had the strict version) so the latter can no longer
run a materially WEAKER per-label check (previously: architecture
fingerprint only, never resolved_config/replay-schema/RNG-basis/Local
identity) and silently drift apart from the A/B gate over time.

Also home to the "formal" cross-cutting preconditions common to both
runners: mode must be "test", the profile's own budget must be used
unmodified (never a smoke-sized override silently shrinking a "formal"
run), and the Local checkpoint feeding the run must be an actually-PROMOTED
one (never a raw/failed/legacy Local checkpoint)."""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Optional

#: A formal A/B or A-G run must cover at least this many scenarios --
#: shared so the two runners can never drift on this number independently.
FORMAL_MIN_SCENARIOS = 20

#: Defect-fix item 10: "total_optimizer_updates > 0 및 최소 학습량 config
#: gate" -- a Global checkpoint that never actually trained (0 optimizer
#: updates, e.g. a freshly-initialized/never-resumed run saved by mistake)
#: must never be usable as a "trained" ablation B/C/.../G policy in a
#: formal benchmark.
DEFAULT_MIN_TOTAL_OPTIMIZER_UPDATES = 1


def validate_global_checkpoint_manifest(
    manifest: Dict[str, Any], profile, current_local_manifest: Dict[str, Any], *,
    min_total_optimizer_updates: int = DEFAULT_MIN_TOTAL_OPTIMIZER_UPDATES,
) -> None:
    """Fail fast on every Global/Local pairing and observation contract.

    Kept independent of ROS/Gazebo so this load-bearing benchmark gate can
    be exhaustively unit-tested without starting a simulator. Raises
    :class:`RuntimeError` (never returns a partial/soft result) the first
    time ANY mismatch is found; every mismatch found is included in one
    message, never just the first.
    """
    from hunter_kinodynamic_rl.evaluation.fingerprint import (
        compare_local_checkpoint_identity,
        hierarchical_architecture_fingerprint,
        sha256_of_obj,
    )
    from hunter_kinodynamic_rl.navigation.global_rl.observation import (
        resolve_candidate_feature_names,
        resolve_map_channel_names,
    )
    from hunter_kinodynamic_rl.navigation.global_rl.replay_schema import (
        SCHEMA_VERSION as GLOBAL_REPLAY_SCHEMA_VERSION,
    )
    from hunter_kinodynamic_rl.navigation.memory.topological_graph import NODE_FEATURE_NAMES

    required_fields = (
        "hierarchical_architecture_fingerprint", "resolved_config", "resolved_config_hash",
        "local_checkpoint_manifest_summary", "map_channel_names", "candidate_feature_names",
        "node_feature_names", "max_nodes_in_observation", "memory_enabled",
        "feasibility_enabled", "global_risk_enabled", "global_replay_schema_version",
        "loop_state", "training_steps", "total_optimizer_updates", "epsilon_schedule_basis",
    )
    mismatches = []
    missing = [name for name in required_fields if name not in manifest]
    if missing:
        mismatches.append(f"missing required field(s): {missing}")

    current_fp = hierarchical_architecture_fingerprint(profile)
    if manifest.get("hierarchical_architecture_fingerprint") != current_fp:
        mismatches.append(
            "hierarchical_architecture_fingerprint: "
            f"checkpoint={manifest.get('hierarchical_architecture_fingerprint')!r} current={current_fp!r}"
        )

    recorded_config = manifest.get("resolved_config")
    if isinstance(recorded_config, dict):
        derived_hash = sha256_of_obj(recorded_config)
        current_hash = sha256_of_obj(dataclasses.asdict(profile))
        if manifest.get("resolved_config_hash") != derived_hash:
            mismatches.append(
                f"resolved_config_hash: manifest={manifest.get('resolved_config_hash')!r} "
                f"derived={derived_hash!r}"
            )
        if derived_hash != current_hash:
            mismatches.append(f"resolved profile hash: checkpoint={derived_hash} current={current_hash}")
    elif "resolved_config" in manifest:
        mismatches.append("resolved_config must be a dict")

    memory_enabled = bool(profile.memory.enabled and profile.global_rl.topology_feedback_enabled)
    expected_values = {
        "map_channel_names": list(resolve_map_channel_names(profile.global_rl)),
        "candidate_feature_names": list(resolve_candidate_feature_names(profile.global_rl)),
        "node_feature_names": list(NODE_FEATURE_NAMES) if memory_enabled else [],
        "max_nodes_in_observation": profile.memory.max_nodes_in_observation if memory_enabled else 0,
        "memory_enabled": memory_enabled,
        "feasibility_enabled": bool(
            profile.feasibility.enabled and profile.global_rl.feasibility_feedback_enabled
        ),
        "global_risk_enabled": bool(
            profile.feasibility.enabled and profile.global_rl.global_risk_feedback_enabled
        ),
        "global_replay_schema_version": GLOBAL_REPLAY_SCHEMA_VERSION,
        "epsilon_schedule_basis": "agent_training_steps",
    }
    for field_name, expected in expected_values.items():
        if field_name in manifest and manifest[field_name] != expected:
            mismatches.append(
                f"{field_name}: checkpoint={manifest[field_name]!r} current={expected!r}"
            )

    mismatches.extend(compare_local_checkpoint_identity(
        manifest.get("local_checkpoint_manifest_summary"), current_local_manifest,
    ))

    total_optimizer_updates = manifest.get("total_optimizer_updates")
    if not isinstance(total_optimizer_updates, int) or total_optimizer_updates < min_total_optimizer_updates:
        mismatches.append(
            f"total_optimizer_updates={total_optimizer_updates!r} < required minimum "
            f"{min_total_optimizer_updates} -- this checkpoint has not actually trained enough (or at all) "
            "to be used as a formal ablation policy"
        )

    if mismatches:
        raise RuntimeError(
            "validate_global_checkpoint_manifest: incompatible Global checkpoint; refusing benchmark:\n  "
            + "\n  ".join(mismatches)
        )


def require_formal_mode_is_test(formal: bool, mode: str) -> None:
    """Defect-fix item 10: "formal에서는 mode는 반드시 test"."""
    if formal and mode != "test":
        raise RuntimeError(f"formal=True requires mode='test' (never a train/validation-pool seed), got {mode!r}")


def require_formal_budget_unmodified(formal: bool, max_options: int, max_local_steps: int) -> None:
    """Defect-fix item 10: "smoke용 max_options/max_local_steps override
    금지 -- 실제 profile 기본 budget 사용" -- a formal run must exercise the
    profile's OWN configured per-mission budget, never a caller-shrunk
    smoke-sized one silently passed off as formal."""
    if formal and (max_options or max_local_steps):
        raise RuntimeError(
            "formal=True must use the profile's own hierarchical_training budget "
            "(max_global_options_per_mission/max_local_steps_per_option) -- max_options/max_local_steps "
            f"overrides (got max_options={max_options}, max_local_steps={max_local_steps}) are smoke-only; "
            "pass 0 for both (the default) under formal=True, or set formal=False"
        )


def require_promoted_local_checkpoint(local_checkpoint_dir: str, local_checkpoint_tag: str, *, context: str) -> None:
    """Defect-fix item 10: "Local checkpoint가 유효하게 promoted되었는지
    검증" -- a formal A/B or A-G run's Local TQC must be the actual
    promoted artifact (structured validator, never a bare file-existence
    check), not a raw/failed/legacy training-run checkpoint."""
    from hunter_kinodynamic_rl.evaluation.local_promotion import validate_promoted_local_checkpoint

    validation = validate_promoted_local_checkpoint(local_checkpoint_dir, local_checkpoint_tag)
    if not validation.promoted:
        raise RuntimeError(
            f"{context}: requires a PROMOTED Local checkpoint at "
            f"{local_checkpoint_dir!r}/{local_checkpoint_tag!r} -- {'; '.join(validation.reasons)}"
        )
