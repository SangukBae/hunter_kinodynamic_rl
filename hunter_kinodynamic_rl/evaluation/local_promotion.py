#!/usr/bin/env python3
"""Requirement B: config-driven acceptance gate + ``local_frozen`` checkpoint
promotion.

A Local checkpoint may be promoted (copied into a ``local_frozen`` target
directory that ``hierarchical_phase4.yaml``/``hierarchical_phase5_*.yaml``'s
``hierarchical_training.local_checkpoint_dir``/``local_checkpoint_name``
point at) ONLY after:

1. The benchmark artifact declares a supported ``schema_version`` (defect-
   fix item 2/3: an artifact from an older schema -- e.g. one still using
   the removed ``infeasible_goal_rejection_rate`` field -- is rejected
   outright, never silently reinterpreted under the new field names).
2. A formal (``benchmark_kind="formal"``, never ``"smoke"``) benchmark
   artifact (:mod:`evaluation.local_subgoal_benchmark`'s schema) exists and
   validates against :data:`REQUIRED_ARTIFACT_FIELDS`.
3. The artifact's recorded ``checkpoint_generation``/``checkpoint_sha256``
   match the checkpoint actually being promoted -- read INDEPENDENTLY from
   the source checkpoint's own on-disk ``manifest.json``/``model.pt``
   (defect-fix item 4: never promote checkpoint X on the strength of a
   caller-supplied manifest dict alone, which could be stale or simply
   wrong; a caller-supplied ``checkpoint_manifest`` is now a cross-check,
   not the source of truth).
4. The artifact's ``local_training_contract_fingerprint``/
   ``architecture_fingerprint`` match values independently RECOMPUTED from
   the source checkpoint's own ``resolved_config`` (never promote a
   checkpoint whose training-distribution or architecture identity has
   drifted from what was benchmarked, and never trust a manifest field
   that merely CLAIMS a fingerprint without the resolved_config backing
   it).
5. :func:`evaluate_local_acceptance` passes every configured threshold.

Any failure refuses promotion outright (returns
``PromotionResult(accepted=False, ...)``, never partially copies files) --
a failed/legacy checkpoint (missing required manifest fields, or one that
simply performed below threshold) can never end up in the ``local_frozen``
target. Promotion itself is atomic (defect-fix item 4): the source
generation is copied into a fresh temp directory under ``target_dir``,
``promotion_manifest.json`` is written into that temp directory, then a
single ``os.rename`` publishes it under its final (generation-UUID-derived,
collision-safe) name, and the ``target_tag`` symlink is swapped via a
temp-symlink + ``os.replace`` -- a crash or exception at any point leaves
either the OLD promoted checkpoint (if one existed) or nothing, never a
half-copied directory the tag could resolve to.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

from hunter_kinodynamic_rl.config.schema import ConfigError
from hunter_kinodynamic_rl.evaluation.fingerprint import (
    architecture_fingerprint_from_resolved_config, local_training_contract_fingerprint_from_resolved_config,
)
from hunter_kinodynamic_rl.evaluation.local_subgoal_benchmark import MIN_SUPPORTED_BENCHMARK_SCHEMA_VERSION
from hunter_kinodynamic_rl.rl.checkpointing.manager import sha256_of_file

#: The ONE canonical Local-promotion tag (defect-fix item 5) -- every
#: profile that consumes a promoted Local checkpoint
#: (``hierarchical_phase4.yaml`` and every ``hierarchical_phase5_*.yaml``)
#: must set ``hierarchical_training.local_checkpoint_name`` to this value.
#: ``promote_local_checkpoint``'s own ``target_tag`` default is this
#: constant, never a bare string literal duplicated at each call site.
DEFAULT_PROMOTION_TAG = "final"

REQUIRED_ARTIFACT_FIELDS = (
    "schema_version", "benchmark_kind", "checkpoint_generation", "checkpoint_sha256",
    "architecture_fingerprint", "local_training_contract_fingerprint", "scenario_manifest_sha256",
    "num_scenarios", "summary",
)
REQUIRED_CHECKPOINT_MANIFEST_FIELDS = (
    "generation", "pt_sha256", "state_dim", "action_dim", "resolved_config",
    "local_training_contract_fingerprint",
)
#: Fields a valid, non-tampered ``promotion_manifest.json`` must carry --
#: an empty ``{}`` marker, or one missing any of these, is never treated as
#: "promoted" (defect-fix item 4).
REQUIRED_PROMOTION_MANIFEST_FIELDS = (
    "promoted_at_unix", "source_checkpoint_dir", "source_checkpoint_tag", "checkpoint_generation",
    "checkpoint_sha256", "local_training_contract_fingerprint", "architecture_fingerprint",
    "acceptance", "benchmark_artifact_summary", "benchmark_num_scenarios", "benchmark_scenario_manifest_sha256",
)


class PromotionValidationError(RuntimeError):
    """Raised only by internal helpers -- every PUBLIC function in this
    module catches this and reports it as a ``PromotionResult``/validation
    dataclass ``reasons`` entry instead of letting it propagate, so a
    caller never needs its own try/except around promotion logic."""


@dataclass
class LocalAcceptanceConfig:
    """Config-driven acceptance thresholds (requirement B: "acceptance
    threshold는 config로 관리한다"). Loaded from a small standalone YAML
    (``config/local_acceptance.yaml`` by default) rather than folded into
    the large training ``Profile`` schema -- promotion/benchmark-acceptance
    is an evaluation/ops-time concern, orthogonal to what a training
    profile itself declares.

    Defect-fix item 2: ``min_subgoal_success_rate``/
    ``min_infeasible_goal_rejection_rate`` are gone (see
    ``local_subgoal_benchmark``'s module docstring) -- replaced by
    ``min_feasible_subgoal_success_rate`` (conditioned on feasible
    scenarios only) and the ``*_infeasible_*``/``min_infeasible_scenarios``
    fields below, all conditioned on a non-empty infeasible-scenario
    sample."""

    min_scenarios: int = 20
    min_feasible_subgoal_success_rate: float = 0.5
    max_collision_rate: float = 0.15
    max_timeout_rate: float = 0.3
    max_high_risk_failure_rate: float = 0.15
    #: A formal benchmark producing FEWER infeasible scenarios than this
    #: cannot support any infeasible-conditional conclusion -- promotion is
    #: refused outright (not silently skipped) rather than evaluating those
    #: thresholds against too small a sample. See config/local_acceptance.yaml's
    #: comment for the binomial-sample-size rationale at the default
    #: scenario.goal_infeasible_fraction/num_scenarios.
    min_infeasible_scenarios: int = 1
    max_infeasible_false_success_rate: float = 0.5
    max_infeasible_collision_rate: float = 0.5
    max_infeasible_high_risk_rate: float = 0.5
    min_infeasible_safe_termination_rate: float = 0.34

    def validate(self) -> None:
        if self.min_scenarios < 1:
            raise ConfigError("local_acceptance.min_scenarios must be >= 1")
        if self.min_infeasible_scenarios < 0:
            raise ConfigError("local_acceptance.min_infeasible_scenarios must be >= 0")
        for name in (
            "min_feasible_subgoal_success_rate", "max_collision_rate", "max_timeout_rate",
            "max_high_risk_failure_rate", "max_infeasible_false_success_rate", "max_infeasible_collision_rate",
            "max_infeasible_high_risk_rate", "min_infeasible_safe_termination_rate",
        ):
            value = getattr(self, name)
            if not (0.0 <= value <= 1.0):
                raise ConfigError(f"local_acceptance.{name} must be in [0, 1], got {value}")


def load_local_acceptance_config(path: Optional[str] = None) -> LocalAcceptanceConfig:
    if path is None:
        here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(here, "config", "local_acceptance.yaml")
    if not os.path.isfile(path):
        cfg = LocalAcceptanceConfig()
        cfg.validate()
        return cfg
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    known = {f.name for f in LocalAcceptanceConfig.__dataclass_fields__.values()}
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"local_acceptance.yaml has unknown field(s): {sorted(unknown)}")
    cfg = LocalAcceptanceConfig(**raw)
    cfg.validate()
    return cfg


@dataclass
class AcceptanceReport:
    accepted: bool
    checks: Dict[str, bool] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)


def evaluate_local_acceptance(artifact: Dict[str, Any], cfg: LocalAcceptanceConfig) -> AcceptanceReport:
    """``artifact``: a :func:`evaluation.local_subgoal_benchmark.build_local_benchmark_artifact`-shaped
    dict (or an equivalent synthetic fixture with the same schema)."""
    missing = [f for f in REQUIRED_ARTIFACT_FIELDS if f not in artifact]
    if missing:
        return AcceptanceReport(accepted=False, reasons=[f"benchmark artifact missing required field(s): {missing}"])
    schema_version = artifact["schema_version"]
    if schema_version < MIN_SUPPORTED_BENCHMARK_SCHEMA_VERSION:
        return AcceptanceReport(
            accepted=False,
            reasons=[
                f"benchmark artifact schema_version={schema_version} is older than the minimum supported "
                f"version {MIN_SUPPORTED_BENCHMARK_SCHEMA_VERSION} -- re-run the benchmark with the current "
                "evaluation.local_subgoal_benchmark to get the current metric definitions "
                "(the old infeasible_goal_rejection_rate field is gone, not reinterpreted)"
            ],
        )
    if artifact["benchmark_kind"] != "formal":
        return AcceptanceReport(
            accepted=False,
            reasons=[f"benchmark_kind={artifact['benchmark_kind']!r} -- only a 'formal' artifact may be "
                     "evaluated for promotion, never 'smoke'"],
        )
    summary = artifact["summary"]
    n_infeasible = summary.get("infeasible_valid_count", 0)
    have_enough_infeasible = n_infeasible >= cfg.min_infeasible_scenarios

    checks = {
        "min_scenarios": artifact["num_scenarios"] >= cfg.min_scenarios,
        "min_feasible_subgoal_success_rate": (
            summary["feasible_subgoal_success_rate"] is not None
            and summary["feasible_subgoal_success_rate"] >= cfg.min_feasible_subgoal_success_rate
        ),
        "max_collision_rate": summary["collision_rate"] <= cfg.max_collision_rate,
        "max_timeout_rate": summary["timeout_rate"] <= cfg.max_timeout_rate,
        "max_high_risk_failure_rate": summary["high_risk_failure_rate"] <= cfg.max_high_risk_failure_rate,
        "min_infeasible_scenarios": have_enough_infeasible,
    }
    reasons = []
    if not checks["min_scenarios"]:
        reasons.append(f"num_scenarios={artifact['num_scenarios']} < min_scenarios={cfg.min_scenarios}")
    if not checks["min_feasible_subgoal_success_rate"]:
        reasons.append(
            f"feasible_subgoal_success_rate={summary['feasible_subgoal_success_rate']!r} < "
            f"min_feasible_subgoal_success_rate={cfg.min_feasible_subgoal_success_rate:.3f} "
            "(None means every scenario in this benchmark was infeasible -- 0 feasible samples)"
        )
    if not checks["max_collision_rate"]:
        reasons.append(f"collision_rate={summary['collision_rate']:.3f} > max_collision_rate={cfg.max_collision_rate:.3f}")
    if not checks["max_timeout_rate"]:
        reasons.append(f"timeout_rate={summary['timeout_rate']:.3f} > max_timeout_rate={cfg.max_timeout_rate:.3f}")
    if not checks["max_high_risk_failure_rate"]:
        reasons.append(
            f"high_risk_failure_rate={summary['high_risk_failure_rate']:.3f} > "
            f"max_high_risk_failure_rate={cfg.max_high_risk_failure_rate:.3f}"
        )
    if not checks["min_infeasible_scenarios"]:
        reasons.append(
            f"infeasible_valid_count={n_infeasible} < min_infeasible_scenarios={cfg.min_infeasible_scenarios} "
            "-- too few infeasible scenarios in this run to trust ANY infeasible-conditional metric; re-run "
            "the benchmark (a different run_seed will draw a different infeasible/feasible split)"
        )

    # The four infeasible-conditional checks are only MEANINGFUL once
    # min_infeasible_scenarios passed -- when it didn't, `checks` above
    # already carries the failure and these are reported as skipped
    # (never silently passed) rather than compared against a None rate.
    infeasible_checks = {
        "max_infeasible_false_success_rate": None,
        "max_infeasible_collision_rate": None,
        "max_infeasible_high_risk_rate": None,
        "min_infeasible_safe_termination_rate": None,
    }
    if have_enough_infeasible:
        infeasible_checks["max_infeasible_false_success_rate"] = (
            summary["infeasible_false_success_rate"] <= cfg.max_infeasible_false_success_rate
        )
        infeasible_checks["max_infeasible_collision_rate"] = (
            summary["infeasible_collision_rate"] <= cfg.max_infeasible_collision_rate
        )
        infeasible_checks["max_infeasible_high_risk_rate"] = (
            summary["infeasible_high_risk_rate"] <= cfg.max_infeasible_high_risk_rate
        )
        infeasible_checks["min_infeasible_safe_termination_rate"] = (
            summary["infeasible_safe_termination_rate"] >= cfg.min_infeasible_safe_termination_rate
        )
        if not infeasible_checks["max_infeasible_false_success_rate"]:
            reasons.append(
                f"infeasible_false_success_rate={summary['infeasible_false_success_rate']:.3f} > "
                f"max_infeasible_false_success_rate={cfg.max_infeasible_false_success_rate:.3f}"
            )
        if not infeasible_checks["max_infeasible_collision_rate"]:
            reasons.append(
                f"infeasible_collision_rate={summary['infeasible_collision_rate']:.3f} > "
                f"max_infeasible_collision_rate={cfg.max_infeasible_collision_rate:.3f}"
            )
        if not infeasible_checks["max_infeasible_high_risk_rate"]:
            reasons.append(
                f"infeasible_high_risk_rate={summary['infeasible_high_risk_rate']:.3f} > "
                f"max_infeasible_high_risk_rate={cfg.max_infeasible_high_risk_rate:.3f}"
            )
        if not infeasible_checks["min_infeasible_safe_termination_rate"]:
            reasons.append(
                f"infeasible_safe_termination_rate={summary['infeasible_safe_termination_rate']:.3f} < "
                f"min_infeasible_safe_termination_rate={cfg.min_infeasible_safe_termination_rate:.3f}"
            )
    checks.update(infeasible_checks)
    # None (skipped) never counts as a pass -- `all()` over a dict with a
    # None value would otherwise treat None as falsy correctly, but be
    # explicit rather than rely on that: a skipped check factors into
    # `accepted` as a fail via min_infeasible_scenarios already having
    # failed above.
    accepted = all(v is True for k, v in checks.items() if k not in infeasible_checks or have_enough_infeasible) \
        if have_enough_infeasible else False
    return AcceptanceReport(accepted=accepted, checks=checks, reasons=reasons)


def _load_source_checkpoint_manifest(source_tag_path: str) -> Dict[str, Any]:
    """Defect-fix item 4: read the checkpoint identity DIRECTLY from the
    resolved source generation directory -- never trust a caller-supplied
    manifest dict as the source of truth. Independently recomputes
    ``model.pt``'s actual SHA-256 and rejects a manifest whose recorded
    ``pt_sha256`` doesn't match the real file on disk (a stale or
    hand-edited ``manifest.json``, or a tampered/corrupted ``model.pt``)."""
    manifest_path = os.path.join(source_tag_path, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise PromotionValidationError(f"source checkpoint has no manifest.json at {manifest_path}")
    with open(manifest_path) as f:
        manifest = json.load(f)
    missing = [f for f in REQUIRED_CHECKPOINT_MANIFEST_FIELDS if f not in manifest]
    if missing:
        raise PromotionValidationError(
            f"source checkpoint manifest.json at {manifest_path} missing required field(s) {missing} -- "
            "legacy/incomplete checkpoints cannot be promoted"
        )
    pt_path = os.path.join(source_tag_path, "model.pt")
    if not os.path.isfile(pt_path):
        raise PromotionValidationError(f"source checkpoint has no model.pt at {pt_path}")
    actual_sha256 = sha256_of_file(pt_path)
    if actual_sha256 != manifest["pt_sha256"]:
        raise PromotionValidationError(
            f"source checkpoint model.pt actual sha256={actual_sha256} != manifest.json recorded "
            f"pt_sha256={manifest['pt_sha256']!r} at {source_tag_path} -- the checkpoint file was modified "
            "or corrupted after being saved"
        )
    return manifest


@dataclass
class PromotionMarkerValidation:
    """Structured replacement for a bare boolean "is this promoted" check
    (defect-fix item 4) -- ``promoted`` is only True once every field/hash
    cross-check below has passed; ``reasons`` explains any failure
    (missing marker, empty ``{}`` marker, tampered/stale marker, or a
    marker whose recorded checkpoint identity no longer matches the actual
    files sitting next to it)."""

    promoted: bool
    reasons: List[str] = field(default_factory=list)
    promotion_manifest: Optional[Dict[str, Any]] = None


def validate_promoted_local_checkpoint(local_checkpoint_dir: str, local_checkpoint_tag: str) -> PromotionMarkerValidation:
    """Requirement C / defect-fix item 4: structured replacement for the
    old file-existence-only ``is_local_checkpoint_promoted``. A checkpoint
    directory is reported ``promoted=True`` only when ALL of the following
    hold:

    1. ``promotion_manifest.json`` exists, parses, and is non-empty (an
       empty ``{}`` marker -- e.g. from an interrupted/failed write -- is
       rejected).
    2. It carries every field in :data:`REQUIRED_PROMOTION_MANIFEST_FIELDS`.
    3. The SAME directory also has a ``manifest.json``/``model.pt`` (the
       full checkpoint content the promotion copied), and
       ``promotion_manifest``'s recorded ``checkpoint_generation``/
       ``checkpoint_sha256`` match that ``manifest.json``'s own
       ``generation``/``pt_sha256`` -- a tampered or stale
       ``promotion_manifest.json`` (edited independently of the checkpoint
       files it describes) is rejected.
    4. ``model.pt``'s ACTUAL sha256 on disk still matches -- catches the
       checkpoint files themselves being swapped out after promotion.
    """
    tag_path = os.path.realpath(os.path.join(local_checkpoint_dir, local_checkpoint_tag))
    marker_path = os.path.join(tag_path, "promotion_manifest.json")
    if not os.path.isfile(marker_path):
        return PromotionMarkerValidation(promoted=False, reasons=[f"no promotion_manifest.json at {marker_path}"])
    try:
        with open(marker_path) as f:
            marker = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return PromotionMarkerValidation(promoted=False, reasons=[f"promotion_manifest.json unreadable: {e!r}"])
    if not marker:
        return PromotionMarkerValidation(promoted=False, reasons=["promotion_manifest.json is empty ({}) -- rejected"])
    missing = [f for f in REQUIRED_PROMOTION_MANIFEST_FIELDS if f not in marker]
    if missing:
        return PromotionMarkerValidation(
            promoted=False, reasons=[f"promotion_manifest.json missing required field(s) {missing}"],
        )
    try:
        checkpoint_manifest = _load_source_checkpoint_manifest(tag_path)
    except PromotionValidationError as e:
        return PromotionMarkerValidation(promoted=False, reasons=[str(e)])
    reasons = []
    if marker["checkpoint_generation"] != checkpoint_manifest["generation"]:
        reasons.append(
            f"promotion_manifest checkpoint_generation={marker['checkpoint_generation']!r} != "
            f"manifest.json generation={checkpoint_manifest['generation']!r} -- stale or tampered marker"
        )
    if marker["checkpoint_sha256"] != checkpoint_manifest["pt_sha256"]:
        reasons.append(
            f"promotion_manifest checkpoint_sha256={marker['checkpoint_sha256']!r} != "
            f"manifest.json pt_sha256={checkpoint_manifest['pt_sha256']!r} -- stale or tampered marker"
        )
    if reasons:
        return PromotionMarkerValidation(promoted=False, reasons=reasons, promotion_manifest=marker)
    return PromotionMarkerValidation(promoted=True, promotion_manifest=marker)


def is_local_checkpoint_promoted(local_checkpoint_dir: str, local_checkpoint_tag: str) -> bool:
    """Backward-compatible boolean wrapper over
    :func:`validate_promoted_local_checkpoint` -- prefer the structured
    validator in new code that needs to explain WHY a checkpoint isn't
    considered promoted (e.g. Global preflight diagnostics)."""
    return validate_promoted_local_checkpoint(local_checkpoint_dir, local_checkpoint_tag).promoted


@dataclass
class PromotionResult:
    accepted: bool
    reasons: List[str] = field(default_factory=list)
    target_dir: Optional[str] = None
    target_tag: Optional[str] = None


def _atomic_publish_generation(source_tag_path: str, target_dir: str, promotion_manifest_payload: Dict[str, Any]) -> str:
    """Defect-fix item 4: copy-to-temp -> write marker -> single atomic
    rename to the final (generation-UUID-derived, collision-safe) name. A
    pre-existing destination for a DIFFERENT checkpoint is refused (never
    overwritten); for the SAME checkpoint (identical generation/sha) it is
    treated as an idempotent no-op, preserving whatever is already there
    untouched."""
    generation = promotion_manifest_payload["checkpoint_generation"]
    dest_generation_dir = os.path.join(target_dir, f"{generation}_promoted")
    if os.path.isdir(dest_generation_dir):
        existing = validate_promoted_local_checkpoint(target_dir, os.path.basename(dest_generation_dir))
        already_same = (
            existing.promotion_manifest is not None
            and existing.promotion_manifest.get("checkpoint_generation") == generation
            and existing.promotion_manifest.get("checkpoint_sha256") == promotion_manifest_payload["checkpoint_sha256"]
        )
        if already_same:
            return dest_generation_dir
        raise PromotionValidationError(
            f"promotion destination {dest_generation_dir} already exists and does not match the checkpoint "
            "currently being promoted -- refusing to overwrite an existing promoted checkpoint"
        )
    tmp_dir = os.path.join(target_dir, f".tmp-promote-{uuid.uuid4().hex}")
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)
    shutil.copytree(source_tag_path, tmp_dir)
    with open(os.path.join(tmp_dir, "promotion_manifest.json"), "w") as f:
        json.dump(promotion_manifest_payload, f, indent=2, sort_keys=True)
    os.rename(tmp_dir, dest_generation_dir)  # atomic on the same filesystem/target_dir
    return dest_generation_dir


def _atomic_replace_tag_symlink(target_dir: str, target_tag: str, link_target_basename: str) -> None:
    """Defect-fix item 4: tag swap via temp-symlink + ``os.replace`` --
    atomic (never a remove-then-create window where the tag resolves to
    nothing), and never silently deletes a real (non-symlink) directory
    that happens to occupy the tag path (that would indicate an unexpected
    on-disk layout, so this raises instead of guessing)."""
    tag_link = os.path.join(target_dir, target_tag)
    if os.path.exists(tag_link) or os.path.islink(tag_link):
        if not os.path.islink(tag_link):
            raise PromotionValidationError(
                f"promotion tag path {tag_link} exists and is NOT a symlink -- refusing to replace an "
                "unexpected on-disk layout automatically"
            )
    tmp_link = os.path.join(target_dir, f".tmp-tag-{uuid.uuid4().hex}")
    if os.path.islink(tmp_link) or os.path.exists(tmp_link):
        os.remove(tmp_link)
    os.symlink(link_target_basename, tmp_link)
    os.replace(tmp_link, tag_link)


def promote_local_checkpoint(
    artifact: Dict[str, Any], checkpoint_manifest: Optional[Dict[str, Any]] = None, *,
    source_checkpoint_dir: str, source_checkpoint_tag: str, target_dir: str, target_tag: str = DEFAULT_PROMOTION_TAG,
    acceptance_cfg: Optional[LocalAcceptanceConfig] = None, copy_files: bool = True,
) -> PromotionResult:
    """``checkpoint_manifest``: OPTIONAL caller-supplied manifest dict
    (e.g. from ``rl.checkpointing.manager.load_generation(...)["manifest"]``)
    used ONLY as a cross-check against the manifest this function reads
    directly from ``source_checkpoint_dir/source_checkpoint_tag`` on disk
    (defect-fix item 4: the on-disk read is authoritative; a caller-passed
    manifest that disagrees with it is treated as a hard mismatch, never
    silently preferred). ``copy_files=False`` runs every check without
    touching the filesystem (used by unit tests to exercise the gate logic
    against synthetic fixtures without real checkpoint files on disk) --
    in that mode the on-disk manifest read is skipped and the caller-
    supplied ``checkpoint_manifest`` (required in that case) is used
    directly, since there is no real checkpoint directory to read from."""
    cfg = acceptance_cfg or load_local_acceptance_config()

    schema_version = artifact.get("schema_version")
    if schema_version is None or schema_version < MIN_SUPPORTED_BENCHMARK_SCHEMA_VERSION:
        return PromotionResult(
            accepted=False,
            reasons=[
                f"benchmark artifact schema_version={schema_version!r} is older than the minimum supported "
                f"version {MIN_SUPPORTED_BENCHMARK_SCHEMA_VERSION} -- re-run the benchmark"
            ],
        )

    if copy_files:
        source_tag_path = os.path.realpath(os.path.join(source_checkpoint_dir, source_checkpoint_tag))
        if not os.path.isdir(source_tag_path):
            return PromotionResult(accepted=False, reasons=[f"source checkpoint directory not found: {source_tag_path}"])
        try:
            on_disk_manifest = _load_source_checkpoint_manifest(source_tag_path)
        except PromotionValidationError as e:
            return PromotionResult(accepted=False, reasons=[str(e)])
        if checkpoint_manifest is not None:
            cross_check_mismatches = [
                f"{field_name}: caller-supplied={checkpoint_manifest.get(field_name)!r} != "
                f"on-disk={on_disk_manifest.get(field_name)!r}"
                for field_name in ("generation", "pt_sha256")
                if checkpoint_manifest.get(field_name) != on_disk_manifest.get(field_name)
            ]
            if cross_check_mismatches:
                return PromotionResult(
                    accepted=False,
                    reasons=["caller-supplied checkpoint_manifest disagrees with the on-disk checkpoint:"]
                    + cross_check_mismatches,
                )
        checkpoint_manifest = on_disk_manifest
    else:
        if checkpoint_manifest is None:
            return PromotionResult(
                accepted=False, reasons=["copy_files=False requires an explicit checkpoint_manifest fixture"],
            )
        missing_ckpt = [f for f in REQUIRED_CHECKPOINT_MANIFEST_FIELDS if f not in checkpoint_manifest]
        if missing_ckpt:
            return PromotionResult(
                accepted=False,
                reasons=[f"checkpoint manifest missing required field(s) {missing_ckpt} -- legacy/incomplete "
                         "checkpoints cannot be promoted"],
            )

    acceptance = evaluate_local_acceptance(artifact, cfg)
    if not acceptance.accepted:
        return PromotionResult(accepted=False, reasons=list(acceptance.reasons))

    identity_mismatches = []
    if artifact["checkpoint_generation"] != checkpoint_manifest["generation"]:
        identity_mismatches.append(
            f"checkpoint_generation: artifact={artifact['checkpoint_generation']!r} != "
            f"checkpoint={checkpoint_manifest['generation']!r}"
        )
    if artifact["checkpoint_sha256"] != checkpoint_manifest["pt_sha256"]:
        identity_mismatches.append(
            f"checkpoint_sha256: artifact={artifact['checkpoint_sha256']!r} != "
            f"checkpoint={checkpoint_manifest['pt_sha256']!r}"
        )
    if artifact["local_training_contract_fingerprint"] != checkpoint_manifest["local_training_contract_fingerprint"]:
        identity_mismatches.append(
            "local_training_contract_fingerprint: artifact="
            f"{artifact['local_training_contract_fingerprint']!r} != checkpoint="
            f"{checkpoint_manifest['local_training_contract_fingerprint']!r}"
        )
    # defect-fix item 4: architecture identity, recomputed from the
    # checkpoint's OWN resolved_config -- never trust a manifest field that
    # merely claims a fingerprint without the config that would produce it
    # (this also transitively covers state_dim/action_dim: any change to
    # either is a change to ARCHITECTURE_SECTIONS, which changes this hash).
    resolved_config = checkpoint_manifest.get("resolved_config")
    if resolved_config is None:
        identity_mismatches.append("checkpoint manifest has no 'resolved_config' -- cannot verify architecture identity")
    else:
        recomputed_arch = architecture_fingerprint_from_resolved_config(resolved_config)
        if artifact["architecture_fingerprint"] != recomputed_arch:
            identity_mismatches.append(
                f"architecture_fingerprint: artifact={artifact['architecture_fingerprint']!r} != "
                f"checkpoint resolved_config-derived={recomputed_arch!r}"
            )
        recomputed_contract = local_training_contract_fingerprint_from_resolved_config(resolved_config)
        if checkpoint_manifest["local_training_contract_fingerprint"] != recomputed_contract:
            identity_mismatches.append(
                "checkpoint manifest local_training_contract_fingerprint="
                f"{checkpoint_manifest['local_training_contract_fingerprint']!r} is internally inconsistent with "
                f"its own resolved_config-derived value={recomputed_contract!r}"
            )
    for dim_field in ("state_dim", "action_dim"):
        value = checkpoint_manifest.get(dim_field)
        if not isinstance(value, int) or value <= 0:
            identity_mismatches.append(f"checkpoint manifest {dim_field}={value!r} is not a positive integer")
    if identity_mismatches:
        return PromotionResult(
            accepted=False,
            reasons=["benchmark artifact does not identify THIS checkpoint:"] + identity_mismatches,
        )

    result = PromotionResult(accepted=True, target_dir=target_dir, target_tag=target_tag)
    if not copy_files:
        return result

    os.makedirs(target_dir, exist_ok=True)
    promotion_manifest_payload = {
        "promoted_at_unix": time.time(),
        "source_checkpoint_dir": source_checkpoint_dir, "source_checkpoint_tag": source_checkpoint_tag,
        "checkpoint_generation": checkpoint_manifest["generation"],
        "checkpoint_sha256": checkpoint_manifest["pt_sha256"],
        "local_training_contract_fingerprint": checkpoint_manifest["local_training_contract_fingerprint"],
        "architecture_fingerprint": artifact["architecture_fingerprint"],
        "acceptance": {"checks": acceptance.checks, "reasons": acceptance.reasons},
        "benchmark_artifact_summary": artifact["summary"],
        "benchmark_num_scenarios": artifact["num_scenarios"],
        "benchmark_scenario_manifest_sha256": artifact["scenario_manifest_sha256"],
    }
    try:
        dest_generation_dir = _atomic_publish_generation(source_tag_path, target_dir, promotion_manifest_payload)
        _atomic_replace_tag_symlink(target_dir, target_tag, os.path.basename(dest_generation_dir))
    except PromotionValidationError as e:
        # Nothing published by THIS call is left half-done -- either the
        # copy-to-temp step never reached os.rename (temp dir remains, tag
        # untouched -- still pointing at whatever it pointed at before) or
        # the destination collision case above returned before touching
        # the filesystem at all.
        return PromotionResult(accepted=False, reasons=[str(e)])

    result.target_dir = dest_generation_dir
    return result
