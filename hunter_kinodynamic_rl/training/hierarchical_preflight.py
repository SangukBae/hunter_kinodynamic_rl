#!/usr/bin/env python3
"""Requirement C: Global (Phase 4/5) hierarchical training preflight
validator -- mirrors ``training.preflight``'s Local counterpart. Runs
without rclpy/Gazebo where possible; only the ``live=True`` ROS-dependency
check needs rclpy importable (not a live Gazebo instance)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

from hunter_kinodynamic_rl.config.loader import load_profile
from hunter_kinodynamic_rl.config.schema import ConfigError
from hunter_kinodynamic_rl.evaluation.fingerprint import architecture_fingerprint, hierarchical_architecture_fingerprint
from hunter_kinodynamic_rl.evaluation.local_promotion import validate_promoted_local_checkpoint
from hunter_kinodynamic_rl.navigation.global_rl.replay_schema import SCHEMA_VERSION as GLOBAL_REPLAY_SCHEMA_VERSION


@dataclass
class GlobalPreflightReport:
    profile_name: str
    ok: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    hierarchical_architecture_fingerprint: Optional[str] = None
    global_replay_schema_version: Optional[int] = None
    local_checkpoint_dir: Optional[str] = None
    local_checkpoint_name: Optional[str] = None
    local_checkpoint_promoted: bool = False
    global_training_device: Optional[str] = None
    local_inference_device: Optional[str] = None
    cuda_available: bool = False
    ros_available: bool = False
    ros_gz_interfaces_available: bool = False
    live: bool = False
    resume: bool = False
    local_checkpoint_promotion_reasons: List[str] = field(default_factory=list)

    def _fail(self, m: str) -> None:
        self.ok = False
        self.errors.append(m)

    def _warn(self, m: str) -> None:
        self.warnings.append(m)

    def raise_if_failed(self) -> None:
        if not self.ok:
            raise GlobalPreflightError(
                f"Global preflight FAILED for profile {self.profile_name!r}:\n" +
                "\n".join(f"  - {e}" for e in self.errors)
            )


class GlobalPreflightError(RuntimeError):
    pass


def run_global_preflight(
    profile_name: str, *, live: bool = False, resume: bool = False, resume_run_dir: Optional[str] = None,
    require_promoted_local: bool = True, require_ros: Optional[bool] = None,
) -> GlobalPreflightReport:
    """``require_ros`` defaults to ``live`` (a ROS-free run has no ROS
    requirement at all; a live run needs rclpy importable at minimum)."""
    if require_ros is None:
        require_ros = live
    report = GlobalPreflightReport(profile_name=profile_name, live=live, resume=resume)

    try:
        profile = load_profile(profile_name)
    except ConfigError as e:
        report._fail(f"profile {profile_name!r} failed to load/validate: {e}")
        return report

    if not profile.hierarchical_training.enabled:
        report._fail(f"profile {profile_name!r} does not enable hierarchical_training")
    if not profile.global_rl.enabled:
        report._fail(f"profile {profile_name!r} does not enable global_rl")

    report.hierarchical_architecture_fingerprint = hierarchical_architecture_fingerprint(profile)
    report.global_replay_schema_version = GLOBAL_REPLAY_SCHEMA_VERSION
    report.local_checkpoint_dir = profile.hierarchical_training.local_checkpoint_dir
    report.local_checkpoint_name = profile.hierarchical_training.local_checkpoint_name

    # Defect-fix item 11: E/F/G-tier profiles (feasibility/global-risk
    # feedback) require a REAL LocalFeasibilityEvaluator, which only the
    # live=True LiveGazeboLocalExecutor path can construct -- the ROS-free
    # SimplifiedKinematicLocalExecutor stand-in has none. Reject BEFORE
    # touching Gazebo/training, never discover this mid-run.
    feasibility_tier = bool(
        profile.feasibility.enabled
        and (profile.global_rl.feasibility_feedback_enabled or profile.global_rl.global_risk_feedback_enabled)
    )
    if feasibility_tier and not live:
        report._fail(
            f"profile {profile_name!r} enables feasibility/global-risk feedback (E/F/G tier) but live=False -- "
            "the ROS-free stand-in executor has no real LocalFeasibilityEvaluator; this profile can only be "
            "trained/evaluated with live=True"
        )

    if live and not report.local_checkpoint_dir:
        report._fail("live=True requires hierarchical_training.local_checkpoint_dir to be set")
    elif live and report.local_checkpoint_dir:
        # Defect-fix item 11: the STRUCTURED validator (never a bare
        # promotion-marker-exists boolean) -- surfaces WHY a checkpoint
        # isn't considered promoted, and additionally cross-checks its
        # recorded architecture_fingerprint against THIS profile's own
        # (never load a checkpoint promoted for a DIFFERENT architecture).
        validation = validate_promoted_local_checkpoint(report.local_checkpoint_dir, report.local_checkpoint_name)
        report.local_checkpoint_promoted = validation.promoted
        report.local_checkpoint_promotion_reasons = list(validation.reasons)
        if validation.promoted:
            recorded_arch = (validation.promotion_manifest or {}).get("architecture_fingerprint")
            expected_arch = architecture_fingerprint(profile)
            if recorded_arch != expected_arch:
                report._fail(
                    f"promoted Local checkpoint architecture_fingerprint={recorded_arch!r} != this profile's "
                    f"own architecture_fingerprint={expected_arch!r} -- the promoted checkpoint was benchmarked "
                    "against a DIFFERENT profile/architecture than the one this run requested"
                )
        if not resume and require_promoted_local and not report.local_checkpoint_promoted:
            report._fail(
                f"fresh (non-resume) live run requires a PROMOTED Local checkpoint under "
                f"{report.local_checkpoint_dir!r}/{report.local_checkpoint_name!r}: "
                f"{'; '.join(report.local_checkpoint_promotion_reasons) or 'no promotion_manifest.json found'} "
                "(requirement C: only newly-promoted Local checkpoints may be used as Global training input)"
            )
        elif not report.local_checkpoint_promoted:
            report._warn(
                f"Local checkpoint at {report.local_checkpoint_dir!r}/{report.local_checkpoint_name!r} is NOT "
                f"promoted ({'; '.join(report.local_checkpoint_promotion_reasons) or 'no promotion_manifest.json'}) "
                "-- proceeding only because require_promoted_local=False or this is a resume"
            )

    # Defect-fix item 11: resume run/checkpoint strict check -- a resumed
    # Global run must not silently continue under a DIFFERENT hierarchical
    # architecture than whatever checkpoint already exists at
    # resume_run_dir.
    if resume and resume_run_dir:
        _check_global_resume_checkpoint_identity(report, resume_run_dir)

    report.global_training_device = profile.hierarchical_training.global_training_device
    report.local_inference_device = profile.hierarchical_training.local_inference_device
    try:
        import torch
        report.cuda_available = bool(torch.cuda.is_available())
    except ImportError:
        report._warn("torch is not importable -- cannot verify CUDA availability")
    for device_field, value in (
        ("global_training_device", report.global_training_device),
        ("local_inference_device", report.local_inference_device),
    ):
        if value == "cuda" and not report.cuda_available:
            report._fail(f"hierarchical_training.{device_field}='cuda' but CUDA is unavailable")

    try:
        import rclpy  # noqa: F401
        report.ros_available = True
    except ImportError:
        report.ros_available = False
        if require_ros:
            report._fail(
                "rclpy is not importable -- a live=True Global training run requires a sourced ROS2 Humble "
                "environment with Gazebo running; pass require_ros=False (or live=False) to skip this check"
            )
    try:
        import ros_gz_interfaces  # noqa: F401
        report.ros_gz_interfaces_available = True
    except ImportError:
        report.ros_gz_interfaces_available = False
        if require_ros:
            report._fail(
                "ros_gz_interfaces is not importable -- a live=True Global training run's LiveGazeboLocalExecutor "
                "requires it; pass require_ros=False (or live=False) to skip this check"
            )

    if resume and not live:
        report._warn(
            "resume=True with live=False -- resuming a ROS-free run only replays the "
            "SimplifiedKinematicLocalExecutor stand-in, never a real Local policy"
        )

    return report


def _check_global_resume_checkpoint_identity(report: GlobalPreflightReport, resume_run_dir: str) -> None:
    """Best-effort, torch-free identity check against whatever Global
    checkpoint generation already exists at ``resume_run_dir`` -- reads
    ``manifest.json`` directly (never loads the actual state_dict here;
    that strict load-time check already exists in
    ``nodes/hierarchical_train_node.py::_check_resume_compatibility``, this
    is the CHEAP pre-Gazebo version of the same claim)."""
    checkpoints_dir = os.path.join(resume_run_dir, "checkpoints")
    if not os.path.isdir(checkpoints_dir):
        report._warn(f"resume_run_dir has no checkpoints/ directory yet at {checkpoints_dir!r} -- nothing to verify")
        return
    candidate_tags: List[str] = ["latest", "final", "best"]
    try:
        for name in sorted(os.listdir(checkpoints_dir)):
            if name not in candidate_tags:
                candidate_tags.append(name)
    except OSError:
        pass
    for tag in candidate_tags:
        tag_path = os.path.realpath(os.path.join(checkpoints_dir, tag))
        manifest_path = os.path.join(tag_path, "manifest.json")
        if not os.path.isfile(manifest_path):
            continue
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            report._warn(f"resume checkpoint manifest at {manifest_path!r} unreadable: {e}")
            return
        recorded_fp = manifest.get("hierarchical_architecture_fingerprint")
        if recorded_fp is not None and recorded_fp != report.hierarchical_architecture_fingerprint:
            report._fail(
                f"resume checkpoint (tag={tag!r} at {tag_path!r}) hierarchical_architecture_fingerprint="
                f"{recorded_fp!r} != current profile's {report.hierarchical_architecture_fingerprint!r} -- "
                "resuming would load an architecture-incompatible Global checkpoint"
            )
        recorded_schema = manifest.get("global_replay_schema_version")
        if recorded_schema is not None and recorded_schema != report.global_replay_schema_version:
            report._fail(
                f"resume checkpoint (tag={tag!r}) global_replay_schema_version={recorded_schema!r} != current "
                f"{report.global_replay_schema_version!r}"
            )
        return
    report._warn(
        f"resume_run_dir has a checkpoints/ directory but no readable generation manifest.json found under "
        f"{checkpoints_dir!r} yet"
    )


def _format_report(report: GlobalPreflightReport) -> str:
    lines = [
        f"Global preflight: profile={report.profile_name!r} live={report.live} resume={report.resume} -> "
        f"{'OK' if report.ok else 'FAILED'}",
        f"  hierarchical_architecture_fingerprint={report.hierarchical_architecture_fingerprint}",
        f"  global_replay_schema_version={report.global_replay_schema_version}",
        f"  local_checkpoint={report.local_checkpoint_dir!r}/{report.local_checkpoint_name!r} "
        f"promoted={report.local_checkpoint_promoted}",
        f"  global_training_device={report.global_training_device} "
        f"local_inference_device={report.local_inference_device} cuda_available={report.cuda_available}",
        f"  ros_available={report.ros_available} ros_gz_interfaces_available={report.ros_gz_interfaces_available}",
    ]
    for w in report.warnings:
        lines.append(f"  WARNING: {w}")
    for e in report.errors:
        lines.append(f"  ERROR: {e}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="hierarchical_phase4")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-run-dir", default=None)
    parser.add_argument("--no-require-promoted-local", action="store_true")
    parser.add_argument("--no-require-ros", action="store_true")
    args, _ = parser.parse_known_args()

    result = run_global_preflight(
        args.profile, live=args.live, resume=args.resume, resume_run_dir=args.resume_run_dir,
        require_promoted_local=not args.no_require_promoted_local,
        require_ros=(False if args.no_require_ros else None),
    )
    print(_format_report(result))
    raise SystemExit(0 if result.ok else 1)
