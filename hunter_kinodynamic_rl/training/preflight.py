#!/usr/bin/env python3
"""Requirement A: Local TQC training preflight validator.

Runs entirely WITHOUT a live Gazebo/ROS environment (unlike
``TrainerBase.__init__``, which needs a live ``/get_dimensions`` service
call) -- every check here is either pure config/schema validation or a
pure-Python empirical sample of ``env.scenarios.procedural_generator``, so
a caller can run this BEFORE launching Gazebo and catch a bad profile,
distribution-drift checkpoint conflict, or accidental-resume mistake in
seconds instead of after a multi-hour training run has already started.

Checks performed (plan requirement A):

1. Profile exists and its schema validates (``config.loader.load_profile``).
2. ``scenario.goal_sampling_mode == "robot_relative_band"`` and
   ``scenario.goal_direction_sectors_deg`` covers the full circle
   ``[-180, 180]`` with no gap.
3. ``scenario.goal_infeasible_fraction > 0`` and a feasibility-check mode is
   configured; the ACTUAL (empirically sampled, not just the configured
   target) infeasible fraction is measured over N synthetic scenario draws
   and must be within tolerance of the configured target.
4. Output run-root path exists or is creatable and writable.
5. ``resume``/``resume_run_dir`` are mutually consistent (mirrors
   ``TrainerBase.__init__``'s own ``resume and resume_run_dir`` gate) --
   catches a caller who passed one without the other before any training
   starts.
6. CPU/CUDA device: reports ``torch.cuda.is_available()``; fails only if a
   specific device was explicitly requested and is unavailable.
7. Required ROS/Gazebo Python dependencies (``rclpy``, ``ros_gz_interfaces``)
   import cleanly -- reported always, fatal only when ``require_ros=True``
   (the default for an actual pre-launch gate; a bare-host CI/config-only
   check passes ``require_ros=False``).
8. Reports obs/action dims, ``architecture_fingerprint``,
   ``local_training_contract_fingerprint`` -- the identity a Local
   checkpoint manifest must carry for downstream promotion/Global-training
   consumption (requirements B/C).
"""

from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import dataclass, field
from typing import List, Optional

from hunter_kinodynamic_rl.config.loader import DEFAULT_RL_PROFILE, load_profile
from hunter_kinodynamic_rl.config.schema import ConfigError, Profile
from hunter_kinodynamic_rl.evaluation.fingerprint import architecture_fingerprint, local_training_contract_fingerprint
from hunter_kinodynamic_rl.trajectory.action_space import action_dim_for_mode

DEFAULT_INFEASIBLE_SAMPLE_COUNT = 300
#: Defect-fix item 11: replaces the old FIXED absolute tolerance (0.15),
#: which had a real boundary bug -- at the common default
#: target=0.15/tolerance=0.15, a measured=0.0 (a COMPLETE generator
#: failure -- zero infeasible scenarios drawn when 15% were expected)
#: computed abs(0.0-0.15)=0.15, and the old check was `> tolerance`
#: (strict), so 0.15 > 0.15 is False -- the check SILENTLY PASSED a
#: completely broken generator. A binomial confidence interval scales
#: correctly with both the sample count and how far the target is from
#: 0.5 (variance is p*(1-p)), so a genuinely miscalibrated/broken
#: generator is caught regardless of which target happens to be
#: configured, while true sampling noise at a healthy sample_count is not
#: flagged. z=4.0 is a deliberately generous ~99.994% two-sided interval
#: -- this check exists to catch a GROSSLY miscalibrated/broken generator,
#: not to replace a real statistical test.
DEFAULT_INFEASIBLE_FRACTION_CONFIDENCE_Z = 4.0


def _binomial_confidence_half_width(p: float, n: int, z: float) -> float:
    """Normal-approximation half-width of a two-sided ``z``-sigma binomial
    confidence interval around a true rate ``p`` measured over ``n``
    Bernoulli draws. ``n <= 0`` returns ``float('inf')`` (no possible
    measurement can be trusted with zero samples) rather than dividing by
    zero or fabricating a finite bound."""
    if n <= 0:
        return float("inf")
    return z * math.sqrt(max(0.0, p * (1.0 - p)) / n)


@dataclass
class LocalPreflightReport:
    profile_name: str
    ok: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # -- distribution contract --------------------------------------------------
    full_circle_goal_direction: bool = False
    goal_direction_sectors_deg: Optional[list] = None
    goal_distance_range_m: Optional[list] = None
    goal_infeasible_fraction_configured: float = 0.0
    goal_infeasible_fraction_measured: Optional[float] = None
    feasibility_check: Optional[str] = None

    # -- identity ----------------------------------------------------------------
    state_dim: Optional[int] = None
    action_dim: Optional[int] = None
    resolved_config_hash: Optional[str] = None
    local_training_contract_fingerprint: Optional[str] = None

    # -- run/output ----------------------------------------------------------------
    run_root: Optional[str] = None
    resume: bool = False
    resume_run_dir: Optional[str] = None
    would_reuse_existing_run_dir: bool = False

    # -- device/deps ----------------------------------------------------------------
    cuda_available: bool = False
    requested_device: Optional[str] = None
    ros_available: bool = False
    ros_gz_interfaces_available: bool = False

    def _fail(self, message: str) -> None:
        self.ok = False
        self.errors.append(message)

    def _warn(self, message: str) -> None:
        self.warnings.append(message)

    def raise_if_failed(self) -> None:
        if not self.ok:
            raise PreflightError(
                f"Local preflight FAILED for profile {self.profile_name!r}:\n" + "\n".join(f"  - {e}" for e in self.errors)
            )


class PreflightError(RuntimeError):
    """Raised by :meth:`LocalPreflightReport.raise_if_failed` -- never
    silently swallowed by a caller that wants a hard gate before training
    starts."""


def _sectors_cover_full_circle(sectors, tol_deg: float = 1e-6) -> bool:
    """``sectors`` is a list of ``[lo_deg, hi_deg]`` intervals (each
    ``lo <= hi``, matching ``ScenarioConfig.goal_direction_sectors_deg``'s
    own contract -- no wraparound-through-180 interval is representable).
    Returns True iff the union of intervals, clipped to ``[-180, 180]``,
    covers that whole range with no gap wider than ``tol_deg``."""
    if not sectors:
        return False
    intervals = sorted((float(lo), float(hi)) for lo, hi in sectors)
    if intervals[0][0] > -180.0 + tol_deg:
        return False
    covered_to = -180.0
    for lo, hi in intervals:
        if lo > covered_to + tol_deg:
            return False
        covered_to = max(covered_to, hi)
    return covered_to >= 180.0 - tol_deg


def _measure_infeasible_fraction(profile: Profile, sample_count: int) -> float:
    from hunter_kinodynamic_rl.env.scenarios.procedural_generator import generate_scenario

    min_turning_radius_m = 1.0 / max(1e-9, profile.robot.max_curvature)
    infeasible_count = 0
    lo, hi = profile.scenario.train_seed_range
    for i in range(sample_count):
        seed = lo + (i % max(1, hi - lo + 1))
        scenario = generate_scenario(
            seed, profile.scenario, robot_radius=profile.robot.collision_radius_m,
            min_turning_radius_m=min_turning_radius_m, wheelbase_m=profile.robot.wheelbase_m,
            start_pose_cfg=profile.start_pose,
        )
        if scenario.realized_infeasible:
            infeasible_count += 1
    return infeasible_count / sample_count if sample_count > 0 else 0.0


def run_local_preflight(
    profile_name: str, *, run_root: str = "runtime/experiments", resume: bool = False,
    resume_run_dir: Optional[str] = None, requested_device: Optional[str] = None, require_ros: bool = True,
    sample_count: int = DEFAULT_INFEASIBLE_SAMPLE_COUNT,
    infeasible_fraction_confidence_z: float = DEFAULT_INFEASIBLE_FRACTION_CONFIDENCE_Z,
) -> LocalPreflightReport:
    report = LocalPreflightReport(profile_name=profile_name, run_root=run_root, resume=resume,
                                   resume_run_dir=resume_run_dir, requested_device=requested_device)
    if sample_count <= 0:
        report._fail(f"sample_count={sample_count} must be > 0 -- cannot empirically measure anything from 0 draws")
    if infeasible_fraction_confidence_z <= 0.0:
        report._fail(
            f"infeasible_fraction_confidence_z={infeasible_fraction_confidence_z} must be > 0"
        )

    # 1. profile existence + schema validity.
    try:
        profile = load_profile(profile_name)
    except ConfigError as e:
        report._fail(f"profile {profile_name!r} failed to load/validate: {e}")
        return report

    if profile.runtime.deployment == "real_hardware":
        report._fail(
            f"profile {profile_name!r} is marked runtime.deployment=real_hardware (inference-only) -- "
            "training must run against a simulation profile"
        )

    if profile.environment_v2.enabled:
        from hunter_kinodynamic_rl.config.loader import default_config_root
        from hunter_kinodynamic_rl.env.randomization.calibration_manifest import validate_calibration_manifest
        calibration = validate_calibration_manifest(profile, default_config_root())
        for warning in calibration.warnings:
            report._warn(f"environment_v2 calibration: {warning}")
        for error in calibration.errors:
            report._fail(f"environment_v2 calibration: {error}")

    # 2. full-circle goal direction.
    report.goal_direction_sectors_deg = list(profile.scenario.goal_direction_sectors_deg)
    report.goal_distance_range_m = list(profile.scenario.goal_distance_range_m)
    report.feasibility_check = profile.scenario.feasibility_check
    report.goal_infeasible_fraction_configured = profile.scenario.goal_infeasible_fraction
    if profile.scenario.goal_sampling_mode != "robot_relative_band":
        report._fail(
            f"scenario.goal_sampling_mode={profile.scenario.goal_sampling_mode!r} -- requirement A's Local "
            "training contract requires 'robot_relative_band' (arbitrary short-range subgoal distribution), "
            "not the legacy 'uniform_world' final-goal distribution"
        )
    else:
        report.full_circle_goal_direction = _sectors_cover_full_circle(profile.scenario.goal_direction_sectors_deg)
        if not report.full_circle_goal_direction:
            report._fail(
                f"scenario.goal_direction_sectors_deg={profile.scenario.goal_direction_sectors_deg} does not "
                "cover the full circle [-180, 180] -- requirement A requires full-circle subgoal-direction "
                "training coverage (Global can select any of 8 directions incl. rear-diagonal)"
            )

    # 3. infeasible fraction + feasibility generator config.
    if profile.scenario.goal_infeasible_fraction <= 0.0:
        report._fail(
            "scenario.goal_infeasible_fraction=0.0 -- requirement A requires the Local training distribution "
            "to include a nonzero fraction of verified-infeasible/blocked subgoals"
        )
    elif profile.scenario.goal_sampling_mode == "robot_relative_band" and sample_count > 0:
        try:
            measured = _measure_infeasible_fraction(profile, sample_count)
            report.goal_infeasible_fraction_measured = measured
            target = profile.scenario.goal_infeasible_fraction
            half_width = _binomial_confidence_half_width(target, sample_count, infeasible_fraction_confidence_z)
            if abs(measured - target) > half_width:
                report._fail(
                    f"empirically measured infeasible fraction {measured:.3f} over {sample_count} synthetic "
                    f"draws deviates from configured goal_infeasible_fraction={target:.3f} by more than the "
                    f"{infeasible_fraction_confidence_z:g}-sigma binomial confidence half-width "
                    f"({half_width:.3f}) -- generator/config may be miscalibrated"
                )
        except Exception as e:  # noqa: BLE001 -- report as a preflight failure, never crash the validator itself
            report._fail(f"failed to empirically sample infeasible fraction: {e}")

    # 4. output path.
    try:
        os.makedirs(run_root, exist_ok=True)
        if not os.access(run_root, os.W_OK):
            report._fail(f"run_root {run_root!r} exists but is not writable")
    except OSError as e:
        report._fail(f"run_root {run_root!r} could not be created: {e}")

    # 5. resume/fresh-start consistency (mirrors TrainerBase.__init__'s
    # `resume and resume_run_dir` gate -- a fresh run NEVER reuses an
    # existing directory, so the only way to "accidentally resume" is to
    # pass resume=True with a stale/wrong resume_run_dir, or to pass
    # resume_run_dir without resume=True and expect it to be honored).
    if resume and not resume_run_dir:
        report._fail("resume=True but resume_run_dir is empty -- nothing to resume from")
    if resume_run_dir and not resume:
        report._warn(
            f"resume_run_dir={resume_run_dir!r} was given but resume=False -- it will be IGNORED and a fresh "
            "run directory will be created instead (this is the intended fresh-start behavior, not a bug, but "
            "confirm this is what you meant)"
        )
    if resume and resume_run_dir:
        report.would_reuse_existing_run_dir = os.path.isdir(resume_run_dir)
        if not report.would_reuse_existing_run_dir:
            report._fail(f"resume=True but resume_run_dir={resume_run_dir!r} does not exist")
    if not resume and resume_run_dir:
        report.would_reuse_existing_run_dir = False

    # 6. device.
    try:
        import torch
        report.cuda_available = bool(torch.cuda.is_available())
    except ImportError:
        report._warn("torch is not importable in this environment -- cannot verify CUDA availability")
    if requested_device == "cuda" and not report.cuda_available:
        report._fail("requested_device='cuda' but CUDA is unavailable in this environment")

    # 7. ROS/Gazebo dependencies -- defect-fix item 11: the module
    # docstring has always claimed BOTH rclpy AND ros_gz_interfaces are
    # checked, but only rclpy actually was; ros_gz_interfaces (the Gazebo
    # entity-pose/spawn service messages this package's live executors
    # depend on) is now genuinely imported too.
    try:
        import rclpy  # noqa: F401
        report.ros_available = True
    except ImportError:
        report.ros_available = False
        if require_ros:
            report._fail(
                "rclpy is not importable -- Local training requires a sourced ROS2 Humble environment "
                "(source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash) with Gazebo running; "
                "pass require_ros=False to skip this check for a config-only validation on a bare host"
            )
    try:
        import ros_gz_interfaces  # noqa: F401
        report.ros_gz_interfaces_available = True
    except ImportError:
        report.ros_gz_interfaces_available = False
        if require_ros:
            report._fail(
                "ros_gz_interfaces is not importable -- Local training's live Gazebo executors require it "
                "(source the sourced ROS2 Humble + workspace install as above); pass require_ros=False to skip "
                "this check for a config-only validation on a bare host"
            )

    # 8. identity.
    history_len = profile.observation.frame_stack if profile.features.temporal_context else 1
    report.state_dim = profile.observation.lidar_bins * history_len + profile.observation.robot_state_dim
    report.action_dim = action_dim_for_mode(profile.action_space)
    report.resolved_config_hash = architecture_fingerprint(profile)
    try:
        report.local_training_contract_fingerprint = local_training_contract_fingerprint(profile)
    except KeyError as e:
        report._fail(f"failed to compute local_training_contract_fingerprint: {e}")

    # 9. resume/fresh checkpoint distribution identity (defect-fix item 11):
    # resuming into an EXISTING run directory must not silently continue
    # training under a DIFFERENT scenario/training-contract distribution
    # than whatever the existing checkpoint(s) there were actually trained
    # under -- checked against the SAME manifest.json a real checkpoint
    # load will use, without needing torch/a full state_dict load.
    if resume and resume_run_dir and report.would_reuse_existing_run_dir:
        _check_resume_checkpoint_identity(report, resume_run_dir)

    return report


def _check_resume_checkpoint_identity(report: LocalPreflightReport, resume_run_dir: str) -> None:
    checkpoints_dir = os.path.join(resume_run_dir, "checkpoints")
    if not os.path.isdir(checkpoints_dir):
        report._warn(
            f"resume_run_dir has no checkpoints/ directory yet at {checkpoints_dir!r} -- nothing to verify "
            "(the first save will create it)"
        )
        return
    candidate_tags: List[str] = []
    for name in ("latest", "final", "best"):
        if name not in candidate_tags:
            candidate_tags.append(name)
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
        recorded_contract = manifest.get("local_training_contract_fingerprint")
        if recorded_contract is not None and recorded_contract != report.local_training_contract_fingerprint:
            report._fail(
                f"resume checkpoint (tag={tag!r} at {tag_path!r}) local_training_contract_fingerprint="
                f"{recorded_contract!r} != current profile's {report.local_training_contract_fingerprint!r} -- "
                "resuming would silently continue training under a DIFFERENT subgoal distribution than what "
                "this checkpoint was actually trained on"
            )
        recorded_arch = manifest.get("resolved_config_hash") or manifest.get("architecture_fingerprint")
        if recorded_arch is not None and recorded_arch != report.resolved_config_hash:
            report._fail(
                f"resume checkpoint (tag={tag!r} at {tag_path!r}) architecture fingerprint={recorded_arch!r} "
                f"!= current profile's {report.resolved_config_hash!r} -- resuming would load an "
                "architecture-incompatible checkpoint"
            )
        return
    report._warn(
        f"resume_run_dir has a checkpoints/ directory but no readable generation manifest.json found under "
        f"{checkpoints_dir!r} yet"
    )


def _format_report(report: LocalPreflightReport) -> str:
    lines = [
        f"Local preflight: profile={report.profile_name!r} -> {'OK' if report.ok else 'FAILED'}",
        f"  full_circle_goal_direction={report.full_circle_goal_direction} sectors={report.goal_direction_sectors_deg}",
        f"  goal_distance_range_m={report.goal_distance_range_m} feasibility_check={report.feasibility_check!r}",
        f"  infeasible_fraction: configured={report.goal_infeasible_fraction_configured:.3f} "
        f"measured={report.goal_infeasible_fraction_measured}",
        f"  state_dim={report.state_dim} action_dim={report.action_dim}",
        f"  architecture_fingerprint={report.resolved_config_hash}",
        f"  local_training_contract_fingerprint={report.local_training_contract_fingerprint}",
        f"  run_root={report.run_root!r} resume={report.resume} resume_run_dir={report.resume_run_dir!r} "
        f"would_reuse_existing_run_dir={report.would_reuse_existing_run_dir}",
        f"  cuda_available={report.cuda_available} ros_available={report.ros_available} "
        f"ros_gz_interfaces_available={report.ros_gz_interfaces_available}",
    ]
    for w in report.warnings:
        lines.append(f"  WARNING: {w}")
    for e in report.errors:
        lines.append(f"  ERROR: {e}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=DEFAULT_RL_PROFILE)
    parser.add_argument("--run-root", default="runtime/experiments")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-run-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-require-ros", action="store_true")
    parser.add_argument("--sample-count", type=int, default=DEFAULT_INFEASIBLE_SAMPLE_COUNT)
    args, _ = parser.parse_known_args()

    result = run_local_preflight(
        args.profile, run_root=args.run_root, resume=args.resume, resume_run_dir=args.resume_run_dir,
        requested_device=args.device, require_ros=not args.no_require_ros, sample_count=args.sample_count,
    )
    print(_format_report(result))
    raise SystemExit(0 if result.ok else 1)
