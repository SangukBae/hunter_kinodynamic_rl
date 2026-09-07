#!/usr/bin/env python3
"""``ros2 run hunter_kinodynamic_rl evaluation_node.py --ros-args -p profile:=evaluation_id -p checkpoint_dir:=<run_dir>/checkpoints -p checkpoint_name:=final``

Loads a trained agent checkpoint and runs it against the profile's fixed
benchmark (``profile.evaluation.benchmark``) via evaluation/benchmark_runner.py.

state_dim/action_dim/max_action are read from the LIVE environment node's
``/get_dimensions`` (never hardcoded -- section 7/11).

section P1-9 ("evaluation must restore the training architecture/profile and
only override scenario/reward/metric config, never observation dimension or
actor architecture"): the AGENT is built from the checkpoint's OWN
``resolved_config`` (the full training-time Profile, saved by every
checkpoint since the P0-1 schema-v2 fix) -- NEVER from the requested
``--profile``'s own action_space/features/risk/counterfactual/hyperparameters
sections. Those sections in ``evaluation_id.yaml`` and friends are
UNUSED for agent construction; the requested profile supplies
``evaluation``/``reward``/``scenario``/``runtime`` (benchmark selection +
metric config + world size/termination + physics/timing). This closes a
real gap the previous "print a warning and continue anyway" check left
open: evaluating an A/B/C/D/E-tier checkpoint through an eval profile that
unconditionally declares F-tier features would have silently constructed a
RiskAgent with an untrained (randomly-initialized) risk critic bolted onto
a vanilla-trained actor -- a meaningless result, not merely a name mismatch
worth a warning.

section item-1 (round 2, "evaluation contract fairness"): restoring the
checkpoint's ARCHITECTURE and merging the requested profile's
evaluation-contract sections into an in-process ``Profile`` object (as
``build_effective_profile`` below does) is only HALF the fairness story --
the LIVE ``environment_node.py`` this node talks to over ROS is a SEPARATE,
already-launched process whose own ``self.profile`` was resolved once at
ITS launch, and does NOT automatically pick up anything this node computes
locally. Before running the benchmark, ``main()`` therefore also WRITES the
effective profile's ``reward``/``scenario``/``runtime``/``evaluation``/
``sensor_noise`` sections to a file (``evaluation/contract_override.py``) and pushes it to
the live environment via ``EnvironmentClient.set_evaluation_contract_override``
-- exactly like the pre-existing ``scenario_override_path`` mechanism
already does for exact scenario placement -- then reads back the live
environment's OWN ``evaluation_contract_fingerprint_sha256`` parameter to
verify it was actually applied (see ``validate_live_evaluation_contract``
below), not merely that the parameter-set RPC returned success. Without
this, the live environment kept using ITS OWN launch-time (== checkpoint
training) profile's world boundary/reward/timeout/physics-timing/
common-metrics settings for every episode, silently unfair between two
checkpoints trained under different profiles -- regardless of the
requested ``--profile``'s own ``evaluation``/``reward``/``scenario``/
``runtime`` sections, which previously affected only this node's own
LOCAL bookkeeping (episode-length client-side loop bound, output
directory, fingerprints recorded in ``summary.json``), never the actual
live episode dynamics.
"""

import argparse
import contextlib
import dataclasses
import json
import os
import tempfile
import uuid
from typing import Optional

import rclpy

from hunter_kinodynamic_rl.config.loader import load_profile, profile_from_dict
from hunter_kinodynamic_rl.evaluation.benchmark_runner import run_benchmark
from hunter_kinodynamic_rl.evaluation.contract_override import write_evaluation_contract_override
from hunter_kinodynamic_rl.evaluation.fingerprint import (
    architecture_fingerprint_from_resolved_config, evaluation_contract_fingerprint,
)
from hunter_kinodynamic_rl.evaluation.result_writer import write_summary_json
from hunter_kinodynamic_rl.rl.algorithms.kinodynamic_tqc.agent import Agent as RiskAgent
from hunter_kinodynamic_rl.rl.algorithms.sac.agent import Agent as SACAgent
from hunter_kinodynamic_rl.rl.algorithms.tqc.agent import Agent as VanillaAgent
from hunter_kinodynamic_rl.rl.checkpointing import manager
from hunter_kinodynamic_rl.trajectory.action_space import action_dim_for_mode
from hunter_kinodynamic_rl.training.trainer_base import EnvServiceError, EnvironmentClient


@dataclasses.dataclass(frozen=True)
class RestoreResult:
    """section item-4 (override-restore-failure fix): a STRUCTURED verdict
    for whether ``_restore_launch_time_evaluation_contract`` actually
    recovered the live environment_node's launch-time evaluation contract --
    previously that function only ever printed a WARNING and returned
    ``None``, so a benchmark run whose evaluation BODY succeeded but whose
    POST-RUN restoration silently failed (a rejected parameter-clear, a
    ``/reset`` that raised, or a fingerprint mismatch after restoring) still
    exited 0 -- indistinguishable, to any caller/CI/orchestration script
    checking only the exit code, from a genuinely clean run. ``main()`` now
    reads this back and raises (after its own ``try/finally`` has already
    let any ORIGINAL exception from the evaluation body propagate
    unmasked) so a caller checking the exit code alone still sees failure."""
    ok: bool
    reason: Optional[str] = None
    restored_fingerprint: Optional[str] = None


def build_effective_profile(manifest: dict, eval_profile_name: str):
    """Reconstruct the checkpoint's OWN training-time architecture from its
    manifest's resolved_config, then layer the requested evaluation
    profile's evaluation/reward/scenario/runtime/sensor_noise/environment_v2 sections on
    top -- never its action_space/features/risk/counterfactual/
    hyperparameters, which would silently mismatch the checkpoint's actual
    weights.

    This IN-PROCESS ``Profile`` merge is only half of the fairness story
    (section item-1, round 2) -- it drives THIS node's own local
    bookkeeping (client-side episode-length budget, output paths,
    fingerprints) directly, but a LIVE ``environment_node.py`` process must
    be told separately (``main()`` below, via
    ``evaluation/contract_override.py`` + ``EnvironmentClient.set_evaluation_contract_override``)
    since it cannot see this in-process object at all."""
    resolved_config = manifest.get("resolved_config")
    if resolved_config is None:
        raise SystemExit(
            "checkpoint manifest has no 'resolved_config' (pre-P0-1 schema_version=1 checkpoint) -- "
            "cannot safely reconstruct its training architecture for evaluation. Retrain, or if this "
            "checkpoint's exact training profile is known, evaluate by hand-constructing the agent "
            "from that profile instead of this node."
        )
    checkpoint_profile_name = manifest.get("profile_name") or "unknown_training_profile"
    # resolved_config is dataclasses.asdict(profile) -- includes the `name`
    # field alongside the section dicts profile_from_dict() expects; strip
    # it (passed separately as profile_from_dict's own `name` arg below).
    sections_only = {k: v for k, v in resolved_config.items() if k != "name"}
    training_profile = profile_from_dict(checkpoint_profile_name, sections_only)
    eval_overrides = load_profile(eval_profile_name)
    if not eval_overrides.evaluation.benchmark:
        raise SystemExit(f"profile {eval_profile_name!r} has no evaluation.benchmark set")
    effective = dataclasses.replace(
        training_profile,
        name=eval_profile_name,
        evaluation=eval_overrides.evaluation,
        reward=eval_overrides.reward,
        scenario=eval_overrides.scenario,
        # section item-1 (round 2): runtime (time_delta_sec/
        # deterministic_stepping/gazebo_max_step_size_sec/...) is an
        # EVALUATION-CONTRACT section exactly like reward/scenario -- see
        # evaluation/fingerprint.py's EVALUATION_CONTRACT_SECTIONS -- so it
        # must be layered here too, never left at the checkpoint's own
        # training-time value.
        runtime=eval_overrides.runtime,
        # requirement 2 (sensor-noise evaluation fairness): sensor_noise is
        # an EVALUATION-CONTRACT section exactly like the others above --
        # every checkpoint benchmarked under this profile must see the
        # SAME sensor/localization noise model the requested profile asks
        # for (on, off, or a specific magnitude), never silently keep its
        # own training-time sensor_noise setting. Without this, `profile`
        # (passed to write_evaluation_contract_override below, and to
        # evaluation_contract_fingerprint above) would carry the
        # CHECKPOINT's training-time sensor_noise instead of the requested
        # evaluation profile's -- both the live environment override and
        # this run's own recorded fingerprint would then be wrong.
        sensor_noise=eval_overrides.sensor_noise,
        environment_v2=eval_overrides.environment_v2,
    )
    effective.validate()
    print(f"evaluation architecture restored from training profile {checkpoint_profile_name!r} "
          f"(features.risk_critic={effective.features.risk_critic}, "
          f"features.counterfactual_risk={effective.features.counterfactual_risk}, "
          f"action_space.mode={effective.action_space.mode!r}); "
          f"benchmark/reward/scenario/runtime/sensor_noise/environment_v2 taken from requested profile "
          f"{eval_profile_name!r}")
    return effective


def expected_dims(profile) -> tuple:
    """The (state_dim, action_dim) THIS profile's own observation/action
    contract implies -- computed independently of whatever a live
    environment_node happens to report, so it can be checked AGAINST that
    report (section P0-1) rather than blindly trusted."""
    history_len = profile.observation.frame_stack if profile.features.temporal_context else 1
    state_dim = profile.observation.lidar_bins * history_len + profile.observation.robot_state_dim
    return state_dim, action_dim_for_mode(profile.action_space)


def build_agent(profile, dims):
    """section P0-1: dispatches on ``algorithm.name`` (SAC vs TQC) THEN
    ``features.risk_critic`` (vanilla vs risk-aware TQC) -- mirrors
    ``nodes/train_node.py``'s own dispatch exactly (see that module's
    docstring), which this function previously did NOT: it only ever
    constructed a TQC-family agent (Vanilla or Risk), silently wrong for
    any checkpoint trained with ``algorithm.name: sac``."""
    if profile.algorithm.name == "sac":
        return SACAgent(dims.state_dim, dims.action_dim, dims.max_action, profile.sac_hyperparameters)
    if profile.features.risk_critic:
        return RiskAgent(dims.state_dim, dims.action_dim, dims.max_action,
                          profile.hyperparameters, profile.risk, profile.counterfactual)
    return VanillaAgent(dims.state_dim, dims.action_dim, dims.max_action, profile.hyperparameters)


def validate_live_environment(
    env: EnvironmentClient, profile, checkpoint_profile_name: str, dims,
    checkpoint_architecture_fingerprint: str,
) -> None:
    """section P0-1/item-2: refuses to proceed (never "random init or shape
    error") when the LIVE environment_node is not actually configured to
    match the checkpoint being evaluated. THREE independent checks, because
    none alone is sufficient:

    1. dimension check -- catches ANY observation/action contract drift
       (lidar_bins, frame_stack, robot_state_dim, action_dim), but CANNOT
       catch two profiles that happen to produce the same numbers while
       decoding ``action_space.mode`` differently (legacy_waypoint vs
       trajectory are BOTH 3-D actions).
    2. live ``profile`` PARAMETER check -- the environment_node process's
       own ``profile`` ROS parameter (set at ITS launch time, immutable
       afterward) is read back directly and compared against the
       checkpoint's OWN training profile name.
    3. **architecture FINGERPRINT check (section item-2, NEW)** -- a name
       match alone does not prove the two sides agree: the same-named
       profile's YAML could have been edited on disk between when the
       checkpoint trained and when the live environment_node launched (or
       between then and now). ``env.get_remote_parameter("architecture_fingerprint_sha256")``
       reads back the SHA-256 the live process computed, ONCE, from
       whatever it ACTUALLY resolved at ITS OWN launch (see
       ``environment_node.py``'s own ``architecture_fingerprint`` call) --
       compared against the SAME fingerprint computed from the
       checkpoint's OWN frozen ``resolved_config`` (never re-loaded from
       disk, so it reflects training time exactly). This is the check that
       actually closes the "profile name matches, but content silently
       drifted" gap; the profile-name check above is kept as a
       human-readable diagnostic on top of it, not a substitute for it.

    A MISMATCH on any of the three is a hard error (no override): the ONLY
    way to fix it is to relaunch environment_node.py with
    ``-p profile:=<checkpoint's training profile>`` (and, for a
    content-drift fingerprint mismatch, to first restore the profile YAML
    to what the checkpoint actually trained under). Unlike the
    EVALUATION-CONTRACT sections (``reward``/``scenario``/``runtime``/
    ``evaluation`` -- see ``validate_live_evaluation_contract`` below and
    ``evaluation/contract_override.py``, which ARE reconfigurable per
    episode without a relaunch), environment_node.py's ARCHITECTURE
    sections (``action_space``/``features``/``observation``/``robot``/
    ``dynamics``/``risk``/``counterfactual``/``hyperparameters``/
    ``algorithm``) are resolved ONCE at construction and genuinely cannot
    be reconfigured at runtime by design (see that module's docstring) --
    changing them requires a fresh Gazebo-side robot/network-shape
    consistent with whatever launches next, not something a running
    process can safely do to itself."""
    expected_state_dim, expected_action_dim = expected_dims(profile)
    if dims.state_dim != expected_state_dim or dims.action_dim != expected_action_dim:
        raise SystemExit(
            f"evaluation_node: live environment_node dimensions (state_dim={dims.state_dim}, "
            f"action_dim={dims.action_dim}) do not match what checkpoint profile "
            f"{checkpoint_profile_name!r} requires (state_dim={expected_state_dim}, "
            f"action_dim={expected_action_dim}) -- refusing to build an agent against mismatched "
            "dimensions. The live environment_node is very likely running under the WRONG profile; "
            f"relaunch it with -p profile:={checkpoint_profile_name}"
        )
    try:
        # round 5 (live-E2E finding): compare against the environment_node's
        # RESOLVED profile name, not its raw `profile` CLI/ROS-param string
        # -- the two differ whenever either side (this checkpoint's own
        # training launch, or the live node's current launch) used an
        # explicit path rather than a short registered name, since
        # `load_profile` strips a path down to its basename-without-extension
        # before it ever becomes `Profile.name` / a checkpoint's manifest
        # `profile_name`. Comparing raw strings would spuriously reject a
        # path-launched profile that is bit-for-bit identical to what the
        # checkpoint trained under. See environment_node.py's own
        # `resolved_profile_name` parameter docstring.
        live_profile_name = env.get_remote_parameter("resolved_profile_name")
    except EnvServiceError as e:
        raise SystemExit(
            f"evaluation_node: could not read the live environment_node's own "
            f"'resolved_profile_name' parameter to verify it matches checkpoint profile "
            f"{checkpoint_profile_name!r}: {e}"
        ) from e
    if live_profile_name != checkpoint_profile_name:
        raise SystemExit(
            f"evaluation_node: live environment_node is running profile={live_profile_name!r}, but "
            f"this checkpoint was trained under profile={checkpoint_profile_name!r}. Dimensions "
            "happened to match (or this check would have already failed above), but action_space.mode/"
            "reward/feature semantics may still differ -- refusing to evaluate under a mismatched "
            f"live environment. Relaunch environment_node.py with -p profile:={checkpoint_profile_name}"
        )
    try:
        live_architecture_fingerprint = env.get_remote_parameter("architecture_fingerprint_sha256")
    except EnvServiceError as e:
        raise SystemExit(
            f"evaluation_node: could not read the live environment_node's own "
            f"'architecture_fingerprint_sha256' parameter (relaunch environment_node.py after rebuilding "
            f"if it predates section item-2): {e}"
        ) from e
    if live_architecture_fingerprint != checkpoint_architecture_fingerprint:
        raise SystemExit(
            "evaluation_node: live environment_node's architecture_fingerprint_sha256="
            f"{live_architecture_fingerprint!r} does not match checkpoint's own "
            f"{checkpoint_architecture_fingerprint!r}, even though both report profile name "
            f"{checkpoint_profile_name!r}. This means profile {checkpoint_profile_name!r}'s YAML content "
            "changed on disk between when this checkpoint trained and when the live environment_node "
            "launched -- the two sides no longer agree on action_space/features/observation/robot/dynamics/"
            "risk/counterfactual/hyperparameters. Restore the profile YAML to what the checkpoint actually "
            "trained under, then relaunch environment_node.py."
        )


def augment_summary_with_physics_calibration(
    summary: dict, env: EnvironmentClient, profile, output_dir: str,
) -> dict:
    """section item-1 (Gazebo physics-step reality-check fix): when the
    effective profile runs with ``runtime.deterministic_stepping``, this
    reads back ``physics_step_calibration_verified``/
    ``physics_step_calibration_observed_dt_sec`` from the LIVE environment_node
    (set once, for real, by ``GazeboRuntimeMixin.verify_physics_step_calibration``
    -- see that method's docstring) and merges them into ``summary`` before
    re-writing ``summary.json`` -- so a reader of the result file can tell
    whether physics was GENUINELY verified for this run, never just assume
    it was because the profile declares ``deterministic_stepping: true``.

    Called AFTER ``run_benchmark`` (which already wrote a first
    ``summary.json`` without this information -- calibration only happens
    lazily, inside the first episode's first step) -- re-writes the SAME
    file with these two fields added. A no-op (returns ``summary``
    unchanged) when the profile doesn't use deterministic stepping at all,
    since these fields would be meaningless for the legacy wall-clock-sleep
    path. If the live parameters can't be read at all (a pre-item-1
    environment_node that never declared them), records
    ``physics_step_calibration_verified=False`` rather than silently
    omitting the field -- an old/incompatible environment_node must never
    look the same as one that verified successfully."""
    if not profile.runtime.deterministic_stepping:
        return summary
    try:
        verified = bool(env.get_remote_parameter("physics_step_calibration_verified"))
        observed_dt_sec = env.get_remote_parameter("physics_step_calibration_observed_dt_sec")
    except EnvServiceError:
        verified, observed_dt_sec = False, None
    summary = dict(summary)
    summary["physics_step_calibration_verified"] = verified
    summary["physics_step_calibration_observed_dt_sec"] = observed_dt_sec
    write_summary_json(os.path.join(output_dir, "summary.json"), summary)
    return summary


def environment_client_kwargs(profile) -> dict:
    """section item-1 (round 3): the risk-telemetry wait budgets
    (``runtime.risk_telemetry_wait_timeout_sec``/
    ``runtime.risk_telemetry_reset_marker_timeout_sec``) ARE part of the
    runtime evaluation contract (hashed into ``evaluation_contract_fingerprint``
    via the whole ``runtime`` section), but ``EnvironmentClient``'s own
    constructor silently falls back to hardcoded defaults (1.0s/5.0s)
    whenever a caller doesn't pass them explicitly -- this call site
    previously never did, a real false guarantee (the fingerprint claimed
    these fields were part of the contract; the ACTUAL client-side wait
    budget used during ``env.step()``/``env.reset()`` never reflected the
    requested profile at all). Factored out purely so this wiring is
    directly unit-testable without a live ROS graph -- mirrors
    ``TrainerBase.__init__``'s own (already-correct) wiring of the same two
    fields."""
    return {
        "telemetry_wait_timeout_sec": profile.runtime.risk_telemetry_wait_timeout_sec,
        "reset_marker_wait_timeout_sec": profile.runtime.risk_telemetry_reset_marker_timeout_sec,
        "wheelbase_m": profile.robot.wheelbase_m,
        "track_width_m": profile.robot.track_width_m,
    }


def validate_live_evaluation_contract(env: EnvironmentClient, requested_evaluation_contract_fingerprint: str) -> str:
    """section item-1/item-2 (round 2): fail-fast verification that the
    live environment_node ACTUALLY applied the evaluation-contract override
    ``main()`` just pushed to it -- reads back its
    ``evaluation_contract_fingerprint_sha256`` parameter (updated by
    ``environment_node.py::_resolve_evaluation_contract_override`` every
    time the active ``reward``/``scenario``/``runtime``/``evaluation``/
    ``sensor_noise`` sections change) and compares it against the fingerprint of what THIS
    node's own effective profile requires.

    This is a DIFFERENT check from ``validate_live_environment``'s
    architecture-fingerprint comparison above -- that one verifies the
    checkpoint's network shape still matches what the live environment
    decodes/observes; this one verifies the BENCHMARK CONDITIONS (world
    size, reward/termination, physics timing, common-metrics config) the
    live environment is actually running right now. A caller must call
    ``EnvironmentClient.set_evaluation_contract_override`` and check its
    own return value FIRST (the parameter-set RPC succeeding does not, by
    itself, prove environment_node.py's own ``_resolve_evaluation_contract_override``
    successfully parsed/applied the file -- e.g. a malformed override file
    would raise INSIDE the next ``/reset`` call, not at parameter-set time)
    -- this function's own mismatch here would also catch that failure
    mode, since a node that never actually applied the override keeps
    reporting its OLD fingerprint."""
    try:
        live_evaluation_contract_fingerprint = env.get_remote_parameter("evaluation_contract_fingerprint_sha256")
    except EnvServiceError as e:
        raise SystemExit(
            "evaluation_node: could not read the live environment_node's own "
            f"'evaluation_contract_fingerprint_sha256' parameter (relaunch environment_node.py after "
            f"rebuilding if it predates section item-1 round 2): {e}"
        ) from e
    if live_evaluation_contract_fingerprint != requested_evaluation_contract_fingerprint:
        raise SystemExit(
            "evaluation_node: live environment_node's evaluation_contract_fingerprint_sha256="
            f"{live_evaluation_contract_fingerprint!r} does not match this evaluation run's own "
            f"REQUESTED effective evaluation-contract fingerprint {requested_evaluation_contract_fingerprint!r} "
            "-- the environment_contract_override was set but does not appear to have been actually "
            "applied (a malformed override file, or a /reset never having run since it was set). "
            "Refusing to evaluate against a live environment that may still be using its own "
            "launch-time (checkpoint-training) reward/scenario/runtime/evaluation/sensor_noise settings."
        )
    return live_evaluation_contract_fingerprint


@dataclasses.dataclass
class _EvalRunState:
    """Mutable bag of everything ``run_eval_body_with_contract_restore``'s
    ``finally``/post-check logic needs, but that only becomes available
    partway through the evaluation body (``main()``'s former inline ``try``
    body) -- kept as a SEPARATE object (rather than closures over ``main()``
    locals) purely so the orchestration below is unit-testable with a tiny
    fake ``body`` callable, without needing a real checkpoint/live ROS
    graph."""
    override_applied: bool = False
    override_path: Optional[str] = None
    launch_time_evaluation_contract_fp: Optional[str] = None
    output_dir: Optional[str] = None


def run_eval_body_with_contract_restore(env: EnvironmentClient, body, state: _EvalRunState) -> None:
    """section item-4 (round 4, override-restore-failure fix): the
    GENERIC try/finally/post-check pattern factored out of ``main()`` so it
    is directly unit-testable (see tests/test_evaluation_node.py) without
    mocking ``rclpy.init``/a live ``EnvironmentClient``/a real checkpoint.

    ``body(state)`` is the evaluation run itself -- it MUST set
    ``state.output_dir``/``state.override_path``/
    ``state.launch_time_evaluation_contract_fp`` as it progresses, and
    ``state.override_applied = True`` only once
    ``env.set_evaluation_contract_override(...)`` has actually succeeded
    (mirrors ``main()``'s own prior inline sequencing exactly).

    Four guarantees, in order of priority:

    1. An exception raised by ``body`` (the evaluation itself) ALWAYS
       propagates UNMASKED -- restoration still runs (in ``finally``), and
       a restoration failure is still reported (printed + an attempted
       write of ``contract_restore_status.json``), but never by RAISING
       from inside this function's own ``finally`` block, which would
       silently replace the body's original exception with a different
       one. This holds even if writing ``contract_restore_status.json``
       ITSELF also fails (code review, restore-status durability fix):
       that write failure is caught locally, right here, and only
       PRINTED (never raised) -- Python would otherwise let a NEW
       exception raised inside an active ``finally`` block permanently
       replace whichever original exception was already propagating
       through it, which is exactly the masking this guarantee forbids.
    2. If ``body`` succeeds but the post-run contract restoration fails
       (``state.override_applied`` was True and
       ``_restore_launch_time_evaluation_contract`` returned
       ``ok=False``), this function raises ``SystemExit`` AFTER the
       ``finally`` block has completed -- so a caller checking only the
       process exit code observes failure, not a false "clean run"
       (the bug this whole mechanism exists to close: previously nothing
       past the WARNING print ever signalled this). If persisting THAT
       failure to ``contract_restore_status.json`` also failed, the
       ``SystemExit`` message says so too (the auxiliary error is
       preserved/reported, never dropped just because a more important
       error is already being raised).
    3. If ``body`` succeeds, the contract restoration succeeds (or no
       override was ever applied), but durably WRITING
       ``contract_restore_status.json`` still fails (code review,
       restore-status durability fix) -- this function ALSO raises
       ``SystemExit`` rather than returning normally: a caller relying on
       that file's mere presence as proof restoration was attempted and
       recorded must never see a clean exit code when it silently isn't
       there.
    4. If ``body`` succeeds, restoration succeeded (or no override was
       ever applied), AND the status file was durably written (or no
       override was ever applied, so nothing needed writing at all),
       this function returns normally."""
    restore_result: Optional[RestoreResult] = None
    status_write_error: Optional[OSError] = None
    try:
        body(state)
    finally:
        if state.override_applied:
            restore_result = _restore_launch_time_evaluation_contract(
                env, state.launch_time_evaluation_contract_fp, state.override_path)
            try:
                _write_contract_restore_status(state.output_dir, restore_result)
            except OSError as e:
                # Deliberately NOT re-raised here -- see guarantee 1 above:
                # raising from an active `finally` while `body` itself may
                # already be propagating an exception would silently
                # replace that original exception with this one. Preserved
                # instead as `status_write_error`, surfaced by the
                # SystemExit checks below WHENEVER control actually
                # reaches them (i.e. `body` did NOT raise) -- and, either
                # way, printed immediately so it is never lost purely
                # because `body` happened to raise too.
                status_write_error = e
                print(
                    f"evaluation_node: WARNING -- failed to durably persist contract_restore_status.json "
                    f"in {state.output_dir!r}: {e!r}"
                )
    # Unreachable when `body` raised (its exception already propagated out
    # of the `try/finally` above the instant `finally` completed without
    # itself raising) -- so every check below only ever fires on a `body`
    # that itself completed successfully.
    if restore_result is not None and not restore_result.ok:
        raise SystemExit(
            "evaluation_node: the evaluation run itself completed successfully, but restoring the live "
            f"environment_node's launch-time evaluation contract afterward FAILED (reason={restore_result.reason!r}"
            f"{f', restored_fingerprint={restore_result.restored_fingerprint!r}' if restore_result.restored_fingerprint else ''}"
            f") -- see {os.path.join(state.output_dir, 'contract_restore_status.json')!r}. The environment_node is "
            "likely left running under this run's evaluation contract; relaunch it before any further "
            "(non-evaluation) use."
            + (f" ADDITIONALLY, persisting this failure to contract_restore_status.json also failed: "
               f"{status_write_error!r} -- that file may be stale, absent, or from a previous run."
               if status_write_error is not None else "")
        )
    if status_write_error is not None:
        raise SystemExit(
            "evaluation_node: the evaluation run (and the live contract restore, if one was applied) completed "
            f"successfully, but durably persisting contract_restore_status.json in {state.output_dir!r} FAILED: "
            f"{status_write_error!r} -- refusing to exit 0 for a run whose durable restore-status record does not "
            "actually reflect what happened."
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", "-p", dest="profile_kv", action="append", default=[])
    args, _ = parser.parse_known_args()

    kv = {p.split(":=", 1)[0]: p.split(":=", 1)[1] for p in args.profile_kv if ":=" in p}
    profile_name = kv.get("profile", "evaluation_id")
    checkpoint_dir = kv.get("checkpoint_dir")
    checkpoint_name = kv.get("checkpoint_name", "final")

    if not checkpoint_dir:
        raise SystemExit("evaluation_node requires -p checkpoint_dir:=<run_dir>/checkpoints")
    # section item-3: the generation-layout manifest lives at
    # <checkpoint_dir>/<checkpoint_name>/manifest.json (checkpoint_name is a
    # SYMLINK directory to .generations/<generation>/, not a flat file --
    # see rl.checkpointing.manager's module docstring).
    manifest_path = os.path.join(checkpoint_dir, checkpoint_name, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise SystemExit(f"missing checkpoint: {manifest_path}")
    with open(manifest_path) as f:
        manifest = json.load(f)
    checkpoint_profile_name = manifest.get("profile_name") or "unknown_training_profile"
    profile = build_effective_profile(manifest, profile_name)
    resolved_config = manifest.get("resolved_config") or {}
    checkpoint_architecture_fp = architecture_fingerprint_from_resolved_config(resolved_config)

    requested_evaluation_contract_fp = evaluation_contract_fingerprint(profile)

    rclpy.init()
    # section item-1 (round 3): the risk-telemetry wait budgets are
    # THEMSELVES part of the runtime evaluation contract
    # (`risk_telemetry_wait_timeout_sec`/`risk_telemetry_reset_marker_timeout_sec`,
    # hashed into `evaluation_contract_fingerprint` via the whole `runtime`
    # section) -- but `EnvironmentClient`'s constructor previously fell
    # back to its own hardcoded defaults (1.0s/5.0s) whenever a caller
    # didn't pass them explicitly, and this call site never did. A false
    # guarantee: the fingerprint claimed these fields were part of the
    # contract, but the ACTUAL client-side wait budget used during
    # `env.step()`/`env.reset()` never reflected the requested profile at
    # all. Mirrors `TrainerBase.__init__`'s own (correct) wiring.
    env = EnvironmentClient(node_name="hunter_kinodynamic_eval_client", **environment_client_kwargs(profile))
    state = _EvalRunState()

    def _body(state: _EvalRunState) -> None:
        dims = env.get_dimensions()
        validate_live_environment(env, profile, checkpoint_profile_name, dims, checkpoint_architecture_fp)
        # section item-2 (round 3): captured BEFORE any override is ever
        # applied -- the ground truth this node restores back to, verified
        # (not just attempted) by run_eval_body_with_contract_restore's own
        # finally block below.
        state.launch_time_evaluation_contract_fp = env.get_remote_parameter("evaluation_contract_fingerprint_sha256")
        agent = build_agent(profile, dims)

        result = manager.load_generation(checkpoint_dir, checkpoint_name, agent.checkpoint_components())
        print(f"loaded checkpoint components: {result['loaded']} (skipped: {result['skipped']})")

        state.output_dir = os.path.join(checkpoint_dir, "..", "evaluation", profile.evaluation.benchmark)
        os.makedirs(state.output_dir, exist_ok=True)

        # section item-1 (round 2): deliver the effective evaluation
        # contract (reward/scenario/runtime/evaluation/sensor_noise) to the LIVE
        # environment_node -- see evaluation/contract_override.py and
        # environment_node.py::_resolve_evaluation_contract_override. Must
        # happen BEFORE run_benchmark's own scenario placement, and its
        # fingerprint must be VERIFIED (not just "the set-parameter RPC
        # returned success") before trusting any episode that follows.
        # section item-2 (round 3): ALWAYS an absolute, per-INVOCATION-
        # unique path (a fresh uuid4 per run, never a fixed filename) --
        # belt-and-suspenders on top of environment_node.py's own
        # content-hash-based staleness fix: even if some future caller
        # reused a path, the content hash alone would already catch a
        # change, but a genuinely unique path also rules out any
        # same-path/different-inode edge case and makes each run's own
        # override file trivially distinguishable on disk.
        state.override_path = os.path.abspath(
            os.path.join(state.output_dir, f"effective_evaluation_contract_{uuid.uuid4().hex}.yaml"))
        write_evaluation_contract_override(state.override_path, profile)
        if not env.set_evaluation_contract_override(state.override_path):
            raise SystemExit(
                f"evaluation_node: live environment_node rejected "
                f"evaluation_contract_override_path={state.override_path!r} -- refusing to evaluate against "
                "whatever contract the live node was already on instead"
            )
        state.override_applied = True
        # `_resolve_evaluation_contract_override` only runs INSIDE /reset --
        # setting the parameter alone does not yet apply it, so a single
        # warm-up reset is needed before the fingerprint it publishes can be
        # trusted to reflect what was just requested (see
        # validate_live_evaluation_contract's own docstring).
        env.reset()
        live_evaluation_contract_fp = validate_live_evaluation_contract(env, requested_evaluation_contract_fp)

        run_metadata = {
            "checkpoint_dir": os.path.abspath(checkpoint_dir), "checkpoint_name": checkpoint_name,
            "training_profile": checkpoint_profile_name, "evaluation_profile": profile_name,
            "algorithm": profile.algorithm.name, "action_space_mode": profile.action_space.mode,
            # section item-2: verifiable-from-the-file identity, beyond
            # scenario_id/seed (see benchmark_runner.run_benchmark) -- two
            # summary.json files with the SAME evaluation_contract_fingerprint
            # were genuinely evaluated under identical conditions (and this
            # round's live_evaluation_contract_fingerprint PROVES the live
            # environment actually applied it, not just that this node
            # requested it); two with the SAME checkpoint_architecture_fingerprint
            # came from bit-for-bit the same training-time architecture
            # config. evaluation_contract_override_path is provenance: the
            # exact file (preserved under output_dir) this run's contract
            # was delivered through.
            "checkpoint_architecture_fingerprint": checkpoint_architecture_fp,
            "evaluation_contract_fingerprint": requested_evaluation_contract_fp,
            "live_evaluation_contract_fingerprint": live_evaluation_contract_fp,
            "evaluation_contract_override_path": state.override_path,
        }
        summary = run_benchmark(profile, agent, state.output_dir, env=env, run_metadata=run_metadata)
        # section item-1 (Gazebo physics-step reality-check fix): record
        # whether physics was ACTUALLY verified for this run, not merely
        # declared -- see augment_summary_with_physics_calibration's own
        # docstring.
        summary = augment_summary_with_physics_calibration(summary, env, profile, state.output_dir)
        print(json.dumps(summary, indent=2, sort_keys=True))

    try:
        # section item-4 (round 4): see run_eval_body_with_contract_restore's
        # own docstring -- raises SystemExit if `_body` succeeded but the
        # post-run contract restoration failed, without masking any
        # exception `_body` itself raises.
        run_eval_body_with_contract_restore(env, _body, state)
    finally:
        env.destroy_node()
        # A SIGINT can already have triggered rclpy's own shutdown before
        # this finally block runs -- guard against calling it twice.
        if rclpy.ok():
            rclpy.shutdown()


def _restore_launch_time_evaluation_contract(
    env: EnvironmentClient, launch_time_evaluation_contract_fp, override_path: str,
) -> RestoreResult:
    """section item-2 (round 3)/item-4 (round 4, override-restore-failure
    fix): restores the live environment_node back to ITS OWN launch-time
    evaluation contract after this run -- called from ``main()``'s
    ``finally`` block, so it must run whether the evaluation run above
    succeeded, raised, or was interrupted.

    The round-2 version only called ``env.set_evaluation_contract_override("")``
    -- clearing the ROS PARAMETER -- and stopped there. But
    ``_resolve_evaluation_contract_override`` only ever runs INSIDE
    ``/reset``, so between that parameter-clear and whenever the NEXT
    ``/reset`` happens (which could be long after this process exits, or
    never, if the environment_node just sits idle), the live environment
    silently kept running under THIS run's stale contract even though the
    parameter itself already said ``""``. Fixed by forcing one more
    ``/reset`` right here, so the restoration ACTUALLY takes effect before
    this function returns, then reading back
    ``evaluation_contract_fingerprint_sha256`` to VERIFY it matches the
    captured launch-time value.

    Restoration failures are REPORTED loudly (never silently swallowed) via
    the SAME ``print(...)`` WARNING messages as before (round-3 callers/tests
    depend on the exact wording), but this function itself never RAISES --
    doing so from a ``finally``-block helper would mask whatever exception,
    if any, is already propagating out of the ``try`` body this cleans up
    after. Instead the verdict is returned as a structured
    :class:`RestoreResult` (section item-4, round 4) -- ``main()`` reads it
    back AFTER its own ``try/finally`` has already let any original
    exception propagate unmasked, and raises there if restoration failed
    while the evaluation body itself succeeded, so a caller checking only
    the process exit code still observes the failure (previously this
    function's WARNING print was the only signal, and the process always
    exited 0 whenever the try body itself didn't raise)."""
    try:
        if not env.set_evaluation_contract_override(""):
            print(
                "evaluation_node: WARNING -- live environment_node REJECTED clearing "
                "evaluation_contract_override_path; it is very likely still running under this run's "
                f"evaluation contract (override file: {override_path!r}). Relaunch it to recover a "
                "known-good state before any further (non-evaluation) use.",
            )
            return RestoreResult(ok=False, reason="set_evaluation_contract_override_rejected")
        env.reset()  # forces _resolve_evaluation_contract_override to actually apply the restoration NOW
        if launch_time_evaluation_contract_fp is None:
            print(
                "evaluation_node: WARNING -- launch-time evaluation_contract_fingerprint_sha256 was never "
                "captured (an earlier failure interrupted this run before it could be read) -- cannot "
                "verify the restoration actually recovered the live environment's ORIGINAL contract."
            )
            return RestoreResult(ok=False, reason="launch_time_fingerprint_never_captured")
        restored_fp = env.get_remote_parameter("evaluation_contract_fingerprint_sha256")
        if restored_fp != launch_time_evaluation_contract_fp:
            print(
                "evaluation_node: WARNING -- FAILED to restore the live environment_node's launch-time "
                f"evaluation contract after this run. live evaluation_contract_fingerprint_sha256="
                f"{restored_fp!r}, expected (this node's own launch-time value) "
                f"{launch_time_evaluation_contract_fp!r}. The environment_node may be left running under a "
                "stale contract -- relaunch it before any further (non-evaluation) use."
            )
            return RestoreResult(ok=False, reason="fingerprint_mismatch_after_restore", restored_fingerprint=restored_fp)
        return RestoreResult(ok=True, restored_fingerprint=restored_fp)
    except Exception as e:  # noqa: BLE001 -- cleanup must never crash the process (or mask an in-flight
                             # exception from the try body above) on top of whatever went wrong
        print(
            f"evaluation_node: WARNING -- could not verify evaluation-contract restoration after this run: "
            f"{e!r}. The environment_node may be left running under a stale contract -- relaunch it before "
            "any further (non-evaluation) use."
        )
        return RestoreResult(ok=False, reason=f"exception_during_restore: {e!r}")


def _atomic_write_json(path: str, payload: dict) -> None:
    """code review (restore-status durability fix): writes ``payload`` to
    ``path`` durably -- the FULL content is written, flushed, and fsynced
    to a temp file in the SAME directory as ``path`` first (same-directory
    is required so the final ``os.replace`` is an atomic same-filesystem
    rename, never a cross-filesystem copy), then published via a single
    atomic ``os.replace``, then the PARENT directory itself is fsynced too
    -- matching ``rl.checkpointing.manager``'s own write-then-publish
    convention (see that module's ``_fsync_path`` docstring for why the
    directory-entry fsync is needed ON TOP OF ``os.replace``'s rename
    atomicity: a crash right after the rename could otherwise still lose
    the fact that the directory entry was ever updated).

    Raises ``OSError`` on ANY failure (temp-file write/flush/fsync, the
    replace itself, or the parent-directory fsync) -- never silently drops
    a partial write the way the pre-fix plain ``open(path, "w")`` +
    swallowed ``OSError`` did. The temp file is best-effort removed on
    failure (never left littering ``output_dir`` on a write/fsync error
    before ``os.replace`` ran); a failure DURING or AFTER ``os.replace``
    itself means ``path`` may or may not have been updated -- the caller
    is responsible for treating this function raising at all as "the
    durable artifact's state is not certain," never as a partial success."""
    directory = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=os.path.basename(path) + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp_path)
        raise
    dir_fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _write_contract_restore_status(output_dir: str, restore_result: RestoreResult) -> None:
    """section item-4 (round 4)/code review (restore-status durability
    fix): persists ``restore_result`` as a durable, machine-readable
    artifact next to ``summary.json`` -- the print-only WARNING elsewhere
    in this module is easy to miss in a captured/redirected log; a caller
    that only checks files after the process exits (a CI job, an
    orchestration script) needs somewhere concrete to look. Written
    unconditionally (both success and failure) so its mere ABSENCE never
    looks like "restoration wasn't attempted" vs. "it succeeded".

    Unlike the pre-fix version, a write failure here is NEVER silently
    swallowed -- it propagates (``OSError``, via :func:`_atomic_write_json`)
    to the caller (:func:`run_eval_body_with_contract_restore`), which
    decides what "this durable artifact did not actually make it to disk"
    means for the process's own exit status, without ever masking the
    evaluation body's own original exception if one is already in
    flight (see that function's own docstring)."""
    path = os.path.join(output_dir, "contract_restore_status.json")
    _atomic_write_json(path, dataclasses.asdict(restore_result))


if __name__ == "__main__":
    main()
