#!/usr/bin/env python3
"""``ros2 run hunter_kinodynamic_rl real_policy_node.py --ros-args -p profile:=real_hunter_safe -p checkpoint_dir:=<run_dir>/checkpoints -p checkpoint_name:=final -p goal_x:=3.0 -p goal_y:=2.0``

Real-Hunter-SE (or live-Gazebo) inference-only control loop: loads a trained
checkpoint, subscribes to the SAME sensor topics environment_node.py itself
consumes (``/scan``, ``/odometry``, ``/hunter_se/joint_states`` -- identical
names whether the publisher is Gazebo or the real robot's own driver stack,
by hunter_se_gazebo's own bridge-config design), and publishes ``/cmd_vel``
at a fixed rate -- INFERENCE ONLY, this node never trains and never touches
Gazebo-specific services (no ``/reset``, ``/step``, ``SetEntityPose``,
``ControlWorld``; those don't exist on real hardware).

Every piece of control-loop LOGIC here is a call into an already-tested,
ROS-free pure module shared with environment_node.py -- this file is thin
ROS glue (sensor subscriptions, a timer, a publisher), matching this
package's mixin/pure-function convention:
  observation build : env/observation/observation_builder.py, sensing/{scan_processor,temporal_stack}.py
  action -> command  : trajectory/trajectory_executor.py (explicitly documented as
                        shared between "the env node, real-robot inference node")
  safety             : env/safety/action_guard.py -- MANDATORY here (section 43),
                        never optional, exactly as on the sim path

Refuses to run against any profile that ISN'T marked
``runtime.deployment: real_hardware`` (section P2: real profile 자체가 그 표시를
갖는다) -- ``real_hunter_safe.yaml`` is the one shipped profile that sets it.

One deliberate, DOCUMENTED sim/real asymmetry: this node never computes or
publishes ``risk_telemetry`` (unlike environment_node.py's ``/step``) --
that computation needs PRIVILEGED ground-truth obstacle positions
(``env/simulation/privileged_snapshot.py``, sourced from Gazebo entity
state) that simply do not exist on real hardware. The safety guard's own
LiDAR-derived collision-proximity stop (``action_guard.apply_collision_proximity_stop``)
is what actually protects the robot here; risk telemetry on the real path
would be a fabricated number with no privileged obstacles to compute it
from, so it is omitted rather than faked.

Section P2-11 safety/operational interface:
  ``-p dry_run:=true``     -- runs the full pipeline (observation, inference,
                              guard, diagnostics) every tick but NEVER
                              publishes to ``cmd_vel_topic`` (the ONE call
                              site that reaches it is ``_publish``).
  ``-p replay_mode:=true`` -- pairs with feeding this node pre-recorded
                              sensor data via ``ros2 bag play <bag> --clock``
                              (this node's interface is plain topic
                              subscriptions, so it needs no bag-reading code
                              of its own) -- REQUIRES ``dry_run:=true``
                              (refuses to start otherwise): replayed data
                              must never be able to actuate a real robot.
  ``-p estop_topic:=...``  -- a ``std_msgs/Bool`` software E-stop kill
                              switch (default
                              ``/hunter_kinodynamic_rl/real_policy/estop``);
                              once latched True, every tick publishes
                              ``STOP_COMMAND`` (never the policy's output)
                              until a False message clears it -- independent
                              of, and in addition to, ``action_guard.guard``'s
                              own sensor/command-freshness and
                              collision-proximity checks.
  ``-p {scan,odom,joint_states,cmd_vel}_topic:=...`` -- every subscribed/
                              published topic name is configurable.
  ``-p sensor_qos_reliability:=best_effort|reliable`` -- scan/odom
                              subscription QoS (default ``best_effort``,
                              matching this package's Gazebo bridge
                              convention); a real driver stack that
                              publishes RELIABLE instead needs this set or
                              the subscription silently receives nothing
                              (the exact live bug this module's sibling,
                              ``evaluation/nav2_mppi_runner.py``, already
                              found and documents for its own ``/scan``
                              subscription).

Section P0-3 safety-responsibility boundary vs. ``hunter_se_cmd_prefilter``:
that package's own ``command_timeout_sec`` watchdog (it zeros ``/cmd_vel``
if not refreshed within its own budget) is a DOWNSTREAM, defense-in-depth
backstop -- it exists whether or not this node is even the one publishing
``/cmd_vel`` (e.g. a human teleop source), and this node MUST NOT rely on it
to satisfy its OWN real-robot safety obligations. Everything this node is
responsible for is self-contained, upstream of the prefilter, and does not
assume the prefilter is even running:
  - LiDAR/odometry staleness, command staleness, and collision-proximity
    stop: ``env/safety/action_guard.py``'s ``guard()``, called every tick.
  - a genuinely HUNG control-tick callback (e.g. inference wedged past its
    own timeout thread's join): the independent watchdog THREAD started in
    ``__init__`` (a plain Python thread, deliberately never an rclpy timer
    -- see ``_start_watchdog_thread``'s docstring for why an rclpy timer
    cannot provide this guarantee under a SingleThreadedExecutor).
  - NaN/Inf or malformed actions, and any unexpected exception anywhere in
    the observation/inference/decode/guard pipeline: caught in
    ``_on_control_tick`` itself, always resulting in a published safe stop.
  - operator-triggered stop: the latching E-stop topic.
  - REPEATED/SUSTAINED inference timeouts, not just one slow tick: both
    ``_infer_with_timeout_thread`` and ``_infer_with_timeout_process`` are
    SINGLE-FLIGHT -- if the previous call has not resolved yet, the next
    control tick does NOT start another worker (thread or IPC request); it
    just counts another timeout and publishes a safe stop. Without this, a
    genuinely stuck inference call under the default 10 Hz control period
    would otherwise leak one new daemon thread per tick for as long as the
    hang lasted. ``runtime.inference_worker_mode='process'`` (opt-in,
    default ``'thread'``) additionally gives this node a REAL kill/restart
    capability for a permanently wedged inference call -- CPython cannot
    forcibly terminate a thread, but ``InferenceWorkerProcess.restart()``
    can SIGTERM/SIGKILL and respawn a stuck OS process, triggered after
    ``runtime.inference_worker_max_consecutive_timeouts`` consecutive
    timeouts. ``_last_successful_inference_time`` (distinct from
    ``_last_command_time``, which a published STOP still refreshes) drives
    a separate, throttled watchdog-thread diagnostic
    (``runtime.policy_health_timeout_sec``) for "the policy itself has
    stopped producing usable actions", even though the robot is already
    guaranteed safely stopped by the mechanisms above regardless.
"""

from __future__ import annotations

import dataclasses
import json
import math
import multiprocessing as mp
import os
import queue
import threading
import time
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState, LaserScan
from std_msgs.msg import Bool, Float32MultiArray

from hunter_kinodynamic_rl.config.loader import load_profile, profile_from_dict
from hunter_kinodynamic_rl.config.schema import Profile, RobotConfig
from hunter_kinodynamic_rl.env.observation.observation_builder import (
    RobotState, build_observation, build_robot_state_vector,
)
from hunter_kinodynamic_rl.env.safety.action_guard import STOP_COMMAND, SafetyLimits, guard
from hunter_kinodynamic_rl.rl.algorithms.kinodynamic_tqc.agent import Agent as RiskAgent
from hunter_kinodynamic_rl.rl.algorithms.tqc.agent import Agent as VanillaAgent
from hunter_kinodynamic_rl.rl.checkpointing import manager as ckpt_manager
from hunter_kinodynamic_rl.sensing.scan_processor import front_and_full_state
from hunter_kinodynamic_rl.sensing.temporal_stack import FrameStack
from hunter_kinodynamic_rl.trajectory import trajectory_executor
from hunter_kinodynamic_rl.trajectory.action_space import ACTION_DIM


class CheckpointProfileMismatchError(SystemExit):
    """Raised (never just logged) when the requested deployment profile's
    ARCHITECTURE-determining sections disagree with the checkpoint's own
    training-time resolved_config -- see build_effective_profile's
    docstring. A SystemExit subclass (not a plain Exception) so it always
    aborts node construction the same way every other startup-time
    misconfiguration in this file does (missing checkpoint_dir, wrong
    runtime.deployment, ...), never something a caller could catch and
    silently continue past."""


class UnsafeDeploymentOverrideError(CheckpointProfileMismatchError):
    """code review (real_policy_node action-space override too permissive):
    raised when the REQUESTED profile's action_space bounds would WIDEN the
    reachable action range beyond what the checkpoint was actually trained
    on -- e.g. a larger kappa_scale, a lower v_min, a wider horizon_length
    range. A strict SUBCLASS of CheckpointProfileMismatchError (still a
    SystemExit, still caught by the exact same
    ``pytest.raises(CheckpointProfileMismatchError)`` pattern
    test_real_policy_node.py already used for architecture mismatches) but
    kept as its own type -- and its own, differently-worded message
    section -- so a caller/log reader can tell "the checkpoint and profile
    are for different network architectures" apart from "this specific
    deployment profile is trying to let the policy do something more than
    it was trained for", per the review's explicit ask that allowed vs.
    rejected overrides be unambiguous in the error message."""


def build_effective_profile(manifest: dict, requested_profile_name: str, requested_profile: Profile) -> Profile:
    """code review (section P2/real-robot safety): the previous version of
    this check only compared ``manifest["profile_name"]`` (a free-form
    string) against the requested ``--profile`` name and WARNED on
    mismatch, then proceeded to build the agent from the REQUESTED
    profile's own action_space/features/risk/counterfactual/hyperparameters
    -- i.e. it could build a RiskAgent-shaped network and then load a
    VanillaAgent checkpoint (or vice versa), or decode actions under the
    wrong action_space.mode, and still go on to publish real robot
    commands. This mirrors evaluation_node.py's
    ``build_effective_profile`` (the same fix already applied to the
    simulation-evaluation path): ARCHITECTURE-determining sections
    (action_space.mode, features.{risk_critic,counterfactual_risk,
    temporal_context}, observation state-dim fields, hyperparameters, robot
    geometry) are read from the checkpoint's OWN ``resolved_config`` and
    checked for exact compatibility with the requested profile -- any
    mismatch raises :class:`CheckpointProfileMismatchError` BEFORE any
    agent is constructed or any checkpoint tensor is loaded, so an
    incompatible pairing can never reach the real robot's ``cmd_vel``.

    DEPLOYMENT-SAFETY sections -- ``robot.max_forward_speed_mps`` (the
    conservative real-hardware speed cap), ``action_space``'s own bounds
    (e.g. ``real_hunter_safe.yaml``'s shorter ``horizon_length_max_m``), and
    ``risk.min_safe_clearance_m`` -- are layered from the REQUESTED profile
    on top of the checkpoint's restored architecture, never overridden back
    to the checkpoint's training-time values -- but code review found the
    PREVIOUS version of this only checked ``action_space.mode`` and then
    blindly accepted the requested profile's ENTIRE ``action_space`` section
    verbatim: a deployment profile could WIDEN ``kappa_scale``, LOWER
    ``v_min_mps``, or widen ``horizon_length_min_m``/``horizon_length_max_m``
    beyond what the checkpoint ever saw during training -- none of that is a
    "cap", it changes what a given raw actor output PHYSICALLY MEANS, and
    for the parts of the action space the policy never explored (e.g. a
    curvature range 50% wider than it ever trained on) its behavior is
    genuinely undefined, not just "less safe". Every action_space bound
    below is therefore explicitly validated SHRINK-ONLY (the requested
    profile's own reachable range must be a SUBSET of the checkpoint's own
    trained range) before being accepted, raising
    :class:`UnsafeDeploymentOverrideError` (never silently clamping or
    warning) on any violation:
      - effective v_max (``action_space.v_max_mps`` or, when that's unset,
        ``robot.max_forward_speed_mps``): requested <= checkpoint's own.
      - ``action_space.v_min_mps``: requested >= checkpoint's own (the
        floor may only be RAISED, never lowered toward/into reverse).
      - ``action_space.kappa_scale``: requested <= checkpoint's own.
      - ``action_space.horizon_length_min_m``: requested >= checkpoint's own.
      - ``action_space.horizon_length_max_m``: requested <= checkpoint's own.
      - (``legacy_waypoint`` mode only) ``legacy_r_min_m``: requested >=
        checkpoint's own; ``legacy_r_max_m``: requested <= checkpoint's
        own; ``legacy_theta_max_rad``: requested <= checkpoint's own;
        ``legacy_yield_enabled``: EXACT match (toggling a whole action
        channel off is a semantics change, not a numeric shrink).
    code review (round 2 -- robot-override too permissive): the PREVIOUS
    version of this function validated only ``wheelbase_m``/
    ``steering_limit_deg`` (exact match) and the derived effective v_max,
    then still took the ENTIRE requested ``robot`` section verbatim into
    ``effective`` -- so ``steering_rate_deg_s``, ``accel_limit_mps2``,
    ``brake_decel_mps2``, ``speed_lag_tau_sec``, and ``collision_radius_m``
    (every one of them read live by ``dynamics/ackermann_rollout.py``'s
    risk rollout, ``dynamics/stopping_model.py``'s stopping-margin/guard
    check, or both) could silently become MORE OPTIMISTIC than what
    training used, with zero validation. Fixed the same way as the
    action-space bounds: every one of these is now validated in whichever
    direction keeps the rollout/guard's assumption AT LEAST AS PESSIMISTIC
    as training (see ``_layer_effective_robot`` and the checks just above
    its call site for the exact per-field direction and rationale), and
    ``effective.robot`` is built by LAYERING only these validated fields
    onto the checkpoint's OWN training-time robot config -- never a
    wholesale ``robot=requested_profile.robot``. Every remaining robot.*
    field (identity/geometry: ``track_width_m``, ``wheel_radius_m``,
    ``length_m``, ``width_m``, ``height_m``, ``mass_kg``,
    ``min_forward_speed_mps``, plus the pre-existing ``wheelbase_m``/
    ``steering_limit_deg`` exact-match checks) must match the checkpoint
    EXACTLY -- confirmed by grep that none of them is read by any
    inference-time (guard/risk/action-decode) code path in this package,
    so there is no legitimate reason for a deployment profile to change
    them, and silently accepting a drifted value there would be
    misleading rather than useful.
    ``risk.min_safe_clearance_m`` is validated shrink-only in the OPPOSITE
    direction from the action-space bounds above -- requested >=
    checkpoint's own (only a LARGER required clearance, i.e. more
    conservative, is accepted; the deployment profile may never accept
    LESS clearance than the risk critic was trained to require)."""
    resolved_config = manifest.get("resolved_config")
    if resolved_config is None:
        raise SystemExit(
            "real_policy_node: checkpoint manifest has no 'resolved_config' (pre-P0-1 schema_version=1 "
            "checkpoint) -- cannot safely verify architecture/action-space compatibility before "
            "publishing real robot commands. Retrain, or hand-verify compatibility and bypass this node."
        )
    checkpoint_profile_name = manifest.get("profile_name") or "unknown_training_profile"
    # resolved_config is dataclasses.asdict(profile) -- includes `name`
    # alongside the section dicts profile_from_dict() expects.
    sections_only = {k: v for k, v in resolved_config.items() if k != "name"}
    training_profile = profile_from_dict(checkpoint_profile_name, sections_only)

    mismatches = []
    if training_profile.action_space.mode != requested_profile.action_space.mode:
        mismatches.append(
            f"action_space.mode: checkpoint={training_profile.action_space.mode!r} "
            f"requested={requested_profile.action_space.mode!r} (different action dimensionality/meaning)"
        )
    if training_profile.features.risk_critic != requested_profile.features.risk_critic:
        mismatches.append(
            f"features.risk_critic: checkpoint={training_profile.features.risk_critic} "
            f"requested={requested_profile.features.risk_critic} (RiskAgent vs. plain TQC network shape)"
        )
    if training_profile.features.counterfactual_risk != requested_profile.features.counterfactual_risk:
        mismatches.append(
            f"features.counterfactual_risk: checkpoint={training_profile.features.counterfactual_risk} "
            f"requested={requested_profile.features.counterfactual_risk}"
        )
    if training_profile.features.temporal_context != requested_profile.features.temporal_context:
        mismatches.append(
            f"features.temporal_context: checkpoint={training_profile.features.temporal_context} "
            f"requested={requested_profile.features.temporal_context} (changes state_dim)"
        )
    if training_profile.observation.lidar_bins != requested_profile.observation.lidar_bins:
        mismatches.append(
            f"observation.lidar_bins: checkpoint={training_profile.observation.lidar_bins} "
            f"requested={requested_profile.observation.lidar_bins} (changes state_dim)"
        )
    # section P0-4: robot_state_dim (7 = drl_agent baseline parity, 8 = +
    # L-memory) changes state_dim exactly like lidar_bins/frame_stack does
    # -- a mismatch here would otherwise only surface as an opaque PyTorch
    # tensor-shape error deep inside the actor's first linear layer.
    if training_profile.observation.robot_state_dim != requested_profile.observation.robot_state_dim:
        mismatches.append(
            f"observation.robot_state_dim: checkpoint={training_profile.observation.robot_state_dim} "
            f"requested={requested_profile.observation.robot_state_dim} (changes state_dim)"
        )
    if training_profile.features.temporal_context and (
        training_profile.observation.frame_stack != requested_profile.observation.frame_stack
    ):
        mismatches.append(
            f"observation.frame_stack: checkpoint={training_profile.observation.frame_stack} "
            f"requested={requested_profile.observation.frame_stack} (changes state_dim under temporal_context)"
        )
    if training_profile.robot.wheelbase_m != requested_profile.robot.wheelbase_m:
        mismatches.append(
            f"robot.wheelbase_m: checkpoint={training_profile.robot.wheelbase_m} "
            f"requested={requested_profile.robot.wheelbase_m} (changes kappa<->steering geometry)"
        )
    if training_profile.robot.steering_limit_deg != requested_profile.robot.steering_limit_deg:
        mismatches.append(
            f"robot.steering_limit_deg: checkpoint={training_profile.robot.steering_limit_deg} "
            f"requested={requested_profile.robot.steering_limit_deg} (changes max_curvature)"
        )
    # code review (real_policy_node robot-override too permissive): every
    # OTHER robot.* field that isn't one of the explicit, validated
    # DEPLOYMENT-SAFETY tunables below (see _layer_effective_robot) is an
    # identity/geometry field this package's inference-time code either
    # never reads (wheel_radius_m, mass_kg, track_width_m, length_m,
    # width_m, height_m -- confirmed by grep: only domain_randomizer.py, a
    # TRAINING-time-only module, touches them) or already requires exact
    # match above (wheelbase_m, steering_limit_deg). Requiring EXACT match
    # here too -- rather than silently accepting whatever the requested
    # profile happens to set -- means a deployment profile can never
    # silently drift these without it being surfaced as an explicit error
    # (previously: build_effective_profile just took `robot=
    # requested_profile.robot` wholesale, so a typo'd or copy-pasted
    # override here would silently apply with zero validation).
    for field in ("name", "track_width_m", "wheel_radius_m", "length_m", "width_m", "height_m", "mass_kg",
                  "min_forward_speed_mps"):
        t_val, r_val = getattr(training_profile.robot, field), getattr(requested_profile.robot, field)
        if t_val != r_val:
            mismatches.append(
                f"robot.{field}: checkpoint={t_val!r} requested={r_val!r} (not one of the deployment-safety "
                "tunables that may be overridden -- see _layer_effective_robot)"
            )

    if mismatches:
        raise CheckpointProfileMismatchError(
            "real_policy_node: checkpoint/profile ARCHITECTURE mismatch -- refusing to construct an "
            f"agent or publish real robot commands. checkpoint trained under profile "
            f"{checkpoint_profile_name!r}, requested profile {requested_profile_name!r}:\n  - "
            + "\n  - ".join(mismatches)
        )

    # code review (real_policy_node action-space override too permissive):
    # SHRINK-ONLY validation for every action_space/robot-speed/risk-
    # clearance bound the requested profile is allowed to override -- see
    # this function's docstring for the exact list and direction each
    # field must move. A tiny epsilon absorbs float round-trip noise
    # (these values pass through YAML -> dataclass -> here), never enough
    # to hide a REAL widening.
    eps = 1e-9

    def _effective_v_max(action_cfg, robot_cfg) -> float:
        return action_cfg.v_max_mps if action_cfg.v_max_mps is not None else robot_cfg.max_forward_speed_mps

    unsafe_overrides = []
    t_action, r_action = training_profile.action_space, requested_profile.action_space
    t_v_max = _effective_v_max(t_action, training_profile.robot)
    r_v_max = _effective_v_max(r_action, requested_profile.robot)
    if r_v_max > t_v_max + eps:
        unsafe_overrides.append(
            f"effective v_max (action_space.v_max_mps or robot.max_forward_speed_mps): "
            f"checkpoint={t_v_max:.4f} requested={r_v_max:.4f} -- a deployment profile may only "
            "SHRINK the speed ceiling, never raise it above what training used"
        )
    # code review (real_policy_node robot-override too permissive): these
    # robot.* fields feed the LIVE inference-time dynamics/risk-rollout
    # (dynamics/ackermann_rollout.py, dynamics/stopping_model.py) and the
    # action guard's hard speed cap (env/safety/action_guard.py) -- unlike
    # the action-space bounds above (which are about the POLICY's action
    # semantics), these describe the robot's ASSUMED physical response.
    # Each is validated in whichever direction makes the assumption more
    # PESSIMISTIC than training, never more optimistic -- an optimistic
    # assumption (faster steering/accel/braking than trained on, less
    # speed lag, a smaller collision footprint) would make the rollout/
    # guard UNDERESTIMATE real risk, exactly the failure mode this review
    # exists to close:
    #   - robot.max_forward_speed_mps: shrink-only (independent of the
    #     effective-v_max check above, which only covers it when
    #     action_space.v_max_mps is unset -- action_guard.guard() enforces
    #     this field directly as a hard cap regardless of v_max_mps).
    #   - steering_rate_deg_s / accel_limit_mps2 / brake_decel_mps2:
    #     shrink-only -- a SLOWER assumed steering/accel/braking response
    #     makes the rollout predict a LESS capable vehicle (more
    #     conservative), never a MORE capable one than training modeled.
    #   - speed_lag_tau_sec: raise-only -- a LARGER tau means the rollout
    #     assumes speed responds MORE sluggishly (more conservative).
    #   - collision_radius_m: raise-only -- a LARGER assumed footprint
    #     means the guard/risk rollout keeps MORE clearance, never less.
    if requested_profile.robot.max_forward_speed_mps > training_profile.robot.max_forward_speed_mps + eps:
        unsafe_overrides.append(
            f"robot.max_forward_speed_mps: checkpoint={training_profile.robot.max_forward_speed_mps} "
            f"requested={requested_profile.robot.max_forward_speed_mps} -- action_guard.guard() enforces this "
            "as a hard cap regardless of action_space.v_max_mps; may only SHRINK, never raise it"
        )
    if requested_profile.robot.steering_rate_deg_s > training_profile.robot.steering_rate_deg_s + eps:
        unsafe_overrides.append(
            f"robot.steering_rate_deg_s: checkpoint={training_profile.robot.steering_rate_deg_s} "
            f"requested={requested_profile.robot.steering_rate_deg_s} -- may only assume a slower-or-equal "
            "steering actuator than training, never a faster one (would underestimate real steering lag)"
        )
    if requested_profile.robot.accel_limit_mps2 > training_profile.robot.accel_limit_mps2 + eps:
        unsafe_overrides.append(
            f"robot.accel_limit_mps2: checkpoint={training_profile.robot.accel_limit_mps2} "
            f"requested={requested_profile.robot.accel_limit_mps2} -- may only assume weaker-or-equal "
            "acceleration than training, never stronger (would underestimate closing speed)"
        )
    if requested_profile.robot.brake_decel_mps2 > training_profile.robot.brake_decel_mps2 + eps:
        unsafe_overrides.append(
            f"robot.brake_decel_mps2: checkpoint={training_profile.robot.brake_decel_mps2} "
            f"requested={requested_profile.robot.brake_decel_mps2} -- may only assume weaker-or-equal braking "
            "than training, never stronger (would underestimate real stopping distance)"
        )
    if requested_profile.robot.speed_lag_tau_sec < training_profile.robot.speed_lag_tau_sec - eps:
        unsafe_overrides.append(
            f"robot.speed_lag_tau_sec: checkpoint={training_profile.robot.speed_lag_tau_sec} "
            f"requested={requested_profile.robot.speed_lag_tau_sec} -- may only assume MORE-or-equal speed "
            "lag than training, never less (less lag means the rollout thinks speed responds faster than reality)"
        )
    if requested_profile.robot.collision_radius_m < training_profile.robot.collision_radius_m - eps:
        unsafe_overrides.append(
            f"robot.collision_radius_m: checkpoint={training_profile.robot.collision_radius_m} "
            f"requested={requested_profile.robot.collision_radius_m} -- may only assume a LARGER-or-equal "
            "footprint than training, never smaller (a smaller assumed footprint keeps less real clearance)"
        )
    if r_action.v_min_mps < t_action.v_min_mps - eps:
        unsafe_overrides.append(
            f"action_space.v_min_mps: checkpoint={t_action.v_min_mps} requested={r_action.v_min_mps} -- "
            "the speed floor may only be RAISED (or left unchanged), never lowered below what training used"
        )
    if r_action.kappa_scale > t_action.kappa_scale + eps:
        unsafe_overrides.append(
            f"action_space.kappa_scale: checkpoint={t_action.kappa_scale} requested={r_action.kappa_scale} -- "
            "may only shrink the curvature range, never widen it beyond training"
        )
    if r_action.horizon_length_min_m < t_action.horizon_length_min_m - eps:
        unsafe_overrides.append(
            f"action_space.horizon_length_min_m: checkpoint={t_action.horizon_length_min_m} "
            f"requested={r_action.horizon_length_min_m} -- may only be raised, never lowered below training"
        )
    if r_action.horizon_length_max_m > t_action.horizon_length_max_m + eps:
        unsafe_overrides.append(
            f"action_space.horizon_length_max_m: checkpoint={t_action.horizon_length_max_m} "
            f"requested={r_action.horizon_length_max_m} -- may only shrink, never widen beyond training"
        )
    if r_action.mode == "legacy_waypoint":
        if r_action.legacy_r_min_m < t_action.legacy_r_min_m - eps:
            unsafe_overrides.append(
                f"action_space.legacy_r_min_m: checkpoint={t_action.legacy_r_min_m} "
                f"requested={r_action.legacy_r_min_m} -- may only be raised, never lowered below training"
            )
        if r_action.legacy_r_max_m > t_action.legacy_r_max_m + eps:
            unsafe_overrides.append(
                f"action_space.legacy_r_max_m: checkpoint={t_action.legacy_r_max_m} "
                f"requested={r_action.legacy_r_max_m} -- may only shrink, never widen beyond training"
            )
        if r_action.legacy_theta_max_rad > t_action.legacy_theta_max_rad + eps:
            unsafe_overrides.append(
                f"action_space.legacy_theta_max_rad: checkpoint={t_action.legacy_theta_max_rad} "
                f"requested={r_action.legacy_theta_max_rad} -- may only shrink, never widen beyond training"
            )
        if r_action.legacy_yield_enabled != t_action.legacy_yield_enabled:
            unsafe_overrides.append(
                f"action_space.legacy_yield_enabled: checkpoint={t_action.legacy_yield_enabled} "
                f"requested={r_action.legacy_yield_enabled} -- toggling a whole action channel is a "
                "semantics change, not a numeric shrink; must match exactly"
            )
    if requested_profile.risk.min_safe_clearance_m < training_profile.risk.min_safe_clearance_m - eps:
        unsafe_overrides.append(
            f"risk.min_safe_clearance_m: checkpoint={training_profile.risk.min_safe_clearance_m} "
            f"requested={requested_profile.risk.min_safe_clearance_m} -- deployment may only REQUIRE MORE "
            "clearance than training did, never less"
        )

    if unsafe_overrides:
        raise UnsafeDeploymentOverrideError(
            "real_policy_node: requested profile's deployment overrides would WIDEN the action space, make "
            "the robot's assumed physical response MORE OPTIMISTIC than training, or weaken the "
            "risk-clearance requirement -- refusing to publish real robot commands under an "
            "undefined-behavior/underestimated-risk configuration. checkpoint trained under profile "
            f"{checkpoint_profile_name!r}, requested profile "
            f"{requested_profile_name!r}. ALLOWED overrides: shrinking v_max/kappa_scale/horizon_length_max/"
            "legacy_r_max/legacy_theta_max/robot.{max_forward_speed_mps,steering_rate_deg_s,accel_limit_mps2,"
            "brake_decel_mps2}, raising v_min/horizon_length_min/legacy_r_min/risk.min_safe_clearance_m/"
            "robot.{speed_lag_tau_sec,collision_radius_m}. REJECTED here:\n  - " + "\n  - ".join(unsafe_overrides)
        )

    effective = dataclasses.replace(
        training_profile,
        name=requested_profile_name,
        robot=_layer_effective_robot(training_profile.robot, requested_profile.robot),
        action_space=requested_profile.action_space,
        runtime=requested_profile.runtime,
        risk=dataclasses.replace(training_profile.risk, min_safe_clearance_m=requested_profile.risk.min_safe_clearance_m),
    )
    effective.validate()
    return effective


def _layer_effective_robot(training_robot: RobotConfig, requested_robot: RobotConfig) -> RobotConfig:
    """Builds the effective ``robot`` config from the checkpoint's OWN
    training-time robot config, with ONLY the deployment-safety tunables
    ALREADY VALIDATED (shrink-only / raise-only, see build_effective_profile
    above) layered on top from the requested profile. Every other field --
    geometry/identity fields already required to match exactly above -- is
    taken from ``training_robot``, never ``requested_robot``, by
    construction (previously this was `robot=requested_profile.robot`
    wholesale, silently accepting ANY unvalidated field the requested
    profile happened to set)."""
    return dataclasses.replace(
        training_robot,
        max_forward_speed_mps=requested_robot.max_forward_speed_mps,
        steering_rate_deg_s=requested_robot.steering_rate_deg_s,
        accel_limit_mps2=requested_robot.accel_limit_mps2,
        brake_decel_mps2=requested_robot.brake_decel_mps2,
        speed_lag_tau_sec=requested_robot.speed_lag_tau_sec,
        collision_radius_m=requested_robot.collision_radius_m,
    )


def _safety_limits_from_profile(profile: Profile) -> SafetyLimits:
    """section P0-3: factored out of ``RealPolicyNode.__init__`` purely so
    the profile -> ``SafetyLimits`` wiring itself is directly unit-testable
    without constructing a live node (see
    tests/test_real_policy_node.py's
    ``test_safety_limits_wired_from_effective_profile``)."""
    return SafetyLimits(
        max_sensor_age_sec=profile.runtime.sensor_freshness_timeout_sec,
        max_odom_age_sec=profile.runtime.sensor_freshness_timeout_sec,
        max_command_age_sec=profile.runtime.watchdog_command_timeout_sec,
        min_obstacle_stop_distance_m=profile.risk.min_safe_clearance_m,
    )


def _twist_from(command) -> Twist:
    """Byte-identical to environment_node.py's own ``_twist_from`` --
    angular.z carries the STEERING ANGLE (radians), not a yaw rate; this is
    hunter_se_cmd_prefilter's own input convention, not a general ROS Twist
    convention (see that package's config, a READ-ONLY reference)."""
    msg = Twist()
    msg.linear.x = command.speed_mps
    msg.angular.z = command.steering_rad
    return msg


def _inference_worker_process_main(request_queue, response_queue,
                                    profile_dict: dict, checkpoint_dir: str, checkpoint_name: str) -> None:
    """section item-5: entry point for the SEPARATE OS process
    ``InferenceWorkerProcess`` spawns -- builds and loads its OWN agent
    instance (entirely independent of the parent process's, so the parent
    never shares live state with it) from the SAME profile/checkpoint the
    parent resolved, then serves inference requests off ``request_queue``
    until it receives the ``None`` shutdown sentinel.

    Must be a plain MODULE-LEVEL function, never a closure or bound method:
    multiprocessing's ``'spawn'`` start method (see ``InferenceWorkerProcess``
    for why ``'fork'`` is unsafe here) pickles the target callable by
    import path, which only works for something importable at module scope.

    Readiness (success OR failure) is signalled with a SINGLE mechanism --
    one ``("__ready__", None, None)`` / ``("__ready_error__", None, <repr>)``
    tuple put onto ``response_queue`` -- deliberately NOT a separate
    ``multiprocessing.Event`` alongside it: an earlier version used an
    ``Event`` for readiness and a best-effort non-blocking
    ``response_queue.get_nowait()`` right after to check for an error,
    which raced against ``Queue.put()``'s OWN internal feeder-thread
    (``put()`` returning in the child does not guarantee the item is
    already visible to a `get()` in the parent) -- confirmed LIVE via a
    genuinely flaky test failure (``DID NOT RAISE``) despite the child
    correctly hitting the error branch every time. A single blocking
    ``Queue.get(timeout=...)`` on the parent side has no such race by
    construction."""
    import numpy as _np

    from hunter_kinodynamic_rl.config.loader import profile_from_dict as _profile_from_dict
    from hunter_kinodynamic_rl.rl.algorithms.kinodynamic_tqc.agent import Agent as _RiskAgent
    from hunter_kinodynamic_rl.rl.algorithms.tqc.agent import Agent as _VanillaAgent
    from hunter_kinodynamic_rl.rl.checkpointing import manager as _ckpt_manager
    from hunter_kinodynamic_rl.trajectory.action_space import ACTION_DIM as _ACTION_DIM

    try:
        name = profile_dict["name"]
        sections_only = {k: v for k, v in profile_dict.items() if k != "name"}
        profile = _profile_from_dict(name, sections_only)
        history_len = profile.observation.frame_stack if profile.features.temporal_context else 1
        state_dim = profile.observation.lidar_bins * history_len + profile.observation.robot_state_dim
        if profile.features.risk_critic:
            agent = _RiskAgent(state_dim, _ACTION_DIM, 1.0, profile.hyperparameters,
                                profile.risk, profile.counterfactual)
        else:
            agent = _VanillaAgent(state_dim, _ACTION_DIM, 1.0, profile.hyperparameters)
        _ckpt_manager.load_generation(checkpoint_dir, checkpoint_name, agent.checkpoint_components())
    except Exception as e:  # noqa: BLE001 -- must report back, never leave the parent's ready-wait hanging
        response_queue.put(("__ready_error__", None, repr(e)))
        return

    response_queue.put(("__ready__", None, None))
    while True:
        item = request_queue.get()  # blocks -- this process has nothing else to do between requests
        if item is None:  # graceful-shutdown sentinel, see InferenceWorkerProcess.shutdown
            break
        request_id, observation = item
        try:
            # tests/test_real_policy_inference_worker.py: a narrow,
            # explicit, OFF-BY-DEFAULT test-only hook -- the only practical
            # way to make a genuinely SEPARATE, freshly-'spawn'-ed
            # interpreter hang predictably (a parent-process monkeypatch on
            # the agent class has no effect on a child that re-imports
            # every module fresh). Exercises InferenceWorkerProcess.restart's
            # real SIGTERM/SIGKILL-and-respawn path against an ACTUAL
            # wedged subprocess, never a mocked one.
            if os.environ.get("HKRL_TEST_FORCE_INFERENCE_HANG") == "1":
                time.sleep(3600.0)
            action = agent.select_action(_np.asarray(observation), deterministic=True)
            response_queue.put((request_id, _np.asarray(action), None))
        except Exception as e:  # noqa: BLE001 -- must report back, never crash the worker loop silently
            response_queue.put((request_id, None, repr(e)))


class InferenceWorkerProcess:
    """section item-5 (opt-in via ``runtime.inference_worker_mode='process'``,
    default remains ``'thread'``): runs policy inference in a SEPARATE OS
    process instead of an in-process worker thread, giving this node a
    REAL termination/restart capability a thread can never provide --
    CPython has no API to forcibly kill a running thread (see
    ``RealPolicyNode._infer_with_timeout``'s thread-mode docstring for that
    honest limitation), but an OS process CAN be terminated
    (SIGTERM, escalating to SIGKILL) and a fresh one respawned in its
    place -- see :meth:`restart`, called by ``RealPolicyNode`` after
    ``runtime.inference_worker_max_consecutive_timeouts`` consecutive
    timeouts.

    ``'spawn'`` start method, NEVER ``'fork'``: this process already has
    rclpy's own DDS threads/sockets/file-descriptors running by the time
    this worker is constructed -- ``fork()`` only duplicates the CALLING
    thread, so any lock another thread happened to hold at fork time stays
    held (and un-releasable) forever in the child, a well-known source of
    child-process deadlocks. ``'spawn'`` starts a genuinely fresh
    interpreter instead, at the cost of re-importing this module and
    reloading the checkpoint from scratch on every ``start``/``restart``.

    Single-flight BY CONSTRUCTION: :meth:`submit` raises if a request is
    already outstanding -- callers must check :attr:`busy` first, exactly
    mirroring the thread-mode path's own single-flight guard.
    """

    def __init__(self, profile: Profile, checkpoint_dir: str, checkpoint_name: str,
                 ready_timeout_sec: float = 60.0):
        self._profile_dict = dataclasses.asdict(profile)
        self._checkpoint_dir = checkpoint_dir
        self._checkpoint_name = checkpoint_name
        self._ready_timeout_sec = ready_timeout_sec
        self._ctx = mp.get_context("spawn")
        self._next_request_id = 0
        self._pending_request_id: Optional[int] = None
        self._process: Optional[mp.process.BaseProcess] = None
        self._request_queue = None
        self._response_queue = None
        self._start()

    def _start(self) -> None:
        self._request_queue = self._ctx.Queue()
        self._response_queue = self._ctx.Queue()
        self._process = self._ctx.Process(
            target=_inference_worker_process_main,
            args=(self._request_queue, self._response_queue,
                  self._profile_dict, self._checkpoint_dir, self._checkpoint_name),
            daemon=True, name="real_policy_inference_worker",
        )
        self._process.start()
        self._pending_request_id = None
        # section item-5: a single blocking get(timeout=...) on the SAME
        # queue the worker reports readiness through -- see
        # _inference_worker_process_main's own docstring for why this
        # replaced an earlier Event-based design that raced Queue.put()'s
        # internal feeder thread.
        try:
            sentinel = self._response_queue.get(timeout=self._ready_timeout_sec)
        except queue.Empty:
            self.shutdown(join_timeout_sec=1.0)
            raise RuntimeError(
                f"inference worker process did not become ready within {self._ready_timeout_sec:.1f}s "
                "(checkpoint load likely stuck) -- refusing to serve inference requests from it"
            )
        if sentinel[0] == "__ready_error__":
            error = sentinel[2]
            self.shutdown(join_timeout_sec=1.0)
            raise RuntimeError(f"inference worker process failed to initialize: {error}")

    @property
    def busy(self) -> bool:
        return self._pending_request_id is not None

    def submit(self, observation) -> None:
        if self.busy:
            raise RuntimeError(
                "InferenceWorkerProcess.submit called while a request is still outstanding "
                "(single-flight violation) -- caller must check `.busy` first"
            )
        self._next_request_id += 1
        self._pending_request_id = self._next_request_id
        self._request_queue.put((self._pending_request_id, np.asarray(observation)))

    def poll(self, timeout_sec: float):
        """Blocks up to ``timeout_sec`` for a response to the CURRENTLY
        outstanding request. Returns ``(action, error)``, both ``None`` if
        no matching response arrived in time (still busy -- caller must
        NOT submit again until this resolves, exactly mirroring the
        thread-mode single-flight contract). Silently discards (and keeps
        waiting out the remaining budget for) any response whose
        ``request_id`` doesn't match the current one -- a stale response
        left over from a request this worker was already :meth:`restart`
        -ed past."""
        if self._pending_request_id is None:
            return None, None
        deadline = time.monotonic() + timeout_sec
        while True:
            remaining = deadline - time.monotonic()
            try:
                if remaining <= 0:
                    # timeout_sec<=0 (or the budget has just run out): ONE
                    # non-blocking check rather than `get(timeout=<=0)`,
                    # whose semantics near/at zero are not guaranteed to
                    # even attempt a read.
                    request_id, action, error = self._response_queue.get_nowait()
                else:
                    request_id, action, error = self._response_queue.get(timeout=remaining)
            except queue.Empty:
                return None, None
            if request_id != self._pending_request_id:
                if remaining <= 0:
                    return None, None
                continue
            self._pending_request_id = None
            return action, error

    def restart(self) -> None:
        """Forcibly terminates the (possibly permanently hung) worker
        process and starts a fresh one in its place -- the real capability
        a thread-based worker structurally cannot provide."""
        self.shutdown(join_timeout_sec=1.0)
        self._start()

    def shutdown(self, join_timeout_sec: float = 2.0) -> None:
        process = self._process
        if process is None:
            return
        if process.is_alive():
            try:
                self._request_queue.put_nowait(None)  # graceful sentinel, best-effort
            except Exception:  # noqa: BLE001 -- queue may already be broken; escalate below regardless
                pass
            process.join(join_timeout_sec)
        if process.is_alive():
            process.terminate()
            process.join(join_timeout_sec)
        if process.is_alive():
            process.kill()
            process.join(join_timeout_sec)
        self._process = None
        self._pending_request_id = None


class RealPolicyNode(Node):
    def __init__(self):
        super().__init__("hunter_kinodynamic_real_policy")

        self.declare_parameter("profile", "real_hunter_safe")
        self.declare_parameter("checkpoint_dir", "")
        self.declare_parameter("checkpoint_name", "final")
        self.declare_parameter("goal_x", 0.0)
        self.declare_parameter("goal_y", 0.0)
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("odom_topic", "/odometry")
        self.declare_parameter("joint_states_topic", "/hunter_se/joint_states")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("estop_topic", "/hunter_kinodynamic_rl/real_policy/estop")
        self.declare_parameter("sensor_qos_reliability", "best_effort")
        # section P2-11: dry_run computes the FULL pipeline (observation,
        # inference, safety guard, diagnostics) every tick but never
        # publishes to cmd_vel_topic -- for validating the pipeline against
        # a live robot/Gazebo's sensors without ever being able to move it.
        self.declare_parameter("dry_run", False)
        # replay_mode: this node's ROS interface is plain topic subscriptions,
        # so it already works verbatim against pre-recorded data played back
        # externally via `ros2 bag play <bag> --clock` (see
        # launch/record_real_trial.launch.py for the topic list) -- no
        # separate bag-reading code needed here. What THIS flag adds is a
        # hard safety pairing: replaying recorded data must NEVER be able to
        # actuate a real robot, so replay_mode=true REQUIRES dry_run=true
        # (raises at startup if dry_run was left/set false) rather than
        # silently trusting the caller to have paired them correctly.
        self.declare_parameter("replay_mode", False)

        profile_name = self.get_parameter("profile").value
        self.profile = load_profile(profile_name)
        if self.profile.runtime.deployment != "real_hardware":
            raise SystemExit(
                f"profile {profile_name!r} has runtime.deployment={self.profile.runtime.deployment!r}, "
                "not 'real_hardware' -- refusing to run the real-robot inference node against a "
                "simulation-only profile (section P2)"
            )

        checkpoint_dir = self.get_parameter("checkpoint_dir").value
        if not checkpoint_dir:
            raise SystemExit("real_policy_node requires -p checkpoint_dir:=<run_dir>/checkpoints")
        checkpoint_name = self.get_parameter("checkpoint_name").value
        # section item-3: the generation-layout manifest lives at
        # <checkpoint_dir>/<checkpoint_name>/manifest.json (checkpoint_name
        # is a SYMLINK directory to .generations/<generation>/, not a flat
        # file -- see rl.checkpointing.manager's module docstring).
        manifest_path = os.path.join(checkpoint_dir, checkpoint_name, "manifest.json")
        if not os.path.isfile(manifest_path):
            raise SystemExit(f"missing checkpoint: {manifest_path}")
        with open(manifest_path) as f:
            manifest = json.load(f)
        # section P2/real-robot-safety: fail-fast architecture/action-space
        # compatibility check (see build_effective_profile's docstring) --
        # replaces the previous warn-and-continue-anyway check. self.profile
        # is REPLACED with the effective one (checkpoint's own restored
        # architecture + this requested profile's deployment-safety
        # overrides layered on top), so every downstream use of
        # self.profile (agent construction, action decoding, safety limits,
        # the control-loop timer period) is already consistent.
        self.profile = build_effective_profile(manifest, profile_name, self.profile)

        # section item-5: 'process' mode builds/loads the agent EXCLUSIVELY
        # inside InferenceWorkerProcess's own subprocess (its
        # _inference_worker_process_main does the exact same construction
        # + ckpt_manager.load_generation call, from the same profile/
        # checkpoint) -- constructing a SECOND copy here would double the
        # checkpoint load for no benefit, since 'process' mode never calls
        # self.agent.select_action at all (see _infer_with_timeout).
        self.agent = None
        self._inference_worker: Optional[InferenceWorkerProcess] = None
        history_len = self.profile.observation.frame_stack if self.profile.features.temporal_context else 1
        state_dim = self.profile.observation.lidar_bins * history_len + self.profile.observation.robot_state_dim
        if self.profile.runtime.inference_worker_mode == "process":
            self._inference_worker = InferenceWorkerProcess(self.profile, checkpoint_dir, checkpoint_name)
            self.get_logger().info(
                "loaded checkpoint into a separate inference worker PROCESS (runtime.inference_worker_mode="
                "'process') -- see InferenceWorkerProcess's docstring for the termination/restart guarantee "
                "this gives over the default thread-mode path"
            )
        else:
            if self.profile.features.risk_critic:
                self.agent = RiskAgent(state_dim, ACTION_DIM, 1.0, self.profile.hyperparameters,
                                        self.profile.risk, self.profile.counterfactual)
            else:
                self.agent = VanillaAgent(state_dim, ACTION_DIM, 1.0, self.profile.hyperparameters)
            result = ckpt_manager.load_generation(
                checkpoint_dir, checkpoint_name, self.agent.checkpoint_components())
            self.get_logger().info(
                f"loaded checkpoint components: {result['loaded']} (skipped: {result['skipped']})")

        self._frame_stack = FrameStack(self.profile.observation.lidar_bins, history_len)
        self._frame_stack_ready = False
        self._prev_action_01 = [0.0, 0.0, 0.0]

        self._latest_scan = None  # (ranges, angle_min, angle_increment)
        self._latest_scan_time = None
        self._latest_odom = None  # (x, y, yaw, v, yaw_rate)
        self._latest_odom_time = None
        self._latest_steering_rad = 0.0
        self._last_command_time = None

        self.goal_x = float(self.get_parameter("goal_x").value)
        self.goal_y = float(self.get_parameter("goal_y").value)
        # section P0-3: previously a bare default-constructed SafetyLimits()
        # -- e.g. real_hunter_safe.yaml's own risk.min_safe_clearance_m=0.5
        # (deliberately more conservative than the 0.3 m sim default) never
        # reached the guard at all; the LiDAR proximity-stop check silently
        # ran against SafetyLimits' hardcoded 0.3 m default instead. Wired
        # directly from self.profile (already the EFFECTIVE profile, i.e.
        # the checkpoint's restored architecture with this deployment
        # profile's safety overrides layered on top by build_effective_profile
        # above) so every one of these three numbers is the one this
        # specific deployment actually configured, not a generic default.
        self._safety_limits = _safety_limits_from_profile(self.profile)
        # section P0-3: tracks the last time a POLICY RESULT was actually
        # obtained (inference succeeded within budget, returned a
        # finite/correctly-shaped action) -- separate from
        # ``_last_command_time`` (updated on EVERY tick, including ones
        # that publish a safe stop because inference failed/timed out).
        # Collapsing these two into one clock would hide a systematically
        # failing policy behind an ever-fresh "command" timestamp (a stop
        # command is still a real, freshly-published command).
        self._last_successful_inference_time: Optional[float] = None
        self._inference_timeouts = 0
        self._inference_errors = 0
        # section item-5: single-flight state. Thread-mode: the live worker
        # Thread from the MOST RECENT call, so the NEXT tick can check
        # `.is_alive()` before ever starting another one (never launch a
        # second worker while the first is still running -- see
        # _infer_with_timeout's docstring for the unbounded-thread-growth
        # bug this closes). Process-mode: counts CONSECUTIVE timeouts so
        # a genuinely wedged worker process gets forcibly restarted after
        # runtime.inference_worker_max_consecutive_timeouts, rather than
        # waited on forever.
        self._inference_thread: Optional[threading.Thread] = None
        self._inference_consecutive_timeouts = 0

        self.replay_mode = bool(self.get_parameter("replay_mode").value)
        self.dry_run = bool(self.get_parameter("dry_run").value)
        if self.replay_mode and not self.dry_run:
            raise SystemExit(
                "-p replay_mode:=true requires -p dry_run:=true -- replaying recorded data must never "
                "be able to actuate a real robot (section P2-11). Pass both, or neither."
            )
        self._estopped = False

        qos_reliability_param = self.get_parameter("sensor_qos_reliability").value
        if qos_reliability_param not in ("best_effort", "reliable"):
            raise SystemExit(
                f"sensor_qos_reliability must be 'best_effort' or 'reliable', got {qos_reliability_param!r}"
            )
        reliability = (ReliabilityPolicy.BEST_EFFORT if qos_reliability_param == "best_effort"
                       else ReliabilityPolicy.RELIABLE)
        scan_qos = QoSProfile(depth=10, reliability=reliability)
        self.create_subscription(LaserScan, self.get_parameter("scan_topic").value, self._on_scan, scan_qos)
        self.create_subscription(Odometry, self.get_parameter("odom_topic").value, self._on_odom, scan_qos)
        self.create_subscription(JointState, self.get_parameter("joint_states_topic").value,
                                  self._on_joint_states, 10)
        # section P2-11: a software E-stop kill-switch -- once latched by a
        # True message, EVERY subsequent tick publishes STOP_COMMAND
        # (never the policy's output) until a False message clears it. This
        # is independent of and in ADDITION to action_guard.guard()'s own
        # per-tick sensor/command-freshness/collision-proximity checks --
        # an operator-triggered stop, not a sensor-derived one.
        self.create_subscription(Bool, self.get_parameter("estop_topic").value, self._on_estop, 10)
        self._cmd_pub = self.create_publisher(Twist, self.get_parameter("cmd_vel_topic").value, 10)
        # Diagnostics for real-trial analysis (docs/SIM2REAL.md's rosbag2
        # topic list explicitly calls out "policy action / goal /
        # collision-event topics", not just the final /cmd_vel -- without
        # this, a recorded bag has no record of what the POLICY chose
        # before the safety guard, or why it stopped) -- same message type
        # environment_node.py's own risk_telemetry uses, same package
        # convention. Layout: [raw_action(3), goal_x, goal_y,
        # nearest_obstacle_dist_m, emergency_stop(0/1)].
        self._diag_pub = self.create_publisher(
            Float32MultiArray, "/hunter_kinodynamic_rl/real_policy_diagnostics", 10)

        self._timer = self.create_timer(self.profile.runtime.time_delta_sec, self._on_control_tick)
        self.get_logger().info(
            f"hunter_kinodynamic_rl real_policy_node up (profile={profile_name}, "
            f"goal=({self.goal_x:.2f},{self.goal_y:.2f}), dry_run={self.dry_run}, replay_mode={self.replay_mode})"
        )

        # section P0-3: `rclpy.spin(node)` in `main()` below runs a plain
        # SingleThreadedExecutor -- an rclpy TIMER-based watchdog would sit
        # in the SAME callback queue as `_on_control_tick` and could never
        # fire while that callback is itself hung (e.g. a genuinely stuck
        # policy-inference call), which is exactly the failure mode this
        # watchdog exists to catch. A plain Python thread, entirely outside
        # rclpy's executor/callback-group machinery, keeps running
        # regardless of what the timer callback is doing. Started LAST
        # (after every attribute the watchdog loop reads/writes already
        # exists) so it can never race __init__ itself.
        self._start_watchdog_thread()

    def _start_watchdog_thread(self) -> None:
        self._watchdog_stop_event = threading.Event()
        self._last_policy_health_warning_time: Optional[float] = None

        def _loop() -> None:
            period = self.profile.runtime.real_policy_watchdog_period_sec
            timeout = self.profile.runtime.watchdog_command_timeout_sec
            health_timeout = self.profile.runtime.policy_health_timeout_sec
            while not self._watchdog_stop_event.wait(period):
                now = time.monotonic()
                if self._last_command_time is not None and (now - self._last_command_time) > timeout:
                    # Deliberately reuses `_publish` (never a raw
                    # `_cmd_pub.publish` bypass) so dry_run/replay_mode's
                    # "no actuation, ever" guarantee (section P2-11) still
                    # applies to this path too. Does NOT update
                    # `_last_command_time` itself -- mirrors
                    # environment_node.py's own `_on_watchdog_tick`: keeps
                    # re-publishing a stop every period until a genuinely
                    # fresh command arrives, rather than going quiet again
                    # after one shot.
                    self._publish(STOP_COMMAND)
                # section item-5: a SEPARATE health signal from the command-
                # freshness check above -- the robot is already guaranteed
                # safely stopped regardless (every tick publishes SOME
                # command, stop or real), but `_last_command_time` alone
                # cannot distinguish "policy healthy" from "policy has been
                # failing every single tick, node just keeps dutifully
                # publishing stops". Uses `_last_successful_inference_time`
                # specifically (see that attribute's own docstring) so a
                # systematically-failing policy is surfaced as its own,
                # loud, THROTTLED (at most once per health_timeout window)
                # diagnostic rather than silently hiding behind an
                # ever-fresh command timestamp.
                since_success = (
                    None if self._last_successful_inference_time is None
                    else now - self._last_successful_inference_time
                )
                if since_success is not None and since_success > health_timeout:
                    already_warned_recently = (
                        self._last_policy_health_warning_time is not None
                        and (now - self._last_policy_health_warning_time) < health_timeout
                    )
                    if not already_warned_recently:
                        self.get_logger().error(
                            f"[real_policy] no SUCCESSFUL policy inference in {since_success:.2f}s "
                            f"(> policy_health_timeout_sec={health_timeout:.2f}s) -- the robot is being "
                            "kept safely stopped by the per-tick timeout/error handling, but the POLICY "
                            "itself is not producing usable actions; investigate the inference worker"
                        )
                        self._last_policy_health_warning_time = now

        self._watchdog_thread = threading.Thread(target=_loop, name="real_policy_watchdog", daemon=True)
        self._watchdog_thread.start()

    def destroy_node(self) -> None:
        stop_event = getattr(self, "_watchdog_stop_event", None)
        if stop_event is not None:
            stop_event.set()
        inference_worker = getattr(self, "_inference_worker", None)
        if inference_worker is not None:
            inference_worker.shutdown()
        super().destroy_node()

    def _on_estop(self, msg: Bool) -> None:
        if bool(msg.data) and not self._estopped:
            self.get_logger().error("E-STOP engaged -- publishing zero command until cleared")
        elif not msg.data and self._estopped:
            self.get_logger().warn("E-STOP cleared")
        self._estopped = bool(msg.data)

    def _on_scan(self, msg: LaserScan) -> None:
        self._latest_scan = (np.asarray(msg.ranges, dtype=np.float32), msg.angle_min, msg.angle_increment)
        self._latest_scan_time = time.monotonic()

    def _on_odom(self, msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self._latest_odom = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw,
                              msg.twist.twist.linear.x, msg.twist.twist.angular.z)
        self._latest_odom_time = time.monotonic()

    def _on_joint_states(self, msg: JointState) -> None:
        """Mirrors environment_node.py's own steering computation exactly
        (Ackermann center steering = mean of the two front wheel angles)."""
        try:
            left = float(msg.position[msg.name.index("front_left_steering")])
            right = float(msg.position[msg.name.index("front_right_steering")])
        except (ValueError, IndexError, TypeError):
            return
        self._latest_steering_rad = 0.5 * (left + right)

    def _nearest_obstacle_dist_m(self) -> float:
        if self._latest_scan is None:
            return float("inf")
        ranges, angle_min, angle_increment = self._latest_scan
        _obs_state, environment_state = front_and_full_state(
            ranges, angle_min, angle_increment, self.profile.observation.lidar_bins,
            self.profile.observation.lidar_max_range_m, self.profile.observation.front_sector_width_rad,
        )
        # No world-boundary term here (unlike environment_node.py's sim-arena
        # collision check, env/simulation/environment_node.py) -- there is no
        # bounded virtual arena on real hardware / an open Gazebo world.
        return float(environment_state.min()) if environment_state.size else float("inf")

    def _publish(self, command) -> None:
        """The ONE call site that actually reaches cmd_vel_topic -- dry_run
        (and therefore replay_mode, which forces it) short-circuits here,
        never earlier, so every other tick of the pipeline (observation
        build, inference, safety guard, diagnostics) still runs exactly as
        it would live, for pipeline validation without being able to move
        a real robot (section P2-11)."""
        if not self.dry_run:
            self._cmd_pub.publish(_twist_from(command))

    def _infer_with_timeout(self, observation: np.ndarray) -> Optional[np.ndarray]:
        """section item-5: dispatches to the process-mode or thread-mode
        worker depending on ``runtime.inference_worker_mode``. Both paths
        share the same SINGLE-FLIGHT contract: if the PREVIOUS call has not
        finished yet, this method does NOT start a new one -- it counts
        another timeout and returns ``None`` immediately, so a genuinely
        sustained hang produces bounded resource growth (at most one live
        worker thread / one worker process, ever), never one new
        thread/request per control tick. Returns ``None`` on timeout OR any
        exception raised inside inference -- callers MUST treat ``None`` as
        "no usable action this tick", never a zero/default action."""
        if self._inference_worker is not None:
            return self._infer_with_timeout_process(observation)
        return self._infer_with_timeout_thread(observation)

    def _infer_with_timeout_thread(self, observation: np.ndarray) -> Optional[np.ndarray]:
        """Default (``runtime.inference_worker_mode='thread'``) path: runs
        ``self.agent.select_action`` on a worker thread and joins it with a
        bounded timeout (``runtime.policy_inference_timeout_sec``), instead
        of calling it directly on the control-tick callback.

        HONEST LIMITATION: CPython cannot forcibly terminate a running
        thread. If ``select_action`` is genuinely hung (not just slow),
        this does NOT kill it -- the worker thread leaks and keeps running
        in the background. What this DOES guarantee is that THIS control
        tick gives up waiting after the configured budget and returns
        control to the timer callback, so the node stays responsive (the
        independent watchdog thread keeps working, subsequent ticks keep
        running, dry_run/E-stop keep functioning) instead of the entire
        process wedging on one stuck inference call.

        section item-5 fix: previously started a BRAND NEW worker thread on
        EVERY call, with no check for whether a PRIOR call's thread was
        still alive -- on a genuinely sustained hang (not just one slow
        tick), every subsequent control tick (10 Hz by default) spawned
        ANOTHER thread that would never be joined, an unbounded
        thread-per-tick leak for as long as the hang lasted. Now checks
        ``self._inference_thread.is_alive()`` FIRST and, if the previous
        worker is still running, does not start a second one at all --
        just counts this tick as another timeout and publishes a safe
        stop, exactly like running out of budget on a freshly-started
        call. Real termination of a genuinely wedged call requires a
        SEPARATE PROCESS (see ``runtime.inference_worker_mode='process'``
        / ``InferenceWorkerProcess``), which this thread-mode path
        structurally cannot provide -- kept as the default because it is
        the more heavily field-verified path (no IPC/subprocess-lifecycle
        surface), with 'process' available as an explicit opt-in for
        deployments that need a real kill/restart guarantee."""
        if self._inference_thread is not None and self._inference_thread.is_alive():
            self._inference_timeouts += 1
            self._inference_consecutive_timeouts += 1
            self.get_logger().error(
                "[real_policy] previous policy inference call is STILL RUNNING past its own timeout budget "
                "(single-flight: NOT starting a second worker thread) -- publishing a safe stop for this tick"
            )
            return None

        result: dict = {}

        def _worker() -> None:
            try:
                result["action"] = self.agent.select_action(observation, deterministic=True)
            except Exception as e:  # noqa: BLE001 -- must never propagate into the timer callback
                result["error"] = e

        thread = threading.Thread(target=_worker, daemon=True)
        self._inference_thread = thread
        thread.start()
        thread.join(self.profile.runtime.policy_inference_timeout_sec)
        if thread.is_alive():
            self._inference_timeouts += 1
            self._inference_consecutive_timeouts += 1
            self.get_logger().error(
                f"[real_policy] policy inference exceeded "
                f"{self.profile.runtime.policy_inference_timeout_sec:.2f}s -- publishing a safe stop for "
                "this tick (the worker thread cannot be forcibly killed and may still be running; the "
                "next tick will not start a new one while this one is still alive -- see this method's "
                "own single-flight docstring)"
            )
            return None
        self._inference_thread = None
        if "error" in result:
            self._inference_errors += 1
            self._inference_consecutive_timeouts = 0
            self.get_logger().error(f"[real_policy] policy inference raised: {result['error']!r}")
            return None
        self._inference_consecutive_timeouts = 0
        return result.get("action")

    def _infer_with_timeout_process(self, observation: np.ndarray) -> Optional[np.ndarray]:
        """Opt-in (``runtime.inference_worker_mode='process'``) path: single-
        flight identically to the thread-mode path above, but backed by
        ``InferenceWorkerProcess`` -- a genuinely wedged call can be
        forcibly terminated (see :meth:`InferenceWorkerProcess.restart`),
        triggered here after ``runtime.inference_worker_max_consecutive_timeouts``
        CONSECUTIVE timeouts (never on the first one alone -- a single slow
        tick is not evidence of a genuine hang, and restarting reloads the
        checkpoint from scratch, itself a multi-second operation not worth
        paying for spuriously)."""
        worker = self._inference_worker

        def _maybe_restart_after_timeout() -> None:
            if (self._inference_consecutive_timeouts
                    >= self.profile.runtime.inference_worker_max_consecutive_timeouts):
                self.get_logger().error(
                    "[real_policy] inference worker process exceeded its consecutive-timeout budget -- "
                    "forcibly terminating and restarting it now"
                )
                worker.restart()
                self._inference_consecutive_timeouts = 0

        if worker.busy:
            # A PREVIOUS tick's request never came back within its own
            # budget -- single-flight: do NOT submit a second one for the
            # CURRENT observation. A quick, non-blocking check reaps the
            # response if it has since arrived (so the NEXT tick is free
            # to submit again), but its action/error is discarded either
            # way -- it was computed from an EARLIER observation, not this
            # tick's, so it must never be adopted as this tick's result.
            worker.poll(0.0)
            self._inference_timeouts += 1
            self._inference_consecutive_timeouts += 1
            self.get_logger().error(
                "[real_policy] previous policy inference request to the worker process is STILL "
                "outstanding (single-flight: not submitting a second one) -- publishing a safe stop "
                "for this tick"
            )
            _maybe_restart_after_timeout()
            return None

        worker.submit(observation)
        action, error = worker.poll(self.profile.runtime.policy_inference_timeout_sec)
        if worker.busy:  # still no response -- a real timeout on THIS request
            self._inference_timeouts += 1
            self._inference_consecutive_timeouts += 1
            self.get_logger().error(
                f"[real_policy] policy inference (process mode) exceeded "
                f"{self.profile.runtime.policy_inference_timeout_sec:.2f}s -- publishing a safe stop for "
                f"this tick ({self._inference_consecutive_timeouts}/"
                f"{self.profile.runtime.inference_worker_max_consecutive_timeouts} consecutive timeouts)"
            )
            _maybe_restart_after_timeout()
            return None
        if error is not None:
            self._inference_errors += 1
            self._inference_consecutive_timeouts = 0
            self.get_logger().error(f"[real_policy] policy inference (process mode) raised: {error}")
            return None
        self._inference_consecutive_timeouts = 0
        return action

    def _on_control_tick(self) -> None:
        now = time.monotonic()
        if self._estopped:
            self._publish(STOP_COMMAND)
            self._last_command_time = now
            return
        if self._latest_scan is None or self._latest_odom is None:
            self._publish(STOP_COMMAND)
            self._last_command_time = now
            return

        # section P0-3: the ENTIRE remaining pipeline (observation build,
        # inference, decode, guard) is wrapped -- an unexpected exception
        # ANYWHERE in it (a malformed observation, a decode error, a bug in
        # a future trajectory mode) must still result in a published safe
        # stop, never a silently-skipped tick that leaves the robot running
        # whatever command it last received. This is this node's OWN
        # guarantee, independent of (not a substitute for)
        # hunter_se_cmd_prefilter's downstream command-timeout watchdog --
        # see this module's docstring for the documented responsibility
        # boundary between the two.
        try:
            obs_cfg = self.profile.observation
            ranges, angle_min, angle_increment = self._latest_scan
            obs_state, _environment_state = front_and_full_state(
                ranges, angle_min, angle_increment, obs_cfg.lidar_bins,
                obs_cfg.lidar_max_range_m, obs_cfg.front_sector_width_rad,
            )
            if not self._frame_stack_ready:
                self._frame_stack.reset(obs_state)
                self._frame_stack_ready = True
            else:
                self._frame_stack.push(obs_state)
            lidar_frame = self._frame_stack.stacked()

            x, y, yaw, v, yaw_rate = self._latest_odom
            robot_state = RobotState(x=x, y=y, yaw=yaw, v=v, yaw_rate=yaw_rate, steering=self._latest_steering_rad)
            robot_state_vector = build_robot_state_vector(
                robot_state, self.goal_x, self.goal_y, self._prev_action_01,
                robot_state_dim=self.profile.observation.robot_state_dim,
            )
            observation = build_observation(lidar_frame, robot_state_vector)

            action = self._infer_with_timeout(observation)
            if action is None:
                self._publish(STOP_COMMAND)
                self._last_command_time = now
                return
            # section P0-3: reject a malformed action (wrong length, or any
            # non-finite element) BEFORE it ever reaches trajectory_executor
            # -- decoding a NaN/Inf or wrong-shaped action could otherwise
            # raise deep inside curvature/steering geometry math, or
            # silently propagate NaN through to the guard's own
            # `sanitize_command` (which only catches it on the FINAL
            # VehicleCommand, one layer too late to also protect
            # trajectory_executor.execute itself).
            action_arr = np.asarray(action, dtype=np.float64).reshape(-1)
            if action_arr.shape[0] != ACTION_DIM or not np.all(np.isfinite(action_arr)):
                self.get_logger().error(
                    f"[real_policy] invalid action from policy (shape={action_arr.shape}, "
                    f"finite={bool(np.all(np.isfinite(action_arr)))}) -- publishing a safe stop"
                )
                self._publish(STOP_COMMAND)
                self._last_command_time = now
                return
            self._last_successful_inference_time = now
            # environment_node.py stores the RAW normalized action directly as
            # its own prev_action ("_01" in build_robot_state_vector's parameter
            # name is just a naming artifact, not a value-range requirement --
            # see tests/test_env_modules.py, which passes negative values).
            self._prev_action_01 = [float(a) for a in action_arr]

            command = trajectory_executor.execute(
                action_arr, self.profile.action_space, self.profile.trajectory, self.profile.robot,
                dynamics_cfg=self.profile.dynamics if self.profile.features.trajectory_l_preview_blend else None,
                current_steering_rad=(
                    self._latest_steering_rad if self.profile.features.trajectory_l_preview_blend else None
                ),
            )
            nearest_obstacle_dist = self._nearest_obstacle_dist_m()
            safe_command = guard(
                command, self.profile.robot, self._safety_limits,
                last_sensor_time_sec=self._latest_scan_time, last_command_time_sec=self._last_command_time or now,
                now_sec=now, nearest_obstacle_distance_m=nearest_obstacle_dist,
                last_odom_time_sec=self._latest_odom_time,
            )
        except Exception as e:  # noqa: BLE001 -- last-resort fail-safe, see docstring above
            self.get_logger().error(f"[real_policy] control tick raised {e!r} -- publishing a safe stop")
            self._publish(STOP_COMMAND)
            self._last_command_time = now
            return

        self._publish(safe_command)
        self._last_command_time = now

        # Byte-identical formula to environment_node.py's own emergency_stop
        # derivation (env/simulation/environment_node.py): the guard forced
        # a stop that the policy itself did not request.
        emergency_stop = command.speed_mps > 1e-3 and safe_command.speed_mps <= 1e-6
        diag = Float32MultiArray()
        diag.data = [
            float(action_arr[0]), float(action_arr[1]), float(action_arr[2]),
            self.goal_x, self.goal_y, float(nearest_obstacle_dist), 1.0 if emergency_stop else 0.0,
        ]
        self._diag_pub.publish(diag)


def main():
    rclpy.init()
    node = RealPolicyNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        # A SIGINT during spin can already have triggered rclpy's own
        # shutdown before this finally block runs -- guard against calling
        # it twice ("rcl_shutdown already called").
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
