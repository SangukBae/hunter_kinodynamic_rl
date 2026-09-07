#!/usr/bin/env python3
"""Gazebo-Ignition-backed environment node -- the ``drl_agent_interfaces``
service contract (``/reset``, ``/step``, ``/get_dimensions``, ``/seed``)
over a REAL synchronized Gazebo simulation.

Every Gazebo interaction (world pause/reset/set_pose/spawn/delete) goes
through :class:`~hunter_kinodynamic_rl.env.simulation.gazebo_runtime.GazeboRuntimeMixin`,
which never blocks forever and never calls ``rclpy.spin_once``/
``spin_until_future_complete`` from inside a service callback (see that
module's docstring for why). ``main()`` MUST run this node under a
``MultiThreadedExecutor`` -- the bounded polling inside reset/step relies on
OTHER executor threads processing the scan/odom/joint_states subscription
callbacks and the pending Gazebo service futures while the callback thread
sleeps.
"""

from __future__ import annotations

import dataclasses
import math
import os
import time
from collections import deque
from typing import List, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy
from ros_gz_interfaces.srv import ControlWorld, DeleteEntity, SetEntityPose, SpawnEntity
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState, LaserScan
from std_msgs.msg import Float32MultiArray

from drl_agent_interfaces.srv import GetDimensions, Reset, Seed, Step

from hunter_kinodynamic_rl.common.geometry import goal_distance_and_heading
from hunter_kinodynamic_rl.config.loader import DEFAULT_RL_PROFILE, load_profile
from hunter_kinodynamic_rl.config.schema import Profile
from hunter_kinodynamic_rl.env.humans.dynamic_obstacle_motion import (
    KinematicMotionState, MotionPattern, RandomWaypointState, apply_pattern, assign_pattern, position_at_elapsed_time,
    motion_substep_durations, realized_velocity,
)
from hunter_kinodynamic_rl.env.observation.observation_builder import (
    RobotState, build_observation, build_robot_state_vector,
)
from hunter_kinodynamic_rl.env.randomization.domain_randomizer import (
    RandomizationDraw, apply_dynamics_overrides, apply_lidar_noise, apply_odometry_noise,
    apply_to_robot_config, check_sensor_overrides_supported, classify_draw_fields, command_latency_steps,
    rate_limit_speed, sample_draw, should_drop_sensor_frame, steering_lag_alpha,
)
from hunter_kinodynamic_rl.env.rewards.reward_calculator import compute_reward, is_goal_reached
from hunter_kinodynamic_rl.evaluation.contract_override import (
    apply_evaluation_contract, load_evaluation_contract_override,
)
from hunter_kinodynamic_rl.evaluation.fingerprint import (
    architecture_fingerprint, evaluation_contract_fingerprint, local_training_contract_fingerprint,
    training_profile_fingerprint,
)
from hunter_kinodynamic_rl.env.safety.action_guard import STOP_COMMAND, SafetyLimits, guard
from hunter_kinodynamic_rl.env.scenarios.benchmark_loader import load_scenario_file
from hunter_kinodynamic_rl.env.scenarios.footprint_geometry import (
    footprint_overlaps_obstacle, oriented_rectangle_boundary_clearance,
)
from hunter_kinodynamic_rl.env.scenarios.procedural_generator import (
    DynamicObstacleSpec, ScenarioSpec, generate_scenario,
)
from hunter_kinodynamic_rl.env.scenarios.seed_scheduler import SeedPoolViolation, SeedScheduler
from hunter_kinodynamic_rl.env.scenarios.tractor_environment_v2 import generate_v2_scenario
from hunter_kinodynamic_rl.env.simulation.gazebo_runtime import GazeboRuntimeMixin
from hunter_kinodynamic_rl.env.simulation.gazebo_service_wait import GazeboServiceError
from hunter_kinodynamic_rl.env.simulation import risk_telemetry as rt
from hunter_kinodynamic_rl.env.simulation import sensor_diagnostics as sd
from hunter_kinodynamic_rl.env.simulation import sensor_noise
from hunter_kinodynamic_rl.env.simulation.risk_computation import (
    compute_common_evaluation_metrics, compute_risk_telemetry, uses_common_evaluation_metrics,
)
from hunter_kinodynamic_rl.risk.boundary import distance_to_boundary_m
from hunter_kinodynamic_rl.rl.checkpointing.manager import sha256_of_file
from hunter_kinodynamic_rl.env.spawning import obstacle_pool, obstacle_spawner
from hunter_kinodynamic_rl.env.spawning.obstacle_catalog import load_catalog
from hunter_kinodynamic_rl.robot.hunter_se import HunterSE
from hunter_kinodynamic_rl.robot.limits import wheel_angles_to_center_steering
from hunter_kinodynamic_rl.sensing.scan_processor import front_and_full_state
from hunter_kinodynamic_rl.sensing.temporal_stack import FrameStack
from hunter_kinodynamic_rl.trajectory import trajectory_executor
from hunter_kinodynamic_rl.trajectory.action_space import TrajectoryCommand, action_dim_for_mode, decode_action
from hunter_kinodynamic_rl.trajectory.pure_pursuit_adapter import VehicleCommand

# section item-1 (round 3): the single, EXPLICIT, documented definition of
# which `RuntimeConfig` fields an evaluation-contract override CANNOT
# safely change on an already-running environment_node -- they describe a
# property of the ALREADY-RUNNING Gazebo world/process this node does not
# control (`gazebo_max_step_size_sec` == the connected world's own SDF
# `<max_step_size>`, baked in at Gazebo's OWN launch, entirely outside this
# ROS node's reach). `_resolve_evaluation_contract_override` FAILS FAST if
# an override's value differs from this node's own launch-time value for
# any field named here, rather than silently accepting an unverifiable
# mismatch into the resolved profile -- see that method's own docstring.
# Every OTHER `RuntimeConfig` field is either read fresh from
# `self.profile.runtime` on each use, or cached-and-re-derived by
# `_apply_runtime_cfg` (including a real ROS-timer recreation for
# `watchdog_period_sec`) -- genuinely, verifiably applied, not just
# accepted into an object nothing consults. Referenced directly by
# `docs/RESEARCH_PROTOCOL.md`'s own "Evaluation-contract delivery" section and by
# `tests/test_environment_node.py` -- keep all three in lock-step if this
# set ever changes.
#
# section item-1 (Gazebo physics-step reality-check fix): this check ALONE
# only ever compares two DECLARED config numbers against each other (an
# override's requested `gazebo_max_step_size_sec` vs. this process's own
# launch-time `gazebo_max_step_size_sec`) -- neither side is ever checked
# against the connected Gazebo world's REAL SDF `<max_step_size>`. That
# verification now happens separately, once, in
# `gazebo_runtime.py::GazeboRuntimeMixin.verify_physics_step_calibration`
# (a real `multi_step[1]` + `/clock` measurement), lazily on the first
# `propagate_state()` call under `runtime.deterministic_stepping` -- see
# that method's own docstring. Combined, the two checks give a genuinely
# verified chain: override value == launch-time declared value (this
# constant) == REAL Gazebo physics step (calibration), rather than only
# the first link. `physics_step_calibration_verified`/
# `physics_step_calibration_observed_dt_sec` (declared below, alongside
# `architecture_fingerprint_sha256`) expose calibration's own outcome for
# a remote client to record in run metadata -- deterministic evaluation
# never advances physics at all until this has genuinely passed (see
# `propagate_state`'s docstring); it is not merely assumed to have passed
# because the profile declares a value.
RUNTIME_FIELDS_FIXED_AT_LAUNCH = ("gazebo_max_step_size_sec",)


def _twist_from(command) -> Twist:
    msg = Twist()
    msg.linear.x = command.speed_mps
    msg.angular.z = command.steering_rad
    return msg


class KinodynamicEnvironmentNode(GazeboRuntimeMixin, Node):
    def __init__(self):
        super().__init__("hunter_kinodynamic_environment")

        self.declare_parameter("profile", DEFAULT_RL_PROFILE)
        self.declare_parameter("world_name", "default")
        self.declare_parameter("robot_entity_name", "hunter_se")
        self.declare_parameter("mode", "train")  # train | validation | test
        # The trainer temporarily borrows the same environment process for
        # held-out validation episodes.  `/seed` validates against this
        # explicit mode, which the client switches to `validation` only for
        # that bounded block and restores to `train` afterward.
        self.declare_parameter("explicit_seed_mode", "train")
        # v2 training curriculum position supplied by TrainerBase before an
        # explicit training seed. -1 means unset and is rejected for v2.
        self.declare_parameter("curriculum_episode_index", -1)
        self.declare_parameter("run_seed", 0)
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("odom_topic", "/odometry")
        self.declare_parameter("joint_states_topic", "/hunter_se/joint_states")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        # Set by evaluation_node.py before /reset to force an EXACT fixed
        # benchmark scenario instead of procedural generation (section 6/11).
        self.declare_parameter("scenario_override_path", "")
        # section item-1 (round 2): set by evaluation_node.py before /reset
        # to apply the REQUESTED evaluation profile's own reward/scenario/
        # runtime/evaluation sections to THIS live environment for the
        # duration of a fixed-benchmark run -- see
        # evaluation/contract_override.py's module docstring for why this
        # exists (this node's OWN launch-time profile's world boundary/
        # reward/timeout/physics-timing/common-metrics settings must NOT
        # silently keep being used just because this process happens to
        # have been launched under the checkpoint's training profile).
        self.declare_parameter("evaluation_contract_override_path", "")

        profile_name = self.get_parameter("profile").value
        self.profile: Profile = load_profile(profile_name)
        self._action_dim = action_dim_for_mode(self.profile.action_space)
        # section item-1 (round 2): the ORIGINAL, launch-time profile --
        # NEVER mutated after this line, unlike `self.profile` itself (which
        # `_resolve_evaluation_contract_override` below may replace's
        # reward/scenario/runtime/evaluation sections with an evaluation
        # profile's own). Restoring `evaluation_contract_override_path` to
        # `""` restores those 4 sections back to exactly this.
        self._launch_profile: Profile = self.profile
        # round 5 (live-E2E finding): `profile_name` (the raw CLI/ROS-param
        # string) and `self.profile.name` (what `load_profile` actually
        # RESOLVES it to) differ whenever this node is launched with an
        # explicit path (e.g. an ad-hoc verification profile living outside
        # config/profiles/, a documented `load_profile` mode -- see its own
        # docstring) rather than a short registered name: `load_profile`
        # strips any path down to `os.path.splitext(os.path.basename(...))[0]`
        # before storing it as `Profile.name`, and that STRIPPED form is
        # exactly what ends up in a checkpoint's own manifest `profile_name`
        # (computed the same way, by the trainer's own `load_profile` call).
        # A remote comparison against the raw `profile` parameter would
        # therefore spuriously mismatch for any path-launched profile even
        # when it is bit-for-bit the checkpoint's own training profile --
        # exposing the RESOLVED name here (mirroring
        # architecture_fingerprint_sha256's own "computed once, exposed for
        # remote verification" pattern) is what evaluation_node.py's
        # validate_live_environment now compares against instead.
        self.declare_parameter("resolved_profile_name", self.profile.name)
        # section item-2: computed ONCE, from whatever this process ACTUALLY
        # resolved at ITS OWN launch (never re-read later) -- exposed as a
        # read-only-in-practice ROS parameter so a remote client
        # (evaluation_node.py's EnvironmentClient.get_remote_parameter) can
        # verify this specific running instance's architecture matches a
        # checkpoint's own frozen resolved_config fingerprint, closing the
        # gap a bare profile-NAME comparison leaves open (the same-named
        # profile's YAML could have been edited between when a checkpoint
        # trained and when this process launched). ARCHITECTURE sections are
        # never affected by an evaluation-contract override (see above), so
        # this value is correctly launch-time-fixed and is deliberately
        # never re-set later, unlike evaluation_contract_fingerprint_sha256
        # below.
        self.declare_parameter("architecture_fingerprint_sha256", architecture_fingerprint(self.profile))
        # defect-fix item 3 (Local benchmark artifact integrity): the same
        # "computed once at launch, exposed for remote verification"
        # pattern as architecture_fingerprint_sha256 above, but for the
        # scenario/subgoal-DISTRIBUTION contract instead of network
        # architecture -- evaluation.local_subgoal_benchmark's
        # verify_environment_server_identity() reads this via
        # EnvironmentClient.get_remote_parameter() to refuse starting a
        # Local benchmark against a live server whose actually-loaded
        # scenario config (goal distance/direction/infeasible-fraction/
        # feasibility-check) doesn't match the profile the benchmark
        # script itself requested -- a same-NAMED profile whose YAML was
        # edited after this process launched would otherwise silently
        # benchmark against the wrong distribution.
        self.declare_parameter(
            "local_training_contract_fingerprint_sha256", local_training_contract_fingerprint(self.profile))
        # Formal training requires a stricter contract than architecture or
        # Local-subgoal compatibility alone: reward, runtime, randomization,
        # sensor-noise and the environment-owned episode length must also be
        # identical between the trainer process and this already-running
        # environment process.  Trainer-only cadence/budget fields are not
        # part of this hash.  This launch-time value never follows
        # evaluation-contract overrides.
        self.declare_parameter(
            "training_profile_fingerprint_sha256", training_profile_fingerprint(self._launch_profile))
        # section item-2 (round 2): unlike architecture_fingerprint_sha256,
        # this one DOES change -- it reflects whichever evaluation-contract
        # sections are ACTUALLY active right now (the launch-time profile's
        # own, or an applied override's), updated by
        # `_resolve_evaluation_contract_override` every time it changes. A
        # remote client (evaluation_node.py) reads this back AFTER applying
        # its own override to verify the live environment genuinely picked
        # it up, rather than trusting that setting the override parameter
        # successfully implies it was actually applied correctly.
        self.declare_parameter(
            "evaluation_contract_fingerprint_sha256", evaluation_contract_fingerprint(self.profile))
        # section item-1 (Gazebo physics-step reality-check fix): whether
        # GazeboRuntimeMixin.verify_physics_step_calibration has ACTUALLY
        # confirmed the connected Gazebo world's real physics step size
        # matches runtime.gazebo_max_step_size_sec -- False/NaN until a
        # deterministic-stepping episode's first propagate_state() call
        # runs it for real (see that method's own docstring). A remote
        # client (e.g. evaluation_node.py) reads these back to record, in
        # summary.json, whether physics was genuinely verified for a given
        # run rather than merely declared by the profile's own runtime
        # section -- exactly the gap RUNTIME_FIELDS_FIXED_AT_LAUNCH's
        # profile-vs-profile comparison alone left open (see that
        # constant's own docstring above).
        self.declare_parameter("physics_step_calibration_verified", False)
        self.declare_parameter("physics_step_calibration_observed_dt_sec", float("nan"))
        self.robot_model = HunterSE(self.profile.robot)
        self._active_robot_config = self.profile.robot

        self.world_name = self.get_parameter("world_name").value
        self.robot_entity_name = self.get_parameter("robot_entity_name").value
        self.mode = self.get_parameter("mode").value
        run_seed = int(self.get_parameter("run_seed").value)

        # section item-1 (round 2): factored into its own method (called
        # again by `_resolve_evaluation_contract_override` below) since
        # several runtime fields are CACHED into their own instance
        # attributes here rather than read fresh from `self.profile.runtime`
        # on every use (gazebo_runtime.py's hot-path timing fields) -- an
        # evaluation-contract override that replaces `self.profile.runtime`
        # wholesale would otherwise silently leave these stale.
        # section item-1 (round 3): tracked as a (path, content_sha256)
        # SIGNATURE, never just the path string -- see
        # `_resolve_evaluation_contract_override`'s own docstring for the
        # same-path-different-content caching bug this closes.
        self._active_evaluation_contract_signature: Optional[tuple] = None
        self._apply_runtime_cfg(self.profile.runtime)
        self._latest_sim_time_sec: Optional[float] = None
        # section item-1 (Gazebo physics-step reality-check fix): whether
        # verify_physics_step_calibration has ACTUALLY confirmed the
        # connected Gazebo world's real physics step against
        # runtime.gazebo_max_step_size_sec yet this process's lifetime --
        # see propagate_state's own docstring. Explicit (not a bare
        # getattr default) so it stays a real, inspectable False even if
        # the very first calibration attempt raises.
        self._physics_step_calibrated = False
        self._physics_step_calibration_observed_dt_sec: Optional[float] = None
        # section P0-8: anchors dynamic-obstacle position computation to
        # REAL elapsed sim time (see position_at_elapsed_time) instead of
        # iteratively accumulated per-tick dt -- set fresh every /reset.
        self._episode_start_sim_time_sec: Optional[float] = None
        # section P0-3: RandomWaypointState.tick() is the one motion
        # pattern that's inherently STATEFUL integration (not reconstructible
        # from spawn-time spec + elapsed time like the other four patterns),
        # so it needs the REAL per-tick sim-time delta, not the nominal
        # config time_delta_sec, to genuinely run "on a simulation-time
        # basis" under non-deterministic (wall-clock-sleep) stepping too --
        # tracks the previous tick's elapsed-since-episode-start reading.
        self._prev_dynamic_tick_elapsed_sec: Optional[float] = None

        self._seed_scheduler = SeedScheduler(run_seed, self.profile.scenario, self.mode)
        self._pending_explicit_seed: Optional[int] = None
        self._pending_curriculum_episode_index: Optional[int] = None

        self._scenario: Optional[ScenarioSpec] = None
        self._episode_seed: Optional[int] = None
        self._episode_step = 0
        self._step_id = 0
        # section item-1 (fixed-benchmark fairness): set from _on_reset's
        # own `is_fixed` every episode -- when True, _on_step computes
        # COMMON, architecture-independent evaluation metrics (see
        # risk_computation.compute_common_evaluation_metrics) instead of
        # the training-architecture-gated compute_risk_telemetry, so every
        # model benchmarked through the same fixed scenario set gets
        # comparable clearance/TTC/stopping-margin/unrecoverable readings
        # regardless of its own action_space.mode/features.risk_critic/
        # risk.enabled.
        self._is_fixed_benchmark = False
        # section P0-5: monotonic count of /reset ATTEMPTS since node
        # startup (incremented at the very TOP of _on_reset, before any
        # work happens, so it counts attempts including ones that later
        # fail/retry-exhaust) -- carried in risk_telemetry so a consumer
        # can detect a message left over from a PREVIOUS reset cycle even
        # when its step_id numerically coincides with the new episode's
        # (step_id alone resets to 0 every episode and can't disambiguate
        # this on its own). See risk_telemetry.py's module docstring.
        self._reset_generation = 0
        self._prev_action = [0.0] * self._action_dim
        self._prev_goal_distance = 0.0
        self._spawned_static_names: List[str] = []
        self._spawned_dynamic_names: List[str] = []
        self._dynamic_obstacles: List[dict] = []  # [{spec, pattern, waypoint_state}]

        self._catalog = load_catalog()
        self._obstacle_rng = np.random.RandomState(run_seed)
        # section (obstacle pool): built from the LAUNCH-time profile only
        # (self.profile here is still self._launch_profile -- no
        # evaluation-contract override has run yet), mirroring
        # RUNTIME_FIELDS_FIXED_AT_LAUNCH's own launch-time-fixed pattern --
        # see obstacle_pool.py's build_pool docstring for why parking
        # distance must not be re-derived from a later override. None when
        # disabled (every profile that doesn't set obstacle_pool.enabled),
        # in which case every obstacle-management code path below stays on
        # the legacy spawn/delete-every-reset behaviour, unchanged.
        self._obstacle_pool = (
            obstacle_pool.build_pool(
                self.profile.obstacle_pool, self.profile.scenario.world_size_m,
                self.profile.observation.lidar_max_range_m,
            ) if self.profile.obstacle_pool.enabled else None
        )

        self.scan_update_count = 0
        self.odom_update_count = 0
        self._latest_scan_ranges: Optional[np.ndarray] = None
        self._latest_scan_meta = (0.0, 0.01)  # (angle_min, angle_increment)
        self._latest_scan_time = 0.0
        # Real previous-command timestamp for the safety guard's command-
        # freshness watchdog -- MUST be the actual previous publish time,
        # never "now" (a "now vs now" comparison can never trip the
        # watchdog, which was found live to be a no-op check; see
        # env/safety/action_guard.py::check_command_freshness and
        # docs/IMPLEMENTATION_PLAN.md).
        self._last_command_time = time.monotonic()
        # Fixed-benchmark command_latency_sec override (section P1-3/P1-5):
        # 0 steps (the default for every profile that doesn't set this) means
        # "publish immediately", byte-identical to having no queue at all.
        self._command_delay_steps = 0
        self._command_delay_queue: deque = deque()
        self._robot_pose = (0.0, 0.0, 0.0)  # x, y, yaw
        self._robot_twist = (0.0, 0.0)      # v (signed forward speed), yaw_rate
        self._front_left_steering = 0.0
        self._front_right_steering = 0.0
        self._center_steering = 0.0
        # section P1-10: this episode's procedural domain-randomization
        # draw (RandomizationDraw() -- the "no randomization" identity --
        # whenever domain_randomization.enabled=False or a FIXED benchmark
        # scenario is active) plus a DEDICATED per-episode RNG stream for
        # the per-STEP noise samples it drives (LiDAR/odometry noise,
        # sensor-frame dropout) -- separate from sample_draw()'s own
        # one-shot per-episode PARAMETER draw, and re-seeded fresh every
        # /reset for reproducibility from the episode seed alone.
        self._active_domain_rand_draw = RandomizationDraw()
        self._domain_rand_step_rng: Optional[np.random.RandomState] = None
        # section (sensor noise): a DEDICATED per-episode RNG stream --
        # never scenario-generation's own rng or _domain_rand_step_rng
        # above -- reset fresh every /reset from the episode seed alone
        # (see sensor_noise.py's module docstring for the reproducibility
        # argument this gives across checkpoint/resume for free). None
        # whenever profile.sensor_noise.enabled is False (every profile
        # that doesn't opt in), in which case every call site below is a
        # byte-identical no-op.
        self._sensor_noise_state: Optional[sensor_noise.SensorNoiseState] = None
        self._filtered_steering_rad = 0.0
        self._filtered_speed_mps = 0.0
        self._last_obs_state: Optional[np.ndarray] = None
        # section (sensor diagnostics race fix): the ground-truth LiDAR
        # frame sampled at the SAME instant as `_last_obs_state` above --
        # see `_observation_obs_state`'s own docstring for why diagnostics
        # must read this instead of calling `_current_scan_states()` again.
        self._last_gt_obs_state: Optional[np.ndarray] = None

        self._frame_stack = FrameStack(
            frame_dim=self.profile.observation.lidar_bins,
            history_len=self.profile.observation.frame_stack if self.profile.features.temporal_context else 1,
        )

        cb_group = ReentrantCallbackGroup()
        # section item-1 (round 3): kept as an instance attribute so
        # `_apply_runtime_cfg` can safely RECREATE the watchdog timer below
        # on the SAME callback group whenever `watchdog_period_sec` changes
        # via an evaluation-contract override.
        self._reentrant_cb_group = cb_group
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(LaserScan, self.get_parameter("scan_topic").value, self._on_scan, qos,
                                  callback_group=cb_group)
        self.create_subscription(Odometry, self.get_parameter("odom_topic").value, self._on_odom, qos,
                                  callback_group=cb_group)
        self.create_subscription(JointState, self.get_parameter("joint_states_topic").value,
                                  self._on_joint_states, 10, callback_group=cb_group)
        # section P0-7: only consumed by multi_step_advance()'s confirmation
        # check when runtime.deterministic_stepping is enabled -- harmless,
        # negligible-overhead to always subscribe (mirrors trainer_base.py's
        # EnvironmentClient, which already does this unconditionally too).
        self.create_subscription(Clock, "/clock", self._on_clock, 10, callback_group=cb_group)

        self._cmd_pub = self.create_publisher(Twist, self.get_parameter("cmd_vel_topic").value, 10)
        self._risk_pub = self.create_publisher(Float32MultiArray, "/hunter_kinodynamic_rl/risk_telemetry", 10)
        # requirement 5: a SEPARATE, versioned diagnostics topic -- see
        # sensor_diagnostics.py's own module docstring for why this is
        # never folded into risk_telemetry's own wire format.
        self._sensor_diag_pub = self.create_publisher(
            Float32MultiArray, "/hunter_kinodynamic_rl/sensor_diagnostics", 10)

        # Gazebo service CLIENTS live in their OWN callback group, separate
        # from the service SERVERS below -- a /reset or /step server callback
        # blocks (time.sleep polling) waiting for a client future to
        # complete; if the client's response were processed in the SAME
        # MutuallyExclusiveCallbackGroup as the blocked server callback, the
        # executor could never run it (self-deadlock, confirmed live: pause_world
        # calls timed out every time until this split was added). Mirrors
        # drl_agent's environment.py `clients_callback_group` split exactly.
        clients_cb_group = MutuallyExclusiveCallbackGroup()
        self.world_control_client = self.create_client(
            ControlWorld, f"/world/{self.world_name}/control", callback_group=clients_cb_group)
        self.set_entity_pose_client = self.create_client(
            SetEntityPose, f"/world/{self.world_name}/set_pose", callback_group=clients_cb_group)
        self.spawn_entity_client = self.create_client(
            SpawnEntity, f"/world/{self.world_name}/create", callback_group=clients_cb_group)
        self.delete_entity_client = self.create_client(
            DeleteEntity, f"/world/{self.world_name}/remove", callback_group=clients_cb_group)

        services_cb_group = MutuallyExclusiveCallbackGroup()
        self.create_service(Reset, "reset", self._on_reset, callback_group=services_cb_group)
        self.create_service(Step, "step", self._on_step, callback_group=services_cb_group)
        self.create_service(GetDimensions, "get_dimensions", self._on_get_dimensions, callback_group=services_cb_group)
        self.create_service(Seed, "seed", self._on_seed, callback_group=services_cb_group)

        # Runs in the SAME reentrant group as the subscriptions (never the
        # services' mutually-exclusive group -- it must keep firing even
        # while a /reset or /step callback is mid-flight/blocked).
        # section item-1 (round 3): `self._watchdog_period_sec` was already
        # cached by `_apply_runtime_cfg` above -- used here (not
        # `self.profile.runtime.watchdog_period_sec` directly) so this
        # INITIAL creation and every later RE-creation (see
        # `_apply_runtime_cfg`'s own watchdog-timer-recreation branch) go
        # through the exact same value.
        self._watchdog_timer = self.create_timer(
            self._watchdog_period_sec, self._on_watchdog_tick, callback_group=cb_group)

        self.get_logger().info(
            f"hunter_kinodynamic_rl environment_node up (profile={profile_name}, "
            f"world={self.world_name}, mode={self.mode}, run_seed={run_seed})")

    # ------------------------------------------------------------ topics
    def _on_scan(self, msg: LaserScan) -> None:
        self._latest_scan_ranges = np.asarray(msg.ranges, dtype=np.float32)
        self._latest_scan_meta = (msg.angle_min, msg.angle_increment)
        self._latest_scan_time = time.monotonic()
        self.scan_update_count += 1

    def _on_odom(self, msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self._robot_pose = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw)
        self._robot_twist = (msg.twist.twist.linear.x, msg.twist.twist.angular.z)
        self.odom_update_count += 1

    def _on_joint_states(self, msg: JointState) -> None:
        """Realized front steering joint angles -> center steering via
        the exact inverse Ackermann geometry. NEVER left at
        a permanent 0 -- section 7's explicit requirement."""
        try:
            left_index = msg.name.index("front_left_steering")
            right_index = msg.name.index("front_right_steering")
            left = float(msg.position[left_index])
            right = float(msg.position[right_index])
        except (ValueError, IndexError, TypeError):
            return
        self._front_left_steering = left
        self._front_right_steering = right
        self._center_steering = wheel_angles_to_center_steering(
            left, right,
            self.profile.robot.wheelbase_m,
            self.profile.robot.track_width_m,
        )

    def _on_clock(self, msg: Clock) -> None:
        self._latest_sim_time_sec = msg.clock.sec + msg.clock.nanosec * 1e-9

    # -------------------------------------------------------- observation
    def _max_range(self) -> float:
        return self.profile.observation.lidar_max_range_m

    def _current_scan_states(self):
        obs_cfg = self.profile.observation
        if self._latest_scan_ranges is None:
            flat = np.full(obs_cfg.lidar_bins, self._max_range(), dtype=np.float32)
            return flat, flat
        angle_min, angle_increment = self._latest_scan_meta
        return front_and_full_state(
            self._latest_scan_ranges, angle_min, angle_increment,
            obs_cfg.lidar_bins, self._max_range(), obs_cfg.front_sector_width_rad,
        )

    def _observation_obs_state(self) -> np.ndarray:
        """The LiDAR frame the AGENT actually observes this tick -- with
        domain-randomization noise/dropout/frame-hold applied when active
        (section P1-10). NEVER used for collision/risk computation, which
        always calls ``_current_scan_states()`` directly for the untouched
        ground truth -- perturbing what the agent SEES must never perturb
        what actually happened.

        section (sensor diagnostics race fix): the ground-truth frame read
        below is also snapshotted into ``self._last_gt_obs_state`` -- the
        SAME instant's ground truth ``self._last_obs_state`` (the noisy
        result) was derived from. A multi-threaded executor can run the
        scan subscription callback between this call and a LATER,
        independent ``self._current_scan_states()`` call (e.g. from
        ``_compute_and_publish_sensor_diagnostics``), so re-querying ground
        truth there instead of reading this snapshot could compare THIS
        tick's noisy observation against a DIFFERENT (newer) scan --
        misattributing real robot/obstacle motion between the two scans to
        noise perturbation. Diagnostics must read this snapshot, never call
        ``_current_scan_states()`` again for the same tick."""
        obs_state, _ = self._current_scan_states()
        self._last_gt_obs_state = obs_state
        if self._domain_rand_step_rng is not None:
            if (self._last_obs_state is not None
                    and should_drop_sensor_frame(self._active_domain_rand_draw, self._domain_rand_step_rng)):
                obs_state = self._last_obs_state  # held frame: simulates a lost sensor message
            else:
                obs_state = apply_lidar_noise(
                    obs_state, self._active_domain_rand_draw, self._domain_rand_step_rng, self._max_range())
        # section (sensor noise): applied AFTER domain_randomization's own
        # LiDAR noise (composable, not exclusive -- see sensor_noise.py's
        # module docstring) -- a no-op whenever profile.sensor_noise.enabled
        # is False.
        if self._sensor_noise_state is not None:
            obs_state = sensor_noise.apply_lidar_noise(
                self._sensor_noise_state, self.profile.sensor_noise, obs_state, self._max_range())
        self._last_obs_state = obs_state
        return obs_state

    def _build_state_vector(self) -> np.ndarray:
        """Per-STEP contract: sample exactly ONE fresh (possibly noisy)
        LiDAR frame, push it onto the frame stack (evicting the oldest),
        and assemble the full observation from the result. Reset uses
        :meth:`_assemble_state_vector` directly instead (see that method's
        own docstring for why) -- never call this method from the reset
        path, or the initial LiDAR frame gets sampled/pushed TWICE (once
        here, once already by whatever filled the frame stack for reset)."""
        obs_state = self._observation_obs_state()
        self._frame_stack.push(obs_state)
        return self._assemble_state_vector()

    def _assemble_state_vector(self) -> np.ndarray:
        """Builds the full observation vector from whatever is ALREADY in
        ``self._frame_stack`` (never samples a new LiDAR frame or pushes
        one) -- the shared tail of both the per-step path
        (:meth:`_build_state_vector`, which pushes a fresh frame first) and
        the reset path (which instead calls ``self._frame_stack.reset(...)``
        with a SINGLE noisy frame sampled once, then calls this method
        directly so that frame is never re-sampled/re-pushed a second
        time -- see requirement 3 / this node's own ``reset`` docstring)."""
        lidar_frame = self._frame_stack.stacked()
        gt_x, gt_y, gt_yaw = self._robot_pose
        gt_v, gt_yaw_rate = self._robot_twist
        gt_steering = self._center_steering
        x, y, yaw, v, yaw_rate, steering = gt_x, gt_y, gt_yaw, gt_v, gt_yaw_rate, gt_steering
        if self._domain_rand_step_rng is not None:
            v, yaw_rate = apply_odometry_noise(v, yaw_rate, self._active_domain_rand_draw, self._domain_rand_step_rng)
        # section (sensor noise): perturbs ONLY these local variables, used
        # below solely to build the AGENT'S observation -- self._robot_pose/
        # self._robot_twist/self._center_steering (ground truth) are never
        # written to, so collision detection, reward, and the privileged
        # risk label (all of which read those attributes directly elsewhere
        # in this node) are completely unaffected regardless of whether
        # sensor_noise is enabled.
        if self._sensor_noise_state is not None:
            cfg = self.profile.sensor_noise
            x, y, yaw = sensor_noise.measured_pose(self._sensor_noise_state, cfg, x, y, yaw)
            v, yaw_rate = sensor_noise.measured_velocity(self._sensor_noise_state, cfg, v, yaw_rate)
            steering = sensor_noise.measured_steering(self._sensor_noise_state, cfg, steering)
        robot_state = RobotState(x=x, y=y, yaw=yaw, v=v, yaw_rate=yaw_rate, steering=steering)
        robot_state_vec = build_robot_state_vector(
            robot_state, self._scenario.goal_x, self._scenario.goal_y, self._prev_action,
            robot_state_dim=self.profile.observation.robot_state_dim,
        )
        self._compute_and_publish_sensor_diagnostics(
            gt_x=gt_x, gt_y=gt_y, gt_yaw=gt_yaw, gt_v=gt_v, gt_yaw_rate=gt_yaw_rate, gt_steering=gt_steering,
            noisy_x=x, noisy_y=y, noisy_yaw=yaw, noisy_v=v, noisy_yaw_rate=yaw_rate, noisy_steering=steering,
        )
        return build_observation(lidar_frame, robot_state_vec)

    def _compute_and_publish_sensor_diagnostics(
        self, gt_x: float, gt_y: float, gt_yaw: float, gt_v: float, gt_yaw_rate: float, gt_steering: float,
        noisy_x: float, noisy_y: float, noisy_yaw: float, noisy_v: float, noisy_yaw_rate: float,
        noisy_steering: float,
    ) -> None:
        """requirement 5: publishes ONE ``sensor_diagnostics.SensorDiagnostics``
        record for THIS tick (reset snapshot at step_id=0, or a real
        ``/step``) -- called from :meth:`_assemble_state_vector` alone, the
        one place ground truth and the agent's actual (possibly noisy)
        pose/velocity/steering are both already available in local
        variables (never re-reads ``self._robot_pose``/``self._robot_twist``,
        which could have moved on by the time a LATER call site read them).
        Mirrors ``_compute_and_publish_risk``'s own try/except-then-publish
        shape: a computation failure here must never crash the /reset or
        /step it's attached to, only degrade THIS diagnostics message to
        ``invalid(..., reason=COMPUTATION_EXCEPTION)``."""
        step_id = self._step_id
        reset_generation = self._reset_generation
        episode_id = self._episode_seed or 0
        sim_timestamp_sec = self._latest_sim_time_sec if self._latest_sim_time_sec is not None else float("nan")
        try:
            # section (sensor diagnostics race fix): NEVER call
            # self._current_scan_states() again here -- self._last_gt_obs_state
            # is the ground-truth frame _observation_obs_state() sampled at
            # the EXACT SAME instant self._last_obs_state (the noisy result)
            # was derived from, earlier this same tick. A fresh call here
            # could observe a DIFFERENT scan if the scan subscription's
            # callback ran (on another executor thread) between that call
            # and this one, which would misattribute real robot/obstacle
            # motion between the two scans to noise perturbation. Both are
            # guaranteed non-None by the time this runs (see
            # _assemble_state_vector's call site, always preceded by
            # _observation_obs_state() this same tick -- including at
            # reset, see KinodynamicEnvironmentNode's own reset docstring).
            ground_truth_obs_state = self._last_gt_obs_state
            noisy_obs_state = self._last_obs_state if self._last_obs_state is not None else ground_truth_obs_state
            dropout_count, perturb_mean, perturb_max = sd.lidar_perturbation_stats(
                ground_truth_obs_state.tolist(), noisy_obs_state.tolist(), self._max_range())
            noise_state = self._sensor_noise_state
            diagnostics = sd.SensorDiagnostics(
                step_id=step_id, valid=True, reset_generation=reset_generation, episode_id=episode_id,
                sim_timestamp_sec=sim_timestamp_sec,
                gt_x=gt_x, gt_y=gt_y, gt_yaw=gt_yaw, noisy_x=noisy_x, noisy_y=noisy_y, noisy_yaw=noisy_yaw,
                gt_v_mps=gt_v, gt_yaw_rate_radps=gt_yaw_rate, gt_steering_rad=gt_steering,
                noisy_v_mps=noisy_v, noisy_yaw_rate_radps=noisy_yaw_rate, noisy_steering_rad=noisy_steering,
                drift_x_m=noise_state.drift_x if noise_state is not None else 0.0,
                drift_y_m=noise_state.drift_y if noise_state is not None else 0.0,
                drift_yaw_rad=noise_state.drift_yaw if noise_state is not None else 0.0,
                localization_latency_steps=self.profile.sensor_noise.localization_latency_steps,
                lidar_beam_count=len(ground_truth_obs_state), lidar_dropout_count=dropout_count,
                lidar_perturbation_mean_m=perturb_mean, lidar_perturbation_max_m=perturb_max,
                invalid_reason=int(sd.DiagnosticsInvalidReason.NONE),
            )
        except Exception as e:
            self.get_logger().warn(f"[sensor_diagnostics] computation failed: {e}")
            diagnostics = sd.invalid(step_id, reset_generation, episode_id, sim_timestamp_sec,
                                      sd.DiagnosticsInvalidReason.COMPUTATION_EXCEPTION)
        self._sensor_diag_pub.publish(Float32MultiArray(data=sd.encode(diagnostics)))

    def _collision_threshold_m(self) -> float:
        return self._active_robot_config.collision_radius_m + self.profile.observation.collision_margin_m

    def _v2_footprint_collision(self) -> bool:
        """Shape-aware v2 termination check using the oriented Hunter body."""
        robot = self._active_robot_config
        # The attested collision circle is larger than the nominal 0.82 m
        # body length because the wheel/axle envelope extends longitudinally.
        # Expand the centered rectangle just enough to retain that attested
        # corner radius while preserving the measured width.
        half_width = 0.5 * robot.width_m
        half_length = max(
            0.5 * robot.length_m,
            math.sqrt(max(robot.collision_radius_m ** 2 - half_width ** 2, 0.0)),
        )
        length_m = 2.0 * half_length
        padding = self.profile.environment_v2.footprint_padding_m
        robot_xy = (self._robot_pose[0], self._robot_pose[1])
        if oriented_rectangle_boundary_clearance(
            robot_xy, self._robot_pose[2], length_m, robot.width_m,
            self.profile.scenario.world_size_m / 2.0, padding,
        ) <= 0.0:
            return True
        obstacles = list(self._scenario.static_obstacles) + [
            entry["spec"] for entry in self._dynamic_obstacles
        ]
        return any(
            footprint_overlaps_obstacle(
                robot_xy, self._robot_pose[2], length_m, robot.width_m, obstacle, padding,
            )
            for obstacle in obstacles
        )

    def _on_watchdog_tick(self) -> None:
        """Fires every ``watchdog_period_sec`` regardless of whether a
        /step call is in flight -- if no NEW command has been published
        within ``watchdog_command_timeout_sec``, publish an explicit zero
        command. Catches a hung/dead TRAINER PROCESS (no one calling /step
        at all), which the per-call safety guard inside _on_step cannot --
        that guard only runs when a call actually arrives."""
        if (time.monotonic() - self._last_command_time) > self.profile.runtime.watchdog_command_timeout_sec:
            self._cmd_pub.publish(Twist())

    def _publish_with_latency(self, command):
        """When ``self._command_delay_steps == 0`` (every profile that
        doesn't set a fixed benchmark's ``command_latency_sec`` override),
        this publishes immediately -- byte-identical to no queue existing
        at all. Otherwise pushes the NEW command and publishes whatever
        command is now exactly ``command_delay_steps`` ticks old, modeling
        a real, measurable command-latency delay (section P1-3/P1-5).

        Returns the command ACTUALLY published this tick (section P0-5):
        callers must use THIS, not the ``command`` argument, for anything
        claiming to describe the real physical command -- with a delay
        queue active the two differ (an older, already-queued command is
        what really reaches the robot, not the one just computed)."""
        if self._command_delay_steps <= 0:
            self._cmd_pub.publish(_twist_from(command))
            return command
        self._command_delay_queue.append(command)
        if len(self._command_delay_queue) > self._command_delay_steps:
            delayed = self._command_delay_queue.popleft()
        else:
            delayed = STOP_COMMAND  # queue not yet full -- hold at a safe stop
        self._cmd_pub.publish(_twist_from(delayed))
        return delayed

    # --------------------------------------------------------- obstacles
    def _clear_previous_obstacles(self) -> None:
        # section (obstacle pool): a pool-prefixed name is NEVER deleted --
        # it stays spawned for the lifetime of the process and is
        # re-teleported/parked by the upcoming _spawn_scenario_obstacles
        # call instead. Filtering by prefix (rather than tracking a
        # separate "did last episode use the pool" flag) is correct
        # regardless of whether the PREVIOUS and the UPCOMING episode agree
        # on pool usage (e.g. a fixed-benchmark episode, which always uses
        # the legacy path, following a pooled procedural one, or vice
        # versa) -- only genuinely legacy-spawned entities ever reach
        # delete_entities.
        legacy_names = [
            n for n in (self._spawned_static_names + self._spawned_dynamic_names)
            if not (n.startswith(obstacle_pool.STATIC_POOL_PREFIX) or n.startswith(obstacle_pool.DYNAMIC_POOL_PREFIX))
        ]
        obstacle_spawner.delete_entities(self, self.delete_entity_client, legacy_names)
        self._spawned_static_names = []
        self._spawned_dynamic_names = []
        self._dynamic_obstacles = []

    def _spawn_scenario_obstacles(self, scenario: ScenarioSpec, is_fixed: bool) -> None:
        # section (obstacle pool): only PROCEDURAL (non-fixed) episodes ever
        # use the pool -- a fixed benchmark always uses the legacy
        # spawn/delete path, so evaluation geometry is never approximated
        # by the pool's quantized static size classes (see
        # obstacle_pool.py's module docstring). use_pool=False also covers
        # "pool configured but this episode is a fixed benchmark" -- in
        # that case ensure_spawned still runs (harmless/idempotent if
        # already done) so any PREVIOUSLY active pool slot gets explicitly
        # parked below via activate_static([])/activate_dynamic([]),
        # rather than being left sitting at its last procedural episode's
        # position while a fixed benchmark runs.
        use_pool = self._obstacle_pool is not None
        if use_pool:
            obstacle_pool.ensure_spawned(self, self.spawn_entity_client, self._obstacle_pool)

        if use_pool and not is_fixed:
            self._spawned_static_names = obstacle_pool.activate_static(self, self._obstacle_pool,
                                                                        scenario.static_obstacles)
        elif use_pool:  # fixed benchmark while the pool is configured -- park every static slot
            self._spawned_static_names = obstacle_pool.activate_static(self, self._obstacle_pool, [])
        else:
            n_static = len(scenario.static_obstacles)
            self._spawned_static_names = [f"{obstacle_spawner.STATIC_ENTITY_PREFIX}{i}" for i in range(n_static)]
            obstacle_spawner.spawn_static_obstacles(
                self, self.spawn_entity_client, scenario.static_obstacles, self._catalog, self._obstacle_rng,
            )

        start_xy, goal_xy = (scenario.start_x, scenario.start_y), (scenario.goal_x, scenario.goal_y)
        self._dynamic_obstacles = []
        resolved_specs = []
        for i, spec in enumerate(scenario.dynamic_obstacles):
            # Fixed benchmark scenarios (section P1-3): NEVER overwrite the
            # YAML-authored (vx, vy) with a seed-derived random pattern --
            # every baseline must see the EXACT same obstacle motion. A
            # fixed scenario's own vx/vy are used as-is (CONSTANT_VELOCITY)
            # unless it explicitly names a `motion_pattern` to resolve
            # against ITS OWN start/goal geometry (still deterministic, no
            # seed-based randomness).
            if self.profile.environment_v2.enabled and spec.motion_pattern:
                # v2 specs are already path-conditioned by the generator;
                # resolving them a second time would destroy their TTC/DCPA.
                pattern = MotionPattern(spec.motion_pattern)
                resolved_spec = spec
            elif is_fixed:
                if spec.motion_pattern:
                    pattern = MotionPattern(spec.motion_pattern)
                    speed = math.hypot(spec.vx, spec.vy) or 0.4
                    resolved_spec = apply_pattern(spec, pattern, start_xy, goal_xy, speed, scenario.seed, i)
                else:
                    pattern = MotionPattern.CONSTANT_VELOCITY
                    resolved_spec = spec
            else:
                pattern = assign_pattern(scenario.seed, i)
                speed = math.hypot(spec.vx, spec.vy) or 0.4
                resolved_spec = apply_pattern(spec, pattern, start_xy, goal_xy, speed, scenario.seed, i)
            speed = math.hypot(resolved_spec.vx, resolved_spec.vy) or 0.4
            waypoint_state = None
            kinematic_state = None
            if self.profile.environment_v2.enabled:
                kinematic_state = KinematicMotionState.from_spec(
                    resolved_spec,
                    half_extent_m=self.profile.scenario.world_size_m / 2.0,
                    seed=(scenario.seed * 131 + i) & 0xFFFFFFFF,
                )
            elif pattern == MotionPattern.RANDOM_WAYPOINT:
                waypoint_state = RandomWaypointState(
                    x=resolved_spec.x0, y=resolved_spec.y0, speed_mps=speed,
                    half_extent_m=self.profile.scenario.world_size_m / 2.0,
                    seed=(scenario.seed * 131 + i) & 0xFFFFFFFF,
                )
                initial_vx, initial_vy = waypoint_state.current_velocity()
                resolved_spec = dataclasses.replace(resolved_spec, vx=initial_vx, vy=initial_vy)
            resolved_specs.append((resolved_spec, pattern, waypoint_state, kinematic_state))

        if use_pool and not is_fixed:
            self._spawned_dynamic_names = obstacle_pool.activate_dynamic(
                self, self._obstacle_pool, [r[0] for r in resolved_specs])
        elif use_pool:  # fixed benchmark while the pool is configured -- park every dynamic slot
            self._spawned_dynamic_names = obstacle_pool.activate_dynamic(self, self._obstacle_pool, [])
        else:
            self._spawned_dynamic_names = []
            for i, (resolved_spec, _pattern, _waypoint_state, _kinematic_state) in enumerate(resolved_specs):
                name = f"{obstacle_spawner.DYNAMIC_ENTITY_PREFIX}{i}"
                obstacle_spawner.spawn_dynamic_obstacle_marker(
                    self, self.spawn_entity_client, i, resolved_spec.radius, resolved_spec.x0, resolved_spec.y0,
                    shape=resolved_spec.shape, length_m=resolved_spec.length_m,
                    width_m=resolved_spec.width_m, yaw_rad=resolved_spec.yaw_rad,
                )
                self._spawned_dynamic_names.append(name)
        for resolved_spec, pattern, waypoint_state, kinematic_state in resolved_specs:
            # "spec0" is the IMMUTABLE spawn-time reference (never
            # reassigned) constant-velocity patterns anchor their
            # elapsed-time-based position to (section P0-8); "spec" is the
            # continuously-updated CURRENT position/velocity (still needed
            # by risk_computation.py's privileged snapshot and by
            # RANDOM_WAYPOINT's own stateful ticking).
            self._dynamic_obstacles.append({
                "spec": resolved_spec, "spec0": resolved_spec, "pattern": pattern,
                "waypoint": waypoint_state, "kinematic": kinematic_state,
            })

    def _tick_dynamic_obstacles(self, dt_sec: float) -> None:
        # section P0-8: elapsed_so_far is REAL (/clock-confirmed) sim time
        # since episode start, as of the START of this tick (physics for
        # THIS tick hasn't advanced yet) -- constant-velocity patterns
        # predict their end-of-tick position as elapsed_so_far + dt_sec
        # from their IMMUTABLE spawn-time spec, so only this one tick's
        # nominal-dt prediction is ever uncertain; every EARLIER tick's
        # contribution is exact, real elapsed time, never iteratively
        # accumulated/drifted. None (e.g. before /clock delivers its first
        # message) falls back to the old iterative x += vx*dt_sec path.
        elapsed_so_far = None
        if self._episode_start_sim_time_sec is not None and self._latest_sim_time_sec is not None:
            elapsed_so_far = self._latest_sim_time_sec - self._episode_start_sim_time_sec

        # section P0-3: RandomWaypointState.tick() needs the REAL per-tick
        # sim-time delta, not the nominal dt_sec, to genuinely run "on a
        # simulation-time basis" under the default (wall-clock-sleep)
        # stepping path -- unlike the constant-velocity patterns above,
        # its position is a STATEFUL integration (spawn spec + elapsed
        # time alone can't reconstruct it), so an inaccurate per-tick dt
        # would accumulate real drift, not just this one tick's
        # uncertainty. Falls back to the nominal dt_sec on the very first
        # tick of an episode (no previous reading yet) or whenever /clock
        # hasn't delivered a message, exactly like the fallback above.
        waypoint_dt_sec = dt_sec
        if elapsed_so_far is not None and self._prev_dynamic_tick_elapsed_sec is not None:
            waypoint_dt_sec = elapsed_so_far - self._prev_dynamic_tick_elapsed_sec
            if not math.isfinite(waypoint_dt_sec) or waypoint_dt_sec <= 0.0:
                waypoint_dt_sec = dt_sec
        if elapsed_so_far is not None:
            self._prev_dynamic_tick_elapsed_sec = elapsed_so_far

        for i, entry in enumerate(self._dynamic_obstacles):
            spec = entry["spec"]
            yaw = spec.yaw_rad
            if entry.get("kinematic") is not None:
                x, y, vx_realized, vy_realized, yaw = entry["kinematic"].tick(
                    dt_sec, ego_xy=(self._robot_pose[0], self._robot_pose[1]),
                )
                label_dt = dt_sec
            elif entry["pattern"] == MotionPattern.RANDOM_WAYPOINT and entry["waypoint"] is not None:
                x, y = entry["waypoint"].tick(waypoint_dt_sec)
            elif elapsed_so_far is not None:
                x, y = position_at_elapsed_time(entry["spec0"], elapsed_so_far + dt_sec)
            else:
                x, y = spec.x0 + spec.vx * dt_sec, spec.y0 + spec.vy * dt_sec
            if entry.get("kinematic") is None:
                label_dt = waypoint_dt_sec if entry["pattern"] == MotionPattern.RANDOM_WAYPOINT else dt_sec
                vx_realized, vy_realized = realized_velocity((spec.x0, spec.y0), (x, y), label_dt)
                if math.hypot(vx_realized, vy_realized) > 1e-9:
                    yaw = math.atan2(vy_realized, vx_realized)
            ax_realized = (vx_realized - spec.vx) / label_dt
            ay_realized = (vy_realized - spec.vy) / label_dt
            motion_time_sec = (
                entry["kinematic"].elapsed_sec if entry.get("kinematic") is not None
                else spec.motion_time_sec + label_dt
            )
            entry["spec"] = dataclasses.replace(
                spec, x0=x, y0=y, vx=vx_realized, vy=vy_realized,
                yaw_rad=yaw, accel_x_mps2=ax_realized, accel_y_mps2=ay_realized,
                motion_time_sec=motion_time_sec,
            )
            entry["realized_velocity_dt_sec"] = label_dt
            name = self._spawned_dynamic_names[i]
            try:
                self.set_entity_pose_ignition(
                    name, x, y, 0.0, 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0),
                )
            except GazeboServiceError as e:
                if self.profile.environment_v2.enabled:
                    raise RuntimeError(f"tractor_env_v2 failed to move {name}: {e}") from e
                self.get_logger().warn(f"[obstacle] failed to move {name}: {e}")

    # --------------------------------------------------------- risk telemetry
    def _compute_and_publish_risk(
        self, command: Optional[TrajectoryCommand], step_id: int,
        robot_pose, robot_v: float, robot_steering: float,
        static_obstacles, dynamic_specs,
        emergency_stop: bool, nominal_speed_mps: float, nominal_steering_rad: float,
        guarded_speed_mps: float, guarded_steering_rad: float,
        published_speed_mps: float, published_steering_rad: float, sensor_stale: bool = False,
        plant_limited: bool = False, guard_intervened: bool = False,
        goal_world_xy=None, reward_breakdown=None, snapshot_timestamp_sec: float = float("nan"),
    ) -> None:
        """Thin wrapper: all the actual math lives in the pure, ROS-free
        ``risk_computation.compute_risk_telemetry``/``compute_common_evaluation_metrics``
        (directly unit-tested -- see tests/test_risk_computation.py). The
        caller (``_on_step``) MUST pass a PRE-ACTION snapshot (state_t),
        never live ``self.*`` attributes that may already reflect state_t+1
        by the time this runs (section P0-2). ``command`` is ``None`` for a
        non-trajectory (``legacy_waypoint``) action -- irrelevant when
        ``self._is_fixed_benchmark`` (the common-metrics path uses the
        REALIZED published command instead, see below). ``emergency_stop``/
        ``plant_limited``/``guard_intervened``/``nominal_*``/``guarded_*``/
        ``published_*``/``sensor_stale`` are the REAL physical-command/sensor
        telemetry -- always attached regardless of whether the risk
        computation itself succeeds (section P1-4/P0-5/P0-9). ``nominal_*`` is
        trajectory_executor's raw, pre-guard output -- the SAME command
        ``command`` (the abstract [kappa,v_ref,L]) decodes to, and what the
        risk label above was itself assessed from (never the guarded/
        published command -- a guard-forced stop must never retroactively
        make a genuinely risky nominal action's label look safe)."""
        # section P0-5: always attached regardless of which branch below
        # runs -- reset_generation/episode_id/sim_timestamp_sec are pure
        # NODE state (compute_risk_telemetry is a pure function with no
        # access to self, so it can't set these itself).
        reset_generation = self._reset_generation
        episode_id = self._episode_seed or 0
        sim_timestamp_sec = self._latest_sim_time_sec if self._latest_sim_time_sec is not None else float("nan")
        privileged_obstacles = [
            rt.PrivilegedObstacleTelemetry(
                float(obstacle.x), float(obstacle.y), float(obstacle.radius), 0,
            ) for obstacle in static_obstacles
        ] + [
            rt.PrivilegedObstacleTelemetry(
                float(obstacle.x0), float(obstacle.y0), float(obstacle.radius), 1,
            ) for obstacle in dynamic_specs
        ]
        world_half_extent_m = self.profile.scenario.world_size_m / 2.0
        snapshot_values = (*robot_pose, snapshot_timestamp_sec, world_half_extent_m)
        privileged_snapshot_valid = all(math.isfinite(float(value)) for value in snapshot_values)
        privileged_snapshot_valid = privileged_snapshot_valid and all(
            math.isfinite(value)
            for obstacle in privileged_obstacles
            for value in (obstacle.x_world, obstacle.y_world, obstacle.radius)
        ) and all(obstacle.radius > 0.0 and obstacle.cause in (0, 1) for obstacle in privileged_obstacles)
        try:
            # The Stage-2 Local benchmark uses procedural scenarios from the
            # explicit TEST pool rather than fixed YAML scenarios. It still
            # needs the same realized-command risk yardstick for L0-L5;
            # otherwise rows without learned trajectory risk would report
            # invalid/zero high-risk outcomes and the comparison is unfair.
            if uses_common_evaluation_metrics(
                self._is_fixed_benchmark, str(self.get_parameter("explicit_seed_mode").value),
            ):
                # section item-1 (fixed-benchmark fairness): common,
                # architecture-independent metrics -- see
                # risk_computation.compute_common_evaluation_metrics's
                # docstring. Computed from the REALIZED (published) command,
                # so this works for EVERY action_space.mode/features
                # combination (including legacy_waypoint, which has no
                # rollout of its own), never gated by
                # profile.risk.enabled/features.ackermann_rollout -- takes
                # priority over the isinstance(command, TrajectoryCommand)
                # branch below whenever a fixed benchmark scenario is active.
                published_command_for_metrics = VehicleCommand(
                    speed_mps=published_speed_mps, steering_rad=published_steering_rad)
                telemetry = compute_common_evaluation_metrics(
                    published_command_for_metrics, step_id, robot_pose, robot_v, robot_steering,
                    static_obstacles, dynamic_specs, self._active_robot_config, self.profile,
                    goal_world_xy=goal_world_xy,
                )
                reason = rt.InvalidReason.NONE
            elif isinstance(command, TrajectoryCommand):
                telemetry = compute_risk_telemetry(
                    command, step_id, robot_pose, robot_v, robot_steering,
                    static_obstacles, dynamic_specs, self._active_robot_config, self.profile,
                    goal_world_xy=goal_world_xy,
                )
                # compute_risk_telemetry's OWN "features disabled" bail-out
                # (risk/ackermann_rollout off, or a non-trajectory action
                # mode) is the only way this call can legitimately return
                # valid=False without raising -- an exception (a DIFFERENT
                # invalid reason) is handled in the except block below.
                reason = rt.InvalidReason.NONE if telemetry.valid else rt.InvalidReason.FEATURES_DISABLED
            else:
                # Legacy waypoint action mode outside a fixed benchmark has
                # no risk framework at all -- the SAME category of "not
                # applicable" compute_risk_telemetry's own bail-out covers,
                # just reached via a different code path (no TrajectoryCommand
                # to even call it with).
                telemetry = rt.invalid(step_id)
                reason = rt.InvalidReason.FEATURES_DISABLED
            telemetry = dataclasses.replace(
                telemetry, emergency_stop=emergency_stop,
                nominal_speed_mps=nominal_speed_mps, nominal_steering_rad=nominal_steering_rad,
                guarded_speed_mps=guarded_speed_mps, guarded_steering_rad=guarded_steering_rad,
                published_speed_mps=published_speed_mps, published_steering_rad=published_steering_rad,
                sensor_stale=sensor_stale, reset_generation=reset_generation, episode_id=episode_id,
                sim_timestamp_sec=sim_timestamp_sec, invalid_reason=int(reason),
                plant_limited=plant_limited, guard_intervened=guard_intervened,
                goal_x=goal_world_xy[0] if goal_world_xy is not None else float("nan"),
                goal_y=goal_world_xy[1] if goal_world_xy is not None else float("nan"),
                reward_goal=reward_breakdown.goal if reward_breakdown is not None else 0.0,
                reward_collision=reward_breakdown.collision if reward_breakdown is not None else 0.0,
                reward_progress=reward_breakdown.progress if reward_breakdown is not None else 0.0,
                reward_step=reward_breakdown.step if reward_breakdown is not None else 0.0,
                reward_control_smoothness=(
                    reward_breakdown.control_smoothness if reward_breakdown is not None else 0.0),
                reward_trajectory_smoothness=(
                    reward_breakdown.trajectory_smoothness if reward_breakdown is not None else 0.0),
                privileged_snapshot_valid=privileged_snapshot_valid,
                snapshot_timestamp_sec=snapshot_timestamp_sec,
                ego_x_world=float(robot_pose[0]), ego_y_world=float(robot_pose[1]),
                ego_yaw_world=float(robot_pose[2]), world_half_extent_m=world_half_extent_m,
                privileged_obstacles=privileged_obstacles,
            )
        except Exception as e:
            self.get_logger().warn(f"[risk] computation failed: {e}")
            telemetry = rt.invalid(step_id, emergency_stop, nominal_speed_mps, nominal_steering_rad,
                                    guarded_speed_mps, guarded_steering_rad,
                                    published_speed_mps, published_steering_rad, sensor_stale,
                                    reset_generation, episode_id, sim_timestamp_sec,
                                    rt.InvalidReason.COMPUTATION_EXCEPTION,
                                    plant_limited, guard_intervened)
            telemetry = dataclasses.replace(
                telemetry,
                goal_x=goal_world_xy[0] if goal_world_xy is not None else float("nan"),
                goal_y=goal_world_xy[1] if goal_world_xy is not None else float("nan"),
                reward_goal=reward_breakdown.goal if reward_breakdown is not None else 0.0,
                reward_collision=reward_breakdown.collision if reward_breakdown is not None else 0.0,
                reward_progress=reward_breakdown.progress if reward_breakdown is not None else 0.0,
                reward_step=reward_breakdown.step if reward_breakdown is not None else 0.0,
                reward_control_smoothness=(
                    reward_breakdown.control_smoothness if reward_breakdown is not None else 0.0),
                reward_trajectory_smoothness=(
                    reward_breakdown.trajectory_smoothness if reward_breakdown is not None else 0.0),
                privileged_snapshot_valid=privileged_snapshot_valid,
                snapshot_timestamp_sec=snapshot_timestamp_sec,
                ego_x_world=float(robot_pose[0]), ego_y_world=float(robot_pose[1]),
                ego_yaw_world=float(robot_pose[2]), world_half_extent_m=world_half_extent_m,
                privileged_obstacles=privileged_obstacles,
            )

        self._risk_pub.publish(Float32MultiArray(data=rt.encode(telemetry)))

    # --------------------------------------------------------------- services
    def _on_seed(self, request, response):
        seed = int(request.seed)
        try:
            explicit_mode = str(self.get_parameter("explicit_seed_mode").value)
            SeedScheduler(0, self.profile.scenario, explicit_mode).validate_explicit_seed(seed)
        except SeedPoolViolation as e:
            self.get_logger().error(f"/seed rejected: {e}")
            response.success = False
            return response
        except ValueError as e:
            self.get_logger().error(f"/seed rejected: invalid explicit_seed_mode: {e}")
            response.success = False
            return response
        if self.profile.environment_v2.enabled and explicit_mode == "train":
            curriculum_index = int(self.get_parameter("curriculum_episode_index").value)
            if curriculum_index < 0:
                self.get_logger().error(
                    "/seed rejected: tractor_env_v2 training requires curriculum_episode_index >= 0 "
                    "to be set before every explicit seed"
                )
                response.success = False
                return response
            self._pending_curriculum_episode_index = curriculum_index
            self.set_parameters([Parameter(
                "curriculum_episode_index", Parameter.Type.INTEGER, -1,
            )])
        self._pending_explicit_seed = seed
        response.success = True
        return response

    def _apply_runtime_cfg(self, rt_cfg) -> None:
        """section item-1 (round 2/3): the ONE place ``self.profile.runtime``
        is read into per-attribute caches -- called from ``__init__`` AND
        from ``_resolve_evaluation_contract_override`` whenever
        ``self.profile.runtime`` changes, so an evaluation-contract
        override's ``time_delta_sec``/``deterministic_stepping``/
        ``gazebo_max_step_size_sec``/etc. actually take effect instead of
        silently keeping this node's OWN launch-time values.

        section item-1 (round 3): ``watchdog_period_sec`` is consumed
        EXACTLY ONCE, at ``rclpy.Timer`` construction time -- swapping the
        cached attribute alone (the pre-round-3 behavior) has ZERO effect
        on the period of an ALREADY-RUNNING timer, a genuine false
        guarantee (the fingerprint changes, nothing observable does).
        Fixed here by safely RECREATING the timer (destroy + create, same
        callback/group) whenever the resolved value actually changes --
        skipped on the very first call, from ``__init__``, before
        ``self._watchdog_timer``/``self._reentrant_cb_group`` exist yet
        (the timer's OWN initial construction, right after ``__init__``'s
        call to this method, uses ``self._watchdog_period_sec`` directly).

        code review (physics-step tolerance/contract bug): a completed
        ``verify_physics_step_calibration()`` is cached indefinitely
        (``self._physics_step_calibrated``) across an entire process's
        episodes -- correct when the calibration-relevant runtime fields
        never change again, but an evaluation-contract override applied by
        a LATER ``/reset`` (this method is also the one
        ``_resolve_evaluation_contract_override`` calls whenever it
        replaces ``self.profile.runtime``) could change
        ``physics_step_calibration_tolerance_sec``/``clock_confirm_timeout_sec``
        for a later episode while an EARLIER episode's now-stale
        calibration result stayed cached as "verified" -- a calibration
        that genuinely passed under one tolerance is not evidence it would
        still pass under a different, possibly stricter one. Detected here
        by comparing against the PREVIOUSLY cached value of each
        calibration-relevant field (never on the very first call, before
        ``self._physics_step_calibrated`` exists at all -- see ``__init__``,
        which sets it immediately after this method's own first call) and
        resetting the calibration cache whenever any of them actually
        changed, forcing :func:`GazeboRuntimeMixin.propagate_state` to
        re-run :func:`GazeboRuntimeMixin.verify_physics_step_calibration`
        for real before trusting deterministic stepping again.
        ``gazebo_max_step_size_sec`` is included defensively even though
        ``RUNTIME_FIELDS_FIXED_AT_LAUNCH`` already rejects any override that
        would change it -- this invalidation must stay correct even if that
        separate guarantee is ever relaxed."""
        calibration_fields_before = (
            getattr(self, "_gazebo_max_step_size_sec", None),
            getattr(self, "_physics_step_calibration_tolerance_sec", None),
            getattr(self, "_clock_confirm_timeout_sec", None),
        )
        self._gz_wait_timeout_sec = rt_cfg.gz_service_wait_timeout_sec
        self._gz_call_timeout_sec = rt_cfg.gz_service_call_timeout_sec
        self._gz_wait_poll_sec = rt_cfg.gz_service_poll_sec
        self.time_delta = rt_cfg.time_delta_sec
        # section P0-7: deterministic (multi_step + /clock-confirmed) vs the
        # legacy wall-clock time.sleep() physics advance -- see
        # gazebo_runtime.py::propagate_state's docstring.
        self._deterministic_stepping = rt_cfg.deterministic_stepping
        self._gazebo_max_step_size_sec = rt_cfg.gazebo_max_step_size_sec
        self._clock_confirm_timeout_sec = rt_cfg.clock_confirm_timeout_sec
        self._physics_step_tolerance_sec = rt_cfg.physics_step_tolerance_sec
        self._physics_step_calibration_tolerance_sec = rt_cfg.physics_step_calibration_tolerance_sec

        if hasattr(self, "_physics_step_calibrated"):
            calibration_fields_after = (
                self._gazebo_max_step_size_sec,
                self._physics_step_calibration_tolerance_sec,
                self._clock_confirm_timeout_sec,
            )
            if calibration_fields_after != calibration_fields_before:
                self._physics_step_calibrated = False
                self._physics_step_calibration_observed_dt_sec = None
                self._publish_physics_step_calibration_status(False, float("nan"))

        existing_timer = getattr(self, "_watchdog_timer", None)
        previous_period = getattr(self, "_watchdog_period_sec", None)
        if existing_timer is not None and rt_cfg.watchdog_period_sec != previous_period:
            self.destroy_timer(existing_timer)
            self._watchdog_timer = self.create_timer(
                rt_cfg.watchdog_period_sec, self._on_watchdog_tick, callback_group=self._reentrant_cb_group)
        self._watchdog_period_sec = rt_cfg.watchdog_period_sec

    def _resolve_evaluation_contract_override(self) -> None:
        """section item-1 (round 2/3): applies the REQUESTED evaluation
        profile's ``reward``/``scenario``/``runtime``/``evaluation``/
        ``sensor_noise`` sections to THIS live environment_node for the
        duration of a fixed-benchmark evaluation run -- set by
        evaluation_node.py (mirrors ``scenario_override_path``'s own
        file-based-override delivery pattern, see
        ``evaluation/contract_override.py``'s module docstring)
        BEFORE calling ``/reset``, so every algorithm type (SAC/vanilla-TQC/
        legacy-waypoint-TQC/risk-aware-TQC) evaluated on the same benchmark
        sees the IDENTICAL world boundary/reward/termination/runtime/
        common-metrics/sensor-noise contract, regardless of what its OWN
        training profile happened to use -- including whether that
        checkpoint was even trained with sensor_noise enabled at all.
        Cleared (empty string) restores this node's own LAUNCH-time
        profile's contract sections -- never silently keeps whatever the
        last override was.

        Deliberately narrow: ONLY `reward`/`scenario`/`runtime`/`evaluation`/
        `sensor_noise` are ever replaced (never `action_space`/`features`/
        `observation`/`robot`/`dynamics`/`risk`/`counterfactual`/
        `hyperparameters`/`algorithm`, which determine the checkpoint's own
        network shape/action-decode semantics and must stay exactly what
        this process
        launched under) -- a deliberately low-risk form of per-episode
        dynamic reconfiguration, not a full node reconstruction.

        section item-1 (round 3), two further fixes:

        1. **Content-hash staleness, not path-string staleness.** Tracking
           only ``override_path`` (the round-2 behavior) means a caller
           that reuses the SAME path across two DIFFERENT evaluation runs
           (e.g. re-running ``evaluation_node.py`` against the same
           ``checkpoint_dir``/benchmark) would silently keep the FIRST
           run's stale contract on this long-lived environment_node
           process -- the path string matches, so
           ``_resolve_evaluation_contract_override`` would return early
           without ever re-reading the file's (changed) content. Fixed by
           tracking ``(path, sha256_of_file(path))`` instead -- ANY content
           change, even at an identical path, is detected and re-applied.
        2. **``gazebo_max_step_size_sec`` fail-fast.** This value describes
           the physics step size of the ALREADY-RUNNING Gazebo world (its
           own SDF ``<max_step_size>``) -- this node cannot verify OR
           change it at runtime. An override that requests a DIFFERENT
           value than this node's own launch-time value is refused
           immediately (before ``self.profile`` is ever mutated) rather
           than silently accepted into the resolved runtime config, where
           it would corrupt ``multi_step_advance``'s physics-step-count
           math without any visible symptom other than wrong timing. See
           ``RUNTIME_FIELDS_FIXED_AT_LAUNCH`` (module-level) -- the single,
           documented, explicit definition of which runtime fields cannot
           be safely re-applied post-launch; kept in lock-step with
           anything referencing this contract (docs, tests)."""
        override_path = self.get_parameter("evaluation_contract_override_path").value
        if override_path:
            if not os.path.isfile(override_path):
                raise RuntimeError(
                    f"evaluation_contract_override_path={override_path!r} does not exist -- refusing to "
                    "silently keep whatever contract was previously active"
                )
            content_sha256 = sha256_of_file(override_path)
            signature = (override_path, content_sha256)
        else:
            signature = None
        if signature == self._active_evaluation_contract_signature:
            return  # content-identical to what's already active -- nothing to do

        if override_path:
            sections = load_evaluation_contract_override(override_path)
            candidate_profile = apply_evaluation_contract(self.profile, sections)
        else:
            candidate_profile = dataclasses.replace(
                self.profile,
                reward=self._launch_profile.reward, scenario=self._launch_profile.scenario,
                runtime=self._launch_profile.runtime, evaluation=self._launch_profile.evaluation,
                sensor_noise=self._launch_profile.sensor_noise,
                environment_v2=self._launch_profile.environment_v2,
            )
        candidate_profile.validate()
        for field_name in RUNTIME_FIELDS_FIXED_AT_LAUNCH:
            candidate_value = getattr(candidate_profile.runtime, field_name)
            launch_value = getattr(self._launch_profile.runtime, field_name)
            if candidate_value != launch_value:
                raise RuntimeError(
                    f"evaluation_contract_override_path={override_path!r} requests "
                    f"runtime.{field_name}={candidate_value!r}, but this environment_node was launched "
                    f"with runtime.{field_name}={launch_value!r} -- this field describes a property of the "
                    "ALREADY-RUNNING Gazebo world/process that this node cannot verify or change at "
                    "runtime (see RUNTIME_FIELDS_FIXED_AT_LAUNCH's own docstring). Refusing to apply an "
                    "override this node cannot actually honor: relaunch environment_node.py (and, if "
                    f"needed, Gazebo itself) with a profile whose runtime.{field_name} matches, or drop "
                    "this field from the evaluation profile's own runtime section."
                )

        # Only commit to self.profile / re-derive caches / republish the
        # fingerprint AFTER every check above has passed -- a rejected
        # override must leave the live environment's state byte-identical
        # to before this call, never partially applied.
        self.profile = candidate_profile
        self._apply_runtime_cfg(self.profile.runtime)
        self._active_evaluation_contract_signature = signature
        self.set_parameters([Parameter(
            "evaluation_contract_fingerprint_sha256", Parameter.Type.STRING,
            evaluation_contract_fingerprint(self.profile),
        )])

    def _resolve_episode_scenario(self):
        override_path = self.get_parameter("scenario_override_path").value
        if override_path:
            benchmark_scenario = load_scenario_file(override_path)
            return (benchmark_scenario.spec, benchmark_scenario.spec.seed, True,
                    benchmark_scenario.dynamics_overrides, benchmark_scenario.sensor_overrides)
        episode_index = self._seed_scheduler.episode_index
        episode_mode = self.mode
        if self._pending_explicit_seed is not None:
            seed = self._pending_explicit_seed
            self._pending_explicit_seed = None
            episode_mode = str(self.get_parameter("explicit_seed_mode").value)
            if episode_mode == "train" and self.profile.environment_v2.enabled:
                if self._pending_curriculum_episode_index is None:
                    raise RuntimeError(
                        "tractor_env_v2 explicit training seed has no pending curriculum episode index"
                    )
                episode_index = self._pending_curriculum_episode_index
                self._pending_curriculum_episode_index = None
        else:
            seed = self._seed_scheduler.next_seed()
        robot_cfg = self._active_robot_config
        max_curvature = robot_cfg.max_curvature
        # section (obstacle pool): quantize every static obstacle's radius
        # UP FRONT to a pool-compatible size class so the feasibility
        # checks inside generate_scenario and the geometry actually
        # spawned into Gazebo (obstacle_pool.activate_static, called from
        # _spawn_scenario_obstacles below) are always for the identical
        # radius -- None (pool disabled) keeps radii continuous, unchanged.
        static_radius_quantizer = None
        if self._obstacle_pool is not None:
            size_classes = self.profile.obstacle_pool.static_size_classes_m
            static_radius_quantizer = lambda r: obstacle_pool.snap_up_to_class(r, size_classes)  # noqa: E731
        if self.profile.environment_v2.enabled:
            scenario = generate_v2_scenario(
                seed, self.profile.scenario, self.profile.environment_v2, robot_cfg,
                self.profile.start_pose, self.profile.reward.goal_threshold_m,
                episode_index=episode_index, mode=episode_mode,
            )
        else:
            scenario = generate_scenario(
                seed, self.profile.scenario, robot_radius=robot_cfg.collision_radius_m,
                min_turning_radius_m=(1.0 / max_curvature) if max_curvature > 0.0 else None,
                wheelbase_m=robot_cfg.wheelbase_m,
                goal_radius_m=self.profile.reward.goal_threshold_m,
                start_pose_cfg=self.profile.start_pose,
                static_radius_quantizer=static_radius_quantizer,
            )
        return scenario, seed, False, {}, {}

    def _on_reset(self, request, response):
        self._reset_generation += 1  # counts ATTEMPTS -- see __init__'s comment
        # section item-1 (round 2): applied FIRST, before anything below
        # reads self.profile.scenario/reward/runtime -- see
        # _resolve_evaluation_contract_override's own docstring.
        self._resolve_evaluation_contract_override()
        self._cmd_pub.publish(Twist())
        self._last_command_time = time.monotonic()
        try:
            self.pause_world(True)
            self.reset_world()
        except GazeboServiceError as e:
            raise RuntimeError(f"reset failed: Gazebo world control error: {e}") from e

        try:
            self._clear_previous_obstacles()
        except GazeboServiceError as e:
            raise RuntimeError(f"reset failed: obstacle cleanup error: {e}") from e

        scenario, seed, is_fixed, dynamics_overrides, sensor_overrides = self._resolve_episode_scenario()
        self._scenario = scenario
        self._episode_seed = seed
        self._episode_step = 0
        self._step_id = 0
        self._is_fixed_benchmark = is_fixed
        self._prev_action = [0.0] * self._action_dim
        # Zero out motion/command state so a timed-out post-reset sensor wait
        # (wait_for_fresh_sensors below) falls back to a sane "stationary"
        # snapshot instead of silently carrying over the PREVIOUS episode's
        # final velocity/steering into this episode's first observation and
        # first risk label (section P1-5 "reset 시 velocity, steering, ...
        # 초기화"). Overwritten by real sensor data as soon as it arrives.
        self._robot_twist = (0.0, 0.0)
        self._front_left_steering = 0.0
        self._front_right_steering = 0.0
        self._center_steering = 0.0

        # section P1-10: reset to the "no randomization" identity every
        # episode -- only the domain_randomization.enabled branch below
        # (never the fixed-benchmark or disabled branches: TRAIN-ONLY,
        # confirmed by these being mutually exclusive branches of the SAME
        # if/elif/else, so a fixed evaluation scenario can never pick up a
        # procedural draw) overwrites these.
        self._active_domain_rand_draw = RandomizationDraw()
        self._domain_rand_step_rng = None
        self._filtered_steering_rad = 0.0
        self._filtered_speed_mps = 0.0
        self._last_obs_state = None
        self._last_gt_obs_state = None
        # section (sensor noise): reset fresh from the EPISODE seed alone,
        # unconditionally (unlike domain_randomization above, this is not
        # gated by is_fixed/domain_randomization.enabled -- it's an
        # orthogonal axis that may run during a fixed benchmark too, since
        # it never touches ground truth/common-metrics computation, see
        # sensor_noise.py's module docstring).
        self._sensor_noise_state = (
            sensor_noise.reset_state(seed, self.profile.sensor_noise)
            if self.profile.sensor_noise.enabled else None
        )

        if is_fixed:
            # Fixed benchmark: apply its OWN dynamics/sensor overrides
            # exactly (never the seed-based procedural domain-randomization
            # draw) -- unsupported keys raise rather than silently no-op
            # (section P1-3/P1-5).
            try:
                self._active_robot_config = apply_dynamics_overrides(self.profile.robot, dynamics_overrides)
                check_sensor_overrides_supported(sensor_overrides)
            except ValueError as e:
                raise RuntimeError(f"reset failed: unsupported benchmark override: {e}") from e
            self._command_delay_steps = command_latency_steps(dynamics_overrides, self.time_delta)
            # Preserve which fixed system overrides must activate the same
            # real command-path branches used by procedural randomization.
            # In particular, a scaled accel/brake config alone is inert
            # unless the speed rate limiter is selected below.
            self._active_domain_rand_draw = RandomizationDraw(
                mass_scale=float(dynamics_overrides.get("mass_scale", 1.0)),
                friction_scale=float(dynamics_overrides.get("friction_scale", 1.0)),
                wheel_radius_scale=float(dynamics_overrides.get("wheel_radius_scale", 1.0)),
                steering_gain=float(dynamics_overrides.get("steering_gain", 1.0)),
                command_latency_sec=float(dynamics_overrides.get("command_latency_sec", 0.0)),
            )
        elif self.profile.domain_randomization.enabled:
            draw = sample_draw(seed, self.profile.domain_randomization)
            self._active_robot_config = apply_to_robot_config(self.profile.robot, draw)
            self._active_domain_rand_draw = draw
            # Decorrelated from sample_draw()'s own use of `seed` directly,
            # so the per-step noise sequence isn't a trivial function of
            # the same draw (still fully deterministic given the episode
            # seed).
            self._domain_rand_step_rng = np.random.RandomState((seed * 7 + 1) & 0xFFFFFFFF)
            # Previously ALWAYS 0 here regardless of the draw -- a real bug
            # (section P1-10): command_latency_sec was sampled every
            # episode but silently discarded for the procedural-randomization
            # path (only fixed-benchmark overrides ever reached the queue).
            self._command_delay_steps = round(draw.command_latency_sec / max(self.time_delta, 1e-6))
        else:
            self._active_robot_config = self.profile.robot
            self._command_delay_steps = 0
        self._command_delay_queue.clear()

        # section P1-10 extension: log WHICH of this episode's randomized
        # axes actually reach the live Gazebo plant vs stay model/
        # observation-only (code review: no silent no-op randomization flag
        # -- a run's logs must make this boundary visible, not just
        # docs/TRACTOR_TQC_MODEL_SPEC.md). steering_gain is reclassified here (not by
        # classify_draw_fields itself, which has no FeatureFlags access) --
        # see domain_randomizer.py's GAZEBO_APPLIED_FIELDS docstring.
        draw_classification = None
        if self.profile.domain_randomization.enabled:
            draw_classification = dict(classify_draw_fields(self._active_domain_rand_draw))
            if self.profile.features.trajectory_l_preview_blend:
                draw_classification["steering_gain"] = "gazebo_applied"
        self.get_logger().info(
            f"[reset] reset_generation={self._reset_generation} episode seed={seed} fixed_benchmark={is_fixed} "
            f"static_obstacles={len(scenario.static_obstacles)} dynamic_obstacles={len(scenario.dynamic_obstacles)} "
            f"domain_rand_draw={self._active_domain_rand_draw if self.profile.domain_randomization.enabled else None} "
            f"domain_rand_classification={draw_classification} "
            f"start_pose_heading_mode={self.profile.start_pose.heading_mode} "
            f"heading_sample_attempts={scenario.heading_sample_attempts} "
            f"sensor_noise_enabled={self.profile.sensor_noise.enabled}")

        prev_scan_updates = self.scan_update_count
        prev_odom_updates = self.odom_update_count

        try:
            half_yaw = scenario.start_yaw / 2.0
            self.set_entity_pose_ignition(
                self.robot_entity_name, scenario.start_x, scenario.start_y, 0.0,
                0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw),
            )
            self._spawn_scenario_obstacles(scenario, is_fixed)
            if self._obstacle_pool is not None:
                active_static = sum(1 for s in self._obstacle_pool.static_slots if s.active)
                active_dynamic = sum(1 for s in self._obstacle_pool.dynamic_slots if s.active)
                self.get_logger().info(
                    f"[reset] obstacle_pool active_static={active_static}/{len(self._obstacle_pool.static_slots)} "
                    f"active_dynamic={active_dynamic}/{len(self._obstacle_pool.dynamic_slots)} "
                    f"parked_static={len(self._obstacle_pool.static_slots) - active_static} "
                    f"parked_dynamic={len(self._obstacle_pool.dynamic_slots) - active_dynamic}")
            self.propagate_state(self.profile.runtime.reset_settle_time_sec)
        except GazeboServiceError as e:
            raise RuntimeError(f"reset failed: Gazebo entity placement error: {e}") from e

        fresh = self.wait_for_fresh_sensors(prev_scan_updates, prev_odom_updates,
                                             timeout_sec=self.profile.runtime.sensor_freshness_timeout_sec)
        # section P0-9: a stale-sensor reset must never silently hand the
        # trainer an initial observation that isn't actually post-reset --
        # retry the physics settle a bounded number of times (transient
        # render/publish lag is common right after a teleport+spawn burst),
        # then FAIL LOUDLY (propagates through the same RuntimeError path
        # every other reset failure already uses) if it never recovers,
        # rather than the old behaviour of just logging a warning and
        # returning the stale state anyway.
        retries_used = 0
        max_retries = self.profile.runtime.sensor_freshness_max_reset_retries
        while not fresh and retries_used < max_retries:
            retries_used += 1
            self.get_logger().warn(
                f"[reset] sensors did not refresh within budget -- retrying settle "
                f"({retries_used}/{max_retries})")
            prev_scan_updates = self.scan_update_count
            prev_odom_updates = self.odom_update_count
            try:
                self.propagate_state(self.profile.runtime.reset_settle_time_sec)
            except GazeboServiceError as e:
                raise RuntimeError(f"reset failed: Gazebo physics retry error: {e}") from e
            fresh = self.wait_for_fresh_sensors(prev_scan_updates, prev_odom_updates,
                                                 timeout_sec=self.profile.runtime.sensor_freshness_timeout_sec)
        if not fresh:
            raise RuntimeError(
                f"reset failed: sensors did not refresh within budget after {1 + max_retries} attempts "
                "(section P0-9) -- refusing to return a stale initial observation to the trainer"
            )

        # section P0-2: residual momentum from the PREVIOUS episode can
        # carry into this "clean" reset -- confirmed live, SetEntityPose
        # only teleports POSE (ros_gz_interfaces/srv/SetEntityPose has no
        # twist field) and WorldReset.model_only=True does not reliably
        # zero reported velocity either (memory:
        # hunter_kinodynamic_rl_reset_velocity_carryover.md). Explicitly
        # WAIT for reported |v| to converge near zero -- publish a stop
        # command and let physics settle further, bounded retries, then
        # FAIL LOUDLY (matching the sensor-freshness retry pattern
        # immediately above) rather than silently handing the trainer a
        # first observation/risk-label that reflects nonzero carried-over
        # motion.
        velocity_threshold = self.profile.runtime.reset_velocity_threshold_mps
        velocity_retries_used = 0
        max_velocity_retries = self.profile.runtime.reset_velocity_max_retries
        while abs(self._robot_twist[0]) > velocity_threshold and velocity_retries_used < max_velocity_retries:
            velocity_retries_used += 1
            self.get_logger().warn(
                f"[reset] residual velocity {self._robot_twist[0]:.4f} m/s exceeds "
                f"{velocity_threshold:.4f} m/s threshold -- publishing stop and re-settling "
                f"({velocity_retries_used}/{max_velocity_retries})"
            )
            self._cmd_pub.publish(Twist())
            prev_scan_updates = self.scan_update_count
            prev_odom_updates = self.odom_update_count
            try:
                self.propagate_state(self.profile.runtime.reset_settle_time_sec)
            except GazeboServiceError as e:
                raise RuntimeError(f"reset failed: Gazebo velocity-settle retry error: {e}") from e
            self.wait_for_fresh_sensors(prev_scan_updates, prev_odom_updates,
                                         timeout_sec=self.profile.runtime.sensor_freshness_timeout_sec)
        if abs(self._robot_twist[0]) > velocity_threshold:
            raise RuntimeError(
                f"reset failed: residual velocity {self._robot_twist[0]:.4f} m/s still exceeds "
                f"{velocity_threshold:.4f} m/s after {1 + max_velocity_retries} settle attempts "
                "(section P0-2) -- refusing to start an episode with carried-over momentum"
            )

        # requirement 3: sample the initial (possibly noisy) LiDAR frame
        # EXACTLY ONCE and fill every slot of the frame stack with that SAME
        # frame (FrameStack.reset's own warm-start contract), then assemble
        # the state from what's already in the stack -- calling
        # _build_state_vector() here instead would sample AND push a
        # SECOND, DIFFERENT noisy frame on top, both double-consuming the
        # sensor_noise/domain_randomization RNG streams at reset and
        # breaking the warm-start invariant that every stacked frame starts
        # identical (the newest slot would silently diverge from the rest).
        self._frame_stack.reset(self._observation_obs_state())
        state = self._assemble_state_vector()
        self._prev_goal_distance, _ = goal_distance_and_heading(
            self._robot_pose[0], self._robot_pose[1], self._robot_pose[2], scenario.goal_x, scenario.goal_y,
        )
        # Re-stamp the command-freshness clock HERE, at the end of reset,
        # not just at line 418 above (section P0-3). The Gazebo round trip
        # above (pause/reset_world/spawn/settle/sensor-wait) routinely takes
        # longer than SafetyLimits.max_command_age_sec (0.5s default) --
        # confirmed live, a single world-service call alone can take up to
        # its own 3s bound. Leaving _last_command_time at its line-418
        # value would make _on_step's VERY FIRST guard() call see
        # `now - last_command_time` already exceeding the freshness
        # threshold and force an emergency stop on step 1 of EVERY episode,
        # even though no control loop actually stalled -- reset itself was
        # just doing legitimate, synchronous simulator housekeeping. The
        # line-418 stamp is still needed so the watchdog timer
        # (_on_watchdog_tick, which runs concurrently on another executor
        # thread throughout reset) doesn't misreport a stale command WHILE
        # reset is still in progress.
        self._last_command_time = time.monotonic()
        # section P0-8: anchor dynamic-obstacle position tracking to
        # sim time as of RIGHT NOW (episode start), not wall-clock -- None
        # if /clock hasn't delivered a message yet, handled gracefully by
        # _tick_dynamic_obstacles's fallback to the old iterative-dt path.
        self._episode_start_sim_time_sec = self._latest_sim_time_sec
        self._prev_dynamic_tick_elapsed_sec = None
        # section P1-5 (shared-interface preservation): publish a
        # RESET_MARKER telemetry message carrying THIS reset's authoritative
        # reset_generation -- the ONLY way a client learns the server's real
        # (process-lifetime-monotonic) counter value, since
        # ``drl_agent_interfaces/srv/Reset.srv`` is intentionally left
        # UNMODIFIED (see that file and trainer_base.EnvironmentClient.reset()
        # /_await_new_reset_marker for the full rationale, and
        # docs/IMPLEMENTATION_PLAN.md for the narrowed guarantee this implies: exactly
        # one client may call ``/reset`` on a given environment_node instance
        # at a time). This is hunter_kinodynamic_rl's OWN topic
        # (``/hunter_kinodynamic_rl/risk_telemetry``), not a shared
        # ``drl_agent_interfaces`` type, so broadcasting the generation here
        # -- instead of returning it in the Reset.srv response -- costs
        # nothing extra and keeps the shared service contract byte-identical
        # to drl_agent's own. Published on EVERY /reset unconditionally (not
        # gated by features.risk_critic / risk.enabled -- reset_generation
        # sync matters for staleness detection regardless of whether risk
        # labels are being consumed).
        self._risk_pub.publish(Float32MultiArray(data=rt.encode(rt.invalid(
            step_id=0, reset_generation=self._reset_generation, episode_id=self._episode_seed or 0,
            sim_timestamp_sec=self._latest_sim_time_sec if self._latest_sim_time_sec is not None else float("nan"),
            reason=rt.InvalidReason.RESET_MARKER,
        ))))
        response.state = state.tolist()
        return response

    def _on_step(self, request, response):
        action: List[float] = list(request.action)
        self._episode_step += 1
        self._step_id += 1
        step_id = self._step_id

        # PRE-ACTION snapshot (state_t) -- captured BEFORE the command is
        # published or physics advances. Used for (a) the risk label, which
        # must be risk(state_t, action_t, obstacles_t) not risk(state_t+1,
        # action_t, obstacles_t+1), and (b) the safety guard's collision-
        # proximity check, which must reflect what's known BEFORE this
        # command is applied, not the outcome AFTER it (section P0-2).
        pre_pose = self._robot_pose
        pre_sim_timestamp_sec = (
            self._latest_sim_time_sec if self._latest_sim_time_sec is not None else float("nan")
        )
        pre_v, _pre_yaw_rate = self._robot_twist
        pre_steering = self._center_steering
        pre_static_obstacles = list(self._scenario.static_obstacles)
        pre_dynamic_specs = [entry["spec"] for entry in self._dynamic_obstacles]
        _pre_obs_state, pre_environment_state = self._current_scan_states()
        pre_min_obstacle_dist = float(np.min(pre_environment_state)) if pre_environment_state.size else float("inf")
        # The world boundary (scenario.world_size_m) is a VIRTUAL limit --
        # no physical Gazebo wall geometry backs it (obstacle markers are
        # spawned into an otherwise open world), so LiDAR-based
        # pre_min_obstacle_dist above can never see it. Fold it in
        # explicitly so the safety guard also slows/stops approaching a
        # wall, not just approaching a spawned obstacle (section P0-2).
        pre_boundary_dist = distance_to_boundary_m(
            pre_pose[0], pre_pose[1], self.profile.scenario.world_size_m / 2.0)
        pre_min_obstacle_dist = min(pre_min_obstacle_dist, pre_boundary_dist)

        command = decode_action(action, self.profile.action_space, self._active_robot_config)
        vehicle_command = trajectory_executor.execute(
            action, self.profile.action_space, self.profile.trajectory, self._active_robot_config,
            dynamics_cfg=self.profile.dynamics if self.profile.features.trajectory_l_preview_blend else None,
            current_steering_rad=pre_steering if self.profile.features.trajectory_l_preview_blend else None,
        )
        safety_limits = SafetyLimits(
            max_sensor_age_sec=self.profile.runtime.sensor_freshness_timeout_sec,
            # Previously omitted -> silently fell back to SafetyLimits'
            # hardcoded 0.5s default, completely decoupled from
            # runtime.watchdog_command_timeout_sec (section P0-3: this
            # guard-side freshness check and the background watchdog timer
            # in _on_watchdog_tick both exist to catch the SAME failure mode
            # -- a stalled trainer process leaving a stale command active --
            # so they must share one config-driven threshold, not a real
            # profile-configured value silently doing nothing here).
            max_command_age_sec=self.profile.runtime.watchdog_command_timeout_sec,
            min_obstacle_stop_distance_m=self._collision_threshold_m(),
        )
        # section P1-10 extension (code review: "domain randomization must
        # reach Gazebo, not just the risk-rollout model" -- see
        # domain_randomizer.py's GAZEBO_APPLIED_FIELDS/rate_limit_speed
        # docstrings): friction_scale/velocity_response_scale fold into
        # self._active_robot_config.accel_limit_mps2/brake_decel_mps2, but
        # nothing previously read those fields on this real command-publish
        # path. Applied to the NOMINAL command BEFORE the safety guard (never
        # after -- a guard-forced emergency stop must stay IMMEDIATE, never
        # softened by a "realistic" ramp), so `guard()` can still override it
        # with zero regardless. A non-randomized draw (friction_scale ==
        # velocity_response_scale == 1.0, true for every profile that
        # doesn't enable domain_randomization, since sample_draw() then
        # returns the identity RandomizationDraw()) makes this branch a
        # byte-identical no-op -- `plant_command is vehicle_command` exactly.
        if self._active_domain_rand_draw.friction_scale != 1.0 or (
            self._active_domain_rand_draw.velocity_response_scale != 1.0
        ):
            self._filtered_speed_mps = rate_limit_speed(
                self._filtered_speed_mps, vehicle_command.speed_mps, self._active_robot_config, self.time_delta,
            )
            plant_command = VehicleCommand(
                speed_mps=self._filtered_speed_mps, steering_rad=vehicle_command.steering_rad,
            )
        else:
            self._filtered_speed_mps = vehicle_command.speed_mps
            plant_command = vehicle_command
        now = time.monotonic()
        safe_command = guard(
            plant_command, self._active_robot_config, safety_limits,
            last_sensor_time_sec=self._latest_scan_time, last_command_time_sec=self._last_command_time,
            now_sec=now, nearest_obstacle_distance_m=pre_min_obstacle_dist,
        )
        self._last_command_time = now
        # code review (emergency_stop metric pollution): captured HERE,
        # immediately after guard() returns and BEFORE the steering_delay
        # filter below reassigns safe_command again -- this is the ONE
        # unambiguous point that isolates the GUARD's own before/after
        # (plant_command -> safe_command), never conflated with the
        # steering-delay filter or the latency queue, both separate
        # mechanisms applied afterward.
        #
        # plant_limited: diagnostic-only -- did the domain-randomization
        # speed rate limiter actually change anything THIS step (see the
        # `plant_command` construction above)? On its own this is NEVER a
        # safety signal.
        #
        # guard_intervened: did the safety GUARD change EITHER speed or
        # steering (sanitize/freshness-stop/collision-proximity-stop/
        # clamp)? Strictly more general than emergency_stop.
        #
        # emergency_stop: the safety GUARD forced a genuinely-commanded
        # forward PLANT speed down to (near) zero -- e.g. collision-
        # proximity stop, stale-sensor/stale-command fallback, or NaN/Inf
        # sanitization (section P1-4: "near_collision_count를 emergency
        # stop으로 이름만 바꾸지 않는다"). Deliberately compares
        # plant_command (the guard's OWN input -- already reflects the
        # domain-randomization plant speed limiter, if any) vs
        # safe_command (the guard's OWN output), NEVER vehicle_command
        # (the RAW pre-plant-limiter nominal command): comparing against
        # vehicle_command was the actual bug this fixes -- a plant limiter
        # ramping UP from a stop on its own (e.g. right after an episode
        # reset, with a heavily-scaled-down velocity_response_scale draw),
        # with NO guard intervention at all, could make
        # `vehicle_command.speed_mps > 1e-3 and safe_command.speed_mps <=
        # 1e-6` misfire simply because the PLANT limiter itself hadn't
        # caught up yet -- a benign realism effect, not a safety
        # intervention. The latency queue delaying an otherwise-unguarded
        # command is ALSO a separate mechanism, not a guard activation, so
        # none of these three compare against published_command either.
        plant_limited = plant_command.speed_mps != vehicle_command.speed_mps
        guard_intervened = (
            safe_command.speed_mps != plant_command.speed_mps
            or safe_command.steering_rad != plant_command.steering_rad
        )
        emergency_stop = plant_command.speed_mps > 1e-3 and safe_command.speed_mps <= 1e-6
        # section P1-10: steering_delay_sec ("actuator gain/delay" domain
        # randomization -- steering_gain already scales the RATE LIMIT via
        # apply_to_robot_config; this separately models the ACTUATOR
        # taking time to track a newly commanded angle) applied as a
        # first-order lag filter on the REAL command sent to Gazebo, so it
        # affects actual simulated motion, not just the observation. A
        # zero draw (every profile that doesn't randomize this) makes
        # steering_lag_alpha return 1.0 -- byte-identical, no filtering.
        if self._active_domain_rand_draw.steering_delay_sec > 0.0:
            alpha = steering_lag_alpha(self._active_domain_rand_draw.steering_delay_sec, self.time_delta)
            self._filtered_steering_rad += alpha * (safe_command.steering_rad - self._filtered_steering_rad)
            safe_command = VehicleCommand(speed_mps=safe_command.speed_mps, steering_rad=self._filtered_steering_rad)
        # published_command is what ACTUALLY reached /cmd_vel this tick --
        # differs from safe_command whenever the command-latency queue is
        # active (section P0-5: telemetry must report this, not the
        # pre-latency "guarded" value it was silently mislabeling before).
        published_command = self._publish_with_latency(safe_command)

        prev_scan_updates = self.scan_update_count
        prev_odom_updates = self.odom_update_count
        # Move dynamic obstacles before each physics advance. v1 uses one
        # zero-order-hold update exactly as before; v2 interleaves configured
        # substeps so acceleration/turning obstacles do not teleport across
        # the whole control interval in a single jump.
        try:
            substeps = self.profile.environment_v2.motion_substeps if self.profile.environment_v2.enabled else 1
            for substep_dt in motion_substep_durations(self.time_delta, substeps):
                self._tick_dynamic_obstacles(substep_dt)
                self.propagate_state(substep_dt)
        except GazeboServiceError as e:
            raise RuntimeError(f"step failed: Gazebo physics advance error: {e}") from e
        sensors_fresh = self.wait_for_fresh_sensors(
            prev_scan_updates, prev_odom_updates, timeout_sec=self.profile.runtime.sensor_freshness_timeout_sec)
        sensor_stale = not sensors_fresh
        if sensor_stale:
            # section P0-9: NEVER silently store a stale post-action
            # observation as an ordinary transition -- the state/risk-label
            # this step produces may still reflect a PRE-action instant
            # (the scan/odom callbacks never delivered a fresher sample
            # within budget). Truncate the episode below (done=True) rather
            # than let training continue to accumulate MORE steps on top of
            # an uncertain observation.
            self.get_logger().warn(
                f"[step] sensors did not refresh within budget -- truncating episode "
                f"(step_id={step_id})")

        # Risk telemetry uses the PRE-ACTION snapshot captured above --
        # published HERE (after the sensor-freshness wait, not before) so
        # sensor_stale can be included in the SAME message, not a second
        # one (section P0-9). Its own VALUES never read live self.* state,
        # so moving the publish point later changes nothing else numerically.
        # section item-1: ALWAYS routed through _compute_and_publish_risk
        # now (never a separate bare rt.invalid publish for the legacy-
        # waypoint case) -- that method itself decides, internally, between
        # the common fixed-benchmark path, the trajectory-command path, and
        # the "not applicable" path (see its own branching).
        _obs_state, environment_state = self._current_scan_states()
        min_obstacle_dist = float(np.min(environment_state)) if environment_state.size else float("inf")
        # Same virtual-boundary rationale as pre_min_obstacle_dist above:
        # no physical Gazebo wall backs scenario.world_size_m, so it must be
        # folded into the POST-action collision determination explicitly,
        # or a policy could drive straight through it with no episode
        # termination at all (section P0-2).
        post_boundary_dist = distance_to_boundary_m(
            self._robot_pose[0], self._robot_pose[1], self.profile.scenario.world_size_m / 2.0)
        min_obstacle_dist = min(min_obstacle_dist, post_boundary_dist)
        if self.profile.environment_v2.enabled and self.profile.environment_v2.use_oriented_robot_footprint:
            collided = self._v2_footprint_collision()
        else:
            collided = min_obstacle_dist < self._collision_threshold_m()

        goal_distance, _heading_err = goal_distance_and_heading(
            self._robot_pose[0], self._robot_pose[1], self._robot_pose[2],
            self._scenario.goal_x, self._scenario.goal_y,
        )
        reached_goal = is_goal_reached(goal_distance, self.profile.reward)
        # section item-1 (round 2): a FIXED-benchmark episode's own timeout
        # budget is `evaluation.max_episode_steps` (the EVALUATION contract,
        # already applied to `self.profile` by
        # `_resolve_evaluation_contract_override` for the whole episode) --
        # NEVER `training.episode_length_steps` (this node's OWN
        # LAUNCH-time/checkpoint-training profile's budget), which would
        # silently give two checkpoints trained under different profiles
        # different server-side timeout budgets on "the same" benchmark,
        # even though `evaluation/benchmark_runner.py::run_episode`'s own
        # CLIENT-side loop bound already correctly uses
        # `evaluation.max_episode_steps`. Non-benchmark (procedural
        # train/validation/test) episodes are unaffected -- they keep using
        # `training.episode_length_steps` exactly as before.
        max_episode_steps = (
            self.profile.evaluation.max_episode_steps if self._is_fixed_benchmark
            else self.profile.training.episode_length_steps
        )
        timed_out = self._episode_step >= max_episode_steps

        reward = compute_reward(
            self.profile.reward, goal_distance, self._prev_goal_distance,
            collided, reached_goal, action, self._prev_action,
            trajectory_kappa=command.kappa if isinstance(command, TrajectoryCommand) else None,
            trajectory_horizon_m=command.horizon_m if isinstance(command, TrajectoryCommand) else None,
        )
        # Publish after the post-action outcome/reward has been computed so
        # the same step-synchronized side channel carries the complete
        # reward breakdown.  Risk itself still uses only the PRE-action
        # snapshot captured above.
        self._compute_and_publish_risk(
            command if isinstance(command, TrajectoryCommand) else None,
            step_id, pre_pose, pre_v, pre_steering, pre_static_obstacles, pre_dynamic_specs,
            emergency_stop, vehicle_command.speed_mps, vehicle_command.steering_rad,
            safe_command.speed_mps, safe_command.steering_rad,
            published_command.speed_mps, published_command.steering_rad, sensor_stale,
            plant_limited, guard_intervened,
            goal_world_xy=(self._scenario.goal_x, self._scenario.goal_y),
            reward_breakdown=reward,
            snapshot_timestamp_sec=pre_sim_timestamp_sec,
        )
        self._prev_goal_distance = goal_distance

        self._prev_action = action
        # section (sensor noise): drift is ticked ONCE per REAL /step (using
        # the real control-tick duration), never at /reset -- an episode's
        # localization drift starts at exactly zero (see _on_reset) and
        # only accumulates once physics has actually advanced.
        if self._sensor_noise_state is not None:
            sensor_noise.tick_drift(self._sensor_noise_state, self.profile.sensor_noise, self.time_delta)
        state = self._build_state_vector()
        response.state = state.tolist()
        response.reward = reward.total
        # section P0-9: a stale-sensor step ALSO truncates the episode --
        # never collision/target (neither is actually known to have
        # happened), just an infrastructure-driven end-of-episode so this
        # one uncertain transition can never be silently followed by MORE
        # steps built on top of it. risk_telemetry's sensor_stale flag
        # (published above) is what lets a reader distinguish this from a
        # genuine collision/goal/timeout.
        response.done = bool(reached_goal or collided or timed_out or sensor_stale)
        response.target = bool(reached_goal)
        response.collision = bool(collided)
        response.min_obstacle_dist_m = min_obstacle_dist
        return response

    def _on_get_dimensions(self, request, response):
        response.environment_dim = self.profile.observation.lidar_bins * self._frame_stack.history_len
        response.agent_dim = self.profile.observation.robot_state_dim
        response.state_dim = response.environment_dim + response.agent_dim
        response.action_dim = self._action_dim
        response.max_action = 1.0
        return response


def main(args=None):
    rclpy.init(args=args)
    node = KinodynamicEnvironmentNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        # A SIGINT during spin can already have triggered rclpy's own
        # shutdown before this finally block runs -- guard against calling
        # it twice ("rcl_shutdown already called").
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
