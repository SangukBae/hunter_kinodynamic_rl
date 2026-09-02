#!/usr/bin/env python3
"""Live-Gazebo ``LocalOptionExecutor`` (detailed spec "실제 hierarchical
학습 경로") -- the real counterpart to
``training.train_hierarchical_dqn.SimplifiedKinematicLocalExecutor``, used
by :class:`~hunter_kinodynamic_rl.training.train_hierarchical_dqn.HierarchicalTrainingLoop`
and ``evaluation/long_horizon_benchmark.py`` whenever they are configured to
train/evaluate against a real frozen Local kinodynamic TQC checkpoint over
live Gazebo instead of the pure-Python point-robot stand-in.

Structurally satisfies BOTH things ``HierarchicalTrainingLoop.run_mission``
actually needs from its ``executor`` local variable:

  - the ``LocalOptionExecutor`` protocol (``run_option(coordinator,
    partial_map, rng, max_local_steps) -> None``), driving
    ``coordinator.record_local_tick(...)`` every control tick exactly like
    ``SimplifiedKinematicLocalExecutor`` does, so ``HierarchyCoordinator``'s
    OWN reached/blocked/timeout/no-progress/high-risk decision logic (Phase
    1/2, already live-verified) is reused verbatim -- this module never
    reimplements subgoal termination logic.
  - a ``pose_world`` property (``run_mission`` reads ``executor.pose_world``
    BETWEEN options too, not just inside ``run_option``), reflecting the
    robot's real, live odometry pose.

Reuses (never reimplements):

  - ``env.simulation.gazebo_runtime.GazeboRuntimeMixin`` for
    pause/reset/set-pose/sensor-freshness-wait -- the SAME hang-avoidance
    discipline (bounded ``time.sleep()`` polling, never a bare
    ``spin_once`` inside a callback) ``environment_node.py`` already relies
    on. This node is NOT itself a ROS service server (no ``/reset``/``/step``
    of its own -- ``reset_episode``/``run_option`` are plain Python methods
    called directly by the training loop's own thread), so the mixin's
    ``time.sleep()``-based waits need SOMETHING ELSE pumping this node's
    callbacks concurrently: a dedicated ``MultiThreadedExecutor`` spun on a
    background daemon thread, started once in ``__init__`` (mirrors
    ``environment_node.py``'s own executor model, just with "a service
    worker thread" replaced by "this node's one background spin thread").
  - ``env.spawning.wall_segment_spawner`` (``build_wall_pool``/
    ``ensure_spawned``/``activate_walls``) for the Phase 3 long-horizon wall
    geometry -- spawned once, repositioned every episode, unused slots
    parked, exactly like ``obstacle_pool.py``'s own design.
  - ``navigation.local_rl.controller.LocalPolicyController`` for
    observation-build/action-decode/safety-guard -- the SAME pure-Python
    contract ``real_policy_node.py``/``hierarchical_environment_node.py``
    already drive, so the active subgoal fed in here is STRUCTURALLY the
    same "never a final goal" contract those modules already guarantee.
  - ``rl.checkpointing.manager.load_generation`` for the frozen checkpoint --
    missing/corrupt/incompatible (dimension or architecture-fingerprint
    mismatch) checkpoints raise immediately in ``__init__``, never silently
    substituting a heuristic or an untrained network (detailed spec:
    "missing/incompatible Local checkpoint ... heuristic이나 untrained
    model로 조용히 대체하지 말고 fail-fast").

Caller contract: exactly one ``rclpy.init()`` must have already happened
before construction, and ``rclpy.shutdown()`` after ``close()`` -- mirrors
every other plain ``Node`` subclass in this package (``EnvironmentClient``
in ``training/trainer_base.py``).
"""

from __future__ import annotations

import math
import threading
import time
from typing import Callable, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from ros_gz_interfaces.msg import Contacts
from ros_gz_interfaces.srv import DeleteEntity, SpawnEntity
from sensor_msgs.msg import JointState, LaserScan

from hunter_kinodynamic_rl.config.loader import load_profile
from hunter_kinodynamic_rl.config.schema import LongHorizonWorldConfig, Profile
from hunter_kinodynamic_rl.env.safety.action_guard import STOP_COMMAND, SafetyLimits
from hunter_kinodynamic_rl.env.scenarios.long_horizon_world import LongHorizonWorld
from hunter_kinodynamic_rl.env.simulation.gazebo_runtime import GazeboRuntimeMixin
from hunter_kinodynamic_rl.env.simulation.gazebo_service_wait import GazeboServiceError
from hunter_kinodynamic_rl.env.spawning.wall_segment_spawner import (
    WallSegmentPool, activate_walls, build_wall_pool, ensure_spawned as ensure_wall_pool_spawned,
)
from hunter_kinodynamic_rl.evaluation.fingerprint import (
    architecture_fingerprint,
    architecture_fingerprint_from_resolved_config,
    local_training_contract_fingerprint,
    local_training_contract_fingerprint_from_resolved_config,
)
from hunter_kinodynamic_rl.navigation.hierarchy.coordinator import HierarchyCoordinator
from hunter_kinodynamic_rl.navigation.hierarchy.local_feasibility_evaluator import (
    FeasibilityEvaluatorTelemetry, FrozenLocalFeasibilityEvaluator, LocalSensorSnapshot,
)
from hunter_kinodynamic_rl.navigation.hierarchy.subgoal_manager import SubgoalStatus
from hunter_kinodynamic_rl.navigation.local_rl.controller import LocalPolicyController
from hunter_kinodynamic_rl.navigation.local_rl.option_telemetry import OptionTelemetry
from hunter_kinodynamic_rl.navigation.local_rl.single_flight_worker import SingleFlightThreadWorker
from hunter_kinodynamic_rl.navigation.localization.gazebo_odom_backend import GazeboOdomLocalizationBackend
from hunter_kinodynamic_rl.navigation.localization.interface import is_pose_usable
from hunter_kinodynamic_rl.navigation.localization.wheel_imu_backend import WheelImuLocalizationBackend
from hunter_kinodynamic_rl.navigation.mapping.partial_map import PartialMap
from hunter_kinodynamic_rl.navigation.mission.mission_frame import MissionFrame, PoseXYYaw
from hunter_kinodynamic_rl.rl.algorithms.kinodynamic_tqc.agent import Agent as RiskAgent
from hunter_kinodynamic_rl.rl.algorithms.tqc.agent import Agent as VanillaAgent
from hunter_kinodynamic_rl.rl.checkpointing import manager as ckpt_manager
from hunter_kinodynamic_rl.trajectory.action_space import ACTION_DIM


SENSOR_QOS = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)


class LocalCheckpointError(RuntimeError):
    """Raised (never swallowed) for a missing, corrupt, or
    architecture/dimension-incompatible frozen Local checkpoint -- see
    module docstring's fail-fast contract."""


def _twist_from(command):
    """Duplicated from ``nodes/hierarchical_environment_node.py`` (a
    4-line, module-private helper there too) rather than imported --
    mirrors that module's own "tiny, load-bearing snippet duplicated with a
    cross-reference comment" convention for a helper too small to be worth
    a shared-module import boundary."""
    msg = Twist()
    msg.linear.x = command.speed_mps
    msg.angular.z = command.steering_rad
    return msg


class LiveGazeboLocalExecutor(Node, GazeboRuntimeMixin):
    """One instance per training/benchmark PROCESS (long-lived across many
    missions -- never reconstructed per mission, unlike
    ``SimplifiedKinematicLocalExecutor``). ``bind_mission(world,
    mission_frame)`` performs the actual Gazebo reset for one mission and
    returns ``self`` (matching the
    ``Callable[[LongHorizonWorld, MissionFrame], LocalOptionExecutor]``
    factory shape ``HierarchicalTrainingLoop``/``long_horizon_benchmark.py``
    inject in place of constructing ``SimplifiedKinematicLocalExecutor``
    directly)."""

    def __init__(
        self, profile: Profile, long_horizon_cfg: LongHorizonWorldConfig,
        local_checkpoint_dir: str, local_checkpoint_name: str,
        node_name: str = "hunter_kinodynamic_live_local_executor",
        world_name: str = "default", robot_entity_name: str = "hunter_se",
        cmd_vel_topic: str = "/cmd_vel", scan_topic: str = "/scan",
        odom_topic: Optional[str] = None, joint_states_topic: str = "/hunter_se/joint_states",
        contact_topic: str = "/hunter_se/chassis_contacts",
        localization_backend_factory: Optional[Callable[[], object]] = None,
    ) -> None:
        """``localization_backend_factory`` (defect-fix item 12, optional --
        every existing caller omits it and gets BYTE-IDENTICAL behavior to
        before): a zero-argument callable constructing the
        :class:`~hunter_kinodynamic_rl.navigation.localization.interface.LocalizationBackend`
        this executor drives, for a live localization sweep condition
        (e.g. ``lambda: WheelImuLocalizationBackend(WheelImuNoiseModel(...))``
        for a "noisy"/"drifting" condition). Defaults to the historical
        hardcoded ``GazeboOdomLocalizationBackend(use_covariance_confidence=True)``
        (ground-truth "ideal" condition) when omitted. See :meth:`bind_mission`
        and :meth:`_on_odom` for how a :class:`WheelImuLocalizationBackend`
        specifically is seeded/integrated -- every OTHER backend type is
        driven exactly like the historical ``GazeboOdomLocalizationBackend``
        path (``on_odometry_msg``)."""
        super().__init__(node_name)
        self.profile = profile
        self.long_horizon_cfg = long_horizon_cfg
        self.world_name = world_name
        self.robot_entity_name = robot_entity_name

        # -- GazeboRuntimeMixin's own attribute contract (see that module's
        # docstring) -- mirrors environment_node.py's __init__ exactly.
        self._gz_wait_timeout_sec = profile.runtime.gz_service_wait_timeout_sec
        self._gz_call_timeout_sec = profile.runtime.gz_service_call_timeout_sec
        self._gz_wait_poll_sec = profile.runtime.gz_service_poll_sec
        self._latest_sim_time_sec: Optional[float] = None
        self.scan_update_count = 0
        self.odom_update_count = 0

        clients_cb_group = MutuallyExclusiveCallbackGroup()
        from ros_gz_interfaces.srv import ControlWorld, SetEntityPose
        self.world_control_client = self.create_client(
            ControlWorld, f"/world/{world_name}/control", callback_group=clients_cb_group)
        self.set_entity_pose_client = self.create_client(
            SetEntityPose, f"/world/{world_name}/set_pose", callback_group=clients_cb_group)
        self._spawn_entity_client = self.create_client(
            SpawnEntity, f"/world/{world_name}/create", callback_group=clients_cb_group)
        self._delete_entity_client = self.create_client(
            DeleteEntity, f"/world/{world_name}/remove", callback_group=clients_cb_group)

        self._latest_scan: Optional[Tuple[np.ndarray, float, float]] = None
        self._latest_scan_receipt_time: Optional[float] = None
        self._latest_odom_receipt_time: Optional[float] = None
        self._latest_steering_rad = 0.0
        self._last_command_time: Optional[float] = None
        self._mission_frame: Optional[MissionFrame] = None
        self._world: Optional[LongHorizonWorld] = None
        # Bound only while a run_option() call is in flight -- see that
        # method and _on_scan's own comment.
        self._active_partial_map: Optional[PartialMap] = None
        # Process-lifetime cumulative counters -- DIAGNOSTIC ONLY (logging),
        # NEVER fed into HierarchyCoordinator/SubgoalManager. This executor
        # instance is long-lived across MANY missions/options (never
        # reconstructed per mission), so a counter that accumulates for the
        # whole process is a fundamentally different clock domain than "how
        # long has THIS option been running" -- run_option() below uses its
        # own LOCAL now_step/now_time_sec variables, reset to zero at the
        # top of every call, for that (code review / live-verified bug: this
        # single shared counter used to feed BOTH roles, so
        # SubgoalResult.elapsed_time_sec silently became
        # process_cumulative_time - option_start_time_sec, i.e. it kept
        # growing across every mission ever run in the process -- a 17.5s
        # budget mission benchmarked at 293-946.5s).
        self._process_tick_counter = 0
        self._process_time_sec = 0.0
        # Mission-local -- reset in bind_mission() -- feeds ONLY
        # PartialMap.record_visit()'s recency bookkeeping (mirrors
        # nodes/hierarchical_environment_node.py's own self._step_counter,
        # also reset once per mission, for the identical call).
        self._mission_tick_counter = 0
        self._latest_v_mps_value = 0.0
        self._latest_yaw_rate_value = 0.0
        self.last_option_telemetry: Optional[OptionTelemetry] = None
        # item 6: single-flight bounded-timeout policy inference (see
        # single_flight_worker.py's own docstring for the pattern this
        # reuses from nodes/real_policy_node.py) + counters surfaced in
        # OptionTelemetry.
        self._inference_worker: SingleFlightThreadWorker = SingleFlightThreadWorker()
        self._inference_timeout_count = 0
        self._inference_error_count = 0
        # item 8: reuses the SAME real Gazebo chassis-contact sensor
        # drl_agent's own environment.py already subscribes to
        # (/hunter_se/chassis_contacts, ros_gz_interfaces/msg/Contacts,
        # already bridged by simulate_hunter_se_ignition.launch.py) --
        # confirmed live in this bring-up's own ros2_gz_bridge log:
        # "Creating GZ->ROS Bridge: [/hunter_se/chassis_contacts ...]".
        # "ground_truth", never a clearance-distance proxy. See
        # _on_contact's own docstring for the latch semantics and
        # option_telemetry.py's docstring for what this string means to a
        # downstream consumer.
        self.collision_signal_source = "ground_truth"
        self._contact_event_count = 0
        self._contact_latched = False
        self._command_timeout_count = 0
        self._closed = False

        # -- Frozen Local TQC (fail-fast: missing/incompatible checkpoint
        # is a hard error here, never a silent untrained fallback -- unlike
        # nodes/hierarchical_environment_node.py's inference deployment
        # path, which deliberately tolerates running untrained for manual
        # profile-validation convenience). Deliberately BEFORE anything
        # ROS-entity-heavy below (subscriptions, the background spin
        # thread, wall-pool spawn): needs no live Gazebo, and raising here
        # must never leave a background thread running (confirmed live: an
        # earlier version started the spin thread before this block, so a
        # rejected checkpoint in a test left an orphaned daemon thread
        # that then raised "cannot schedule new futures after interpreter
        # shutdown" at the OS-process's own exit, well after the test that
        # triggered it had already finished). --
        if not local_checkpoint_dir or not local_checkpoint_name:
            raise LocalCheckpointError(
                "LiveGazeboLocalExecutor requires a non-empty local_checkpoint_dir/local_checkpoint_name "
                "-- a live-Gazebo hierarchical trainer/benchmark must drive a REAL frozen Local TQC "
                "checkpoint, never an untrained network"
            )
        history_len = profile.observation.frame_stack if profile.features.temporal_context else 1
        self.state_dim = profile.observation.lidar_bins * history_len + profile.observation.robot_state_dim
        requested_device = profile.hierarchical_training.local_inference_device
        if requested_device == "cuda":
            import torch
            if not torch.cuda.is_available():
                raise LocalCheckpointError(
                    "hierarchical_training.local_inference_device='cuda' but CUDA is unavailable"
                )
        agent_device = None if requested_device == "auto" else requested_device
        if profile.features.risk_critic:
            self.local_agent = RiskAgent(self.state_dim, ACTION_DIM, 1.0, profile.hyperparameters,
                                          profile.risk, profile.counterfactual, device=agent_device)
        else:
            self.local_agent = VanillaAgent(
                self.state_dim, ACTION_DIM, 1.0, profile.hyperparameters, device=agent_device)
        try:
            result = ckpt_manager.load_generation(
                local_checkpoint_dir, local_checkpoint_name, self.local_agent.checkpoint_components(),
                map_location=str(self.local_agent.device),
            )
        except (FileNotFoundError, RuntimeError) as e:
            raise LocalCheckpointError(
                f"LiveGazeboLocalExecutor: failed to load frozen Local checkpoint from "
                f"{local_checkpoint_dir!r} tag={local_checkpoint_name!r}: {e}"
            ) from e
        manifest = result.get("manifest", {})
        required_manifest_fields = (
            "generation", "pt_sha256", "state_dim", "action_dim", "resolved_config",
            "local_training_contract_fingerprint",
        )
        missing = [name for name in required_manifest_fields if name not in manifest or manifest[name] is None]
        if missing:
            raise LocalCheckpointError(
                "LiveGazeboLocalExecutor: frozen Local checkpoint manifest is missing required field(s) "
                f"{missing}; legacy/incomplete checkpoints are not accepted implicitly"
            )
        ckpt_state_dim, ckpt_action_dim = manifest["state_dim"], manifest["action_dim"]
        if ckpt_state_dim != self.state_dim:
            raise LocalCheckpointError(
                f"LiveGazeboLocalExecutor: checkpoint state_dim={ckpt_state_dim} != this profile's "
                f"state_dim={self.state_dim} -- refusing to drive an observation-incompatible checkpoint"
            )
        if ckpt_action_dim != ACTION_DIM:
            raise LocalCheckpointError(
                f"LiveGazeboLocalExecutor: checkpoint action_dim={ckpt_action_dim} != ACTION_DIM={ACTION_DIM} "
                "-- refusing to drive an action-incompatible checkpoint"
            )
        ckpt_resolved_config = manifest["resolved_config"]
        ckpt_fp = architecture_fingerprint_from_resolved_config(ckpt_resolved_config)
        this_fp = architecture_fingerprint(profile)
        if ckpt_fp != this_fp:
            raise LocalCheckpointError(
                f"LiveGazeboLocalExecutor: checkpoint architecture_fingerprint={ckpt_fp} != this "
                f"profile's architecture_fingerprint={this_fp} -- checkpoint was trained under a "
                "different action_space/features/risk/counterfactual configuration; refusing to "
                "drive it as if it matched"
            )
        try:
            recorded_contract = manifest["local_training_contract_fingerprint"]
            derived_contract = local_training_contract_fingerprint_from_resolved_config(ckpt_resolved_config)
            expected_profile = load_profile(profile.hierarchical_training.local_training_profile_name)
            expected_contract = local_training_contract_fingerprint(expected_profile)
        except (KeyError, ValueError) as e:
            raise LocalCheckpointError(
                f"LiveGazeboLocalExecutor: invalid Local training-contract metadata: {e}"
            ) from e
        if recorded_contract != derived_contract:
            raise LocalCheckpointError(
                "LiveGazeboLocalExecutor: Local checkpoint's recorded training-contract fingerprint "
                f"{recorded_contract!r} does not match its own resolved_config-derived value "
                f"{derived_contract!r}; refusing inconsistent metadata"
            )
        if recorded_contract != expected_contract:
            raise LocalCheckpointError(
                "LiveGazeboLocalExecutor: Local checkpoint training distribution is incompatible with "
                f"profile {profile.hierarchical_training.local_training_profile_name!r}: "
                f"checkpoint={recorded_contract}, expected={expected_contract}"
            )
        self.checkpoint_manifest = manifest
        self.get_logger().info(
            f"[live_local_executor] loaded frozen Local checkpoint {local_checkpoint_dir}/{local_checkpoint_name} "
            f"on device={self.local_agent.device} (loaded={result.get('loaded')}, skipped={result.get('skipped')})"
        )

        # -- Must exist BEFORE the subscriptions below are created: the
        # background spin thread (started further down) can deliver a
        # message and invoke _on_odom/_on_scan the MOMENT these
        # subscriptions exist, and both callbacks read self._localization
        # (confirmed live: creating this AFTER the subscriptions crashed
        # the spin thread on the very first odom message with
        # AttributeError, which then silently killed the thread -- every
        # subsequent Gazebo service call hung forever with nothing left to
        # process its response). --
        self._local_controller = LocalPolicyController(profile)
        self._localization = (
            localization_backend_factory() if localization_backend_factory is not None
            else GazeboOdomLocalizationBackend(use_covariance_confidence=True)
        )
        # Defect-fix item 12: only meaningful for a WheelImuLocalizationBackend
        # (dead-reckoning, needs an explicit initialize() before any
        # integrate() call) -- see bind_mission()/_on_odom() below. Reset
        # per-mission in bind_mission(), never carried over.
        self._wheel_imu_initialized = False
        self._latest_odom_time: Optional[float] = None
        self._safety_limits = SafetyLimits(
            max_sensor_age_sec=profile.runtime.sensor_freshness_timeout_sec,
            max_odom_age_sec=profile.runtime.sensor_freshness_timeout_sec,
            max_command_age_sec=profile.runtime.watchdog_command_timeout_sec,
            min_obstacle_stop_distance_m=profile.risk.min_safe_clearance_m,
        )

        # SENSOR_QOS (BEST_EFFORT) -- matches hierarchical_environment_node.py's
        # identical constant. The default RELIABLE QoS this used before is
        # incompatible with these topics' real BEST_EFFORT publishers
        # (confirmed live: "New publisher discovered ... incompatible QoS
        # ... RELIABILITY -- No messages will be received from it"), so no
        # scan/odom message would ever have reached these callbacks at all.
        self.create_subscription(Odometry, odom_topic or profile.localization.odom_topic, self._on_odom, SENSOR_QOS)
        self.create_subscription(LaserScan, scan_topic, self._on_scan, SENSOR_QOS)
        self.create_subscription(JointState, joint_states_topic, self._on_joint_states, SENSOR_QOS)
        self.create_subscription(Contacts, contact_topic, self._on_contact, SENSOR_QOS)
        self._cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)

        # -- background spin thread -- MUST start before ANY blocking
        # Gazebo service call below (wall-pool spawn): nothing else pumps
        # this node's callbacks/service-response futures while
        # __init__/bind_mission/run_option run on the CALLER's own thread
        # and use GazeboRuntimeMixin's time.sleep()-based waits (confirmed
        # live -- starting this thread AFTER ensure_wall_pool_spawned()
        # below, as a first version of this file did, made every
        # spawn/set_pose call time out: nothing was ever processing the
        # response). Mirrors environment_node.py's own executor model,
        # just with "a service worker thread" replaced by "this node's one
        # background spin thread" (this node is not itself a ROS service
        # server). Deliberately AFTER every fail-fast check above --
        # nothing here can raise, so this thread is never orphaned by an
        # __init__ that aborts partway through. --
        self._executor = MultiThreadedExecutor(num_threads=4)
        self._executor.add_node(self)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True, name=f"{node_name}_spin")
        self._spin_thread.start()

        # -- Phase 3: wall-segment pool (spawned once, repositioned per
        # mission via bind_mission -> activate_walls) -- requires a live
        # Gazebo world reachable NOW (blocks/fails on gz_service_wait_timeout_sec).
        # item 6 (code review): this is the first call below the spin
        # thread's start() that can raise (Gazebo service timeout, spawn
        # rejected, wall-pool capacity exceeded) -- if it does, __init__
        # must not return an object with an orphaned background spin
        # thread/node still alive (nothing left holding a reference to
        # close() it). Wrapped so ANY failure here tears the spin
        # thread/executor/node down before propagating, exactly like the
        # fail-fast checkpoint block above already does for its own
        # earlier failure window.
        try:
            self._wall_pool: WallSegmentPool = build_wall_pool(
                profile.wall_segment_pool, long_horizon_cfg.wall_thickness_m, long_horizon_cfg.wall_height_m,
                long_horizon_cfg.size_m, profile.observation.lidar_max_range_m,
            )
            ensure_wall_pool_spawned(self, self._spawn_entity_client, self._wall_pool)
        except Exception:
            self._shutdown_background_threads()
            self._executor.shutdown()
            self.destroy_node()
            raise

        # -- independent watchdog thread (item 6, mirrors
        # nodes/real_policy_node.py's own `_start_watchdog_thread` exactly:
        # a plain Python thread entirely outside rclpy's executor/callback-
        # group machinery, so it keeps publishing STOP even if the CALLER's
        # thread (run_option/bind_mission, or whatever invokes them) is
        # itself hung on something outside the bounded inference call --
        # single_flight_worker.py's timeout alone only bounds ONE inference
        # call; this bounds the command-freshness guarantee independent of
        # the caller's control flow entirely). Started LAST, after every
        # attribute it reads/writes already exists and nothing above it can
        # still raise. --
        self._start_watchdog_thread()

    def _start_watchdog_thread(self) -> None:
        self._watchdog_stop_event = threading.Event()

        def _loop() -> None:
            period = self.profile.runtime.watchdog_period_sec
            timeout = self.profile.runtime.watchdog_command_timeout_sec
            while not self._watchdog_stop_event.wait(period):
                if self._last_command_time is not None and (time.monotonic() - self._last_command_time) > timeout:
                    self._publish(STOP_COMMAND)
                    # item 8: surfaced via OptionTelemetry.command_timeout_count
                    # -- counts every watchdog-triggered stop, not just the
                    # in-loop inference-timeout STOPs run_option() itself
                    # already tracks (this one specifically catches the
                    # caller/inference thread being hung on something
                    # OUTSIDE the bounded inference call, see close()'s own
                    # docstring on why this watchdog exists at all).
                    self._command_timeout_count += 1

        self._watchdog_thread = threading.Thread(
            target=_loop, daemon=True, name=f"{self.get_name()}_watchdog")
        self._watchdog_thread.start()

    def _shutdown_background_threads(self) -> None:
        """Idempotent: stops+joins the watchdog thread (if it was ever
        started) and joins the spin thread after the executor itself has
        been asked to shut down. Safe to call from a partially-constructed
        instance (constructor-failure cleanup) or from :meth:`close`."""
        stop_event = getattr(self, "_watchdog_stop_event", None)
        if stop_event is not None:
            stop_event.set()
        watchdog_thread = getattr(self, "_watchdog_thread", None)
        if watchdog_thread is not None:
            watchdog_thread.join(timeout=2.0)

    def close(self) -> None:
        """Idempotent -- safe to call more than once (item 6). Publishes a
        final STOP, stops the watchdog, shuts the background spin executor
        down and joins its thread, then destroys the node -- no thread,
        future, or node is left running afterward."""
        if self._closed:
            return
        self._closed = True
        try:
            self._publish(STOP_COMMAND)
        except Exception:  # noqa: BLE001 -- best-effort final stop, never blocks shutdown
            pass
        self._shutdown_background_threads()
        self._executor.shutdown()
        spin_thread = getattr(self, "_spin_thread", None)
        if spin_thread is not None:
            spin_thread.join(timeout=2.0)
        self.destroy_node()

    # ------------------------------------------------------------------ sensor callbacks
    def _on_odom(self, msg: Odometry) -> None:
        self._latest_odom_receipt_time = time.monotonic()
        stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if isinstance(self._localization, WheelImuLocalizationBackend):
            # Defect-fix item 12: mirrors hierarchical_navigation_node.py's
            # own WheelImu dispatch exactly -- integrate() needs a prior
            # initialize() (done explicitly in bind_mission() once
            # world.start_pose is known, so the dead-reckoning origin lines
            # up with mission_frame's own origin rather than an arbitrary
            # (0,0,0)). Any _on_odom that fires BEFORE that explicit
            # initialize() (e.g. during bind_mission()'s own reset/settle
            # window) is simply not integrated -- a safe no-op, never a
            # placeholder-origin initialize() that bind_mission would then
            # have to detect and undo.
            if self._wheel_imu_initialized:
                previous_stamp = self._latest_odom_time if self._latest_odom_time is not None else stamp_sec
                dt_sec = max(0.0, stamp_sec - previous_stamp)
                self._localization.integrate(msg.twist.twist.linear.x, msg.twist.twist.angular.z, dt_sec, stamp_sec)
        else:
            self._localization.on_odometry_msg(msg, stamp_sec)
        self._latest_odom_time = stamp_sec
        # v/yaw_rate come from the raw Odometry message's own twist, never
        # from the localization backend's pose (which carries position/yaw
        # only) -- mirrors hierarchical_environment_node.py's identical
        # _on_odom split.
        self._latest_v_mps_value = msg.twist.twist.linear.x
        self._latest_yaw_rate_value = msg.twist.twist.angular.z
        self.odom_update_count += 1

    def _on_joint_states(self, msg: JointState) -> None:
        try:
            left = float(msg.position[msg.name.index("front_left_steering")])
            right = float(msg.position[msg.name.index("front_right_steering")])
        except (ValueError, IndexError, TypeError):
            return
        self._latest_steering_rad = 0.5 * (left + right)

    def _on_contact(self, msg: Contacts) -> None:
        """item 8: real Gazebo chassis-contact collision signal -- mirrors
        drl_agent's own environment.py::_update_contact_collision LATCH
        pattern (count a NEW event only on the empty->non-empty edge), but
        additionally UN-latches on the non-empty->empty edge (drl_agent's
        own latch is deliberately permanent, for "definitive episode
        termination" -- this executor instead wants a per-OPTION EVENT
        COUNT for telemetry, so a bump/separate/bump-again sequence must
        count as two events, not one)."""
        if msg.contacts:
            if not self._contact_latched:
                self._contact_event_count += 1
            self._contact_latched = True
        else:
            self._contact_latched = False

    def _on_scan(self, msg: LaserScan) -> None:
        self._latest_scan_receipt_time = time.monotonic()
        self._latest_scan = (np.asarray(msg.ranges, dtype=np.float64), msg.angle_min, msg.angle_increment)
        self.scan_update_count += 1
        # Continuous partial-map accumulation (mirrors
        # hierarchical_environment_node.py's own _on_scan) -- only while a
        # mission is bound AND an option is active, so a scan arriving
        # between options (before run_option's next call) is simply not
        # integrated (run_option's own tick loop integrates the freshest
        # scan explicitly at each tick instead -- see there); this callback
        # covers scans that arrive DURING the time.sleep() pacing inside a
        # tick, so no scan is silently skipped.
        # Captured into a LOCAL once -- never re-read self._active_partial_map
        # again below. run_option() (main/caller thread) can set it back to
        # None between two statements here (confirmed live: an in-flight
        # _on_scan call that already passed this None-check saw
        # self._active_partial_map flip to None between its integrate_scan
        # and record_visit calls, crashing record_visit on a NoneType --
        # silently, in the background thread, as an "exception was never
        # retrieved" warning rather than a visible failure). A local
        # reference can't be mutated out from under this callback by
        # another thread the way re-reading the attribute can.
        partial_map = self._active_partial_map
        if partial_map is None or self._mission_frame is None:
            return
        pose = self._localization.latest_pose()
        if not is_pose_usable(
            pose, now_sec=pose.stamp_sec, timeout_sec=self.profile.localization.pose_timeout_sec,
            min_confidence=self.profile.localization.minimum_confidence,
        ):
            return
        robot_pose_mission = self._mission_frame.odom_pose_to_mission(PoseXYYaw(x=pose.x, y=pose.y, yaw=pose.yaw))
        n = len(msg.ranges)
        if n == 0:
            return
        beam_yaws_robot = msg.angle_min + msg.angle_increment * np.arange(n, dtype=np.float64)
        beam_angles_mission = robot_pose_mission.yaw + beam_yaws_robot
        partial_map.integrate_scan(
            (robot_pose_mission.x, robot_pose_mission.y), beam_angles_mission,
            np.asarray(msg.ranges, dtype=np.float64), float(msg.range_max), range_min=float(msg.range_min),
        )
        partial_map.record_visit(robot_pose_mission.x, robot_pose_mission.y, self._mission_tick_counter)

    # ------------------------------------------------------------------ pose_world (run_mission reads this between options)
    @property
    def dt_sec(self) -> float:
        """The nominal control period run_option() paces itself at --
        exposed so a caller (e.g. long_horizon_benchmark.run_ablation_mission's
        navigation_time_sec sanity check, item 1) can compute this
        executor's own per-option elapsed-time upper bound
        (``max_local_steps * dt_sec``) without hardcoding/duplicating it."""
        return self.profile.runtime.time_delta_sec

    @property
    def pose_world(self) -> PoseXYYaw:
        pose = self._localization.latest_pose()
        # ``INVALID_POSE`` (the backend's own pre-first-odom sentinel) is
        # x=0.0/y=0.0 -- FINITE -- so checking finiteness alone would
        # silently treat "no real odom has arrived yet" as a genuine
        # (0, 0, 0) reading. ``valid`` is the backend's own documented flag
        # for exactly this distinction; only fall through to it once a real
        # (finite) reading has actually arrived.
        if pose is not None and pose.valid and math.isfinite(pose.x) and math.isfinite(pose.y):
            return PoseXYYaw(x=pose.x, y=pose.y, yaw=pose.yaw)
        if self._world is not None:
            return PoseXYYaw(*self._world.start_pose)
        return PoseXYYaw(0.0, 0.0, 0.0)

    def _localization_valid(self) -> bool:
        pose = self._localization.latest_pose()
        if not is_pose_usable(
            pose, now_sec=pose.stamp_sec, timeout_sec=self.profile.localization.pose_timeout_sec,
            min_confidence=self.profile.localization.minimum_confidence,
        ):
            return False
        if self._latest_odom_receipt_time is None:
            return False
        return (time.monotonic() - self._latest_odom_receipt_time) <= self.profile.localization.pose_timeout_sec

    # ------------------------------------------------------------------ reset (bind_mission == the LocalOptionExecutor factory)
    def bind_mission(
        self, world: LongHorizonWorld, mission_frame: MissionFrame,
        on_event: Optional[Callable[[str, dict], None]] = None,
    ) -> "LiveGazeboLocalExecutor":
        """Physically applies ``world`` to live Gazebo (spawn/reposition
        walls, teleport the robot to ``world.start_pose``) and stores
        ``mission_frame`` for subsequent ``run_option`` calls. Mirrors
        ``environment_node.py``'s own ``_on_reset`` sequence (pause -> world
        reset -> place entities -> settle -> wait for fresh sensors), then
        leaves the world UNPAUSED (this node has no ``/step``-style discrete
        advance -- ``run_option`` paces itself in real time, matching
        ``hierarchical_environment_node.py``'s continuous live-deployment
        design). ``on_event`` (defect-fix item 9, optional -- every existing
        caller passes none and is unaffected): called at each named
        sub-step (``wall_activation_complete``, ``robot_teleport_complete``,
        ``sensor_freshness_wait_started``, ``sensor_freshness_wait_complete``
        with ``elapsed_sec``) so a live-evidence recorder can log the actual
        sequence instead of only this method's own bundled start/end."""
        def _emit(name: str, **fields) -> None:
            if on_event is not None:
                on_event(name, fields)

        # Defect-fix item 12: a fresh mission must never inherit the
        # PREVIOUS mission's WheelImu dead-reckoning state -- any _on_odom
        # firing during THIS mission's own reset/settle window below (before
        # the explicit (re)initialize() a few lines down) is a safe no-op
        # (see _on_odom's own comment), never an integrate() against a
        # stale prior-mission origin.
        self._wheel_imu_initialized = False
        try:
            self.pause_world(True)
            self.reset_world()
            activate_walls(self, self._wall_pool, list(world.wall_segments))
            _emit("wall_activation_complete", wall_segment_count=len(world.wall_segments))
            half_yaw = world.start_pose[2] / 2.0
            self.set_entity_pose_ignition(
                self.robot_entity_name, world.start_pose[0], world.start_pose[1], 0.0,
                0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw),
            )
            _emit("robot_teleport_complete", start_pose=world.start_pose)
            prev_scan, prev_odom = self.scan_update_count, self.odom_update_count
            self.propagate_state(self.profile.runtime.reset_settle_time_sec)
            self.pause_world(False)
        except GazeboServiceError as e:
            raise RuntimeError(f"LiveGazeboLocalExecutor.bind_mission: Gazebo reset failed: {e}") from e
        _emit("sensor_freshness_wait_started", timeout_sec=self.profile.runtime.sensor_freshness_timeout_sec)
        wait_t0 = time.monotonic()
        fresh = self.wait_for_fresh_sensors(prev_scan, prev_odom,
                                             timeout_sec=self.profile.runtime.sensor_freshness_timeout_sec)
        _emit("sensor_freshness_wait_complete", elapsed_sec=time.monotonic() - wait_t0, fresh=fresh)
        if not fresh:
            raise RuntimeError(
                "LiveGazeboLocalExecutor.bind_mission: scan/odom did not refresh after reset within "
                f"{self.profile.runtime.sensor_freshness_timeout_sec:.1f}s -- Gazebo may have stalled"
            )
        self._maybe_initialize_wheel_imu(world.start_pose)
        self._world = world
        self._mission_frame = mission_frame
        self._local_controller.reset()
        self._last_command_time = time.monotonic()
        # A new mission must never inherit the PREVIOUS mission's recency
        # bookkeeping (item 1: "새 option 및 새 mission 시작 시 이전
        # mission의 누적 시간이 elapsed/timeout에 영향을 주지 않게 한다") --
        # self._process_tick_counter/_process_time_sec are deliberately
        # NOT reset here (they are process-lifetime diagnostics, see
        # __init__'s own comment), only this mission-local one is.
        self._mission_tick_counter = 0
        return self

    def _maybe_initialize_wheel_imu(self, start_pose: Tuple[float, float, float]) -> None:
        """Defect-fix item 12: seed the dead-reckoning origin at the ACTUAL
        world start pose (never (0,0,0)) so this backend's coordinate frame
        lines up with ``mission_frame``'s own origin (``mission_frame`` is
        built by the CALLER directly from ``world.start_pose``, independent
        of this localization backend) -- a mismatched origin here would
        silently corrupt every mission-frame conversion the control loop
        does. No-op for any other backend type."""
        if not isinstance(self._localization, WheelImuLocalizationBackend):
            return
        # By the time bind_mission() calls this (after wait_for_fresh_sensors
        # succeeded), at least one real _on_odom already landed this
        # mission, so self._latest_odom_time is a real stamp in the SAME
        # clock domain integrate() will keep using; the time.monotonic()
        # fallback only matters for a direct/test call before any odom.
        seed_stamp = self._latest_odom_time if self._latest_odom_time is not None else time.monotonic()
        self._localization.initialize(start_pose[0], start_pose[1], start_pose[2], seed_stamp)
        self._wheel_imu_initialized = True

    # ------------------------------------------------------------------ LocalOptionExecutor protocol
    def run_option(
        self, coordinator: HierarchyCoordinator, partial_map: PartialMap, rng: np.random.RandomState,
        max_local_steps: int, on_tick: Optional[Callable[[dict], None]] = None,
    ) -> None:
        """``on_tick`` (defect-fix item 9, optional -- every existing caller
        passes none and is unaffected): invoked once per LOCAL control tick
        with a plain dict of exactly the fields a live-evidence/diagnostic
        recorder needs (step/time/pose/subgoal/action/command/risk/guard/
        collision) -- this executor's own control loop never depends on the
        callback's return value or holds a reference to it beyond this
        call, so a slow/raising ``on_tick`` only affects the CALLER's own
        recording, never this executor's actual control behavior (any
        exception it raises propagates normally, exactly like any other
        per-tick bug would -- it is not swallowed, since a live-evidence
        recorder failing silently would be worse than it failing loudly)."""
        if self._mission_frame is None:
            raise RuntimeError("LiveGazeboLocalExecutor.run_option called before bind_mission()")
        dt_sec = self.profile.runtime.time_delta_sec
        inference_timeout_sec = self.profile.runtime.policy_inference_timeout_sec
        self._active_partial_map = partial_map
        self.last_option_telemetry = OptionTelemetry(collision_signal_source=self.collision_signal_source)
        # item 8: command_timeout_count/collision_count are per-OPTION
        # deltas of process-lifetime counters (mirrors
        # _process_tick_counter/_mission_tick_counter's identical
        # domain-separation discipline, item 1) -- captured here and
        # finalized in the `finally` block below.
        command_timeout_count_before = self._command_timeout_count
        contact_event_count_before = self._contact_event_count
        # item 1: OPTION-LOCAL clock -- plain local variables, reset to
        # zero on EVERY run_option() call, never carried over from a
        # previous option or mission and never mixed with
        # self._process_tick_counter/_process_time_sec (see __init__'s own
        # comment on that distinction). These are exactly what feeds
        # coordinator.record_local_tick()/force_terminate_active_subgoal()
        # below, so SubgoalResult.elapsed_time_sec is always within
        # [0, max_local_steps * dt_sec] -- the actual option control-time
        # range -- regardless of how many missions/options this
        # long-lived executor instance has already driven.
        local_tick = 0
        local_time_sec = 0.0
        try:
            for _ in range(max_local_steps):
                if coordinator.stop_required or coordinator.mission_done:
                    return

                # item 2: every one of the fault branches below MUST leave
                # the subgoal in one of the six terminal SubgoalStatus
                # values (never a silent `continue`/fall-through that
                # abandons an ACTIVATED subgoal with last_subgoal_result
                # still None) -- see force_terminate_active_subgoal's own
                # docstring for the full rationale.
                if not self._localization_valid():
                    self._publish(STOP_COMMAND)
                    coordinator.force_terminate_active_subgoal(
                        SubgoalStatus.CANCELLED_BY_REPLAN, "localization_confidence_degraded",
                        now_step=local_tick, now_time_sec=local_time_sec,
                    )
                    return
                scan_age_sec = (
                    math.inf if self._latest_scan_receipt_time is None
                    else time.monotonic() - self._latest_scan_receipt_time
                )
                if self._latest_scan is None or scan_age_sec > self.profile.runtime.sensor_freshness_timeout_sec:
                    self._publish(STOP_COMMAND)
                    coordinator.force_terminate_active_subgoal(
                        SubgoalStatus.CANCELLED_BY_REPLAN, "scan_stale",
                        now_step=local_tick, now_time_sec=local_time_sec,
                    )
                    return

                subgoal_mission = coordinator.active_subgoal_mission
                robot_pose_mission = self._mission_frame.odom_pose_to_mission(self.pose_world)
                subgoal_robot = MissionFrame.mission_to_robot(subgoal_mission, robot_pose_mission)
                ranges, angle_min, angle_increment = self._latest_scan
                # defect-fix item 1: subgoal_robot is already ROBOT-frame
                # (MissionFrame.mission_to_robot above) -- RobotState's
                # pose MUST be the origin to match, never the live
                # odom/mission-frame pose (see
                # LocalPolicyController.build_observation's docstring).
                robot_state = LocalPolicyController.robot_relative_state(
                    v=self._latest_v_mps(), yaw_rate=self._latest_yaw_rate(), steering=self._latest_steering_rad,
                )
                # observed_before/after (below, post-sleep) straddles a
                # background-thread _on_scan write to the SAME partial_map
                # (no lock) -- a benign, low-severity race: newly_explored
                # can occasionally be off by a few cells (telemetry/
                # exploration-bonus precision only, never a crash or a
                # training-invalidating error).
                observed_before = partial_map.observed_count()
                local_obs = self._local_controller.build_observation(
                    ranges, angle_min, angle_increment, robot_state, subgoal_robot[0], subgoal_robot[1],
                )

                # item 6: single-flight, bounded-timeout policy inference
                # (never a direct, unbounded self.local_agent.select_action()
                # call) -- see single_flight_worker.py's own docstring for
                # the pattern reused from nodes/real_policy_node.py.
                raw_action, infer_error, timed_out = self._inference_worker.call(
                    lambda obs=local_obs.observation: self.local_agent.select_action(obs, deterministic=True),
                    inference_timeout_sec,
                )
                if timed_out:
                    self._inference_timeout_count += 1
                    self.last_option_telemetry.inference_timeout_count += 1
                    self._publish(STOP_COMMAND)
                    coordinator.force_terminate_active_subgoal(
                        SubgoalStatus.CANCELLED_BY_REPLAN, "inference_timeout", robot_pose_mission,
                        now_step=local_tick, now_time_sec=local_time_sec,
                    )
                    return
                if infer_error is not None:
                    self._inference_error_count += 1
                    self.last_option_telemetry.inference_error_count += 1
                    self._publish(STOP_COMMAND)
                    coordinator.force_terminate_active_subgoal(
                        SubgoalStatus.CANCELLED_BY_REPLAN, "inference_error", robot_pose_mission,
                        now_step=local_tick, now_time_sec=local_time_sec,
                    )
                    return

                action_arr = LocalPolicyController.validate_action(raw_action)
                if action_arr is None:
                    self._publish(STOP_COMMAND)
                    coordinator.force_terminate_active_subgoal(
                        SubgoalStatus.CANCELLED_BY_REPLAN, "invalid_action_output", robot_pose_mission,
                        now_step=local_tick, now_time_sec=local_time_sec,
                    )
                    return

                self._local_controller.commit_action(action_arr)
                now = time.monotonic()
                nominal_command, safe_command = self._local_controller.decode_and_guard(
                    action_arr, current_steering_rad=self._latest_steering_rad, safety_limits=self._safety_limits,
                    nearest_obstacle_dist_m=local_obs.nearest_obstacle_dist_m,
                    last_sensor_time_sec=self._latest_scan_receipt_time or 0.0,
                    last_command_time_sec=self._last_command_time or now,
                    now_sec=now, last_odom_time_sec=self._latest_odom_receipt_time,
                    localization_valid=True, subgoal_valid=True,
                )
                self._publish(safe_command)
                self._last_command_time = now

                time.sleep(dt_sec)

                local_tick += 1
                local_time_sec += dt_sec
                self._process_tick_counter += 1
                self._process_time_sec += dt_sec
                self._mission_tick_counter += 1
                next_pose_mission = self._mission_frame.odom_pose_to_mission(self.pose_world)
                emergency_stop = bool(nominal_command.speed_mps > 1e-3 and safe_command.speed_mps <= 1e-6)
                # TTC/stopping-margin (detailed spec's local benchmark
                # metrics) -- clearance/speed-derived, updated every tick
                # this option runs; None fields stay None if the option
                # never records a finite speed (e.g. it stops immediately).
                clearance_m = local_obs.nearest_obstacle_dist_m
                v_mps = robot_state.v
                if math.isfinite(clearance_m) and v_mps > 1e-3:
                    ttc = clearance_m / v_mps
                    tel = self.last_option_telemetry
                    tel.min_time_to_collision_sec = (
                        ttc if tel.min_time_to_collision_sec is None else min(tel.min_time_to_collision_sec, ttc)
                    )
                    brake_decel = self.profile.robot.brake_decel_mps2
                    if brake_decel > 0.0:
                        margin = clearance_m - (v_mps * v_mps) / (2.0 * brake_decel)
                        tel.min_stopping_margin_m = (
                            margin if tel.min_stopping_margin_m is None else min(tel.min_stopping_margin_m, margin)
                        )
                steering_saturated = bool(
                    abs(safe_command.steering_rad) >= self.profile.robot.steering_limit_rad - 1e-3
                )
                predicted_risk = None
                predict_risk_fn = getattr(self.local_agent, "predict_risk", None)
                if predict_risk_fn is not None:
                    predicted_risk = predict_risk_fn(local_obs.observation, action_arr)
                newly_explored = partial_map.observed_count() - observed_before
                subgoal_cell = partial_map.world_to_cell(*subgoal_mission)
                subgoal_endpoint_blocked = (
                    False if subgoal_cell is None else bool(partial_map.channels().inflated[subgoal_cell])
                )

                coordinator.record_local_tick(
                    next_pose_mission, now_step=local_tick, now_time_sec=local_time_sec,
                    speed_mps=robot_state.v, clearance_m=local_obs.nearest_obstacle_dist_m,
                    predicted_risk=predicted_risk, emergency_stop=emergency_stop,
                    steering_saturated=steering_saturated, newly_explored_cells=newly_explored,
                    subgoal_endpoint_blocked=subgoal_endpoint_blocked,
                    localization_confidence=self._localization.latest_pose().confidence,
                )
                if on_tick is not None:
                    on_tick({
                        "step": local_tick, "time_sec": local_time_sec,
                        "pose_mission": (next_pose_mission.x, next_pose_mission.y, next_pose_mission.yaw),
                        "subgoal_mission": subgoal_mission, "action": [float(a) for a in action_arr],
                        "command": {"speed_mps": safe_command.speed_mps, "steering_rad": safe_command.steering_rad},
                        "predicted_risk": predicted_risk, "emergency_stop": emergency_stop,
                        "steering_saturated": steering_saturated, "clearance_m": local_obs.nearest_obstacle_dist_m,
                        "collision_count_cumulative": self._contact_event_count,
                    })
                if coordinator.subgoal_reached or coordinator.subgoal_failed or coordinator.mission_done:
                    return
            # item 2: max_local_steps exhausted WITHOUT any of the above
            # triggers -- including evaluate_replanning's own
            # local_option_timeout_steps, whenever it is configured LARGER
            # than this call's own max_local_steps budget -- ever firing.
            # The subgoal is still ACTIVE at this point; never fall
            # through silently (code review / live-verified bug: this used
            # to leave last_subgoal_result permanently None whenever a
            # benchmark capped max_local_steps below
            # local_option_timeout_steps/no_progress_window_steps).
            if coordinator.subgoal_manager.active:
                robot_pose_mission = self._mission_frame.odom_pose_to_mission(self.pose_world)
                coordinator.force_terminate_active_subgoal(
                    SubgoalStatus.FAILED_TIMEOUT, "max_local_steps_exhausted", robot_pose_mission,
                    now_step=local_tick, now_time_sec=local_time_sec,
                )
        finally:
            self._active_partial_map = None
            self.last_option_telemetry.command_timeout_count = (
                self._command_timeout_count - command_timeout_count_before
            )
            self.last_option_telemetry.collision_count = (
                self._contact_event_count - contact_event_count_before
            )
            self._publish(STOP_COMMAND)

    # ------------------------------------------------------------------ requirement E: LocalFeasibilityEvaluator wiring
    def _local_feasibility_snapshot(self) -> Optional[LocalSensorSnapshot]:
        """``snapshot_provider`` for :meth:`build_local_feasibility_evaluator`
        -- reads this executor's own live odom/steering state plus a
        frozen copy of ``self._local_controller``'s CURRENT temporal state
        (defect-fix item 6: the REAL control loop's own LiDAR frame
        history and last committed action, never a second independently-
        accumulated stack) exactly once per call and returns ``None`` (a
        tracked fallback) whenever either sensor hasn't arrived yet or the
        temporal context isn't seeded yet."""
        scan_receipt = self._latest_scan_receipt_time
        odom_receipt = self._latest_odom_receipt_time
        if scan_receipt is None or odom_receipt is None:
            return None
        temporal_context = self._local_controller.snapshot_temporal_context()
        if temporal_context is None:
            return None
        return LocalSensorSnapshot(
            lidar_frame=temporal_context.lidar_frame, prev_action=temporal_context.prev_action,
            speed_mps=self._latest_v_mps(), yaw_rate_rps=self._latest_yaw_rate(),
            steering_rad=self._latest_steering_rad, stamp_monotonic=min(scan_receipt, odom_receipt),
        )

    def build_local_feasibility_evaluator(
        self, *, inference_timeout_sec: float, max_snapshot_age_sec: float,
        telemetry: Optional[FeasibilityEvaluatorTelemetry] = None,
    ) -> FrozenLocalFeasibilityEvaluator:
        """Requirement E: constructs a real ``LocalFeasibilityEvaluator``
        bound to THIS executor's own frozen ``self.local_agent`` and live
        sensor state -- the piece a live Global-training/hierarchical-
        benchmark caller passes as ``local_evaluator`` (to
        ``HierarchicalTrainingLoop``/``run_ablation_mission``) whenever the
        ablation in use needs ``feasibility_feedback_enabled``/
        ``global_risk_feedback_enabled`` over THIS live Gazebo executor,
        instead of leaving those features zero-filled."""
        return FrozenLocalFeasibilityEvaluator(
            self.profile, self.local_agent, inference_timeout_sec=inference_timeout_sec,
            max_snapshot_age_sec=max_snapshot_age_sec, snapshot_provider=self._local_feasibility_snapshot,
            telemetry=telemetry,
        )

    def _publish(self, command) -> None:
        self._cmd_pub.publish(_twist_from(command))

    def _latest_v_mps(self) -> float:
        return self._latest_v_mps_value

    def _latest_yaw_rate(self) -> float:
        return self._latest_yaw_rate_value
