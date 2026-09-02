"""ROS-independent local-policy contract: sensor frame -> observation vector,
and (raw policy action) -> guarded ``VehicleCommand`` (plan section 6.2/6.7).

This is the observation-build / action-decode / safety-guard slice of
``nodes/real_policy_node.py``'s own ``_on_control_tick`` pipeline, factored
out so both that node and a future hierarchy control loop (test harness,
live-sim runner, or a Phase 4+ ROS coordinator) drive the SAME local TQC
through byte-identical logic -- only sensor I/O, timing, and the actual
network-inference call stay in the caller.

Critical Phase 2 contract (plan section 6.3/6.9): every goal coordinate this
module ever sees is the ACTIVE SUBGOAL, never the final mission goal --
callers (``HierarchyCoordinator``) own that separation; nothing here reads
or stores a final goal at all, so a final-goal leak into the local
observation is structurally impossible from this module's own API.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from hunter_kinodynamic_rl.config.schema import Profile
from hunter_kinodynamic_rl.env.observation.observation_builder import (
    RobotState, build_observation, build_robot_state_vector,
)
from hunter_kinodynamic_rl.env.safety.action_guard import STOP_COMMAND, SafetyLimits, guard
from hunter_kinodynamic_rl.sensing.scan_processor import front_and_full_state
from hunter_kinodynamic_rl.sensing.temporal_stack import FrameStack
from hunter_kinodynamic_rl.trajectory import trajectory_executor
from hunter_kinodynamic_rl.trajectory.action_space import ACTION_DIM, LegacyWaypointCommand, TrajectoryCommand, decode_action
from hunter_kinodynamic_rl.trajectory.pure_pursuit_adapter import VehicleCommand
from hunter_kinodynamic_rl.trajectory.trajectory_primitive import ConstantCurvatureArc


@dataclass(frozen=True)
class LocalTemporalContext:
    """Immutable snapshot of :class:`LocalPolicyController`'s own temporal
    state -- the already-stacked LiDAR frame history and the last committed
    action -- for a caller (Global candidate feasibility scoring) that needs
    to read the REAL control loop's current context WITHOUT being able to
    mutate it (defect-fix item 6: candidate-order-dependent contamination).
    Both fields are plain copies, never a view into the controller's own
    mutable buffers -- see :meth:`LocalPolicyController.snapshot_temporal_context`."""

    lidar_frame: np.ndarray
    prev_action: Tuple[float, float, float]


@dataclass(frozen=True)
class LocalObservationResult:
    """``observation`` is what the policy network consumes; ``obs_state``/
    ``environment_state`` (front-sector / full-360 LiDAR bins, see
    ``sensing/scan_processor.py``) and ``nearest_obstacle_dist_m`` are
    exposed separately since callers (the safety guard, replanning's
    "known occupied" / risk checks) need them independent of the stacked
    policy observation."""

    observation: np.ndarray
    obs_state: np.ndarray
    environment_state: np.ndarray
    nearest_obstacle_dist_m: float


class LocalPolicyController:
    """Pure Python (no rclpy, no Gazebo). Owns the frame-stack and
    previous-action memory that must persist ACROSS ticks -- exactly the
    state ``RealPolicyNode`` used to keep on ``self`` directly. One
    instance per running local-policy loop; call :meth:`reset` when a new
    subgoal/episode begins a fresh temporal window (mirrors
    ``environment_node.py``'s own per-episode frame-stack reset)."""

    def __init__(self, profile: Profile) -> None:
        self.profile = profile
        history_len = profile.observation.frame_stack if profile.features.temporal_context else 1
        self._frame_stack = FrameStack(profile.observation.lidar_bins, history_len)
        self._frame_stack_ready = False
        self.prev_action = [0.0, 0.0, 0.0]

    def reset(self) -> None:
        """Re-arms the frame stack for a fresh temporal window and clears
        previous-action memory. Does NOT reallocate the FrameStack -- the
        next :meth:`build_observation` call reseeds it from that tick's
        scan (same lazy-reset convention ``RealPolicyNode`` already used)."""
        self._frame_stack_ready = False
        self.prev_action = [0.0, 0.0, 0.0]

    def build_observation(
        self, ranges, angle_min: float, angle_increment: float, robot_state: RobotState,
        subgoal_x: float, subgoal_y: float,
    ) -> LocalObservationResult:
        """``subgoal_x``/``subgoal_y`` MUST be the active subgoal, and
        ``robot_state``'s pose (x, y, yaw) MUST be expressed in the SAME
        frame as that subgoal -- ``build_robot_state_vector``'s
        ``goal_distance_and_heading`` call has no independent way to detect
        a frame mismatch, so passing mismatched frames silently produces a
        wrong distance/heading with no error (defect-fix item 1: this is
        exactly the historical bug -- a robot-frame subgoal paired with an
        odom/mission-frame pose). Exactly two contracts are valid anywhere
        in this package:

        1. **Robot-relative subgoal** (every hierarchical-navigation call
           site: ``nodes/hierarchical_navigation_node.py``,
           ``nodes/hierarchical_environment_node.py``,
           ``navigation/local_rl/live_gazebo_executor.py``,
           ``navigation/hierarchy/local_feasibility_evaluator.py``) --
           ``subgoal_x``/``subgoal_y`` come from
           ``MissionFrame.mission_to_robot(...)``, and ``robot_state``'s
           pose MUST be the origin ``(0, 0, 0)`` (velocities/yaw-rate/
           steering stay real measurements) -- see
           :meth:`robot_relative_state`, which every one of those call
           sites uses to construct it so this can never drift back into a
           mismatched pair.
        2. **Single shared frame** (the non-hierarchical standalone path:
           ``env/simulation/environment_node.py``,
           ``nodes/real_policy_node.py``) -- ``robot_state``'s pose AND the
           goal are both given in the same odom/world frame (there is no
           separate "final goal" to convert; the node's own tracked goal
           already IS the active subgoal in that frame).

        Never mix the two: a non-origin ``robot_state`` pose paired with a
        robot-relative subgoal (or vice versa) is the defect this docstring
        exists to prevent from recurring."""
        obs_cfg = self.profile.observation
        obs_state, environment_state = front_and_full_state(
            ranges, angle_min, angle_increment, obs_cfg.lidar_bins,
            obs_cfg.lidar_max_range_m, obs_cfg.front_sector_width_rad,
        )
        if not self._frame_stack_ready:
            self._frame_stack.reset(obs_state)
            self._frame_stack_ready = True
        else:
            self._frame_stack.push(obs_state)
        lidar_frame = self._frame_stack.stacked()

        robot_state_vector = build_robot_state_vector(
            robot_state, subgoal_x, subgoal_y, self.prev_action,
            robot_state_dim=obs_cfg.robot_state_dim,
        )
        observation = build_observation(lidar_frame, robot_state_vector)
        nearest = float(environment_state.min()) if environment_state.size else float("inf")
        return LocalObservationResult(
            observation=observation, obs_state=obs_state, environment_state=environment_state,
            nearest_obstacle_dist_m=nearest,
        )

    @staticmethod
    def robot_relative_state(v: float, yaw_rate: float, steering: float) -> RobotState:
        """Canonical constructor for the "robot-relative subgoal" contract
        documented on :meth:`build_observation` (defect-fix item 1) --
        pose is pinned to the origin ``(0, 0, 0)`` since the subgoal handed
        to ``build_observation`` alongside it is already expressed in the
        robot's own frame; only real measurements (speed, yaw rate,
        steering) are carried through. Every hierarchical-navigation call
        site MUST build its ``RobotState`` through this helper rather than
        re-constructing one from a live odom/mission pose directly -- that
        re-construction is exactly how the historical frame-mismatch bug
        was introduced independently at four call sites."""
        return RobotState(x=0.0, y=0.0, yaw=0.0, v=v, yaw_rate=yaw_rate, steering=steering)

    def snapshot_temporal_context(self) -> Optional[LocalTemporalContext]:
        """Immutable copy of this controller's CURRENT LiDAR frame-stack
        window and previous action -- for a caller (Global candidate
        feasibility scoring, ``navigation.hierarchy.local_feasibility_evaluator``)
        that needs the REAL control loop's up-to-date temporal state
        without holding a reference into its mutable buffers and without
        being able to push into/reset them (defect-fix item 6). Returns
        ``None`` before the frame stack has been seeded by a first
        ``build_observation`` call this episode/mission -- callers must
        treat that exactly like any other "no evaluation possible yet"
        fallback, never substitute a fabricated all-zero frame."""
        if not self._frame_stack_ready:
            return None
        return LocalTemporalContext(
            lidar_frame=self._frame_stack.stacked().copy(),
            prev_action=(float(self.prev_action[0]), float(self.prev_action[1]), float(self.prev_action[2])),
        )

    @staticmethod
    def validate_action(action) -> Optional[np.ndarray]:
        """Rejects a wrong-shaped or non-finite raw policy action BEFORE it
        reaches trajectory decoding -- mirrors ``RealPolicyNode``'s own
        pre-decode guard exactly. Returns ``None``, NEVER RAISES, on
        rejection -- including malformed input ``np.asarray`` itself
        cannot convert (a ragged/inhomogeneous nested sequence, or a plain
        object with no array interface) -- so callers can uniformly fall
        back to a safe stop without their own try/except around this call.
        ``real_policy_node.py`` happens to also wrap its own call in an
        outer try/except, but this is a general-purpose pure API a future
        hierarchy node may call directly (code review finding: the
        documented "never raises" contract did not actually hold for
        ragged/object input before this fix)."""
        try:
            action_arr = np.asarray(action, dtype=np.float64).reshape(-1)
            if action_arr.shape[0] != ACTION_DIM or not np.all(np.isfinite(action_arr)):
                return None
            return action_arr
        except (TypeError, ValueError):
            return None

    def decode_and_guard(
        self, action_arr: np.ndarray, current_steering_rad: Optional[float],
        safety_limits: SafetyLimits, nearest_obstacle_dist_m: Optional[float],
        last_sensor_time_sec: float, last_command_time_sec: float, now_sec: float,
        last_odom_time_sec: Optional[float] = None,
        localization_valid: bool = True, subgoal_valid: bool = True,
    ) -> Tuple[VehicleCommand, VehicleCommand]:
        """Action -> trajectory -> guarded ``VehicleCommand``. Returns
        ``(nominal_command, safe_command)`` -- ``nominal_command`` is the
        pre-guard decode (``STOP_COMMAND`` when a safety precondition below
        short-circuits, since no real decode happens in that case), needed
        by callers that derive an ``emergency_stop`` diagnostic the same
        way ``environment_node.py`` does (``nominal.speed_mps > 1e-3 and
        safe.speed_mps <= 1e-6``).

        Section 6's two hard safety preconditions -- "stale/invalid
        localization이면 physical motion 금지" and "invalid subgoal이면
        physical motion 금지" -- are enforced FIRST, before the action is
        even decoded: either flag False returns ``(STOP_COMMAND,
        STOP_COMMAND)`` immediately, independent of (and never overridden
        by) whatever ``guard()`` itself would otherwise compute from
        sensor/command freshness."""
        if not localization_valid or not subgoal_valid:
            return STOP_COMMAND, STOP_COMMAND
        command = trajectory_executor.execute(
            action_arr, self.profile.action_space, self.profile.trajectory, self.profile.robot,
            dynamics_cfg=self.profile.dynamics if self.profile.features.trajectory_l_preview_blend else None,
            current_steering_rad=(
                current_steering_rad if self.profile.features.trajectory_l_preview_blend else None
            ),
        )
        safe_command = guard(
            command, self.profile.robot, safety_limits,
            last_sensor_time_sec=last_sensor_time_sec, last_command_time_sec=last_command_time_sec,
            now_sec=now_sec, nearest_obstacle_distance_m=nearest_obstacle_dist_m,
            last_odom_time_sec=last_odom_time_sec,
        )
        return command, safe_command

    def terminal_point_robot_frame(self, action_arr: np.ndarray) -> Tuple[float, float]:
        """Decodes ``action_arr`` (``decode_action``, mode-dispatched --
        never re-implements curvature/waypoint math) into the (x, y)
        terminal point the action steers toward, in the ROBOT's own frame
        (x forward, y left) -- used by
        ``navigation.local_rl.feasibility_evaluator`` to judge whether a
        local action conditioned on a Global candidate still progresses
        toward it (requirement E's "progress-preserving" feature), never
        by the control loop itself (``decode_and_guard`` already owns the
        real command path via ``trajectory_executor.execute``)."""
        command = decode_action(action_arr, self.profile.action_space, self.profile.robot)
        if isinstance(command, TrajectoryCommand):
            point = ConstantCurvatureArc(
                kappa=command.kappa, v_ref=command.v_ref, horizon_m=command.horizon_m,
            ).point_at(command.horizon_m)
            return point.x, point.y
        if isinstance(command, LegacyWaypointCommand):
            return command.r * math.cos(command.theta), command.r * math.sin(command.theta)
        raise ValueError(f"terminal_point_robot_frame: unsupported decoded command type {type(command)!r}")

    def commit_action(self, action_arr: np.ndarray) -> None:
        """Stores the just-decoded action as ``prev_action`` for the NEXT
        tick's observation -- call once per successful tick, after
        inference resolves but the caller still owns whether to call this
        for a tick that timed out/errored (real_policy_node only commits
        on a successful inference, matching its ``_prev_action_01``
        update site)."""
        self.prev_action = [float(a) for a in action_arr]
