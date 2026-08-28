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
from hunter_kinodynamic_rl.trajectory.action_space import ACTION_DIM
from hunter_kinodynamic_rl.trajectory.pure_pursuit_adapter import VehicleCommand


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
        """``subgoal_x``/``subgoal_y`` MUST be the active subgoal (mission
        frame converted to whatever frame ``robot_state``/goal math already
        uses elsewhere in this package -- robot/odom frame, matching
        ``build_robot_state_vector``'s existing goal_x/goal_y convention),
        never the final mission goal."""
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

    def commit_action(self, action_arr: np.ndarray) -> None:
        """Stores the just-decoded action as ``prev_action`` for the NEXT
        tick's observation -- call once per successful tick, after
        inference resolves but the caller still owns whether to call this
        for a tick that timed out/errored (real_policy_node only commits
        on a successful inference, matching its ``_prev_action_01``
        update site)."""
        self.prev_action = [float(a) for a in action_arr]
