"""Assemble the final policy observation vector: [lidar bins (optionally
temporally stacked)] + [goal distance, heading error, previous action (2),
speed, yaw rate, steering] -- section 13.

Pure function of its inputs (no ROS, no state beyond the caller-owned
FrameStack), so it's independently testable and identical whether called
from the simulator env node or the real-robot inference node.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hunter_kinodynamic_rl.common.geometry import goal_distance_and_heading


# section P0-4 (baseline/reference parity): drl_agent's OWN 87D observation
# contract (CLAUDE.md's State/Action Space section) carries a 7D robot-state
# tail -- goal_dist, heading_err, prev_r, prev_theta, v, yaw_rate, steering
# -- deliberately only the first TWO previous-action components (its own
# 3rd, hybrid stop/yield action component, is NOT included in the state
# drl_agent's own ObservationBuilder produces). ``baseline_tqc`` and
# ``legacy_waypoint_tqc`` MUST match this exactly (87D / 327D temporal) --
# see docs/IMPLEMENTATION_PLAN.md and tests/test_baseline_observation_parity.py.
ROBOT_STATE_DIM_LEGACY_PARITY = 7
# section P1-11: a 3rd previous-action component (L, for
# ``[kappa, v_ref, L]``, or the hybrid legacy contract's own yield) is a
# DELIBERATE, OPT-IN research extension beyond drl_agent's baseline -- a
# real MDP-completeness improvement for a package whose central research
# question is specifically about L, but never silently applied to a
# profile that's supposed to be parity-comparable with drl_agent (section
# P0-4: "8D observation이 연구상 필요하면 명시적인 config feature/version으로
# 분리하고 baseline에 몰래 적용하지 않는다"). Selected per-profile via
# ``observation.robot_state_dim`` (7 or 8 -- see ObservationConfig.validate),
# never a bare global default.
ROBOT_STATE_DIM_WITH_L_MEMORY = 8


@dataclass(frozen=True)
class RobotState:
    x: float
    y: float
    yaw: float
    v: float
    yaw_rate: float
    steering: float


def build_robot_state_vector(
    robot: RobotState, goal_x: float, goal_y: float, prev_action_01,
    robot_state_dim: int = ROBOT_STATE_DIM_LEGACY_PARITY,
) -> np.ndarray:
    """``robot_state_dim`` (from ``profile.observation.robot_state_dim``)
    selects between drl_agent's 7D baseline contract (the default, for
    ``baseline_tqc``/``legacy_waypoint_tqc`` parity) and the 8D
    L-memory-extended contract (``ROBOT_STATE_DIM_WITH_L_MEMORY`` --
    ``kinodynamic_tqc*`` profiles opt in explicitly via their own YAML).
    Any other value is rejected by ``ObservationConfig.validate()`` before
    this function is ever reached."""
    dist, heading_err = goal_distance_and_heading(robot.x, robot.y, robot.yaw, goal_x, goal_y)
    prev_a0 = float(prev_action_01[0]) if len(prev_action_01) > 0 else 0.0
    prev_a1 = float(prev_action_01[1]) if len(prev_action_01) > 1 else 0.0
    tail = [dist, heading_err, prev_a0, prev_a1]
    if robot_state_dim >= ROBOT_STATE_DIM_WITH_L_MEMORY:
        prev_a2 = float(prev_action_01[2]) if len(prev_action_01) > 2 else 0.0
        tail.append(prev_a2)
    tail.extend([robot.v, robot.yaw_rate, robot.steering])
    return np.array(tail, dtype=np.float32)


def build_observation(lidar_frame: np.ndarray, robot_state_vector: np.ndarray) -> np.ndarray:
    """lidar_frame may already be temporally-stacked (see sensing/temporal_stack.py) --
    this function doesn't care, it only concatenates."""
    return np.concatenate([
        np.asarray(lidar_frame, dtype=np.float32).reshape(-1),
        np.asarray(robot_state_vector, dtype=np.float32).reshape(-1),
    ])
