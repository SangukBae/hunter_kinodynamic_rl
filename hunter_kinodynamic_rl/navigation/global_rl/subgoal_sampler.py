"""Discrete robot-relative candidate-subgoal grid (plan section 8.3/9).

:func:`build_candidate_set` returns a FIXED-length, FIXED-ORDER tuple of
:class:`SubgoalCandidate` -- ``len(direction_degrees) * len(distances_m)``
direction/distance combinations, in row-major (direction outer, distance
inner) order, followed by exactly ONE fallback candidate at the last index
(``config.fallback_index``). This ordering is the contract every other
Global RL module (action mask, observation, network output, replay) indexes
into -- it must never depend on dict iteration order or be re-derived
independently elsewhere.

Candidates are generated ROBOT-RELATIVE (angle=0 is straight ahead, CCW
positive, radius is a straight-line polar distance) -- plan section 9's
"translation invariance/local geometry와 직접 연결" rationale.
:func:`candidate_endpoint_mission` converts one candidate to a MISSION-frame
point given the robot's current mission-frame pose, via
``MissionFrame.robot_to_mission`` (never re-implemented here).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

from hunter_kinodynamic_rl.config.schema import GlobalRLConfig
from hunter_kinodynamic_rl.navigation.mission.mission_frame import MissionFrame, PoseXYYaw

FALLBACK_LABEL_BACKTRACK = "BACKTRACK"
FALLBACK_LABEL_STOP_RECOVERY = "STOP_RECOVERY"


@dataclass(frozen=True)
class SubgoalCandidate:
    index: int
    angle_rad: float
    radius_m: float
    is_fallback: bool
    label: str


def _fallback_candidate(config: GlobalRLConfig, index: int) -> SubgoalCandidate:
    if config.fallback_mode == "stop_recovery":
        return SubgoalCandidate(
            index=index, angle_rad=0.0, radius_m=0.0, is_fallback=True, label=FALLBACK_LABEL_STOP_RECOVERY,
        )
    # "backtrack" -- a short step directly behind the robot (robot-relative
    # angle=pi), never toward whatever direction the other candidates cover.
    return SubgoalCandidate(
        index=index, angle_rad=math.pi, radius_m=float(config.fallback_backtrack_distance_m),
        is_fallback=True, label=FALLBACK_LABEL_BACKTRACK,
    )


def build_candidate_set(config: GlobalRLConfig) -> Tuple[SubgoalCandidate, ...]:
    """Fixed-order candidate tuple: direction-major, distance-minor, then
    the single fallback candidate last. Length is always
    ``config.n_candidates`` -- callers (observation/action-mask/network) may
    rely on this shape being stable across calls for the SAME config."""
    candidates = []
    index = 0
    for deg in config.direction_degrees:
        for dist in config.distances_m:
            candidates.append(SubgoalCandidate(
                index=index, angle_rad=math.radians(float(deg)), radius_m=float(dist),
                is_fallback=False, label=f"dir{deg:g}_dist{dist:g}",
            ))
            index += 1
    candidates.append(_fallback_candidate(config, index))
    return tuple(candidates)


def candidate_endpoint_robot_frame(candidate: SubgoalCandidate) -> Tuple[float, float]:
    """Robot-body-frame (x forward, y left) endpoint of ``candidate``."""
    x = candidate.radius_m * math.cos(candidate.angle_rad)
    y = candidate.radius_m * math.sin(candidate.angle_rad)
    return x, y


def candidate_endpoint_mission(candidate: SubgoalCandidate, robot_pose_mission: PoseXYYaw) -> Tuple[float, float]:
    """Mission-frame endpoint of ``candidate``, given the robot's current
    mission-frame pose -- the "선택 즉시 mission-frame subgoal로 변환" step
    (plan section 9)."""
    point_robot = candidate_endpoint_robot_frame(candidate)
    return MissionFrame.robot_to_mission(point_robot, robot_pose_mission)
