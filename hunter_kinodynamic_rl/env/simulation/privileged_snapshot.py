"""Pure construction of the privileged, robot-LOCAL-frame obstacle snapshot
risk assessment consumes.

Kept separate from ``environment_node.py`` (a single, directly-testable
source of truth) for two reasons found by code review:

1. The static obstacles placed by ``env/scenarios/procedural_generator.py``
   were previously OMITTED from the privileged snapshot entirely (only
   dynamic obstacles were converted) -- every risk label in a static-only
   scene (the common case: most scenarios have zero dynamic obstacles) was
   silently ``risk_target=0`` regardless of how close the robot's rollout
   passed a wall or box.
2. The snapshot must be built from a SPECIFIC instant's robot pose +
   obstacle positions -- environment_node.py must capture it BEFORE
   publishing a command / advancing physics (the PRE-ACTION state the
   action was actually chosen from), never from live instance attributes
   that have since been mutated by the same step's physics advance.

Privileged obstacle ground truth (position, velocity, true radius) is
ONLY ever used here, to build the SUPERVISED risk-critic target and
counterfactual candidate ranking -- never appended to the policy
observation (see risk/labels.py's module docstring for the same rule).
"""

from __future__ import annotations

from typing import List

from hunter_kinodynamic_rl.common.geometry import to_robot_frame
from hunter_kinodynamic_rl.env.scenarios.procedural_generator import DynamicObstacleSpec, StaticObstacle
from hunter_kinodynamic_rl.risk.labels import DynamicObstacle


def build_privileged_obstacles(
    robot_x: float, robot_y: float, robot_yaw: float,
    static_obstacles: List[StaticObstacle],
    dynamic_specs: List[DynamicObstacleSpec],
) -> List[DynamicObstacle]:
    """Static obstacles: ``vx=vy=0``, the scenario's own conservative
    ``radius`` (the same radius used for collision-free placement -- see
    ``procedural_generator.generate_scenario``'s clearance checks, and
    ``docs/TRACTOR_TQC_MODEL_SPEC.md``'s note on the gap between this radius and the
    actual spawned Gazebo asset geometry). Dynamic obstacles: world-frame
    ``(x0, y0, vx, vy)`` rotated (not translated -- velocity is a vector)
    into the robot's CURRENT heading frame."""
    out: List[DynamicObstacle] = []
    for obstacle in static_obstacles:
        lx, ly = to_robot_frame(obstacle.x - robot_x, obstacle.y - robot_y, robot_yaw)
        out.append(DynamicObstacle(
            x0=lx, y0=ly, vx=0.0, vy=0.0, radius=obstacle.radius, cause=0,
        ))
    for spec in dynamic_specs:
        lx, ly = to_robot_frame(spec.x0 - robot_x, spec.y0 - robot_y, robot_yaw)
        lvx, lvy = to_robot_frame(spec.vx, spec.vy, robot_yaw)
        out.append(DynamicObstacle(
            x0=lx, y0=ly, vx=lvx, vy=lvy, radius=spec.radius, cause=1,
        ))
    return out
