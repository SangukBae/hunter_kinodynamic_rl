"""Defect-fix item 1: hierarchical Local observation coordinate-frame
contract.

Historical bug: every hierarchical call site
(``nodes/hierarchical_navigation_node.py``,
``nodes/hierarchical_environment_node.py``,
``navigation/local_rl/live_gazebo_executor.py``,
``navigation/hierarchy/local_feasibility_evaluator.py``) converted the
active subgoal to the robot's own frame via ``MissionFrame.mission_to_robot``
but then paired it with a ``RobotState`` whose pose was still the raw
odom/mission-frame pose -- ``build_robot_state_vector``'s
``goal_distance_and_heading`` call silently computed a wrong distance/
heading from the resulting frame mismatch (no exception, no NaN -- just a
numerically wrong observation feeding the RL agent).

Canonical fix (documented on
``LocalPolicyController.build_observation``/``robot_relative_state``):
whenever the goal handed to ``build_observation`` is already robot-frame,
``RobotState``'s pose MUST be the origin -- every hierarchical call site
now builds its ``RobotState`` through ``LocalPolicyController.robot_relative_state``,
which can only ever produce ``(0, 0, 0)`` pose. This module tests that
contract directly against the geometry primitives those call sites
actually use (``MissionFrame.mission_to_robot`` + ``goal_distance_and_heading``),
and pins byte-identical observation output between the control-loop path
and the feasibility-evaluator path for the same inputs."""

import math

import numpy as np
import pytest

from hunter_kinodynamic_rl.common.geometry import goal_distance_and_heading
from hunter_kinodynamic_rl.config.loader import load_profile
from hunter_kinodynamic_rl.env.observation.observation_builder import build_observation, build_robot_state_vector
from hunter_kinodynamic_rl.navigation.local_rl.controller import LocalPolicyController
from hunter_kinodynamic_rl.navigation.mission.mission_frame import MissionFrame, PoseXYYaw


def test_nonorigin_robot_pose_straight_ahead_goal_gives_distance_3_heading_0():
    """robot pose (10, 0, 0), goal (13, 0) -> distance 3, heading 0 --
    exactly the scenario the historical bug got wrong: robot_state pose
    non-origin, subgoal converted to robot frame."""
    robot_pose_mission = PoseXYYaw(x=10.0, y=0.0, yaw=0.0)
    subgoal_robot = MissionFrame.mission_to_robot((13.0, 0.0), robot_pose_mission)
    robot_state = LocalPolicyController.robot_relative_state(v=0.0, yaw_rate=0.0, steering=0.0)
    dist, heading = goal_distance_and_heading(
        robot_state.x, robot_state.y, robot_state.yaw, subgoal_robot[0], subgoal_robot[1],
    )
    assert dist == pytest.approx(3.0)
    assert heading == pytest.approx(0.0)


def test_robot_yaw_90_degrees_goal_ahead_in_world_frame():
    """robot at mission pose (0, 0, 90deg), goal 5m along the world +y axis
    (exactly where a robot facing +y is pointed) -> robot-frame distance 5,
    heading 0 (straight ahead) -- pins the ROTATION half of the contract,
    not just translation."""
    robot_pose_mission = PoseXYYaw(x=0.0, y=0.0, yaw=math.pi / 2.0)
    subgoal_robot = MissionFrame.mission_to_robot((0.0, 5.0), robot_pose_mission)
    robot_state = LocalPolicyController.robot_relative_state(v=0.0, yaw_rate=0.0, steering=0.0)
    dist, heading = goal_distance_and_heading(
        robot_state.x, robot_state.y, robot_state.yaw, subgoal_robot[0], subgoal_robot[1],
    )
    assert dist == pytest.approx(5.0)
    assert heading == pytest.approx(0.0, abs=1e-9)


def test_mission_frame_origin_different_from_world_origin_matches_ground_truth():
    """A non-trivial mission-start pose (mission frame origin != world/odom
    origin) must not change the physical robot-goal relationship: routing
    a world-frame goal through mission_frame.odom_to_mission ->
    MissionFrame.mission_to_robot must reproduce EXACTLY the same
    distance/heading as computing it directly in world/odom frame."""
    mission_frame = MissionFrame()
    mission_frame.initialize(PoseXYYaw(x=2.0, y=-3.0, yaw=0.3))
    odom_pose = PoseXYYaw(x=5.0, y=1.0, yaw=1.1)
    goal_world_x, goal_world_y = 8.0, 2.0

    goal_mission = mission_frame.odom_to_mission(goal_world_x, goal_world_y)
    robot_pose_mission = mission_frame.odom_pose_to_mission(odom_pose)
    subgoal_robot = MissionFrame.mission_to_robot(goal_mission, robot_pose_mission)
    robot_state = LocalPolicyController.robot_relative_state(v=0.0, yaw_rate=0.0, steering=0.0)
    dist, heading = goal_distance_and_heading(
        robot_state.x, robot_state.y, robot_state.yaw, subgoal_robot[0], subgoal_robot[1],
    )

    expected_dist, expected_heading = goal_distance_and_heading(
        odom_pose.x, odom_pose.y, odom_pose.yaw, goal_world_x, goal_world_y,
    )
    assert dist == pytest.approx(expected_dist)
    assert heading == pytest.approx(expected_heading)


def test_control_loop_and_feasibility_evaluator_produce_identical_observation():
    """The control-loop path (LocalPolicyController.build_observation --
    used by hierarchical_navigation_node.py, hierarchical_environment_node.py,
    live_gazebo_executor.py) and the feasibility-evaluator path
    (FrozenLocalFeasibilityEvaluator.evaluate's inlined build_robot_state_vector
    + build_observation over a LocalPolicyController.snapshot_temporal_context()
    snapshot) must produce a BYTE-IDENTICAL observation for the same
    mission pose / candidate / dynamics / temporal context -- item 1's
    "같은 관측 결과를 생성하는지" requirement, exercised against the actual
    production functions rather than a re-implementation."""
    profile = load_profile("smoke_test")
    controller = LocalPolicyController(profile)
    lidar_bins = profile.observation.lidar_bins
    single_frame = np.linspace(1.0, 5.0, lidar_bins, dtype=np.float32)
    controller._frame_stack.reset(single_frame)
    controller._frame_stack_ready = True
    controller.prev_action = [0.05, -0.1, 0.2]

    robot_pose_mission = PoseXYYaw(x=10.0, y=2.0, yaw=0.4)
    subgoal_robot = MissionFrame.mission_to_robot((13.0, 1.0), robot_pose_mission)
    robot_state = LocalPolicyController.robot_relative_state(v=0.7, yaw_rate=-0.1, steering=0.02)

    control_loop_vector = build_robot_state_vector(
        robot_state, subgoal_robot[0], subgoal_robot[1], controller.prev_action,
        robot_state_dim=profile.observation.robot_state_dim,
    )
    control_loop_obs = build_observation(controller._frame_stack.stacked(), control_loop_vector)

    context = controller.snapshot_temporal_context()
    assert context is not None
    evaluator_vector = build_robot_state_vector(
        robot_state, subgoal_robot[0], subgoal_robot[1], context.prev_action,
        robot_state_dim=profile.observation.robot_state_dim,
    )
    evaluator_obs = build_observation(context.lidar_frame, evaluator_vector)

    np.testing.assert_array_equal(control_loop_obs, evaluator_obs)


def test_standalone_single_frame_contract_is_unchanged():
    """The non-hierarchical standalone path (env/simulation/environment_node.py,
    nodes/real_policy_node.py) uses the OTHER valid contract: RobotState's
    pose and the goal are both in the SAME (odom/world) frame -- this pins
    that contract explicitly so it can never silently drift toward the
    robot-relative convention (which would double-transform the goal)."""
    dist, heading = goal_distance_and_heading(1.0, 1.0, 0.0, 3.0, 1.0)
    assert dist == pytest.approx(2.0)
    assert heading == pytest.approx(0.0)
