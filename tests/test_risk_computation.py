"""Regression tests for the two P0 bugs code review found in
environment_node.py's risk pipeline:

1. Static obstacles were OMITTED from the privileged snapshot entirely
   (only dynamic obstacles were converted) -- every risk label in a
   static-only scene was silently risk_target=0 regardless of proximity.
2. The risk label was computed from the POST-action (state_t+1) robot pose
   and obstacle positions instead of the PRE-action (state_t) snapshot the
   action was actually chosen from.

Both are exercised here WITHOUT any ROS/Gazebo dependency -- see
env/simulation/risk_computation.py's module docstring for why this is a
pure function in the first place.
"""

import dataclasses

import pytest

from hunter_kinodynamic_rl.config.loader import load_profile
from hunter_kinodynamic_rl.env.scenarios.procedural_generator import DynamicObstacleSpec, StaticObstacle
from hunter_kinodynamic_rl.env.simulation.risk_computation import (
    compute_risk_telemetry,
    uses_common_evaluation_metrics,
)
from hunter_kinodynamic_rl.trajectory.action_space import TrajectoryCommand


def _risk_profile(world_size_m: float = None):
    profile = load_profile("kinodynamic_tqc_counterfactual")
    if world_size_m is not None:
        # Some tests below place the robot at synthetic poses tens of
        # metres from the origin purely to be "far from the obstacle" --
        # unrelated to the world-boundary risk contribution (section
        # P0-2) this profile's default (small) world_size_m would
        # otherwise also trip. Widening it here isolates those tests'
        # actual subject (pre- vs post-action snapshot selection) from
        # the boundary feature, which has its own dedicated coverage in
        # tests/test_boundary_risk.py.
        profile = dataclasses.replace(
            profile, scenario=dataclasses.replace(profile.scenario, world_size_m=world_size_m))
    return profile


def test_formal_test_pool_and_fixed_scenarios_use_common_metrics():
    assert uses_common_evaluation_metrics(False, "train") is False
    assert uses_common_evaluation_metrics(False, "validation") is False
    assert uses_common_evaluation_metrics(False, "test") is True
    assert uses_common_evaluation_metrics(True, "train") is True


# ------------------------------------------------------- P0-1: static obstacles
def test_static_only_scene_produces_finite_clearance_when_obstacle_is_close():
    profile = _risk_profile()
    command = TrajectoryCommand(kappa=0.0, v_ref=1.0, horizon_m=2.0)
    static_obstacles = [StaticObstacle(x=1.5, y=0.0, radius=0.3)]  # directly ahead, close

    telemetry = compute_risk_telemetry(
        command, step_id=1, robot_pose=(0.0, 0.0, 0.0), robot_v=1.0, robot_steering=0.0,
        static_obstacles=static_obstacles, dynamic_specs=[],
        active_robot_config=profile.robot, profile=profile,
    )
    assert telemetry.valid is True
    assert telemetry.min_clearance_m < 5.0  # finite, not the "no obstacles" infinity


def test_static_only_scene_with_no_obstacles_is_bounded_by_world_wall_not_infinite():
    """With zero obstacles, the ONLY thing bounding min_clearance_m is now
    the world boundary (section P0-2) -- it must be finite, never infinite,
    since the robot is never actually unconstrained in the bounded
    procedural world. Still far from the wall here, so risk_target stays
    ~0 (see tests/test_boundary_risk.py for the case where it doesn't)."""
    profile = _risk_profile()
    command = TrajectoryCommand(kappa=0.0, v_ref=1.0, horizon_m=2.0)

    telemetry = compute_risk_telemetry(
        command, step_id=1, robot_pose=(0.0, 0.0, 0.0), robot_v=1.0, robot_steering=0.0,
        static_obstacles=[], dynamic_specs=[],
        active_robot_config=profile.robot, profile=profile,
    )
    assert telemetry.min_clearance_m < float("inf")
    assert telemetry.min_clearance_m > profile.risk.min_safe_clearance_m
    assert telemetry.risk_target == pytest.approx(0.0)


def test_rollout_into_static_obstacle_is_detected_as_collision_with_nonzero_risk():
    profile = _risk_profile()
    command = TrajectoryCommand(
        kappa=0.0, v_ref=profile.robot.max_forward_speed_mps, horizon_m=3.0
    )  # straight ahead at the manual speed limit
    static_obstacles = [StaticObstacle(x=1.0, y=0.0, radius=0.3)]  # squarely in the path

    telemetry = compute_risk_telemetry(
        command, step_id=1, robot_pose=(0.0, 0.0, 0.0),
        robot_v=profile.robot.max_forward_speed_mps, robot_steering=0.0,
        static_obstacles=static_obstacles, dynamic_specs=[],
        active_robot_config=profile.robot, profile=profile,
    )
    assert telemetry.collision_within_horizon is True
    assert telemetry.risk_target != pytest.approx(0.0), (
        "a static-obstacle collision transition must NOT be stored with risk_target=0"
    )


def test_static_and_dynamic_obstacles_are_both_included_in_one_snapshot():
    profile = _risk_profile()
    command = TrajectoryCommand(kappa=0.0, v_ref=1.0, horizon_m=2.0)
    # Static obstacle far away (irrelevant); dynamic obstacle close and directly ahead.
    static_obstacles = [StaticObstacle(x=10.0, y=10.0, radius=0.3)]
    dynamic_specs = [DynamicObstacleSpec(x0=1.0, y0=0.0, vx=0.0, vy=0.0, radius=0.3)]

    telemetry = compute_risk_telemetry(
        command, step_id=1, robot_pose=(0.0, 0.0, 0.0), robot_v=1.0, robot_steering=0.0,
        static_obstacles=static_obstacles, dynamic_specs=dynamic_specs,
        active_robot_config=profile.robot, profile=profile,
    )
    assert telemetry.collision_within_horizon is True  # from the dynamic obstacle, not the far static one


# --------------------------------------------------- P0-2: pre- vs post-action state
def test_risk_label_differs_between_pre_action_and_post_action_snapshots():
    """Simulates the exact bug: an obstacle is close to the robot's
    PRE-action pose but far from where the robot ends up (state_t+1, e.g.
    after a large step or a teleport-like jump). Using the post-action
    snapshot must NOT reproduce the same (correct) pre-action risk -- if it
    did, this test would be unable to tell the two code paths apart, which
    is exactly the coverage gap that let the original bug through."""
    profile = _risk_profile(world_size_m=1000.0)
    command = TrajectoryCommand(kappa=0.0, v_ref=1.0, horizon_m=2.0)
    obstacle = StaticObstacle(x=1.0, y=0.0, radius=0.3)  # close to the ORIGIN

    pre_action_pose = (0.0, 0.0, 0.0)     # robot at origin when action was chosen
    post_action_pose = (20.0, 20.0, 0.0)  # robot far away after the step advanced

    pre_label = compute_risk_telemetry(
        command, step_id=1, robot_pose=pre_action_pose, robot_v=1.0, robot_steering=0.0,
        static_obstacles=[obstacle], dynamic_specs=[], active_robot_config=profile.robot, profile=profile,
    )
    post_label = compute_risk_telemetry(
        command, step_id=1, robot_pose=post_action_pose, robot_v=1.0, robot_steering=0.0,
        static_obstacles=[obstacle], dynamic_specs=[], active_robot_config=profile.robot, profile=profile,
    )
    assert pre_label.collision_within_horizon is True
    assert post_label.collision_within_horizon is False
    assert pre_label.risk_target != post_label.risk_target


def test_regression_using_post_action_pose_would_hide_a_real_collision():
    """Directly documents the failure mode of the original bug: computing
    risk from the ROBOT'S POST-STEP POSE (as the buggy code did) for an
    obstacle that was actually dangerous at the PRE-action pose silently
    reports "safe" -- this is the exact defect this suite guards against."""
    profile = _risk_profile(world_size_m=1000.0)
    command = TrajectoryCommand(kappa=0.0, v_ref=1.0, horizon_m=2.0)
    obstacle = StaticObstacle(x=0.5, y=0.0, radius=0.3)  # directly in front of the pre-action pose

    correct_label = compute_risk_telemetry(
        command, step_id=5, robot_pose=(0.0, 0.0, 0.0), robot_v=1.0, robot_steering=0.0,
        static_obstacles=[obstacle], dynamic_specs=[], active_robot_config=profile.robot, profile=profile,
    )
    buggy_label = compute_risk_telemetry(
        command, step_id=5, robot_pose=(5.0, 5.0, 0.0), robot_v=1.0, robot_steering=0.0,
        static_obstacles=[obstacle], dynamic_specs=[], active_robot_config=profile.robot, profile=profile,
    )
    assert correct_label.collision_within_horizon is True
    assert buggy_label.collision_within_horizon is False, (
        "sanity check on the test itself: the post-action pose must indeed miss the obstacle, "
        "otherwise this test cannot distinguish correct from buggy behaviour"
    )


def test_features_disabled_returns_invalid_regardless_of_obstacles():
    profile = load_profile("baseline_tqc")  # features.ackermann_rollout=False
    command = TrajectoryCommand(kappa=0.0, v_ref=1.0, horizon_m=2.0)
    telemetry = compute_risk_telemetry(
        command, step_id=1, robot_pose=(0.0, 0.0, 0.0), robot_v=1.0, robot_steering=0.0,
        static_obstacles=[StaticObstacle(x=0.5, y=0.0, radius=0.3)], dynamic_specs=[],
        active_robot_config=profile.robot, profile=profile,
    )
    assert telemetry.valid is False
