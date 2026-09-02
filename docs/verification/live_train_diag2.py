#!/usr/bin/env python3
"""Diagnostic: does constructing a CUDA-resident GlobalDQNAgent BEFORE
LiveGazeboLocalExecutor (i.e. before its background rclpy spin thread
starts) avoid the 'publisher's context is invalid' crash?"""
import rclpy

from hunter_kinodynamic_rl.common.seed import enable_torch_determinism, seed_all
from hunter_kinodynamic_rl.config.loader import load_profile
from hunter_kinodynamic_rl.navigation.global_rl.agent import GlobalDQNAgent
from hunter_kinodynamic_rl.navigation.global_rl.observation import resolve_map_channel_names
from hunter_kinodynamic_rl.navigation.local_rl.live_gazebo_executor import LiveGazeboLocalExecutor
from hunter_kinodynamic_rl.env.scenarios.long_horizon_generator import (
    LongHorizonSeedScheduler, generate_long_horizon_world,
)
from hunter_kinodynamic_rl.navigation.hierarchy.coordinator import HierarchyCoordinator
from hunter_kinodynamic_rl.navigation.mapping.partial_map import PartialMap
from hunter_kinodynamic_rl.navigation.mission.mission_frame import MissionFrame, PoseXYYaw
from hunter_kinodynamic_rl.training.train_hierarchical_dqn import hierarchy_config_from

profile = load_profile("hierarchical_phase4")
seed_all(profile.training.seed)
enable_torch_determinism(warn_only=True)

rclpy.init()

map_channels = len(resolve_map_channel_names(profile.global_rl))
print("[diag2] constructing GlobalDQNAgent on CUDA BEFORE LiveGazeboLocalExecutor...", flush=True)
agent = GlobalDQNAgent(profile.global_rl, map_channels, device=None, max_nodes=0)
print(f"[diag2] agent.device={agent.device}", flush=True)

print("[diag2] constructing LiveGazeboLocalExecutor...", flush=True)
executor = LiveGazeboLocalExecutor(
    profile, profile.long_horizon_world,
    profile.hierarchical_training.local_checkpoint_dir, profile.hierarchical_training.local_checkpoint_name,
)
print("[diag2] executor constructed", flush=True)

scheduler = LongHorizonSeedScheduler(555, profile.long_horizon_world, mode="test")
min_turning_radius_m = 1.0 / profile.robot.max_curvature
world = generate_long_horizon_world(
    scheduler.next_seed(), profile.long_horizon_world, profile.robot.collision_radius_m,
    min_turning_radius_m, profile.robot.wheelbase_m, mode="test",
)
mission_frame = MissionFrame()
mission_frame.initialize(PoseXYYaw(*world.start_pose))
partial_map = PartialMap(profile.mapping)
goal_mission = mission_frame.odom_to_mission(*world.goal_pose)

try:
    executor.bind_mission(world, mission_frame)
    pose_mission = mission_frame.odom_pose_to_mission(executor.pose_world)
    coordinator = HierarchyCoordinator(hierarchy_config_from(profile.hierarchy))
    coordinator.start_mission(goal_mission[0], goal_mission[1], now_step=0, now_time_sec=0.0)
    coordinator.enqueue_subgoal(*goal_mission)
    import numpy as np
    activated = coordinator.activate_next_subgoal(pose_mission, now_step=0, now_time_sec=0.0)
    print(f"[diag2] activated={activated}, running option...", flush=True)
    if activated:
        executor.run_option(coordinator, partial_map, np.random.RandomState(0), 5)
    result = coordinator.last_subgoal_result
    print(f"[diag2] result={None if result is None else result.status.value} -> PASS (no publish crash)", flush=True)
finally:
    executor.close()
    if rclpy.ok():
        rclpy.shutdown()
