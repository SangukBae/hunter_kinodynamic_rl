#!/usr/bin/env python3
"""Live-Gazebo smoke verification for LiveGazeboLocalExecutor (item 11.2-11.6).

Constructs the real executor against the already-running headless Gazebo,
binds MULTIPLE missions (non-zero start pose/yaw teleport), runs several
options per mission, and prints diagnostics for:
 - wall pool spawn / world reset success
 - fresh /scan and /odometry after each bind_mission
 - partial map UNKNOWN -> FREE/OCCUPIED growth
 - terminal SubgoalResult status for every option (item 2)
 - option-local clock sanity (item 1)
 - graceful close() with no leftover threads
"""
import sys
import time

import numpy as np
import rclpy

from hunter_kinodynamic_rl.config.loader import load_profile
from hunter_kinodynamic_rl.env.scenarios.long_horizon_generator import (
    LongHorizonSeedScheduler, generate_long_horizon_world,
)
from hunter_kinodynamic_rl.navigation.hierarchy.coordinator import HierarchyCoordinator
from hunter_kinodynamic_rl.navigation.local_rl.live_gazebo_executor import LiveGazeboLocalExecutor
from hunter_kinodynamic_rl.navigation.mapping.partial_map import PartialMap
from hunter_kinodynamic_rl.navigation.mission.mission_frame import MissionFrame, PoseXYYaw
from hunter_kinodynamic_rl.training.train_hierarchical_dqn import hierarchy_config_from

rclpy.init()
profile = load_profile("hierarchical_phase4")
print(f"[smoke] loaded profile hierarchical_phase4, local_checkpoint_dir="
      f"{profile.hierarchical_training.local_checkpoint_dir!r} name="
      f"{profile.hierarchical_training.local_checkpoint_name!r}", flush=True)

t0 = time.time()
executor = LiveGazeboLocalExecutor(
    profile, profile.long_horizon_world,
    profile.hierarchical_training.local_checkpoint_dir, profile.hierarchical_training.local_checkpoint_name,
)
print(f"[smoke] LiveGazeboLocalExecutor constructed in {time.time()-t0:.2f}s "
      f"(checkpoint generation={executor.checkpoint_manifest.get('generation')}, "
      f"sha256={executor.checkpoint_manifest.get('pt_sha256')})", flush=True)

scheduler = LongHorizonSeedScheduler(4242, profile.long_horizon_world, mode="test")
min_turning_radius_m = 1.0 / profile.robot.max_curvature

results = []
try:
    for mission_index in range(3):
        seed = scheduler.next_seed()
        world = generate_long_horizon_world(
            seed, profile.long_horizon_world, profile.robot.collision_radius_m,
            min_turning_radius_m, profile.robot.wheelbase_m, mode="test",
        )
        mission_frame = MissionFrame()
        mission_frame.initialize(PoseXYYaw(*world.start_pose))
        partial_map = PartialMap(profile.mapping)
        goal_mission = mission_frame.odom_to_mission(*world.goal_pose)

        t_bind = time.time()
        executor.bind_mission(world, mission_frame)
        bind_elapsed = time.time() - t_bind
        pose = executor.pose_world
        print(f"[smoke] mission {mission_index}: seed={seed} start_pose={world.start_pose} "
              f"bind_elapsed={bind_elapsed:.2f}s post-bind pose_world=({pose.x:.3f},{pose.y:.3f},{pose.yaw:.3f}) "
              f"scan_count={executor.scan_update_count} odom_count={executor.odom_update_count}", flush=True)
        # non-zero start pose/yaw sanity
        assert abs(world.start_pose[0]) + abs(world.start_pose[1]) > 0.0 or abs(world.start_pose[2]) > 0.0 \
            or mission_index == 0, "expected at least one non-origin start pose across missions"

        for option_index in range(3):
            pose_mission = mission_frame.odom_pose_to_mission(executor.pose_world)
            coordinator = HierarchyCoordinator(hierarchy_config_from(profile.hierarchy))
            coordinator.start_mission(goal_mission[0], goal_mission[1], now_step=0, now_time_sec=0.0)
            coordinator.enqueue_subgoal(*goal_mission)

            def _is_valid(x, y, _pm=partial_map):
                cell = _pm.world_to_cell(x, y)
                return True if cell is None else not bool(_pm.channels().inflated[cell])

            activated = coordinator.activate_next_subgoal(pose_mission, now_step=0, now_time_sec=0.0, is_valid=_is_valid)
            observed_before = partial_map.observed_count()
            t_opt = time.time()
            max_local_steps = 15
            if activated:
                executor.run_option(coordinator, partial_map, np.random.RandomState(0), max_local_steps)
            opt_elapsed = time.time() - t_opt
            observed_after = partial_map.observed_count()
            result = coordinator.last_subgoal_result
            telemetry = executor.last_option_telemetry
            print(
                f"[smoke]   option {option_index}: activated={activated} wall_elapsed={opt_elapsed:.2f}s "
                f"result={'None' if result is None else result.status.value} "
                f"reason={None if result is None else result.reason} "
                f"local_steps={None if result is None else result.local_steps} "
                f"elapsed_time_sec={None if result is None else round(result.elapsed_time_sec, 3)} "
                f"observed_delta={observed_after - observed_before} "
                f"telemetry_collision_count={telemetry.collision_count if telemetry else None} "
                f"telemetry_source={telemetry.collision_signal_source if telemetry else None} "
                f"inference_timeouts={telemetry.inference_timeout_count if telemetry else None}",
                flush=True,
            )
            results.append({
                "mission": mission_index, "option": option_index, "activated": activated,
                "status": None if result is None else result.status.value,
                "reason": None if result is None else result.reason,
                "elapsed_time_sec": None if result is None else result.elapsed_time_sec,
                "max_local_steps": max_local_steps, "dt_sec": executor.dt_sec,
            })
            if not activated or coordinator.mission_done:
                break
finally:
    t_close = time.time()
    executor.close()
    print(f"[smoke] executor.close() completed in {time.time()-t_close:.2f}s", flush=True)
    if rclpy.ok():
        rclpy.shutdown()

# Sanity: every activated option produced a non-None terminal result (item 2),
# and elapsed_time_sec never exceeds max_local_steps*dt_sec (item 1).
bad = [r for r in results if r["activated"] and r["status"] is None]
clock_violations = [
    r for r in results
    if r["elapsed_time_sec"] is not None and not (0.0 <= r["elapsed_time_sec"] <= r["max_local_steps"] * r["dt_sec"] + 0.5)
]
print(f"[smoke] SUMMARY: {len(results)} options run, {len(bad)} activated-but-no-terminal-result, "
      f"{len(clock_violations)} clock-budget violations", flush=True)
if bad or clock_violations:
    print("[smoke] FAIL", flush=True)
    sys.exit(1)
print("[smoke] PASS", flush=True)
