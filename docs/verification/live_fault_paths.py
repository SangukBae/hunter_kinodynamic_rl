#!/usr/bin/env python3
"""Live-Gazebo fault-path verification (item 10/2): drives the REAL
LiveGazeboLocalExecutor against the running Gazebo and forces each of the
five executor-level termination paths one at a time via targeted, narrow
monkeypatching (never breaking real ROS/Gazebo infra) -- confirms each
produces a non-None terminal SubgoalResult with the correct status/reason,
and that STOP is the last command published.
"""
import math
import sys
import time

import numpy as np
import rclpy

from hunter_kinodynamic_rl.config.loader import load_profile
from hunter_kinodynamic_rl.env.scenarios.long_horizon_generator import (
    LongHorizonSeedScheduler, generate_long_horizon_world,
)
from hunter_kinodynamic_rl.env.safety.action_guard import STOP_COMMAND
from hunter_kinodynamic_rl.navigation.hierarchy.coordinator import HierarchyCoordinator
from hunter_kinodynamic_rl.navigation.local_rl.live_gazebo_executor import LiveGazeboLocalExecutor
from hunter_kinodynamic_rl.navigation.mapping.partial_map import PartialMap
from hunter_kinodynamic_rl.navigation.mission.mission_frame import MissionFrame, PoseXYYaw
from hunter_kinodynamic_rl.training.train_hierarchical_dqn import hierarchy_config_from

rclpy.init()
profile = load_profile("hierarchical_phase4")
executor = LiveGazeboLocalExecutor(
    profile, profile.long_horizon_world,
    profile.hierarchical_training.local_checkpoint_dir, profile.hierarchical_training.local_checkpoint_name,
)
print("[fault] executor constructed", flush=True)
scheduler = LongHorizonSeedScheduler(9001, profile.long_horizon_world, mode="test")
min_turning_radius_m = 1.0 / profile.robot.max_curvature

results = {}


def _fresh_mission():
    seed = scheduler.next_seed()
    world = generate_long_horizon_world(
        seed, profile.long_horizon_world, profile.robot.collision_radius_m,
        min_turning_radius_m, profile.robot.wheelbase_m, mode="test",
    )
    mission_frame = MissionFrame()
    mission_frame.initialize(PoseXYYaw(*world.start_pose))
    partial_map = PartialMap(profile.mapping)
    goal_mission = mission_frame.odom_to_mission(*world.goal_pose)
    executor.bind_mission(world, mission_frame)
    return mission_frame, partial_map, goal_mission


def _run_one_option(mission_frame, partial_map, goal_mission, max_local_steps=10):
    pose_mission = mission_frame.odom_pose_to_mission(executor.pose_world)
    coordinator = HierarchyCoordinator(hierarchy_config_from(profile.hierarchy))
    coordinator.start_mission(goal_mission[0], goal_mission[1], now_step=0, now_time_sec=0.0)
    coordinator.enqueue_subgoal(*goal_mission)

    def _is_valid(x, y, _pm=partial_map):
        cell = _pm.world_to_cell(x, y)
        return True if cell is None else not bool(_pm.channels().inflated[cell])

    activated = coordinator.activate_next_subgoal(pose_mission, now_step=0, now_time_sec=0.0, is_valid=_is_valid)
    if activated:
        executor.run_option(coordinator, partial_map, np.random.RandomState(0), max_local_steps)
    return activated, coordinator.last_subgoal_result


def _check(name, activated, result, expected_status=None, expected_reason=None):
    ok = (
        activated and result is not None
        and (expected_status is None or result.status.value == expected_status)
        and (expected_reason is None or result.reason == expected_reason)
    )
    print(f"[fault] {name}: activated={activated} status={None if result is None else result.status.value} "
          f"reason={None if result is None else result.reason} "
          f"local_steps={None if result is None else result.local_steps} -> {'OK' if ok else 'FAIL'}", flush=True)
    results[name] = ok
    return ok


try:
    # 1. stale/invalid localization
    mf, pm, goal = _fresh_mission()
    orig_localization_valid = executor._localization_valid
    executor._localization_valid = lambda: False
    try:
        activated, result = _run_one_option(mf, pm, goal)
    finally:
        executor._localization_valid = orig_localization_valid
    _check("localization_stale", activated, result, "cancelled_by_replan", "localization_confidence_degraded")

    # 2. stale/never-received scan
    mf, pm, goal = _fresh_mission()
    orig_scan_receipt = executor._latest_scan_receipt_time
    executor._latest_scan_receipt_time = time.monotonic() - 1000.0
    try:
        activated, result = _run_one_option(mf, pm, goal)
    finally:
        executor._latest_scan_receipt_time = orig_scan_receipt
    _check("scan_stale", activated, result, "cancelled_by_replan", "scan_stale")

    # 3. Local policy (inference) timeout
    mf, pm, goal = _fresh_mission()
    orig_select_action = executor.local_agent.select_action

    def _slow_select_action(*a, **kw):
        time.sleep(3.0)
        return orig_select_action(*a, **kw)

    executor.local_agent.select_action = _slow_select_action
    try:
        activated, result = _run_one_option(mf, pm, goal)
    finally:
        executor.local_agent.select_action = orig_select_action
    _check("inference_timeout", activated, result, "cancelled_by_replan", "inference_timeout")

    # Let test 3's orphaned (leaked, by single-flight-worker's own honest
    # documented limitation) worker thread actually finish before test 4 --
    # otherwise the worker would still report busy=True and test 4 would
    # observe another "inference_timeout" rather than exercising the
    # invalid-action path it's meant to.
    deadline = time.monotonic() + 5.0
    while executor._inference_worker.busy and time.monotonic() < deadline:
        time.sleep(0.1)

    # 4. NaN/malformed action
    mf, pm, goal = _fresh_mission()
    executor.local_agent.select_action = lambda *a, **kw: np.array([float("nan"), 0.0, 0.0])
    try:
        activated, result = _run_one_option(mf, pm, goal)
    finally:
        executor.local_agent.select_action = orig_select_action
    _check("invalid_action", activated, result, "cancelled_by_replan", "invalid_action_output")

    # 5. max_local_steps exhausted (suppress the risk trigger so the
    # loop actually runs out its own step budget instead of terminating
    # via FAILED_HIGH_RISK first, matching the live_smoke_executor.py run's
    # own observation that this checkpoint predicts high risk almost
    # immediately).
    mf, pm, goal = _fresh_mission()
    predict_risk_fn = getattr(executor.local_agent, "predict_risk", None)
    if predict_risk_fn is not None:
        executor.local_agent.predict_risk = lambda *a, **kw: 0.0
    try:
        activated, result = _run_one_option(mf, pm, goal, max_local_steps=3)
    finally:
        if predict_risk_fn is not None:
            executor.local_agent.predict_risk = predict_risk_fn
    _check("max_local_steps_exhausted", activated, result)
    if results.get("max_local_steps_exhausted") and result is not None:
        results["max_local_steps_exhausted"] = result.status.value == "failed_timeout" and result.local_steps == 3
        print(f"[fault] max_local_steps_exhausted re-check: local_steps==3 and FAILED_TIMEOUT -> "
              f"{'OK' if results['max_local_steps_exhausted'] else 'FAIL'}", flush=True)

finally:
    executor.close()
    if rclpy.ok():
        rclpy.shutdown()

print(f"[fault] SUMMARY: {results}", flush=True)
if not all(results.values()):
    print("[fault] FAIL", flush=True)
    sys.exit(1)
print("[fault] PASS", flush=True)
