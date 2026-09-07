#!/usr/bin/env python3
"""Live Gazebo verification script for the item-1 (Gazebo physics-step
reality-check) fix -- see docs/verification/README.md
for the full writeup, exact command used to invoke this, and its actual
captured output.

Exercises the REAL, unmocked GazeboRuntimeMixin.verify_physics_step_calibration
/ multi_step_advance code path (env/simulation/gazebo_runtime.py) against a
live, already-running Gazebo Ignition world (launched via
simulate_hunter_se_ignition.launch.py; the shipped drl_arena.world/empty.world
both declare <max_step_size>0.001</max_step_size>, <real_time_factor>1.0</real_time_factor>).

Three live cases:
1. CORRECT declared gazebo_max_step_size_sec (0.001s, matching the real
   world) -- calibration must succeed and report an observed dt matching
   the real physics step.
2. REALISTIC near-miss declared gazebo_max_step_size_sec (0.002s, exactly
   double the real world's 0.001s -- the exact class of misconfiguration
   the pre-fix physics_step_tolerance_sec=0.005 default silently masked,
   since abs(0.002-0.001)=0.001 <= 0.005 -- see RuntimeConfig's own
   docstring in config/schema.py) -- calibration must now fail fast with a
   clear over/under-step error, judged against the NEW, tighter,
   single-step-scaled physics_step_calibration_tolerance_sec (default
   0.0002s), and deterministic evaluation must never proceed.
3. A real 100-step (0.1s) multi_step_advance call, matching a normal
   control-period physics advance.

Prerequisites (run inside the Docker container):
    source /opt/ros/humble/setup.bash
    source /root/DRL_Robot_Path_Planning/ros2_ws/install/setup.bash
    export PYTHONPATH=/root/DRL_Robot_Path_Planning/ros2_ws/src/hunter_kinodynamic_rl:$PYTHONPATH
    # (PYTHONPATH prepend is REQUIRED -- see the
    # hunter_kinodynamic_rl_live_verification_pythonpath memory note: a
    # bare `python3 -m hunter_kinodynamic_rl...` from the workspace root
    # silently resolves the STALE install/ copy, not source edits.)
    ros2 launch hunter_se_gazebo simulate_hunter_se_ignition.launch.py rviz:=false headless:=true &

Then:
    python3 docs/verification/live_physics_calibration_check.py

Exit code 0 iff all three cases behaved as expected.
"""
import sys
import threading
import time

import rclpy
from rclpy.executors import MultiThreadedExecutor

from hunter_kinodynamic_rl.env.simulation.environment_node import KinodynamicEnvironmentNode
from hunter_kinodynamic_rl.env.simulation.gazebo_service_wait import GazeboServiceError, compute_physics_step_count


def wait_for_clock(node, timeout_sec=10.0):
    deadline = time.time() + timeout_sec
    while node._latest_sim_time_sec is None and time.time() < deadline:
        time.sleep(0.1)
    return node._latest_sim_time_sec is not None


def main():
    rclpy.init()
    node = KinodynamicEnvironmentNode()
    # multi_step_advance's _await_future polls via time.sleep(), relying on
    # OTHER executor threads to process the pending service future -- must
    # run under a MultiThreadedExecutor, exactly like production's own
    # main() (see environment_node.py).
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    try:
        ok = wait_for_clock(node)
        print(f"[live] /clock received: {ok}, latest_sim_time_sec={node._latest_sim_time_sec}")
        if not ok:
            print("[live] FAIL: /clock never arrived -- cannot run live check")
            sys.exit(1)

        # multi_step_advance/verify_physics_step_calibration assume the
        # world is PAUSED (Ignition's multi_step semantics: advance exactly
        # N steps then re-pause) -- in production this precondition is
        # established by _on_reset's own unconditional pause_world(True)
        # (see environment_node.py::_on_reset) before any deterministic
        # stepping ever happens. Replicated here since this script never
        # calls /reset.
        node.pause_world(True)
        print("[live] world paused")
        settle_deadline = time.time() + 1.0
        last = node._latest_sim_time_sec
        stable_since = time.time()
        while time.time() < settle_deadline:
            time.sleep(0.02)
            if node._latest_sim_time_sec != last:
                print(f"[live] /clock still draining: {last} -> {node._latest_sim_time_sec}")
                last = node._latest_sim_time_sec
                stable_since = time.time()
            elif time.time() - stable_since > 0.2:
                break
        print(f"[live] /clock stabilized at {node._latest_sim_time_sec} after pause")

        # ---- Case 1: correct declared step size (matches the real world's 0.001s) ----
        node._gazebo_max_step_size_sec = 0.001
        node._clock_confirm_timeout_sec = 2.0
        node._physics_step_tolerance_sec = 0.005
        node._physics_step_calibration_tolerance_sec = 0.0002
        node._physics_step_calibrated = False
        try:
            observed = node.verify_physics_step_calibration()
            print(f"[live] CASE 1 (correct 0.001s declared): calibration PASSED, observed_dt={observed:.6f}s")
            case1_ok = True
        except GazeboServiceError as e:
            print(f"[live] CASE 1 UNEXPECTEDLY FAILED: {e}")
            case1_ok = False

        # ---- Case 2: REALISTIC near-miss declared step size (world is really
        # 0.001s, we declare 0.002s -- exactly double) -- the specific
        # misconfiguration class the pre-fix physics_step_tolerance_sec=0.005
        # default masked (abs(0.002-0.001)=0.001 <= 0.005 passed undetected);
        # the new physics_step_calibration_tolerance_sec=0.0002 must now
        # catch it. ----
        node._gazebo_max_step_size_sec = 0.002
        node._clock_confirm_timeout_sec = 1.0
        node._physics_step_tolerance_sec = 0.005
        node._physics_step_calibration_tolerance_sec = 0.0002
        node._physics_step_calibrated = False
        try:
            observed = node.verify_physics_step_calibration()
            print(f"[live] CASE 2 UNEXPECTEDLY PASSED: observed_dt={observed:.6f}s (should have raised!)")
            case2_ok = False
        except GazeboServiceError as e:
            print(f"[live] CASE 2 (realistic 0.002s declared, real world 0.001s): calibration correctly FAILED: {e}")
            case2_ok = True

        # ---- Case 3: real multi_step_advance for a full control-period advance (0.1s at 0.001s steps = 100 steps) ----
        node._gazebo_max_step_size_sec = 0.001
        node._clock_confirm_timeout_sec = 2.0
        node._physics_step_tolerance_sec = 0.005
        node._physics_step_calibration_tolerance_sec = 0.0002
        n_steps = compute_physics_step_count(0.1, 0.001)
        observed = node.multi_step_advance(n_steps, 0.1, 2.0)
        print(f"[live] CASE 3 (real 100-step 0.1s advance): observed_dt={observed:.6f}s, n_steps={n_steps}")
        case3_ok = abs(observed - 0.1) <= 0.005

        print(f"[live] SUMMARY: case1_ok={case1_ok} case2_ok={case2_ok} case3_ok={case3_ok}")
        result = 0 if (case1_ok and case2_ok and case3_ok) else 1
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=5.0)
    sys.exit(result)


if __name__ == "__main__":
    main()
