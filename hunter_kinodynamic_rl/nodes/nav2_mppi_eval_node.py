#!/usr/bin/env python3
"""``ros2 run hunter_kinodynamic_rl nav2_mppi_eval_node.py --ros-args -p profile:=evaluation_id``

Runs the Nav2-MPPI classical baseline (section 34/35) against the profile's
fixed benchmark, mirroring evaluation_node.py's contract exactly (same
profile format, same output layout) but with NO agent/checkpoint -- Nav2
drives the robot, not a trained policy. REQUIRES ``nav2_mppi.launch.py`` (or
equivalent) already running alongside Gazebo + environment_node.py -- this
node only sends navigate_to_pose goals and monitors telemetry, it never
launches Nav2 itself.
"""

import json
import os

import rclpy

from hunter_kinodynamic_rl.config.loader import load_profile
from hunter_kinodynamic_rl.evaluation.nav2_mppi_runner import run_benchmark
from hunter_kinodynamic_rl.training.trainer_base import EnvironmentClient


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", "-p", dest="profile_kv", action="append", default=[])
    args, _ = parser.parse_known_args()

    kv = {p.split(":=", 1)[0]: p.split(":=", 1)[1] for p in args.profile_kv if ":=" in p}
    profile_name = kv.get("profile", "evaluation_id")
    output_root = kv.get("output_dir")

    profile = load_profile(profile_name)
    if not profile.evaluation.benchmark:
        raise SystemExit(f"profile {profile_name!r} has no evaluation.benchmark set")

    rclpy.init()
    env = EnvironmentClient(node_name="hunter_kinodynamic_nav2_mppi_eval_client")
    try:
        output_dir = output_root or os.path.join("runtime", "nav2_mppi_baseline", profile.evaluation.benchmark)
        os.makedirs(output_dir, exist_ok=True)
        summary = run_benchmark(profile, output_dir, env=env)
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        env.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
