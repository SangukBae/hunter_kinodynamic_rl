#!/usr/bin/env python3
"""Run the Nav2-MPPI classical baseline (docs/RESEARCH_PROTOCOL.md section
34/35) against the SAME fixed benchmark scenarios ``evaluation/
benchmark_runner.py`` runs the RL agents against, producing the SAME
``evaluation/metrics.py::aggregate()``-shaped summary for a direct,
apples-to-apples comparison table.

Architecturally different from ``benchmark_runner.py`` in one fundamental
way: the RL agent is driven through environment_node.py's own ``/step`` RPC
(one discrete action -> one physics tick), but Nav2's MPPI controller drives
``/cmd_vel`` CONTINUOUSLY and autonomously via its own control loop once a
``navigate_to_pose`` action goal is sent -- there is no discrete "step" to
call into. This runner therefore:

  1. Places each scenario via the SAME mechanism benchmark_runner.py uses
     (``EnvironmentClient.set_scenario_override`` + ``reset()``) so obstacle/
     robot placement is byte-identical to the RL evaluation runs.
  2. Sends ONE ``navigate_to_pose`` action goal (start -> scenario goal) via
     ``nav2_msgs/action/NavigateToPose`` and polls the action's status
     instead of stepping.
  3. Detects collision INDEPENDENTLY from live ``/scan`` -- reusing the
     EXACT SAME pure functions environment_node.py's own collision check
     uses (``sensing.scan_processor.front_and_full_state`` for the nearest-
     obstacle distance, ``risk.boundary.distance_to_boundary_m`` for the
     world-edge distance, ``robot.collision_radius_m + observation.
     collision_margin_m`` for the threshold) rather than re-deriving the
     formula, since environment_node.py's own collision check only runs
     inside its ``/step`` handler, which nothing calls here.

``min_clearance_m`` here is a REALIZED, INSTANTANEOUS measurement (live
nearest-obstacle distance minus the robot's collision radius, sampled every
control tick) -- NOT directly the same quantity as ``benchmark_runner.py``'s
``min_clearance_m`` (the RL policy's own PREDICTED forward-rollout clearance
for the action it just committed to). Both are genuine "how close did/would
this get to an obstacle" measurements in the same units (meters), but one is
realized-instantaneous and the other is predicted-forward -- worth noting
when reading a comparison table, not a like-for-like risk-model output.
Nav2 has no counterfactual/TTC-rollout equivalent at all, so
``ttc_values_sec``/``risk_valid_steps``/``collision_free_steps`` are left
unset -- ``metrics.aggregate()`` reports those columns as "no data"
(``*_valid_count: 0``), not zero.

LIVE-VERIFICATION STATUS (honest, as of this package's last live-Gazebo
session -- see docs/RESEARCH_PROTOCOL.md): the full map-free Nav2 stack
(config/nav2_mppi/nav2_mppi_params.yaml + launch/nav2_mppi.launch.py) DOES
configure/activate cleanly against this exact simulation, accepts
navigate_to_pose goals with correctly-placed scenario coordinates, and this
runner's independent collision/telemetry pipeline DOES produce real,
plausible measurements end-to-end (verified: a genuine near-collision was
recorded at 0.05 m worst clearance, matching environment_node.py's own
collision math). Four real integration bugs were found and fixed live in
this exact process: a QoS mismatch that silently starved the /scan
subscription, an empty default_nav_to_pose_bt_xml that crashes the BT with
"Empty Tree", environment_node.py's world being left PAUSED between /step
calls (Nav2 needs it running continuously, not stepped), and a
use_sim_time mismatch between Nav2's clock and Gazebo-bridged /odometry's
sim-time stamps. What was NOT achieved in the live attempts made: a full
episode reaching the scenario goal within its timeout -- average commanded
velocity in the last live run was implausibly low (~0.03 m/s vs. this
robot's 2.0 m/s max), a genuine unresolved tuning/performance question
(possibly MPPI critic weights inherited from scout_nav2's much slower
platform, possibly Gazebo real-time-factor degradation from a long-running
session) that further live iteration would be needed to root-cause. Treat
this baseline as CODE-COMPLETE and PARTIALLY LIVE-VERIFIED, not as a
confirmed-working navigation stack -- unlike every other live-Gazebo claim
in this package's history, "the robot successfully reached a benchmark
goal via Nav2-MPPI" has not been observed.
"""

from __future__ import annotations

import math
import os
import time
from typing import TYPE_CHECKING, List, Optional

from hunter_kinodynamic_rl.config.schema import Profile
from hunter_kinodynamic_rl.env.scenarios.benchmark_loader import BenchmarkScenario, load_benchmark
from hunter_kinodynamic_rl.evaluation import metrics as metrics_mod
from hunter_kinodynamic_rl.evaluation.result_writer import write_episode_csv, write_episode_jsonl, write_summary_json
from hunter_kinodynamic_rl.risk.boundary import distance_to_boundary_m
from hunter_kinodynamic_rl.sensing.scan_processor import front_and_full_state

if TYPE_CHECKING:
    # rclpy-dependent -- deferred so collision_threshold_m/
    # nearest_obstacle_distance_m above stay importable (and unit-testable)
    # on a bare host checkout with no ROS install, matching this package's
    # convention for every other pure-logic module. Nav2MPPIRunner/
    # run_benchmark below import EnvironmentClient lazily at call time.
    from hunter_kinodynamic_rl.training.trainer_base import EnvironmentClient


def collision_threshold_m(profile: Profile) -> float:
    """The EXACT SAME formula environment_node.py's own
    ``_collision_threshold_m`` uses (env/simulation/environment_node.py) --
    duplicated here (not imported) because that method lives on the node
    class itself; both sides read the identical two config fields."""
    return profile.robot.collision_radius_m + profile.observation.collision_margin_m


def navigation_time_sec(episode_elapsed_sim_time_sec: Optional[float], wall_start: float, wall_now: float) -> float:
    """section P1-9: prefer /clock-derived SIMULATION time (matching
    benchmark_runner.py's RL-agent metric exactly) over wall-clock time,
    which silently diverges from it whenever Gazebo's real-time-factor
    isn't exactly 1.0. Falls back to wall time only when /clock was never
    received at all (``episode_elapsed_sim_time_sec`` is None, e.g.
    use_sim_time off)."""
    if episode_elapsed_sim_time_sec is not None:
        return episode_elapsed_sim_time_sec
    return wall_now - wall_start


def nearest_obstacle_distance_m(ranges, angle_min: float, angle_increment: float,
                                 profile: Profile, robot_x: float, robot_y: float) -> float:
    """The SAME 360-degree-binned-minimum + world-boundary formula
    environment_node.py's ``/step`` handler uses for ``min_obstacle_dist``
    -- see this module's docstring for why it can't just call that method
    directly (nothing here ever calls ``/step``)."""
    _obs_state, environment_state = front_and_full_state(
        ranges, angle_min, angle_increment, profile.observation.lidar_bins,
        profile.observation.lidar_max_range_m, profile.observation.front_sector_width_rad,
    )
    nearest = float(environment_state.min()) if environment_state.size else float("inf")
    half_extent = profile.scenario.world_size_m / 2.0
    boundary_dist = distance_to_boundary_m(robot_x, robot_y, half_extent)
    return min(nearest, boundary_dist)


class Nav2MPPIRunner:
    """Wraps an already-constructed :class:`EnvironmentClient` (for scenario
    placement + odometry/joint-state telemetry, exactly like
    ``benchmark_runner.run_episode``'s ``env`` argument) with a
    ``NavigateToPose`` action client and a standalone ``/scan`` collision
    monitor. Constructed lazily (see :meth:`_ensure_ros_wiring`) so this
    module stays importable (and its pure helper functions above stay unit-
    testable) without rclpy/nav2_msgs on a bare host checkout."""

    def __init__(self, env: EnvironmentClient, profile: Profile, world_name: str = "default"):
        self.env = env
        self.profile = profile
        self.world_name = world_name
        self._latest_scan = None  # (ranges, angle_min, angle_increment)
        self._min_obstacle_dist_seen = float("inf")
        self._nav_client = None
        self._scan_sub = None
        self._world_control_client = None
        self._ensure_ros_wiring()

    def _ensure_ros_wiring(self) -> None:
        from nav2_msgs.action import NavigateToPose
        from rclpy.action import ActionClient
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from ros_gz_interfaces.srv import ControlWorld
        from sensor_msgs.msg import LaserScan

        self._nav_client = ActionClient(self.env, NavigateToPose, "navigate_to_pose")
        # Same raw Gazebo world-control service environment_node.py's own
        # GazeboRuntimeMixin uses (env/simulation/gazebo_runtime.py) -- NOT
        # exclusive to that node, any client can call it. Needed because
        # environment_node.py keeps Gazebo PAUSED except during its own
        # discrete /step calls (determinism/perf); Nav2's controller_server
        # drives cmd_vel continuously and autonomously at its own
        # controller_frequency and needs the simulation actually running in
        # real time for that whole episode -- found live: with the world
        # left paused (nothing here ever calls /step), /odometry and /tf
        # simply stop publishing new messages entirely, and Nav2's costmaps
        # time out waiting on a transform that will never update.
        self._world_control_client = self.env.create_client(ControlWorld, f"/world/{self.world_name}/control")

        def _on_scan(msg: LaserScan) -> None:
            self._latest_scan = (msg.ranges, msg.angle_min, msg.angle_increment)

        # MUST match environment_node.py's own /scan QoS (BEST_EFFORT,
        # env/simulation/environment_node.py) -- found live: the default
        # RELIABLE subscription QoS is incompatible with pointcloud_to_
        # laserscan's BEST_EFFORT publisher, so this subscription silently
        # never received a single message (no error, just permanently empty
        # _latest_scan -- every episode's collision check was a no-op).
        scan_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._scan_sub = self.env.create_subscription(LaserScan, "/scan", _on_scan, scan_qos)

    def _set_world_paused(self, paused: bool) -> None:
        import rclpy
        from ros_gz_interfaces.srv import ControlWorld

        if not self._world_control_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(f"/world/{self.world_name}/control service unavailable")
        req = ControlWorld.Request()
        req.world_control.pause = bool(paused)
        future = self._world_control_client.call_async(req)
        rclpy.spin_until_future_complete(self.env, future, timeout_sec=5.0)

    def _sample_min_obstacle_dist(self) -> Optional[float]:
        if self._latest_scan is None or self.env.latest_pose is None:
            return None
        ranges, angle_min, angle_increment = self._latest_scan
        x, y, _yaw = self.env.latest_pose
        return nearest_obstacle_distance_m(ranges, angle_min, angle_increment, self.profile, x, y)

    def run_episode(self, scenario: BenchmarkScenario, goal_timeout_sec: float,
                     poll_period_sec: float = 0.2) -> dict:
        """Caller MUST call ``env.set_scenario_override(<scenario's YAML
        path>)`` before invoking this -- see :func:`run_benchmark`, the only
        production call site (mirrors benchmark_runner.run_episode's own
        contract)."""
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from nav2_msgs.action import NavigateToPose

        state = self.env.reset()  # places the scenario; state itself unused (no RL policy here)
        del state
        self._latest_scan = None
        self._min_obstacle_dist_seen = float("inf")
        # /reset leaves the world PAUSED (environment_node.py's normal
        # between-steps convention) -- Nav2 needs it running continuously
        # for this whole episode, see _ensure_ros_wiring's comment.
        self._set_world_paused(False)

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = "odom"
        goal_msg.pose.pose.position.x = scenario.spec.goal_x
        goal_msg.pose.pose.position.y = scenario.spec.goal_y
        # Straight-line heading start->goal -- success in this whole research
        # protocol is position-only (reward_calculator.is_goal_reached never
        # checks heading), so any reasonable orientation is fine; this just
        # avoids handing MPPI/the goal-angle critic an arbitrary/backwards
        # target heading to additionally chase.
        heading = math.atan2(scenario.spec.goal_y - scenario.spec.start_y,
                              scenario.spec.goal_x - scenario.spec.start_x)
        goal_msg.pose.pose.orientation.z = math.sin(heading / 2.0)
        goal_msg.pose.pose.orientation.w = math.cos(heading / 2.0)

        if not self._nav_client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError("navigate_to_pose action server unavailable -- is nav2_mppi.launch.py running?")

        send_future = self._nav_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self.env, send_future, timeout_sec=10.0)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError(f"navigate_to_pose goal rejected for scenario {scenario.scenario_id!r}")

        result_future = goal_handle.get_result_async()
        start_wall = time.monotonic()
        collided = False
        success = False
        clearance_series: List[float] = []
        velocities: List[float] = []
        steering_values: List[float] = []
        threshold = collision_threshold_m(self.profile)

        # Spin with a SHORT timeout so callbacks (odometry, scan, the result
        # future) are processed promptly, but only SAMPLE at poll_period_sec
        # cadence -- found live: spinning with timeout_sec=poll_period_sec
        # directly returns almost immediately whenever a message is already
        # pending (which it constantly is, at these topics' real rates), so
        # `steps` ballooned to tens of thousands of near-instant iterations
        # instead of a count comparable in scale to the RL system's own
        # fixed-size episode_length_steps -- decoupling spin cadence from
        # sample cadence fixes both the busy-spin CPU waste and the metric.
        last_sample_wall = 0.0
        while True:
            rclpy.spin_once(self.env, timeout_sec=0.05)
            now_wall = time.monotonic()
            if now_wall - last_sample_wall < poll_period_sec:
                continue
            last_sample_wall = now_wall

            dist = self._sample_min_obstacle_dist()
            if dist is not None:
                clearance_series.append(dist - self.profile.robot.collision_radius_m)
                if dist < threshold:
                    collided = True
                    break
            velocities.append(self.env.latest_v_mps)
            steering_values.append(self.env.latest_center_steering_rad)

            if result_future.done():
                success = bool(result_future.result().status == 4)  # GoalStatus.STATUS_SUCCEEDED
                break
            if now_wall - start_wall > goal_timeout_sec:
                goal_handle.cancel_goal_async()
                break

        # Restore environment_node.py's normal paused-between-episodes
        # convention -- the NEXT env.reset() (a subsequent scenario, or a
        # different consumer of this same environment_node entirely) should
        # not find the world spuriously still running.
        self._set_world_paused(True)

        timeout = not (success or collided)
        start = (scenario.spec.start_x, scenario.spec.start_y)
        goal = (scenario.spec.goal_x, scenario.spec.goal_y)
        straight_line = math.hypot(goal[0] - start[0], goal[1] - start[1])
        finite_clearance = [c for c in clearance_series if math.isfinite(c)]

        return {
            "scenario_id": scenario.scenario_id, "success": success, "collision": collided,
            "timeout": timeout, "unrecoverable": False,
            "steps": len(velocities), "path_length_m": self.env.episode_path_length_m,
            "straight_line_distance_m": straight_line,
            "navigation_time_sec": navigation_time_sec(
                self.env.episode_elapsed_sim_time_sec, start_wall, time.monotonic()),
            "velocities_mps": velocities,
            "min_clearance_m": min(finite_clearance) if finite_clearance else None,
            "min_clearance_valid_count": len(finite_clearance),
            "ttc_values_sec": [], "steering_values_rad": steering_values,
            "steering_limit_rad": self.profile.robot.steering_limit_rad, "control_deltas": [],
            "emergency_stops": 0, "total_reward": 0.0,
            "risk_valid_steps": 0, "collision_free_steps": 0,
        }


def run_benchmark(profile: Profile, output_dir: str, env: EnvironmentClient = None,
                   goal_timeout_sec: Optional[float] = None) -> dict:
    """Mirrors ``benchmark_runner.run_benchmark``'s contract exactly (same
    output files, same summary shape) -- see that function's docstring."""
    from hunter_kinodynamic_rl.config.loader import default_config_root
    from hunter_kinodynamic_rl.training.trainer_base import EnvironmentClient
    import glob
    import yaml

    benchmark_dir = os.path.join(default_config_root(), "benchmarks", profile.evaluation.benchmark)
    scenario_paths = {}
    for path in sorted(glob.glob(os.path.join(benchmark_dir, "*.yaml"))):
        with open(path) as f:
            sid = (yaml.safe_load(f) or {}).get("scenario_id", os.path.splitext(os.path.basename(path))[0])
        scenario_paths[sid] = path

    scenarios = load_benchmark(profile.evaluation.benchmark)
    owns_env = env is None
    if owns_env:
        env = EnvironmentClient()
    runner = Nav2MPPIRunner(env, profile)
    timeout = goal_timeout_sec or (profile.training.episode_length_steps * profile.runtime.time_delta_sec)
    episodes: List[dict] = []
    try:
        for scenario in scenarios:
            path = scenario_paths.get(scenario.scenario_id)
            if path is None:
                raise RuntimeError(f"could not resolve source path for scenario_id={scenario.scenario_id!r}")
            for _ in range(profile.evaluation.episodes_per_scenario):
                env.set_scenario_override(path)
                episode = runner.run_episode(scenario, goal_timeout_sec=timeout)
                episodes.append(episode)
    finally:
        env.set_scenario_override("")
        if owns_env:
            env.destroy_node()

    summary = metrics_mod.aggregate(episodes)
    write_episode_csv(f"{output_dir}/episodes.csv", episodes)
    write_episode_jsonl(f"{output_dir}/episodes.jsonl", episodes)
    write_summary_json(f"{output_dir}/summary.json", summary)
    return summary
