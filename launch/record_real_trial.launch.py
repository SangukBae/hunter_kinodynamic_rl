"""Rosbag2 recording for a real-robot (or live-Gazebo) inference trial
(section P2-15) -- run this ALONGSIDE ``real_policy_node.py``:

    ros2 launch hunter_kinodynamic_rl record_real_trial.launch.py \
        output_dir:=runtime/real_trials/trial_001

Records every topic needed to reconstruct and analyze the trial afterward
without needing to have been watching live (matches the original research
brief's section 42 topic list, adapted to this system's actual topic names
-- ``/odom``/``/ouster/points`` in that list are this package's
``/odometry``/``/ouster/points`` respectively, see CLAUDE.md's Topics
section):
  - ``/ouster/points`` -- the RAW point cloud, preserved for later
    reprocessing even though real_policy_node.py itself only ever consumes
    the derived ``/scan``.
  - ``/scan``, ``/odometry``, ``/hunter_se/joint_states`` -- the SAME
    sensor topics real_policy_node.py itself subscribes to (its exact
    observed input).
  - ``/cmd_vel`` -- the safety-guarded command real_policy_node.py actually
    published each tick (post-guard, what the robot was TOLD to do).
  - ``/cmd_vel_filtered`` -- hunter_se_cmd_prefilter's shaped output (what
    the robot ACTUALLY received after throttle/steering-rate shaping,
    section P0-5's "telemetry must reflect what was actually published").
  - ``/hunter_kinodynamic_rl/real_policy_diagnostics`` -- the RAW policy
    action + goal + nearest-obstacle-distance + emergency-stop flag, BEFORE
    the safety guard -- the brief's "policy action / goal / collision-event
    topics" (real_policy_node.py never publishes risk_telemetry on the real
    path, see that node's module docstring for why; this is its
    replacement diagnostics channel).
  - ``/tf``, ``/tf_static``, ``/clock`` -- full pose history + time base,
    needed to replay the trial's trajectory afterward.

Plain ``ros2 bag record``, no dataset_builder-style custom segmenting/
metadata (that package's own recording pipeline is a separate, unrelated
system for offline-RL dataset collection, not real-robot trial logging).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration

_TOPICS = [
    "/ouster/points", "/scan", "/odometry", "/hunter_se/joint_states",
    "/cmd_vel", "/cmd_vel_filtered", "/hunter_kinodynamic_rl/real_policy_diagnostics",
    "/tf", "/tf_static", "/clock",
]


def generate_launch_description():
    output_dir_arg = DeclareLaunchArgument(
        "output_dir", default_value="runtime/real_trials/trial",
        description="rosbag2 output directory (must not already exist)",
    )
    output_dir = LaunchConfiguration("output_dir")

    record = ExecuteProcess(
        cmd=["ros2", "bag", "record", "-o", output_dir] + _TOPICS,
        output="screen",
    )

    return LaunchDescription([output_dir_arg, record])
