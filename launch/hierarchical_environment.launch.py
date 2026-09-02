#!/usr/bin/env python3
"""``ros2 launch hunter_kinodynamic_rl hierarchical_environment.launch.py profile:=hierarchical_phase4 goal_x:=10.0 goal_y:=0.0``

Launches ONLY the Phase 4 live-Gazebo hierarchical navigation adapter
(``nodes/hierarchical_environment_node.py``) -- the single-mission
inference/deployment node: Global candidate selection + frozen Local TQC,
driven continuously against whatever Gazebo world is ALREADY running (this
launch file does not start Gazebo itself -- see
``environment.launch.py``'s identical "no combined launch file" convention).

Requires, in order, in separate terminals:

  1. ``ros2 launch hunter_se_gazebo simulate_hunter_se_ignition.launch.py rviz:=false``
  2. THIS launch file.

``local_checkpoint_dir``/``local_checkpoint_name`` (empty by default --
falls back to ``profile``'s own ``hierarchical_training`` section) let a
caller point at a specific frozen Local checkpoint without editing the
profile file."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    profile_arg = DeclareLaunchArgument("profile", default_value="hierarchical_phase4")
    local_checkpoint_dir_arg = DeclareLaunchArgument("local_checkpoint_dir", default_value="")
    local_checkpoint_name_arg = DeclareLaunchArgument("local_checkpoint_name", default_value="")
    global_checkpoint_dir_arg = DeclareLaunchArgument("global_checkpoint_dir", default_value="")
    global_checkpoint_name_arg = DeclareLaunchArgument("global_checkpoint_name", default_value="final")
    goal_x_arg = DeclareLaunchArgument("goal_x", default_value="10.0")
    goal_y_arg = DeclareLaunchArgument("goal_y", default_value="0.0")

    hierarchical_environment_node = Node(
        package="hunter_kinodynamic_rl",
        executable="hierarchical_environment_node.py",
        name="hierarchical_environment_node",
        output="screen",
        parameters=[{
            "profile": LaunchConfiguration("profile"),
            "local_checkpoint_dir": LaunchConfiguration("local_checkpoint_dir"),
            "local_checkpoint_name": LaunchConfiguration("local_checkpoint_name"),
            "global_checkpoint_dir": LaunchConfiguration("global_checkpoint_dir"),
            "global_checkpoint_name": LaunchConfiguration("global_checkpoint_name"),
            "goal_x": LaunchConfiguration("goal_x"),
            "goal_y": LaunchConfiguration("goal_y"),
        }],
    )

    return LaunchDescription([
        profile_arg, local_checkpoint_dir_arg, local_checkpoint_name_arg,
        global_checkpoint_dir_arg, global_checkpoint_name_arg, goal_x_arg, goal_y_arg,
        hierarchical_environment_node,
    ])
