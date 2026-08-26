#!/usr/bin/env python3
"""``ros2 launch hunter_kinodynamic_rl environment.launch.py profile:=kinodynamic_tqc``

Launches ONLY the environment node -- Gazebo is launched separately via
``hunter_se_gazebo``'s own launch file (CLAUDE.md: "no combined launch file"
is the established convention in this monorepo; drl_agent's env/trainer
nodes are launched the same way, in separate terminals)."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    profile_arg = DeclareLaunchArgument("profile", default_value="kinodynamic_tqc")
    # section P1-1: this is the Gazebo SDF's <world name="..."> (what every
    # ros_gz service is actually namespaced under, e.g.
    # `/world/default/control`), NOT hunter_se_gazebo's own `world:=` launch
    # argument (which selects a .world FILE, e.g. `drl_arena.world` --
    # confusingly similarly named). Every world file this package's launch
    # convention supports (drl_arena.world, the AWS hospital world, ...)
    # declares `<world name="default">` internally -- see
    # hunter_se_gazebo/launch/simulate_hunter_se_ignition.launch.py's own
    # "Supported worlds" docstring and worlds/drl_arena.world's own comment.
    # A default of "drl_arena" here (the world FILE's name, matching a
    # different launch file's `world:=` argument by coincidence of naming,
    # never the SDF world name) made every `/world/drl_arena/...` service
    # call time out against the real `/world/default/...` services Gazebo
    # actually advertises -- confirmed live.
    world_name_arg = DeclareLaunchArgument("world_name", default_value="default")

    environment_node = Node(
        package="hunter_kinodynamic_rl",
        executable="environment_node.py",
        name="hunter_kinodynamic_environment",
        output="screen",
        parameters=[{
            "profile": LaunchConfiguration("profile"),
            "world_name": LaunchConfiguration("world_name"),
        }],
    )

    return LaunchDescription([profile_arg, world_name_arg, environment_node])
