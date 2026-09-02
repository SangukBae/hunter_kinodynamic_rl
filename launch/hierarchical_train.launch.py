#!/usr/bin/env python3
"""``ros2 launch hunter_kinodynamic_rl hierarchical_train.launch.py profile:=hierarchical_phase4 live:=true``

Launches ONLY the Phase 4 Global-RL training driver -- mirrors
``environment.launch.py``'s own "no combined launch file" convention
(CLAUDE.md): Gazebo (``hunter_se_gazebo``'s own launch file) is started
separately, in its own terminal, BEFORE this one when ``live:=true``.

``live:=false`` (default) trains the ROS-free
``SimplifiedKinematicLocalExecutor`` path and needs no Gazebo at all --
matches ``training/train_hierarchical_dqn.py``'s own historical usage.
``live:=true`` requires:

  1. ``ros2 launch hunter_se_gazebo simulate_hunter_se_ignition.launch.py rviz:=false``
     already running.
  2. ``profile``'s ``hierarchical_training.local_checkpoint_dir``/
     ``local_checkpoint_name`` resolving to a REAL frozen Local kinodynamic
     TQC checkpoint -- this launch fails fast (``LocalCheckpointError``,
     propagated as a node startup failure) if it does not.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    profile_arg = DeclareLaunchArgument("profile", default_value="hierarchical_phase4")
    run_root_arg = DeclareLaunchArgument("run_root", default_value="runtime/hierarchical_experiments")
    num_missions_arg = DeclareLaunchArgument("num_missions", default_value="1000")
    live_arg = DeclareLaunchArgument(
        "live", default_value="false", choices=["true", "false"],
        description="true: drive the real frozen Local TQC over live Gazebo. false (default): the "
                    "ROS-free SimplifiedKinematicLocalExecutor stand-in, no Gazebo needed.",
    )
    resume_arg = DeclareLaunchArgument("resume", default_value="false", choices=["true", "false"])
    resume_run_dir_arg = DeclareLaunchArgument("resume_run_dir", default_value="")
    resume_checkpoint_tag_arg = DeclareLaunchArgument("resume_checkpoint_tag", default_value="latest")

    train_node = Node(
        package="hunter_kinodynamic_rl",
        executable="hierarchical_train_node.py",
        name="hierarchical_train_node",
        output="screen",
        parameters=[{
            "profile": LaunchConfiguration("profile"),
            "run_root": LaunchConfiguration("run_root"),
            "num_missions": LaunchConfiguration("num_missions"),
            "live": LaunchConfiguration("live"),
            "resume": LaunchConfiguration("resume"),
            "resume_run_dir": LaunchConfiguration("resume_run_dir"),
            "resume_checkpoint_tag": LaunchConfiguration("resume_checkpoint_tag"),
        }],
    )

    return LaunchDescription([
        profile_arg, run_root_arg, num_missions_arg, live_arg,
        resume_arg, resume_run_dir_arg, resume_checkpoint_tag_arg,
        train_node,
    ])
