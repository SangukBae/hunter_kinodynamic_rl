"""Nav2-MPPI classical baseline bring-up (section 34/35) -- controller_server
(RotationShim + MPPI, Ackermann motion model), planner_server (NavFn),
behavior_server (recovery actions the default BT falls back to), bt_navigator
(navigate_to_pose action interface), and their lifecycle_manager. Map-free
(no AMCL, no map_server -- see config/nav2_mppi/nav2_mppi_params.yaml's
header comment for why): assumes Gazebo + hunter_se_gazebo (odom ->
base_footprint TF, /odometry, /scan) are ALREADY running, exactly like
environment_node.py itself does not launch Gazebo.

    ros2 launch hunter_kinodynamic_rl nav2_mppi.launch.py
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    params_file = os.path.join(
        get_package_share_directory("hunter_kinodynamic_rl"), "config", "nav2_mppi", "nav2_mppi_params.yaml",
    )
    # Found live: /odometry and /tf are bridged straight from Gazebo and
    # ALWAYS carry simulation-time stamps (small values like sec=37, not a
    # wall-clock epoch) regardless of environment_node.py's OWN use_sim_time
    # setting (that node deliberately runs use_sim_time=false for its own
    # wall-clock step-sleep logic, section P0-7 -- a separate, unrelated
    # design decision). Nav2's nodes must read the SAME clock the data they
    # consume is stamped with, or every TF lookup looks like a request for a
    # point in the future ("Extrapolation Error ... latest data is at time
    # 37.015000" while Nav2 asked using a real wall-clock epoch second).
    use_sim_time = {"use_sim_time": True}
    # Resolved here (not pinned in the static YAML, see nav2_mppi_params.yaml's
    # bt_navigator comment) so this launch file works across ROS distros /
    # install locations without editing the params file.
    bt_xml_param = {"default_nav_to_pose_bt_xml": os.path.join(
        get_package_share_directory("nav2_bt_navigator"), "behavior_trees",
        "navigate_to_pose_w_replanning_and_recovery.xml",
    )}

    lifecycle_nodes = ["controller_server", "planner_server", "behavior_server", "bt_navigator"]

    return LaunchDescription([
        Node(
            package="nav2_controller", executable="controller_server", name="controller_server",
            output="screen", parameters=[params_file, use_sim_time],
        ),
        Node(
            package="nav2_planner", executable="planner_server", name="planner_server",
            output="screen", parameters=[params_file, use_sim_time],
        ),
        Node(
            package="nav2_behaviors", executable="behavior_server", name="behavior_server",
            output="screen", parameters=[params_file, use_sim_time],
        ),
        Node(
            package="nav2_bt_navigator", executable="bt_navigator", name="bt_navigator",
            output="screen", parameters=[params_file, bt_xml_param, use_sim_time],
        ),
        Node(
            package="nav2_lifecycle_manager", executable="lifecycle_manager",
            name="lifecycle_manager_navigation", output="screen",
            parameters=[{"autostart": True, "node_names": lifecycle_nodes}, use_sim_time],
        ),
    ])
