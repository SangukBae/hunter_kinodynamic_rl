#!/usr/bin/env python3
"""``ros2 run hunter_kinodynamic_rl mission_map_node.py`` entrypoint --
thin wrapper, all logic in navigation/ros/mission_map_node.py."""

from hunter_kinodynamic_rl.navigation.ros.mission_map_node import main

if __name__ == "__main__":
    main()
