#!/usr/bin/env python3
"""``ros2 run hunter_kinodynamic_rl environment_node.py`` entrypoint --
thin wrapper, all logic in env/simulation/environment_node.py."""

from hunter_kinodynamic_rl.env.simulation.environment_node import main

if __name__ == "__main__":
    main()
