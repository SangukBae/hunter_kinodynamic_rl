"""Package-level pytest bootstrap (collection guard) -- mirrors drl_agent's
conftest.py so the ROS launch_testing plugin never tries to import
launch/*.launch.py, and colcon build artifacts are never collected."""

collect_ignore = ["launch", "build", "install", "log"]
