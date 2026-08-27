cmake_minimum_required(VERSION 3.8)
project(hunter_kinodynamic_rl)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(drl_agent_interfaces REQUIRED)
find_package(ros_gz_interfaces REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  set(ament_cmake_flake8_FOUND TRUE)
  set(ament_cmake_pep257_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(hunter_kinodynamic_rl_pytests tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    TIMEOUT 600
    ENV PYTHONDONTWRITEBYTECODE=1
  )
endif()

install(
  DIRECTORY
    config
    launch
  DESTINATION
    share/${PROJECT_NAME}/
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
)

# Importable Python package -- the only home of the implementation
# (import hunter_kinodynamic_rl.<...>), mirroring drl_agent's convention.
ament_python_install_package(${PROJECT_NAME})

# ros2 run hunter_kinodynamic_rl <name>.py entrypoints.
install(
  PROGRAMS
    hunter_kinodynamic_rl/nodes/environment_node.py
    hunter_kinodynamic_rl/nodes/train_node.py
    hunter_kinodynamic_rl/nodes/evaluation_node.py
    hunter_kinodynamic_rl/nodes/nav2_mppi_eval_node.py
    hunter_kinodynamic_rl/nodes/real_policy_node.py
    hunter_kinodynamic_rl/nodes/system_id_node.py
    hunter_kinodynamic_rl/nodes/mission_map_node.py
  DESTINATION
    lib/${PROJECT_NAME}
)

ament_package()
