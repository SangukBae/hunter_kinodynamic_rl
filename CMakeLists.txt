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
    hunter_kinodynamic_rl/nodes/hierarchical_environment_node.py
    hunter_kinodynamic_rl/nodes/hierarchical_train_node.py
    hunter_kinodynamic_rl/nodes/hierarchical_navigation_node.py
    hunter_kinodynamic_rl/evaluation/run_live_hierarchical_benchmark.py
    hunter_kinodynamic_rl/evaluation/run_live_ablation_suite.py
    hunter_kinodynamic_rl/evaluation/live_evidence_runner.py
    hunter_kinodynamic_rl/evaluation/run_stage2_local_baselines.py
    hunter_kinodynamic_rl/evaluation/tractor_latency.py
    hunter_kinodynamic_rl/evaluation/tractor_acceptance.py
    hunter_kinodynamic_rl/evaluation/materialize_environment_v2.py
    hunter_kinodynamic_rl/training/preflight.py
    hunter_kinodynamic_rl/training/hierarchical_preflight.py
    hunter_kinodynamic_rl/training/tractor_preflight.py
    hunter_kinodynamic_rl/training/tractor_dataset_validation.py
    hunter_kinodynamic_rl/training/tractor_scenario_plan.py
    hunter_kinodynamic_rl/training/train_tractor_tqc.py
    hunter_kinodynamic_rl/training/train_comparison_baselines.py
  DESTINATION
    lib/${PROJECT_NAME}
)

ament_package()
