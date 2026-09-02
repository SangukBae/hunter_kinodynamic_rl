# 2026-08-28: Hierarchical navigation Phase 1 (mission frame / localization / mapping)

> **Historical verification artifact.** 당시 source/명령/결과를 보존한다. 최신
> 상태는 [`../CURRENT_STATUS.md`](../CURRENT_STATUS.md)를 따른다.

> **Superseded in part** by `2026-08-28_hierarchical_navigation_phase1_review_fixes.md`
> -- a code review found this initial pass had never actually been run
> against a live Gazebo instance and several defects that would have kept
> `/mission_map` from updating in practice (QoS mismatch, sim/wall clock
> mixing, others). The test count and 3-state map-channel description
> below are stale; see that document for the corrected/current state,
> including a real live-Gazebo verification run.

Implements `docs/HIERARCHICAL_NAVIGATION_IMPLEMENTATION_PLAN.md` section 5
end to end: `navigation/mission/` (`MissionFrame`, `GoalManager`),
`navigation/localization/` (`LocalizationBackend` protocol + odom/Gazebo
adapters), `navigation/mapping/` (Bresenham ray tracing, `PartialMap` with
explicit UNKNOWN/FREE/OCCUPIED channels, rolling crop, visited/failure
maps), plus `MissionConfig`/`LocalizationConfig`/`MappingConfig` and the
`mission_map_node.py` RViz verification node. No Global RL / subgoal
hierarchy yet (Phase 2+).

## Environment

Docker container `7a2702b311a1` (`drl_robot_path_planning:first`), ROS2
Humble, workspace `/root/DRL_Robot_Path_Planning/ros2_ws`.

## Scope boundaries kept

- `drl_agent_interfaces` untouched; no shared service/message changed.
- New config sections (`mission`, `localization`, `mapping`) are read only by
  `navigation/` -- every pre-existing profile validates unchanged and
  inherits inert defaults (`test_navigation_config.py::
  test_unrelated_existing_profile_still_loads_with_navigation_defaults`).
- `mission_map_node.py` is read-only w.r.t. the simulation: subscribes to
  `/odometry` + `/scan` only, never publishes `/cmd_vel`, never reads
  simulator ground-truth obstacle/map state.
- UNKNOWN is computed as `NOT observed`, never `1 - occupied`
  (`PartialMap.channels`); `MappingConfig.validate()` structurally rejects a
  `free_threshold >= occupied_threshold` config that could make the same
  log-odds value satisfy both predicates.
- NaN/Inf/non-positive beam ranges make no map update at all (not even
  free-space); a genuine max-range return marks free space through to
  `range_max` but never an occupied endpoint.
- `mission_map_node.py::_on_scan` rejects the whole scan (no map write) when
  `is_pose_usable` fails on the current localization reading -- stale/low-
  confidence/invalid pose never contributes to the map.

## Docker verification

```
colcon build --packages-select hunter_kinodynamic_rl   # clean, 0 errors
python3 -m hunter_kinodynamic_rl.config.validation <every config/profiles/*.yaml>  # all OK, incl. hierarchical_phase1
colcon test --packages-select hunter_kinodynamic_rl
colcon test-result --test-result-base build/hunter_kinodynamic_rl --verbose
```

Result: `1286 tests, 0 errors, 0 failures, 0 skipped` (full existing suite +
124 new Phase 1 tests across `test_mission_frame.py`, `test_goal_manager.py`,
`test_localization_interface.py`, `test_mapping_raytracing.py`,
`test_partial_map.py`, `test_visited_map.py`, `test_rolling_map.py`,
`test_navigation_config.py`, `test_mission_map_node.py`).

## Known Phase 1 limitations (deferred to later phases per the plan)

- No subgoal hierarchy / Global RL yet -- `mission_map_node.py` only
  accumulates the map and reports final-goal reach, it does not drive the
  robot.
- LiDAR extrinsic is a configurable flat xy/yaw offset (`lidar_offset_x/y`,
  `lidar_yaw_offset_rad` node parameters), not a full 3D transform -- fine
  for the 2D occupancy grid this phase produces.
- Only one live localization backend (`GazeboOdomLocalizationBackend`,
  `nav_msgs/Odometry`-based) exists; wheel+IMU / LiDAR-odometry / LIO
  backends are Phase 6 scope.
- `mapping.inflation_radius_m` drives `PartialMap.channels().inflated`
  (occupied dilated by that radius, published as `/inflated_map`) -- the
  exact "known occupied/inflated cell" test Phase 4's action mask will use;
  no candidate/action-mask code exists yet to consume it beyond this
  channel/visualization.
