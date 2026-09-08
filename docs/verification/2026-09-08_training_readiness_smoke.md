# Training Readiness Smoke — 2026-09-08

Evidence level: **development Gazebo smoke / code-health**, not formal experiment, navigation
performance, simulator fidelity, target-hardware timing or real-robot evidence.

## Environment

- audit base: `b4de2e4e3523e8210ffe078c887bbd38f4d97bde` plus the remediation working tree;
- container: `DRL_Robot_Path_Planning`, local image reference `drl_robot_path_planning:first`;
- local content-addressed image ID: `sha256:c5fb6840b4bf9a4e674da607d5a318986f4d06b22ba33eb4538cd8b40b8f64c6`;
- ROS 2 Humble overlay, Python 3.10.12, Gazebo Ignition 6;
- profile: `tractor_local_dynamic_v2`;
- deterministic stepping: 0.001 s physics step, 0.0002 s general/calibration tolerance.

The local image has no registry `RepoDigest`, so its image ID is recorded here but must not be
silently substituted for the exact formal-run container identity selected by the operator.

## Checks and observations

1. `colcon build --packages-select hunter_kinodynamic_rl --symlink-install` completed: 1 package,
   exit 0.
2. Full sourced Docker regression completed: **2,253 passed in 273.95 s**, exit 0.
3. Procedural curriculum L0 collection used run seed `91827`, 3 episodes and a 40-step operator cap:
   3 episode NPZ files, 120 rows, 75 sequence windows, exit 0.
4. Independent dataset validation reported `ok=true`, `errors=[]`, 3 distinct scenario geometries,
   consistent observation/action/trajectory/robot/environment/resolved-config hashes, and 120
   `nominal_preaction_rollout_summary_v1` labels. The expected warnings state that this small
   development dataset lacks calibration/locked splits and is not formal realized-track evidence.
5. A separate curriculum episode index 6000 (L4) reset generated **4 static + 8 dynamic** obstacles.
   Twenty control steps completed with a 328D state, valid risk telemetry on every step, 12
   privileged obstacle records, no collision and no environment-node failure. A following L0 reset
   deleted all 12 runtime-spawned models and returned a 328D state. The final single-instance Gazebo
   log contained no `Entity ... not found` errors.

The first attempt exposed a real `/clock` queue race: a requested 0.020 s substep could be measured as
0.025 s because the previous call returned at the lower tolerance boundary. Tightening the tolerance
then exposed a 0.001 s stale reset origin. The delivered implementation drains changing `/clock`
values before capturing the origin and after reaching the target; regression tests cover both queue
directions. The final L0 and L4 checks above were rerun after this fix with the 0.0002 s tolerance.
Obstacle deletion now also sends `Entity.MODEL` explicitly and runs before model reset; the former
default `Entity.NONE` request was the source of misleading successful-delete responses and Gazebo
`not found` errors.

## Reproduction shape

Launch Gazebo and the environment in separate terminals, then run the development collector:

```bash
ros2 launch hunter_se_gazebo simulate_hunter_se_ignition.launch.py rviz:=false headless:=true
ros2 launch hunter_kinodynamic_rl environment.launch.py profile:=tractor_local_dynamic_v2
ros2 run hunter_kinodynamic_rl train_tractor_tqc.py \
  --profile tractor_local_dynamic_v2 --variant a7 \
  --dataset-root <new-development-root> --seed 91827 \
  collect --episodes 3 --max-steps-per-episode 40
ros2 run hunter_kinodynamic_rl tractor_dataset_validation.py \
  --dataset-root <new-development-root> --loss-window 16
```

Do not reuse the temporary roots from this smoke for formal work. Formal collection must start from a
new immutable root and bind the frozen 616-scenario manifest, committed source identity and selected
container digest.
