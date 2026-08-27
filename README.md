# hunter_kinodynamic_rl

**Risk-Aware Kinodynamic Reinforcement Learning for Real-World Ackermann Robot Navigation.**

An independent ROS2 research package: a TQC policy that outputs an
Ackermann-feasible local trajectory (curvature, speed, horizon) instead of an
abstract waypoint, whose future risk is predicted by rolling the candidate
trajectory through Hunter SE's actual dynamics -- and is trained to prefer
safer alternatives via a counterfactual risk-aware objective.

```
Ouster LiDAR + robot state + goal
        |
Temporal observation encoder
        |
TQC Actor -> [kappa, v_ref, L]
        |
Ackermann-feasible local trajectory (dynamics/, trajectory/)
        |
Hunter SE dynamics rollout (dynamics/ackermann_rollout.py)
        |
Future clearance / TTC / stopping margin / collision risk (risk/)
        |
Reward critic + Risk critic (rl/)
        |
Counterfactual candidate comparison (risk/counterfactual_sampler.py)
        |
Risk-aware policy learning
        |
Pure Pursuit (trajectory/pure_pursuit_adapter.py) -> cmd_vel
        |
hunter_se_cmd_prefilter -> Gazebo AckermannSteering -> Hunter SE
```

See `docs/ARCHITECTURE.md` for the full data-flow explanation (including
training vs. inference differences) and `docs/SOURCE_MAP.md` for exactly
which pieces of `drl_agent` were reused and how.

## Relationship to `drl_agent`

`drl_agent` (the sibling package) is treated as a **read-only reference
implementation** -- nothing here modifies it. `hunter_kinodynamic_rl` reuses
`hunter_se_gazebo`, `drl_agent_interfaces`, `pointcloud_to_laserscan`, and
`drl_obstacle_assets` as ROS dependencies, and copies a small number of
`drl_agent` modules in verbatim (hash-pinned, see `docs/SOURCE_MAP.md`) --
it does **not** depend on `drl_agent` itself at runtime.

## Build

```bash
# Inside the Docker container (see CLAUDE.md's Active Docker Environment)
cd /root/DRL_Robot_Path_Planning/ros2_ws
colcon build --packages-select hunter_kinodynamic_rl drl_agent_interfaces hunter_se_gazebo
source install/setup.bash
```

## Test (no Gazebo needed)

```bash
cd ros2_ws/src/hunter_kinodynamic_rl
python3 -m pytest -q tests/                    # ROS-free unit suite (85 tests)
python3 -m hunter_kinodynamic_rl.config.validation kinodynamic_tqc_counterfactual
```

`tests/test_tqc_parity.py` cross-checks this package's copied TQC networks
against the live `drl_agent` source for exact numerical parity;
`tests/test_dynamics.py`, `test_trajectory.py`, `test_risk.py` cover the
three core research contributions independent of any simulator.

## Simulate + train

```bash
# Terminal 1: Gazebo + Hunter SE (reused from hunter_se_gazebo, unmodified)
ros2 launch hunter_se_gazebo simulate_hunter_se_ignition.launch.py rviz:=false

# Terminal 2: environment node
ros2 run hunter_kinodynamic_rl environment_node.py --ros-args -p profile:=kinodynamic_tqc_counterfactual

# Terminal 3: trainer (dispatches to vanilla or risk-aware TQC based on the profile's features.risk_critic)
ros2 run hunter_kinodynamic_rl train_node.py --ros-args -p profile:=kinodynamic_tqc_counterfactual
```

Fast implementation smoke-check (small warmup/batch/episode counts, NOT for
research results -- see `config/profiles/smoke_test.yaml`):

```bash
ros2 run hunter_kinodynamic_rl environment_node.py --ros-args -p profile:=smoke_test
ros2 run hunter_kinodynamic_rl train_node.py --ros-args -p profile:=smoke_test
```

## Evaluate against a fixed benchmark

```bash
ros2 run hunter_kinodynamic_rl evaluation_node.py --ros-args \
  -p profile:=evaluation_id -p checkpoint_dir:=<run_dir>/models -p checkpoint_name:=ckpt
```

## Profiles

| Profile | What it is |
|---|---|
| `baseline_tqc` / `legacy_waypoint_tqc` | Ablation A -- vanilla TQC, legacy `[r, theta, yield]` waypoint action (fair comparison point vs. drl_agent) |
| `kinodynamic_tqc` | Ablation B -- Ackermann trajectory action, no temporal/risk |
| `kinodynamic_tqc_temporal` | Ablation C -- + temporal LiDAR context |
| `kinodynamic_tqc_risk_supervised_only` | Ablation D -- + future risk prediction, supervised only (`risk.actor_lambda=0.0`, no actor penalty) |
| `kinodynamic_tqc_risk` | Ablation E -- D + risk-aware actor penalty (`risk.actor_lambda=0.1`) |
| `kinodynamic_tqc_counterfactual` | Ablation F -- full proposed system (+ counterfactual risk ranking) |
| `kinodynamic_tqc_domain_rand` | Ablation B + opt-in dynamics/sensor domain randomization |
| `sac_baseline` | Algorithm-choice comparison point -- Vanilla SAC on the SAME task as ablation B (`algorithm.name: sac`) |
| `smoke_test` | Fast end-to-end implementation check, not a research config |
| `evaluation_{id,ood_geometry,ood_dynamics,dynamic}` | Fixed-benchmark evaluation (`config/benchmarks/`) -- also the scenario set the Nav2-MPPI classical baseline (`evaluation/nav2_mppi_runner.py`) runs against |
| `real_hunter_safe` | Real-robot inference profile (`nodes/real_policy_node.py`) -- conservative speed/lookahead + mandatory `env/safety/action_guard.py` |
| `hierarchical_phase1` | Phase 1 hierarchical-navigation verification profile (mission frame / localization / mapping only, see below) |

## Hierarchical navigation (Phase 1: mission frame / localization / mapping)

Opt-in, separate from the local-only kinodynamic-TQC path above -- see
`docs/HIERARCHICAL_NAVIGATION_IMPLEMENTATION_PLAN.md` and
`hunter_se_unknown_gps_denied_hierarchical_navigation_detailed_spec.txt` for
the full multi-phase design. Phase 1 implements only the foundation: a fixed
mission-start frame + relative final goal (`navigation/mission/`), a
pluggable localization backend protocol (`navigation/localization/`), and an
online LiDAR partial map with explicit UNKNOWN/FREE/OCCUPIED/
OBSERVED_UNCERTAIN/visited/failure channels (`navigation/mapping/`; see
`partial_map.py`'s module docstring for why there are four map states, not
three). No Global RL / subgoal hierarchy yet. Live-verified against two real
Gazebo runs, including a teleported non-zero start pose/yaw to confirm the
mission-frame rotation math against a real Gazebo odometry quaternion, not
just synthetic unit tests
(`docs/verification/2026-08-28_hierarchical_navigation_phase1_review_fixes.md`).

```bash
ros2 launch hunter_se_gazebo simulate_hunter_se_ignition.launch.py rviz:=true
ros2 run hunter_kinodynamic_rl mission_map_node.py --ros-args \
  -p profile:=hierarchical_phase1 -p goal_x:=5.0 -p goal_y:=0.0
```

Publishes `/mission_map`, `/rolling_map`, `/visited_map`, `/inflated_map`
(`nav_msgs/OccupancyGrid`) and `/mission_goal` (`geometry_msgs/PointStamped`,
frame `mission`), plus the `odom -> mission` TF -- add all five to RViz to
watch the partial map grow as the robot explores. Read-only w.r.t. the
simulation: never publishes `/cmd_vel` and never touches
`drl_agent_interfaces`.

## System identification (real Hunter SE)

```bash
ros2 run hunter_kinodynamic_rl system_id_node.py --ros-args -p output:=system_id_results/circle_20deg.csv
python3 -c "
from hunter_kinodynamic_rl.dynamics.system_identification import samples_from_csv, analyze_circle_test
print(analyze_circle_test(samples_from_csv('system_id_results/circle_20deg.csv')))
"
```

Feed the results into `config/robot/hunter_se_identified.yaml` (copy
`hunter_se.yaml` and overwrite the measured fields) and point a profile's
`robot_file` at it once real-robot measurements exist.

## Verification status

309 tests pass under `colcon test` (0 errors, 0 failures, 0 skipped, incl.
inside the Docker container with real CUDA); 245 pass on a bare host
checkout with no ROS/torch (11 skipped there, all clean
`pytest.importorskip` module skips). The FULL live pipeline has been run
against a real Gazebo + Hunter SE instance: Gazebo launch -> spawn ->
`/reset`/`/step` -> real risk telemetry + counterfactual candidates ->
replay -> CUDA critic/actor/risk updates -> checkpoint save -> resume ->
periodic held-out validation + best-checkpoint selection -> exact
fixed-benchmark evaluation with real odometry-based metrics -> the
real-robot inference node driving real Gazebo-confirmed motion. See
**`docs/DELIVERY_REPORT.md`** for the full, item-by-item command-level
evidence, including what's live-verified vs. unit-tested-only and what
remains genuinely open (Nav2-MPPI goal-reaching is unresolved; no real
Hunter SE hardware trial has been run, no hardware available in this
development environment).

**2026-08-27 defect-fix pass** (start-pose wall clearance, sensor-noise
evaluation fairness, reset initial-frame noise duplication, obstacle-pool
active/parked/retry/exact-class consistency, a new GT-vs-noisy observation
diagnostics side channel, exact discrete-time OU localization drift): see
**`docs/verification/2026-08-27_start_pose_noise_pool_diagnostics_ou.md`**
for command-level evidence -- 1162 tests green under `colcon test` (0
errors/failures/skipped), plus a live Gazebo session (40 resets, a full
200-step training run, an interrupted+resumed run, two fixed-benchmark
evaluation runs differing only in `sensor_noise`).

**2026-08-26 defect-fix pass** (Gazebo physics-step reality-check, system-ID
stale-data handling, checkpoint-prune safety, evaluation-contract restore
failure propagation): see **`docs/verification/2026-08-26_item1-4_fixes.md`**
for command-level evidence, including a real live-Gazebo A/B run of the new
physics-step calibration mechanism. A per-episode `train_*.log` is NOT where
step-level data lives -- the actual per-step evidence for a training/
evaluation run is `<run_dir>/logs/steps.jsonl` (structured, one JSON object
per step) plus `<run_dir>/logs/episodes.csv`; any `.log` file under a run
directory is plain rclpy console output, not the structured record.
