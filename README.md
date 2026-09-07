# hunter_kinodynamic_rl

**Uncertainty-Calibrated Counterfactual Kinodynamic Learning for Ackermann
Robot Navigation.**

An independent ROS2 research package: a TQC policy that outputs an
Ackermann-feasible local trajectory (curvature, speed, horizon) instead of an
abstract waypoint, whose future risk is predicted by rolling the candidate
trajectory through Hunter SE's actual dynamics -- and is trained to prefer
safer alternatives via a counterfactual risk-aware objective.

> **Current status (2026-09-02):** the corrected baseline is frozen at
> `hunter-kinodynamic-rl-r0-20260902`. The bounded Local Gazebo
> reset→step/update→save→resume smoke passed. A live Global smoke was
> correctly blocked because the only `local_frozen` artifact is legacy and
> unpromoted; train and promote a research-valid Local before Global. Formal
> Local/Global results, the proposed risk/residual uncertainty extensions and
> real-Hunter trials remain open. See
> [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md).

The research program keeps the existing package structure and develops it in
two stages:

```text
Local:  Kinodynamic Policy + Uncertainty-Calibrated Risk
      + Physics/Residual Dynamics + Counterfactual Policy Improvement

Global: Online Partial Map + Experience-Aware Topological Memory
      + Frozen-Local Capability Distribution + Localization Uncertainty
```

[`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) is the authoritative
forward-looking roadmap. It clearly separates what is already implemented
from the target method; the current architecture and experiment contracts live
in [`docs/TRACTOR_TQC_MODEL_SPEC.md`](docs/TRACTOR_TQC_MODEL_SPEC.md) and
[`docs/RESEARCH_PROTOCOL.md`](docs/RESEARCH_PROTOCOL.md).

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

See `docs/TRACTOR_TQC_MODEL_SPEC.md` for the full data-flow explanation (including
training vs. inference differences) and `docs/IMPLEMENTATION_PLAN.md` for exactly
which pieces of `drl_agent` were reused and how.

## Relationship to `drl_agent`

`drl_agent` (the sibling package) is treated as a **read-only reference
implementation** -- nothing here modifies it. `hunter_kinodynamic_rl` reuses
`hunter_se_gazebo`, `drl_agent_interfaces`, `pointcloud_to_laserscan`, and
`drl_obstacle_assets` as ROS dependencies, and copies a small number of
`drl_agent` modules in verbatim (hash-pinned, see `docs/IMPLEMENTATION_PLAN.md`) --
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
python3 -m pytest -q tests/
python3 -m hunter_kinodynamic_rl.config.validation kinodynamic_tqc_counterfactual
```

The 2026-09-02 Docker release audit reports 2,093 direct `pytest` tests,
29/29 profiles valid, and 2,098 `colcon test-result` checks with zero
errors/failures/skips. These are regression evidence, not navigation-performance
evidence; exact commands and caveats are in `docs/CURRENT_STATUS.md` and
`docs/IMPLEMENTATION_PLAN.md`.

`tests/test_tqc_parity.py` cross-checks this package's copied TQC networks
against the live `drl_agent` source for exact numerical parity;
`tests/test_dynamics.py`, `test_trajectory.py`, `test_risk.py` cover the
three core research contributions independent of any simulator.

## Simulate + train

```bash
# Terminal 1: Gazebo + improved Hunter SE (the no-argument development default)
ros2 launch hunter_se_gazebo simulate_hunter_se_ignition.launch.py rviz:=false

# Terminal 2: environment node (defaults to kinodynamic_tqc_improved)
ros2 run hunter_kinodynamic_rl environment_node.py

# Terminal 3: trainer (same improved profile; dispatches to risk-aware TQC)
ros2 run hunter_kinodynamic_rl train_node.py
```

Fast implementation smoke-check (small warmup/batch/episode counts, NOT for
research results -- see `config/profiles/smoke_test.yaml`):

```bash
ros2 run hunter_kinodynamic_rl environment_node.py --ros-args -p profile:=smoke_test
ros2 run hunter_kinodynamic_rl train_node.py --ros-args -p profile:=smoke_test
```

For the full-circle/infeasible arbitrary-subgoal preflight and hierarchy Local
contract, use the dedicated bounded profile instead:

```bash
ros2 run hunter_kinodynamic_rl environment_node.py --ros-args -p profile:=smoke_test_arbitrary_subgoal
ros2 run hunter_kinodynamic_rl train_node.py --ros-args -p profile:=smoke_test_arbitrary_subgoal
```

## Evaluate against a fixed benchmark

```bash
ros2 run hunter_kinodynamic_rl evaluation_node.py --ros-args \
  -p profile:=evaluation_id -p checkpoint_dir:=<run_dir>/checkpoints -p checkpoint_name:=final
```

## Profiles

| Profile | What it is |
|---|---|
| `kinodynamic_tqc_improved` | Default active-development profile -- full trajectory/temporal/risk/counterfactual stack, arbitrary full-circle Local subgoals, and `hunter_se_improved.yaml` |
| `baseline_tqc` / `legacy_waypoint_tqc` | Ablation A -- vanilla TQC, legacy `[r, theta, yield]` waypoint action (fair comparison point vs. drl_agent) |
| `kinodynamic_tqc` | Ablation B -- Ackermann trajectory action, no temporal/risk |
| `kinodynamic_tqc_temporal` | Ablation C -- + temporal LiDAR context |
| `kinodynamic_tqc_risk_supervised_only` | Ablation D -- + future risk prediction, supervised only (`risk.actor_lambda=0.0`, no actor penalty) |
| `kinodynamic_tqc_risk` | Ablation E -- D + risk-aware actor penalty (`risk.actor_lambda=0.1`) |
| `kinodynamic_tqc_counterfactual` | Ablation F -- full currently implemented A-F lineage (candidate supervision + safer-margin weighting), not the planned L6-L8 uncertainty/residual/direct-target method |
| `kinodynamic_tqc_stability` | Three-feature stability experiment profile; not the uncertainty-calibrated target method |
| `kinodynamic_tqc_domain_rand` | Ablation B + opt-in dynamics/sensor domain randomization |
| `sac_baseline` | Algorithm-choice comparison point -- Vanilla SAC on the SAME task as ablation B (`algorithm.name: sac`) |
| `smoke_test` / `smoke_test_stability` | Fast legacy/local end-to-end implementation checks, not research configs |
| `smoke_test_arbitrary_subgoal` | Bounded hierarchy-compatible Local save/resume check with the same full-circle/infeasible distribution gate as the research profile; never promote it |
| `local_l0_direct_control` ... `local_l5_counterfactual` | Stage 2 fair Local ladder: direct `(v, steering)` → trajectory action → temporal context → supervised risk → actor risk penalty → structured counterfactual weighting; same task/budget, five generated seeds each |
| `evaluation_{id,ood_geometry,ood_dynamics,dynamic}` | Fixed-benchmark evaluation (`config/benchmarks/`) -- also the scenario set the Nav2-MPPI classical baseline (`evaluation/nav2_mppi_runner.py`) runs against |
| `real_hunter_safe` | Real-robot inference profile (`nodes/real_policy_node.py`) -- conservative speed/lookahead + mandatory `env/safety/action_guard.py` |
| `hierarchical_phase1` | Phase 1 hierarchical-navigation verification profile (mission frame / localization / mapping only, see below) |
| `kinodynamic_tqc_arbitrary_subgoal` | Phase 2 -- same architecture/action/risk contract as `kinodynamic_tqc_counterfactual`, trained on full-circle robot-relative short-range subgoals (2-6m, including deliberately infeasible cases); the candidate/evaluator frame contract is fixed, while multi-tick reverse/U-turn recovery remains future work |
| `hierarchical_phase3` / `hierarchical_phase4` | Phase 3/4 hierarchical-navigation verification profiles (long-horizon world / Global RL, see below) |
| `hierarchical_phase5_a` ... `hierarchical_phase5_g` | Formal hierarchy ablations A-G; strict mode, budget, checkpoint, and frozen-Local capability gates apply |

## Hierarchical navigation (Phase 1: mission frame / localization / mapping)

Opt-in, separate from the local-only kinodynamic-TQC path above -- see
`docs/IMPLEMENTATION_PLAN.md` and
`hunter_se_unknown_gps_denied_hierarchical_navigation_detailed_spec.txt` for
the full multi-phase design. Phase 1 implements only the foundation: a fixed
mission-start frame + relative final goal (`navigation/mission/`), a
pluggable localization backend protocol (`navigation/localization/`), and an
online LiDAR partial map with explicit UNKNOWN/FREE/OCCUPIED/
OBSERVED_UNCERTAIN/visited/failure channels (`navigation/mapping/`; see
`partial_map.py`'s module docstring for why there are four map states, not
three). This Phase-1 entry point intentionally runs no Global RL or subgoal
hierarchy; the later-phase entry points below add those layers. It was
live-verified against two real Gazebo runs, including a teleported non-zero
start pose/yaw to confirm the mission-frame rotation math against a real
Gazebo odometry quaternion, not
just synthetic unit tests
(`docs/verification/README.md`).

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

## Hierarchical navigation (Phase 2-5: Local TQC, long-horizon world, Global RL, ablations)

Opt-in, builds on Phase 1 above. See
`docs/IMPLEMENTATION_PLAN.md` sections 6-8 for the
full design and their `2026-08-31 갱신`/`2026-09-01` status notes for
exactly what is live-verified vs. still pending. For a full fresh-Local
→ Phase 6 runbook (preflight, promotion, formal A/B, A-G ablation,
localization sweep, rosbag dry-run), see
`docs/IMPLEMENTATION_PLAN.md`.

**Phase 2 -- train the Local TQC on arbitrary short-range subgoals** (2-6m,
full-circle directions and some deliberately infeasible cases -- never the
final mission goal, see `navigation/local_rl/controller.py`'s own contract).
The candidate geometry and evaluator-frame contract is fixed in the current
code. Rear candidates are represented honestly for one-decision capability
evaluation; a learned multi-tick reverse/U-turn recovery option is still a
future extension, so rear-candidate results must not be described as that
capability.

The canonical Stage 2 baseline campaign is automated and resumable:

```bash
ros2 run hunter_kinodynamic_rl run_stage2_local_baselines.py prepare
ros2 run hunter_kinodynamic_rl run_stage2_local_baselines.py execute --promote
ros2 run hunter_kinodynamic_rl run_stage2_local_baselines.py status
```

It trains L0-L5 with seeds 0-4, evaluates every final checkpoint on one
pre-registered 20-scenario manifest, aggregates across seeds, and permits only
the pre-registered `L5/seed_0/final` checkpoint to enter the existing
acceptance/promotion gate. `runtime/stage2_local_baselines/` is resumable; an
incomplete matrix cannot aggregate or promote.

```bash
ros2 launch hunter_se_gazebo simulate_hunter_se_ignition.launch.py rviz:=false headless:=true
ros2 run hunter_kinodynamic_rl environment_node.py --ros-args -p profile:=kinodynamic_tqc_arbitrary_subgoal
ros2 run hunter_kinodynamic_rl train_node.py --ros-args -p profile:=kinodynamic_tqc_arbitrary_subgoal
```

(`headless:=true` on `simulate_hunter_se_ignition.launch.py` runs Ignition
Gazebo server-only, `-s` -- needed on a host with no working GL context for
the GUI; physics/sensors/topics/services are unaffected.) Writes to
`runtime/experiments/<timestamp>_kinodynamic_tqc_arbitrary_subgoal_seed<seed>/`;
point `hierarchical_phase4.yaml`'s `hierarchical_training.local_checkpoint_dir`/
`local_checkpoint_name` at that run's `checkpoints/` dir once it has a
promoted, canonical `final` checkpoint. Formal hierarchy paths intentionally
reject unpromoted, mutable, or architecture-incompatible Local checkpoints.

**Phase 3/4 -- live-Gazebo long-horizon world + Global RL, real frozen Local
TQC** (`navigation/local_rl/live_gazebo_executor.LiveGazeboLocalExecutor`,
replacing the ROS-free `SimplifiedKinematicLocalExecutor` stand-in):

```bash
# Gazebo (same as above)
ros2 launch hunter_se_gazebo simulate_hunter_se_ignition.launch.py rviz:=false headless:=true

# Train the Global masked-DQN against live Gazebo + the frozen Local TQC
# (fails fast -- LocalCheckpointError -- if local_checkpoint_dir/name in the
# profile doesn't resolve to a real, architecture-compatible checkpoint):
ros2 launch hunter_kinodynamic_rl hierarchical_train.launch.py \
  profile:=hierarchical_phase4 live:=true num_missions:=1000

# Single-mission live deployment/inference (Global candidate selection +
# frozen Local TQC, continuous):
ros2 launch hunter_kinodynamic_rl hierarchical_environment.launch.py \
  profile:=hierarchical_phase4 goal_x:=10.0 goal_y:=0.0 \
  global_checkpoint_dir:=runtime/hierarchical_experiments/<run>/checkpoints global_checkpoint_name:=latest

# Fixed unseen-manifest benchmark: A (local-only/no-memory baseline) vs.
# B (Phase 4 Global DQN + frozen Local TQC), both over live Gazebo:
ros2 run hunter_kinodynamic_rl run_live_hierarchical_benchmark.py --ros-args \
  -p profile:=hierarchical_phase4 -p num_scenarios:=20 \
  -p global_checkpoint_dir:=runtime/hierarchical_experiments/<run>/checkpoints \
  -p global_checkpoint_name:=latest -p output_dir:=runtime/hierarchical_benchmark
```

`live:=false` (`hierarchical_train.launch.py`'s default) keeps training
against `SimplifiedKinematicLocalExecutor` (ROS-free, no Gazebo needed) --
useful for iterating on Global RL/reward/replay logic without paying for a
live simulation. Both `HierarchicalTrainingLoop` and
`evaluation/long_horizon_benchmark.run_ablation_mission` accept a
`local_executor_factory` injection point for this swap; the default
(`None`) is byte-identical to the pre-existing ROS-free behaviour.

**Current limitations:** the corrected release passed a fresh bounded Local
Gazebo reset/step, replay/update, checkpoint save and resume smoke. The Global
preflight then rejected the legacy, unpromoted `local_frozen` artifact as
designed, so no corrected-release live Global mission/update/save/resume has run.
Formal Local/Global training and benchmarks have not run; existing Local/Global
checkpoints and smoke artifacts are not paper results.

## System identification (real Hunter SE)

This node actively commands the vehicle; follow the containment and E-stop
procedure in `docs/SIM2REAL.md` before running it.

```bash
ros2 run hunter_kinodynamic_rl system_id_node.py --ros-args \
  -p trial:=all -p output_dir:=system_id_results
```

The `all` trial writes per-trial CSV/JSON artifacts and
`system_id_results/hunter_se_identified.yaml`. Review that generated file,
then copy the accepted parameters into `config/robot/hunter_se_identified.yaml`
and point a profile's `robot_file` at it once real-robot measurements exist.

## Verification status

The dated material below is retained as historical evidence for the specific
commands and source state used at the time. It does not override the latest
readiness verdict in `docs/CURRENT_STATUS.md`.

The 2026-09-02 release audit passed 2,093 direct Docker `pytest` tests,
29/29 profile validations and 2,098 `colcon` checks. A real Gazebo Local smoke
reached 60 steps, saved a checkpoint, resumed from `best` at step 53 and reached
60 again with 7/7 telemetry matches and no timeouts. Global preflight failed
closed on the missing promotion manifest, and teardown left no Gazebo/ROS
processes. See
[`docs/verification/README.md`](docs/verification/README.md).

**Initial delivery evidence:** 309 tests passed under `colcon test` (0 errors,
0 failures, 0 skipped, incl.
inside the Docker container with real CUDA); 245 pass on a bare host
checkout with no ROS/torch (11 skipped there, all clean
`pytest.importorskip` module skips). The FULL live pipeline has been run
against a real Gazebo + Hunter SE instance: Gazebo launch -> spawn ->
`/reset`/`/step` -> real risk telemetry + counterfactual candidates ->
replay -> CUDA critic/actor/risk updates -> checkpoint save -> resume ->
periodic held-out validation + best-checkpoint selection -> exact
fixed-benchmark evaluation with real odometry-based metrics -> the
real-robot inference node driving real Gazebo-confirmed motion. See
**`docs/CURRENT_STATUS.md`** for the full, item-by-item command-level
evidence, including what's live-verified vs. unit-tested-only and what
remains genuinely open (Nav2-MPPI goal-reaching is unresolved; no real
Hunter SE hardware trial has been run, no hardware available in this
development environment).

**2026-08-27 defect-fix pass** (start-pose wall clearance, sensor-noise
evaluation fairness, reset initial-frame noise duplication, obstacle-pool
active/parked/retry/exact-class consistency, a new GT-vs-noisy observation
diagnostics side channel, exact discrete-time OU localization drift): see
**`docs/verification/README.md`**
for command-level evidence -- 1162 tests green under `colcon test` (0
errors/failures/skipped), plus a live Gazebo session (40 resets, a full
200-step training run, an interrupted+resumed run, two fixed-benchmark
evaluation runs differing only in `sensor_noise`).

**2026-08-26 defect-fix pass** (Gazebo physics-step reality-check, system-ID
stale-data handling, checkpoint-prune safety, evaluation-contract restore
failure propagation): see **`docs/verification/README.md`**
for command-level evidence, including a real live-Gazebo A/B run of the new
physics-step calibration mechanism. A per-episode `train_*.log` is NOT where
step-level data lives -- the actual per-step evidence for a training/
evaluation run is `<run_dir>/logs/steps.jsonl` (structured, one JSON object
per step) plus `<run_dir>/logs/episodes.csv`; any `.log` file under a run
directory is plain rclpy console output, not the structured record.
