# Delivery Report

> **Historical artifact.** 이 문서는 작성 당시 source와 실행 증거를 보존한다.
> 최신 구현·연구 준비 상태는 [`CURRENT_STATUS.md`](CURRENT_STATUS.md)를 따른다.

Round-4 code-review remediation for `hunter_kinodynamic_rl` -- 16 items
(P0-1..9, P1-10..13, P2-14, P2-15, this Final-2 report). This document is
the honest record of what was done, what was verified LIVE against Gazebo
vs. only unit-tested, and what remains open. Per the session's own ground
rule: nothing below claims success for anything not actually run/observed.

## Current test suite state (last run)

- Host (`ros-free`, no torch/rclpy): `PYTHONPATH=. python3 -m pytest -p no:anyio -q tests/`
  -> **245 passed, 11 skipped** (skips are all `pytest.importorskip("rclpy")`/
  `pytest.importorskip("torch")` module-level skips on files that need a
  built ROS workspace or GPU-capable torch -- not failures).
- Docker/`colcon test --packages-select hunter_kinodynamic_rl` (torch + rclpy
  both available): **309 tests, 0 errors, 0 failures, 0 skipped**.

## P0 (critical correctness) -- all fixed, live-verified against Gazebo

1. **L bind physical control** -- the trajectory action's third component
   (L, horizon length) now caps commanded SPEED (`pure_pursuit_adapter.py`'s
   `_l_bound_speed`), not just the risk-assessment horizon -- otherwise L
   was a dead action dimension outside of risk scoring.
2. **World boundary in collision/risk** -- `risk/boundary.py::distance_to_boundary_m`
   folded into both the live `/step` collision check and risk-telemetry
   computation; a robot near the world edge is now correctly treated as
   near an obstacle even with an all-clear LiDAR sweep.
3. **False emergency-stop right after reset** -- fixed a spurious
   `emergency_stop` flag on the very first step of a fresh episode.
4. **Deterministic checkpoint resume** -- checkpoints only ever save at true
   episode boundaries (`training/checkpoint_policy.py::checkpoint_due`);
   `_last_checkpoint_step` is updated BEFORE saving (an ordering bug that
   corrupted post-resume save cadence). Verified via a from-scratch vs.
   interrupted-and-resumed run producing a bit-identical action sequence
   across every python-random/numpy/torch RNG stream
   (`tests/test_checkpoint_resume_determinism.py`).
5. **Command telemetry reflects the actually-published (delayed) command**,
   not the pre-latency-buffer value.
6. **TTC censoring fix** -- "no collision found within the risk horizon"
   is no longer folded into TTC statistics as if the horizon itself were a
   measured time-to-collision (`risk/ttc.py`,
   `evaluation/metrics.py::collision_free_rate_mean` reports the censoring
   rate as its own explicit metric instead).
7. **Deterministic Gazebo stepping** investigated -- root cause of an
   earlier stepping/executor-hang class of bugs was a bare
   `rclpy.spin_once(self, ...)` call inside a node's own callback silently
   detaching that node from its `MultiThreadedExecutor` (a real,
   confirmed, non-obvious rclpy footgun -- see the persistent memory file
   `cf_st_step_executor_hang.md`), not the two earlier-suspected/superseded
   causes. Fixed by switching to `time.sleep()`-based polling.
8. **Continuous dynamic obstacle motion** -- moving obstacles now animate
   continuously between steps instead of one-shot teleporting to their
   next waypoint.
9. **Sensor-freshness failure handling** -- a stale-sensor `/reset` now
   retries a bounded number of times then fails LOUDLY (`RuntimeError`)
   instead of silently returning a stale initial observation; a stale-
   sensor `/step` truncates the episode and the trainer never stores that
   one transition in the replay buffer
   (`tests/test_trainer_sensor_stale.py`).

## P1 (completeness) -- all fixed

10. **Domain randomization wired to real consumers** -- 4 new pure
    functions (`apply_lidar_noise`, `apply_odometry_noise`,
    `should_drop_sensor_frame`, `steering_lag_alpha`) now actually consume
    the previously-sampled-but-unused LiDAR-noise/dropout/odometry-noise/
    steering-delay/command-latency fields; a genuine bug (`command_latency_sec`
    hardcoded to 0 regardless of the sampled draw) was found and fixed.
    Config layering: `config/domain_randomization/default.yaml` (opt-in,
    disabled by default) + a new `kinodynamic_tqc_domain_rand.yaml` profile.
11. **Complete observation/logging** -- `ROBOT_STATE_DIM` 7->8 to stop
    silently dropping the policy's own memory of its previous L/yield
    action component; `RunLogger` now records real pose, decoded
    trajectory, `actual_dt_sec`, guarded vs. published commands,
    sensor-stale/emergency-stop flags, and the full candidate array per
    step.
12. **Real Ackermann steering in system identification** -- `system_id_node.py`
    rewritten to read `/hunter_se/joint_states` (center steering = mean of
    the two front wheel joints) instead of odometry's `angular.z` (yaw
    rate, a different physical quantity); active excitation via `_hold_for()`
    (continuous republish -- the previous one-shot publish decayed to zero
    after `hunter_se_cmd_prefilter.yaml`'s 2 s `command_timeout_sec`).
    Live-verified: a circle-test trial's steering converged to ~0.305 rad
    and its fitted turning radius (1.87 m) matched the wheelbase-predicted
    value (1.77 m) within ~5%.
13. **Periodic held-out validation + best-checkpoint selection** -- a
    SEPARATE `validation_seed_scheduler` (never train/test seeds) runs
    `eval_episodes` deterministic-policy episodes at the same
    episode-boundary checkpoint cadence; best-checkpoint criterion is
    `success_rate - collision_rate`, documented; validation transitions
    NEVER enter the replay buffer. **Live-verified against Gazebo**: a
    smoke-scale training run produced 3 real validation events at steps
    52/132/200 with the exact expected `is_new_best` semantics (first
    validation always counts; a tied metric does NOT count as a new best;
    an improved metric does), and a real `best.json`/`best.pt` on disk
    whose `best_eval_metric` matched the logged value exactly.
    **Config-validation completeness** (the second half of this item): a
    full audit of `config/schema.py` found and fixed:
    - A genuinely DEAD flag (`features.risk_prediction`) declared in every
      shipped profile but read by zero code paths anywhere -- removed
      entirely (not just documented), matching this codebase's own
      `loader.py` typo-guard philosophy ("an ablation flag that silently
      does nothing is worse than a startup error").
    - `features.curriculum` (also zero consumers) now explicitly rejected
      at load time instead of silently accepted.
    - A new cross-check: `evaluation.benchmark` set + `domain_randomization.enabled`
      together now raises -- `environment_node.py`'s reset branch always
      prefers the fixed-benchmark path, so randomization would silently
      never apply despite looking enabled.
    - Reward sign-convention checks (`goal_reached_reward>=0`,
      `collision_penalty<=0`, `step_penalty<=0`) -- these three fields are
      added to the total reward DIRECTLY (unlike the weight fields, which
      scale an already-signed term), so a flipped sign would have silently
      rewarded collisions or punished reaching the goal with nothing else
      in the pipeline able to catch it.
    - `counterfactual.{kappa_offsets_frac,speed_fractions}` must be
      non-empty when `counterfactual.enabled` (an empty list silently
      collapses candidate generation to just [actor, stop]).
    12 new regression tests, all `pytest.raises(ConfigError)`-style.

## P2 (baselines + deployment)

### P2-14: Vanilla SAC -- implemented, unit-tested, **fully live-verified**

`rl/algorithms/sac/agent.py` -- standard twin-Q SAC (Haarnoja et al. 2018),
reusing `rl/networks/tqc.py`'s `Actor`/`Critic` verbatim
(`Critic(n_quantiles=1)` collapses to a plain scalar twin-Q pair) rather
than duplicating a second network file. New `SACHyperparameters` dataclass
(no quantile fields), new `AlgorithmConfig` (`algorithm.name: tqc|sac`),
`sac_baseline` profile mirrors ablation row B exactly so only the algorithm
differs. `nodes/train_node.py` dispatches on `algorithm.name`.

**Live-verified**: a smoke-scale training run against Gazebo completed all
200 steps, produced real checkpoints (`final`/`latest`/`best`, the correct
6-component set including `ent_coef_optimizer`), and a real periodic-
validation event -- confirming the algorithm-selection dispatch, the SAC
update math, and checkpoint save/load all work end-to-end, not just in
isolated unit tests.

### P2-14: Nav2-MPPI classical baseline -- CODE-COMPLETE, PARTIALLY live-verified

`evaluation/nav2_mppi_runner.py` + `config/nav2_mppi/` +
`launch/nav2_mppi.launch.py` + `nodes/nav2_mppi_eval_node.py`. Map-free (no
AMCL/map_server -- the RL benchmark scenarios have no pre-built occupancy
map matching their procedurally-placed obstacles), Ackermann motion model
(scout_nav2's own reference config uses DiffDrive despite modeling an
Ackermann platform -- corrected here), `min_turning_r` derived from this
robot's actual wheelbase/steering-limit geometry (regression-tested against
schema.py's own `RobotConfig.max_curvature`).

Four REAL integration bugs were found and fixed during live debugging:
- A `/scan` subscription QoS mismatch that silently starved the collision
  monitor (default RELIABLE vs. the publisher's BEST_EFFORT).
- `default_nav_to_pose_bt_xml: ""` does NOT fall back to bt_navigator's
  compiled-in default -- it crashes the BT with "Empty Tree". Fixed by
  resolving the real installed path via `ament_index` at launch time.
- `environment_node.py` leaves Gazebo PAUSED between `/step` calls
  (determinism/perf); Nav2's controller needs it running continuously.
  Fixed by having the harness directly call the same raw
  `/world/<name>/control` Gazebo service `environment_node.py`'s own
  `GazeboRuntimeMixin` uses.
- A `use_sim_time` mismatch -- Nav2's nodes were reading wall-clock time
  while `/odometry`/`/tf` (bridged straight from Gazebo) carry
  simulation-time stamps, so every TF lookup looked like a request for a
  point in the future.

After all four fixes: the full Nav2 stack configures/activates cleanly,
accepts `navigate_to_pose` goals with correctly-placed scenario
coordinates, and the harness's independent collision/telemetry pipeline
produces real, plausible measurements (a genuine near-collision was
recorded at 0.05 m worst clearance, matching `environment_node.py`'s own
collision formula reused verbatim). **What was NOT achieved**: a live
attempt reaching a benchmark goal within its timeout -- the last live run's
average commanded velocity was implausibly low (~0.03 m/s vs. this
robot's 2.0 m/s max), an unresolved tuning/performance question (possibly
inherited MPPI critic weights tuned for a much slower platform, possibly
Gazebo real-time-factor degradation from an extremely long-running Docker
session) that would need further live iteration to root-cause. Treat this
baseline as implemented and partially verified, not as a confirmed-working
navigation stack -- see `evaluation/nav2_mppi_runner.py`'s module docstring
for the exact status.

### P2-15: Real-robot inference node + rosbag2 -- implemented, **live-verified against Gazebo**

`nodes/real_policy_node.py` -- subscribes the SAME sensor topics
`environment_node.py` itself consumes (`/scan`, `/odometry`,
`/hunter_se/joint_states` -- identical names whether the publisher is
Gazebo or real hardware, by `hunter_se_gazebo`'s own bridge-config design),
loads a trained checkpoint, and publishes `/cmd_vel` at
`runtime.time_delta_sec`. Refuses to run against any profile not marked
`runtime.deployment: real_hardware`. Composes ONLY already-tested
ROS-free pure functions (`observation_builder`, `trajectory_executor`
-- explicitly documented in that module as shared between "the env node,
real-robot inference node" -- and `action_guard.guard`, mandatory here).
One deliberate, documented sim/real asymmetry: it never computes/publishes
risk telemetry, since that needs privileged ground-truth obstacle
positions that don't exist on real hardware -- a new
`/hunter_kinodynamic_rl/real_policy_diagnostics` topic (raw policy action +
goal + nearest-obstacle-distance + emergency-stop flag) is the honest
replacement.

`launch/record_real_trial.launch.py` -- `ros2 bag record` wrapping the
original brief's section-42 topic list, adapted to this system's actual
topic names, plus the new diagnostics topic.

**Live-verified against Gazebo** (a live-Gazebo instance is topologically
indistinguishable from real hardware to this node -- it never touches any
Gazebo-specific service): loaded a real checkpoint (all components,
including `risk_critic`), published plausible `/cmd_vel` respecting
`real_hunter_safe.yaml`'s conservative 0.8 m/s speed cap, and the FULL
command pipeline (`real_policy_node -> /cmd_vel -> hunter_se_cmd_prefilter
-> /cmd_vel_filtered -> Gazebo`) produced REAL, CONFIRMED robot motion (3.4 m
of travel over 8 s, matching the commanded ~0.4 m/s). An initial "robot not
moving" observation during this verification was root-caused to a stale,
heavily pause/reset/kill-cycled long-running Gazebo instance (many hours of
this same session's prior experiments) rather than a code defect --
confirmed by reproducing clean motion in a freshly-launched Gazebo
instance. `record_real_trial.launch.py` was also live-verified: it
successfully subscribed to and recorded 9 of its 10 target topics (the
10th, the diagnostics topic, had no active publisher at that specific
moment because `real_policy_node.py` had already exited its test timeout
window -- not a recording-config defect).
No real Hunter SE hardware trial was performed (none available in this
development environment) -- this remains the one genuinely un-closeable
gap, unchanged from `docs/SIM2REAL.md`'s prior status.

## Documentation reconciled with actual delivered state

- `docs/RESEARCH_PROTOCOL.md` -- the "Vanilla SAC and Nav2-MPPI not
  implemented" line replaced with the actual status of each (see above).
- `docs/ARCHITECTURE.md` -- the stale "no sensor-noise consumer yet" claim
  (contradicted by P1-10, already done in an earlier part of this same
  session) corrected; "not yet done" list updated to only genuinely
  outstanding items.
- `docs/SIM2REAL.md` -- the real-robot node and rosbag2 config moved from
  "still needs" to a dedicated "Section P2-15" status section with the
  exact live-verification claims made above.

## Housekeeping found during this Final-2 pass

- `ros2_ws/src/hunter_kinodynamic_rl/` had no `.gitignore` of its own; the
  repo ROOT `.gitignore` already covers `__pycache__/` everywhere but was
  missing `ros2_ws/runtime/` (this package's default run-output location,
  ~30 MB of checkpoints/logs accumulated from this session's live
  verification passes) -- added.
- Removed a stray leftover run directory
  (`ros2_ws/runtime/experiments/20260816_234019_smoke_test_seed0`, ~30 MB)
  missed by an earlier cleanup pass in this same session.

## What remains genuinely open (not fixed in this round)

- **Nav2-MPPI goal-reaching**: implemented and partially live-debugged (see
  above), but no live attempt reached a benchmark goal -- an unresolved
  low-effective-velocity issue needs further live iteration to root-cause.
- **Real Hunter SE hardware**: no physical robot was available in this
  development environment at any point in this session. Every "live-
  verified" claim above means "verified against Gazebo," never against
  real hardware. `config/robot/hunter_se_identified.yaml` (measured, not
  nominal, dynamics parameters) still does not exist for the same reason.
- Everything else in this round's 16-item list is fixed, tested, and
  (where a live-Gazebo check was meaningful and feasible within this
  session) live-verified as described above.
