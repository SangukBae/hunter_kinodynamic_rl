# 2026-08-27: safe start pose / sensor-noise fairness / reset-noise dedup /
# obstacle-pool consistency / GT-vs-noisy diagnostics / exact discrete OU

Six requirements fixed in one pass (numbered as requested):
1. start-pose wall clearance (`robot_radius + start_pose.min_wall_clearance_m`)
2. sensor-noise evaluation fairness (fingerprint + contract override)
3. reset initial-frame noise duplication
4. obstacle-pool active/parked/rollback/exact-class consistency
5. GT-vs-noisy observation diagnostics (new side channel)
6. exact discrete-time Ornstein-Uhlenbeck localization drift

## Environment

Docker container `7a2702b311a1` (`drl_robot_path_planning:first`), ROS2
Humble, workspace `/root/DRL_Robot_Path_Planning/ros2_ws`, Gazebo Sim
(Ignition Fortress), RGL GPU LiDAR on an RTX 4080 (confirmed `/clock` at
~888 Hz, i.e. real-time-factor healthy, not the CPU-fallback ~0.001x RTF
failure mode). Build: `colcon build --packages-select hunter_kinodynamic_rl`
(plain build, not symlink-install) run once at the end of this pass, after
all source edits below.

## item 1: start-pose wall clearance

**Root cause.** `procedural_generator.generate_scenario` inset the START
position from the world boundary by `robot_radius` only -- never
`start_pose.min_wall_clearance_m`. `safe_start._is_heading_safe`'s
front-safety-distance wall projection subtracted `min_wall_clearance_m`
but never `robot_radius`, so a heading could pass with the robot's
*footprint* already inside the margin.

**Fix** (`env/scenarios/procedural_generator.py`, `env/scenarios/safe_start.py`):
- New `start_inset_half = half - (robot_radius + start_pose_cfg.min_wall_clearance_m)`,
  used ONLY for the start-position draw (goal sampling untouched). Raises
  `RuntimeError` immediately if this leaves no usable region --
  `min_wall_clearance_m=0.0` (every pre-existing profile's default) makes
  this byte-identical to the old `inset_half`.
- `_is_heading_safe`'s wall-projection limit is now
  `world_half_extent_m - cfg.min_wall_clearance_m - robot_radius`.
- `_min_obstacle_or_wall_clearance` (the fallback-heading ranking score)
  now also subtracts `robot_radius` from its wall-clearance term, matching
  `obstacle_clearance`'s own units.

**Tests** (`tests/test_start_pose_sampling.py`, 291 total incl. 178 new):
multi-seed (25 seeds x 4 heading modes) start-position and projected-
footprint clearance checks, a zero-clearance byte-identical-to-legacy
regression, and two new fail-fast tests (`min_wall_clearance_m` alone, and
combined with `robot_radius`, leaving no usable region).

## item 2: sensor-noise evaluation fairness

**Root cause.** `evaluation/fingerprint.py::EVALUATION_CONTRACT_SECTIONS`
and `evaluation/contract_override.py::CONTRACT_SECTION_NAMES` never
included `sensor_noise` -- a benchmark run's fingerprint and the live
override delivered to `environment_node.py` both silently ignored it, so
every checkpoint kept its OWN training-time `sensor_noise` setting during
evaluation regardless of what the requested evaluation profile asked for.
Separately, `nodes/evaluation_node.py::build_effective_profile` (which
layers the requested profile's contract sections onto the checkpoint's
reconstructed architecture, and is what actually gets fingerprinted/written
to the override file) never included `sensor_noise` in its
`dataclasses.replace(...)` call -- the missing piece that would have made
even a *fixed* fingerprint/override list fail to matter in practice.

**Fix:**
- `evaluation/fingerprint.py`: `EVALUATION_CONTRACT_SECTIONS` now includes
  `"sensor_noise"`.
- `evaluation/contract_override.py`: `CONTRACT_SECTION_NAMES` +
  `_CONTRACT_SECTION_TYPES` include `sensor_noise` / `SensorNoiseConfig`.
- `env/simulation/environment_node.py::_resolve_evaluation_contract_override`'s
  restore branch (override cleared) now also restores
  `sensor_noise=self._launch_profile.sensor_noise`.
- `nodes/evaluation_node.py::build_effective_profile` now also sets
  `sensor_noise=eval_overrides.sensor_noise`.
- Clarified (module docstrings + the `check_sensor_overrides_supported`
  error message in `env/randomization/domain_randomizer.py`) that this is
  UNRELATED to that module's own TRAIN-ONLY, per-episode-randomized
  `sensor:` benchmark-scenario override axis (still rejected outright for
  fixed benchmarks) -- only the evaluation PROFILE's `sensor_noise` section
  may make a fixed benchmark noisy.

**Unit tests** (`tests/test_fingerprint.py`, `tests/test_contract_override.py`,
166 combined incl. new): fingerprint changes when `sensor_noise` changes
and is otherwise ignored correctly by the other fingerprint; write/load/
apply round-trip including a noise-ENABLED section; the exact fairness
regression (`build_effective_profile` applies the requested profile's
`sensor_noise`, never the checkpoint's own).

**Live evidence.** Same trained checkpoint (`smoke_test_stability`,
`sensor_noise.enabled=true` at training time), evaluated against the SAME
5-scenario `id` benchmark under two eval profiles differing ONLY in
`sensor_noise`:

```
ros2 run hunter_kinodynamic_rl evaluation_node.py --ros-args \
  -p profile:=evaluation_id_noise_check_disabled \
  -p checkpoint_dir:=runtime/experiments/20260827_082405_smoke_test_stability_seed0/checkpoints \
  -p checkpoint_name:=final
# evaluation_contract_fingerprint == live_evaluation_contract_fingerprint == aa662c27...

ros2 run hunter_kinodynamic_rl evaluation_node.py --ros-args \
  -p profile:=evaluation_id_noise_check \
  -p checkpoint_dir:=runtime/experiments/20260827_082405_smoke_test_stability_seed0/checkpoints \
  -p checkpoint_name:=final
# evaluation_contract_fingerprint == live_evaluation_contract_fingerprint == ace0b997...
```

Both runs exit 0, both self-verify `live_evaluation_contract_fingerprint ==
evaluation_contract_fingerprint` (evaluation_node.py's own fail-fast check),
and the two fingerprints differ from each other -- proving the SAME
checkpoint is evaluated noise-free or noisy purely based on the requested
eval profile, never its own training-time setting.

## item 3: reset initial-frame noise duplication

**Root cause.** `_on_reset` called `self._frame_stack.reset(self._observation_obs_state())`
(1st noisy LiDAR sample, fills every history slot with it), then
`state = self._build_state_vector()`, whose OWN body called
`self._observation_obs_state()` AGAIN (2nd, DIFFERENT noisy sample) and
pushed it -- double RNG consumption at reset, and the frame stack's newest
slot ended up different from every older slot even at t=0 (violating
`FrameStack`'s own documented warm-start contract).

**Fix** (`env/simulation/environment_node.py`): split `_build_state_vector`
(per-step: sample once, push, delegate) from a new `_assemble_state_vector`
(builds the observation from whatever's ALREADY in the frame stack, never
samples/pushes). Reset now calls `_frame_stack.reset(self._observation_obs_state())`
(one sample) then `_assemble_state_vector()` directly -- never
`_build_state_vector()`.

**Tests** (`tests/test_environment_node.py`, new): frame-stack-identical-
after-reset under both domain-randomization and `sensor_noise` LiDAR noise;
a direct RNG-position proof that the `sensor_noise` stream advances by
exactly one `apply_lidar_noise` call at reset, not two.

**Live evidence.** 40 live resets against a running `environment_node`
(`smoke_test_stability`, `sensor_noise.enabled=true`) via a script driving
`EnvironmentClient` directly: 0 errors, 0 risk/diagnostics mismatches.

## item 4: obstacle-pool consistency

Four separate bugs in `env/spawning/obstacle_pool.py`:

1. **`activate_dynamic` never set `.active`** -- every dynamic slot kept
   whatever its dataclass default was forever, so the active/parked COUNT
   `environment_node.py` logs every reset (`active_dynamic=X/Y`) was
   disconnected from reality for the dynamic pool. Fixed: every slot's
   `.active` is now set explicitly (True/False) on every call.
2. **`ensure_spawned` had no per-slot retry recovery** -- a partial
   failure (some entities spawned, then one `SpawnEntity` call failed) left
   `pool.ready=False`, and the NEXT call re-ran the spawn loop from
   scratch, re-issuing `SpawnEntity` for entities that already existed.
   Fixed: new `_StaticSlot.spawned`/`_DynamicSlot.spawned` per-slot flags;
   a retry only (re-)attempts unspawned slots; `pool.ready` is set True
   only once every slot is individually confirmed.
3. **`activate_static` escalated to a LARGER size class** when its exact
   class ran out of free slots -- spawns geometry bigger than
   `ScenarioSpec.radius`, a silent mismatch against what
   feasibility/risk computation assumed. Fixed: only searches the EXACT
   needed class; raises `RuntimeError` immediately if none free (never
   escalates). `Profile.validate()`'s existing per-class worst-case
   capacity check (`obstacle_pool.max_static` split across
   `static_size_classes_m` must individually cover `scenario.max_obstacles`)
   already gives the correct necessary-and-sufficient guarantee for this
   no-escalation policy -- comment updated to describe the new contract,
   logic unchanged (it was already exactly right).

**Tests** (`tests/test_obstacle_pool.py`, `tests/test_config.py`, 74
combined incl. new): dynamic slot active/parked correctness across varying
counts; partial-spawn-failure-then-retry never re-spawns an already-spawned
slot (a `_FlakyClient`/`_FlakyNode` simulate a mid-loop `SpawnEntity`
failure); exact-class-only allocation now raises instead of escalating.

**Live evidence.** All 40 resets above logged
`obstacle_pool active_static=N/6 active_dynamic=0/2 parked_static=(6-N)
parked_dynamic=2` with `active+parked` always summing to the configured
pool size, and zero `"not found, so not removed"` lines anywhere in
`environment_node`'s log across the whole session (Gazebo launch, 40
resets, a full 200-step training run, an interrupted+resumed run, two
evaluation runs).

## item 5: GT-vs-noisy observation diagnostics

New module `env/simulation/sensor_diagnostics.py` -- a SEPARATE, versioned
(`schema_version=1`) `Float32MultiArray` wire format on
`/hunter_kinodynamic_rl/sensor_diagnostics` (mirrors `risk_telemetry.py`'s
own approach; never folded into that channel's wire format). Fields: schema
version, `step_id`, `valid`/`invalid_reason`, `reset_generation`,
`episode_id` (seed), `sim_timestamp_sec`, GT and noisy `(x, y, yaw)`, GT and
noisy `(v, yaw_rate, steering)`, current OU localization drift
`(x, y, yaw)`, configured `localization_latency_steps`, LiDAR beam count,
LiDAR dropout count (inferred from the "reads as max_range" convention),
and LiDAR perturbation mean/max (noisy-vs-ground-truth per-beam deviation).

**Wiring:**
- `environment_node.py::_assemble_state_vector` (the ONE place ground truth
  and the agent's actual observation are both already local variables,
  shared by reset and every `/step`) computes and publishes exactly one
  record per call via a new `_compute_and_publish_sensor_diagnostics`,
  mirroring `_compute_and_publish_risk`'s try/except-then-publish shape --
  a computation failure degrades to `invalid(..., COMPUTATION_EXCEPTION)`,
  never crashes the reset/step it's attached to.
- `training/trainer_base.py`: `_RiskTelemetryListener` now ALSO subscribes
  to `sensor_diagnostics` (same isolated node/executor, away from `/clock`'s
  extreme publish rate -- the documented starvation problem was specific to
  `/clock`, not general multi-subscription contention). New
  `sensor_diagnostics_matches_step` pure predicate +
  `_await_matching_sensor_diagnostics` (same bounded-poll/staleness-
  rejection shape as risk_telemetry's own, independent timeout/matched
  counters). `EnvironmentClient.step()` now returns an 8-tuple (telemetry,
  diagnostics) -- every call site updated (`trainer_base.py`'s main loop
  and validation loop, `evaluation/benchmark_runner.py::run_episode`).
- `training/run_logger.py::log_step` gained a `sensor_diagnostics` param:
  when provided, `steps.jsonl` gets a `sensor_diagnostics` object with GT
  and noisy values recorded SEPARATELY (`gt_pose`/`noisy_pose`,
  `gt_velocity_mps`/`noisy_velocity_mps`, etc.), NaN->`null` on a poll
  timeout (never a stale/fabricated value) with `invalid_reason` recording
  why.

**Tests** (`tests/test_sensor_diagnostics.py` new, 13; `tests/test_environment_node.py`
new; `tests/test_run_logger.py` new; `tests/test_trainer_telemetry_sync.py`
new + fixture updated for the new 8-tuple `step()` return; `tests/test_benchmark_runner.py`,
`tests/test_checkpoint_resume_determinism.py`, `tests/test_trainer_sensor_stale.py`,
`tests/test_checkpoint_boundary_wiring.py`, `tests/test_trainer_validation.py`,
`tests/test_cross_algorithm_benchmark_fairness.py` fake-env fixtures updated
for the new arity): encode/decode round-trip, GT-vs-noisy divergence when
`sensor_noise` is enabled and byte-identity when disabled, step_id/
reset_generation alignment against risk_telemetry, timeout -> null+reason.

**Live evidence.** The 40-reset script above: 0 mismatches between
`risk_telemetry` and `sensor_diagnostics` `(reset_generation, step_id)`
pairs across 240 steps, 0 diagnostics timeouts, 0 telemetry timeouts, and
every single step showed `gt_x != noisy_x` (sensor_noise genuinely
perturbing the observation). The real 200-step training run's
`logs/steps.jsonl` contains real `sensor_diagnostics` records end-to-end,
e.g.:
```json
{"valid": true, "step_id": 1, "reset_generation": 41,
 "gt_pose": {"x": 2.7688, "y": -2.0314, "yaw": -2.1474},
 "noisy_pose": {"x": 2.7707, "y": -2.0236, "yaw": -2.1475},
 "localization_drift": {"x_m": -0.00176, "y_m": 0.00353, "yaw_rad": 0.0},
 "lidar_beam_count": 80, "lidar_dropout_count": 0,
 "lidar_perturbation_mean_m": 0.00858, "lidar_perturbation_max_m": 0.0602}
```

## item 6: exact discrete-time OU drift

**Root cause.** `sensor_noise._ou_step` used first-order Euler-Maruyama
(`current - theta*current*dt + N(0, sigma*sqrt(dt))`) -- accumulates
discretization error growing with `theta*dt`, and can numerically
oscillate/diverge once `theta*dt > 2` (the ordinary explicit-Euler
instability threshold), unlike the true OU process, which always decays.

**Fix** (`env/simulation/sensor_noise.py`): closed-form exact discrete
update `X' = X*exp(-theta*dt) + N(0, sqrt(sigma^2/(2*theta) * (1 -
exp(-2*theta*dt))))` for `theta > 0`; `theta <= 0` (only ever exactly 0 in
practice, schema validation rejects negative) reduces to plain driftless
Brownian motion (`X + N(0, sigma*sqrt(dt))`), identical to what
Euler-Maruyama also computes for `theta=0`. `sigma_m <= 0` or `dt_sec <= 0`
remain no-ops, preserving `reset_state`'s documented "sigma=0 disables the
drift axis entirely" byte-identical contract. `sigma_m` is now explicitly
documented as the SDE's diffusion coefficient (units
`<state>/sqrt(sec)`), not a per-step standard deviation --
`config/schema.py`'s `SensorNoiseConfig` docstring updated to match.

**Tests** (`tests/test_sensor_noise.py`, new): `theta=0` matches the
Euler-equivalent closed form exactly (same RNG stream); `dt=0` and `sigma=0`
no-ops (including the documented "no decay either" contract for
`sigma=0`); large `theta*dt` (`1e6 x 0.1 = 1e5`) stays finite and collapses
to the stationary noise band instead of the explicit-Euler
oscillate-then-diverge failure mode; same-seed reproducibility and
finiteness over a 20,000-tick run for both `theta>0` and `theta=0`.

## Full-suite verification

```
docker exec 7a2702b311a1 bash -lc "source /opt/ros/humble/setup.bash; \
  source /root/DRL_Robot_Path_Planning/ros2_ws/install/setup.bash; \
  cd /root/DRL_Robot_Path_Planning/ros2_ws && \
  colcon build --packages-select hunter_kinodynamic_rl --cmake-args -DCMAKE_BUILD_TYPE=Release && \
  colcon test --packages-select hunter_kinodynamic_rl && \
  colcon test-result --all --verbose"
```
Result: **1162 tests, 0 errors, 0 failures, 0 skipped** (1157 pytest +
3 package-structure tests + `lint_cmake` + `xmllint`, all green). Every
existing test that asserted the PRE-fix `activate_static` escalation
behavior was rewritten (not deleted) to assert the new fail-fast contract,
since that behavior change was an explicit, deliberate part of item 4.

## Live Gazebo/ROS2 smoke test

```
ros2 launch hunter_se_gazebo simulate_hunter_se_ignition.launch.py rviz:=false
ros2 run hunter_kinodynamic_rl environment_node.py --ros-args -p profile:=smoke_test_stability
# 40 resets x 6 steps via a script driving EnvironmentClient directly (reset/step/seed)
ros2 run hunter_kinodynamic_rl train_node.py --ros-args -p profile:=smoke_test_stability          # full 200-step run
ros2 run hunter_kinodynamic_rl train_node.py --ros-args -p profile:=smoke_test_stability_resume_check  # interrupted (SIGINT) at global_step=59, resumed, reached 368 before being stopped
ros2 run hunter_kinodynamic_rl evaluation_node.py --ros-args -p profile:=evaluation_id_noise_check[_disabled] ...
```
- Gazebo: RGL GPU LiDAR loaded (RTX 4080), `/clock` at ~888 Hz -- no
  CPU-fallback low-RTF failure mode.
- 0 tracebacks, 0 `"not found, so not removed"` lines, anywhere in
  `environment_node`'s log or the Gazebo-side `~/.ros/log/*` session logs
  across the entire session (Gazebo launch through both evaluation runs).
- Checkpoint/resume: the interrupted run's `global_step` was 59 when
  killed; resuming from `checkpoints/latest` continued to 368 (monotonic,
  never restarted from 0), consistent with `SeedScheduler`'s own persisted
  state (seed/noise continuity, since `sensor_noise.reset_state` derives
  purely from the episode seed -- see that module's own docstring).
- The completed run's `checkpoints/{best,latest,final}` symlinks all
  resolved correctly to distinct `.generations/<hash>/` entries.

## Known limitations (not fixed in this pass -- out of scope for
## requirements 1-6, discovered incidentally during live verification)

- **Pre-existing, unrelated schema gap**: `Profile.validate()`'s
  obstacle-pool worst-case-capacity cross-check
  (`obstacle_pool.enabled` -> `max_static` split across
  `static_size_classes_m` must cover `scenario.max_obstacles`) runs
  UNCONDITIONALLY, even though `obstacle_pool.py`'s own module docstring
  states the pool is NEVER used for fixed-benchmark evaluation. Evaluating
  the `smoke_test_stability` checkpoint (`obstacle_pool.max_static=6`,
  tuned for its own `scenario.max_obstacles=2`) against the shipped
  `evaluation_id` profile (default `scenario.max_obstacles=8`) fails this
  check even though the pool is never actually touched at runtime for that
  path. Worked around for this pass's own live evaluation by using a
  `scenario.max_obstacles=2`-compatible eval profile; not fixed, since it
  is unrelated to any of requirements 1-6 and touching it risked scope
  creep into `Profile.validate()`'s cross-section logic.
- A `rcl_shutdown already called on the given context` cosmetic
  double-shutdown `RCLError` was observed in the exception-handling path
  when SIGINT interrupted a blocked `/step` service call mid-training
  (pre-existing in `trainer_base.py`'s shutdown-on-failure path, unrelated
  to items 1-6) -- not fixed.
- The interrupted/resumed training run was stopped (SIGTERM) at
  `global_step=368` of its `3000` target rather than run to full
  completion, once continuity past the interruption point was confirmed --
  a full-length run was not required to prove the resume mechanism works.
- Dynamic-obstacle-pool active/parked correctness is thoroughly unit-tested
  (`tests/test_obstacle_pool.py`), but the live smoke session's scenario
  config never actually populated a dynamic obstacle (`active_dynamic=0`
  throughout, since `scenario.dynamic_obstacle_count` was left at its
  profile default) -- the live logs never show a live nonzero
  `active_dynamic` count as independent confirmation.
- `colcon test` registered `lint_cmake` + `xmllint` + the pytest suite;
  whether `ament_flake8`/`ament_pep8` are wired into this package's own
  lint targets was not separately confirmed (they did not appear as
  distinct test entries in `colcon test-result`).
