# Benchmark

This document defines evaluation datasets, artifact contracts and metrics. It
does not declare the current implementation complete; current readiness and
known blockers are tracked in `CURRENT_STATUS.md`. The research questions and
planned L0–L8/G0–G8 ladder are defined in `RESEARCH_ROADMAP.md` and
`RESEARCH_PROTOCOL.md`; this file defines how those comparisons are executed
and recorded.

## Fixed scenario sets

`config/benchmarks/{id,ood_geometry,ood_dynamics,dynamic}/*.yaml` -- loaded
by `env/scenarios/benchmark_loader.load_benchmark(name)`. Each file:

```yaml
scenario_id: <string>
seed: <int, from scenario.test_seed_range>
start: {x, y, yaw}
goal: {x, y}
static_obstacles: [{x, y, radius}, ...]
moving_obstacles: [{x0, y0, vx, vy, radius}, ...]
dynamics: {}      # OOD-dynamics overrides (friction_scale, steering_delay_sec, ...)
sensor: {}        # sensor-noise overrides
```

The current engineering fixture contains 5 ID, 4 OOD-geometry, 1
OOD-dynamics and 4 dynamic scenarios. `tests/test_benchmark_scenario_coverage.py`
loads the files directly, checks minimum ID/OOD-geometry/dynamic counts and
verifies start/goal clearance and grid reachability for those three sets. The
single OOD-dynamics case and the overall fixture size are still insufficient
for a paper claim;
freeze a larger versioned manifest under the protocol below before formal use.

## Research benchmark matrix (target)

The current four benchmark directories are engineering fixtures, not a paper
dataset. Before formal Local uncertainty/residual results, freeze a larger
manifest with these orthogonal condition groups:

| Group | Purpose | Minimum controlled axes |
|---|---|---|
| ID-static | in-distribution navigation and calibration | seen parameter ranges, unseen test seeds |
| OOD-geometry | structural generalization | held-out layout family, corridor width, nonholonomic traps |
| OOD-dynamics | model mismatch and residual uncertainty | friction, payload, steering delay/rate/offset, acceleration/braking, command latency |
| OOD-sensor | observation robustness | LiDAR noise/dropout/frame loss, odometry noise/yaw bias |
| OOD-localization | covariance/risk propagation | injected pose error and accumulated drift, not confidence-only change |
| Dynamic | moving-obstacle robustness | speed, crossing direction, occlusion and encounter timing |
| Real | sim-to-real evidence | open, corridor and dead-end/loop sites with repeated missions |

One-axis OOD sweeps establish attribution; a separately named combined-stress
set measures robustness. Training randomization ranges, validation calibration
ranges and test-only OOD ranges must be stored in the result artifact. A held-
out seed from the same topology generator is not by itself an OOD-layout claim.

## Running a benchmark

```bash
ros2 run hunter_kinodynamic_rl evaluation_node.py --ros-args \
  -p profile:=evaluation_ood_geometry \
  -p checkpoint_dir:=<run_dir>/checkpoints -p checkpoint_name:=final
```

Writes `episodes.csv`, `episodes.jsonl`, and `summary.json` (via
`evaluation/result_writer.py`) under `<checkpoint_dir>/../evaluation/<benchmark>/`.

## Exact scenario placement

`benchmark_runner.py` places each scenario EXACTLY, not procedurally:
before every `/reset`, `EnvironmentClient.set_scenario_override(path)` sets
`environment_node.py`'s `scenario_override_path` ROS parameter to that
scenario's own YAML file. `_resolve_episode_scenario()` (`env/simulation/
environment_node.py`) checks this parameter FIRST -- when set, it loads
the file's exact `start`/`goal`/`static_obstacles`/`moving_obstacles`
coordinates via `load_scenario_file()` and returns `is_fixed=True`
(skipping procedural generation, and domain randomization, entirely --
see `test_domain_randomization_never_applies_to_a_fixed_benchmark_scenario`
in `tests/test_environment_node.py`). `run_benchmark()` clears the override
(`set_scenario_override("")`) when done, restoring the environment node's
normal seed-driven procedural mode for any subsequent (non-benchmark) use.
`set_scenario_override()` returning `False` (the environment refused it --
e.g. the target scenario file doesn't exist server-side) now raises
immediately (`run_benchmark`) rather than silently proceeding against
whatever scenario the environment happened to already have loaded.

## Fairness across model types (2026-08-25, extended 2026-08-26)

A benchmark is only meaningful if every model type evaluated on it saw the
IDENTICAL world size, episode length, termination conditions, AND physics/
timing -- previously, `run_episode`'s episode-length cutoff read
`profile.training.episode_length_steps` (the CHECKPOINT's OWN training
profile), so two checkpoints trained with different episode lengths were
silently evaluated under DIFFERENT cutoffs on the "same" benchmark.

- **`evaluation.max_episode_steps`** (`EvaluationConfig` field) is the sole
  episode-length cutoff `run_episode` uses CLIENT-side -- a property of the
  BENCHMARK/evaluation profile, never the checkpoint's training config.
  Set it explicitly (or accept the shared default) in every
  `evaluation_*.yaml` profile used to compare models on the same
  benchmark.
- **The LIVE environment's own SERVER-side timeout is ALSO
  `evaluation.max_episode_steps` during a fixed-benchmark episode** (2026-08-26
  fix, `environment_node.py`'s own `/step` handler) -- not
  `training.episode_length_steps` (this node's OWN launch/checkpoint-
  training profile's budget). Previously only the client-side loop bound
  used the evaluation budget; the live server could still force `done=True`
  earlier (if its own training profile's episode length was shorter) or
  later (if longer), silently truncating/extending a benchmark episode
  differently per checkpoint. Non-benchmark (procedural train/validation/
  test) episodes are unaffected -- they still use
  `training.episode_length_steps` exactly as before.
- **The live environment's world boundary/reward/termination/runtime ALSO
  come from the requested evaluation profile now** (2026-08-26 fix) -- see
  "Evaluation-contract delivery" below. Previously these ONLY affected this
  node's own local bookkeeping; the live environment kept using whatever
  ITS OWN launch-time (== checkpoint training) profile set for
  `scenario.world_size_m`/`reward`/`runtime`, regardless of the requested
  `--profile`.
- **Every episode is now classified**: if the step loop exhausts without
  the environment ever reporting `done`, `timeout=True` is set (a Python
  `for...else` clause) -- `success`/`collided`/`timeout` can no longer all
  be `False` for the same episode.
- **Common, architecture-independent metrics**
  (`env/simulation/risk_computation.py::compute_common_evaluation_metrics`,
  gated on `evaluation.common_metrics_*` config, active whenever
  `environment_node.py` is running a fixed-benchmark episode): rolls out
  the REALIZED, ACTUALLY-published physical command (not the checkpoint's
  own internal action representation) through the same
  clearance/TTC/unrecoverable-state machinery every risk-aware model uses,
  so `SAC`/`legacy_waypoint`/vanilla-TQC checkpoints -- none of which have
  a native trajectory-rollout concept of their own -- get the SAME
  clearance/TTC/stopping-margin/collision/unrecoverable telemetry fields
  populated as a risk-aware model, without ever handing privileged
  obstacle ground truth to the policy's own observation. Since
  `evaluation.common_metrics_*` is itself part of the delivered evaluation
  contract (see below), this is computed IDENTICALLY across algorithms too.

## Evaluation-contract delivery to the live environment (2026-08-26 addition)

`evaluation_node.py` restores the checkpoint's own training ARCHITECTURE
locally (`build_effective_profile`), but the LIVE `environment_node.py` is
a SEPARATE, already-launched ROS process whose own `self.profile` was
resolved once at ITS launch -- it cannot see what `evaluation_node.py`
computed in-process. `main()` therefore also WRITES the effective profile's
`reward`/`scenario`/`runtime`/`evaluation` sections to a file
(`evaluation/contract_override.py::write_evaluation_contract_override`) and
pushes it to the live environment via
`EnvironmentClient.set_evaluation_contract_override` -- mirroring
`scenario_override_path`'s own file-based-override delivery pattern.
`environment_node.py::_resolve_evaluation_contract_override` applies it at
the TOP of every `/reset` (before anything reads `self.profile.scenario`/
`reward`/`runtime`), including re-deriving the runtime-CACHED fields
`gazebo_runtime.py`'s hot path uses (`time_delta`,
`deterministic_stepping`, ...) -- never just `self.profile.runtime` alone.
Clearing the override (`""`) restores the environment's own launch-time
contract sections. Deliberately narrow: only `reward`/`scenario`/`runtime`/
`evaluation` are ever replaced -- `action_space`/`features`/`observation`/
`robot`/`dynamics`/`risk`/`counterfactual`/`hyperparameters`/`algorithm`
(everything that determines the checkpoint's own network shape) are never
touched by this mechanism.

`main()` verifies the override was ACTUALLY applied (not just that the
parameter-set RPC returned success) by forcing one warm-up `/reset` and
reading back the live environment's `evaluation_contract_fingerprint_sha256`
parameter -- see "Fingerprint verification" below.

## Fingerprint verification (2026-08-25, extended 2026-08-26)

Comparing scenario IDs or profile NAMES cannot detect the underlying YAML
CONTENT silently drifting while the name stays the same. Two independent
questions, both SHA-256-based (`evaluation/fingerprint.py`):

- **Architecture fingerprint** (`architecture_fingerprint(profile)` /
  `architecture_fingerprint_from_resolved_config(resolved_config)`): hashes
  the network-shape/action-decode-relevant sections
  (`action_space`/`features`/`observation`/`robot`/`dynamics`/`risk`/
  `counterfactual`/`hyperparameters`/`sac_hyperparameters`/`algorithm`).
  `environment_node.py` exposes its own resolved value as the
  `architecture_fingerprint_sha256` ROS parameter ONCE at launch (never
  re-set afterward -- an evaluation-contract override never touches
  architecture); `evaluation_node.py` compares it against the checkpoint's
  own (computed from the manifest's frozen `resolved_config`) and raises if
  a live environment reports the SAME profile name but a DIFFERENT
  fingerprint -- the YAML changed content without the name changing.
- **Evaluation-contract fingerprint** (`evaluation_contract_fingerprint(profile)`):
  hashes the benchmark-CONDITION sections -- `evaluation`/`reward`/
  `scenario`/**`runtime`** (added 2026-08-26: `time_delta_sec`/
  `deterministic_stepping`/etc. change episode dynamics just as much as
  reward/world-size do) -- a deliberately DIFFERENT question from the
  architecture fingerprint. UNLIKE `architecture_fingerprint_sha256`,
  `environment_node.py`'s `evaluation_contract_fingerprint_sha256`
  parameter is RE-SET every time the active contract sections change
  (`_resolve_evaluation_contract_override`), so it always reflects what the
  live environment is ACTUALLY running, not just launch time.
  `evaluation_node.py::validate_live_evaluation_contract` reads it back
  after applying its own override and raises if it doesn't match this
  run's own REQUESTED effective evaluation-contract fingerprint --
  verifying the override was genuinely applied, not merely accepted.
  `summary.json` records both `evaluation_contract_fingerprint` (requested)
  and `live_evaluation_contract_fingerprint` (what the live environment
  reported back) plus `evaluation_contract_override_path` (provenance: the
  exact file the contract was delivered through, preserved alongside the
  run's other output).
- `run_benchmark()` also records each scenario YAML's own SHA-256
  (`scenario_file_sha256`, per scenario entry) and an overall
  `benchmark_manifest_sha256` (hash of the sorted `{scenario_id:
  scenario_file_sha256}` map) in `summary.json` -- two independently
  produced result files can be verified, from the files alone, to have
  used byte-identical scenario definitions, not just matching IDs/seeds.

## Checkpoint layout (2026-08-25 change)

`-p checkpoint_name:=<tag>` now resolves to a DIRECTORY (a symlink to a
`.generations/<uuid>/` staging directory containing `model.pt` +
`manifest.json` + `replay.npz` together), not `<tag>.pt`/`<tag>.json`
sibling files -- see `rl/checkpointing/manager.py`'s module docstring. The
CLI itself is unchanged; only checkpoints saved by an EARLIER (pre-this-fix)
session need the explicit `manager.load_legacy_flat` reader instead of the
default `load_generation`.

`load_generation` resolves the `<tag>` symlink to its concrete generation
directory EXACTLY ONCE (2026-08-26 fix) -- every file it reads, and the
`replay_path`/`generation_dir` it hands back to the caller, all target that
one resolved directory, immune to a CONCURRENT `save_generation` call
republishing `<tag>` to a different generation mid-load (previously, a
caller's later `ReplayBuffer.load(result["replay_path"])` could silently
read a DIFFERENT generation's replay file than the one `load_generation`
had just verified). `manager.prune_orphan_generations` is a separate,
EXPLICIT, opt-in maintenance utility (never called automatically) for
removing old unreferenced `.generations/<uuid>/` directories.

## Metrics (`evaluation/metrics.py::aggregate`)

Every field name embeds its own unit; `*_valid_count` sits beside any
`*_mean`/`*_worst` metric that can be legitimately undefined (e.g. zero
risk-valid steps in the whole benchmark), so a reader can tell "no data"
apart from a genuine 0.0 -- undefined metrics serialize as JSON `null`,
never a bare zero or a non-standard `Infinity` literal.

| Field | Definition | Source |
|---|---|---|
| `success_rate` | fraction of episodes reaching the goal (`reward_calculator.is_goal_reached`, position-only) | env `/step` `target` flag |
| `collision_rate` | fraction of episodes ending in collision | env `/step` `collision` flag |
| `timeout_rate` | fraction of episodes classified `timeout` -- explicitly set whenever the step loop exhausts `evaluation.max_episode_steps` without the environment ever reporting `done` (never left `False` alongside `success`/`collided` also `False`) | `run_episode`'s own bookkeeping |
| `unrecoverable_state_rate` | fraction of episodes where any step's risk label reported `unrecoverable=True` | risk telemetry |
| `spl` | Success weighted by Path Length: `straight_line_distance_m / max(path_length_m, straight_line_distance_m)` for successes, 0 for failures | `success_path_length()` |
| `navigation_time_steps_mean` | mean episode length, control ticks (not seconds) | `run_episode`'s step counter |
| `navigation_time_sec_mean` | mean episode duration, **simulation** time (`/clock`-derived `episode_elapsed_sim_time_sec`), never wall-clock (Nav2-MPPI baseline included -- see `evaluation/nav2_mppi_runner.py::navigation_time_sec`) | `EnvironmentClient.episode_elapsed_sim_time_sec` |
| `path_length_m_mean` | mean odometry-integrated traveled distance, meters | `EnvironmentClient.episode_path_length_m` (cumulative `hypot` of consecutive `/odometry` positions) |
| `average_velocity_mps_mean` | mean of each episode's mean odometry-derived speed, m/s (signed) | `EnvironmentClient.latest_v_mps` samples |
| `min_clearance_m_mean` / `_worst` | mean / minimum of the risk label's `min_clearance_m` (predicted forward-rollout clearance, meters, **not** the robot's collision radius) across all risk-valid steps | risk telemetry |
| `ttc_sec_mean` / `_worst` | mean / minimum time-to-collision, seconds -- computed ONLY from steps where a collision was actually found within the rollout horizon (real, uncensored measurements); steps where none was found are excluded here, not averaged in as if the horizon itself were a measured TTC | risk telemetry, `risk/ttc.py` |
| `collision_free_rate_mean` | fraction of risk-valid steps where the rollout found no collision within its horizon (the TTC "censoring" rate) -- reported separately from `ttc_sec_mean` precisely because those two must never be conflated | risk telemetry |
| `steering_saturation_rate_mean` | fraction of steps commanding steering within 1e-3 rad of `robot.steering_limit_rad` | joint-state-derived steering |
| `steering_smoothness_mean` | mean absolute steering change between consecutive steps, rad (lower = smoother) | joint-state-derived steering |
| `control_smoothness_mean` | mean absolute change in `(published_speed_mps, published_steering_rad)` between consecutive steps -- the ACTUALLY-published command (post safety-guard, post command-latency queue), not the policy's raw/nominal action | risk telemetry's `published_*` fields |
| `emergency_stop_count_total` | total count of steps where the safety guard forced a stop, summed across all episodes | risk telemetry's `emergency_stop` flag |

Every distance is meters, every duration is seconds (simulation time unless
explicitly a step count), every angle is radians, every velocity is m/s --
consistent with `CLAUDE.md`'s "DRL State/Action Space" unit conventions
used throughout the rest of this package.

## Hierarchical Local benchmark contract

The arbitrary-subgoal Local benchmark uses an immutable test-pool manifest and
records checkpoint generation/SHA, architecture fingerprint, Local training
contract, package provenance and per-episode termination. Promotion must verify
the artifact against the actual source generation rather than trust caller-
supplied identity strings.

Feasible and intentionally infeasible episodes have different denominators.
An infeasible goal that merely exhausts the episode budget is a timeout, not an
explicit policy rejection. Unless the policy/environment exposes a real reject
signal, report observable conditional metrics instead:

- `infeasible_scenario_count`
- `infeasible_false_success_rate`
- `infeasible_collision_rate`
- `infeasible_high_risk_rate`
- a precisely defined collision-free/safe-termination rate

Every infeasible-only rate uses `infeasible_scenario_count` as its denominator
and carries a valid count. A formal artifact requires exact one-to-one equality
between manifest and episode `(scenario_id, seed)` pairs, no duplicate scenario
IDs, test mode only, matching summary counts and a supported schema version.

**Current status:** the audited implementation now uses schema v2 conditional
metrics (`feasible_subgoal_success_rate` and infeasible-only false-success,
collision, high-risk and safe-termination rates), preserves `None` for a zero
denominator, and rejects the obsolete rejection-rate schema during promotion.
This closes the code-level metric defect but does not constitute a completed
formal benchmark; see `CURRENT_STATUS.md`.

## Hierarchical A/B and A–G formal contract

`formal` means all of the following, not merely a JSON label:

- test split and one immutable manifest with at least 20 scenarios;
- complete results for every requested scenario and ablation;
- trained, strict-compatible Global checkpoints rather than heuristics;
- the exact promoted Local generation used by Global training;
- no smoke-only option/local-step budget overrides;
- strict resolved-config, observation/replay schema, RNG, SHA and Local-identity
  checks;
- package SHA/dirty/diff provenance and manifest content hash;
- per-scenario episode rows, not summaries alone.

Missing B–G checkpoints may be reported as skipped in a smoke suite. A formal
A–G suite with any requested label missing is incomplete and must not be marked
formal.

The current frozen-Local evaluator records timeout/error/non-finite fallback
reasons in artifact telemetry, but its candidate feature tensor still
zero-fills `predicted_action_risk` and `progress_preserving` when evaluation is
invalid. That zero is not evidence of low risk. Until the observation schema
has an explicit per-candidate validity/unknown mask, a formal E/F/G claim must
require zero raw action/risk fallback counts, report every fallback reason,
and include a sensitivity analysis if any affected decision is retained for
diagnosis. Do not gate on the emitted legacy `fallback_rate`: its denominator
is per decision while some numerator fields are per candidate, so it is not a
bounded probability. Any nonzero formal fallback makes the artifact incomplete.

## Long-horizon datasets

The 24–40 m procedural worlds are engineering and ablation environments. A
strong long-horizon paper claim additionally requires 50–100 m routes, a held-
out layout generator or real floor plans, multiple consecutive dead ends and
loops, nonholonomic traps and missions long enough for localization drift to
accumulate. The simulator's full map remains evaluation-only for solvability,
shortest feasible path and difficulty labels.

## Local research ablation artifact

Every Local L0–L8 row records, in addition to the existing navigation metrics:

- exact action contract and path primitive (`(v, steering)`, `[kappa,v]`,
  `[kappa,v,L]`, constant-curvature/spline/Bezier);
- risk output schema, factor aggregation and ensemble member identities;
- nominal/residual model generation, training-data hash and calibrator hash;
- counterfactual candidate generator, candidate budget, constraints
  `(rho, Delta_R, epsilon_u)` and target/abstention activation counts;
- raw, nominal, guarded and published command streams;
- inference latency for actor, candidate rollout, risk ensemble and full tick.

Compare all learned rows on identical scenario IDs and paired condition draws.
Do not use a larger rollout/candidate budget only for the proposed method
without reporting the compute-matched result.

## Risk and dynamics model metrics

Navigation return is not a model-validation metric. A formal risk artifact
reports, per factor and condition:

- AUROC/AUPRC and false-negative rate for collision/unrecoverable events;
- Brier score, ECE, reliability bins and sample count;
- clearance, TTC and stopping-margin error with explicit TTC censoring;
- ensemble spread versus actual error, selective risk versus retained coverage,
  and OOD detection AUROC where an OOD detector claim is made;
- calibration before/after a validation-only calibrator;
- mean, tail and worst-case inference latency.

A formal residual-dynamics artifact reports one-step response error, open-loop
multi-step pose/heading/endpoint error versus horizon, and downstream risk-
factor error for `nominal`, `single residual` and `residual ensemble` models.
Split system-ID data by complete run/rosbag so adjacent samples from one
trajectory cannot leak across train and test.

## Safety and risk reporting

Keep these command stages separate in every result:

1. raw policy action;
2. nominal decoded vehicle command;
3. guarded command;
4. actually published command.

Report guard intervention rate, proposed high-risk-action rate, emergency-stop
recovery and raw-vs-guarded collision/risk. Risk-critic evaluation additionally
reports AUROC, AUPRC, Brier score, ECE, reliability diagram, false-negative rate
and latency/error against exact rollout. Ensemble methods additionally report
uncertainty-error correlation, selective-risk/coverage curves and calibration
under every registered OOD axis.

The guard-attribution table contains four conditions: Raw Actor, Raw Actor +
Counterfactual learning, Actor + Guard, and Counterfactual Actor + Guard.
Hardware never disables the mandatory guard; raw hardware safety is evaluated
from proposal telemetry or replay. A collision reduction accompanied only by a
higher unconditional stop rate is reported as conservative stopping, not
policy improvement.

`time_to_goal` includes successful episodes only. Failed episodes contribute to
`time_to_termination`, never to time-to-goal. All optional metrics carry valid
counts so unavailable data is distinct from a real zero.
