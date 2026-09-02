# Research Protocol

This document defines the hypotheses, comparisons, statistics and claim
boundaries for the research program in `RESEARCH_ROADMAP.md`. A listed target
comparison is a requirement for a future result, not evidence that its profile
or implementation already exists. Current readiness is tracked separately in
`CURRENT_STATUS.md`.

## Seeds

Configured per-profile via `scenario.{train,validation,test}_seed_range`
(default: train `[0, 9999]`, validation `[10000, 10999]`, test `[20000,
20999]`) -- see `config/training/defaults.yaml`. Ranges are validated
non-overlapping at profile-load time
(`config/schema.py::ScenarioConfig.validate`), and
`env/scenarios/procedural_generator.seed_split()` raises rather than
silently misclassifying a seed outside all three ranges. **Test seeds never
enter the replay buffer** -- the training loop (`training/trainer_base.py`)
only ever calls `env.seed()` with seeds from
`training.seed`-derived training draws; benchmark evaluation
(`evaluation/benchmark_runner.py`) only ever seeds from a loaded benchmark
scenario's own `seed` field (`config/benchmarks/*/*.yaml`, all in the
`[20000, 20999]` test range).

## Scenarios

- **Training**: procedural (`env/scenarios/procedural_generator.py`),
  regenerated fresh every episode from `env.seed()`.
- **Evaluation**: FIXED, hand-authored YAML files under
  `config/benchmarks/{id,ood_geometry,ood_dynamics,dynamic}/` -- every
  baseline runs the identical scenario set (section 32's "모든 baseline이
  정확히 같은 scenario에서 평가되어야 한다").

## Baselines / ablation matrix

| Letter | Profile | Trajectory action | Temporal | Risk prediction | Risk-aware critic/actor | Counterfactual |
|---|---|---|---|---|---|---|
| A | `baseline_tqc` / `legacy_waypoint_tqc` | no (legacy waypoint) | no | no | no | no |
| B | `kinodynamic_tqc` | yes | no | no | no | no |
| C | `kinodynamic_tqc_temporal` | yes | yes | no | no | no |
| D | `kinodynamic_tqc_risk_supervised_only` | yes | yes | yes | no (`actor_lambda=0.0`) | no |
| E | `kinodynamic_tqc_risk` | yes | yes | yes | yes (`actor_lambda=0.1`) | no |
| F | `kinodynamic_tqc_counterfactual` | yes | yes | yes | yes | yes |

Every row is the SAME code path (`training/train_tqc.py` for A/B/C,
`training/train_kinodynamic_tqc.py` for D-F, selected automatically by
`nodes/train_node.py` from `features.risk_critic`) -- ablations are config
flags (`config/schema.py::FeatureFlags`), never a forked implementation
(section 36).

Section 34/35's remaining two comparison points are now implemented:

- **Vanilla SAC** (`rl/algorithms/sac/agent.py`, `training/train_sac.py`,
  `sac_baseline` profile, selected via `algorithm.name=sac`): standard
  twin-Q SAC sharing this package's trajectory action space/observation/
  reward with ablation row B, so only the algorithm differs. Fully
  live-verified (a real training run against Gazebo produced real
  checkpoints and a real periodic-validation event).
- **Nav2-MPPI classical baseline** (`evaluation/nav2_mppi_runner.py`,
  `config/nav2_mppi/`, `launch/nav2_mppi.launch.py`,
  `nodes/nav2_mppi_eval_node.py`): prior-map-free (no AMCL/map_server -- see that
  config file's header comment), Ackermann motion model, adapted from
  scout_nav2's verified MPPI controller block. CODE-COMPLETE and
  PARTIALLY live-verified only -- see `evaluation/nav2_mppi_runner.py`'s
  module docstring for the exact live-verification status (the stack
  configures/activates/accepts goals and its collision/telemetry pipeline
  produces real measurements, but no live attempt achieved a full
  goal-reaching episode; an unresolved low-effective-velocity issue would
  need further live iteration to root-cause).

The A–F table is the implemented engineering lineage. Paper figures use the
following research ladder and must not relabel missing variants as completed:

| Label | Required configuration | Status at roadmap adoption |
|---|---|---|
| L0 | direct-control SAC/TQC `(v, steering)` | dedicated fair comparison required |
| L1 | `[kappa, v_ref, L]` trajectory action | implemented basis |
| L2 | L1 + temporal observation | implemented basis |
| L3 | L2 + supervised risk, no actor penalty | implemented scalar-risk basis |
| L4 | L3 + risk actor penalty | implemented scalar-risk basis |
| L5 | L4 + structured candidate supervision/margin weighting | implemented basis; formal training pending |
| L6 | L5 + multi-task risk ensemble uncertainty | planned |
| L7 | L6 + learned residual dynamics | planned |
| L8 | L7 + residual ensemble uncertainty and uncertainty-gated direct counterfactual target | planned |

`[kappa,v]`, fixed-short/long `L`, adaptive `L`, and equal-capacity spline or
Bezier variants remain required factor comparisons even though they do not map
one-to-one onto existing A–F profiles.

## Metrics

Computed by `evaluation/metrics.py::aggregate()` from per-episode result
dicts: success rate, collision rate, timeout rate, Unrecoverable-State rate,
SPL, navigation time, path length, average velocity, minimum clearance, TTC
statistics, steering saturation rate, steering smoothness, control
smoothness, emergency-stop count (section 37's full list).

## Checkpoint policy

`rl/checkpointing/manager.py` saves every `nn.Module`/`Optimizer` component
an agent exposes via `checkpoint_components()` (actor, critic, critic
target, optimizers, risk critic + its optimizer when present) plus a JSON
manifest recording which components were present, training step, and seed.
Loading a checkpoint saved WITHOUT the risk critic into a risk-aware agent
degrades gracefully (the risk critic stays freshly-initialised, reported in
the manifest's `skipped` list) rather than erroring -- see
`tests/test_checkpointing.py::test_checkpoint_load_reports_skipped_component_when_absent`.

## Domain randomization

Opt-in (`domain_randomization.enabled`, default false) -- see
`config/domain_randomization/default.yaml` for the default ranges and
`env/randomization/domain_randomizer.py` for the sampler. A `RandomizationDraw`
is deterministic given `(seed, config)`
(`tests/test_env_modules.py::test_randomization_is_deterministic_per_seed`).

### OOD split contract

Training randomization bounds and test-only OOD bounds must be explicit,
non-overlapping where the hypothesis calls for extrapolation, and stored in the
resolved experiment artifact. At minimum sweep:

- tire friction and robot mass/payload;
- steering delay, rate, offset/gain and actuator dead zone;
- acceleration/braking response and command latency;
- LiDAR range noise, point/ray dropout and complete frame dropout;
- odometry noise, yaw bias, covariance and accumulated pose drift;
- held-out topology/layout family, not only unseen seeds from one generator.

Use one-axis sweeps to attribute failure and combined stress tests to measure
robustness. The primary OOD question is whether risk/uncertainty rises before
unsafe execution when the rollout model is wrong. A policy success curve alone
does not answer it. Simulator fields classified as model-only may not be
reported as physical dynamics randomization without a Gazebo-side consumer.

## Real-robot trials

Not run in this development session -- no real Hunter SE is available here.
See `docs/SIM2REAL.md`.

## Paper scope and hypotheses

The first paper, tentatively *Risk-Calibrated Counterfactual Policy Improvement
for Kinodynamic Ackermann Navigation*, is Local-method-first. Its primary
hypotheses are:

1. `[kappa, v_ref, L]` improves feasibility and smoothness over direct
   `(v, steering)` control and fixed-horizon trajectory actions.
2. Multi-task risk factors are calibrated, and ensemble uncertainty predicts
   risk-model error/OOD well enough to improve conservative action selection.
3. Physics + learned residual dynamics reduces one-step, multi-step and risk
   rollout error relative to nominal physics without sacrificing interpretability.
4. Progress-preserving, uncertainty-gated counterfactual targets reduce raw
   unsafe proposals and guard dependence without collapsing progress or
   increasing indiscriminate stops.

The second paper, tentatively *Capability-Aware Hierarchical Navigation with
Experience Memory under Localization and Dynamics Uncertainty*, tests whether
frozen-Local capability distributions and edge-level experience memory reduce
infeasible subgoal selection, repeated dead-end entry and revisit distance. A
separate hypothesis tests whether propagating pose covariance into candidate
risk produces safer degradation under GPS-denied localization drift. Do not
use one large Global table to substitute for proving the Local hypotheses first.

## Required Local comparisons

| Question | Minimum comparison |
|---|---|
| trajectory action vs direct control | TQC/SAC `(v, steering)`, `[kappa,v]`, `[kappa,v,L]` |
| learned horizon | fixed short L, fixed long L, adaptive L |
| path representation | constant-curvature and equal-capacity spline/Bezier action |
| risk critic | no risk, supervised risk only, actor risk penalty |
| counterfactual contribution | risk penalty only, random candidate augmentation, structured candidate supervision/weighting |
| learned vs exact risk | exact rollout, learned critic, hybrid |
| risk representation | scalar risk, multi-task factors, factor ensemble |
| uncertainty use | mean only, mean + uncertainty penalty, calibrated conservative bound, abstention gate |
| dynamics model | nominal physics, single residual, residual ensemble |
| counterfactual target | current margin weighting, direct target without progress constraint, progress-constrained target, progress + uncertainty gate |
| safety attribution | raw policy, guard-only, policy+guard |
| classical navigation | Nav2 MPPI or Hybrid-A* family; document unresolved baseline failures |

TQC is the implementation base, not the claimed novelty. The proposed method
should retain its research meaning if another off-policy continuous-control
algorithm replaces TQC.

## Required Global comparisons

Use the same immutable test manifest for all rows:

1. `G0`: frozen Local-only final-goal pursuit.
2. classical optimistic/frontier exploration with A*/D* Lite or Hybrid-A*.
3. `G1`: online partial-map Global RL.
4. `G2`: G1 + visited channel.
5. `G3`: G2 + topology/dead-end memory.
6. `G4`: G3 + Local success probability; compare geometry-only, oracle and learned feasibility.
7. `G5`: G4 + Local expected risk; compare no risk, oracle rollout and learned risk.
8. `G6`: G5 + calibrated Local risk uncertainty.
9. `G7`: G6 + localization-aware candidate risk.
10. `G8`: full capability distribution + edge experience posterior.

The B/C distinction is specifically the visited channel. Every learned B–G
row needs its own architecture fingerprint and independently trained
checkpoint; a heuristic or another row's checkpoint is never a substitute.
The same rule applies to G0–G8; they are the target research ladder and do not
rename the historical Phase-5 A–G labels.

## Risk calibration protocol

Navigation success alone does not validate a risk critic. Report:

- AUROC and AUPRC; accuracy is insufficient for rare collision events.
- Brier score, Expected Calibration Error and a reliability diagram.
- false-negative rate at every deployed risk threshold.
- observed collision/high-risk frequency in predicted-risk bins.
- error and latency relative to exact online rollout.
- calibration under localization noise, actuator delay and dynamics mismatch.
- ensemble spread versus absolute prediction error and selective-risk/coverage
  curves when high-uncertainty predictions are rejected.
- ID versus every one-axis OOD condition, followed by combined-stress OOD.

The unit of analysis and label horizon must be stated. Calibration metrics must
come from held-out test scenarios, never replay-buffer training rows. Fit any
temperature/isotonic/conformal calibrator only on a separate validation split
and evaluate it once on the locked test split. Report aleatoric prediction targets
and epistemic ensemble disagreement separately; do not name an unvalidated
standard deviation a calibrated confidence bound.

## Dynamics-model validation protocol

Residual dynamics is validated independently of navigation return:

- one-step MAE/RMSE for velocity, yaw rate and steering response;
- open-loop multi-step pose/heading and endpoint error versus horizon;
- trajectory clearance/TTC/stopping-margin error induced by the model;
- nominal physics versus one residual model versus residual ensemble;
- ID and test-only OOD dynamics conditions;
- ensemble spread versus actual rollout error and inference latency.

Split temporally contiguous runs by complete trial/rosbag, not randomly by
individual adjacent samples. A real-Hunter test trial must never be used to fit
the residual model or its uncertainty calibrator. Preserve the nominal-only
fallback and report when it is used.

## Counterfactual policy-improvement protocol

For actor action $a=\pi_\theta(s)$, the selected candidate must minimize the
conservative risk $R^+=\mu_R+\beta\sigma_R$ subject to feasibility, progress
retention $P(a')\ge\rho P(a)$ and uncertainty $U(a')\le\epsilon_u$. Store the
candidate set, each constraint outcome, selected candidate, risk improvement,
progress ratio, uncertainty, activation weight and abstention reason.

Compare identical candidate budgets and rollout compute. Report how often no
candidate is feasible, how often only stop is feasible, target activation
rate, erroneous-target rate under privileged evaluation, progress loss and
extra latency. “Safer” is not established if the method simply selects stop
more often.

## Guard attribution protocol

Record raw policy action, nominal decoded command, guarded command and actually
published command separately. At minimum report guard intervention rate,
policy-proposed high-risk action rate, intervention-induced stop/failure rate,
and recovery success after emergency stop. A lower guarded collision rate is
not evidence of a safer learned policy unless raw risk or intervention rate
also improves.

Use the same checkpoint/manifest to report four named conditions:

1. Raw Actor;
2. Raw Actor + Counterfactual learning;
3. Actor + Guard;
4. Counterfactual Actor + Guard.

If disabling the guard is unsafe on hardware, run raw conditions only in
simulation/replay and use proposal-level metrics on hardware. The guard remains
mandatory on the physical robot.

## Statistical design

- Use at least five independent training seeds per learned method for paper
  results; a one-seed engineering smoke is not a research result.
- Evaluate every method on the same paired scenario IDs and immutable content
  hashes.
- Report 95% confidence intervals for success and collision rates.
- Use paired bootstrap intervals or a justified non-parametric paired test for
  SPL, time, path length, clearance and revisit metrics.
- Report mean, median, spread and a worst-tail statistic where safety matters.
- Classify failure as collision, timeout, stuck/no-progress, infeasible goal,
  localization loss, planner loop or infrastructure failure.
- Add a held-out layout family or real floor plan beyond procedural seed splits.
- Include 50–100 m routes, consecutive dead ends/loops and nonholonomic traps
  before making a strong long-horizon claim.

## Hierarchical checkpoint and formal-result policy

Graceful partial loading is useful for development, but it is forbidden for a
formal hierarchical result. Formal artifacts require strict component loading,
resolved config/schema hashes, immutable checkpoint SHA, Local generation and
training-contract identity, replay/RNG schema, package provenance and a cleanly
identified scenario manifest. Missing labels or partially completed scenarios
make an A–G suite incomplete, not formal.

For feasibility/capability rows E/F/G, evaluator failure is missing evidence,
not a low-risk observation. The current artifact records fallback reasons, but
the current candidate tensor has no explicit validity channel and zero-fills
the two policy-conditioned fields. Until that schema is versioned, each formal
run must require zero raw action/risk fallback counts and report all reasons
per label and condition. The emitted legacy `fallback_rate` is not an
acceptance statistic because it divides some per-candidate counts by a
per-decision query count and can exceed one. Any nonzero fallback makes the
formal result incomplete; retain affected-decision sensitivity analysis only
as diagnosis. Silently treating the zero-fill as valid low risk invalidates
the claim.

The audited code-level formal gates are closed; see `CURRENT_STATUS.md` for the
latest evidence. Formal training/results still require a fresh short
save/resume/live smoke on the corrected release and then actual multi-seed
execution under this protocol.
