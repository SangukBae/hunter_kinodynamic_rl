# TRACTOR-TQC Research Protocol

Status: **FROZEN as `tractor_protocol_v1` before formal runs**
Frozen payload SHA-256: `fe2c1cfbd175c12c333173a7c9ccd25edf885fbfc18e519640c4d7e597ba9ee1`
Scope: Local kinodynamic control first; Global is a dependent follow-up

## 1. Objective and non-claims

Research question:

> 동일한 Hunter observation/action/guard와 학습 budget에서, candidate swept tube와
> action-independent future occupancy의 명시적 상호작용이 flat/recurrent/world-model
> baselines보다 candidate ranking, calibrated risk와 navigation을 개선하는가?

The paper does not claim to solve SLAM, prove formal safety, invent TQC/residual dynamics/attention,
or validate Global navigation before a Local model is promoted. Reward shaping, hyperparameter tuning
and a guard are controls—not the central novelty.

## 2. Prerequisites

Formal TRACTOR runs cannot start until [CURRENT_STATUS.md](CURRENT_STATUS.md)의 P0-01…P0-07,
dataset validation, checkpoint interruption/resume tests and baseline contract freeze are complete.
The current 0/30 Stage-2 matrix is run in a separate immutable root before being cited as evidence.

## 3. Falsifiable hypotheses

| ID | Hypothesis | Primary evidence | Failure condition |
|---|---|---|---|
| H1 | factorized ego-warped belief improves held-out static/dynamic future quality | occupancy IoU/F1, flow EPE, calibration | matched recurrent vector/BEV baseline not worse within CI |
| H2 | explicit tube×occupancy product improves candidate ordering | ranking regret, NDCG, unsafe top-1 rate | implicit concat/cross-attention matches it |
| H3 | cause-time hazard improves risk usefulness | Brier/NLL/ECE, time/cause accuracy, selective risk | scalar/endpoint risk matches calibration and ranking |
| H4 | structural gains transfer to navigation | success, collision, SPL, time/length, interventions | no benefit across locked ID/OOD matrix |
| H5 | gains survive equal-compute and target deployment | parameter/FLOP matched results, p99 latency | benefit disappears when budget/deadline is matched |

The executable thresholds are frozen in `config/tractor/protocol.yaml`. A missing metric, denominator,
target-hardware flag or confidence bound fails closed; it is never interpreted as a pass.

### Frozen primary promotion gates

| Metric | Absolute gate | Paired A7−B1 gate over seed-level 95% CI |
|---|---|---|
| success rate | lower CI ≥ `0.80` | lower CI ≥ `+0.05` |
| collision rate | upper CI ≤ `0.05` | upper CI ≤ `+0.01` non-inferiority |
| successful time-to-goal | — | upper CI of relative change ≤ `+10%` |
| realized minimum-clearance mean | lower CI ≥ `0.15 m` | lower CI ≥ `−0.02 m` non-inferiority |
| runtime | target hardware, at least `10,000` decisions | p99 ≤ `100 ms`, miss rate ≤ `1%` |

All navigation gates are joint gates. The independent training seed—not an episode—is the replication
unit. These are preregistered research promotion criteria, not a guarantee of ICRA/IROS acceptance.

## 4. Baselines

All core baselines share robot model, sensor history, reward/termination, action decoder, control rate,
guard, train transitions, seeds and locked scenarios unless explicitly labelled a contract-difference
reference.

| ID | Model | Purpose |
|---|---|---|
| B0 | current direct-control TQC reference | historical contract-difference anchor |
| B1 | current 328D trajectory-action TQC | same-action minimum baseline |
| B2 | parameter-matched flat MLP TQC | capacity control |
| B3 | GRU/temporal TQC | sequence-memory control |
| B4 | BEV belief + implicit concatenation | representation without explicit operator |
| B5 | BEV + generic cross-attention | attention/control-operator comparison |
| B6 | compact latent world-model RL | model-based alternative with equal data/compute report |
| B7 | risk-sensitive/CVaR TQC | tail-return comparison |
| B8 | scalar/endpoint risk critic | risk-representation baseline |
| A7 | TRACTOR core | primary method |
| A8 | TRACTOR + bounded vehicle residual | physics-support extension |
| A9 | TRACTOR ensemble/UCB | uncertainty extension |
| B12 | Nav2/MPPI-style classical controller | controller-specific external reference |

B0/B12 are not direct architecture ablations because their action/controller contracts differ.
Publication tables label those differences.

## 5. Training protocol

### Common fairness

- at least five registered seeds for every headline method;
- identical development transitions and locked-test episodes where contract permits;
- both fixed-update and fixed-environment-step budgets;
- parameter count, FLOPs, wall-clock, GPU-hours and peak memory reported;
- tuning budget and search space logged per method;
- failure/timeout runs retained and excluded only by preregistered rules.

### TRACTOR stages

| Stage | Train | Frozen/controlled output | Exit gate |
|---|---|---|---|
| 0 contract repair | P0 code/tests | current baseline behavior | full regression + attestations |
| 1 baseline freeze | B1–B8 training protocol | scenario/split/reward/action | complete baseline artifacts |
| 2 data plane | episode store/index/sampler | schema and split | leakage/durability tests |
| 3 representation | belief, scene forecast, vehicle residual if enabled | actor/risk headline claims | held-out identifiability |
| 4 risk/ranking | interaction/aggregator + active risk heads | return critics/actor | calibration/ranking improvement |
| 5 joint RL | value+risk then actor/entropy | fixed protocol | stable learning + deterministic resume |
| 6 formal eval | no training | locked checkpoints/calibrator | complete seed×scenario matrix |

Stage 3/4/5 are separate experiment lineages linked by explicit warm-start maps. Stage 5 TQC uses
real executed transitions only. Auxiliary losses may update the registered value-path modules; actor
updates detach belief but preserve gradients through action-conditioned scoring.

Core phase truth table:

| Phase/transaction | Exact objective | Owner/eligible modules | Ineligible updates |
|---|---|---|---|
| Stage 3 representation | `w_occ L_occupancy + w_flow L_flow + w_resp L_vehicle_response` | one `optimizer_value_path` step; belief/future-scene/vehicle-response and enabled residual keys selected in the phase manifest | actor, return critics, tube/aggregator, risk heads and entropy remain frozen |
| Stage 4 feature side | `L_R[risk_contract_id]` with risk-head weights frozen | `optimizer_value_path` steps only tube interaction+temporal aggregator | encoders/scene/residual/return critics, actor and entropy remain frozen |
| Stage 4 head side | `L_R[risk_contract_id]` on stop-gradient interaction features | `optimizer_risk_heads` only | every value-path module, actor and entropy remain frozen |
| Stage 5 value side | `L_TQC +` valid representation auxiliaries `+ w_rfeat L_R[risk_contract_id]` with risk-head weights frozen | one `optimizer_value_path` step over the complete online value path | risk-head, actor and entropy weights do not step |
| Stage 5 risk side | `L_R[risk_contract_id]` on stop-gradient value features | one `optimizer_risk_heads` step | value-path, actor and entropy weights do not step |
| Stage 5 actor | `alpha_ent log pi - lower-tail value +` registered risk/smoothness terms | one `optimizer_actor` step; value/risk weights frozen, action gradient preserved | representation, value and risk weights do not step |
| Stage 5 entropy | registered temperature objective | `optimizer_entropy` only | all network weights remain frozen |

Stage-4 feature/head sides and Stage-5 value/risk sides each form an atomic transaction; both prechecks
pass and both optimizer steps
commit before EMA. A batch with no valid risk label skips that whole transaction without consuming its
target-action RNG. A missing auxiliary label sets only that masked auxiliary term to zero and records a
zero valid count; it never invents a target. The core sampler is uniform over valid windows, so the
masked risk estimators are unweighted sample means. Event-balanced training requires a separately
registered sampling/loss contract and is not part of A7 core.

`L_R` dispatches exactly by model-spec row: R0 masked MSE, R1 masked BCE, R2 survival NLL, R3 masked
class CE, R4 competing-risk NLL, and R5/R6 that NLL plus their enabled masked severity-quantile losses.

Exact weights, masks, reductions, schedules, clipping and update ratios belong to the frozen training
fingerprint. No weight is changed after locked-test inspection under the same protocol version.

## 6. Evaluation matrix

The frozen `tractor_scenario_plan_v1` contains 616 unique geometries: 352 development, 88 calibration
and 176 locked-test instances across six static and five dynamic families. Seed intervals and canonical
geometry hashes are disjoint across splits. The plan SHA-256 is
`7ae919676a408ab08bdbc686a646a634a3789856f62a1d15ebaafd91fa2a8626`.

### Scenario axes

- topology: open, corridor, corner, choke point, clutter, dead end;
- obstacle: static density, crossing, head-on, overtaking, occlusion, mixed motion;
- vehicle: mass/friction/lag/steering/braking perturbations;
- sensing: dropout, range noise, latency, stale frames, partial observability;
- localization: drift, jump, covariance inflation, invalid periods;
- goal: distance/bearing including rear/full-circle observability cases.

ID varies registered ranges seen in development. OOD holds out layouts, trajectories and combinations;
each OOD axis is reported separately before any pooled score.

### Metrics

| Family | Metrics |
|---|---|
| navigation | success, collision/event rate by cause, SPL, path/time, min clearance, progress |
| control | curvature/speed/steering smoothness, saturation, command tracking |
| proposal/ranking | top-1 unsafe rate, ranking regret, NDCG, candidate coverage |
| prediction | occupancy IoU/F1/NLL, dynamic flow EPE, trajectory ADE/FDE, tube coverage |
| risk | NLL, Brier, ECE/reliability, cause/time accuracy, calibration by horizon |
| uncertainty | risk-coverage/selective curves, OOD AUROC where labels support it |
| operations | guard intervention, fallback cause, p50/p95/p99 latency, deadline miss, memory |

Report numerator/denominator and event counts, not percentages alone. A run with insufficient event
support cannot establish calibration or safety improvement.

## 7. Ablations

Run core isolation before optional extensions:

| Axis | Required comparison | Question |
|---|---|---|
| representation | raw vector vs recurrent vector vs factorized BEV | is spatial factorization useful? |
| ego motion | concatenate vs warp; valid vs mask-removed | is alignment/validity causal? |
| interaction | implicit concat, generic attention, explicit product, full operator | what produces ranking gain? |
| tube | centreline vs deterministic footprint vs probabilistic tube | does swept uncertainty matter? |
| future | current-only vs action-independent future | is prediction actually used? |
| risk | scalar, endpoint, time hazard, cause-time, +severity | which structure improves calibration/action? |
| physics | nominal vs residual vs ensemble | support value and model-bias cost |
| learning | auxiliary-only vs actor-through-score; real vs imagined Bellman reference | gradient-path contribution |
| compute | parameter/FLOP/candidate/horizon matched | rule out capacity/planning advantage |

Every ablation changes one registered axis. If a change forces another contract change, record it as a
compound variant and do not use it for a clean causal claim.

## 8. Statistics

- Unit of replication is the independent training seed, not an episode.
- Report per-seed results, mean/median, 95% confidence intervals and paired differences on shared
  scenario seeds.
- Use bootstrap intervals over seeds/scenario groups where distributional assumptions are weak.
- Multiple headline comparisons use a declared correction or a small preregistered primary set.
- Safety metrics include event counts and confidence bounds; zero observed collisions is not zero risk.
- Practical effect thresholds are decided before significance testing.

## 9. Evidence ladder and claim gate

```text
unit/property tests
< offline held-out prediction/ranking
< deterministic simulation smoke
< formal multi-seed locked simulation
< target-device timing/HIL
< contained real static trials
< controlled dynamic real trials
< GPS-denied integrated mission
```

Higher levels do not automatically validate unrelated lower assumptions. A paper sentence is allowed
only when [evidence/README.md](evidence/README.md)의 claim ledger points to immutable artifacts with
complete provenance.

## 10. Invalidating conditions

Results are invalid for the frozen protocol if any of the following occurs:

- split leakage or locked-test-guided tuning;
- mixed action/footprint/reward/terminal/data semantics under one experiment ID;
- missing seed/run, overwritten output or unverifiable source/container identity;
- unequal train data/budget presented as a direct architecture comparison;
- privileged simulator input reaches inference;
- candidate or ensemble members are dropped/renormalized after failures;
- calibration is fit on headline test data;
- timeout/deadline failures are omitted from aggregates.

Protocol-invalid runs remain archived with `INVALID`; they are never silently repaired or resumed into
the same output root.

## 11. Promotion and stop criteria

TRACTOR is promoted to deployment evaluation only if it passes preregistered navigation,
calibration/OOD and p99 deadline gates without regression in guard/fallback behavior. Redesign is
triggered when the explicit operator is not better than implicit equal-compute baselines, future belief
is not identifiable, risk is materially miscalibrated, or gains require imagined Bellman targets.

Real motion stops immediately for stale/invalid localization or sensors, manifest mismatch, command
tracking outside the envelope, repeated deadline miss, guard escalation, geofence breach or operator
intervention. Details are in [SIM2REAL.md](SIM2REAL.md).

## 12. Publication-ready contribution gate

Only after complete evidence may the paper claim:

1. a new explicit tube-occupancy interaction representation;
2. its isolated contribution over equal-budget implicit alternatives;
3. calibrated cause-time risk and candidate-ranking improvement;
4. navigation/runtime transfer within the exact tested simulation or real ODD.

Each contribution needs a primary table/figure, ablation, uncertainty interval, failure analysis and
artifact link. Until then all manuscript language uses future/proposed tense.
