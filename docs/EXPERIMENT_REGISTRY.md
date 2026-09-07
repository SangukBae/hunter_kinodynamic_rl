# Experiment and Decision Registry

This is the human-readable index. Machine-readable manifests and raw artifacts are authoritative for
individual runs. Rows are append-only; corrections create a new row linked to the old one.

## 1. Current snapshot

| Campaign | Planned matrix | Training complete | Benchmark complete | Status |
|---|---:|---:|---:|---|
| legacy Stage-2 L0–L5 | 6×5 = 30 | **0/30** | **0/30** | NOT STARTED in inspected runtime |
| same-contract paper comparison | B1–B8/A7–A9 × 5 seeds; 176 locked scenarios | **0/55** | **0/55 runs; 0/9,680 episodes** | FORMAL-BLOCKED / UNTRAINED |
| comparison risk calibration | A7/A8/A9/B8 × 5 seeds; 88 calibration scenarios | — | **0/20** | IMPLEMENTED / UNFITTED |
| formal realized-track corpus | 352 development + 88 calibration + 176 locked | — | **0/616** | CORE COLLECTOR PRESENT / FULL SCHEMA BLOCKED |
| TRACTOR simulation env v2 | 6 fixed suites × 8 scenarios | 0 | 0 | CONFIGURED / NO ROLLOUT EVIDENCE |
| formal Global | blocked by Local promotion | 0 | 0 | BLOCKED |
| Hunter real navigation | not approved | 0 | 0 | NO EVIDENCE |

The prior `2,138 passed` regression report is evidence metadata, not a training row.

## 2. Experiment identity

Canonical ID:

```text
<campaign>__<method>__<protocol>__seed-<n>__<timestamp-utc>
```

Before execution, each record fixes:

```text
experiment_id, parent/warm-start lineage
research question and hypothesis
method/baseline/ablation IDs
protocol, scenario, split and seed manifests
model/data/training/robot fingerprints
source commit, dirty digest, container and hardware
train/evaluation budget and preregistered thresholds
output root, owner and expected artifacts
```

## 3. Run registry

| Experiment ID | Method | Seed | Phase | Train | Eval | Evidence | Artifact root |
|---|---|---:|---|---|---|---|---|
| _none_ | — | — | — | NOT STARTED | NOT STARTED | none | — |

Allowed lifecycle:

```text
REGISTERED → PREFLIGHT_PASSED → RUNNING → TRAIN_COMPLETE
→ BENCHMARK_COMPLETE → AGGREGATED → ACCEPTED/REJECTED
```

`FAILED`, `INTERRUPTED` and `INVALID` are terminal records unless an exact resume is permitted by the
checkpoint contract. A retry with changed semantics gets a new experiment ID.

## 4. Required completion evidence

Training completion requires final atomic checkpoint, manifest/root hash, counters, logs, resolved
config/attestations, deterministic load probe and no missing registered seed. Benchmark completion
requires every locked scenario episode, raw per-episode metrics, failure attribution, timing and a
complete matrix check. Aggregation requires per-seed values, event counts, uncertainty intervals and
documented exclusions.

Missing metrics remain `missing`; they are never written as zero. Crashed or unsafe runs remain in
denominators according to the frozen rule.

## 5. Baseline and ablation registry

| ID | Frozen definition | Status |
|---|---|---|
| B0, B12 | contract-difference references; see frozen protocol | FROZEN / UNRUN |
| B1 | current direct 328D TQC network, `[kappa,v_ref,L]` data/action contract | IMPLEMENTED / UNTRAINED |
| B2–B8 | registered comparison representations/objectives on the shared data contract | IMPLEMENTED / UNTRAINED |
| A7 core | nominal TRACTOR, `Q_m=0,Q_r=1` | IMPLEMENTED / UNTRAINED |
| A8 residual | one bounded vehicle residual | IMPLEMENTED / UNTRAINED |
| A9 ensemble | canonical `Q_m=3,Q_r=3` | IMPLEMENTED / UNTRAINED |

Each ablation row records exactly one changed axis, expected mechanism, compute difference and isolation
check. Compound changes are labelled and cannot support a single-factor causal claim.

## 6. Decision log

| ID | Decision | Rationale | Revisit when |
|---|---|---|---|
| D-01 | retain `[kappa,v_ref,L]` | action parity and current executor/guard reuse | separate physical-action study |
| D-02 | factorized ego-warped BEV | exposes alignment, occupancy and failure modes | matched evidence shows no benefit |
| D-03 | explicit read-only-base tube interaction | testable causal/permutation contract | operator fails isolation |
| D-04 | primary Bellman updates use executed transitions | isolate imagined-model bias | follow-up after calibrated model evidence |
| D-05 | single Gaussian actor in core | isolate interaction contribution | after core acceptance |
| D-06 | residual dynamics is supporting | extensive prior art and confounding risk | only with distinct new mechanism |
| D-07 | fixed guard remains | learned risk is not a safety proof | no planned removal |
| D-08 | Local promotion precedes Global training | prevent moving capability contract | after immutable Local promotion |
| D-09 | seed-level CI is the headline replication rule | episodes from one trained policy are not independent replicas | new protocol version only |
| D-10 | split identity is canonical geometry SHA-256 | renamed copies must still trigger leakage rejection | new data schema only |
| D-11 | preserve v1 and add opt-in `tractor_env_v2` | avoid invalidating 616-scenario frozen evidence while increasing motion/geometry diversity | only with a new environment contract |
| D-12 | curriculum index is trainer-delivered and fail-fast | exact resume must restore difficulty stage; reset count is not a reliable proxy | service/interface revision only |

New decisions use `D-<number>`, list alternatives, affected fingerprints and migration/test impact in a
new section or linked artifact. Do not create another standalone decision document.

## 7. Result record

For each aggregated experiment append or link a record containing:

```text
question/hypothesis and preregistered gate
all seed/run IDs and exclusions
fingerprints and source/container/hardware identity
train budget and actual resource use
ID/OOD navigation, prediction, ranking, risk/calibration and timing metrics
effect sizes, confidence intervals and event counts
failure cases and protocol deviations
claim status: unsupported / exploratory / accepted within stated scope
artifact manifest/root hash
```

Use [templates/EXPERIMENT_CARD.md](templates/EXPERIMENT_CARD.md) for a new run and
[templates/MODEL_CARD.md](templates/MODEL_CARD.md) for promotion.

## 8. Completion rule

Documentation or partial implementation completion is not research completion. Formal collection and
training remain blocked while `formal_research_implementation_readiness()` reports any gap. The project may be marked
complete only when every claim has a valid artifact, the registered matrix is complete, target timing
and calibration gates pass, and any reported real claim has approved trial cards. Until then the
registry remains explicit about zero/missing evidence.
