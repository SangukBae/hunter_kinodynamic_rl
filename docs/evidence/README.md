# Evidence and Claim Ledger

Snapshot: **2026-09-07 KST**

This file consolidates the current baseline audit, source anchors, regression/runtime status, claim
ledger and legacy manifest. It is provenance metadata, not a performance result.

## 1. Audited checkout

| Field | Value |
|---|---|
| repository | `hunter_kinodynamic_rl` package working tree |
| branch | `phase1-hierarchical-navigation` |
| commit at audit | `42e192f2f038c8cc9f214ff15ed6f8e57c00c5f6` |
| worktree | dirty; tracked and untracked research changes present |
| audit date | 2026-09-06 KST |

This snapshot must not be described as a clean tagged release. Source line numbers may drift; symbol
and file references below should be resolved against the recorded commit/dirty digest before citation.

## 2. Source anchors

| Contract | Primary anchors | Audited interpretation |
|---|---|---|
| scan/history | `sensing/scan_processor.py`, `sensing/temporal_stack.py` | 4×80 current-first front scan stack |
| observation | `env/observation/observation_builder.py` | 320 scan + 8 tail = 328D |
| TQC | `rl/networks/tqc.py`, `rl/algorithms/kinodynamic_tqc/agent.py` | 2×25 quantiles; truncated target |
| risk | `rl/networks/risk_critic.py`, agent/environment label path | scalar supervised risk, not cause-time calibrated hazard |
| trajectory/action | `hunter_kinodynamic_rl/trajectory/*` | normalized 3D → `[kappa,v_ref,L]` → Pure-Pursuit command |
| safety | `hunter_kinodynamic_rl/env/safety/*` and publisher call sites | fixed final command checks remain required |
| replay/checkpoint | `rl/replay/*`, agent/training checkpoint code | transition replay; TRACTOR sequence schema absent |
| localization/map | `hunter_kinodynamic_rl/navigation/localization/*`, `navigation/mapping/*` | interfaces/infrastructure, not a complete GPS-denied result |
| hierarchy | `hunter_kinodynamic_rl/navigation/hierarchy/*` and associated runners | Global infrastructure; no accepted-Local formal campaign |
| TRACTOR model | `rl/networks/tractor/*` | untrained A7/A8/A9 implementation with typed causal scoring path |
| TRACTOR data/checkpoint | `rl/replay/sequence_*`, `rl/checkpointing/tractor.py` | synthetic contract evidence only; no collected dataset/promoted checkpoint |
| TRACTOR evaluation | `evaluation/tractor_*` | metric/matrix/latency tooling; no formal result rows |

TRACTOR-specific packages are present in the current dirty working tree. They have not produced a
trained, calibrated, benchmarked or promoted artifact.

## 3. Regression and runtime evidence

| Evidence | Observed | Allowed interpretation |
|---|---|---|
| prior Docker regression report | `2,138 passed` | broad code-health snapshot; exact run provenance incomplete |
| current combined regression | `2,186 passed, 1 skipped` in `126.73 s` | ROS mounted into `drl_path:final`, CPU-only code correctness snapshot |
| Local bounded/Gazebo historical runs | reset/step/save/resume and command-tracking records | smoke/wiring and simulator behavior only |
| Stage-2 L0–L5 artifacts | no completed training/benchmark directories in inspected runtime | training 0/30, benchmark 0/30 |
| TRACTOR artifacts | none | implementation/training/benchmark/timing unproven |
| current TRACTOR contract tests | synthetic unit/property/checkpoint checks pass | implementation correctness within covered fixtures only |
| default A7 forward/timing smoke | untrained CPU synthetic process run | executable path only; not target-device real-time evidence |
| Hunter real navigation artifacts | none | real and GPS-denied claims unavailable |

To promote the regression number beyond this record, reconstruct the exact commit+dirty digest,
container digest, command, complete log and exit code. Test count alone is not navigation evidence.

## 4. Claim ledger

| Candidate statement | Current status | Required evidence |
|---|---|---|
| current package contains a TQC kinodynamic Local baseline | allowed | source anchors and contract tests |
| hierarchical navigation infrastructure exists | allowed with qualification | source anchors; do not say formally validated |
| prior audit reported 2,138 passing tests | allowed with provenance caveat | this snapshot plus recovered raw log for stronger wording |
| TRACTOR-TQC architecture and untrained implementation exist | allowed | model specification + current contract tests |
| TRACTOR is trained or performance-improving | prohibited | valid training + complete comparative artifacts |
| TRACTOR improves navigation/ranking | prohibited | complete preregistered multi-seed comparisons |
| risk is calibrated or safer | prohibited | held-out calibration, event counts, CI and risk-coverage evidence |
| target deployment is real-time | prohibited | named-hardware p50/p95/p99 and miss/fallback logs |
| Global navigation is validated | prohibited | accepted Local + complete Global formal protocol |
| Hunter GPS-denied navigation works | prohibited | approved real trials with separate localization evidence |
| Gazebo tracking proves real fidelity | prohibited | real system-ID and controlled transfer evidence |

Strongest currently supportable summary:

> The dirty working tree contains a TQC-based Local baseline, hierarchical infrastructure and an
> untrained TRACTOR-TQC code path. Formal training, comparative benchmarking, calibration,
> target-hardware timing, Global evaluation and real-robot validation remain open.

## 5. Historical evidence policy

Pre-TRACTOR records covered defect fixes, hierarchy phases, Stage-1 freeze, Stage-2 setup and Hunter
Gazebo/controller checks. They may justify regression requirements or identify old artifacts. They do
not automatically establish current performance because code/config/container/robot contracts may
have changed.

Use [../verification/README.md](../verification/README.md) as the consolidated index. For an exact old
number or statement, recover the original file from Git history and verify its referenced artifact,
commit and execution environment. If that evidence cannot be recovered, label the statement
`historical, provenance incomplete`.

## 6. Evidence acceptance checklist

Before linking evidence to a claim, verify:

- immutable experiment/trial ID and complete artifact manifest;
- source commit, dirty digest, container/hardware and resolved config;
- model/data/training/robot fingerprints and checkpoint lineage;
- full registered seed×scenario matrix or explicit missing rows;
- raw episode/run metrics, denominators, event counts and failure attribution;
- calibration/test separation and statistical uncertainty;
- evidence label matches its level: regression, smoke, simulation or real.

No documentation statement may upgrade evidence beyond the lowest verified item in this chain.
