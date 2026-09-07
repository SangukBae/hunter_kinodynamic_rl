# TRACTOR-TQC Implementation and Operations Plan

Status: **core, matched baselines and formal comparison pipeline landed; data/training/evidence open**

This document is the single code-change, integration and runbook map. It does not report completion;
actual state belongs to [CURRENT_STATUS.md](CURRENT_STATUS.md).

## 1. Current source anchors

| Responsibility | Current path |
|---|---|
| scan processing / stack | `hunter_kinodynamic_rl/sensing/scan_processor.py`, `hunter_kinodynamic_rl/sensing/temporal_stack.py` |
| observation | `hunter_kinodynamic_rl/env/observation/observation_builder.py` |
| action/trajectory/controller | `hunter_kinodynamic_rl/trajectory/*` |
| safety guard | `hunter_kinodynamic_rl/env/safety/*` |
| TQC / risk | `hunter_kinodynamic_rl/rl/networks/tqc.py`, `hunter_kinodynamic_rl/rl/networks/risk_critic.py`, `hunter_kinodynamic_rl/rl/algorithms/kinodynamic_tqc/agent.py` |
| replay/checkpoint | `hunter_kinodynamic_rl/rl/replay/*`, `hunter_kinodynamic_rl/rl/checkpointing/*` |
| localization/map/hierarchy | `hunter_kinodynamic_rl/navigation/localization/*`, `hunter_kinodynamic_rl/navigation/mapping/*`, `hunter_kinodynamic_rl/navigation/hierarchy/*` |
| training/evaluation | `hunter_kinodynamic_rl/training/*`, `hunter_kinodynamic_rl/evaluation/*` |
| matched baseline families | `hunter_kinodynamic_rl/rl/algorithms/comparison_baselines/*`, `config/tractor/baseline_models.yaml` |
| robot/profile config | `config/robot/*`, `config/profiles/*` |

The current `rl/networks/tqc.py` remains code-identical to its `drl_agent` source except for the
documentation link redirected to the consolidated compatibility contract. This comment-only adaptation
is guarded by `tests/test_source_map.py`; it is not an algorithmic baseline change.

Use `find`, `git grep` and tests to resolve current symbols before editing; paths above are ownership
anchors, not a substitute for inspecting the working tree.

## 2. Proposed ownership

```text
hunter_kinodynamic_rl/
  rl/networks/tractor/
    contracts.py, input_adapter.py, ray_lift.py, se2_warp.py
    belief_encoder.py, scene_forecast.py
    nominal_rollout_adapter.py, residual_dynamics.py
    tube_rasterizer.py, tube_interaction.py, temporal_aggregator.py
    actor.py, return_critic.py, hazard_critic.py, selector.py, calibration.py
  rl/algorithms/tractor_tqc/
    agent.py, losses.py, optimizer_groups.py, target_update.py
  rl/replay/
    sequence_schema.py, episode_store.py, sequence_index.py, sequence_buffer.py
  training/
    train_tractor_tqc.py, tractor_episode_collector.py, tractor_sequence_training.py
    tractor_preflight.py, tractor_dataset_validation.py
  evaluation/
    tractor_*_metrics.py, tractor_latency.py, tractor_bundle_export.py
config/tractor/{model,data,training,inference}.yaml
config/profiles/tractor_*.yaml
```

New modules consume typed dataclasses from `contracts.py`; no dict with optional semantic fields may
cross the `encode/propose/score` boundary.

As of 2026-09-07, the listed TRACTOR network, B1–B8 matched baselines, formal realized-track collector,
reset-prefix replay adapter, ordered learning runners, checkpoint, calibration, common locked evaluator,
campaign orchestrator, preflight and runtime boundary exist.
Work-package completion still follows the exit gates
below; source presence must not be interpreted as trained or paper-ready evidence.

## 3. Work packages

### P0 — repair current contracts

1. centralize footprint and assert Local/map/Global equality;
2. extend fingerprints over trajectory, observation/replay, model family and physical attestation;
3. separate static baseline and dynamic-paper scenario manifests;
4. generate velocity/risk labels from realized timestamp-aligned motion;
5. split `terminated` and `truncated` throughout env, replay and target;
6. make export/promotion/live loading atomic and fail closed;
7. freeze front/rear observability behavior.

Exit: targeted tests, full regression, resolved config/robot attestation and current baseline smoke.

### P1 — freeze baselines

Immutable B1–B8 profiles, seed schedule, train budget and locked scenario manifest are implemented.
Weights/results remain absent. The existing L0–L5 0/30 campaign stays in its own output root and is
reported separately because it does not share the sequence-training contract.

### P2 — sequence data plane

Implement versioned episode writer, checksummed chunks, index, sampler, burn-in and migration report.
Add scripted label fixtures and a dataset validator. Do not begin long training while any leakage,
timestamp or mask error remains.

### P3 — belief and dynamics

Implement ray lift, cumulative SE(2) warp, scene ConvGRU, four-class occupancy, flow, ego/health and
vehicle-response encoders. Verify offline identifiability before connecting RL. Adapt the current
nominal rollout only after numerical parity tests; add bounded residual as a separate variant.

### P4 — interaction and risk

Implement tube rasterization, candidate-private sparse gathers, explicit products, temporal aggregator,
two quantile critics and registered risk heads. Property tests for action-invariance, permutation and
single-vs-batched equivalence are release blockers.

### P5 — RL integration

Implement disjoint optimizer ownership, complete target EMA, deterministic target-action RNG,
real-transition Bellman target and actor-through-score gradient path. Add exact interruption/resume
tests before multi-hour jobs.

### P6 — formal evaluation

Run development smoke, then complete baselines, ablations and TRACTOR seeds under the frozen protocol.
Aggregate only complete verified manifests. Fit calibration on its dedicated split and evaluate ID/OOD
once.

### P7 — runtime/export

Batch all `K` candidates, reuse one encoded belief/base future, avoid dense candidate copies, profile
target hardware and export only promoted checkpoint+calibrator bundles. Deadline/fallback reasons are
telemetry fields.

### P8 — sim-to-real

Follow [SIM2REAL.md](SIM2REAL.md) from system-ID through replay, HIL and contained trials. Do not add
Global training until the Local bundle is frozen and promoted.

## 4. Integration invariants

1. observation tail and action decoder remain backward-compatible only inside their declared family;
2. one coherent ROS snapshot advances recurrence once;
3. candidate scoring never mutates shared posterior/base future;
4. published command—not actor intent—is the plant-input feedback field;
5. ActionGuard remains downstream of learned selection;
6. localization owns pose validity; Local owns short-horizon action selection;
7. Global consumes only a frozen promoted Local and explicit capability validity;
8. all model/data/robot identities flow from training artifact to evaluation and live node.

## 5. Minimal command workflow

Exact entry points may evolve; resolve available CLI help before use.

```bash
# preflight
git status --short
find config -type f | sort
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q

# TRACTOR contract/data gate
python3 -m hunter_kinodynamic_rl.training.tractor_preflight --help
python3 -m hunter_kinodynamic_rl.training.train_tractor_tqc --help
python3 -m hunter_kinodynamic_rl.evaluation.tractor_latency --help

# terminal A: live Environment v2 (after Gazebo is running)
ros2 launch hunter_kinodynamic_rl environment.launch.py profile:=tractor_local_dynamic_v2

# terminal B: collect immutable development episodes; no learning yet
python3 -m hunter_kinodynamic_rl.training.train_tractor_tqc \
  --profile tractor_local_dynamic_v2 \
  --variant a7 \
  --dataset-root runtime/tractor_sequence_v2_seed11 \
  --seed 11 collect --episodes 352

# freeze the replay population and train; use a fresh run root per seed
python3 -m hunter_kinodynamic_rl.training.train_tractor_tqc \
  --variant a7 \
  --dataset-root runtime/tractor_sequence_v2_seed11 \
  --run-root runtime/tractor_tqc/a7_seed11 \
  --seed 11 --device cuda train --updates 150000
```

`pipeline` is a convenience command that performs collection and then training in the same process.
It still closes collection before constructing the sequence index. `train --resume` restores the
model, all optimizers, entropy state, target-policy RNG and exact sampler state, and rejects a changed
dataset/index. The current live collector labels are explicitly marked
`nominal_preaction_rollout_summary_v1`; they exercise the complete development path but are not a
substitute for the formal realized timestamp-aligned candidate corpus.

The strict paper matrix has one orchestrator and one fresh immutable root:

```bash
# freeze manifests and expected B1–B8/A7–A9 × five-seed matrix
ros2 run hunter_kinodynamic_rl run_paper_comparison_campaign.py \
  --campaign-root runtime/tractor_paper_v1 prepare

# with the registered simulator/profile running: collect all 616 scenarios once
ros2 run hunter_kinodynamic_rl run_paper_comparison_campaign.py \
  --campaign-root runtime/tractor_paper_v1 collect

# 55 matched training runs; commands are restartable with --skip-complete
ros2 run hunter_kinodynamic_rl run_paper_comparison_campaign.py \
  --campaign-root runtime/tractor_paper_v1 train --device cuda --skip-complete

# calibration split only, then one locked-test pass per method/seed
ros2 run hunter_kinodynamic_rl run_paper_comparison_campaign.py \
  --campaign-root runtime/tractor_paper_v1 calibrate --device cuda --skip-complete
ros2 run hunter_kinodynamic_rl run_paper_comparison_campaign.py \
  --campaign-root runtime/tractor_paper_v1 evaluate --device cuda --skip-complete

# fail closed unless 55 runs and 9,680 locked episode records are verified
ros2 run hunter_kinodynamic_rl run_paper_comparison_campaign.py \
  --campaign-root runtime/tractor_paper_v1 aggregate \
  --runtime-json <target-hardware-latency.json>
ros2 run hunter_kinodynamic_rl run_paper_comparison_campaign.py \
  --campaign-root runtime/tractor_paper_v1 status
```

Formal runs require an explicit fresh output root. Run long stages in `tmux`, preserve stdout/stderr,
record GPU/driver/container/commit identity and verify the first checkpoint in a separate process.
`aggregate` reports paired seed-level B1 comparisons; it cannot turn incomplete or non-target timing
artifacts into a pass.

## 6. Troubleshooting order

| Symptom | Check first | Required response |
|---|---|---|
| shape mismatch | schema/order/fingerprint, not only tensor shape | reject or explicit migration |
| NaN/non-finite | raw units, masks, covariance PSD, survival logits | stop update; preserve failing batch |
| recurrent drift | reset epoch, burn-in, one-commit rule, timestamp chain | reproduce with deterministic fixture |
| risk looks too safe | censoring, class prevalence, missing masks, calibration split | block promotion |
| candidate order changes output | in-place shared state, dropout/RNG, mask reorder | fail property test |
| resume diverges | target/optimizer/replay/RNG/counter restoration | reject generation |
| Gazebo command mismatch | requested→guarded→published trace, controller semantics | separate model vs executor fault |
| latency miss | stage profile, synchronization, dense copies, ROS scheduling | fallback and optimize before export |
| GPU OOM | batch/window/K/H/M and retained dense futures | new run config/output root; do not corrupt resume |

Repeated faults are logged by experiment ID and promoted into a regression test when reproducible.

## 7. Version and artifact policy

- Source/config/schema changes are committed or represented by a dirty-state digest.
- Large datasets/checkpoints/results live outside Git; Git tracks manifests, small summaries and hashes.
- One experiment ID never mixes semantic contracts.
- Protocol-invalid or OOM-invalidated runs remain read-only and a corrected run uses a new versioned
  output root.
- Promotion is atomic; a symlink/pointer changes only after bundle validation and startup probe.

## 8. Definition of done

Implementation is complete only when model/replay/checkpoint/runtime tests pass and an independent
process can train, save, resume, evaluate and export the same fingerprinted family. Research is
complete only after the full multi-seed protocol and claim ledger are satisfied. Real deployment is
complete only after the staged evidence in `SIM2REAL`—not after a Gazebo smoke run.
