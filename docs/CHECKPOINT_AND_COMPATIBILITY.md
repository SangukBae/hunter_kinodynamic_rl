# Checkpoint and Compatibility Contract

Status: **SEMANTIC TRAINING/CALIBRATION/DEPLOYMENT LINEAGE IMPLEMENTED; no trained or promoted artifact**

Checkpoint compatibility is semantic, not shape-only. Current TQC and TRACTOR are different model
families; neither may resume the other. Partial weight import is a recorded warm start into a new
experiment.

`rl/checkpointing/tractor.py` implements atomic generation-directory publication,
path/role/generation/model-fingerprint/payload-hash validation before deserialization,
component semantic hashes, parent/root/stage lineage, exact agent/sampler/global-RNG restoration,
online-only Stage warm start, finite-weight startup probes and inference-only calibrated bundle
serialization. Export cryptographically binds the in-memory online weights and calibrator to the exact
Stage-5 source generation. Tests use synthetic untrained state; an actually trained/promoted artifact,
target latency evidence and live ROS deployment evidence remain open.

## 1. Artifact roles

| Role | Contains | Must not contain |
|---|---|---|
| `training_checkpoint` | online/target weights, optimizers, scaler, replay/sampler, RNG, counters | fitted deployment calibrator |
| `calibration_artifact` | source checkpoint identity, fitted mapping/thresholds, calibration split hashes | neural optimizer/replay |
| `deployment_bundle` | approved inference weights, calibrator, resolved robot/config and manifest | targets, optimizers, replay |

Training generations are immutable after publication. The tensor payload has a byte SHA-256 and every
component has a serialization-independent semantic state hash. Role, path, generation, model fingerprint
and byte hash are validated before tensor deserialization.

## 2. Executable identity and canonical extension

The current manifest/metadata records or binds the operational subset below:

```text
schema_id, artifact_role, experiment_id, generation, parent_generation
source checkpoint/calibration generation and SHA-256 where applicable
model_family, architecture_revision, variant_id, model_semver
observation/action/trajectory/replay schemas
protocol_version, ablation_contract
software commit, dirty-state digest, container image digest
resolved config and hash
architecture, data and training fingerprints
environment and robot attestations
training phase, step/episode/resume counters
present component names, shapes and hashes
optimizer ownership hashes, target-network contract
calibration context, evidence level and promotion status
```

The checkpoint schema is `tractor_training_checkpoint_v2`; calibration and deployment use role-separated
v2 schemas. The training entrypoint supplies dataset manifest/index/validation hashes, scenario/contract/
protocol, method/seed/stage, source/container identity and device metadata. Loaders can require exact
metadata fields. A larger always-present canonical superset with explicit nulls is a future schema
hardening item, not a claim about the current manifest.

The architecture fingerprint includes grid axes/bounds, observation order, action decoder, trajectory
and commit semantics, belief/operator/head variants, `K/H/M/Q_m/Q_r`, recurrence policy and footprint.
The data fingerprint includes row/window schema, masks, label generators, split manifest and
normalization. The training fingerprint includes losses, optimizer ownership, update order, target
EMA, RNG and schedule.

The active risk bundle is a tagged manifest object:

```text
risk_contract_id: R0 | R1 | R2 | R3 | R4 | R5 | R6
Q_r, ordered_member_ids, enabled_subhead_keys, output_shape_schema
loss_and_label_schema_hashes
risk_member_training_policy, ordered_member_initialization_seeds
risk_aggregation_contract, expected_calibration_context_id
```

R0–R5 require `Q_r=1`; R0–R4 enable exactly their registered scalar/event/hazard keys, R5 enables one
matched hazard+clearance+stopping bundle, and R6 enables that indivisible bundle for every ordered
member. R6 uses independently initialized members on the same registered rows. Inactive component keys are
absent. Output/label/loss shapes are normative in `TRACTOR_TQC_MODEL_SPEC` section 8.
Registered deployment families map exactly as A7→R5, A8→R5 and A9→R6; a different pair is a new
architecture fingerprint, not a load-time option.
For A9, `risk_aggregation_contract` is exactly
`vehicle_outer_risk_inner_uniform_v1` and stores the lexicographic Cartesian `(m,r)` order, fixed
denominator, `ddof=1`, calibrate-before-UCB/LCB order, beta/clipping and no-drop invalidation rule.

| Risk field | training checkpoint | calibration artifact | deployment bundle |
|---|---|---|---|
| tag/`Q_r`/member order/schema hashes | exact active values | exact source copy | exact source copy |
| neural risk component keys | exact active bundle | absent | exact inference bundle |
| member training policy/order | exact active values | source copy | source copy; no sampler state |
| expected calibration context | non-null for R5/R6, otherwise null | exact source copy | exact source copy |
| fitted calibration payload/hash | null | required | required exact copy |
| deployment eligibility | never implied | never implied | only R5/R6 with approved calibration |

## 3. Component inventory

A Stage-5 training checkpoint contains the exact enabled runtime set of:

- belief, ego-goal and vehicle-response encoders;
- actor and action-independent future scene model;
- enabled vehicle residual members;
- tube interaction and temporal aggregator;
- two online return critics and active risk-member bundle;
- a complete EMA target copy of the value path, including encoders and interaction;
- four disjoint optimizer states: value path, risk heads, actor, entropy;
- entropy state, optional AMP scaler and `target_policy_torch_generator_exact_resume_v1` state;
- replay/index/sampler generation references.

Load rejects any missing or extra component relative to the instantiated agent and rejects any semantic
state hash mismatch. Inactive variant components are absent from the model state. Runtime recurrent
hidden states are caller-owned and are not serialized as model parameters. Core A7 uses exact-zero
initial hidden state.

## 4. Implemented phase and initialization rules

| Phase | Trainable ownership | Target behavior |
|---|---|---|
| Stage 3 representation | value-path subset registered by phase | full target initialized once, not consumed or updated |
| Stage 4 risk/ranking | atomic selected interaction/aggregator value subset + risk heads | full target initialized once, not consumed or updated |
| Stage 5 joint RL | atomic value+risk, then actor/entropy | full value-path EMA after successful joint transaction |

Stage transitions are warm starts with new experiment IDs. They import only declared compatible online
components; optimizer, target, replay, counters and RNG are fresh. After import/fresh initialization,
the complete online value path is copied once to target before any update.

Exact resume restores both online and target weights, all optimizers/scaler, replay/sampler, RNG and
counters. It never overwrites the restored target with online weights.

## 5. Compatibility decisions

| Change | Resume | Warm start | New data/protocol |
|---|---:|---:|---:|
| only logging/output path | yes if fingerprints equal | n/a | no |
| optimizer/LR/update schedule | no | compatible weights may import | training fingerprint |
| actor/head width or risk contract | no | explicit mapped subset only | architecture fingerprint |
| observation/action/decoder/footprint | no | usually no | architecture + data |
| replay fields/window/terminal meaning | no | weights only | data + training |
| calibration mapping/threshold | training unchanged | n/a | new calibration artifact/bundle |
| robot/controller attestation | no deployment reuse | model may be reevaluated | new deployment evidence |

Shape equality never overrides a semantic mismatch. Evaluation may load an unpromoted checkpoint only
in explicit development mode and must label its outputs accordingly.

## 6. Atomic save and load

Implemented save protocol:

1. freeze a coherent step boundary;
2. write payloads and manifest to a new temporary generation;
3. hash the tensor payload and every component semantic state;
4. validate component ownership, parent lineage and counters;
5. fsync where supported and atomically rename;
6. update the current pointer only after validation.

The load protocol validates path confinement, role/schema, payload checksum, model fingerprint and
component inventory before applying tensor state. It then verifies semantic component hashes, restores
state and runs finite-weight startup checks. Training entrypoints additionally validate dataset/source/
protocol metadata. Corrupt, partial, foreign-family or lineage mismatches fail closed.

## 7. Calibration and deployment

Calibration is fitted on the dedicated calibration split against one immutable Stage-5 checkpoint.
Its artifact stores source hash, episode IDs, member order, method, fitted parameters and metrics.
Refitting creates a new artifact; it never mutates training generations.

Deployment export accepts only a promoted Stage-5 checkpoint plus approved calibration artifact. The
bundle includes inference components, resolved robot/config, contract hashes, latency evidence and
fallback thresholds. On the robot, mismatched footprint/controller/action decoder, missing calibrator,
stale source identity or failed startup probe prevents command publication.

## 8. Required evidence before deployment promotion

- atomic publication leaves the previous pointer valid until a complete generation is published;
- exact save-resume restores sampling, proposal RNG, optimizer and update counters;
- semantic component mutation, missing/extra inventory and metadata mismatch are rejected;
- Stage transition imports online weights only and reinitializes target from online;
- calibration and deployment hashes resolve to the exact Stage-5 source artifact;
- path traversal, corrupted byte/semantic hash and wrong artifact role fail before state application;
- before real promotion, record target-hardware latency, HIL startup/load and operator-approved trial evidence.
