# Checkpoint and Compatibility Contract

Status: **IMPLEMENTED core save/load/export contracts; no promoted trained artifact**

Checkpoint compatibility is semantic, not shape-only. Current TQC and TRACTOR are different model
families; neither may resume the other. Partial weight import is a recorded warm start into a new
experiment.

`rl/checkpointing/tractor.py` implements atomic training generations, path/role/fingerprint/hash
validation before deserialization, exact agent/sampler/RNG restoration and inference-only calibrated
bundle export. Tests use synthetic untrained state; live ROS loading and promotion evidence remain open.

## 1. Artifact roles

| Role | Contains | Must not contain |
|---|---|---|
| `training_checkpoint` | online/target weights, optimizers, scaler, replay/sampler, RNG, counters | fitted deployment calibrator |
| `calibration_artifact` | source checkpoint identity, fitted mapping/thresholds, calibration split hashes | neural optimizer/replay |
| `deployment_bundle` | approved inference weights, calibrator, resolved robot/config and manifest | targets, optimizers, replay |

Each artifact is an immutable generation with a root checksum. Role is validated before deserializing
payload files.

## 2. Required identity

The manifest records:

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

Keys in the canonical superset are always present: inactive fields are JSON `null`, `[]` or `{}` as
specified by schema. Missing and disabled are not synonymous.

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
bootstrap_policy, ordered_member_seeds, inclusion_digest
risk_aggregation_contract, expected_calibration_context_id
```

R0–R5 require `Q_r=1`; R0–R4 enable exactly their registered scalar/event/hazard keys, R5 enables one
matched hazard+clearance+stopping bundle, and R6 enables that indivisible bundle for every ordered
member. Only R6 has non-null stochastic bootstrap seeds/inclusion digest. Inactive component keys are
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
| bootstrap policy/seeds/digest | exact active values | source copy | source copy; no sampler state |
| expected calibration context | non-null for R5/R6, otherwise null | exact source copy | exact source copy |
| fitted calibration payload/hash | null | required | required exact copy |
| deployment eligibility | never implied | never implied | only R5/R6 with approved calibration |

## 3. Component inventory

A Stage-5 training checkpoint contains the exact enabled set of:

- belief, ego-goal and vehicle-response encoders;
- actor and action-independent future scene model;
- enabled vehicle residual members;
- tube interaction and temporal aggregator;
- two online return critics and active risk-member bundle;
- a complete EMA target copy of the value path, including encoders and interaction;
- four disjoint optimizer states: value path, risk heads, actor, entropy;
- entropy state, optional AMP scaler, `target_policy_online_actor_philox_v1` RNG state/counter and
  sorted-join-key contract;
- replay/index/sampler generation references.

Inactive variant components must be absent. Runtime recurrent hidden states are caller-owned and are
not serialized as model parameters. Core A7 uses exact-zero initial hidden state.

## 4. Phase and initialization rules

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

Save protocol:

1. freeze a coherent step boundary;
2. write payloads and manifest to a new temporary generation;
3. hash every file and canonical manifest;
4. validate component ownership, parent lineage and counters;
5. fsync where supported and atomically rename;
6. update the current pointer only after validation.

Load protocol validates path confinement, role/schema, root checksum, lineage, all fingerprints,
attestations and component inventory before tensor deserialization. It then restores state and runs a
deterministic probe. Corrupt, partial, foreign-family or dirty-identity mismatches fail closed.

## 7. Calibration and deployment

Calibration is fitted on the dedicated calibration split against one immutable Stage-5 checkpoint.
Its artifact stores source hash, episode IDs, member order, method, fitted parameters and metrics.
Refitting creates a new artifact; it never mutates training generations.

Deployment export accepts only a promoted Stage-5 checkpoint plus approved calibration artifact. The
bundle includes inference components, resolved robot/config, contract hashes, latency evidence and
fallback thresholds. On the robot, mismatched footprint/controller/action decoder, missing calibrator,
stale source identity or failed startup probe prevents command publication.

## 8. Required tests

- interruption at each save stage leaves the previous generation loadable;
- bitwise/deterministic save-resume reproduces sampling, proposals and updates;
- every semantic fingerprint mutation causes the intended reject/warm-start decision;
- missing/extra/overlapping optimizer ownership is rejected;
- Stage transition imports no target/optimizer/replay state;
- calibration and deployment lineage hashes resolve to exact source artifacts;
- path traversal, corrupted hash and wrong artifact role fail before deserialization.
