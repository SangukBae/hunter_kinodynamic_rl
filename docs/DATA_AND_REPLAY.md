# Data and Replay Contract

Status: **FORMAL SCHEMA/VALIDATOR IMPLEMENTED; no collected rollout dataset**
Schema: `tractor_sequence_v2`

Current IID transition replay cannot train recurrent ego-warped belief or cause-time risk. TRACTOR
therefore uses episode-oriented durable storage plus sampled sequence windows in `rl/replay/`.
Durability, checksum, reset-boundary, terminal and exact sampler-resume tests pass. Dense supervision,
candidate lineage and the immutable corpus-root manifest are enforced for formal data. This schema
does not silently reinterpret legacy replay, and no 616-episode formal dataset has yet been collected.

## 1. Storage layers

| Layer | Role |
|---|---|
| raw episode store | immutable packed observation, timestamps, masks, outcomes, dense targets and provenance |
| replay index | valid sequence starts, event strata, split and sidecar references |
| sampled batch | burn-in + loss window, normalized only after loading |
| optional sidecar | additional large targets keyed by episode and step; core formal dense targets are in episode rows |

Candidate trajectories and command telemetry use physical units, but the observation remains the exact
packed policy vector plus validity/motion/covariance fields rather than a second decomposed copy of all
raw sensor messages. Formal provenance binds the generator and packed contract; normalization parameters
remain model-manifest fields.

## 2. Episode header

Each episode records at least:

```text
schema_id, episode_id, scenario_id, scenario_family, obstacle_contract
scenario_geometry_sha256, group_id, split_id, seed
software_commit, dirty_state_digest, container_image_digest
resolved_config_hash, protocol_version
environment_attestation_hash, robot_attestation_hash
observation/action/trajectory contract hashes
start/end UTC, termination reason, step count
sensor/localization/controller sources and clock domains
```

An episode belongs to exactly one development, calibration or locked-test split. Derived windows and
sidecars inherit that assignment. In v2, `group_id` equals the canonical scenario geometry SHA-256;
renaming a copied scenario therefore cannot hide cross-split geometry leakage. The corpus-root manifest
additionally records every episode byte hash, row/split counts, sequence-index hash, protocol/contract,
behavior seed and committed source/container identity.

## 3. Per-step record

The currently enforced `CORE_STEP_FIELDS` store the packed observation and masks, motion delta,
previous published command, covariance/localization fields, requested normalized action, reward,
measured `dt`/discount and terminal semantics. `CANDIDATE_FIELDS` store fixed-capacity physical
trajectories, candidate masks, cause/time/censor/severity/progress labels and one candidate-set hash.
`PRIVILEGED_LABEL_FIELDS` store the pre-action ego/obstacle snapshot. Formal collection additionally
requires the all-or-none `FORMAL_SUPERVISION_FIELDS` and validates every shape, mask and SHA-256 field.

### Observation and timing fields

```text
decision_timestamp_ns
packed observation[328], scan_valid[4,80]
vehicle_response_valid[3]
pose_xyyaw, pose_covariance[3,3], localization/confidence validity
sensor_freshness_sec and validity
motion_delta_from_previous[dx,dy,dyaw,dt] and validity
reset_epoch, scene_reset, response_reset
```

The packed vector preserves the current four-frame scan and 8D tail contract. `dt` is timestamp-derived and
must be positive. Missing motion, pose or scan validity is stored explicitly; it is never encoded as
valid zero.

### Action and response fields

```text
action_normalized_requested[3]
candidate_actions_normalized[K,3]
candidate_trajectories_physical[K,3]
candidate_set_sha256 and candidate selection masks
```

The runtime telemetry distinguishes requested, guarded and confirmed-published values. The formal
sequence row stores the executed transition's requested normalized action, complete candidate set and
previous confirmed-published command. The next row consumes the previous confirmed-published command
exactly once. Runtime selector diagnostics remain evaluation telemetry; replay does not reconstruct
them from a near-equal action.

### Outcome and risk

```text
reward
transition_dt_sec, discount_factor
next_observation reference, next_observation_valid
terminated, truncated, termination_reason, bellman_sample_valid
candidate event/time/cause/censor labels and validity
candidate clearance/stopping margin and validity
privileged source snapshot and timestamp
```

Terminal semantics are closed and validated before sampling:

| `terminated` | `truncated` | next valid | Bellman handling |
|---:|---:|---:|---|
| true | false | either | reward-only branch; next tensors are never evaluated |
| false | false | true | bootstrap with measured-time `discount_factor` |
| false | true, time limit | true | bootstrap; mark truncation for reporting |
| false | true, infrastructure/invalid snapshot | false | `bellman_sample_valid=false`, exclude from Bellman loss |
| false | false | false | invalid transition, reject/quarantine |
| true | true | either | reject in v1; reason contract is ambiguous |

`termination_reason` is a versioned enum whose values declare true environment terminal, time limit,
operator/guard stop, reset, sensor/localization failure and infrastructure failure. `discount_factor`
is deterministically derived from `transition_dt_sec` and the fingerprinted reference discount rule;
it is not silently fixed to nominal 0.1 s. No-event sequence end is censored unless the observation
window proves survival to its horizon.

### Scene, flow and response supervision — implemented for formal rows

Formal rows store the following with independent validity masks:

```text
current_bev_class_target[64,64]          # F/S/D/U integer class
current_bev_class_valid[64,64]
current_dynamic_flow_target[2,64,64]     # ground-relative in E_t^ego
current_dynamic_flow_valid[64,64]
future_bev_class_target[H,64,64] and validity
future_dynamic_flow_target[H,2,64,64] and validity
tube_oob_mass_target[K,H], tube_coverage_valid[K,H]
vehicle_response_target[3] and validity
label_frame_id, generator_version, source timestamp/hash
```

Current labels supervise `O_t/V_t`; future labels supervise the action-independent `O_base/V_base`.
Unknown/unobserved is a class, while supervision validity is a separate mask. A missing label never
becomes free occupancy or zero flow. `tube_oob_mass_target` is probability mass outside half-open grid
bounds before gathering; in-grid weights retain their original scale and are not renormalized.
Boundary-or-unknown supervision is in-grid unknown overlap plus this OOB mass.

### Candidate supervision and lineage — implemented

Formal episode rows use the fixed-capacity tensor portion of this schema (`K=8`, `H=15`) and bind it
to the base row and all generating contracts:

```text
episode_id, step_index, base_transition_row_sha256
candidate_actions_normalized       float32[K,3]
candidate_trajectories_physical    float32[K,3]   # kappa,v_ref,L
candidate_present                  bool[K]
candidate_is_stop                  bool[K]
candidate_model_valid              bool[K]
candidate_horizon_mask             bool[K,H]
candidate_event_observed           bool[K]
candidate_event_step               int16[K]
candidate_event_cause              int8[K]
candidate_censor_step              int16[K]
candidate_event_label_valid        bool[K]
candidate_clearance_m              float32[K,H]
candidate_clearance_valid          bool[K,H]
candidate_stopping_margin_m        float32[K,H]
candidate_stopping_margin_valid    bool[K,H]
candidate_progress_m               float32[K]
candidate_progress_valid           bool[K]
candidate_set_sha256, action/trajectory/decoder hashes
candidate_execution/robot/scenario/label-generator/source-artifact hashes
```

The sidecar joins exactly one base row by all three join fields. Event tuples are total:

| label state | `valid` | `observed` | `event_step` | `cause` | `censor_step` |
|---|---:|---:|---:|---:|---:|
| invalid/unlabelled | false | false | `-2` | `-2` | `-2` |
| event | true | true | `q∈[0,H-1]` | `0..2` | exactly `q` |
| no event through censor | true | false | `-1` | `-1` | `q∈[0,H-1]` |

Every other tuple is rejected. Cause enum is `{0:static,1:dynamic,2:boundary_or_unknown}`.
`candidate_event_label_valid` is true iff the candidate is present/model-valid, `censor_step` is in its
true horizon, source labels are valid and fields match the event/no-event row; an event must also satisfy
`event_step<=censor_step` (equality in v1).
Per-time severity validity additionally requires `candidate_horizon_mask` and source geometry.

Absent/padded actions use finite storage zero with `candidate_present=false`; label floats use NaN with
their masks false. Consumers gather valid indices before arithmetic and never rely on `0*NaN`. Exactly
one present candidate may carry stop identity. Candidate index is meaningful only with
`candidate_set_sha256`; reordering actions must reorder every mask/label and update the hash.

Counterfactual labels simulate the **unguarded constant-candidate commitment** from the recorded
pre-action measured vehicle state using the fingerprinted commit/actuator model. A later guard change
cannot retroactively alter them. Requested/guarded/published actual-transition fields remain separate.
The executable formal row binds every label to the base transition, action, trajectory, decoder,
executor, effective robot, scenario, generator and source artifact hashes. The validator recomputes
the decoder, rollout, label-generator and base-row identities and rejects a mismatch. Labels remain
training/evaluation targets, never deployment inputs.

The following separate R0 legacy fields are also a target extension; the current sequence schema does
not store them:

R0 does not infer a legacy scalar label from MC-3. The target main transition row stores
`legacy_risk_target:float32`, `legacy_risk_valid:bool`, the **pre-guard
`action_normalized_requested`** hash, and exact current nominal risk-target generator/horizon hashes.
The post-guard published 2D command has no unique `L` and is not mislabeled as a normalized 3D action.

## 4. Sequence windows

Formal core uses `sequence_contract=reset_prefix_exact_v1`:

```text
burn_in = every row from the most recent recurrent reset through loss_start-1
loss_window = proposed 16 consecutive rows
```

Burn-in runs without gradient and reconstructs the current-weight online and EMA-target states exactly;
a fixed eight-row prefix is only an explicitly labelled approximate ablation, not core. Windows may not cross
episode, reset, schema, clock-discontinuity or physical-contract boundaries. Padding has an explicit
mask and never becomes a terminal event. Proposed lengths are configuration, fingerprint and ablation
variables—not measured optima.

Time axis is exact: output bin `q` covers `(q*0.2,(q+1)*0.2] s`, with time zero included in bin 0;
`q=max(0,ceil(event_time/0.2)-1)`. Current-default 0.1 s simulator substeps contribute two samples per
bin. The earliest event wins; an exact timestamp tie uses fingerprinted priority
`boundary_or_unknown > dynamic > static`. Future occupancy/flow targets are the bin-end state expressed
in fixed decision frame `E_t^ego`. Clearance and stopping targets are the minimum over fine substeps in
the bin. Candidate progress is initial-to-final goal-distance reduction at the last valid grid endpoint.
Non-integer substep ratios use the model spec's interpolation rule; no partial final bin exists because
the scoring horizon is conservatively grid-ceiled.

## 5. Sampling

- Core Bellman and auxiliary batches sample uniformly from valid indexed windows; masked risk and
  representation losses use unweighted means over their valid elements.
- Every sampled window occurrence receives a monotonically generated `sample_draw_ordinal:uint64` from
  checkpointed sampler state. Sampling the same replay row twice therefore creates two ordered,
  reproducible occurrences; target-action RNG reproducibility is guaranteed on the recorded
  runtime/device, not claimed bitwise across different devices.
- Event-balanced/priority sampling is disabled for A7 core. Enabling it creates a separate contract
  that must store exact inclusion probability and define clipped/normalized inverse-probability
  estimators plus dedicated unbiasedness/calibration tests.
- R6 risk members are independently initialized complete bundles trained on the same registered rows;
  hazard and severity outputs remain member-aligned during loss and aggregation.
- Repeated windows from one episode never appear in different data splits.
- RNG state, sampler cursor, replay generation and priority state are checkpointed for exact resume.

## 6. Label correctness

Release-blocking fixtures cover:

1. static objects under pure ego translation/rotation yield zero ground-relative object flow;
2. constant-velocity tracks integrate to displacement/actual `dt`;
3. random-waypoint obstacle labels use realized timestamp-aligned motion, not commanded velocity;
4. event time/cause/censoring match scripted static, dynamic and boundary cases;
5. stopping margin uses vehicle speed at the matching closest/event time;
6. out-of-grid and unobserved cells are boundary/unknown, never free;
7. a hand-built clipped tube conserves `in_grid_mass + oob_mass = 1` and reproduces the
   boundary-or-unknown target without renormalization;
8. privileged fields are absent from deployment input tensors.

## 7. Split and leakage policy

Split by scenario episode/group before generating windows. The locked test contains held-out layout,
obstacle trajectory, dynamics/noise and localization-degradation axes. Calibration gets its own
episodes and cannot be reused for threshold tuning or headline evaluation. Real system-identification,
calibration and real test runs are separated by session/day where practical.

No checkpoint, early stopping, normalization statistic, calibrator, scenario selection or ablation
choice may consume locked-test outcomes. A protocol change after inspection creates a new version and
new output root.

`config/tractor/scenario_plan.yaml` freezes six static and five dynamic families into disjoint pools:

| Split | Instances | Seed interval | Permitted use |
|---|---:|---:|---|
| development | 352 | 0–9999 | training and model selection |
| calibration | 88 | 10000–10999 | risk calibration only |
| locked_test | 176 | 20000–20999 | final evaluation only |

`tractor_scenario_plan.py` deterministically creates each fixed YAML, applies the bounded Ackermann
feasibility filter, records geometry/artifact hashes and refuses a non-empty output directory. The
materialized manifest is validated before collection; modifying one scenario byte invalidates it.

```bash
ros2 run hunter_kinodynamic_rl tractor_scenario_plan.py \
  --config-root <install-share>/config --output-root <new-immutable-scenario-root>
```

## 8. Serialization and durability

Episode chunks and indexes are written to temporary files, fsynced where supported, checksummed and
atomically renamed. The root manifest stores ordered chunk hashes, schema/version, row counts, episode
counts, split counts and sidecar lineage. Recovery accepts only complete generations and ignores
partial temporary files.

Legacy transition replay may warm-start same-contract baseline training only through an explicit
migration tool and report. It cannot fabricate motion validity, censoring or command provenance; rows
missing required fields are excluded from TRACTOR sequence training.

## 9. Dataset validation report

Before training, `tractor_dataset_validation.py --formal --scenario-manifest ...` emits machine-readable
counts for:

- episodes/windows per split and scenario family;
- event/cause/censoring prevalence and missing-mask rates;
- `dt`, freshness, motion and covariance validity distributions;
- action/command/response ranges and decoder round-trip error;
- candidate/sidecar completeness and checksum failures;
- duplicate/group/geometry leakage across splits;
- footprint, environment and robot attestation hashes.

Any split leakage, non-monotonic timestamp, contract-hash mixture, silent missing-value fill or
privileged-input leak blocks training.

The executable development path is:

```text
EnvironmentClient (tractor_env_v2)
  -> TractorEpisodeRecorder
  -> immutable EpisodeStore
  -> frozen SequenceIndex / reset-prefix SequenceBuffer
  -> TractorSequenceBatchAssembler
  -> TractorAgent value -> risk -> actor/entropy -> EMA
  -> atomic training generation
```

The live collector conservatively invalidates an entire current scan frame when diagnostics report
any dropped beam, records measured transition time and planar odometry covariance, and stores the
previous confirmed-published command separately from the current requested action. It also stores
explicit `next_*` input columns so a time-limit truncation can bootstrap from the real final post-step
observation. Per-transition requested/guarded/published attribution remains a target extension.

The development-only collector can still emit `nominal_preaction_rollout_summary_v1`. The formal
comparison collector instead records a privileged pre-action world snapshot outside policy inputs and,
after episode completion, aligns every candidate with actual future obstacle states by timestamp. It
emits `realized_timestamp_aligned_counterfactual_v1` cause/time/censor/severity sidecars. Formal
validation requires this realized source for candidate labels. Complete per-label lineage hashes are
not yet enforced, so the formal readiness gate blocks collection. The relabel generator and core source
check are implemented, but no 616-scenario corpus has been collected; therefore label quality and
navigation performance remain unmeasured.
