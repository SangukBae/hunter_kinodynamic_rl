# TRACTOR-TQC Test Plan

Status: **formal code-contract regression implemented; formal experiment, target timing, HIL and real gates open**

Tests establish code and contract correctness. Navigation, calibration and real-world claims still
require the formal protocol and artifacts.

Bullets describe the required contract; current code-level items are covered by the regression suite
unless marked as experiment/HIL/real evidence. Formal commands additionally require clean committed
source, a recorded container digest and complete immutable artifacts.

## 1. Test ladder

| Level | Required coverage |
|---|---|
| unit | transforms, decoder, dynamics, tube, hazards, losses |
| property | causality, invariance/equivariance, monotonicity, gradient routing |
| contract | schema, fingerprints, checkpoint compatibility, provenance |
| integration | env→replay→agent→save/resume; ROS snapshot→command |
| system | Gazebo scenarios, fault injection, timing |
| statistical | complete seeds, artifacts, CI and exclusion rules |
| HIL/real | staged safety and robot-contract checks |

## 2. P0 regression

- one authoritative footprint is used by collision, tube, map inflation and Global mask;
- dynamic labels follow realized displacement/actual `dt` across waypoint/reset transitions;
- stopping margin uses time-aligned speed;
- `terminated` and `truncated` produce hand-computed targets;
- observation/action/trajectory changes mutate fingerprints and reject incompatible resume;
- live executor accepts only a promoted verified bundle;
- rear/full-circle candidates obey the frozen observability rule.

## 3. Model unit and property tests

### Geometry and belief

- grid origin, axis, half-open bounds and zero tie match fixtures;
- normalized↔physical action round-trip stays within tolerance;
- straight/constant-curvature/brake rollout matches analytic results and current nominal adapter;
- footprint raster mass is finite/nonnegative and respects covariance growth;
- localization covariance is transformed and added exactly once;
- pure ego translation/rotation preserves zero flow for static objects;
- invalid motion chain masks older frames and hidden state hold/reset is exact.

### Causal operator

- candidate permutation produces the identical output permutation;
- single-candidate and batched scores agree;
- scoring/proposal cannot mutate current posterior or dense base future;
- perturbing candidate `i` cannot change candidate `j` private feature;
- explicit product equals a tiny hand-computed grid;
- a clipped tube conserves original in-grid mass plus OOB mass, performs no in-grid renormalization and
  yields `unknown_overlap + oob_mass` for boundary-or-unknown evidence.

### Risk and learning

- hazards are finite/nonnegative, total cause hazard is below one and survival is monotone;
- event, cause, time and censored NLL match hand calculations;
- severity quantiles remain ordered and masks exclude missing labels;
- TQC target uses actual `terminated`, configured truncation and target path;
- Bellman current critic uses only the stored pre-guard requested action; arbitrary MC-3 changes leave
  its quantiles/loss unchanged while guard outcome remains in reward/next state;
- imagined samples never reach the primary Bellman loss;
- optimizer parameter sets are complete and pairwise disjoint;
- actor step changes actor only while action gradient through scorer is finite/nonzero;
- failed/overflowed Stage-4 or Stage-5 joint transaction changes no affected weight/optimizer/RNG state and performs no EMA update.

## 4. Replay and checkpoint

- windows never cross episode/reset/schema boundaries;
- reset-prefix burn-in reconstruction equals online recurrence for both online and EMA target weights;
- split/group/geometry leakage detector catches renamed duplicates and derived sidecars;
- interrupted chunk/save writes recover the previous complete generation (core publication test exists);
- exact resume restores current online/target/optimizers/scaler/replay/RNG/counters;
- semantic component mutation and required metadata/parent/root lineage mismatch trigger rejection;
- wrong role/hash/path traversal fails before deserialization;
- deployment bundle has no optimizer, target or replay payload and is bound to exact Stage-5/calibration hashes.

## 5. ROS/Gazebo integration

Trace one coherent decision:

```text
sensor/localization snapshot
→ encode once
→ K candidate batch
→ selected normalized/physical action
→ requested command
→ guarded command
→ confirmed published command
→ next replay row
```

Inject stale scan, invalid covariance, localization jump, NaN action, empty candidate set, controller
delay and missed deadline. Each must produce the registered fallback, diagnostic reason and no hidden
state double-advance.

Gazebo smoke covers reset→step and coherent data collection. Offline integration covers
replay→Stage-3/4/5 update→save/resume. Scenario tests cover static/dynamic feasibility, collisions,
near misses, braking, occlusion and OOD perturbations. Smoke success is not a benchmark result.

Current readiness snapshot (2026-09-08): sourced Docker regression **2,253 passed in 273.95 s**;
ROS package build passed; live Gazebo development collection produced 3 L0 episodes/120 rows/75
windows with validator errors 0; a separate L4 static-4/dynamic-8 scene completed 20 steps with valid
telemetry and a following cleanup reset. See `verification/2026-09-08_training_readiness_smoke.md`. These counts are code/runtime
readiness evidence only and do not fill any formal campaign cell.

## 6. Timing and resource tests

Measure warmed-up end-to-end and per-stage p50/p95/p99 on named hardware, with realistic ROS message
load and synchronization. Record candidate count, horizon, precision, CPU/GPU utilization, peak memory,
deadline misses and fallback outcome. Frozen v2 release gate: target hardware, at least 10,000 timed
decisions, p99 `≤100 ms` and misses `≤1%`.

## 7. Statistical artifact checks

Automated aggregation rejects missing/duplicate seed-scenario pairs, mixed fingerprints, incomplete
episodes, inconsistent denominators, unregistered exclusions, calibration/test overlap and unavailable
raw evidence. It reports individual seeds, event counts, confidence intervals and paired effects.

## 8. HIL and real checks

Before wheel motion verify bundle/robot/controller hashes, command bounds, E-stop, geofence, watchdog,
sensor/localization freshness and operator communication. Progress through stopped-wheel, lifted-wheel,
contained static and controlled dynamic trials; any [SIM2REAL.md](SIM2REAL.md) stop condition aborts.

## 9. Completion gates

- **Implementation complete:** all unit/property/contract/integration tests pass in a recorded environment.
- **Formal simulation complete:** every registered seed×scenario artifact is valid and aggregate gates pass.
- **Real validation complete:** the exact promoted bundle passes the preregistered ODD and trial matrix.

No lower gate may be renamed as a higher one.
