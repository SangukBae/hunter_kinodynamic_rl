# Model Promotion Card

Use one card for each candidate, promoted or deployment bundle.

## Identity

```text
model/bundle ID:
status: candidate | rejected | promoted-sim | promoted-HIL | promoted-real
model family/revision/variant:
checkpoint and calibration generation/root hashes:
source experiment IDs:
architecture/data/training/robot fingerprints:
software/container/hardware:
risk contract ID, Q_r, ordered member IDs and enabled subheads:
bootstrap policy/seeds/inclusion digest:
calibration context and payload hashes:
risk aggregation pair order, ddof, beta/clipping and missing-member rule:
```

## Intended use

Describe Local/Global role, robot/controller, sensor/localization contract, ODD, action/command path and
explicit prohibited uses.

## Evidence

```text
formal seed×scenario completeness:
navigation/prediction/ranking metrics:
risk calibration, event counts and intervals:
OOD/selective-risk evidence:
p50/p95/p99 latency, deadline misses and memory:
regression/integration/HIL/real artifact links:
known failures and operator interventions:
```

## Safety and compatibility

- [ ] exact footprint/action/controller/guard attestations match
- [ ] startup/load probe fails closed on mismatch
- [ ] calibrator and thresholds match this checkpoint
- [ ] stale/invalid/deadline/no-feasible fallbacks verified
- [ ] rollback bundle and approval owner recorded

## Decision

```text
promotion gate and outcome:
approver/date:
permitted next stage:
reason for rejection or limitations:
```
