# Real-Robot Trial Card

One card is required per Hunter session/scenario. Approval applies only to the exact identities below.

## Identity and approval

```text
trial_id, date/time/location:
operators/spotters/approver:
model bundle and calibration hashes:
robot/controller/sensor/localization attestations:
software/container and bag/log destinations:
ODD, route, obstacle plan, repetitions and speed cap:
```

## Pre-motion gate

- [ ] prior sim, target timing and required HIL gates passed
- [ ] physical inspection, battery, brakes and steering passed
- [ ] E-stop, watchdog, geofence and communications tested
- [ ] scan/localization clocks, extrinsics, freshness and covariance valid
- [ ] requested→guarded→published command trace verified
- [ ] abort roles, exclusion zone and unexpected-person procedure briefed

## Stop rules

Stop for E-stop/spotter request, identity mismatch, stale or invalid sensor/localization, non-finite or
empty candidate output, repeated deadline miss, response outside attested bounds, geofence breach,
unexpected entry, collision/contact or registered near-miss.

## Execution record

```text
start/end UTC and repetitions attempted/completed:
fallbacks, guard interventions and operator interventions:
contacts/near misses and minimum clearance:
deadline misses and latency summary:
deviations or environmental changes:
raw bag/log/video manifests and hashes:
```

## Decision

State whether the trial is valid, what exact claim/next gate it supports, limitations and required
follow-up. A completed drive does not automatically authorize promotion.
