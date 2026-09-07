# Simulation-to-Real and Safety Plan

Status: **no Hunter SE real-navigation evidence**

Gazebo command tracking, smoke tests and domain randomization do not establish real-robot fidelity or
safety. Real motion uses only a promoted deployment bundle and advances through independent gates.

## 1. Robot contract

The deployed model is identified by one resolved, hashed robot attestation:

```text
wheelbase, track, wheel radius
body/bumper/tyre geometry and collision footprint
mass/inertia and centre of gravity
speed, curvature, steering, acceleration/braking/rate limits
actuator/controller path and command semantics
sensor frames, rates, range and latency
localization source, covariance/validity semantics
control period, watchdog and E-stop behavior
```

The current improved Local collision envelope (`0.58 m`) and historical hierarchy footprint
(`0.45 m`) are unresolved until P0-01 establishes one source. CAD/manual-informed parameters are
priors; only system identification and controlled measurement support real values.

Profiles have distinct roles:

| Profile class | Use | Restriction |
|---|---|---|
| baseline | preserve comparison with current TQC | not a real-safe claim |
| improved simulation | plausible Hunter geometry/dynamics | Gazebo evidence only |
| real deployment | measured limits/controller/sensors | cannot inherit unverified sim defaults |

## 2. Responsibility boundary

- Localization/SLAM estimates pose and uncertainty; TRACTOR consumes relative motion and validity.
- Local TRACTOR chooses short-horizon `[kappa,v_ref,L]`.
- Pure-Pursuit/controller converts trajectory to requested command.
- Fixed guard, watchdog, geofence and E-stop retain final authority.
- Global uses only a frozen, promoted Local capability interface.

Neural risk is decision support, not a safety certificate. Unknown, stale and invalid states are not
encoded as safe zeros.

## 3. Evidence gates

| Gate | Activity | Required evidence to advance |
|---|---|---|
| SR-0 | contract repair | footprint/action/controller/fingerprint consistency, full regression |
| SR-1 | formal simulation | complete multi-seed ID/OOD, calibration and latency gates |
| SR-2 | system identification | timestamped command-response/brake/steering trials, fitted uncertainty |
| SR-3 | recorded-data replay | no command output; perception/risk/latency evaluated on bags |
| SR-4 | HIL/stopped or lifted wheels | controller, watchdog, E-stop, bundle and fault-injection checks |
| SR-5 | contained static motion | low-speed geofenced course with physical clearance |
| SR-6 | controlled dynamic obstacle | trained spotters, repeatable actor paths, abort envelope |
| SR-7 | GPS-denied localization | separate pose-error/drift metrics and induced-degradation trials |
| SR-8 | Global integration | Local frozen; capability validity and mission failure attribution |

Failure at one gate does not erase data, but blocks promotion. Any model/config change affecting
behavior returns to the earliest invalidated gate.

## 4. Sim-to-real data policy

Partition real sessions into system-ID, development, calibration and locked test. Do not tune on real
test outcomes. Randomize only identified or defensible ranges and report them; overly broad
randomization is not evidence. Vehicle residual learns command-response mismatch only and is bounded
around the nominal model. Localization errors remain explicit inputs/evaluation axes.

Recorded replay checks scan validity, ego warp, occupancy/flow, risk calibration, candidate ranking and
target-device timing without actuation. HIL then checks the exact requested→guarded→published command
path and hidden-state lifecycle.

## 5. Defense layers

1. schema/fingerprint/robot-bundle validation at startup;
2. synchronized sensor and localization validity checks;
3. candidate feasibility, uncertainty and calibrated-risk filters;
4. bounded physical decoder and controller rate limits;
5. fixed ActionGuard and command watchdog;
6. geofence, speed cap and independent E-stop;
7. trained operator and trial-specific abort plan.

No learned module may suppress a guard reason or publish directly around the controller boundary.

## 6. Trial checklist

Before each trial record model/calibrator/bundle hashes, robot/controller configuration, battery,
weather/floor, sensor extrinsics, localization health, E-stop and geofence checks, personnel roles,
speed cap and expected scenario timeline.

During the trial retain raw sensor/localization topics, candidate/value/risk/uncertainty tensors,
selected trajectory, requested/guarded/published commands, controller feedback, guard/fallback reasons,
latency/deadlines and operator events on a common clock.

After the trial hash logs, record deviations/interventions, verify no missing interval, compute metrics
with denominators and link the immutable trial card. A successful anecdotal run is not promotion.

## 7. Immediate stop conditions

- E-stop, geofence or spotter request;
- stale/invalid scan or localization beyond the registered grace policy;
- bundle, robot, controller or calibration identity mismatch;
- non-finite action, empty feasible set or repeated deadline miss;
- speed/steering/braking response outside the attested envelope;
- unexpected obstacle/person entry or loss of communication;
- collision, contact, near-miss threshold or persistent guard escalation.

After a stop, preserve logs and diagnose offline. Do not continue by widening thresholds during the same
trial identity.

## 8. Allowed claim scope

Report the exact robot, software, bundle, localization backend, ODD, number of trials, interventions,
event counts and uncertainty. A contained static trial supports only that contained ODD; it does not
support general GPS-denied navigation or dynamic-obstacle safety.
