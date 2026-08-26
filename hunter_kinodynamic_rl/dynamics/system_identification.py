"""Real-Hunter-SE system identification (section 16): analysis functions
(pure, unit-tested against synthetic data) + a thin ROS-optional CSV
recorder for the four trial types (velocity step response, steering step
response, constant-circle turning radius, stop response).

HONESTY NOTE: the ANALYSIS functions below are exercised by
tests/test_system_identification.py against synthetic (noise-free)
trajectories. The recorder class (bottom of file) that logs a REAL robot's
odometry/cmd during a live trial has NOT been run against real hardware in
this session -- there is no real Hunter SE available to this development
environment. Results only become available once someone runs an actual
trial and feeds the output through :func:`analyze_circle_test` etc.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
import os
from dataclasses import dataclass
from typing import List, Optional

import yaml

from hunter_kinodynamic_rl.config.schema import RobotConfig


@dataclass(frozen=True)
class Sample:
    t_sec: float
    x: float
    y: float
    yaw: float
    v_mps: float
    steering_rad: float


def analyze_velocity_step_response(samples: List[Sample], target_v_mps: float) -> dict:
    """First-order-response time constant: time to reach 63.2% of the step."""
    if not samples:
        return {}
    v0 = samples[0].v_mps
    target_delta = target_v_mps - v0
    if abs(target_delta) < 1e-6:
        return {"tau_sec": 0.0, "steady_state_v_mps": v0, "valid": True, "reason": None}
    threshold = v0 + 0.632 * target_delta
    tau = samples[-1].t_sec
    reached = False
    for s in samples:
        if (target_delta > 0 and s.v_mps >= threshold) or (target_delta < 0 and s.v_mps <= threshold):
            tau = s.t_sec
            reached = True
            break
    return {
        "tau_sec": tau, "steady_state_v_mps": samples[-1].v_mps,
        # section P1-2: `reached=False` means the 63.2% threshold was never
        # crossed within the trial window -- `tau_sec` is then just the
        # LAST sample's timestamp, a lower bound, not a real time constant.
        "valid": reached, "reason": None if reached else "threshold_not_reached_within_window",
    }


def analyze_steering_step_response(samples: List[Sample], target_steering_rad: float) -> dict:
    if not samples:
        return {}
    s0 = samples[0].steering_rad
    target_delta = target_steering_rad - s0
    if abs(target_delta) < 1e-6:
        return {"tau_sec": 0.0, "steady_state_steering_rad": s0, "valid": True, "reason": None}
    threshold = s0 + 0.632 * target_delta
    tau = samples[-1].t_sec
    reached = False
    for s in samples:
        if (target_delta > 0 and s.steering_rad >= threshold) or (target_delta < 0 and s.steering_rad <= threshold):
            tau = s.t_sec
            reached = True
            break
    return {
        "tau_sec": tau, "steady_state_steering_rad": samples[-1].steering_rad,
        "valid": reached, "reason": None if reached else "threshold_not_reached_within_window",
    }


def analyze_circle_test(samples: List[Sample]) -> dict:
    """Measured turning radius from a constant-steering trial: fit a circle
    through the (x, y) trace via the algebraic (Kasa) least-squares method."""
    if len(samples) < 3:
        return {}
    xs = [s.x for s in samples]
    ys = [s.y for s in samples]
    n = len(xs)
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    u = [x - mean_x for x in xs]
    v = [y - mean_y for y in ys]
    suu = sum(ui * ui for ui in u)
    svv = sum(vi * vi for vi in v)
    suv = sum(ui * vi for ui, vi in zip(u, v))
    suuu = sum(ui ** 3 for ui in u)
    svvv = sum(vi ** 3 for vi in v)
    suvv = sum(ui * vi * vi for ui, vi in zip(u, v))
    svuu = sum(vi * ui * ui for ui, vi in zip(u, v))

    det = suu * svv - suv * suv
    if abs(det) < 1e-9:
        return {"turning_radius_m": math.inf}
    rhs_u = 0.5 * (suuu + suvv)
    rhs_v = 0.5 * (svvv + svuu)
    uc = (rhs_u * svv - rhs_v * suv) / det
    vc = (rhs_v * suu - rhs_u * suv) / det
    radius = math.sqrt(uc * uc + vc * vc + (suu + svv) / n)
    return {"turning_radius_m": radius, "center_x": mean_x + uc, "center_y": mean_y + vc}


def _extrapolate_forward_at_constant_velocity(sample: Sample, t_sec: float) -> Sample:
    """item-4 (round 2): advances ``sample`` to ``t_sec`` assuming the
    robot continued in a straight line at ITS OWN (still steady-state,
    NOT YET braking) ``v_mps``/``yaw`` -- the fallback estimate for the
    true brake-onset reference point when no live ``brake_onset_state``
    snapshot is available (offline CSV reprocessing). Physically justified
    specifically because ``sample`` is the LAST recorded instant known to
    still be BEFORE the brake command -- unlike interpolating toward the
    first POST-onset sample (already mid-deceleration), extrapolating
    forward from steady-state cruise does not borrow any of the
    deceleration's own bias."""
    dt = t_sec - sample.t_sec
    return Sample(
        t_sec=t_sec, x=sample.x + sample.v_mps * math.cos(sample.yaw) * dt,
        y=sample.y + sample.v_mps * math.sin(sample.yaw) * dt,
        yaw=sample.yaw, v_mps=sample.v_mps, steering_rad=sample.steering_rad,
    )


def analyze_stop_test(
    samples: List[Sample],
    *,
    brake_onset_t_sec: Optional[float] = None,
    brake_onset_state: Optional[Sample] = None,
    brake_onset_snapshot_failure_reason: Optional[str] = None,
    allow_fallback_despite_live_snapshot_failure: bool = False,
    initial_speed_target_mps: Optional[float] = None,
    steady_state_tolerance_mps: float = 0.08,
    stop_speed_threshold_mps: float = 1e-3,
    stop_hysteresis_samples: int = 1,
    max_sample_gap_sec: float = 0.5,
) -> dict:
    """Measured stopping distance/time from an initial speed to v==0.

    section P1-2 fix: the ORIGINAL implementation used ``samples[0]`` as the
    stop trial's "start" and scanned for the first ``|v| < 1e-3`` sample --
    correct ONLY if ``samples`` already covers exactly the braking segment
    (which is what every existing unit test's synthetic data does, and
    what a caller with no ``brake_onset_t_sec`` still gets, preserving
    backward compatibility). ``system_id_node.py``'s live recorder,
    however, records CONTINUOUSLY from trial start -- including the initial
    acceleration-to-target-speed ramp, which itself starts at v=0 -- so
    ``samples[0]`` was actually the very first (v=0, t=0) sample of the
    ACCELERATION phase, and the "first sample below threshold" scan matched
    it immediately, reporting a bogus 0m/0s stop before the brake command
    was ever issued. Passing ``brake_onset_t_sec`` (the recorder's own
    timestamp for when it switched from commanding ``target_v_mps`` to
    commanding 0) trims the analysis window to the braking segment only.

    item-4 (round 1) fix: anchoring at the LAST sample strictly BEFORE
    brake onset (instead of the first sample AT/AFTER it) avoided
    misclassifying a genuinely-normal trial as ``steady_state_not_reached``
    purely because the first POST-onset sample already showed some
    deceleration.

    item-4 (round 2) fix, the one THIS docstring section is about: round
    1's fix introduced a DIFFERENT bug -- the last PRE-onset sample's own
    timestamp is itself usually strictly EARLIER than ``brake_onset_t_sec``
    (odometry does not tick every millisecond), so anchoring there means
    whatever the robot travelled during that gap (still at steady speed,
    NOT braking yet) is silently counted as part of ``stopping_distance_m``/
    ``stopping_time_sec``. Concretely: ``brake_onset_t_sec=1.5`` with the
    last pre-onset sample at ``t_sec=1.1`` previously left the 1.1->1.5
    (0.4 s) pre-braking segment INCLUDED in the measurement. Two ways to
    get an "at ``t_sec==brake_onset``, exactly" reference point instead, in
    priority order:

    1. **``brake_onset_state``** (preferred, when available): a
       :class:`Sample` for the EXACT instant the brake command was issued,
       snapshotted DIRECTLY by ``SystemIdRecorder`` from live odometry/
       joint-state readings right before it publishes the stop command
       (see ``system_id_node.py::SystemIdRecorder.run_trial``) -- not
       derived from the discrete ``samples`` list at all, so there is no
       gap to close in the first place. Used AS-IS as the trial's
       ``start`` whenever given (``brake_onset_t_sec`` is then only
       consulted for the reason above -- it is unused once
       ``brake_onset_state`` is provided, since ``brake_onset_state.t_sec``
       is authoritative).
    2. **``brake_onset_t_sec``-only fallback** (offline CSV reprocessing
       with no live snapshot recorded): the last pre-onset sample is
       EXTRAPOLATED FORWARD to ``t_sec=brake_onset_t_sec`` at ITS OWN
       (steady-state, still pre-braking) velocity/heading -- see
       :func:`_extrapolate_forward_at_constant_velocity`. An explicit,
       DOCUMENTED estimate, not a silent one: rejected outright
       (``reason="onset_reference_gap_too_large"``) if the gap being
       extrapolated over exceeds ``max_sample_gap_sec``, the SAME
       reliability bound already applied to gaps BETWEEN ordinary
       samples -- an unbounded extrapolation is exactly as unreliable as
       an unbounded odometry dropout.

    section item-2 (system-ID stale-data fix): ``brake_onset_snapshot_failure_reason``
    is the LIVE caller's (``system_id_node.py``) own explanation for why its
    PREFERRED live snapshot mechanism (``build_brake_onset_snapshot``)
    explicitly failed (stale odometry/joint-state, missing data, or an
    inconsistent receipt time) -- distinct from simply not being given a
    ``brake_onset_state`` at all, which offline CSV reprocessing (no live
    receipt-time tracking exists for a replayed CSV) does unconditionally.
    When this reason is provided (non-``None``) and no ``brake_onset_state``
    was recovered, the sample-based ``brake_onset_t_sec`` extrapolation
    fallback below is a STRICTLY LOOSER check (it only bounds gaps BETWEEN
    RECORDED SAMPLES, never against the live snapshot's own, already-failed,
    receipt-time freshness bound) -- silently falling through to it would
    let a genuinely stale live reading get re-validated as ``valid=True``
    through a laxer path. So by default this immediately returns
    ``valid=False`` with the SAME reason, before ever reaching the
    extrapolation fallback -- a live run's own explicit staleness
    determination is authoritative, never silently overridden by a looser
    check. Passing ``allow_fallback_despite_live_snapshot_failure=True``
    opts back into the sample-based fallback anyway (still bounded by the
    SAME ``max_sample_gap_sec`` used everywhere else in this function), but
    a resulting ``valid=True`` result is tagged ``estimated=True`` and
    ``reference_source="sample_extrapolation_despite_stale_live_snapshot"``
    so nothing downstream can mistake it for an as-good-as-the-preferred-path
    measurement.

    Always returns ``valid``/``reason`` -- ``valid=False`` (never a
    fabricated 0/garbage distance-and-time pair) whenever:
    - the live caller's own preferred brake-onset snapshot mechanism
      explicitly failed and no fallback was opted into (see immediately
      below -- checked FIRST, before the generic empty-samples check, so
      this specific, actionable reason is never masked by it),
    - there are no samples in the (possibly trimmed) analysis window,
    - the pre-brake steady-state speed was never reached (so "initial
      speed" would understate the actual commanded step),
    - the onset-reference gap (fallback path only) exceeds
      ``max_sample_gap_sec``,
    - a gap between consecutive samples exceeds ``max_sample_gap_sec``
      (odometry dropout), or
    - no ``stop_hysteresis_samples``-long run of ``|v| < stop_speed_threshold_mps``
      is ever observed before the window ends (never stopped / trial timed
      out).

    code review (system-ID error-origin ordering fix): the
    ``brake_onset_snapshot_failure_reason`` check below runs BEFORE the
    generic ``not samples`` check -- a live run that never received ANY
    odometry/joint-state at all naturally has both an empty ``samples``
    list AND a live-snapshot failure (e.g. ``"no_odometry_received"``,
    ``"odometry_stale_at_brake_onset"``, ``"joint_state_stale_at_brake_onset"``,
    or an out-of-order receipt-time reason). Checking ``not samples`` first
    (the pre-fix order) would report the far less actionable generic
    ``"no_samples"`` in exactly that case, discarding the real, diagnostic
    root cause -- while still safely returning ``valid=False`` either way,
    which reason string is preserved matters for actually debugging a
    failed system-ID run.
    """
    # section item-2 (system-ID stale-data fix): a live caller's OWN
    # explicit "the preferred live snapshot failed" determination is
    # authoritative -- never silently re-validated by the strictly looser
    # sample-based extrapolation fallback below. See this function's own
    # docstring section for the full rationale. Deliberately checked BEFORE
    # the generic empty-``samples`` check (see the ordering-fix docstring
    # section above) so this specific failure reason is never masked by
    # the far less actionable ``"no_samples"``.
    used_stale_fallback = False
    if (brake_onset_snapshot_failure_reason is not None and brake_onset_state is None
            and not allow_fallback_despite_live_snapshot_failure):
        return {
            "valid": False, "reason": brake_onset_snapshot_failure_reason,
            "brake_onset_t_sec": brake_onset_t_sec,
        }
    if brake_onset_snapshot_failure_reason is not None and brake_onset_state is None:
        used_stale_fallback = True  # allow_fallback_despite_live_snapshot_failure=True, opted in explicitly

    if not samples and brake_onset_state is None:
        return {"valid": False, "reason": "no_samples"}

    window = samples
    if brake_onset_state is not None:
        # section item-4 (round 2): the PREFERRED, gap-free path -- see
        # this function's own docstring, priority 1.
        onset_t_sec = brake_onset_state.t_sec
        post_onset = [s for s in samples if s.t_sec > onset_t_sec]
        if not post_onset:
            return {"valid": False, "reason": "no_samples_after_brake_onset", "brake_onset_t_sec": onset_t_sec}
        start = brake_onset_state
        window = [start] + post_onset
    elif brake_onset_t_sec is not None:
        pre_onset = [s for s in samples if s.t_sec <= brake_onset_t_sec]
        post_onset = [s for s in samples if s.t_sec > brake_onset_t_sec]
        if not post_onset:
            return {"valid": False, "reason": "no_samples_after_brake_onset", "brake_onset_t_sec": brake_onset_t_sec}
        if pre_onset:
            last_pre = pre_onset[-1]
            onset_reference_gap_sec = brake_onset_t_sec - last_pre.t_sec
            if onset_reference_gap_sec > max_sample_gap_sec:
                return {
                    "valid": False, "reason": "onset_reference_gap_too_large",
                    "gap_sec": onset_reference_gap_sec, "brake_onset_t_sec": brake_onset_t_sec,
                }
            # section item-4 (round 2): EXTRAPOLATE forward to
            # t_sec==brake_onset_t_sec exactly -- see
            # _extrapolate_forward_at_constant_velocity's own docstring for
            # why this (never a plain anchor at last_pre's own, earlier,
            # timestamp) is the correct fallback.
            start = _extrapolate_forward_at_constant_velocity(last_pre, brake_onset_t_sec)
            window = [start] + post_onset
        else:
            # No odometry at all before brake onset (recording began at/
            # after the brake command) -- there is no way to establish a
            # genuinely pre-brake reference point, so fall back to the
            # first available sample (equivalent to the pre-item-4
            # behavior) rather than fabricating one.
            start = post_onset[0]
            window = post_onset
    else:
        start = window[0]

    if (initial_speed_target_mps is not None
            and abs(start.v_mps - initial_speed_target_mps) > steady_state_tolerance_mps):
        return {
            "valid": False, "reason": "steady_state_not_reached",
            "initial_speed_mps": start.v_mps, "initial_speed_target_mps": initial_speed_target_mps,
        }

    for prev, cur in zip(window, window[1:]):
        if (cur.t_sec - prev.t_sec) > max_sample_gap_sec:
            return {
                "valid": False, "reason": "odometry_dropout",
                "gap_sec": cur.t_sec - prev.t_sec, "gap_start_t_sec": prev.t_sec,
            }

    stop_sample = None
    n = max(1, stop_hysteresis_samples)
    for i in range(len(window) - n + 1):
        run = window[i:i + n]
        if all(abs(s.v_mps) < stop_speed_threshold_mps for s in run):
            stop_sample = run[0]
            break

    if stop_sample is None:
        return {
            "valid": False, "reason": "did_not_stop_within_window",
            "initial_speed_mps": start.v_mps, "window_duration_sec": window[-1].t_sec - start.t_sec,
        }

    distance = math.hypot(stop_sample.x - start.x, stop_sample.y - start.y)
    result = {
        "valid": True, "reason": None,
        "initial_speed_mps": start.v_mps,
        "stopping_distance_m": distance,
        "stopping_time_sec": stop_sample.t_sec - start.t_sec,
        "brake_onset_t_sec": brake_onset_t_sec,
    }
    if used_stale_fallback:
        # section item-2: an explicitly opted-in fallback despite the
        # preferred live snapshot having failed -- tagged so nothing
        # downstream can mistake this for an as-reliable-as-normal
        # measurement (see this function's docstring).
        result["estimated"] = True
        result["reference_source"] = "sample_extrapolation_despite_stale_live_snapshot"
        result["live_onset_snapshot_failure_reason"] = brake_onset_snapshot_failure_reason
    return result


def samples_to_csv(path: str, samples: List[Sample]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t_sec", "x", "y", "yaw", "v_mps", "steering_rad"])
        for s in samples:
            writer.writerow([s.t_sec, s.x, s.y, s.yaw, s.v_mps, s.steering_rad])


def samples_from_csv(path: str) -> List[Sample]:
    with open(path) as f:
        reader = csv.DictReader(f)
        return [Sample(**{k: float(v) for k, v in row.items()}) for row in reader]


def write_results(path: str, results: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)


def build_identified_robot_config(
    base: RobotConfig,
    velocity_step_result: Optional[dict] = None,
    steering_step_result: Optional[dict] = None,
    circle_result: Optional[dict] = None,
    stop_result: Optional[dict] = None,
    commanded_steering_rad: float = 0.0,
) -> RobotConfig:
    """section P1-12: merge REAL trial measurements into a NEW RobotConfig
    (``base`` is never mutated) -- only the fields these 4 trial types can
    actually measure are updated; everything else (chassis dimensions,
    mass, steering limit, etc -- not identifiable from velocity-step/
    steering-step/circle/stop trials alone) is carried over unchanged from
    ``base``. Any missing/empty trial result dict leaves its corresponding
    field(s) at the base value.

    - ``speed_lag_tau_sec``: velocity-step response's own tau directly.
    - ``steering_rate_deg_s``: steering-step's commanded angular DELTA over
      its own tau -- an equivalent CONSTANT rate covering the step within
      ~1 tau (a conservative, simple characterization consistent with how
      this field is already used elsewhere as a RATE LIMIT, not a fitted
      second-order response).
    - ``brake_decel_mps2``: stop-test's initial speed / measured stopping
      distance, via v0^2 = 2*a*d (constant-deceleration kinematics).
    - ``wheelbase_m``: circle-test's measured turning radius and the
      trial's OWN commanded steering angle, via the standard bicycle-model
      relation turning_radius = wheelbase / tan(steering_angle).
    """
    velocity_step_result = velocity_step_result or {}
    steering_step_result = steering_step_result or {}
    circle_result = circle_result or {}
    stop_result = stop_result or {}

    speed_lag_tau_sec = velocity_step_result.get("tau_sec", base.speed_lag_tau_sec)

    steering_rate_deg_s = base.steering_rate_deg_s
    steering_tau = steering_step_result.get("tau_sec")
    steering_delta_rad = steering_step_result.get("steady_state_steering_rad")
    if steering_tau is not None and steering_tau > 1e-6 and steering_delta_rad is not None:
        steering_rate_deg_s = math.degrees(abs(steering_delta_rad)) / steering_tau

    brake_decel_mps2 = base.brake_decel_mps2
    stopping_distance_m = stop_result.get("stopping_distance_m")
    initial_speed_mps = stop_result.get("initial_speed_mps")
    if stopping_distance_m is not None and stopping_distance_m > 1e-6 and initial_speed_mps is not None:
        brake_decel_mps2 = (initial_speed_mps ** 2) / (2.0 * stopping_distance_m)

    wheelbase_m = base.wheelbase_m
    turning_radius_m = circle_result.get("turning_radius_m")
    if (turning_radius_m is not None and math.isfinite(turning_radius_m)
            and abs(commanded_steering_rad) > 1e-6):
        wheelbase_m = turning_radius_m * abs(math.tan(commanded_steering_rad))

    return dataclasses.replace(
        base, wheelbase_m=wheelbase_m, brake_decel_mps2=brake_decel_mps2,
        steering_rate_deg_s=steering_rate_deg_s, speed_lag_tau_sec=speed_lag_tau_sec,
    )


def write_identified_robot_yaml(path: str, config: RobotConfig) -> None:
    """Same shape as ``config/robot/hunter_se.yaml`` (a single top-level
    ``robot:`` key) so it can be dropped in as a ``robot_file`` override
    for any profile without further transformation."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump({"robot": dataclasses.asdict(config)}, f, sort_keys=False)
