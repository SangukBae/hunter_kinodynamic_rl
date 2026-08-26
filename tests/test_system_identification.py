import math
import os

import pytest
import yaml

from hunter_kinodynamic_rl.config.schema import RobotConfig
from hunter_kinodynamic_rl.dynamics.system_identification import (
    Sample, analyze_circle_test, analyze_steering_step_response,
    analyze_stop_test, analyze_velocity_step_response, build_identified_robot_config,
    samples_from_csv, samples_to_csv, write_identified_robot_yaml,
)


def _base_robot() -> RobotConfig:
    return RobotConfig(
        name="hunter_se", wheelbase_m=0.547696, track_width_m=0.503404,
        wheel_radius_m=0.136, length_m=0.76, width_m=0.4, height_m=0.14,
        mass_kg=42.0, steering_limit_deg=21.58, max_forward_speed_mps=2.0,
        min_forward_speed_mps=0.0, accel_limit_mps2=6.0, brake_decel_mps2=6.0,
        steering_rate_deg_s=200.0, speed_lag_tau_sec=0.05,
    )


def test_velocity_step_response_tau():
    # v(t) = v_target * (1 - exp(-t/tau)), tau=0.5
    tau_true = 0.5
    samples = [Sample(t_sec=t, x=0, y=0, yaw=0, v_mps=2.0 * (1 - math.exp(-t / tau_true)), steering_rad=0.0)
               for t in [i * 0.05 for i in range(60)]]
    result = analyze_velocity_step_response(samples, target_v_mps=2.0)
    assert result["tau_sec"] == pytest.approx(tau_true, abs=0.1)


def test_steering_step_response_tau():
    tau_true = 0.2
    samples = [Sample(t_sec=t, x=0, y=0, yaw=0, v_mps=0.0,
                       steering_rad=0.3 * (1 - math.exp(-t / tau_true)))
               for t in [i * 0.02 for i in range(60)]]
    result = analyze_steering_step_response(samples, target_steering_rad=0.3)
    assert result["tau_sec"] == pytest.approx(tau_true, abs=0.05)


def test_circle_test_recovers_known_radius():
    radius = 2.5
    samples = [
        Sample(t_sec=i * 0.1, x=radius * math.cos(theta), y=radius * math.sin(theta),
               yaw=theta + math.pi / 2, v_mps=1.0, steering_rad=0.2)
        for i, theta in enumerate([k * 0.1 for k in range(60)])
    ]
    result = analyze_circle_test(samples)
    assert result["turning_radius_m"] == pytest.approx(radius, rel=0.02)


def test_stop_test_measures_distance_and_time():
    samples = [
        Sample(t_sec=0.0, x=0.0, y=0.0, yaw=0.0, v_mps=2.0, steering_rad=0.0),
        Sample(t_sec=0.1, x=0.19, y=0.0, yaw=0.0, v_mps=1.4, steering_rad=0.0),
        Sample(t_sec=0.2, x=0.33, y=0.0, yaw=0.0, v_mps=0.6, steering_rad=0.0),
        Sample(t_sec=0.3, x=0.38, y=0.0, yaw=0.0, v_mps=0.0, steering_rad=0.0),
    ]
    result = analyze_stop_test(samples)
    assert result["initial_speed_mps"] == pytest.approx(2.0)
    assert result["stopping_distance_m"] == pytest.approx(0.38)
    assert result["stopping_time_sec"] == pytest.approx(0.3)


def test_stop_test_first_post_onset_sample_already_decelerating_still_reaches_steady_state():
    """item-4 regression: the ORIGINAL brake_onset_t_sec fix anchored the
    trial's "start" reference at the first sample with t_sec >=
    brake_onset_t_sec -- if THAT sample already shows some deceleration
    (odometry sampled at discrete intervals, arriving up to one sample
    period after the true onset instant), a genuinely-normal trial (steady
    2.0 m/s WAS reached well before braking) got misclassified as
    steady_state_not_reached purely because of which sample happened to be
    picked as the reference, not because steady state was actually never
    reached."""
    samples = [
        Sample(t_sec=0.0, x=0.0, y=0.0, yaw=0.0, v_mps=0.0, steering_rad=0.0),
        Sample(t_sec=0.5, x=0.6, y=0.0, yaw=0.0, v_mps=2.0, steering_rad=0.0),
        Sample(t_sec=1.0, x=1.6, y=0.0, yaw=0.0, v_mps=2.0, steering_rad=0.0),
        # Last pre-onset sample: still solidly at steady state.
        Sample(t_sec=1.45, x=2.5, y=0.0, yaw=0.0, v_mps=2.0, steering_rad=0.0),
        # brake_onset_t_sec=1.5 falls BETWEEN this sample and the next --
        # the first POST-onset sample below already shows real
        # deceleration (0.3 m/s drop in one 0.1s sample period), which
        # would exceed steady_state_tolerance_mps (0.08) if it were
        # (incorrectly) used as the reference point.
        Sample(t_sec=1.55, x=2.69, y=0.0, yaw=0.0, v_mps=1.7, steering_rad=0.0),
        Sample(t_sec=1.65, x=2.83, y=0.0, yaw=0.0, v_mps=1.0, steering_rad=0.0),
        Sample(t_sec=1.75, x=2.90, y=0.0, yaw=0.0, v_mps=0.3, steering_rad=0.0),
        Sample(t_sec=1.85, x=2.92, y=0.0, yaw=0.0, v_mps=0.0, steering_rad=0.0),
    ]
    result = analyze_stop_test(samples, brake_onset_t_sec=1.5, initial_speed_target_mps=2.0)
    assert result["valid"] is True
    assert result["reason"] is None
    # Anchored at t=1.5 EXACTLY (extrapolated forward from the last
    # pre-onset sample at t=1.45, x=2.5, at its own steady v=2.0 m/s: x(1.5)
    # = 2.5 + 2.0*0.05 = 2.6) -- never the first post-onset sample (t=1.55,
    # v=1.7), so the reported "initial speed" is the genuine pre-brake
    # steady-state value, AND the 1.45->1.5 pre-braking segment (0.1 m) is
    # correctly excluded from stopping_distance_m/stopping_time_sec (item-4
    # round-2 fix).
    assert result["initial_speed_mps"] == pytest.approx(2.0)
    assert result["stopping_distance_m"] == pytest.approx(2.92 - 2.6)
    assert result["stopping_time_sec"] == pytest.approx(1.85 - 1.5)


def test_stop_test_brake_onset_with_no_pre_onset_samples_falls_back_to_first_post_onset_sample():
    """If the recording genuinely starts at/after the brake command (no
    pre-onset odometry exists at all), there is no genuinely pre-brake
    reference point to anchor on -- falls back to the first available
    (post-onset) sample rather than fabricating one, same as if no
    brake_onset_t_sec had been trimmed away anything."""
    samples = [
        Sample(t_sec=2.0, x=0.0, y=0.0, yaw=0.0, v_mps=2.0, steering_rad=0.0),
        Sample(t_sec=2.1, x=0.19, y=0.0, yaw=0.0, v_mps=0.0, steering_rad=0.0),
    ]
    result = analyze_stop_test(samples, brake_onset_t_sec=1.5, initial_speed_target_mps=2.0)
    assert result["valid"] is True
    assert result["initial_speed_mps"] == pytest.approx(2.0)
    assert result["stopping_distance_m"] == pytest.approx(0.19)
    assert result["stopping_time_sec"] == pytest.approx(0.1)


def test_stop_test_odometry_gap_and_timeout_still_detected_with_brake_onset_anchoring():
    """The item-4 anchoring change must not weaken the pre-existing
    odometry-dropout / never-stopped checks."""
    gap_samples = [
        Sample(t_sec=1.4, x=0.0, y=0.0, yaw=0.0, v_mps=2.0, steering_rad=0.0),
        Sample(t_sec=1.6, x=0.3, y=0.0, yaw=0.0, v_mps=1.9, steering_rad=0.0),
        Sample(t_sec=3.0, x=0.4, y=0.0, yaw=0.0, v_mps=0.0, steering_rad=0.0),  # >0.5s gap
    ]
    gap_result = analyze_stop_test(gap_samples, brake_onset_t_sec=1.5, initial_speed_target_mps=2.0)
    assert gap_result == {
        "valid": False, "reason": "odometry_dropout", "gap_sec": pytest.approx(1.4), "gap_start_t_sec": 1.6,
    }

    never_stops = [
        Sample(t_sec=1.4, x=0.0, y=0.0, yaw=0.0, v_mps=2.0, steering_rad=0.0),
        Sample(t_sec=1.6, x=0.3, y=0.0, yaw=0.0, v_mps=1.9, steering_rad=0.0),
        Sample(t_sec=1.7, x=0.5, y=0.0, yaw=0.0, v_mps=1.8, steering_rad=0.0),
    ]
    timeout_result = analyze_stop_test(never_stops, brake_onset_t_sec=1.5, initial_speed_target_mps=2.0)
    assert timeout_result["valid"] is False
    assert timeout_result["reason"] == "did_not_stop_within_window"


def test_stop_test_end_to_end_through_a_recorder_style_continuous_sample_stream():
    """Integration-style regression (item-4): feeds a FULL
    acceleration-then-braking sample stream shaped exactly like
    SystemIdRecorder.run_trial's "stop" trial produces -- continuous
    odometry recorded from t=0 (v=0) through an exponential ramp to steady
    speed, held, then an exponential decel to a full stop after
    brake_onset_t_sec -- through analyze_stop_test end-to-end, at a fixed
    sample period that does NOT align with brake_onset_t_sec (replicating
    the real recorder's odometry-callback-driven sampling, which has no
    reason to land exactly on the moment run_trial() stamps
    _stop_trial_brake_onset_t_sec)."""
    dt = 0.073  # deliberately NOT a divisor of brake_onset_t_sec
    target_v = 1.5
    accel_tau = 0.15
    brake_onset_t_sec = 2.0
    decel_tau = 0.2
    total_duration = 4.0

    samples = []
    t = 0.0
    x = 0.0
    while t <= total_duration:
        if t < brake_onset_t_sec:
            v = target_v * (1.0 - math.exp(-t / accel_tau))
        else:
            v = target_v * math.exp(-(t - brake_onset_t_sec) / decel_tau)
        if v < 1e-4:
            v = 0.0
        samples.append(Sample(t_sec=t, x=x, y=0.0, yaw=0.0, v_mps=v, steering_rad=0.0))
        x += v * dt
        t += dt

    result = analyze_stop_test(
        samples, brake_onset_t_sec=brake_onset_t_sec, initial_speed_target_mps=target_v,
        stop_speed_threshold_mps=0.02, stop_hysteresis_samples=3,
    )
    assert result["valid"] is True, result
    assert result["reason"] is None
    # Reported initial speed is the genuine pre-brake steady-state value
    # (the exponential ramp has essentially converged by t=2.0, well past
    # accel_tau=0.15), not whatever partial-decel value the nearest
    # post-onset odometry sample happened to land on.
    assert result["initial_speed_mps"] == pytest.approx(target_v, abs=0.01)
    assert result["stopping_distance_m"] > 0.0
    assert result["stopping_time_sec"] > 0.0


# --------------------------------------------------- item-4 (round 2)
def test_stop_test_excludes_the_pre_braking_gap_between_last_pre_onset_sample_and_true_onset():
    """The EXACT scenario named in the round-2 governing instruction:
    onset=1.5, last pre-onset sample at t=1.1 -- the 0.4s (still at steady
    2.0 m/s) segment between them must NOT be counted as part of the
    stopping distance/time."""
    samples = [
        Sample(t_sec=1.0, x=2.0, y=0.0, yaw=0.0, v_mps=2.0, steering_rad=0.0),
        # Last pre-onset sample: steady state, 0.4s before the true onset.
        Sample(t_sec=1.1, x=2.2, y=0.0, yaw=0.0, v_mps=2.0, steering_rad=0.0),
        Sample(t_sec=1.6, x=3.0, y=0.0, yaw=0.0, v_mps=0.5, steering_rad=0.0),
        Sample(t_sec=1.8, x=3.2, y=0.0, yaw=0.0, v_mps=0.0, steering_rad=0.0),
    ]
    result = analyze_stop_test(samples, brake_onset_t_sec=1.5, initial_speed_target_mps=2.0)
    assert result["valid"] is True, result
    # start is extrapolated to t=1.5 at v=2.0 from (t=1.1, x=2.2):
    # x(1.5) = 2.2 + 2.0*(1.5-1.1) = 3.0
    assert result["initial_speed_mps"] == pytest.approx(2.0)
    # The 1.1->1.5 pre-braking segment (0.8 m at 2.0 m/s) must be EXCLUDED:
    # a buggy anchor at t=1.1 (x=2.2) would report distance=3.2-2.2=1.0 and
    # time=1.8-1.1=0.7 -- both measurably larger than the correct values.
    assert result["stopping_distance_m"] == pytest.approx(3.2 - 3.0)
    assert result["stopping_time_sec"] == pytest.approx(1.8 - 1.5)


def test_stop_test_rejects_an_onset_reference_gap_that_is_too_large_to_extrapolate():
    samples = [
        Sample(t_sec=0.0, x=0.0, y=0.0, yaw=0.0, v_mps=2.0, steering_rad=0.0),  # 1.5s before onset -- too stale
        Sample(t_sec=1.6, x=3.0, y=0.0, yaw=0.0, v_mps=0.5, steering_rad=0.0),
        Sample(t_sec=1.8, x=3.2, y=0.0, yaw=0.0, v_mps=0.0, steering_rad=0.0),
    ]
    result = analyze_stop_test(samples, brake_onset_t_sec=1.5, initial_speed_target_mps=2.0,
                                max_sample_gap_sec=0.5)
    assert result["valid"] is False
    assert result["reason"] == "onset_reference_gap_too_large"
    assert result["gap_sec"] == pytest.approx(1.5)


def test_stop_test_prefers_the_live_brake_onset_state_snapshot_over_extrapolation():
    """When the recorder's own live snapshot (brake_onset_state) is
    available, it is used AS-IS -- no extrapolation needed, and any
    brake_onset_t_sec passed alongside it is ignored (brake_onset_state's
    own t_sec is authoritative)."""
    onset_state = Sample(t_sec=1.5, x=3.0, y=0.0, yaw=0.0, v_mps=2.0, steering_rad=0.0)
    samples = [
        Sample(t_sec=1.1, x=2.2, y=0.0, yaw=0.0, v_mps=2.0, steering_rad=0.0),  # ignored -- not the anchor
        Sample(t_sec=1.6, x=3.0, y=0.0, yaw=0.0, v_mps=0.5, steering_rad=0.0),
        Sample(t_sec=1.8, x=3.2, y=0.0, yaw=0.0, v_mps=0.0, steering_rad=0.0),
    ]
    result = analyze_stop_test(samples, brake_onset_state=onset_state, initial_speed_target_mps=2.0)
    assert result["valid"] is True, result
    assert result["initial_speed_mps"] == pytest.approx(2.0)
    assert result["stopping_distance_m"] == pytest.approx(3.2 - 3.0)
    assert result["stopping_time_sec"] == pytest.approx(1.8 - 1.5)


def test_stop_test_brake_onset_state_with_no_post_onset_samples_is_invalid():
    onset_state = Sample(t_sec=5.0, x=0.0, y=0.0, yaw=0.0, v_mps=2.0, steering_rad=0.0)
    samples = [Sample(t_sec=1.0, x=0.0, y=0.0, yaw=0.0, v_mps=2.0, steering_rad=0.0)]
    result = analyze_stop_test(samples, brake_onset_state=onset_state, initial_speed_target_mps=2.0)
    assert result["valid"] is False
    assert result["reason"] == "no_samples_after_brake_onset"


# --------------------------------------------------------------- item-2 (system-ID stale-data fix)
def _dense_sample_stream_around_onset(onset_t_sec: float = 1.0) -> list:
    """A sample stream dense enough that the OLD (buggy) sample-based
    fallback would happily recover a valid=True result -- used to prove the
    fix actually closes the loophole, not just that a NEW code path exists
    that happens to look right."""
    samples = []
    t = 0.0
    while t <= onset_t_sec:
        samples.append(Sample(t_sec=t, x=t * 2.0, y=0.0, yaw=0.0, v_mps=2.0, steering_rad=0.0))
        t += 0.05
    t = onset_t_sec + 0.05
    v = 2.0
    x = samples[-1].x
    while t <= onset_t_sec + 2.0:
        v = max(0.0, v - 0.2)
        x += v * 0.05
        samples.append(Sample(t_sec=t, x=x, y=0.0, yaw=0.0, v_mps=v, steering_rad=0.0))
        t += 0.05
    return samples


def test_stop_test_stale_live_snapshot_failure_is_not_masked_by_the_sample_fallback():
    """The core item-2 regression: a live caller (system_id_node.py) whose
    OWN preferred live-snapshot mechanism (build_brake_onset_snapshot)
    explicitly failed (e.g. odometry_stale_at_brake_onset) must not have
    that failure silently overridden by analyze_stop_test's own, strictly
    LOOSER, sample-based brake_onset_t_sec extrapolation fallback -- even
    when the sample stream is dense enough that fallback WOULD have
    recovered a valid=True result (see test just above, the pre-fix repro)."""
    samples = _dense_sample_stream_around_onset(onset_t_sec=1.0)
    result = analyze_stop_test(
        samples, brake_onset_t_sec=1.0, brake_onset_state=None,
        brake_onset_snapshot_failure_reason="odometry_stale_at_brake_onset",
        initial_speed_target_mps=2.0,
    )
    assert result["valid"] is False
    assert result["reason"] == "odometry_stale_at_brake_onset"


def test_stop_test_live_snapshot_failure_reason_survives_when_no_samples_were_recorded_either():
    """code review (system-ID error-origin ordering fix): the exact
    counterexample this fix responds to -- a live run that received NO
    odometry/joint-state at all naturally has BOTH an empty ``samples``
    list AND a failed live brake-onset snapshot. Before the fix, the
    generic ``not samples`` check ran first and reported ``"no_samples"``,
    discarding the real, actionable reason
    (``build_brake_onset_snapshot``'s own diagnosis, e.g.
    ``"odometry_stale_at_brake_onset"``). The result must stay
    ``valid=False`` either way -- what changes is WHICH reason string
    survives."""
    result = analyze_stop_test(
        [], brake_onset_t_sec=1.0, brake_onset_state=None,
        brake_onset_snapshot_failure_reason="odometry_stale_at_brake_onset",
        initial_speed_target_mps=2.0,
    )
    assert result["valid"] is False
    assert result["reason"] == "odometry_stale_at_brake_onset"


def test_stop_test_opt_in_fallback_despite_stale_snapshot_is_tagged_estimated():
    """The documented escape hatch: a caller that explicitly opts back into
    the sample-based fallback despite a failed live snapshot still gets a
    result (bounded by the same max_sample_gap_sec as everywhere else), but
    it must be clearly tagged as an estimate from a degraded path, never
    indistinguishable from an ordinary, fully-trusted measurement."""
    samples = _dense_sample_stream_around_onset(onset_t_sec=1.0)
    result = analyze_stop_test(
        samples, brake_onset_t_sec=1.0, brake_onset_state=None,
        brake_onset_snapshot_failure_reason="odometry_stale_at_brake_onset",
        allow_fallback_despite_live_snapshot_failure=True,
        initial_speed_target_mps=2.0,
    )
    assert result["valid"] is True, result
    assert result["estimated"] is True
    assert result["reference_source"] == "sample_extrapolation_despite_stale_live_snapshot"
    assert result["live_onset_snapshot_failure_reason"] == "odometry_stale_at_brake_onset"


def test_stop_test_offline_csv_style_call_without_a_failure_reason_still_uses_the_fallback():
    """Backward compatibility / the explicit offline-CSV allowance: a caller
    that never had a live snapshot mechanism to begin with (no
    brake_onset_snapshot_failure_reason passed at all -- e.g. reprocessing a
    plain CSV via samples_from_csv) is UNAFFECTED by this fix and keeps
    using the sample-based fallback exactly as before."""
    samples = _dense_sample_stream_around_onset(onset_t_sec=1.0)
    result = analyze_stop_test(samples, brake_onset_t_sec=1.0, initial_speed_target_mps=2.0)
    assert result["valid"] is True, result
    assert "estimated" not in result
    assert "reference_source" not in result


def test_csv_roundtrip(tmp_path):
    samples = [Sample(t_sec=0.0, x=1.0, y=2.0, yaw=0.1, v_mps=1.5, steering_rad=0.05)]
    path = os.path.join(str(tmp_path), "trial.csv")
    samples_to_csv(path, samples)
    restored = samples_from_csv(path)
    assert restored == samples


# --------------------------------------------------- P1-12: identified robot config
def test_build_identified_robot_config_updates_speed_lag_tau():
    base = _base_robot()
    identified = build_identified_robot_config(base, velocity_step_result={"tau_sec": 0.35})
    assert identified.speed_lag_tau_sec == pytest.approx(0.35)
    # every other field carried over UNCHANGED from base
    assert identified.mass_kg == base.mass_kg
    assert identified.wheelbase_m == base.wheelbase_m


def test_build_identified_robot_config_updates_steering_rate():
    base = _base_robot()
    # commanded 0.3 rad step reached in 0.2s tau -> equivalent rate.
    identified = build_identified_robot_config(
        base, steering_step_result={"tau_sec": 0.2, "steady_state_steering_rad": 0.3})
    assert identified.steering_rate_deg_s == pytest.approx(math.degrees(0.3) / 0.2)


def test_build_identified_robot_config_updates_brake_decel_from_stop_test():
    base = _base_robot()
    # v0=2.0 m/s, stopped over 0.5m -> a = v0^2/(2*d) = 4/1 = 4.0 m/s^2.
    identified = build_identified_robot_config(
        base, stop_result={"initial_speed_mps": 2.0, "stopping_distance_m": 0.5})
    assert identified.brake_decel_mps2 == pytest.approx(4.0)


def test_build_identified_robot_config_updates_wheelbase_from_circle_test():
    base = _base_robot()
    commanded_steering = math.radians(15.0)
    turning_radius = base.wheelbase_m / math.tan(commanded_steering)  # what base WOULD produce
    identified = build_identified_robot_config(
        base, circle_result={"turning_radius_m": turning_radius}, commanded_steering_rad=commanded_steering)
    assert identified.wheelbase_m == pytest.approx(base.wheelbase_m, rel=1e-6)


def test_build_identified_robot_config_ignores_missing_trial_results():
    base = _base_robot()
    identified = build_identified_robot_config(base)  # every trial omitted
    assert identified == base


def test_build_identified_robot_config_ignores_infinite_turning_radius():
    base = _base_robot()
    identified = build_identified_robot_config(
        base, circle_result={"turning_radius_m": math.inf}, commanded_steering_rad=0.2)
    assert identified.wheelbase_m == base.wheelbase_m


def test_write_identified_robot_yaml_round_trips_through_the_loader_shape(tmp_path):
    base = _base_robot()
    identified = build_identified_robot_config(base, velocity_step_result={"tau_sec": 0.4})
    path = os.path.join(str(tmp_path), "hunter_se_identified.yaml")
    write_identified_robot_yaml(path, identified)

    with open(path) as f:
        data = yaml.safe_load(f)
    assert set(data.keys()) == {"robot"}  # same single-top-level-key shape config/robot/*.yaml uses
    assert data["robot"]["speed_lag_tau_sec"] == pytest.approx(0.4)
    assert data["robot"]["wheelbase_m"] == pytest.approx(base.wheelbase_m)
