import math

import pytest

from hunter_kinodynamic_rl.env.simulation.risk_telemetry import (
    _HEADER_LEN, CandidateTelemetry, InvalidReason, RiskTelemetry, decode, encode, invalid,
)


def test_invalid_telemetry_roundtrip():
    t = invalid(step_id=5)
    decoded = decode(encode(t))
    assert decoded.step_id == 5
    assert decoded.valid is False
    assert math.isnan(decoded.risk_target)


def test_valid_telemetry_with_candidates_roundtrip():
    t = RiskTelemetry(
        step_id=42, valid=True, risk_target=0.3, min_clearance_m=1.2, ttc_sec=2.5,
        collision_within_horizon=False, stopping_margin_m=0.5, unrecoverable=False,
        safer_alternative_margin=0.1, actor_candidate_index=0,
        emergency_stop=True, guarded_speed_mps=0.5, guarded_steering_rad=0.15,
        published_speed_mps=0.0, published_steering_rad=0.15,
        candidates=[
            CandidateTelemetry(kappa=0.1, v_ref=1.0, horizon_m=1.5, risk_score=0.3),
            CandidateTelemetry(kappa=-0.1, v_ref=1.0, horizon_m=1.5, risk_score=0.05),
        ],
    )
    decoded = decode(encode(t))
    assert decoded == t


def test_v8_raw_risk_progress_goal_and_reward_fields_roundtrip():
    t = RiskTelemetry(
        step_id=8, valid=True, risk_target=0.4, min_clearance_m=0.8, ttc_sec=1.2,
        collision_within_horizon=False, stopping_margin_m=0.2, unrecoverable=False,
        safer_alternative_margin=0.1, actor_candidate_index=0,
        steering_saturation=True, goal_progress_m=0.7, goal_x=4.0, goal_y=-2.0,
        reward_goal=0.0, reward_collision=0.0, reward_progress=0.2, reward_step=-0.01,
        reward_control_smoothness=-0.02, reward_trajectory_smoothness=-0.03,
        candidates=[CandidateTelemetry(0.1, 1.0, 1.5, 0.4, 0.7)],
    )
    decoded = decode(encode(t))
    assert decoded.steering_saturation is True
    assert decoded.goal_progress_m == pytest.approx(0.7)
    assert decoded.goal_x == pytest.approx(4.0)
    assert decoded.reward_trajectory_smoothness == pytest.approx(-0.03)
    assert decoded.candidates[0].goal_progress_m == pytest.approx(0.7)


def test_invalid_telemetry_carries_real_physical_command_fields():
    """Physical-command fields are ALWAYS meaningful, independent of
    risk-label validity (section P1-4)."""
    t = invalid(step_id=9, emergency_stop=True, guarded_speed_mps=0.0, guarded_steering_rad=0.2,
                published_speed_mps=0.0, published_steering_rad=0.2)
    decoded = decode(encode(t))
    assert decoded.emergency_stop is True
    assert decoded.guarded_steering_rad == pytest.approx(0.2)
    assert decoded.published_steering_rad == pytest.approx(0.2)
    assert decoded.valid is False


def test_guarded_and_published_commands_can_differ():
    """The core P0-5 regression: a command-latency queue can make the
    ACTUALLY-published command differ from the guarded (pre-latency) one
    -- both must round-trip independently, never collapsed into one
    field."""
    t = invalid(step_id=3, emergency_stop=False, guarded_speed_mps=1.5, guarded_steering_rad=0.1,
                published_speed_mps=0.0, published_steering_rad=0.0)
    decoded = decode(encode(t))
    assert decoded.guarded_speed_mps == pytest.approx(1.5)
    assert decoded.published_speed_mps == pytest.approx(0.0)
    assert decoded.guarded_speed_mps != decoded.published_speed_mps


def test_nominal_guarded_and_published_all_three_round_trip_independently():
    """Command pipeline has THREE distinct stages (nominal -> guarded ->
    published) -- a safety-guard stop (nominal!=guarded) combined with an
    active command-latency queue (guarded!=published) must round-trip all
    three as genuinely different values, never collapsing any pair."""
    t = invalid(
        step_id=4, emergency_stop=True,
        nominal_speed_mps=2.0, nominal_steering_rad=0.3,
        guarded_speed_mps=0.0, guarded_steering_rad=0.3,
        published_speed_mps=0.5, published_steering_rad=0.1,
    )
    decoded = decode(encode(t))
    assert decoded.nominal_speed_mps == pytest.approx(2.0)
    assert decoded.guarded_speed_mps == pytest.approx(0.0)
    assert decoded.published_speed_mps == pytest.approx(0.5)
    assert len({decoded.nominal_speed_mps, decoded.guarded_speed_mps, decoded.published_speed_mps}) == 3
    assert decoded.nominal_steering_rad == pytest.approx(0.3)
    assert decoded.guarded_steering_rad == pytest.approx(0.3)
    assert decoded.published_steering_rad == pytest.approx(0.1)


def test_nominal_fields_default_to_zero_when_unspecified():
    t = invalid(step_id=1)
    decoded = decode(encode(t))
    assert decoded.nominal_speed_mps == pytest.approx(0.0)
    assert decoded.nominal_steering_rad == pytest.approx(0.0)


def test_decode_rejects_wrong_schema_version():
    payload = encode(invalid(0))
    payload[0] = 999.0
    with pytest.raises(ValueError):
        decode(payload)


def test_decode_rejects_truncated_payload():
    with pytest.raises(ValueError):
        decode([1.0, 2.0])


def test_decode_rejects_length_mismatch_for_declared_candidate_count():
    payload = encode(invalid(0))
    payload[_HEADER_LEN - 1] = 3.0  # num_candidates field: claims 3 candidates but none are appended
    with pytest.raises(ValueError):
        decode(payload)


# ------------------------------------------------------------ P0-9: sensor_stale
def test_sensor_stale_flag_roundtrips():
    t = invalid(step_id=7, sensor_stale=True)
    decoded = decode(encode(t))
    assert decoded.sensor_stale is True


def test_sensor_stale_defaults_to_false():
    t = invalid(step_id=7)
    decoded = decode(encode(t))
    assert decoded.sensor_stale is False


def test_sensor_stale_independent_of_valid():
    """A stale-sensor step can occur whether or not the risk framework
    itself is enabled (section P0-9) -- the flag must round-trip
    regardless of `valid`."""
    t = RiskTelemetry(
        step_id=11, valid=True, risk_target=0.1, min_clearance_m=2.0, ttc_sec=3.0,
        collision_within_horizon=False, stopping_margin_m=1.0, unrecoverable=False,
        safer_alternative_margin=0.0, actor_candidate_index=0, sensor_stale=True,
    )
    decoded = decode(encode(t))
    assert decoded.valid is True
    assert decoded.sensor_stale is True


# ---------------------------------------------------------------- P0-5: reset_generation / episode_id / sim_timestamp_sec / invalid_reason
def test_reset_generation_and_episode_id_round_trip_independently_of_step_id():
    """The core P0-5 staleness-detection property: a telemetry message from
    a PREVIOUS episode's step_id=K must be distinguishable from a NEW
    episode's own step_id=K message via reset_generation, which step_id
    alone (resets to 0 every episode) cannot provide."""
    t = invalid(step_id=3, reset_generation=7, episode_id=20431)
    decoded = decode(encode(t))
    assert decoded.reset_generation == 7
    assert decoded.episode_id == 20431
    assert decoded.step_id == 3


def test_sim_timestamp_sec_round_trips():
    t = invalid(step_id=1, sim_timestamp_sec=123.456)
    decoded = decode(encode(t))
    assert decoded.sim_timestamp_sec == pytest.approx(123.456)


def test_sim_timestamp_sec_nan_when_clock_unavailable():
    """Default (no /clock message delivered yet) is an explicit NaN, never
    a misleading 0.0 that looks like "sim time zero" (matches this
    module's existing NaN-sentinel convention for other unavailable
    fields)."""
    t = invalid(step_id=1)
    decoded = decode(encode(t))
    assert math.isnan(decoded.sim_timestamp_sec)


def test_invalid_reason_defaults_to_none():
    t = invalid(step_id=1)
    decoded = decode(encode(t))
    assert decoded.invalid_reason == int(InvalidReason.NONE)


def test_invalid_reason_round_trips_each_enum_member():
    for reason in InvalidReason:
        t = invalid(step_id=1, reason=reason)
        decoded = decode(encode(t))
        assert decoded.invalid_reason == int(reason)


def test_valid_telemetry_reports_invalid_reason_none():
    """A genuinely valid risk label must report NO invalid reason -- proves
    this field doesn't just default-leak a stale/wrong code on the success
    path."""
    t = RiskTelemetry(
        step_id=1, valid=True, risk_target=0.1, min_clearance_m=2.0, ttc_sec=3.0,
        collision_within_horizon=False, stopping_margin_m=1.0, unrecoverable=False,
        safer_alternative_margin=0.0, actor_candidate_index=0,
        invalid_reason=int(InvalidReason.NONE),
    )
    decoded = decode(encode(t))
    assert decoded.valid is True
    assert decoded.invalid_reason == int(InvalidReason.NONE)


def test_reset_marker_round_trips_step_id_zero_and_reset_generation():
    """code review (risk-telemetry correctness bug): the RESET_MARKER
    message environment_node.py now publishes at the end of every /reset
    (see risk_telemetry.py's module docstring) -- step_id=0 (never
    collides with a real per-step message, which start at 1) carrying the
    AUTHORITATIVE reset_generation a fresh EnvironmentClient learns from."""
    t = invalid(step_id=0, reset_generation=17, episode_id=4242, reason=InvalidReason.RESET_MARKER)
    decoded = decode(encode(t))
    assert decoded.step_id == 0
    assert decoded.valid is False
    assert decoded.reset_generation == 17
    assert decoded.episode_id == 4242
    assert decoded.invalid_reason == int(InvalidReason.RESET_MARKER)
