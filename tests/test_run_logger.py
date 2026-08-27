"""Regression tests for run_logger.py::RunLogger.log_step (section P1-11):
JSON-safe NaN->null conversion, and the newly-added pose/trajectory/
actual_dt_sec/guarded-published-command/candidates fields.

run_logger.py has no rclpy dependency (pure Python + config/risk_telemetry,
both ROS-free) -- host-testable directly.
"""

import json

import pytest

from hunter_kinodynamic_rl.config.loader import load_profile
from hunter_kinodynamic_rl.env.randomization.domain_randomizer import RandomizationDraw
from hunter_kinodynamic_rl.env.simulation import risk_telemetry as rt
from hunter_kinodynamic_rl.env.simulation import sensor_diagnostics as sd
from hunter_kinodynamic_rl.training.run_logger import RunLogger, run_directory
from hunter_kinodynamic_rl.trajectory.action_space import TrajectoryCommand


def _make_logger(tmp_path):
    profile = load_profile("smoke_test")
    run_dir = run_directory(profile, str(tmp_path))
    return RunLogger(run_dir, profile), run_dir


def _read_last_step_record(run_dir):
    path = f"{run_dir}/logs/steps.jsonl"
    with open(path) as f:
        lines = f.readlines()
    return json.loads(lines[-1])


def test_invalid_telemetry_emits_null_not_nan_in_json(tmp_path):
    """The core P1-11 regression: risk fields that are NaN in-memory
    (rt.invalid()'s "no label this step" convention) must serialize as
    JSON `null`, never the non-standard `NaN` token most JSON parsers
    reject."""
    logger, run_dir = _make_logger(tmp_path)
    telemetry = rt.invalid(step_id=1)
    logger.log_step(1, 1, [0.0, 0.5, 0.5], reward=-0.1, telemetry=telemetry, collision=False, target=False)

    raw_text = open(f"{run_dir}/logs/steps.jsonl").read()
    assert "NaN" not in raw_text

    record = _read_last_step_record(run_dir)
    assert record["risk_target"] is None
    assert record["min_clearance_m"] is None
    assert record["ttc_sec"] is None
    assert record["stopping_margin_m"] is None
    assert record["safer_alternative_margin"] is None
    assert record["risk_valid"] is False


def test_valid_telemetry_reports_real_numbers_not_null(tmp_path):
    logger, run_dir = _make_logger(tmp_path)
    telemetry = rt.RiskTelemetry(
        step_id=2, valid=True, risk_target=0.3, min_clearance_m=1.2, ttc_sec=2.5,
        collision_within_horizon=False, stopping_margin_m=0.5, unrecoverable=False,
        safer_alternative_margin=0.1, actor_candidate_index=0,
        candidates=[rt.CandidateTelemetry(kappa=0.1, v_ref=1.0, horizon_m=1.5, risk_score=0.3)],
    )
    logger.log_step(2, 1, [0.0, 0.5, 0.5], reward=0.0, telemetry=telemetry, collision=False, target=False)

    record = _read_last_step_record(run_dir)
    assert record["risk_target"] == pytest.approx(0.3)
    assert record["min_clearance_m"] == pytest.approx(1.2)
    assert record["num_candidates"] == 1
    assert record["candidates"] == [{
        "kappa": 0.1, "v_ref": 1.0, "horizon_m": 1.5,
        "risk_score": 0.3, "goal_progress_m": 0.0,
    }]


# --------------------------------------------------------- requirement 5
def test_sensor_diagnostics_omitted_when_not_provided(tmp_path):
    logger, run_dir = _make_logger(tmp_path)
    logger.log_step(1, 1, [0.0, 0.5, 0.5], reward=-0.1, telemetry=rt.invalid(step_id=1),
                     collision=False, target=False)
    record = _read_last_step_record(run_dir)
    assert "sensor_diagnostics" not in record


def test_sensor_diagnostics_gt_and_noisy_are_logged_separately(tmp_path):
    logger, run_dir = _make_logger(tmp_path)
    diag = sd.SensorDiagnostics(
        step_id=1, valid=True, reset_generation=1, episode_id=42, sim_timestamp_sec=1.0,
        gt_x=1.0, gt_y=2.0, gt_yaw=0.1, noisy_x=1.05, noisy_y=1.9, noisy_yaw=0.12,
        gt_v_mps=0.5, gt_yaw_rate_radps=0.0, gt_steering_rad=0.0,
        noisy_v_mps=0.55, noisy_yaw_rate_radps=0.01, noisy_steering_rad=0.02,
        drift_x_m=0.05, drift_y_m=-0.1, drift_yaw_rad=0.02,
        localization_latency_steps=2, lidar_beam_count=80, lidar_dropout_count=1,
        lidar_perturbation_mean_m=0.03, lidar_perturbation_max_m=0.2,
    )
    logger.log_step(1, 1, [0.0, 0.5, 0.5], reward=0.0, telemetry=rt.invalid(step_id=1),
                     collision=False, target=False, sensor_diagnostics=diag)

    record = _read_last_step_record(run_dir)
    sd_record = record["sensor_diagnostics"]
    assert sd_record["valid"] is True
    assert sd_record["gt_pose"] == {"x": 1.0, "y": 2.0, "yaw": 0.1}
    assert sd_record["noisy_pose"] == {"x": 1.05, "y": 1.9, "yaw": 0.12}
    assert sd_record["gt_velocity_mps"] == pytest.approx(0.5)
    assert sd_record["noisy_velocity_mps"] == pytest.approx(0.55)
    assert sd_record["localization_drift"] == {"x_m": 0.05, "y_m": -0.1, "yaw_rad": 0.02}
    assert sd_record["localization_latency_steps"] == 2
    assert sd_record["lidar_beam_count"] == 80
    assert sd_record["lidar_dropout_count"] == 1
    assert sd_record["lidar_perturbation_mean_m"] == pytest.approx(0.03)


def test_sensor_diagnostics_timeout_emits_null_gt_noisy_fields_with_reason(tmp_path):
    """The item-5 requirement: on a poll timeout, GT/noisy fields must be
    null (never a stale/fabricated value), with invalid_reason recording
    WHY -- mirrors risk_telemetry's own NaN->null convention."""
    logger, run_dir = _make_logger(tmp_path)
    diag = sd.invalid(step_id=3, reset_generation=1, episode_id=42,
                       reason=sd.DiagnosticsInvalidReason.POLL_TIMEOUT)
    logger.log_step(3, 1, [0.0, 0.5, 0.5], reward=0.0, telemetry=rt.invalid(step_id=3),
                     collision=False, target=False, sensor_diagnostics=diag)

    record = _read_last_step_record(run_dir)
    sd_record = record["sensor_diagnostics"]
    assert sd_record["valid"] is False
    assert sd_record["invalid_reason"] == int(sd.DiagnosticsInvalidReason.POLL_TIMEOUT)
    assert sd_record["gt_pose"] == {"x": None, "y": None, "yaw": None}
    assert sd_record["noisy_pose"] == {"x": None, "y": None, "yaw": None}
    assert sd_record["gt_velocity_mps"] is None
    assert sd_record["noisy_velocity_mps"] is None
    raw_text = open(f"{run_dir}/logs/steps.jsonl").read()
    assert "NaN" not in raw_text


def test_sensor_diagnostics_invalid_nulls_every_measurement_field_never_zero(tmp_path):
    """Full item-5 regression: sensor_diagnostics.invalid() defaults
    drift_*/localization_latency_steps/lidar_beam_count/
    lidar_dropout_count/lidar_perturbation_{mean,max}_m to a concrete
    0/0.0 on the dataclass -- previously that leaked straight into the
    JSONL record, indistinguishable from "the measurement really was
    zero". Every one of those fields must be null here instead, exactly
    like the pose/velocity fields already were."""
    logger, run_dir = _make_logger(tmp_path)
    diag = sd.invalid(step_id=5, reset_generation=1, episode_id=42, sim_timestamp_sec=12.5,
                       reason=sd.DiagnosticsInvalidReason.COMPUTATION_EXCEPTION)
    logger.log_step(5, 1, [0.0, 0.5, 0.5], reward=0.0, telemetry=rt.invalid(step_id=5),
                     collision=False, target=False, sensor_diagnostics=diag)

    record = _read_last_step_record(run_dir)
    sd_record = record["sensor_diagnostics"]
    assert sd_record["valid"] is False
    assert sd_record["localization_drift"] == {"x_m": None, "y_m": None, "yaw_rad": None}
    assert sd_record["localization_latency_steps"] is None
    assert sd_record["lidar_beam_count"] is None
    assert sd_record["lidar_dropout_count"] is None
    assert sd_record["lidar_perturbation_mean_m"] is None
    assert sd_record["lidar_perturbation_max_m"] is None
    # Traceability fields are always present, valid or not.
    assert sd_record["schema_version"] == sd.SCHEMA_VERSION
    assert sd_record["episode_id"] == 42
    assert sd_record["sim_timestamp_sec"] == pytest.approx(12.5)


def test_sensor_diagnostics_valid_record_includes_traceability_fields(tmp_path):
    logger, run_dir = _make_logger(tmp_path)
    diag = sd.SensorDiagnostics(
        step_id=1, valid=True, reset_generation=1, episode_id=42, sim_timestamp_sec=1.0,
        gt_x=1.0, gt_y=2.0, gt_yaw=0.1, noisy_x=1.05, noisy_y=1.9, noisy_yaw=0.12,
        gt_v_mps=0.5, gt_yaw_rate_radps=0.0, gt_steering_rad=0.0,
        noisy_v_mps=0.55, noisy_yaw_rate_radps=0.01, noisy_steering_rad=0.02,
        drift_x_m=0.05, drift_y_m=-0.1, drift_yaw_rad=0.02,
        localization_latency_steps=2, lidar_beam_count=80, lidar_dropout_count=1,
        lidar_perturbation_mean_m=0.03, lidar_perturbation_max_m=0.2,
    )
    logger.log_step(1, 1, [0.0, 0.5, 0.5], reward=0.0, telemetry=rt.invalid(step_id=1),
                     collision=False, target=False, sensor_diagnostics=diag)

    record = _read_last_step_record(run_dir)
    sd_record = record["sensor_diagnostics"]
    assert sd_record["schema_version"] == sd.SCHEMA_VERSION
    assert sd_record["episode_id"] == 42
    assert sd_record["sim_timestamp_sec"] == pytest.approx(1.0)


def test_physical_command_and_stale_flags_are_logged(tmp_path):
    logger, run_dir = _make_logger(tmp_path)
    telemetry = rt.invalid(
        step_id=3, emergency_stop=True, guarded_speed_mps=1.5, guarded_steering_rad=0.1,
        published_speed_mps=0.0, published_steering_rad=0.0, sensor_stale=True,
    )
    logger.log_step(3, 1, [0.0, 0.5, 0.5], reward=-1.0, telemetry=telemetry, collision=False, target=False)

    record = _read_last_step_record(run_dir)
    assert record["emergency_stop"] is True
    assert record["sensor_stale"] is True
    assert record["guarded_speed_mps"] == pytest.approx(1.5)
    assert record["published_speed_mps"] == pytest.approx(0.0)


def test_pose_trajectory_and_dt_default_to_none_when_not_provided(tmp_path):
    logger, run_dir = _make_logger(tmp_path)
    telemetry = rt.invalid(step_id=4)
    logger.log_step(4, 1, [0.0, 0.5, 0.5], reward=0.0, telemetry=telemetry, collision=False, target=False)

    record = _read_last_step_record(run_dir)
    assert record["pose"] is None
    assert record["trajectory"] is None
    assert record["actual_dt_sec"] is None


def test_pose_trajectory_and_dt_are_logged_when_provided(tmp_path):
    logger, run_dir = _make_logger(tmp_path)
    telemetry = rt.invalid(step_id=5)
    logger.log_step(
        5, 1, [0.0, 0.5, 0.5], reward=0.0, telemetry=telemetry, collision=False, target=False,
        pose=(1.0, 2.0, 0.5), trajectory_command=TrajectoryCommand(kappa=0.1, v_ref=1.0, horizon_m=1.5),
        actual_dt_sec=0.098,
    )

    record = _read_last_step_record(run_dir)
    assert record["pose"] == {"x": 1.0, "y": 2.0, "yaw": 0.5}
    assert record["trajectory"] == {"kappa": 0.1, "v_ref": 1.0, "horizon_m": 1.5}
    assert len(record["trajectory_points"]) == logger.profile.trajectory.num_samples
    assert record["actual_dt_sec"] == pytest.approx(0.098)


def test_full_state_motion_goal_reward_and_predicted_ground_truth_risk_are_logged(tmp_path):
    profile = load_profile("smoke_test")
    run_dir = run_directory(profile, str(tmp_path))
    logger = RunLogger(run_dir, profile, log_full_state=True)
    telemetry = rt.RiskTelemetry(
        step_id=6, valid=True, risk_target=0.35, min_clearance_m=1.0, ttc_sec=2.0,
        collision_within_horizon=False, stopping_margin_m=0.4, unrecoverable=False,
        safer_alternative_margin=0.1, actor_candidate_index=0,
        steering_saturation=True, goal_progress_m=0.6, goal_x=4.0, goal_y=2.0,
        reward_progress=0.2, reward_step=-0.01, reward_trajectory_smoothness=-0.03,
    )
    logger.log_step(
        6, 1, [0.0, 0.5, 0.5], reward=0.16, telemetry=telemetry,
        collision=False, target=False, state=[1.0, 2.0], next_state=[1.5, 2.5],
        measured_velocity_mps=0.8, measured_yaw_rate_rad_s=0.2,
        measured_steering_rad=0.1, predicted_risk=0.3,
    )
    record = _read_last_step_record(run_dir)
    assert record["state"] == [1.0, 2.0]
    assert record["next_state"] == [1.5, 2.5]
    assert record["goal"] == {"x": 4.0, "y": 2.0}
    assert record["measured_yaw_rate_rad_s"] == pytest.approx(0.2)
    assert record["risk_ground_truth"] == pytest.approx(0.35)
    assert record["risk_predicted"] == pytest.approx(0.3)
    assert record["reward_components"]["progress"] == pytest.approx(0.2)
    assert record["steering_saturation"] is True


def test_full_state_remains_null_when_opted_out(tmp_path):
    logger, run_dir = _make_logger(tmp_path)
    logger.log_step(
        7, 1, [0.0, 0.0, 0.0], reward=0.0, telemetry=rt.invalid(7),
        collision=False, target=False, state=[1.0], next_state=[2.0],
    )
    record = _read_last_step_record(run_dir)
    assert record["state"] is None
    assert record["next_state"] is None


# ------------------------------------------------------- P1-8: episode-start domain-rand logging
def test_log_episode_start_with_no_draw_records_null(tmp_path):
    """domain_randomization.enabled=False (or unspecified) -- the episode
    still gets a structured start record, with an explicit null draw
    rather than the event being silently skipped."""
    logger, run_dir = _make_logger(tmp_path)
    logger.log_episode_start(episode_index=1, seed=42, global_step=0)

    record = _read_last_step_record(run_dir)
    assert record["event"] == "episode_start"
    assert record["episode_index"] == 1
    assert record["seed"] == 42
    assert record["domain_rand_draw"] is None


def test_log_episode_start_records_the_sampled_draw_values(tmp_path):
    """section P1-8: sampled/applied domain-randomization values must be
    logged per episode -- proves every RandomizationDraw field round-trips
    into the structured JSONL record, not just a free-text console line."""
    logger, run_dir = _make_logger(tmp_path)
    draw = RandomizationDraw(
        mass_scale=1.05, friction_scale=0.9, wheel_radius_scale=1.01, steering_gain=0.95,
        steering_delay_sec=0.03, velocity_response_scale=0.98, command_latency_sec=0.02,
        lidar_range_noise_std_m=0.01, lidar_dropout_prob=0.005, odometry_noise_std=0.004,
        sensor_frame_drop_prob=0.002,
    )
    logger.log_episode_start(episode_index=3, seed=99, global_step=120, domain_rand_draw=draw)

    record = _read_last_step_record(run_dir)
    assert record["event"] == "episode_start"
    assert record["global_step"] == 120
    assert record["domain_rand_draw"]["mass_scale"] == pytest.approx(1.05)
    assert record["domain_rand_draw"]["command_latency_sec"] == pytest.approx(0.02)
    assert record["domain_rand_draw"]["sensor_frame_drop_prob"] == pytest.approx(0.002)
