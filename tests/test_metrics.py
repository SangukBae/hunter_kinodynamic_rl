"""evaluation/metrics.py: JSON-safe aggregation (None instead of a
non-standard Infinity literal when a metric is undefined, always paired
with an explicit valid_count) -- section P1-4."""

import json

import pytest

from hunter_kinodynamic_rl.evaluation.metrics import aggregate, steering_saturation_rate, success_path_length


def _episode(**overrides):
    base = {
        "success": False, "collision": False, "timeout": True, "unrecoverable": False,
        "steps": 10, "path_length_m": 5.0, "straight_line_distance_m": 4.0,
        "navigation_time_sec": 1.0,
        "velocities_mps": [1.0, 1.0], "min_clearance_m": 0.5, "min_clearance_valid_count": 1,
        "ttc_values_sec": [2.0], "steering_values_rad": [0.0, 0.1],
        "steering_limit_rad": 0.377, "control_deltas": [0.05], "emergency_stops": 0,
        "risk_valid_steps": 1, "collision_free_steps": 0,
        "stopping_margin_values_m": [0.2, 0.4],
        "steering_rates_rad_s": [0.5, 1.5],
        "goal_progress_m": 3.0, "goal_progress_ratio": 0.75,
        "curvature_feasibility_violations": 1, "physical_command_steps": 10,
    }
    base.update(overrides)
    return base


def test_aggregate_empty_returns_empty_dict():
    assert aggregate([]) == {}


def test_aggregate_with_no_obstacles_reports_none_not_infinity():
    episodes = [_episode(min_clearance_m=None, ttc_values_sec=[])]
    summary = aggregate(episodes)
    assert summary["min_clearance_m_worst"] is None
    assert summary["min_clearance_valid_count"] == 0
    assert summary["ttc_sec_worst"] is None
    assert summary["ttc_valid_count"] == 0
    # Must be genuinely JSON-serializable with `null`, not a bare Infinity token.
    text = json.dumps(summary)
    assert "Infinity" not in text
    assert json.loads(text)["min_clearance_m_worst"] is None


def test_aggregate_with_valid_clearance_reports_real_values():
    episodes = [_episode(min_clearance_m=0.4), _episode(min_clearance_m=0.9)]
    summary = aggregate(episodes)
    assert summary["min_clearance_m_worst"] == pytest.approx(0.4)
    assert summary["min_clearance_valid_count"] == 2


def test_aggregate_navigation_time_uses_real_seconds():
    episodes = [_episode(navigation_time_sec=3.0), _episode(navigation_time_sec=5.0)]
    summary = aggregate(episodes)
    assert summary["navigation_time_sec_mean"] == pytest.approx(4.0)
    assert summary["navigation_time_sec_valid_count"] == 2


def test_aggregate_navigation_time_none_when_clock_unavailable():
    episodes = [_episode(navigation_time_sec=None)]
    summary = aggregate(episodes)
    assert summary["navigation_time_sec_mean"] is None
    assert summary["navigation_time_sec_valid_count"] == 0


def test_success_path_length_zero_for_failed_episode():
    assert success_path_length(_episode(success=False)) == 0.0


def test_success_path_length_ratio_for_successful_episode():
    ep = _episode(success=True, path_length_m=8.0, straight_line_distance_m=4.0)
    assert success_path_length(ep) == pytest.approx(0.5)


def test_steering_saturation_rate_counts_near_limit_samples():
    ep = _episode(steering_values_rad=[0.0, 0.377, -0.377, 0.1], steering_limit_rad=0.377)
    assert steering_saturation_rate(ep) == pytest.approx(0.5)


def test_aggregate_reports_progress_steering_rate_feasibility_and_stopping_margin():
    summary = aggregate([_episode()])
    assert summary["goal_progress_m_mean"] == pytest.approx(3.0)
    assert summary["goal_progress_ratio_mean"] == pytest.approx(0.75)
    assert summary["average_steering_magnitude_rad"] == pytest.approx(0.05)
    assert summary["steering_rate_rad_s_mean"] == pytest.approx(1.0)
    assert summary["steering_rate_rad_s_max"] == pytest.approx(1.5)
    assert summary["curvature_feasibility_violation_rate"] == pytest.approx(0.1)
    assert summary["stopping_margin_m_mean"] == pytest.approx(0.3)
    assert summary["stopping_margin_m_worst"] == pytest.approx(0.2)


# ---------------------------------------------------- P0-6: TTC censoring
def test_ttc_mean_is_not_contaminated_by_censored_no_collision_steps():
    """The core P0-6 regression: benchmark_runner.py only ever appends a
    REAL (collision_within_horizon=True) ttc_sec into ttc_values_sec -- an
    episode where every risk-assessed step found no collision must report
    ttc_sec_mean as undefined (None), NOT as some horizon-clamp constant
    silently averaged in as if it were a measurement."""
    episodes = [_episode(ttc_values_sec=[], risk_valid_steps=50, collision_free_steps=50)]
    summary = aggregate(episodes)
    assert summary["ttc_sec_mean"] is None
    assert summary["ttc_valid_count"] == 0
    assert summary["collision_free_rate_mean"] == pytest.approx(1.0)
    assert summary["collision_free_rate_valid_count"] == 1


def test_ttc_mean_reflects_only_real_near_miss_measurements():
    episodes = [
        _episode(ttc_values_sec=[0.4, 0.6], risk_valid_steps=10, collision_free_steps=8),
        _episode(ttc_values_sec=[], risk_valid_steps=10, collision_free_steps=10),
    ]
    summary = aggregate(episodes)
    assert summary["ttc_sec_mean"] == pytest.approx(0.5)
    assert summary["ttc_valid_count"] == 2
    # episode 1: 8/10 censored=0.8, episode 2: 10/10=1.0 -> mean 0.9
    assert summary["collision_free_rate_mean"] == pytest.approx(0.9)
    assert summary["collision_free_rate_valid_count"] == 2


def test_collision_free_rate_undefined_when_risk_framework_never_produced_a_valid_sample():
    episodes = [_episode(risk_valid_steps=0, collision_free_steps=0, ttc_values_sec=[])]
    summary = aggregate(episodes)
    assert summary["collision_free_rate_mean"] is None
    assert summary["collision_free_rate_valid_count"] == 0
