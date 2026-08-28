import pytest

from hunter_kinodynamic_rl.navigation.hierarchy.replanning import (
    ReplanningConfig, evaluate_replanning, junction_detected_stub,
)
from hunter_kinodynamic_rl.navigation.hierarchy.subgoal_manager import SubgoalStatus


def _cfg(**overrides) -> ReplanningConfig:
    cfg = ReplanningConfig(
        local_option_timeout_steps=20, no_progress_window_steps=5, no_progress_min_delta_m=0.2,
        consecutive_emergency_stop_limit=3, local_risk_threshold=0.7, localization_min_confidence=0.4,
    )
    for k, v in overrides.items():
        cfg = ReplanningConfig(**{**cfg.__dict__, k: v})
    return cfg


def _no_trigger_kwargs():
    return dict(
        reached=False, local_steps=1, subgoal_distance_history=(5.0,),
        subgoal_endpoint_blocked=False, consecutive_emergency_stops=0,
        latest_predicted_risk=None, localization_confidence=None, junction_detected=False,
    )


def test_defaults_produce_no_trigger():
    assert evaluate_replanning(_cfg(), **_no_trigger_kwargs()) is None


def test_reached_wins_over_every_other_condition():
    kwargs = _no_trigger_kwargs()
    kwargs.update(reached=True, subgoal_endpoint_blocked=True, local_steps=999)
    trigger = evaluate_replanning(_cfg(), **kwargs)
    assert trigger.status == SubgoalStatus.REACHED
    assert trigger.reason == "subgoal_tolerance_reached"


@pytest.mark.parametrize("confidence", [0.1, float("nan")])
def test_degraded_localization_wins_over_reached(confidence):
    kwargs = _no_trigger_kwargs()
    kwargs.update(reached=True, localization_confidence=confidence)
    trigger = evaluate_replanning(_cfg(), **kwargs)
    assert trigger.status == SubgoalStatus.CANCELLED_BY_REPLAN
    assert trigger.reason == "localization_confidence_degraded"


def test_local_option_timeout():
    kwargs = _no_trigger_kwargs()
    kwargs.update(local_steps=20)
    trigger = evaluate_replanning(_cfg(), **kwargs)
    assert trigger.status == SubgoalStatus.FAILED_TIMEOUT
    assert trigger.reason == "local_option_timeout"


def test_local_option_timeout_not_triggered_below_threshold():
    kwargs = _no_trigger_kwargs()
    kwargs.update(local_steps=19)
    assert evaluate_replanning(_cfg(), **kwargs) is None


def test_no_progress_within_window():
    kwargs = _no_trigger_kwargs()
    # 5-sample window, distance barely moves (< 0.2 m required decrease).
    kwargs.update(subgoal_distance_history=(5.0, 4.95, 4.92, 4.91, 4.90))
    trigger = evaluate_replanning(_cfg(), **kwargs)
    assert trigger.status == SubgoalStatus.FAILED_NO_PROGRESS
    assert trigger.reason == "no_progress_within_window"


def test_no_progress_not_triggered_with_enough_progress():
    kwargs = _no_trigger_kwargs()
    kwargs.update(subgoal_distance_history=(5.0, 4.5, 4.0, 3.5, 3.0))
    assert evaluate_replanning(_cfg(), **kwargs) is None


def test_no_progress_not_triggered_before_window_fills():
    kwargs = _no_trigger_kwargs()
    kwargs.update(subgoal_distance_history=(5.0, 5.0, 5.0))  # only 3 samples, window is 5
    assert evaluate_replanning(_cfg(), **kwargs) is None


def test_subgoal_endpoint_blocked():
    kwargs = _no_trigger_kwargs()
    kwargs.update(subgoal_endpoint_blocked=True)
    trigger = evaluate_replanning(_cfg(), **kwargs)
    assert trigger.status == SubgoalStatus.FAILED_BLOCKED
    assert trigger.reason == "subgoal_endpoint_occupied_or_inflated"


def test_consecutive_emergency_stops():
    kwargs = _no_trigger_kwargs()
    kwargs.update(consecutive_emergency_stops=3)
    trigger = evaluate_replanning(_cfg(), **kwargs)
    assert trigger.status == SubgoalStatus.FAILED_BLOCKED
    assert trigger.reason == "consecutive_emergency_stops"


def test_consecutive_emergency_stops_below_limit_no_trigger():
    kwargs = _no_trigger_kwargs()
    kwargs.update(consecutive_emergency_stops=2)
    assert evaluate_replanning(_cfg(), **kwargs) is None


def test_local_risk_threshold_exceeded():
    kwargs = _no_trigger_kwargs()
    kwargs.update(latest_predicted_risk=0.71)
    trigger = evaluate_replanning(_cfg(), **kwargs)
    assert trigger.status == SubgoalStatus.FAILED_HIGH_RISK
    assert trigger.reason == "local_risk_threshold_exceeded"


def test_risk_at_threshold_not_triggered():
    kwargs = _no_trigger_kwargs()
    kwargs.update(latest_predicted_risk=0.7)
    assert evaluate_replanning(_cfg(), **kwargs) is None


def test_localization_confidence_degraded_cancels_not_fails():
    kwargs = _no_trigger_kwargs()
    kwargs.update(localization_confidence=0.1)
    trigger = evaluate_replanning(_cfg(), **kwargs)
    assert trigger.status == SubgoalStatus.CANCELLED_BY_REPLAN
    assert trigger.reason == "localization_confidence_degraded"


@pytest.mark.parametrize("confidence", [float("nan"), float("inf"), float("-inf"), "not-a-number"])
def test_non_finite_or_malformed_localization_confidence_cancels_by_replan(confidence):
    kwargs = _no_trigger_kwargs()
    kwargs.update(localization_confidence=confidence)
    trigger = evaluate_replanning(_cfg(), **kwargs)
    assert trigger.status == SubgoalStatus.CANCELLED_BY_REPLAN
    assert trigger.reason == "localization_confidence_degraded"


def test_localization_confidence_at_minimum_not_triggered():
    kwargs = _no_trigger_kwargs()
    kwargs.update(localization_confidence=0.4)
    assert evaluate_replanning(_cfg(), **kwargs) is None


def test_junction_detected_cancels():
    kwargs = _no_trigger_kwargs()
    kwargs.update(junction_detected=True)
    trigger = evaluate_replanning(_cfg(), **kwargs)
    assert trigger.status == SubgoalStatus.CANCELLED_BY_REPLAN
    assert trigger.reason == "new_junction_detected"


def test_junction_detected_stub_always_false():
    """Phase 5 stub -- Phase 2 never wires a real junction detector, so the
    stub itself must always report 'no junction' regardless of input."""
    assert junction_detected_stub() is False
    assert junction_detected_stub(1, 2, 3, foo="bar") is False


@pytest.mark.parametrize("field,bad_value", [
    ("local_option_timeout_steps", 0),
    ("no_progress_window_steps", 0),
    ("no_progress_min_delta_m", -0.1),
    ("consecutive_emergency_stop_limit", 0),
    ("local_risk_threshold", 0.0),
    ("localization_min_confidence", 1.5),
])
def test_replanning_config_validate_rejects_invalid_fields(field, bad_value):
    cfg = ReplanningConfig(**{field: bad_value})
    with pytest.raises(ValueError):
        cfg.validate()


def test_timeout_and_blocked_and_no_progress_and_high_risk_and_cancelled_have_distinct_reasons():
    """Explicit cross-check of the required test item: every one of the
    five failure/cancel triggers is independently distinguishable by
    reason string, not just by (grouped) status enum."""
    reasons = set()
    cfg = _cfg()

    kwargs = _no_trigger_kwargs(); kwargs["local_steps"] = 20
    reasons.add(evaluate_replanning(cfg, **kwargs).reason)

    kwargs = _no_trigger_kwargs(); kwargs["subgoal_endpoint_blocked"] = True
    reasons.add(evaluate_replanning(cfg, **kwargs).reason)

    kwargs = _no_trigger_kwargs(); kwargs["subgoal_distance_history"] = (5.0, 5.0, 5.0, 5.0, 5.0)
    reasons.add(evaluate_replanning(cfg, **kwargs).reason)

    kwargs = _no_trigger_kwargs(); kwargs["latest_predicted_risk"] = 0.9
    reasons.add(evaluate_replanning(cfg, **kwargs).reason)

    kwargs = _no_trigger_kwargs(); kwargs["localization_confidence"] = 0.0
    reasons.add(evaluate_replanning(cfg, **kwargs).reason)

    assert len(reasons) == 5
