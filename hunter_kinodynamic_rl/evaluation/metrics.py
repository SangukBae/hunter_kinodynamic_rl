"""Aggregate evaluation metrics (section 37) from a list of per-episode
result dicts. Pure functions -- no ROS, no filesystem -- so directly
unit-tested; result_writer.py owns turning these into CSV/JSON on disk.

Each episode result dict is expected to carry (at minimum):
    success: bool, collision: bool, timeout: bool, unrecoverable: bool,
    steps: int, path_length_m: float, straight_line_distance_m: float,
    velocities_mps: list[float], min_clearance_m: float,
    ttc_values_sec: list[float], steering_values_rad: list[float],
    steering_limit_rad: float, control_deltas: list[float],
    emergency_stops: int
"""

from __future__ import annotations

import math
from typing import Dict, List


def _mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _rate(episodes: List[dict], key: str) -> float:
    if not episodes:
        return 0.0
    return sum(1 for e in episodes if e.get(key)) / len(episodes)


def success_path_length(episode: dict) -> float:
    """SPL (Success weighted by Path Length) contribution for ONE episode --
    0 for a failed episode, else straight_line/max(path, straight_line)."""
    if not episode.get("success"):
        return 0.0
    path = max(episode.get("path_length_m", 0.0), 1e-6)
    straight = episode.get("straight_line_distance_m", 0.0)
    return straight / max(path, straight)


def steering_saturation_rate(episode: dict) -> float:
    values = episode.get("steering_values_rad", [])
    limit = episode.get("steering_limit_rad", 1.0)
    if not values or limit <= 0.0:
        return 0.0
    return sum(1 for s in values if abs(s) >= limit - 1e-3) / len(values)


def steering_smoothness(episode: dict) -> float:
    """Mean absolute steering-rate proxy (radians between consecutive
    commands) -- lower is smoother."""
    values = episode.get("steering_values_rad", [])
    if len(values) < 2:
        return 0.0
    deltas = [abs(values[i] - values[i - 1]) for i in range(1, len(values))]
    return _mean(deltas)


def _episode_collision_free_rate(episode: dict) -> float:
    """Fraction of this episode's VALID risk-assessed steps where the
    rollout found NO collision within its horizon -- i.e. the TTC label
    for that step was censored (clamped at the horizon), not a real
    measurement (section P0-6: see risk/ttc.py's docstring). None when the
    episode had zero valid risk samples at all (risk framework disabled,
    or ``action_space.mode != "trajectory"``) -- distinct from a rate of
    0.0, which would mean "every single step found a collision"."""
    valid = episode.get("risk_valid_steps", 0)
    if valid <= 0:
        return None
    return episode.get("collision_free_steps", 0) / valid


def aggregate(episodes: List[dict]) -> Dict[str, float]:
    """Every field is JSON-safe (``None``, never a non-standard ``Infinity``
    literal, for an "undefined because no obstacle/label was ever recorded"
    case -- section P1-4): each ``*_worst``/``*_mean`` metric that can be
    undefined is paired with an explicit ``*_valid_count`` so a reader can
    tell "no data" apart from "zero".

    ``ttc_sec_mean``/``ttc_sec_worst`` are computed ONLY from episodes'
    ``ttc_values_sec`` -- which benchmark_runner.py populates with REAL,
    UNCENSORED time-to-collision measurements only (steps where a
    collision was actually found within the assessment horizon). Steps
    where no collision was found (the common, desired case for a
    well-behaved policy) are NEVER folded into this average as if
    ``horizon_sec`` itself were a measured TTC (section P0-6) -- their
    prevalence is instead reported separately via
    ``collision_free_rate_mean``, computed from EVERY valid risk-assessed
    step regardless of whether a collision was found."""
    n = len(episodes)
    if n == 0:
        return {}

    all_ttc = [t for e in episodes for t in e.get("ttc_values_sec", [])]
    all_clearance = [e["min_clearance_m"] for e in episodes
                      if e.get("min_clearance_m") is not None and math.isfinite(e["min_clearance_m"])]
    all_control_deltas = [d for e in episodes for d in e.get("control_deltas", [])]
    all_steering = [abs(s) for e in episodes for s in e.get("steering_values_rad", [])]
    all_steering_rates = [r for e in episodes for r in e.get("steering_rates_rad_s", [])]
    all_stopping_margins = [m for e in episodes for m in e.get("stopping_margin_values_m", [])
                            if math.isfinite(m)]
    goal_progress = [e["goal_progress_m"] for e in episodes if e.get("goal_progress_m") is not None]
    goal_progress_ratio = [e["goal_progress_ratio"] for e in episodes
                           if e.get("goal_progress_ratio") is not None]
    command_steps = sum(e.get("physical_command_steps", e.get("steps", 0)) for e in episodes)
    feasibility_violations = sum(e.get("curvature_feasibility_violations", 0) for e in episodes)
    nav_times = [e["navigation_time_sec"] for e in episodes if e.get("navigation_time_sec") is not None]
    collision_free_rates = [r for r in (_episode_collision_free_rate(e) for e in episodes) if r is not None]

    return {
        "num_episodes": n,
        "success_rate": _rate(episodes, "success"),
        "collision_rate": _rate(episodes, "collision"),
        "timeout_rate": _rate(episodes, "timeout"),
        "unrecoverable_state_rate": _rate(episodes, "unrecoverable"),
        "spl": _mean([success_path_length(e) for e in episodes]),
        "navigation_time_steps_mean": _mean([e.get("steps", 0) for e in episodes]),
        "navigation_time_sec_mean": _mean(nav_times) if nav_times else None,
        "navigation_time_sec_valid_count": len(nav_times),
        "path_length_m_mean": _mean([e.get("path_length_m", 0.0) for e in episodes]),
        "goal_progress_m_mean": _mean(goal_progress) if goal_progress else None,
        "goal_progress_ratio_mean": _mean(goal_progress_ratio) if goal_progress_ratio else None,
        "goal_progress_valid_count": len(goal_progress),
        "average_velocity_mps_mean": _mean(
            [_mean(e.get("velocities_mps", [])) for e in episodes if e.get("velocities_mps")]
        ),
        "min_clearance_m_mean": _mean(all_clearance) if all_clearance else None,
        "min_clearance_m_worst": min(all_clearance) if all_clearance else None,
        "min_clearance_valid_count": len(all_clearance),
        "ttc_sec_mean": _mean(all_ttc) if all_ttc else None,
        "ttc_sec_worst": min(all_ttc) if all_ttc else None,
        "ttc_valid_count": len(all_ttc),
        "collision_free_rate_mean": _mean(collision_free_rates) if collision_free_rates else None,
        "collision_free_rate_valid_count": len(collision_free_rates),
        "steering_saturation_rate_mean": _mean([steering_saturation_rate(e) for e in episodes]),
        "average_steering_magnitude_rad": _mean(all_steering),
        "steering_rate_rad_s_mean": _mean(all_steering_rates),
        "steering_rate_rad_s_max": max(all_steering_rates) if all_steering_rates else 0.0,
        "steering_smoothness_mean": _mean([steering_smoothness(e) for e in episodes]),
        "curvature_feasibility_violation_rate": (
            feasibility_violations / command_steps if command_steps > 0 else 0.0),
        "curvature_feasibility_violation_count": feasibility_violations,
        "stopping_margin_m_mean": _mean(all_stopping_margins) if all_stopping_margins else None,
        "stopping_margin_m_worst": min(all_stopping_margins) if all_stopping_margins else None,
        "stopping_margin_valid_count": len(all_stopping_margins),
        "control_smoothness_mean": _mean(all_control_deltas),
        "emergency_stop_count_total": sum(e.get("emergency_stops", 0) for e in episodes),
    }
