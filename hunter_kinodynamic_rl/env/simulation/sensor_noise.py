"""Fixed-shape, curriculum-independent sensor/localization error model
(requirement 3 of the drl_agent -> hunter_kinodynamic_rl port -- see
``config/schema.py``'s ``SensorNoiseConfig``).

Unlike ``env/randomization/domain_randomizer.py`` (which draws a NEW
magnitude every episode from a configured RANGE, for sim-to-real TRAINING
ROBUSTNESS), every magnitude here is a single, profile-authored CONSTANT
applied uniformly for the whole run -- this package has no drl_agent-style
curriculum stages, so there is no notion of "this noise grows/shrinks by
stage" to begin with. It models a REALISTIC, PERSISTENT sensor/localization
imperfection (e.g. "this SLAM stack has a slowly-drifting few-centimeter
bias"), composed ON TOP OF (i.e. applied strictly AFTER) whatever
domain_randomization noise is already active -- see
``environment_node.py``'s ``_observation_obs_state``/``_build_state_vector``
call sites for the exact composition order.

**Ground-truth/observation separation (the load-bearing invariant):** every
function here is called EXCLUSIVELY from the observation-construction path
in ``environment_node.py`` (``_observation_obs_state`` for LiDAR,
``_build_state_vector`` for pose/velocity/steering). Nothing in this module
is ever called from the collision check, the risk-label computation (which
uses the PRE-noise ground-truth ``self._robot_pose``/dynamic-obstacle
specs), scenario feasibility, or reset verification -- those all keep using
``self._robot_pose``/``self._robot_twist``/the real LiDAR scan directly.
This module has NO access to (and does not need) those ground-truth values
at all beyond what's explicitly passed in for perturbation.

**Reproducibility:** :class:`SensorNoiseState` owns a DEDICATED
``numpy.random.RandomState`` stream, re-seeded fresh every ``/reset`` from
``(episode_seed, a fixed salt)`` by :func:`reset_state` -- never the
scenario-generation RNG (``procedural_generator``'s ``rng``) or
domain_randomization's own ``_domain_rand_step_rng``. Because it derives
purely from the episode seed (never from a running counter that persists
across episodes), it is automatically correct across checkpoint/resume: the
trainer's own :class:`~hunter_kinodynamic_rl.env.scenarios.seed_scheduler.SeedScheduler`
is what's actually persisted in a checkpoint (see
``training/trainer_base.py``'s ``_save_checkpoint``) -- resuming it
naturally continues the same episode-seed sequence, which in turn
regenerates the exact same noise-RNG seed for every subsequent episode, with
no extra state for this module to save or restore on its own.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Tuple, TYPE_CHECKING

import numpy as np

from hunter_kinodynamic_rl.common.geometry import wrap_to_pi

if TYPE_CHECKING:
    from hunter_kinodynamic_rl.config.schema import SensorNoiseConfig

# Distinguishes this module's RNG stream from every other per-episode RNG
# derived from the same seed (_obstacle_rng uses the run_seed directly;
# _domain_rand_step_rng uses (seed*7+1); this uses (seed*1000003 + salt)) --
# the exact multiplier/salt values only need to be MUTUALLY distinct, not
# cryptographically meaningful.
_SEED_SALT = 0x5E4501


@dataclass
class SensorNoiseState:
    """Per-episode mutable state -- constructed fresh by :func:`reset_state`
    at every ``/reset``. Never persisted/restored explicitly (see module
    docstring for why that's unnecessary)."""

    rng: np.random.RandomState
    drift_x: float = 0.0
    drift_y: float = 0.0
    drift_yaw: float = 0.0
    pose_history: Deque[Tuple[float, float, float]] = field(default_factory=deque)


def reset_state(episode_seed: int, cfg: "SensorNoiseConfig") -> SensorNoiseState:
    """Deterministic given ``episode_seed`` alone -- ``cfg`` is accepted
    (unused beyond documenting the call site's intent) purely so a future
    per-config seed derivation could use it without changing every caller;
    the seed itself deliberately depends on nothing but the episode seed,
    so two runs with different sensor_noise settings but the SAME episode
    seed still draw from bit-identical RNG streams (their noise SEQUENCES
    only diverge because they consume different numbers of draws per
    field, not because the seed itself changed)."""
    del cfg
    seed = (int(episode_seed) * 1000003 + _SEED_SALT) & 0xFFFFFFFF
    return SensorNoiseState(rng=np.random.RandomState(seed))


def _ou_step(current: float, theta: float, sigma_m: float, dt_sec: float, rng: np.random.RandomState) -> float:
    """Exact discrete-time solution of the Ornstein-Uhlenbeck SDE
    ``dX = -theta*X*dt + sigma*dW`` (requirement 6), rather than a
    first-order Euler-Maruyama approximation -- Euler's local error grows
    with ``theta*dt``, so it drifts away from the true OU distribution (and
    can even go numerically unstable) for a large ``theta`` and/or a
    coarse ``dt_sec``; the closed-form update below is exact for every
    finite ``theta >= 0`` and ``dt_sec >= 0``, so state quality never
    depends on how fine the control tick is. ``sigma_m`` is the SDE's own
    DIFFUSION COEFFICIENT (units: <state-unit> / sqrt(sec)) -- it is NOT a
    per-step standard deviation; the actual per-step noise standard
    deviation is derived below from ``sigma_m``, ``theta``, and ``dt_sec``
    together.

    Guarded no-ops (both preserve :func:`reset_state`'s own documented
    byte-identical-when-disabled contract -- see ``SensorNoiseConfig``'s
    docstring, "a zero sigma disables its own drift axis entirely"):
    ``sigma_m <= 0`` disables this drift axis entirely (no mean-reversion
    decay is applied either -- the state simply never moves away from the
    0.0 ``reset_state`` initializes it to), and ``dt_sec <= 0`` is a
    zero-duration tick (nothing to integrate over, state unchanged).

    ``theta <= 0`` (no mean reversion -- pure driftless diffusion; schema
    validation already rejects a negative ``theta``, so ``<= 0`` in
    practice only ever means exactly 0 here) is special-cased to avoid a
    0/0 in the closed-form variance below; it reduces to plain Brownian
    motion, i.e. exactly what Euler-Maruyama also computes for
    ``theta == 0``. Every other ``theta`` uses the exact OU mean/variance --
    for a very large ``theta*dt``, ``exp(-theta*dt)``/``exp(-2*theta*dt)``
    underflow gracefully toward 0.0 (Python floats never raise/overflow on
    that side), so the state decays cleanly to (noise around) zero instead
    of overshooting the way an explicit-Euler step (`current - theta *
    current * dt`) would once ``theta * dt > 2`` (oscillating and, past
    that, diverging)."""
    if sigma_m <= 0.0 or dt_sec <= 0.0:
        return current
    if theta <= 0.0:
        return current + float(rng.normal(0.0, sigma_m * math.sqrt(dt_sec)))
    decay = math.exp(-theta * dt_sec)
    variance = (sigma_m ** 2) / (2.0 * theta) * (1.0 - math.exp(-2.0 * theta * dt_sec))
    return current * decay + float(rng.normal(0.0, math.sqrt(max(variance, 0.0))))


def tick_drift(state: SensorNoiseState, cfg: "SensorNoiseConfig", dt_sec: float) -> None:
    """Advances the Ornstein-Uhlenbeck localization-drift state by one
    control tick. Called ONCE per real ``/step`` (never at ``/reset`` --
    the initial observation has no elapsed control time to drift over, so
    drift starts every episode at exactly zero, set by
    :func:`reset_state`'s default field values). A no-op whenever
    ``cfg.enabled`` is False or both sigma fields are non-positive."""
    if not cfg.enabled:
        return
    state.drift_x = _ou_step(state.drift_x, cfg.localization_drift_theta, cfg.localization_drift_xy_sigma_m,
                              dt_sec, state.rng)
    state.drift_y = _ou_step(state.drift_y, cfg.localization_drift_theta, cfg.localization_drift_xy_sigma_m,
                              dt_sec, state.rng)
    state.drift_yaw = _ou_step(state.drift_yaw, cfg.localization_drift_theta, cfg.localization_drift_yaw_sigma_rad,
                                dt_sec, state.rng)


def measured_pose(state: SensorNoiseState, cfg: "SensorNoiseConfig",
                   x: float, y: float, yaw: float) -> Tuple[float, float, float]:
    """Ground-truth ``(x, y, yaw)`` in -> the AGENT'S localization reading
    out: current OU drift + i.i.d. Gaussian noise, then (if
    ``localization_latency_steps > 0``) delayed by that many ticks. Returns
    the untouched input unchanged when ``cfg.enabled`` is False (the
    default) -- byte-identical to not calling this at all."""
    if not cfg.enabled:
        return x, y, yaw
    nx, ny, nyaw = x + state.drift_x, y + state.drift_y, yaw + state.drift_yaw
    if cfg.localization_xy_noise_std_m > 0.0:
        nx += float(state.rng.normal(0.0, cfg.localization_xy_noise_std_m))
        ny += float(state.rng.normal(0.0, cfg.localization_xy_noise_std_m))
    if cfg.localization_yaw_noise_std_rad > 0.0:
        nyaw += float(state.rng.normal(0.0, cfg.localization_yaw_noise_std_rad))
    nyaw = wrap_to_pi(nyaw)
    if cfg.localization_latency_steps <= 0:
        return nx, ny, nyaw
    state.pose_history.append((nx, ny, nyaw))
    max_len = cfg.localization_latency_steps + 1
    while len(state.pose_history) > max_len:
        state.pose_history.popleft()
    return state.pose_history[0]  # oldest retained == exactly latency_steps ago once the buffer has filled


def measured_velocity(state: SensorNoiseState, cfg: "SensorNoiseConfig",
                       v_mps: float, yaw_rate_radps: float) -> Tuple[float, float]:
    if not cfg.enabled:
        return v_mps, yaw_rate_radps
    mv = v_mps + (float(state.rng.normal(0.0, cfg.velocity_noise_std_mps))
                  if cfg.velocity_noise_std_mps > 0.0 else 0.0)
    myr = yaw_rate_radps + (float(state.rng.normal(0.0, cfg.yaw_rate_noise_std_radps))
                            if cfg.yaw_rate_noise_std_radps > 0.0 else 0.0)
    return mv, myr


def measured_steering(state: SensorNoiseState, cfg: "SensorNoiseConfig", steering_rad: float) -> float:
    if not cfg.enabled or cfg.steering_noise_std_rad <= 0.0:
        return steering_rad
    return steering_rad + float(state.rng.normal(0.0, cfg.steering_noise_std_rad))


def apply_lidar_noise(state: SensorNoiseState, cfg: "SensorNoiseConfig",
                       ranges: np.ndarray, max_range_m: float) -> np.ndarray:
    """Additive constant bias + Gaussian range noise + per-beam dropout (a
    dropped beam reads as ``max_range_m``, the standard "no return"
    convention -- matches ``domain_randomizer.apply_lidar_noise``'s own
    dropout convention). Applied AFTER whatever domain_randomization LiDAR
    noise the caller already applied (composable, not exclusive) --
    callers MUST pass the AGENT'S observation array here, never the
    ground-truth scan used for collision/risk."""
    if not cfg.enabled:
        return ranges
    if (cfg.lidar_bias_m == 0.0 and cfg.lidar_range_noise_std_m <= 0.0 and cfg.lidar_dropout_prob <= 0.0):
        return ranges
    noisy = ranges.astype(np.float32, copy=True)
    if cfg.lidar_bias_m != 0.0:
        noisy = noisy + np.float32(cfg.lidar_bias_m)
    if cfg.lidar_range_noise_std_m > 0.0:
        noisy = noisy + state.rng.normal(0.0, cfg.lidar_range_noise_std_m, size=noisy.shape).astype(np.float32)
    noisy = np.clip(noisy, 0.0, max_range_m)
    if cfg.lidar_dropout_prob > 0.0:
        dropped = state.rng.uniform(0.0, 1.0, size=noisy.shape) < cfg.lidar_dropout_prob
        noisy = np.where(dropped, np.float32(max_range_m), noisy)
    return noisy.astype(ranges.dtype)
