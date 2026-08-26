"""Per-episode domain randomization draw (section 30) -- sim-to-real prep.
Pure sampling: what to DO with each sampled value (perturb the RobotConfig
used for that episode's dynamics rollout, inject LiDAR noise, etc.) is the
caller's job; this module only owns "given a seed and the configured
ranges, produce one consistent randomization draw"."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from hunter_kinodynamic_rl.config.schema import DomainRandomizationConfig, RobotConfig


@dataclass(frozen=True)
class RandomizationDraw:
    mass_scale: float = 1.0
    friction_scale: float = 1.0
    wheel_radius_scale: float = 1.0
    steering_gain: float = 1.0
    steering_delay_sec: float = 0.0
    velocity_response_scale: float = 1.0
    command_latency_sec: float = 0.0
    lidar_range_noise_std_m: float = 0.0
    lidar_dropout_prob: float = 0.0
    odometry_noise_std: float = 0.0
    sensor_frame_drop_prob: float = 0.0


# Which RandomizationDraw fields actually reach the LIVE Ignition Fortress
# plant (a real /cmd_vel-visible or physics-visible consequence) vs which
# stay MODEL-ONLY (fold into the RobotConfig used for action decoding /
# the offline dynamics-rollout risk model, and/or the observation, but never
# touch simulated motion) -- code review: "silent no-op randomization flag"
# concern. Ignition Fortress's public ros_gz_interfaces service surface used
# by this package (SpawnEntity/DeleteEntity/SetEntityPose/ControlWorld --
# see env/simulation/gazebo_runtime.py) has NO runtime service for per-link
# mass/inertia or wheel geometry (that would require either a custom
# world/model plugin or a full delete+respawn with an edited SDF, both out
# of scope here); it also has no generic tyre-friction service. So:
#   - mass_scale, wheel_radius_scale: MODEL-ONLY, unconditionally --
#     genuinely not runtime-settable with the tools this package has.
#   - friction_scale, velocity_response_scale: GAZEBO-APPLIED via
#     rate_limit_speed() below (applied to the PRE-safety-guard nominal
#     command in environment_node.py, whenever the episode's draw is
#     non-identity -- see that call site's docstring) -- these two scale
#     robot.accel_limit_mps2/brake_decel_mps2, which nothing previously
#     read on the real command-publish path (only the risk-model rollout
#     did).
#   - steering_gain: GAZEBO-APPLIED, but only when
#     features.trajectory_l_preview_blend is also enabled (default True for
#     the trajectory-mode profiles that use domain randomization) -- it
#     scales robot.steering_rate_deg_s, which trajectory/pure_pursuit_adapter.py's
#     L-derived steering commit window (see its module docstring) uses as a
#     REAL actuator rate limit on the published steering command. With that
#     feature OFF, steering_gain reverts to MODEL-ONLY (dynamics-rollout risk
#     model only).
#   - steering_delay_sec, command_latency_sec: GAZEBO-APPLIED unconditionally
#     (steering_lag_alpha / the command-delay queue, both always active).
#   - lidar_*/odometry_*/sensor_frame_drop_prob: OBSERVATION-ONLY by design
#     (apply_lidar_noise/apply_odometry_noise/should_drop_sensor_frame all
#     explicitly perturb only the AGENT'S OBSERVATION, never ground truth or
#     the published command -- this is the intended, not a missing, boundary).
MODEL_ONLY_FIELDS = frozenset({"mass_scale", "wheel_radius_scale"})
GAZEBO_APPLIED_FIELDS = frozenset({
    "friction_scale", "velocity_response_scale", "steering_gain",
    "steering_delay_sec", "command_latency_sec",
})
OBSERVATION_ONLY_FIELDS = frozenset({
    "lidar_range_noise_std_m", "lidar_dropout_prob", "odometry_noise_std", "sensor_frame_drop_prob",
})


def classify_draw_fields(draw: RandomizationDraw) -> dict:
    """Per-field {name: category} for THIS draw -- used by
    environment_node.py's reset logging so a training run's logs show which
    randomization axes actually reached the simulated plant this episode,
    not just which ranges are configured (no silently-inert flag hiding in
    the logs). ``steering_gain`` reports "model_only" here always -- the
    caller (environment_node.py, which knows
    ``features.trajectory_l_preview_blend``) upgrades it to "gazebo_applied"
    in its own log line when that feature is active; this function has no
    access to FeatureFlags and must not guess."""
    all_fields = MODEL_ONLY_FIELDS | GAZEBO_APPLIED_FIELDS | OBSERVATION_ONLY_FIELDS
    assert all_fields == set(RandomizationDraw.__dataclass_fields__), (
        "domain_randomizer.py: RandomizationDraw field added without updating its "
        "MODEL_ONLY/GAZEBO_APPLIED/OBSERVATION_ONLY classification"
    )
    result = {}
    for name in RandomizationDraw.__dataclass_fields__:
        if name in MODEL_ONLY_FIELDS:
            result[name] = "model_only"
        elif name in GAZEBO_APPLIED_FIELDS:
            result[name] = "gazebo_applied"
        else:
            result[name] = "observation_only"
    return result


def _sample(rng: np.random.RandomState, r) -> float:
    return float(rng.uniform(r[0], r[1]))


def sample_draw(seed: int, cfg: DomainRandomizationConfig) -> RandomizationDraw:
    if not cfg.enabled:
        return RandomizationDraw()
    rng = np.random.RandomState(seed)
    return RandomizationDraw(
        mass_scale=_sample(rng, cfg.mass_scale_range),
        friction_scale=_sample(rng, cfg.friction_scale_range),
        wheel_radius_scale=_sample(rng, cfg.wheel_radius_scale_range),
        steering_gain=_sample(rng, cfg.steering_gain_range),
        steering_delay_sec=_sample(rng, cfg.steering_delay_sec_range),
        velocity_response_scale=_sample(rng, cfg.velocity_response_scale_range),
        command_latency_sec=_sample(rng, cfg.command_latency_sec_range),
        lidar_range_noise_std_m=_sample(rng, cfg.lidar_range_noise_std_m_range),
        lidar_dropout_prob=_sample(rng, cfg.lidar_dropout_prob_range),
        odometry_noise_std=_sample(rng, cfg.odometry_noise_std_range),
        sensor_frame_drop_prob=_sample(rng, cfg.sensor_frame_drop_prob_range),
    )


# Fixed-benchmark `dynamics:` override keys this package can ACTUALLY apply
# (section P1-3/P1-5). Any OTHER key in a benchmark YAML's `dynamics:` block
# raises -- silently ignoring an override the user asked for would make the
# benchmark's stated OOD condition a no-op, invisibly. See
# docs/ARCHITECTURE.md's "Domain randomization: where it actually applies"
# section for the honest boundary this whitelist encodes: mass/friction/
# wheel-radius/steering-gain are real RobotConfig scale factors;
# command_latency_sec is a real N-step command delay in
# environment_node.py's control loop.
#
# steering-specific delay and LiDAR/odometry noise DO have real consumers
# now (section P1-10: apply_lidar_noise/apply_odometry_noise/
# steering_lag_alpha, wired into environment_node.py's PROCEDURAL
# domain-randomization path only) -- but they are DELIBERATELY still
# excluded from THIS whitelist (fixed-BENCHMARK dynamics/sensor overrides),
# because fixed benchmarks are for EVALUATION and domain randomization is
# TRAIN-ONLY by requirement; a benchmark YAML requesting sensor noise would
# contradict that boundary, not just be an unimplemented feature.
SUPPORTED_DYNAMICS_OVERRIDE_KEYS = frozenset({
    "friction_scale", "mass_scale", "wheel_radius_scale", "steering_gain", "command_latency_sec",
})


def apply_dynamics_overrides(base: RobotConfig, overrides: dict) -> RobotConfig:
    """Fixed-benchmark ``dynamics:`` overrides -> a new RobotConfig (the
    ``command_latency_sec`` key is NOT a RobotConfig field -- read it
    separately via :func:`command_latency_steps`). Raises ``ValueError``
    (never silently ignores) for any unsupported key."""
    unknown = set(overrides) - SUPPORTED_DYNAMICS_OVERRIDE_KEYS
    if unknown:
        raise ValueError(
            f"unsupported dynamics override key(s) {sorted(unknown)} -- "
            f"supported: {sorted(SUPPORTED_DYNAMICS_OVERRIDE_KEYS)}"
        )
    friction_scale = float(overrides.get("friction_scale", 1.0))
    mass_scale = float(overrides.get("mass_scale", 1.0))
    wheel_radius_scale = float(overrides.get("wheel_radius_scale", 1.0))
    steering_gain = float(overrides.get("steering_gain", 1.0))
    return RobotConfig(
        name=base.name, wheelbase_m=base.wheelbase_m, track_width_m=base.track_width_m,
        wheel_radius_m=base.wheel_radius_m * wheel_radius_scale,
        length_m=base.length_m, width_m=base.width_m, height_m=base.height_m,
        mass_kg=base.mass_kg * mass_scale,
        steering_limit_deg=base.steering_limit_deg,
        max_forward_speed_mps=base.max_forward_speed_mps, min_forward_speed_mps=base.min_forward_speed_mps,
        accel_limit_mps2=base.accel_limit_mps2 * friction_scale,
        brake_decel_mps2=base.brake_decel_mps2 * friction_scale,
        steering_rate_deg_s=base.steering_rate_deg_s * steering_gain,
        speed_lag_tau_sec=base.speed_lag_tau_sec, collision_radius_m=base.collision_radius_m,
    )


def command_latency_steps(overrides: dict, time_delta_sec: float) -> int:
    """``command_latency_sec`` -> integer control-tick delay (rounded), used
    by ``environment_node.py``'s command-delay queue. 0 (the default, and
    the case for every profile that doesn't set this override) means "no
    queue, publish immediately" -- byte-identical to the pre-override
    control loop."""
    latency = float(overrides.get("command_latency_sec", 0.0))
    if latency < 0.0:
        raise ValueError(f"dynamics.command_latency_sec must be >= 0, got {latency}")
    return max(0, round(latency / max(time_delta_sec, 1e-6)))


# Fixed-BENCHMARK sensor overrides stay unsupported (section P1-5's
# disclosed gap) even though a real LiDAR/odometry-noise consumer now
# exists for the PROCEDURAL domain-randomization path (section P1-10) --
# benchmarks are for EVALUATION, randomization is TRAIN-ONLY, so wiring a
# benchmark `sensor:` override into the SAME noise consumer would
# contradict that boundary. A non-empty `sensor:` override block in a
# benchmark YAML must still raise, not silently do nothing.
def check_sensor_overrides_supported(overrides: dict) -> None:
    if overrides:
        raise ValueError(
            f"sensor overrides {sorted(overrides)} are not supported for FIXED benchmark scenarios "
            "(evaluation must stay noise-free/reproducible; sensor noise is a TRAIN-ONLY domain-"
            "randomization axis, see sample_draw/apply_lidar_noise/apply_odometry_noise) -- remove "
            "them from this scenario's `sensor:` block."
        )


def apply_lidar_noise(ranges: np.ndarray, draw: RandomizationDraw, rng: np.random.RandomState,
                       max_range_m: float) -> np.ndarray:
    """section P1-10: additive Gaussian range noise + per-beam dropout, for
    the AGENT'S OBSERVATION only -- callers MUST NOT use this for
    collision/risk computation, which needs the real (noise-free) ground
    truth. A dropped beam reads as max_range_m (no return), the standard
    LiDAR dropout convention -- matching how a genuinely absent reflection
    already looks to front_and_full_state's own binning."""
    if draw.lidar_range_noise_std_m <= 0.0 and draw.lidar_dropout_prob <= 0.0:
        return ranges
    noisy = ranges.astype(np.float32, copy=True)
    if draw.lidar_range_noise_std_m > 0.0:
        noisy = noisy + rng.normal(0.0, draw.lidar_range_noise_std_m, size=noisy.shape).astype(np.float32)
    noisy = np.clip(noisy, 0.0, max_range_m)
    if draw.lidar_dropout_prob > 0.0:
        dropped = rng.uniform(0.0, 1.0, size=noisy.shape) < draw.lidar_dropout_prob
        noisy = np.where(dropped, np.float32(max_range_m), noisy)
    return noisy.astype(ranges.dtype)


def apply_odometry_noise(v_mps: float, yaw_rate_radps: float, draw: RandomizationDraw,
                          rng: np.random.RandomState) -> tuple:
    """section P1-10: additive Gaussian noise on the odometry-sourced
    speed/yaw-rate OBSERVATION fields (state[84]/state[85] -- see
    CLAUDE.md's State layout) -- for the AGENT'S OBSERVATION only, never
    the ground-truth self._robot_twist used elsewhere (reward, collision,
    privileged risk labels)."""
    if draw.odometry_noise_std <= 0.0:
        return v_mps, yaw_rate_radps
    return (
        v_mps + float(rng.normal(0.0, draw.odometry_noise_std)),
        yaw_rate_radps + float(rng.normal(0.0, draw.odometry_noise_std)),
    )


def should_drop_sensor_frame(draw: RandomizationDraw, rng: np.random.RandomState) -> bool:
    """section P1-10: True with probability draw.sensor_frame_drop_prob --
    the caller's job is to hold the PREVIOUS observation frame instead of
    this tick's fresh one (a lost LiDAR/odometry message, the most common
    real-world sensor-driver failure mode), for the AGENT'S OBSERVATION
    only."""
    if draw.sensor_frame_drop_prob <= 0.0:
        return False
    return bool(rng.uniform(0.0, 1.0) < draw.sensor_frame_drop_prob)


def steering_lag_alpha(steering_delay_sec: float, time_delta_sec: float) -> float:
    """section P1-10: first-order (exponential) lag filter coefficient for
    ``steering_delay_sec`` (an "actuator gain/delay" domain-randomization
    axis -- steering_gain already scales the RATE LIMIT; this separately
    models a lag in how fast the ACTUATOR tracks a newly commanded angle,
    applied as ``filtered += alpha * (target - filtered)`` each tick).
    ``steering_delay_sec`` is treated as the filter's time CONSTANT tau:
    alpha = time_delta_sec / tau, clamped to [0, 1] (alpha=1 means "no lag,
    reaches target within one tick" -- tau <= time_delta_sec). 0 (the
    default, and the case for every profile that doesn't randomize this)
    returns 1.0 -- byte-identical to no filtering at all."""
    if steering_delay_sec <= 0.0:
        return 1.0
    return max(0.0, min(1.0, time_delta_sec / steering_delay_sec))


def rate_limit_speed(current_speed_mps: float, target_speed_mps: float,
                      robot: RobotConfig, dt_sec: float) -> float:
    """code review: friction_scale/velocity_response_scale (via
    apply_to_robot_config) scale robot.accel_limit_mps2/brake_decel_mps2,
    but nothing previously READ those two fields on the real command-publish
    path -- only the offline dynamics-rollout risk model did, so a
    randomized friction/velocity-response draw never actually changed how
    fast the robot's PUBLISHED speed command could ramp in Gazebo. This
    closes that gap: a first-order rate limiter (same accel-while-speeding-
    up / brake-while-slowing-down rate selection as
    ``trajectory/pure_pursuit.ackermann_swept_path``/hunter_se_cmd_prefilter),
    applied to the NOMINAL (pre-safety-guard) command in environment_node.py
    -- see that call site's docstring for why it must run BEFORE, never
    after, the safety guard (a guard-forced emergency stop must always be
    IMMEDIATE, never softened by a "realistic" ramp)."""
    rate = robot.accel_limit_mps2 if abs(target_speed_mps) >= abs(current_speed_mps) else robot.brake_decel_mps2
    max_delta = max(rate, 0.0) * max(dt_sec, 0.0)
    delta = float(target_speed_mps) - float(current_speed_mps)
    if abs(delta) <= max_delta:
        return float(target_speed_mps)
    return float(current_speed_mps) + math.copysign(max_delta, delta)


def apply_to_robot_config(base: RobotConfig, draw: RandomizationDraw) -> RobotConfig:
    """Returns a NEW RobotConfig with the draw applied -- the base config
    (and therefore config/robot/hunter_se.yaml) is never mutated."""
    return RobotConfig(
        name=base.name,
        wheelbase_m=base.wheelbase_m,
        track_width_m=base.track_width_m,
        wheel_radius_m=base.wheel_radius_m * draw.wheel_radius_scale,
        length_m=base.length_m, width_m=base.width_m, height_m=base.height_m,
        mass_kg=base.mass_kg * draw.mass_scale,
        steering_limit_deg=base.steering_limit_deg,
        max_forward_speed_mps=base.max_forward_speed_mps,
        min_forward_speed_mps=base.min_forward_speed_mps,
        accel_limit_mps2=base.accel_limit_mps2 * draw.friction_scale * draw.velocity_response_scale,
        brake_decel_mps2=base.brake_decel_mps2 * draw.friction_scale * draw.velocity_response_scale,
        steering_rate_deg_s=base.steering_rate_deg_s * draw.steering_gain,
        speed_lag_tau_sec=base.speed_lag_tau_sec,
        collision_radius_m=base.collision_radius_m,
    )
