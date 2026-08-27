"""Coverage for the fixed-shape sensor/localization noise model (drl_agent
-> hunter_kinodynamic_rl requirement 3): ``config/schema.py``'s
``SensorNoiseConfig`` and ``env/simulation/sensor_noise.py``.

Pure-function module -- no ROS needed, runs on a bare host checkout.
"""

import numpy as np
import pytest

from hunter_kinodynamic_rl.config.schema import ConfigError, ScenarioConfig, SensorNoiseConfig
from hunter_kinodynamic_rl.env.scenarios.seed_scheduler import SeedScheduler
from hunter_kinodynamic_rl.env.simulation import sensor_noise as sn


# --------------------------------------------------------------- schema
def test_sensor_noise_config_defaults_are_disabled_and_validate():
    cfg = SensorNoiseConfig()
    cfg.validate()
    assert cfg.enabled is False


@pytest.mark.parametrize("field_name", [
    "lidar_range_noise_std_m", "lidar_dropout_prob", "localization_xy_noise_std_m",
    "localization_yaw_noise_std_rad", "localization_drift_theta", "localization_drift_xy_sigma_m",
    "localization_drift_yaw_sigma_rad", "velocity_noise_std_mps", "yaw_rate_noise_std_radps",
    "steering_noise_std_rad",
])
def test_sensor_noise_config_rejects_negative_magnitudes(field_name):
    with pytest.raises(ConfigError):
        SensorNoiseConfig(**{field_name: -1.0}).validate()


def test_sensor_noise_config_rejects_out_of_range_dropout_and_negative_latency():
    with pytest.raises(ConfigError):
        SensorNoiseConfig(lidar_dropout_prob=1.5).validate()
    with pytest.raises(ConfigError):
        SensorNoiseConfig(localization_latency_steps=-1).validate()


# ----------------------------------------------------------- disabled == no-op
def test_disabled_config_leaves_every_quantity_byte_identical():
    cfg = SensorNoiseConfig(enabled=False, lidar_range_noise_std_m=0.5, localization_xy_noise_std_m=0.5,
                             velocity_noise_std_mps=0.5, steering_noise_std_rad=0.5,
                             localization_drift_xy_sigma_m=0.5)
    state = sn.reset_state(42, cfg)
    ranges = np.full(80, 5.0, dtype=np.float32)

    assert sn.measured_pose(state, cfg, 1.0, 2.0, 0.3) == (1.0, 2.0, 0.3)
    assert sn.measured_velocity(state, cfg, 0.5, 0.1) == (0.5, 0.1)
    assert sn.measured_steering(state, cfg, 0.05) == 0.05
    assert np.array_equal(sn.apply_lidar_noise(state, cfg, ranges, 10.0), ranges)
    sn.tick_drift(state, cfg, 0.1)
    assert state.drift_x == 0.0 and state.drift_y == 0.0 and state.drift_yaw == 0.0


# --------------------------------------------------------------- determinism
def test_reset_state_is_deterministic_given_the_same_episode_seed():
    cfg = SensorNoiseConfig(enabled=True, localization_xy_noise_std_m=0.05, lidar_range_noise_std_m=0.02)
    s1 = sn.reset_state(777, cfg)
    s2 = sn.reset_state(777, cfg)
    ranges = np.full(80, 5.0, dtype=np.float32)
    seq1 = [sn.measured_pose(s1, cfg, 0.0, 0.0, 0.0) for _ in range(5)]
    seq2 = [sn.measured_pose(s2, cfg, 0.0, 0.0, 0.0) for _ in range(5)]
    assert seq1 == seq2
    assert np.array_equal(sn.apply_lidar_noise(s1, cfg, ranges, 10.0), sn.apply_lidar_noise(s2, cfg, ranges, 10.0))


def test_different_episode_seeds_produce_different_noise_sequences():
    cfg = SensorNoiseConfig(enabled=True, localization_xy_noise_std_m=0.05)
    s1 = sn.reset_state(1, cfg)
    s2 = sn.reset_state(2, cfg)
    v1 = sn.measured_pose(s1, cfg, 0.0, 0.0, 0.0)
    v2 = sn.measured_pose(s2, cfg, 0.0, 0.0, 0.0)
    assert v1 != v2


def test_sensor_noise_rng_stream_is_independent_of_a_same_seed_domain_rand_stream():
    """The dedicated RNG derivation ((seed*1000003 + salt)) must not
    coincide with domain_randomization's own per-episode stream seed
    ((seed*7 + 1)) for any small seed -- a collision would silently
    couple two RNG streams meant to be independent."""
    for seed in range(200):
        sensor_seed = (seed * 1000003 + sn._SEED_SALT) & 0xFFFFFFFF
        domain_rand_seed = (seed * 7 + 1) & 0xFFFFFFFF
        assert sensor_seed != domain_rand_seed


# ------------------------------------------------------------ observation only
def test_measured_pose_and_velocity_perturb_only_their_own_return_values():
    """A pure-function sanity check standing in for the full
    observation/ground-truth separation (which requires the ROS
    environment_node and is exercised by the Docker-only integration
    check) -- these functions take explicit ground-truth values as
    arguments and have no way to mutate anything outside their own
    return value, so the caller's own ground-truth variables are
    necessarily untouched by construction."""
    cfg = SensorNoiseConfig(enabled=True, localization_xy_noise_std_m=0.1, localization_yaw_noise_std_rad=0.05,
                             velocity_noise_std_mps=0.1, yaw_rate_noise_std_radps=0.05,
                             steering_noise_std_rad=0.02)
    state = sn.reset_state(5, cfg)
    gt_x, gt_y, gt_yaw = 3.0, -2.0, 0.7
    gt_v, gt_yaw_rate = 0.8, 0.1
    gt_steering = 0.05

    mx, my, myaw = sn.measured_pose(state, cfg, gt_x, gt_y, gt_yaw)
    mv, myr = sn.measured_velocity(state, cfg, gt_v, gt_yaw_rate)
    msteer = sn.measured_steering(state, cfg, gt_steering)

    # Ground truth locals are untouched (still exactly what was passed in).
    assert (gt_x, gt_y, gt_yaw) == (3.0, -2.0, 0.7)
    assert (gt_v, gt_yaw_rate) == (0.8, 0.1)
    assert gt_steering == 0.05
    # The measured values are actually perturbed (noise magnitudes are
    # large enough here that an exact match would indicate a wiring bug).
    assert (mx, my, myaw) != (gt_x, gt_y, gt_yaw)
    assert (mv, myr) != (gt_v, gt_yaw_rate)
    assert msteer != gt_steering


def test_apply_lidar_noise_never_mutates_the_input_array():
    cfg = SensorNoiseConfig(enabled=True, lidar_range_noise_std_m=0.1, lidar_bias_m=0.05)
    state = sn.reset_state(1, cfg)
    ranges = np.full(80, 5.0, dtype=np.float32)
    original = ranges.copy()
    noisy = sn.apply_lidar_noise(state, cfg, ranges, 10.0)
    assert np.array_equal(ranges, original)  # input untouched
    assert not np.array_equal(noisy, ranges)  # output actually perturbed


def test_lidar_dropout_sets_dropped_beams_to_max_range():
    cfg = SensorNoiseConfig(enabled=True, lidar_dropout_prob=1.0)  # every beam drops
    state = sn.reset_state(1, cfg)
    ranges = np.full(80, 3.0, dtype=np.float32)
    noisy = sn.apply_lidar_noise(state, cfg, ranges, 10.0)
    assert np.all(noisy == np.float32(10.0))


def test_lidar_noise_clips_to_valid_range():
    cfg = SensorNoiseConfig(enabled=True, lidar_bias_m=100.0)  # huge bias -- must clip
    state = sn.reset_state(1, cfg)
    ranges = np.full(80, 5.0, dtype=np.float32)
    noisy = sn.apply_lidar_noise(state, cfg, ranges, 10.0)
    assert np.all(noisy <= 10.0)


# ------------------------------------------------------------------ latency
def test_localization_latency_ramps_up_then_delays_by_the_configured_step_count():
    cfg = SensorNoiseConfig(enabled=True, localization_latency_steps=2)
    state = sn.reset_state(1, cfg)
    # Feed a distinguishable sequence of "current tick" ground-truth poses.
    inputs = [(float(i), 0.0, 0.0) for i in range(6)]
    outputs = [sn.measured_pose(state, cfg, *p) for p in inputs]
    # Not enough history yet on the first two calls -- ramps from the
    # oldest available reading rather than blocking/raising.
    assert outputs[0] == (0.0, 0.0, 0.0)
    # Once the buffer has filled (latency_steps + 1 == 3 entries), output
    # at tick i must equal the ground-truth input from i - latency_steps.
    for i in range(2, 6):
        assert outputs[i] == (float(i - 2), 0.0, 0.0)


def test_zero_latency_returns_the_current_tick_unchanged():
    cfg = SensorNoiseConfig(enabled=True, localization_latency_steps=0)
    state = sn.reset_state(1, cfg)
    assert sn.measured_pose(state, cfg, 1.0, 2.0, 0.0) == (1.0, 2.0, 0.0)
    assert sn.measured_pose(state, cfg, 9.0, 9.0, 0.0) == (9.0, 9.0, 0.0)


# --------------------------------------------------------------------- drift
def test_ou_drift_is_zero_at_episode_start_and_only_accumulates_after_ticking():
    cfg = SensorNoiseConfig(enabled=True, localization_drift_xy_sigma_m=0.05, localization_drift_yaw_sigma_rad=0.02)
    state = sn.reset_state(1, cfg)
    assert (state.drift_x, state.drift_y, state.drift_yaw) == (0.0, 0.0, 0.0)
    for _ in range(50):
        sn.tick_drift(state, cfg, 0.1)
    assert (state.drift_x, state.drift_y, state.drift_yaw) != (0.0, 0.0, 0.0)


def test_ou_drift_stays_bounded_by_mean_reversion_over_a_long_episode():
    """theta > 0 mean-reverts -- drift must not grow unbounded (a plain
    Brownian/random-walk bug would let it wander arbitrarily far over
    thousands of ticks)."""
    cfg = SensorNoiseConfig(enabled=True, localization_drift_theta=2.0, localization_drift_xy_sigma_m=0.1)
    state = sn.reset_state(1, cfg)
    max_abs = 0.0
    for _ in range(5000):
        sn.tick_drift(state, cfg, 0.1)
        max_abs = max(max_abs, abs(state.drift_x), abs(state.drift_y))
    # Stationary std of an OU process is sigma/sqrt(2*theta*dt-independent) --
    # for these params that's well under 1m; a random-walk bug would blow
    # well past this over 5000 ticks.
    assert max_abs < 2.0


# -------------------------------------------- checkpoint/resume determinism
def test_sensor_noise_state_derives_purely_from_episode_seed_so_resume_needs_no_extra_state():
    """The core reproducibility claim this module's docstring makes: since
    reset_state() depends on NOTHING but the episode seed, resuming a
    trainer from a checkpoint (which persists only SeedScheduler's own
    episode_index -- see training/trainer_base.py's _save_checkpoint)
    automatically reproduces the correct noise RNG for every subsequent
    episode, with zero additional state for THIS module to save/restore.
    Mirrors test_seed_scheduler.py's own
    test_episode_index_advances_and_resume_continues_sequence."""
    cfg = ScenarioConfig(train_seed_range=[0, 9999])
    noise_cfg = SensorNoiseConfig(enabled=True, localization_xy_noise_std_m=0.05)

    sched = SeedScheduler(run_seed=3, scenario_cfg=cfg, mode="train")
    for _ in range(5):
        sched.next_seed()
    checkpoint_state = sched.state_dict()

    # Resumed process: reconstruct the scheduler from checkpointed state,
    # draw the next episode seed, and build the sensor-noise state from it.
    resumed_sched = SeedScheduler.from_state_dict(checkpoint_state, cfg)
    resumed_seed = resumed_sched.next_seed()
    resumed_noise_state = sn.reset_state(resumed_seed, noise_cfg)
    resumed_sample = sn.measured_pose(resumed_noise_state, noise_cfg, 1.0, 2.0, 0.0)

    # Uninterrupted process: never "restarted", just keeps drawing.
    fresh_sched = SeedScheduler(run_seed=3, scenario_cfg=cfg, mode="train")
    for _ in range(5):
        fresh_sched.next_seed()
    fresh_seed = fresh_sched.next_seed()
    fresh_noise_state = sn.reset_state(fresh_seed, noise_cfg)
    fresh_sample = sn.measured_pose(fresh_noise_state, noise_cfg, 1.0, 2.0, 0.0)

    assert resumed_seed == fresh_seed
    assert resumed_sample == fresh_sample


def test_drift_disabled_when_sigma_is_zero_even_if_enabled():
    cfg = SensorNoiseConfig(enabled=True, localization_drift_xy_sigma_m=0.0, localization_drift_yaw_sigma_rad=0.0)
    state = sn.reset_state(1, cfg)
    for _ in range(20):
        sn.tick_drift(state, cfg, 0.1)
    assert (state.drift_x, state.drift_y, state.drift_yaw) == (0.0, 0.0, 0.0)


# ---------------------------------------------- exact discrete OU (requirement 6)
def test_ou_step_theta_zero_reduces_to_plain_brownian_motion():
    """theta=0 (no mean reversion) must exactly match the closed-form
    driftless-diffusion case (current + N(0, sigma*sqrt(dt))), which is also
    what Euler-Maruyama computes for theta=0 -- this is the one case where
    the exact-OU rewrite and the old Euler formula must agree exactly."""
    # Same RNG stream, same math -- both must draw the identical sequence.
    rng_a, rng_b = np.random.RandomState(11), np.random.RandomState(11)
    x_a = x_b = 0.0
    for _ in range(10):
        x_a = sn._ou_step(x_a, 0.0, 0.2, 0.1, rng_a)
        x_b = x_b - 0.0 * x_b * 0.1 + float(rng_b.normal(0.0, 0.2 * np.sqrt(0.1)))
    assert x_a == pytest.approx(x_b)


def test_ou_step_dt_zero_is_a_no_op():
    rng = np.random.RandomState(0)
    assert sn._ou_step(1.234, 2.0, 0.5, 0.0, rng) == 1.234


def test_ou_step_sigma_zero_is_a_no_op_even_with_nonzero_theta():
    """Mirrors reset_state's documented "a zero sigma disables its own
    drift axis entirely (theta is then irrelevant)" contract -- the exact
    discrete update must NOT apply mean-reversion decay on its own when
    sigma is zero (state should stay put, not decay toward 0)."""
    rng = np.random.RandomState(0)
    assert sn._ou_step(5.0, 3.0, 0.0, 0.1, rng) == 5.0


def test_ou_step_large_theta_dt_stays_finite_and_decays_toward_zero_noise_band():
    """A large theta*dt must not overflow/diverge the way an explicit-Euler
    step (`current - theta*current*dt`) would once theta*dt > 2 (it
    oscillates, then diverges) -- exp(-theta*dt) underflows gracefully to
    0.0 instead."""
    rng = np.random.RandomState(0)
    current = 1000.0
    for _ in range(200):
        current = sn._ou_step(current, theta=1e6, sigma_m=0.1, dt_sec=0.1, rng=rng)
        assert np.isfinite(current)
    # theta*dt = 1e5 -- decay is essentially total, state should have
    # collapsed near the (small) stationary noise band, nowhere near 1000.
    assert abs(current) < 10.0


def test_ou_step_reproducible_for_same_seed_and_finite_over_a_long_run():
    cfg = SensorNoiseConfig(enabled=True, localization_drift_theta=0.5, localization_drift_xy_sigma_m=0.2)

    def run():
        state = sn.reset_state(42, cfg)
        for _ in range(20000):
            sn.tick_drift(state, cfg, 0.1)
        return state.drift_x, state.drift_y, state.drift_yaw

    a = run()
    b = run()
    assert a == b
    assert all(np.isfinite(v) for v in a)


def test_ou_step_zero_theta_long_run_stays_finite():
    """theta=0 is a driftless random walk (unbounded variance as t grows,
    unlike theta>0's mean-reverting stationary distribution) -- it must
    still never produce a non-finite value over a long run."""
    rng = np.random.RandomState(3)
    current = 0.0
    for _ in range(20000):
        current = sn._ou_step(current, 0.0, 0.05, 0.1, rng)
    assert np.isfinite(current)
