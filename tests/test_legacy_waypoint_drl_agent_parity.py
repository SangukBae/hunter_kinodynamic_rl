"""section P0-4: legacy_waypoint controller parity against drl_agent's own
hybrid stop/yield contract (CLAUDE.md's Action section /
drl_agent/config/environment_curriculum.yaml). The PREVIOUS
trajectory_executor.py implementation hand-rolled its own yield logic on
top of the SUPERSEDED ``pure_pursuit.waypoint_to_command`` ramp -- wrong
yield threshold (0.0 instead of drl_agent's 0.3), no MOVE-mode lookahead
floor at all, a speed FLOOR instead of a CAP in yield mode, and a
hardcoded 0.5 speed_steer_factor instead of drl_agent's 0.6. This file
proves ``trajectory_executor.execute()`` now calls
``pure_pursuit.hybrid_action_to_command`` (the verbatim-copied,
ALREADY-parity-correct function that was simply left unused) with
drl_agent's own constants.
"""

import math

import pytest

from hunter_kinodynamic_rl.config.schema import ActionSpaceConfig, RobotConfig, TrajectoryConfig
from hunter_kinodynamic_rl.trajectory import pure_pursuit, trajectory_executor


def make_robot() -> RobotConfig:
    return RobotConfig(
        name="hunter_se", wheelbase_m=0.547696, track_width_m=0.503404,
        wheel_radius_m=0.136, length_m=0.76, width_m=0.4, height_m=0.14,
        mass_kg=42.0, steering_limit_deg=21.58, max_forward_speed_mps=2.0,
        min_forward_speed_mps=0.0, accel_limit_mps2=6.0, brake_decel_mps2=6.0,
        steering_rate_deg_s=200.0, speed_lag_tau_sec=0.0,
    )


def _cfg(**overrides) -> ActionSpaceConfig:
    return ActionSpaceConfig(mode="legacy_waypoint", **overrides)


def test_default_legacy_controller_constants_match_drl_agent_exactly():
    """The literal drl_agent numbers (environment_curriculum.yaml) this
    package's ActionSpaceConfig defaults must reproduce."""
    cfg = ActionSpaceConfig()
    assert cfg.legacy_yield_threshold == pytest.approx(0.3)       # yield_reward.action_threshold
    assert cfg.legacy_lookahead_min_m == pytest.approx(0.8)       # controller_lookahead_min_m
    assert cfg.legacy_v_move_min_mps == pytest.approx(0.35)       # controller_v_move_min_mps
    assert cfg.legacy_yield_creep_mps == pytest.approx(0.0)       # controller_yield_creep_mps
    assert cfg.legacy_speed_steer_factor == pytest.approx(0.6)    # controller_speed_steer_factor


def test_trajectory_executor_delegates_to_hybrid_action_to_command_directly():
    """trajectory_executor.execute() must produce BYTE-IDENTICAL output to
    calling pure_pursuit.hybrid_action_to_command() directly with the same
    action_cfg-derived constants -- proving the executor is a thin wrapper,
    not a re-implementation that could silently drift again."""
    robot = make_robot()
    cfg = _cfg()
    action = [0.2, -0.6, -0.1]

    got = trajectory_executor.execute(action, cfg, TrajectoryConfig(), robot)

    want_speed, want_steering, _theta, _info = pure_pursuit.hybrid_action_to_command(
        action,
        actions_low=[cfg.legacy_r_min_m, -cfg.legacy_theta_max_rad, -1.0],
        actions_high=[cfg.legacy_r_max_m, cfg.legacy_theta_max_rad, 1.0],
        wheelbase_m=robot.wheelbase_m, steering_limit_rad=robot.steering_limit_rad,
        cruise_speed_mps=robot.max_forward_speed_mps, speed_steer_factor=cfg.legacy_speed_steer_factor,
        yield_enabled=cfg.legacy_yield_enabled, yield_threshold=cfg.legacy_yield_threshold,
        lookahead_min_m=cfg.legacy_lookahead_min_m, v_move_min_mps=cfg.legacy_v_move_min_mps,
        yield_creep_speed_mps=cfg.legacy_yield_creep_mps,
    )
    assert got.speed_mps == pytest.approx(want_speed)
    assert got.steering_rad == pytest.approx(want_steering)


def test_move_mode_floors_lookahead_and_speed():
    """A short-r, non-yielding action must have its EFFECTIVE waypoint
    distance floored to legacy_lookahead_min_m -- the policy cannot collapse
    the lookahead to near-zero outside yield mode. r=0.01m (near the
    legacy_r_min_m=0.0 floor) with yield well below threshold."""
    robot = make_robot()
    cfg = _cfg()
    # action[0] normalized near -1 -> r near legacy_r_min_m=0.0; action[2]=-1.0 -> yield=False.
    cmd = trajectory_executor.execute([-0.99, 0.0, -1.0], cfg, TrajectoryConfig(), robot)
    # With r floored to 0.8m and theta=0 (straight ahead), steering ~ 0 and
    # speed should be at/near cruise (no steer penalty) -- well above the
    # v_move_min floor, proving the lookahead floor let a real speed through
    # rather than collapsing near-zero-L behavior.
    assert cmd.speed_mps >= cfg.legacy_v_move_min_mps - 1e-9


def test_move_mode_never_produces_the_yield_creep_speed_from_geometry_alone():
    """Even a maximally-steered MOVE-mode command must never fall below
    v_move_min_mps -- the policy cannot stop outside yield mode."""
    robot = make_robot()
    cfg = _cfg()
    cmd = trajectory_executor.execute([0.5, 1.0, -1.0], cfg, TrajectoryConfig(), robot)  # full steer, not yielding
    assert cmd.speed_mps >= cfg.legacy_v_move_min_mps - 1e-9


def test_yield_threshold_matches_drl_agent_not_zero():
    """The core regression: yield engages only at action[2] >= 0.3 (drl_agent's
    threshold), NOT >= 0.0 (the previous, wrong hunter_kinodynamic_rl
    behavior). A yield scalar of 0.1 (between the two thresholds) must
    still be MOVE mode."""
    robot = make_robot()
    cfg = _cfg()
    move_cmd = trajectory_executor.execute([0.5, 0.0, 0.1], cfg, TrajectoryConfig(), robot)
    assert move_cmd.speed_mps >= cfg.legacy_v_move_min_mps - 1e-9  # still MOVE mode -> floored, never near-zero

    yield_cmd = trajectory_executor.execute([0.5, 0.0, 0.31], cfg, TrajectoryConfig(), robot)
    assert yield_cmd.speed_mps == pytest.approx(cfg.legacy_yield_creep_mps, abs=1e-6)  # now YIELD mode


def test_yield_mode_caps_speed_at_yield_creep_not_just_floors_it():
    """The core regression: drl_agent's yield mode CAPS speed at
    yield_creep_mps (a ceiling on an otherwise-computed geometric speed),
    not a FLOOR (which would let speed stay high whenever geometry alone
    would already produce something above the creep value) -- with the
    default yield_creep_mps=0.0, yielding must mean a genuine full stop
    regardless of the commanded waypoint geometry."""
    robot = make_robot()
    cfg = _cfg()
    # theta=0 (straight ahead, zero steer penalty) would otherwise drive at
    # full cruise speed -- yield must override that down to 0, not just
    # "raise it to at least 0" (a no-op floor, since speed is already >= 0).
    cmd = trajectory_executor.execute([0.9, 0.0, 1.0], cfg, TrajectoryConfig(), robot)
    assert cmd.speed_mps == pytest.approx(0.0, abs=1e-6)


def test_speed_steer_factor_matches_drl_agent():
    """A steered (non-zero theta) command's speed must reflect drl_agent's
    0.6 speed_steer_factor, not the previous hardcoded 0.5."""
    robot = make_robot()
    cfg = _cfg()
    # theta near its max, not yielding, r generous enough to avoid the
    # lookahead floor dominating the geometry.
    cmd = trajectory_executor.execute([1.0, 1.0, -1.0], cfg, TrajectoryConfig(), robot)
    r = cfg.legacy_r_max_m
    theta = cfg.legacy_theta_max_rad
    x_wp, y_wp = r * math.cos(theta), r * math.sin(theta)
    steering = math.atan2(2.0 * y_wp * robot.wheelbase_m, (x_wp ** 2 + y_wp ** 2))
    steering = max(-robot.steering_limit_rad, min(robot.steering_limit_rad, steering))
    steer_ratio = abs(steering) / robot.steering_limit_rad
    expected_speed = max(
        robot.max_forward_speed_mps * (1.0 - cfg.legacy_speed_steer_factor * steer_ratio),
        cfg.legacy_v_move_min_mps,
    )
    assert cmd.speed_mps == pytest.approx(expected_speed, abs=1e-4)
