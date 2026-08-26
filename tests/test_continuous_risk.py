"""section item-6: continuous (segment-based, not just sampled-endpoint)
closest-approach/collision-time regression coverage.

The core scenario every ``_continuous`` function exists for: a
:class:`~hunter_kinodynamic_rl.dynamics.ackermann_rollout.Rollout` only
samples the ego path at discrete instants, and a moving obstacle only needs
to cross the ego's path BETWEEN two samples for a genuine collision to be
completely invisible to the pre-item-6 (sampled-endpoint-only) checks --
both endpoints individually report large clearance, yet the two are
provably co-located somewhere strictly between them.
"""

import math

import pytest

from hunter_kinodynamic_rl.config.schema import RiskConfig, RobotConfig
from hunter_kinodynamic_rl.dynamics.ackermann_rollout import Rollout, RolloutPoint
from hunter_kinodynamic_rl.risk import boundary, future_clearance, segment_math, trajectory_risk, ttc
from hunter_kinodynamic_rl.risk.labels import DynamicObstacle
from hunter_kinodynamic_rl.robot.interface import VehicleState


def make_robot() -> RobotConfig:
    return RobotConfig(
        name="hunter_se", wheelbase_m=0.547696, track_width_m=0.503404,
        wheel_radius_m=0.136, length_m=0.76, width_m=0.4, height_m=0.14,
        mass_kg=42.0, steering_limit_deg=21.58, max_forward_speed_mps=2.0,
        min_forward_speed_mps=0.0, accel_limit_mps2=6.0, brake_decel_mps2=6.0,
        steering_rate_deg_s=200.0, speed_lag_tau_sec=0.0,
    )


# --------------------------------------------------------- segment_math primitives
def test_closest_approach_finds_the_interior_minimum():
    s, dist = segment_math.closest_approach_on_segment(-1.0, -5.0, 1.0, 5.0)
    assert s == pytest.approx(0.5)
    assert dist == pytest.approx(0.0, abs=1e-9)


def test_closest_approach_clamps_to_an_endpoint_when_the_unclamped_minimum_is_outside_0_1():
    # P(s) = (2,2) + s*(1,1) -- moving AWAY from the origin for all s>=0,
    # so the true closest point on the infinite line is at s<0, clamped to 0.
    s, dist = segment_math.closest_approach_on_segment(2.0, 2.0, 3.0, 3.0)
    assert s == pytest.approx(0.0)
    assert dist == pytest.approx(math.hypot(2.0, 2.0))


def test_first_crossing_below_radius_finds_the_interior_crossing():
    s = segment_math.first_crossing_below_radius(-1.0, -5.0, 1.0, 5.0, radius=0.6)
    assert s is not None
    assert 0.0 < s < 0.5  # crosses INTO the radius before the exact-zero midpoint


def test_first_crossing_below_radius_none_when_segment_never_gets_close_enough():
    s = segment_math.first_crossing_below_radius(10.0, 10.0, 11.0, 11.0, radius=0.5)
    assert s is None


def test_first_crossing_below_radius_handles_a_stationary_relative_position():
    # dx=dy=0 -- P(s) constant over the whole segment.
    assert segment_math.first_crossing_below_radius(0.0, 0.0, 0.0, 0.0, radius=0.1) == pytest.approx(0.0)
    assert segment_math.first_crossing_below_radius(5.0, 0.0, 5.0, 0.0, radius=0.1) is None


# ------------------------------------------------------- the core item-6 regression
def _crossing_paths_rollout_and_obstacle():
    """Ego travels in a straight line from (0,0) at t=0 to (2,0) at t=1
    (one rollout sample, interpolated with t0_state). A y=+5 -> y=-5
    obstacle at x=1.0 crosses y=0 EXACTLY at t=0.5, s=0.5 -- precisely
    where the ego is at that instant too ((1,0)). Both the t=0 and t=1
    samples show the two ~5.1 m apart (obviously "safe" to any
    endpoint-only check); the exact midpoint is a dead-center overlap."""
    t0_state = VehicleState(x=0.0, y=0.0)
    rollout = Rollout(points=[RolloutPoint(t_sec=1.0, state=VehicleState(x=2.0, y=0.0, v=2.0))])
    obstacle = DynamicObstacle(x0=1.0, y0=5.0, vx=0.0, vy=-10.0, radius=0.3)
    return t0_state, rollout, obstacle


def test_discrete_endpoints_both_report_large_clearance_and_no_collision():
    """Sanity check establishing the bug this whole test module is about:
    the PRE-item-6 (sampled-endpoint-only) primitives see nothing wrong."""
    t0_state, rollout, obstacle = _crossing_paths_rollout_and_obstacle()
    clearance = future_clearance.min_clearance(rollout, 0.3, [obstacle], t0_state=t0_state)
    assert clearance > 4.0  # ~5.1 - 0.6, both endpoints "far"
    assert ttc.time_to_collision_or_none(rollout, 0.3, [obstacle], horizon_sec=1.0, t0_state=t0_state) is None
    assert ttc.collision_within_horizon(rollout, 0.3, [obstacle], horizon_sec=1.0, t0_state=t0_state) is False


def test_continuous_min_clearance_detects_the_mid_segment_collision():
    t0_state, rollout, obstacle = _crossing_paths_rollout_and_obstacle()
    clearance = future_clearance.min_clearance_continuous(rollout, 0.3, [obstacle], t0_state=t0_state)
    # exact overlap at the midpoint minus both radii: 0 - 0.3 - 0.3
    assert clearance == pytest.approx(-0.6, abs=1e-6)


def test_continuous_ttc_detects_the_mid_segment_collision():
    t0_state, rollout, obstacle = _crossing_paths_rollout_and_obstacle()
    t = ttc.time_to_collision_or_none_continuous(rollout, 0.3, [obstacle], horizon_sec=1.0, t0_state=t0_state)
    assert t is not None
    assert 0.0 < t < 1.0  # a genuine interior crossing time, not snapped to either sample
    assert ttc.collision_within_horizon_continuous(rollout, 0.3, [obstacle], horizon_sec=1.0, t0_state=t0_state)


def test_continuous_never_reports_a_worse_result_than_discrete_on_a_clearly_safe_trajectory():
    """The continuous check must never be MORE optimistic than the discrete
    one -- checking a continuum can only find an equal-or-worse (smaller
    clearance / earlier collision) result, never a better one."""
    robot = make_robot()
    from hunter_kinodynamic_rl.dynamics import ackermann_rollout
    from hunter_kinodynamic_rl.config.schema import DynamicsConfig
    dyn = DynamicsConfig(horizon_sec=2.0, dt_sec=0.1, model_actuator_lag=False)
    state = VehicleState(v=1.0, steering=0.0)
    rollout = ackermann_rollout.rollout_constant_target(state, 1.0, 0.0, robot, dyn)
    far_obstacle = DynamicObstacle(x0=50.0, y0=50.0, radius=0.3)

    discrete = future_clearance.min_clearance(rollout, 0.3, [far_obstacle], t0_state=state)
    continuous = future_clearance.min_clearance_continuous(rollout, 0.3, [far_obstacle], t0_state=state)
    assert continuous <= discrete

    discrete_ttc = ttc.time_to_collision(rollout, 0.3, [far_obstacle], horizon_sec=2.0, t0_state=state)
    continuous_ttc = ttc.time_to_collision_continuous(rollout, 0.3, [far_obstacle], horizon_sec=2.0, t0_state=state)
    assert continuous_ttc <= discrete_ttc


# ------------------------------------------------------------- horizon-inclusive rule
def test_continuous_min_clearance_respects_the_clearance_horizon():
    """The continuous check must clip to horizon_sec exactly like the
    discrete one -- a mid-segment close approach that occurs AFTER the
    configured horizon must not count."""
    t0_state, rollout, obstacle = _crossing_paths_rollout_and_obstacle()
    # collision is at t=0.5 -- a horizon of 0.3 must exclude it entirely
    clipped = future_clearance.min_clearance_continuous(
        rollout, 0.3, [obstacle], horizon_sec=0.3, t0_state=t0_state)
    assert clipped > 0.0
    full = future_clearance.min_clearance_continuous(rollout, 0.3, [obstacle], t0_state=t0_state)
    assert full < clipped


def test_continuous_ttc_never_reports_a_time_past_the_horizon():
    t0_state, rollout, obstacle = _crossing_paths_rollout_and_obstacle()
    t = ttc.time_to_collision_or_none_continuous(rollout, 0.3, [obstacle], horizon_sec=0.3, t0_state=t0_state)
    assert t is None  # collision at t=0.5 is past the 0.3s horizon


def test_continuous_collision_exactly_at_the_ttc_horizon_is_not_lost():
    """Mirrors test_p0_2_risk_correctness.py's exact-horizon-boundary
    coverage for the discrete primitive -- the continuous one must keep the
    same horizon-INCLUSIVE convention."""
    t0_state = VehicleState(x=0.0, y=0.0)
    rollout = Rollout(points=[RolloutPoint(t_sec=2.0, state=VehicleState(x=1.0, y=0.0))])
    obstacle = DynamicObstacle(x0=1.0, y0=0.0, radius=0.0)  # touches exactly at t=2.0 (ego reaches x=1 then)
    t = ttc.time_to_collision_or_none_continuous(rollout, 0.0, [obstacle], horizon_sec=2.0, t0_state=t0_state)
    assert t == pytest.approx(2.0)


# ------------------------------------------------------------------- world boundary
def test_boundary_continuous_never_reports_a_worse_result_than_discrete():
    """See segment_math/boundary.py's own docstrings: for a STRAIGHT-line
    interpolated segment against a CONVEX region (the square world
    boundary), the continuous minimum can never actually be worse than the
    discrete endpoint minimum (a straight chord between two interior points
    of a convex region never exits it) -- but must never be BETTER
    (over-optimistic) either. This asserts exact equality on a representative
    scenario, documenting that guarantee."""
    rollout = Rollout(points=[
        RolloutPoint(t_sec=1.0, state=VehicleState(x=1.0, y=0.0)),
        RolloutPoint(t_sec=2.0, state=VehicleState(x=2.0, y=0.0)),
    ])
    robot_pose = (0.0, 0.0, 0.0)
    discrete = boundary.min_boundary_clearance(rollout, 0.3, robot_pose, half_extent_m=5.0)
    continuous = boundary.min_boundary_clearance_continuous(rollout, 0.3, robot_pose, half_extent_m=5.0)
    assert continuous == pytest.approx(discrete)


def test_boundary_continuous_ttc_reports_an_exact_interior_crossing_time():
    """The continuous boundary TTC reports the EXACT wall-crossing instant
    via linear interpolation, not merely the next sample's timestamp --
    a real precision improvement even though (per the convexity argument
    above) the discrete check already correctly detects THAT a crossing
    happened, just not exactly WHEN."""
    rollout = Rollout(points=[RolloutPoint(t_sec=1.0, state=VehicleState(x=3.0, y=0.0))])
    robot_pose = (0.0, 0.0, 0.0)
    half_extent_m = 1.5  # wall at world x=1.5, ego crosses it at local x=1.5 -> t=0.5 (linear, v constant)
    discrete_t = boundary.boundary_time_to_exit_or_none(rollout, 0.0, robot_pose, half_extent_m, horizon_sec=1.0)
    continuous_t = boundary.boundary_time_to_exit_or_none_continuous(
        rollout, 0.0, robot_pose, half_extent_m, horizon_sec=1.0)
    assert discrete_t == pytest.approx(1.0)  # only sample is AT t=1.0 (already past the wall)
    assert continuous_t == pytest.approx(0.5)  # the true crossing instant
    assert continuous_t < discrete_t


def test_continuous_boundary_and_obstacle_checks_apply_the_same_t0_inclusive_rule():
    """item-6: "apply the SAME t=0-inclusive... rule... to the world
    boundary as well" -- an already-out-of-bounds t=0 state must be caught
    by the continuous boundary check exactly like the discrete one."""
    rollout = Rollout(points=[RolloutPoint(t_sec=0.1, state=VehicleState(x=0.0, y=0.0))])
    robot_pose = (10.0, 0.0, 0.0)  # already 10m out, half_extent=5
    clearance = boundary.min_boundary_clearance_continuous(rollout, 0.3, robot_pose, half_extent_m=5.0)
    assert clearance < 0.0
    t = boundary.boundary_time_to_exit_or_none_continuous(rollout, 0.3, robot_pose, half_extent_m=5.0,
                                                            horizon_sec=1.0)
    assert t == pytest.approx(0.0)


# -------------------------------------------------------- production wiring (assess_trajectory)
def test_assess_trajectory_catches_the_mid_segment_collision_end_to_end():
    """The actual production entry point (trajectory_risk.assess_trajectory)
    must now score the crossing-paths scenario as a genuine collision --
    proving item-6's fix is wired into REAL risk assessment, not just
    available as unused standalone functions."""
    robot = make_robot()
    t0_state, rollout, obstacle = _crossing_paths_rollout_and_obstacle()
    risk_cfg = RiskConfig(enabled=True, ttc_horizon_sec=1.0, clearance_horizon_sec=1.0, min_safe_clearance_m=0.3)
    label = trajectory_risk.assess_trajectory(
        rollout, ego_radius=0.3, obstacles=[obstacle], steering_rad=0.0, robot=robot, risk_cfg=risk_cfg,
        initial_state=t0_state,
    )
    assert label.collision_within_horizon is True
    assert label.risk_score == pytest.approx(1.0)
    assert label.min_clearance_m < 0.0
