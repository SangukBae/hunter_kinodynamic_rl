"""section P0-2 regression coverage: clearance/TTC/risk-label correctness.

Two confirmed bugs fixed:

1. A :class:`~hunter_kinodynamic_rl.dynamics.ackermann_rollout.Rollout`'s
   own points are samples at ``t in (dt, 2*dt, ..., horizon_sec]`` -- they
   NEVER include ``t=0`` (see ``rollout_constant_target``'s own docstring).
   ``future_clearance.min_clearance``/``ttc.time_to_collision_or_none``/
   ``boundary.min_boundary_clearance``/``boundary.boundary_time_to_exit_or_none``
   previously only ever scanned ``rollout.points``, so an obstacle already
   overlapping the ego AT THE CURRENT INSTANT (t=0) was invisible until (if
   ever) a later sample happened to still show it.
2. ``future_clearance.min_clearance`` never accepted (or respected) a
   ``clearance_horizon_sec`` at all -- it always scanned the WHOLE rollout,
   even when the caller's risk config wanted a SHORTER clearance horizon
   than the rollout's own duration (e.g. a long-L trajectory candidate
   assessed against a short ``risk.clearance_horizon_sec``).
3. ``time_to_collision``/``boundary_time_to_exit`` return a single float
   that collapses "no collision" and "collision exactly at the horizon"
   into the SAME sentinel value (``horizon_sec``), so
   ``collision_within_horizon``'s old ``< horizon_sec`` comparison silently
   misclassified an exactly-at-horizon collision as "no collision".
"""

import math

import pytest

from hunter_kinodynamic_rl.config.schema import DynamicsConfig, RiskConfig, RobotConfig
from hunter_kinodynamic_rl.dynamics.ackermann_rollout import Rollout, RolloutPoint
from hunter_kinodynamic_rl.risk import boundary, future_clearance, trajectory_risk, ttc
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


def _rollout(points_t_sec, x_of_t):
    """A synthetic straight-line-on-x rollout: state.x = x_of_t(t), y=0."""
    return Rollout(points=[
        RolloutPoint(t_sec=t, state=VehicleState(x=x_of_t(t), y=0.0, yaw=0.0, v=1.0, steering=0.0))
        for t in points_t_sec
    ])


# ------------------------------------------------------------------ t=0 overlap
def test_min_clearance_detects_t0_overlap_invisible_to_rollout_points():
    """The core bug: an obstacle sitting exactly at the ego's CURRENT
    position (t=0) but that the ego then drives AWAY from FAST enough that
    even the very first rollout sample already looks clear is invisible to
    every rollout point -- without t0_state, min_clearance would wrongly
    report "safe"."""
    rollout = _rollout([0.1, 0.2, 0.3], lambda t: t * 20.0)  # driving away, fast
    obstacle = DynamicObstacle(x0=0.0, y0=0.0, radius=0.3)
    t0_state = VehicleState(x=0.0, y=0.0)

    without_t0 = future_clearance.min_clearance(rollout, ego_radius=0.3, obstacles=[obstacle])
    assert without_t0 > 0.0  # the bug: looks perfectly safe

    with_t0 = future_clearance.min_clearance(
        rollout, ego_radius=0.3, obstacles=[obstacle], t0_state=t0_state)
    assert with_t0 == pytest.approx(-0.6)  # 0 - 0.3 - 0.3, already overlapping


def test_time_to_collision_detects_t0_overlap():
    rollout = _rollout([0.1, 0.2, 0.3], lambda t: t * 20.0)
    obstacle = DynamicObstacle(x0=0.0, y0=0.0, radius=0.3)
    t0_state = VehicleState(x=0.0, y=0.0)

    assert ttc.collision_within_horizon(rollout, 0.3, [obstacle], horizon_sec=1.0) is False  # the bug
    assert ttc.collision_within_horizon(
        rollout, 0.3, [obstacle], horizon_sec=1.0, t0_state=t0_state) is True
    assert ttc.time_to_collision_or_none(
        rollout, 0.3, [obstacle], horizon_sec=1.0, t0_state=t0_state) == pytest.approx(0.0)


# --------------------------------------------------------- horizon boundary cases
def test_collision_before_horizon_is_detected():
    rollout = _rollout([0.5, 1.0, 1.5], lambda t: 1.0)  # obstacle at x=1.0, ego reaches it early
    obstacle = DynamicObstacle(x0=1.0, y0=0.0, radius=0.3)
    assert ttc.collision_within_horizon(rollout, 0.3, [obstacle], horizon_sec=2.0) is True
    t = ttc.time_to_collision_or_none(rollout, 0.3, [obstacle], horizon_sec=2.0)
    assert t == pytest.approx(0.5)


def test_collision_exactly_at_horizon_is_distinguished_from_no_collision():
    """The core boundary-ambiguity bug: a collision sampled at EXACTLY
    horizon_sec must be reported as a real collision, not conflated with
    "never collides" (both previously produced the identical clamped
    value)."""
    # x=0.5 at t=1 (clear, dist=0.5), x=1.0 at t=2 (dist=0.0 exactly --
    # touching precisely at the horizon). Zero radii keep the boundary
    # value an exact binary float (0.0), avoiding float-rounding noise at
    # the exact-equality boundary this test is specifically about.
    rollout = _rollout([1.0, 2.0], lambda t: 0.5 * t)
    obstacle = DynamicObstacle(x0=1.0, y0=0.0, radius=0.0)

    at_horizon = ttc.time_to_collision_or_none(rollout, 0.0, [obstacle], horizon_sec=2.0)
    assert at_horizon == pytest.approx(2.0)
    assert ttc.collision_within_horizon(rollout, 0.0, [obstacle], horizon_sec=2.0) is True

    # A genuinely non-colliding trajectory reports the SAME clamped float
    # via time_to_collision(), but the _or_none/boolean primitives must
    # still tell the two apart.
    far_obstacle = DynamicObstacle(x0=100.0, y0=0.0, radius=0.3)
    never = ttc.time_to_collision_or_none(rollout, 0.0, [far_obstacle], horizon_sec=2.0)
    assert never is None
    assert ttc.time_to_collision(rollout, 0.0, [far_obstacle], horizon_sec=2.0) == pytest.approx(2.0)
    assert ttc.collision_within_horizon(rollout, 0.0, [far_obstacle], horizon_sec=2.0) is False


def test_collision_after_horizon_is_not_counted():
    rollout = _rollout([1.0, 2.0, 3.0], lambda t: 0.4 * t)  # x=0.4, 0.8, 1.2 at t=1,2,3
    obstacle = DynamicObstacle(x0=1.2, y0=0.0, radius=0.1)  # only reached (dist<=0.1) at t=3
    assert ttc.time_to_collision_or_none(rollout, 0.0, [obstacle], horizon_sec=2.0) is None
    assert ttc.collision_within_horizon(rollout, 0.0, [obstacle], horizon_sec=2.0) is False
    # ... but IS detected once the horizon is widened to actually cover it.
    assert ttc.time_to_collision_or_none(rollout, 0.0, [obstacle], horizon_sec=3.0) == pytest.approx(3.0)


# ------------------------------------------------------------- no / static / moving
def test_no_obstacle_never_collides():
    rollout = _rollout([0.5, 1.0], lambda t: t)
    assert future_clearance.min_clearance(rollout, 0.3, []) == math.inf
    assert ttc.time_to_collision_or_none(rollout, 0.3, [], horizon_sec=2.0) is None
    assert ttc.collision_within_horizon(rollout, 0.3, [], horizon_sec=2.0) is False


def test_static_obstacle_collision():
    rollout = _rollout([0.5, 1.0, 1.5], lambda t: t)
    static = DynamicObstacle(x0=1.0, y0=0.0, vx=0.0, vy=0.0, radius=0.3)
    assert ttc.collision_within_horizon(rollout, 0.3, [static], horizon_sec=2.0) is True


def test_moving_obstacle_collision_uses_its_own_trajectory():
    """A moving obstacle that starts far away but drives INTO the ego's
    path must still be caught -- clearance_at must evaluate the obstacle's
    OWN position_at(t), not its t=0 position, at every sample time."""
    rollout = _rollout([0.5, 1.0, 1.5, 2.0], lambda t: 0.0)  # ego holds still at x=0
    moving = DynamicObstacle(x0=5.0, y0=0.0, vx=-3.0, vy=0.0, radius=0.3)  # closes in at 3 m/s
    # at t=1.5, obstacle is at x=5-4.5=0.5 (clearance 0.5-0.3-0.3=-0.1 -> collision)
    t = ttc.time_to_collision_or_none(rollout, 0.3, [moving], horizon_sec=2.0)
    assert t == pytest.approx(1.5)
    # A STATIC copy of the same obstacle at its t=0 position would never collide.
    frozen = DynamicObstacle(x0=5.0, y0=0.0, vx=0.0, vy=0.0, radius=0.3)
    assert ttc.time_to_collision_or_none(rollout, 0.3, [frozen], horizon_sec=2.0) is None


# --------------------------------------------------------------- world boundary
def test_boundary_t0_overlap_is_detected():
    """Mirrors the obstacle t=0 regression: a rollout that starts ALREADY
    outside the world boundary (e.g. spawned right at the edge) but
    immediately turns back inward must still be flagged, since every
    future rollout point would otherwise show improving (inward) clearance."""
    rollout = _rollout([0.1, 0.2], lambda t: 0.0)  # placeholder x; y stays 0 throughout
    robot_pose = (10.0, 0.0, 0.0)  # world position already 10m out, half_extent=5 -> already past the wall
    clearance = boundary.min_boundary_clearance(rollout, ego_radius=0.3, robot_pose=robot_pose, half_extent_m=5.0)
    assert clearance < 0.0
    t = boundary.boundary_time_to_exit_or_none(rollout, 0.3, robot_pose, half_extent_m=5.0, horizon_sec=2.0)
    assert t == pytest.approx(0.0)


def test_boundary_exactly_at_horizon_is_distinguished():
    dyn_robot = make_robot()
    rollout = _rollout([1.0, 2.0], lambda t: t)  # local x=1 at t=1, x=2 at t=2
    robot_pose = (0.0, 0.0, 0.0)
    # half_extent=2.5, ego_radius=0.5 -> wall reached (world x=2.0) exactly when local x=2.0, t=2.0
    exit_time = boundary.boundary_time_to_exit_or_none(
        rollout, ego_radius=0.5, robot_pose=robot_pose, half_extent_m=2.5, horizon_sec=2.0)
    assert exit_time == pytest.approx(2.0)
    del dyn_robot


def test_boundary_horizon_clips_min_clearance():
    """min_boundary_clearance's horizon_sec kwarg must exclude points
    beyond it, mirroring future_clearance.min_clearance's own horizon
    clipping."""
    rollout = _rollout([1.0, 2.0, 3.0], lambda t: t)  # local x grows to 3.0
    robot_pose = (0.0, 0.0, 0.0)
    half_extent_m = 10.0  # never actually collides -- purely testing which points count
    full = boundary.min_boundary_clearance(rollout, 0.0, robot_pose, half_extent_m, check_t0=False)
    clipped = boundary.min_boundary_clearance(
        rollout, 0.0, robot_pose, half_extent_m, horizon_sec=1.5, check_t0=False)
    assert full < clipped  # the t=3.0 point (closer to the wall) is excluded from the clipped version


# --------------------------------------------------- empty / single-point rollouts
def test_empty_rollout_with_t0_still_checks_the_current_instant():
    empty = Rollout(points=[])
    obstacle = DynamicObstacle(x0=0.0, y0=0.0, radius=0.3)
    t0_state = VehicleState(x=0.0, y=0.0)
    assert future_clearance.min_clearance(empty, 0.3, [obstacle]) == math.inf  # no t0 given: unchanged
    assert future_clearance.min_clearance(
        empty, 0.3, [obstacle], t0_state=t0_state) == pytest.approx(-0.6)
    assert ttc.time_to_collision_or_none(
        empty, 0.3, [obstacle], horizon_sec=1.0, t0_state=t0_state) == pytest.approx(0.0)


def test_single_point_rollout():
    rollout = _rollout([1.0], lambda t: 1.0)
    obstacle = DynamicObstacle(x0=1.0, y0=0.0, radius=0.3)
    assert ttc.time_to_collision_or_none(rollout, 0.3, [obstacle], horizon_sec=2.0) == pytest.approx(1.0)
    far = DynamicObstacle(x0=50.0, y0=0.0, radius=0.3)
    assert ttc.time_to_collision_or_none(rollout, 0.3, [far], horizon_sec=2.0) is None


# ------------------------------------------------- different clearance/TTC horizons
def test_min_clearance_horizon_kwarg_excludes_late_close_approach():
    """The core clearance_horizon_sec bug: a rollout that stays far from an
    obstacle EARLY but swings close LATE (beyond the intended clearance
    horizon) must not have that late close approach counted, when a
    horizon_sec is given."""
    rollout = _rollout([0.5, 1.0, 1.5, 2.0], lambda t: 10.0 if t < 1.5 else 1.0)
    obstacle = DynamicObstacle(x0=1.0, y0=0.0, radius=0.3)
    unclipped = future_clearance.min_clearance(rollout, 0.3, [obstacle])
    assert unclipped < 0.0  # the late (t=2.0) close approach dominates
    clipped = future_clearance.min_clearance(rollout, 0.3, [obstacle], horizon_sec=1.0)
    assert clipped > 0.0  # only t=0.5, t=1.0 count -- both far away


def test_assess_trajectory_command_clamps_clearance_to_the_configured_horizon():
    """End-to-end: assess_trajectory's own clearance computation must
    respect risk_cfg.clearance_horizon_sec, not silently scan the whole
    (possibly longer) rollout -- this is the exact bug confirmed in
    risk/trajectory_risk.py's pre-fix assess_trajectory."""
    robot = make_robot()
    dyn = DynamicsConfig(horizon_sec=2.0, dt_sec=0.1, model_actuator_lag=False)
    # A long, slow trajectory: ego barely moves early, but a close obstacle
    # sits right where the LATE part of the rollout passes.
    from hunter_kinodynamic_rl.dynamics import ackermann_rollout
    state = VehicleState(v=0.0, steering=0.0)
    rollout = ackermann_rollout.rollout_constant_target(state, 1.0, 0.0, robot, dyn)
    obstacle = DynamicObstacle(x0=1.8, y0=0.0, radius=0.1)  # reached only late in the 2.0s rollout

    short_horizon_cfg = RiskConfig(enabled=True, clearance_horizon_sec=0.5, ttc_horizon_sec=2.0,
                                    min_safe_clearance_m=0.3)
    label = trajectory_risk.assess_trajectory(
        rollout, ego_radius=0.3, obstacles=[obstacle], steering_rad=0.0, robot=robot, risk_cfg=short_horizon_cfg,
        initial_state=state,
    )
    assert label.min_clearance_m > 0.0  # the late close pass is outside the 0.5s clearance horizon

    long_horizon_cfg = RiskConfig(enabled=True, clearance_horizon_sec=2.0, ttc_horizon_sec=2.0,
                                   min_safe_clearance_m=0.3)
    label_long = trajectory_risk.assess_trajectory(
        rollout, ego_radius=0.3, obstacles=[obstacle], steering_rad=0.0, robot=robot, risk_cfg=long_horizon_cfg,
        initial_state=state,
    )
    assert label_long.min_clearance_m < label.min_clearance_m
