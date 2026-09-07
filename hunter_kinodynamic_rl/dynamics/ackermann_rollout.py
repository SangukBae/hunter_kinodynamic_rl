"""Forward-simulate Hunter SE's actual short-horizon trajectory under a
commanded target (v, steering), accounting for actuator dynamics -- NOT an
instantaneous jump to the target. This is the "Hunter SE dynamics rollout"
step in the research brief's core data-flow diagram (section 68): candidate
trajectories are scored using THIS rollout, not the idealised kinematic-only
path.

Composes dynamics/actuator_model.py (how v/steering approach the target) with
dynamics/bicycle_model.py (how pose evolves given the actual, ramped v/steering)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

from hunter_kinodynamic_rl.config.schema import DynamicsConfig, RobotConfig
from hunter_kinodynamic_rl.dynamics import actuator_model, bicycle_model
from hunter_kinodynamic_rl.robot.interface import VehicleState


@dataclass
class RolloutPoint:
    t_sec: float
    state: VehicleState


@dataclass
class Rollout:
    points: List[RolloutPoint] = field(default_factory=list)

    @property
    def final_state(self) -> VehicleState:
        return self.points[-1].state if self.points else VehicleState()

    def states(self) -> List[VehicleState]:
        return [p.state for p in self.points]

    def xy(self):
        return [(p.state.x, p.state.y) for p in self.points]


def rollout_constant_target(
    initial_state: VehicleState,
    target_v_mps: float,
    target_steering_rad: float,
    robot: RobotConfig,
    dynamics_cfg: DynamicsConfig,
) -> Rollout:
    """Rollout toward ONE constant (v, steering) target held for the whole
    horizon -- the common case for scoring a single trajectory-primitive
    action. Returns samples at every dt_sec step, t in (0, horizon_sec]."""
    n_steps = max(1, int(round(dynamics_cfg.horizon_sec / dynamics_cfg.dt_sec)))
    dt = dynamics_cfg.horizon_sec / n_steps

    actuator = actuator_model.ActuatorState(v=initial_state.v, steering=initial_state.steering,
                                             v_lagged=initial_state.v)
    state = initial_state
    points: List[RolloutPoint] = []
    for i in range(1, n_steps + 1):
        if dynamics_cfg.model_actuator_lag:
            v_prev, steer_prev = actuator_model.effective_speed(actuator, robot), actuator.steering
            actuator = actuator_model.step_actuator(actuator, target_v_mps, target_steering_rad, dt, robot)
            v_eff = actuator_model.effective_speed(actuator, robot)
            state = bicycle_model.step_midpoint(state, v_prev, steer_prev, v_eff, actuator.steering,
                                                 dt, robot.wheelbase_m)
        else:
            state = bicycle_model.step(state, target_v_mps, target_steering_rad, dt, robot.wheelbase_m)
        points.append(RolloutPoint(t_sec=dt * i, state=state))
    return Rollout(points=points)


def rollout_constant_curvature(
    initial_state: VehicleState,
    kappa: float,
    v_ref_mps: float,
    robot: RobotConfig,
    dynamics_cfg: DynamicsConfig,
) -> Rollout:
    """Convenience wrapper: curvature -> steering, then
    :func:`rollout_constant_target`, using ``dynamics_cfg.horizon_sec``
    (the FIXED fallback horizon -- see :func:`horizon_from_trajectory` for
    the L-derived horizon used by trajectory-mode candidate scoring)."""
    from hunter_kinodynamic_rl.robot.limits import curvature_to_steering
    steering = curvature_to_steering(kappa, robot.wheelbase_m)
    return rollout_constant_target(initial_state, v_ref_mps, steering, robot, dynamics_cfg)


_V_FLOOR_MPS = 0.05


def l_derived_horizon_sec(horizon_m: float, v_ref_mps: float, dynamics_cfg: DynamicsConfig) -> float:
    """L (arc-length horizon) -> ``clip(L / v_ref, horizon_min_sec, horizon_max_sec)``,
    WITHOUT the ``min_safety_horizon_sec`` floor :func:`horizon_from_trajectory` applies
    on top. This is the raw "how long does the policy intend to commit to this
    trajectory" quantity -- used by :func:`horizon_from_trajectory` for risk-window
    sizing (padded with the safety floor there) AND, directly (unpadded), by
    ``trajectory/pure_pursuit_adapter.py`` to size the L-derived STEERING commit
    window (padding that with a risk-motivated safety floor would mute L's control
    effect at short horizons -- the floor's purpose is risk-hiding prevention, not
    control smoothness, so it belongs only in the risk-facing function).

    v_ref below ``_V_FLOOR_MPS`` (including exactly 0, a commanded stop) maps to
    ``horizon_max_sec`` rather than dividing by ~0.
    """
    if v_ref_mps <= _V_FLOOR_MPS:
        return dynamics_cfg.horizon_max_sec
    raw = horizon_m / v_ref_mps
    return max(dynamics_cfg.horizon_min_sec, min(dynamics_cfg.horizon_max_sec, raw))


def horizon_from_trajectory(horizon_m: float, v_ref_mps: float, dynamics_cfg: DynamicsConfig) -> float:
    """L (arc-length horizon) -> rollout TIME horizon:
    ``max(l_derived_horizon_sec(...), min_safety_horizon_sec)``.

    Two effects, deliberately BOTH present (found missing the second one in
    code review -- see docs/TRACTOR_TQC_MODEL_SPEC.md's "L semantics" section):

    1. A policy that commits to a LONGER L is scored on a longer
       future-risk window (:func:`l_derived_horizon_sec`) -- this is what
       makes L a non-dead action dimension.
    2. The horizon is NEVER shorter than ``min_safety_horizon_sec``
       REGARDLESS of L -- this is what stops a policy from picking an
       arbitrarily small L purely to shrink the rollout below the distance
       to a real obstacle, which would otherwise let it report risk=0 for
       an actually-unsafe heading/speed simply by not looking far enough
       ahead (a reward/risk-hacking exploit; see
       ``tests/test_l_semantics.py::test_short_L_cannot_hide_a_distant_collision``).
    """
    return max(l_derived_horizon_sec(horizon_m, v_ref_mps, dynamics_cfg), dynamics_cfg.min_safety_horizon_sec)


def l_commit_blended_steering(
    target_steering_rad: float, current_steering_rad: float,
    horizon_m: float, v_ref_mps: float, robot: RobotConfig,
    dynamics_cfg: DynamicsConfig, dt_sec: float,
    *, apply_rate_limit: bool = True,
) -> float:
    """The SAME "L-derived steering commit window" formula
    ``trajectory/pure_pursuit_adapter.py`` uses to compute the actually-
    PUBLISHED steering command (see that module's "STEERING COMMIT WINDOW"
    docstring section for the full rationale) -- lives here, not there, so
    :func:`rollout_trajectory_command`'s ``commit_blend=True`` mode below
    can reuse it directly without ``dynamics`` importing FROM
    ``trajectory`` (which would be circular: ``pure_pursuit_adapter.py``
    already imports ``l_derived_horizon_sec`` from this module). A short L
    means "commit quickly" -> most of the target is applied THIS tick; a
    long L means "commit gradually" -> only a small fraction is applied.

    ``apply_rate_limit`` (default True) additionally bounds the blended
    delta by the robot's real ``steering_rate_deg_s`` actuator limit so the
    blend can never command a physically unrealizable angular rate. This is
    correct as the FINAL word on steering whenever nothing downstream also
    rate-limits (``pure_pursuit_adapter.py``'s publish path, and
    :func:`rollout_trajectory_command`'s ``commit_blend=True`` /
    ``model_actuator_lag=False`` path). Pass ``apply_rate_limit=False`` when
    the caller's OWN actuator model (``dynamics/actuator_model.py``) will
    immediately rate-limit this function's return value again as ITS
    target -- see ``rollout_trajectory_command``'s ``model_actuator_lag=True``
    branch below, which needs this to avoid double rate-limiting."""
    dt = max(dt_sec, 1e-6)
    commit_sec = max(l_derived_horizon_sec(horizon_m, v_ref_mps, dynamics_cfg), dt)
    blend = max(0.0, min(1.0, dt / commit_sec))
    desired_delta = blend * (target_steering_rad - current_steering_rad)
    if not apply_rate_limit:
        return current_steering_rad + desired_delta
    rate_limit_rad = math.radians(max(robot.steering_rate_deg_s, 0.0)) * dt
    applied_delta = max(-rate_limit_rad, min(rate_limit_rad, desired_delta))
    return current_steering_rad + applied_delta


def rollout_trajectory_command(
    initial_state: VehicleState,
    kappa: float,
    v_ref_mps: float,
    horizon_m: float,
    robot: RobotConfig,
    dynamics_cfg: DynamicsConfig,
    *,
    commit_blend: bool = False,
) -> Rollout:
    """Roll out a full ``[kappa, v_ref, L]`` trajectory command using the
    L-DERIVED horizon (:func:`horizon_from_trajectory`) instead of the fixed
    ``dynamics_cfg.horizon_sec`` -- the entry point
    risk/trajectory_risk.py's ``assess_trajectory_command`` and
    risk/counterfactual_sampler.py both use, so every risk/counterfactual
    computation is L-consistent by construction.

    ``commit_blend=False`` (default): the LEGACY rollout -- every tick
    jumps straight to the full target curvature's steering angle, as if
    instantly achieved. Byte-identical to this function's pre-existing
    behavior.

    ``commit_blend=True`` (code review: "risk label과 실제 실행 command가
    불일치" -- risk labels assumed instant target-curvature tracking while
    the ACTUALLY-published command, once
    ``features.trajectory_l_preview_blend`` shipped, only partially
    commits toward it each tick via :func:`l_commit_blended_steering`):
    EVERY tick's steering is computed with that SAME formula, starting
    from ``initial_state.steering`` (the caller's job to pass the REAL,
    PRE-ACTION measured current steering -- NEVER a safety-guard-adjusted
    value, so a guard intervention can never retroactively make a
    genuinely risky nominal action look safe; see
    risk/trajectory_risk.py's docstring). Tick 1 therefore reproduces --
    up to the same dt-discretization rounding this rollout's horizon
    stepping already has (n_steps = round(horizon_sec / dynamics_cfg.dt_sec),
    generally close to but not bit-identical to
    trajectory_cfg.dt_sec-driven execution) -- what environment_node.py
    actually published this step; later ticks model the SAME commit-window
    control law continuing to apply as if the policy kept re-issuing this
    same nominal action, the standard "constant candidate persistence"
    assumption this rollout already makes for kappa/v_ref.

    ``commit_blend=True`` COMBINED with ``dynamics_cfg.model_actuator_lag=True``
    (code review: a risk rollout using the DEFAULT profile -- see
    ``config/training/defaults.yaml``'s ``dynamics.model_actuator_lag: true``
    -- must not silently drop speed lag / accel-brake limiting just because
    the L-commit steering blend is also engaged): the two features compose
    rather than one shadowing the other, with a clear division of labor to
    avoid double rate-limiting steering --
      * :func:`l_commit_blended_steering` still produces each tick's TARGET
        steering command profile (the L-commit fraction of the delta toward
        the candidate's target curvature), but with ``apply_rate_limit=False``
        -- it no longer also clips to the physical ``steering_rate_deg_s``
        limit itself.
      * ``dynamics/actuator_model.py``'s ``step_actuator`` takes that
        per-tick blended value as ITS target and is the ONE place that
        applies the physical steering-rate limit (its own ``move_towards``
        clamp) -- and, being the actuator model, ALSO applies speed
        rate-limiting (accel/brake) and, if configured, the first-order
        speed lag, exactly as ``rollout_constant_target``'s
        ``model_actuator_lag=True`` path already does for the non-L-commit
        rollout. Net effect: steering is rate-limited EXACTLY ONCE (by the
        actuator model, not by the blend formula), and speed is no longer
        silently idealised.
    When ``model_actuator_lag=False`` (opt-out), behavior is UNCHANGED from
    before this fix: :func:`l_commit_blended_steering` is the sole rate
    limiter (``apply_rate_limit=True``, its default) and speed is applied
    directly every tick via plain ``bicycle_model.step`` -- byte-identical
    to this function's pre-existing ``commit_blend=True`` behavior."""
    horizon_sec = horizon_from_trajectory(horizon_m, v_ref_mps, dynamics_cfg)
    from hunter_kinodynamic_rl.robot.limits import curvature_to_steering
    target_steering = curvature_to_steering(kappa, robot.wheelbase_m)

    if commit_blend:
        n_steps = max(1, int(round(horizon_sec / dynamics_cfg.dt_sec)))
        dt = horizon_sec / n_steps
        state = initial_state
        points: List[RolloutPoint] = []

        if dynamics_cfg.model_actuator_lag:
            # See this function's docstring: the L-commit blend supplies each
            # tick's TARGET steering (rate-limit NOT applied here -- the
            # actuator model applies it, exactly once) while ALSO gaining
            # speed accel/brake limiting + lag, matching the DEFAULT profile
            # (config/training/defaults.yaml's model_actuator_lag: true).
            actuator = actuator_model.ActuatorState(
                v=initial_state.v, steering=initial_state.steering, v_lagged=initial_state.v,
            )
            for i in range(1, n_steps + 1):
                target_tick_steering = l_commit_blended_steering(
                    target_steering, actuator.steering, horizon_m, v_ref_mps, robot, dynamics_cfg, dt,
                    apply_rate_limit=False,
                )
                v_prev, steer_prev = actuator_model.effective_speed(actuator, robot), actuator.steering
                actuator = actuator_model.step_actuator(actuator, v_ref_mps, target_tick_steering, dt, robot)
                v_eff = actuator_model.effective_speed(actuator, robot)
                state = bicycle_model.step_midpoint(state, v_prev, steer_prev, v_eff, actuator.steering,
                                                     dt, robot.wheelbase_m)
                points.append(RolloutPoint(t_sec=dt * i, state=state))
            return Rollout(points=points)

        for i in range(1, n_steps + 1):
            effective_steering = l_commit_blended_steering(
                target_steering, state.steering, horizon_m, v_ref_mps, robot, dynamics_cfg, dt,
            )
            state = bicycle_model.step(state, v_ref_mps, effective_steering, dt, robot.wheelbase_m)
            points.append(RolloutPoint(t_sec=dt * i, state=state))
        return Rollout(points=points)

    effective_cfg = DynamicsConfig(
        horizon_sec=horizon_sec, horizon_min_sec=dynamics_cfg.horizon_min_sec,
        horizon_max_sec=dynamics_cfg.horizon_max_sec, dt_sec=dynamics_cfg.dt_sec,
        model_actuator_lag=dynamics_cfg.model_actuator_lag,
    )
    return rollout_constant_target(initial_state, v_ref_mps, target_steering, robot, effective_cfg)
