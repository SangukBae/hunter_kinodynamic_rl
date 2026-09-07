"""Package-owned side-channel wire format for privileged risk data AND
real per-step physical-command telemetry: environment_node.py -> trainer/
evaluator (section 5's explicit requirement to keep ``drl_agent_interfaces``'s
``Step.srv`` untouched, since privileged/diagnostic data doesn't belong in
the policy-observation service contract).

Transported as a plain ``std_msgs/Float32MultiArray`` on
``/hunter_kinodynamic_rl/risk_telemetry`` (no new .srv/.msg generation
needed -- this module is the actual "schema", pure Python, unit-tested
without ROS). Synchronized by ``step_id``: both environment_node.py and the
trainer's ``EnvironmentClient`` maintain an identical monotonic counter,
reset to 0 at ``/reset`` and incremented once per ``/step`` -- the trainer
bounded-polls its subscription cache until the received ``step_id`` matches
its own local counter, so a transition is NEVER paired with a stale or
mismatched telemetry sample (section 5: "step ID 또는 transition ID로 label
동기화, stale label과 action mismatch 방지").

``reset_generation``/``episode_id`` (schema_version=6) give a consumer a
SECOND, cross-episode axis of staleness detection that ``step_id`` alone
cannot: ``step_id`` resets to 0 on every ``/reset``, so a telemetry message
genuinely left over from episode N-1's LAST step (step_id=K) could still
numerically match a NEW episode N's step_id=K message if a reader only
checks step_id equality without ALSO confirming it's from the SAME reset
cycle. ``reset_generation`` is a monotonic counter incremented once at the
very START of every ``_on_reset`` call (including one that later fails/
retries -- it counts reset ATTEMPTS, so a telemetry message from before a
reset attempt is recognized as stale the moment the NEXT attempt begins,
independent of whether that attempt itself succeeds). ``episode_id`` is the
semantically-stable identifier for WHICH scenario this is (the episode's
own seed, ``environment_node.py``'s ``self._episode_seed``) -- useful for
correlating telemetry against a specific, reproducible scenario replay
rather than just "the Nth reset since the node started."

Sent on EVERY ``/step`` regardless of whether the risk framework is enabled
(``valid=False`` when disabled) -- ``emergency_stop``/``guarded_*``/
``published_*`` are always meaningful (section P1-4/P0-5: evaluation
metrics must read the REAL physical command, not re-derive an approximation
from the normalized action).

ALSO sent exactly once at the very end of every ``/reset`` (code review
fix): an ``invalid(step_id=0, reason=RESET_MARKER, reset_generation=...)``
message carrying THIS reset's authoritative ``reset_generation`` -- the
sole purpose is letting a freshly-constructed ``EnvironmentClient`` LEARN
the server's real (process-lifetime, never client-local) counter value
instead of assuming it starts at 0 (see
``EnvironmentClient._await_reset_marker``'s docstring). ``step_id=0`` never
collides with a real per-step message (those start at 1).

Command pipeline has THREE distinct stages, ALL THREE now carried on the
wire (schema_version=5 -- an earlier round (P0-5) fixed the guarded/
published conflation described below but deliberately left "nominal" out,
reasoning it was "directly reconstructable from the logged normalized
action"; a later research-direction pass explicitly requires all three
recorded directly, not re-derivable-in-principle, so it was added here
too): trajectory_executor's raw output ("nominal": the actor's OWN intended
command, BEFORE any safety intervention -- this is also what the risk
label itself is computed from, see compute_risk_telemetry's docstring, so
a risky nominal command is never retroactively "graded safe" just because
the guard stopped it) -> env/safety/action_guard.guard()'s output
("guarded": sanitized/clamped/freshness-checked, but NOT yet delayed) ->
environment_node.py's command-latency queue's output ("published": what was
ACTUALLY handed to ``/cmd_vel`` this tick, which on a profile/benchmark with
``command_latency_sec`` > 0 is an OLDER command sitten in the queue, or an
explicit hold-at-STOP while the queue is still filling -- see
``environment_node.py::_publish_with_latency``). Before the P0-5 fix, the
wire only ever carried the "guarded" value mislabeled as "commanded" --
exactly correct with no latency queue active (the overwhelming majority of
profiles, where guarded == published and command_delay_steps == 0), but
silently WRONG for any fixed-benchmark scenario overriding
``command_latency_sec`` (section P1-3/ood_dynamics benchmarks): evaluation
metrics like ``control_deltas`` in ``evaluation/benchmark_runner.py`` were
computing smoothness against a value the robot never actually received that
tick.

section P0-9 adds ``sensor_stale``: True whenever this step's
``wait_for_fresh_sensors`` timed out (the LiDAR/odom observation folded
into ``state``/the risk assessment may reflect a PRE-action instant, not
truly post-action) -- environment_node.py forces ``done=True`` in the SAME
step's ``Step.srv`` response whenever this is True (a truncation, not a
silent continuation), and this flag is what lets the trainer/logs
distinguish that truncation reason from a genuine collision/goal/timeout
looking at risk_telemetry alone (``Step.srv`` itself carries no such
distinction -- see this module's section-5 docstring on why physical/
diagnostic fields live here instead of in that read-only interface).

``invalid_reason`` (schema_version=6) is an :class:`InvalidReason` int code
-- WHY ``valid`` is False, not just THAT it is. A plain float32 wire can't
carry a free-form string, so this is a small closed enum (the actual finite
set of reasons this codebase ever produces one) rather than text -- see
each member's docstring for exactly which call site produces it.

Wire layout (flat float32 list), schema_version=9:

    [0]  schema_version
    [1]  step_id
    [2]  valid                    (risk fields [3..10] meaningful iff true)
    [3]  risk_target              (actor's own candidate's risk_score)
    [4]  min_clearance_m
    [5]  ttc_sec
    [6]  collision_within_horizon (1.0 / 0.0)
    [7]  stopping_margin_m
    [8]  unrecoverable            (1.0 / 0.0)
    [9]  safer_alternative_margin
    [10] actor_candidate_index
    [11] emergency_stop           (1.0 / 0.0 -- code review fix, schema_version=7:
                                    the safety GUARD forced the PLANT command's
                                    speed down to ~0 this step -- compares
                                    plant_limited_speed_mps (see [24] below) to
                                    guarded_speed_mps, NEVER nominal_speed_mps.
                                    Before this fix it compared nominal_speed_mps
                                    (the RAW policy-intended speed, pre-plant-
                                    limiter) to guarded_speed_mps, so a domain-
                                    randomization plant speed limiter (see [24])
                                    ramping up from a stop on its own -- no guard
                                    intervention at all -- could get misreported
                                    as an emergency stop. ALWAYS meaningful,
                                    independent of `valid`)
    [12] nominal_speed_mps        (trajectory_executor's raw output --
                                    BEFORE the safety guard; this is what
                                    the risk label itself was computed from)
    [13] nominal_steering_rad     (ditto)
    [14] guarded_speed_mps        (post-safety-guard, PRE-latency-queue)
    [15] guarded_steering_rad     (post-safety-guard, PRE-latency-queue)
    [16] published_speed_mps      (REAL command actually sent to /cmd_vel
                                    this tick -- post-latency-queue; equals
                                    guarded_speed_mps whenever
                                    command_delay_steps == 0)
    [17] published_steering_rad   (ditto)
    [18] sensor_stale             (1.0 / 0.0 -- see section P0-9 above;
                                    ALWAYS meaningful, independent of `valid`)
    [19] reset_generation         (monotonic count of /reset ATTEMPTS since
                                    node startup -- staleness detection
                                    ACROSS episode boundaries, see module
                                    docstring)
    [20] episode_id                (this episode's seed -- semantic
                                    scenario identity, not a raw counter)
    [21] sim_timestamp_sec         (absolute /clock reading this step was
                                    computed at -- NaN if /clock hasn't
                                    delivered a message yet)
    [22] invalid_reason            (InvalidReason int code; 0/NONE when
                                    `valid` is True)
    [23] plant_limited             (1.0 / 0.0, schema_version=7 -- code review:
                                    the domain-randomization speed rate limiter
                                    (env/randomization/domain_randomizer.rate_limit_speed)
                                    actually changed the command THIS step --
                                    diagnostic-only, NEVER a safety signal on its
                                    own; distinguishes "the plant limiter is
                                    active" from "the guard intervened", the two
                                    causes emergency_stop used to conflate)
    [24] guard_intervened          (1.0 / 0.0, schema_version=7 -- the safety
                                    GUARD changed EITHER speed or steering this
                                    step (sanitize/freshness-stop/collision-
                                    proximity-stop/clamp) -- strictly more
                                    general than emergency_stop (which is only
                                    the "forced speed to ~0" special case);
                                    compares plant-limited-speed/nominal-
                                    steering (the guard's OWN input) to
                                    guarded_speed_mps/guarded_steering_rad (its
                                    output))
    [25] steering_saturation
    [26] goal_progress_m          (actor rollout's predicted distance reduction)
    [27] goal_x
    [28] goal_y
    [29..34] reward terms         (goal, collision, progress, step,
                                   control_smoothness, trajectory_smoothness)
    [35] num_candidates (N)
    [36 .. 36+10N) per-candidate
                     [kappa, v_ref, horizon_m, risk_score, goal_progress_m,
                      min_clearance_m, ttc_sec, collision_within_horizon,
                      stopping_margin_m, event_cause] x N
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List

SCHEMA_VERSION = 9
_HEADER_LEN = 36
_CANDIDATE_STRIDE = 10


class InvalidReason(IntEnum):
    NONE = 0  # valid=True (or no reason recorded -- pre-P0-5 callers)
    FEATURES_DISABLED = 1  # risk/ackermann_rollout disabled, or action_space.mode != "trajectory"
    COMPUTATION_EXCEPTION = 2  # compute_risk_telemetry raised (environment_node.py's except block)
    POLL_TIMEOUT = 3  # EnvironmentClient never received a step_id-matching message within budget
    # code review (risk-telemetry correctness bug): step_id=0, published ONCE at
    # the very end of every /reset -- carries this /reset's AUTHORITATIVE
    # reset_generation so a FRESH EnvironmentClient can learn the server's
    # actual (process-lifetime-monotonic, never client-local) counter value
    # instead of wrongly assuming it starts at 0 for every new client. See
    # EnvironmentClient._await_reset_marker's docstring for the exact bug
    # this closes (a long-lived/reused environment_node process -- e.g. a
    # trainer restarted against an already-running node -- silently desynced
    # the client's local reset_generation from the server's forever after,
    # making EVERY step's telemetry match fail, i.e. risk_valid=False for the
    # entire run, confirmed live). Never returned by compute_risk_telemetry
    # or any /step-time call site -- RESET_MARKER messages are recognizable
    # by step_id==0 alone in practice, but this explicit reason makes that
    # non-accidental for any code reading invalid_reason directly.
    RESET_MARKER = 4


@dataclass(frozen=True)
class CandidateTelemetry:
    kappa: float
    v_ref: float
    horizon_m: float
    risk_score: float
    goal_progress_m: float = 0.0
    min_clearance_m: float = float("nan")
    ttc_sec: float = float("nan")
    collision_within_horizon: bool = False
    stopping_margin_m: float = float("nan")
    event_cause: int = -1


@dataclass(frozen=True)
class RiskTelemetry:
    step_id: int
    valid: bool
    risk_target: float
    min_clearance_m: float
    ttc_sec: float
    collision_within_horizon: bool
    stopping_margin_m: float
    unrecoverable: bool
    safer_alternative_margin: float
    actor_candidate_index: int
    steering_saturation: bool = False
    goal_progress_m: float = float("nan")
    goal_x: float = float("nan")
    goal_y: float = float("nan")
    reward_goal: float = 0.0
    reward_collision: float = 0.0
    reward_progress: float = 0.0
    reward_step: float = 0.0
    reward_control_smoothness: float = 0.0
    reward_trajectory_smoothness: float = 0.0
    emergency_stop: bool = False
    nominal_speed_mps: float = 0.0
    nominal_steering_rad: float = 0.0
    guarded_speed_mps: float = 0.0
    guarded_steering_rad: float = 0.0
    published_speed_mps: float = 0.0
    published_steering_rad: float = 0.0
    sensor_stale: bool = False
    reset_generation: int = 0
    episode_id: int = 0
    sim_timestamp_sec: float = float("nan")
    invalid_reason: int = int(InvalidReason.NONE)
    plant_limited: bool = False
    guard_intervened: bool = False
    candidates: List[CandidateTelemetry] = field(default_factory=list)


def invalid(step_id: int, emergency_stop: bool = False,
            nominal_speed_mps: float = 0.0, nominal_steering_rad: float = 0.0,
            guarded_speed_mps: float = 0.0, guarded_steering_rad: float = 0.0,
            published_speed_mps: float = 0.0, published_steering_rad: float = 0.0,
            sensor_stale: bool = False, reset_generation: int = 0, episode_id: int = 0,
            sim_timestamp_sec: float = float("nan"),
            reason: "InvalidReason" = InvalidReason.NONE,
            plant_limited: bool = False, guard_intervened: bool = False) -> RiskTelemetry:
    """The explicit "no RISK label this step" value (risk fields NaN/False)
    -- NEVER represented by silently omitting the message or by a bare NaN
    with no companion flag (section 5: "NaN sentinel만으로 오류를 숨기지
    않음"). The physical-command/sensor-staleness fields are independent of
    risk validity and still meaningful here."""
    return RiskTelemetry(
        step_id=step_id, valid=False, risk_target=float("nan"),
        min_clearance_m=float("nan"), ttc_sec=float("nan"),
        collision_within_horizon=False, stopping_margin_m=float("nan"),
        unrecoverable=False, safer_alternative_margin=float("nan"),
        actor_candidate_index=0, emergency_stop=emergency_stop,
        nominal_speed_mps=nominal_speed_mps, nominal_steering_rad=nominal_steering_rad,
        guarded_speed_mps=guarded_speed_mps, guarded_steering_rad=guarded_steering_rad,
        published_speed_mps=published_speed_mps, published_steering_rad=published_steering_rad,
        sensor_stale=sensor_stale, reset_generation=reset_generation, episode_id=episode_id,
        sim_timestamp_sec=sim_timestamp_sec, invalid_reason=int(reason),
        plant_limited=plant_limited, guard_intervened=guard_intervened, candidates=[],
    )


def encode(t: RiskTelemetry) -> List[float]:
    out = [
        float(SCHEMA_VERSION), float(t.step_id), 1.0 if t.valid else 0.0,
        float(t.risk_target), float(t.min_clearance_m), float(t.ttc_sec),
        1.0 if t.collision_within_horizon else 0.0, float(t.stopping_margin_m),
        1.0 if t.unrecoverable else 0.0, float(t.safer_alternative_margin),
        float(t.actor_candidate_index), 1.0 if t.emergency_stop else 0.0,
        float(t.nominal_speed_mps), float(t.nominal_steering_rad),
        float(t.guarded_speed_mps), float(t.guarded_steering_rad),
        float(t.published_speed_mps), float(t.published_steering_rad),
        1.0 if t.sensor_stale else 0.0,
        float(t.reset_generation), float(t.episode_id), float(t.sim_timestamp_sec),
        float(t.invalid_reason),
        1.0 if t.plant_limited else 0.0, 1.0 if t.guard_intervened else 0.0,
        1.0 if t.steering_saturation else 0.0,
        float(t.goal_progress_m), float(t.goal_x), float(t.goal_y),
        float(t.reward_goal), float(t.reward_collision), float(t.reward_progress),
        float(t.reward_step), float(t.reward_control_smoothness),
        float(t.reward_trajectory_smoothness),
        float(len(t.candidates)),
    ]
    for c in t.candidates:
        out.extend([
            c.kappa, c.v_ref, c.horizon_m, c.risk_score, c.goal_progress_m,
            c.min_clearance_m, c.ttc_sec,
            1.0 if c.collision_within_horizon else 0.0,
            c.stopping_margin_m, float(c.event_cause),
        ])
    return out


def decode(data: List[float]) -> RiskTelemetry:
    if len(data) < _HEADER_LEN:
        raise ValueError(f"risk_telemetry payload too short: {len(data)} < {_HEADER_LEN}")
    version = int(round(data[0]))
    if version != SCHEMA_VERSION:
        raise ValueError(f"risk_telemetry schema_version mismatch: got {version}, expected {SCHEMA_VERSION}")
    n = int(round(data[35]))
    expected_len = _HEADER_LEN + _CANDIDATE_STRIDE * n
    if len(data) != expected_len:
        raise ValueError(f"risk_telemetry payload length {len(data)} != expected {expected_len} for N={n}")
    candidates = [
        CandidateTelemetry(
            kappa=data[_HEADER_LEN + _CANDIDATE_STRIDE * i],
            v_ref=data[_HEADER_LEN + _CANDIDATE_STRIDE * i + 1],
            horizon_m=data[_HEADER_LEN + _CANDIDATE_STRIDE * i + 2],
            risk_score=data[_HEADER_LEN + _CANDIDATE_STRIDE * i + 3],
            goal_progress_m=data[_HEADER_LEN + _CANDIDATE_STRIDE * i + 4],
            min_clearance_m=data[_HEADER_LEN + _CANDIDATE_STRIDE * i + 5],
            ttc_sec=data[_HEADER_LEN + _CANDIDATE_STRIDE * i + 6],
            collision_within_horizon=bool(data[_HEADER_LEN + _CANDIDATE_STRIDE * i + 7] >= 0.5),
            stopping_margin_m=data[_HEADER_LEN + _CANDIDATE_STRIDE * i + 8],
            event_cause=int(round(data[_HEADER_LEN + _CANDIDATE_STRIDE * i + 9])),
        )
        for i in range(n)
    ]
    return RiskTelemetry(
        step_id=int(round(data[1])), valid=bool(data[2] >= 0.5), risk_target=data[3],
        min_clearance_m=data[4], ttc_sec=data[5], collision_within_horizon=bool(data[6] >= 0.5),
        stopping_margin_m=data[7], unrecoverable=bool(data[8] >= 0.5), safer_alternative_margin=data[9],
        actor_candidate_index=int(round(data[10])), emergency_stop=bool(data[11] >= 0.5),
        nominal_speed_mps=data[12], nominal_steering_rad=data[13],
        guarded_speed_mps=data[14], guarded_steering_rad=data[15],
        published_speed_mps=data[16], published_steering_rad=data[17],
        sensor_stale=bool(data[18] >= 0.5),
        reset_generation=int(round(data[19])), episode_id=int(round(data[20])),
        sim_timestamp_sec=data[21], invalid_reason=int(round(data[22])),
        plant_limited=bool(data[23] >= 0.5), guard_intervened=bool(data[24] >= 0.5),
        steering_saturation=bool(data[25] >= 0.5), goal_progress_m=data[26],
        goal_x=data[27], goal_y=data[28], reward_goal=data[29], reward_collision=data[30],
        reward_progress=data[31], reward_step=data[32], reward_control_smoothness=data[33],
        reward_trajectory_smoothness=data[34],
        candidates=candidates,
    )
