"""GT/noisy observation diagnostics side channel (requirement 5).

A SEPARATE, VERSIONED wire format/topic from ``risk_telemetry.py`` (never
folded into it) -- section 5's own rule ("privileged/diagnostic data
doesn't belong in the policy-observation service contract") applies here
too, and this is a genuinely different concern from risk labeling: risk
telemetry answers "was the actor's candidate action risky", this answers
"what did the sensor pipeline actually do to ground truth this tick" (the
sensor_noise/domain_randomization perturbation applied between the real
Gazebo state and what the policy's observation vector actually contains).
Keeping them on separate topics means a consumer of one is never forced to
understand or pay the encode/decode/bandwidth cost of the other, and a
schema change to this diagnostics channel can never accidentally break
risk-telemetry's own (already externally-depended-on) wire contract.

Transported as a plain ``std_msgs/Float32MultiArray`` on
``/hunter_kinodynamic_rl/sensor_diagnostics`` (no new .srv/.msg generation
needed, mirroring risk_telemetry.py's own approach for the identical
reason -- section 5's ``drl_agent_interfaces`` stays untouched). Published
by ``environment_node.py`` from EXACTLY the same place ground truth and the
agent's actual (possibly noisy) observation are both already locally
available in one call -- see ``_assemble_state_vector``'s own docstring --
so it fires once at ``/reset`` (``step_id=0``, ``drift_*`` always 0.0: an
episode's OU drift starts at exactly zero, see ``sensor_noise.py``) and
once per real ``/step`` (``step_id`` == that step's ``risk_telemetry``
``step_id`` exactly, so a consumer can align the two channels the same way
it already aligns risk_telemetry across episode boundaries: by
``(reset_generation, step_id)`` -- see risk_telemetry.py's own module
docstring for why ``step_id`` ALONE is not staleness-safe across a reset).

Ground truth here is ALWAYS ``self._robot_pose``/``self._robot_twist``/
``self._center_steering``/the untouched LiDAR scan -- the exact same
values collision detection/reward/the privileged risk label already use
elsewhere in this node. "Noisy" is whatever the agent's OWN observation
vector for this tick actually contains: ``sensor_noise``'s
``measured_pose``/``measured_velocity``/``measured_steering``/
``apply_lidar_noise`` outputs, composed on top of whatever
``domain_randomization`` noise was already applied (see
``environment_node.py::_observation_obs_state``/``_assemble_state_vector``)
-- so the LiDAR perturbation stats below capture the TOTAL agent-observed
deviation from ground truth, from every noise source combined, not
``sensor_noise`` alone. When ``profile.sensor_noise.enabled`` is False AND
``domain_randomization`` never touched this tick's LiDAR/odometry, noisy
fields are bit-identical to their ground-truth counterparts (a genuine
"no noise happened" result, not a placeholder).

Wire layout (flat float32 list, fixed length -- no variable-length
section, unlike risk_telemetry's per-candidate tail), schema_version=1:

    [0]  schema_version
    [1]  step_id                  (0 == this episode's /reset snapshot,
                                    exactly like risk_telemetry's own
                                    step_id==0 reset-marker convention)
    [2]  valid                    (fields [6..25] meaningful iff true)
    [3]  reset_generation
    [4]  episode_id               (this episode's seed)
    [5]  sim_timestamp_sec        (NaN if /clock hasn't delivered yet)
    [6]  gt_x
    [7]  gt_y
    [8]  gt_yaw
    [9]  noisy_x
    [10] noisy_y
    [11] noisy_yaw
    [12] gt_v_mps
    [13] gt_yaw_rate_radps
    [14] gt_steering_rad
    [15] noisy_v_mps
    [16] noisy_yaw_rate_radps
    [17] noisy_steering_rad
    [18] drift_x_m                (current OU localization-drift state --
                                    0.0 whenever sensor_noise is disabled)
    [19] drift_y_m
    [20] drift_yaw_rad
    [21] localization_latency_steps (the CONFIGURED delay -- the realized
                                    per-tick delay is always exactly this
                                    many ticks once the latency buffer has
                                    filled, see sensor_noise.measured_pose)
    [22] lidar_beam_count
    [23] lidar_dropout_count      (beams that read as max_range in the
                                    noisy observation but did NOT in ground
                                    truth -- the standard "no return"
                                    dropout convention, see
                                    sensor_noise.apply_lidar_noise)
    [24] lidar_perturbation_mean_m (mean absolute noisy-vs-ground-truth
                                    per-beam deviation)
    [25] lidar_perturbation_max_m
    [26] invalid_reason           (DiagnosticsInvalidReason int code; 0/NONE
                                    when `valid` is True)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Sequence

SCHEMA_VERSION = 1
_WIRE_LEN = 27


class DiagnosticsInvalidReason(IntEnum):
    NONE = 0  # valid=True
    COMPUTATION_EXCEPTION = 1  # environment_node.py's own computation raised
    # Client-side-only sentinels (never produced by environment_node.py
    # itself -- assigned by the TRAINER when it could not pair a real
    # message to a given (reset_generation, step_id) within its poll
    # budget, mirroring risk_telemetry's own POLL_TIMEOUT convention).
    POLL_TIMEOUT = 2


@dataclass(frozen=True)
class SensorDiagnostics:
    step_id: int
    valid: bool
    reset_generation: int = 0
    episode_id: int = 0
    sim_timestamp_sec: float = float("nan")
    gt_x: float = float("nan")
    gt_y: float = float("nan")
    gt_yaw: float = float("nan")
    noisy_x: float = float("nan")
    noisy_y: float = float("nan")
    noisy_yaw: float = float("nan")
    gt_v_mps: float = float("nan")
    gt_yaw_rate_radps: float = float("nan")
    gt_steering_rad: float = float("nan")
    noisy_v_mps: float = float("nan")
    noisy_yaw_rate_radps: float = float("nan")
    noisy_steering_rad: float = float("nan")
    drift_x_m: float = 0.0
    drift_y_m: float = 0.0
    drift_yaw_rad: float = 0.0
    localization_latency_steps: int = 0
    lidar_beam_count: int = 0
    lidar_dropout_count: int = 0
    lidar_perturbation_mean_m: float = 0.0
    lidar_perturbation_max_m: float = 0.0
    invalid_reason: int = int(DiagnosticsInvalidReason.NONE)


def invalid(step_id: int, reset_generation: int = 0, episode_id: int = 0,
            sim_timestamp_sec: float = float("nan"),
            reason: "DiagnosticsInvalidReason" = DiagnosticsInvalidReason.NONE) -> SensorDiagnostics:
    """The explicit "no diagnostics this step" value -- NEVER represented
    by silently omitting the message or a bare NaN with no companion flag
    (mirrors risk_telemetry.invalid's own convention/rationale)."""
    return SensorDiagnostics(
        step_id=step_id, valid=False, reset_generation=reset_generation, episode_id=episode_id,
        sim_timestamp_sec=sim_timestamp_sec, invalid_reason=int(reason),
    )


def lidar_perturbation_stats(ground_truth: Sequence[float], noisy: Sequence[float],
                              max_range_m: float, dropout_eps_m: float = 1e-6):
    """Returns ``(dropout_count, perturbation_mean_m, perturbation_max_m)``
    -- pure/ROS-free so it's directly unit-testable. A beam counts as
    DROPPED iff the noisy reading is (within ``dropout_eps_m`` of) exactly
    ``max_range_m`` while ground truth was NOT -- the same "no return"
    convention ``sensor_noise.apply_lidar_noise``/
    ``domain_randomizer.apply_lidar_noise`` both already use, inferred here
    rather than threaded through as a separate return value so this stays
    usable regardless of which noise source(s) actually produced the
    dropout. Perturbation mean/max are computed over ALL beams (dropped or
    not) -- a dropped beam's own large gt-vs-max_range delta is a genuine
    part of "how much did the agent's observation deviate from reality"."""
    if len(ground_truth) != len(noisy):
        raise ValueError(f"lidar_perturbation_stats: length mismatch gt={len(ground_truth)} noisy={len(noisy)}")
    n = len(ground_truth)
    if n == 0:
        return 0, 0.0, 0.0
    dropout_count = 0
    abs_deltas: List[float] = []
    for gt, ny in zip(ground_truth, noisy):
        delta = abs(float(ny) - float(gt))
        abs_deltas.append(delta)
        if abs(float(ny) - max_range_m) <= dropout_eps_m and abs(float(gt) - max_range_m) > dropout_eps_m:
            dropout_count += 1
    return dropout_count, sum(abs_deltas) / n, max(abs_deltas)


def encode(d: SensorDiagnostics) -> List[float]:
    return [
        float(SCHEMA_VERSION), float(d.step_id), 1.0 if d.valid else 0.0,
        float(d.reset_generation), float(d.episode_id), float(d.sim_timestamp_sec),
        float(d.gt_x), float(d.gt_y), float(d.gt_yaw),
        float(d.noisy_x), float(d.noisy_y), float(d.noisy_yaw),
        float(d.gt_v_mps), float(d.gt_yaw_rate_radps), float(d.gt_steering_rad),
        float(d.noisy_v_mps), float(d.noisy_yaw_rate_radps), float(d.noisy_steering_rad),
        float(d.drift_x_m), float(d.drift_y_m), float(d.drift_yaw_rad),
        float(d.localization_latency_steps),
        float(d.lidar_beam_count), float(d.lidar_dropout_count),
        float(d.lidar_perturbation_mean_m), float(d.lidar_perturbation_max_m),
        float(d.invalid_reason),
    ]


def decode(data: List[float]) -> SensorDiagnostics:
    if len(data) != _WIRE_LEN:
        raise ValueError(f"sensor_diagnostics payload length {len(data)} != expected {_WIRE_LEN}")
    version = int(round(data[0]))
    if version != SCHEMA_VERSION:
        raise ValueError(f"sensor_diagnostics schema_version mismatch: got {version}, expected {SCHEMA_VERSION}")
    return SensorDiagnostics(
        step_id=int(round(data[1])), valid=bool(data[2] >= 0.5),
        reset_generation=int(round(data[3])), episode_id=int(round(data[4])), sim_timestamp_sec=data[5],
        gt_x=data[6], gt_y=data[7], gt_yaw=data[8],
        noisy_x=data[9], noisy_y=data[10], noisy_yaw=data[11],
        gt_v_mps=data[12], gt_yaw_rate_radps=data[13], gt_steering_rad=data[14],
        noisy_v_mps=data[15], noisy_yaw_rate_radps=data[16], noisy_steering_rad=data[17],
        drift_x_m=data[18], drift_y_m=data[19], drift_yaw_rad=data[20],
        localization_latency_steps=int(round(data[21])),
        lidar_beam_count=int(round(data[22])), lidar_dropout_count=int(round(data[23])),
        lidar_perturbation_mean_m=data[24], lidar_perturbation_max_m=data[25],
        invalid_reason=int(round(data[26])),
    )
