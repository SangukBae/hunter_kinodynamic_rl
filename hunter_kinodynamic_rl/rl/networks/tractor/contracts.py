"""Typed tensor and configuration contracts for TRACTOR-TQC.

No untyped dictionary crosses the encode/propose/score boundary.  Runtime
checks are deliberately strict because a shape-compatible tensor with a
different meaning is an invalid research sample, not a recoverable input.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Optional, Tuple

import torch


TRACTOR_ARCHITECTURE_REVISION = "tractor-tqc-r1"
TRACTOR_ACTION_SCHEMA = "trajectory_kappa_vref_L_v1"


@dataclass(frozen=True)
class TractorConfig:
    variant_id: str = "tractor_core_v1"
    t_obs: int = 4
    n_scan: int = 80
    tail_dim: int = 8
    grid_height: int = 64
    grid_width: int = 64
    resolution_m: float = 0.25
    x_min_m: float = -8.0
    y_min_m: float = -8.0
    max_range_m: float = 10.0
    ray_free_samples: int = 16
    scene_channels: int = 32
    ego_dim: int = 64
    plant_dim: int = 32
    interaction_dim: int = 128
    num_candidates: int = 8
    horizon_steps: int = 15
    dt_out_sec: float = 0.2
    dt_dyn_sec: float = 0.1
    sparse_tube_samples: int = 32
    n_critics: int = 2
    n_quantiles: int = 25
    n_causes: int = 3
    n_severity_quantiles: int = 7
    severity_quantile_levels: Tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
    residual_members: int = 0
    risk_members: int = 1
    wheelbase_m: float = 0.550
    steering_limit_rad: float = math.radians(18.8883201008)
    steering_rate_rad_s: float = math.radians(35.0)
    min_speed_mps: float = 0.0
    max_speed_mps: float = 1.3333333333
    accel_limit_mps2: float = 2.0
    brake_decel_mps2: float = 2.0
    speed_lag_tau_sec: float = 0.25
    min_arc_m: float = 0.5
    max_arc_m: float = 3.0
    min_horizon_sec: float = 0.5
    max_horizon_sec: float = 3.0
    min_safety_horizon_sec: float = 1.5
    footprint_radius_m: float = 0.58
    process_noise_xy_m2: float = 0.0025
    process_noise_yaw_rad2: float = 0.0004
    log_std_min: float = -5.0
    log_std_max: float = 1.0

    def validate(self) -> None:
        positive_ints = (
            "t_obs", "n_scan", "tail_dim", "grid_height", "grid_width",
            "ray_free_samples", "scene_channels", "ego_dim", "plant_dim",
            "interaction_dim", "num_candidates", "horizon_steps",
            "sparse_tube_samples", "n_critics", "n_quantiles", "n_causes",
            "n_severity_quantiles", "risk_members",
        )
        for name in positive_ints:
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.residual_members < 0:
            raise ValueError("residual_members must be >= 0")
        if self.num_candidates < 2:
            raise ValueError("TRACTOR requires at least a base and stop candidate")
        if len(self.severity_quantile_levels) != self.n_severity_quantiles:
            raise ValueError("severity quantile level count does not match n_severity_quantiles")
        if any(
            not 0.0 < float(level) < 1.0 for level in self.severity_quantile_levels
        ) or any(
            left >= right for left, right in zip(
                self.severity_quantile_levels, self.severity_quantile_levels[1:]
            )
        ):
            raise ValueError("severity quantile levels must be strictly increasing inside (0,1)")
        if self.variant_id == "tractor_core_v1" and self.residual_members != 0:
            raise ValueError("tractor_core_v1 requires residual_members=0")
        if self.variant_id == "tractor_residual_v1" and self.residual_members != 1:
            raise ValueError("tractor_residual_v1 requires residual_members=1")
        if self.variant_id == "tractor_ensemble_v1" and (
            self.residual_members < 2 or self.risk_members < 2
        ):
            raise ValueError("tractor_ensemble_v1 requires residual_members>=2 and risk_members>=2")
        if self.grid_height != self.grid_width:
            raise ValueError("v1 requires a square BEV grid")
        if self.resolution_m <= 0.0 or self.dt_out_sec <= 0.0 or self.dt_dyn_sec <= 0.0:
            raise ValueError("resolution and time steps must be positive")
        if self.dt_dyn_sec > self.dt_out_sec:
            raise ValueError("dt_dyn_sec must be <= dt_out_sec")
        if not (0.0 < self.wheelbase_m and 0.0 < self.steering_limit_rad < math.pi / 2):
            raise ValueError("invalid Ackermann geometry")
        if not (0.0 <= self.min_speed_mps < self.max_speed_mps):
            raise ValueError("speed bounds must satisfy 0 <= min < max")
        if not (0.0 < self.min_arc_m < self.max_arc_m):
            raise ValueError("arc bounds must satisfy 0 < min < max")
        if not (
            0.0 < self.min_horizon_sec
            <= self.min_safety_horizon_sec
            <= self.max_horizon_sec
            <= self.horizon_steps * self.dt_out_sec + 1e-9
        ):
            raise ValueError("inconsistent horizon contract")
        if self.footprint_radius_m <= 0.0:
            raise ValueError("footprint_radius_m must be positive")

    @property
    def observation_dim(self) -> int:
        return self.t_obs * self.n_scan + self.tail_dim

    @property
    def q_m_eff(self) -> int:
        return max(1, self.residual_members)

    def fingerprint(self) -> str:
        self.validate()
        payload = {
            "architecture_revision": TRACTOR_ARCHITECTURE_REVISION,
            "action_schema": TRACTOR_ACTION_SCHEMA,
            "config": asdict(self),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class TractorInputs:
    observation: torch.Tensor
    scan_valid: torch.Tensor
    motion_delta: torch.Tensor
    motion_valid: torch.Tensor
    decision_timestamp_sec: torch.Tensor
    previous_intent_valid: torch.Tensor
    previous_command_published: torch.Tensor
    previous_command_valid: torch.Tensor
    vehicle_response_valid: torch.Tensor
    localization_covariance: torch.Tensor
    localization_valid: torch.Tensor
    localization_confidence: torch.Tensor
    localization_confidence_valid: torch.Tensor
    sensor_freshness_sec: torch.Tensor
    sensor_freshness_valid: torch.Tensor
    scene_reset: torch.Tensor
    response_reset: torch.Tensor
    previous_scene_hidden: Optional[torch.Tensor] = None
    previous_scene_hidden_valid: Optional[torch.Tensor] = None
    previous_response_hidden: Optional[torch.Tensor] = None
    previous_response_hidden_valid: Optional[torch.Tensor] = None

    def index_select(self, indices: torch.Tensor) -> "TractorInputs":
        """Select batch rows without inventing values for absent recurrent state."""
        def take(value):
            return None if value is None else value.index_select(0, indices)

        return TractorInputs(**{
            name: take(getattr(self, name))
            for name in self.__dataclass_fields__
        })

    def validate(self, cfg: TractorConfig) -> None:
        cfg.validate()
        if self.observation.ndim != 2 or self.observation.shape[1] != cfg.observation_dim:
            raise ValueError(
                f"observation must be (B,{cfg.observation_dim}), got {tuple(self.observation.shape)}"
            )
        b = self.observation.shape[0]
        expected = {
            "scan_valid": (b, cfg.t_obs, cfg.n_scan),
            "motion_delta": (b, cfg.t_obs - 1, 4),
            "motion_valid": (b, cfg.t_obs - 1),
            "decision_timestamp_sec": (b, 1),
            "previous_intent_valid": (b, 1),
            "previous_command_published": (b, 2),
            "previous_command_valid": (b, 1),
            "vehicle_response_valid": (b, 3),
            "localization_covariance": (b, 3, 3),
            "localization_valid": (b, 1),
            "localization_confidence": (b, 1),
            "localization_confidence_valid": (b, 1),
            "sensor_freshness_sec": (b, 1),
            "sensor_freshness_valid": (b, 1),
            "scene_reset": (b, 1),
            "response_reset": (b, 1),
        }
        for name, shape in expected.items():
            if tuple(getattr(self, name).shape) != shape:
                raise ValueError(f"{name} must be {shape}, got {tuple(getattr(self, name).shape)}")
        mask_fields = (
            "scan_valid", "motion_valid", "previous_intent_valid",
            "previous_command_valid", "vehicle_response_valid", "localization_valid",
            "localization_confidence_valid", "sensor_freshness_valid",
            "scene_reset", "response_reset",
        )
        for name in mask_fields:
            if getattr(self, name).dtype != torch.bool:
                raise ValueError(f"{name} must have bool dtype")
        optional_shapes = {
            "previous_scene_hidden": (b, cfg.scene_channels, cfg.grid_height, cfg.grid_width),
            "previous_scene_hidden_valid": (b, 1),
            "previous_response_hidden": (b, cfg.plant_dim),
            "previous_response_hidden_valid": (b, 1),
        }
        for name, shape in optional_shapes.items():
            value = getattr(self, name)
            if value is not None and tuple(value.shape) != shape:
                raise ValueError(f"{name} must be {shape}, got {tuple(value.shape)}")
        for name in ("previous_scene_hidden_valid", "previous_response_hidden_valid"):
            value = getattr(self, name)
            if value is not None and value.dtype != torch.bool:
                raise ValueError(f"{name} must have bool dtype")
        finite_fields = (
            "observation", "motion_delta", "decision_timestamp_sec",
            "previous_command_published", "localization_covariance",
            "localization_confidence", "sensor_freshness_sec",
        )
        for name in finite_fields:
            if not torch.isfinite(getattr(self, name)).all():
                raise ValueError(f"{name} contains non-finite values")
        for name in ("previous_scene_hidden", "previous_response_hidden"):
            value = getattr(self, name)
            if value is not None and not torch.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values")
        if not torch.allclose(
            self.localization_covariance,
            self.localization_covariance.transpose(-1, -2),
            atol=1e-5,
            rtol=0.0,
        ):
            raise ValueError("localization_covariance must be symmetric")
        if torch.linalg.eigvalsh(self.localization_covariance).amin() < -1e-6:
            raise ValueError("localization_covariance must be positive semidefinite")
        valid_motion_dt = self.motion_delta[..., 3][self.motion_valid.bool()]
        if valid_motion_dt.numel() and (valid_motion_dt <= 0.0).any():
            raise ValueError("valid motion edges require positive dt")
        valid_confidence = self.localization_confidence[self.localization_confidence_valid.bool()]
        if valid_confidence.numel() and (
            (valid_confidence < 0.0) | (valid_confidence > 1.0)
        ).any():
            raise ValueError("valid localization confidence must be in [0,1]")


@dataclass(frozen=True)
class DecisionContext:
    goal_xy_m: torch.Tensor
    response: torch.Tensor
    response_valid: torch.Tensor
    previous_command_published: torch.Tensor
    localization_covariance: torch.Tensor
    timestamp_sec: torch.Tensor


@dataclass(frozen=True)
class HealthState:
    scene_valid: torch.Tensor
    response_valid: torch.Tensor
    localization_valid: torch.Tensor
    sensor_valid: torch.Tensor

    @property
    def model_valid(self) -> torch.Tensor:
        return self.scene_valid & self.response_valid & self.localization_valid & self.sensor_valid


@dataclass(frozen=True)
class BeliefState:
    scene_feature: torch.Tensor
    occupancy: torch.Tensor
    dynamic_flow: torch.Tensor
    future_feature: torch.Tensor
    future_occupancy: torch.Tensor
    future_dynamic_flow: torch.Tensor
    ego_latent: torch.Tensor
    plant_latent: torch.Tensor
    next_scene_hidden: torch.Tensor
    next_scene_hidden_valid: torch.Tensor
    next_response_hidden: torch.Tensor
    next_response_hidden_valid: torch.Tensor
    health: HealthState


@dataclass(frozen=True)
class CandidateSet:
    normalized_actions: torch.Tensor
    present: torch.Tensor
    is_stop: torch.Tensor
    source: str

    def permute(self, order: torch.Tensor) -> "CandidateSet":
        return CandidateSet(
            normalized_actions=self.normalized_actions[:, order],
            present=self.present[:, order],
            is_stop=self.is_stop[:, order],
            source=self.source,
        )


@dataclass(frozen=True)
class RolloutBatch:
    physical_actions: torch.Tensor
    poses: torch.Tensor
    covariance: torch.Tensor
    horizon_mask: torch.Tensor
    model_valid: torch.Tensor


@dataclass(frozen=True)
class TubeBatch:
    sample_xy: torch.Tensor
    sample_weight: torch.Tensor
    sample_valid: torch.Tensor
    oob_mass: torch.Tensor
    horizon_mask: torch.Tensor


@dataclass(frozen=True)
class InteractionOutput:
    per_time: torch.Tensor
    aggregated: torch.Tensor
    overlap_by_cause: torch.Tensor
    candidate_valid: torch.Tensor


@dataclass(frozen=True)
class ReturnRiskOutput:
    return_quantiles: torch.Tensor
    hazard: torch.Tensor
    clearance_quantiles: torch.Tensor
    stopping_quantiles: torch.Tensor
    candidate_valid: torch.Tensor


@dataclass(frozen=True)
class SelectionOutput:
    selected_index: torch.Tensor
    selected_action: torch.Tensor
    feasible: torch.Tensor
    score: torch.Tensor
    event_probability_ucb: torch.Tensor
    clearance_lcb: torch.Tensor
    stopping_lcb: torch.Tensor
    event_probability_std: torch.Tensor
    clearance_std: torch.Tensor
    stopping_std: torch.Tensor
    dispersion_available: torch.Tensor
    reason: Tuple[str, ...]
