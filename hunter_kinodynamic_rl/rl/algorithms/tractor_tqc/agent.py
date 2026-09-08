"""Training-ready TRACTOR-TQC agent; importing it never starts training."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn

from hunter_kinodynamic_rl.rl.networks.tractor import TractorConfig, TractorInputs, TractorTQC
from hunter_kinodynamic_rl.rl.networks.tractor.contracts import CandidateSet
from hunter_kinodynamic_rl.rl.networks.tractor.model import TractorValueTarget
from hunter_kinodynamic_rl.rl.networks.tractor.selector import cumulative_event_probability

from .losses import (
    competing_risk_nll, masked_flow_loss, masked_occupancy_nll,
    masked_pinball_loss, masked_response_loss, quantile_huber_loss,
    truncated_bellman_target,
)
from .optimizer_groups import assert_disjoint_complete, freeze_modules
from .target_update import ema_update


class EntropyTemperature(nn.Module):
    def __init__(self, initial_alpha: float):
        super().__init__()
        if initial_alpha <= 0.0:
            raise ValueError("initial_alpha must be positive")
        self.log_alpha = nn.Parameter(torch.tensor(float(initial_alpha)).log())

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()


@dataclass(frozen=True)
class TractorAgentConfig:
    representation_lr: float = 3e-4
    value_lr: float = 3e-4
    risk_lr: float = 3e-4
    actor_lr: float = 3e-4
    entropy_lr: float = 3e-4
    initial_alpha: float = 0.2
    target_entropy: float = -3.0
    top_quantiles_to_drop_per_net: int = 2
    target_tau: float = 0.005
    actor_risk_weight: float = 1.0
    max_gradient_norm: float = 20.0

    def validate(self) -> None:
        rates = (self.representation_lr, self.value_lr, self.risk_lr, self.actor_lr, self.entropy_lr)
        if any(rate <= 0.0 for rate in rates):
            raise ValueError("all learning rates must be positive")
        if self.top_quantiles_to_drop_per_net < 0 or not 0.0 < self.target_tau <= 1.0:
            raise ValueError("invalid target truncation or EMA coefficient")


@dataclass(frozen=True)
class TractorTrainingBatch:
    current: TractorInputs
    next: TractorInputs
    action_normalized_requested: torch.Tensor
    reward: torch.Tensor
    discount_factor: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    bellman_sample_valid: torch.Tensor

    def validate(self, config: TractorConfig) -> None:
        self.current.validate(config)
        self.next.validate(config)
        b = self.action_normalized_requested.shape[0]
        expected = {
            "action_normalized_requested": (b, 3), "reward": (b, 1),
            "discount_factor": (b, 1), "terminated": (b, 1),
            "truncated": (b, 1), "bellman_sample_valid": (b, 1),
        }
        for name, shape in expected.items():
            if tuple(getattr(self, name).shape) != shape:
                raise ValueError(f"{name} must be {shape}")
        if self.current.observation.shape[0] != b or self.next.observation.shape[0] != b:
            raise ValueError("input batch dimensions do not match transition tensors")
        if (self.terminated.bool() & self.truncated.bool()).any():
            raise ValueError("terminated and truncated are mutually exclusive")


@dataclass(frozen=True)
class TractorRiskBatch:
    """Candidate-aligned R5/R6 supervision for one flattened sequence batch."""

    current: TractorInputs
    candidates: CandidateSet
    event_observed: torch.Tensor
    event_step: torch.Tensor
    event_cause: torch.Tensor
    censor_step: torch.Tensor
    event_label_valid: torch.Tensor
    clearance_m: torch.Tensor
    clearance_valid: torch.Tensor
    stopping_margin_m: torch.Tensor
    stopping_margin_valid: torch.Tensor

    def validate(self, config: TractorConfig) -> None:
        self.current.validate(config)
        b = self.current.observation.shape[0]
        k, h = config.num_candidates, config.horizon_steps
        if tuple(self.candidates.normalized_actions.shape) != (b, k, 3):
            raise ValueError(f"candidate actions must be {(b, k, 3)}")
        if tuple(self.candidates.present.shape) != (b, k):
            raise ValueError(f"candidate present mask must be {(b, k)}")
        if tuple(self.candidates.is_stop.shape) != (b, k):
            raise ValueError(f"candidate stop mask must be {(b, k)}")
        expected = {
            "event_observed": (b, k), "event_step": (b, k),
            "event_cause": (b, k), "censor_step": (b, k),
            "event_label_valid": (b, k), "clearance_m": (b, k, h),
            "clearance_valid": (b, k, h),
            "stopping_margin_m": (b, k, h),
            "stopping_margin_valid": (b, k, h),
        }
        for name, shape in expected.items():
            if tuple(getattr(self, name).shape) != shape:
                raise ValueError(f"{name} must be {shape}")
        for name in (
            "event_observed", "event_label_valid", "clearance_valid",
            "stopping_margin_valid",
        ):
            if getattr(self, name).dtype != torch.bool:
                raise ValueError(f"{name} must have bool dtype")
        if self.candidates.present.dtype != torch.bool or self.candidates.is_stop.dtype != torch.bool:
            raise ValueError("candidate masks must have bool dtype")
        if (self.event_label_valid & ~self.candidates.present).any():
            raise ValueError("valid event labels require present candidates")


@dataclass(frozen=True)
class TractorRepresentationBatch:
    """Dense Stage-3 supervision aligned to flattened sequence rows."""

    current: TractorInputs
    current_bev_class_target: torch.Tensor
    current_bev_class_valid: torch.Tensor
    current_dynamic_flow_target: torch.Tensor
    current_dynamic_flow_valid: torch.Tensor
    future_bev_class_target: torch.Tensor
    future_bev_class_valid: torch.Tensor
    future_dynamic_flow_target: torch.Tensor
    future_dynamic_flow_valid: torch.Tensor
    vehicle_response_target: torch.Tensor
    vehicle_response_target_valid: torch.Tensor
    action_normalized_requested: torch.Tensor
    transition_dt_sec: torch.Tensor

    def validate(self, config: TractorConfig) -> None:
        self.current.validate(config)
        b = self.current.observation.shape[0]
        g = (config.grid_height, config.grid_width)
        expected = {
            "current_bev_class_target": (b, *g),
            "current_bev_class_valid": (b, *g),
            "current_dynamic_flow_target": (b, 2, *g),
            "current_dynamic_flow_valid": (b, *g),
            "future_bev_class_target": (b, config.horizon_steps, *g),
            "future_bev_class_valid": (b, config.horizon_steps, *g),
            "future_dynamic_flow_target": (b, config.horizon_steps, 2, *g),
            "future_dynamic_flow_valid": (b, config.horizon_steps, *g),
            "vehicle_response_target": (b, 3),
            "vehicle_response_target_valid": (b, 3),
            "action_normalized_requested": (b, 3),
            "transition_dt_sec": (b, 1),
        }
        for name, shape in expected.items():
            if tuple(getattr(self, name).shape) != shape:
                raise ValueError(f"{name} must be {shape}")
        for name in (
            "current_bev_class_valid", "current_dynamic_flow_valid",
            "future_bev_class_valid", "future_dynamic_flow_valid",
            "vehicle_response_target_valid",
        ):
            if getattr(self, name).dtype != torch.bool:
                raise ValueError(f"{name} must have bool dtype")
        if not torch.isfinite(self.action_normalized_requested).all() or not torch.isfinite(
            self.transition_dt_sec
        ).all() or (self.transition_dt_sec <= 0.0).any():
            raise ValueError("representation action/dt inputs must be finite with positive dt")


def _one_candidate(actions: torch.Tensor, source: str) -> CandidateSet:
    present = torch.ones(actions.shape[0], 1, dtype=torch.bool, device=actions.device)
    return CandidateSet(actions[:, None, :], present, present.new_zeros(present.shape), source)


class TractorAgent:
    def __init__(
        self,
        model_config: TractorConfig,
        agent_config: TractorAgentConfig = TractorAgentConfig(),
        *,
        device: str | torch.device = "cpu",
        target_seed: int = 0,
    ):
        model_config.validate()
        agent_config.validate()
        if agent_config.top_quantiles_to_drop_per_net >= model_config.n_quantiles:
            raise ValueError("top_quantiles_to_drop_per_net must be smaller than n_quantiles")
        self.model_config = model_config
        self.config = agent_config
        self.device = torch.device(device)
        self.online = TractorTQC(model_config).to(self.device)
        self.target = TractorValueTarget(self.online).to(self.device).eval()
        self.target.requires_grad_(False)
        self.temperature = EntropyTemperature(agent_config.initial_alpha).to(self.device)
        self.target_generator = torch.Generator(device=self.device.type)
        self.target_generator.manual_seed(int(target_seed))
        self.update_step = 0

        groups = self.parameter_groups()
        assert_disjoint_complete(self.online, groups)
        self.value_path_optimizer = torch.optim.Adam([
            {"params": groups["representation"], "lr": agent_config.representation_lr},
            {"params": groups["value"], "lr": agent_config.value_lr},
        ])
        # Backward-compatible runtime aliases.  Checkpoints expose only the
        # single semantic owner below, never duplicate optimizer payloads.
        self.representation_optimizer = self.value_path_optimizer
        self.value_optimizer = self.value_path_optimizer
        self.risk_optimizer = torch.optim.Adam(groups["risk"], lr=agent_config.risk_lr)
        self.actor_optimizer = torch.optim.Adam(groups["actor"], lr=agent_config.actor_lr)
        self.entropy_optimizer = torch.optim.Adam(self.temperature.parameters(), lr=agent_config.entropy_lr)

    def parameter_groups(self) -> Dict[str, list[nn.Parameter]]:
        value_modules = (
            self.online.rollout, self.online.interaction,
            self.online.temporal_aggregator, self.online.return_critic,
        )
        return {
            "representation": list(self.online.belief_encoder.parameters()),
            "value": [p for module in value_modules for p in module.parameters()],
            "risk": list(self.online.risk_head.parameters()),
            "actor": list(self.online.actor.parameters()),
        }

    def _critic_objective(self, batch: TractorTrainingBatch) -> torch.Tensor:
        batch.validate(self.model_config)
        belief, context = self.online.encode(batch.current)
        current = self.online.score(
            belief, context, _one_candidate(batch.action_normalized_requested, "stored_requested")
        ).return_quantiles[:, 0]
        target_count = self.model_config.n_critics * self.model_config.n_quantiles
        total_drop = self.config.top_quantiles_to_drop_per_net * self.model_config.n_critics
        target_count -= total_drop
        target = batch.reward.expand(-1, target_count).clone()
        bootstrap = batch.bellman_sample_valid.reshape(-1).bool() & ~batch.terminated.reshape(-1).bool()
        with torch.no_grad():
            if bootstrap.any():
                indices = torch.nonzero(bootstrap, as_tuple=False).flatten()
                next_belief, next_context = self.target.encode(batch.next.index_select(indices))
                next_action, next_log_prob = self.online.actor.sample(
                    next_belief, generator=self.target_generator
                )
                next_values = self.target.score(
                    next_belief, next_context, _one_candidate(next_action, "target_policy")
                )[:, 0]
                selected_target = truncated_bellman_target(
                    batch.reward.index_select(0, indices),
                    batch.discount_factor.index_select(0, indices),
                    batch.terminated.index_select(0, indices), next_values,
                    next_log_prob, self.temperature.alpha,
                    total_drop,
                )
                target.index_copy_(0, indices, selected_target)
        return quantile_huber_loss(current, target, batch.bellman_sample_valid)

    def critic_step(self, batch: TractorTrainingBatch, *, update_target: bool = True) -> Dict[str, float]:
        loss = self._critic_objective(batch)
        self.value_path_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        parameters = self.parameter_groups()["representation"] + self.parameter_groups()["value"]
        finite = bool(torch.isfinite(loss).item()) and all(
            parameter.grad is None or torch.isfinite(parameter.grad).all().item() for parameter in parameters
        )
        if not finite:
            self.value_path_optimizer.zero_grad(set_to_none=True)
            return {"loss/critic": float("nan"), "update/applied": 0.0}
        torch.nn.utils.clip_grad_norm_(parameters, self.config.max_gradient_norm)
        self.value_path_optimizer.step()
        if update_target:
            self.update_target()
        self.update_step += 1
        return {"loss/critic": float(loss.detach()), "update/applied": 1.0}

    def _representation_objective(
        self,
        batch: TractorRepresentationBatch,
        *,
        occupancy_weight: float = 1.0,
        flow_weight: float = 0.5,
        response_weight: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        batch.validate(self.model_config)
        if min(occupancy_weight, flow_weight, response_weight) < 0.0:
            raise ValueError("representation loss weights must be non-negative")
        belief, _context = self.online.encode(batch.current)
        current_occupancy = masked_occupancy_nll(
            belief.occupancy, batch.current_bev_class_target,
            batch.current_bev_class_valid,
        )
        future_occupancy = masked_occupancy_nll(
            belief.future_occupancy, batch.future_bev_class_target,
            batch.future_bev_class_valid,
        )
        occupancy_loss = 0.5 * (current_occupancy + future_occupancy)
        current_flow = masked_flow_loss(
            belief.dynamic_flow, batch.current_dynamic_flow_target,
            batch.current_dynamic_flow_valid,
        )
        future_flow = masked_flow_loss(
            belief.future_dynamic_flow, batch.future_dynamic_flow_target,
            batch.future_dynamic_flow_valid,
        )
        flow_loss = 0.5 * (current_flow + future_flow)
        response_loss = masked_response_loss(
            belief.response_prediction, batch.vehicle_response_target,
            batch.vehicle_response_target_valid,
        )
        residual_loss = self._residual_supervision_loss(batch, belief)
        loss = (
            occupancy_weight * occupancy_loss
            + flow_weight * flow_loss
            + response_weight * response_loss
            + response_weight * residual_loss
        )
        valid_count = sum(int(mask.sum().item()) for mask in (
            batch.current_bev_class_valid, batch.future_bev_class_valid,
            batch.current_dynamic_flow_valid, batch.future_dynamic_flow_valid,
            batch.vehicle_response_target_valid,
        ))
        return loss, occupancy_loss, flow_loss, response_loss, residual_loss, valid_count

    def _residual_supervision_loss(
        self, batch: TractorRepresentationBatch, belief,
    ) -> torch.Tensor:
        """One measured-dt teacher-forced response target for A8/A9 residual members."""
        if self.model_config.residual_members == 0:
            return belief.plant_latent.sum() * 0.0
        physical = self.online.rollout.decode(batch.action_normalized_requested)
        kappa, v_ref, arc = physical.unbind(-1)
        tail = batch.current.observation[:, self.model_config.t_obs * self.model_config.n_scan :]
        response = tail[:, 5:8]
        v, steering = response[:, 0], response[:, 2]
        dt = batch.transition_dt_sec[:, 0].clamp_max(self.model_config.dt_out_sec)
        target_steering = torch.atan(self.model_config.wheelbase_m * kappa)
        raw_horizon = torch.where(
            v_ref <= 0.05,
            torch.full_like(v_ref, self.model_config.max_horizon_sec),
            arc / v_ref.clamp_min(0.05),
        )
        commit_horizon = raw_horizon.clamp(
            self.model_config.min_horizon_sec, self.model_config.max_horizon_sec,
        )
        blend = (dt / commit_horizon.clamp_min(dt)).clamp(0.0, 1.0)
        requested_steering = steering + blend * (target_steering - steering)
        rate = torch.where(
            v_ref.abs() >= v.abs(),
            torch.full_like(v, self.model_config.accel_limit_mps2),
            torch.full_like(v, self.model_config.brake_decel_mps2),
        )
        v_rate = v + (v_ref - v).clamp(-rate * dt, rate * dt)
        steering_rate = self.model_config.steering_rate_rad_s * dt
        steering_next = steering + (requested_steering - steering).clamp(
            -steering_rate, steering_rate,
        )
        if self.model_config.speed_lag_tau_sec > 0.0:
            alpha = 1.0 - torch.exp(-dt / self.model_config.speed_lag_tau_sec)
            v_next = v + alpha * (v_rate - v)
        else:
            v_next = v_rate
        nominal = torch.stack((
            v_next,
            v_next * torch.tan(steering_next) / self.model_config.wheelbase_m,
            steering_next,
        ), dim=-1)
        valid = (
            batch.vehicle_response_target_valid.all(dim=-1)
            & batch.current.vehicle_response_valid.all(dim=-1)
        )
        if not valid.any():
            return nominal.sum() * 0.0
        elapsed = (dt / (self.model_config.horizon_steps * self.model_config.dt_out_sec)).unsqueeze(-1)
        scaled_dt = (dt / self.model_config.dt_dyn_sec).unsqueeze(-1)
        losses = []
        for residual in self.online.rollout.residuals.members:
            delta, variance = residual(
                belief.plant_latent, physical, response,
                batch.current.vehicle_response_valid, nominal, scaled_dt, elapsed,
            )
            predicted_v = nominal[:, 0] + delta[:, 0]
            predicted_steering = nominal[:, 2] + delta[:, 1]
            target = batch.vehicle_response_target
            error = torch.stack((predicted_v - target[:, 0], predicted_steering - target[:, 2]), -1)
            gaussian = 0.5 * (error.square() / variance + variance.log()).mean(dim=-1)
            predicted_yaw_rate = (
                predicted_v * torch.tan(predicted_steering) / self.model_config.wheelbase_m
            )
            yaw_consistency = torch.nn.functional.smooth_l1_loss(
                predicted_yaw_rate[valid], target[:, 1][valid], reduction="none",
            )
            losses.append((gaussian[valid] + yaw_consistency).mean())
        return torch.stack(losses).mean()

    def representation_step(
        self,
        batch: TractorRepresentationBatch,
        *,
        occupancy_weight: float = 1.0,
        flow_weight: float = 0.5,
        response_weight: float = 0.5,
    ) -> Dict[str, float]:
        """Stage-3 update of only the registered belief/forecast path."""
        eligible = {"belief_encoder"}
        if self.model_config.residual_members:
            eligible.add("rollout")
        frozen = [
            module for name, module in self.online.named_children() if name not in eligible
        ]
        with freeze_modules(frozen):
            loss, occupancy_loss, flow_loss, response_loss, residual_loss, valid_count = (
                self._representation_objective(
                    batch, occupancy_weight=occupancy_weight, flow_weight=flow_weight,
                    response_weight=response_weight,
                )
            )
        if valid_count == 0:
            return {"representation/update_applied": 0.0, "representation/valid_count": 0.0}
        self.value_path_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        parameters = self.parameter_groups()["representation"] + list(
            self.online.rollout.residuals.parameters()
        )
        finite = bool(torch.isfinite(loss).item()) and all(
            parameter.grad is None or torch.isfinite(parameter.grad).all().item()
            for parameter in parameters
        )
        if not finite:
            self.value_path_optimizer.zero_grad(set_to_none=True)
            return {
                "loss/representation": float("nan"),
                "representation/update_applied": 0.0,
                "representation/valid_count": float(valid_count),
            }
        torch.nn.utils.clip_grad_norm_(parameters, self.config.max_gradient_norm)
        self.value_path_optimizer.step()
        self.update_step += 1
        return {
            "loss/representation": float(loss.detach()),
            "loss/occupancy": float(occupancy_loss.detach()),
            "loss/dynamic_flow": float(flow_loss.detach()),
            "loss/vehicle_response": float(response_loss.detach()),
            "loss/residual_response": float(residual_loss.detach()),
            "representation/valid_count": float(valid_count),
            "representation/update_applied": 1.0,
        }

    def _risk_objective(
        self, batch: TractorRiskBatch, clearance_weight: float, stopping_weight: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        belief, context = self.online.encode(batch.current)
        output = self.online.score(belief, context, batch.candidates)
        q_m, q_r = output.hazard.shape[1:3]

        def expand_member_axes(value: torch.Tensor) -> torch.Tensor:
            return value[:, None, None].expand(-1, q_m, q_r, *value.shape[1:])

        model_valid = output.candidate_valid[:, None, None, :]
        event_valid = expand_member_axes(batch.event_label_valid) & model_valid
        clearance_valid = expand_member_axes(batch.clearance_valid) & model_valid.unsqueeze(-1)
        stopping_valid = (
            expand_member_axes(batch.stopping_margin_valid) & model_valid.unsqueeze(-1)
        )
        hazard_loss = competing_risk_nll(
            output.hazard, expand_member_axes(batch.event_observed),
            expand_member_axes(batch.event_step), expand_member_axes(batch.event_cause),
            expand_member_axes(batch.censor_step), event_valid,
        )
        clearance_loss = masked_pinball_loss(
            output.clearance_quantiles, expand_member_axes(batch.clearance_m), clearance_valid,
        )
        stopping_loss = masked_pinball_loss(
            output.stopping_quantiles, expand_member_axes(batch.stopping_margin_m), stopping_valid,
        )
        loss = hazard_loss + clearance_weight * clearance_loss + stopping_weight * stopping_loss
        return (
            loss, hazard_loss, clearance_loss, stopping_loss,
            int(event_valid.sum().item()), int((clearance_valid | stopping_valid).sum().item()),
        )

    def risk_step(
        self,
        batch: TractorRiskBatch,
        *,
        clearance_weight: float = 0.25,
        stopping_weight: float = 0.25,
    ) -> Dict[str, float]:
        """Update only the cause-time/severity head from candidate labels.

        Representation, rollout and interaction modules are intentionally
        frozen here.  This keeps optimizer ownership disjoint and prevents
        privileged labels from changing the policy observation encoder.
        """
        batch.validate(self.model_config)
        if clearance_weight < 0.0 or stopping_weight < 0.0:
            raise ValueError("risk severity loss weights must be non-negative")
        frozen = [module for name, module in self.online.named_children() if name != "risk_head"]
        with freeze_modules(frozen):
            loss, hazard_loss, clearance_loss, stopping_loss, valid_count, severity_count = (
                self._risk_objective(batch, clearance_weight, stopping_weight)
            )
        if valid_count + severity_count == 0:
            return {
                "loss/risk": 0.0, "loss/risk_hazard": 0.0,
                "loss/risk_clearance": 0.0, "loss/risk_stopping": 0.0,
                "risk/valid_count": 0.0, "risk/severity_valid_count": 0.0,
                "risk/update_applied": 0.0,
            }
        self.risk_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        parameters = self.parameter_groups()["risk"]
        finite = bool(torch.isfinite(loss).item()) and all(
            parameter.grad is None or torch.isfinite(parameter.grad).all().item()
            for parameter in parameters
        )
        if not finite:
            self.risk_optimizer.zero_grad(set_to_none=True)
            return {
                "loss/risk": float("nan"), "risk/valid_count": float(valid_count),
                "risk/severity_valid_count": float(severity_count), "risk/update_applied": 0.0,
            }
        torch.nn.utils.clip_grad_norm_(parameters, self.config.max_gradient_norm)
        self.risk_optimizer.step()
        return {
            "loss/risk": float(loss.detach()),
            "loss/risk_hazard": float(hazard_loss.detach()),
            "loss/risk_clearance": float(clearance_loss.detach()),
            "loss/risk_stopping": float(stopping_loss.detach()),
            "risk/valid_count": float(valid_count),
            "risk/severity_valid_count": float(severity_count),
            "risk/update_applied": 1.0,
        }

    def _risk_feature_step(
        self, batch: TractorRiskBatch, clearance_weight: float, stopping_weight: float,
    ) -> Dict[str, float]:
        """Stage-4 feature-side update; risk head and all other modules stay fixed."""
        eligible_names = {"interaction", "temporal_aggregator"}
        frozen = [
            module for name, module in self.online.named_children()
            if name not in eligible_names
        ]
        with freeze_modules(frozen):
            loss, _hazard, _clearance, _stopping, valid_count, severity_count = (
                self._risk_objective(batch, clearance_weight, stopping_weight)
            )
        if valid_count + severity_count == 0:
            return {"risk_feature/update_applied": 0.0}
        self.value_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        parameters = [
            parameter
            for name in eligible_names
            for parameter in getattr(self.online, name).parameters()
        ]
        finite = bool(torch.isfinite(loss).item()) and all(
            parameter.grad is None or torch.isfinite(parameter.grad).all().item()
            for parameter in parameters
        )
        if not finite:
            self.value_optimizer.zero_grad(set_to_none=True)
            return {"loss/risk_feature": float("nan"), "risk_feature/update_applied": 0.0}
        torch.nn.utils.clip_grad_norm_(parameters, self.config.max_gradient_norm)
        self.value_optimizer.step()
        return {
            "loss/risk_feature": float(loss.detach()),
            "risk_feature/update_applied": 1.0,
        }

    def atomic_stage4_risk_step(
        self,
        batch: TractorRiskBatch,
        *,
        clearance_weight: float = 0.25,
        stopping_weight: float = 0.25,
    ) -> Dict[str, float]:
        """Commit Stage-4 feature/head sides together or restore both."""
        batch.validate(self.model_config)
        if clearance_weight < 0.0 or stopping_weight < 0.0:
            raise ValueError("risk severity loss weights must be non-negative")
        if not (
            batch.event_label_valid.any()
            or batch.clearance_valid.any()
            or batch.stopping_margin_valid.any()
        ):
            return {"stage4/update_applied": 0.0}
        rollback = {
            "online": copy.deepcopy(self.online.state_dict()),
            "value_optimizer": copy.deepcopy(self.value_optimizer.state_dict()),
            "risk_optimizer": copy.deepcopy(self.risk_optimizer.state_dict()),
            "update_step": self.update_step,
        }
        try:
            metrics = self._risk_feature_step(batch, clearance_weight, stopping_weight)
            if metrics.get("risk_feature/update_applied") != 1.0:
                raise FloatingPointError("TRACTOR Stage-4 feature update was rejected")
            head = self.risk_step(
                batch, clearance_weight=clearance_weight, stopping_weight=stopping_weight,
            )
            metrics.update(head)
            if head.get("risk/update_applied") != 1.0:
                raise FloatingPointError("TRACTOR Stage-4 head update was rejected")
        except BaseException:
            self.online.load_state_dict(rollback["online"])
            self.value_optimizer.load_state_dict(rollback["value_optimizer"])
            self.risk_optimizer.load_state_dict(rollback["risk_optimizer"])
            self.update_step = int(rollback["update_step"])
            self.value_optimizer.zero_grad(set_to_none=True)
            self.risk_optimizer.zero_grad(set_to_none=True)
            raise
        self.update_step += 1
        metrics["stage4/update_applied"] = 1.0
        return metrics

    def atomic_value_risk_step(
        self,
        value_batch: TractorTrainingBatch,
        risk_batch: TractorRiskBatch,
        representation_batch: TractorRepresentationBatch | None = None,
        *,
        occupancy_weight: float = 1.0,
        flow_weight: float = 0.5,
        response_weight: float = 0.5,
        risk_feature_weight: float = 0.25,
        clearance_weight: float = 0.25,
        stopping_weight: float = 0.25,
    ) -> Dict[str, float]:
        """Commit the Stage-5 value/risk pair as one recoverable transaction.

        Both batches and risk-label eligibility are checked before the target
        policy RNG is consumed.  If either numerical update is rejected or
        raises, online weights, both optimizer states, update ordinal and the
        dedicated target RNG are restored byte-for-byte.  Actor, entropy and
        EMA therefore only run after a complete value+risk commit.
        """
        value_batch.validate(self.model_config)
        risk_batch.validate(self.model_config)
        if representation_batch is not None:
            representation_batch.validate(self.model_config)
        if min(
            occupancy_weight, flow_weight, response_weight, risk_feature_weight,
            clearance_weight, stopping_weight,
        ) < 0.0:
            raise ValueError("Stage-5 loss weights must be non-negative")
        event_count = int(risk_batch.event_label_valid.sum().item())
        severity_count = int(
            (risk_batch.clearance_valid | risk_batch.stopping_margin_valid).sum().item()
        )
        if event_count + severity_count == 0:
            return {
                "update/applied": 0.0, "risk/update_applied": 0.0,
                "update/transaction_applied": 0.0,
                "risk/valid_count": 0.0, "risk/severity_valid_count": 0.0,
            }

        rollback = {
            "online": copy.deepcopy(self.online.state_dict()),
            "value_path_optimizer": copy.deepcopy(self.value_path_optimizer.state_dict()),
            "risk_optimizer": copy.deepcopy(self.risk_optimizer.state_dict()),
            "target_generator_state": self.target_generator.get_state().clone(),
            "update_step": self.update_step,
        }
        try:
            # Value-side graph: complete value path is trainable; the risk
            # head supplies feature gradients but its weights are frozen.
            with freeze_modules([self.online.risk_head]):
                critic_loss = self._critic_objective(value_batch)
                if representation_batch is None:
                    representation_loss = critic_loss.new_zeros(())
                    occupancy_loss = flow_loss = response_loss = representation_loss
                    representation_valid = 0
                else:
                    (
                        representation_loss, occupancy_loss, flow_loss,
                        response_loss, residual_loss, representation_valid,
                    ) = self._representation_objective(
                        representation_batch, occupancy_weight=occupancy_weight,
                        flow_weight=flow_weight, response_weight=response_weight,
                    )
                (
                    risk_feature_loss, _feature_hazard, _feature_clearance,
                    _feature_stopping, feature_valid, feature_severity,
                ) = self._risk_objective(risk_batch, clearance_weight, stopping_weight)
                value_loss = (
                    critic_loss + representation_loss
                    + risk_feature_weight * risk_feature_loss
                )

            # Risk-head graph: every value-path parameter is frozen, so this
            # is the executable stop-gradient side of the same transaction.
            frozen_value = [
                module for name, module in self.online.named_children()
                if name != "risk_head"
            ]
            with freeze_modules(frozen_value):
                (
                    risk_loss, hazard_loss, clearance_loss, stopping_loss,
                    risk_valid, risk_severity,
                ) = self._risk_objective(risk_batch, clearance_weight, stopping_weight)

            self.value_path_optimizer.zero_grad(set_to_none=True)
            self.risk_optimizer.zero_grad(set_to_none=True)
            value_loss.backward()
            risk_loss.backward()
            value_parameters = (
                self.parameter_groups()["representation"] + self.parameter_groups()["value"]
            )
            risk_parameters = self.parameter_groups()["risk"]
            finite = all(torch.isfinite(item).item() for item in (value_loss, risk_loss))
            finite = finite and all(
                parameter.grad is None or torch.isfinite(parameter.grad).all().item()
                for parameter in value_parameters + risk_parameters
            )
            if not finite:
                raise FloatingPointError("TRACTOR Stage-5 value/risk gradients are non-finite")
            torch.nn.utils.clip_grad_norm_(value_parameters, self.config.max_gradient_norm)
            torch.nn.utils.clip_grad_norm_(risk_parameters, self.config.max_gradient_norm)
            self.value_path_optimizer.step()
            self.risk_optimizer.step()
            self.update_step += 1
            metrics = {
                "loss/critic": float(critic_loss.detach()),
                "loss/representation": float(representation_loss.detach()),
                "loss/occupancy": float(occupancy_loss.detach()),
                "loss/dynamic_flow": float(flow_loss.detach()),
                "loss/vehicle_response": float(response_loss.detach()),
                "loss/residual_response": float(
                    residual_loss.detach() if representation_batch is not None
                    else representation_loss.detach()
                ),
                "loss/risk_feature": float(risk_feature_loss.detach()),
                "loss/risk": float(risk_loss.detach()),
                "loss/risk_hazard": float(hazard_loss.detach()),
                "loss/risk_clearance": float(clearance_loss.detach()),
                "loss/risk_stopping": float(stopping_loss.detach()),
                "representation/valid_count": float(representation_valid),
                "risk_feature/valid_count": float(feature_valid + feature_severity),
                "risk/valid_count": float(risk_valid),
                "risk/severity_valid_count": float(risk_severity),
                "update/applied": 1.0, "risk/update_applied": 1.0,
            }
        except BaseException:
            self.online.load_state_dict(rollback["online"])
            self.value_path_optimizer.load_state_dict(rollback["value_path_optimizer"])
            self.risk_optimizer.load_state_dict(rollback["risk_optimizer"])
            self.target_generator.set_state(rollback["target_generator_state"])
            self.update_step = int(rollback["update_step"])
            self.value_path_optimizer.zero_grad(set_to_none=True)
            self.risk_optimizer.zero_grad(set_to_none=True)
            raise
        metrics["update/transaction_applied"] = 1.0
        return metrics

    def actor_step(self, current: TractorInputs, *, update_target: bool = True) -> Dict[str, float]:
        current.validate(self.model_config)
        rollback = {
            "actor": copy.deepcopy(self.online.actor.state_dict()),
            "temperature": copy.deepcopy(self.temperature.state_dict()),
            "actor_optimizer": copy.deepcopy(self.actor_optimizer.state_dict()),
            "entropy_optimizer": copy.deepcopy(self.entropy_optimizer.state_dict()),
            "target": copy.deepcopy(self.target.state_dict()) if update_target else None,
            "torch_cpu_rng": torch.get_rng_state(),
            "torch_cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        try:
            frozen = [module for name, module in self.online.named_children() if name != "actor"]
            with freeze_modules(frozen):
                belief, context = self.online.encode(current)
                action, log_probability = self.online.actor.sample(belief)
                output = self.online.score(belief, context, _one_candidate(action, "actor_live"))
                value = output.return_quantiles[:, 0].mean(dim=(-1, -2))
                event_probability = cumulative_event_probability(output.hazard).mean(dim=(1, 2))[:, 0]
                actor_loss = (
                    self.temperature.alpha.detach() * log_probability[:, 0]
                    - value + self.config.actor_risk_weight * event_probability
                ).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_parameters = list(self.online.actor.parameters())
            actor_finite = bool(torch.isfinite(actor_loss).item()) and all(
                parameter.grad is None or torch.isfinite(parameter.grad).all().item()
                for parameter in actor_parameters
            )
            if not actor_finite:
                raise FloatingPointError("TRACTOR actor loss/gradients are non-finite")
            torch.nn.utils.clip_grad_norm_(actor_parameters, self.config.max_gradient_norm)
            self.actor_optimizer.step()

            entropy_loss = -(
                self.temperature.log_alpha * (log_probability.detach() + self.config.target_entropy)
            ).mean()
            self.entropy_optimizer.zero_grad(set_to_none=True)
            entropy_loss.backward()
            entropy_finite = bool(torch.isfinite(entropy_loss).item()) and all(
                parameter.grad is None or torch.isfinite(parameter.grad).all().item()
                for parameter in self.temperature.parameters()
            )
            if not entropy_finite:
                raise FloatingPointError("TRACTOR entropy loss/gradients are non-finite")
            self.entropy_optimizer.step()
            if any(
                not torch.isfinite(parameter).all().item()
                for parameter in (*actor_parameters, *self.temperature.parameters())
            ):
                raise FloatingPointError("TRACTOR actor/entropy step produced non-finite weights")
            if update_target:
                self.update_target()
        except BaseException:
            self.online.actor.load_state_dict(rollback["actor"])
            self.temperature.load_state_dict(rollback["temperature"])
            self.actor_optimizer.load_state_dict(rollback["actor_optimizer"])
            self.entropy_optimizer.load_state_dict(rollback["entropy_optimizer"])
            if rollback["target"] is not None:
                self.target.load_state_dict(rollback["target"])
            torch.set_rng_state(rollback["torch_cpu_rng"])
            if rollback["torch_cuda_rng"] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rollback["torch_cuda_rng"])
            self.actor_optimizer.zero_grad(set_to_none=True)
            self.entropy_optimizer.zero_grad(set_to_none=True)
            raise
        return {
            "loss/actor": float(actor_loss.detach()),
            "loss/entropy": float(entropy_loss.detach()),
            "entropy/alpha": float(self.temperature.alpha.detach()),
        }

    def update_target(self) -> None:
        ema_update(self.target, self.online, self.config.target_tau)

    def checkpoint_components(self) -> Dict[str, object]:
        return {
            "online": self.online,
            "target": self.target,
            "temperature": self.temperature,
            "value_path_optimizer": self.value_path_optimizer,
            "risk_optimizer": self.risk_optimizer,
            "actor_optimizer": self.actor_optimizer,
            "entropy_optimizer": self.entropy_optimizer,
        }

    def extra_state_dict(self) -> dict:
        return {
            "update_step": self.update_step,
            "target_generator_state": self.target_generator.get_state(),
            "model_fingerprint": self.model_config.fingerprint(),
        }

    def load_extra_state_dict(self, state: dict) -> None:
        if state.get("model_fingerprint") != self.model_config.fingerprint():
            raise RuntimeError("TRACTOR model fingerprint mismatch")
        self.target_generator.set_state(state["target_generator_state"])
        self.update_step = int(state["update_step"])
