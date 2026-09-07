"""Training-ready TRACTOR-TQC agent; importing it never starts training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn

from hunter_kinodynamic_rl.rl.networks.tractor import TractorConfig, TractorInputs, TractorTQC
from hunter_kinodynamic_rl.rl.networks.tractor.contracts import CandidateSet
from hunter_kinodynamic_rl.rl.networks.tractor.model import TractorValueTarget
from hunter_kinodynamic_rl.rl.networks.tractor.selector import cumulative_event_probability

from .losses import (
    competing_risk_nll, masked_pinball_loss, quantile_huber_loss,
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
        self.representation_optimizer = torch.optim.Adam(groups["representation"], lr=agent_config.representation_lr)
        self.value_optimizer = torch.optim.Adam(groups["value"], lr=agent_config.value_lr)
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

    def critic_step(self, batch: TractorTrainingBatch, *, update_target: bool = True) -> Dict[str, float]:
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
        loss = quantile_huber_loss(current, target, batch.bellman_sample_valid)
        self.representation_optimizer.zero_grad(set_to_none=True)
        self.value_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        parameters = self.parameter_groups()["representation"] + self.parameter_groups()["value"]
        finite = bool(torch.isfinite(loss).item()) and all(
            parameter.grad is None or torch.isfinite(parameter.grad).all().item() for parameter in parameters
        )
        if not finite:
            self.representation_optimizer.zero_grad(set_to_none=True)
            self.value_optimizer.zero_grad(set_to_none=True)
            return {"loss/critic": float("nan"), "update/applied": 0.0}
        torch.nn.utils.clip_grad_norm_(parameters, self.config.max_gradient_norm)
        self.representation_optimizer.step()
        self.value_optimizer.step()
        if update_target:
            self.update_target()
        self.update_step += 1
        return {"loss/critic": float(loss.detach()), "update/applied": 1.0}

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
            belief, context = self.online.encode(batch.current)
            output = self.online.score(belief, context, batch.candidates)
            q_m, q_r = output.hazard.shape[1:3]

            def expand_member_axes(value: torch.Tensor) -> torch.Tensor:
                return value[:, None, None].expand(-1, q_m, q_r, *value.shape[1:])

            model_valid = output.candidate_valid[:, None, None, :]
            event_valid = expand_member_axes(batch.event_label_valid) & model_valid
            clearance_valid = (
                expand_member_axes(batch.clearance_valid)
                & model_valid.unsqueeze(-1)
            )
            stopping_valid = (
                expand_member_axes(batch.stopping_margin_valid)
                & model_valid.unsqueeze(-1)
            )
            hazard_loss = competing_risk_nll(
                output.hazard,
                expand_member_axes(batch.event_observed),
                expand_member_axes(batch.event_step),
                expand_member_axes(batch.event_cause),
                expand_member_axes(batch.censor_step),
                event_valid,
            )
            clearance_loss = masked_pinball_loss(
                output.clearance_quantiles,
                expand_member_axes(batch.clearance_m),
                clearance_valid,
            )
            stopping_loss = masked_pinball_loss(
                output.stopping_quantiles,
                expand_member_axes(batch.stopping_margin_m),
                stopping_valid,
            )
            loss = hazard_loss + clearance_weight * clearance_loss + stopping_weight * stopping_loss

        valid_count = int(event_valid.sum().item())
        severity_count = int((clearance_valid | stopping_valid).sum().item())
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

    def actor_step(self, current: TractorInputs, *, update_target: bool = True) -> Dict[str, float]:
        current.validate(self.model_config)
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
        torch.nn.utils.clip_grad_norm_(self.online.actor.parameters(), self.config.max_gradient_norm)
        self.actor_optimizer.step()

        entropy_loss = -(
            self.temperature.log_alpha * (log_probability.detach() + self.config.target_entropy)
        ).mean()
        self.entropy_optimizer.zero_grad(set_to_none=True)
        entropy_loss.backward()
        self.entropy_optimizer.step()
        if update_target:
            self.update_target()
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
            "representation_optimizer": self.representation_optimizer,
            "value_optimizer": self.value_optimizer,
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
