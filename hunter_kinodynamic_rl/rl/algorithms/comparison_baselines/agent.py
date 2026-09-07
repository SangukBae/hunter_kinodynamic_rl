"""Training agent for the executable B1--B8 paper comparison baselines."""

from __future__ import annotations

import copy
import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from hunter_kinodynamic_rl.config.comparison import (
    BASELINE_ARCHITECTURE_REVISION, ComparisonModelConfig,
)
from hunter_kinodynamic_rl.rl.algorithms.tractor_tqc.agent import (
    EntropyTemperature, TractorAgentConfig, TractorRiskBatch, TractorTrainingBatch,
)
from hunter_kinodynamic_rl.rl.algorithms.tractor_tqc.losses import (
    quantile_huber_loss, truncated_bellman_target,
)
from hunter_kinodynamic_rl.rl.algorithms.tractor_tqc.optimizer_groups import (
    assert_disjoint_complete, freeze_modules,
)
from hunter_kinodynamic_rl.rl.algorithms.tractor_tqc.target_update import ema_update
from hunter_kinodynamic_rl.rl.networks.tractor import TractorConfig, TractorInputs

from .model import ComparisonModel


class ComparisonAgent:
    """One implementation whose frozen method contract selects B1--B8 axes."""

    model_family = "TRACTOR comparison baseline"
    architecture_revision = BASELINE_ARCHITECTURE_REVISION

    def __init__(
        self,
        model_config: ComparisonModelConfig,
        input_config: TractorConfig,
        agent_config: TractorAgentConfig = TractorAgentConfig(),
        *,
        device: str | torch.device = "cpu",
        target_seed: int = 0,
    ):
        model_config.validate()
        input_config.validate()
        agent_config.validate()
        if agent_config.top_quantiles_to_drop_per_net >= model_config.n_quantiles:
            raise ValueError("top_quantiles_to_drop_per_net must be smaller than n_quantiles")
        self.model_config = model_config
        self.input_config = input_config
        self.config = agent_config
        self.method_id = model_config.method_id
        self.device = torch.device(device)
        self.online = ComparisonModel(model_config, input_config).to(self.device)
        self.target = copy.deepcopy(self.online).to(self.device).eval()
        self.target.requires_grad_(False)
        self.temperature = EntropyTemperature(agent_config.initial_alpha).to(self.device)
        self.target_generator = torch.Generator(device=self.device.type)
        self.target_generator.manual_seed(int(target_seed))
        self.update_step = 0

        groups = self.parameter_groups()
        assert_disjoint_complete(self.online, groups)
        self.representation_optimizer = torch.optim.Adam(
            groups["representation"], lr=agent_config.representation_lr,
        )
        self.value_optimizer = torch.optim.Adam(groups["value"], lr=agent_config.value_lr)
        self.actor_optimizer = torch.optim.Adam(groups["actor"], lr=agent_config.actor_lr)
        self.risk_optimizer = None
        if groups["risk"]:
            self.risk_optimizer = torch.optim.Adam(groups["risk"], lr=agent_config.risk_lr)
        self.entropy_optimizer = torch.optim.Adam(
            self.temperature.parameters(), lr=agent_config.entropy_lr,
        )

    def parameter_groups(self) -> Dict[str, list[nn.Parameter]]:
        representation = list(self.online.encoder.parameters())
        value_modules = []
        for module in (
            self.online.critics, self.online.cross_attention, self.online.latent_transition,
        ):
            if module is not None:
                value_modules.append(module)
        return {
            "representation": representation,
            "value": [parameter for module in value_modules for parameter in module.parameters()],
            "risk": [] if self.online.risk_head is None else list(self.online.risk_head.parameters()),
            "actor": list(self.online.actor.parameters()),
        }

    def critic_step(self, batch: TractorTrainingBatch, *, update_target: bool = True) -> Dict[str, float]:
        batch.validate(self.input_config)
        state = self.online.encode(batch.current)
        current = self.online.quantiles_from_state(state, batch.action_normalized_requested)
        total_drop = self.config.top_quantiles_to_drop_per_net * self.model_config.n_critics
        target_count = self.model_config.n_critics * self.model_config.n_quantiles - total_drop
        target = batch.reward.expand(-1, target_count).clone()
        bootstrap = batch.bellman_sample_valid.reshape(-1).bool() & ~batch.terminated.reshape(-1).bool()
        with torch.no_grad():
            if bootstrap.any():
                indices = torch.nonzero(bootstrap, as_tuple=False).flatten()
                next_state = self.target.encode(batch.next.index_select(indices))
                next_action, next_log_prob = self.online.sample_action(
                    next_state, generator=self.target_generator,
                )
                next_values = self.target.quantiles_from_state(next_state, next_action)
                selected_target = truncated_bellman_target(
                    batch.reward.index_select(0, indices),
                    batch.discount_factor.index_select(0, indices),
                    batch.terminated.index_select(0, indices),
                    next_values, next_log_prob, self.temperature.alpha, total_drop,
                )
                target.index_copy_(0, indices, selected_target)
        critic_loss = quantile_huber_loss(current, target, batch.bellman_sample_valid)
        dynamics_loss = current.sum() * 0.0
        if self.online.latent_transition is not None:
            next_state_online = self.online.encode(batch.next)
            prediction = self.online.dynamics_prediction(state, batch.action_normalized_requested)
            mask = batch.bellman_sample_valid.reshape(-1).bool()
            if mask.any():
                dynamics_loss = F.smooth_l1_loss(
                    prediction[mask], next_state_online.vector.detach()[mask],
                )
        loss = critic_loss + self.model_config.dynamics_loss_weight * dynamics_loss
        self.representation_optimizer.zero_grad(set_to_none=True)
        self.value_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        parameters = self.parameter_groups()["representation"] + self.parameter_groups()["value"]
        finite = bool(torch.isfinite(loss).item()) and all(
            parameter.grad is None or torch.isfinite(parameter.grad).all().item()
            for parameter in parameters
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
        return {
            "loss/critic": float(critic_loss.detach()),
            "loss/dynamics": float(dynamics_loss.detach()),
            "update/applied": 1.0,
        }

    def risk_step(self, batch: TractorRiskBatch) -> Dict[str, float]:
        if self.online.risk_head is None or self.risk_optimizer is None:
            return {"loss/risk": 0.0, "risk/update_applied": 0.0, "risk/valid_count": 0.0}
        batch.validate(self.input_config)
        valid = batch.event_label_valid.bool() & batch.candidates.present.bool()
        if not valid.any():
            return {"loss/risk": 0.0, "risk/update_applied": 0.0, "risk/valid_count": 0.0}
        frozen = [module for name, module in self.online.named_children() if name != "risk_head"]
        with freeze_modules(frozen):
            state = self.online.encode(batch.current)
            probability = self.online.risk_probability_from_state(
                state, batch.candidates.normalized_actions,
            )
            loss = F.binary_cross_entropy(probability[valid], batch.event_observed.float()[valid])
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
                "loss/risk": float("nan"), "risk/update_applied": 0.0,
                "risk/valid_count": float(valid.sum()),
            }
        torch.nn.utils.clip_grad_norm_(parameters, self.config.max_gradient_norm)
        self.risk_optimizer.step()
        return {
            "loss/risk": float(loss.detach()), "risk/update_applied": 1.0,
            "risk/valid_count": float(valid.sum()),
        }

    def actor_step(self, current: TractorInputs, *, update_target: bool = True) -> Dict[str, float]:
        current.validate(self.input_config)
        frozen = [module for name, module in self.online.named_children() if name != "actor"]
        with freeze_modules(frozen):
            state = self.online.encode(current)
            action, log_probability = self.online.sample_action(state)
            quantiles = self.online.quantiles_from_state(state, action)
            flattened = quantiles.flatten(start_dim=1).sort(dim=-1).values
            count = max(1, int(math.ceil(
                flattened.shape[-1] * self.model_config.cvar_fraction,
            )))
            value = flattened[:, :count].mean(dim=-1)
            risk = torch.zeros_like(value)
            if self.online.risk_head is not None:
                risk = self.online.risk_probability_from_state(state, action)[:, 0]
            actor_loss = (
                self.temperature.alpha.detach() * log_probability[:, 0]
                - value + self.model_config.actor_risk_weight * risk
            ).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.online.actor.parameters(), self.config.max_gradient_norm,
        )
        self.actor_optimizer.step()
        entropy_loss = -(
            self.temperature.log_alpha
            * (log_probability.detach() + self.config.target_entropy)
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
            "actor/cvar_fraction": self.model_config.cvar_fraction,
            "actor/risk_mean": float(risk.detach().mean()),
        }

    def update_target(self) -> None:
        ema_update(self.target, self.online, self.config.target_tau)

    def checkpoint_components(self) -> Dict[str, object]:
        components = {
            "online": self.online,
            "target": self.target,
            "temperature": self.temperature,
            "representation_optimizer": self.representation_optimizer,
            "value_optimizer": self.value_optimizer,
            "actor_optimizer": self.actor_optimizer,
            "entropy_optimizer": self.entropy_optimizer,
        }
        if self.risk_optimizer is not None:
            components["risk_optimizer"] = self.risk_optimizer
        return components

    def extra_state_dict(self) -> dict:
        return {
            "update_step": self.update_step,
            "target_generator_state": self.target_generator.get_state(),
            "model_fingerprint": self.model_config.fingerprint(),
            "method_id": self.method_id,
        }

    def load_extra_state_dict(self, state: dict) -> None:
        if state.get("model_fingerprint") != self.model_config.fingerprint():
            raise RuntimeError("comparison model fingerprint mismatch")
        if state.get("method_id") != self.method_id:
            raise RuntimeError("comparison method identity mismatch")
        self.target_generator.set_state(state["target_generator_state"])
        self.update_step = int(state["update_step"])
