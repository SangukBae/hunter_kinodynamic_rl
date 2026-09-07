"""Vanilla TQC agent -- the reference baseline this package's risk-aware
extension (rl/algorithms/kinodynamic_tqc/agent.py) builds on top of.

The NETWORK definitions (``Actor``, ``Critic``, ``quantile_huber_loss``) are
copied VERBATIM from drl_agent (hash-verified, see docs/IMPLEMENTATION_PLAN.md) --
that is the actual "TQC math" and must not drift. This agent/training-loop
file is a CLEAN reimplementation of the vanilla update rule (no
aux_prediction / action_risk_head / temporal-fusion machinery, none of which
belongs in an independent reference baseline for THIS research package), not
a line-for-line copy of drl_agent's agent.py/update.py -- see
tests/test_tqc_parity.py for what is checked and how, and
docs/IMPLEMENTATION_PLAN.md for the exact rationale.

Target-quantile truncation and actor-loss form follow the standard TQC
update (Kuznetsov et al. 2020), matching drl_agent's own implementation:

    n_target_quantiles = n_critics * n_quantiles - top_quantiles_to_drop_per_net * n_critics
    target = reward + not_done * discount * (truncated_next_quantiles - ent_coef * next_log_prob)
    critic_loss = quantile_huber_loss(current_quantiles, target)
    actor_loss  = (ent_coef * log_prob - Q_pi.mean(quantiles).mean(critics)).mean()
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from hunter_kinodynamic_rl.config.schema import TQCHyperparameters
from hunter_kinodynamic_rl.rl.networks.tqc import Actor, Critic, quantile_huber_loss


def _polyak_update(src_params, tgt_params, tau: float) -> None:
    with torch.no_grad():
        for src, tgt in zip(src_params, tgt_params):
            tgt.data.mul_(1.0 - tau).add_(src.data, alpha=tau)


class Agent:
    def __init__(self, state_dim: int, action_dim: int, max_action: float,
                 hyperparameters: TQCHyperparameters, device: Optional[str] = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = float(max_action)
        self.hp = hyperparameters
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        actor_activ = F.relu if self.hp.actor_activ == "relu" else F.elu
        critic_activ = F.elu if self.hp.critic_activ == "elu" else F.relu

        self.actor = Actor(state_dim, action_dim, hdim=self.hp.actor_hdim, activ=actor_activ).to(self.device)
        self.critic = Critic(state_dim, action_dim, hdim=self.hp.critic_hdim, activ=critic_activ,
                              n_quantiles=self.hp.n_quantiles, n_critics=self.hp.n_critics).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, hdim=self.hp.critic_hdim, activ=critic_activ,
                                     n_quantiles=self.hp.n_quantiles, n_critics=self.hp.n_critics).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.hp.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.hp.critic_lr)

        self.ent_coef_auto = isinstance(self.hp.ent_coef, str) and self.hp.ent_coef.startswith("auto")
        self.target_entropy = -float(action_dim)
        if self.ent_coef_auto:
            init_value = float(self.hp.ent_coef.split("_")[1]) if "_" in self.hp.ent_coef else 1.0
            self.log_ent_coef = torch.log(torch.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = torch.optim.Adam([self.log_ent_coef], lr=self.hp.ent_coef_lr)
        else:
            self.ent_coef_tensor = torch.tensor(float(self.hp.ent_coef), device=self.device)
            self.log_ent_coef = None
            self.ent_coef_optimizer = None

        self.training_steps = 0

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device).view(1, -1)
            action = self.actor.forward(s, deterministic=deterministic)
        return (action.cpu().numpy().flatten() * self.max_action)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def _current_ent_coef(self) -> torch.Tensor:
        return torch.exp(self.log_ent_coef.detach()) if self.ent_coef_auto else self.ent_coef_tensor

    def train_step(self, state, action, next_state, reward, not_done) -> Dict[str, float]:
        self.training_steps += 1

        if self.ent_coef_auto:
            with torch.no_grad():
                _, log_prob_ent = self.actor.action_log_prob(state)
            ent_coef_loss = -(self.log_ent_coef * (log_prob_ent + self.target_entropy).detach()).mean()
            self.ent_coef_optimizer.zero_grad()
            ent_coef_loss.backward()
            self.ent_coef_optimizer.step()
        ent_coef = self._current_ent_coef()

        with torch.no_grad():
            next_actions, next_log_prob = self.actor.action_log_prob(next_state)
            next_quantiles = self.critic_target(next_state, next_actions)
            batch_size = state.shape[0]
            next_quantiles_flat = next_quantiles.reshape(batch_size, -1)
            next_quantiles_sorted, _ = torch.sort(next_quantiles_flat, dim=1)
            n_target = self.hp.n_critics * self.hp.n_quantiles - self.hp.top_quantiles_to_drop_per_net * self.hp.n_critics
            next_quantiles_truncated = next_quantiles_sorted[:, :n_target]
            target_quantiles = reward + not_done * self.hp.discount * (
                next_quantiles_truncated - ent_coef * next_log_prob.view(-1, 1)
            )
            target_quantiles = target_quantiles.unsqueeze(1)

        current_quantiles = self.critic(state, action)
        critic_loss = quantile_huber_loss(current_quantiles, target_quantiles, sum_over_quantiles=False)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actions_pi, log_prob = self.actor.action_log_prob(state)
        for p in self.critic.parameters():
            p.requires_grad_(False)
        qf_pi = self.critic(state, actions_pi)
        for p in self.critic.parameters():
            p.requires_grad_(True)
        qf_pi = qf_pi.mean(dim=2).mean(dim=1, keepdim=True)
        actor_loss = (ent_coef * log_prob - qf_pi).mean()
        # Extension point (kinodynamic_tqc.Agent overrides this to add the
        # risk-penalty term) -- combined into ONE backward/step, matching
        # drl_agent's single-combined-actor_loss pattern rather than two
        # sequential optimizer steps on the same parameters.
        extra_loss, extra_metrics = self._actor_extra_loss(state, actions_pi)
        actor_loss = actor_loss + extra_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.training_steps % self.hp.target_update_interval == 0:
            _polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.hp.tau)

        metrics = {
            "loss/critic": float(critic_loss.item()),
            "loss/actor": float(actor_loss.item()),
            "ent_coef": float(ent_coef.item()),
            "training_steps": self.training_steps,
        }
        metrics.update(extra_metrics)
        return metrics

    def _actor_extra_loss(self, state: torch.Tensor, actions_pi: torch.Tensor):
        """Hook for subclasses (kinodynamic_tqc.Agent) to add an extra term
        to the actor loss BEFORE the single combined backward/step. Vanilla
        TQC contributes zero -- returning it as a tensor (not a Python
        float) keeps this a true no-op graph node, not a separate branch."""
        return actions_pi.new_zeros(()), {}

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def checkpoint_components(self) -> Dict[str, object]:
        components = {
            "actor": self.actor, "critic": self.critic, "critic_target": self.critic_target,
            "actor_optimizer": self.actor_optimizer, "critic_optimizer": self.critic_optimizer,
        }
        if self.ent_coef_auto:
            components["ent_coef_optimizer"] = self.ent_coef_optimizer
        return components

    def ent_coef_state(self) -> Optional[float]:
        """``log_ent_coef`` is a bare tensor (not an nn.Module), so it isn't
        covered by checkpoint_components()'s state_dict() calls -- persist
        its scalar value in the checkpoint's JSON meta instead and restore
        it via :meth:`load_ent_coef_state`."""
        return float(self.log_ent_coef.detach().item()) if self.ent_coef_auto else None

    def load_ent_coef_state(self, value: Optional[float]) -> None:
        if self.ent_coef_auto and value is not None:
            with torch.no_grad():
                self.log_ent_coef.copy_(torch.tensor([value], device=self.device))
