"""Vanilla twin-Q SAC (Haarnoja et al. 2018) -- the "Vanilla SAC" algorithm-
choice comparison point (docs/RESEARCH_PROTOCOL.md section 34/35). Shares
this package's [kappa, v_ref, L] trajectory action space, observation
builder, and reward with the TQC-based ablation rows (A-F) unchanged, so
this is a controlled comparison of ALGORITHM choice alone, not a different
task setup.

Reuses ``rl/networks/tqc.py``'s ``Actor``/``Critic`` verbatim rather than
duplicating a second Gaussian-policy/MLP-critic definition:
``Actor`` is already algorithm-agnostic (a squashed-Gaussian policy with no
TQC-specific logic), and ``Critic(n_quantiles=1, n_critics=2)`` collapses
exactly to a standard scalar twin-Q critic pair -- SAC just never sorts/
truncates/quantile-regresses that single "quantile" per net, it uses it
directly as a Q-value and takes ``min`` across critics (the actual SAC
update, not TQC's distributional one). No quantile fields anywhere in this
module or in ``SACHyperparameters`` -- there is nothing here for a quantile
count to mean, unlike TQCHyperparameters.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from hunter_kinodynamic_rl.config.schema import SACHyperparameters
from hunter_kinodynamic_rl.rl.networks.tqc import Actor, Critic


def _polyak_update(src_params, tgt_params, tau: float) -> None:
    with torch.no_grad():
        for src, tgt in zip(src_params, tgt_params):
            tgt.data.mul_(1.0 - tau).add_(src.data, alpha=tau)


class Agent:
    def __init__(self, state_dim: int, action_dim: int, max_action: float,
                 hyperparameters: SACHyperparameters, device: Optional[str] = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = float(max_action)
        self.hp = hyperparameters
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        actor_activ = F.relu if self.hp.actor_activ == "relu" else F.elu
        critic_activ = F.elu if self.hp.critic_activ == "elu" else F.relu

        self.actor = Actor(state_dim, action_dim, hdim=self.hp.actor_hdim, activ=actor_activ).to(self.device)
        # n_quantiles=1 -> Critic's per-net output is a single scalar Q-value,
        # i.e. a plain twin-Q critic pair (see module docstring).
        self.critic = Critic(state_dim, action_dim, hdim=self.hp.critic_hdim, activ=critic_activ,
                              n_quantiles=1, n_critics=self.hp.n_critics).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, hdim=self.hp.critic_hdim, activ=critic_activ,
                                     n_quantiles=1, n_critics=self.hp.n_critics).to(self.device)
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

    def _q_min(self, critic_module, state, action) -> torch.Tensor:
        """``Critic(...).forward`` -> (batch, n_critics, 1); squeeze the
        trailing quantile-of-one dim and take the min across critics --
        the standard SAC pessimistic target/actor-objective operator (as
        opposed to TQC's sorted-and-truncated-mean over all critics x
        quantiles)."""
        q = critic_module(state, action).squeeze(-1)  # (batch, n_critics)
        return q.min(dim=1, keepdim=True).values  # (batch, 1)

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
            next_q_min = self._q_min(self.critic_target, next_state, next_actions)
            target_q = reward + not_done * self.hp.discount * (next_q_min - ent_coef * next_log_prob)

        current_q = self.critic(state, action).squeeze(-1)  # (batch, n_critics)
        critic_loss = F.mse_loss(current_q, target_q.expand_as(current_q))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actions_pi, log_prob = self.actor.action_log_prob(state)
        for p in self.critic.parameters():
            p.requires_grad_(False)
        qf_pi_min = self._q_min(self.critic, state, actions_pi)
        for p in self.critic.parameters():
            p.requires_grad_(True)
        actor_loss = (ent_coef * log_prob - qf_pi_min).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.training_steps % self.hp.target_update_interval == 0:
            _polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.hp.tau)

        return {
            "loss/critic": float(critic_loss.item()),
            "loss/actor": float(actor_loss.item()),
            "ent_coef": float(ent_coef.item()),
            "training_steps": self.training_steps,
        }

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
        return float(self.log_ent_coef.detach().item()) if self.ent_coef_auto else None

    def load_ent_coef_state(self, value: Optional[float]) -> None:
        if self.ent_coef_auto and value is not None:
            with torch.no_grad():
                self.log_ent_coef.copy_(torch.tensor([value], device=self.device))
