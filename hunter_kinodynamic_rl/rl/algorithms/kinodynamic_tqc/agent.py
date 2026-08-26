"""Risk-aware kinodynamic TQC (Contributions 2+3): extends the vanilla TQC
reference (rl/algorithms/tqc/agent.py) with a risk critic trained on
dynamics-grounded future-risk labels (risk/trajectory_risk.py) plus
counterfactual candidate supervision, and an actor-side risk penalty.

Gradient rule for the risk penalty (mirrors drl_agent's Action-Risk Head --
freeze-not-detach): the risk critic's OWN parameters are frozen during the
actor's forward pass through it, so ``d(risk)/d(action)`` still reaches the
actor, but no gradient updates the risk critic's weights from the actor
loss -- only its own supervised loss does that.

Counterfactual objective (section 6), two parts:
  1. Candidate-augmented supervision: the risk critic is ALSO trained to
     predict each valid counterfactual candidate's risk_score from its own
     (state, candidate_action) pair -- not just the actor's stored action --
     so it learns risk as a function of ACTION, not just of the state it
     happened to observe alongside one action.
  2. Margin-reweighted actor penalty: ``safer_alternative_margin`` (ground
     truth, precomputed by the environment's candidate search -- NOT
     differentiable w.r.t. the current actor) is used to UP-WEIGHT the
     existing risk-critic-mediated actor penalty on transitions where a
     safer alternative existed, rather than being backpropagated through
     directly (which isn't possible since it's a stored scalar, not a
     function of the actor's live output).

When ``risk.enabled=False`` this class is BYTE-IDENTICAL in its update math
to the vanilla TQC agent -- see tests/test_kinodynamic_tqc.py's
``test_risk_disabled_is_bytewise_identical_to_vanilla``.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from hunter_kinodynamic_rl.config.schema import CounterfactualConfig, RiskConfig, TQCHyperparameters
from hunter_kinodynamic_rl.rl.algorithms.tqc.agent import Agent as VanillaAgent
from hunter_kinodynamic_rl.rl.networks.risk_critic import RiskCritic


class Agent(VanillaAgent):
    def __init__(self, state_dim: int, action_dim: int, max_action: float,
                 hyperparameters: TQCHyperparameters, risk_config: RiskConfig,
                 counterfactual_config: Optional[CounterfactualConfig] = None,
                 device: Optional[str] = None):
        super().__init__(state_dim, action_dim, max_action, hyperparameters, device=device)
        self.risk_cfg = risk_config
        self.cf_cfg = counterfactual_config or CounterfactualConfig(enabled=False)
        self.risk_critic: Optional[RiskCritic] = None
        self.risk_critic_optimizer = None
        self.risk_supervised_updates = 0
        if self.risk_cfg.enabled:
            self.risk_critic = RiskCritic(state_dim, action_dim).to(self.device)
            self.risk_critic_optimizer = torch.optim.Adam(self.risk_critic.parameters(), lr=self.hp.critic_lr)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        state, action = batch["state"], batch["action"]
        next_state, reward, not_done = batch["next_state"], batch["reward"], batch["not_done"]

        # Stash the batch's counterfactual/margin tensors where
        # _actor_extra_loss (called INSIDE super().train_step()'s single
        # combined actor backward) can read them without changing the base
        # class's hook signature.
        self._pending_margin = batch.get("safer_alternative_margin")
        metrics = super().train_step(state, action, next_state, reward, not_done)
        self._pending_margin = None

        if self.risk_critic is None:
            return metrics

        valid = batch["valid"].view(-1) >= 0.5
        n_valid = int(valid.sum().item())
        risk_loss_value = 0.0
        candidate_loss_value = 0.0
        if n_valid >= self.risk_cfg.min_valid_labels_per_batch:
            risk_loss = self.risk_critic.supervised_loss(
                state[valid], action[valid], batch["risk_target"][valid],
            )
            total_loss = risk_loss

            if self.cf_cfg.enabled and self.cf_cfg.candidate_supervision_weight > 0.0:
                cand_loss = self._candidate_supervision_loss(state, batch)
                if cand_loss is not None:
                    total_loss = total_loss + self.cf_cfg.candidate_supervision_weight * cand_loss
                    candidate_loss_value = float(cand_loss.item())

            self.risk_critic_optimizer.zero_grad()
            total_loss.backward()
            self.risk_critic_optimizer.step()
            self.risk_supervised_updates += 1
            risk_loss_value = float(risk_loss.item())

        metrics["loss/risk_critic"] = risk_loss_value
        metrics["loss/risk_candidate_supervision"] = candidate_loss_value
        metrics["risk/valid_labels_in_batch"] = n_valid
        metrics["risk/supervised_updates_total"] = self.risk_supervised_updates
        return metrics

    def _candidate_supervision_loss(self, state: torch.Tensor, batch: Dict[str, torch.Tensor]):
        """MSE over every VALID (state, candidate_normalized_action) ->
        candidate_risk triple, flattened across the batch. Candidates are
        already stored as NORMALIZED actions (trainer_base.py converts them
        at collection time via trajectory.action_space.trajectory_command_to_normalized) --
        this module never needs the action-space config itself.

        ``candidate_risk``/``candidate_valid_mask`` are (B, K) -- the same
        shape ReplayBuffer.sample() returns them in; ``candidate_actions_normalized``
        is (B, K, action_dim), assembled by the trainer (not stored per-scalar
        in the buffer, which only holds kappa/v_ref/horizon)."""
        cand_actions = batch.get("candidate_actions_normalized")
        cand_risk = batch.get("candidate_risk")
        cand_valid = batch.get("candidate_valid_mask")
        if cand_actions is None or cand_risk is None or cand_valid is None:
            return None
        mask = cand_valid > 0.5  # (B, K)
        if mask.sum() < 1:
            return None
        b, k, _a_dim = cand_actions.shape
        state_expanded = state.unsqueeze(1).expand(b, k, state.shape[-1])[mask]
        actions_flat = cand_actions[mask]
        targets_flat = cand_risk[mask].view(-1, 1)
        pred = self.risk_critic(state_expanded, actions_flat)
        return F.mse_loss(pred, targets_flat)

    def _actor_extra_loss(self, state: torch.Tensor, actions_pi: torch.Tensor):
        if self.risk_critic is None or self.risk_cfg.actor_lambda <= 0.0:
            return actions_pi.new_zeros(()), {}
        if self.risk_supervised_updates < self.risk_cfg.actor_penalty_warmup_updates:
            return actions_pi.new_zeros(()), {"risk/actor_penalty_active": 0.0}

        for p in self.risk_critic.parameters():
            p.requires_grad_(False)
        predicted_risk = self.risk_critic(state, actions_pi).view(-1)
        for p in self.risk_critic.parameters():
            p.requires_grad_(True)

        weight = torch.ones_like(predicted_risk)
        if self.cf_cfg.enabled and self._pending_margin is not None:
            margin = self._pending_margin.view(-1)
            finite = torch.isfinite(margin)
            margin = torch.where(finite, margin, torch.zeros_like(margin))
            weight = 1.0 + self.cf_cfg.counterfactual_weight_scale * F.relu(margin)

        risk_penalty = self.risk_cfg.actor_lambda * (predicted_risk * weight).mean()
        return risk_penalty, {
            "loss/risk_actor_penalty": float(risk_penalty.item()),
            "risk/actor_penalty_active": 1.0,
        }

    def predict_risk(self, state, action) -> Optional[float]:
        """Inference-only scalar used by structured experiment logging.

        Ground-truth privileged risk remains in telemetry; this method
        records the learned critic's prediction for the exact normalized
        action selected at the same state, without creating gradients or
        changing model mode/state.
        """
        if self.risk_critic is None:
            return None
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).view(1, -1)
        action_t = torch.as_tensor(action, dtype=torch.float32, device=self.device).view(1, -1)
        with torch.no_grad():
            return float(self.risk_critic(state_t, action_t).view(-1)[0].item())

    def checkpoint_components(self) -> Dict[str, object]:
        components = super().checkpoint_components()
        if self.risk_critic is not None:
            components["risk_critic"] = self.risk_critic
            components["risk_critic_optimizer"] = self.risk_critic_optimizer
        return components
