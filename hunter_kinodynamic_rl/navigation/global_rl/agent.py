"""Masked Double DQN agent: epsilon-greedy masked action selection + the
SMDP training step (plan section 8.6/8.8)::

    target = option_reward + gamma^local_steps * (1 - mission_done) * max_valid Q_target(next_state)

Double-DQN action selection (the ONLINE network picks ``next_action``, the
TARGET network evaluates it) -- both restricted to valid candidates via
``MaskedDuelingDQN.forward``'s own ``-inf`` masking, so an invalid candidate
can never be selected as ``next_action`` even transiently. Requires torch --
see ``networks.py``'s own docstring for the import-gating convention.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from hunter_kinodynamic_rl.config.schema import GlobalRLConfig
from hunter_kinodynamic_rl.navigation.global_rl.networks import MaskedDuelingDQN
from hunter_kinodynamic_rl.navigation.global_rl.observation import GlobalObservation


def smdp_target(
    option_reward: torch.Tensor, gamma: float, local_steps: torch.Tensor, mission_done: torch.Tensor,
    next_q_target: torch.Tensor,
) -> torch.Tensor:
    """``option_reward + gamma^local_steps * (1 - mission_done) * next_q_target``
    (plan section 8.8) -- factored out of :meth:`GlobalDQNAgent.train_step`
    purely so the SMDP-specific arithmetic (variable-length discounting,
    bootstrapping gated on MISSION done rather than subgoal success/done)
    is directly unit-testable without a network forward pass. All tensors
    broadcast against ``option_reward``'s shape (typically ``(B, 1)``)."""
    gamma_pow = torch.pow(torch.as_tensor(gamma, dtype=option_reward.dtype, device=option_reward.device), local_steps)
    return option_reward + gamma_pow * (1.0 - mission_done) * next_q_target


class GlobalDQNAgent:
    def __init__(self, config: GlobalRLConfig, map_channels: int, device: Optional[str] = None, max_nodes: int = 0):
        self.config = config
        self.max_nodes = int(max_nodes)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.online = MaskedDuelingDQN(config, map_channels, max_nodes).to(self.device)
        self.target = MaskedDuelingDQN(config, map_channels, max_nodes).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
        self.training_steps = 0

    def _observation_to_tensors(self, obs: GlobalObservation):
        map_t = torch.as_tensor(obs.map_tensor, dtype=torch.float32, device=self.device).unsqueeze(0)
        scalar_t = torch.as_tensor(obs.scalar_tensor, dtype=torch.float32, device=self.device).unsqueeze(0)
        cand_t = torch.as_tensor(obs.candidate_tensor, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.as_tensor(obs.action_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        node_t = torch.as_tensor(obs.node_tensor, dtype=torch.float32, device=self.device).unsqueeze(0)
        node_mask_t = torch.as_tensor(obs.node_validity_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        return map_t, scalar_t, cand_t, mask_t, node_t, node_mask_t

    def q_values(self, obs: GlobalObservation) -> np.ndarray:
        with torch.no_grad():
            q = self.online(*self._observation_to_tensors(obs))
        return q.squeeze(0).cpu().numpy()

    def select_action(self, obs: GlobalObservation, *, epsilon: float, rng: np.random.RandomState) -> int:
        """Epsilon-greedy over VALID candidates only -- both branches
        (random and greedy) respect ``obs.action_mask``; the fallback
        candidate guarantees at least one valid index always exists."""
        valid_indices = np.flatnonzero(obs.action_mask)
        if valid_indices.size == 0:
            raise RuntimeError(
                "GlobalDQNAgent.select_action(): action_mask has no valid candidate -- the fallback "
                "candidate must always be valid (see action_mask.compute_action_mask); this indicates a bug "
                "upstream, not a state this agent should silently paper over"
            )
        if rng.random() < epsilon:
            return int(rng.choice(valid_indices))
        q = self.q_values(obs)
        return int(np.argmax(q))

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        map_state, scalar_state = batch["map_state"], batch["scalar_state"]
        cand_feat, mask = batch["candidate_features"], batch["action_mask"]
        next_map, next_scalar = batch["next_map_state"], batch["next_scalar_state"]
        next_cand, next_mask = batch["next_candidate_features"], batch["next_action_mask"]
        action = batch["action"]
        reward = batch["option_reward"]
        mission_done = batch["mission_done"]
        local_steps = batch["local_steps"]
        node_t = batch.get("node_tensor")
        node_mask_t = batch.get("node_validity_mask")
        next_node_t = batch.get("next_node_tensor")
        next_node_mask_t = batch.get("next_node_validity_mask")

        q_all = self.online(map_state, scalar_state, cand_feat, mask, node_t, node_mask_t)
        q_sa = q_all.gather(1, action)

        with torch.no_grad():
            next_q_online = self.online(next_map, next_scalar, next_cand, next_mask, next_node_t, next_node_mask_t)
            next_action = torch.argmax(next_q_online, dim=-1, keepdim=True)
            next_q_target_all = self.target(next_map, next_scalar, next_cand, next_mask, next_node_t, next_node_mask_t)
            next_q_target = next_q_target_all.gather(1, next_action)
            target = smdp_target(reward, self.config.gamma, local_steps, mission_done, next_q_target)

        loss = F.smooth_l1_loss(q_sa, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.training_steps += 1
        if self.training_steps % self.config.target_update_interval_steps == 0:
            self.target.load_state_dict(self.online.state_dict())

        return {
            "loss": float(loss.item()), "mean_q": float(q_sa.mean().item()),
            "mean_target": float(target.mean().item()),
        }

    def checkpoint_components(self) -> Dict[str, object]:
        return {"online": self.online, "target": self.target, "optimizer": self.optimizer}
