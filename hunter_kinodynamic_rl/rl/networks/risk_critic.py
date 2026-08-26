"""Risk critic: a small supervised head predicting a candidate action's
normalized future risk (section 19/20), separate from the TQC reward critic.

Interface deliberately narrow (state/latent, action -> scalar risk in [0,1])
so a distributional/quantile/CVaR risk critic can be swapped in later behind
the same ``forward(z, action) -> (batch, 1)`` contract (section 19: "Vanilla
TQC의 reward critic과 별도로 safety/risk critic을 추가... interface는 향후
expected risk / quantile risk / CVaR / distributional safety critic으로
확장 가능하게 한다").
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RiskCritic(nn.Module):
    """MLP regressor: [state, action] -> risk in [0, 1] (sigmoid output).

    Trained by its OWN supervised MSE loss against the privileged risk label
    stored in the replay buffer (risk/trajectory_risk.py's ``risk_score``) --
    NEVER by the TQC critic/actor loss (mirrors drl_agent's Action-Risk Head
    gradient rule: the actor sees this network's OUTPUT via a frozen-weights
    forward pass so d(risk)/d(action) still reaches the actor, but the risk
    critic's own weights are updated only by its supervised loss).
    """

    def __init__(self, state_dim: int, action_dim: int, hdim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hdim),
            nn.ReLU(),
            nn.Linear(hdim, hdim),
            nn.ReLU(),
            nn.Linear(hdim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=1)
        return torch.sigmoid(self.net(x))

    def supervised_loss(self, state: torch.Tensor, action: torch.Tensor, target_risk: torch.Tensor) -> torch.Tensor:
        pred = self.forward(state, action)
        return F.mse_loss(pred, target_risk.view(-1, 1))
