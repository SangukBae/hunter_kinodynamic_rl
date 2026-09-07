"""Twin candidate-conditioned return-quantile critics."""

from __future__ import annotations

import torch
import torch.nn as nn

from .contracts import BeliefState, InteractionOutput, TractorConfig


class _QuantileHead(nn.Module):
    def __init__(self, input_dim: int, n_quantiles: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ELU(), nn.Linear(256, 256), nn.ELU(),
            nn.Linear(256, n_quantiles),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TractorReturnCritic(nn.Module):
    def __init__(self, config: TractorConfig):
        super().__init__()
        self.cfg = config
        input_dim = config.interaction_dim + config.ego_dim + config.plant_dim
        self.heads = nn.ModuleList([
            _QuantileHead(input_dim, config.n_quantiles) for _ in range(config.n_critics)
        ])

    def forward(self, belief: BeliefState, interaction: InteractionOutput) -> torch.Tensor:
        b, q, k, _ = interaction.aggregated.shape
        context = torch.cat((belief.ego_latent, belief.plant_latent), dim=-1)
        context = context[:, None, None, :].expand(-1, q, k, -1)
        features = torch.cat((interaction.aggregated, context), dim=-1)
        # Fixed uniform vehicle-member aggregation: no silent member drops.
        values = torch.stack([head(features) for head in self.heads], dim=-2)
        return values.mean(dim=1)
