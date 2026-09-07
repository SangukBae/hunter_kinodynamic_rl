"""Cause-time competing hazard and ordered severity heads."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .contracts import BeliefState, InteractionOutput, TractorConfig


def _ordered_quantiles(raw: torch.Tensor) -> torch.Tensor:
    first = raw[..., :1]
    if raw.shape[-1] == 1:
        return first
    increments = F.softplus(raw[..., 1:])
    return torch.cat((first, first + torch.cumsum(increments, dim=-1)), dim=-1)


class _RiskBundle(nn.Module):
    def __init__(self, input_dim: int, config: TractorConfig):
        super().__init__()
        self.cfg = config
        self.body = nn.Sequential(nn.Linear(input_dim, 192), nn.ELU(), nn.Linear(192, 128), nn.ELU())
        self.hazard_logits = nn.Linear(128, config.n_causes + 1)
        self.clearance = nn.Linear(128, config.n_severity_quantiles)
        self.stopping = nn.Linear(128, config.n_severity_quantiles)

    def forward(self, x: torch.Tensor):
        hidden = self.body(x)
        event_or_none = F.softmax(self.hazard_logits(hidden), dim=-1)
        return (
            event_or_none[..., : self.cfg.n_causes],
            _ordered_quantiles(self.clearance(hidden)),
            _ordered_quantiles(self.stopping(hidden)),
        )


class CauseTimeRiskHead(nn.Module):
    def __init__(self, config: TractorConfig):
        super().__init__()
        self.cfg = config
        input_dim = config.interaction_dim + config.ego_dim + config.plant_dim
        self.members = nn.ModuleList([_RiskBundle(input_dim, config) for _ in range(config.risk_members)])

    def forward(self, belief: BeliefState, interaction: InteractionOutput):
        b, q, k, h, _ = interaction.per_time.shape
        context = torch.cat((belief.ego_latent, belief.plant_latent), dim=-1)
        context = context[:, None, None, None, :].expand(-1, q, k, h, -1)
        x = torch.cat((interaction.per_time, context), dim=-1)
        hazard, clearance, stopping = [], [], []
        for member in self.members:
            hz, cl, st = member(x)
            hazard.append(hz)
            clearance.append(cl)
            stopping.append(st)
        # B,Qm,Qr,K,H,...
        return (
            torch.stack(hazard, dim=2),
            torch.stack(clearance, dim=2),
            torch.stack(stopping, dim=2),
        )
