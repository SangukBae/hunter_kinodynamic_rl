"""Bounded, vehicle-response-only residual dynamics for A8/A9."""

from __future__ import annotations

import torch
import torch.nn as nn

from .contracts import TractorConfig


class VehicleResidualStep(nn.Module):
    def __init__(self, config: TractorConfig, delta_v_bound: float = 0.25,
                 delta_steer_bound_rad: float = 0.08):
        super().__init__()
        self.cfg = config
        self.delta_v_bound = float(delta_v_bound)
        self.delta_steer_bound_rad = float(delta_steer_bound_rad)
        self.network = nn.Sequential(
            nn.Linear(config.plant_dim + 3 + 3 + 3 + 3 + 2, 128), nn.ELU(),
            nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 4),
        )

    def forward(
        self,
        plant_latent: torch.Tensor,
        physical_action: torch.Tensor,
        current_response: torch.Tensor,
        response_valid: torch.Tensor,
        nominal_next_response: torch.Tensor,
        dt_sec: torch.Tensor,
        elapsed_fraction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat((
            plant_latent, physical_action, current_response,
            response_valid.to(plant_latent.dtype), nominal_next_response,
            dt_sec, elapsed_fraction,
        ), dim=-1)
        raw = self.network(x)
        delta = torch.stack((
            self.delta_v_bound * torch.tanh(raw[..., 0]),
            self.delta_steer_bound_rad * torch.tanh(raw[..., 1]),
        ), dim=-1)
        q = 1e-6 + (0.05 - 1e-6) * torch.sigmoid(raw[..., 2:4])
        return delta, q


class VehicleResidualEnsemble(nn.Module):
    def __init__(self, config: TractorConfig):
        super().__init__()
        self.members = nn.ModuleList([
            VehicleResidualStep(config) for _ in range(config.residual_members)
        ])

    def __len__(self) -> int:
        return len(self.members)
