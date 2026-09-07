"""Split the legacy 328-D observation without changing field order."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .contracts import TractorConfig


class LegacyObservationAdapter(nn.Module):
    def __init__(self, config: TractorConfig):
        super().__init__()
        self.config = config

    def forward(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if observation.ndim != 2 or observation.shape[-1] != self.config.observation_dim:
            raise ValueError(
                f"expected (B,{self.config.observation_dim}) observation, got {tuple(observation.shape)}"
            )
        split = self.config.t_obs * self.config.n_scan
        scans = observation[:, :split].reshape(-1, self.config.t_obs, self.config.n_scan)
        tail = observation[:, split:]
        return scans, tail


def goal_vector_from_tail(tail: torch.Tensor) -> torch.Tensor:
    distance = tail[:, 0]
    heading = tail[:, 1]
    return torch.stack((distance * torch.cos(heading), distance * torch.sin(heading)), dim=-1)
