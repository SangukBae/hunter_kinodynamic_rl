"""Action-independent dense future scene predictor."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .contracts import TractorConfig


class ConvGRUCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int):
        super().__init__()
        total = input_channels + hidden_channels
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(total, 2 * hidden_channels, 3, padding=1)
        self.candidate = nn.Conv2d(total, hidden_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid(self.gates(torch.cat((x, hidden), dim=1)))
        reset, update = gates.chunk(2, dim=1)
        proposal = torch.tanh(self.candidate(torch.cat((x, reset * hidden), dim=1)))
        return (1.0 - update) * hidden + update * proposal


class SceneForecast(nn.Module):
    def __init__(self, config: TractorConfig):
        super().__init__()
        c = config.scene_channels
        self.cfg = config
        self.cell = ConvGRUCell(c, c)
        self.transition = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1), nn.ELU(), nn.Conv2d(c, c, 3, padding=1), nn.ELU()
        )
        self.occupancy_head = nn.Conv2d(c, 4, 1)
        self.flow_head = nn.Conv2d(c, 2, 1)

    def forward(self, current_feature: torch.Tensor):
        hidden = current_feature
        features, occupancy, flow = [], [], []
        zero_input = torch.zeros_like(current_feature)
        for _ in range(self.cfg.horizon_steps):
            # The recurrent state, not the input slot, carries the predicted
            # scene through time.  Keeping the input action-independent makes
            # the shared future immutable across candidate scoring.
            hidden = self.cell(zero_input, self.transition(hidden))
            features.append(hidden)
            occupancy.append(F.softmax(self.occupancy_head(hidden), dim=1))
            flow.append(self.flow_head(hidden))
        return (
            torch.stack(features, dim=1),
            torch.stack(occupancy, dim=1),
            torch.stack(flow, dim=1),
        )
