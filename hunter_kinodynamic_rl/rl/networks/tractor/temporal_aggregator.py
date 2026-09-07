"""Shared temporal aggregation applied independently to every candidate."""

from __future__ import annotations

import torch.nn as nn

from .contracts import InteractionOutput, TractorConfig


class CandidateTemporalAggregator(nn.Module):
    def __init__(self, config: TractorConfig):
        super().__init__()
        self.cfg = config
        self.gru = nn.GRU(config.interaction_dim, config.interaction_dim, batch_first=True)

    def forward(self, interaction: InteractionOutput) -> InteractionOutput:
        b, q, k, h, d = interaction.per_time.shape
        sequence = interaction.per_time.reshape(b * q * k, h, d)
        output, _ = self.gru(sequence)
        output = output.reshape(b, q, k, h, d)
        aggregated = output[:, :, :, -1]
        valid = interaction.candidate_valid
        output = output * valid.unsqueeze(-1).unsqueeze(-1).to(output.dtype)
        aggregated = aggregated * valid.unsqueeze(-1).to(aggregated.dtype)
        return InteractionOutput(output, aggregated, interaction.overlap_by_cause, valid)
