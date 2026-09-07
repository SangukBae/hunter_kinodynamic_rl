"""Candidate-private explicit tube-occupancy interaction operator."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .contracts import BeliefState, InteractionOutput, RolloutBatch, TractorConfig, TubeBatch


def _normalized_grid(xy: torch.Tensor, cfg: TractorConfig) -> torch.Tensor:
    x_extent = cfg.grid_width * cfg.resolution_m
    y_extent = cfg.grid_height * cfg.resolution_m
    gx = 2.0 * (xy[..., 0] - cfg.x_min_m) / x_extent - 1.0
    gy = 2.0 * (xy[..., 1] - cfg.y_min_m) / y_extent - 1.0
    return torch.stack((gx, gy), dim=-1)


class CausalTubeOccupancyInteraction(nn.Module):
    """Read-only shared future, candidate-private sparse feature updates.

    The operator contains no operation over the candidate axis.  Therefore a
    candidate permutation can only permute outputs; it cannot change their
    values.  This structural property is backed by property tests.
    """

    def __init__(self, config: TractorConfig):
        super().__init__()
        self.cfg = config
        in_dim = config.scene_channels + 2 + 3 + 3 + 3
        self.private_update = nn.Sequential(
            nn.Linear(in_dim, config.interaction_dim), nn.ELU(),
            nn.Linear(config.interaction_dim, config.interaction_dim), nn.ELU(),
        )

    def _sample_map(self, dense: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        # dense: B,H,C,Y,X; grid: B,Q,K,H,M,2
        b, h, c, gy, gx = dense.shape
        q, k, m = grid.shape[1], grid.shape[2], grid.shape[4]
        dense_expanded = dense[:, None, None].expand(-1, q, k, -1, -1, -1, -1)
        flat_dense = dense_expanded.reshape(b * q * k * h, c, gy, gx)
        flat_grid = grid.permute(0, 1, 2, 3, 4, 5).reshape(b * q * k * h, m, 1, 2)
        sampled = F.grid_sample(
            flat_dense, flat_grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampled = sampled.squeeze(-1).transpose(1, 2)
        return sampled.reshape(b, q, k, h, m, c)

    def forward(
        self, belief: BeliefState, rollout: RolloutBatch, tube: TubeBatch
    ) -> InteractionOutput:
        grid = _normalized_grid(tube.sample_xy, self.cfg)
        sampled_occ = self._sample_map(belief.future_occupancy, grid)
        sampled_feature = self._sample_map(belief.future_feature, grid)
        sampled_flow = self._sample_map(belief.future_dynamic_flow, grid)
        weights = tube.sample_weight.unsqueeze(-1)
        valid_weight = weights * tube.sample_valid.unsqueeze(-1).to(weights.dtype)
        weighted_occ = (sampled_occ * valid_weight).sum(dim=-2)
        weighted_feature = (sampled_feature * valid_weight).sum(dim=-2)
        weighted_flow = (sampled_flow * valid_weight).sum(dim=-2)
        # Occupancy class order is F,S,D,U.  Boundary/OOB is folded into U.
        overlaps = torch.stack((
            weighted_occ[..., 1],
            weighted_occ[..., 2],
            weighted_occ[..., 3] + tube.oob_mass,
        ), dim=-1).clamp(0.0, 1.0)
        pose = rollout.poses
        action = rollout.physical_actions[:, None, :, None, :].expand(
            -1, self.cfg.q_m_eff, -1, self.cfg.horizon_steps, -1
        )
        private_input = torch.cat((weighted_feature, weighted_flow, overlaps, pose, action), dim=-1)
        per_time = self.private_update(private_input)
        candidate_valid = rollout.model_valid & belief.health.model_valid[:, None, :]
        per_time = per_time * tube.horizon_mask.unsqueeze(-1).to(per_time.dtype)
        per_time = per_time * candidate_valid.unsqueeze(-1).unsqueeze(-1).to(per_time.dtype)
        # A masked mean is deterministic and cannot mix candidates/members.
        count = tube.horizon_mask.sum(dim=-1, keepdim=True).clamp_min(1).to(per_time.dtype)
        aggregated = per_time.sum(dim=-2) / count
        return InteractionOutput(per_time, aggregated, overlaps, candidate_valid)
