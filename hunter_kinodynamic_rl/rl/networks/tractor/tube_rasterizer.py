"""Sparse probabilistic swept-footprint rasterization with explicit OOB mass."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .contracts import RolloutBatch, TractorConfig, TubeBatch


class TubeRasterizer(nn.Module):
    def __init__(self, config: TractorConfig):
        super().__init__()
        self.cfg = config
        m = config.sparse_tube_samples
        idx = torch.arange(m, dtype=torch.float32)
        angle = idx * (math.pi * (3.0 - math.sqrt(5.0)))
        radius = torch.sqrt((idx + 0.5) / m)
        offsets = torch.stack((radius * torch.cos(angle), radius * torch.sin(angle)), dim=-1)
        self.register_buffer("unit_disk_offsets", offsets, persistent=False)

    def forward(self, rollout: RolloutBatch) -> TubeBatch:
        pose = rollout.poses
        xy = pose[..., :2]
        yaw = pose[..., 2]
        xy_cov = rollout.covariance[..., :2, :2]
        eigen = torch.linalg.eigvalsh(xy_cov).clamp_min(0.0)
        uncertainty = 2.0 * torch.sqrt(eigen[..., -1])
        radius = self.cfg.footprint_radius_m + uncertainty
        offsets = self.unit_disk_offsets.to(dtype=pose.dtype)
        ox = offsets[:, 0].view(*([1] * yaw.ndim), -1) * radius.unsqueeze(-1)
        oy = offsets[:, 1].view(*([1] * yaw.ndim), -1) * radius.unsqueeze(-1)
        c, s = torch.cos(yaw).unsqueeze(-1), torch.sin(yaw).unsqueeze(-1)
        sample_x = xy[..., 0].unsqueeze(-1) + c * ox - s * oy
        sample_y = xy[..., 1].unsqueeze(-1) + s * ox + c * oy
        samples = torch.stack((sample_x, sample_y), dim=-1)
        col = torch.floor((sample_x - self.cfg.x_min_m) / self.cfg.resolution_m)
        row = torch.floor((sample_y - self.cfg.y_min_m) / self.cfg.resolution_m)
        valid = (
            (row >= 0) & (row < self.cfg.grid_height)
            & (col >= 0) & (col < self.cfg.grid_width)
            & rollout.horizon_mask.unsqueeze(-1)
        )
        weights = pose.new_full(sample_x.shape, 1.0 / self.cfg.sparse_tube_samples)
        weights = weights * rollout.horizon_mask.unsqueeze(-1).to(weights.dtype)
        oob = (weights * ~valid).sum(dim=-1)
        return TubeBatch(samples, weights, valid, oob, rollout.horizon_mask)
