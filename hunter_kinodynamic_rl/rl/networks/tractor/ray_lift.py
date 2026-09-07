"""Deterministic LiDAR ray lifting into hit/free/observed/age BEV evidence."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .contracts import TractorConfig


class RayLift(nn.Module):
    def __init__(self, config: TractorConfig):
        super().__init__()
        self.cfg = config
        angles = torch.linspace(-math.pi / 2.0, math.pi / 2.0, config.n_scan)
        fractions = torch.linspace(0.0, 1.0, config.ray_free_samples + 1)[:-1]
        self.register_buffer("angles", angles, persistent=False)
        self.register_buffer("free_fractions", fractions, persistent=False)

    def _indices(self, x: torch.Tensor, y: torch.Tensor):
        col = torch.floor((x - self.cfg.x_min_m) / self.cfg.resolution_m).long()
        row = torch.floor((y - self.cfg.y_min_m) / self.cfg.resolution_m).long()
        valid = (
            (row >= 0) & (row < self.cfg.grid_height)
            & (col >= 0) & (col < self.cfg.grid_width)
        )
        flat = row.clamp(0, self.cfg.grid_height - 1) * self.cfg.grid_width
        flat = flat + col.clamp(0, self.cfg.grid_width - 1)
        return flat, valid

    def forward(self, ranges: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        if ranges.ndim != 3 or tuple(ranges.shape) != tuple(valid.shape):
            raise ValueError("ranges and valid must have identical (B,T,N) shapes")
        b, t, n = ranges.shape
        if t != self.cfg.t_obs or n != self.cfg.n_scan:
            raise ValueError("scan shape does not match TractorConfig")
        ranges = ranges.clamp(0.0, self.cfg.max_range_m)
        valid = valid.bool()
        hit_valid = valid & (ranges < self.cfg.max_range_m - 1e-4)
        cos_a = torch.cos(self.angles).view(1, 1, n)
        sin_a = torch.sin(self.angles).view(1, 1, n)

        hit_x, hit_y = ranges * cos_a, ranges * sin_a
        hit_idx, hit_in_grid = self._indices(hit_x, hit_y)

        frac = self.free_fractions.view(1, 1, 1, -1)
        free_r = ranges.unsqueeze(-1) * frac
        free_x = free_r * cos_a.unsqueeze(-1)
        free_y = free_r * sin_a.unsqueeze(-1)
        free_idx, free_in_grid = self._indices(free_x, free_y)

        cells = self.cfg.grid_height * self.cfg.grid_width
        hit = ranges.new_zeros((b, t, cells))
        free = ranges.new_zeros((b, t, cells))
        observed = ranges.new_zeros((b, t, cells))
        hit.scatter_add_(2, hit_idx, (hit_valid & hit_in_grid).to(ranges.dtype))
        free_mask = valid.unsqueeze(-1) & free_in_grid
        free.scatter_add_(2, free_idx.reshape(b, t, -1), free_mask.to(ranges.dtype).reshape(b, t, -1))
        observed.scatter_add_(
            2, free_idx.reshape(b, t, -1), free_mask.to(ranges.dtype).reshape(b, t, -1)
        )
        observed.scatter_add_(2, hit_idx, (valid & hit_in_grid).to(ranges.dtype))

        hit = hit.clamp(0.0, 1.0)
        free = free.clamp(0.0, 1.0) * (1.0 - hit)
        observed = observed.clamp(0.0, 1.0)
        age = torch.linspace(0.0, 1.0, t, device=ranges.device, dtype=ranges.dtype)
        age = age.view(1, t, 1).expand(b, -1, cells) * observed
        evidence = torch.stack((hit, free, observed, age), dim=2)
        return evidence.reshape(b, t, 4, self.cfg.grid_height, self.cfg.grid_width)
