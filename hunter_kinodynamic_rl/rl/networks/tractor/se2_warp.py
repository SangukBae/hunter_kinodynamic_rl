"""SE(2) history alignment into the current ego frame."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .contracts import TractorConfig


def compose_pose(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compose batched planar transforms ``a`` then ``b`` as [x,y,yaw]."""
    c, s = torch.cos(b[:, 2]), torch.sin(b[:, 2])
    x = b[:, 0] + c * a[:, 0] - s * a[:, 1]
    y = b[:, 1] + s * a[:, 0] + c * a[:, 1]
    return torch.stack((x, y, a[:, 2] + b[:, 2]), dim=-1)


class SE2HistoryWarp(torch.nn.Module):
    def __init__(self, config: TractorConfig):
        super().__init__()
        self.cfg = config

    def _warp(self, image: torch.Tensor, older_to_current: torch.Tensor) -> torch.Tensor:
        b = image.shape[0]
        yaw = older_to_current[:, 2]
        c, s = torch.cos(yaw), torch.sin(yaw)
        # affine_grid maps output(current) coordinates to input(older).  The
        # inverse of older->current is R^T(current - t).
        tx = older_to_current[:, 0] / (self.cfg.grid_width * self.cfg.resolution_m / 2.0)
        ty = older_to_current[:, 1] / (self.cfg.grid_height * self.cfg.resolution_m / 2.0)
        theta = image.new_zeros((b, 2, 3))
        theta[:, 0, 0] = c
        theta[:, 0, 1] = s
        theta[:, 1, 0] = -s
        theta[:, 1, 1] = c
        theta[:, 0, 2] = -(c * tx + s * ty)
        theta[:, 1, 2] = -(-s * tx + c * ty)
        grid = F.affine_grid(theta, image.shape, align_corners=False)
        return F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    def forward(
        self, evidence: torch.Tensor, motion_delta: torch.Tensor, motion_valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, t = evidence.shape[:2]
        if t != self.cfg.t_obs:
            raise ValueError("evidence history length does not match config")
        aligned = [evidence[:, 0]]
        chain_valid = torch.ones((b,), dtype=torch.bool, device=evidence.device)
        aligned_valid = [chain_valid]
        cumulative = evidence.new_zeros((b, 3))
        for older_index in range(1, t):
            edge = motion_delta[:, older_index - 1, :3]
            cumulative = edge if older_index == 1 else compose_pose(edge, cumulative)
            chain_valid = chain_valid & motion_valid[:, older_index - 1].bool()
            warped = self._warp(evidence[:, older_index], cumulative)
            # grid_sample introduces small interpolation noise even for the
            # exact identity transform. Preserve identity histories bitwise;
            # this matters for deterministic replay and resume comparisons.
            identity = (cumulative == 0.0).all(dim=-1)
            warped = torch.where(
                identity[:, None, None, None], evidence[:, older_index], warped,
            )
            aligned.append(torch.where(chain_valid[:, None, None, None], warped, torch.zeros_like(warped)))
            aligned_valid.append(chain_valid)
        return torch.stack(aligned, dim=1), torch.stack(aligned_valid, dim=1)
