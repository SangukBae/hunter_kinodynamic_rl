"""Differentiable normalized-action decode and Ackermann response rollout."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .contracts import CandidateSet, DecisionContext, RolloutBatch, TractorConfig
from .residual_dynamics import VehicleResidualEnsemble


def _move_towards(current: torch.Tensor, target: torch.Tensor, max_delta: float) -> torch.Tensor:
    return current + (target - current).clamp(-max_delta, max_delta)


class NominalRolloutAdapter(nn.Module):
    def __init__(self, config: TractorConfig):
        super().__init__()
        self.cfg = config
        self.residuals = VehicleResidualEnsemble(config)

    def decode(self, actions: torch.Tensor) -> torch.Tensor:
        actions = actions.clamp(-1.0, 1.0)
        kappa_max = math.tan(self.cfg.steering_limit_rad) / self.cfg.wheelbase_m
        kappa = actions[..., 0] * kappa_max
        speed = self.cfg.min_speed_mps + 0.5 * (actions[..., 1] + 1.0) * (
            self.cfg.max_speed_mps - self.cfg.min_speed_mps
        )
        arc = self.cfg.min_arc_m + 0.5 * (actions[..., 2] + 1.0) * (
            self.cfg.max_arc_m - self.cfg.min_arc_m
        )
        return torch.stack((kappa, speed, arc), dim=-1)

    def encode(self, physical_actions: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`decode` for manifest/data round-trip checks."""
        kappa, speed, arc = physical_actions.unbind(-1)
        kappa_max = math.tan(self.cfg.steering_limit_rad) / self.cfg.wheelbase_m
        normalized_kappa = kappa / kappa_max
        normalized_speed = 2.0 * (speed - self.cfg.min_speed_mps) / (
            self.cfg.max_speed_mps - self.cfg.min_speed_mps
        ) - 1.0
        normalized_arc = 2.0 * (arc - self.cfg.min_arc_m) / (
            self.cfg.max_arc_m - self.cfg.min_arc_m
        ) - 1.0
        return torch.stack((normalized_kappa, normalized_speed, normalized_arc), dim=-1)

    def forward(
        self,
        candidates: CandidateSet,
        context: DecisionContext,
        plant_latent: torch.Tensor,
    ) -> RolloutBatch:
        actions = self.decode(candidates.normalized_actions)
        b, k = actions.shape[:2]
        q_m = self.cfg.q_m_eff
        device, dtype = actions.device, actions.dtype
        poses_all, cov_all = [], []
        horizon_all, valid_all = [], []
        substeps = max(1, int(math.ceil(self.cfg.dt_out_sec / self.cfg.dt_dyn_sec)))
        dt = self.cfg.dt_out_sec / substeps

        kappa, v_ref, arc = actions.unbind(-1)
        raw_horizon = torch.where(
            v_ref <= 0.05,
            torch.full_like(v_ref, self.cfg.max_horizon_sec),
            arc / v_ref.clamp_min(0.05),
        )
        commit_horizon = raw_horizon.clamp(self.cfg.min_horizon_sec, self.cfg.max_horizon_sec)
        score_horizon = torch.maximum(
            commit_horizon,
            torch.full_like(raw_horizon, self.cfg.min_safety_horizon_sec),
        )
        horizon_steps = torch.ceil(score_horizon / self.cfg.dt_out_sec).long().clamp(1, self.cfg.horizon_steps)
        time_ids = torch.arange(1, self.cfg.horizon_steps + 1, device=device).view(1, 1, -1)
        horizon_mask = time_ids <= horizon_steps.unsqueeze(-1)

        for member_index in range(q_m):
            x = actions.new_zeros((b, k))
            y = actions.new_zeros((b, k))
            yaw = actions.new_zeros((b, k))
            v = context.response[:, 0:1].expand(-1, k).clone()
            steer = context.response[:, 2:3].expand(-1, k).clone()
            lagged_v = v.clone()
            cov = context.localization_covariance[:, None].expand(-1, k, -1, -1).clone()
            member_poses, member_cov = [], []
            for h in range(self.cfg.horizon_steps):
                for substep in range(substeps):
                    target_steer = torch.atan(self.cfg.wheelbase_m * kappa)
                    blend = (dt / commit_horizon.clamp_min(dt)).clamp(0.0, 1.0)
                    requested_steer = steer + blend * (target_steer - steer)
                    speed_delta = torch.where(
                        v_ref.abs() >= v.abs(),
                        torch.full_like(v, self.cfg.accel_limit_mps2 * dt),
                        torch.full_like(v, self.cfg.brake_decel_mps2 * dt),
                    )
                    v_rate = v + (v_ref - v).clamp(-speed_delta, speed_delta)
                    steer_next = _move_towards(
                        steer, requested_steer, self.cfg.steering_rate_rad_s * dt
                    ).clamp(-self.cfg.steering_limit_rad, self.cfg.steering_limit_rad)
                    if self.cfg.speed_lag_tau_sec > 0.0:
                        alpha = 1.0 - math.exp(-dt / self.cfg.speed_lag_tau_sec)
                        v_next = lagged_v + alpha * (v_rate - lagged_v)
                    else:
                        v_next = v_rate
                    if self.cfg.residual_members > 0:
                        residual = self.residuals.members[member_index]
                        pa = actions.reshape(b * k, 3)
                        zp = plant_latent[:, None, :].expand(-1, k, -1).reshape(b * k, -1)
                        response = torch.stack((v, v * torch.tan(steer) / self.cfg.wheelbase_m, steer), -1)
                        response_mask = context.response_valid[:, None, :].expand(-1, k, -1)
                        nominal = torch.stack((
                            v_next, v_next * torch.tan(steer_next) / self.cfg.wheelbase_m, steer_next,
                        ), -1)
                        dt_tensor = actions.new_full((b * k, 1), dt / self.cfg.dt_dyn_sec)
                        elapsed = actions.new_full(
                            (b * k, 1), (h * substeps + substep + 1) / (self.cfg.horizon_steps * substeps)
                        )
                        delta, q = residual(
                            zp, pa, response.reshape(b * k, 3), response_mask.reshape(b * k, 3),
                            nominal.reshape(b * k, 3),
                            dt_tensor, elapsed,
                        )
                        delta = delta.reshape(b, k, 2)
                        v_next = (v_next + delta[..., 0]).clamp(
                            self.cfg.min_speed_mps, self.cfg.max_speed_mps
                        )
                        steer_next = (steer_next + delta[..., 1]).clamp(
                            -self.cfg.steering_limit_rad, self.cfg.steering_limit_rad
                        )
                        q = q.reshape(b, k, 2)
                    else:
                        q = actions.new_zeros((b, k, 2))
                    v_bar = 0.5 * (lagged_v + v_next)
                    steer_bar = 0.5 * (steer + steer_next)
                    omega = v_bar * torch.tan(steer_bar) / self.cfg.wheelbase_m
                    yaw_mid = yaw + 0.5 * omega * dt
                    x = x + v_bar * torch.cos(yaw_mid) * dt
                    y = y + v_bar * torch.sin(yaw_mid) * dt
                    yaw = yaw + omega * dt
                    v, lagged_v, steer = v_rate, v_next, steer_next
                    jacobian = torch.eye(3, dtype=dtype, device=device).view(1, 1, 3, 3).expand(b, k, -1, -1).clone()
                    jacobian[..., 0, 2] = -v_bar * torch.sin(yaw_mid) * dt
                    jacobian[..., 1, 2] = v_bar * torch.cos(yaw_mid) * dt
                    cov = jacobian @ cov @ jacobian.transpose(-1, -2)
                    process = cov.new_zeros(cov.shape)
                    process[..., 0, 0] = self.cfg.process_noise_xy_m2 * dt + q[..., 0] * dt
                    process[..., 1, 1] = self.cfg.process_noise_xy_m2 * dt + q[..., 0] * dt
                    process[..., 2, 2] = self.cfg.process_noise_yaw_rad2 * dt + q[..., 1] * dt
                    cov = 0.5 * (cov + cov.transpose(-1, -2)) + process
                member_poses.append(torch.stack((x, y, yaw), dim=-1))
                member_cov.append(cov)
            poses_all.append(torch.stack(member_poses, dim=2))
            cov_all.append(torch.stack(member_cov, dim=2))
            horizon_all.append(horizon_mask)
            valid_all.append(candidates.present & torch.isfinite(actions).all(dim=-1))
        return RolloutBatch(
            physical_actions=actions,
            poses=torch.stack(poses_all, dim=1),
            covariance=torch.stack(cov_all, dim=1),
            horizon_mask=torch.stack(horizon_all, dim=1),
            model_valid=torch.stack(valid_all, dim=1),
        )
