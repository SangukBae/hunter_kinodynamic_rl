"""Single-Gaussian actor and deterministic structured candidate proposal."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .contracts import BeliefState, CandidateSet, TractorConfig


class TractorActor(nn.Module):
    def __init__(self, config: TractorConfig):
        super().__init__()
        self.cfg = config
        dim = config.scene_channels + config.ego_dim + config.plant_dim
        self.body = nn.Sequential(nn.Linear(dim, 256), nn.ELU(), nn.Linear(256, 256), nn.ELU())
        self.mean = nn.Linear(256, 3)
        self.log_std = nn.Linear(256, 3)

    def distribution(self, belief: BeliefState):
        pooled = belief.scene_feature.mean(dim=(-2, -1))
        hidden = self.body(torch.cat((pooled, belief.ego_latent, belief.plant_latent), dim=-1))
        mean = self.mean(hidden)
        log_std = self.log_std(hidden).clamp(self.cfg.log_std_min, self.cfg.log_std_max)
        return mean, log_std

    def sample(
        self,
        belief: BeliefState,
        deterministic: bool = False,
        generator: torch.Generator | None = None,
    ):
        mean, log_std = self.distribution(belief)
        noise = torch.zeros_like(mean) if deterministic else torch.randn(
            mean.shape, dtype=mean.dtype, device=mean.device, generator=generator
        )
        pre_tanh = mean + log_std.exp() * noise
        action = torch.tanh(pre_tanh)
        normal_log_prob = -0.5 * (
            ((pre_tanh - mean) / log_std.exp()).pow(2) + 2.0 * log_std + math.log(2.0 * math.pi)
        )
        log_prob = normal_log_prob.sum(-1, keepdim=True)
        log_prob -= (2.0 * (math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh))).sum(-1, keepdim=True)
        return action, log_prob

    def propose(self, belief: BeliefState) -> CandidateSet:
        mean, _ = self.distribution(belief)
        base = torch.tanh(mean)
        offsets = base.new_tensor([
            [0.0, 0.0, 0.0],
            [0.0, -2.0, 0.0],
            [-0.45, 0.0, 0.0],
            [0.45, 0.0, 0.0],
            [-0.9, -0.35, 0.0],
            [0.9, -0.35, 0.0],
            [0.0, -0.2, -0.6],
            [0.0, -0.2, 0.6],
        ])
        if self.cfg.num_candidates > offsets.shape[0]:
            repeats = math.ceil(self.cfg.num_candidates / offsets.shape[0])
            offsets = offsets.repeat(repeats, 1)
        offsets = offsets[: self.cfg.num_candidates]
        actions = (base[:, None, :] + offsets[None, :, :]).clamp(-1.0, 1.0)
        # Slot 1 is the unique physical stop unless K=1.  Preserve base
        # curvature and arc while forcing normalized speed to its lower bound.
        stop_index = 0 if self.cfg.num_candidates == 1 else 1
        stop = torch.stack((base[:, 0], torch.full_like(base[:, 1], -1.0), base[:, 2]), dim=-1)
        slot = torch.arange(self.cfg.num_candidates, device=actions.device) == stop_index
        actions = torch.where(slot[None, :, None], stop[:, None, :], actions)
        present = torch.ones(actions.shape[:2], dtype=torch.bool, device=actions.device)
        is_stop = torch.zeros_like(present)
        base_is_stop = base[:, 1] <= -1.0 + 1e-6
        is_stop[:, 0] = base_is_stop
        is_stop[:, stop_index] = ~base_is_stop
        # Mask later duplicates while retaining fixed-capacity storage and
        # the actor base in slot 0. Comparisons are metadata-only and do not
        # alter the action gradient of any present candidate.
        detached = actions.detach()
        for index in range(1, self.cfg.num_candidates):
            duplicate = torch.zeros(actions.shape[0], dtype=torch.bool, device=actions.device)
            for earlier in range(index):
                duplicate |= present[:, earlier] & torch.isclose(
                    detached[:, index], detached[:, earlier], atol=1e-6, rtol=0.0
                ).all(dim=-1)
            present[:, index] &= ~duplicate
            is_stop[:, index] &= present[:, index]
        return CandidateSet(actions, present, is_stop, "actor_structured_v1")
