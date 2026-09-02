"""Masked Dueling Double DQN network (plan section 8.6) for the Global RL
MVP -- a SEPARATE architecture from the continuous TQC actor/critic (plan:
"현재 continuous TQC Actor/Critic을 Global policy에 재사용하지 않는다").

::

    map_tensor    -> CNN encoder    ---+
    scalar_tensor -> MLP encoder    ---+-> fused MLP -> V(s)
    candidate_tensor -> shared MLP  ---+                 |
                                                          v
                        fused (broadcast per candidate) + candidate_feature
                                    -> MLP -> A(s, candidate)
                        Q(s, candidate) = V(s) + (A(s, candidate) - mean_valid(A))
                        Q(s, invalid candidate) := -inf   (never selected)

Requires torch -- import this module only after ``pytest.importorskip("torch")``
(mirrors ``rl/networks/tqc.py``'s own gating convention); the rest of
``navigation.global_rl`` (observation/action_mask/subgoal_sampler/reward/
replay) is plain numpy and importable without it.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from hunter_kinodynamic_rl.config.schema import GlobalRLConfig
from hunter_kinodynamic_rl.navigation.global_rl.observation import (
    N_SCALARS, resolve_candidate_feature_names,
)
from hunter_kinodynamic_rl.navigation.memory.topological_graph import N_NODE_FEATURES

NEG_INF = float("-inf")


class MapEncoder(nn.Module):
    def __init__(self, in_channels: int, cnn_channels: Sequence[int], out_dim: int, crop_size: int):
        super().__init__()
        layers = []
        c_in = in_channels
        for c_out in cnn_channels:
            layers.append(nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1))
            layers.append(nn.ReLU(inplace=True))
            c_in = c_out
        self.conv = nn.Sequential(*layers)
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, crop_size, crop_size)
            flat_dim = int(self.conv(dummy).flatten(1).shape[1])
        self.fc = nn.Linear(flat_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv(x).flatten(1)
        return F.relu(self.fc(z))


class ScalarEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.ReLU(inplace=True), nn.Linear(out_dim, out_dim), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CandidateEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """``values``/``mask``: ``(B, n)`` -> ``(B, 1)`` mean of ``values`` over
    entries where ``mask`` is True. The fallback candidate is always valid
    (see ``action_mask.compute_action_mask``), so the denominator is always
    >= 1 -- never divides by zero."""
    mask_f = mask.to(values.dtype)
    denom = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)
    return (values * mask_f).sum(dim=1, keepdim=True) / denom


def masked_argmax(q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """``q``/``mask``: ``(B, n)`` -> ``(B,)`` argmax index restricted to
    ``mask``-True entries. Never picks an invalid candidate: masked-out
    entries are set to ``-inf`` before the argmax."""
    q_masked = q.masked_fill(~mask, NEG_INF)
    return torch.argmax(q_masked, dim=-1)


def masked_mean_features(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """``values``: ``(B, n, D)``, ``mask``: ``(B, n)`` -> ``(B, D)`` mean of
    ``values`` over entries where ``mask`` is True, per feature dimension --
    the ``masked_mean`` pooling above generalized to feature VECTORS (Phase
    5's node-tensor pooling into one fixed-size topology summary, plan 9.5
    step 2's "GNN encoder는 나중에" -- masked-mean pooling is the ungraphed
    stand-in). ``n == 0`` (topology disabled / empty graph) degrades
    gracefully to an all-zero ``(B, D)`` result, never a division error --
    the denominator clamp below applies exactly as it does for
    :func:`masked_mean`."""
    mask_f = mask.to(values.dtype).unsqueeze(-1)
    denom = mask_f.sum(dim=1).clamp(min=1.0)
    return (values * mask_f).sum(dim=1) / denom


class MaskedDuelingDQN(nn.Module):
    def __init__(self, config: GlobalRLConfig, map_channels: int, max_nodes: int = 0):
        """``max_nodes`` (Phase 5, plan 9.5) is the node-tensor width this
        network should expect -- ``0`` (default) reproduces Phase 4's exact
        architecture byte-for-byte (no node encoder submodule constructed
        at all, ``forward``'s ``node_tensor``/``node_validity_mask``
        arguments are then optional and ignored). Pass
        ``profile.memory.max_nodes_in_observation`` when
        ``config.topology_feedback_enabled`` is True, ``0`` otherwise --
        mirrors how ``map_channels`` is already computed by the caller from
        a DIFFERENT config section (``resolve_map_channel_names``) rather
        than hardcoded here."""
        super().__init__()
        self.config = config
        self.max_nodes = int(max_nodes)
        n_candidate_features = len(resolve_candidate_feature_names(config))
        self.map_encoder = MapEncoder(map_channels, config.cnn_channels, config.map_feature_dim,
                                       config.map_crop_size_cells)
        self.scalar_encoder = ScalarEncoder(N_SCALARS, config.scalar_feature_dim)
        self.candidate_encoder = CandidateEncoder(n_candidate_features, config.candidate_feature_dim)
        fused_in = config.map_feature_dim + config.scalar_feature_dim
        if self.max_nodes > 0:
            self.node_encoder = CandidateEncoder(N_NODE_FEATURES, config.candidate_feature_dim)
            fused_in += config.candidate_feature_dim
        else:
            self.node_encoder = None
        self.fuse = nn.Sequential(nn.Linear(fused_in, config.fused_feature_dim), nn.ReLU(inplace=True))
        self.value_head = nn.Linear(config.fused_feature_dim, 1)
        self.advantage_head = nn.Sequential(
            nn.Linear(config.fused_feature_dim + config.candidate_feature_dim, config.fused_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(config.fused_feature_dim, 1),
        )

    def forward(
        self, map_tensor: torch.Tensor, scalar_tensor: torch.Tensor, candidate_tensor: torch.Tensor,
        action_mask: torch.Tensor, node_tensor: torch.Tensor = None, node_validity_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """``map_tensor``: ``(B, C, H, W)``; ``scalar_tensor``: ``(B, N_SCALARS)``;
        ``candidate_tensor``: ``(B, n, len(resolve_candidate_feature_names(config)))``;
        ``action_mask``: ``(B, n)`` bool; ``node_tensor``/``node_validity_mask``
        (``(B, max_nodes, N_NODE_FEATURES)``/``(B, max_nodes)``) are REQUIRED
        when this network was constructed with ``max_nodes > 0``, ignored
        (may be omitted) otherwise. Returns ``(B, n)`` Q-values, ``-inf`` at
        every masked-out (invalid) candidate -- selection code needs no
        separate masking step (plan section 8.6: "invalid Q = -inf at
        selection")."""
        map_feat = self.map_encoder(map_tensor)
        scalar_feat = self.scalar_encoder(scalar_tensor)
        fuse_inputs = [map_feat, scalar_feat]
        if self.node_encoder is not None:
            if node_tensor is None or node_validity_mask is None:
                raise ValueError(
                    "MaskedDuelingDQN.forward(): this network was constructed with max_nodes > 0 -- "
                    "node_tensor/node_validity_mask are required, not optional"
                )
            node_feat = self.node_encoder(node_tensor)
            fuse_inputs.append(masked_mean_features(node_feat, node_validity_mask))
        fused = self.fuse(torch.cat(fuse_inputs, dim=1))
        cand_feat = self.candidate_encoder(candidate_tensor)
        n = cand_feat.shape[1]
        fused_expand = fused.unsqueeze(1).expand(-1, n, -1)
        advantage = self.advantage_head(torch.cat([fused_expand, cand_feat], dim=-1)).squeeze(-1)
        value = self.value_head(fused)
        adv_mean = masked_mean(advantage, action_mask)
        q = value + (advantage - adv_mean)
        return q.masked_fill(~action_mask, NEG_INF)
