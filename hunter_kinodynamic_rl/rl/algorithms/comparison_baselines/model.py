"""Fair-comparison B1--B8 models sharing the A7 observation/action contract."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from hunter_kinodynamic_rl.config.comparison import ComparisonModelConfig
from hunter_kinodynamic_rl.rl.networks.tqc import Actor as CurrentTQCActor
from hunter_kinodynamic_rl.rl.networks.tqc import Critic as CurrentTQCCritic
from hunter_kinodynamic_rl.rl.networks.tractor import TractorConfig, TractorInputs
from hunter_kinodynamic_rl.rl.networks.tractor.belief_encoder import BeliefEncoder


@dataclass(frozen=True)
class EncodedComparisonState:
    vector: torch.Tensor
    spatial: torch.Tensor | None = None

    def index_select(self, indices: torch.Tensor) -> "EncodedComparisonState":
        return EncodedComparisonState(
            self.vector.index_select(0, indices),
            None if self.spatial is None else self.spatial.index_select(0, indices),
        )


class SquashedGaussianActor(nn.Module):
    def __init__(self, input_dim: int, cfg: ComparisonModelConfig):
        super().__init__()
        self.cfg = cfg
        self.body = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ELU(), nn.Linear(256, 256), nn.ELU(),
        )
        self.mean = nn.Linear(256, 3)
        self.log_std = nn.Linear(256, 3)

    def distribution(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.body(state)
        return self.mean(hidden), self.log_std(hidden).clamp(
            self.cfg.log_std_min, self.cfg.log_std_max,
        )

    def sample(
        self, state: torch.Tensor, deterministic: bool = False,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.distribution(state)
        noise = torch.zeros_like(mean) if deterministic else torch.randn(
            mean.shape, dtype=mean.dtype, device=mean.device, generator=generator,
        )
        pre_tanh = mean + log_std.exp() * noise
        action = torch.tanh(pre_tanh)
        log_prob = -0.5 * (
            ((pre_tanh - mean) / log_std.exp()).square()
            + 2.0 * log_std + math.log(2.0 * math.pi)
        )
        log_prob = log_prob.sum(-1, keepdim=True)
        log_prob -= (2.0 * (
            math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh)
        )).sum(-1, keepdim=True)
        return action, log_prob


class CurrentTQCActorAdapter(nn.Module):
    """Expose the package's current three-layer ReLU TQC actor via the common API."""

    def __init__(self, cfg: ComparisonModelConfig):
        super().__init__()
        self.actor = CurrentTQCActor(
            cfg.observation_dim, 3, hdim=256, activ=F.relu,
            log_std_min=cfg.current_tqc_log_std_min,
            log_std_max=cfg.current_tqc_log_std_max,
        )

    def distribution(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.actor.activ(self.actor.l1(state))
        hidden = self.actor.activ(self.actor.l2(hidden))
        hidden = self.actor.activ(self.actor.l3(hidden))
        return self.actor.mean(hidden), self.actor.log_std(hidden).clamp(
            self.actor.log_std_min, self.actor.log_std_max,
        )

    def sample(
        self, state: torch.Tensor, deterministic: bool = False,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.distribution(state)
        noise = torch.zeros_like(mean) if deterministic else torch.randn(
            mean.shape, dtype=mean.dtype, device=mean.device, generator=generator,
        )
        pre_tanh = mean + log_std.exp() * noise
        action = torch.tanh(pre_tanh)
        log_prob = -0.5 * (
            ((pre_tanh - mean) / log_std.exp()).square()
            + 2.0 * log_std + math.log(2.0 * math.pi)
        )
        log_prob = log_prob.sum(-1, keepdim=True)
        log_prob -= (2.0 * (
            math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh)
        )).sum(-1, keepdim=True)
        return action, log_prob


class CurrentTQCObservation(nn.Module):
    """Pass the 328D observation directly, as the current TQC does."""

    def forward(self, inputs: TractorInputs) -> EncodedComparisonState:
        return EncodedComparisonState(inputs.observation)


class FlatEncoder(nn.Module):
    def __init__(self, cfg: ComparisonModelConfig):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(cfg.observation_dim, cfg.encoder_hidden), nn.ELU(),
            nn.Linear(cfg.encoder_hidden, cfg.latent_dim), nn.ELU(),
        )

    def forward(self, inputs: TractorInputs) -> EncodedComparisonState:
        return EncodedComparisonState(self.network(inputs.observation))


class RecurrentVectorEncoder(nn.Module):
    """B3 recurrent-vector baseline over the same four LiDAR frames."""

    def __init__(self, cfg: ComparisonModelConfig):
        super().__init__()
        self.cfg = cfg
        self.scan_projection = nn.Sequential(
            nn.Linear(cfg.n_scan, cfg.encoder_hidden), nn.ELU(),
        )
        self.scan_gru = nn.GRU(cfg.encoder_hidden, cfg.recurrent_hidden, batch_first=True)
        self.tail = nn.Sequential(nn.Linear(cfg.tail_dim, 64), nn.ELU())
        self.output = nn.Sequential(
            nn.Linear(cfg.recurrent_hidden + 64, cfg.latent_dim), nn.ELU(),
        )

    def forward(self, inputs: TractorInputs) -> EncodedComparisonState:
        scan = inputs.observation[:, : self.cfg.t_obs * self.cfg.n_scan].reshape(
            -1, self.cfg.t_obs, self.cfg.n_scan,
        )
        # Observation order is newest first; recurrent processing is chronological.
        scan = torch.flip(scan, dims=(1,))
        valid = torch.flip(inputs.scan_valid.any(dim=-1), dims=(1,)).to(scan.dtype)
        embedded = self.scan_projection(scan) * valid.unsqueeze(-1)
        recurrent, _ = self.scan_gru(embedded)
        tail = inputs.observation[:, -self.cfg.tail_dim :]
        vector = self.output(torch.cat((recurrent[:, -1], self.tail(tail)), dim=-1))
        return EncodedComparisonState(vector)


class BevEncoder(nn.Module):
    """B4/B5 use A7's perception representation but no explicit trajectory tube."""

    def __init__(self, cfg: ComparisonModelConfig, input_cfg: TractorConfig):
        super().__init__()
        self.belief = BeliefEncoder(input_cfg)
        source_dim = input_cfg.scene_channels + input_cfg.ego_dim + input_cfg.plant_dim
        self.vector = nn.Sequential(nn.Linear(source_dim, cfg.latent_dim), nn.ELU())

    def forward(self, inputs: TractorInputs) -> EncodedComparisonState:
        belief, _ = self.belief(inputs)
        pooled = belief.scene_feature.mean(dim=(-2, -1))
        vector = self.vector(torch.cat((pooled, belief.ego_latent, belief.plant_latent), dim=-1))
        return EncodedComparisonState(vector, belief.scene_feature)


class QuantileHeads(nn.Module):
    def __init__(self, input_dim: int, cfg: ComparisonModelConfig):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim + 3, 256), nn.ELU(),
                nn.Linear(256, 256), nn.ELU(),
                nn.Linear(256, 256), nn.ELU(),
                nn.Linear(256, cfg.n_quantiles),
            ) for _ in range(cfg.n_critics)
        ])

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        features = torch.cat((state, action), dim=-1)
        return torch.stack([head(features) for head in self.heads], dim=1)


class CrossAttentionQuantileHeads(nn.Module):
    """B5 generic action-query attention, intentionally without a rollout/tube."""

    def __init__(self, cfg: ComparisonModelConfig, scene_channels: int):
        super().__init__()
        if scene_channels % cfg.attention_heads:
            raise ValueError("scene channels must be divisible by attention_heads")
        self.query = nn.Linear(3, scene_channels)
        self.attention = nn.MultiheadAttention(
            scene_channels, cfg.attention_heads, batch_first=True,
        )
        self.fusion = nn.Sequential(
            nn.Linear(cfg.latent_dim + scene_channels, cfg.latent_dim), nn.ELU(),
        )
        self.heads = QuantileHeads(cfg.latent_dim, cfg)

    def forward(self, state: EncodedComparisonState, action: torch.Tensor) -> torch.Tensor:
        if state.spatial is None:
            raise RuntimeError("B5 cross-attention requires spatial BEV features")
        tokens = F.adaptive_avg_pool2d(state.spatial, (8, 8)).flatten(2).transpose(1, 2)
        query = self.query(action).unsqueeze(1)
        attended, _ = self.attention(query, tokens, tokens, need_weights=False)
        fused = self.fusion(torch.cat((state.vector, attended[:, 0]), dim=-1))
        return self.heads(fused, action)


class ComparisonModel(nn.Module):
    def __init__(self, cfg: ComparisonModelConfig, input_cfg: TractorConfig):
        super().__init__()
        cfg.validate()
        input_cfg.validate()
        if cfg.observation_dim != input_cfg.observation_dim:
            raise ValueError("baseline and A7 input dimensions differ")
        self.config = cfg
        self.input_config = input_cfg
        if cfg.method_id == "B1":
            self.encoder = CurrentTQCObservation()
        elif cfg.method_id == "B3":
            self.encoder = RecurrentVectorEncoder(cfg)
        elif cfg.method_id in {"B4", "B5"}:
            self.encoder = BevEncoder(cfg, input_cfg)
        else:
            self.encoder = FlatEncoder(cfg)
        self.actor = (
            CurrentTQCActorAdapter(cfg)
            if cfg.method_id == "B1"
            else SquashedGaussianActor(cfg.latent_dim, cfg)
        )
        self.latent_transition = None
        if cfg.method_id == "B6":
            self.latent_transition = nn.Sequential(
                nn.Linear(cfg.latent_dim + 3, cfg.latent_dim), nn.ELU(),
                nn.Linear(cfg.latent_dim, cfg.latent_dim),
            )
        self.cross_attention = None
        if cfg.method_id == "B5":
            self.cross_attention = CrossAttentionQuantileHeads(cfg, input_cfg.scene_channels)
            self.critics = None
        elif cfg.method_id == "B1":
            self.critics = CurrentTQCCritic(
                cfg.observation_dim, 3, hdim=256, activ=F.elu,
                n_quantiles=cfg.n_quantiles, n_critics=cfg.n_critics,
            )
        else:
            self.critics = QuantileHeads(cfg.latent_dim, cfg)
        self.risk_head = None
        if cfg.method_id == "B8":
            self.risk_head = nn.Sequential(
                nn.Linear(cfg.latent_dim + 3, 256), nn.ELU(),
                nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 1),
            )

    def encode(self, inputs: TractorInputs) -> EncodedComparisonState:
        inputs.validate(self.input_config)
        return self.encoder(inputs)

    def sample_action(
        self, state: EncodedComparisonState, deterministic: bool = False,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actor.sample(state.vector, deterministic=deterministic, generator=generator)

    def quantiles_from_state(
        self, state: EncodedComparisonState, action: torch.Tensor,
    ) -> torch.Tensor:
        if self.cross_attention is not None:
            return self.cross_attention(state, action)
        value_state = state.vector
        if self.latent_transition is not None:
            value_state = value_state + self.latent_transition(torch.cat((value_state, action), dim=-1))
        assert self.critics is not None
        return self.critics(value_state, action)

    def quantiles(self, inputs: TractorInputs, action: torch.Tensor) -> torch.Tensor:
        return self.quantiles_from_state(self.encode(inputs), action)

    def risk_probability_from_state(
        self, state: EncodedComparisonState, action: torch.Tensor,
    ) -> torch.Tensor:
        if self.risk_head is None:
            raise RuntimeError("scalar endpoint risk exists only for B8")
        if action.ndim == 2:
            return torch.sigmoid(self.risk_head(torch.cat((state.vector, action), dim=-1)))
        if action.ndim != 3:
            raise ValueError("candidate action must be (B,3) or (B,K,3)")
        b, k, _ = action.shape
        vector = state.vector[:, None].expand(-1, k, -1)
        return torch.sigmoid(self.risk_head(torch.cat((vector, action), dim=-1))).reshape(b, k)

    def dynamics_prediction(
        self, state: EncodedComparisonState, action: torch.Tensor,
    ) -> torch.Tensor:
        if self.latent_transition is None:
            raise RuntimeError("latent transition exists only for B6")
        return state.vector + self.latent_transition(torch.cat((state.vector, action), dim=-1))

    def deterministic_action(self, inputs: TractorInputs) -> torch.Tensor:
        state = self.encode(inputs)
        return self.sample_action(state, deterministic=True)[0]
