"""Learned feasibility predictor (plan section 9.7) -- a SEPARATE, strictly
OPTIONAL upgrade path over ``hierarchy/feasibility.py``'s heuristic features,
trained from REALIZED :class:`~hunter_kinodynamic_rl.navigation.hierarchy.subgoal_manager.SubgoalResult`
outcomes once the heuristic features have stabilized (plan: "Heuristic
feature가 안정화된 뒤 실제 option 결과를 label로 다음 predictor를 학습할 수
있다"). No shipped profile enables this by default -- it exists as a
trainable component a future experiment opts into, never a hidden
dependency of the base Phase 5 pipeline.

::

    input:  local map crop (small, centered on the candidate endpoint) +
            candidate [radius, angle] + vehicle speed + topology context
            (e.g. nearby dead-end density / mean node risk)
    output: success probability + expected local steps + expected risk

Labels are always REALIZED option outcomes (``SubgoalResult.status ==
REACHED``, ``local_steps``, ``mean_predicted_risk``) -- this predictor never
consumes simulator ground-truth success likelihood as an input OR a label
source (plan's information-boundary requirement applies here exactly as it
does to the Global observation itself).

Requires torch -- import only after ``pytest.importorskip("torch")``,
mirroring ``networks.py``'s own gating convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from hunter_kinodynamic_rl.config.schema import ConfigError

FEASIBILITY_PREDICTOR_SCALAR_NAMES = ("radius_norm", "angle_norm", "speed_norm", "topology_context")
N_PREDICTOR_SCALARS = len(FEASIBILITY_PREDICTOR_SCALAR_NAMES)


@dataclass(frozen=True)
class FeasibilityPredictorConfig:
    map_crop_channels: int = 5
    map_crop_size_cells: int = 32
    cnn_channels: List[int] = field(default_factory=lambda: [8, 16])
    feature_dim: int = 64
    learning_rate: float = 1e-3
    expected_steps_norm: float = 150.0

    def validate(self) -> None:
        if self.map_crop_channels <= 0:
            raise ConfigError("feasibility_predictor.map_crop_channels must be > 0")
        if self.map_crop_size_cells <= 0:
            raise ConfigError("feasibility_predictor.map_crop_size_cells must be > 0")
        if not self.cnn_channels or any(c <= 0 for c in self.cnn_channels):
            raise ConfigError("feasibility_predictor.cnn_channels must be a non-empty list of positive ints")
        if self.feature_dim <= 0:
            raise ConfigError("feasibility_predictor.feature_dim must be > 0")
        if self.learning_rate <= 0.0:
            raise ConfigError("feasibility_predictor.learning_rate must be > 0")
        if self.expected_steps_norm <= 0.0:
            raise ConfigError("feasibility_predictor.expected_steps_norm must be > 0")


@dataclass(frozen=True)
class FeasibilityLabel:
    """One REALIZED option outcome -- built from a terminated
    ``SubgoalResult``, never from simulator ground truth."""

    success: bool
    local_steps: int
    mean_risk: float


class FeasibilityPredictorNet(nn.Module):
    def __init__(self, config: FeasibilityPredictorConfig) -> None:
        super().__init__()
        self.config = config
        layers = []
        c_in = config.map_crop_channels
        for c_out in config.cnn_channels:
            layers.append(nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1))
            layers.append(nn.ReLU(inplace=True))
            c_in = c_out
        self.conv = nn.Sequential(*layers)
        with torch.no_grad():
            dummy = torch.zeros(1, config.map_crop_channels, config.map_crop_size_cells, config.map_crop_size_cells)
            flat_dim = int(self.conv(dummy).flatten(1).shape[1])
        self.map_fc = nn.Linear(flat_dim, config.feature_dim)
        self.scalar_fc = nn.Sequential(
            nn.Linear(N_PREDICTOR_SCALARS, config.feature_dim), nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Linear(2 * config.feature_dim, config.feature_dim), nn.ReLU(inplace=True),
        )
        self.success_head = nn.Linear(config.feature_dim, 1)
        self.steps_head = nn.Linear(config.feature_dim, 1)
        self.risk_head = nn.Linear(config.feature_dim, 1)

    def forward(self, map_crop: torch.Tensor, scalars: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``map_crop``: ``(B, C, H, W)``; ``scalars``: ``(B, N_PREDICTOR_SCALARS)``.
        Returns ``(success_logit, expected_steps_norm, expected_risk)`` each
        ``(B, 1)`` -- ``success_logit`` is a raw logit (apply ``sigmoid``
        for a probability), ``expected_risk`` is already sigmoid-squashed
        into ``[0, 1]``, ``expected_steps_norm`` is a nonnegative
        (``softplus``) normalized step count."""
        map_feat = F.relu(self.map_fc(self.conv(map_crop).flatten(1)))
        scalar_feat = self.scalar_fc(scalars)
        fused = self.fuse(torch.cat([map_feat, scalar_feat], dim=1))
        success_logit = self.success_head(fused)
        expected_steps_norm = F.softplus(self.steps_head(fused))
        expected_risk = torch.sigmoid(self.risk_head(fused))
        return success_logit, expected_steps_norm, expected_risk


class FeasibilityPredictor:
    def __init__(self, config: FeasibilityPredictorConfig, device: Optional[str] = None) -> None:
        config.validate()
        self.config = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.net = FeasibilityPredictorNet(config).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config.learning_rate)
        self.training_steps = 0

    def predict(self, map_crop, scalars) -> Tuple[float, float, float]:
        """Single-sample inference. Returns
        ``(success_probability, expected_local_steps, expected_risk)`` --
        ``expected_local_steps`` is de-normalized back to raw step units."""
        map_t = torch.as_tensor(map_crop, dtype=torch.float32, device=self.device).unsqueeze(0)
        scalar_t = torch.as_tensor(scalars, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            success_logit, steps_norm, risk = self.net(map_t, scalar_t)
        success_prob = float(torch.sigmoid(success_logit).item())
        expected_steps = float(steps_norm.item()) * self.config.expected_steps_norm
        expected_risk = float(risk.item())
        return success_prob, expected_steps, expected_risk

    def train_step(self, map_crop_batch, scalar_batch, labels: List[FeasibilityLabel]) -> Dict[str, float]:
        map_t = torch.as_tensor(map_crop_batch, dtype=torch.float32, device=self.device)
        scalar_t = torch.as_tensor(scalar_batch, dtype=torch.float32, device=self.device)
        success_target = torch.as_tensor(
            [[1.0 if l.success else 0.0] for l in labels], dtype=torch.float32, device=self.device,
        )
        steps_target = torch.as_tensor(
            [[l.local_steps / self.config.expected_steps_norm] for l in labels], dtype=torch.float32, device=self.device,
        )
        risk_target = torch.as_tensor(
            [[float(min(max(l.mean_risk, 0.0), 1.0))] for l in labels], dtype=torch.float32, device=self.device,
        )

        success_logit, steps_pred, risk_pred = self.net(map_t, scalar_t)
        success_loss = F.binary_cross_entropy_with_logits(success_logit, success_target)
        steps_loss = F.smooth_l1_loss(steps_pred, steps_target)
        risk_loss = F.smooth_l1_loss(risk_pred, risk_target)
        loss = success_loss + steps_loss + risk_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.training_steps += 1
        return {
            "loss": float(loss.item()), "success_loss": float(success_loss.item()),
            "steps_loss": float(steps_loss.item()), "risk_loss": float(risk_loss.item()),
        }

    def checkpoint_components(self) -> Dict[str, object]:
        return {"net": self.net, "optimizer": self.optimizer}
