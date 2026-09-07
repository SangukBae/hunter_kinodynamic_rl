"""Post-aggregate risk calibration artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import torch


@dataclass(frozen=True)
class PlattCalibration:
    scale: float
    bias: float
    source_checkpoint_sha256: str
    split_sha256: str

    def apply(self, probability: torch.Tensor) -> torch.Tensor:
        eps = torch.finfo(probability.dtype).eps
        p = probability.clamp(eps, 1.0 - eps)
        logits = torch.log(p) - torch.log1p(-p)
        return torch.sigmoid(self.scale * logits + self.bias)

    def sha256(self) -> str:
        data = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(data).hexdigest()


def fit_platt(
    probability: torch.Tensor,
    target: torch.Tensor,
    *,
    source_checkpoint_sha256: str,
    split_sha256: str,
    max_iter: int = 100,
) -> PlattCalibration:
    if probability.numel() == 0 or probability.shape != target.shape:
        raise ValueError("calibration probability/target must be non-empty and shape-matched")
    p = probability.detach().double().clamp(1e-8, 1.0 - 1e-8)
    y = target.detach().double()
    scale = torch.ones((), dtype=torch.double, requires_grad=True)
    bias = torch.zeros((), dtype=torch.double, requires_grad=True)
    optimizer = torch.optim.LBFGS((scale, bias), max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        logits = scale * (torch.log(p) - torch.log1p(-p)) + bias
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        return loss

    optimizer.step(closure)
    return PlattCalibration(
        float(scale.detach()), float(bias.detach()), source_checkpoint_sha256, split_sha256
    )
