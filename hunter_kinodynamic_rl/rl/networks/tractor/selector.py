"""Fixed conservative aggregation, calibration and candidate selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .calibration import PlattCalibration
from .contracts import CandidateSet, ReturnRiskOutput, SelectionOutput, TractorConfig


@dataclass(frozen=True)
class SelectorConfig:
    tail_fraction: float = 0.2
    max_event_probability: float = 0.35
    minimum_clearance_m: float = 0.15
    risk_penalty: float = 1.0
    smoothness_penalty: float = 0.05
    probability_beta: float = 1.0
    margin_beta: float = 1.0

    def validate(self) -> None:
        if not (0.0 < self.tail_fraction <= 1.0):
            raise ValueError("tail_fraction must be in (0,1]")
        if not (0.0 <= self.max_event_probability <= 1.0):
            raise ValueError("max_event_probability must be in [0,1]")


def cumulative_event_probability(hazard: torch.Tensor) -> torch.Tensor:
    total_hazard = hazard.sum(dim=-1).clamp(0.0, 1.0 - 1e-7)
    survival = torch.cumprod(1.0 - total_hazard, dim=-1)
    return 1.0 - survival[..., -1]


class CandidateSelector:
    def __init__(self, model_config: TractorConfig, selector_config: SelectorConfig):
        self.model_cfg = model_config
        self.cfg = selector_config
        self.cfg.validate()

    def __call__(
        self,
        candidates: CandidateSet,
        output: ReturnRiskOutput,
        previous_action: torch.Tensor,
        previous_action_valid: torch.Tensor,
        calibration: Optional[PlattCalibration],
    ) -> SelectionOutput:
        # risk tensors: B,Qm,Qr,K,H,C -> B,S,K
        p_samples = cumulative_event_probability(output.hazard)
        b, qm, qr, k = p_samples.shape
        p_samples = p_samples.reshape(b, qm * qr, k)
        p_mean = p_samples.mean(dim=1)
        p_std = p_samples.std(dim=1, unbiased=True) if p_samples.shape[1] > 1 else torch.zeros_like(p_mean)
        calibrated = calibration.apply(p_mean) if calibration is not None else p_mean
        p_ucb = (calibrated + self.cfg.probability_beta * p_std).clamp(0.0, 1.0)

        clear_samples = output.clearance_quantiles[..., 0].reshape(b, qm * qr, k, -1)
        stop_samples = output.stopping_quantiles[..., 0].reshape(b, qm * qr, k, -1)
        clear_point = clear_samples.amin(dim=-1)
        stop_point = stop_samples.amin(dim=-1)
        clear_mean, stop_mean = clear_point.mean(1), stop_point.mean(1)
        if clear_point.shape[1] > 1:
            clear_std = clear_point.std(1, unbiased=True)
            stop_std = stop_point.std(1, unbiased=True)
        else:
            clear_std, stop_std = torch.zeros_like(clear_mean), torch.zeros_like(stop_mean)
        clear_lcb = clear_mean - self.cfg.margin_beta * clear_std
        stop_lcb = stop_mean - self.cfg.margin_beta * stop_std

        quantiles = output.return_quantiles.reshape(b, k, -1).sort(dim=-1).values
        n_tail = max(1, int(self.cfg.tail_fraction * quantiles.shape[-1] + 0.999999))
        tail_value = quantiles[..., :n_tail].mean(dim=-1)
        delta = candidates.normalized_actions - previous_action[:, None, :]
        smooth = delta.square().sum(-1) * previous_action_valid.to(delta.dtype)
        score = tail_value - self.cfg.risk_penalty * p_ucb - self.cfg.smoothness_penalty * smooth
        feasible = (
            candidates.present & output.candidate_valid
            & (p_ucb <= self.cfg.max_event_probability)
            & (clear_lcb >= self.cfg.minimum_clearance_m)
            & (stop_lcb >= 0.0)
        )
        masked_score = torch.where(feasible, score, torch.full_like(score, -torch.inf))
        selected = masked_score.argmax(dim=-1)
        any_feasible = feasible.any(dim=-1)
        selected = torch.where(any_feasible, selected, torch.full_like(selected, -1))
        gather_index = selected.clamp_min(0).view(b, 1, 1).expand(-1, 1, 3)
        action = candidates.normalized_actions.gather(1, gather_index).squeeze(1)
        action = torch.where(any_feasible[:, None], action, torch.zeros_like(action))
        reason = tuple("selected" if bool(ok) else "no_feasible_candidate" for ok in any_feasible.tolist())
        return SelectionOutput(selected, action, feasible, score, p_ucb, clear_lcb, stop_lcb, reason)
