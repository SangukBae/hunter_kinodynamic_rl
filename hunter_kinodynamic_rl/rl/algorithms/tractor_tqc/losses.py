"""Numerically explicit losses for TRACTOR-TQC."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def truncated_bellman_target(
    reward: torch.Tensor,
    discount_factor: torch.Tensor,
    terminated: torch.Tensor,
    next_quantiles: torch.Tensor,
    next_log_probability: torch.Tensor,
    entropy_temperature: torch.Tensor,
    top_quantiles_to_drop: int,
) -> torch.Tensor:
    """Build targets while bootstrapping time-limit truncations.

    ``terminated`` alone gates bootstrap. Infrastructure truncations are
    removed later by ``bellman_sample_valid`` and never enter this function.
    """
    if next_quantiles.ndim != 3:
        raise ValueError("next_quantiles must be (B,n_critics,n_quantiles)")
    flattened = next_quantiles.reshape(next_quantiles.shape[0], -1).sort(dim=-1).values
    if not 0 <= top_quantiles_to_drop < flattened.shape[-1]:
        raise ValueError("top_quantiles_to_drop removes every target quantile")
    kept = flattened[:, : flattened.shape[-1] - top_quantiles_to_drop]
    soft_value = kept - entropy_temperature.detach() * next_log_probability
    return reward + discount_factor * (~terminated.bool()).to(reward.dtype) * soft_value


def quantile_huber_loss(
    prediction: torch.Tensor,
    target_samples: torch.Tensor,
    valid: torch.Tensor,
    kappa: float = 1.0,
) -> torch.Tensor:
    if prediction.ndim != 3 or target_samples.ndim != 2:
        raise ValueError("prediction must be (B,C,N) and targets (B,T)")
    if kappa <= 0.0:
        raise ValueError("kappa must be positive")
    delta = target_samples[:, None, None, :] - prediction[:, :, :, None]
    absolute = delta.abs()
    huber = torch.where(absolute <= kappa, 0.5 * delta.square(), kappa * (absolute - 0.5 * kappa))
    n = prediction.shape[-1]
    tau = (torch.arange(n, device=prediction.device, dtype=prediction.dtype) + 0.5) / n
    weight = (tau[None, None, :, None] - (delta.detach() < 0.0).to(prediction.dtype)).abs()
    per_sample = (weight * huber / kappa).mean(dim=(1, 2, 3))
    mask = valid.reshape(-1).bool()
    if not mask.any():
        return prediction.sum() * 0.0
    return per_sample[mask].mean()


def competing_risk_nll(
    hazard: torch.Tensor,
    event_observed: torch.Tensor,
    event_step: torch.Tensor,
    event_cause: torch.Tensor,
    censor_step: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Discrete cause-time likelihood with right censoring."""
    if hazard.ndim < 3:
        raise ValueError("hazard must end in (H,C)")
    h, causes = hazard.shape[-2:]
    flat = hazard.reshape(-1, h, causes)
    observed = event_observed.reshape(-1).bool()
    step = event_step.reshape(-1).long()
    cause = event_cause.reshape(-1).long()
    censor = censor_step.reshape(-1).long()
    mask = valid.reshape(-1).bool()
    if flat.shape[0] != mask.numel():
        raise ValueError("risk labels do not match hazard leading dimensions")
    if not mask.any():
        return hazard.sum() * 0.0
    eps = torch.finfo(hazard.dtype).eps
    total = flat.sum(dim=-1).clamp(0.0, 1.0 - eps)
    log_survival = torch.log1p(-total)
    losses = []
    for row in torch.nonzero(mask, as_tuple=False).flatten().tolist():
        q = int(censor[row])
        if not 0 <= q < h:
            raise ValueError("valid censor_step lies outside hazard horizon")
        if observed[row]:
            event_q, event_c = int(step[row]), int(cause[row])
            if event_q != q or not 0 <= event_c < causes:
                raise ValueError("invalid observed cause-time tuple")
            survival_before = log_survival[row, :event_q].sum()
            event_log_probability = torch.log(flat[row, event_q, event_c].clamp_min(eps))
            losses.append(-(survival_before + event_log_probability))
        else:
            if int(step[row]) != -1 or int(cause[row]) != -1:
                raise ValueError("invalid censored no-event tuple")
            losses.append(-log_survival[row, :q + 1].sum())
    return torch.stack(losses).mean()


def masked_pinball_loss(
    ordered_prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    if ordered_prediction.shape[:-1] != target.shape or target.shape != valid.shape:
        raise ValueError("severity target/mask must match prediction without quantile axis")
    mask = valid.bool()
    if not mask.any():
        return ordered_prediction.sum() * 0.0
    # Gather before arithmetic: invalid storage is allowed to contain NaN.
    ordered_prediction = ordered_prediction[mask]
    target = target[mask]
    count = ordered_prediction.shape[-1]
    tau = (torch.arange(count, device=ordered_prediction.device, dtype=ordered_prediction.dtype) + 0.5) / count
    error = target.unsqueeze(-1) - ordered_prediction
    loss = torch.maximum(tau * error, (tau - 1.0) * error).mean(dim=-1)
    return loss.mean()


def masked_flow_loss(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.shape[-3] != 2:
        raise ValueError("flow tensors must match and use a two-channel axis")
    mask = valid.bool()
    if not mask.any():
        return prediction.sum() * 0.0
    # Move the two-channel axis last, then gather valid pixels before loss.
    prediction = prediction.movedim(-3, -1)[mask]
    target = target.movedim(-3, -1)[mask]
    return F.smooth_l1_loss(prediction, target)


def masked_occupancy_nll(
    probability: torch.Tensor, target: torch.Tensor, valid: torch.Tensor,
) -> torch.Tensor:
    """Class NLL for already-normalized occupancy probabilities."""
    if probability.ndim < 4 or probability.shape[-3] != 4:
        raise ValueError("occupancy probability must use a four-class channel axis")
    expected = probability.shape[:-3] + probability.shape[-2:]
    if tuple(target.shape) != tuple(expected) or target.shape != valid.shape:
        raise ValueError("occupancy target/mask shapes do not match probability")
    mask = valid.bool()
    if not mask.any():
        return probability.sum() * 0.0
    if ((target[mask] < 0) | (target[mask] >= 4)).any():
        raise ValueError("valid occupancy classes must be in [0,3]")
    log_probability = probability.clamp_min(torch.finfo(probability.dtype).tiny).log()
    selected = log_probability.movedim(-3, -1)[mask]
    return F.nll_loss(selected, target[mask].long())


def masked_response_loss(
    prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape or target.shape != valid.shape:
        raise ValueError("vehicle-response target/mask shapes do not match prediction")
    mask = valid.bool()
    if not mask.any():
        return prediction.sum() * 0.0
    return F.smooth_l1_loss(prediction[mask], target[mask])
