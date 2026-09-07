"""Complete parameter-and-buffer EMA for TRACTOR target networks."""

from __future__ import annotations

import torch
import torch.nn as nn


@torch.no_grad()
def ema_update(target: nn.Module, online: nn.Module, tau: float) -> None:
    if not 0.0 < tau <= 1.0:
        raise ValueError("tau must be in (0,1]")
    target_parameters = dict(target.named_parameters())
    online_parameters = dict(online.named_parameters())
    target_buffers = dict(target.named_buffers())
    online_buffers = dict(online.named_buffers())
    if not target_parameters.keys() <= online_parameters.keys() or not target_buffers.keys() <= online_buffers.keys():
        raise RuntimeError("target value path is not a named subset of the online model")
    for name, target_value in target_parameters.items():
        target_value.lerp_(online_parameters[name], tau)
    for name, target_value in target_buffers.items():
        source = online_buffers[name]
        if torch.is_floating_point(target_value):
            target_value.lerp_(source, tau)
        else:
            target_value.copy_(source)
