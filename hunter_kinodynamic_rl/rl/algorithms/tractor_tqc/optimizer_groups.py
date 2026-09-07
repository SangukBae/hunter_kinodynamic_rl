"""Optimizer ownership checks and temporary freeze helpers."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, Mapping

import torch.nn as nn


def assert_disjoint_complete(model: nn.Module, groups: Mapping[str, Iterable[nn.Parameter]]) -> None:
    expected = {id(parameter): name for name, parameter in model.named_parameters() if parameter.requires_grad}
    owners = {}
    duplicates = []
    for group_name, parameters in groups.items():
        for parameter in parameters:
            identifier = id(parameter)
            if identifier in owners:
                duplicates.append((expected.get(identifier, "<external>"), owners[identifier], group_name))
            owners[identifier] = group_name
    missing = sorted(name for identifier, name in expected.items() if identifier not in owners)
    external = sorted(str(identifier) for identifier in owners if identifier not in expected)
    if duplicates or missing or external:
        raise ValueError(
            f"invalid optimizer ownership: duplicates={duplicates}, missing={missing}, external={external}"
        )


@contextmanager
def freeze_modules(modules: Iterable[nn.Module]):
    previous = []
    for module in modules:
        for parameter in module.parameters():
            previous.append((parameter, parameter.requires_grad))
            parameter.requires_grad_(False)
    try:
        yield
    finally:
        for parameter, required in previous:
            parameter.requires_grad_(required)
