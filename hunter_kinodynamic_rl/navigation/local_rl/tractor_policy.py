"""One-snapshot/one-recurrence inference wrapper for TRACTOR-TQC."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import torch

from hunter_kinodynamic_rl.rl.networks.tractor import TractorInputs, TractorTQC
from hunter_kinodynamic_rl.rl.networks.tractor.calibration import PlattCalibration
from hunter_kinodynamic_rl.rl.networks.tractor.contracts import CandidateSet, ReturnRiskOutput, SelectionOutput
from hunter_kinodynamic_rl.rl.networks.tractor.selector import CandidateSelector, SelectorConfig


@dataclass(frozen=True)
class PolicyDecision:
    snapshot_id: int
    candidates: CandidateSet
    model_output: ReturnRiskOutput
    selection: SelectionOutput
    publish_allowed: bool
    fallback_reason: str | None


class TractorPolicy:
    """Own recurrent state while leaving command execution/guard downstream."""

    def __init__(
        self,
        model: TractorTQC,
        selector_config: SelectorConfig,
        calibration: Optional[PlattCalibration],
        *,
        deployment: bool = True,
    ):
        if deployment and calibration is None:
            raise ValueError("deployment TRACTOR policy requires an approved calibration artifact")
        self.model = model.eval()
        self.selector = CandidateSelector(model.config, selector_config)
        self.calibration = calibration
        self.deployment = deployment
        self.reset()

    def reset(self) -> None:
        self._last_snapshot_id: int | None = None
        self._scene_hidden = None
        self._scene_hidden_valid = None
        self._response_hidden = None
        self._response_hidden_valid = None
        self._previous_action = None
        self._previous_action_valid = None

    def step(self, snapshot_id: int, inputs: TractorInputs) -> PolicyDecision:
        if self._last_snapshot_id is not None and snapshot_id <= self._last_snapshot_id:
            raise RuntimeError("snapshot ids must increase strictly; recurrence was not advanced")
        bound = replace(
            inputs,
            previous_scene_hidden=self._scene_hidden,
            previous_scene_hidden_valid=self._scene_hidden_valid,
            previous_response_hidden=self._response_hidden,
            previous_response_hidden_valid=self._response_hidden_valid,
        )
        with torch.inference_mode():
            belief, context = self.model.encode(bound)
            candidates = self.model.propose(belief)
            output = self.model.score(belief, context, candidates)
            batch = inputs.observation.shape[0]
            if batch != 1:
                raise ValueError("runtime TractorPolicy accepts exactly one coherent snapshot")
            previous = self._previous_action
            previous_valid = self._previous_action_valid
            if previous is None:
                previous = inputs.observation.new_zeros((1, 3))
                previous_valid = torch.zeros((1, 1), dtype=torch.bool, device=previous.device)
            selection = self.selector(candidates, output, previous, previous_valid, self.calibration)

        # Commit recurrent state exactly once after the complete decision.
        self._scene_hidden = belief.next_scene_hidden.detach()
        self._scene_hidden_valid = belief.next_scene_hidden_valid.detach()
        self._response_hidden = belief.next_response_hidden.detach()
        self._response_hidden_valid = belief.next_response_hidden_valid.detach()
        self._last_snapshot_id = int(snapshot_id)
        selected = int(selection.selected_index[0].item())
        model_healthy = bool(belief.health.model_valid[0, 0].item())
        publish_allowed = model_healthy and selected >= 0
        fallback_reason = None
        if not model_healthy:
            fallback_reason = "invalid_belief"
        elif selected < 0:
            fallback_reason = "no_feasible_candidate"
        if publish_allowed:
            self._previous_action = selection.selected_action.detach()
            self._previous_action_valid = torch.ones(
                (1, 1), dtype=torch.bool, device=selection.selected_action.device
            )
        else:
            self._previous_action_valid = torch.zeros(
                (1, 1), dtype=torch.bool, device=selection.selected_action.device
            )
        return PolicyDecision(
            int(snapshot_id), candidates, output, selection, publish_allowed, fallback_reason
        )
