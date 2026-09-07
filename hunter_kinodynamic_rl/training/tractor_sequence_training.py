"""Sequence-replay adapter and one ordered TRACTOR-TQC update transaction.

This module is deliberately ROS-free.  It is the executable boundary from
``tractor_sequence_v2`` numpy columns to typed ``TractorInputs`` and then to
the value, risk, actor/entropy and EMA transactions owned by ``TractorAgent``.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Mapping, Sequence

import numpy as np
import torch

from hunter_kinodynamic_rl.rl.algorithms.tractor_tqc import (
    TractorAgent, TractorRiskBatch, TractorTrainingBatch,
)
from hunter_kinodynamic_rl.rl.networks.tractor import TractorConfig, TractorInputs
from hunter_kinodynamic_rl.rl.networks.tractor.contracts import CandidateSet
from hunter_kinodynamic_rl.rl.replay import SequenceSample


_INPUT_COLUMN_MAP = {
    "observation": "observation",
    "scan_valid": "scan_valid",
    "motion_delta": "motion_delta_from_previous",
    "motion_valid": "motion_valid",
    "decision_timestamp_sec": "decision_timestamp_ns",
    "previous_intent_valid": "previous_intent_valid",
    "previous_command_published": "previous_command_published",
    "previous_command_valid": "previous_command_valid",
    "vehicle_response_valid": "vehicle_response_valid",
    "localization_covariance": "pose_covariance",
    "localization_valid": "localization_valid",
    "localization_confidence": "localization_confidence",
    "localization_confidence_valid": "localization_confidence_valid",
    "sensor_freshness_sec": "sensor_freshness_sec",
    "sensor_freshness_valid": "sensor_freshness_valid",
    "scene_reset": "scene_reset",
    "response_reset": "response_reset",
}
_BOOL_INPUTS = {
    "scan_valid", "motion_valid", "previous_intent_valid", "previous_command_valid",
    "vehicle_response_valid", "localization_valid", "localization_confidence_valid",
    "sensor_freshness_valid", "scene_reset", "response_reset",
}
_SCALAR_INPUTS = {
    "decision_timestamp_sec", "previous_intent_valid", "previous_command_valid",
    "localization_valid", "localization_confidence", "localization_confidence_valid",
    "sensor_freshness_sec", "sensor_freshness_valid", "scene_reset", "response_reset",
}


def _tensor(value, *, device: torch.device, boolean: bool = False) -> torch.Tensor:
    return torch.as_tensor(
        np.asarray(value), device=device, dtype=torch.bool if boolean else torch.float32,
    )


def _row_inputs(
    columns: Mapping[str, np.ndarray],
    row: int,
    config: TractorConfig,
    device: torch.device,
    *,
    prefix: str = "",
    previous_scene_hidden: torch.Tensor | None = None,
    previous_scene_hidden_valid: torch.Tensor | None = None,
    previous_response_hidden: torch.Tensor | None = None,
    previous_response_hidden_valid: torch.Tensor | None = None,
) -> TractorInputs:
    values = {}
    for input_name, column_name in _INPUT_COLUMN_MAP.items():
        key = f"{prefix}{column_name}"
        if key not in columns:
            raise KeyError(f"sequence row is missing required input column {key!r}")
        value = np.asarray(columns[key])[row]
        if input_name == "decision_timestamp_sec":
            value = np.asarray(value, dtype=np.float64) * 1e-9
        tensor = _tensor(value, device=device, boolean=input_name in _BOOL_INPUTS)
        if input_name in _SCALAR_INPUTS:
            tensor = tensor.reshape(1, 1)
        else:
            tensor = tensor.unsqueeze(0)
        values[input_name] = tensor
    result = TractorInputs(
        **values,
        previous_scene_hidden=previous_scene_hidden,
        previous_scene_hidden_valid=previous_scene_hidden_valid,
        previous_response_hidden=previous_response_hidden,
        previous_response_hidden_valid=previous_response_hidden_valid,
    )
    result.validate(config)
    return result


def _concat_inputs(items: Sequence[TractorInputs]) -> TractorInputs:
    if not items:
        raise ValueError("cannot concatenate an empty TractorInputs sequence")
    payload = {}
    for field in fields(TractorInputs):
        values = [getattr(item, field.name) for item in items]
        if all(value is None for value in values):
            payload[field.name] = None
        elif any(value is None for value in values):
            template = next(value for value in values if value is not None)
            values = [torch.zeros_like(template) if value is None else value for value in values]
            payload[field.name] = torch.cat(values, dim=0)
        else:
            payload[field.name] = torch.cat(values, dim=0)
    return TractorInputs(**payload)


class TractorSequenceBatchAssembler:
    """Reconstruct current-weight online/target recurrence from reset prefixes."""

    def __init__(self, agent: TractorAgent):
        self.agent = agent
        self.config = agent.model_config
        self.device = agent.device

    def _episode_columns(self, sample: SequenceSample) -> Mapping[str, np.ndarray]:
        _header, columns = self.agent_sequence_store.load(
            sample.window.episode_id, sample.window.episode_sha256,
        )
        return columns

    @property
    def agent_sequence_store(self):
        if not hasattr(self, "_store"):
            raise RuntimeError("assembler is not bound to an EpisodeStore")
        return self._store

    def bind_store(self, store) -> "TractorSequenceBatchAssembler":
        self._store = store
        return self

    def _recurrent_inputs(self, sample: SequenceSample, *, target: bool) -> dict[int, TractorInputs]:
        columns = self._episode_columns(sample)
        module = self.agent.target if target else self.agent.online
        scene_hidden = response_hidden = None
        scene_valid = response_valid = None
        cache: dict[int, TractorInputs] = {}
        stop = min(len(np.asarray(columns["observation"])), sample.window.loss_end + 1)
        with torch.no_grad():
            for row in range(sample.window.burn_start, stop):
                inputs = _row_inputs(
                    columns, row, self.config, self.device,
                    previous_scene_hidden=scene_hidden,
                    previous_scene_hidden_valid=scene_valid,
                    previous_response_hidden=response_hidden,
                    previous_response_hidden_valid=response_valid,
                )
                cache[row] = inputs
                belief, _context = module.encode(inputs)
                scene_hidden = belief.next_scene_hidden.detach()
                scene_valid = belief.next_scene_hidden_valid.detach()
                response_hidden = belief.next_response_hidden.detach()
                response_valid = belief.next_response_hidden_valid.detach()
        return cache

    def current_inputs(self, samples: Sequence[SequenceSample]) -> TractorInputs:
        rows = []
        for sample in samples:
            cache = self._recurrent_inputs(sample, target=False)
            rows.extend(cache[row] for row in range(sample.window.loss_start, sample.window.loss_end))
        current = _concat_inputs(rows)
        current.validate(self.config)
        return current

    def training_batch(self, samples: Sequence[SequenceSample]) -> TractorTrainingBatch:
        current_rows, next_rows = [], []
        action, reward, discount, terminated, truncated, bellman = [], [], [], [], [], []
        for sample in samples:
            columns = self._episode_columns(sample)
            online_cache = self._recurrent_inputs(sample, target=False)
            target_cache = self._recurrent_inputs(sample, target=True)
            for row in range(sample.window.loss_start, sample.window.loss_end):
                current_rows.append(online_cache[row])
                if row + 1 in target_cache:
                    next_rows.append(target_cache[row + 1])
                elif bool(np.asarray(columns["terminated"])[row]) or not bool(
                    np.asarray(columns["bellman_sample_valid"])[row]
                ):
                    # This tensor is contract-checked but never evaluated by
                    # the reward-only/invalid Bellman branch.
                    next_rows.append(target_cache[row])
                else:
                    # Time-limit truncations must bootstrap from the actual
                    # post-step observation, never a fabricated duplicate.
                    with torch.no_grad():
                        preceding_belief, _ = self.agent.target.encode(target_cache[row])
                    next_rows.append(_row_inputs(
                        columns, row, self.config, self.device, prefix="next_",
                        previous_scene_hidden=preceding_belief.next_scene_hidden.detach(),
                        previous_scene_hidden_valid=preceding_belief.next_scene_hidden_valid.detach(),
                        previous_response_hidden=preceding_belief.next_response_hidden.detach(),
                        previous_response_hidden_valid=preceding_belief.next_response_hidden_valid.detach(),
                    ))
                action.append(np.asarray(columns["action_normalized_requested"])[row])
                reward.append(np.asarray(columns["reward"])[row])
                discount.append(np.asarray(columns["discount_factor"])[row])
                terminated.append(np.asarray(columns["terminated"])[row])
                truncated.append(np.asarray(columns["truncated"])[row])
                bellman.append(np.asarray(columns["bellman_sample_valid"])[row])
        batch = TractorTrainingBatch(
            current=_concat_inputs(current_rows), next=_concat_inputs(next_rows),
            action_normalized_requested=_tensor(action, device=self.device),
            reward=_tensor(reward, device=self.device).reshape(-1, 1),
            discount_factor=_tensor(discount, device=self.device).reshape(-1, 1),
            terminated=_tensor(terminated, device=self.device, boolean=True).reshape(-1, 1),
            truncated=_tensor(truncated, device=self.device, boolean=True).reshape(-1, 1),
            bellman_sample_valid=_tensor(bellman, device=self.device, boolean=True).reshape(-1, 1),
        )
        batch.validate(self.config)
        return batch

    def risk_batch(self, samples: Sequence[SequenceSample]) -> TractorRiskBatch:
        current = self.current_inputs(samples)
        names = (
            "candidate_actions_normalized", "candidate_present", "candidate_is_stop",
            "candidate_model_valid", "candidate_event_observed", "candidate_event_step",
            "candidate_event_cause", "candidate_censor_step", "candidate_event_label_valid",
            "candidate_clearance_m", "candidate_clearance_valid",
            "candidate_stopping_margin_m", "candidate_stopping_margin_valid",
        )
        gathered = {name: [] for name in names}
        for sample in samples:
            columns = self._episode_columns(sample)
            missing = sorted(set(names) - set(columns))
            if missing:
                raise ValueError(f"TRACTOR risk transaction requires complete candidate sidecars: {missing}")
            selection = slice(sample.window.loss_start, sample.window.loss_end)
            for name in names:
                gathered[name].append(np.asarray(columns[name])[selection])
        joined = {name: np.concatenate(values, axis=0) for name, values in gathered.items()}
        present = joined["candidate_present"].astype(bool) & joined["candidate_model_valid"].astype(bool)
        candidates = CandidateSet(
            normalized_actions=_tensor(joined["candidate_actions_normalized"], device=self.device),
            present=_tensor(present, device=self.device, boolean=True),
            is_stop=_tensor(joined["candidate_is_stop"], device=self.device, boolean=True),
            source="tractor_sequence_v2",
        )
        batch = TractorRiskBatch(
            current=current, candidates=candidates,
            event_observed=_tensor(joined["candidate_event_observed"], device=self.device, boolean=True),
            event_step=torch.as_tensor(joined["candidate_event_step"], device=self.device, dtype=torch.long),
            event_cause=torch.as_tensor(joined["candidate_event_cause"], device=self.device, dtype=torch.long),
            censor_step=torch.as_tensor(joined["candidate_censor_step"], device=self.device, dtype=torch.long),
            event_label_valid=_tensor(joined["candidate_event_label_valid"], device=self.device, boolean=True),
            clearance_m=_tensor(joined["candidate_clearance_m"], device=self.device),
            clearance_valid=_tensor(joined["candidate_clearance_valid"], device=self.device, boolean=True),
            stopping_margin_m=_tensor(joined["candidate_stopping_margin_m"], device=self.device),
            stopping_margin_valid=_tensor(
                joined["candidate_stopping_margin_valid"], device=self.device, boolean=True,
            ),
        )
        batch.validate(self.config)
        return batch


class TractorSequenceTrainer:
    """Execute one reproducible update in the frozen transaction order."""

    def __init__(self, agent: TractorAgent, sequence_buffer, loss_weights: Mapping[str, float]):
        self.agent = agent
        self.sequence_buffer = sequence_buffer
        self.loss_weights = dict(loss_weights)
        self.assembler = TractorSequenceBatchAssembler(agent).bind_store(sequence_buffer.store)
        self.risk_updates = 0

    def update(self, batch_size: int) -> dict[str, float]:
        samples = self.sequence_buffer.sample(batch_size, split_id="development")
        value = self.agent.critic_step(self.assembler.training_batch(samples), update_target=False)
        metrics = dict(value)
        if value.get("update/applied") != 1.0:
            raise RuntimeError("TRACTOR value update rejected; stopping before checkpoint publication")
        risk = self.agent.risk_step(
            self.assembler.risk_batch(samples),
            clearance_weight=float(self.loss_weights.get("clearance", 0.25)),
            stopping_weight=float(self.loss_weights.get("stopping_margin", 0.25)),
        )
        metrics.update(risk)
        if risk.get("risk/update_applied") != 1.0:
            raise RuntimeError("TRACTOR risk update rejected; stopping before checkpoint publication")
        self.risk_updates += 1
        actor = self.agent.actor_step(self.assembler.current_inputs(samples), update_target=False)
        metrics.update(actor)
        self.agent.update_target()
        metrics["update/transaction_applied"] = 1.0
        metrics["replay/sample_draw_ordinal"] = float(self.sequence_buffer.draw_ordinal)
        return metrics
