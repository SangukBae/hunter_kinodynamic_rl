"""Shared immutable sequence-replay transaction for B1--B8."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import torch

from hunter_kinodynamic_rl.rl.algorithms.comparison_baselines import ComparisonAgent
from hunter_kinodynamic_rl.rl.algorithms.tractor_tqc import TractorRiskBatch, TractorTrainingBatch
from hunter_kinodynamic_rl.rl.networks.tractor.contracts import CandidateSet, TractorInputs
from hunter_kinodynamic_rl.rl.replay import SequenceSample
from hunter_kinodynamic_rl.training.tractor_sequence_training import (
    _concat_inputs, _row_inputs, _tensor,
)


class ComparisonSequenceBatchAssembler:
    """Build baseline tensors without giving a baseline privileged side-channel input."""

    def __init__(self, agent: ComparisonAgent):
        self.agent = agent
        self.config = agent.input_config
        self.device = agent.device

    def bind_store(self, store) -> "ComparisonSequenceBatchAssembler":
        self.store = store
        return self

    def _columns(self, sample: SequenceSample):
        if not hasattr(self, "store"):
            raise RuntimeError("comparison assembler is not bound to an EpisodeStore")
        return self.store.load(sample.window.episode_id, sample.window.episode_sha256)[1]

    def current_inputs(self, samples: Sequence[SequenceSample]) -> TractorInputs:
        rows = []
        for sample in samples:
            columns = self._columns(sample)
            rows.extend(
                _row_inputs(columns, row, self.config, self.device)
                for row in range(sample.window.loss_start, sample.window.loss_end)
            )
        result = _concat_inputs(rows)
        result.validate(self.config)
        return result

    def training_batch(self, samples: Sequence[SequenceSample]) -> TractorTrainingBatch:
        current, next_rows = [], []
        action, reward, discount, terminated, truncated, bellman = [], [], [], [], [], []
        for sample in samples:
            columns = self._columns(sample)
            row_count = len(np.asarray(columns["observation"]))
            for row in range(sample.window.loss_start, sample.window.loss_end):
                current.append(_row_inputs(columns, row, self.config, self.device))
                if row + 1 < row_count:
                    next_rows.append(_row_inputs(columns, row + 1, self.config, self.device))
                elif bool(np.asarray(columns["terminated"])[row]) or not bool(
                    np.asarray(columns["bellman_sample_valid"])[row]
                ):
                    next_rows.append(_row_inputs(columns, row, self.config, self.device))
                else:
                    next_rows.append(_row_inputs(
                        columns, row, self.config, self.device, prefix="next_",
                    ))
                action.append(np.asarray(columns["action_normalized_requested"])[row])
                reward.append(np.asarray(columns["reward"])[row])
                discount.append(np.asarray(columns["discount_factor"])[row])
                terminated.append(np.asarray(columns["terminated"])[row])
                truncated.append(np.asarray(columns["truncated"])[row])
                bellman.append(np.asarray(columns["bellman_sample_valid"])[row])
        batch = TractorTrainingBatch(
            current=_concat_inputs(current), next=_concat_inputs(next_rows),
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
            columns = self._columns(sample)
            missing = sorted(set(names) - set(columns))
            if missing:
                raise ValueError(f"B8 requires complete candidate sidecars: {missing}")
            selection = slice(sample.window.loss_start, sample.window.loss_end)
            for name in names:
                gathered[name].append(np.asarray(columns[name])[selection])
        joined = {name: np.concatenate(values, axis=0) for name, values in gathered.items()}
        present = joined["candidate_present"].astype(bool) & joined["candidate_model_valid"].astype(bool)
        batch = TractorRiskBatch(
            current=current,
            candidates=CandidateSet(
                normalized_actions=_tensor(joined["candidate_actions_normalized"], device=self.device),
                present=_tensor(present, device=self.device, boolean=True),
                is_stop=_tensor(joined["candidate_is_stop"], device=self.device, boolean=True),
                source="tractor_sequence_v2",
            ),
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


class ComparisonSequenceTrainer:
    def __init__(self, agent: ComparisonAgent, sequence_buffer):
        self.agent = agent
        self.sequence_buffer = sequence_buffer
        self.assembler = ComparisonSequenceBatchAssembler(agent).bind_store(sequence_buffer.store)
        self.risk_updates = 0

    def update(self, batch_size: int) -> dict[str, float]:
        samples = self.sequence_buffer.sample(batch_size, split_id="development")
        metrics = self.agent.critic_step(
            self.assembler.training_batch(samples), update_target=False,
        )
        if metrics.get("update/applied") != 1.0:
            raise RuntimeError("comparison value update rejected")
        if self.agent.method_id == "B8":
            risk = self.agent.risk_step(self.assembler.risk_batch(samples))
            metrics.update(risk)
            if risk.get("risk/update_applied") != 1.0:
                raise RuntimeError("B8 risk update rejected")
            self.risk_updates += 1
        metrics.update(self.agent.actor_step(
            self.assembler.current_inputs(samples), update_target=False,
        ))
        self.agent.update_target()
        metrics["update/transaction_applied"] = 1.0
        metrics["replay/sample_draw_ordinal"] = float(self.sequence_buffer.draw_ordinal)
        return metrics
