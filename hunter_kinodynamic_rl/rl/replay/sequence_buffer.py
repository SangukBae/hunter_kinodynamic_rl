"""Uniform reproducible sampler over reset-prefix exact windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np

from .episode_store import EpisodeStore
from .sequence_index import SequenceIndex, SequenceWindow


@dataclass(frozen=True)
class SequenceSample:
    window: SequenceWindow
    sample_draw_ordinal: int
    burn_in: Dict[str, np.ndarray]
    loss: Dict[str, np.ndarray]


class SequenceBuffer:
    def __init__(self, store: EpisodeStore, index: SequenceIndex, seed: int):
        index.validate_split_isolation()
        if not index.windows:
            raise ValueError("sequence index contains no windows")
        self.store = store
        self.index = index
        self.rng = np.random.default_rng(seed)
        self.draw_ordinal = 0

    def sample(self, batch_size: int, split_id: str = "development") -> Sequence[SequenceSample]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        eligible = [window for window in self.index.windows if window.split_id == split_id]
        if not eligible:
            raise ValueError(f"no indexed windows for split {split_id!r}")
        selected = self.rng.integers(0, len(eligible), size=batch_size)
        result = []
        for selected_index in selected:
            window = eligible[int(selected_index)]
            _, columns = self.store.load(window.episode_id, window.episode_sha256)
            burn = {name: value[window.burn_start:window.loss_start] for name, value in columns.items()}
            loss = {name: value[window.loss_start:window.loss_end] for name, value in columns.items()}
            result.append(SequenceSample(window, self.draw_ordinal, burn, loss))
            self.draw_ordinal += 1
        return tuple(result)

    def state_dict(self) -> dict:
        return {
            "index_sha256": self.index.sha256(),
            "draw_ordinal": self.draw_ordinal,
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("index_sha256") != self.index.sha256():
            raise RuntimeError("sampler state belongs to a different sequence index")
        draw_ordinal = int(state["draw_ordinal"])
        if draw_ordinal < 0:
            raise ValueError("draw_ordinal cannot be negative")
        self.rng.bit_generator.state = state["rng_state"]
        self.draw_ordinal = draw_ordinal
