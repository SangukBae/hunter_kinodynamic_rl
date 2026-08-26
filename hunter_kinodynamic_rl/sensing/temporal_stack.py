"""Frame-stacking temporal observation buffer -- section 14. Pure numpy, no
ROS, so it is swappable behind :class:`TemporalEncoder` implementations
(stack/MLP, 1D CNN, GRU, Transformer -- only the NETWORK differs; the
buffering here stays the same regardless)."""

from __future__ import annotations

from collections import deque
from typing import List

import numpy as np


class FrameStack:
    """Keeps the last ``history_len`` LiDAR frames, warm-starting a new
    episode by REPEATING the first frame (matches drl_agent's
    ObsTimeContext warm-start convention -- avoids an artificial "nothing
    seen yet" transient at episode start)."""

    def __init__(self, frame_dim: int, history_len: int):
        if history_len < 1:
            raise ValueError("history_len must be >= 1")
        self.frame_dim = frame_dim
        self.history_len = history_len
        self._frames: deque = deque(maxlen=history_len)

    def reset(self, first_frame: np.ndarray) -> None:
        frame = np.asarray(first_frame, dtype=np.float32).reshape(self.frame_dim)
        self._frames.clear()
        for _ in range(self.history_len):
            self._frames.append(frame.copy())

    def push(self, frame: np.ndarray) -> None:
        if not self._frames:
            self.reset(frame)
            return
        self._frames.append(np.asarray(frame, dtype=np.float32).reshape(self.frame_dim))

    def stacked(self) -> np.ndarray:
        """Current frame FIRST, oldest last -- matches
        observation_time_context.yaml's "current frame first" convention."""
        if not self._frames:
            raise RuntimeError("FrameStack.reset() must be called before stacked()")
        ordered: List[np.ndarray] = list(reversed(self._frames))
        return np.concatenate(ordered, axis=0)

    @property
    def stacked_dim(self) -> int:
        return self.frame_dim * self.history_len
