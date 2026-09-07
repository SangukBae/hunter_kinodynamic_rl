"""Deterministic reset-prefix sequence index and leakage checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import numpy as np

from .episode_store import EpisodeStore, sha256_file
from .sequence_schema import SCHEMA_ID, SEQUENCE_CONTRACT


@dataclass(frozen=True)
class SequenceWindow:
    episode_id: str
    group_id: str
    split_id: str
    reset_epoch: int
    burn_start: int
    loss_start: int
    loss_end: int
    episode_sha256: str


@dataclass(frozen=True)
class SequenceIndex:
    schema_id: str
    sequence_contract: str
    loss_window: int
    windows: tuple[SequenceWindow, ...]

    @classmethod
    def build(cls, store: EpisodeStore, loss_window: int = 16) -> "SequenceIndex":
        if loss_window <= 0:
            raise ValueError("loss_window must be positive")
        windows = []
        for path in store.iter_paths():
            digest = sha256_file(path)
            header, columns = store.load(path.stem, expected_sha256=digest)
            resets = np.asarray(columns["reset_epoch"], dtype=np.int64)
            for loss_start in range(0, header.step_count - loss_window + 1):
                loss_end = loss_start + loss_window
                epoch = int(resets[loss_start])
                if np.any(resets[loss_start:loss_end] != epoch):
                    continue
                burn_start = loss_start
                while burn_start > 0 and int(resets[burn_start - 1]) == epoch:
                    burn_start -= 1
                windows.append(SequenceWindow(
                    header.episode_id, header.group_id, header.split_id, epoch, burn_start,
                    loss_start, loss_end, digest,
                ))
        return cls(SCHEMA_ID, SEQUENCE_CONTRACT, loss_window, tuple(windows))

    def validate_split_isolation(self) -> None:
        assignments = {}
        for window in self.windows:
            previous = assignments.setdefault(window.group_id, window.split_id)
            if previous != window.split_id:
                raise ValueError(f"group {window.group_id} leaks across splits")

    def sha256(self) -> str:
        payload = {
            "schema_id": self.schema_id,
            "sequence_contract": self.sequence_contract,
            "loss_window": self.loss_window,
            "windows": [asdict(item) for item in self.windows],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def save(self, path: str | Path) -> None:
        target = Path(path)
        payload = {
            "schema_id": self.schema_id,
            "sequence_contract": self.sequence_contract,
            "loss_window": self.loss_window,
            "windows": [asdict(item) for item in self.windows],
            "sha256": self.sha256(),
        }
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SequenceIndex":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        index = cls(
            str(payload["schema_id"]), str(payload["sequence_contract"]), int(payload["loss_window"]),
            tuple(SequenceWindow(**item) for item in payload["windows"]),
        )
        if index.schema_id != SCHEMA_ID or index.sequence_contract != SEQUENCE_CONTRACT:
            raise RuntimeError("incompatible sequence index contract")
        if payload.get("sha256") != index.sha256():
            raise RuntimeError("sequence index checksum mismatch")
        index.validate_split_isolation()
        return index
