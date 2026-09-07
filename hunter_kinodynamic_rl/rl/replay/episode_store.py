"""Immutable, checksummed and atomically published TRACTOR episode chunks."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Dict, Iterable, Tuple

import numpy as np

from .sequence_schema import EpisodeHeader, SCHEMA_ID, validate_episode_columns


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class EpisodeStore:
    def __init__(self, root: str | os.PathLike[str]):
        self.root = Path(root)
        self.episodes_dir = self.root / "episodes"
        self.episodes_dir.mkdir(parents=True, exist_ok=True)

    def episode_path(self, episode_id: str) -> Path:
        if not episode_id or "/" in episode_id or ".." in episode_id:
            raise ValueError("unsafe episode_id")
        return self.episodes_dir / f"{episode_id}.npz"

    def append(self, header: EpisodeHeader, columns: Dict[str, np.ndarray]) -> Tuple[Path, str]:
        validate_episode_columns(header, columns)
        destination = self.episode_path(header.episode_id)
        if destination.exists():
            raise FileExistsError(f"immutable episode already exists: {destination}")
        fd, temp_name = tempfile.mkstemp(prefix=f".{header.episode_id}.", suffix=".tmp", dir=self.episodes_dir)
        try:
            with os.fdopen(fd, "wb") as stream:
                np.savez_compressed(
                    stream,
                    __schema_id__=np.asarray(SCHEMA_ID),
                    __header_json__=np.asarray(header.canonical_json()),
                    **{name: np.asarray(value) for name, value in columns.items()},
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, destination)
            directory_fd = os.open(self.episodes_dir, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except BaseException:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
            raise
        return destination, sha256_file(destination)

    def load(self, episode_id: str, expected_sha256: str | None = None):
        path = self.episode_path(episode_id)
        if expected_sha256 is not None and sha256_file(path) != expected_sha256:
            raise RuntimeError(f"episode checksum mismatch: {episode_id}")
        with np.load(path, allow_pickle=False) as data:
            if str(data["__schema_id__"].item()) != SCHEMA_ID:
                raise RuntimeError(f"unsupported episode schema in {path}")
            header = EpisodeHeader(**json.loads(str(data["__header_json__"].item())))
            columns = {
                name: data[name].copy() for name in data.files if not name.startswith("__")
            }
        validate_episode_columns(header, columns)
        return header, columns

    def iter_paths(self) -> Iterable[Path]:
        return sorted(self.episodes_dir.glob("*.npz"))
