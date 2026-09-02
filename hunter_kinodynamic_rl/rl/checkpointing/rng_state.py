"""Full RNG-state bundle capture/restore for deterministic resume (item 4:
"resume 시 action-selection RNG/Torch RNG를 복원하지 않는다" -- Global
training resume previously re-seeded every RNG from the base training seed
regardless of how many draws had already happened before the checkpoint, so
a resumed run's action-selection/exploration/torch-init streams never
matched an uninterrupted run's).

Bundles Python ``random``, NumPy's global RNG, one caller-supplied
"action-selection" ``np.random.RandomState`` (the stream
``HierarchicalTrainingLoop.run_mission`` actually draws epsilon-greedy
actions from -- kept SEPARATE from the global NumPy RNG so restoring it
never has side effects on whatever else numpy's global stream is used for),
and Torch CPU/CUDA RNG state -- everything
``hunter_kinodynamic_rl.common.seed.seed_all`` seeds, plus the one
additional stream that module never covers.

Saved as ONE opaque ``torch.save``-serialized blob (mirrors this package's
own ``model.pt`` convention for a mixed tensor/tuple/ndarray bundle -- never
a bespoke binary format) with its own sha256 recorded by the caller into the
checkpoint's JSON manifest, exactly like ``pt_sha256``/``replay_sha256``
already are (see ``rl.checkpointing.manager``'s module docstring). Not part
of ``rl.checkpointing.manager``'s own atomic generation-publish machinery
(a separate, deliberately narrower concern) -- callers write this file
BEFORE calling ``save_generation`` and embed its path/sha256/generation id
into ``meta`` so it publishes as part of the SAME atomic ``manifest.json``;
see ``nodes/hierarchical_train_node.py`` for the actual call site.
"""

from __future__ import annotations

import random
import os
import uuid
from typing import Any, Dict, Optional

import numpy as np


def capture_rng_state(action_rng: Optional[np.random.RandomState] = None) -> Dict[str, Any]:
    """Snapshots every RNG stream item 4 requires. ``torch_cpu``/``torch_cuda``
    are omitted entirely (not set to ``None``) when torch isn't importable,
    so :func:`restore_rng_state` can tell "torch state was never captured"
    apart from "torch state was captured as empty"."""
    state: Dict[str, Any] = {
        "python_random": random.getstate(),
        "numpy_global": np.random.get_state(),
        "action_rng": action_rng.get_state() if action_rng is not None else None,
    }
    try:
        import torch
    except Exception:
        return state
    state["torch_cpu"] = torch.get_rng_state()
    state["torch_cuda"] = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return state


def restore_rng_state(state: Dict[str, Any], action_rng: Optional[np.random.RandomState] = None) -> None:
    """Inverse of :func:`capture_rng_state`. ``action_rng`` (if given) is
    restored in place via ``set_state`` -- the caller's existing
    ``RandomState`` object identity is preserved (every other reference to
    it, e.g. already threaded through a running training loop, sees the
    restored stream immediately, never a NEW object it would need to
    re-thread)."""
    random.setstate(state["python_random"])
    np.random.set_state(state["numpy_global"])
    if action_rng is not None and state.get("action_rng") is not None:
        action_rng.set_state(state["action_rng"])
    if "torch_cpu" not in state:
        return
    try:
        import torch
    except Exception:
        return
    torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_rng_state_file(path: str, state: Dict[str, Any]) -> str:
    """Serializes ``state`` (from :func:`capture_rng_state`) to ``path`` via
    ``torch.save`` and returns its sha256 -- the caller embeds both
    ``path``/``sha256`` into the checkpoint's JSON manifest (see this
    module's own docstring for why this file lives outside
    ``rl.checkpointing.manager``'s atomic generation unit)."""
    import torch

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = f"{path}.{uuid.uuid4().hex}.tmp"
    try:
        torch.save(state, tmp_path)
        with open(tmp_path, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        directory_fd = os.open(os.path.dirname(os.path.abspath(path)), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    from hunter_kinodynamic_rl.rl.checkpointing.manager import sha256_of_file

    return sha256_of_file(path)


def load_rng_state_file(path: str) -> Dict[str, Any]:
    import torch

    return torch.load(path, weights_only=False)
