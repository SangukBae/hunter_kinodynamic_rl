"""Uniform-sampling replay buffer, schema v2 (see schema.py): native
``risk_target`` + explicit validity mask + counterfactual candidate fields.

This is a deliberately SIMPLER buffer than drl_agent's ``rl/replay/buffer.py``
(no LAP/PER, no risk-balanced stratified sampling) -- see docs/IMPLEMENTATION_PLAN.md
for why: this package's research contribution is the risk-aware TRAINING
OBJECTIVE, not a replay-sampling scheme.

Torch is imported lazily inside :meth:`sample_torch` only -- everything else
is numpy-only so this module is usable and unit-tested without torch
(mirrors drl_agent's torch-gating convention).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from hunter_kinodynamic_rl.rl.replay.schema import SCHEMA_VERSION


@dataclass
class RiskTransition:
    """Optional per-``add()`` risk/counterfactual payload -- everything
    None/empty means "no label this transition" (the explicit ``valid=False``
    case, never inferred from NaN alone)."""

    risk_target: Optional[float] = None
    min_clearance_m: float = float("nan")
    ttc_sec: float = float("nan")
    collision_within_horizon: bool = False
    stopping_margin_m: float = float("nan")
    steering_saturation: bool = False
    goal_progress_m: float = float("nan")
    unrecoverable: bool = False
    safer_alternative_margin: float = float("nan")
    candidate_kappa: List[float] = field(default_factory=list)
    candidate_v_ref: List[float] = field(default_factory=list)
    candidate_horizon: List[float] = field(default_factory=list)
    candidate_risk: List[float] = field(default_factory=list)
    candidate_goal_progress: List[float] = field(default_factory=list)
    actor_candidate_index: int = 0

    @property
    def valid(self) -> bool:
        return self.risk_target is not None


class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, capacity: int = 1_000_000, seed: int = 0,
                 max_candidates: int = 8):
        if state_dim <= 0 or action_dim <= 0:
            raise ValueError("state_dim/action_dim must be > 0")
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        if max_candidates < 0:
            raise ValueError("max_candidates must be >= 0")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.capacity = capacity
        self.max_candidates = max_candidates
        self.ptr = 0
        self.size = 0
        self._rng = np.random.RandomState(seed)

        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)

        self.risk_target = np.full((capacity, 1), np.nan, dtype=np.float32)
        self.valid = np.zeros((capacity, 1), dtype=np.float32)  # explicit mask -- section 5
        self.min_clearance_m = np.full((capacity, 1), np.nan, dtype=np.float32)
        self.ttc_sec = np.full((capacity, 1), np.nan, dtype=np.float32)
        self.collision_within_horizon = np.zeros((capacity, 1), dtype=np.float32)
        self.stopping_margin_m = np.full((capacity, 1), np.nan, dtype=np.float32)
        self.steering_saturation = np.zeros((capacity, 1), dtype=np.float32)
        self.goal_progress_m = np.full((capacity, 1), np.nan, dtype=np.float32)
        self.unrecoverable = np.zeros((capacity, 1), dtype=np.float32)
        self.safer_alternative_margin = np.full((capacity, 1), np.nan, dtype=np.float32)

        self.candidate_kappa = np.zeros((capacity, max_candidates), dtype=np.float32)
        self.candidate_v_ref = np.zeros((capacity, max_candidates), dtype=np.float32)
        self.candidate_horizon = np.zeros((capacity, max_candidates), dtype=np.float32)
        self.candidate_risk = np.zeros((capacity, max_candidates), dtype=np.float32)
        self.candidate_goal_progress = np.zeros((capacity, max_candidates), dtype=np.float32)
        self.candidate_valid_mask = np.zeros((capacity, max_candidates), dtype=np.float32)
        self.actor_candidate_index = np.zeros((capacity, 1), dtype=np.float32)

    def __len__(self) -> int:
        return self.size

    def add(self, state, action, next_state, reward: float, done: bool,
            risk: Optional[RiskTransition] = None) -> None:
        i = self.ptr
        self.state[i] = state
        self.action[i] = action
        self.next_state[i] = next_state
        self.reward[i] = reward
        self.done[i] = float(done)

        risk = risk or RiskTransition()
        self.valid[i] = 1.0 if risk.valid else 0.0
        self.risk_target[i] = np.nan if risk.risk_target is None else float(risk.risk_target)
        self.min_clearance_m[i] = risk.min_clearance_m
        self.ttc_sec[i] = risk.ttc_sec
        self.collision_within_horizon[i] = 1.0 if risk.collision_within_horizon else 0.0
        self.stopping_margin_m[i] = risk.stopping_margin_m
        self.steering_saturation[i] = 1.0 if risk.steering_saturation else 0.0
        self.goal_progress_m[i] = risk.goal_progress_m
        self.unrecoverable[i] = 1.0 if risk.unrecoverable else 0.0
        self.safer_alternative_margin[i] = risk.safer_alternative_margin

        self.candidate_valid_mask[i] = 0.0
        self.candidate_kappa[i] = 0.0
        self.candidate_v_ref[i] = 0.0
        self.candidate_horizon[i] = 0.0
        self.candidate_risk[i] = 0.0
        self.candidate_goal_progress[i] = 0.0
        n = min(len(risk.candidate_kappa), self.max_candidates)
        if n > 0:
            self.candidate_kappa[i, :n] = risk.candidate_kappa[:n]
            self.candidate_v_ref[i, :n] = risk.candidate_v_ref[:n]
            self.candidate_horizon[i, :n] = risk.candidate_horizon[:n]
            self.candidate_risk[i, :n] = risk.candidate_risk[:n]
            if risk.candidate_goal_progress:
                self.candidate_goal_progress[i, :n] = risk.candidate_goal_progress[:n]
            self.candidate_valid_mask[i, :n] = 1.0
        self.actor_candidate_index[i] = risk.actor_candidate_index

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_indices(self, batch_size: int) -> np.ndarray:
        if self.size == 0:
            raise RuntimeError("cannot sample from an empty replay buffer")
        return self._rng.randint(0, self.size, size=batch_size)

    def sample(self, batch_size: int) -> dict:
        idx = self.sample_indices(batch_size)
        return {
            "state": self.state[idx], "action": self.action[idx], "next_state": self.next_state[idx],
            "reward": self.reward[idx], "not_done": 1.0 - self.done[idx],
            "risk_target": self.risk_target[idx], "valid": self.valid[idx],
            "min_clearance_m": self.min_clearance_m[idx], "ttc_sec": self.ttc_sec[idx],
            "collision_within_horizon": self.collision_within_horizon[idx],
            "stopping_margin_m": self.stopping_margin_m[idx], "unrecoverable": self.unrecoverable[idx],
            "steering_saturation": self.steering_saturation[idx],
            "goal_progress_m": self.goal_progress_m[idx],
            "safer_alternative_margin": self.safer_alternative_margin[idx],
            "candidate_kappa": self.candidate_kappa[idx], "candidate_v_ref": self.candidate_v_ref[idx],
            "candidate_horizon": self.candidate_horizon[idx], "candidate_risk": self.candidate_risk[idx],
            "candidate_goal_progress": self.candidate_goal_progress[idx],
            "candidate_valid_mask": self.candidate_valid_mask[idx],
            "actor_candidate_index": self.actor_candidate_index[idx],
        }

    def valid_ratio(self) -> float:
        """Fraction of CURRENTLY STORED transitions with a valid risk label
        -- section 5: "risk label 통계와 valid ratio 기록"."""
        if self.size == 0:
            return 0.0
        return float(self.valid[: self.size].mean())

    def sample_torch(self, batch_size: int, device=None) -> dict:
        import torch
        batch = self.sample(batch_size)
        return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in batch.items()}

    def save(self, path: str, *, generation: str = "") -> None:
        """``generation`` (section item-3): an opaque caller-supplied tag
        (e.g. a uuid4 hex shared with the SAME save call's ``.pt``/manifest
        via ``rl.checkpointing.manager.save_generation``) embedded directly
        in this file -- lets a reader verify this specific replay file
        belongs to the SAME checkpoint generation as the model/manifest it's
        paired with, without needing to separately track a hash mapping.
        Empty string (the default) for callers that don't use the
        generation-based checkpoint layout."""
        n = self.size
        # numpy's savez_compressed auto-appends ".npz" when the target path
        # doesn't already end with it -- pin down the REAL final path first
        # so the atomic tmp-then-replace below targets exactly that file,
        # not a name numpy silently derives.
        final_path = path if path.endswith(".npz") else path + ".npz"
        # Ends in ".npz" so numpy does NOT append its own ".npz" suffix --
        # the file numpy actually writes is EXACTLY this path.
        tmp_path = final_path[: -len(".npz")] + ".tmp.npz"
        rng_state = self._rng.get_state()
        np.savez_compressed(
            tmp_path,
            schema_version=SCHEMA_VERSION,
            generation=str(generation),
            state=self.state[:n], action=self.action[:n], next_state=self.next_state[:n],
            reward=self.reward[:n], done=self.done[:n],
            risk_target=self.risk_target[:n], valid=self.valid[:n],
            min_clearance_m=self.min_clearance_m[:n], ttc_sec=self.ttc_sec[:n],
            collision_within_horizon=self.collision_within_horizon[:n],
            stopping_margin_m=self.stopping_margin_m[:n], unrecoverable=self.unrecoverable[:n],
            steering_saturation=self.steering_saturation[:n], goal_progress_m=self.goal_progress_m[:n],
            safer_alternative_margin=self.safer_alternative_margin[:n],
            candidate_kappa=self.candidate_kappa[:n], candidate_v_ref=self.candidate_v_ref[:n],
            candidate_horizon=self.candidate_horizon[:n], candidate_risk=self.candidate_risk[:n],
            candidate_goal_progress=self.candidate_goal_progress[:n],
            candidate_valid_mask=self.candidate_valid_mask[:n],
            actor_candidate_index=self.actor_candidate_index[:n],
            ptr=self.ptr, size=self.size, max_candidates=self.max_candidates,
            state_dim=self.state_dim, action_dim=self.action_dim, capacity=self.capacity,
            # Sampling RNG state -- WITHOUT this, resume restarts the
            # sample_indices() draw sequence from a fresh reseed instead of
            # continuing it (section P1-1).
            rng_state_name=rng_state[0], rng_state_keys=rng_state[1],
            rng_state_pos=rng_state[2], rng_state_has_gauss=rng_state[3], rng_state_cached_gaussian=rng_state[4],
        )
        os.replace(tmp_path, final_path)

    @classmethod
    def load(cls, path: str, seed: int = 0) -> "ReplayBuffer":
        data = np.load(path)
        version = int(data["schema_version"])
        if version not in (3, SCHEMA_VERSION):
            raise ValueError(
                f"replay buffer file schema_version={version} != this code's SCHEMA_VERSION={SCHEMA_VERSION}; "
                "only the explicit v3->v4 migration is supported."
            )
        buf = cls(int(data["state_dim"]), int(data["action_dim"]), int(data["capacity"]),
                   seed=seed, max_candidates=int(data["max_candidates"]))
        if "rng_state_name" in data:
            buf._rng.set_state((
                str(data["rng_state_name"]), data["rng_state_keys"],
                int(data["rng_state_pos"]), int(data["rng_state_has_gauss"]),
                float(data["rng_state_cached_gaussian"]),
            ))
        n = int(data["size"])
        for field_name in (
            "state", "action", "next_state", "reward", "done", "risk_target", "valid",
            "min_clearance_m", "ttc_sec", "collision_within_horizon", "stopping_margin_m",
            "unrecoverable", "safer_alternative_margin", "candidate_kappa", "candidate_v_ref",
            "candidate_horizon", "candidate_risk", "candidate_valid_mask", "actor_candidate_index",
        ):
            getattr(buf, field_name)[:n] = data[field_name]
        if version >= 4:
            for field_name in ("steering_saturation", "goal_progress_m", "candidate_goal_progress"):
                getattr(buf, field_name)[:n] = data[field_name]
        buf.ptr = int(data["ptr"])
        buf.size = n
        buf.generation = str(data["generation"]) if "generation" in data else ""
        return buf


def peek_replay_generation(path: str) -> str:
    """Reads just the ``generation`` field (section item-3) WITHOUT
    constructing a full :class:`ReplayBuffer` -- cheap enough to call from
    ``rl.checkpointing.manager.load_generation``'s own consistency check on
    every load. Empty string if the file predates this field."""
    data = np.load(path)
    return str(data["generation"]) if "generation" in data else ""
