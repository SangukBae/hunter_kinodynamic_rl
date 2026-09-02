"""Global (option-level SMDP) replay buffer (plan section 8.8) -- entirely
separate from ``rl.replay.buffer.ReplayBuffer`` (the LOCAL, per-control-tick
buffer): different time scale, different reward, different state shape.
Never share a buffer instance or an on-disk file between the two.

Map channels are stored as ``uint8`` (plan: "map은 uint8 또는 compact
representation으로 저장") -- :class:`~hunter_kinodynamic_rl.navigation.global_rl.observation.GlobalObservation`'s
``map_tensor`` is already float32 in ``[0, 1]`` (every channel is either a
0/1 boolean or the ``[0, 1]``-ranged goal-direction/failure field), so
quantizing to ``round(x * 255)`` and dequantizing to ``x / 255.0`` on sample
is lossless enough for a training signal while cutting storage 4x versus
float32.

``add()`` requires an explicit ``mode="train"`` keyword (plan: "test/
benchmark mode transition은 Global replay에 저장하지 않도록 API에 guard...
추가") -- this is an enforced guard, not just a caller convention: any other
value raises immediately, so a validation/test rollout accidentally reusing
this buffer's ``add()`` fails loudly instead of silently contaminating
training data.

Phase 5 (plan section 9): channel/scalar/candidate-feature NAMES are no
longer fixed module constants (Phase 5's ablation flags make them a
function of the profile's ``GlobalRLConfig`` -- see ``observation.resolve_*``)
so ``GlobalReplayBuffer`` accepts them as constructor parameters (defaulting
to the Phase-4-identical base tuples) instead of importing fixed globals;
callers building a Phase-5-configured buffer pass the RESOLVED tuples so a
saved file's metadata always matches the code that wrote it. The
topological-memory node tensor (``node_tensor``/``node_validity_mask``,
shape ``(0, N_NODE_FEATURES)``/``(0,)`` when topology feedback is disabled)
is stored alongside every transition unconditionally -- a zero-width array
costs nothing and needs no special-casing in save/load.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from hunter_kinodynamic_rl.navigation.global_rl.observation import (
    CANDIDATE_FEATURE_NAMES, GlobalObservation, MAP_CHANNEL_NAMES, SCALAR_NAMES,
)
from hunter_kinodynamic_rl.navigation.global_rl.replay_schema import (
    SCHEMA_VERSION, decode_failure_reason, encode_failure_reason,
)
from hunter_kinodynamic_rl.navigation.hierarchy.subgoal_manager import SubgoalStatus
from hunter_kinodynamic_rl.navigation.memory.topological_graph import N_NODE_FEATURES


def _quantize_map(map_tensor: np.ndarray) -> np.ndarray:
    return np.clip(np.round(np.asarray(map_tensor, dtype=np.float32) * 255.0), 0, 255).astype(np.uint8)


def _check_channel_metadata(field_name: str, saved: tuple, current: tuple) -> None:
    if saved != current:
        raise ValueError(
            f"GlobalReplayBuffer file {field_name}={saved!r} does not match the current buffer's "
            f"{field_name}={current!r} -- this file's columns no longer mean what this buffer instance "
            "(built against a specific resolved GlobalRLConfig) expects (a channel was renamed, reordered, "
            "or inserted since this file was saved). Refusing to load it as if the shapes matching were enough."
        )


def _dequantize_map(map_uint8: np.ndarray) -> np.ndarray:
    return map_uint8.astype(np.float32) / 255.0


@dataclass
class GlobalTransition:
    map_state: np.ndarray
    scalar_state: np.ndarray
    candidate_features: np.ndarray
    action_mask: np.ndarray
    action: int
    option_reward: float
    next_map_state: np.ndarray
    next_scalar_state: np.ndarray
    next_candidate_features: np.ndarray
    next_action_mask: np.ndarray
    mission_done: bool
    subgoal_success: bool
    failure_reason: Optional[SubgoalStatus]
    local_steps: int
    exploration_gain: float
    risk_integral: float
    # Phase 5 -- default to the empty (0-node) shape, matching
    # GlobalObservation's own "topology disabled" default, so a caller that
    # constructs a GlobalTransition directly (e.g. an existing Phase 4 test)
    # without ever mentioning topology still works unchanged.
    node_tensor: np.ndarray = field(default_factory=lambda: np.zeros((0, N_NODE_FEATURES), dtype=np.float32))
    node_validity_mask: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=bool))
    next_node_tensor: np.ndarray = field(default_factory=lambda: np.zeros((0, N_NODE_FEATURES), dtype=np.float32))
    next_node_validity_mask: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=bool))

    @classmethod
    def from_observations(
        cls, obs: GlobalObservation, action: int, option_reward: float, next_obs: GlobalObservation,
        *, mission_done: bool, subgoal_success: bool, failure_reason: Optional[SubgoalStatus],
        local_steps: int, exploration_gain: float, risk_integral: float,
    ) -> "GlobalTransition":
        return cls(
            map_state=obs.map_tensor, scalar_state=obs.scalar_tensor, candidate_features=obs.candidate_tensor,
            action_mask=obs.action_mask, action=action, option_reward=option_reward,
            next_map_state=next_obs.map_tensor, next_scalar_state=next_obs.scalar_tensor,
            next_candidate_features=next_obs.candidate_tensor, next_action_mask=next_obs.action_mask,
            mission_done=mission_done, subgoal_success=subgoal_success, failure_reason=failure_reason,
            local_steps=local_steps, exploration_gain=exploration_gain, risk_integral=risk_integral,
            node_tensor=obs.node_tensor, node_validity_mask=obs.node_validity_mask,
            next_node_tensor=next_obs.node_tensor, next_node_validity_mask=next_obs.node_validity_mask,
        )


class GlobalReplayBuffer:
    def __init__(
        self, map_shape: Tuple[int, int, int], n_scalars: int, n_candidates: int, n_candidate_features: int,
        capacity: int = 50_000, seed: int = 0, node_shape: Tuple[int, int] = (0, N_NODE_FEATURES),
        map_channel_names: tuple = MAP_CHANNEL_NAMES, scalar_names: tuple = SCALAR_NAMES,
        candidate_feature_names: tuple = CANDIDATE_FEATURE_NAMES,
    ):
        if capacity <= 0:
            raise ValueError("GlobalReplayBuffer capacity must be > 0")
        self.map_shape = tuple(int(x) for x in map_shape)
        self.n_scalars = int(n_scalars)
        self.n_candidates = int(n_candidates)
        self.n_candidate_features = int(n_candidate_features)
        self.node_shape = (int(node_shape[0]), int(node_shape[1]))
        self.map_channel_names = tuple(str(x) for x in map_channel_names)
        self.scalar_names = tuple(str(x) for x in scalar_names)
        self.candidate_feature_names = tuple(str(x) for x in candidate_feature_names)
        self.capacity = int(capacity)
        self.ptr = 0
        self.size = 0
        self._rng = np.random.RandomState(seed)

        c = self.capacity
        self.map_state = np.zeros((c, *self.map_shape), dtype=np.uint8)
        self.next_map_state = np.zeros_like(self.map_state)
        self.scalar_state = np.zeros((c, self.n_scalars), dtype=np.float32)
        self.next_scalar_state = np.zeros_like(self.scalar_state)
        self.candidate_features = np.zeros((c, self.n_candidates, self.n_candidate_features), dtype=np.float32)
        self.next_candidate_features = np.zeros_like(self.candidate_features)
        self.action_mask = np.zeros((c, self.n_candidates), dtype=bool)
        self.next_action_mask = np.zeros_like(self.action_mask)
        self.action = np.zeros((c, 1), dtype=np.int64)
        self.option_reward = np.zeros((c, 1), dtype=np.float32)
        self.mission_done = np.zeros((c, 1), dtype=np.float32)
        self.subgoal_success = np.zeros((c, 1), dtype=np.float32)
        self.failure_reason_code = np.full((c, 1), -1, dtype=np.int32)
        self.local_steps = np.zeros((c, 1), dtype=np.int32)
        self.exploration_gain = np.zeros((c, 1), dtype=np.float32)
        self.risk_integral = np.zeros((c, 1), dtype=np.float32)
        self.node_tensor = np.zeros((c, *self.node_shape), dtype=np.float32)
        self.next_node_tensor = np.zeros_like(self.node_tensor)
        self.node_validity_mask = np.zeros((c, self.node_shape[0]), dtype=bool)
        self.next_node_validity_mask = np.zeros_like(self.node_validity_mask)
        self.generation = ""

    def __len__(self) -> int:
        return self.size

    def add(self, transition: GlobalTransition, *, mode: str) -> None:
        if mode != "train":
            raise ValueError(
                f"GlobalReplayBuffer.add() refuses mode={mode!r} -- only 'train' transitions may be stored "
                "(plan section 8.8: validation/test/benchmark rollouts must never be written into the "
                "Global replay buffer)"
            )
        i = self.ptr
        self.map_state[i] = _quantize_map(transition.map_state)
        self.next_map_state[i] = _quantize_map(transition.next_map_state)
        self.scalar_state[i] = transition.scalar_state
        self.next_scalar_state[i] = transition.next_scalar_state
        self.candidate_features[i] = transition.candidate_features
        self.next_candidate_features[i] = transition.next_candidate_features
        self.action_mask[i] = transition.action_mask
        self.next_action_mask[i] = transition.next_action_mask
        self.action[i] = int(transition.action)
        self.option_reward[i] = float(transition.option_reward)
        self.mission_done[i] = 1.0 if transition.mission_done else 0.0
        self.subgoal_success[i] = 1.0 if transition.subgoal_success else 0.0
        self.failure_reason_code[i] = encode_failure_reason(transition.failure_reason)
        self.local_steps[i] = int(transition.local_steps)
        self.exploration_gain[i] = float(transition.exploration_gain)
        self.risk_integral[i] = float(transition.risk_integral)
        if self.node_shape[0] > 0:
            self.node_tensor[i] = transition.node_tensor
            self.next_node_tensor[i] = transition.next_node_tensor
            self.node_validity_mask[i] = transition.node_validity_mask
            self.next_node_validity_mask[i] = transition.next_node_validity_mask

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_indices(self, batch_size: int) -> np.ndarray:
        if self.size == 0:
            raise RuntimeError("cannot sample from an empty GlobalReplayBuffer")
        return self._rng.randint(0, self.size, size=batch_size)

    def sample(self, batch_size: int) -> dict:
        idx = self.sample_indices(batch_size)
        return {
            "map_state": _dequantize_map(self.map_state[idx]),
            "scalar_state": self.scalar_state[idx],
            "candidate_features": self.candidate_features[idx],
            "action_mask": self.action_mask[idx],
            "action": self.action[idx],
            "option_reward": self.option_reward[idx],
            "next_map_state": _dequantize_map(self.next_map_state[idx]),
            "next_scalar_state": self.next_scalar_state[idx],
            "next_candidate_features": self.next_candidate_features[idx],
            "next_action_mask": self.next_action_mask[idx],
            "mission_done": self.mission_done[idx],
            "subgoal_success": self.subgoal_success[idx],
            "failure_reason_code": self.failure_reason_code[idx],
            "local_steps": self.local_steps[idx],
            "exploration_gain": self.exploration_gain[idx],
            "risk_integral": self.risk_integral[idx],
            "node_tensor": self.node_tensor[idx],
            "node_validity_mask": self.node_validity_mask[idx],
            "next_node_tensor": self.next_node_tensor[idx],
            "next_node_validity_mask": self.next_node_validity_mask[idx],
        }

    def sample_torch(self, batch_size: int, device=None) -> dict:
        import torch
        batch = self.sample(batch_size)
        out = {}
        for key, value in batch.items():
            if key in ("action_mask", "next_action_mask", "node_validity_mask", "next_node_validity_mask"):
                out[key] = torch.as_tensor(value, dtype=torch.bool, device=device)
            elif key == "action":
                out[key] = torch.as_tensor(value, dtype=torch.int64, device=device)
            else:
                out[key] = torch.as_tensor(value, dtype=torch.float32, device=device)
        return out

    def save(self, path: str, *, generation: str = "") -> None:
        n = self.size
        final_path = path if path.endswith(".npz") else path + ".npz"
        tmp_path = final_path[: -len(".npz")] + ".tmp.npz"
        rng_state = self._rng.get_state()
        np.savez_compressed(
            tmp_path,
            schema_version=SCHEMA_VERSION, generation=str(generation),
            map_shape=np.array(self.map_shape, dtype=np.int64),
            n_scalars=self.n_scalars, n_candidates=self.n_candidates,
            n_candidate_features=self.n_candidate_features, capacity=self.capacity,
            node_shape=np.array(self.node_shape, dtype=np.int64),
            # v2: the ACTUAL channel/scalar/candidate-feature names+order
            # THIS BUFFER INSTANCE was constructed with -- load() rejects a
            # mismatch against the current code's tuples outright, so a
            # file saved before a channel got renamed/reordered/inserted
            # can never be silently sampled as if its columns still meant
            # the same thing (plan: "replay schema version과 map/channel
            # metadata 저장"). Plain numpy unicode arrays -- no
            # allow_pickle needed for either write or read.
            map_channel_names=np.array(self.map_channel_names), scalar_names=np.array(self.scalar_names),
            candidate_feature_names=np.array(self.candidate_feature_names),
            map_state=self.map_state[:n], next_map_state=self.next_map_state[:n],
            scalar_state=self.scalar_state[:n], next_scalar_state=self.next_scalar_state[:n],
            candidate_features=self.candidate_features[:n], next_candidate_features=self.next_candidate_features[:n],
            action_mask=self.action_mask[:n], next_action_mask=self.next_action_mask[:n],
            action=self.action[:n], option_reward=self.option_reward[:n],
            mission_done=self.mission_done[:n], subgoal_success=self.subgoal_success[:n],
            failure_reason_code=self.failure_reason_code[:n], local_steps=self.local_steps[:n],
            exploration_gain=self.exploration_gain[:n], risk_integral=self.risk_integral[:n],
            node_tensor=self.node_tensor[:n], next_node_tensor=self.next_node_tensor[:n],
            node_validity_mask=self.node_validity_mask[:n], next_node_validity_mask=self.next_node_validity_mask[:n],
            ptr=self.ptr, size=self.size,
            rng_state_name=rng_state[0], rng_state_keys=rng_state[1], rng_state_pos=rng_state[2],
            rng_state_has_gauss=rng_state[3], rng_state_cached_gaussian=rng_state[4],
        )
        os.replace(tmp_path, final_path)

    @classmethod
    def load(
        cls, path: str, seed: int = 0, *,
        expected_map_channel_names: tuple = MAP_CHANNEL_NAMES, expected_scalar_names: tuple = SCALAR_NAMES,
        expected_candidate_feature_names: tuple = CANDIDATE_FEATURE_NAMES,
    ) -> "GlobalReplayBuffer":
        """``expected_*`` default to the Phase-4-identical base tuples (a
        caller that never touches Phase 5's ablation flags gets the exact
        same check the v2 schema always did); a Phase-5-configured caller
        should pass its OWN resolved ``observation.resolve_map_channel_names(config)``/
        ``resolve_candidate_feature_names(config)`` here so a file saved
        under a DIFFERENT ablation configuration is rejected outright,
        never silently reinterpreted."""
        data = np.load(path)
        version = int(data["schema_version"])
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"GlobalReplayBuffer file schema_version={version} != this code's SCHEMA_VERSION="
                f"{SCHEMA_VERSION}; no migration is implemented for this version."
            )
        map_channel_names = tuple(str(x) for x in data["map_channel_names"])
        scalar_names = tuple(str(x) for x in data["scalar_names"])
        candidate_feature_names = tuple(str(x) for x in data["candidate_feature_names"])
        _check_channel_metadata("map_channel_names", map_channel_names, tuple(expected_map_channel_names))
        _check_channel_metadata("scalar_names", scalar_names, tuple(expected_scalar_names))
        _check_channel_metadata(
            "candidate_feature_names", candidate_feature_names, tuple(expected_candidate_feature_names),
        )
        map_shape = tuple(int(x) for x in data["map_shape"])
        node_shape = tuple(int(x) for x in data["node_shape"]) if "node_shape" in data else (0, 0)
        buf = cls(
            map_shape, int(data["n_scalars"]), int(data["n_candidates"]), int(data["n_candidate_features"]),
            capacity=int(data["capacity"]), seed=seed, node_shape=node_shape,
            map_channel_names=map_channel_names, scalar_names=scalar_names,
            candidate_feature_names=candidate_feature_names,
        )
        buf._rng.set_state((
            str(data["rng_state_name"]), data["rng_state_keys"], int(data["rng_state_pos"]),
            int(data["rng_state_has_gauss"]), float(data["rng_state_cached_gaussian"]),
        ))
        n = int(data["size"])
        for field_name in (
            "map_state", "next_map_state", "scalar_state", "next_scalar_state", "candidate_features",
            "next_candidate_features", "action_mask", "next_action_mask", "action", "option_reward",
            "mission_done", "subgoal_success", "failure_reason_code", "local_steps", "exploration_gain",
            "risk_integral", "node_tensor", "next_node_tensor", "node_validity_mask", "next_node_validity_mask",
        ):
            getattr(buf, field_name)[:n] = data[field_name]
        buf.ptr = int(data["ptr"])
        buf.size = n
        buf.generation = str(data["generation"]) if "generation" in data else ""
        return buf


__all__ = [
    "GlobalReplayBuffer", "GlobalTransition", "decode_failure_reason", "encode_failure_reason",
]
