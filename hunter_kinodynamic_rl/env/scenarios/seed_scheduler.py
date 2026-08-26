"""Per-episode seed scheduler with train/validation/test pool isolation
(section 3.4). A trainer/evaluator asks for "the next seed for this mode";
the scheduler:

  * draws seeds from ONLY the requested split's configured range
    (``ScenarioConfig.{train,validation,test}_seed_range``),
  * is deterministic given ``(run_seed, mode)`` -- the SAME run_seed always
    reproduces the SAME episode-seed sequence, a DIFFERENT run_seed produces
    a DIFFERENT sequence (section 3.4's explicit acceptance criterion),
  * is resumable: ``episode_index`` is the only state a checkpoint needs to
    persist to continue the exact same sequence after resume,
  * REJECTS (raises, does not silently clamp) an explicit seed from the
    wrong pool -- e.g. a training run handed a seed from the test range.
"""

from __future__ import annotations

import numpy as np

from hunter_kinodynamic_rl.env.scenarios.procedural_generator import SeedSplitError, seed_split
from hunter_kinodynamic_rl.config.schema import ScenarioConfig


class SeedPoolViolation(ValueError):
    """Raised when a mode is handed (or would draw) a seed outside its own
    configured pool -- section 3.4: "training mode에서 test seed를 받으면
    즉시 오류"."""


_MODE_RANGE_ATTR = {
    "train": "train_seed_range",
    "validation": "validation_seed_range",
    "test": "test_seed_range",
}


class SeedScheduler:
    def __init__(self, run_seed: int, scenario_cfg: ScenarioConfig, mode: str, episode_index: int = 0):
        if mode not in _MODE_RANGE_ATTR:
            raise ValueError(f"mode must be one of {sorted(_MODE_RANGE_ATTR)}, got {mode!r}")
        self.run_seed = int(run_seed)
        self.scenario_cfg = scenario_cfg
        self.mode = mode
        self.episode_index = int(episode_index)
        lo, hi = getattr(scenario_cfg, _MODE_RANGE_ATTR[mode])
        self._lo, self._hi = int(lo), int(hi)
        if self._hi < self._lo:
            raise ValueError(f"{mode} seed range is empty: [{self._lo}, {self._hi}]")

    def validate_explicit_seed(self, seed: int) -> None:
        """Raise SeedPoolViolation if ``seed`` is not in THIS scheduler's
        mode's own range -- call this whenever an external caller (e.g. a
        ``/seed`` ROS request) supplies a seed rather than letting the
        scheduler draw the next one itself."""
        try:
            actual_split = seed_split(seed, self.scenario_cfg)
        except SeedSplitError as e:
            raise SeedPoolViolation(f"seed {seed} is outside ALL configured seed ranges") from e
        if actual_split != self.mode:
            raise SeedPoolViolation(
                f"seed {seed} belongs to the {actual_split!r} pool, but this scheduler is "
                f"running in {self.mode!r} mode -- refusing to cross train/validation/test isolation"
            )

    def next_seed(self) -> int:
        """Deterministic sequence: seed_i = _lo + (draw from a private
        RandomState seeded by (run_seed, mode, episode_index)) mod range_size.
        Advances episode_index."""
        range_size = self._hi - self._lo + 1
        mode_salt = {"train": 0, "validation": 1, "test": 2}[self.mode]
        ss = np.random.SeedSequence([self.run_seed & 0xFFFFFFFF, mode_salt, self.episode_index & 0xFFFFFFFF])
        draw = int(ss.generate_state(1, dtype=np.uint32)[0]) % range_size
        seed = self._lo + draw
        self.episode_index += 1
        return seed

    def state_dict(self) -> dict:
        return {"run_seed": self.run_seed, "mode": self.mode, "episode_index": self.episode_index}

    @classmethod
    def from_state_dict(cls, state: dict, scenario_cfg: ScenarioConfig) -> "SeedScheduler":
        return cls(state["run_seed"], scenario_cfg, state["mode"], episode_index=state["episode_index"])
