"""Strict executable contracts for the B1--B8 comparison baselines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from hunter_kinodynamic_rl.config.loader import default_config_root
from hunter_kinodynamic_rl.config.tractor import load_tractor_contract


BASELINE_SCHEMA_ID = "tractor_comparison_baselines_v1"
BASELINE_ARCHITECTURE_REVISION = "tractor-comparison-baselines-r1"
BASELINE_METHODS = tuple(f"B{index}" for index in range(1, 9))


@dataclass(frozen=True)
class ComparisonModelConfig:
    method_id: str
    representation: str
    interaction: str
    risk: str
    observation_dim: int
    t_obs: int
    n_scan: int
    tail_dim: int
    n_critics: int
    n_quantiles: int
    log_std_min: float
    log_std_max: float
    encoder_hidden: int = 256
    latent_dim: int = 128
    recurrent_hidden: int = 128
    attention_heads: int = 4
    cvar_fraction: float = 1.0
    dynamics_loss_weight: float = 0.0
    actor_risk_weight: float = 0.0

    @property
    def variant_id(self) -> str:
        return f"comparison_{self.method_id.lower()}_v1"

    def validate(self) -> None:
        if self.method_id not in BASELINE_METHODS:
            raise ValueError(f"method_id must be one of {BASELINE_METHODS}")
        if self.observation_dim != self.t_obs * self.n_scan + self.tail_dim:
            raise ValueError("comparison observation dimensions are inconsistent")
        if min(
            self.encoder_hidden, self.latent_dim, self.recurrent_hidden,
            self.attention_heads, self.n_critics, self.n_quantiles,
        ) <= 0:
            raise ValueError("comparison network dimensions must be positive")
        if not 0.0 < self.cvar_fraction <= 1.0:
            raise ValueError("cvar_fraction must be in (0,1]")
        if self.dynamics_loss_weight < 0.0 or self.actor_risk_weight < 0.0:
            raise ValueError("comparison loss weights must be non-negative")
        expected = {
            "B1": ("flat_328d", "none", "none"),
            "B2": ("flat_parameter_matched", "none", "none"),
            "B3": ("recurrent_vector", "none", "none"),
            "B4": ("factorized_ego_warped_bev", "implicit_concat", "none"),
            "B5": ("factorized_ego_warped_bev", "cross_attention", "none"),
            "B6": ("compact_latent_world_model", "latent_rollout", "none"),
            "B7": ("flat_328d", "none", "cvar_return"),
            "B8": ("flat_328d", "none", "scalar_endpoint"),
        }
        if (self.representation, self.interaction, self.risk) != expected[self.method_id]:
            raise ValueError(f"{self.method_id} architecture axes do not match the frozen registry")
        if self.method_id != "B7" and self.cvar_fraction != 1.0:
            raise ValueError("only B7 may change the return-tail objective")
        if self.method_id != "B6" and self.dynamics_loss_weight != 0.0:
            raise ValueError("only B6 may use latent dynamics supervision")
        if self.method_id != "B8" and self.actor_risk_weight != 0.0:
            raise ValueError("only B8 may use a scalar endpoint-risk penalty")

    def fingerprint(self) -> str:
        self.validate()
        payload = {
            "architecture_revision": BASELINE_ARCHITECTURE_REVISION,
            "action_schema": "trajectory_kappa_vref_L_v1",
            "config": asdict(self),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


def load_comparison_contract(config_root: str | None, method_id: str) -> dict:
    """Load one baseline while inheriting A7's data/training/action contract."""
    import yaml

    method_id = str(method_id).upper()
    if method_id not in BASELINE_METHODS:
        raise ValueError(f"method_id must be one of {BASELINE_METHODS}")
    root = Path(config_root or default_config_root()) / "tractor"
    source = root / "baseline_models.yaml"
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if set(raw) != {"schema_id", "input_contract_variant", "parameter_match", "methods"}:
        raise ValueError("baseline_models.yaml has unknown or missing top-level fields")
    if raw["schema_id"] != BASELINE_SCHEMA_ID or raw["input_contract_variant"] != "a7":
        raise ValueError("baseline model schema/input contract is incompatible")
    if set(raw["methods"]) != set(BASELINE_METHODS):
        raise ValueError("baseline_models.yaml must define exactly B1--B8")
    a7 = load_tractor_contract(config_root, "a7")
    model_values = dict(raw["methods"][method_id])
    model = ComparisonModelConfig(
        method_id=method_id,
        observation_dim=a7["model"].observation_dim,
        t_obs=a7["model"].t_obs,
        n_scan=a7["model"].n_scan,
        tail_dim=a7["model"].tail_dim,
        n_critics=a7["model"].n_critics,
        n_quantiles=a7["model"].n_quantiles,
        log_std_min=a7["model"].log_std_min,
        log_std_max=a7["model"].log_std_max,
        **model_values,
    )
    model.validate()
    contract_payload = {
        "baseline_schema": raw,
        "method_id": method_id,
        "model": asdict(model),
        "a7_input_contract_sha256": a7["contract_sha256"],
    }
    return {
        **a7,
        "method_id": method_id,
        "model": model,
        "input_model": a7["model"],
        "baseline_registry": raw,
        "parameter_match": raw["parameter_match"],
        "contract_sha256": hashlib.sha256(
            json.dumps(contract_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
