"""Strict loader for the standalone TRACTOR research contract files."""

from __future__ import annotations

from dataclasses import fields
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Type, TypeVar

import yaml

from hunter_kinodynamic_rl.rl.algorithms.tractor_tqc.agent import TractorAgentConfig
from hunter_kinodynamic_rl.rl.networks.tractor.contracts import TractorConfig
from hunter_kinodynamic_rl.rl.networks.tractor.selector import SelectorConfig
from hunter_kinodynamic_rl.rl.replay.sequence_schema import SCHEMA_ID, SEQUENCE_CONTRACT

from .loader import default_config_root


T = TypeVar("T")
PROTOCOL_SCHEMA_ID = "tractor_research_protocol_v1"


def _strict_dataclass(cls: Type[T], values: Dict[str, Any], source: str) -> T:
    known = {item.name for item in fields(cls)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"{source} has unknown fields: {unknown}")
    return cls(**values)


def _read_mapping(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a top-level mapping")
    return value


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _protocol_payload_sha256(protocol: dict) -> str:
    payload = dict(protocol)
    payload.pop("frozen_payload_sha256", None)
    return canonical_sha256(payload)


def load_tractor_protocol(config_root: str | None = None) -> dict:
    """Load the frozen, executable research acceptance contract.

    The in-file digest deliberately excludes only the digest field itself.
    Changing any threshold, statistical rule or method identity therefore
    requires an explicit re-freeze instead of silently changing a formal run.
    """
    root = Path(config_root or default_config_root()) / "tractor"
    protocol = _read_mapping(root / "protocol.yaml")
    required = {
        "schema_id", "protocol_version", "status", "frozen_utc", "change_control",
        "frozen_payload_sha256", "primary_method", "primary_baseline", "statistics",
        "headline_gates", "runtime_gates", "support_gates",
    }
    unknown = sorted(set(protocol) - required)
    missing = sorted(required - set(protocol))
    if unknown or missing:
        raise ValueError(f"protocol.yaml unknown={unknown}, missing={missing}")
    if protocol["schema_id"] != PROTOCOL_SCHEMA_ID:
        raise ValueError("protocol.yaml schema_id is incompatible")
    if protocol["status"] != "frozen":
        raise ValueError("research protocol must be frozen before formal use")
    expected_digest = _protocol_payload_sha256(protocol)
    if protocol["frozen_payload_sha256"] != expected_digest:
        raise ValueError(
            "protocol.yaml frozen_payload_sha256 mismatch; create a new protocol version "
            "and output root before accepting changed thresholds"
        )
    statistics = protocol["statistics"]
    expected_statistics = {
        "unit_of_replication", "confidence", "bootstrap_samples", "bootstrap_seed",
        "minimum_unique_seeds", "required_split",
    }
    if set(statistics) != expected_statistics:
        raise ValueError("protocol statistics fields are incomplete or unknown")
    if statistics["unit_of_replication"] != "independent_training_seed":
        raise ValueError("training seed must remain the statistical replication unit")
    if not 0.0 < float(statistics["confidence"]) < 1.0:
        raise ValueError("protocol confidence must be in (0,1)")
    if int(statistics["bootstrap_samples"]) < 1000:
        raise ValueError("formal bootstrap requires at least 1000 resamples")
    if int(statistics["minimum_unique_seeds"]) < 5:
        raise ValueError("formal protocol requires at least five independent seeds")
    if statistics["required_split"] != "locked_test":
        raise ValueError("headline evidence must use the locked_test split")
    gate_fields = {
        "id", "metric", "direction", "absolute_bound", "absolute_threshold",
        "paired_transform", "paired_bound", "paired_threshold",
    }
    gates = protocol["headline_gates"]
    if not isinstance(gates, list) or not gates:
        raise ValueError("protocol headline_gates must be a non-empty list")
    gate_ids = []
    for gate in gates:
        if set(gate) != gate_fields:
            raise ValueError(f"headline gate fields invalid for {gate.get('id')!r}")
        gate_ids.append(str(gate["id"]))
        if gate["direction"] not in {"maximize", "minimize"}:
            raise ValueError(f"invalid direction for gate {gate['id']!r}")
        if gate["absolute_bound"] not in {"lower", "upper", "none"}:
            raise ValueError(f"invalid absolute bound for gate {gate['id']!r}")
        if (gate["absolute_bound"] == "none") != (gate["absolute_threshold"] is None):
            raise ValueError(f"absolute threshold/bound mismatch for gate {gate['id']!r}")
        if gate["paired_transform"] not in {"additive_difference", "relative_change"}:
            raise ValueError(f"invalid paired transform for gate {gate['id']!r}")
        if gate["paired_bound"] not in {"lower", "upper"}:
            raise ValueError(f"invalid paired bound for gate {gate['id']!r}")
    if len(set(gate_ids)) != len(gate_ids):
        raise ValueError("headline gate ids must be unique")
    if {gate["metric"] for gate in gates} != {
        "success_rate", "collision_rate", "time_to_goal_sec_mean", "min_clearance_m_mean",
    }:
        raise ValueError("protocol must freeze success, collision, time-to-goal and clearance gates")
    runtime = protocol["runtime_gates"]
    if set(runtime) != {
        "target_hardware_required", "minimum_timed_decisions", "p99_latency_ms_max",
        "deadline_miss_rate_max",
    }:
        raise ValueError("runtime gate fields are incomplete or unknown")
    if not runtime["target_hardware_required"] or int(runtime["minimum_timed_decisions"]) < 1000:
        raise ValueError("runtime gate requires a sufficiently long target-hardware measurement")
    if float(runtime["p99_latency_ms_max"]) <= 0.0 or not (
        0.0 <= float(runtime["deadline_miss_rate_max"]) < 1.0
    ):
        raise ValueError("runtime thresholds are outside their valid range")
    support = protocol["support_gates"]
    if set(support) != {
        "minimum_locked_scenarios_per_seed", "require_static_and_dynamic",
        "require_complete_method_seed_scenario_matrix",
    }:
        raise ValueError("support gate fields are incomplete or unknown")
    if int(support["minimum_locked_scenarios_per_seed"]) <= 0:
        raise ValueError("minimum locked scenario support must be positive")
    return protocol


def load_tractor_contract(config_root: str | None = None, variant: str = "a7") -> dict:
    root = Path(config_root or default_config_root()) / "tractor"
    model_files = {"a7": "model.yaml", "a8": "model_a8.yaml", "a9": "model_a9.yaml"}
    if variant not in model_files:
        raise ValueError(f"variant must be one of {sorted(model_files)}, got {variant!r}")
    model_raw = _read_mapping(root / model_files[variant])
    data_raw = _read_mapping(root / "data.yaml")
    training_raw = _read_mapping(root / "training.yaml")
    inference_raw = _read_mapping(root / "inference.yaml")
    protocol = load_tractor_protocol(config_root)
    scenario_plan_raw = _read_mapping(root / "scenario_plan.yaml")
    static_scenarios_raw = _read_mapping(root / "scenarios_static.yaml")
    dynamic_scenarios_raw = _read_mapping(root / "scenarios_dynamic.yaml")
    if scenario_plan_raw.get("protocol_version") != protocol["protocol_version"]:
        raise ValueError("scenario plan and research protocol version disagree")
    allowed_training = {"agent", "campaign", "loss_weights"}
    if set(training_raw) - allowed_training:
        raise ValueError(f"training.yaml has unknown sections: {sorted(set(training_raw) - allowed_training)}")
    allowed_inference = {"selector", "runtime"}
    if set(inference_raw) - allowed_inference:
        raise ValueError(f"inference.yaml has unknown sections: {sorted(set(inference_raw) - allowed_inference)}")
    model_values = dict(model_raw)
    if "severity_quantile_levels" in model_values:
        model_values["severity_quantile_levels"] = tuple(model_values["severity_quantile_levels"])
    model = _strict_dataclass(TractorConfig, model_values, "model.yaml")
    agent = _strict_dataclass(TractorAgentConfig, training_raw.get("agent", {}), "training.agent")
    selector = _strict_dataclass(SelectorConfig, inference_raw.get("selector", {}), "inference.selector")
    model.validate()
    agent.validate()
    selector.validate()
    required_data = {
        "schema_id", "sequence_contract", "loss_window", "sampling",
        "event_balancing", "splits", "privileged_fields_allowed_in_inference",
    }
    if required_data - set(data_raw):
        raise ValueError(f"data.yaml missing fields: {sorted(required_data - set(data_raw))}")
    if data_raw["schema_id"] != SCHEMA_ID or data_raw["sequence_contract"] != SEQUENCE_CONTRACT:
        raise ValueError("data.yaml replay/sequence contract does not match executable code")
    if int(data_raw["loss_window"]) <= 0 or data_raw["sampling"] != "uniform_valid_windows":
        raise ValueError("invalid core loss-window or sampling contract")
    if bool(data_raw["event_balancing"]) or bool(data_raw["privileged_fields_allowed_in_inference"]):
        raise ValueError("A7 core forbids event balancing and privileged inference fields")
    if set(data_raw["splits"]) != {"development", "calibration", "locked_test"}:
        raise ValueError("data.yaml must declare the three isolated formal splits")
    campaign = training_raw.get("campaign", {})
    if campaign.get("protocol_status") not in {"draft", "frozen"}:
        raise ValueError("campaign.protocol_status must be draft or frozen")
    if campaign.get("protocol_status") != protocol["status"]:
        raise ValueError("training campaign and protocol status disagree")
    if campaign.get("protocol_version") != protocol["protocol_version"]:
        raise ValueError("training campaign and protocol version disagree")
    seeds = list(campaign.get("seeds", []))
    if len(seeds) < 5 or len(set(seeds)) != len(seeds):
        raise ValueError("headline campaign requires at least five unique seeds")
    runtime = inference_raw.get("runtime", {})
    if float(runtime.get("decision_deadline_ms", 0.0)) <= 0.0:
        raise ValueError("runtime decision deadline must be positive")
    raw = {
        "model": model_raw,
        "data": data_raw,
        "training": training_raw,
        "inference": inference_raw,
        "protocol": protocol,
        "scenario_plan": scenario_plan_raw,
        "static_scenarios": static_scenarios_raw,
        "dynamic_scenarios": dynamic_scenarios_raw,
    }
    return {
        "root": root,
        "variant": variant,
        "model": model,
        "agent": agent,
        "selector": selector,
        "data": data_raw,
        "campaign": campaign,
        "loss_weights": training_raw.get("loss_weights", {}),
        "runtime": runtime,
        "protocol": protocol,
        "protocol_sha256": protocol["frozen_payload_sha256"],
        "scenario_plan": scenario_plan_raw,
        "scenario_manifest_sha256": scenario_plan_raw.get("expected_manifest_sha256"),
        "contract_sha256": canonical_sha256(raw),
        "raw": raw,
    }
