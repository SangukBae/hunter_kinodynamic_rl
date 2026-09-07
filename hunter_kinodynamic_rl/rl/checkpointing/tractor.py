"""Atomic, role-separated checkpoint and deployment artifacts for TRACTOR."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import re
import tempfile
import uuid
from typing import Any, Mapping

import numpy as np
import torch

from hunter_kinodynamic_rl.rl.algorithms.tractor_tqc.agent import TractorAgent
from hunter_kinodynamic_rl.rl.networks.tractor.calibration import PlattCalibration


CHECKPOINT_SCHEMA = "tractor_training_checkpoint_v1"
BUNDLE_SCHEMA = "tractor_deployment_bundle_v1"
CALIBRATION_SCHEMA = "tractor_calibration_artifact_v1"
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _validate_identifier(value: str, label: str) -> None:
    if not _SAFE_IDENTIFIER.fullmatch(value):
        raise ValueError(f"unsafe {label}: {value!r}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _capture_global_rng() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_global_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _publish_pointer(root: Path, tag: str, generation: str) -> None:
    _validate_identifier(tag, "tag")
    staging = root / f".{tag}.{generation}.tmp"
    if staging.exists() or staging.is_symlink():
        staging.unlink()
    os.symlink(str(Path(".generations") / generation), staging)
    os.replace(staging, root / tag)
    _fsync_dir(root)


def _resolve_generation(root: Path, tag: str) -> Path:
    _validate_identifier(tag, "tag")
    pointer = root / tag
    if not pointer.is_symlink():
        raise RuntimeError(f"checkpoint pointer is not a symlink: {pointer}")
    target = Path(os.readlink(pointer))
    if target.is_absolute() or len(target.parts) != 2 or target.parts[0] != ".generations":
        raise RuntimeError("checkpoint pointer escapes the generation root")
    _validate_identifier(target.parts[1], "generation")
    generation = (root / target).resolve()
    allowed = (root / ".generations").resolve()
    if generation.parent != allowed or not generation.is_dir():
        raise RuntimeError("checkpoint generation is missing or outside its root")
    return generation


def save_training_generation(
    root: str | os.PathLike[str],
    tag: str,
    agent: TractorAgent,
    sampler,
    metadata: Mapping[str, Any],
    *,
    scaler=None,
    generation: str | None = None,
) -> str:
    _validate_identifier(tag, "tag")
    generation = generation or uuid.uuid4().hex
    _validate_identifier(generation, "generation")
    root_path = Path(root)
    generations = root_path / ".generations"
    root_path.mkdir(parents=True, exist_ok=True)
    generations.mkdir(exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{generation}.", dir=generations))
    final = generations / generation
    if final.exists():
        raise FileExistsError(f"checkpoint generation exists: {final}")
    try:
        components = agent.checkpoint_components()
        payload = {
            "schema_id": CHECKPOINT_SCHEMA,
            "artifact_role": "training_checkpoint",
            "generation": generation,
            "model_fingerprint": agent.model_config.fingerprint(),
            "components": {name: component.state_dict() for name, component in components.items()},
            "agent_extra": agent.extra_state_dict(),
            "sampler_state": sampler.state_dict(),
            "global_rng": _capture_global_rng(),
            "scaler": None if scaler is None else scaler.state_dict(),
        }
        model_path = staging / "training.pt"
        torch.save(payload, model_path)
        with model_path.open("rb") as stream:
            os.fsync(stream.fileno())
        manifest = {
            "schema_id": CHECKPOINT_SCHEMA,
            "artifact_role": "training_checkpoint",
            "generation": generation,
            "model_family": "TRACTOR-TQC",
            "architecture_revision": "tractor-tqc-r1",
            "variant_id": agent.model_config.variant_id,
            "model_fingerprint": agent.model_config.fingerprint(),
            "present_components": sorted(components),
            "training_payload_sha256": _sha256(model_path),
            "training_payload_size_bytes": model_path.stat().st_size,
            "metadata": dict(metadata),
        }
        _write_json(staging / "manifest.json", manifest)
        _fsync_dir(staging)
        os.rename(staging, final)
        _fsync_dir(generations)
        _publish_pointer(root_path, tag, generation)
    except BaseException:
        if staging.exists():
            import shutil
            shutil.rmtree(staging)
        raise
    return generation


def load_training_generation(
    root: str | os.PathLike[str],
    tag: str,
    agent: TractorAgent,
    sampler,
    *,
    scaler=None,
    restore_rng: bool = True,
) -> dict:
    generation = _resolve_generation(Path(root), tag)
    manifest_path = generation / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_id") != CHECKPOINT_SCHEMA or manifest.get("artifact_role") != "training_checkpoint":
        raise RuntimeError("wrong checkpoint schema or artifact role")
    if manifest.get("generation") != generation.name:
        raise RuntimeError("checkpoint generation identity mismatch")
    if manifest.get("model_fingerprint") != agent.model_config.fingerprint():
        raise RuntimeError("checkpoint model fingerprint mismatch")
    payload_path = generation / "training.pt"
    if _sha256(payload_path) != manifest.get("training_payload_sha256"):
        raise RuntimeError("checkpoint payload checksum mismatch")
    # Tensor deserialization occurs only after role, path, generation,
    # fingerprint and byte hash validation above.
    payload = torch.load(payload_path, map_location=agent.device, weights_only=False)
    if payload.get("schema_id") != CHECKPOINT_SCHEMA or payload.get("artifact_role") != "training_checkpoint":
        raise RuntimeError("payload role does not match its validated manifest")
    if payload.get("generation") != generation.name:
        raise RuntimeError("payload generation does not match its directory")
    components = agent.checkpoint_components()
    if set(payload["components"]) != set(components):
        raise RuntimeError("checkpoint component inventory mismatch")
    if manifest.get("present_components") != sorted(components):
        raise RuntimeError("manifest component inventory mismatch")
    for name, component in components.items():
        component.load_state_dict(payload["components"][name])
    agent.load_extra_state_dict(payload["agent_extra"])
    sampler.load_state_dict(payload["sampler_state"])
    if scaler is not None:
        if payload.get("scaler") is None:
            raise RuntimeError("checkpoint is missing required AMP scaler state")
        scaler.load_state_dict(payload["scaler"])
    elif payload.get("scaler") is not None:
        raise RuntimeError("checkpoint has scaler state but caller supplied no scaler")
    if restore_rng:
        _restore_global_rng(payload["global_rng"])
    return manifest


def export_deployment_bundle(
    output_root: str | os.PathLike[str],
    bundle_id: str,
    agent: TractorAgent,
    calibration: PlattCalibration,
    metadata: Mapping[str, Any],
    *,
    calibration_artifact_path: str | os.PathLike[str],
) -> Path:
    _validate_identifier(bundle_id, "bundle_id")
    required_metadata = {
        "promotion_status", "source_checkpoint_sha256", "calibration_split_sha256",
        "calibration_artifact_sha256",
        "robot_attestation_hash", "controller_attestation_hash", "resolved_config",
        "protocol_version", "approval_owner", "latency_evidence",
    }
    missing = sorted(required_metadata - set(metadata))
    if missing:
        raise ValueError(f"deployment export metadata is incomplete: {missing}")
    if metadata["promotion_status"] not in {"promoted-sim", "promoted-hil", "promoted-real"}:
        raise ValueError("deployment export requires an approved promotion status")
    if metadata["source_checkpoint_sha256"] != calibration.source_checkpoint_sha256:
        raise ValueError("calibrator does not belong to the promoted source checkpoint")
    if metadata["calibration_split_sha256"] != calibration.split_sha256:
        raise ValueError("calibration split lineage mismatch")
    artifact_calibration, artifact_manifest = load_calibration_artifact(
        calibration_artifact_path,
        expected_checkpoint_sha256=metadata["source_checkpoint_sha256"],
    )
    if artifact_calibration != calibration:
        raise ValueError("supplied calibration differs from its immutable artifact")
    if _sha256(Path(calibration_artifact_path)) != metadata["calibration_artifact_sha256"]:
        raise ValueError("calibration artifact byte hash mismatch")
    if artifact_manifest["calibration"]["split_sha256"] != metadata["calibration_split_sha256"]:
        raise ValueError("calibration artifact split mismatch")
    latency = metadata["latency_evidence"]
    if not isinstance(latency, Mapping) or not latency.get("target_hardware_evidence", False):
        raise ValueError("deployment export requires target-hardware latency evidence")
    if not latency.get("gate_passed", False):
        raise ValueError("target-hardware latency gate did not pass")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    final = root / bundle_id
    if final.exists():
        raise FileExistsError(f"immutable deployment bundle exists: {final}")
    staging = Path(tempfile.mkdtemp(prefix=f".{bundle_id}.", dir=root))
    try:
        weights = staging / "inference.pt"
        torch.save({
            "schema_id": BUNDLE_SCHEMA,
            "artifact_role": "deployment_bundle",
            "model_fingerprint": agent.model_config.fingerprint(),
            "online": agent.online.state_dict(),
        }, weights)
        with weights.open("rb") as stream:
            os.fsync(stream.fileno())
        manifest = {
            "schema_id": BUNDLE_SCHEMA,
            "artifact_role": "deployment_bundle",
            "bundle_id": bundle_id,
            "model_family": "TRACTOR-TQC",
            "variant_id": agent.model_config.variant_id,
            "model_fingerprint": agent.model_config.fingerprint(),
            "inference_sha256": _sha256(weights),
            "calibration": asdict(calibration),
            "calibration_sha256": calibration.sha256(),
            "forbidden_training_payloads": [],
            "metadata": dict(metadata),
        }
        _write_json(staging / "manifest.json", manifest)
        _fsync_dir(staging)
        os.rename(staging, final)
        _fsync_dir(root)
    except BaseException:
        if staging.exists():
            import shutil
            shutil.rmtree(staging)
        raise
    return final


def save_calibration_artifact(
    output_root: str | os.PathLike[str],
    artifact_id: str,
    calibration: PlattCalibration,
    *,
    episode_ids: list[str],
    metrics: Mapping[str, Any],
    calibration_context_id: str,
) -> Path:
    _validate_identifier(artifact_id, "artifact_id")
    if not episode_ids or len(set(episode_ids)) != len(episode_ids):
        raise ValueError("calibration artifact requires unique non-empty episode ids")
    if not calibration_context_id:
        raise ValueError("calibration_context_id must be non-empty")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    final = root / f"{artifact_id}.json"
    if final.exists():
        raise FileExistsError(f"immutable calibration artifact exists: {final}")
    payload = {
        "schema_id": CALIBRATION_SCHEMA,
        "artifact_role": "calibration_artifact",
        "artifact_id": artifact_id,
        "calibration": asdict(calibration),
        "calibration_sha256": calibration.sha256(),
        "calibration_context_id": calibration_context_id,
        "split_id": "calibration",
        "episode_ids": sorted(episode_ids),
        "metrics": dict(metrics),
    }
    fd, temp_name = tempfile.mkstemp(prefix=f".{artifact_id}.", suffix=".tmp", dir=root)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, final)
        _fsync_dir(root)
    except BaseException:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
        raise
    return final


def load_calibration_artifact(
    path: str | os.PathLike[str], *, expected_checkpoint_sha256: str | None = None
) -> tuple[PlattCalibration, dict]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_id") != CALIBRATION_SCHEMA or payload.get("artifact_role") != "calibration_artifact":
        raise RuntimeError("wrong calibration artifact role/schema")
    if payload.get("split_id") != "calibration" or not payload.get("episode_ids"):
        raise RuntimeError("calibration artifact has no isolated calibration split lineage")
    calibration = PlattCalibration(**payload["calibration"])
    if calibration.sha256() != payload.get("calibration_sha256"):
        raise RuntimeError("calibration payload checksum mismatch")
    if expected_checkpoint_sha256 is not None and calibration.source_checkpoint_sha256 != expected_checkpoint_sha256:
        raise RuntimeError("calibration source checkpoint mismatch")
    return calibration, payload


def validate_deployment_bundle(
    path: str | os.PathLike[str], expected_model_fingerprint: str | None = None
) -> dict:
    bundle = Path(path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_id") != BUNDLE_SCHEMA or manifest.get("artifact_role") != "deployment_bundle":
        raise RuntimeError("wrong deployment artifact role/schema")
    if expected_model_fingerprint is not None and manifest.get("model_fingerprint") != expected_model_fingerprint:
        raise RuntimeError("deployment model fingerprint mismatch")
    if manifest.get("forbidden_training_payloads") != []:
        raise RuntimeError("deployment manifest declares forbidden training payloads")
    if _sha256(bundle / "inference.pt") != manifest.get("inference_sha256"):
        raise RuntimeError("deployment inference checksum mismatch")
    calibration = PlattCalibration(**manifest["calibration"])
    if calibration.sha256() != manifest.get("calibration_sha256"):
        raise RuntimeError("deployment calibration checksum mismatch")
    disallowed = {"replay.npz", "training.pt", "optimizer.pt", "target.pt"}
    found = sorted(item.name for item in bundle.iterdir() if item.name in disallowed)
    if found:
        raise RuntimeError(f"deployment bundle contains training-only files: {found}")
    return manifest


def load_deployment_policy(
    path: str | os.PathLike[str], model_config, selector_config, *, device="cpu"
):
    """Validate role/lineage bytes, then build the strict runtime policy."""
    from hunter_kinodynamic_rl.navigation.local_rl.tractor_policy import TractorPolicy

    bundle = Path(path)
    manifest = validate_deployment_bundle(bundle, model_config.fingerprint())
    payload = torch.load(bundle / "inference.pt", map_location=device, weights_only=True)
    if payload.get("schema_id") != BUNDLE_SCHEMA or payload.get("artifact_role") != "deployment_bundle":
        raise RuntimeError("deployment tensor payload role/schema mismatch")
    from hunter_kinodynamic_rl.rl.networks.tractor.model import TractorTQC

    model = TractorTQC(model_config).to(device)
    model.load_state_dict(payload["online"], strict=True)
    calibration = PlattCalibration(**manifest["calibration"])
    return TractorPolicy(model, selector_config, calibration, deployment=True), manifest
