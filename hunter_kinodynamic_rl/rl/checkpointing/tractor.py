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
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np
import torch

from hunter_kinodynamic_rl.rl.networks.tractor.calibration import PlattCalibration

if TYPE_CHECKING:
    from hunter_kinodynamic_rl.rl.algorithms.tractor_tqc import TractorAgent


CHECKPOINT_SCHEMA = "tractor_training_checkpoint_v2"
BUNDLE_SCHEMA = "tractor_deployment_bundle_v2"
CALIBRATION_SCHEMA = "tractor_calibration_artifact_v2"
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


def semantic_state_sha256(value: Any) -> str:
    """Deterministic hash of nested tensor/optimizer state independent of torch.save bytes."""
    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, Mapping):
            digest.update(b"mapping\0")
            for key in sorted(item, key=lambda entry: str(entry)):
                update(str(key))
                update(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(b"sequence\0")
            for entry in item:
                update(entry)
        else:
            digest.update(b"scalar\0")
            digest.update(json.dumps(item, sort_keys=True, default=str).encode("utf-8"))

    update(value)
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
    agent,
    sampler,
    metadata: Mapping[str, Any],
    *,
    scaler=None,
    generation: str | None = None,
    parent_tag: str | None = None,
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
        component_hashes = {
            name: semantic_state_sha256(component.state_dict())
            for name, component in components.items()
        }
        parent_generation = None
        parent_payload_sha256 = None
        root_generation = generation
        lineage_tag = parent_tag or tag
        _validate_identifier(lineage_tag, "parent tag")
        pointer = root_path / lineage_tag
        if pointer.is_symlink():
            parent = _resolve_generation(root_path, lineage_tag)
            parent_manifest = json.loads((parent / "manifest.json").read_text(encoding="utf-8"))
            if parent_manifest.get("schema_id") != CHECKPOINT_SCHEMA:
                raise RuntimeError("cannot extend an incompatible checkpoint lineage")
            parent_generation = parent.name
            parent_payload_sha256 = parent_manifest["training_payload_sha256"]
            root_generation = parent_manifest["semantic_lineage"]["root_generation"]
        training_stage = str(metadata.get("training_stage", "development_unspecified"))
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
            "component_state_sha256": component_hashes,
        }
        model_path = staging / "training.pt"
        torch.save(payload, model_path)
        with model_path.open("rb") as stream:
            os.fsync(stream.fileno())
        manifest = {
            "schema_id": CHECKPOINT_SCHEMA,
            "artifact_role": "training_checkpoint",
            "generation": generation,
            "model_family": getattr(agent, "model_family", "TRACTOR-TQC"),
            "architecture_revision": getattr(agent, "architecture_revision", "tractor-tqc-r2"),
            "variant_id": agent.model_config.variant_id,
            "model_fingerprint": agent.model_config.fingerprint(),
            "present_components": sorted(components),
            "component_state_sha256": component_hashes,
            "training_payload_sha256": _sha256(model_path),
            "training_payload_size_bytes": model_path.stat().st_size,
            "metadata": dict(metadata),
            "semantic_lineage": {
                "root_generation": root_generation,
                "parent_generation": parent_generation,
                "parent_training_payload_sha256": parent_payload_sha256,
                "training_stage": training_stage,
                "warm_start": metadata.get("warm_start"),
            },
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
    agent,
    sampler,
    *,
    scaler=None,
    restore_rng: bool = True,
    expected_metadata: Mapping[str, Any] | None = None,
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
    component_hashes = {
        name: semantic_state_sha256(state)
        for name, state in payload["components"].items()
    }
    if payload.get("component_state_sha256") != component_hashes:
        raise RuntimeError("checkpoint payload component semantic hash mismatch")
    if manifest.get("component_state_sha256") != component_hashes:
        raise RuntimeError("checkpoint manifest component semantic hash mismatch")
    if expected_metadata is not None:
        for name, expected in expected_metadata.items():
            if manifest.get("metadata", {}).get(name) != expected:
                raise RuntimeError(f"checkpoint metadata lineage mismatch for {name!r}")
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
    for module_name in ("online", "target"):
        module = getattr(agent, module_name)
        if any(not torch.isfinite(parameter).all().item() for parameter in module.parameters()):
            raise RuntimeError(f"checkpoint startup probe found non-finite {module_name} weights")
    return manifest


def load_inference_weights(
    root: str | os.PathLike[str], tag: str, agent,
) -> dict:
    """Load only validated online weights; never restore replay, optimizers or RNG."""
    generation = _resolve_generation(Path(root), tag)
    manifest = json.loads((generation / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_id") != CHECKPOINT_SCHEMA or manifest.get("artifact_role") != "training_checkpoint":
        raise RuntimeError("wrong checkpoint schema or artifact role")
    if manifest.get("generation") != generation.name:
        raise RuntimeError("checkpoint generation identity mismatch")
    if manifest.get("model_fingerprint") != agent.model_config.fingerprint():
        raise RuntimeError("checkpoint model fingerprint mismatch")
    payload_path = generation / "training.pt"
    if _sha256(payload_path) != manifest.get("training_payload_sha256"):
        raise RuntimeError("checkpoint payload checksum mismatch")
    payload = torch.load(payload_path, map_location=agent.device, weights_only=False)
    if payload.get("schema_id") != CHECKPOINT_SCHEMA or payload.get("generation") != generation.name:
        raise RuntimeError("checkpoint payload identity mismatch")
    if "online" not in payload.get("components", {}):
        raise RuntimeError("checkpoint is missing online inference weights")
    online_hash = semantic_state_sha256(payload["components"]["online"])
    if payload.get("component_state_sha256", {}).get("online") != online_hash:
        raise RuntimeError("checkpoint payload online semantic hash mismatch")
    if manifest.get("component_state_sha256", {}).get("online") != online_hash:
        raise RuntimeError("checkpoint manifest online semantic hash mismatch")
    agent.online.load_state_dict(payload["components"]["online"], strict=True)
    if any(not torch.isfinite(parameter).all().item() for parameter in agent.online.parameters()):
        raise RuntimeError("checkpoint startup probe found non-finite online weights")
    return manifest


def load_warm_start_generation(
    root: str | os.PathLike[str], tag: str, agent, *,
    source_stage: str, target_stage: str,
) -> dict:
    """Import only online weights across one registered training-stage boundary."""
    allowed = {("stage3", "stage4"), ("stage4", "stage5")}
    if (source_stage, target_stage) not in allowed:
        raise ValueError(
            f"unsupported TRACTOR warm start {source_stage!r}->{target_stage!r}"
        )
    generation = _resolve_generation(Path(root), tag)
    manifest = json.loads((generation / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("semantic_lineage", {}).get("training_stage") != source_stage:
        raise RuntimeError("warm-start source training stage mismatch")
    manifest = load_inference_weights(root, tag, agent)
    from hunter_kinodynamic_rl.rl.algorithms.tractor_tqc.target_update import ema_update

    if set(agent.target.state_dict()) == set(agent.online.state_dict()):
        agent.target.load_state_dict(agent.online.state_dict(), strict=True)
    else:
        ema_update(agent.target, agent.online, 1.0)
    return {
        "source_generation": manifest["generation"],
        "source_training_payload_sha256": manifest["training_payload_sha256"],
        "source_online_state_sha256": manifest["component_state_sha256"]["online"],
        "source_stage": source_stage, "target_stage": target_stage,
    }


def export_deployment_bundle(
    output_root: str | os.PathLike[str],
    bundle_id: str,
    agent: TractorAgent,
    calibration: PlattCalibration,
    metadata: Mapping[str, Any],
    *,
    calibration_artifact_path: str | os.PathLike[str],
    source_checkpoint_root: str | os.PathLike[str],
    source_checkpoint_tag: str = "final",
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
    source_generation = _resolve_generation(Path(source_checkpoint_root), source_checkpoint_tag)
    source_manifest = json.loads(
        (source_generation / "manifest.json").read_text(encoding="utf-8")
    )
    if source_manifest.get("schema_id") != CHECKPOINT_SCHEMA:
        raise ValueError("deployment source is not a semantic TRACTOR training checkpoint")
    source_payload = source_generation / "training.pt"
    if _sha256(source_payload) != source_manifest.get("training_payload_sha256"):
        raise ValueError("deployment source checkpoint payload checksum mismatch")
    if source_manifest.get("semantic_lineage", {}).get("training_stage") != "stage5":
        raise ValueError("deployment promotion requires a Stage-5 source checkpoint")
    in_memory_hash = semantic_state_sha256(agent.online.state_dict())
    source_online_hash = source_manifest.get("component_state_sha256", {}).get("online")
    if in_memory_hash != source_online_hash:
        raise ValueError("in-memory inference weights do not match the promoted Stage-5 checkpoint")
    if metadata["source_checkpoint_sha256"] != source_manifest["training_payload_sha256"]:
        raise ValueError("declared source checkpoint hash does not match the promoted artifact")
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
            "source_training_lineage": dict(source_manifest["semantic_lineage"]),
            "source_checkpoint_generation": source_generation.name,
            "source_online_state_sha256": source_online_hash,
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
    provenance: Mapping[str, Any] | None = None,
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
        "provenance": dict(provenance or {}),
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
    lineage = manifest.get("source_training_lineage", {})
    if lineage.get("training_stage") != "stage5" or not manifest.get(
        "source_checkpoint_generation"
    ):
        raise RuntimeError("deployment bundle lacks semantic Stage-5 promotion lineage")
    inference_payload = torch.load(bundle / "inference.pt", map_location="cpu", weights_only=True)
    if semantic_state_sha256(inference_payload.get("online", {})) != manifest.get(
        "source_online_state_sha256"
    ):
        raise RuntimeError("deployment weights differ from the promoted source state")
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
