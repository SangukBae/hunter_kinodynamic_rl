"""Immutable root identity for a complete formal TRACTOR episode corpus."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile

from hunter_kinodynamic_rl.rl.replay import EpisodeStore, SequenceIndex
from hunter_kinodynamic_rl.rl.replay.episode_store import sha256_file


FORMAL_DATASET_MANIFEST_SCHEMA = "tractor_formal_dataset_manifest_v1"
_EMPTY_BYTES_SHA256 = hashlib.sha256(b"").hexdigest()
_EMPTY_JSON_SHA256 = hashlib.sha256(b"{}").hexdigest()


def _canonical_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def formal_source_identity(provenance: dict) -> dict:
    """Return a reproducible source/container identity or fail before a formal run."""
    if provenance.get("tracked_diff_sha256") != _EMPTY_BYTES_SHA256:
        raise RuntimeError("formal research requires a committed tracked source tree")
    if provenance.get("untracked_source_manifest_sha256") != _EMPTY_JSON_SHA256:
        raise RuntimeError("formal research forbids untracked source files")
    container = os.environ.get("HUNTER_CONTAINER_IMAGE_DIGEST", "")
    if not container.startswith("sha256:") or len(container) != 71:
        raise RuntimeError(
            "formal research requires HUNTER_CONTAINER_IMAGE_DIGEST=sha256:<64 hex>"
        )
    if any(character not in "0123456789abcdef" for character in container[7:]):
        raise RuntimeError("HUNTER_CONTAINER_IMAGE_DIGEST must be a lowercase SHA-256 digest")
    if provenance.get("execution_matches_source_module") is not True:
        raise RuntimeError("formal execution module does not match the committed package source")
    return {
        "package_git_commit_sha": str(provenance["package_git_commit_sha"]),
        "tracked_diff_sha256": str(provenance["tracked_diff_sha256"]),
        "untracked_source_manifest_sha256": str(
            provenance["untracked_source_manifest_sha256"]
        ),
        "source_content_manifest_sha256": str(
            provenance["source_content_manifest_sha256"]
        ),
        "source_content_file_count": int(provenance["source_content_file_count"]),
        "execution_module_sha256": str(provenance["execution_module_sha256"]),
        "container_image_digest": container,
    }


def validate_runtime_source_against_dataset(dataset_manifest: dict, source_identity: dict) -> None:
    frozen = dataset_manifest.get("source_identity", {})
    for field in (
        "package_git_commit_sha", "tracked_diff_sha256",
        "untracked_source_manifest_sha256", "source_content_manifest_sha256",
        "container_image_digest",
    ):
        if source_identity.get(field) != frozen.get(field):
            raise RuntimeError(f"formal runtime source differs from dataset field {field!r}")


def _atomic_json(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"immutable formal dataset manifest exists: {path}")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def build_formal_dataset_manifest(
    dataset_root: str | Path, *, scenario_manifest_sha256: str,
    contract_sha256: str, protocol_version: str, behavior_seed: int,
    source_identity: dict, loss_window: int,
) -> dict:
    root = Path(dataset_root)
    store = EpisodeStore(root)
    artifacts = []
    split_counts = Counter()
    row_count = 0
    for path in store.iter_paths():
        digest = sha256_file(path)
        header, _columns = store.load(path.stem, expected_sha256=digest)
        artifacts.append({
            "episode_id": header.episode_id, "scenario_id": header.scenario_id,
            "split_id": header.split_id, "step_count": header.step_count,
            "artifact_sha256": digest,
        })
        split_counts[header.split_id] += 1
        row_count += header.step_count
    if not artifacts:
        raise RuntimeError("cannot freeze an empty formal dataset")
    index = SequenceIndex.build(store, loss_window=loss_window)
    payload = {
        "schema_id": FORMAL_DATASET_MANIFEST_SCHEMA,
        "scenario_manifest_sha256": scenario_manifest_sha256,
        "contract_sha256": contract_sha256,
        "protocol_version": protocol_version,
        "behavior_seed": int(behavior_seed),
        "source_identity": dict(source_identity),
        "loss_window": int(loss_window),
        "sequence_index_sha256": index.sha256(),
        "episode_count": len(artifacts), "row_count": row_count,
        "split_episode_counts": dict(sorted(split_counts.items())),
        "episode_artifacts": artifacts,
    }
    payload["dataset_manifest_sha256"] = _canonical_sha256(payload)
    _atomic_json(root / "dataset_manifest.json", payload)
    return payload


def validate_formal_dataset_manifest(
    dataset_root: str | Path, *, scenario_manifest_sha256: str,
    contract_sha256: str, protocol_version: str, loss_window: int,
) -> dict:
    root = Path(dataset_root)
    path = root / "dataset_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    digest_payload = dict(payload)
    stored_digest = digest_payload.pop("dataset_manifest_sha256", None)
    if payload.get("schema_id") != FORMAL_DATASET_MANIFEST_SCHEMA:
        raise RuntimeError("formal dataset manifest schema mismatch")
    if stored_digest != _canonical_sha256(digest_payload):
        raise RuntimeError("formal dataset manifest checksum mismatch")
    expected = {
        "scenario_manifest_sha256": scenario_manifest_sha256,
        "contract_sha256": contract_sha256,
        "protocol_version": protocol_version,
        "loss_window": int(loss_window),
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise RuntimeError(f"formal dataset manifest changed frozen field {name!r}")
    store = EpisodeStore(root)
    observed_paths = {path.stem: path for path in store.iter_paths()}
    entries = payload.get("episode_artifacts")
    if not isinstance(entries, list) or len(entries) != int(payload.get("episode_count", -1)):
        raise RuntimeError("formal dataset manifest episode inventory is invalid")
    if set(observed_paths) != {str(entry.get("episode_id")) for entry in entries}:
        raise RuntimeError("formal dataset episode inventory changed")
    row_count = 0
    split_counts = Counter()
    for entry in entries:
        episode_id = str(entry["episode_id"])
        path = observed_paths[episode_id]
        if sha256_file(path) != entry.get("artifact_sha256"):
            raise RuntimeError(f"formal dataset episode checksum mismatch: {episode_id}")
        header, _columns = store.load(episode_id, entry["artifact_sha256"])
        if (
            header.scenario_id != entry.get("scenario_id")
            or header.split_id != entry.get("split_id")
            or header.step_count != int(entry.get("step_count", -1))
        ):
            raise RuntimeError(f"formal dataset episode identity mismatch: {episode_id}")
        row_count += header.step_count
        split_counts[header.split_id] += 1
    if row_count != int(payload.get("row_count", -1)):
        raise RuntimeError("formal dataset row count changed")
    if dict(sorted(split_counts.items())) != payload.get("split_episode_counts"):
        raise RuntimeError("formal dataset split counts changed")
    index = SequenceIndex.build(store, loss_window=loss_window)
    if index.sha256() != payload.get("sequence_index_sha256"):
        raise RuntimeError("formal dataset sequence index identity changed")
    return payload
