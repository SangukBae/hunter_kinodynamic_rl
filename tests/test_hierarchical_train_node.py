"""``nodes/hierarchical_train_node.py`` (items 4/9): resume fail-fast
compatibility checks and the Global-training update cadence fix.

``_check_resume_compatibility`` is pure Python (no ROS/Gazebo needed to call
it directly) -- only the module-level import of ``rclpy``/``rclpy.node.Node``
requires ``rclpy`` to be importable at all (mirrors every other node test in
this package's own ``importorskip`` convention)."""

from __future__ import annotations

import dataclasses
import types

import pytest

pytest.importorskip("rclpy")
pytest.importorskip("torch")

from hunter_kinodynamic_rl.config.loader import load_profile  # noqa: E402
from hunter_kinodynamic_rl.evaluation.fingerprint import (  # noqa: E402
    hierarchical_architecture_fingerprint, sha256_of_obj,
)
from hunter_kinodynamic_rl.navigation.global_rl.replay_schema import (  # noqa: E402
    SCHEMA_VERSION as GLOBAL_REPLAY_SCHEMA_VERSION,
)
from hunter_kinodynamic_rl.nodes.hierarchical_train_node import (  # noqa: E402
    ResumeIncompatibleError, _check_resume_compatibility,
)


def _profile():
    return load_profile("hierarchical_phase4")


def _fake_loop(profile):
    return types.SimpleNamespace(
        map_channel_names=["occupied", "free", "unknown", "visited"],
        candidate_feature_names=["distance_m"],
        max_nodes=0,
        memory_enabled=False, feasibility_enabled=False, global_risk_enabled=False,
    )


def _valid_manifest(profile, loop, *, live_executor=None) -> dict:
    resolved_config = dataclasses.asdict(profile)
    manifest = {
        "hierarchical_architecture_fingerprint": hierarchical_architecture_fingerprint(profile),
        "resolved_config": resolved_config,
        "resolved_config_hash": sha256_of_obj(resolved_config),
        "map_channel_names": list(loop.map_channel_names),
        "candidate_feature_names": list(loop.candidate_feature_names),
        "node_feature_names": [],
        "max_nodes_in_observation": loop.max_nodes,
        "memory_enabled": loop.memory_enabled, "feasibility_enabled": loop.feasibility_enabled,
        "global_risk_enabled": loop.global_risk_enabled,
        "loop_state": {}, "training_steps": 0, "total_optimizer_updates": 0,
        "epsilon_schedule_basis": "agent_training_steps",
        "global_replay_schema_version": GLOBAL_REPLAY_SCHEMA_VERSION,
        "rng_state_path": ".rng_states/rng_state-test.pt",
        "rng_state_sha256": "test-sha256", "rng_state_schema_version": 1,
    }
    if live_executor is not None:
        manifest["local_checkpoint_manifest_summary"] = dict(live_executor.checkpoint_manifest)
    else:
        manifest["local_checkpoint_manifest_summary"] = None
    return manifest


def _fake_live_executor(**overrides) -> types.SimpleNamespace:
    manifest = {
        "generation": "gen-a", "pt_sha256": "sha-a", "state_dim": 327, "action_dim": 3,
        "local_training_contract_fingerprint": "contract-a",
    }
    manifest.update(overrides)
    return types.SimpleNamespace(checkpoint_manifest=manifest)


def test_passes_when_everything_matches_live_false():
    profile = _profile()
    loop = _fake_loop(profile)
    manifest = _valid_manifest(profile, loop)
    _check_resume_compatibility(manifest, profile, loop, None)  # must not raise


def test_passes_when_everything_matches_live_true():
    profile = _profile()
    loop = _fake_loop(profile)
    live_executor = _fake_live_executor()
    manifest = _valid_manifest(profile, loop, live_executor=live_executor)
    _check_resume_compatibility(manifest, profile, loop, live_executor)  # must not raise


def test_rejects_fingerprint_mismatch():
    profile = _profile()
    loop = _fake_loop(profile)
    manifest = _valid_manifest(profile, loop)
    manifest["hierarchical_architecture_fingerprint"] = "deliberately-wrong"
    with pytest.raises(ResumeIncompatibleError, match="hierarchical_architecture_fingerprint"):
        _check_resume_compatibility(manifest, profile, loop, None)


def test_rejects_missing_fingerprint_field_exactly_like_a_mismatch():
    """A checkpoint saved before item 4's checks existed (no fingerprint
    field at all) must be rejected the SAME way a genuine mismatch is --
    never implicitly trusted through."""
    profile = _profile()
    loop = _fake_loop(profile)
    manifest = _valid_manifest(profile, loop)
    del manifest["hierarchical_architecture_fingerprint"]
    with pytest.raises(ResumeIncompatibleError, match="hierarchical_architecture_fingerprint"):
        _check_resume_compatibility(manifest, profile, loop, None)


@pytest.mark.parametrize("field_name", [
    "resolved_config", "resolved_config_hash", "map_channel_names", "candidate_feature_names",
    "node_feature_names", "max_nodes_in_observation", "loop_state", "training_steps",
    "total_optimizer_updates", "epsilon_schedule_basis", "global_replay_schema_version",
    "rng_state_path", "rng_state_sha256", "rng_state_schema_version",
    "local_checkpoint_manifest_summary",
])
def test_rejects_every_missing_required_resume_field(field_name):
    profile = _profile()
    loop = _fake_loop(profile)
    manifest = _valid_manifest(profile, loop)
    del manifest[field_name]
    with pytest.raises(ResumeIncompatibleError, match="missing required manifest field"):
        _check_resume_compatibility(manifest, profile, loop, None)


def test_rejects_resolved_config_hash_mismatch():
    profile = _profile()
    loop = _fake_loop(profile)
    manifest = _valid_manifest(profile, loop)
    tampered_profile = dataclasses.replace(
        profile, hierarchy=dataclasses.replace(profile.hierarchy, mission_timeout_steps=999999),
    )
    manifest["resolved_config"] = dataclasses.asdict(tampered_profile)
    with pytest.raises(ResumeIncompatibleError, match="resolved_config hash"):
        _check_resume_compatibility(manifest, profile, loop, None)


def test_rejects_observation_schema_mismatch():
    profile = _profile()
    loop = _fake_loop(profile)
    manifest = _valid_manifest(profile, loop)
    manifest["map_channel_names"] = ["occupied", "free", "unknown", "visited", "extra_channel"]
    with pytest.raises(ResumeIncompatibleError, match="map_channel_names"):
        _check_resume_compatibility(manifest, profile, loop, None)


def test_rejects_local_checkpoint_generation_mismatch_when_live():
    profile = _profile()
    loop = _fake_loop(profile)
    live_executor = _fake_live_executor(generation="gen-current")
    manifest = _valid_manifest(profile, loop, live_executor=_fake_live_executor(generation="gen-old"))
    with pytest.raises(ResumeIncompatibleError, match="local_checkpoint.generation"):
        _check_resume_compatibility(manifest, profile, loop, live_executor)


def test_rejects_local_checkpoint_sha256_mismatch_when_live():
    profile = _profile()
    loop = _fake_loop(profile)
    live_executor = _fake_live_executor(pt_sha256="sha-current")
    manifest = _valid_manifest(profile, loop, live_executor=_fake_live_executor(pt_sha256="sha-old"))
    with pytest.raises(ResumeIncompatibleError, match="local_checkpoint.pt_sha256"):
        _check_resume_compatibility(manifest, profile, loop, live_executor)


def test_rejects_resuming_live_run_as_non_live():
    """Checkpoint recorded a real Local-checkpoint-driven run (live=True)
    but this resume attempt passes live_executor=None (live=False) --
    silently dropping the Local checkpoint's own provenance must be
    refused, never treated as compatible."""
    profile = _profile()
    loop = _fake_loop(profile)
    manifest = _valid_manifest(profile, loop, live_executor=_fake_live_executor())
    with pytest.raises(ResumeIncompatibleError, match="live=True"):
        _check_resume_compatibility(manifest, profile, loop, None)


def test_rejects_resuming_non_live_run_as_live():
    profile = _profile()
    loop = _fake_loop(profile)
    live_executor = _fake_live_executor()
    manifest = _valid_manifest(profile, loop)  # live_executor=None at save time
    with pytest.raises(ResumeIncompatibleError, match="not recorded"):
        _check_resume_compatibility(manifest, profile, loop, live_executor)


def _load_model_payload(run_dir: str):
    import os

    import torch

    gen_dir = os.path.realpath(os.path.join(run_dir, "checkpoints", "latest"))
    return torch.load(os.path.join(gen_dir, "model.pt"), weights_only=True)


def _state_dicts_equal(a, b) -> bool:
    """Deep tensor-value equality between two component payloads (online/
    target/optimizer state_dicts) -- NEVER raw-file sha256/byte comparison:
    ``torch.save``'s zip-based container is not byte-stable across two
    independent save calls for byte-IDENTICAL tensor content (confirmed by
    direct investigation while building this test -- two back-to-back saves
    of the exact same in-memory state_dict produced different file bytes
    yet ``torch.equal``-identical tensors once reloaded). A file-hash
    comparison would report a false divergence here regardless of how
    deterministic the actual training math is."""
    import torch

    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return False
        return all(_state_dicts_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_state_dicts_equal(x, y) for x, y in zip(a, b))
    if torch.is_tensor(a) and torch.is_tensor(b):
        return torch.equal(a, b)
    return a == b


def _low_warmup_profile_patch(monkeypatch, *, cadence: str = "per_transition"):
    """Patches nodes.hierarchical_train_node.load_profile to return
    hierarchical_phase4 with a drastically reduced warmup_options/batch_size
    -- the shipped profile's warmup_options=200 needs several full-budget
    missions before ANY optimizer update ever runs (this world/hierarchy
    config's missions routinely end well short of the 40-option budget), so
    a short, fast test would otherwise exercise ZERO actual training and
    vacuously "pass" without checking anything real."""
    import dataclasses

    import hunter_kinodynamic_rl.nodes.hierarchical_train_node as node_mod
    from hunter_kinodynamic_rl.config.loader import load_profile as real_load_profile

    def patched(name, config_root=None):
        profile = real_load_profile(name, config_root)
        return dataclasses.replace(
            profile,
            global_rl=dataclasses.replace(profile.global_rl, warmup_options=5, batch_size=4),
            hierarchical_training=dataclasses.replace(
                profile.hierarchical_training, training_update_cadence=cadence,
            ),
        )

    monkeypatch.setattr(node_mod, "load_profile", patched)


def test_deterministic_resume_matches_an_uninterrupted_run(tmp_path, monkeypatch):
    """item 4's core determinism requirement: an uninterrupted N-mission run
    and a K-mission run resumed for the remaining N-K missions must produce
    an IDENTICAL final Global checkpoint (online/target/optimizer state
    dicts, tensor-value compared -- see _state_dicts_equal's own docstring
    for why NOT a raw file hash) -- proof that action sequence, replay
    sampling, and training steps all matched exactly across the resume
    boundary, not just that the run didn't crash. Forces CPU
    (torch.cuda.is_available monkeypatched False) so the comparison isn't
    sensitive to GPU-kernel nondeterminism this test isn't trying to
    characterize, and forces a low warmup so real optimizer updates
    actually happen within a handful of missions (see
    _low_warmup_profile_patch)."""
    import torch

    from hunter_kinodynamic_rl.nodes.hierarchical_train_node import train_hierarchical_dqn

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    _low_warmup_profile_patch(monkeypatch)

    run_a = train_hierarchical_dqn(
        "hierarchical_phase4", run_root=str(tmp_path / "a"), num_missions=6, live=False, logger=lambda *a: None,
    )
    run_b_phase1 = train_hierarchical_dqn(
        "hierarchical_phase4", run_root=str(tmp_path / "b"), num_missions=3, live=False, logger=lambda *a: None,
    )
    run_b_phase2 = train_hierarchical_dqn(
        "hierarchical_phase4", run_root=str(tmp_path / "b"), num_missions=3,
        resume_run_dir=run_b_phase1["run_dir"], resume_checkpoint_tag="latest", live=False, logger=lambda *a: None,
    )

    assert run_a["global_step"] == run_b_phase2["global_step"]
    payload_a = _load_model_payload(run_a["run_dir"])
    payload_b = _load_model_payload(run_b_phase2["run_dir"])
    for key in ("online", "target", "optimizer"):
        assert _state_dicts_equal(payload_a[key], payload_b[key]), (
            f"uninterrupted 6-mission run and 3+3 resumed run produced a DIFFERENT final '{key}' state -- "
            "action sequence / replay sampling / training steps diverged across the resume boundary"
        )


def test_training_update_cadence_per_transition_runs_proportionally_more_updates(tmp_path, monkeypatch):
    """item 9: training_update_cadence='per_transition' must perform MORE
    optimizer updates than 'per_mission' for the SAME mission sequence --
    the exact regression this fixes (previously exactly one update per
    mission regardless of how many Global option-transitions it stored)."""
    import json
    import os

    import torch

    from hunter_kinodynamic_rl.nodes.hierarchical_train_node import train_hierarchical_dqn

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def _run(cadence):
        _low_warmup_profile_patch(monkeypatch, cadence=cadence)
        result = train_hierarchical_dqn(
            "hierarchical_phase4", run_root=str(tmp_path / cadence), num_missions=6, live=False,
            logger=lambda *a: None,
        )
        with open(os.path.join(result["run_dir"], "hierarchical_metadata.json")) as f:
            return json.load(f)

    meta_per_mission = _run("per_mission")
    meta_per_transition = _run("per_transition")

    assert meta_per_mission["total_optimizer_updates"] > 0, "test setup did not exercise any training at all"
    assert meta_per_transition["total_optimizer_updates"] > meta_per_mission["total_optimizer_updates"], (
        f"per_transition ({meta_per_transition['total_optimizer_updates']} updates) did not run more optimizer "
        f"updates than per_mission ({meta_per_mission['total_optimizer_updates']} updates) over the same "
        "6-mission sequence"
    )


def test_hierarchical_architecture_fingerprint_is_a_pure_function_of_hierarchical_sections():
    """Sanity check underpinning every fingerprint-mismatch test above --
    the SAME profile always fingerprints identically, and a genuinely
    different hierarchical-section profile fingerprints differently."""
    profile = _profile()
    assert hierarchical_architecture_fingerprint(profile) == hierarchical_architecture_fingerprint(profile)
    different = dataclasses.replace(
        profile, mapping=dataclasses.replace(profile.mapping, mission_size_cells=profile.mapping.mission_size_cells + 10),
    )
    assert hierarchical_architecture_fingerprint(profile) != hierarchical_architecture_fingerprint(different)
