"""Role, integrity, resume and deployment separation tests."""

import json
import hashlib
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from hunter_kinodynamic_rl.rl.algorithms.tractor_tqc import TractorAgent, TractorAgentConfig
from hunter_kinodynamic_rl.rl.checkpointing.tractor import (
    export_deployment_bundle, load_deployment_policy, load_training_generation, save_training_generation,
    load_calibration_artifact, load_inference_weights, load_warm_start_generation,
    save_calibration_artifact,
    validate_deployment_bundle,
)
from hunter_kinodynamic_rl.rl.networks.tractor import TractorConfig
from hunter_kinodynamic_rl.rl.networks.tractor.calibration import PlattCalibration
from hunter_kinodynamic_rl.rl.networks.tractor.selector import SelectorConfig
from hunter_kinodynamic_rl.rl.replay import EpisodeHeader, EpisodeStore, SequenceBuffer, SequenceIndex


def _config():
    return TractorConfig(
        t_obs=2, n_scan=8, grid_height=8, grid_width=8, resolution_m=1.0,
        x_min_m=-4.0, y_min_m=-4.0, ray_free_samples=2, scene_channels=8,
        ego_dim=8, plant_dim=8, interaction_dim=8, num_candidates=2,
        horizon_steps=2, dt_out_sec=0.2, dt_dyn_sec=0.1, sparse_tube_samples=4,
        n_quantiles=3, n_severity_quantiles=2, max_speed_mps=1.0,
        severity_quantile_levels=(0.1, 0.9),
        min_arc_m=0.1, max_arc_m=0.4, min_horizon_sec=0.2,
        min_safety_horizon_sec=0.2, max_horizon_sec=0.4,
    )


def _sampler(tmp_path: Path):
    steps = 2
    store = EpisodeStore(tmp_path / "replay")
    header = EpisodeHeader(
        episode_id="ep", scenario_id="fixture", split_id="development", seed=0,
        software_commit="commit", dirty_state_digest="clean", container_image_digest="container",
        resolved_config_hash="config", protocol_version="protocol",
        environment_attestation_hash="env", robot_attestation_hash="robot",
        observation_contract_hash="obs", action_contract_hash="action",
        trajectory_contract_hash="trajectory", start_utc="start", end_utc="end",
        termination_reason="goal", step_count=steps, sensor_source="scan",
        localization_source="localization", controller_source="controller", clock_domain="sim",
        group_id="9" * 64, scenario_family="open_static", obstacle_contract="static_only",
        scenario_geometry_sha256="9" * 64,
    )
    columns = {
        "decision_timestamp_ns": np.asarray([1, 2], dtype=np.int64),
        "observation": np.zeros((steps, 24), np.float32),
        "scan_valid": np.ones((steps, 2, 8), bool),
        "motion_delta_from_previous": np.tile(
            np.asarray([0.0, 0.0, 0.0, 0.1], np.float32), (steps, 1, 1)
        ),
        "motion_valid": np.ones((steps, 1), bool),
        "previous_intent_valid": np.ones(steps, bool),
        "previous_command_published": np.zeros((steps, 2), np.float32),
        "previous_command_valid": np.ones(steps, bool),
        "vehicle_response_valid": np.ones((steps, 3), bool),
        "pose_covariance": np.zeros((steps, 3, 3), np.float32),
        "localization_valid": np.ones(steps, bool),
        "localization_confidence": np.ones(steps, np.float32),
        "localization_confidence_valid": np.ones(steps, bool),
        "sensor_freshness_sec": np.zeros(steps, np.float32),
        "sensor_freshness_valid": np.ones(steps, bool),
        "reset_epoch": np.zeros(steps, np.int64),
        "scene_reset": np.asarray([True, False]),
        "response_reset": np.asarray([True, False]),
        "action_normalized_requested": np.zeros((steps, 3), np.float32),
        "reward": np.zeros(steps, np.float32),
        "transition_dt_sec": np.ones(steps, np.float32),
        "discount_factor": np.full(steps, 0.99, np.float32),
        "next_observation_valid": np.ones(steps, bool),
        "terminated": np.asarray([False, True]),
        "truncated": np.zeros(steps, bool),
        "termination_reason": np.asarray(["none", "goal"]),
        "bellman_sample_valid": np.ones(steps, bool),
    }
    store.append(header, columns)
    return SequenceBuffer(store, SequenceIndex.build(store, loss_window=1), seed=9)


def test_training_checkpoint_exact_state_round_trip(tmp_path: Path):
    config = _config()
    first = TractorAgent(config, TractorAgentConfig(top_quantiles_to_drop_per_net=1), target_seed=4)
    sampler = _sampler(tmp_path)
    sampler.sample(1)
    first.update_step = 12
    generation = save_training_generation(
        tmp_path / "checkpoints", "latest", first, sampler,
        {"experiment_id": "fixture", "evidence_level": "unit_test"},
    )

    restored = TractorAgent(config, TractorAgentConfig(top_quantiles_to_drop_per_net=1), target_seed=999)
    restored_sampler = SequenceBuffer(sampler.store, sampler.index, seed=999)
    manifest = load_training_generation(
        tmp_path / "checkpoints", "latest", restored, restored_sampler, restore_rng=False
    )
    assert manifest["generation"] == generation
    assert restored.update_step == 12
    assert restored_sampler.draw_ordinal == 1
    for name, value in first.online.state_dict().items():
        assert torch.equal(value, restored.online.state_dict()[name]), name


def test_final_checkpoint_links_to_latest_semantic_lineage(tmp_path: Path):
    config = _config()
    agent = TractorAgent(config, TractorAgentConfig(top_quantiles_to_drop_per_net=1))
    sampler = _sampler(tmp_path)
    root = tmp_path / "checkpoints"
    latest = save_training_generation(
        root, "latest", agent, sampler, {"training_stage": "stage5"},
    )
    final = save_training_generation(
        root, "final", agent, sampler, {"training_stage": "stage5"},
        parent_tag="latest",
    )
    latest_manifest = json.loads(
        (root / ".generations" / latest / "manifest.json").read_text()
    )
    final_manifest = json.loads(
        (root / ".generations" / final / "manifest.json").read_text()
    )
    assert final_manifest["semantic_lineage"]["root_generation"] == latest
    assert final_manifest["semantic_lineage"]["parent_generation"] == latest
    assert (
        final_manifest["semantic_lineage"]["parent_training_payload_sha256"]
        == latest_manifest["training_payload_sha256"]
    )


def test_inference_loader_checks_semantic_hash_after_byte_integrity(tmp_path: Path):
    config = _config()
    agent = TractorAgent(config, TractorAgentConfig(top_quantiles_to_drop_per_net=1))
    sampler = _sampler(tmp_path)
    root = tmp_path / "checkpoints"
    generation = save_training_generation(root, "final", agent, sampler, {})
    manifest_path = root / ".generations" / generation / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["component_state_sha256"]["online"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="semantic hash"):
        load_inference_weights(root, "final", agent)


def test_stage_warm_start_imports_online_only_and_reinitializes_target(tmp_path: Path):
    config = _config()
    source = TractorAgent(config, TractorAgentConfig(top_quantiles_to_drop_per_net=1))
    with torch.no_grad():
        next(source.online.parameters()).fill_(0.25)
    root = tmp_path / "checkpoints"
    generation = save_training_generation(
        root, "final", source, _sampler(tmp_path), {"training_stage": "stage3"},
    )
    target = TractorAgent(config, TractorAgentConfig(top_quantiles_to_drop_per_net=1))
    lineage = load_warm_start_generation(
        root, "final", target, source_stage="stage3", target_stage="stage4",
    )
    assert lineage["source_generation"] == generation
    for name, value in source.online.state_dict().items():
        assert torch.equal(value, target.online.state_dict()[name])
    for name, value in target.target.state_dict().items():
        assert torch.equal(value, target.online.state_dict()[name])
    assert target.update_step == 0
    with pytest.raises(ValueError, match="unsupported"):
        load_warm_start_generation(
            root, "final", target, source_stage="stage3", target_stage="stage5",
        )


def test_corrupt_payload_and_escaping_pointer_fail_before_deserialization(tmp_path: Path, monkeypatch):
    config = _config()
    agent = TractorAgent(config, TractorAgentConfig(top_quantiles_to_drop_per_net=1))
    sampler = _sampler(tmp_path)
    root = tmp_path / "checkpoints"
    generation = save_training_generation(root, "latest", agent, sampler, {})
    payload = root / ".generations" / generation / "training.pt"
    with payload.open("ab") as stream:
        stream.write(b"corruption")
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("torch.load must not run before hash validation")

    monkeypatch.setattr(torch, "load", forbidden)
    with pytest.raises(RuntimeError, match="checksum"):
        load_training_generation(root, "latest", agent, sampler)
    assert not called

    (root / "latest").unlink()
    os.symlink("../outside", root / "latest")
    with pytest.raises(RuntimeError, match="escapes"):
        load_training_generation(root, "latest", agent, sampler)


def test_wrong_role_is_rejected_before_tensor_load(tmp_path: Path, monkeypatch):
    config = _config()
    agent = TractorAgent(config, TractorAgentConfig(top_quantiles_to_drop_per_net=1))
    sampler = _sampler(tmp_path)
    root = tmp_path / "checkpoints"
    generation = save_training_generation(root, "latest", agent, sampler, {})
    manifest_path = root / ".generations" / generation / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifact_role"] = "deployment_bundle"
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("unexpected deserialize"))
    with pytest.raises(RuntimeError, match="role"):
        load_training_generation(root, "latest", agent, sampler)


def test_deployment_bundle_contains_only_inference_role(tmp_path: Path):
    config = _config()
    agent = TractorAgent(config, TractorAgentConfig(top_quantiles_to_drop_per_net=1))
    source_root = tmp_path / "source-checkpoints"
    source_generation = save_training_generation(
        source_root, "final", agent, _sampler(tmp_path / "source-data"),
        {"training_stage": "stage5"},
    )
    source_manifest = json.loads(
        (source_root / ".generations" / source_generation / "manifest.json").read_text()
    )
    source_hash = source_manifest["training_payload_sha256"]
    calibration = PlattCalibration(1.0, 0.0, source_hash, "calibration-split-hash")
    calibration_path = save_calibration_artifact(
        tmp_path / "calibration", "cal-for-bundle", calibration,
        episode_ids=["cal-episode"], metrics={"ece": 0.05, "event_count": 10},
        calibration_context_id="r5-post-aggregate-v1",
    )
    calibration_artifact_sha256 = hashlib.sha256(calibration_path.read_bytes()).hexdigest()
    bundle = export_deployment_bundle(
        tmp_path / "bundles", "tractor-a7-test", agent, calibration,
        {
            "promotion_status": "promoted-sim",
            "source_checkpoint_sha256": source_hash,
            "calibration_split_sha256": "calibration-split-hash",
            "calibration_artifact_sha256": calibration_artifact_sha256,
            "robot_attestation_hash": "robot", "controller_attestation_hash": "controller",
            "resolved_config": {"fixture": True}, "protocol_version": "fixture-v1",
            "approval_owner": "unit-test",
            "latency_evidence": {"target_hardware_evidence": True, "gate_passed": True},
        }, calibration_artifact_path=calibration_path,
        source_checkpoint_root=source_root,
    )
    manifest = validate_deployment_bundle(bundle, config.fingerprint())
    assert manifest["artifact_role"] == "deployment_bundle"
    payload = torch.load(bundle / "inference.pt", weights_only=False)
    assert set(payload) == {"schema_id", "artifact_role", "model_fingerprint", "online"}
    policy, loaded_manifest = load_deployment_policy(
        bundle, config, SelectorConfig(),
    )
    assert loaded_manifest["bundle_id"] == "tractor-a7-test"
    assert policy.deployment


def test_deployment_export_rejects_unapproved_or_unprofiled_state(tmp_path: Path):
    config = _config()
    agent = TractorAgent(config, TractorAgentConfig(top_quantiles_to_drop_per_net=1))
    calibration = PlattCalibration(1.0, 0.0, "checkpoint", "split")
    with pytest.raises(ValueError, match="incomplete"):
        export_deployment_bundle(
            tmp_path, "invalid", agent, calibration, {},
            calibration_artifact_path=tmp_path / "missing.json",
            source_checkpoint_root=tmp_path / "missing-checkpoint",
        )


def test_calibration_artifact_is_role_separated_and_source_bound(tmp_path: Path):
    calibration = PlattCalibration(0.9, -0.1, "checkpoint", "split")
    path = save_calibration_artifact(
        tmp_path, "cal-v1", calibration, episode_ids=["ep-2", "ep-1"],
        metrics={"ece": 0.04, "event_count": 12}, calibration_context_id="r5-post-aggregate-v1",
    )
    loaded, manifest = load_calibration_artifact(path, expected_checkpoint_sha256="checkpoint")
    assert loaded == calibration
    assert manifest["episode_ids"] == ["ep-1", "ep-2"]
    assert "optimizer" not in manifest
    with pytest.raises(RuntimeError, match="source checkpoint"):
        load_calibration_artifact(path, expected_checkpoint_sha256="other")
