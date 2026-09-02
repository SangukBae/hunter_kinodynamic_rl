"""ROS-free gates for loading a trained Global policy into the live benchmark."""

import dataclasses

import pytest

from hunter_kinodynamic_rl.config.loader import load_profile
from hunter_kinodynamic_rl.evaluation.fingerprint import (
    LOCAL_CHECKPOINT_IDENTITY_FIELDS,
    hierarchical_architecture_fingerprint,
    sha256_of_obj,
)
from hunter_kinodynamic_rl.evaluation.run_live_hierarchical_benchmark import (
    _validate_global_checkpoint_manifest,
)
from hunter_kinodynamic_rl.navigation.global_rl.observation import (
    resolve_candidate_feature_names,
    resolve_map_channel_names,
)
from hunter_kinodynamic_rl.navigation.global_rl.replay_schema import SCHEMA_VERSION


def _profile():
    return load_profile("hierarchical_phase4")


def _local_identity():
    return {
        "generation": "local-generation", "pt_sha256": "local-sha", "state_dim": 327,
        "action_dim": 3, "local_training_contract_fingerprint": "local-contract",
    }


def _valid_manifest(profile, local_identity):
    resolved = dataclasses.asdict(profile)
    return {
        "hierarchical_architecture_fingerprint": hierarchical_architecture_fingerprint(profile),
        "resolved_config": resolved,
        "resolved_config_hash": sha256_of_obj(resolved),
        "local_checkpoint_manifest_summary": dict(local_identity),
        "map_channel_names": list(resolve_map_channel_names(profile.global_rl)),
        "candidate_feature_names": list(resolve_candidate_feature_names(profile.global_rl)),
        "node_feature_names": [], "max_nodes_in_observation": 0,
        "memory_enabled": False, "feasibility_enabled": False, "global_risk_enabled": False,
        "global_replay_schema_version": SCHEMA_VERSION,
        "loop_state": {"global_step": 1}, "training_steps": 1, "total_optimizer_updates": 1,
        "epsilon_schedule_basis": "agent_training_steps",
    }


def test_matching_global_and_local_contract_passes():
    profile = _profile()
    local = _local_identity()
    _validate_global_checkpoint_manifest(_valid_manifest(profile, local), profile, local)


@pytest.mark.parametrize("field_name", LOCAL_CHECKPOINT_IDENTITY_FIELDS)
def test_missing_local_identity_field_is_rejected_on_either_side(field_name):
    profile = _profile()
    local = _local_identity()
    manifest = _valid_manifest(profile, local)
    del manifest["local_checkpoint_manifest_summary"][field_name]
    with pytest.raises(RuntimeError, match=rf"local_checkpoint\.{field_name}"):
        _validate_global_checkpoint_manifest(manifest, profile, local)

    manifest = _valid_manifest(profile, local)
    current = dict(local)
    del current[field_name]
    with pytest.raises(RuntimeError, match=rf"local_checkpoint\.{field_name}"):
        _validate_global_checkpoint_manifest(manifest, profile, current)


def test_global_observation_schema_mismatch_is_rejected():
    profile = _profile()
    local = _local_identity()
    manifest = _valid_manifest(profile, local)
    manifest["map_channel_names"].append("stale_channel")
    with pytest.raises(RuntimeError, match="map_channel_names"):
        _validate_global_checkpoint_manifest(manifest, profile, local)


def test_missing_required_global_field_is_rejected():
    profile = _profile()
    local = _local_identity()
    manifest = _valid_manifest(profile, local)
    del manifest["resolved_config"]
    with pytest.raises(RuntimeError, match="missing required field"):
        _validate_global_checkpoint_manifest(manifest, profile, local)
