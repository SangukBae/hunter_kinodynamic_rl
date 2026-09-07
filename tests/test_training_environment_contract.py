"""Fail-fast contract checks for formal Local L0-L5 training."""

import dataclasses

import pytest

pytest.importorskip("rclpy")
pytest.importorskip("drl_agent_interfaces")

from hunter_kinodynamic_rl.config.loader import load_profile  # noqa: E402
from hunter_kinodynamic_rl.evaluation.fingerprint import (  # noqa: E402
    architecture_fingerprint,
    local_training_contract_fingerprint,
    training_profile_fingerprint,
)
from hunter_kinodynamic_rl.training.trainer_base import (  # noqa: E402
    EnvServiceError,
    TrainingEnvironmentContractError,
    expected_training_dimensions,
    robot_name_from_urdf,
    robot_state_publisher_node,
    validate_training_environment_contract,
)


class _Dims:
    def __init__(self, state_dim, action_dim, max_action=1.0):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action


class _LiveEnvironment:
    def __init__(self, profile, *, robot_name=None, parameter_overrides=None, model_error=None):
        self.parameters = {
            "resolved_profile_name": profile.name,
            "architecture_fingerprint_sha256": architecture_fingerprint(profile),
            "local_training_contract_fingerprint_sha256": local_training_contract_fingerprint(profile),
            "training_profile_fingerprint_sha256": training_profile_fingerprint(profile),
        }
        self.parameters.update(parameter_overrides or {})
        self.robot_name = robot_name or profile.robot.name
        self.model_error = model_error
        self.publisher_node = None

    def get_remote_parameter(self, name):
        return self.parameters[name]

    def get_robot_description_model_name(self, publisher_node):
        self.publisher_node = publisher_node
        if self.model_error is not None:
            raise self.model_error
        return self.robot_name


def _dims(profile):
    return _Dims(*expected_training_dimensions(profile))


def test_improved_stage2_profile_passes_complete_live_contract():
    profile = load_profile("local_l5_counterfactual")
    env = _LiveEnvironment(profile)

    attestation = validate_training_environment_contract(env, profile, _dims(profile))

    assert attestation["urdf_robot_name"] == "hunter_se_improved"
    assert env.publisher_node == "/hunter_se/robot_state_publisher"
    assert attestation["training_profile_fingerprint_sha256"] == training_profile_fingerprint(profile)


def test_live_training_fingerprint_catches_reward_drift_with_same_shapes_and_architecture():
    live_profile = load_profile("local_l5_counterfactual")
    trainer_profile = dataclasses.replace(
        live_profile,
        reward=dataclasses.replace(
            live_profile.reward,
            collision_penalty=live_profile.reward.collision_penalty - 1.0,
        ),
    )
    assert architecture_fingerprint(live_profile) == architecture_fingerprint(trainer_profile)
    assert local_training_contract_fingerprint(live_profile) == local_training_contract_fingerprint(trainer_profile)

    with pytest.raises(TrainingEnvironmentContractError, match="training_profile_fingerprint_sha256"):
        validate_training_environment_contract(_LiveEnvironment(live_profile), trainer_profile, _dims(trainer_profile))


def test_training_contract_rejects_shape_compatible_wrong_profile_name():
    profile = load_profile("local_l5_counterfactual")
    env = _LiveEnvironment(profile, parameter_overrides={"resolved_profile_name": "stale_profile"})
    with pytest.raises(TrainingEnvironmentContractError, match="resolved_profile_name"):
        validate_training_environment_contract(env, profile, _dims(profile))


def test_training_contract_rejects_wrong_action_bound():
    profile = load_profile("local_l0_direct_control")
    bad_dims = _Dims(*expected_training_dimensions(profile), max_action=2.0)
    with pytest.raises(TrainingEnvironmentContractError, match="max_action"):
        validate_training_environment_contract(_LiveEnvironment(profile), profile, bad_dims)


def test_training_contract_rejects_baseline_urdf_for_improved_profile():
    profile = load_profile("local_l3_supervised_risk")
    env = _LiveEnvironment(profile, robot_name="hunter_se")
    with pytest.raises(TrainingEnvironmentContractError, match="wrong physical model"):
        validate_training_environment_contract(env, profile, _dims(profile))


def test_training_contract_rejects_missing_improved_robot_state_publisher():
    profile = load_profile("local_l1_trajectory")
    env = _LiveEnvironment(profile, model_error=EnvServiceError("service unavailable"))
    with pytest.raises(TrainingEnvironmentContractError, match="model_variant:=improved"):
        validate_training_environment_contract(env, profile, _dims(profile))


def test_model_variant_routes_to_distinct_robot_state_publishers():
    assert robot_state_publisher_node(load_profile("local_l0_direct_control")) == \
        "/hunter_se/robot_state_publisher"
    assert robot_state_publisher_node(load_profile("kinodynamic_tqc")) == \
        "/robot_state_publisher"


def test_robot_name_from_urdf_parses_resolved_description_and_fails_closed():
    assert robot_name_from_urdf("<robot name='hunter_se_improved'><link name='base_link'/></robot>") == \
        "hunter_se_improved"
    with pytest.raises(EnvServiceError, match="malformed XML"):
        robot_name_from_urdf("<robot")
    with pytest.raises(EnvServiceError, match="root must be"):
        robot_name_from_urdf("<robot></robot>")
