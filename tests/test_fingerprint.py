"""section item-2: config/benchmark fingerprinting -- architecture vs
evaluation-contract fingerprints must be independently sensitive to their
own sections and independent of everything else, and a checkpoint's frozen
``resolved_config`` fingerprint must match re-fingerprinting the SAME
profile freshly loaded (the "nothing changed" baseline every drift-detection
test below is contrasted against).
"""

import dataclasses

from hunter_kinodynamic_rl.config.loader import load_profile
from hunter_kinodynamic_rl.evaluation.fingerprint import (
    architecture_fingerprint, architecture_fingerprint_from_resolved_config, evaluation_contract_fingerprint,
    local_training_contract_fingerprint, local_training_contract_fingerprint_from_resolved_config,
)


def test_architecture_fingerprint_matches_resolved_config_variant():
    profile = load_profile("kinodynamic_tqc")
    resolved_config = dataclasses.asdict(profile)
    assert architecture_fingerprint(profile) == architecture_fingerprint_from_resolved_config(resolved_config)


def test_architecture_fingerprint_is_deterministic_and_profile_specific():
    a = architecture_fingerprint(load_profile("baseline_tqc"))
    b = architecture_fingerprint(load_profile("baseline_tqc"))
    c = architecture_fingerprint(load_profile("kinodynamic_tqc"))
    assert a == b
    assert a != c  # different action_space.mode/observation contract


def test_architecture_fingerprint_changes_when_action_space_changes():
    """The core item-2 regression: a profile's YAML content changing (here
    simulated via dataclasses.replace, standing in for an on-disk edit)
    must change the architecture fingerprint -- a name match alone must
    never be trusted."""
    base = load_profile("kinodynamic_tqc")
    edited = dataclasses.replace(
        base, action_space=dataclasses.replace(base.action_space, kappa_scale=0.5))
    assert architecture_fingerprint(base) != architecture_fingerprint(edited)


def test_architecture_fingerprint_changes_when_robot_geometry_changes():
    base = load_profile("kinodynamic_tqc")
    edited = dataclasses.replace(base, robot=dataclasses.replace(base.robot, wheelbase_m=base.robot.wheelbase_m + 0.1))
    assert architecture_fingerprint(base) != architecture_fingerprint(edited)


def test_architecture_fingerprint_changes_when_observation_changes():
    base = load_profile("kinodynamic_tqc")
    edited = dataclasses.replace(base, observation=dataclasses.replace(base.observation, lidar_bins=40))
    assert architecture_fingerprint(base) != architecture_fingerprint(edited)


def test_architecture_fingerprint_ignores_evaluation_contract_changes():
    """Editing evaluation/reward/scenario (the EVALUATION-contract's own
    sections) must NOT change the architecture fingerprint -- these are
    legitimately allowed to differ between the checkpoint's training
    profile and the requested eval profile."""
    base = load_profile("kinodynamic_tqc")
    edited = dataclasses.replace(
        base,
        evaluation=dataclasses.replace(base.evaluation, benchmark="id", max_episode_steps=999),
        reward=dataclasses.replace(base.reward, goal_threshold_m=base.reward.goal_threshold_m + 1.0),
        scenario=dataclasses.replace(base.scenario, world_size_m=base.scenario.world_size_m + 5.0),
    )
    assert architecture_fingerprint(base) == architecture_fingerprint(edited)


def test_local_training_contract_detects_distribution_change_architecture_ignores():
    base = load_profile("kinodynamic_tqc_arbitrary_subgoal")
    edited = dataclasses.replace(
        base,
        scenario=dataclasses.replace(
            base.scenario,
            goal_direction_sectors_deg=[[-90.0, 90.0]],
            goal_infeasible_fraction=0.0,
        ),
    )
    assert architecture_fingerprint(base) == architecture_fingerprint(edited)
    assert local_training_contract_fingerprint(base) != local_training_contract_fingerprint(edited)
    assert local_training_contract_fingerprint(base) == \
        local_training_contract_fingerprint_from_resolved_config(dataclasses.asdict(base))


def test_architecture_fingerprint_ignores_runtime_and_training_loop_changes():
    base = load_profile("kinodynamic_tqc")
    edited = dataclasses.replace(
        base,
        runtime=dataclasses.replace(base.runtime, time_delta_sec=base.runtime.time_delta_sec * 2),
        training=dataclasses.replace(base.training, seed=base.training.seed + 1, max_timesteps=1),
    )
    assert architecture_fingerprint(base) == architecture_fingerprint(edited)


def test_evaluation_contract_fingerprint_changes_when_scenario_or_reward_changes():
    base = load_profile("evaluation_id")
    edited_scenario = dataclasses.replace(
        base, scenario=dataclasses.replace(base.scenario, world_size_m=base.scenario.world_size_m + 1.0))
    edited_reward = dataclasses.replace(
        base, reward=dataclasses.replace(base.reward, goal_threshold_m=base.reward.goal_threshold_m + 1.0))
    edited_eval = dataclasses.replace(
        base, evaluation=dataclasses.replace(base.evaluation, max_episode_steps=1))
    baseline_fp = evaluation_contract_fingerprint(base)
    assert evaluation_contract_fingerprint(edited_scenario) != baseline_fp
    assert evaluation_contract_fingerprint(edited_reward) != baseline_fp
    assert evaluation_contract_fingerprint(edited_eval) != baseline_fp


def test_evaluation_contract_fingerprint_ignores_architecture_changes():
    """The dual of the architecture test above: editing action_space/robot/
    observation (architecture-owned) must NOT change the evaluation-contract
    fingerprint -- two DIFFERENT checkpoints (different architectures)
    evaluated through the SAME eval profile must report the SAME
    evaluation_contract_fingerprint."""
    base = load_profile("evaluation_id")
    edited = dataclasses.replace(
        base,
        action_space=dataclasses.replace(base.action_space, kappa_scale=0.3),
        robot=dataclasses.replace(base.robot, wheelbase_m=base.robot.wheelbase_m + 0.2),
    )
    assert evaluation_contract_fingerprint(base) == evaluation_contract_fingerprint(edited)


# --------------------------------------------------------------- item-2 (round 2)
def test_evaluation_contract_fingerprint_changes_when_runtime_changes():
    """runtime (time_delta_sec/deterministic_stepping/...) is now an
    EVALUATION-CONTRACT section (round 2's fairness fix) -- editing it must
    change the evaluation-contract fingerprint, exactly like reward/
    scenario/evaluation already do."""
    base = load_profile("evaluation_id")
    edited = dataclasses.replace(
        base, runtime=dataclasses.replace(base.runtime, time_delta_sec=base.runtime.time_delta_sec * 2))
    assert evaluation_contract_fingerprint(base) != evaluation_contract_fingerprint(edited)


def test_evaluation_contract_fingerprint_changes_when_sensor_noise_changes():
    """requirement 2: sensor_noise is now an EVALUATION-CONTRACT section --
    editing it must change the evaluation-contract fingerprint, exactly
    like reward/scenario/runtime/evaluation already do (and must NOT be
    confused with domain_randomization, which is not part of this
    contract at all)."""
    base = load_profile("evaluation_id")
    edited = dataclasses.replace(
        base, sensor_noise=dataclasses.replace(base.sensor_noise, enabled=True,
                                                 localization_xy_noise_std_m=0.05))
    assert evaluation_contract_fingerprint(base) != evaluation_contract_fingerprint(edited)


def test_evaluation_contract_fingerprint_catches_a_same_named_profile_content_edit():
    """The EXPLICIT item-2 (round 2) regression: two profiles with the
    SAME name but DIFFERENT world/reward/runtime/metric content (simulated
    via dataclasses.replace, standing in for an on-disk edit) must produce
    DIFFERENT evaluation-contract fingerprints -- a name match alone must
    never be trusted for benchmark-condition fairness, exactly as
    architecture_fingerprint already guarantees for the architecture side."""
    original = load_profile("evaluation_id")
    edited_on_disk = dataclasses.replace(
        original,
        scenario=dataclasses.replace(original.scenario, world_size_m=original.scenario.world_size_m + 10.0),
        reward=dataclasses.replace(original.reward, goal_threshold_m=original.reward.goal_threshold_m + 5.0),
        runtime=dataclasses.replace(original.runtime, time_delta_sec=original.runtime.time_delta_sec * 5),
        evaluation=dataclasses.replace(original.evaluation, max_episode_steps=1),
    )
    assert evaluation_contract_fingerprint(original) != evaluation_contract_fingerprint(edited_on_disk)
    # The profile "name" field itself was never touched -- proving the
    # fingerprint, not the name, is what actually caught the drift.
    assert original.name == edited_on_disk.name


def test_two_different_checkpoints_share_evaluation_contract_fingerprint_via_build_effective_profile():
    """End-to-end (still pure/ROS-free): two checkpoints trained under
    DIFFERENT profiles (different architectures) but evaluated through the
    SAME requested eval profile must produce the SAME
    evaluation_contract_fingerprint -- and DIFFERENT architecture_fingerprints
    (since they really are different architectures) -- proving
    build_effective_profile's own section-layering achieves what item-1/
    item-2 require."""
    from hunter_kinodynamic_rl.nodes.evaluation_node import build_effective_profile

    manifest_a = {"profile_name": "baseline_tqc",
                  "resolved_config": dataclasses.asdict(load_profile("baseline_tqc"))}
    manifest_b = {"profile_name": "kinodynamic_tqc_counterfactual",
                  "resolved_config": dataclasses.asdict(load_profile("kinodynamic_tqc_counterfactual"))}
    effective_a = build_effective_profile(manifest_a, "evaluation_id")
    effective_b = build_effective_profile(manifest_b, "evaluation_id")

    assert evaluation_contract_fingerprint(effective_a) == evaluation_contract_fingerprint(effective_b)
    assert architecture_fingerprint(effective_a) != architecture_fingerprint(effective_b)


def test_build_effective_profile_applies_the_requested_profiles_sensor_noise_not_the_checkpoints_own():
    """requirement 2, the exact fairness bug: a checkpoint trained WITH
    sensor_noise enabled, evaluated through a profile that leaves it at the
    (disabled) default, must be evaluated NOISE-FREE -- and vice versa --
    never silently keep the checkpoint's own training-time setting."""
    from hunter_kinodynamic_rl.config.schema import SensorNoiseConfig
    from hunter_kinodynamic_rl.nodes.evaluation_node import build_effective_profile

    noisy_trained = dataclasses.replace(
        load_profile("baseline_tqc"),
        sensor_noise=SensorNoiseConfig(enabled=True, localization_xy_noise_std_m=0.1))
    manifest = {"profile_name": "baseline_tqc", "resolved_config": dataclasses.asdict(noisy_trained)}

    eval_profile = load_profile("evaluation_id")
    assert eval_profile.sensor_noise.enabled is False  # the requested profile's own (default) setting

    effective = build_effective_profile(manifest, "evaluation_id")
    assert effective.sensor_noise == eval_profile.sensor_noise
    assert effective.sensor_noise.enabled is False
