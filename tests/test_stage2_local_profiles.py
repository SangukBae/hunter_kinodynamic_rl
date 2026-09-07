"""Stage 2 L0-L5 profiles must be a fair, executable Local ladder."""

import dataclasses

from hunter_kinodynamic_rl.config.loader import load_profile
from hunter_kinodynamic_rl.evaluation.fingerprint import (
    architecture_fingerprint,
    local_training_contract_fingerprint,
)
from hunter_kinodynamic_rl.trajectory.action_space import action_dim_for_mode


LABELS = {
    "L0": "local_l0_direct_control",
    "L1": "local_l1_trajectory",
    "L2": "local_l2_temporal",
    "L3": "local_l3_supervised_risk",
    "L4": "local_l4_risk_actor",
    "L5": "local_l5_counterfactual",
}


def test_stage2_rows_share_task_budget_algorithm_and_seed_pools():
    profiles = {label: load_profile(name) for label, name in LABELS.items()}
    reference = profiles["L0"]
    for label, profile in profiles.items():
        assert profile.robot.name == "hunter_se_improved", label
        assert profile.algorithm.name == "tqc", label
        assert profile.training.max_timesteps == 150_000, label
        assert profile.training.timesteps_before_training == 3_000, label
        assert profile.training.eval_freq == 10_000, label
        assert profile.training.eval_episodes == 10, label
        assert profile.training.episode_length_steps == 600, label
        assert dataclasses.asdict(profile.scenario) == dataclasses.asdict(reference.scenario), label
        assert dataclasses.asdict(profile.reward) == dataclasses.asdict(reference.reward), label
        assert dataclasses.asdict(profile.hyperparameters) == dataclasses.asdict(reference.hyperparameters), label
        assert dataclasses.asdict(profile.robot) == dataclasses.asdict(reference.robot), label
        assert dataclasses.asdict(profile.start_pose) == dataclasses.asdict(reference.start_pose), label
        assert dataclasses.asdict(profile.runtime) == dataclasses.asdict(reference.runtime), label
        assert dataclasses.asdict(profile.sensor_noise) == dataclasses.asdict(reference.sensor_noise), label
        assert dataclasses.asdict(profile.domain_randomization) == dataclasses.asdict(reference.domain_randomization), label
        assert local_training_contract_fingerprint(profile) == local_training_contract_fingerprint(reference), label
        assert profile.scenario.goal_sampling_mode == "robot_relative_band", label
        assert profile.scenario.goal_direction_sectors_deg == [[-180.0, 180.0]], label
        assert profile.scenario.goal_infeasible_fraction == 0.15, label


def test_stage2_rows_add_exactly_the_declared_method_components():
    p = {label: load_profile(name) for label, name in LABELS.items()}
    assert p["L0"].action_space.mode == "direct_control"
    assert action_dim_for_mode(p["L0"].action_space) == 2
    assert p["L0"].observation.robot_state_dim == 7
    assert not p["L0"].features.ackermann_rollout

    assert p["L1"].action_space.mode == "trajectory"
    assert action_dim_for_mode(p["L1"].action_space) == 3
    assert not p["L1"].features.temporal_context

    assert p["L2"].features.temporal_context
    assert not p["L2"].features.risk_critic

    assert p["L3"].features.risk_critic and p["L3"].risk.enabled
    assert p["L3"].risk.actor_lambda == 0.0
    assert not p["L3"].counterfactual.enabled

    assert p["L4"].risk.actor_lambda > 0.0
    assert not p["L4"].counterfactual.enabled

    assert p["L5"].risk.actor_lambda == p["L4"].risk.actor_lambda
    assert p["L5"].features.counterfactual_risk
    assert p["L5"].counterfactual.enabled

    assert len({architecture_fingerprint(profile) for profile in p.values()}) == len(p)


def test_stage2_l5_is_compatible_with_the_hierarchy_local_architecture():
    l5 = load_profile(LABELS["L5"])
    hierarchy = load_profile("hierarchical_phase4")
    expected_training_profile = load_profile(hierarchy.hierarchical_training.local_training_profile_name)
    assert hierarchy.hierarchical_training.local_training_profile_name == LABELS["L5"]
    assert hierarchy.robot.name == "hunter_se_improved"
    assert hierarchy.global_rl.robot_footprint_radius_m == hierarchy.robot.collision_radius_m
    assert architecture_fingerprint(l5) == architecture_fingerprint(hierarchy)
    assert local_training_contract_fingerprint(l5) == local_training_contract_fingerprint(expected_training_profile)
