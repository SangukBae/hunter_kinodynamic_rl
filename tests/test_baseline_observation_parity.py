"""section P0-4: baseline/reference observation-contract parity against
drl_agent's own 87D contract (CLAUDE.md's State/Action Space section:
LiDAR 80D + 7D robot-state tail -- goal_dist, heading_err, prev_r,
prev_theta, v, yaw_rate, steering; only the first TWO previous-action
components, even though drl_agent's own hybrid action is 3D). Every
SHIPPED profile's ``observation.robot_state_dim`` (loaded exactly as
train_node.py/environment_node.py do, through the real layered
config/training/defaults.yaml -> profile.yaml chain -- never a hand-built
dataclass) is checked against the deliberate, documented rule this package
uses to decide it: ``action_space.mode == "legacy_waypoint"`` -> 7
(drl_agent parity), ``"trajectory"`` -> 8 (opt-in L-memory extension)."""

import pytest

from hunter_kinodynamic_rl.config.loader import load_profile
from hunter_kinodynamic_rl.env.observation.observation_builder import (
    ROBOT_STATE_DIM_LEGACY_PARITY, ROBOT_STATE_DIM_WITH_L_MEMORY,
)

DRL_AGENT_LIDAR_BINS = 80
DRL_AGENT_STATE_DIM = 87  # 80 + 7 -- CLAUDE.md's documented drl_agent contract
DRL_AGENT_TEMPORAL_STATE_DIM = 80 * 4 + 7  # 327


@pytest.mark.parametrize("profile_name", ["baseline_tqc", "legacy_waypoint_tqc"])
def test_legacy_waypoint_profiles_match_drl_agent_observation_contract_exactly(profile_name):
    profile = load_profile(profile_name)
    assert profile.action_space.mode == "legacy_waypoint"
    assert profile.observation.robot_state_dim == ROBOT_STATE_DIM_LEGACY_PARITY == 7
    assert profile.observation.lidar_bins == DRL_AGENT_LIDAR_BINS
    state_dim = profile.observation.lidar_bins * (
        profile.observation.frame_stack if profile.features.temporal_context else 1
    ) + profile.observation.robot_state_dim
    assert state_dim == DRL_AGENT_STATE_DIM


@pytest.mark.parametrize("profile_name", [
    "kinodynamic_tqc", "kinodynamic_tqc_temporal", "kinodynamic_tqc_risk",
    "kinodynamic_tqc_risk_supervised_only", "kinodynamic_tqc_counterfactual",
    "kinodynamic_tqc_domain_rand", "sac_baseline", "smoke_test", "real_hunter_safe",
    "evaluation_id", "evaluation_ood_geometry", "evaluation_ood_dynamics", "evaluation_dynamic",
])
def test_trajectory_mode_profiles_opt_into_l_memory_explicitly(profile_name):
    """Every trajectory-action profile in this package deliberately opts
    INTO the 8D (+L-memory) contract -- verifies the opt-in actually landed
    (not a silent fallback to the drl_agent-parity default), for every
    shipped profile a checkpoint/deployment/evaluation could plausibly use."""
    profile = load_profile(profile_name)
    assert profile.action_space.mode == "trajectory"
    assert profile.observation.robot_state_dim == ROBOT_STATE_DIM_WITH_L_MEMORY == 8


def test_baseline_and_kinodynamic_share_every_other_observation_field():
    """The ONLY observation-contract difference between the drl_agent-parity
    baseline and the kinodynamic profiles must be robot_state_dim -- LiDAR
    bins/range/front-sector-width/collision-margin stay identical, so this
    is a genuinely fair, apples-to-apples comparison on everything except
    the one deliberate research variable."""
    baseline = load_profile("baseline_tqc").observation
    kinodynamic = load_profile("kinodynamic_tqc").observation
    assert baseline.lidar_bins == kinodynamic.lidar_bins
    assert baseline.lidar_max_range_m == kinodynamic.lidar_max_range_m
    assert baseline.front_sector_width_rad == kinodynamic.front_sector_width_rad
    assert baseline.collision_margin_m == kinodynamic.collision_margin_m
    assert baseline.robot_state_dim != kinodynamic.robot_state_dim
