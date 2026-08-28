import pytest

from hunter_kinodynamic_rl.config.loader import load_profile
from hunter_kinodynamic_rl.config.schema import (
    ConfigError, HierarchyConfig, LocalizationConfig, MappingConfig, MissionConfig,
)


def test_mission_config_defaults_are_valid():
    MissionConfig().validate()


def test_mission_config_rejects_negative_position_tolerance():
    with pytest.raises(ConfigError):
        MissionConfig(position_tolerance_m=-0.1).validate()


def test_mission_config_rejects_heading_tolerance_above_pi():
    with pytest.raises(ConfigError):
        MissionConfig(heading_tolerance_rad=4.0).validate()


def test_localization_config_defaults_are_valid():
    LocalizationConfig().validate()


def test_localization_config_rejects_unknown_backend():
    with pytest.raises(ConfigError):
        LocalizationConfig(backend="lio_sam").validate()


def test_localization_config_rejects_out_of_range_confidence():
    with pytest.raises(ConfigError):
        LocalizationConfig(minimum_confidence=1.5).validate()


def test_mapping_config_defaults_are_valid():
    MappingConfig().validate()


def test_mapping_config_rejects_free_threshold_at_or_above_occupied_threshold():
    with pytest.raises(ConfigError):
        MappingConfig(free_threshold=0.2, occupied_threshold=0.2).validate()
    with pytest.raises(ConfigError):
        MappingConfig(free_threshold=0.3, occupied_threshold=0.2).validate()


def test_mapping_config_rejects_non_negative_free_log_odds_delta():
    with pytest.raises(ConfigError):
        MappingConfig(free_log_odds_delta=0.0).validate()


def test_mapping_config_rejects_non_positive_occupied_log_odds_delta():
    with pytest.raises(ConfigError):
        MappingConfig(occupied_log_odds_delta=0.0).validate()


def test_mapping_config_rejects_thresholds_outside_log_odds_bounds():
    with pytest.raises(ConfigError):
        MappingConfig(log_odds_min=-1.0, log_odds_max=1.0, occupied_threshold=2.0).validate()


def test_mapping_config_rejects_saturation_out_of_dtype_range():
    with pytest.raises(ConfigError):
        MappingConfig(visited_count_saturation=100000).validate()
    with pytest.raises(ConfigError):
        MappingConfig(failure_count_saturation=1000).validate()


def test_hierarchical_phase1_profile_loads():
    profile = load_profile("hierarchical_phase1")
    assert profile.mapping.mission_size_cells == 128
    assert profile.localization.backend == "gazebo_odom"


def test_unrelated_existing_profile_still_loads_with_navigation_defaults():
    """New mission/localization/mapping sections must be opt-in / inert for
    every pre-existing profile -- loading one unrelated to hierarchical
    navigation must still succeed and simply carry the defaults."""
    profile = load_profile("smoke_test")
    assert profile.mission.position_tolerance_m == pytest.approx(0.6)
    assert profile.mapping.resolution_m == pytest.approx(0.2)
    assert profile.localization.backend == "gazebo_odom"


# ------------------------------------------------------------ HierarchyConfig (Phase 2)

def test_hierarchy_config_defaults_are_valid():
    HierarchyConfig().validate()


def test_hierarchy_config_rejects_non_positive_subgoal_position_tolerance():
    with pytest.raises(ConfigError):
        HierarchyConfig(subgoal_position_tolerance_m=0.0).validate()


def test_hierarchy_config_rejects_heading_tolerance_above_pi():
    with pytest.raises(ConfigError):
        HierarchyConfig(subgoal_heading_tolerance_rad=4.0).validate()


def test_hierarchy_config_rejects_non_positive_timeout_steps():
    with pytest.raises(ConfigError):
        HierarchyConfig(local_option_timeout_steps=0).validate()


def test_hierarchy_config_rejects_negative_no_progress_delta():
    with pytest.raises(ConfigError):
        HierarchyConfig(no_progress_min_delta_m=-0.1).validate()


def test_hierarchy_config_rejects_out_of_range_localization_confidence():
    with pytest.raises(ConfigError):
        HierarchyConfig(localization_min_confidence=1.5).validate()


def test_hierarchy_config_rejects_negative_max_retries():
    with pytest.raises(ConfigError):
        HierarchyConfig(max_retries_per_subgoal=-1).validate()


def test_hierarchy_config_accepts_none_mission_timeouts_but_rejects_non_positive_when_set():
    HierarchyConfig(mission_timeout_steps=None, mission_timeout_sec=None).validate()
    with pytest.raises(ConfigError):
        HierarchyConfig(mission_timeout_steps=0).validate()
    with pytest.raises(ConfigError):
        HierarchyConfig(mission_timeout_sec=0.0).validate()


def test_unrelated_existing_profile_still_loads_with_hierarchy_defaults():
    """Same opt-in guarantee as mission/localization/mapping: an unrelated
    pre-existing profile must load fine and simply carry HierarchyConfig's
    defaults, never requiring a YAML edit."""
    profile = load_profile("smoke_test")
    assert profile.hierarchy.subgoal_position_tolerance_m == pytest.approx(0.5)
    assert profile.hierarchy.local_option_timeout_steps == 200
    assert profile.hierarchy.mission_timeout_steps is None
