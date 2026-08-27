import pytest

from hunter_kinodynamic_rl.config.loader import load_profile
from hunter_kinodynamic_rl.config.schema import ConfigError, LocalizationConfig, MappingConfig, MissionConfig


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
