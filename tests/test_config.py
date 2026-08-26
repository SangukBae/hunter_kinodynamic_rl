import pytest

from hunter_kinodynamic_rl.config.loader import default_config_root, deep_merge, load_profile, profile_from_dict
from hunter_kinodynamic_rl.config.schema import ConfigError, RuntimeConfig


def test_deep_merge_overrides_nested_keys_only():
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    override = {"a": {"y": 20}}
    merged = deep_merge(base, override)
    assert merged == {"a": {"x": 1, "y": 20}, "b": 3}


def test_deep_merge_replaces_lists_outright():
    base = {"a": {"items": [1, 2, 3]}}
    override = {"a": {"items": [9]}}
    assert deep_merge(base, override)["a"]["items"] == [9]


def test_deep_merge_does_not_mutate_inputs():
    base = {"a": {"x": 1}}
    override = {"a": {"x": 2}}
    deep_merge(base, override)
    assert base == {"a": {"x": 1}}


def test_profile_from_dict_rejects_unknown_section():
    with pytest.raises(ConfigError):
        profile_from_dict("test", {"not_a_real_section": {}})


def test_profile_from_dict_rejects_unknown_key_in_known_section():
    with pytest.raises(ConfigError):
        profile_from_dict("test", {"robot": {"not_a_real_field": 1}})


def test_profile_from_dict_rejects_the_removed_dead_risk_prediction_flag():
    """features.risk_prediction was found (P1-13 config-validation audit) to
    be declared in every profile YAML but never read by any code -- toggling
    it had zero effect, exactly the "ablation flag that silently does
    nothing" hazard loader.py's own typo guard exists to prevent. Removed
    from FeatureFlags entirely; this documents that a stray leftover key
    (e.g. from an un-migrated profile) is now caught as unknown rather than
    silently accepted and ignored."""
    with pytest.raises(ConfigError):
        profile_from_dict("test", {"features": {"risk_prediction": True}})


def test_profile_from_dict_rejects_the_removed_dead_critic_extra_dim_flag():
    """risk.critic_extra_dim was declared (default 1, i.e. ON) and threaded
    into RiskLabel construction (risk/trajectory_risk.py), but NOTHING ever
    constructed a Critic with extra_dim>0 or passed `extra` to its forward()
    -- the reward critic's input shape and every forward call were
    unaffected regardless of this flag's value (section P0-7: "no dead
    config flags may remain"). Removed from RiskConfig entirely rather than
    wired up, since a risk-conditioned critic input is not part of the
    stated A-F ablation matrix."""
    with pytest.raises(ConfigError):
        profile_from_dict("test", {"risk": {"critic_extra_dim": 1}})


def test_load_profile_by_path(tmp_path):
    profile_path = tmp_path / "custom.yaml"
    profile_path.write_text("action_space:\n  mode: legacy_waypoint\n")
    profile = load_profile(str(profile_path))
    assert profile.action_space.mode == "legacy_waypoint"
    assert profile.robot.wheelbase_m == pytest.approx(0.547696)  # defaults layer still applied


def test_load_profile_unknown_name_raises():
    with pytest.raises(ConfigError):
        load_profile("this_profile_does_not_exist")


def test_cross_section_validation_risk_critic_requires_risk_enabled():
    with pytest.raises(ConfigError):
        profile_from_dict("test", {"features": {"risk_critic": True}, "risk": {"enabled": False}})


def test_cross_section_validation_counterfactual_requires_risk_enabled():
    with pytest.raises(ConfigError):
        profile_from_dict("test", {
            "features": {"counterfactual_risk": True},
            "counterfactual": {"enabled": True},
            "risk": {"enabled": False},
        })


# ------------------------------------- P1-13: config-validation completeness audit
def test_cross_section_validation_benchmark_and_domain_randomization_conflict():
    """environment_node.py's _on_reset branches `if is_fixed_benchmark: ...
    elif domain_randomization.enabled: ...` -- the benchmark branch always
    wins, so a profile enabling both would silently never randomize despite
    looking like it does."""
    with pytest.raises(ConfigError):
        profile_from_dict("test", {
            "evaluation": {"benchmark": "id"},
            "domain_randomization": {"enabled": True},
        })


def test_cross_section_validation_benchmark_alone_is_fine(tmp_path):
    # profile_from_dict() alone has no robot/* layering (robot fields default
    # to 0.0 and would fail RobotConfig.validate() regardless of this check)
    # -- go through load_profile() for a fully-layered, otherwise-valid profile.
    profile_path = tmp_path / "benchmark_only.yaml"
    profile_path.write_text("evaluation:\n  benchmark: id\n")
    load_profile(str(profile_path))  # must not raise


def test_cross_section_validation_domain_randomization_alone_is_fine(tmp_path):
    profile_path = tmp_path / "dr_only.yaml"
    profile_path.write_text("domain_randomization:\n  enabled: true\n")
    load_profile(str(profile_path))  # must not raise


def test_cross_section_validation_curriculum_flag_rejected_until_implemented():
    """features.curriculum has zero consumers in this package -- every
    shipped profile leaves it at the default False; catch it explicitly if
    a future profile edit turns it on before the feature is wired."""
    with pytest.raises(ConfigError):
        profile_from_dict("test", {"features": {"curriculum": True}})


def test_reward_goal_reached_reward_rejects_negative():
    from hunter_kinodynamic_rl.config.schema import RewardConfig
    with pytest.raises(ConfigError):
        RewardConfig(goal_reached_reward=-1.0).validate()


def test_reward_collision_penalty_rejects_positive():
    from hunter_kinodynamic_rl.config.schema import RewardConfig
    with pytest.raises(ConfigError):
        RewardConfig(collision_penalty=1.0).validate()


def test_reward_step_penalty_rejects_positive():
    from hunter_kinodynamic_rl.config.schema import RewardConfig
    with pytest.raises(ConfigError):
        RewardConfig(step_penalty=0.01).validate()


def test_reward_default_signs_are_accepted():
    from hunter_kinodynamic_rl.config.schema import RewardConfig
    RewardConfig().validate()  # goal_reached_reward=100 >=0, collision_penalty=-100 <=0, step_penalty=-0.01 <=0


def test_counterfactual_rejects_empty_kappa_offsets_when_enabled():
    from hunter_kinodynamic_rl.config.schema import CounterfactualConfig
    with pytest.raises(ConfigError):
        CounterfactualConfig(enabled=True, kappa_offsets_frac=[]).validate()


def test_counterfactual_rejects_empty_speed_fractions_when_enabled():
    from hunter_kinodynamic_rl.config.schema import CounterfactualConfig
    with pytest.raises(ConfigError):
        CounterfactualConfig(enabled=True, speed_fractions=[]).validate()


def test_counterfactual_empty_lists_are_fine_when_disabled():
    from hunter_kinodynamic_rl.config.schema import CounterfactualConfig
    CounterfactualConfig(enabled=False, kappa_offsets_frac=[], speed_fractions=[]).validate()


def test_counterfactual_rejects_invalid_progress_and_horizon_settings():
    from hunter_kinodynamic_rl.config.schema import CounterfactualConfig
    with pytest.raises(ConfigError):
        CounterfactualConfig(enabled=True, horizon_fractions=[]).validate()
    with pytest.raises(ConfigError):
        CounterfactualConfig(enabled=True, horizon_fractions=[0.0]).validate()
    with pytest.raises(ConfigError):
        CounterfactualConfig(enabled=True, min_progress_ratio=1.1).validate()
    with pytest.raises(ConfigError):
        CounterfactualConfig(enabled=True, max_progress_loss_m=-0.1).validate()


def test_counterfactual_enabled_rejects_disabled_feature_flag(tmp_path):
    profile_path = tmp_path / "bad_counterfactual_flag.yaml"
    profile_path.write_text(
        "features:\n"
        "  risk_critic: true\n"
        "  counterfactual_risk: false\n"
        "risk:\n"
        "  enabled: true\n"
        "counterfactual:\n"
        "  enabled: true\n"
    )
    with pytest.raises(ConfigError, match="counterfactual.enabled=true"):
        load_profile(str(profile_path))


# --------------------------------------------- P2-14: Vanilla SAC algorithm choice
def test_algorithm_config_rejects_unknown_name():
    from hunter_kinodynamic_rl.config.schema import AlgorithmConfig
    with pytest.raises(ConfigError):
        AlgorithmConfig(name="ppo").validate()


def test_algorithm_config_default_is_tqc():
    from hunter_kinodynamic_rl.config.schema import AlgorithmConfig
    AlgorithmConfig().validate()  # must not raise
    assert AlgorithmConfig().name == "tqc"


def test_sac_hyperparameters_rejects_non_positive_tau():
    from hunter_kinodynamic_rl.config.schema import SACHyperparameters
    with pytest.raises(ConfigError):
        SACHyperparameters(tau=0.0).validate()


def test_sac_hyperparameters_rejects_non_positive_n_critics():
    from hunter_kinodynamic_rl.config.schema import SACHyperparameters
    with pytest.raises(ConfigError):
        SACHyperparameters(n_critics=0).validate()


def test_cross_section_validation_sac_rejects_risk_critic():
    with pytest.raises(ConfigError):
        profile_from_dict("test", {
            "algorithm": {"name": "sac"},
            "features": {"risk_critic": True},
            "risk": {"enabled": True},
        })


def test_cross_section_validation_sac_rejects_counterfactual_risk():
    with pytest.raises(ConfigError):
        profile_from_dict("test", {
            "algorithm": {"name": "sac"},
            "features": {"counterfactual_risk": True},
            "counterfactual": {"enabled": True},
            "risk": {"enabled": True},
        })


def test_sac_baseline_profile_loads_and_selects_sac():
    profile = load_profile("sac_baseline")
    assert profile.algorithm.name == "sac"
    assert profile.features.risk_critic is False
    assert profile.features.counterfactual_risk is False


# --------------------------------------------------------- P0-9: sensor freshness
def test_sensor_freshness_max_reset_retries_rejects_negative():
    cfg = RuntimeConfig(sensor_freshness_max_reset_retries=-1)
    with pytest.raises(ConfigError):
        cfg.validate()


def test_sensor_freshness_max_reset_retries_accepts_zero():
    cfg = RuntimeConfig(sensor_freshness_max_reset_retries=0)
    cfg.validate()  # 0 (no retry, fail immediately) is a valid, deliberate choice


# --------------------------------------------------------------- item-5
def test_inference_worker_mode_rejects_unknown_value():
    cfg = RuntimeConfig(inference_worker_mode="subprocess")
    with pytest.raises(ConfigError, match="inference_worker_mode"):
        cfg.validate()


def test_inference_worker_mode_accepts_thread_and_process():
    RuntimeConfig(inference_worker_mode="thread").validate()
    RuntimeConfig(inference_worker_mode="process").validate()


def test_inference_worker_max_consecutive_timeouts_rejects_zero():
    cfg = RuntimeConfig(inference_worker_max_consecutive_timeouts=0)
    with pytest.raises(ConfigError, match="inference_worker_max_consecutive_timeouts"):
        cfg.validate()


def test_policy_health_timeout_must_exceed_policy_inference_timeout():
    """Otherwise the watchdog's health diagnostic would fire on every
    single ordinary per-tick timeout, not a genuinely sustained one."""
    cfg = RuntimeConfig(policy_inference_timeout_sec=1.0, policy_health_timeout_sec=1.0)
    with pytest.raises(ConfigError, match="policy_health_timeout_sec"):
        cfg.validate()

    cfg_ok = RuntimeConfig(policy_inference_timeout_sec=0.4, policy_health_timeout_sec=5.0)
    cfg_ok.validate()


# --------------------------------------------------- P0-7: deterministic stepping
def test_deterministic_stepping_disabled_ignores_step_size_mismatch():
    cfg = RuntimeConfig(deterministic_stepping=False, time_delta_sec=0.1, gazebo_max_step_size_sec=0.003)
    cfg.validate()  # must not raise -- the divisibility check only applies when enabled


def test_deterministic_stepping_requires_exact_multiple_of_max_step_size():
    cfg = RuntimeConfig(deterministic_stepping=True, time_delta_sec=0.1, gazebo_max_step_size_sec=0.003)
    with pytest.raises(ConfigError):
        cfg.validate()


def test_deterministic_stepping_accepts_a_clean_multiple():
    cfg = RuntimeConfig(deterministic_stepping=True, time_delta_sec=0.1, reset_settle_time_sec=0.2,
                         gazebo_max_step_size_sec=0.001)
    cfg.validate()  # 0.1/0.001=100, 0.2/0.001=200 -- both exact


# ------------------------- code review: physics-step tolerance/contract bug
def test_physics_step_calibration_tolerance_rejects_the_old_masked_0_001_vs_0_002_default():
    """The exact pre-fix contract this section responds to: the shipped
    gazebo_max_step_size_sec=0.001 alongside the OLD physics_step_tolerance_sec
    default (0.005) would have silently accepted a connected world whose
    real physics step was 0.002s -- schema validation must now reject any
    calibration tolerance at or above half the declared step size, since
    that's exactly the bound that guarantees a whole-step-scale mismatch
    like 0.001-vs-0.002 is always caught."""
    cfg = RuntimeConfig(gazebo_max_step_size_sec=0.001, physics_step_calibration_tolerance_sec=0.005)
    with pytest.raises(ConfigError, match="physics_step_calibration_tolerance_sec"):
        cfg.validate()


def test_physics_step_calibration_tolerance_accepts_the_shipped_default():
    RuntimeConfig().validate()  # gazebo_max_step_size_sec=0.001, calibration tolerance=0.0002 -- 20% of one step


def test_physics_step_calibration_tolerance_boundary_at_exactly_half_the_step_size():
    cfg_boundary = RuntimeConfig(gazebo_max_step_size_sec=0.001,
                                 physics_step_calibration_tolerance_sec=0.0005)
    with pytest.raises(ConfigError, match="physics_step_calibration_tolerance_sec"):
        cfg_boundary.validate()  # equality would let a real 0.0005s step pass a declared 0.001s step

    cfg_bad = RuntimeConfig(gazebo_max_step_size_sec=0.001,
                            physics_step_calibration_tolerance_sec=0.0005001)
    with pytest.raises(ConfigError, match="physics_step_calibration_tolerance_sec"):
        cfg_bad.validate()

    cfg_ok = RuntimeConfig(gazebo_max_step_size_sec=0.001,
                           physics_step_calibration_tolerance_sec=0.0004999)
    cfg_ok.validate()


def test_physics_step_calibration_tolerance_must_be_positive():
    cfg = RuntimeConfig(physics_step_calibration_tolerance_sec=0.0)
    with pytest.raises(ConfigError, match="physics_step_calibration_tolerance_sec"):
        cfg.validate()


def test_physics_step_calibration_tolerance_bound_scales_with_a_reconfigured_step_size():
    """A profile that changes gazebo_max_step_size_sec to something SMALLER
    must also re-choose physics_step_calibration_tolerance_sec -- the
    shipped absolute default (0.0002s) is deliberately NOT auto-scaled
    (see the field's own docstring), so keeping it against a smaller
    declared step (here 0.0001s, whose half-step bound is 0.00005s) must
    fail rather than silently inheriting a now-disproportionate tolerance."""
    cfg = RuntimeConfig(gazebo_max_step_size_sec=0.0001)  # calibration tolerance stays at the 0.001-scaled default
    with pytest.raises(ConfigError, match="physics_step_calibration_tolerance_sec"):
        cfg.validate()

    cfg_ok = RuntimeConfig(gazebo_max_step_size_sec=0.0001, physics_step_calibration_tolerance_sec=0.00002)
    cfg_ok.validate()


# ---------------------------------------------- P1-10: domain_randomization layering
def test_domain_randomization_defaults_are_layered_in_but_disabled():
    """config/domain_randomization/default.yaml (section P1-10) must be
    part of the layering chain -- but every profile that doesn't itself
    say otherwise stays enabled=false regardless (the file's own
    enabled=false), so this layer is a pure no-op for the ~all existing
    profiles that never reference domain_randomization at all."""
    profile = load_profile("kinodynamic_tqc")
    assert profile.domain_randomization.enabled is False
    # The default.yaml's own non-trivial ranges ARE present (proving the
    # layer loaded), just inert while disabled.
    assert profile.domain_randomization.mass_scale_range == [0.9, 1.1]


def test_profile_can_enable_domain_randomization_and_inherit_default_ranges(tmp_path):
    # profile_from_dict() takes an ALREADY-MERGED dict (no file layering at
    # all) -- to actually exercise the layering chain (the thing being
    # tested here), this goes through load_profile()'s real file-path
    # branch instead, which still resolves the SAME domain_randomization/
    # default.yaml layer regardless of profile_name being a bare name or
    # an explicit path (see load_profile's `root = config_root or
    # default_config_root()`, computed before that branch).
    profile_path = tmp_path / "dr_enabled_test.yaml"
    profile_path.write_text("domain_randomization:\n  enabled: true\n")
    profile = load_profile(str(profile_path))
    assert profile.domain_randomization.enabled is True
    assert profile.domain_randomization.mass_scale_range == [0.9, 1.1]
    assert profile.domain_randomization.lidar_range_noise_std_m_range == [0.0, 0.02]


def test_default_config_root_resolves_to_a_directory_containing_profiles():
    root = default_config_root()
    import os
    assert os.path.isdir(os.path.join(root, "profiles")), (
        f"default_config_root() returned {root!r}, which has no profiles/ subdirectory -- "
        "this is the exact bug caught when validating the INSTALLED package (ament_index_python "
        "resolves share/hunter_kinodynamic_rl/, not a path relative to the Python module)."
    )
