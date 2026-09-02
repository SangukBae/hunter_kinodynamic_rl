"""Defect-fix item 12: localization sweep live driver -- pure/importable
pieces only (backend-factory selection, noise-param defaults). The actual
live Gazebo drive (:func:`run_localization_sweep_condition`/
:func:`run_live_localization_sweep`) requires a live Gazebo instance and is
NOT executed in this session (see module docstring's own honest-limitation
note); this file verifies the code imports cleanly and its pure logic is
correct, mirroring test_live_evidence_runner.py's own scope discipline."""

import pytest

from hunter_kinodynamic_rl.evaluation.localization_sweep_live_driver import (
    DEFAULT_CONDITION_NOISE_PARAMS, DEFAULT_CONDITIONS, DRIFTING_CONDITION, IDEAL_CONDITION, NOISY_CONDITION,
    localization_backend_factory_for_condition, run_live_localization_sweep, run_localization_sweep_condition,
)


def test_default_conditions_are_ideal_noisy_drifting():
    assert DEFAULT_CONDITIONS == (IDEAL_CONDITION, NOISY_CONDITION, DRIFTING_CONDITION)


def test_ideal_condition_factory_builds_gazebo_odom_backend():
    from hunter_kinodynamic_rl.navigation.localization.gazebo_odom_backend import GazeboOdomLocalizationBackend

    factory = localization_backend_factory_for_condition(IDEAL_CONDITION)
    backend = factory()
    assert isinstance(backend, GazeboOdomLocalizationBackend)


def test_noisy_condition_factory_builds_wheel_imu_backend_with_default_params():
    from hunter_kinodynamic_rl.navigation.localization.wheel_imu_backend import WheelImuLocalizationBackend

    factory = localization_backend_factory_for_condition(NOISY_CONDITION)
    backend = factory()
    assert isinstance(backend, WheelImuLocalizationBackend)
    expected = DEFAULT_CONDITION_NOISE_PARAMS[NOISY_CONDITION]
    assert backend._noise_model.position_process_noise_m2_per_m == pytest.approx(
        expected["position_process_noise_m2_per_m"]
    )


def test_drifting_condition_has_strictly_larger_noise_than_noisy():
    """drifting must be a strictly MORE degraded version of noisy, never
    an unrelated/smaller noise profile -- otherwise the sweep's own
    drift-magnitude-vs-performance curve (drift_curve) would be
    nonsensical."""
    noisy = DEFAULT_CONDITION_NOISE_PARAMS[NOISY_CONDITION]
    drifting = DEFAULT_CONDITION_NOISE_PARAMS[DRIFTING_CONDITION]
    for key in noisy:
        assert drifting[key] > noisy[key], f"{key}: drifting={drifting[key]} must be > noisy={noisy[key]}"


def test_explicit_noise_params_override_defaults():
    from hunter_kinodynamic_rl.navigation.localization.wheel_imu_backend import WheelImuLocalizationBackend

    factory = localization_backend_factory_for_condition(
        NOISY_CONDITION, noise_params={"position_process_noise_m2_per_m": 0.999},
    )
    backend = factory()
    assert isinstance(backend, WheelImuLocalizationBackend)
    assert backend._noise_model.position_process_noise_m2_per_m == pytest.approx(0.999)
    # Overriding only one field must not silently zero the others out --
    # WheelImuNoiseModel's own defaults fill the rest.
    assert backend._noise_model.heading_process_noise_rad2_per_rad > 0.0


def test_unknown_condition_name_falls_back_to_wheel_imu_with_no_extra_noise():
    from hunter_kinodynamic_rl.navigation.localization.wheel_imu_backend import WheelImuLocalizationBackend

    factory = localization_backend_factory_for_condition("some_other_condition")
    backend = factory()
    assert isinstance(backend, WheelImuLocalizationBackend)


def test_run_localization_sweep_condition_is_importable_with_correct_signature():
    import inspect

    sig = inspect.signature(run_localization_sweep_condition)
    for name in ("profile", "scenarios", "agent", "condition", "seed"):
        assert name in sig.parameters


def test_run_live_localization_sweep_is_importable_with_correct_signature():
    import inspect

    sig = inspect.signature(run_live_localization_sweep)
    for name in ("profile", "agent", "num_scenarios", "seed", "conditions"):
        assert name in sig.parameters
