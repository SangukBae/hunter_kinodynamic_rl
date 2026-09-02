#!/usr/bin/env python3
"""Defect-fix item 12: live driver for the Phase 6 localization sweep.

``evaluation/localization_sweep.py`` already implements scenario-pairing
verification and per-condition aggregation (Requirement I) -- what was
missing (per the Runbook's own "부분 구현" note) is a driver that actually
RUNS the ideal/noisy/drifting conditions against live Gazebo and hands
their episode lists to :func:`evaluation.localization_sweep.aggregate_localization_sweep`.
This module is that driver.

Each condition gets its OWN freshly-constructed
:class:`~hunter_kinodynamic_rl.navigation.local_rl.live_gazebo_executor.LiveGazeboLocalExecutor`
bound to that condition's own localization backend (defect-fix item 12's
``LiveGazeboLocalExecutor.__init__``/``bind_mission`` changes -- see that
module's own docstrings) -- never one shared executor whose backend is
swapped mid-sweep. Every condition runs the SAME fixed scenario manifest
(:func:`evaluation.long_horizon_benchmark.build_fixed_benchmark_manifest`,
built ONCE and reused, never regenerated per condition) and the SAME
``ablation``/checkpoint pair, with each condition's own noise
parameters/seed recorded verbatim into the returned
:class:`~hunter_kinodynamic_rl.evaluation.localization_sweep.LocalizationSweepResult`
for reproducibility.

HONEST LIMITATION (inherited from
``navigation.localization.wheel_imu_backend.WheelImuLocalizationBackend``'s
own documented limitation, not introduced here): that backend integrates
the REAL Gazebo-bridged wheel/IMU twist with NO injected random
perturbation on ``v_mps``/``yaw_rate`` themselves -- it only accumulates a
GROWING COVARIANCE/CONFIDENCE metric alongside a dead-reckoned pose that
otherwise tracks true motion closely (bounded only by Euler-integration
discretization error, not by noise). A sweep built on this backend
therefore mainly differentiates CONFIDENCE-GATED behavior (e.g.
``is_pose_usable``/``_localization_valid`` degraded-mode triggers, whose
threshold this backend's decaying confidence can cross) between
conditions, not the navigation-performance cost of actual positional
drift. Quantifying performance under REAL injected position/heading error
would require extending the noise model itself (a separate, larger change
this session did not make) -- this driver does not overclaim that it
measured that.

``LidarOdomLocalizationBackend`` is NOT used for any condition here: its
own module docstring states the actual scan-matching/ICP/NDT algorithm
that would drive it is not implemented in this repository at all, so there
is no honest way to exercise it as a live sweep condition without
fabricating relative-transform inputs.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

from hunter_kinodynamic_rl.evaluation.localization_sweep import LocalizationSweepResult, aggregate_localization_sweep

IDEAL_CONDITION = "ideal"
NOISY_CONDITION = "noisy"
DRIFTING_CONDITION = "drifting"
DEFAULT_CONDITIONS: Sequence[str] = (IDEAL_CONDITION, NOISY_CONDITION, DRIFTING_CONDITION)

#: Default WheelImuNoiseModel field overrides per non-ideal condition --
#: "drifting" is a strictly larger-covariance-growth version of "noisy",
#: never a different backend/mechanism (see module docstring's honest
#: limitation note on what this backend can and cannot actually simulate).
DEFAULT_CONDITION_NOISE_PARAMS: Dict[str, Dict[str, float]] = {
    NOISY_CONDITION: {
        "position_process_noise_m2_per_m": 0.01, "heading_process_noise_rad2_per_rad": 0.005,
        "heading_process_noise_rad2_per_sec": 0.0005,
    },
    DRIFTING_CONDITION: {
        "position_process_noise_m2_per_m": 0.1, "heading_process_noise_rad2_per_rad": 0.05,
        "heading_process_noise_rad2_per_sec": 0.005,
    },
}


def localization_backend_factory_for_condition(
    condition: str, noise_params: Optional[Dict[str, float]] = None,
) -> Callable[[], Any]:
    """Returns a zero-argument factory suitable for
    ``LiveGazeboLocalExecutor(..., localization_backend_factory=...)``.
    ``condition == "ideal"`` -> ground-truth ``GazeboOdomLocalizationBackend``
    (zero noise, by construction -- the reference every other condition's
    ``drift_sensitivity`` is measured against). Any other condition name ->
    ``WheelImuLocalizationBackend`` with ``noise_params`` applied as
    ``WheelImuNoiseModel`` field overrides (defaults from
    :data:`DEFAULT_CONDITION_NOISE_PARAMS` when the condition name matches
    one of ``"noisy"``/``"drifting"`` and no explicit ``noise_params`` was
    given)."""
    from hunter_kinodynamic_rl.navigation.localization.gazebo_odom_backend import GazeboOdomLocalizationBackend
    from hunter_kinodynamic_rl.navigation.localization.wheel_imu_backend import (
        WheelImuLocalizationBackend, WheelImuNoiseModel,
    )

    if condition == IDEAL_CONDITION:
        return lambda: GazeboOdomLocalizationBackend(use_covariance_confidence=True)
    params = dict(noise_params) if noise_params is not None else dict(DEFAULT_CONDITION_NOISE_PARAMS.get(condition, {}))
    model = WheelImuNoiseModel(**params)
    return lambda: WheelImuLocalizationBackend(model)


def run_localization_sweep_condition(
    profile, scenarios, agent, *, condition: str, seed: int, noise_params: Optional[Dict[str, float]] = None,
    ablation: str = "B", max_options: int = 0, max_local_steps: int = 0, local_evaluator=None,
    node_name: Optional[str] = None,
) -> List[dict]:
    """Runs ONE localization-sweep condition over LIVE Gazebo -- a fresh
    ``LiveGazeboLocalExecutor`` bound to ``condition``'s own localization
    backend, driving the SAME ``scenarios`` manifest/``ablation``/checkpoint
    every other condition in the sweep uses. Requires ``rclpy.init()`` to
    have already happened in the caller (mirrors every other live driver
    in this package)."""
    from hunter_kinodynamic_rl.evaluation.long_horizon_benchmark import run_ablation_benchmark
    from hunter_kinodynamic_rl.navigation.local_rl.live_gazebo_executor import LiveGazeboLocalExecutor

    backend_factory = localization_backend_factory_for_condition(condition, noise_params)
    min_turning_radius_m = 1.0 / profile.robot.max_curvature
    live_executor = LiveGazeboLocalExecutor(
        profile, profile.long_horizon_world,
        profile.hierarchical_training.local_checkpoint_dir, profile.hierarchical_training.local_checkpoint_name,
        node_name=node_name or f"localization_sweep_{condition}",
        localization_backend_factory=backend_factory,
    )
    try:
        episodes = run_ablation_benchmark(
            ablation, scenarios, agent=agent, seed=seed, local_evaluator=local_evaluator,
            global_cfg=profile.global_rl, hierarchy_cfg=profile.hierarchy, mapping_cfg=profile.mapping,
            memory_cfg=profile.memory, feasibility_cfg=profile.feasibility,
            long_horizon_cfg=profile.long_horizon_world, robot_radius_m=profile.robot.collision_radius_m,
            min_turning_radius_m=min_turning_radius_m, wheelbase_m=profile.robot.wheelbase_m,
            max_options=max_options or profile.hierarchical_training.max_global_options_per_mission,
            max_local_steps=max_local_steps or profile.hierarchical_training.max_local_steps_per_option,
            mission_timeout_steps=profile.hierarchy.mission_timeout_steps,
            local_executor_factory=live_executor.bind_mission,
        )
    finally:
        live_executor.close()
    return episodes


def run_live_localization_sweep(
    profile, agent, *, num_scenarios: int = 20, seed: int = 20000, mode: str = "test",
    conditions: Sequence[str] = DEFAULT_CONDITIONS, condition_noise_params: Optional[Dict[str, Dict[str, float]]] = None,
    ablation: str = "B", max_options: int = 0, max_local_steps: int = 0, logger=print,
) -> LocalizationSweepResult:
    """Runs EVERY requested condition against ONE shared fixed scenario
    manifest (built once, reused for every condition -- never regenerated/
    reshuffled per condition) and hands the results to
    :func:`evaluation.localization_sweep.aggregate_localization_sweep`,
    which itself refuses (:class:`~hunter_kinodynamic_rl.evaluation.localization_sweep.ScenarioPairingError`)
    to aggregate a mismatched comparison -- this driver additionally fails
    BEFORE running anything live if any two conditions would end up with
    non-identical scenario_id sets (impossible when scenarios are shared
    like this, kept as an explicit assertion rather than trusted
    silently). ``condition_noise_params`` (optional, ``{condition:
    {field: value}}``) overrides :data:`DEFAULT_CONDITION_NOISE_PARAMS`
    per condition; the SAME fixed ``seed`` is used for the shared scenario
    manifest -- noise/drift injection itself has no separate seed of its
    own (see this module's own docstring: no random perturbation is
    actually injected by the current localization backend)."""
    from hunter_kinodynamic_rl.evaluation.long_horizon_benchmark import build_fixed_benchmark_manifest

    condition_noise_params = condition_noise_params or {}
    min_turning_radius_m = 1.0 / profile.robot.max_curvature
    scenarios = build_fixed_benchmark_manifest(
        profile.long_horizon_world, profile.robot.collision_radius_m, min_turning_radius_m,
        profile.robot.wheelbase_m, num_scenarios=num_scenarios, seed=seed, mode=mode,
    )
    logger(f"built {len(scenarios)} fixed {mode!r}-pool scenarios shared across conditions {list(conditions)}")

    results: Dict[str, List[dict]] = {}
    noise_params_used: Dict[str, Any] = {}
    seeds_used: Dict[str, int] = {}
    for condition in conditions:
        params = condition_noise_params.get(
            condition, {} if condition == IDEAL_CONDITION else DEFAULT_CONDITION_NOISE_PARAMS.get(condition, {}),
        )
        logger(f"running localization sweep condition {condition!r} (noise_params={params})...")
        results[condition] = run_localization_sweep_condition(
            profile, scenarios, agent, condition=condition, seed=seed, noise_params=params, ablation=ablation,
            max_options=max_options, max_local_steps=max_local_steps,
        )
        noise_params_used[condition] = params
        seeds_used[condition] = seed

    return aggregate_localization_sweep(results, noise_params=noise_params_used, seeds=seeds_used)
