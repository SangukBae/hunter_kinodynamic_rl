#!/usr/bin/env python3
"""Run a trained policy against a FIXED benchmark (section 32/33) via a live
environment_node.py, aggregate REAL metrics, write results.

Each benchmark scenario is placed EXACTLY (section 6/11): before every
``/reset``, the runner writes the scenario's own YAML file path to
environment_node.py's ``scenario_override_path`` parameter (via
``EnvironmentClient.set_scenario_override``), so Gazebo gets the
hand-authored start/goal/obstacle layout, not a seed-driven procedural
regeneration. Metrics are computed from REAL odometry (path length via
``EnvironmentClient.episode_path_length_m``, itself integrated from live
``/odometry`` samples) and the REAL per-step risk telemetry stream
(clearance/TTC series, unrecoverable-state flag) -- never approximated from
the normalized action or a single last-step reading.
"""

from __future__ import annotations

import math
import os
from typing import List

from hunter_kinodynamic_rl.config.schema import Profile
from hunter_kinodynamic_rl.env.scenarios.benchmark_loader import BenchmarkScenario, load_benchmark
from hunter_kinodynamic_rl.evaluation import metrics as metrics_mod
from hunter_kinodynamic_rl.evaluation.fingerprint import sha256_of_obj
from hunter_kinodynamic_rl.evaluation.result_writer import write_episode_csv, write_episode_jsonl, write_summary_json
from hunter_kinodynamic_rl.rl.checkpointing import manager
from hunter_kinodynamic_rl.training.trainer_base import EnvironmentClient


def run_episode(env: EnvironmentClient, agent, profile: Profile, scenario: BenchmarkScenario) -> dict:
    """Caller MUST call ``env.set_scenario_override(<this scenario's YAML
    path>)`` before invoking this function -- see ``run_benchmark`` below,
    which is the only production call site.

    Every physical metric below is a REAL measurement (odometry velocity,
    joint-state-derived steering, the safety guard's actual post-clamp
    command, /clock-based elapsed time) -- NEVER the normalized action
    re-interpreted as if it were already a physical quantity (section
    P1-4's explicit correction: the previous version read
    ``action[0]*steering_limit`` as "steering" and ``action[1]`` as
    "velocity [m/s]", both wrong in unit AND in which action component
    means what under ``action_space.mode: trajectory``)."""
    state = env.reset()

    steering_values, steering_rates, velocities, control_deltas = [], [], [], []
    clearance_series, ttc_series, stopping_margin_series = [], [], []
    # section P0-6: time_to_collision() returns the assessment HORIZON
    # itself (not a real measurement) whenever no collision is found within
    # it -- risk/ttc.py's own docstring calls this out explicitly. Lumping
    # that censored sentinel in with genuine near-miss TTC measurements
    # would make "mean/worst TTC" mostly report the constant horizon value
    # whenever a policy is behaving safely (the common, desired case),
    # which is not a meaningful statistic. risk_valid_steps/
    # collision_free_steps track the CENSORING rate itself as its own
    # explicit, separate metric (see collision_free_rate_mean in
    # evaluation/metrics.py::aggregate) instead of silently folding it into
    # ttc_series.
    risk_valid_steps = 0
    collision_free_steps = 0
    prev_command = None
    prev_steering = None
    prev_sim_timestamp = None
    curvature_feasibility_violations = 0
    total_reward = 0.0
    collided = timeout = success = unrecoverable = False
    emergency_stop_count = 0
    last_step_index = -1

    # section item-1 (fixed-benchmark fairness): the EVALUATION-CONTRACT's
    # own step budget -- NEVER `profile.training.episode_length_steps`
    # (checkpoint/training-profile-owned, and previously used here: two
    # models trained under different profiles with different
    # episode_length_steps would silently get different benchmark episode
    # budgets purely from that, confounding "navigated better" with
    # "got a longer episode"). Every model benchmarked through the SAME
    # evaluation profile shares this one value.
    episode_length_steps = profile.evaluation.max_episode_steps
    for step in range(episode_length_steps):
        action = agent.select_action(state, deterministic=True)
        state, reward, done, target, collision, _min_dist, telemetry, _diagnostics = env.step(action)
        total_reward += reward
        last_step_index = step

        # REAL measurements, sampled right after the step that just ran.
        measured_steering = env.latest_center_steering_rad
        steering_values.append(measured_steering)
        velocities.append(env.latest_v_mps)
        if (prev_steering is not None and prev_sim_timestamp is not None
                and math.isfinite(telemetry.sim_timestamp_sec)
                and telemetry.sim_timestamp_sec > prev_sim_timestamp):
            steering_rates.append(
                abs(measured_steering - prev_steering)
                / (telemetry.sim_timestamp_sec - prev_sim_timestamp)
            )
        prev_steering = measured_steering
        if math.isfinite(telemetry.sim_timestamp_sec):
            prev_sim_timestamp = telemetry.sim_timestamp_sec
        nominal_feasible = (
            math.isfinite(telemetry.nominal_speed_mps)
            and math.isfinite(telemetry.nominal_steering_rad)
            and -1e-9 <= telemetry.nominal_speed_mps <= profile.robot.max_forward_speed_mps + 1e-9
            and abs(telemetry.nominal_steering_rad) <= profile.robot.steering_limit_rad + 1e-9
        )
        curvature_feasibility_violations += int(not nominal_feasible)
        # published_* (not guarded_*): control smoothness must reflect what
        # the robot ACTUALLY received each tick -- with a command-latency
        # benchmark override active (e.g. ood_dynamics scenarios), the
        # guarded value can differ from what was really published this
        # step (section P0-5).
        published = (telemetry.published_speed_mps, telemetry.published_steering_rad)
        if prev_command is not None:
            control_deltas.append(abs(published[0] - prev_command[0]) + abs(published[1] - prev_command[1]))
        prev_command = published
        if telemetry.emergency_stop:
            emergency_stop_count += 1

        if telemetry.valid:
            clearance_series.append(telemetry.min_clearance_m)
            stopping_margin_series.append(telemetry.stopping_margin_m)
            risk_valid_steps += 1
            if telemetry.collision_within_horizon:
                ttc_series.append(telemetry.ttc_sec)  # a REAL, uncensored measurement
            else:
                collision_free_steps += 1  # censored -- horizon-clamped, never averaged as if real
            unrecoverable = unrecoverable or telemetry.unrecoverable
        if collision:
            collided = True
        if target:
            success = True
        if done:
            # The ENVIRONMENT node is authoritative about when an episode
            # times out (its own profile's episode_length_steps, which may
            # differ from THIS evaluation profile's) -- so "timeout" is
            # simply "done fired without success or collision", never
            # re-derived from comparing step counts across two configs
            # (that comparison silently under-counted timeouts whenever the
            # two profiles' episode_length_steps differed -- see
            # docs/RESEARCH_PROTOCOL.md's known-limitations note).
            timeout = not (collided or success)
            break
    else:
        # section item-1: the for-loop exhausted `episode_length_steps`
        # WITHOUT the environment ever reporting `done=True` (e.g. the live
        # environment_node's own training-profile episode_length_steps is
        # LONGER than this evaluation profile's `evaluation.max_episode_steps`
        # budget) -- previously left `success`/`collided`/`timeout` all
        # False, silently recording this episode as neither a success, a
        # collision, NOR a timeout. A `for...else` (executes only when the
        # loop completes without `break`) is the exact "no early exit ever
        # happened" signal.
        timeout = not (collided or success)

    start = (scenario.spec.start_x, scenario.spec.start_y)
    goal = (scenario.spec.goal_x, scenario.spec.goal_y)
    straight_line = math.hypot(goal[0] - start[0], goal[1] - start[1])
    finite_clearance = [c for c in clearance_series if math.isfinite(c)]
    finite_ttc = [t for t in ttc_series if math.isfinite(t)]
    finite_stopping_margin = [m for m in stopping_margin_series if math.isfinite(m)]
    path_length = env.episode_path_length_m
    elapsed_sim_time = env.episode_elapsed_sim_time_sec
    final_pose = getattr(env, "latest_pose", None)
    final_goal_distance = (
        math.hypot(goal[0] - final_pose[0], goal[1] - final_pose[1])
        if final_pose is not None else None
    )
    goal_progress = straight_line - final_goal_distance if final_goal_distance is not None else None

    # Physical-plausibility sanity check (section P1-4): a path length wildly
    # inconsistent with steps*measured-average-velocity indicates a metric
    # computation bug (e.g. path length reset mid-episode, or velocity
    # samples from the wrong topic) -- fail loudly rather than publish a
    # silently-wrong number.
    avg_v = (sum(velocities) / len(velocities)) if velocities else 0.0
    steps_run = last_step_index + 1
    expected_path = abs(avg_v) * steps_run * profile.runtime.time_delta_sec
    if steps_run > 5 and expected_path > 0.5 and not (0.1 * expected_path <= path_length <= 10.0 * expected_path):
        raise RuntimeError(
            f"evaluation metric sanity check failed for scenario {scenario.scenario_id!r}: "
            f"path_length_m={path_length:.3f} is inconsistent with avg_velocity*time "
            f"({expected_path:.3f}) -- a metric computation bug is more likely than physically "
            f"plausible robot behaviour this far off"
        )

    return {
        # section P0-1: scenario_id ALONE doesn't prove two runs used the
        # SAME scenario if the benchmark's own YAML files were ever edited
        # between runs -- the scenario's own embedded seed is a second,
        # independent identity check (see run_benchmark's summary metadata).
        "scenario_id": scenario.scenario_id, "scenario_seed": scenario.spec.seed,
        "success": success, "collision": collided,
        "timeout": timeout, "unrecoverable": unrecoverable, "steps": steps_run,
        "path_length_m": path_length, "straight_line_distance_m": straight_line,
        "final_goal_distance_m": final_goal_distance,
        "goal_progress_m": goal_progress,
        "goal_progress_ratio": (goal_progress / straight_line
                                if goal_progress is not None and straight_line > 0.0 else None),
        "navigation_time_sec": elapsed_sim_time,
        "velocities_mps": velocities,
        "min_clearance_m": min(finite_clearance) if finite_clearance else None,
        "min_clearance_valid_count": len(finite_clearance),
        "ttc_values_sec": finite_ttc, "stopping_margin_values_m": finite_stopping_margin,
        "steering_values_rad": steering_values, "steering_rates_rad_s": steering_rates,
        "steering_limit_rad": profile.robot.steering_limit_rad, "control_deltas": control_deltas,
        "curvature_feasibility_violations": curvature_feasibility_violations,
        "physical_command_steps": steps_run,
        "emergency_stops": emergency_stop_count, "total_reward": total_reward,
        "risk_valid_steps": risk_valid_steps, "collision_free_steps": collision_free_steps,
    }


def run_benchmark(profile: Profile, agent, output_dir: str, env: EnvironmentClient = None,
                   run_metadata: dict = None) -> dict:
    """``env``: an already-constructed, already-``rclpy.init()``-ed client
    (evaluation_node.py's normal call path, so it can query dimensions
    BEFORE building the agent -- section 7/11: never hardcode state_dim).
    If omitted, a private client is created and destroyed internally
    (standalone/test use).

    ``run_metadata`` (section P0-1): arbitrary caller-supplied identifying
    fields (e.g. ``checkpoint_dir``, ``checkpoint_name``, ``training_profile``,
    ``algorithm``, ``action_space_mode``) merged directly into
    ``summary.json`` -- lets a reader comparing MULTIPLE baselines' result
    files verify after the fact which checkpoint/algorithm produced each
    one, on top of the ``benchmark``/``scenarios`` (scenario_id + seed)
    fields this function always adds itself."""
    from hunter_kinodynamic_rl.config.loader import default_config_root
    import glob
    import yaml

    benchmark_dir = os.path.join(default_config_root(), "benchmarks", profile.evaluation.benchmark)
    scenario_paths = {}
    # section item-2: a scenario_id/seed pair (already recorded per-episode)
    # proves two runs used a scenario with the SAME logical identity, but
    # NOT that the underlying YAML file's actual content was byte-identical
    # (e.g. someone hand-edited a benchmark scenario's obstacle layout
    # without renaming it) -- the file's own SHA-256 closes that gap.
    scenario_file_sha256 = {}
    for path in sorted(glob.glob(os.path.join(benchmark_dir, "*.yaml"))):
        with open(path) as f:
            sid = (yaml.safe_load(f) or {}).get("scenario_id", os.path.splitext(os.path.basename(path))[0])
        scenario_paths[sid] = path
        scenario_file_sha256[sid] = manager.sha256_of_file(path)

    scenarios = load_benchmark(profile.evaluation.benchmark)
    owns_env = env is None
    if owns_env:
        env = EnvironmentClient()
    episodes: List[dict] = []
    try:
        for scenario in scenarios:
            path = scenario_paths.get(scenario.scenario_id)
            if path is None:
                raise RuntimeError(f"could not resolve source path for scenario_id={scenario.scenario_id!r}")
            for _ in range(profile.evaluation.episodes_per_scenario):
                # section item-1: MUST fail immediately if the live
                # environment_node rejected the override (e.g. an unknown
                # parameter, or the service call itself failing) -- the
                # previous code ignored this return value entirely, so a
                # silently-rejected override would run `run_episode`
                # against WHATEVER scenario the node was already on
                # (a stale/procedural one), producing a plausible-looking
                # but wrong result with no error anywhere.
                if not env.set_scenario_override(path):
                    raise RuntimeError(
                        f"evaluation_node: environment_node rejected scenario_override_path={path!r} "
                        f"(scenario_id={scenario.scenario_id!r}) -- refusing to evaluate against "
                        "whatever scenario the live node was already on instead"
                    )
                episode = run_episode(env, agent, profile, scenario)
                episodes.append(episode)
    finally:
        env.set_scenario_override("")  # leave the env node back in normal (procedural) mode
        if owns_env:
            env.destroy_node()

    summary = metrics_mod.aggregate(episodes)
    # section P0-1: verifiable-from-the-result-file proof that this run
    # (and, when compared against another baseline's summary.json, THAT
    # run too) actually evaluated the same benchmark scenario set -- a
    # scenario_id string alone doesn't prove the underlying YAML wasn't
    # edited between runs; the scenario's own embedded seed is independent
    # corroboration. Deduplicated by scenario_id (episodes_per_scenario > 1
    # repeats the same scenario/seed pair multiple times).
    seen = {}
    for e in episodes:
        seen.setdefault(e["scenario_id"], e["scenario_seed"])
    summary["benchmark"] = profile.evaluation.benchmark
    summary["scenarios"] = [
        {"scenario_id": sid, "seed": seed, "scenario_file_sha256": scenario_file_sha256.get(sid)}
        for sid, seed in sorted(seen.items())
    ]
    summary["episodes_per_scenario"] = profile.evaluation.episodes_per_scenario
    # section item-2: ONE hash summarizing every scenario file this benchmark
    # NAME (`profile.evaluation.benchmark`) actually resolved to at run time
    # -- two summary.json files with the same benchmark_manifest_sha256 used
    # byte-identical scenario files, not just identically-named ones.
    summary["benchmark_manifest_sha256"] = sha256_of_obj(
        {sid: scenario_file_sha256[sid] for sid in sorted(scenario_file_sha256)})
    if run_metadata:
        summary.update(run_metadata)
    write_episode_csv(f"{output_dir}/episodes.csv", episodes)
    write_episode_jsonl(f"{output_dir}/episodes.jsonl", episodes)
    write_summary_json(f"{output_dir}/summary.json", summary)
    return summary
