"""Regression coverage for evaluation/benchmark_runner.py::run_episode --
specifically the P0-6 TTC-censoring fix (real vs no-collision-within-horizon
samples must be tracked separately, never blended into one "mean TTC").

benchmark_runner.py imports EnvironmentClient from training/trainer_base.py,
which imports rclpy at module scope -- self-skips cleanly on a bare host
checkout (mirrors test_environment_node.py's pattern). No live ROS graph or
Gazebo needed: run_episode only ever calls duck-typed methods/attributes on
its `env` argument, so a plain fake object satisfying that surface exercises
the real function logic without any ROS dependency at runtime.
"""

import dataclasses

import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("drl_agent_interfaces")

from hunter_kinodynamic_rl.config.loader import load_profile  # noqa: E402
from hunter_kinodynamic_rl.env.scenarios.benchmark_loader import BenchmarkScenario  # noqa: E402
from hunter_kinodynamic_rl.env.scenarios.procedural_generator import ScenarioSpec  # noqa: E402
from hunter_kinodynamic_rl.env.simulation import risk_telemetry as rt  # noqa: E402
from hunter_kinodynamic_rl.evaluation import metrics as metrics_mod  # noqa: E402
from hunter_kinodynamic_rl.evaluation.benchmark_runner import run_episode  # noqa: E402
from hunter_kinodynamic_rl.evaluation.fingerprint import sha256_of_obj  # noqa: E402
from hunter_kinodynamic_rl.rl.checkpointing import manager  # noqa: E402


class _FakeAgent:
    def select_action(self, state, deterministic=True):
        return [0.0, 0.5, 0.5]


class _FakeEnv:
    """``telemetry_script``: one rt.RiskTelemetry (or None -> a plain
    invalid() sample) per step, consumed in order; the last entry repeats
    once exhausted."""

    def __init__(self, telemetry_script):
        self._telemetry_script = telemetry_script
        self._step_i = 0
        self.episode_path_length_m = 1.0
        self.episode_elapsed_sim_time_sec = 1.0
        self.latest_v_mps = 0.5
        self.latest_center_steering_rad = 0.0

    def reset(self):
        self._step_i = 0
        return [0.0] * 4

    def step(self, action):
        idx = min(self._step_i, len(self._telemetry_script) - 1)
        telemetry = self._telemetry_script[idx]
        self._step_i += 1
        done = self._step_i >= len(self._telemetry_script)
        return [0.0] * 4, 0.0, done, False, False, 10.0, telemetry, None


def _scenario() -> BenchmarkScenario:
    spec = ScenarioSpec(seed=1, start_x=0.0, start_y=0.0, start_yaw=0.0, goal_x=5.0, goal_y=0.0)
    return BenchmarkScenario(scenario_id="fake_scenario", spec=spec)


def _valid_telemetry(step_id, collision_within_horizon, ttc_sec):
    return rt.RiskTelemetry(
        step_id=step_id, valid=True, risk_target=0.0, min_clearance_m=1.0, ttc_sec=ttc_sec,
        collision_within_horizon=collision_within_horizon, stopping_margin_m=1.0, unrecoverable=False,
        safer_alternative_margin=0.0, actor_candidate_index=0,
    )


def _profile():
    profile = load_profile("smoke_test")
    return dataclasses.replace(
        profile, training=dataclasses.replace(profile.training, episode_length_steps=5),
    )


def test_ttc_values_only_include_real_collision_within_horizon_samples():
    """The core P0-6 regression: a step where the risk assessment found NO
    collision within its horizon must NOT contribute its (censored,
    horizon-clamped) ttc_sec into ttc_values_sec."""
    profile = _profile()
    telemetry_script = [
        _valid_telemetry(1, collision_within_horizon=False, ttc_sec=3.0),  # censored (horizon clamp)
        _valid_telemetry(2, collision_within_horizon=True, ttc_sec=0.4),   # REAL near-miss
        _valid_telemetry(3, collision_within_horizon=False, ttc_sec=3.0),  # censored
        _valid_telemetry(4, collision_within_horizon=True, ttc_sec=0.6),   # REAL near-miss
        _valid_telemetry(5, collision_within_horizon=False, ttc_sec=3.0),  # censored
    ]
    env = _FakeEnv(telemetry_script)
    result = run_episode(env, _FakeAgent(), profile, _scenario())

    assert result["ttc_values_sec"] == pytest.approx([0.4, 0.6])
    assert result["risk_valid_steps"] == 5
    assert result["collision_free_steps"] == 3


def test_all_censored_episode_reports_zero_real_ttc_samples():
    profile = _profile()
    telemetry_script = [_valid_telemetry(i, collision_within_horizon=False, ttc_sec=3.0) for i in range(1, 6)]
    env = _FakeEnv(telemetry_script)
    result = run_episode(env, _FakeAgent(), profile, _scenario())

    assert result["ttc_values_sec"] == []
    assert result["risk_valid_steps"] == 5
    assert result["collision_free_steps"] == 5


def test_invalid_telemetry_steps_are_excluded_from_both_counters():
    profile = _profile()
    telemetry_script = [
        rt.invalid(step_id=1),
        _valid_telemetry(2, collision_within_horizon=True, ttc_sec=0.7),
        rt.invalid(step_id=3),
    ]
    env = _FakeEnv(telemetry_script)
    result = run_episode(env, _FakeAgent(), profile, _scenario())

    assert result["ttc_values_sec"] == pytest.approx([0.7])
    assert result["risk_valid_steps"] == 1
    assert result["collision_free_steps"] == 0


# --------------------------------------------------------------- section P0-1
def test_episode_result_records_the_scenario_seed():
    """section P0-1: the scenario's own embedded seed must be recorded per
    episode -- a second, independent identity check beyond scenario_id
    string equality (which can't detect a benchmark YAML edited between
    two runs being compared)."""
    profile = _profile()
    scenario = BenchmarkScenario(
        scenario_id="fake_scenario",
        spec=ScenarioSpec(seed=4242, start_x=0.0, start_y=0.0, start_yaw=0.0, goal_x=5.0, goal_y=0.0),
    )
    env = _FakeEnv([_valid_telemetry(1, collision_within_horizon=False, ttc_sec=3.0)])
    result = run_episode(env, _FakeAgent(), profile, scenario)
    assert result["scenario_id"] == "fake_scenario"
    assert result["scenario_seed"] == 4242


def test_run_benchmark_summary_records_scenarios_and_run_metadata(tmp_path, monkeypatch):
    """section P0-1: summary.json must carry (scenario_id, seed) pairs for
    every scenario actually evaluated, plus any caller-supplied run
    identity metadata (checkpoint/algorithm/profile) -- so two baselines'
    result files can be verifiably compared after the fact."""
    import yaml as yaml_mod

    from hunter_kinodynamic_rl.evaluation import benchmark_runner

    benchmark_name = "fake_benchmark"
    profile = dataclasses.replace(
        _profile(), evaluation=dataclasses.replace(_profile().evaluation, benchmark=benchmark_name))

    benchmark_dir = tmp_path / "benchmarks" / benchmark_name
    benchmark_dir.mkdir(parents=True)
    for sid in ("scenario_a", "scenario_b"):
        with open(benchmark_dir / f"{sid}.yaml", "w") as f:
            yaml_mod.safe_dump({"scenario_id": sid}, f)
    monkeypatch.setattr("hunter_kinodynamic_rl.config.loader.default_config_root", lambda: str(tmp_path))

    scenario_a = BenchmarkScenario(
        scenario_id="scenario_a", spec=ScenarioSpec(seed=1, start_x=0.0, start_y=0.0, start_yaw=0.0,
                                                      goal_x=5.0, goal_y=0.0))
    scenario_b = BenchmarkScenario(
        scenario_id="scenario_b", spec=ScenarioSpec(seed=2, start_x=0.0, start_y=0.0, start_yaw=0.0,
                                                      goal_x=5.0, goal_y=0.0))
    monkeypatch.setattr(benchmark_runner, "load_benchmark", lambda name: [scenario_a, scenario_b])

    class _ScenarioFakeEnv(_FakeEnv):
        def __init__(self):
            super().__init__([_valid_telemetry(1, collision_within_horizon=False, ttc_sec=3.0)])

        def set_scenario_override(self, path):
            return True

    summary = benchmark_runner.run_benchmark(
        profile, _FakeAgent(), str(tmp_path / "output"), env=_ScenarioFakeEnv(),
        run_metadata={"checkpoint_dir": "/fake/checkpoints", "algorithm": "tqc"},
    )
    assert summary["benchmark"] == benchmark_name
    scenario_a_sha = manager.sha256_of_file(str(benchmark_dir / "scenario_a.yaml"))
    scenario_b_sha = manager.sha256_of_file(str(benchmark_dir / "scenario_b.yaml"))
    assert summary["scenarios"] == [
        {"scenario_id": "scenario_a", "seed": 1, "scenario_file_sha256": scenario_a_sha},
        {"scenario_id": "scenario_b", "seed": 2, "scenario_file_sha256": scenario_b_sha},
    ]
    # section item-2: a real, independently-computable hash -- not a
    # placeholder -- summarizing every scenario file this run resolved.
    assert summary["benchmark_manifest_sha256"] == sha256_of_obj(
        {"scenario_a": scenario_a_sha, "scenario_b": scenario_b_sha})
    assert summary["checkpoint_dir"] == "/fake/checkpoints"
    assert summary["algorithm"] == "tqc"
