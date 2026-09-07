"""section P1-9: evaluation must restore the CHECKPOINT'S OWN training
architecture (action_space/features/risk/counterfactual/hyperparameters),
never the requested `--profile`'s own declarations for those sections --
only `evaluation`/`reward`/`scenario` may come from the requested profile.

nodes/evaluation_node.py has an rclpy import at module scope, but
build_effective_profile() itself is pure config-dict-in/Profile-out --
ROS-free and host-testable directly.
"""

import dataclasses
import json
import os

import pytest

pytest.importorskip("rclpy")  # evaluation_node.py imports rclpy at module scope
pytest.importorskip("drl_agent_interfaces")  # ...and training.trainer_base.EnvironmentClient at module scope
pytest.importorskip("torch")  # ...and the RiskAgent/VanillaAgent kinodynamic_tqc/tqc modules at module scope

from hunter_kinodynamic_rl.config.loader import load_profile  # noqa: E402
from hunter_kinodynamic_rl.evaluation.fingerprint import (  # noqa: E402
    architecture_fingerprint, evaluation_contract_fingerprint,
)
from hunter_kinodynamic_rl.nodes.evaluation_node import (  # noqa: E402
    RestoreResult, _atomic_write_json, _EvalRunState, _restore_launch_time_evaluation_contract,
    _write_contract_restore_status, augment_summary_with_physics_calibration,
    build_agent, build_effective_profile, environment_client_kwargs, expected_dims,
    run_eval_body_with_contract_restore, validate_live_environment, validate_live_evaluation_contract,
)
from hunter_kinodynamic_rl.rl.algorithms.kinodynamic_tqc.agent import Agent as RiskAgent  # noqa: E402
from hunter_kinodynamic_rl.rl.algorithms.sac.agent import Agent as SACAgent  # noqa: E402
from hunter_kinodynamic_rl.rl.algorithms.tqc.agent import Agent as VanillaAgent  # noqa: E402
from hunter_kinodynamic_rl.training.trainer_base import EnvServiceError  # noqa: E402


def _manifest_for(training_profile_name: str) -> dict:
    profile = load_profile(training_profile_name)
    return {
        "profile_name": training_profile_name,
        "resolved_config": dataclasses.asdict(profile),
    }


def test_missing_resolved_config_raises():
    """A pre-P0-1 checkpoint (schema_version=1) has no resolved_config --
    must fail loudly rather than silently falling back to the requested
    eval profile's own (potentially mismatched) architecture."""
    with pytest.raises(SystemExit):
        build_effective_profile({"profile_name": "kinodynamic_tqc"}, "evaluation_id")


def test_a_tier_checkpoint_architecture_survives_an_f_tier_eval_profile():
    """The core regression: baseline_tqc (ablation A, no risk critic at
    all) evaluated via evaluation_id.yaml (which unconditionally declares
    features.risk_critic=True, counterfactual_risk=True) must NOT pick up
    evaluation_id's architecture -- evaluation_node.main() would otherwise
    construct a RiskAgent with a randomly-initialized risk critic bolted
    onto an actor that was never trained with one."""
    manifest = _manifest_for("baseline_tqc")
    effective = build_effective_profile(manifest, "evaluation_id")

    assert effective.features.risk_critic is False
    assert effective.features.counterfactual_risk is False
    assert effective.action_space.mode == "legacy_waypoint"


def test_d_tier_checkpoint_keeps_actor_lambda_zero_not_es_nonzero():
    """D (kinodynamic_tqc_risk_supervised_only, actor_lambda=0.0) must not
    silently pick up E's/F's actor_lambda=0.1 by virtue of being evaluated
    through evaluation_id.yaml."""
    manifest = _manifest_for("kinodynamic_tqc_risk_supervised_only")
    effective = build_effective_profile(manifest, "evaluation_id")

    assert effective.features.risk_critic is True
    assert effective.risk.actor_lambda == pytest.approx(0.0)
    assert effective.features.counterfactual_risk is False


def test_evaluation_benchmark_and_episodes_come_from_the_requested_profile():
    """The ONE thing that SHOULD come from the requested --profile: which
    fixed benchmark to run and how many episodes per scenario."""
    manifest = _manifest_for("kinodynamic_tqc")
    effective = build_effective_profile(manifest, "evaluation_ood_geometry")

    assert effective.evaluation.benchmark == "ood_geometry"
    assert effective.evaluation.episodes_per_scenario == 1


def test_f_tier_checkpoint_evaluated_through_its_own_matching_profile_still_works():
    """Sanity check: the common case (evaluating a checkpoint through an
    eval profile whose architecture happens to already match) must still
    produce a valid, full-featured effective profile."""
    manifest = _manifest_for("kinodynamic_tqc_counterfactual")
    effective = build_effective_profile(manifest, "evaluation_id")

    assert effective.features.risk_critic is True
    assert effective.features.counterfactual_risk is True
    assert effective.counterfactual.num_candidates == 8


def test_eval_profile_missing_benchmark_still_raises():
    manifest = _manifest_for("kinodynamic_tqc")
    with pytest.raises(SystemExit):
        build_effective_profile(manifest, "kinodynamic_tqc")  # a training profile, no evaluation.benchmark set


# --------------------------------------------------------------- item-1 (round 2)
def test_runtime_comes_from_the_requested_eval_profile_not_the_checkpoint_training_profile():
    """The core round-2 fairness regression: a checkpoint trained with a
    non-default runtime (e.g. a different time_delta_sec) must NOT carry
    that into evaluation -- runtime is an evaluation-CONTRACT section
    exactly like reward/scenario (see evaluation/fingerprint.py's
    EVALUATION_CONTRACT_SECTIONS), overridden from the requested profile,
    never the checkpoint's own training-time value."""
    training_profile = load_profile("kinodynamic_tqc")
    edited_training = dataclasses.replace(
        training_profile,
        runtime=dataclasses.replace(training_profile.runtime, time_delta_sec=training_profile.runtime.time_delta_sec
                                     * 3, deterministic_stepping=not training_profile.runtime.deterministic_stepping),
    )
    manifest = {"profile_name": "kinodynamic_tqc", "resolved_config": dataclasses.asdict(edited_training)}
    effective = build_effective_profile(manifest, "evaluation_id")

    eval_profile_runtime = load_profile("evaluation_id").runtime
    assert effective.runtime == eval_profile_runtime
    assert effective.runtime.time_delta_sec != edited_training.runtime.time_delta_sec


# --------------------------------------------------------------- item-2 (round 2)
class _FakeEnvForContractValidation:
    def __init__(self, remote_evaluation_contract_fingerprint):
        self._fp = remote_evaluation_contract_fingerprint

    def get_remote_parameter(self, name):
        if name == "evaluation_contract_fingerprint_sha256":
            return self._fp
        raise AssertionError(f"unexpected parameter name {name!r}")


def test_validate_live_evaluation_contract_passes_when_fingerprints_match():
    profile = load_profile("evaluation_id")
    fp = evaluation_contract_fingerprint(profile)
    env = _FakeEnvForContractValidation(fp)
    assert validate_live_evaluation_contract(env, fp) == fp


def test_validate_live_evaluation_contract_raises_when_remote_parameter_unreadable():
    class _BrokenEnv:
        def get_remote_parameter(self, name):
            raise EnvServiceError("simulated: get_parameters service unavailable")

    with pytest.raises(SystemExit):
        validate_live_evaluation_contract(_BrokenEnv(), "any-fingerprint")


def test_validate_live_evaluation_contract_raises_on_mismatch_from_a_same_named_yaml_content_edit():
    """The EXPLICIT item-2 (round 2) regression: two profiles with the SAME
    name but DIFFERENT world/reward/runtime/metric content (simulated via
    dataclasses.replace, standing in for an on-disk edit -- see
    test_fingerprint.py for the pure fingerprint-level version of this same
    regression) must be caught by the live evaluation-contract check, never
    silently accepted just because both are called 'evaluation_id'."""
    original = load_profile("evaluation_id")
    edited_on_disk = dataclasses.replace(
        original,
        scenario=dataclasses.replace(original.scenario, world_size_m=original.scenario.world_size_m + 10.0),
        reward=dataclasses.replace(original.reward, goal_threshold_m=original.reward.goal_threshold_m + 5.0),
        runtime=dataclasses.replace(original.runtime, time_delta_sec=original.runtime.time_delta_sec * 5),
        evaluation=dataclasses.replace(original.evaluation, max_episode_steps=1),
    )
    requested_fp = evaluation_contract_fingerprint(original)
    # The LIVE environment applied the (edited-on-disk) content instead.
    live_fp = evaluation_contract_fingerprint(edited_on_disk)
    assert requested_fp != live_fp  # sanity: the edit actually changes the fingerprint

    env = _FakeEnvForContractValidation(live_fp)
    with pytest.raises(SystemExit, match="evaluation_contract_fingerprint_sha256"):
        validate_live_evaluation_contract(env, requested_fp)


# --------------------------------------------------------------- section P0-1


class _FakeDims:
    def __init__(self, state_dim, action_dim, max_action=1.0):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action


def test_expected_dims_matches_baseline_87d_contract():
    profile = load_profile("baseline_tqc")
    state_dim, action_dim = expected_dims(profile)
    assert (state_dim, action_dim) == (87, 3)


def test_expected_dims_matches_temporal_327d_contract():
    profile = load_profile("kinodynamic_tqc_temporal")
    state_dim, action_dim = expected_dims(profile)
    assert (state_dim, action_dim) == (80 * 4 + 8, 3)


def test_build_agent_dispatches_sac():
    """The core P0-1 regression: a checkpoint trained with
    algorithm.name=='sac' must construct a SACAgent -- previously this
    function (inlined in main()) only ever built RiskAgent/VanillaAgent
    (both TQC-family), silently wrong for a SAC checkpoint."""
    profile = load_profile("sac_baseline")
    assert profile.algorithm.name == "sac"
    dims = _FakeDims(*expected_dims(profile))
    agent = build_agent(profile, dims)
    assert isinstance(agent, SACAgent)


def test_build_agent_dispatches_vanilla_tqc():
    profile = load_profile("baseline_tqc")
    dims = _FakeDims(*expected_dims(profile))
    agent = build_agent(profile, dims)
    assert isinstance(agent, VanillaAgent)
    assert not isinstance(agent, RiskAgent)


def test_build_agent_dispatches_risk_aware_tqc():
    profile = load_profile("kinodynamic_tqc_counterfactual")
    dims = _FakeDims(*expected_dims(profile))
    agent = build_agent(profile, dims)
    assert isinstance(agent, RiskAgent)


class _FakeEnvForValidation:
    def __init__(self, remote_profile_name, remote_architecture_fingerprint=None):
        self._remote_profile_name = remote_profile_name
        self._remote_architecture_fingerprint = remote_architecture_fingerprint

    def get_remote_parameter(self, name):
        if name == "resolved_profile_name":
            return self._remote_profile_name
        if name == "architecture_fingerprint_sha256":
            return self._remote_architecture_fingerprint
        raise AssertionError(f"unexpected parameter name {name!r}")


def test_validate_live_environment_passes_when_everything_matches():
    profile = load_profile("baseline_tqc")
    dims = _FakeDims(*expected_dims(profile))
    fp = architecture_fingerprint(profile)
    env = _FakeEnvForValidation("baseline_tqc", fp)
    validate_live_environment(env, profile, "baseline_tqc", dims, fp)  # must not raise


def test_validate_live_environment_raises_on_dimension_mismatch():
    """section P0-1: must fail with a CLEAR error identifying the mismatch
    -- never proceed to construct an agent with garbage-shaped weights."""
    profile = load_profile("baseline_tqc")
    wrong_dims = _FakeDims(state_dim=999, action_dim=3)
    fp = architecture_fingerprint(profile)
    env = _FakeEnvForValidation("baseline_tqc", fp)
    with pytest.raises(SystemExit, match="dimensions"):
        validate_live_environment(env, profile, "baseline_tqc", wrong_dims, fp)


def test_validate_live_environment_raises_on_profile_name_mismatch_even_with_matching_dims():
    """The core regression this check exists for: legacy_waypoint_tqc and
    baseline_tqc BOTH have action_dim=3 and (after P0-4's fix)
    robot_state_dim=7 -- dimensions alone cannot tell them apart, but they
    decode action[0] as a totally different physical quantity (r vs kappa
    under different action_space.mode values in general; here both happen
    to be legacy_waypoint, so use a genuinely different-mode pair instead)."""
    trajectory_profile = load_profile("kinodynamic_tqc")  # robot_state_dim=8, action_space.mode=trajectory
    baseline_profile = load_profile("baseline_tqc")        # robot_state_dim=7, action_space.mode=legacy_waypoint
    assert expected_dims(trajectory_profile) != expected_dims(baseline_profile)  # sanity: dims DO differ here

    dims = _FakeDims(*expected_dims(trajectory_profile))
    fp = architecture_fingerprint(trajectory_profile)
    env = _FakeEnvForValidation("some_other_profile_name", fp)  # live env running under a DIFFERENT profile
    with pytest.raises(SystemExit, match="some_other_profile_name"):
        validate_live_environment(env, trajectory_profile, "kinodynamic_tqc", dims, fp)


def test_validate_live_environment_raises_when_remote_parameter_unreadable():
    class _BrokenEnv:
        def get_remote_parameter(self, name):
            raise EnvServiceError("simulated: get_parameters service unavailable")

    profile = load_profile("baseline_tqc")
    dims = _FakeDims(*expected_dims(profile))
    with pytest.raises(SystemExit):
        validate_live_environment(_BrokenEnv(), profile, "baseline_tqc", dims, architecture_fingerprint(profile))


def test_validate_live_environment_raises_on_architecture_fingerprint_mismatch():
    """section item-2: the core new regression -- profile NAME and
    dimensions both match, but the live environment_node's own
    architecture_fingerprint_sha256 differs from the checkpoint's own
    (frozen resolved_config) fingerprint, e.g. because the profile YAML on
    disk was edited between training and now. Must fail fast, never
    silently proceed on a name-only match."""
    profile = load_profile("baseline_tqc")
    dims = _FakeDims(*expected_dims(profile))
    env = _FakeEnvForValidation("baseline_tqc", "deadbeef" * 8)  # a fingerprint that can never match
    with pytest.raises(SystemExit, match="architecture_fingerprint_sha256"):
        validate_live_environment(env, profile, "baseline_tqc", dims, architecture_fingerprint(profile))


# --------------------------------------------------------------- item-4 (round 5, live-E2E finding)
def test_validate_live_environment_passes_when_both_sides_were_launched_by_an_explicit_path():
    """The exact failure a real live-Gazebo E2E run hit (round 5, item 4):
    a checkpoint trained via `-p profile:=/abs/path/to/smoke_sac.yaml` gets
    manifest profile_name='smoke_sac' (load_profile strips a path down to
    its basename-without-extension). Before this fix, evaluation_node.py
    compared that stripped name against the LIVE environment_node's raw
    `profile` ROS parameter -- which, when the live node was ALSO
    (correctly) relaunched with that exact same path for architecture
    consistency, is still the full path string, never 'smoke_sac' --
    causing a spurious rejection of a live environment that was, in fact,
    running the checkpoint's own exact training profile. Comparing against
    `resolved_profile_name` instead (this fix) closes the gap."""
    profile = load_profile("sac_baseline")
    dims = _FakeDims(*expected_dims(profile))
    fp = architecture_fingerprint(profile)
    # The live node's raw CLI value would have been the full path; its
    # RESOLVED name (what it now exposes) is the stripped short form --
    # exactly what a path-launched checkpoint's manifest also records.
    env = _FakeEnvForValidation("sac_baseline", fp)
    validate_live_environment(env, profile, "sac_baseline", dims, fp)  # must not raise


# --------------------------------------------------------------- item-1 (round 3)
def test_environment_client_kwargs_carries_the_requested_profiles_own_telemetry_timeouts():
    """The core round-3 regression: EnvironmentClient's constructor
    previously fell back to its own hardcoded defaults (1.0s/5.0s) for the
    risk-telemetry wait budgets -- a false guarantee, since these fields
    ARE part of the hashed runtime evaluation contract. A profile with
    DELIBERATELY non-default values must produce THOSE exact values here,
    never the EnvironmentClient default."""
    base = load_profile("evaluation_id")
    edited = dataclasses.replace(
        base, runtime=dataclasses.replace(
            base.runtime, risk_telemetry_wait_timeout_sec=7.25, risk_telemetry_reset_marker_timeout_sec=13.5))
    kwargs = environment_client_kwargs(edited)
    assert kwargs == {
        "telemetry_wait_timeout_sec": 7.25,
        "reset_marker_wait_timeout_sec": 13.5,
        "wheelbase_m": edited.robot.wheelbase_m,
        "track_width_m": edited.robot.track_width_m,
    }
    # sanity: genuinely different from EnvironmentClient's own hardcoded defaults
    assert kwargs["telemetry_wait_timeout_sec"] != 1.0
    assert kwargs["reset_marker_wait_timeout_sec"] != 5.0


# --------------------------------------------------------------- item-2 (round 3)
class _FakeEnvForRestore:
    def __init__(self, set_override_success=True, reset_raises=None, restored_fp="fp-restored"):
        self.set_override_calls = []
        self.reset_calls = 0
        self._set_override_success = set_override_success
        self._reset_raises = reset_raises
        self._restored_fp = restored_fp

    def set_evaluation_contract_override(self, path):
        self.set_override_calls.append(path)
        return self._set_override_success

    def reset(self):
        self.reset_calls += 1
        if self._reset_raises is not None:
            raise self._reset_raises
        return None

    def get_remote_parameter(self, name):
        assert name == "evaluation_contract_fingerprint_sha256"
        return self._restored_fp


def test_restore_forces_a_reset_so_the_restoration_actually_applies_before_returning():
    """The core round-3 regression: the round-2 version only called
    set_evaluation_contract_override("") and returned -- since
    _resolve_evaluation_contract_override only runs INSIDE /reset, the
    restoration would not actually take effect until whenever the NEXT
    /reset happened (possibly never). Must force one right here."""
    env = _FakeEnvForRestore(restored_fp="launch-fp")
    _restore_launch_time_evaluation_contract(env, "launch-fp", "/tmp/some_override.yaml")
    assert env.set_override_calls == [""]
    assert env.reset_calls == 1  # NOT zero


def test_restore_reports_a_rejected_parameter_clear_and_does_not_force_a_reset(capsys):
    env = _FakeEnvForRestore(set_override_success=False)
    _restore_launch_time_evaluation_contract(env, "launch-fp", "/tmp/some_override.yaml")
    assert env.reset_calls == 0
    captured = capsys.readouterr()
    assert "WARNING" in captured.out
    assert "REJECTED" in captured.out


def test_restore_reports_a_fingerprint_mismatch_after_reset_loudly(capsys):
    """A restoration that runs (parameter cleared, reset forced) but does
    NOT actually recover the launch-time contract (fingerprint mismatch)
    must be REPORTED, never silently accepted as success."""
    env = _FakeEnvForRestore(restored_fp="some-other-fp")
    _restore_launch_time_evaluation_contract(env, "launch-fp", "/tmp/some_override.yaml")
    assert env.reset_calls == 1
    captured = capsys.readouterr()
    assert "WARNING" in captured.out
    assert "FAILED to restore" in captured.out


def test_restore_matching_fingerprint_reports_no_warning(capsys):
    env = _FakeEnvForRestore(restored_fp="launch-fp")
    _restore_launch_time_evaluation_contract(env, "launch-fp", "/tmp/some_override.yaml")
    captured = capsys.readouterr()
    assert "WARNING" not in captured.out


def test_restore_never_raises_even_if_env_reset_itself_raises(capsys):
    """Cleanup (a `finally`-block helper) must never crash the process on
    top of -- or mask -- whatever exception, if any, is already
    propagating out of the try body it cleans up after."""
    env = _FakeEnvForRestore(reset_raises=RuntimeError("simulated: reset failed"))
    _restore_launch_time_evaluation_contract(env, "launch-fp", "/tmp/some_override.yaml")  # must not raise
    captured = capsys.readouterr()
    assert "WARNING" in captured.out
    assert "simulated: reset failed" in captured.out


def test_restore_with_no_captured_launch_fingerprint_reports_it_cannot_verify(capsys):
    """If an earlier failure interrupted main() before the launch-time
    fingerprint could ever be captured, restoration must say so explicitly
    -- never silently claim success it cannot actually verify."""
    env = _FakeEnvForRestore()
    _restore_launch_time_evaluation_contract(env, None, "/tmp/some_override.yaml")
    assert env.reset_calls == 1
    captured = capsys.readouterr()
    assert "WARNING" in captured.out
    assert "never captured" in captured.out


# --------------------------------------------------------------- item-4 (round 4, override-restore-failure)
def test_restore_returns_ok_true_on_a_clean_restore():
    """section item-4: _restore_launch_time_evaluation_contract now returns
    a STRUCTURED RestoreResult, not just a print -- the happy path must
    report ok=True with the restored fingerprint attached."""
    env = _FakeEnvForRestore(restored_fp="launch-fp")
    result = _restore_launch_time_evaluation_contract(env, "launch-fp", "/tmp/some_override.yaml")
    assert result == RestoreResult(ok=True, reason=None, restored_fingerprint="launch-fp")


def test_restore_returns_ok_false_on_rejected_parameter_clear():
    env = _FakeEnvForRestore(set_override_success=False)
    result = _restore_launch_time_evaluation_contract(env, "launch-fp", "/tmp/some_override.yaml")
    assert result.ok is False
    assert result.reason == "set_evaluation_contract_override_rejected"


def test_restore_returns_ok_false_when_reset_raises():
    env = _FakeEnvForRestore(reset_raises=RuntimeError("simulated: reset failed"))
    result = _restore_launch_time_evaluation_contract(env, "launch-fp", "/tmp/some_override.yaml")
    assert result.ok is False
    assert "simulated: reset failed" in result.reason


def test_restore_returns_ok_false_on_fingerprint_mismatch():
    env = _FakeEnvForRestore(restored_fp="some-other-fp")
    result = _restore_launch_time_evaluation_contract(env, "launch-fp", "/tmp/some_override.yaml")
    assert result.ok is False
    assert result.reason == "fingerprint_mismatch_after_restore"
    assert result.restored_fingerprint == "some-other-fp"


class _FakeEnvForOrchestration(_FakeEnvForRestore):
    """Only the restore-related surface is used by
    run_eval_body_with_contract_restore's own finally block -- `_body` in
    these tests never calls env.* itself, it only mutates `state`."""


def test_body_success_and_restore_failure_raises_systemexit_not_exit_zero(tmp_path):
    """The core item-4 regression this whole mechanism exists to close: the
    evaluation BODY succeeds cleanly, but the post-run contract restoration
    fails (set_evaluation_contract_override rejected) -- the overall call
    must NOT return normally (which main() would then exit 0 from); it must
    raise, so a caller checking only the process exit code observes
    failure."""
    env = _FakeEnvForOrchestration(set_override_success=False)
    state = _EvalRunState()

    def _body(state):
        state.output_dir = str(tmp_path)
        state.override_path = str(tmp_path / "override.yaml")
        state.launch_time_evaluation_contract_fp = "launch-fp"
        state.override_applied = True
        # ... the rest of a real body would run the benchmark here; this
        # fake body returning normally IS "the evaluation body succeeded".

    with pytest.raises(SystemExit, match="evaluation contract afterward FAILED"):
        run_eval_body_with_contract_restore(env, _body, state)
    # section item-4: the failure was also persisted as a durable artifact,
    # not just printed -- see _write_contract_restore_status.
    status_path = tmp_path / "contract_restore_status.json"
    assert status_path.is_file()
    import json as _json
    status = _json.loads(status_path.read_text())
    assert status["ok"] is False
    assert status["reason"] == "set_evaluation_contract_override_rejected"


def test_body_exception_propagates_unmasked_even_when_restore_also_fails(tmp_path):
    """The other half of the item-4 guarantee: when the evaluation BODY
    itself raises, that ORIGINAL exception must propagate -- even if the
    cleanup-time contract restoration ALSO fails. A RestoreResult(ok=False)
    must never replace/mask the real failure with a generic SystemExit
    about contract restoration; the restore failure is still reported
    (printed + written to contract_restore_status.json), just not by
    raising over the original exception."""
    env = _FakeEnvForOrchestration(reset_raises=RuntimeError("simulated: reset failed during restore"))
    state = _EvalRunState()

    def _body(state):
        state.output_dir = str(tmp_path)
        state.override_path = str(tmp_path / "override.yaml")
        state.launch_time_evaluation_contract_fp = "launch-fp"
        state.override_applied = True
        raise RuntimeError("simulated: the evaluation body itself failed")

    with pytest.raises(RuntimeError, match="the evaluation body itself failed"):
        run_eval_body_with_contract_restore(env, _body, state)
    status_path = tmp_path / "contract_restore_status.json"
    assert status_path.is_file()  # restore was still attempted and its failure still recorded


def test_body_success_and_restore_success_returns_normally(tmp_path):
    env = _FakeEnvForOrchestration(restored_fp="launch-fp")
    state = _EvalRunState()

    def _body(state):
        state.output_dir = str(tmp_path)
        state.override_path = str(tmp_path / "override.yaml")
        state.launch_time_evaluation_contract_fp = "launch-fp"
        state.override_applied = True

    run_eval_body_with_contract_restore(env, _body, state)  # must not raise
    status_path = tmp_path / "contract_restore_status.json"
    assert status_path.is_file()
    import json as _json
    assert _json.loads(status_path.read_text())["ok"] is True


def test_body_success_with_no_override_ever_applied_skips_restore_entirely(tmp_path):
    """If the body never even reached the point of applying an override
    (state.override_applied stays False), there is nothing to restore --
    must return normally without calling env.reset()/writing a status
    file."""
    env = _FakeEnvForOrchestration()
    state = _EvalRunState()

    def _body(state):
        pass  # never touches state.override_applied

    run_eval_body_with_contract_restore(env, _body, state)  # must not raise
    assert env.reset_calls == 0
    assert not (tmp_path / "contract_restore_status.json").is_file()


# --------------- code review: restore-status file durability/atomicity fix
def test_atomic_write_json_leaves_only_the_final_file_no_tmp_droppings(tmp_path):
    path = str(tmp_path / "contract_restore_status.json")
    _atomic_write_json(path, {"ok": True, "reason": None})

    assert json.loads((tmp_path / "contract_restore_status.json").read_text()) == {"ok": True, "reason": None}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["contract_restore_status.json"]  # no leftover .tmp


def test_atomic_write_json_raises_and_leaves_no_tmp_file_when_the_write_itself_fails(tmp_path, monkeypatch):
    """A failure partway through writing the temp file's content (before
    os.replace ever runs) must never leave a half-written .tmp file behind,
    and must never create the final path at all."""
    path = str(tmp_path / "contract_restore_status.json")

    def _raising_dump(*args, **kwargs):
        raise OSError("simulated: disk full mid-write")

    monkeypatch.setattr(json, "dump", _raising_dump)
    with pytest.raises(OSError, match="disk full mid-write"):
        _atomic_write_json(path, {"ok": True})

    assert list(tmp_path.iterdir()) == []  # no .tmp droppings, final path never created


def test_atomic_write_json_raises_and_leaves_no_tmp_file_when_fsync_fails(tmp_path, monkeypatch):
    def _raising_fsync(fd):
        raise OSError("simulated: fsync failed")

    monkeypatch.setattr(os, "fsync", _raising_fsync)
    with pytest.raises(OSError, match="fsync failed"):
        _atomic_write_json(str(tmp_path / "contract_restore_status.json"), {"ok": True})

    assert list(tmp_path.iterdir()) == []


def test_atomic_write_json_raises_and_preserves_the_previous_file_when_replace_fails(tmp_path, monkeypatch):
    """A failure in os.replace itself (already-written, fsynced temp file,
    but the atomic rename fails) must never touch whatever was previously
    at `path` -- a partial/half-applied durable artifact would be worse
    than simply keeping the old one and raising."""
    path = tmp_path / "contract_restore_status.json"
    path.write_text('{"ok": true, "reason": "previous_run"}')

    real_replace = os.replace

    def _raising_replace(src, dst):
        raise OSError("simulated: rename failed")

    monkeypatch.setattr(os, "replace", _raising_replace)
    with pytest.raises(OSError, match="rename failed"):
        _atomic_write_json(str(path), {"ok": True, "reason": "new_run"})

    monkeypatch.setattr(os, "replace", real_replace)
    assert json.loads(path.read_text()) == {"ok": True, "reason": "previous_run"}  # untouched
    assert sorted(p.name for p in tmp_path.iterdir()) == ["contract_restore_status.json"]  # tmp cleaned up


def test_body_success_and_restore_success_but_status_write_fails_still_raises_systemexit(tmp_path, monkeypatch):
    """code review (restore-status durability fix): a durability failure
    while persisting a SUCCESSFUL restore's status must still prevent a
    clean (exit 0) return -- a caller relying on contract_restore_status.json's
    mere presence as proof this was recorded must never see success when
    it silently wasn't written."""
    import hunter_kinodynamic_rl.nodes.evaluation_node as eval_node_mod

    env = _FakeEnvForOrchestration(restored_fp="launch-fp")
    state = _EvalRunState()

    def _body(state):
        state.output_dir = str(tmp_path)
        state.override_path = str(tmp_path / "override.yaml")
        state.launch_time_evaluation_contract_fp = "launch-fp"
        state.override_applied = True

    def _raising_write(output_dir, restore_result):
        raise OSError("simulated: disk full persisting contract_restore_status.json")

    monkeypatch.setattr(eval_node_mod, "_write_contract_restore_status", _raising_write)
    with pytest.raises(SystemExit, match="durably persisting contract_restore_status.json"):
        run_eval_body_with_contract_restore(env, _body, state)


def test_body_success_and_restore_failure_and_status_write_also_fails_reports_both(tmp_path, monkeypatch):
    """Both the restore failure AND the auxiliary status-write failure must
    be visible in the raised SystemExit -- neither one should silently
    swallow the other."""
    import hunter_kinodynamic_rl.nodes.evaluation_node as eval_node_mod

    env = _FakeEnvForOrchestration(set_override_success=False)
    state = _EvalRunState()

    def _body(state):
        state.output_dir = str(tmp_path)
        state.override_path = str(tmp_path / "override.yaml")
        state.launch_time_evaluation_contract_fp = "launch-fp"
        state.override_applied = True

    def _raising_write(output_dir, restore_result):
        raise OSError("simulated: disk full persisting contract_restore_status.json")

    monkeypatch.setattr(eval_node_mod, "_write_contract_restore_status", _raising_write)
    with pytest.raises(SystemExit) as exc_info:
        run_eval_body_with_contract_restore(env, _body, state)
    message = str(exc_info.value)
    assert "evaluation contract afterward FAILED" in message
    assert "set_evaluation_contract_override_rejected" in message
    assert "persisting this failure to contract_restore_status.json also failed" in message
    assert "disk full persisting contract_restore_status.json" in message


def test_body_exception_propagates_unmasked_even_when_status_write_also_fails(tmp_path, monkeypatch, capsys):
    """The strictest version of the no-masking guarantee: the evaluation
    BODY raises, restoration itself succeeds, but persisting the status
    file ALSO fails -- the original body exception must still be exactly
    what propagates, never replaced by anything related to the write
    failure (which is merely printed, per run_eval_body_with_contract_restore's
    own docstring)."""
    import hunter_kinodynamic_rl.nodes.evaluation_node as eval_node_mod

    env = _FakeEnvForOrchestration(restored_fp="launch-fp")
    state = _EvalRunState()

    def _body(state):
        state.output_dir = str(tmp_path)
        state.override_path = str(tmp_path / "override.yaml")
        state.launch_time_evaluation_contract_fp = "launch-fp"
        state.override_applied = True
        raise RuntimeError("simulated: the evaluation body itself failed")

    def _raising_write(output_dir, restore_result):
        raise OSError("simulated: disk full persisting contract_restore_status.json")

    monkeypatch.setattr(eval_node_mod, "_write_contract_restore_status", _raising_write)
    with pytest.raises(RuntimeError, match="the evaluation body itself failed"):
        run_eval_body_with_contract_restore(env, _body, state)
    # The auxiliary write failure was still surfaced (printed), just never
    # allowed to replace the original exception's type/message.
    assert "disk full persisting contract_restore_status.json" in capsys.readouterr().out


# --------------------------------------------------------------- item-1 (Gazebo physics-step reality-check fix)
class _FakeEnvForPhysicsCalibration:
    def __init__(self, verified, observed_dt_sec, raises=None):
        self._verified = verified
        self._observed_dt_sec = observed_dt_sec
        self._raises = raises

    def get_remote_parameter(self, name):
        if self._raises is not None:
            raise self._raises
        if name == "physics_step_calibration_verified":
            return self._verified
        if name == "physics_step_calibration_observed_dt_sec":
            return self._observed_dt_sec
        raise AssertionError(f"unexpected parameter name {name!r}")


def test_augment_summary_is_a_noop_for_a_non_deterministic_stepping_profile(tmp_path):
    profile = load_profile("evaluation_id")
    assert profile.runtime.deterministic_stepping is False  # sanity: every profile's default
    env = _FakeEnvForPhysicsCalibration(verified=True, observed_dt_sec=0.1)
    summary = {"foo": "bar"}
    result = augment_summary_with_physics_calibration(summary, env, profile, str(tmp_path))
    assert result == {"foo": "bar"}  # unchanged
    assert not (tmp_path / "summary.json").is_file()  # never re-written for a no-op


def test_augment_summary_records_verified_true_and_the_observed_dt(tmp_path):
    profile = dataclasses.replace(
        load_profile("evaluation_id"),
        runtime=dataclasses.replace(load_profile("evaluation_id").runtime, deterministic_stepping=True),
    )
    env = _FakeEnvForPhysicsCalibration(verified=True, observed_dt_sec=0.1)
    summary = {"foo": "bar"}
    result = augment_summary_with_physics_calibration(summary, env, profile, str(tmp_path))
    assert result["physics_step_calibration_verified"] is True
    assert result["physics_step_calibration_observed_dt_sec"] == pytest.approx(0.1)
    assert result["foo"] == "bar"  # original summary fields preserved

    written = json.loads((tmp_path / "summary.json").read_text())
    assert written["physics_step_calibration_verified"] is True


def test_augment_summary_records_verified_false_when_the_live_env_reports_it_never_ran(tmp_path):
    """The core item-1 regression: a summary.json for a deterministic-
    stepping run must NEVER claim physics was verified unless the live
    environment_node genuinely ran verify_physics_step_calibration and
    reported success."""
    profile = dataclasses.replace(
        load_profile("evaluation_id"),
        runtime=dataclasses.replace(load_profile("evaluation_id").runtime, deterministic_stepping=True),
    )
    env = _FakeEnvForPhysicsCalibration(verified=False, observed_dt_sec=None)
    summary = augment_summary_with_physics_calibration({}, env, profile, str(tmp_path))
    assert summary["physics_step_calibration_verified"] is False


def test_augment_summary_records_verified_false_when_the_live_env_predates_this_parameter(tmp_path):
    """An old/incompatible environment_node that never declared these
    parameters must not look indistinguishable from a genuinely verified
    run -- must record False, never silently omit the field."""
    profile = dataclasses.replace(
        load_profile("evaluation_id"),
        runtime=dataclasses.replace(load_profile("evaluation_id").runtime, deterministic_stepping=True),
    )
    env = _FakeEnvForPhysicsCalibration(verified=False, observed_dt_sec=None, raises=EnvServiceError("simulated"))
    summary = augment_summary_with_physics_calibration({}, env, profile, str(tmp_path))
    assert summary["physics_step_calibration_verified"] is False
    assert summary["physics_step_calibration_observed_dt_sec"] is None
