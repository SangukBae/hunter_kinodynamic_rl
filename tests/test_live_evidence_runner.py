"""Requirement H / defect-fix item 9: live-evidence smoke runner --
pure/importable pieces only (EventLog JSONL emission, leftover-process
check, preflight, default-profile fix) -- the live mission driver itself
requires Gazebo and is not executed here."""

import json
import os
import subprocess
import sys
import time

from hunter_kinodynamic_rl.evaluation.live_evidence_runner import (
    DEFAULT_LIVE_EVIDENCE_PROFILE, EventLog, PreflightResult, _check_leftover_processes,
    preflight_live_evidence,
)


def test_event_log_writes_one_json_object_per_line(tmp_path):
    path = str(tmp_path / "events.jsonl")
    log = EventLog(path)
    log.emit("reset_complete", elapsed_sec=1.5, foo="bar")
    log.emit("mission_start", goal_mission=(1.0, 2.0))
    log.close()

    with open(path) as f:
        lines = [json.loads(line) for line in f if line.strip()]
    assert len(lines) == 2
    assert lines[0]["event"] == "reset_complete"
    assert lines[0]["elapsed_sec"] == 1.5
    assert lines[0]["foo"] == "bar"
    assert "ts_unix" in lines[0]
    assert lines[1]["event"] == "mission_start"


def test_event_log_creates_parent_directory(tmp_path):
    path = str(tmp_path / "nested" / "dir" / "events.jsonl")
    log = EventLog(path)
    log.emit("x")
    log.close()
    assert os.path.isfile(path)


def test_default_profile_actually_exists():
    """Defect-fix item 9's core gap: the old default ('hierarchical_phase3')
    named a profile with no corresponding config file -- load_profile()
    failed on every default invocation. This pins that the CURRENT default
    is a real, loadable profile."""
    from hunter_kinodynamic_rl.config.loader import load_profile

    profile = load_profile(DEFAULT_LIVE_EVIDENCE_PROFILE)
    assert profile.name == DEFAULT_LIVE_EVIDENCE_PROFILE
    assert profile.long_horizon_world.enabled


def test_check_leftover_processes_with_no_pgid_reports_not_applicable():
    """Defect-fix item 9: launch_gazebo=False means this run owns no
    process group -- must never silently claim "clean" or check unrelated
    processes; reports an explicit not-applicable marker instead."""
    result = _check_leftover_processes(None)
    assert isinstance(result, list)
    assert any("not_applicable" in r for r in result)


def test_check_leftover_processes_scoped_to_own_process_group_finds_survivor():
    """Defect-fix item 9: the check must be SCOPED to this run's own
    process group, not a blanket system-wide pgrep -- proven here by
    spawning a real child process in its OWN new session/process group and
    confirming the scoped check finds it by pgid (never by matching some
    unrelated system process by name alone)."""
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(5)"], start_new_session=True,
    )
    try:
        pgid = os.getpgid(proc.pid)
        time.sleep(0.2)  # let it actually start
        survivors = _check_leftover_processes(pgid)
        assert any(str(proc.pid) in line for line in survivors)
    finally:
        proc.kill()
        proc.wait(timeout=5.0)


def test_check_leftover_processes_scoped_group_with_no_survivors_is_clean():
    proc = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    pgid = os.getpgid(proc.pid)
    proc.wait(timeout=5.0)
    time.sleep(0.2)
    survivors = _check_leftover_processes(pgid)
    assert survivors == []


class _FakeLongHorizonWorldConfig:
    def __init__(self, enabled):
        self.enabled = enabled


class _FakeHierarchicalTrainingConfig:
    def __init__(self, local_checkpoint_dir, local_checkpoint_name):
        self.local_checkpoint_dir = local_checkpoint_dir
        self.local_checkpoint_name = local_checkpoint_name


class _FakeRuntimeConfig:
    time_delta_sec = 0.1


class _FakeProfile:
    """A minimal stand-in exposing only what preflight_live_evidence reads
    -- avoids needing a full real Profile + a real promoted checkpoint on
    disk just to test the preflight's own decision logic."""

    def __init__(self, *, long_horizon_enabled, checkpoint_dir, checkpoint_name, name="fake"):
        self.name = name
        self.long_horizon_world = _FakeLongHorizonWorldConfig(long_horizon_enabled)
        self.hierarchical_training = _FakeHierarchicalTrainingConfig(checkpoint_dir, checkpoint_name)
        self.runtime = _FakeRuntimeConfig()


def test_preflight_rejects_long_horizon_world_disabled(tmp_path):
    profile = _FakeProfile(long_horizon_enabled=False, checkpoint_dir=str(tmp_path), checkpoint_name="final")
    result = preflight_live_evidence(profile)
    assert not result.ok
    assert any("long_horizon_world" in e for e in result.errors)


def test_preflight_rejects_unpromoted_checkpoint(tmp_path):
    profile = _FakeProfile(long_horizon_enabled=True, checkpoint_dir=str(tmp_path), checkpoint_name="final")
    result = preflight_live_evidence(profile)
    assert not result.ok
    assert not result.info["local_checkpoint_promoted"]
    assert any("not a valid PROMOTED checkpoint" in e for e in result.errors)


def test_preflight_reports_timeout_budget():
    profile = _FakeProfile(long_horizon_enabled=False, checkpoint_dir="/tmp/x", checkpoint_name="final")
    result = preflight_live_evidence(profile)
    assert result.info["local_control_period_sec"] == 0.1


def test_preflight_result_is_a_dataclass_with_ok_errors_info():
    r = PreflightResult(ok=True)
    assert r.errors == []
    assert r.info == {}
