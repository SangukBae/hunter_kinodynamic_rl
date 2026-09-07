"""Sequence replay durability, semantics and exact-resume tests."""

from dataclasses import replace
import hashlib
from pathlib import Path

import numpy as np
import pytest

from hunter_kinodynamic_rl.rl.replay import (
    EpisodeHeader, EpisodeStore, SequenceBuffer, SequenceIndex,
)
from hunter_kinodynamic_rl.rl.replay.sequence_schema import (
    TerminationReason, discount_from_dt, validate_candidate_event_tuple,
    validate_terminal_semantics,
)
from hunter_kinodynamic_rl.training.tractor_dataset_validation import validate_dataset


def _header(episode_id: str, steps: int, split_id: str = "development"):
    geometry_sha256 = hashlib.sha256(episode_id.encode("utf-8")).hexdigest()
    return EpisodeHeader(
        episode_id=episode_id,
        scenario_id="static_fixture",
        split_id=split_id,
        seed=7,
        software_commit="a" * 40,
        dirty_state_digest="clean",
        container_image_digest="sha256:" + "b" * 64,
        resolved_config_hash="c" * 64,
        protocol_version="tractor_protocol_v1",
        environment_attestation_hash="d" * 64,
        robot_attestation_hash="e" * 64,
        observation_contract_hash="f" * 64,
        action_contract_hash="1" * 64,
        trajectory_contract_hash="2" * 64,
        start_utc="2026-09-07T00:00:00Z",
        end_utc="2026-09-07T00:00:01Z",
        termination_reason=TerminationReason.GOAL.value,
        step_count=steps,
        sensor_source="fixture",
        localization_source="fixture",
        controller_source="fixture",
        clock_domain="sim_time",
        group_id=geometry_sha256,
        scenario_family="open_static",
        obstacle_contract="static_only",
        scenario_geometry_sha256=geometry_sha256,
    )


def _columns(steps: int, reset_at: int | None = None):
    timestamps = np.arange(steps, dtype=np.int64) * 100_000_000 + 1
    terminated = np.zeros(steps, dtype=bool)
    terminated[-1] = True
    reasons = np.full(steps, TerminationReason.NONE.value, dtype="U16")
    reasons[-1] = TerminationReason.GOAL.value
    epochs = np.zeros(steps, dtype=np.int64)
    scene_reset = np.zeros(steps, dtype=bool)
    response_reset = np.zeros(steps, dtype=bool)
    scene_reset[0] = response_reset[0] = True
    if reset_at is not None:
        epochs[reset_at:] = 1
        scene_reset[reset_at] = response_reset[reset_at] = True
    return {
        "decision_timestamp_ns": timestamps,
        "observation": np.zeros((steps, 328), dtype=np.float32),
        "scan_valid": np.ones((steps, 4, 80), dtype=bool),
        "motion_delta_from_previous": np.tile(
            np.asarray([0.0, 0.0, 0.0, 0.1], dtype=np.float32), (steps, 3, 1)
        ),
        "motion_valid": np.ones((steps, 3), dtype=bool),
        "previous_intent_valid": np.ones(steps, dtype=bool),
        "previous_command_published": np.zeros((steps, 2), dtype=np.float32),
        "previous_command_valid": np.ones(steps, dtype=bool),
        "vehicle_response_valid": np.ones((steps, 3), dtype=bool),
        "pose_covariance": np.zeros((steps, 3, 3), dtype=np.float32),
        "localization_valid": np.ones(steps, dtype=bool),
        "localization_confidence": np.ones(steps, dtype=np.float32),
        "localization_confidence_valid": np.ones(steps, dtype=bool),
        "sensor_freshness_sec": np.zeros(steps, dtype=np.float32),
        "sensor_freshness_valid": np.ones(steps, dtype=bool),
        "reset_epoch": epochs,
        "scene_reset": scene_reset,
        "response_reset": response_reset,
        "action_normalized_requested": np.zeros((steps, 3), dtype=np.float32),
        "reward": np.arange(steps, dtype=np.float32),
        "transition_dt_sec": np.full(steps, 0.1, dtype=np.float32),
        "discount_factor": np.full(steps, 0.99, dtype=np.float32),
        "next_observation_valid": np.ones(steps, dtype=bool),
        "terminated": terminated,
        "truncated": np.zeros(steps, dtype=bool),
        "termination_reason": reasons,
        "bellman_sample_valid": np.ones(steps, dtype=bool),
    }


def test_terminal_and_event_semantics_are_closed():
    validate_terminal_semantics(True, False, False, "goal", True)
    validate_terminal_semantics(False, True, True, "time_limit", True)
    validate_terminal_semantics(False, True, False, "sensor_failure", False)
    with pytest.raises(ValueError):
        validate_terminal_semantics(True, True, True, "goal", True)
    validate_candidate_event_tuple(
        valid=True, observed=True, event_step=2, cause=1, censor_step=2, horizon=3
    )
    validate_candidate_event_tuple(
        valid=True, observed=False, event_step=-1, cause=-1, censor_step=2, horizon=3
    )
    with pytest.raises(ValueError):
        validate_candidate_event_tuple(
            valid=True, observed=True, event_step=1, cause=1, censor_step=2, horizon=3
        )


def test_measured_time_discount_rule():
    result = discount_from_dt(np.asarray([0.1, 0.2]), 0.99, 0.1)
    assert np.allclose(result, [0.99, 0.99 ** 2])


def test_episode_is_immutable_checksummed_and_fail_closed(tmp_path: Path):
    store = EpisodeStore(tmp_path)
    header = _header("episode-1", 5)
    path, digest = store.append(header, _columns(5))
    loaded_header, loaded = store.load("episode-1", digest)
    assert loaded_header == header
    assert loaded["reward"].tolist() == list(range(5))
    with pytest.raises(FileExistsError):
        store.append(header, _columns(5))
    with pytest.raises(RuntimeError, match="checksum"):
        store.load("episode-1", "0" * 64)


def test_windows_never_cross_reset_and_burn_in_starts_at_epoch(tmp_path: Path):
    store = EpisodeStore(tmp_path)
    store.append(_header("episode-reset", 8), _columns(8, reset_at=4))
    index = SequenceIndex.build(store, loss_window=3)
    assert len(index.windows) == 4
    assert all(window.reset_epoch in (0, 1) for window in index.windows)
    for window in index.windows:
        assert not (window.loss_start < 4 < window.loss_end)
        assert window.burn_start == (0 if window.reset_epoch == 0 else 4)


def test_sampler_exact_resume_restores_rng_and_draw_ordinal(tmp_path: Path):
    store = EpisodeStore(tmp_path)
    store.append(_header("episode-a", 7), _columns(7))
    index = SequenceIndex.build(store, loss_window=2)
    first = SequenceBuffer(store, index, seed=21)
    first.sample(3)
    state = first.state_dict()
    expected = first.sample(4)

    resumed = SequenceBuffer(store, index, seed=999)
    resumed.load_state_dict(state)
    actual = resumed.sample(4)
    assert [item.window for item in actual] == [item.window for item in expected]
    assert [item.sample_draw_ordinal for item in actual] == [3, 4, 5, 6]


def test_split_assignment_is_episode_level(tmp_path: Path):
    store = EpisodeStore(tmp_path)
    store.append(_header("development-episode", 4), _columns(4))
    store.append(_header("test-episode", 4, "locked_test"), _columns(4))
    index = SequenceIndex.build(store, loss_window=2)
    index.validate_split_isolation()
    buffer = SequenceBuffer(store, index, seed=0)
    assert all(sample.window.split_id == "locked_test" for sample in buffer.sample(2, "locked_test"))


def test_dataset_validator_detects_geometry_leak_even_if_scenario_ids_differ(tmp_path: Path):
    store = EpisodeStore(tmp_path)
    first = _header("development-geometry", 4, "development")
    second = replace(
        _header("test-renamed-copy", 4, "locked_test"),
        group_id=first.group_id,
        scenario_geometry_sha256=first.scenario_geometry_sha256,
    )
    store.append(first, _columns(4))
    store.append(second, _columns(4))
    report = validate_dataset(tmp_path, loss_window=2)
    assert not report.ok
    assert any("geometry" in error and "leaks" in error for error in report.errors)
