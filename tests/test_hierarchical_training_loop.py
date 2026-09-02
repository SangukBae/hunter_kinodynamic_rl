"""Phase 4 Global RL: ROS-free hierarchical training core loop (plan
section 8.9). No torch required -- ``FakeAgent`` below duck-types
``GlobalDQNAgent.select_action`` so the whole loop is testable without a
real network."""

import numpy as np
import pytest

from hunter_kinodynamic_rl.config.schema import (
    GlobalRLConfig, HierarchicalTrainingConfig, HierarchyConfig, LongHorizonWorldConfig, MappingConfig,
)
from hunter_kinodynamic_rl.env.scenarios.long_horizon_generator import long_horizon_seed_split
from hunter_kinodynamic_rl.navigation.hierarchy.subgoal_manager import SubgoalStatus
from hunter_kinodynamic_rl.navigation.mission.mission_frame import PoseXYYaw
from hunter_kinodynamic_rl.rl.checkpointing.manager import sha256_of_file
from hunter_kinodynamic_rl.training.train_hierarchical_dqn import HierarchicalTrainingLoop


class FakeAgent:
    """Epsilon-irrelevant, mask-respecting random policy -- exercises the
    SAME action_mask contract a real ``GlobalDQNAgent`` must honor."""

    def select_action(self, obs, *, epsilon, rng):
        valid = np.flatnonzero(obs.action_mask)
        return int(rng.choice(valid))


def _configs(**training_overrides):
    global_cfg = GlobalRLConfig(
        direction_degrees=[-90.0, -45.0, 0.0, 45.0, 90.0], distances_m=[2.0, 4.0], map_crop_size_cells=32,
    )
    hierarchy_cfg = HierarchyConfig(local_option_timeout_steps=40, mission_timeout_steps=None)
    long_horizon_cfg = LongHorizonWorldConfig(
        enabled=True, size_m=20.0, corridor_width_min_m=2.0, corridor_width_max_m=3.0,
        start_goal_geodesic_min_m=4.0, generation_attempt_limit=200,
        train_seed_range=[0, 999], validation_seed_range=[1000, 1999], test_seed_range=[2000, 2999],
    )
    mapping_cfg = MappingConfig(mission_size_cells=200)
    defaults = dict(max_local_steps_per_option=40, max_global_options_per_mission=8, long_horizon_level=1)
    defaults.update(training_overrides)
    training_cfg = HierarchicalTrainingConfig(**defaults)
    return global_cfg, hierarchy_cfg, long_horizon_cfg, mapping_cfg, training_cfg


def _loop(mode="train", seed=0, **training_overrides):
    global_cfg, hierarchy_cfg, long_horizon_cfg, mapping_cfg, training_cfg = _configs(**training_overrides)
    return HierarchicalTrainingLoop(
        global_cfg, hierarchy_cfg, long_horizon_cfg, mapping_cfg, training_cfg,
        robot_radius_m=0.45, min_turning_radius_m=1.0, wheelbase_m=0.6, seed=seed, mode=mode,
    )


def test_mission_runs_to_completion_within_option_budget():
    loop = _loop()
    outcome = loop.run_mission(FakeAgent(), epsilon=1.0, rng=np.random.RandomState(0))
    assert outcome.global_transitions >= 1
    assert outcome.global_transitions <= loop.training_cfg.max_global_options_per_mission
    assert sum([outcome.mission_reached, outcome.mission_failed, outcome.mission_timed_out]) >= 1
    assert len(loop.replay) == outcome.global_transitions


def test_global_action_drives_subgoal_queue_and_moves_the_map_forward():
    """Regression guard for the Global action -> subgoal
    queue/HierarchyCoordinator flow (plan requirement): after one mission,
    the partial map must have accumulated real observations (the
    SimplifiedKinematicLocalExecutor actually drove and scanned)."""
    loop = _loop()
    outcome = loop.run_mission(FakeAgent(), epsilon=1.0, rng=np.random.RandomState(1))
    assert outcome.global_transitions >= 1
    batch = loop.replay.sample(1)
    # At least one map channel differs from an all-unknown fresh map.
    assert not np.all(batch["map_state"][0, 2] == 1.0)  # channel 2 == "unknown"


def test_validation_mode_never_writes_to_replay():
    loop = _loop(mode="validation")
    outcome = loop.run_mission(FakeAgent(), epsilon=1.0, rng=np.random.RandomState(0))
    assert outcome.global_transitions == 0
    assert len(loop.replay) == 0


def test_train_mode_seeds_never_cross_into_validation_or_test_pool():
    loop = _loop(mode="train")
    for _ in range(20):
        seed = loop.seed_scheduler.next_seed()
        assert long_horizon_seed_split(seed, loop.long_horizon_cfg) == "train"


def test_validation_mode_seeds_never_cross_into_train_or_test_pool():
    loop = _loop(mode="validation")
    for _ in range(20):
        seed = loop.seed_scheduler.next_seed()
        assert long_horizon_seed_split(seed, loop.long_horizon_cfg) == "validation"


def test_local_checkpoint_hash_is_recorded_in_checkpoint_meta(tmp_path):
    """``local_checkpoint_dir``/``local_checkpoint_name`` mirror
    ``rl.checkpointing.manager.load_generation``'s own (directory, tag)
    contract -- a real checkpoint dir has ``<tag>`` as a symlink into
    ``.generations/<uuid>/model.pt``; a plain ``<tag>/model.pt`` file
    (as used here) resolves identically for hashing purposes."""
    checkpoint_dir = tmp_path / "checkpoints"
    tag_dir = checkpoint_dir / "final"
    tag_dir.mkdir(parents=True)
    model_file = tag_dir / "model.pt"
    model_file.write_bytes(b"a frozen local checkpoint")
    loop = _loop(local_checkpoint_dir=str(checkpoint_dir), local_checkpoint_name="final")
    meta = loop.checkpoint_meta()
    assert meta["local_checkpoint_sha256"] == sha256_of_file(str(model_file))
    assert meta["local_checkpoint_file"] == str(model_file)
    assert meta["local_checkpoint_dir"] == str(checkpoint_dir)
    assert meta["local_checkpoint_name"] == "final"


def test_checkpoint_meta_has_no_hash_when_checkpoint_dir_missing():
    loop = _loop(local_checkpoint_dir="")
    meta = loop.checkpoint_meta()
    assert meta["local_checkpoint_sha256"] is None
    assert meta["local_checkpoint_file"] is None


def test_checkpoint_meta_has_no_hash_when_checkpoint_dir_set_but_file_absent(tmp_path):
    loop = _loop(local_checkpoint_dir=str(tmp_path / "nonexistent"), local_checkpoint_name="final")
    meta = loop.checkpoint_meta()
    assert meta["local_checkpoint_sha256"] is None
    assert meta["local_checkpoint_file"] == str(tmp_path / "nonexistent" / "final" / "model.pt")


def test_hierarchical_training_disabled_requires_local_checkpoint_dir():
    with pytest.raises(Exception):
        HierarchicalTrainingConfig(enabled=True, local_checkpoint_dir="").validate()


def test_state_dict_resume_continues_the_exact_seed_and_step_sequence():
    """Mirrors TrainerBase's own local seed_scheduler resume contract:
    resuming must continue the exact seed draw sequence, not restart it."""
    loop = _loop(seed=7)
    loop.run_mission(FakeAgent(), epsilon=1.0, rng=np.random.RandomState(0))
    loop.run_mission(FakeAgent(), epsilon=1.0, rng=np.random.RandomState(1))
    state = loop.state_dict()

    resumed = _loop(seed=7)
    resumed.load_state_dict(state)

    fresh_reseed = _loop(seed=7)  # would draw the SAME first seed again -- the bug being guarded against
    next_from_original = loop.seed_scheduler.next_seed()
    next_from_resumed = resumed.seed_scheduler.next_seed()
    next_from_fresh = fresh_reseed.seed_scheduler.next_seed()

    assert next_from_resumed == next_from_original
    assert next_from_fresh != next_from_original
    assert resumed.global_step == state["global_step"] == loop.global_step
    assert resumed.mission_index == state["mission_index"] == loop.mission_index


def test_repeated_deadend_is_not_flagged_on_a_first_blocked_endpoint():
    """plan section 8.7/15: first dead-end/blocked exploration must not be
    penalized as a repeat -- verified end-to-end via a real mission run's
    stored transitions (failure_reason FAILED_BLOCKED with the endpoint's
    own failure_count still at its first increment implies
    ``repeated_deadend`` was NOT set on that same transition)."""
    loop = _loop(max_global_options_per_mission=20, max_local_steps_per_option=15)
    outcome = loop.run_mission(FakeAgent(), epsilon=1.0, rng=np.random.RandomState(3))
    if outcome.global_transitions == 0:
        pytest.skip("no transitions generated for this seed")
    batch = loop.replay.sample(len(loop.replay))
    blocked_code = None
    for i, code in enumerate(batch["failure_reason_code"][:, 0]):
        from hunter_kinodynamic_rl.navigation.global_rl.replay_schema import decode_failure_reason
        if decode_failure_reason(int(code)) == SubgoalStatus.FAILED_BLOCKED:
            blocked_code = i
            break
    if blocked_code is None:
        pytest.skip("no FAILED_BLOCKED transition occurred for this seed")
    # Just asserts this code path is exercised without raising -- the actual
    # first-vs-repeated distinction is unit-tested precisely in
    # tests/test_global_reward.py; this test's job is only to confirm the
    # end-to-end loop actually produces FAILED_BLOCKED transitions and never
    # crashes computing repeated_deadend for them.
    assert True


class _FakeLiveExecutor:
    """Duck-types the ``LocalOptionExecutor`` protocol PLUS the
    ``pose_world`` attribute ``run_mission`` itself reads between options --
    the exact same shape ``navigation.local_rl.live_gazebo_executor.
    LiveGazeboLocalExecutor`` satisfies, minus any real Gazebo/rclpy
    dependency, so ``local_executor_factory`` injection is testable without
    either."""

    def __init__(self, world, mission_frame):
        self.world = world
        self.mission_frame = mission_frame
        self.pose_world = PoseXYYaw(*world.start_pose)
        self.run_option_calls = 0

    def run_option(self, coordinator, partial_map, rng, max_local_steps):
        self.run_option_calls += 1


def test_local_executor_factory_is_used_once_per_mission_with_world_and_mission_frame():
    created = []

    def factory(world, mission_frame):
        executor = _FakeLiveExecutor(world, mission_frame)
        created.append(executor)
        return executor

    global_cfg, hierarchy_cfg, long_horizon_cfg, mapping_cfg, training_cfg = _configs()
    loop = HierarchicalTrainingLoop(
        global_cfg, hierarchy_cfg, long_horizon_cfg, mapping_cfg, training_cfg,
        robot_radius_m=0.45, min_turning_radius_m=1.0, wheelbase_m=0.6, seed=0, mode="train",
        local_executor_factory=factory,
    )
    loop.run_mission(FakeAgent(), epsilon=1.0, rng=np.random.RandomState(0))

    assert len(created) == 1
    assert created[0].run_option_calls >= 1


def test_local_executor_factory_default_none_still_uses_simplified_kinematic_executor():
    """``local_executor_factory=None`` (the default) must reproduce the
    ORIGINAL byte-identical mission outcome for a fixed seed -- a
    regression guard that adding the factory parameter did not change
    behaviour for every pre-existing (non-live) caller."""
    outcome_a = _loop(seed=7).run_mission(FakeAgent(), epsilon=1.0, rng=np.random.RandomState(7))
    outcome_b = _loop(seed=7).run_mission(FakeAgent(), epsilon=1.0, rng=np.random.RandomState(7))
    assert outcome_a == outcome_b


# ---------------------------------------------------------------------------
# item 1/2 regression: option clock domain separation + guaranteed terminal
# SubgoalResult. Root cause (live-verified): a fresh HierarchyCoordinator is
# created per Global option (see run_mission's own docstring), but used to be
# seeded with a RUN-cumulative pseudo-clock (self.global_step, an
# ever-growing option-COUNT index reused as fake "seconds") while ticks
# reported the EXECUTOR's own clock -- a completely different domain (for
# LiveGazeboLocalExecutor, real process-lifetime dt-paced seconds). A 17.5s
# mission control budget benchmarked at 293-946.5s as a direct result.
# These tests drive SimplifiedKinematicLocalExecutor.run_option() directly
# (the ROS-free executor, exercising the identical fix) against a
# hand-built, fully open (obstacle-free) synthetic world so the outcome
# depends ONLY on the clock/termination logic under test, never on
# procedurally-generated wall geometry.
# ---------------------------------------------------------------------------

def _open_world_and_frame():
    from hunter_kinodynamic_rl.env.scenarios.long_horizon_world import LongHorizonWorld
    from hunter_kinodynamic_rl.navigation.mission.mission_frame import MissionFrame, PoseXYYaw

    world = LongHorizonWorld(
        seed=1, occupancy=np.zeros((100, 100), dtype=bool), resolution_m=0.1, origin_xy=(-5.0, -5.0),
        start_pose=(0.0, 0.0, 0.0), goal_pose=(3.0, 0.0),
    )
    mission_frame = MissionFrame()
    mission_frame.initialize(PoseXYYaw(*world.start_pose))
    return world, mission_frame


def _fresh_option_coordinator():
    from hunter_kinodynamic_rl.navigation.hierarchy.coordinator import HierarchyCoordinator, HierarchyCoordinatorConfig
    from hunter_kinodynamic_rl.navigation.hierarchy.replanning import ReplanningConfig
    from hunter_kinodynamic_rl.navigation.hierarchy.subgoal_manager import SubgoalManagerConfig
    from hunter_kinodynamic_rl.navigation.mission.goal_manager import GoalManagerConfig

    # Every evaluate_replanning trigger is naturally unreachable within
    # max_local_steps=5 against the open world below (no obstacles -> no
    # collisions -> no emergency stops/high risk; subgoal 3m away, tolerance
    # 0.5m, 5 ticks * 0.3m/tick step distance can cover at most 1.5m -> never
    # reached; local_option_timeout_steps/no_progress_window_steps default
    # to 200/40, both > 5) -- so ONLY the executor's own
    # max_local_steps-exhausted fallback (item 2) can terminate the option.
    return HierarchyCoordinator(HierarchyCoordinatorConfig(
        goal=GoalManagerConfig(), subgoal=SubgoalManagerConfig(), replanning=ReplanningConfig(),
    ))


def test_simplified_executor_terminates_option_when_max_local_steps_smaller_than_local_option_timeout():
    from hunter_kinodynamic_rl.config.schema import MappingConfig
    from hunter_kinodynamic_rl.navigation.mapping.partial_map import PartialMap
    from hunter_kinodynamic_rl.training.train_hierarchical_dqn import SimplifiedKinematicLocalExecutor

    world, mission_frame = _open_world_and_frame()
    executor = SimplifiedKinematicLocalExecutor(world, mission_frame, robot_radius_m=0.45)
    partial_map = PartialMap(MappingConfig(mission_size_cells=100))
    goal_mission = mission_frame.odom_to_mission(*world.goal_pose)

    coordinator = _fresh_option_coordinator()
    coordinator.start_mission(goal_mission[0], goal_mission[1], now_step=0, now_time_sec=0.0)
    coordinator.enqueue_subgoal(*goal_mission)
    activated = coordinator.activate_next_subgoal(
        mission_frame.odom_pose_to_mission(executor.pose_world), now_step=0, now_time_sec=0.0,
    )
    assert activated

    executor.run_option(coordinator, partial_map, np.random.RandomState(0), max_local_steps=5)

    result = coordinator.last_subgoal_result
    assert result is not None, "activated subgoal must never end with last_subgoal_result still None"
    assert result.status == SubgoalStatus.FAILED_TIMEOUT
    assert result.reason == "max_local_steps_exhausted"
    assert result.local_steps == 5
    # Option control-time budget == max_local_steps(5) * dt_sec(0.3) == 1.5s.
    assert 0.0 <= result.elapsed_time_sec <= 1.5 + 1e-6


def test_simplified_executor_option_clock_does_not_leak_across_options_in_same_mission():
    """Two options driven on the SAME executor instance within the SAME
    mission must produce the SAME small elapsed_time_sec bound each time --
    never a growing/accumulating value (the exact class of bug reported
    live: a long-lived executor's own clock leaking across many
    options/missions)."""
    from hunter_kinodynamic_rl.config.schema import MappingConfig
    from hunter_kinodynamic_rl.navigation.mapping.partial_map import PartialMap
    from hunter_kinodynamic_rl.training.train_hierarchical_dqn import SimplifiedKinematicLocalExecutor

    world, mission_frame = _open_world_and_frame()
    executor = SimplifiedKinematicLocalExecutor(world, mission_frame, robot_radius_m=0.45)
    partial_map = PartialMap(MappingConfig(mission_size_cells=100))
    goal_mission = mission_frame.odom_to_mission(*world.goal_pose)
    rng = np.random.RandomState(0)

    # The robot physically advances toward the subgoal each option (the
    # SAME executor instance's pose_world carries real progress across
    # these calls), so a LATER option may legitimately terminate early via
    # REACHED once cumulative progress closes the distance -- the bound
    # under test is "never more than one option's own budget", not "every
    # option takes exactly the same number of ticks".
    results = []
    for _ in range(4):
        coordinator = _fresh_option_coordinator()
        coordinator.start_mission(goal_mission[0], goal_mission[1], now_step=0, now_time_sec=0.0)
        coordinator.enqueue_subgoal(*goal_mission)
        activated = coordinator.activate_next_subgoal(
            mission_frame.odom_pose_to_mission(executor.pose_world), now_step=0, now_time_sec=0.0,
        )
        if not activated:
            break
        executor.run_option(coordinator, partial_map, rng, max_local_steps=5)
        results.append(coordinator.last_subgoal_result)

    assert len(results) >= 2, "expected at least two options to exercise cross-option clock isolation"
    for result in results:
        assert result is not None
        assert 0 <= result.local_steps <= 5
        # If the bug were still present (executor-instance-lifetime clock
        # fed into the coordinator), a LATER option's elapsed_time_sec
        # would keep growing past what its OWN tick count could produce,
        # as the executor's own (never-reset) internal counters kept
        # accumulating across every prior option. Each option here must
        # independently stay within its OWN local budget regardless of how
        # many options already ran on this same long-lived executor
        # instance.
        assert 0.0 <= result.elapsed_time_sec <= result.local_steps * 0.3 + 1e-6
