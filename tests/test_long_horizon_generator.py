"""Coverage for the Phase 3 long-horizon procedural world generator
(``env/scenarios/long_horizon_generator.py`` + ``config/schema.py``'s
``LongHorizonWorldConfig``). Pure Python/numpy -- no ROS import anywhere in
the generator, so this runs on a bare host checkout exactly like
``test_ackermann_feasibility.py``/``test_start_pose_sampling.py``.
"""

import numpy as np
import pytest

from hunter_kinodynamic_rl.config.schema import ConfigError, LongHorizonWorldConfig
from hunter_kinodynamic_rl.env.scenarios import long_horizon_generator as G
from hunter_kinodynamic_rl.env.scenarios import long_horizon_solvability as S

ROBOT_RADIUS_M = 0.3
MIN_TURNING_RADIUS_M = 0.8
WHEELBASE_M = 0.65


def _fast_cfg(**overrides) -> LongHorizonWorldConfig:
    """Small, fast-generating config for ordinary unit tests -- verified
    (see verification doc) to reliably succeed within a small fraction of
    generation_attempt_limit."""
    base = dict(
        enabled=True, size_m=16.0, resolution_m=0.25,
        corridor_width_min_m=1.5, corridor_width_max_m=2.5, wall_thickness_m=0.15,
        room_count_range=[1, 3], dead_end_count_range=[1, 3], loop_count_range=[1, 2],
        alternative_route_min_count=1, start_goal_geodesic_min_m=6.0,
        require_ackermann_feasibility=True, generation_attempt_limit=100,
        train_seed_range=[0, 999], validation_seed_range=[1000, 1099], test_seed_range=[1100, 1199],
    )
    base.update(overrides)
    return LongHorizonWorldConfig(**base)


def _generate(cfg: LongHorizonWorldConfig, seed: int):
    return G.generate_long_horizon_world(
        seed=seed, cfg=cfg, robot_radius_m=ROBOT_RADIUS_M,
        min_turning_radius_m=MIN_TURNING_RADIUS_M, wheelbase_m=WHEELBASE_M,
    )


# --------------------------------------------------------------- config
def test_long_horizon_world_config_defaults_are_disabled_and_validate():
    cfg = LongHorizonWorldConfig()
    cfg.validate()
    assert cfg.enabled is False


@pytest.mark.parametrize("kwargs", [
    {"size_m": 0.0},
    {"resolution_m": 0.0},
    {"wall_thickness_m": 0.0},
    {"corridor_width_min_m": 3.0, "corridor_width_max_m": 2.0},
    {"room_count_range": [5, 2]},
    {"dead_end_count_range": [-1, 2]},
    {"loop_count_range": [0, 0], "alternative_route_min_count": 1},
    {"generation_attempt_limit": 0},
    {"goal_radius_m": 0.0},
    {"train_seed_range": [0, 100], "validation_seed_range": [50, 150]},
    {"size_m": 5.0, "corridor_width_max_m": 4.0, "wall_thickness_m": 0.2},
])
def test_long_horizon_world_config_rejects_invalid_values(kwargs):
    with pytest.raises(ConfigError):
        LongHorizonWorldConfig(**kwargs).validate()


def test_seed_split_matches_configured_ranges_and_rejects_out_of_range():
    cfg = _fast_cfg()
    assert G.long_horizon_seed_split(0, cfg) == "train"
    assert G.long_horizon_seed_split(999, cfg) == "train"
    assert G.long_horizon_seed_split(1000, cfg) == "validation"
    assert G.long_horizon_seed_split(1199, cfg) == "test"
    with pytest.raises(G.LongHorizonSeedSplitError):
        G.long_horizon_seed_split(100000, cfg)


# --------------------------------------------------------------- seed-pool enforcement (code review)
# Previously long_horizon_seed_split()/LongHorizonWorldConfig.validate()
# established train/validation/test pool separation, but nothing on the
# actual generation PATH ever enforced it -- generate_long_horizon_world(
# seed=<out-of-pool>, cfg=...) happily generated a world regardless. These
# tests reproduce the exact failure mode found in review (a tight
# train=[0,9]/validation=[10,19]/test=[20,29] config still generating a
# world for seed=999) and confirm the fix.
def _tight_pool_cfg(**overrides) -> LongHorizonWorldConfig:
    return _fast_cfg(
        train_seed_range=[0, 9], validation_seed_range=[10, 19], test_seed_range=[20, 29], **overrides,
    )


def test_generate_long_horizon_world_mode_none_does_not_enforce_pool_membership():
    """Backward-compatible default: no mode given, no check -- an ad hoc/
    manual verification script that doesn't care about pool separation
    keeps working exactly as before."""
    cfg = _tight_pool_cfg()
    w = _generate(cfg, seed=999)  # 999 is outside every configured range
    assert w.seed == 999


def test_generate_long_horizon_world_mode_rejects_out_of_pool_seed():
    cfg = _tight_pool_cfg()
    with pytest.raises(G.LongHorizonSeedSplitError, match="outside all configured"):
        G.generate_long_horizon_world(
            seed=999, cfg=cfg, robot_radius_m=ROBOT_RADIUS_M,
            min_turning_radius_m=MIN_TURNING_RADIUS_M, wheelbase_m=WHEELBASE_M, mode="train",
        )


def test_generate_long_horizon_world_mode_rejects_seed_from_the_wrong_pool():
    """A seed that IS in some pool, just not the one the caller asked for."""
    cfg = _tight_pool_cfg()
    with pytest.raises(G.LongHorizonSeedSplitError, match="belongs to the 'validation' pool"):
        G.generate_long_horizon_world(
            seed=15, cfg=cfg, robot_radius_m=ROBOT_RADIUS_M,
            min_turning_radius_m=MIN_TURNING_RADIUS_M, wheelbase_m=WHEELBASE_M, mode="train",
        )


def test_generate_long_horizon_world_mode_accepts_seed_from_the_matching_pool():
    cfg = _tight_pool_cfg()
    w = G.generate_long_horizon_world(
        seed=3, cfg=cfg, robot_radius_m=ROBOT_RADIUS_M,
        min_turning_radius_m=MIN_TURNING_RADIUS_M, wheelbase_m=WHEELBASE_M, mode="train",
    )
    assert w.seed == 3


# --------------------------------------------------------------- LongHorizonSeedScheduler
def test_seed_scheduler_rejects_unknown_mode():
    with pytest.raises(ValueError):
        G.LongHorizonSeedScheduler(run_seed=0, cfg=_fast_cfg(), mode="bogus")


def test_seed_scheduler_draws_only_from_its_own_mode_pool():
    cfg = _tight_pool_cfg()
    sched = G.LongHorizonSeedScheduler(run_seed=42, cfg=cfg, mode="validation")
    for _ in range(20):
        seed = sched.next_seed()
        assert 10 <= seed <= 19
        sched.validate_explicit_seed(seed)  # must never raise for its own draw


def test_seed_scheduler_is_deterministic_given_run_seed_and_resumable_via_state_dict():
    cfg = _tight_pool_cfg()
    a = G.LongHorizonSeedScheduler(run_seed=7, cfg=cfg, mode="train")
    seq_a = [a.next_seed() for _ in range(10)]

    b = G.LongHorizonSeedScheduler(run_seed=7, cfg=cfg, mode="train")
    seq_b = [b.next_seed() for _ in range(10)]
    assert seq_a == seq_b  # same run_seed -> identical sequence

    c = G.LongHorizonSeedScheduler(run_seed=8, cfg=cfg, mode="train")
    seq_c = [c.next_seed() for _ in range(10)]
    assert seq_c != seq_a  # different run_seed -> different sequence

    # Resume from episode_index=5 continues the SAME sequence as `a`.
    resumed = G.LongHorizonSeedScheduler.from_state_dict(
        {"run_seed": 7, "mode": "train", "episode_index": 5}, cfg,
    )
    assert [resumed.next_seed() for _ in range(5)] == seq_a[5:10]


def test_seed_scheduler_validate_explicit_seed_rejects_wrong_pool():
    cfg = _tight_pool_cfg()
    sched = G.LongHorizonSeedScheduler(run_seed=0, cfg=cfg, mode="train")
    with pytest.raises(G.LongHorizonSeedSplitError):
        sched.validate_explicit_seed(15)  # validation-pool seed


def test_seed_scheduler_seeds_generate_successfully_with_mode_enforced():
    """End-to-end: every seed a scheduler produces for its own mode must be
    directly usable with generate_long_horizon_world(..., mode=<same mode>)
    without ever tripping the seed-pool-isolation check."""
    cfg = _tight_pool_cfg()
    sched = G.LongHorizonSeedScheduler(run_seed=1, cfg=cfg, mode="train")
    for _ in range(3):
        seed = sched.next_seed()
        w = G.generate_long_horizon_world(
            seed=seed, cfg=cfg, robot_radius_m=ROBOT_RADIUS_M,
            min_turning_radius_m=MIN_TURNING_RADIUS_M, wheelbase_m=WHEELBASE_M, mode="train",
        )
        assert w.seed == seed


# --------------------------------------------------------------- determinism / diversity
def test_same_seed_produces_byte_identical_world():
    cfg = _fast_cfg()
    w1 = _generate(cfg, seed=7)
    w2 = _generate(cfg, seed=7)
    assert np.array_equal(w1.occupancy, w2.occupancy)
    assert w1.start_pose == w2.start_pose
    assert w1.goal_pose == w2.goal_pose
    assert w1.shortest_path_length_m == w2.shortest_path_length_m
    assert w1.topology_metadata == w2.topology_metadata
    assert [(s.center_x_m, s.center_y_m, s.length_m, s.yaw_rad) for s in w1.wall_segments] == \
        [(s.center_x_m, s.center_y_m, s.length_m, s.yaw_rad) for s in w2.wall_segments]


def test_different_seeds_produce_structurally_different_worlds():
    cfg = _fast_cfg()
    worlds = [_generate(cfg, seed=s) for s in range(5)]
    occupancies = [tuple(w.occupancy.flatten().tolist()) for w in worlds]
    # Not every seed pair need differ, but they must not ALL collapse to the
    # same layout.
    assert len(set(occupancies)) > 1
    starts = {w.start_pose for w in worlds}
    assert len(starts) > 1


# --------------------------------------------------------------- start/goal safety
@pytest.mark.parametrize("seed", range(10))
def test_start_and_goal_never_land_on_occupied_or_inflated_cell(seed):
    cfg = _fast_cfg()
    w = _generate(cfg, seed=seed)
    h, wd = w.occupancy.shape
    inflated = S.inflate_occupancy(w.occupancy, w.resolution_m, ROBOT_RADIUS_M)
    start_cell = S.world_to_cell(w.start_pose[0], w.start_pose[1], w.resolution_m, *w.origin_xy, h, wd)
    goal_cell = S.world_to_cell(w.goal_pose[0], w.goal_pose[1], w.resolution_m, *w.origin_xy, h, wd)
    assert start_cell is not None and goal_cell is not None
    assert not inflated[start_cell]
    assert not inflated[goal_cell]


def test_start_goal_geodesic_distance_meets_configured_minimum():
    cfg = _fast_cfg()
    for seed in range(10):
        w = _generate(cfg, seed=seed)
        assert w.shortest_path_length_m >= cfg.start_goal_geodesic_min_m - 1e-6


# --------------------------------------------------------------- topology contract
def test_dead_end_loop_and_alternative_route_conditions_are_met():
    cfg = _fast_cfg()
    for seed in range(10):
        w = _generate(cfg, seed=seed)
        meta = w.topology_metadata
        assert cfg.dead_end_count_range[0] <= meta["dead_end_count"] <= cfg.dead_end_count_range[1]
        assert cfg.loop_count_range[0] <= meta["loop_count"] <= cfg.loop_count_range[1]
        assert meta["alternative_route_count"] >= cfg.alternative_route_min_count


def test_topology_metadata_is_a_mapping_not_leaked_into_geometry():
    import collections.abc

    w = _generate(_fast_cfg(), seed=0)
    # A read-only Mapping (types.MappingProxyType), never a plain dict --
    # see test_world_information_boundary.py's mutation-attempt tests for
    # the immutability contract itself; this test only confirms every
    # expected key is still readable through the proxy.
    assert isinstance(w.topology_metadata, collections.abc.Mapping)
    for key in ("room_count", "dead_end_count", "loop_count", "alternative_route_count", "grid_n", "pitch_m"):
        assert key in w.topology_metadata


# --------------------------------------------------------------- shortest path recomputation
def test_shortest_path_length_matches_independent_recomputation():
    cfg = _fast_cfg()
    for seed in range(8):
        w = _generate(cfg, seed=seed)
        recomputed = S.shortest_path_length_m(
            w.occupancy, w.resolution_m, w.origin_xy, w.start_pose[:2], w.goal_pose,
            inflation_radius_m=ROBOT_RADIUS_M,
        )
        assert recomputed is not None
        assert abs(recomputed - w.shortest_path_length_m) < 1e-6


# --------------------------------------------------------------- wall segment / occupancy consistency
def test_occupancy_is_exactly_the_rasterization_of_wall_segments():
    cfg = _fast_cfg()
    for seed in range(8):
        w = _generate(cfg, seed=seed)
        n_cells = w.occupancy.shape[0]
        rebuilt = G.build_occupancy_from_segments(w.wall_segments, w.resolution_m, w.origin_xy, n_cells)
        assert np.array_equal(rebuilt, w.occupancy)


def test_every_wall_segment_has_a_unique_entity_id_and_positive_length():
    w = _generate(_fast_cfg(), seed=0)
    ids = [seg.entity_id for seg in w.wall_segments]
    assert len(ids) == len(set(ids))
    assert all(seg.length_m > 0.0 for seg in w.wall_segments)
    assert all(seg.semantic in ("corridor", "room_boundary", "boundary") for seg in w.wall_segments)


# --------------------------------------------------------------- per-class wall-pool capacity (code review, round 2)
def test_max_possible_wall_segment_counts_by_class_bound_holds_across_many_seeds():
    """Empirical check that the analytical per-class worst-case bound
    (config/schema.py's validate_wall_pool_capacity now depends on this)
    never underestimates what a real generated world can need -- the exact
    defect code review found: a validated profile still failing inside
    wall_segment_spawner.activate_walls because one specific length class
    needed more segments than its guaranteed slot share, even though the
    single TOTAL capacity was never exceeded."""
    cfg = LongHorizonWorldConfig(
        enabled=True, size_m=40.0, resolution_m=0.25,
        corridor_width_min_m=2.0, corridor_width_max_m=4.0,
        room_count_range=[4, 10], dead_end_count_range=[1, 5], loop_count_range=[1, 4],
        alternative_route_min_count=1, start_goal_geodesic_min_m=20.0,
        require_ackermann_feasibility=True, generation_attempt_limit=100,
    )
    classes = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    bound = G.max_possible_wall_segment_counts_by_class(cfg, classes)

    def snap_up(length, classes):
        for c in sorted(classes):
            if c >= length - 1e-9:
                return c
        raise RuntimeError("exceeds classes")

    for seed in range(20):
        w = G.generate_long_horizon_world(
            seed=seed, cfg=cfg, robot_radius_m=0.45, min_turning_radius_m=1.2, wheelbase_m=0.65,
        )
        actual = {c: 0 for c in classes}
        for seg in w.wall_segments:
            actual[snap_up(seg.length_m, classes)] += 1
        for c in classes:
            assert actual[c] <= bound[c], (seed, c, actual[c], bound[c])


def test_max_possible_wall_segment_counts_by_class_matches_code_review_repro_shape():
    """The exact scenario from code review: max_segments=100 split evenly
    across 10 classes (10 slots/class) is nowhere near enough -- this
    function must report a per-class worst case far above 10 for at least
    one class, which is what makes config/schema.py's
    validate_wall_pool_capacity correctly reject the profile."""
    cfg = LongHorizonWorldConfig(
        enabled=True, size_m=40.0, room_count_range=[4, 10],
        dead_end_count_range=[1, 5], loop_count_range=[1, 4],
    )
    classes = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    bound = G.max_possible_wall_segment_counts_by_class(cfg, classes)
    assert max(bound.values()) > 10  # far exceeds the naive 100/10=10-per-class allocation


# --------------------------------------------------------------- bounded attempt semantics
def test_generation_attempt_limit_exhaustion_raises_clear_runtimeerror():
    # An impossible combination (loop_count fixed at 0 while
    # alternative_route_min_count demands >= 1 is rejected by validate()
    # itself; instead force impossibility via a tiny attempt budget against
    # an otherwise-normal config, which is what actually exhausts here).
    cfg = _fast_cfg(generation_attempt_limit=1, start_goal_geodesic_min_m=15.9)
    # 15.9 m minimum geodesic in a 16 m world is very unlikely to be found
    # within a single attempt.
    with pytest.raises(RuntimeError, match="no feasible long-horizon world found"):
        _generate(cfg, seed=999999)


def test_ackermann_feasibility_required_without_robot_params_raises():
    cfg = _fast_cfg(require_ackermann_feasibility=True)
    with pytest.raises(RuntimeError, match="require_ackermann_feasibility"):
        G.generate_long_horizon_world(seed=0, cfg=cfg, robot_radius_m=ROBOT_RADIUS_M)


def test_world_too_small_for_minimal_lattice_raises_immediately():
    cfg = _fast_cfg(size_m=5.0, corridor_width_max_m=4.0, wall_thickness_m=0.2, require_ackermann_feasibility=False)
    with pytest.raises(RuntimeError):
        G.generate_long_horizon_world(seed=0, cfg=cfg, robot_radius_m=ROBOT_RADIUS_M)


def test_successful_generation_records_which_attempt_succeeded():
    w = _generate(_fast_cfg(), seed=0)
    assert w.topology_metadata["attempt"] >= 0


# --------------------------------------------------------------- completion criterion
def test_at_least_one_hundred_seeds_generate_successfully_within_bounded_attempts():
    """Plan section 7.8's own completion criterion: >= 100 seeds generated
    consecutively, each within generation_attempt_limit."""
    cfg = LongHorizonWorldConfig(
        enabled=True, size_m=40.0, resolution_m=0.25,
        corridor_width_min_m=2.0, corridor_width_max_m=4.0,
        room_count_range=[4, 10], dead_end_count_range=[1, 5], loop_count_range=[1, 4],
        alternative_route_min_count=1, start_goal_geodesic_min_m=20.0,
        require_ackermann_feasibility=True, generation_attempt_limit=100,
    )
    cfg.validate()
    for seed in range(100):
        w = G.generate_long_horizon_world(
            seed=seed, cfg=cfg, robot_radius_m=0.45, min_turning_radius_m=1.2, wheelbase_m=0.65,
        )
        assert w.topology_metadata["attempt"] < cfg.generation_attempt_limit
