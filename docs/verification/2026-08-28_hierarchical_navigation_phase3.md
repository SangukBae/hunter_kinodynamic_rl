# 2026-08-28: Phase 3 -- Long-horizon procedural world

Implements `docs/HIERARCHICAL_NAVIGATION_IMPLEMENTATION_PLAN.md` section 7
against `hunter_se_unknown_gps_denied_hierarchical_navigation_detailed_spec.txt`
section 16, on top of the completed Phase 1 (mission frame, localization,
mapping) and Phase 2 (local subgoal controller, hierarchy coordinator)
delivery. No Global RL was added -- out of scope for this phase, per the
plan and the task instructions for this session.

> **Round 2 (code review, same date).** A review pass on the initial Phase
> 3 delivery (code-reading + reproduction scripts, no colcon re-run) found
> five real defects, all fixed below under "Round 2 fixes". Full colcon
> regression re-run after the fixes: **1590 tests, 0 errors, 0 failures, 0
> skipped** (19/19 profiles still OK).
>
> **Round 3 (code review, same date).** A follow-up review on Round 2's own
> fixes found the wall-pool capacity check (Round 2 fix 4) was still
> insufficient -- it checked the TOTAL segment count but not per-CLASS
> demand, which `wall_segment_spawner.activate_walls` actually needs (no
> escalation to a larger class on exhaustion) -- and confirmed Round 2 fix
> 2's `LongHorizonWorld.__post_init__` nested-metadata freeze claim didn't
> match the actual code (only the generator's own tuple conversion
> provided it, not `__post_init__` itself). Both fixed below under "Round 3
> fixes". Full colcon regression re-run after the fixes: **1597 tests, 0
> errors, 0 failures, 0 skipped** (19/19 profiles still OK).

## Round 3 fixes

**1 (P1, wall-pool capacity check missed per-class slot exhaustion) --
Round 2's `validate_wall_pool_capacity` only checked the TOTAL segment
count against `max_segments`, never each length CLASS's own share.**
`wall_segment_spawner.activate_walls` requires a free slot in a segment's
OWN snapped length class and never escalates to a larger one when that
exact class is exhausted -- reproduced live: `max_segments=100` split
evenly across 10 classes (10 slots/class, `WallSegmentPool`'s round-robin
allocation) validated cleanly (the grand total, 100, was never exceeded by
any generated world's ~88-100 segments), but `activate_walls` failed at
runtime with `RuntimeError: no free slot in the length class 1.0` because
that ONE class alone needed far more than its 10-slot share (up to 80 in
the worst case, confirmed both analytically and by generating 60 worlds
and observing real per-class counts up to 42). Fixed: new
`long_horizon_generator.max_possible_wall_segment_counts_by_class(cfg,
length_classes_m)` computes, per class, the worst-case count of segments
that could specifically need it -- derived from `_build_walls`'s own
emission rule (boundary+closed-edge segments all need the "pitch class";
open-edge remainder segments range over an analytically bounded length
window, so every class within/just above that window is credited the
worst-case remainder count) -- verified empirically against 60+ generated
worlds across the default profile and all 6 curriculum levels, always
an upper bound, never exceeded. `config/schema.py`'s
`validate_wall_pool_capacity` now requires `max_segments //
len(length_classes_m)` (the guaranteed per-class MINIMUM the round-robin
allocation gives every class -- the same worst-populated-class reasoning
`validate_procedural_pool_capacity` already uses for `ObstaclePoolConfig`)
to cover the worst NEEDED class. `config/profiles/hierarchical_phase3.yaml`'s
`wall_segment_pool` was re-sized against this (`length_classes_m: [3.0,
4.0, 13.4]`, `max_segments: 240` -- each class needs <= 80 in the worst
case, `240 // 3 = 80`). New regression tests: the exact zero-margin
repro shape now rejected
(`test_profile_rejects_the_exact_per_class_exhaustion_repro_from_code_review`),
an end-to-end confirmation that a validated profile's pool never fails
`activate_walls` across 20 generated worlds
(`test_a_validated_wall_pool_profile_never_fails_activate_walls_across_many_seeds`),
plus the per-class bound function's own empirical-holds-across-seeds and
repro-shape unit tests in `test_long_horizon_generator.py`.

**2 (P2, documentation/code mismatch) -- nested `topology_metadata` values
were only immutable because the GENERATOR happened to pre-convert them,
not because `LongHorizonWorld.__post_init__` itself enforced it.** Round
2's `__post_init__` wrapped only the TOP-LEVEL `topology_metadata` dict in
`MappingProxyType` -- an external caller passing
`topology_metadata={"room_cells": [1, 2]}` (a plain list, unlike the
generator's own pre-converted tuple) could still mutate it in place
(`world.topology_metadata["room_cells"].append(3)`), even though the
Round 2 summary described this as handled by `__post_init__`. Fixed: a new
`_deep_freeze()` helper in `long_horizon_world.py` recursively converts
every `Mapping` (-> `MappingProxyType` over a freshly built dict),
`list`/`tuple` (-> `tuple`), and `numpy.ndarray` (-> a read-only copy)
found ANYWHERE inside `topology_metadata`, regardless of what the caller
supplied; `LongHorizonWorld.__post_init__` now calls this instead of the
shallow `MappingProxyType(dict(...))` wrap. `long_horizon_generator.py`'s
own `room_cells` construction was simplified back to a plain
`sorted(...)` list, since the class itself now guarantees the freeze --
one fewer place this invariant has to be independently maintained. New
regression tests in `test_world_information_boundary.py`: constructing a
`LongHorizonWorld` directly with a deliberately mutable nested
list/dict/list-inside-a-dict and confirming each raises on mutation, plus
a copy-not-alias test for the nested list specifically.

## Round 2 fixes

**1 (P0, ROS import boundary) -- `wall_segment_spawner.py` failed to
import at all on a bare, ROS-free host.** It imported
`obstacle_pool._ensure_slot_spawned`/`snap_up_to_class` at MODULE SCOPE,
which transitively pulls in `obstacle_pool -> obstacle_spawner ->
ros_gz_interfaces.srv` -- `import
hunter_kinodynamic_rl.env.spawning.wall_segment_spawner` raised
`ModuleNotFoundError: ros_gz_interfaces` on a bare host, contradicting this
module's own "pure-Python/no ROS import at module scope" documentation
(the pre-existing `test_wall_segment_spawner.py`'s module-level
`pytest.importorskip("ros_gz_interfaces")` masked this by skipping the
whole file on such a host instead of catching it). Fixed: `snap_up_to_class`
is now a small, local, genuinely pure reimplementation
(`_snap_length_up_to_class`) instead of an imported one;
`obstacle_pool._ensure_slot_spawned` is imported LAZILY, inside
`ensure_spawned()` itself (the one function that actually needs it) --
confirmed live on this session's own bare host (no `ros_gz_interfaces`
installed) that `wall_segment_spawner` now imports and every non-Gazebo
function in it (`build_wall_pool`, `_snap_length_up_to_class`, the SDF
builder) is directly callable. New regression tests:
`tests/test_wall_segment_spawner_import_boundary.py` (a dedicated file
with NO `importorskip` guard, since testing "importable without ROS" would
otherwise be skipped on exactly the host it needs to run on -- simulates
"not installed" via `sys.modules[name] = None`, which holds regardless of
whether `ros_gz_interfaces` happens to be installed on the machine running
the test).

**2 (P1, seed-pool isolation not enforced) -- `generate_long_horizon_world`
never checked its own `seed` against `cfg`'s train/validation/test ranges.**
`long_horizon_seed_split`/`LongHorizonWorldConfig.validate()` established
non-overlapping pools, but nothing on the actual generation PATH enforced
membership -- `generate_long_horizon_world(seed=999, cfg=<train=[0,9],
validation=[10,19], test=[20,29]>)` generated a world anyway (reproduced
live). Fixed: `generate_long_horizon_world` gained an optional `mode:
Optional[str] = None` parameter -- when given (`"train"`/`"validation"`/
`"test"`), it validates `seed` against that pool via `long_horizon_seed_split`
and raises `LongHorizonSeedSplitError` immediately, before any generation
work, on a mismatch; `None` (default) keeps the prior no-check behaviour
for callers that don't need it (e.g. an ad hoc verification script). Also
added `LongHorizonSeedScheduler`, mirroring
`env.scenarios.seed_scheduler.SeedScheduler` exactly (deterministic
`(run_seed, mode, episode_index)`-seeded draw, `state_dict`/
`from_state_dict` resumability, `validate_explicit_seed`) so a future
trainer/evaluator has a ready-made seed source that always satisfies its
own pool by construction, rather than needing to re-derive this pattern.
New regression tests in `tests/test_long_horizon_generator.py`: seed-pool
enforcement (accept/reject-own-pool/reject-wrong-pool/`mode=None`
backward-compat), and scheduler determinism/resumability/pool-membership/
end-to-end-with-`generate_long_horizon_world` coverage.

**3 (P1, `frozen=True` doesn't stop in-place mutation) --
`LongHorizonWorld`'s "immutable" fields were all still mutable in place.**
`frozen=True` only blocks REASSIGNING a field; it does nothing to a
mutable field's own contents. `world.occupancy[:] = False`,
`world.wall_segments.append(...)`, and `world.topology_metadata["x"] = 1`
were all previously possible (reproduced live) despite `frozen=True` --
`test_world_information_boundary.py`'s prior check (dataclass `frozen`
flag only) could not catch this class of bug at all. Fixed:
`LongHorizonWorld.__post_init__` (using `object.__setattr__`, the standard
pattern for a frozen dataclass needing post-init normalization) now
converts every field into a genuinely immutable value regardless of what
the caller passed in: `occupancy` -> a COPY (never aliases the caller's
own array) with `.setflags(write=False)` (an in-place write now raises
`ValueError`); `wall_segments` -> `tuple` (its elements were already
frozen `WallSegment`s); `topology_metadata` -> `types.MappingProxyType`
wrapping a COPIED `dict` (item assignment now raises `TypeError`), with
its own nested `"room_cells"` value changed from `list` to `tuple` at the
generator (the one nested mutable value that would otherwise have slipped
through the top-level-only `MappingProxyType` wrap). New regression tests
in `tests/test_world_information_boundary.py`: in-place-mutation-raises
for all three fields, plus copy-not-alias tests confirming a caller
mutating the object it originally passed in after construction cannot
affect the already-returned `LongHorizonWorld`.

**4 (P1, wall-pool capacity validation too weak) --
`wall_segment_pool.enabled=True, max_segments=0` (and, separately,
`max_segments` too small for the worst-case wall SEGMENT COUNT even when
`length_classes_m` covered the worst-case pitch) both validated cleanly**
and would only have failed loudly much later, at runtime, inside
`activate_walls` (reproduced live: exactly this profile shape passed
`Profile.validate()`). Fixed two ways: `WallSegmentPoolConfig.validate()`
now requires `max_segments > 0` when `enabled`; and a new
`long_horizon_generator.max_possible_wall_segment_count(cfg)` (worst-case
total segment count for `cfg`'s lattice size -- `4 * grid_n(cfg)^2`,
derived from `_build_walls`'s own emission rule: `4n` boundary segments
always, up to `2` per interior edge across `2n(n-1)` interior edges;
verified live never exceeded across 30 generated worlds) is checked by
`config/schema.py`'s `validate_wall_pool_capacity` alongside the existing
pitch-coverage check, sharing the same "computed from the generator's own
formula, can never silently drift out of sync" pattern
`max_possible_pitch_m` already used. New regression tests in
`tests/test_wall_segment_spawner.py`: the exact zero-`max_segments`
profile shape from review now rejected, a too-small-for-segment-count (but
pitch-sufficient) profile now rejected, plus a
`WallSegmentPoolConfig`-level `max_segments=0` case.

**5 (documentation accuracy) -- "independently verified" overclaimed what
the shortest-path tests actually check.** `long_horizon_generator` calls
`long_horizon_solvability.shortest_path_length_m` internally for its own
start/goal search, and the Round 1 test
(`test_shortest_path_length_matches_independent_recomputation`) recomputes
via the SAME function -- real regression protection (the two can never
silently diverge, since they're the same code), but not independent
verification of the Dijkstra algorithm itself, as the doc previously
implied. Fixed: `long_horizon_generator`'s own module docstring (step 8)
now states this distinction explicitly, and a new test,
`test_shortest_path_length_matches_a_hand_computed_single_gap_detour` (in
`tests/test_long_horizon_solvability.py`, which never calls the generator
at all), provides GENUINE independent verification: a single-cell gap in
an otherwise solid wall forces the shortest path through one known point
(computed via plain coordinate arithmetic, `cell_to_world`, not any
path-planning call); start/goal are chosen so both straight-line legs are
pure 45-degree diagonals -- the one case where an 8-connected grid's
distance metric reproduces true Euclidean distance to float precision (no
octile-vs-Euclidean approximation error at all) -- so the by-hand
`hypot(gap - start) + hypot(goal - gap)` sum is asserted against the
algorithm's output to `1e-9`, not just "within some tolerance" (verified:
the two values differed by ~2e-16, i.e. pure floating-point noise).

## Scope delivered

New pure-Python modules (no ROS import at module scope; all independently
unit-testable on a bare host checkout, exactly like the Phase 1/2 modules):

- `env/scenarios/long_horizon_world.py` -- `WallSegment` + `LongHorizonWorld`
  frozen dataclasses, matching the plan's data contract exactly
  (`seed`, `occupancy`, `resolution_m`, `origin_xy`, `wall_segments`,
  `start_pose`, `goal_pose`, `shortest_path_length_m`, `topology_metadata`).
  `topology_metadata` is documented as PRIVILEGED (generator/evaluation/
  teacher only).
- `env/scenarios/long_horizon_generator.py` -- `generate_long_horizon_world`:
  a seed-deterministic room/corridor/junction/loop/dead-end maze built on a
  random spanning tree over a coarse lattice graph (randomized Kruskal +
  greedy dead-end reduction + loop-edge augmentation), rasterized into an
  occupancy grid that is ALWAYS the exact rasterization of the returned
  `wall_segments` list (`build_occupancy_from_segments`, also independently
  callable). Start/goal are sampled via Dijkstra shortest-path distance
  over the robot-footprint-inflated free grid
  (`long_horizon_solvability.dijkstra_distances`), with a geodesic (not
  straight-line) minimum-distance gate, a safe-heading sampler, and an
  optional grid-based Ackermann feasibility gate. One shared, seeded
  `numpy.random.RandomState` per seed drives every draw across a bounded
  `generation_attempt_limit` retry loop, mirroring
  `env/scenarios/procedural_generator.generate_scenario`'s own "redraw the
  whole attempt on any infeasibility" structure.
- `env/scenarios/long_horizon_solvability.py` -- pure grid functions:
  `inflate_occupancy` (reuses `navigation.mapping.visited_map.rasterize_circle_cells`,
  the same circle rasterization `PartialMap._inflate` uses),
  `dijkstra_distances`/`shortest_path_length_m`/`is_connected` (8-connected
  weighted Dijkstra), and `is_ackermann_feasible_grid` (a bounded forward
  Hybrid-A*-style search over `dynamics.bicycle_model.step`, in the same
  algorithm family as `env.scenarios.ackermann_feasibility.is_ackermann_feasible`
  but checked against a fine occupancy grid instead of a circular-obstacle
  list -- a deliberately separate implementation, see that module's
  docstring for why).
- `env/scenarios/long_horizon_curriculum.py` -- `apply_level`/
  `level_overrides`: pure config overlays for the plan's Level 1-6
  difficulty progression (section 7.5). Level 6's "dynamic obstacle 추가"
  is explicitly out of this module's scope (documented in its own
  docstring) -- Phase 3 delivers only the static long-horizon geometry.
- `env/spawning/wall_segment_spawner.py` -- `WallSegmentPool` +
  `build_wall_pool`/`ensure_spawned`/`activate_walls`: a fixed-size,
  reusable Gazebo wall-segment entity pool mirroring
  `env/spawning/obstacle_pool.py`'s "pre-spawn once, teleport per episode,
  park the unused rest" design (directly reuses `obstacle_pool`'s own
  `_ensure_slot_spawned`/`snap_up_to_class` helpers rather than
  duplicating them). Wall length is quantized (rounded up to the nearest
  configured class) at ACTIVATION time, not generation time -- see that
  module's own docstring for the exact, documented approximation this
  implies for the live Gazebo geometry (never for the pure-Python
  `LongHorizonWorld.occupancy` contract, which is unaffected).
- `config/schema.py` -- new opt-in `LongHorizonWorldConfig` and
  `WallSegmentPoolConfig` sections (mirror the `ObstaclePoolConfig`
  pattern: `enabled=False` default, byte-identical to pre-existing
  behaviour for every profile that doesn't set them), wired into
  `Profile`/`config/loader.py`. Cross-section checks:
  `wall_segment_pool.enabled` requires `long_horizon_world.enabled`, and
  `validate_wall_pool_capacity` requires `length_classes_m`'s largest
  class to cover the worst-case maze cell pitch
  (`long_horizon_generator.max_possible_pitch_m`, shared verbatim between
  the two modules so the bound can never silently drift out of sync).
- `config/profiles/hierarchical_phase3.yaml` -- a verification profile
  with `long_horizon_world`/`wall_segment_pool` both enabled, matching the
  task's own example config numbers.

New tests (all ROS-free except `test_wall_segment_spawner.py`, which
self-skips on a bare host and runs for real once `ros_gz_interfaces` is
resolvable post-`colcon build`, mirroring `test_obstacle_pool.py`):

- `tests/test_long_horizon_generator.py` (36 cases) -- config validation,
  seed-pool membership/rejection, determinism, seed diversity, start/goal
  clearance (occupied + inflated), geodesic-distance minimum, dead-end/
  loop/alternative-route range compliance, shortest-path independent-
  recomputation match, occupancy/wall-segment rasterization consistency,
  bounded-attempt-limit exhaustion raising a clear error, and the plan's
  own >= 100-consecutive-seed completion criterion (section 7.8).
- `tests/test_long_horizon_solvability.py` (21 cases) -- cell/world
  round-trip, inflation, connectivity/shortest-path on hand-built synthetic
  grids (open field, fully-blocked wall, gapped wall), Dijkstra edge cases,
  and grid-Ackermann-feasibility open-field/blocked/gap/occupied-endpoint/
  invalid-parameter cases.
- `tests/test_wall_segment_spawner.py` (22 cases) -- pool config
  validation, `Profile`-level cross-checks (missing `long_horizon_world`,
  insufficient `length_classes_m`), pool construction/parking-distance,
  idempotent `ensure_spawned`, spawn-failure propagation,
  `activate_walls`' teleport/park split, length-class snapping, yaw-to-
  quaternion orientation, over-capacity/class-exhaustion/oversized-segment
  rejection, and cross-episode slot reuse (no new `SpawnEntity` calls).
- `tests/test_world_information_boundary.py` (6 cases) -- Phase 3 slice of
  the plan's information-boundary requirement: `topology_metadata` is a
  plain dict structurally separate from `occupancy`/`wall_segments`,
  neither dataclass exposes any observation-shaped method, and the only
  two Phase 1/2 policy-facing surfaces (`PartialMap.channels()`,
  `LocalPolicyController.build_observation`) have no parameter through
  which `LongHorizonWorld`/its privileged fields could be threaded in.

No new ROS `.srv`/`.msg`/nodes were added; `drl_agent_interfaces`'
`Reset.srv`/`Step.srv`/`GetDimensions.srv` are untouched. Phase 3
deliberately stops short of wiring `long_horizon_world`/`wall_segment_pool`
into `environment_node.py`'s live `/reset` flow -- that live-simulation
integration (spawning the pool, driving episodes against a generated
maze, mission-map partial-map accumulation over it) is naturally Phase 4's
territory once a Global RL policy actually needs a live long-horizon
episode to train against; this phase's own completion criteria (section
7.8) are about the generator/solvability/spawner-spec contracts, which are
what is delivered and tested here.

## A real bug found and fixed during this pass

**Wall rasterization silently produced zero occupied cells for the
default config.** `wall_thickness_m=0.2` is THINNER than
`resolution_m=0.25` -- the original `rasterize_wall_segment` tested
whether a grid cell's CENTER point fell inside the wall's exact geometric
box, but grid-cell centers are spaced a full `resolution_m` apart, so a
sub-resolution-thickness wall can fall entirely BETWEEN two consecutive
cell centers with neither one inside it (confirmed live: a straightforward
4 m x 0.2 m wall segment rasterized to 0 occupied cells with the original
point-test). Fixed by testing cell-FOOTPRINT overlap instead (inflating
both local half-extents by `resolution_m / 2` before the box test) --
conservative (a wall can only grow slightly, up to one cell in each
direction, never disappear or open an unintended gap), documented in the
function's own docstring.

## A design defect found and fixed: dead-end/loop-count coupling

The first working version of `_build_spanning_topology` sized its lattice
purely from `size_m` and `corridor_width_max_m` (independent of the
requested `dead_end_count_range`/`loop_count_range`), which for the task's
own example profile (40 m world, `corridor_width_max_m=4.0`) produced an
81-node lattice. A random spanning tree over a grid graph that size has
~25-31 leaves on average (empirically measured) -- reducing that down to
the configured `dead_end_count_range` maximum of 5 requires ~14-20 extra
edges, and EVERY edge added past a spanning tree also increases the
cyclomatic number (this module's own `loop_count`) by one, making the
configured `loop_count_range` maximum of 4 essentially unreachable. Every
seed failed after exhausting `generation_attempt_limit=100`.

Fixed two ways:
1. `_grid_n` now sizes the lattice primarily from the TARGET topology
   counts (`room_count_range[1] + dead_end_count_range[1] +
   loop_count_range[1] + 4`), never larger than what `size_m` can
   physically support at `corridor_width_max_m` -- see that function's own
   docstring for the exact reasoning.
2. The dead-end-reduction phase now prefers an edge connecting TWO current
   leaves to each other (removes 2 leaves per added edge) before falling
   back to a leaf-to-any-neighbor edge (removes 1) -- roughly halving how
   many extra (cyclomatic-increasing) edges reduction needs.

Verified live: the task's own example config (40 m, `dead_end_count_range:
[1,5]`, `loop_count_range: [1,4]`) now generates 100 consecutive seeds in
~12 s (down from failing every single seed before the fix), and all six
`long_horizon_curriculum` levels generate successfully for 10 seeds each.

## A performance defect found and fixed: Ackermann search budget

`is_ackermann_feasible_grid`'s default `max_expansions=6000` (inherited
from `ackermann_feasibility.is_ackermann_feasible`'s own default, sized for
a <15 m arena) was observed live to reject an otherwise-feasible
long-horizon route (a real, findable path through an ordinary 2-2.5 m wide
corridor maze) purely from search-budget exhaustion on a ~33 m geodesic
path -- confirmed by re-running the identical case with a larger budget
(20000: found; 60000: found; a finer `xy_resolution_m`: found). Raised the
default to `max_expansions=20000` -- the smallest tested value that
reliably found the same route; this ALSO made the overall generator
faster in practice (fewer wasted retries), not slower: the full six-level
x ten-seed sweep dropped from ~40 s to ~7 s once feasible worlds stopped
being spuriously rejected.

## Key contracts verified by tests

- Same seed -> byte-identical `occupancy`, `wall_segments` geometry,
  `start_pose`/`goal_pose`, `shortest_path_length_m`, and
  `topology_metadata`; different seeds -> structurally different worlds
  (not every pair need differ, but the whole set never collapses to one
  layout).
- `long_horizon_generator.long_horizon_seed_split` matches
  `LongHorizonWorldConfig`'s train/validation/test ranges exactly and
  raises for any seed outside all three; `LongHorizonWorldConfig.validate()`
  independently rejects overlapping ranges (mirrors
  `env.scenarios.procedural_generator.seed_split`/`ScenarioConfig`'s own
  pattern, but is a fully separate seed pool).
- Start and goal never land on an occupied OR robot-footprint-inflated
  cell, for every generated world.
- `shortest_path_length_m` on the returned world always matches
  `long_horizon_solvability.shortest_path_length_m` recomputed
  independently from the SAME world's own `occupancy` -- there is only one
  shortest-path implementation in the package (the generator calls the
  exact same function internally), so this can never silently drift.
- `occupancy` is always EXACTLY `build_occupancy_from_segments(world.wall_segments,
  ...)` -- never merely similar.
- `dead_end_count`/`loop_count`/`alternative_route_count` (the last always
  equal to the cyclomatic number, by construction -- see
  `long_horizon_generator`'s module docstring step 4 for why this is a
  valid proxy) land inside their configured ranges for every generated
  world across 10 seeds per curriculum level (60 world/level combinations)
  plus the 100-seed completion-criterion sweep.
- `generation_attempt_limit` exhaustion raises a clear `RuntimeError`
  naming the likely cause, never a silent partial/invalid world.
- `require_ackermann_feasibility=true` without a valid
  `min_turning_radius_m`/`wheelbase_m` raises immediately (mirrors
  `procedural_generator.generate_scenario`'s own `feasibility_check ==
  "ackermann"` contract) -- never silently falls back to a weaker check.
- `WallSegmentPool.ensure_spawned` is idempotent (no new `SpawnEntity`
  calls on a second call, or across two different episodes'
  `activate_walls` calls) and propagates a genuine spawn failure via
  `GazeboServiceError` rather than silently marking a slot spawned.
- `activate_walls` never escalates a wall segment to a larger size class
  than its own snapped-up one when that exact class is exhausted (raises
  instead), and never spawns more segments than `max_segments` allows.
- Neither `PartialMap.channels()` nor `LocalPolicyController.build_observation`
  (the only two Phase 1/2 policy-facing surfaces) has any parameter a
  `LongHorizonWorld`/its privileged `topology_metadata` could be threaded
  through, and `LongHorizonWorld`/`WallSegment` themselves expose no
  observation-shaped method.

## Test results (Docker, container `DRL_Robot_Path_Planning`, workspace
`/root/DRL_Robot_Path_Planning/ros2_ws`)

```
source /opt/ros/humble/setup.bash
colcon build --packages-select hunter_kinodynamic_rl        # 1 package finished, no errors
python3 -m hunter_kinodynamic_rl.config.loader <every config/profiles/*.yaml>.validate()   # 19/19 OK
python3 -m pytest -q tests/test_long_horizon_generator.py tests/test_long_horizon_solvability.py \
  tests/test_wall_segment_spawner.py tests/test_wall_segment_spawner_import_boundary.py \
  tests/test_world_information_boundary.py   # 112 passed
colcon test --packages-select hunter_kinodynamic_rl
colcon test-result --test-result-base build/hunter_kinodynamic_rl --verbose
```

Result (Round 3, final): **1597 tests, 0 errors, 0 failures, 0 skipped**
(up from Phase 2's 1475 -- net new: the 112 cases in the five Phase 3 test
files listed above (the five-file split, not four, is itself a Round 2
fix -- `test_wall_segment_spawner_import_boundary.py` is new), plus
`test_config.py`/`test_navigation_config.py`-style config-schema coverage
folded into `test_long_horizon_generator.py`/`test_wall_segment_spawner.py`
themselves, plus lint checks over the five implementation modules and one
new profile -- zero regressions in any pre-existing Phase 1/2 or
local-only suite). Profile count is now 19 (18 from Phase 2 + this phase's
own `hierarchical_phase3.yaml`).

Additional standalone verification (Docker, same environment):

```
100 consecutive seeds (task's own example config, 40 m world) generated successfully
  -- 12.6 s total (Round 2, after the __post_init__ immutability copy), max attempt used = 6 of 100
6 curriculum levels x 10 seeds each generated successfully -- 7.3 s total
Determinism, seed-diversity, start/goal-clearance, occupancy/wall-segment-
  consistency, and shortest-path-recomputation-match spot-checked directly
  (not just via pytest) before the test files were written
```

## Remaining limitations

- **No live Gazebo verification was performed for Phase 3.** No Gazebo/
  Ignition instance was running in the container at the start of this
  session (`ps aux` confirmed), and standing one up, spawning a generated
  maze's wall-segment pool into it, and confirming `/scan` actually
  observes the walls (plus a `mission_map_node` UNKNOWN -> FREE/OCCUPIED
  check over it) would be a substantial separate live-verification pass in
  its own right -- exactly the kind of multi-round live session Phase 1's
  own verification history shows is needed to do properly (QoS mismatches,
  sim/wall clock mixing, and similar live-only failure modes were all
  found that way, not via unit tests). Per this task's own explicit
  allowance ("실제 Gazebo spawn 연동이 현재 테스트 환경에서 어렵다면
  ROS-free spawn request/spec 생성까지 구현하고, live Gazebo 미검증은
  verification doc에 명확히 남겨라"), this phase delivers the complete
  ROS-free generation + spawn-spec/pool-activation contract (tested against
  a real `ros_gz_interfaces`-resolvable Docker environment, using the exact
  same fake-node-duck-typing test harness `obstacle_pool.py`'s own tests
  use) but does NOT confirm the physical Gazebo spawn, `/scan` visibility,
  or `mission_map_node` partial-map expansion live. This is the single
  largest open item before Phase 3 can be considered fully closed out.
- **`wall_segment_spawner`'s length-class quantization happens at
  activation time, not generation time** (unlike `obstacle_pool`'s own
  static-obstacle radius, which is pre-quantized during generation so the
  declared and spawned geometry always match exactly) -- documented in
  detail in that module's own docstring. The physically-spawned Gazebo
  wall can be up to one length-class step longer than declared, which for
  one of a passage-flanking remainder-segment pair can narrow the PHYSICAL
  passage by up to half that overflow. Never affects the pure-Python
  `LongHorizonWorld.occupancy`/connectivity/shortest-path contract (every
  test here reasons about that, unaffected); only matters for a live
  Gazebo run, where `wall_segment_pool.length_classes_m`'s step size
  should be kept small relative to `corridor_width_min_m` to keep this
  negligible.
- **Seed-pool enforcement (`generate_long_horizon_world(..., mode=...)` /
  `LongHorizonSeedScheduler`, Round 2) is opt-in, not mandatory.** A caller
  that omits `mode` (the default) still gets the pre-Round-2 no-check
  behaviour -- by design, so an ad hoc script can generate any seed it
  wants, but it also means nothing FORCES a future trainer to actually use
  `mode`/the scheduler; that trainer's own code review should confirm it
  does.
- **`environment_node.py` is not wired to `long_horizon_world`/
  `wall_segment_pool` at all** -- no profile flag makes a live training
  episode actually run against a generated long-horizon maze yet. This
  phase's own completion criteria (section 7.8) are about the generator/
  solvability/spawner contracts, not a live-episode integration; wiring
  that in is naturally scoped to whichever of Phase 4/5 first needs a live
  long-horizon episode to train Global RL against.
- **`long_horizon_curriculum`'s Level 1 cannot literally reach zero dead
  ends** -- a grid-lattice spanning tree always has >= 2 leaves, so
  Level 1's `dead_end_count_range` is `[0, 2]` (documented in that level's
  own dict comment as the closest this algorithm can reliably produce to
  the plan's "no dead end" intent) rather than a literal `[0, 0]`.
- **`alternative_route_count` is defined as the graph's cyclomatic number**
  (every edge added past a spanning tree creates at least one alternative
  route) rather than an explicit enumerated-route count -- a documented,
  intentional simplification (see `long_horizon_generator`'s module
  docstring step 4), not a limitation discovered after the fact, but
  worth restating here: a caller wanting an exact alternative-route COUNT
  (rather than a lower bound proxy) would need a separate route-enumeration
  pass this phase does not implement.
- **`generate_long_horizon_world`'s corridor/room representation is
  lattice-cell-based, not a general polygon/graph-embedding one** -- rooms
  are lattice cells whose open edges use the maximum passage width rather
  than a separately-carved larger footprint; this keeps geometry simple
  and exactly rasterization-consistent (see the design rationale in
  `long_horizon_generator`'s module docstring step 6) at the cost of
  visual/geometric room diversity a dedicated room-shape generator would
  have.
