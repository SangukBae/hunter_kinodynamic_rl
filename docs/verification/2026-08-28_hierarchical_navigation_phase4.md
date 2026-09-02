# 2026-08-28: Phase 4 -- Global RL MVP

> **Historical verification artifact.** 당시 source/명령/결과를 보존한다. 최신
> 상태는 [`../CURRENT_STATUS.md`](../CURRENT_STATUS.md)를 따른다.

Implements `docs/HIERARCHICAL_NAVIGATION_IMPLEMENTATION_PLAN.md` section 8
against `hunter_se_unknown_gps_denied_hierarchical_navigation_detailed_spec.txt`
sections 8-10/14/18-19, on top of the completed Phase 1 (mission frame,
localization, mapping), Phase 2 (local subgoal controller, hierarchy
coordinator), and Phase 3 (long-horizon procedural world) delivery.

> **Review round (same date).** A review pass found three real defects
> before "완료" sign-off: (1) `HierarchicalTrainingConfig.local_checkpoint_path`
> was documented/used as a `model.pt` FILE path in the training loop's own
> `checkpoint_meta()` but treated as a `ckpt_manager.load_generation()`
> DIRECTORY in the live node -- the live node could never actually load a
> real checkpoint. (2) `action_mask.py`'s rollout check sampled a straight
> line between the robot and a candidate's endpoint, not the plan's own
> "Ackermann short rollout" -- an off-axis candidate whose straight chord
> was clear but whose actual turning-radius-constrained arc clipped an
> obstacle passed as valid. (3) `GlobalReplayBuffer` saved `map_shape`/
> field COUNTS but never the channel/scalar/candidate-feature NAMES those
> counts stand for, so a shape-compatible file whose channel semantics had
> drifted would load and sample silently wrong. All three fixed below
> under "Review round fixes"; the rest of the MVP (masked Dueling DQN,
> `gamma^local_steps` SMDP target, train-mode replay guard, seed-scheduler
> pool isolation, ROS-free loop/adapter split) was confirmed already
> correct by the review's own targeted test run (46 passed, 2 skipped --
> the 2 skips being the torch-gated files run without torch on a bare
> host). Full regression re-run after the fixes: **1662 tests, 0 errors, 0
> failures, 0 skipped** (Docker, `colcon test`), 20/20 profiles still
> valid.
>
> **Review round 2 (2026-08-29).** Round 1's Ackermann-arc fix (#2 above)
> was itself still incomplete: it computed the curvature toward the
> candidate's endpoint correctly but only swept the arc out to
> `candidate.radius_m` (the STRAIGHT-line chord length), not the actual arc
> length needed to reach that endpoint along the curved path -- for a
> 90-degree/6m candidate at `kappa=1/3`, the real arc needs ~9.42m to reach
> `(0, 6)`, so the ~3.4m tail (`s` in `(6, 9.42)`) was never checked; a
> known obstacle placed there (reviewer's own repro, e.g. around `(1.37,
> 5.67)`) was reported as valid. Fixed below under "Review round 2 fix" by
> computing the arc length that actually reaches the declared endpoint
> (`s = r*phi/sin(phi)`, a standard circular-arc identity) and rejecting --
> never clamping -- any candidate whose required curvature exceeds the
> vehicle's limit. Full regression re-run: **1660 passed** (host+Docker
> `pytest -q`), **1665 tests, 0 errors, 0 failures, 0 skipped** (Docker
> `colcon test`), 20/20 profiles still valid.

## Review round 2 fix

**Ackermann rollout still stopped short of the endpoint (round 1's fix was
incomplete).** `action_mask._ackermann_feasible_arc_points_robot_frame`
(replacing round 1's `_ackermann_arc_points_robot_frame`) now derives, per
candidate at polar angle `phi` and distance `r`: `kappa = (2/r)*sin(phi)`
(equivalent to the earlier `2*dy/(dx^2+dy^2)` form) and the arc length that
ACTUALLY reaches the endpoint, `s = r*phi/sin(phi)` (from the circular-arc
identity that the chord subtends angle `2*phi` at the circle's center) --
never the candidate's straight-line `radius_m`. Two cases are now REJECTED
outright rather than clamped-and-truncated:

- `abs(kappa) > robot_max_curvature`: no single feasible arc reaches the
  endpoint at all -- clamping curvature would sweep a shorter, DIFFERENT
  arc that still doesn't reach the declared endpoint, reintroducing the
  exact same "silently incomplete rollout" bug in a different guise. This
  is the reviewer's own recommended fix (rejecting is safer than clamping
  for the plan's "endpoint까지의 Ackermann rollout" requirement).
- `sin(phi) ~ 0` with `cos(phi) < 0` (candidate directly, or nearly
  directly, behind the robot): the required arc length diverges to
  infinity in this limit (looping almost all the way around to reach a
  point behind the start) -- no FORWARD arc reaches it at all. This
  affects exactly the `-180` degree candidate in the default 8-direction
  grid; `tests/test_global_action_mask.py` documents it as an explicit,
  map-independent invalid candidate
  (`test_directly_behind_candidate_is_unreachable_via_any_forward_arc`)
  rather than a silent edge case.

Verified against the reviewer's exact numbers: for the 90-degree/6m
candidate, `kappa=1/3` and the fixed arc length is `3*pi ~= 9.4248` m
(reviewer: "약 9.425m"), and the arc's own last sampled point is
`(~0, 6.0)` -- the true endpoint, not `(2.73, 4.25)` (round 1's truncated
sample at `s=6`). New regression tests added to
`tests/test_global_action_mask.py`:
`test_full_rollout_reaches_the_actual_endpoint_not_truncated_at_the_straight_chord_length`
(places an obstacle at the reviewer's own repro coordinates, `(1.37,
5.67)`, and confirms it now invalidates the candidate),
`test_candidate_requiring_curvature_beyond_vehicle_limit_is_invalid`, and
`test_directly_behind_candidate_is_unreachable_via_any_forward_arc`.
`test_ackermann_rollout_differs_from_straight_line_for_a_lateral_candidate`
(round 1's original regression test) and
`test_all_reachable_candidates_are_valid_on_a_fresh_unknown_map` (renamed
from round 1's `test_all_valid_on_a_fresh_unknown_map`) were updated for
the reject-not-clamp behavior change.

## Review round fixes

**1 (checkpoint path/directory mismatch).** Split
`HierarchicalTrainingConfig.local_checkpoint_path: str` into
`local_checkpoint_dir: str` + `local_checkpoint_name: str = "final"` -- the
same `(directory, tag)` shape `rl.checkpointing.manager.load_generation`
actually needs (and the same shape `real_policy_node.py`'s own
`checkpoint_dir`/`checkpoint_name` ROS parameters already use elsewhere in
this package). Added `training/train_hierarchical_dqn.resolve_local_checkpoint_file(dir, name)`
(`<dir>/<name>/model.pt` -- the exact file a `tag` symlink resolves to) so
`checkpoint_meta()`'s sha256 hashing and `hierarchical_environment_node.py`'s
`ckpt_manager.load_generation(dir, name, ...)` call both resolve the SAME
file from the SAME two fields, instead of one side parsing a combined path
string. `hierarchical_phase4.yaml` updated to
`local_checkpoint_dir: ".../checkpoints"` / `local_checkpoint_name: "final"`.
Verified live in Docker: constructed `HierarchicalEnvironmentNode` against a
profile with these fields; ran `train_hierarchical_dqn(...)` against a real
`<dir>/final/model.pt` layout and confirmed the saved checkpoint manifest's
`local_checkpoint_sha256` matches `hashlib.sha256` of that exact file.
`tests/test_hierarchical_training_loop.py` updated
(`test_local_checkpoint_hash_is_recorded_in_checkpoint_meta` now builds a
real `<dir>/<tag>/model.pt` layout; added
`test_checkpoint_meta_has_no_hash_when_checkpoint_dir_set_but_file_absent`).

**2 (straight-line rollout, not Ackermann).** `action_mask.py` now checks
TWO separate conditions per the plan's own two bullets: the candidate's
DECLARED endpoint occupancy (unchanged, a direct cell lookup), and a
SEPARATE short rollout that follows the constant-curvature arc a real
Ackermann vehicle would actually sweep toward the candidate --
`_ackermann_arc_points_robot_frame` derives the arc's curvature from the
standard pure-pursuit formula (`kappa = 2*dy/(dx^2+dy^2)`, well-defined for
ANY candidate direction including behind the robot), clamps it to the
caller-supplied `robot_max_curvature` (`compute_action_mask` gained a new
REQUIRED parameter, never defaulted -- `RobotConfig.max_curvature` /
`1/min_turning_radius_m`, threaded through every call site:
`train_hierarchical_dqn.HierarchicalTrainingLoop` derives it once from its
own `min_turning_radius_m` constructor arg,
`hierarchical_environment_node.py` from `profile.robot.max_curvature`), and
sweeps the resulting arc via the standard constant-curvature kinematics
(`x(s)=sin(kappa*s)/kappa`, `y(s)=(1-cos(kappa*s))/kappa`). A straight-ahead
candidate (`dy=0`) always yields `kappa=0`, so this is BYTE-IDENTICAL to the
old straight-line check for the common forward case -- only off-axis/behind
candidates change behavior, and only when the required curvature exceeds
what the vehicle can do. New regression tests:
`test_ackermann_rollout_differs_from_straight_line_for_a_sharp_lateral_candidate`
(places an obstacle exactly on the clamped arc's own midpoint, off the
straight chord, and confirms it invalidates the candidate) and
`test_straight_ahead_candidate_is_unaffected_by_curvature_clamping`.

**3 (replay channel metadata not persisted).** `replay_schema.py` bumped to
`SCHEMA_VERSION = 2` (v1 never shipped against a real training run, only
this package's own tests -- rejected outright, same as an unknown future
version, never silently migrated). `GlobalReplayBuffer.save()` now writes
`map_channel_names`/`scalar_names`/`candidate_feature_names` (plain numpy
unicode arrays -- no `allow_pickle` needed) alongside the existing shape/
count fields; `load()` compares each against the CURRENT
`observation.MAP_CHANNEL_NAMES`/`SCALAR_NAMES`/`CANDIDATE_FEATURE_NAMES`
tuples and raises `ValueError` on any mismatch (a renamed/reordered/
inserted channel can no longer silently pass a shape-only check). New
tests: `test_save_records_current_channel_metadata`,
`test_load_rejects_a_renamed_or_reordered_map_channel`.

## Files added

```
navigation/global_rl/
├── __init__.py
├── subgoal_sampler.py       # robot-relative 8x2 candidate grid + backtrack/stop_recovery fallback
├── action_mask.py           # occupied/inflated + short-rollout + footprint + localization/map checks
├── observation.py           # map/scalar/candidate tensors, fixed shape contract
├── reward.py                # option-level Global reward (goal/progress/exploration/penalties)
├── replay_schema.py         # v1 field contract + failure-reason int codes
├── replay.py                # GlobalReplayBuffer (uint8 map, mode="train" guard, RNG-resumable)
├── networks.py               # masked Dueling Double DQN (torch)
└── agent.py                  # GlobalDQNAgent + smdp_target() (torch)

training/train_hierarchical_dqn.py   # ROS-free HierarchicalTrainingLoop + SimplifiedKinematicLocalExecutor
nodes/hierarchical_environment_node.py  # live-Gazebo inference adapter (structurally complete, unverified live)
nodes/hierarchical_train_node.py        # ros2 run wrapper around the ROS-free training loop
config/profiles/hierarchical_phase4.yaml

tests/test_subgoal_sampler.py
tests/test_global_action_mask.py
tests/test_global_observation.py
tests/test_global_reward.py
tests/test_global_replay.py
tests/test_global_smdp_target.py
tests/test_masked_dqn.py
tests/test_hierarchical_training_loop.py
```

`config/schema.py` gained `GlobalRLConfig` (action grid, network sizing,
replay/agent hyperparameters, and the Global reward weights -- bundled the
same way `HierarchyConfig` already bundles Phase 2's own thresholds) and
`HierarchicalTrainingConfig` (loop-level: frozen local-checkpoint path,
per-option/per-mission step budgets, `long_horizon_curriculum` level). Both
are opt-in (`enabled: false` default) and wired into `Profile`/`loader.py`
exactly like every prior phase's sections; `Profile.validate()` cross-checks
`hierarchical_training.enabled` against `global_rl.enabled` and
`long_horizon_world.enabled` the same way it already does for
`wall_segment_pool`/`long_horizon_world`.

## Key contracts

- **Candidate grid** (`subgoal_sampler.build_candidate_set`): fixed
  direction-major/distance-minor order, `len(direction_degrees) *
  len(distances_m) + 1` candidates (the `+1` fallback always last,
  `config.fallback_index`). Robot-relative; `candidate_endpoint_mission`
  converts to mission frame via `MissionFrame.robot_to_mission` (never a
  re-implementation). Fallback is `backtrack` (short step directly behind
  the robot) or `stop_recovery` (zero-distance), config-selected.
- **Action mask** (`action_mask.compute_action_mask`): invalidates a
  candidate only for endpoint-in-inflated-cell, a short Ackermann arc
  rollout crossing an inflated cell -- swept the FULL arc length actually
  needed to reach the candidate's endpoint (`s = r*phi/sin(phi)`, never
  truncated at the straight-line `radius_m`), with the candidate REJECTED
  outright (never clamped-and-truncated) when the required curvature
  exceeds `robot_max_curvature` or the candidate is effectively directly
  behind the robot (no forward arc reaches it) -- non-finite/non-positive
  geometry, localization/map invalid, or the robot's OWN footprint already
  being in collision. An UNKNOWN endpoint is never invalidated for being
  unknown. The fallback candidate is unconditionally `True` regardless of
  every other check.
- **Observation** (`observation.build_global_observation`): fixed
  `(5, H, W)` map tensor (occupied/free/unknown/visited/goal-direction --
  the last a smooth `cos`-alignment field, never a hard heatmap clamped to
  the crop edge), fixed `(5,)` scalar tensor (final-goal distance/bearing
  ALWAYS the true geometry, never approximated from the map crop alone --
  satisfies "goal 밖에 있을 때도 distance/bearing scalar 포함"), fixed
  `(n_candidates, 4)` candidate tensor (mask + known-free ratio + unknown
  gain + curvature-difficulty proxy). Never reads `LongHorizonWorld`
  ground truth -- only the online `PartialMap` + mission-frame pose/goal
  (`tests/test_global_observation.py::test_no_privileged_ground_truth_leaks_into_observation`
  pins `GlobalObservation`'s own field set).
- **Network/agent** (`networks.MaskedDuelingDQN`, `agent.GlobalDQNAgent`):
  CNN map encoder + MLP scalar encoder + shared per-candidate MLP -> fused
  dueling head; `Q(invalid candidate) = -inf` is applied INSIDE
  `forward()`, so both greedy (`argmax`) and epsilon-random
  (`rng.choice(valid_indices)`) selection structurally cannot pick an
  invalid candidate, and an all-but-fallback-invalid state still selects
  the fallback. `agent.smdp_target()` implements
  `option_reward + gamma^local_steps * (1 - mission_done) * next_q_target`
  exactly, gated on `mission_done` alone -- `subgoal_success`/`failure_reason`
  never affect bootstrapping (tested explicitly).
- **Replay** (`replay.GlobalReplayBuffer`): map channels stored `uint8`
  (quantized/dequantized losslessly enough, `atol=1/255`), schema-versioned
  `.npz` with map-shape/channel-count metadata, sampling RNG persisted
  across save/load (mirrors `rl.replay.buffer.ReplayBuffer`'s own resume
  test). `add()` raises on any `mode != "train"` -- the training loop only
  ever calls it from a `mode="train"` loop instance, a second, independent
  enforcement layer on top of the API guard.
- **Reward** (`reward.compute_global_reward`): `R_goal` only on
  `mission_reached`; `R_final_progress` signed by distance delta;
  `R_exploration` clipped to `exploration_reward_clip` (never allowed to
  dominate `goal_reward`); `R_repeated_revisit`/`R_repeated_deadend` driven
  by caller-supplied flags (this module never detects dead-ends itself --
  out of Phase 4's scope, a Phase 5 topological-memory concern); reason-
  specific `R_local_failure` (timeout/no_progress/blocked/high_risk/
  cancelled_by_replan, each independently configurable, `0` for
  `REACHED`/`None`); `R_risk`/`R_elapsed` scale with the option's own
  risk-integral/local-step count.
- **Training loop** (`train_hierarchical_dqn.HierarchicalTrainingLoop`):
  generates one `LongHorizonWorld` per mission via a mode-scoped
  `LongHorizonSeedScheduler` (train/validation/test pools never cross --
  regression-tested), then repeats Global-decision -> local-option ->
  reward cycles. A FRESH `HierarchyCoordinator` is created per Global
  option (see the method's own docstring for why: Phase 2's
  `FailureRecoveryPolicy` was designed around an externally pre-populated
  queue, where "no next candidate" genuinely means exhaustion -- Phase 4
  enqueues exactly one candidate per Global decision by design, so reusing
  one coordinator across options would abort the mission on the very first
  local failure). True mission termination is decided by the loop itself
  from `coordinator.mission_reached` (fires correctly on a fresh
  coordinator, since `GoalManager` is re-armed with the identical final
  goal every option) plus the option/step budgets.
- **Local execution stand-in** (`SimplifiedKinematicLocalExecutor`): a
  deliberately simplified point-robot steer-then-drive model -- NOT the
  real frozen kinodynamic TQC -- that drives real Phase 1/2 code
  (`PartialMap.integrate_scan` fed from a genuine simulated LiDAR ray-cast
  against the `LongHorizonWorld` occupancy, `HierarchyCoordinator.record_local_tick`
  for termination) so the Global RL <-> map <-> hierarchy plumbing is
  trainable/testable end to end without Gazebo. See the Limitations section.
- **Local-checkpoint hash in Global manifest**:
  `HierarchicalTrainingLoop.checkpoint_meta()` resolves
  `<hierarchical_training.local_checkpoint_dir>/<local_checkpoint_name>/model.pt`
  (`train_hierarchical_dqn.resolve_local_checkpoint_file` -- the exact file
  a `ckpt_manager.load_generation(dir, tag, ...)` call would read) and
  computes `rl.checkpointing.manager.sha256_of_file` over it when it
  exists; `nodes/hierarchical_train_node.py` embeds it in every saved
  Global checkpoint's manifest.

## Tests

69 new tests across 8 files (numpy-only where possible; `test_masked_dqn.py`/
`test_global_smdp_target.py` are torch-gated via `pytest.importorskip`).
Required checks per the task's own list, and where each lives:

- invalid action never selected (greedy/random) -- `test_masked_dqn.py`
- all-invalid selects fallback -- `test_masked_dqn.py::test_all_but_fallback_invalid_still_selects_a_valid_index`
- UNKNOWN endpoint stays valid -- `test_global_action_mask.py::test_all_valid_on_a_fresh_unknown_map`
- fixed map/scalar/candidate/mask shapes -- `test_global_observation.py`
- no GT/topology leak into observation -- `test_global_observation.py::test_no_privileged_ground_truth_leaks_into_observation`
- `gamma^local_steps` applied exactly -- `test_global_smdp_target.py`
- `mission_done` vs `subgoal_done` bootstrapping difference -- `test_global_smdp_target.py::test_subgoal_success_does_not_gate_bootstrapping_only_mission_done_does`
- first vs. repeated dead-end penalty -- `test_global_reward.py::test_repeated_deadend_penalty_only_applied_when_flagged`
- replay save/load preserves schema/metadata/sampling RNG -- `test_global_replay.py`
- local checkpoint hash in Global manifest -- `test_hierarchical_training_loop.py::test_local_checkpoint_hash_is_recorded_in_checkpoint_meta`
- Phase 3 seed-pool mode never missing/crossed -- `test_hierarchical_training_loop.py::test_{train,validation}_mode_seeds_never_cross_into_*`

## Docker verification

```
docker exec 7a2702b311a1 bash -c "source /opt/ros/humble/setup.bash && \
  cd /root/DRL_Robot_Path_Planning/ros2_ws && colcon build --packages-select hunter_kinodynamic_rl"
# Starting >>> hunter_kinodynamic_rl / Finished <<< hunter_kinodynamic_rl [0.6s]

docker exec 7a2702b311a1 bash -c "source /opt/ros/humble/setup.bash && \
  source /root/DRL_Robot_Path_Planning/ros2_ws/install/setup.bash && \
  cd .../hunter_kinodynamic_rl && python3 -m pytest -q"
# 1660 passed in 52.09s

docker exec 7a2702b311a1 bash -c "source /opt/ros/humble/setup.bash && \
  cd /root/DRL_Robot_Path_Planning/ros2_ws && colcon test --packages-select hunter_kinodynamic_rl && \
  colcon test-result --test-result-base build/hunter_kinodynamic_rl --verbose"
# Summary: 1665 tests, 0 errors, 0 failures, 0 skipped
```

All 20 `config/profiles/*.yaml` (19 pre-existing + `hierarchical_phase4`)
load and validate via `config.loader.load_profile`.

Live smoke checks (Docker, ROS sourced, no Gazebo needed):

- `HierarchicalEnvironmentNode(...)` constructs successfully via
  `rclpy.init()` (no spin -- confirms profile load, localization/mapping/
  coordinator/local-agent/safety-limits wiring, and subscription/publisher
  setup raise nothing).
- `nodes.hierarchical_train_node.train_hierarchical_dqn("hierarchical_phase4",
  run_root="/tmp/...", num_missions=2)` runs end to end (Global RL
  candidate selection -> `HierarchyCoordinator` subgoal activation ->
  `SimplifiedKinematicLocalExecutor` local driving/LiDAR/map integration ->
  `compute_global_reward` -> `GlobalReplayBuffer.add(mode="train")` ->
  `GlobalDQNAgent.train_step`) in ~2.4s/mission, writes a generation
  checkpoint (`rl.checkpointing.manager.save_generation`), and a SECOND
  invocation with `resume_run_dir` set correctly continues
  `global_step` (80 -> 160) rather than restarting it.

## Final review fix: adaptive Ackermann rollout sampling

After the endpoint-length fix, a final review found one remaining action-mask
edge case: long feasible arcs could still skip a known obstacle located between
the profile's sparse fixed `rollout_sample_count` points. The concrete repro was
a 135-degree/6m candidate whose arc is nearly 20m long while
`hierarchical_phase4.yaml` used `rollout_sample_count: 8`, leaving multi-meter
gaps between adjacent samples.

Fix applied:

- `_ackermann_feasible_arc_points_robot_frame()` now treats
  `rollout_sample_count` as a minimum only.
- Long arcs are adaptively subdivided so adjacent samples are no farther apart
  than `partial_map.resolution_m * 0.5`.
- `_rollout_hits_inflated()` grid-traces between adjacent arc samples with
  `trace_clipped()`, so inflated cells between samples are checked too.
- Added
  `test_long_arc_rollout_does_not_skip_obstacles_between_sparse_profile_samples`
  covering the 135-degree/6m sparse-sample repro.

Local focused verification after this final fix:

```
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  tests/test_global_action_mask.py \
  tests/test_global_observation.py \
  tests/test_global_replay.py \
  tests/test_global_reward.py \
  tests/test_global_smdp_target.py \
  tests/test_hierarchical_training_loop.py \
  tests/test_masked_dqn.py \
  tests/test_subgoal_sampler.py
# 55 passed, 2 skipped in 1.27s

PYTHONPYCACHEPREFIX=/tmp/hunter_kinodynamic_rl_pycache python3 -m compileall -q \
  hunter_kinodynamic_rl/navigation/global_rl/action_mask.py \
  tests/test_global_action_mask.py
# OK
```

## Limitations (explicit, MVP scope boundary)

- `SimplifiedKinematicLocalExecutor` is a lightweight point-robot stand-in
  for the real frozen kinodynamic TQC -- it validates the Global RL/map/
  hierarchy integration, not whether the ACTUAL Hunter SE kinodynamics can
  safely track a Global-selected subgoal. Swapping in a live-Gazebo-backed
  `LocalOptionExecutor` (reusing `hierarchical_environment_node.py`'s own
  sensor/coordinator wiring) is the next integration step.
- `nodes/hierarchical_environment_node.py`/`hierarchical_train_node.py` are
  structurally complete ROS adapters (construction verified in Docker) but
  were NOT exercised against a live Gazebo instance this session -- same
  documented limitation Phase 3 already carries.
- No actual frozen local kinodynamic TQC checkpoint was trained/used this
  session; `hierarchical_phase4.yaml`'s `local_checkpoint_dir` is a
  placeholder path (schema validation only requires it non-empty, never
  that `<local_checkpoint_dir>/<local_checkpoint_name>/model.pt` actually
  exists, so the profile still validates cleanly). The review round DID
  verify, with a real `<dir>/<tag>/model.pt` layout, that
  `hierarchical_environment_node.py`'s `ckpt_manager.load_generation(dir,
  tag, ...)` call and `checkpoint_meta()`'s sha256 hashing both resolve to
  the SAME file -- what remains untested is loading an ACTUAL trained TQC
  checkpoint through that path.
- Phase 5 scope (topological memory, learned feasibility predictor, global
  risk map) was NOT implemented -- `reward.compute_global_reward`'s
  `repeated_deadend`/`revisit_amount` are caller-supplied flags with no
  detector behind them yet, exactly as the task instructions require.
