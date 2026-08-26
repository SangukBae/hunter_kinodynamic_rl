# 2026-08-26 (round 2): residual-defect fix verification

Addendum to `2026-08-26_item1-4_fixes.md`. That report's item-1/item-2/item-3/item-4
fixes were real but each had a residual gap the report's own live/unit
evidence did not actually exercise. This pass closes those gaps, adds one
new concurrency defect the report never addressed (checkpoint-save-side
locking), reproduces each with a failing counterexample against the
pre-fix code, and re-verifies the full suite. **Do not trust the round-1
report's "Full-suite verification" section as proof these residual gaps
were closed -- it predates this pass.**

## Environment

Same as round 1: Docker container `7a2702b311a1`
(`drl_robot_path_planning:first`), ROS2 Humble, workspace
`/root/DRL_Robot_Path_Planning/ros2_ws`, Gazebo Sim (Ignition Fortress)
6.16.0. `docs/verification/live_physics_calibration_check.py` was edited
in place for this pass (case 2 now uses a *realistic* near-miss, 0.002s,
instead of round 1's 0.05s -- see item 1 below).

## item 1: physics-step tolerance/contract bug

**Root cause.** Round 1 added `verify_physics_step_calibration` (a real
`multi_step[1]` probe) but judged it against `runtime.physics_step_tolerance_sec`
-- a tolerance sized for a FULL control-period advance (~0.1s, default
`0.005`), reused unchanged for the single-step (`gazebo_max_step_size_sec`,
default `0.001s`) calibration probe. `abs(0.002 - 0.001) = 0.001 <= 0.005`
-- a connected world whose real physics step was exactly DOUBLE the
declared value passed calibration undetected. Round 1's own live A/B test
used a declared value of `0.05s` (25x off), which trivially failed and
never exercised this near-miss.

**Fix.**
- `config/schema.py`: new `RuntimeConfig.physics_step_calibration_tolerance_sec`
  (default `0.0002s`, 20% of the shipped step size), separate from
  `physics_step_tolerance_sec`. `validate()` now rejects any calibration
  tolerance `>= 0.5 * gazebo_max_step_size_sec` -- the strict bound needed
  to GUARANTEE
  any whole-step-scale mismatch (declared vs. real off by >=2x) is always
  caught, not just today's default numbers.
- `env/simulation/gazebo_runtime.py::verify_physics_step_calibration` now
  passes this new tolerance explicitly to `multi_step_advance` instead of
  falling back to `physics_step_tolerance_sec`.
- `env/simulation/environment_node.py::_apply_runtime_cfg`: caches the new
  field, AND now invalidates `_physics_step_calibrated`/
  `_physics_step_calibration_observed_dt_sec` (republishing
  `physics_step_calibration_verified=False`) whenever
  `gazebo_max_step_size_sec`/`physics_step_calibration_tolerance_sec`/
  `clock_confirm_timeout_sec` actually changes on a later call -- this
  fires naturally from `_resolve_evaluation_contract_override` (called
  from every `/reset`) whenever a genuine override change reaches
  `self.profile.runtime`. Never fires on the first (`__init__`) call
  (`_physics_step_calibrated` doesn't exist yet) or when unrelated fields
  change.

**Failing-repro -> fix (unit, `tests/test_config.py`/`tests/test_environment_node.py`):**
```
docker exec 7a2702b311a1 bash -lc "source /opt/ros/humble/setup.bash; \
  source /root/DRL_Robot_Path_Planning/ros2_ws/install/setup.bash; \
  cd /root/DRL_Robot_Path_Planning/ros2_ws/src/hunter_kinodynamic_rl && \
  python3 -m pytest -q tests/test_config.py tests/test_environment_node.py"
# 106 passed (was 96 before this pass's 10 new tests)
```
Re-verified against the pre-fix source (`git checkout --` the three files
back to the pre-round-2 staged index, ran the new tests, restored): the
original 8 counterexamples failed (`test_physics_step_calibration_tolerance_*`
x4 -> TypeError/DID NOT RAISE; `test_verify_physics_step_calibration_catches_a_declared_0_001_vs_real_0_002_world_at_shipped_defaults`
-> DID NOT RAISE; `test_apply_runtime_cfg_invalidates_a_stale_calibration_when_*`
x2 -> AttributeError/assert True is False; the end-to-end
`test_evaluation_contract_override_invalidates_a_stale_physics_step_calibration`
-> AttributeError).
The later reverse-direction boundary counterexample additionally proved
that accepting tolerance exactly equal to half the declared step let a
real 0.0005s step pass a declared 0.001s step; the strict `< 0.5 * step`
schema bound and its regression test now reject it.

**Live-Gazebo A/B (real Ignition, `drl_arena.world`, `<max_step_size>0.001</max_step_size>`):**
```
docker exec -d 7a2702b311a1 bash -lc "source /opt/ros/humble/setup.bash; \
  source /root/DRL_Robot_Path_Planning/ros2_ws/install/setup.bash; \
  export IGN_GAZEBO_SYSTEM_PLUGIN_PATH=/root/DRL_Robot_Path_Planning/third_party/rgl/RGLGazeboPlugin/install/RGLServerPlugin; \
  cd /root/DRL_Robot_Path_Planning; \
  ros2 launch hunter_se_gazebo simulate_hunter_se_ignition.launch.py rviz:=false headless:=true"

docker exec 7a2702b311a1 bash -lc "source /opt/ros/humble/setup.bash; \
  source /root/DRL_Robot_Path_Planning/ros2_ws/install/setup.bash; \
  export PYTHONPATH=/root/DRL_Robot_Path_Planning/ros2_ws/src/hunter_kinodynamic_rl:\$PYTHONPATH; \
  cd /root/DRL_Robot_Path_Planning/ros2_ws/src/hunter_kinodynamic_rl; \
  python3 docs/verification/live_physics_calibration_check.py"
```
Output (AFTER fix, realistic near-miss 0.002s declared vs. real 0.001s world):
```
[live] CASE 1 (correct 0.001s declared): calibration PASSED, observed_dt=0.001000s
[live] CASE 2 (realistic 0.002s declared, real world 0.001s): calibration correctly FAILED:
  multi_step[1]: observed sim-time advance 0.001000s does not match expected 0.002000s
  within tolerance 0.000200s (under-step).
[live] CASE 3 (real 100-step 0.1s advance): observed_dt=0.100000s, n_steps=100
[live] SUMMARY: case1_ok=True case2_ok=True case3_ok=True
```
A direct BEFORE-fix simulation (same live world, same node, manually set
`_physics_step_calibration_tolerance_sec=0.005` -- the value the pre-fix
code would have reused unconditionally):
```
[live][PRE-FIX SIMULATION] CASE 2 with old tolerance=0.005: calibration WRONGLY PASSED,
  observed_dt=0.001000s -- confirms the pre-fix mismatch was real
```
Gazebo cleanly torn down afterward (`pkill -f "ign gazebo"`); `pgrep`/
`ros2 node list` confirmed clean (one stale `/ros2_gz_bridge` daemon-cache
entry with no backing process -- cleared via `ros2 daemon stop`).

## item 2: checkpoint save/load/prune concurrency

**Root cause (two gaps, neither addressed by round 1's item-3).**
1. `save_generation` never acquired ANY lock on its own generation
   directory while writing it. A freshly-`os.makedirs`'d generation is, by
   construction, unreferenced by any tag until
   `_publish_generation_symlink` runs at the very end -- indistinguishable
   from a genuine orphan to `prune_orphan_generations`' scan. A concurrent
   prune (small `min_age_sec`) could mark it orphaned and, on a LATER call
   once that marker aged out, `rmtree` it out from under the still-running
   save, before the tag was ever published.
2. `load_generation` releases its shared lock the instant it returns --
   before a caller (`trainer_base.py::_resume_from`) goes on to call
   `ReplayBuffer.load(result["replay_path"])`. A concurrent prune racing
   exactly that window (the generation just became unreferenced and was
   already aged past `min_age_sec`) could delete `replay.npz` before the
   caller's own deserialization ever ran.

**Fix (`rl/checkpointing/manager.py`):**
- `save_generation` now acquires an EXCLUSIVE, blocking lock on its own
  `.lock` file immediately after `os.makedirs(gen_dir)`, held through
  every write/fsync AND the final `_publish_generation_symlink` call.
- A directory-level `.generation-lifecycle.lock` now closes the remaining
  mkdir-before-generation-lock window: saves hold it shared from before
  generation-directory creation through publication, while prune requires
  it exclusively for its complete scan/mark/sweep. A prune racing an
  in-flight save returns without touching that save's directory.
- Orphan-marker create/read/remove operations now also require the
  generation's exclusive lock. Consequently an unpublished save cannot
  receive a marker that survives publication and later causes the first
  prune after a new orphan transition to skip the intended grace period.
- New `load_generation_lease` (context manager): identical read/verify to
  `load_generation`, but keeps the SAME shared lock held for the caller's
  ENTIRE `with` block -- callers needing `result["replay_path"]` must
  deserialize it INSIDE that block. `load_generation` itself is unchanged
  (narrower, function-scoped lock; documented as such).
- `training/trainer_base.py::_resume_from` now uses
  `ckpt_manager.load_generation_lease(...)` and moved the
  `ReplayBuffer.load(...)` call inside its `with` block.

**Failing-repro -> fix:**
```
docker exec 7a2702b311a1 bash -lc "source /opt/ros/humble/setup.bash; \
  cd /root/DRL_Robot_Path_Planning/ros2_ws/src/hunter_kinodynamic_rl && \
  python3 -m pytest -q tests/test_checkpointing.py"
# 40 passed (was 37 before this pass's 3 new tests)
```
Re-verified against pre-fix `manager.py` (restored via `git checkout --`
from the staged index, then reapplied): `test_save_generation_lifecycle_excludes_prune_and_marker_until_publish`
failed with `AssertionError: assert ['<uuid>'] == []` -- the second prune
call genuinely deleted the in-flight, unpublished generation.
`test_load_generation_lease_blocks_a_concurrent_prune_during_replay_deserialization`
failed with `AttributeError: module ... has no attribute 'load_generation_lease'`.
Both restored and re-passed after reapplying the fix; full 40/40 re-run
clean.

New tests: `test_save_generation_lifecycle_excludes_prune_and_marker_until_publish`
(genuine threading test -- pauses inside `_publish_generation_symlink`,
drives two real `prune_orphan_generations` calls around it),
`test_load_generation_lease_blocks_a_concurrent_prune_during_replay_deserialization`
(genuine threading test -- pauses inside a real `ReplayBuffer.load`, races
a real prune), `test_load_generation_alone_does_not_protect_a_caller_reading_replay_path_after_return`
(documents `load_generation`'s intentionally narrower, still-vulnerable-if-misused
contract).

The save/prune race test additionally asserts that neither of its two
prune attempts creates `.orphaned_since` while publication is paused and
that no stale marker remains after publication. A direct post-fix
counterexample run reported
`marker_created_during_save=False`; before the lifecycle/marker-lock fix it
reported `True`, and that stale marker allowed the old generation to be
deleted on the first prune after a later tag replacement.
The two save/prune and load/prune race tests were also repeated together
five times after the fix: every run passed (`2 passed, 38 deselected`).

## item 3: system-ID error-origin ordering

**Root cause.** `analyze_stop_test` checked `if not samples: return
{"valid": False, "reason": "no_samples"}` BEFORE checking
`brake_onset_snapshot_failure_reason`. A live run that never received ANY
odometry/joint-state at all naturally has both an empty `samples` list AND
a failed live brake-onset snapshot (e.g. `"no_odometry_received_before_onset"`,
`"odometry_stale_at_brake_onset"`) -- the generic, far less actionable
`"no_samples"` silently won every time, discarding the real diagnostic.

**Fix (`dynamics/system_identification.py::analyze_stop_test`):** swapped
the order -- the `brake_onset_snapshot_failure_reason` check (and its
`allow_fallback_despite_live_snapshot_failure` escape hatch) now runs
FIRST; the generic `not samples` check runs after. `valid=False` either
way; only which `reason` string survives changes.

**Failing-repro -> fix:**
```
docker exec 7a2702b311a1 bash -lc "source /opt/ros/humble/setup.bash; \
  cd /root/DRL_Robot_Path_Planning/ros2_ws/src/hunter_kinodynamic_rl && \
  python3 -m pytest -q tests/test_system_identification.py tests/test_system_id_node.py"
# 44 passed (was 42 before this pass's 2 new tests)
```
Both new tests (`test_stop_test_live_snapshot_failure_reason_survives_when_no_samples_were_recorded_either`,
`test_final_json_preserves_the_real_failure_reason_when_no_samples_were_recorded_either`)
failed against the pre-fix ordering (`assert 'no_samples' ==
'odometry_stale_at_brake_onset'` / `'no_odometry_received_before_onset'`),
confirmed via `git checkout --` to the pre-fix file and back.

## item 4: evaluation-contract restore-status durability

**Root cause.** `_write_contract_restore_status` wrote directly to
`contract_restore_status.json` via a plain `open(path, "w")` (no temp
file, no flush/fsync, no parent-directory fsync -- not atomic, not
durable), and caught `OSError` by only PRINTING a warning and returning
normally. A write failure here was invisible to
`run_eval_body_with_contract_restore`'s own control flow: if the
evaluation body and the contract restore BOTH succeeded but persisting
that success failed, the process still exited 0.

**Fix (`nodes/evaluation_node.py`):**
- New `_atomic_write_json`: writes to a temp file in the SAME directory,
  flushes + `os.fsync`s it, `os.replace`s it into place, then fsyncs the
  parent directory too (mirrors `rl.checkpointing.manager`'s own
  write-then-publish convention). Raises `OSError` on any failure; cleans
  up the temp file on the way out.
- `_write_contract_restore_status` now delegates to it and no longer
  swallows the exception.
- `run_eval_body_with_contract_restore`: the write is now attempted inside
  the existing `finally` block but its `OSError` is caught LOCALLY (never
  re-raised from `finally`, which would silently replace an
  already-propagating exception from `body` -- Python's own finally/raise
  semantics). After the `try/finally` completes (i.e. only when `body`
  itself did NOT raise): a restore failure OR a status-write failure (or
  both) now raises `SystemExit`, with both reasons reported together when
  both occurred. A `body` exception always propagates completely
  unmodified; the write failure is merely printed in that case.

**Failing-repro -> fix:**
```
docker exec 7a2702b311a1 bash -lc "source /opt/ros/humble/setup.bash; \
  cd /root/DRL_Robot_Path_Planning/ros2_ws/src/hunter_kinodynamic_rl && \
  python3 -m pytest -q tests/test_evaluation_node.py"
# 47 passed (was 40 before this pass's 7 new tests)
```
Re-verified against pre-fix `evaluation_node.py`: import of
`_atomic_write_json`/`_write_contract_restore_status` failed outright
(`ImportError: cannot import name '_atomic_write_json'`) -- the API
(and, per the pre-fix source read in full, the non-atomic/swallowing
behavior it replaces) genuinely did not exist before this pass.

New tests: `test_atomic_write_json_leaves_only_the_final_file_no_tmp_droppings`,
`test_atomic_write_json_raises_and_leaves_no_tmp_file_when_the_write_itself_fails`,
`..._when_fsync_fails`, `test_atomic_write_json_raises_and_preserves_the_previous_file_when_replace_fails`,
`test_body_success_and_restore_success_but_status_write_fails_still_raises_systemexit`,
`test_body_success_and_restore_failure_and_status_write_also_fails_reports_both`,
`test_body_exception_propagates_unmasked_even_when_status_write_also_fails`.

## Full-suite verification (this pass)

```
cd /root/DRL_Robot_Path_Planning/ros2_ws/src/hunter_kinodynamic_rl && python3 -m pytest -q
# 765 passed (was 741; +24 new tests across the 4 items above)

cd /root/DRL_Robot_Path_Planning/ros2_ws/src/drl_agent && python3 -m pytest -q
# 1059 passed -- untouched by this pass, sanity check only

cd /root/DRL_Robot_Path_Planning/ros2_ws && colcon build \
  --packages-select drl_agent_interfaces hunter_se_gazebo drl_agent hunter_kinodynamic_rl
# Summary: 4 packages finished [4.50s]

colcon test --packages-select hunter_kinodynamic_rl
colcon test-result --test-result-base build/hunter_kinodynamic_rl --verbose
# Summary: 770 tests, 0 errors, 0 failures, 0 skipped
```

Process cleanliness (before/after the live Gazebo run):
```
docker exec 7a2702b311a1 pgrep -af 'ign gazebo|gz sim|ros2 launch|parameter_bridge|robot_state_publisher|hunter_se_cmd_prefilter|spawn_hunter_se|environment_node|system_id_node|train_node|evaluation_node'
# (no output, before AND after)
```
One stale `ros2 node list` entry (`/ros2_gz_bridge`) survived the
`pkill -f "ign gazebo"` teardown with no backing process in `ps aux` --
a `ros2-daemon` discovery-cache artifact, not a real leftover process;
cleared with `ros2 daemon stop`.

## Git scope

This pass touched exactly 15 files, all inside `hunter_kinodynamic_rl`:
14 show as `AM` (modified on top of the round-1 session's already-staged,
not-yet-committed new package), while this newly-created report itself is
still untracked (`??`):
`config/schema.py`, `dynamics/system_identification.py`,
`env/simulation/environment_node.py`, `env/simulation/gazebo_runtime.py`,
`nodes/evaluation_node.py`, `rl/checkpointing/manager.py`,
`training/trainer_base.py`, `tests/test_checkpointing.py`,
`tests/test_config.py`, `tests/test_environment_node.py`,
`tests/test_evaluation_node.py`, `tests/test_system_identification.py`,
`tests/test_system_id_node.py`, plus this doc and one edit to
`docs/verification/live_physics_calibration_check.py`.
`ros2_ws/src/drl_agent_interfaces/package.xml` (pre-existing, unrelated
unstaged edit) and `.gitignore` (pre-existing staged
`ros2_ws/runtime/`-related change) were NOT touched, staged, or included
in any diff produced by this pass. No commit, no push.

## Limitations (genuinely not done in this pass)

- **No second Gazebo world with a different real `<max_step_size>`** --
  same limitation as round 1; the realistic-near-miss A/B above still
  uses a deliberately-wrong DECLARED value against the one shipped
  0.001s world, not a literal second SDF.
- **No full SAC/TQC training+evaluation E2E rerun** with
  `deterministic_stepping: true` or against a live checkpoint
  save/load/prune race under real training load -- the concurrency fixes
  are verified via genuine (real-thread, real-lock) unit-level races, not
  a live multi-process training run.
- **Checkpoint durability is verified at the fsync-call/atomic-rename
  level**, not via an actual kill -9-mid-write + remount test (same
  disclosed limitation as round 1).
- **Real Hunter SE hardware** was never available in this environment
  (pre-existing, unrelated to this pass).
