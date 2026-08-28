# 2026-08-28: Phase 2 -- Local goal / hierarchy separation

Implements `docs/HIERARCHICAL_NAVIGATION_IMPLEMENTATION_PLAN.md` section 6
against `hunter_se_unknown_gps_denied_hierarchical_navigation_detailed_spec.txt`,
on top of the completed Phase 1 delivery (mission frame, localization,
mapping -- see `2026-08-28_hierarchical_navigation_phase1_review_fixes.md`).

> **Round 2 (code review, same date).** A review pass on the initial Phase 2
> delivery found two state-safety defects and one documentation-contract
> gap, fixed below under "Round 2 fixes"; the doc's own profile count (17
> vs. the actual 18) is also corrected throughout. Full regression re-run
> after the fixes: **1472 tests, 0 errors, 0 failures, 0 skipped** (18/18
> profiles still OK).
>
> **Round 3 (code review, same date).** A follow-up review on Round 2's
> localization-confidence fix found it closed "never confirm success from a
> bad pose" but not the stronger "no physical motion while degraded"
> guarantee -- fixed below under "Round 3 fixes". Full regression re-run
> after the fix: **1475 tests, 0 errors, 0 failures, 0 skipped** (18/18
> profiles still OK).
>
> **Round 4 (code review follow-up).** A review of Round 3 found two remaining
> confidence-gating holes: non-finite/malformed confidence values still
> bypassed ordinary Python `<` comparisons, and the first
> `activate_next_subgoal()` call after mission start ignored a provided
> unhealthy confidence reading. Fixed below under "Round 4 fixes". Targeted
> local verification for the touched hierarchy/replanning tests:
> **59 passed** with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`; profile validation:
> **18/18 OK**. The previous full Docker/colcon result remains the Round 3
> **1475 tests, 0 errors, 0 failures, 0 skipped** run; a full colcon rerun was
> not performed in this follow-up session.

## Round 4 fixes

**1 (state safety) -- non-finite confidence bypassed degraded-localization
checks.** Round 3 used direct comparisons like
`localization_confidence < hierarchy.localization_min_confidence`. That is
not sufficient in Python: `float("nan") < threshold` is `False`, so a NaN
confidence could still confirm `mission_reached` if the pose happened to be
inside tolerance, and could also clear a degraded pause inside
`activate_next_subgoal()`. Fixed: `HierarchyCoordinator` now routes every
confidence decision through finite-aware helpers. A confidence is healthy
only when it is provided, can be converted to `float`, is finite, and is at
or above the configured minimum. A provided NaN/Inf/malformed confidence is
treated as degraded. The standalone `evaluate_replanning()` helper now uses
the same finite-aware degraded check.

**2 (state safety) -- first subgoal activation ignored a provided unhealthy
confidence reading.** Round 3 gated `activate_next_subgoal()` only when the
coordinator was already in `localization_degraded`. On the first activation
after `start_mission()`, a caller could pass `localization_confidence=0.1`
and still receive an active subgoal with `stop_required=False`. Fixed:
whenever `activate_next_subgoal()` is called with a provided unhealthy
confidence reading, it refuses activation, leaves the queue untouched, sets
`localization_degraded=True`, and keeps `stop_required=True`. A later call
with finite confidence at/above the configured minimum resumes normal
activation.

New regression tests:
`test_nan_confidence_pose_inside_final_goal_tolerance_never_confirms_mission_success`,
`test_localization_degraded_pause_does_not_clear_on_nan_confidence`,
`test_initial_activation_with_low_confidence_is_refused_and_queue_preserved`,
`test_initial_activation_with_nan_confidence_is_refused_and_queue_preserved`,
`test_non_finite_or_malformed_localization_confidence_cancels_by_replan`,
and `test_degraded_localization_wins_over_reached`.

## Round 3 fixes

**1 (state safety) -- a degraded-confidence cancel could still hand out a
fresh active subgoal in the SAME tick.** Round 2 fix 2 made
`record_local_tick` cancel the current subgoal
(`CANCELLED_BY_REPLAN`/`"localization_confidence_degraded"`) before any
reached/success check, but that cancel still went through the SAME
`_apply_recovery` path every other failure/cancel trigger uses -- if the
coordinator's subgoal queue already had a next candidate,
`RecoveryAction.ADVANCE_NEXT` activated it immediately, in the same tick,
handing the local policy a fresh active subgoal (`stop_required` back to
`False`) while localization was STILL reporting degraded confidence.
"Never confirm success from a bad pose" (Round 2) is necessary but not
sufficient for GPS-denied/drift safety -- "no physical motion while
localization is degraded" is the stronger guarantee actually required (plan
section 6 safety conditions: "stale/invalid localization이면 physical
motion 금지").

Fixed: the localization-degraded branch no longer calls `_apply_recovery`
at all. It sets a new `HierarchyCoordinator._localization_degraded` flag
(exposed read-only as `localization_degraded`) instead, leaving the
coordinator paused -- `stop_required` stays `True`, and
`activate_next_subgoal()` now refuses to activate ANYTHING (queued
candidates included, left untouched) while that flag is set. The flag only
clears when `activate_next_subgoal()` is called AGAIN with a
`localization_confidence` reading at/above
`hierarchy.localization_min_confidence`, at which point normal queue
popping resumes from wherever it left off. Because `record_local_tick`
cannot run at all while `stop_required` is `True` (it raises), the mission
timeout is additionally checked directly inside `activate_next_subgoal()`
while paused, so a localization that never recovers still eventually sets
`mission_timed_out` rather than stalling forever.

The intended caller contract (documented on
`HierarchyCoordinator`'s own module docstring): a live control loop whose
sensor feed keeps sampling localization while `stop_required` is `True`
should keep calling `activate_next_subgoal()` every tick with its current
confidence reading -- the same poll a normal "waiting for the next subgoal"
gap already needs, so no extra caller-side state machine is required. A
future Phase 4 ROS node driving this coordinator should also pass
`LocalPolicyController.decode_and_guard(..., localization_valid=False)`
whenever `coordinator.localization_degraded` is `True`, as defense in depth
alongside `stop_required` -- both checks agree by construction (the
coordinator can never report `stop_required=False` while
`localization_degraded=True`), but a caller that (incorrectly) skips the
`stop_required` check entirely should still be stopped by the guard.

New regression tests:
`test_localization_degraded_blocks_physical_motion_even_with_a_candidate_still_queued`,
`test_localization_degraded_pause_clears_once_confidence_recovers`,
`test_localization_degraded_pause_still_bounded_by_mission_timeout`.

## Round 2 fixes

**1 (state safety) -- `start_mission()` did not reset a still-active
subgoal.** A mission restarted while the PREVIOUS mission's subgoal was
still `ACTIVE` (never reached/failed/cancelled) left that stale subgoal
active: `active_subgoal_mission` kept the OLD coordinates and
`stop_required` stayed `False` across the restart, so a brand-new mission's
local policy could be handed a subgoal belonging to an entirely different
mission. Fixed: `SubgoalManager` gained an explicit `reset()` (clears
`status`/`subgoal_mission_xy` back to pre-`activate()` state and re-zeros
every accumulator, unconditionally -- unlike `finish()`, which requires
`ACTIVE` and would raise on an already-terminated subgoal);
`HierarchyCoordinator.start_mission()` now calls it, and also re-zeros
`previous_final_goal_distance`/`previous_subgoal_distance`/the internal
last-tick distance cache. `stop_required` is now always `True` immediately
after `start_mission()`, regardless of the previous mission's subgoal state.

**2 (state safety) -- final-goal success could be confirmed from a
degraded-confidence pose.** `record_local_tick` checked
`GoalManager.check_reached()` (and, further down, the subgoal-level
`SubgoalManager.check_reached()`) BEFORE ever looking at
`localization_confidence`, so a pose sampled from a low-confidence
(drifting/GPS-denied) localization estimate that happened to land inside
tolerance could confirm `mission_reached` -- or a subgoal `REACHED` --
outright. Fixed: `record_local_tick` now checks
`localization_confidence < hierarchy.localization_min_confidence` FIRST
(right after this tick's stats are recorded, before either the final-goal
or subgoal reached check), and short-circuits straight to
`CANCELLED_BY_REPLAN` / `"localization_confidence_degraded"` when degraded
-- exactly the same outcome the mid-route case already used, now also
covering the "pose happens to be inside tolerance" case. Neither
`mission_reached` nor a subgoal `REACHED` can ever be confirmed on a tick
whose reported localization confidence is below the configured minimum.

**3 (documentation contract) -- `LocalPolicyController.validate_action()`'s
"returns None, never raises" docstring did not actually hold.**
`np.asarray(action, dtype=np.float64).reshape(-1)` raises on a
ragged/inhomogeneous nested sequence (`ValueError`) or a plain object with
no array interface (`TypeError`) -- `real_policy_node.py` happens to wrap
its own call in an outer try/except, but this is a general-purpose pure API
a future hierarchy node may call directly. Fixed: the whole body is now
wrapped in `try/except (TypeError, ValueError): return None`.

**4 (documentation accuracy) -- verification doc profile count and
`test_hierarchical_interface.py` wording.** The doc claimed "17/17 OK" for
`config/profiles/*.yaml`; the actual, current count is 18 (confirmed via
`ls config/profiles/*.yaml | wc -l` and re-running the validator against
every file) -- corrected throughout this document and in the plan doc's own
Phase 2 completion note. The "was not added" wording for
`test_hierarchical_interface.py` is also tightened to explicitly say this
is a scoping decision, not a `pytest` skip, and plays no part in the "0
skipped" aggregate.

New regression tests (all in the files listed below):
`test_start_mission_clears_a_still_active_subgoal_from_the_previous_mission`,
`test_start_mission_clears_a_terminated_subgoal_from_the_previous_mission`,
`test_low_confidence_pose_inside_final_goal_tolerance_never_confirms_mission_success`,
`test_confidence_at_or_above_minimum_does_not_block_a_genuine_final_goal_reach`,
`test_validate_action_never_raises_on_a_ragged_nested_sequence`,
`test_validate_action_never_raises_on_a_plain_object_with_no_array_interface`,
`test_validate_action_never_raises_on_a_string`,
`test_validate_action_never_raises_on_none`.

## Scope delivered

New pure-Python modules (no ROS import at module scope; all independently
unit-testable):

- `navigation/local_rl/controller.py` -- `LocalPolicyController`: the
  ROS-independent slice of `nodes/real_policy_node.py`'s observation-build /
  action-decode / safety-guard pipeline, factored out so a future hierarchy
  control loop drives the local TQC through byte-identical logic. Only ever
  accepts an **active subgoal** coordinate -- it has no final-goal state to
  leak from, structurally.
- `navigation/hierarchy/subgoal_manager.py` -- `SubgoalManager` +
  `SubgoalStatus` (CREATED/ACTIVE/REACHED/FAILED_BLOCKED/FAILED_TIMEOUT/
  FAILED_NO_PROGRESS/FAILED_HIGH_RISK/CANCELLED_BY_REPLAN) + `SubgoalResult`
  (every stat the plan requires: local_steps, elapsed_time_sec,
  path_length_m, start/end final-goal and subgoal distances, minimum
  clearance, mean/max predicted risk, emergency-stop count, steering-
  saturation count, newly-explored cells -- optional fields are `None` when
  the caller never supplied a sample, never a fabricated sentinel).
- `navigation/hierarchy/replanning.py` -- `evaluate_replanning`: the seven
  live replanning conditions (tolerance reached, local timeout, no-progress
  window, endpoint blocked, consecutive emergency stops, risk threshold,
  localization confidence) plus a `junction_detected_stub` (always `False`,
  Phase 5 hook).
- `navigation/hierarchy/failure_recovery.py` -- `FailureRecoveryPolicy`:
  retry-same / advance-next / abort-mission decision from a terminated
  subgoal's outcome and the caller's own subgoal queue -- the "heuristic 또는
  외부 입력 subgoal sequence" stand-in for a future Global RL action.
- `navigation/hierarchy/coordinator.py` -- `HierarchyCoordinator`: owns the
  final `GoalManager`, the active `SubgoalManager`, a candidate-subgoal
  queue, and ties replanning + recovery together. Exposes
  `final_goal_mission` / `active_subgoal_mission` / `previous_final_goal_distance`
  / `previous_subgoal_distance` / `subgoal_reached` / `subgoal_failed` /
  `mission_reached` / `mission_failed` / `mission_timed_out` / `stop_required`
  exactly as the plan's data contract lists.
- `config/schema.py` -- new opt-in `HierarchyConfig` section (mirrors
  `MissionConfig`/`LocalizationConfig`/`MappingConfig`'s Phase 1 pattern),
  wired into `Profile`/`config/loader.py`; every existing profile keeps
  loading unchanged and simply carries its defaults.
- `nodes/real_policy_node.py` refactored: `_on_control_tick` now delegates
  observation-build and action-decode+guard to `LocalPolicyController`
  instead of holding the frame-stack/prev-action state inline. Behavior is
  unchanged -- verified byte-for-byte via the full existing
  `test_real_policy_node.py` / `test_real_policy_inference_worker.py` suites
  (updated only where they poked the now-internal `_frame_stack`/
  `_prev_action_01` attributes directly; assertions unchanged).

No new ROS `.srv`/`.msg`/nodes were added. `drl_agent_interfaces`'
`Reset.srv`/`Step.srv`/`GetDimensions.srv` are untouched. A Phase 2
`HierarchyCoordinator` + `LocalPolicyController` pairing is driven today by
a test harness (see the coordinator's own docstring for the intended
per-tick wiring); a live ROS node wiring them into Gazebo/real-hardware
sensor I/O is Phase 4's `hierarchical_environment_node.py`/
`hierarchical_train_node.py` (the plan's own module list). Because of this,
`tests/test_hierarchical_interface.py` was never created at all -- this is
a deliberate scoping decision (no new ROS interface exists yet to test),
not a `pytest.mark.skip`/collection skip; it plays no part in the "0
skipped" figure in the colcon summary below, which is the aggregate over
every test that DOES exist and ran.

## Key contracts verified by tests

- Subgoal REACHED is never treated as mission success -- only
  `GoalManager`'s own final-goal tolerance check sets `mission_reached`.
- `FAILED_TIMEOUT` / `FAILED_BLOCKED` / `FAILED_NO_PROGRESS` /
  `FAILED_HIGH_RISK` / `CANCELLED_BY_REPLAN` are five independently
  triggerable, distinctly-reasoned outcomes.
- `LocalPolicyController.decode_and_guard`'s `localization_valid=False` /
  `subgoal_valid=False` short-circuits to `STOP_COMMAND` before any
  trajectory decode happens at all.
- `HierarchyCoordinator.stop_required` is `True` before the first subgoal is
  activated, while awaiting the next subgoal after a terminal outcome, and
  once the mission itself has ended -- the caller-side contract for "never
  publish a stale command while replanning".
- `LocalPolicyController.build_observation`'s goal-distance/bearing tail is
  computed exclusively from whatever `(x, y)` the caller passes as the
  active subgoal; the module carries no separate final-goal state, so a
  final-goal leak is structurally impossible from this API.
- Multiple subgoals run consecutively without a Global RL agent, driven by
  a plain FIFO queue (`enqueue_subgoal`/`activate_next_subgoal`); a failed
  subgoal retries once (configurable) before the coordinator advances to
  the next queued candidate, and aborts the mission (`mission_failed`) only
  once the queue and retry budget are both exhausted.
- `mission_timed_out` is tracked independently of `mission_failed` (a
  mission-level step/time budget, separate from the local-option timeout).
- A degraded-confidence pose can never confirm `mission_reached` or a
  subgoal `REACHED`, even when it happens to land inside tolerance (Round 2
  fix 2, above) -- it always cancels-by-replan instead.
- Restarting a mission via `start_mission()` always leaves `stop_required`
  `True` and `active_subgoal_mission` `None`, regardless of whether the
  previous mission's subgoal was still active or already terminated (Round
  2 fix 1, above).
- `LocalPolicyController.validate_action()` returns `None` -- never raises
  -- for every malformed input tried, including ragged/inhomogeneous
  sequences and plain objects with no array interface (Round 2 fix 3,
  above).
- While `localization_degraded` is `True`, `activate_next_subgoal()`
  refuses to activate anything -- even an already-queued candidate --
  until called again with a recovered confidence reading; the pause is
  still bounded by the mission timeout (Round 3 fix 1, above).

## Test results (Docker, container `hunter_kinodynamic_rl` workspace)

```
source /opt/ros/humble/setup.bash
source install/setup.bash
colcon build --packages-select hunter_kinodynamic_rl        # 1 package finished, no errors
python3 -m hunter_kinodynamic_rl.config.validation <every config/profiles/*.yaml>   # 18/18 OK
colcon test --packages-select hunter_kinodynamic_rl
colcon test-result --test-result-base build/hunter_kinodynamic_rl --verbose
```

Result: **1475 tests, 0 errors, 0 failures, 0 skipped** (up from Phase 1's
1379 -- net new: `test_subgoal_manager.py`, `test_replanning.py`,
`test_hierarchy_coordinator.py`, `test_local_controller_contract.py`, 8 new
`HierarchyConfig` cases in `test_navigation_config.py`, plus the
`test_real_policy_node.py`/`test_real_policy_inference_worker.py` fixture
updates and the Round 2/Round 3 regression tests listed above -- all passing, zero
regressions in the pre-existing Phase 1 or local-only suites).

Round 4 targeted verification in the local host environment:

```
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=... python3 -m pytest -q \
  tests/test_hierarchy_coordinator.py tests/test_replanning.py
# 59 passed

PYTHONPATH=... python3 - <<'PY'
from pathlib import Path
from hunter_kinodynamic_rl.config.loader import load_profile
base = Path("config/profiles")
profiles = sorted(p.stem for p in base.glob("*.yaml"))
for name in profiles:
    load_profile(name).validate()
print(f"{len(profiles)}/{len(profiles)} profiles valid")
PY
# 18/18 profiles valid
```

The local host's default pytest plugin autoload path is broken by an
unrelated installed `anyio`/`pytest` plugin mismatch
(`ModuleNotFoundError: No module named '_pytest.scope'`), so the targeted
pytest run disables third-party plugin autoload. A full Docker/colcon rerun
was not performed for Round 4; the most recent full colcon result remains
the Round 3 result above.

## Remaining limitations

- No live Gazebo run was performed for Phase 2 -- unlike Phase 1, the plan's
  own section 6 completion criteria and this task's verification commands
  don't call for one (Phase 2's deliverable is the pure coordination
  contract + local-controller refactor, not a new sensor/ROS integration
  surface). `LocalPolicyController` reuses Phase-1-adjacent, already
  live-verified building blocks (`observation_builder`, `action_guard`,
  `trajectory_executor`) unchanged.
- `SubgoalManager`/`HierarchyCoordinator` are not yet wired to
  `navigation/mapping`'s `PartialMap.channels().occupied`/`.inflated` for a
  real "known occupied/inflated" `subgoal_endpoint_blocked` signal --
  `record_local_tick`/`activate_next_subgoal` accept it (and `is_valid`) as
  a caller-supplied boolean/callback, exercised in tests with synthetic
  values; wiring an actual `PartialMap` query through is natural but
  deliberately left to whichever of Phase 3/4 first has a live map to query
  against, keeping this module's own test surface focused on the
  hierarchy-state-machine contract itself.
- `junction_detected` is wired only as the documented Phase 5 stub
  (`junction_detected_stub`, always `False`).
- No new checkpoint/replay schema changes -- Phase 2 introduces no new
  network or trained component, so there is nothing to version here yet.
