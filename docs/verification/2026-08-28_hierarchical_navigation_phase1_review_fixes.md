# 2026-08-28: Phase 1 code-review fixes + live Gazebo verification

> **Historical verification artifact.** 당시 source/명령/결과를 보존한다. 최신
> 상태는 [`../CURRENT_STATUS.md`](../CURRENT_STATUS.md)를 따른다.

Follow-up to `2026-08-28_hierarchical_navigation_phase1.md`. An external code
review of that Phase 1 delivery found the runtime was NOT actually verified
against a live Gazebo instance and identified several real defects that
would have prevented `/mission_map` from ever updating in practice. All are
fixed here, and (unlike the original delivery) this pass includes genuine
live Gazebo runs, not just unit/node tests with synthetic messages.

> **Round 2 (same date)**: a second review pass on THIS document's own
> fixes found four more real defects (long-gap `pose_at` interpolation,
> `confidence` missing from finiteness checks, the low-speed-goal gate
> never actually reading real speed, mission-frame init skipping the
> confidence gate) plus two process items (LocalizationBackend protocol
> missing `pose_at`, and the 4-state/3-state plan-doc mismatch left
> undocumented). See "Round 2 fixes" and "Second live Gazebo run" below.
>
> **Round 3 (same date)**: a third review pass found the round-2 speed-
> tracking fix itself had a gap -- a NaN/Inf `/odometry` twist bypassed the
> low-speed-on-goal gate (`abs(nan) > threshold` is `False` in Python).
> Fixed in "Round 3 fix" below, along with the two remaining doc
> mismatches (the plan's `LocalizationBackend` example and
> `interface.py`'s own module docstring both still said "`latest_pose()`
> alone"). This closes out every reviewed code item except item 10 (git
> tracking, a repo-management decision left to the user).

## Fixes (by review item number)

**1 (P0) -- `/scan` QoS mismatch.** `mission_map_node.py` subscribed to
`/scan`/`/odometry` with the rclpy default (RELIABLE) QoS; the real
`pointcloud_to_laserscan` and Ignition odometry-bridge publishers use
`SensorDataQoS`/BEST_EFFORT, which is incompatible per DDS rules -- exactly
the bug already once found and fixed in this codebase's own Nav2-MPPI
baseline (`evaluation/nav2_mppi_runner.py`'s docstring: "a QoS mismatch that
silently starved the /scan subscription"). Fixed: both subscriptions now use
a `QoSProfile(depth=10, reliability=BEST_EFFORT)` constant (`SENSOR_QOS`),
matching `environment_node.py`'s own established convention exactly.

**2 (P0) -- sim-time/wall-time clock mixing.** The scan-freshness check
compared a Gazebo-bridged (sim-time) scan stamp against `time.time()`
(wall-clock) -- the same class of bug previously found live in this exact
codebase (`nav2_mppi_runner.py`'s docstring: "a use_sim_time mismatch
between Nav2's clock and Gazebo-bridged /odometry's sim-time stamps").
Fixed structurally, not by adding `use_sim_time` plumbing (which the
existing `environment_node.py` deliberately avoids for the same reason):
`_on_scan` never reads any external clock at all now. `OdomLocalizationBackend`
gained a bounded pose history (`deque`, `DEFAULT_HISTORY_SIZE=50`) and
`pose_at(stamp_sec, max_dt_sec)`, which looks up (linearly interpolating,
wrapped yaw) the pose AT the scan's own timestamp rather than "whatever
odometry happened to be most recently received" -- scan and odometry are
independent topics with no ordering guarantee. `is_pose_usable`'s age check
is now symmetric (`abs(age)`) since a synchronized pose can legitimately
land a hair after `now_sec`.

**3 (P1) -- non-finite localization passed as valid.** `is_pose_usable`
checked only `valid`/`confidence`/age, not whether `x`/`y`/`yaw`/`stamp_sec`/
`covariance` were finite. Added `is_pose_finite`, called from
`is_pose_usable` AND independently in `mission_map_node.py` before
`MissionFrame.initialize()`. `OdomLocalizationBackend.update()` now forces
`valid=False` on any non-finite field regardless of what the caller passed,
and excludes non-finite readings from the interpolation history entirely.
`MissionFrame.initialize()` itself now raises `ValueError` on a non-finite
start pose (defense in depth beyond the node-level check).

**4 (P1) -- boundary-clipped ray fabricating a wall.** `integrate_beam`
clipped a hit ray to the grid boundary and marked the CLIPPED cell occupied
even when the PHYSICAL endpoint was outside the map -- fabricating an
obstacle at the map edge for something that might be much further away.
Fixed: the occupied mark is now gated on `world_to_cell(physical_endpoint)
is not None`; an out-of-bounds physical hit still marks the boundary-clipped
cell FREE (matches max-range semantics), never occupied.

**5 (P1) -- `+Inf` beams treated as no-information.** Real LiDAR
drivers/this project's own Gazebo bridge publish a "no return" beam as
`+inf`, not NaN. `integrate_beam` previously skipped `+inf` identically to
NaN (leaving open space permanently UNKNOWN). Fixed: `+inf` is now treated
exactly like a max-range beam (FREE out to `range_max`, no occupied
endpoint); `-inf`/NaN are still skipped entirely (no information at all);
`range_m < range_min` is also now skipped (new `range_min` parameter on
`integrate_beam`/`integrate_scan`, wired from `LaserScan.range_min` in the
node).

**6 (P1) -- undocumented 4th map state.** An observed cell whose log-odds
falls strictly between `free_threshold` and `occupied_threshold` was
neither `occupied` nor `free` internally, but the node exported it as
occupancy value 50 with no name anywhere. This state is real (the plan's
own two-threshold formula produces it) -- made explicit as
`MapChannels.observed_uncertain` (also threaded through `RollingCrop`),
documented in `partial_map.py`'s and `MappingConfig`'s docstrings as the
actual, exhaustive 4-way partition (`occupied | free | observed_uncertain |
unknown` covers every cell exactly once -- see
`test_four_channels_exhaustively_and_disjointly_partition_every_cell`).
`channel_to_occupancy_data` now takes all four masks explicitly instead of
defaulting unclassified cells to a magic constant.

**7 (P2) -- dead config fields.** `localization.backend` always constructed
`GazeboOdomLocalizationBackend`, and `localization.publish_mission_tf`'s TF
publish was unconditional. Fixed: `GazeboOdomLocalizationBackend` gained
`use_covariance_confidence` (True for `"gazebo_odom"`, False -- confidence
pinned to 1.0 -- for `"odom"`), selected from config in the node
constructor; `_publish_all` now skips `_publish_tf` when
`publish_mission_tf` is False.

**8 (P2) -- silent mission-frame re-initialization.** `MissionFrame.initialize()`
silently overwrote an existing origin. Now raises `MissionFrameError` on a
second call without an intervening `reset()`.

**9 (P2) -- verification/report mismatch.** Addressed by this document: an
accurate current test count, and a live Gazebo run (below) exercising the
exact failure modes the previous report's synthetic-only node tests
happened to mask (both node tests previously fed `time.time()` to BOTH
odometry and scan, which cannot reproduce a sim/wall clock mismatch at all
-- rewritten to use arbitrary message-domain stamps, including one
regression test parametrized over both small sim-time-like and huge
wall-clock-epoch-shaped values).

**10 (P1) -- git tracking.** Not a code defect; flagged to the user
directly (this package is `.gitignore`d from the main repo with no nested
`.git` present, so these changes are not currently tracked anywhere) rather
than acted on unilaterally, since initializing a repository/remote is the
user's call.

## Live Gazebo verification

Container `7a2702b311a1`, `hunter_se_gazebo simulate_hunter_se_ignition.launch.py`
(`drl_arena` world, RGL GPU LiDAR, confirmed real `/clock` advancing --
not the CPU-fallback ~0.001x RTF failure mode), `mission_map_node.py
--ros-args -p profile:=hierarchical_phase1 -p goal_x:=5.0 -p goal_y:=2.0`.

- **`/scan` reception**: confirmed receiving (BEST_EFFORT `/scan` at
  ~18 Hz, `/odometry` at ~44 Hz measured via `ros2 topic hz`); the node's
  scan callback fires and integrates continuously (verified via temporary
  instrumentation, reverted after use).
- **`/mission_map` updates / UNKNOWN -> FREE/OCCUPIED**: confirmed. A
  direct subscriber reading the full (untruncated) `OccupancyGrid.data`
  array observed `{-1: 3488, 100: 330, 0: 12566}` out of 16384 cells after
  the room's static geometry was scanned -- a real UNKNOWN -> FREE/OCCUPIED
  transition, not the all-UNKNOWN state a first naive `ros2 topic echo
  --once` check appeared to show (that tool truncates large arrays to a
  short prefix by default -- a red herring caught by cross-checking with a
  dedicated Python subscriber that reads the whole array).
- **`/visited_map` changes with movement**: confirmed. Publishing
  `/cmd_vel` (linear.x=0.8) for several seconds while watching
  `/visited_map` live: nonzero visited-cell count grew from 13 (stationary)
  to 138 as `robot_mission.x` advanced from 0.0 to 5.18 m, tracking the
  robot's actual mission-frame trajectory step by step (confirmed via the
  same temporary instrumentation).
- **Goal fixed in mission frame despite motion**: confirmed. `/mission_goal`
  stayed exactly `(5.0, 2.0)` in frame `mission` throughout, while
  `/odometry`'s `pose.pose.position.x` moved from 4.5 -> 9.31 -> 11.4 m and
  the robot turned.
- **`odom -> mission` TF**: confirmed via `ros2 run tf2_ros tf2_echo odom
  mission` -- translation `(4.5, 0.0, 0.0)`, identity rotation, matching
  the logged spawn pose exactly.
- **`/inflated_map`**: confirmed publishing real dilated data
  (`{100: 1648, -1: 2818, 0: 11918}`), consistent with
  `mapping.inflation_radius_m=0.45` over the 330 raw occupied cells.
- **Stale/invalid localization halting map updates**: NOT separately
  live-triggered (would require deliberately killing/freezing the odometry
  feed mid-run) -- covered by
  `test_node_rejects_stale_localization_for_map_update_pure_message_domain`
  and the `OdomLocalizationBackend`/`is_pose_usable` unit tests using
  synthetic messages with large, purely message-domain timestamp gaps.
- **Different start pose/yaw mission-frame consistency**: only ONE live
  spawn pose was exercised (yaw ~ 0). The general claim (arbitrary start
  pose/yaw -> correct relative-goal transform) is covered by
  `test_mission_odom_round_trip_arbitrary_start_pose`'s parametrization
  over multiple `(x, y)` x multiple yaw values (including non-trivial
  yaws), not by a second live Gazebo spawn -- restarting Gazebo with a
  different spawn pose per run was out of scope for this pass's time
  budget.

## New finding: scan-integration performance

Not one of the review's 10 items, but discovered live: a REAL Ouster-derived
`/scan` on this world has ~2048 beams (`range_max=50.0`), and
`PartialMap.integrate_beam`'s pure-Python per-beam Bresenham trace took
~0.33 s/scan against a 128x128 `hierarchical_phase1` grid -- i.e. the node
can sustain only ~3 scans/s against an 18 Hz publisher (BEST_EFFORT QoS
means excess scans are simply dropped, not queued, so this does not break
correctness, only reduces effective map-update rate). Acceptable for a
Phase 1 verification node; a vectorized (numpy-batched) ray-tracing
implementation would be needed before this becomes a real-time Global-RL
observation source in Phase 4+.

## Round 2 fixes

A second review pass on the fixes above found four more real defects (all
in the `navigation/localization/` + `mission_map_node.py` path) and two
process items.

**1 (P1) -- `pose_at()` interpolated across an unbounded gap.** The
"before AND after both exist" branch interpolated regardless of how far
apart the two bracketing samples were -- odometry at t=0 (x=0) and t=100
(x=100) bracketing a scan at t=50 with `max_dt_sec=0.5` still produced a
"valid" `x=50` pose, silently fabricating a straight-line path across what
could be a 100s localization outage or rosbag stall. Fixed:
`OdomLocalizationBackend.pose_at` now requires EACH side of the bracket
(`stamp_sec - before.stamp_sec` and `after.stamp_sec - stamp_sec`) to
independently be `<= max_dt_sec` before interpolating; if only one side is
close, it falls back to that single nearest sample (still gated by the same
bound), and if neither side is close it returns `INVALID_POSE`.

**2 (P1) -- `confidence` excluded from finiteness checks.** `is_pose_finite`
checked `x`/`y`/`yaw`/`stamp_sec`/`covariance` but not `confidence` --
`NaN < min_confidence` is `False` in Python, so a NaN confidence silently
passed `is_pose_usable`'s `confidence < min_confidence` rejection even at
`min_confidence=0.0`. Fixed: `is_pose_finite` now also requires
`math.isfinite(confidence)` AND `0.0 <= confidence <= 1.0`;
`OdomLocalizationBackend`'s own `_is_finite_pose` gate (used by `update()`
to force `valid=False`) checks the same.

**3 (P1) -- low-speed-on-goal gate always saw speed=0.0.** `_last_speed_mps`
was initialized to `0.0` and never updated anywhere -- a leftover from an
earlier refactor of `_on_odometry`. With `hierarchical_phase1`'s
`require_low_speed_on_goal: true`, a robot driving through the goal
tolerance radius at full speed would have registered as goal-reached.
Fixed: `_on_odometry` now records `(stamp_sec, speed_mps)` from the
message's own `twist.twist.linear` into a bounded history (mirroring the
pose-history pattern), and `_speed_at`/`_current_speed_at` look up the
nearest sample by timestamp -- returning `inf` (not `0.0`) when no sample
is close enough, so a missing/stale speed reading fails toward REJECTING a
goal-reached rather than falsely accepting one.

**4 (P1) -- mission-frame init skipped the confidence gate.** `_on_odometry`
initialized the mission frame from `pose.valid and is_pose_finite(pose)`
alone -- a first reading with a huge covariance (hence low
confidence) could permanently anchor the mission origin even though every
LATER map update would correctly get rejected by the same confidence gate;
by then the origin itself is already wrong. Fixed: uses the same
`is_pose_usable(pose, now_sec=pose.stamp_sec, timeout_sec=..., min_confidence=...)`
gate every ordinary map update goes through (`now_sec=pose.stamp_sec` makes
the age term exactly 0, so this purely adds the confidence check on top of
the existing valid/finite checks).

**5 (P2) -- `LocalizationBackend` protocol missing `pose_at`.** The Protocol
only declared `latest_pose()`, even though `mission_map_node.py` depends on
`pose_at()` directly -- an incomplete abstraction that would make a Phase 6
backend swap (wheel+IMU, LiDAR odometry, LIO) a navigation-core change
instead of a backend-only one. Fixed: `pose_at(self, stamp_sec, max_dt_sec)
-> PoseEstimate` added to the Protocol.

**6 (P2) -- 4-state map vs. the plan/spec's stated 3 channels.** Not fixed
in code (a design decision, not a bug) -- annotated instead. The
implementation plan (`docs/HIERARCHICAL_NAVIGATION_IMPLEMENTATION_PLAN.md`
section 5.4) now carries an explicit note next to its original 3-channel
formula explaining that the formula ITSELF (two distinct thresholds)
produces the `observed_uncertain` 4th state, and listing the three options
Phase 4 must pick from before designing the Global RL observation tensor
(separate channel / merge into policy input / collapse to a single
threshold). Phase 1's own deliverable is the internal partial map and
mission/localization plumbing, not the Global observation contract, so this
is deferred rather than resolved now.

## Second live Gazebo run (arbitrary start pose/yaw)

The first live run (above) only exercised the default spawn pose
(`x=4.5, y=0, yaw~=0`) -- insufficient to verify the mission-frame rotation
math against a REAL, non-trivial Gazebo-sourced orientation quaternion (unit
tests cover arbitrary yaw with synthetic poses, but not the live
quaternion-to-yaw extraction path). Rather than edit `hunter_se_gazebo`'s
launch file (a different, shared package) to add spawn-pose arguments, the
robot was teleported via the existing `/world/default/set_pose`
(`ros_gz_interfaces/srv/SetEntityPose`) service BEFORE starting
`mission_map_node.py`, to `x=1.0, y=3.0, yaw=-pi/3` (confirmed via
`/odometry`: `position=(1.0, 3.0)`, `orientation=(z=-0.5, w=0.866)`, exactly
`yaw=-60 deg`).

- **Mission frame origin**: node log reported
  `mission frame initialized at (1.000, 3.000, -1.047)` -- exact match.
- **`odom -> mission` TF**: `tf2_echo odom mission` -> translation
  `(1.0, 3.0, 0.0)`, rotation `yaw=-1.047 rad (-60 deg)` -- exact match.
- **Relative goal fixed + correctly rotated**: launched with
  `goal_x:=3.0 goal_y:=-1.0`; `/mission_goal` read back exactly `(3.0,
  -1.0)` in frame `mission`. Applying the TF's own rotation matrix to that
  point by hand (`R = [[0.5, 0.866, 0], [-0.866, 0.5, 0]]`, `T = [1.0,
  3.0]`) gives `odom_goal ~= (1.634, -0.098)`, which matches the plan's own
  formula (section 3.1) computed independently:
  `x0 + cos(yaw0)*gx - sin(yaw0)*gy = 1.634`,
  `y0 + sin(yaw0)*gx + cos(yaw0)*gy = -0.098`. Confirms the coordinate
  transform is correct for a real, non-zero-origin, non-zero-yaw Gazebo
  pose, not just the synthetic unit-test cases.
- **`/mission_map` still populates correctly**: `{-1: 3479, 100: 315, 0:
  12590}` of 16384 cells -- a real UNKNOWN -> FREE/OCCUPIED transition,
  proportions closely matching the first (yaw~=0) run, as expected for a
  similarly-open region of the same arena viewed from a rotated frame.

## Process-cleanup lesson (this pass)

After the second live run, `pkill -9 -f <pattern>` against the Gazebo/bridge
process names silently failed to actually terminate them (unclear why --
possibly a quoting/matching interaction inside the `bash -lc` wrapper), and
this was NOT re-verified before moving on. The stale (but still fully live)
Gazebo instance kept publishing real `/joint_states` on the same
`ROS_DOMAIN_ID` used by the next `colcon test` run, and two unrelated
`test_system_id_node.py` tests -- which publish their OWN synthetic
JointState message and assert on receiving THAT exact value -- instead
received the real robot's actual (near-zero) steering angle from the
leftover live simulation, producing two spurious failures
(`assert 5.48e-10 == 0.21 +/- 2.1e-07`) that had nothing to do with any
code changed in this pass. Root-caused by killing every matching PID
explicitly (`kill -9 <pid> ...`), confirming via `ps aux` AND `ros2 node
list` (the latter needed a `ros2 daemon stop` to drop its own stale
discovery cache), then re-running the full suite clean. Lesson: after any
live-Gazebo session, verify cleanup with `ros2 node list` (not just `ps
aux`, and not just trusting a `pkill` exit code) before trusting the next
`colcon test` run's result.

## Round 3 fix

**1 (P1) -- NaN/Inf `/odometry` twist bypassed the low-speed-goal gate.**
Round 2's `_speed_history`/`_speed_at` fix tracked real speed correctly,
but `_on_odometry` stored `math.hypot(msg.twist.twist.linear.x,
msg.twist.twist.linear.y)` unguarded -- a NaN/Inf twist component (e.g. a
corrupted message) produces a NaN/Inf `speed_mps`, and
`abs(float("nan")) > threshold` is `False` in Python, so
`GoalManager.check_reached`'s low-speed rejection silently never fired.
Reproduced live before the fix: a robot within goal tolerance reporting a
NaN speed registered `goal_reached=True`. Fixed at BOTH the ingestion point
(`_on_odometry` forces a non-finite `speed_mps` to `+inf` -- the same
fail-toward-"moving" value `_speed_at` already returns for a
missing/stale sample -- before appending to history) and, as defense in
depth, at the read point (`_speed_at` re-checks `math.isfinite` on the
value it's about to return, in case a non-finite value ever reaches
history some other way). Five new regression tests cover NaN/+Inf/-Inf
twist ingestion, the full reported goal-reached scenario end-to-end, and
`_speed_at`'s defense-in-depth path directly.

**2 (doc-only) -- stale `pose_at()`-less protocol references.** The
`LocalizationBackend` example in
`HIERARCHICAL_NAVIGATION_IMPLEMENTATION_PLAN.md` (section 5.3) and
`interface.py`'s own module docstring both still described `latest_pose()`
alone, even though `pose_at()` was added to the real Protocol in round 2.
Both updated so a future Phase 6 backend implementer isn't working from a
stale contract.

## Docker verification

```
colcon build --packages-select hunter_kinodynamic_rl   # 0 errors
config validation, every config/profiles/*.yaml         # all OK
colcon test --packages-select hunter_kinodynamic_rl
colcon test-result --test-result-base build/hunter_kinodynamic_rl --verbose
```

Result (final, after round 3 fix + confirmed-clean process state):
`1379 tests, 0 errors, 0 failures, 0 skipped` (1354 after round 1 + 20
round-2 + 5 round-3 regression tests). Two live Gazebo runs performed in
total (default spawn pose, then teleported to an arbitrary non-zero
pose/yaw); all spawned Gazebo/node processes confirmed cleanly terminated
(`ps aux` + `ros2 node list` both empty) both after the live runs and
before this final test run.
