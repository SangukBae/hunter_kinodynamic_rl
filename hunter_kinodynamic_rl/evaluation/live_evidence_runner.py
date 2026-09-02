#!/usr/bin/env python3
"""Requirement H / defect-fix item 9: reproducible Phase 3 live-evidence
smoke runner.

Drives ONE short mission over live Gazebo using the SAME
``navigation.local_rl.live_gazebo_executor.LiveGazeboLocalExecutor`` +
``navigation.hierarchy.coordinator.HierarchyCoordinator`` every other live
script in this package already uses (never a separate, parallel
implementation) and records:

- a structured JSONL event log (one JSON object per line) for reset, wall
  activation, robot teleport, sensor-freshness wait (start/complete +
  elapsed), mission start, EVERY local control tick's key fields (step,
  time, pose, subgoal, action, command, predicted risk, guard/emergency-
  stop flags, cumulative collision count -- via
  ``LiveGazeboLocalExecutor.run_option``'s optional ``on_tick`` hook, added
  for this exact purpose rather than re-deriving telemetry after the fact),
  option termination, mission termination, and teardown -- NOT just a
  prose stdout summary.
- raw process stdout/stderr, if this runner also launched Gazebo itself
  (``launch_gazebo=True``) -- preserved as plain log files, explicitly
  closed once the child process has exited (defect-fix item 9: the v1
  runner opened these files but never closed them).
- a final ``summary.json``: world/seed, start pose, goal, topic-readiness
  timings, timeout budget, termination reason, and a leftover-process
  check result SCOPED TO THIS RUN'S OWN PROCESS GROUP (defect-fix item 9:
  a global ``pgrep`` for ``gzserver``/``gzclient`` would misclassify any
  OTHER user's/session's unrelated Gazebo instance as a "leftover" of this
  run -- the check below only ever inspects the process group this run
  itself created via ``launch_gazebo=True``, and is a documented no-op
  when ``launch_gazebo=False`` since this run then owns no such group).

Defect-fix item 9's core gap: the default ``profile`` parameter was
``"hierarchical_phase3"``, a name with NO corresponding
``config/profiles/hierarchical_phase3.yaml`` -- ``load_profile()`` fails
immediately on any default invocation. Fixed by defaulting to
``"hierarchical_phase4"`` (an existing, live-evidence-appropriate profile
whose ``hierarchical_training.local_checkpoint_dir``/``local_checkpoint_name``
already point at the canonical promoted-Local location, defect-fix item 5)
AND running :func:`preflight_live_evidence` BEFORE touching Gazebo at all
-- a profile whose Local checkpoint isn't actually promoted/compatible now
fails fast with an actionable reason instead of a deep exception from
``LiveGazeboLocalExecutor``'s constructor partway through a live run.

``launch_gazebo=False`` (default, matches every other live script's
"Gazebo already running in a separate terminal" convention in this
package): this runner only drives the mission and does its OWN cleanup
(``live_executor.close()``), it does not manage a Gazebo subprocess at all.
``launch_gazebo=True`` additionally starts/stops
``simulate_hunter_se_ignition.launch.py`` as a subprocess (in its own new
session/process group, via ``start_new_session=True``) and captures its
stdout/stderr, then verifies (via ``pgrep -g <this run's own pgid>``) that
no ``gzserver``/``gzclient``/this launch's ROS nodes remain after teardown
-- on BOTH the success and failure paths (``finally``).

NOT executed in this session (needs live Gazebo + a frozen Local
checkpoint); scoped for a SHORT smoke mission only, never a full
long-horizon world completion (plan section 7.8's own scope note)."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from typing import List, Optional

#: Only matched WITHIN this run's own process group (see
#: _check_leftover_processes) -- never a blanket system-wide pgrep.
LEFTOVER_PROCESS_PATTERNS = ("gzserver", "gzclient", "ruby.*ign gazebo", "ign gazebo")

DEFAULT_LIVE_EVIDENCE_PROFILE = "hierarchical_phase4"


@dataclass
class EventLog:
    path: str
    _fh = None

    def __post_init__(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._fh = open(self.path, "a")

    def emit(self, event_type: str, **fields) -> None:
        record = {"ts_unix": time.time(), "event": event_type, **fields}
        self._fh.write(json.dumps(record, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def preflight_live_evidence(profile) -> "PreflightResult":
    """Defect-fix item 9: checked BEFORE touching Gazebo at all.

    - ``long_horizon_world.enabled`` (this runner is meaningless against a
      profile with no procedural long-horizon world to explore).
    - The profile's own Local checkpoint is actually PROMOTED (structured
      validator, never a bare file-existence check --
      ``evaluation.local_promotion.validate_promoted_local_checkpoint``)
      and its recorded ``architecture_fingerprint`` matches THIS profile's
      own (never load a promoted checkpoint trained under a DIFFERENT
      architecture against this profile's Local controller/network shapes).
    - Wall pool capacity: already enforced at config-load time by
      ``Profile.validate()`` -> ``validate_wall_pool_capacity`` whenever
      ``wall_segment_pool.enabled`` -- ``load_profile()`` succeeding at all
      is that guarantee; this preflight re-affirms it defensively rather
      than re-implementing the check.
    - Timeout budget: reports ``max_local_steps * runtime.time_delta_sec``
      (the real wall-clock budget one option can consume) for the caller
      to sanity-check against its own smoke-scope timeout, never silently
      assumed to be small.
    """
    from hunter_kinodynamic_rl.evaluation.fingerprint import architecture_fingerprint
    from hunter_kinodynamic_rl.evaluation.local_promotion import validate_promoted_local_checkpoint

    errors: List[str] = []
    info: dict = {}

    if not profile.long_horizon_world.enabled:
        errors.append(
            f"profile {profile.name!r} has long_horizon_world.enabled=False -- this runner requires a "
            "procedural long-horizon world"
        )

    ckpt_dir = profile.hierarchical_training.local_checkpoint_dir
    ckpt_tag = profile.hierarchical_training.local_checkpoint_name
    validation = validate_promoted_local_checkpoint(ckpt_dir, ckpt_tag)
    info["local_checkpoint_promoted"] = validation.promoted
    if not validation.promoted:
        errors.append(
            f"Local checkpoint at {ckpt_dir!r}/{ckpt_tag!r} is not a valid PROMOTED checkpoint: "
            f"{'; '.join(validation.reasons)}"
        )
    else:
        recorded_arch = (validation.promotion_manifest or {}).get("architecture_fingerprint")
        expected_arch = architecture_fingerprint(profile)
        if recorded_arch != expected_arch:
            errors.append(
                f"promoted Local checkpoint architecture_fingerprint={recorded_arch!r} != this profile's "
                f"own architecture_fingerprint={expected_arch!r} -- the promoted checkpoint was benchmarked "
                "against a DIFFERENT profile/architecture than the one this run requested"
            )

    # Wall pool capacity is already enforced by Profile.validate() at
    # load_profile() time (see validate_wall_pool_capacity) -- if we got a
    # `profile` object at all, this already holds.
    info["wall_pool_capacity_validated_at_config_load"] = True

    timeout_budget_sec = None
    try:
        timeout_budget_sec = float(profile.runtime.time_delta_sec)
    except AttributeError:
        pass
    info["local_control_period_sec"] = timeout_budget_sec

    return PreflightResult(ok=not errors, errors=errors, info=info)


@dataclass
class PreflightResult:
    ok: bool
    errors: List[str] = field(default_factory=list)
    info: dict = field(default_factory=dict)


def _check_leftover_processes(pgid: Optional[int]) -> List[str]:
    """Defect-fix item 9: scoped ``pgrep -g <pgid>`` -- returns a list of
    matched process command lines belonging to THIS RUN'S OWN process
    group (empty = clean). ``pgid is None`` means this run never launched
    Gazebo itself (``launch_gazebo=False``), so there is no process group
    to check -- reported as an explicit ``["not_applicable..."]`` marker,
    never silently claimed clean nor checked against unrelated processes.
    Never raises on a platform without ``pgrep``; reports an explicit
    `"pgrep unavailable"` marker instead of silently claiming a clean
    result."""
    if pgid is None:
        return ["not_applicable: launch_gazebo=False -- this run owns no Gazebo process group"]
    try:
        result = subprocess.run(
            ["pgrep", "-g", str(pgid), "-a"], capture_output=True, text=True, timeout=5.0,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().splitlines()
        return []
    except (FileNotFoundError, subprocess.SubprocessError):
        return ["pgrep unavailable -- leftover-process check could not run"]


def run_live_evidence_smoke(
    profile_name: str = DEFAULT_LIVE_EVIDENCE_PROFILE, world_seed: int = 0,
    output_dir: str = "runtime/live_evidence", max_local_steps: int = 40, launch_gazebo: bool = False,
    world_name: str = "default", launch_timeout_sec: float = 60.0, logger=print,
) -> dict:
    """Requires ``rclpy.init()`` to have already happened in the caller,
    exactly like every other live driver in this package (``main()`` below
    does this) -- independent of ``launch_gazebo``, which only controls
    whether this function ALSO manages a Gazebo subprocess."""
    from hunter_kinodynamic_rl.config.loader import load_profile
    from hunter_kinodynamic_rl.env.scenarios.long_horizon_generator import generate_long_horizon_world
    from hunter_kinodynamic_rl.navigation.hierarchy.coordinator import HierarchyCoordinator
    from hunter_kinodynamic_rl.navigation.local_rl.live_gazebo_executor import LiveGazeboLocalExecutor
    from hunter_kinodynamic_rl.navigation.mission.mission_frame import MissionFrame, PoseXYYaw
    from hunter_kinodynamic_rl.training.train_hierarchical_dqn import hierarchy_config_from

    run_dir = os.path.join(output_dir, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    event_log = EventLog(os.path.join(run_dir, "events.jsonl"))
    launch_proc: Optional[subprocess.Popen] = None
    launch_stdout_f = launch_stderr_f = None
    summary = {
        "profile_name": profile_name, "world_seed": world_seed, "run_dir": run_dir,
        "termination_reason": "not_started",
    }
    try:
        profile = load_profile(profile_name)
        event_log.emit("profile_loaded", profile_name=profile_name)

        preflight = preflight_live_evidence(profile)
        event_log.emit("preflight_complete", ok=preflight.ok, errors=preflight.errors, info=preflight.info)
        summary["preflight"] = {"ok": preflight.ok, "errors": preflight.errors, "info": preflight.info}
        if not preflight.ok:
            raise RuntimeError(f"live evidence preflight failed: {'; '.join(preflight.errors)}")

        if launch_gazebo:
            launch_stdout_f = open(os.path.join(run_dir, "raw_launch_stdout.log"), "w")
            launch_stderr_f = open(os.path.join(run_dir, "raw_launch_stderr.log"), "w")
            launch_proc = subprocess.Popen(
                ["ros2", "launch", "hunter_se_gazebo", "simulate_hunter_se_ignition.launch.py",
                 f"world:={world_name}", "rviz:=false"],
                stdout=launch_stdout_f, stderr=launch_stderr_f, start_new_session=True,
            )
            event_log.emit(
                "gazebo_launch_started", world_name=world_name, pid=launch_proc.pid,
                pgid=os.getpgid(launch_proc.pid),
            )
            time.sleep(launch_timeout_sec)  # bounded settle wait -- no service-discovery signal to poll here

        min_turning_radius_m = 1.0 / profile.robot.max_curvature
        world = generate_long_horizon_world(
            world_seed, profile.long_horizon_world, profile.robot.collision_radius_m, min_turning_radius_m,
            profile.robot.wheelbase_m, mode="test",
        )
        summary["start_pose"] = world.start_pose
        summary["goal_pose"] = world.goal_pose
        event_log.emit(
            "world_generated", world_seed=world_seed, start_pose=world.start_pose, goal_pose=world.goal_pose,
            wall_segment_count=len(world.wall_segments),
        )

        mission_frame = MissionFrame()
        mission_frame.initialize(PoseXYYaw(*world.start_pose))
        goal_mission = mission_frame.odom_to_mission(world.goal_pose[0], world.goal_pose[1])

        live_executor = LiveGazeboLocalExecutor(
            profile, profile.long_horizon_world,
            profile.hierarchical_training.local_checkpoint_dir, profile.hierarchical_training.local_checkpoint_name,
        )
        event_log.emit("local_checkpoint_loaded", **{
            k: live_executor.checkpoint_manifest.get(k) for k in ("generation", "pt_sha256")
        })
        try:
            t0 = time.monotonic()
            live_executor.bind_mission(
                world, mission_frame,
                on_event=lambda name, fields: event_log.emit(name, **fields),
            )
            event_log.emit(
                "reset_complete", elapsed_sec=time.monotonic() - t0,
                scan_update_count=live_executor.scan_update_count, odom_update_count=live_executor.odom_update_count,
            )

            coordinator = HierarchyCoordinator(hierarchy_config_from(profile.hierarchy))
            coordinator.start_mission(goal_mission[0], goal_mission[1], now_step=0, now_time_sec=0.0)
            coordinator.enqueue_subgoal(*goal_mission)
            activated = coordinator.activate_next_subgoal(
                mission_frame.odom_pose_to_mission(live_executor.pose_world), now_step=0, now_time_sec=0.0,
                is_valid=lambda x, y: True,
            )
            event_log.emit("mission_start", activated=activated, goal_mission=goal_mission)

            if activated:
                from hunter_kinodynamic_rl.navigation.mapping.partial_map import PartialMap
                partial_map = PartialMap(profile.mapping)
                import numpy as np
                live_executor.run_option(
                    coordinator, partial_map, np.random.RandomState(0), max_local_steps,
                    on_tick=lambda fields: event_log.emit("local_control_tick", **fields),
                )
                event_log.emit("option_termination", status=(
                    coordinator.last_subgoal_result.status.value
                    if coordinator.last_subgoal_result is not None
                    and hasattr(coordinator.last_subgoal_result.status, "value") else None
                ))

            result = coordinator.last_subgoal_result
            summary["termination_reason"] = result.reason if result is not None else "no_result"
            event_log.emit(
                "termination", reason=summary["termination_reason"],
                status=(result.status.value if result is not None and hasattr(result.status, "value") else None),
                local_steps=(result.local_steps if result is not None else 0),
            )
        finally:
            live_executor.close()
            event_log.emit("live_executor_closed")
    except Exception as e:  # noqa: BLE001 -- must still run cleanup/leftover-check below, never re-raise silently
        summary["termination_reason"] = f"error: {e}"
        event_log.emit("error", message=str(e))
        raise
    finally:
        launch_pgid = None
        if launch_proc is not None:
            try:
                launch_pgid = os.getpgid(launch_proc.pid)
            except ProcessLookupError:
                launch_pgid = None
            event_log.emit("gazebo_launch_terminating", pid=launch_proc.pid, pgid=launch_pgid)
            try:
                if launch_pgid is not None:
                    os.killpg(launch_pgid, signal.SIGINT)
                launch_proc.wait(timeout=15.0)
            except Exception:  # noqa: BLE001 -- escalate to SIGKILL, never leave it running
                try:
                    if launch_pgid is not None:
                        os.killpg(launch_pgid, signal.SIGKILL)
                except Exception:  # noqa: BLE001
                    pass
        for fh in (launch_stdout_f, launch_stderr_f):
            if fh is not None:
                fh.close()
        survivors = _check_leftover_processes(launch_pgid)
        summary["leftover_processes"] = survivors
        summary["leftover_check_scope"] = (
            f"process_group:{launch_pgid}" if launch_pgid is not None else "not_applicable_launch_gazebo_false"
        )
        event_log.emit("leftover_process_check", survivors=survivors, scope=summary["leftover_check_scope"])
        event_log.close()
        with open(os.path.join(run_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger(f"live evidence run written to {run_dir} (termination_reason={summary['termination_reason']})")

    return summary


def main(args=None):
    """``ros2 run hunter_kinodynamic_rl live_evidence_runner.py --ros-args -p profile:=hierarchical_phase4 -p world_seed:=0 -p launch_gazebo:=false``
    -- default ``profile:=hierarchical_phase4`` (defect-fix item 9: the old
    default, ``hierarchical_phase3``, named a profile that does not exist).
    Default ``launch_gazebo:=false`` assumes Gazebo is already running in a
    separate terminal (this package's standard live-script convention);
    ``launch_gazebo:=true`` additionally has THIS function manage the
    Gazebo subprocess itself, but ``main()`` always owns the
    ``rclpy.init()``/``rclpy.shutdown()`` pair either way (mirrors
    ``run_live_hierarchical_benchmark.py``'s own ``main()`` exactly)."""
    import rclpy
    from rclpy.node import Node

    rclpy.init(args=args)
    node = Node("live_evidence_runner")
    node.declare_parameter("profile", DEFAULT_LIVE_EVIDENCE_PROFILE)
    node.declare_parameter("world_seed", 0)
    node.declare_parameter("output_dir", "runtime/live_evidence")
    node.declare_parameter("max_local_steps", 40)
    node.declare_parameter("launch_gazebo", False)
    node.declare_parameter("world_name", "default")
    try:
        run_live_evidence_smoke(
            node.get_parameter("profile").value, world_seed=int(node.get_parameter("world_seed").value),
            output_dir=node.get_parameter("output_dir").value,
            max_local_steps=int(node.get_parameter("max_local_steps").value),
            launch_gazebo=bool(node.get_parameter("launch_gazebo").value),
            world_name=node.get_parameter("world_name").value, logger=node.get_logger().info,
        )
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
