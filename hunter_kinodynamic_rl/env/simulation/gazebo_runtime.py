"""Gazebo Ignition world control: bounded pause/unpause/reset, entity
teleport, physics-step advancement, and post-advance sensor-freshness wait.

Adapted from (not a verbatim copy of -- different class shape/attribute
names) drl_agent's ``env/simulation/gazebo_runtime.py`` +
``gazebo_entity_manager.py::_await_future``. The algorithm is reused
DELIBERATELY UNCHANGED because it encodes a hard-won fix documented in that
package's memory notes: a bare ``rclpy.spin_once(self, ...)`` call inside a
service callback that is ALREADY running under a ``MultiThreadedExecutor``
permanently detaches the node from that executor (``rclpy`` ``Node.executor``
setter side effect) -- every subsequent service call then hangs forever. The
fix is to NEVER call ``spin_once``/``spin_until_future_complete`` from inside
``/reset`` or ``/step``; instead poll ``future.done()`` with ``time.sleep()``
while the OTHER executor worker threads keep servicing the response and the
scan/odom subscription callbacks in the background. See
docs/IMPLEMENTATION_PLAN.md and the ``cf_st_step_executor_hang`` note this mirrors.

``multi_step_advance``/the ``runtime.deterministic_stepping`` opt-in
(section P0-7) revisit an approach `drl_agent` had EXCLUDED for hanging --
but the specific thing that hung there was a bare-``spin_once``-based
``/clock``-only wait with no service-response completion signal, i.e. the
exact same root cause this whole module already worked around for
``pause_world``/``reset_world``/etc. Reusing this module's OWN
already-hang-safe ``_call_world_service``/``_await_future`` machinery for
the ``multi_step`` call (the real completion signal) plus a bounded
``time.sleep()``-based ``/clock`` check purely for verification (never the
sole wait mechanism) is a different, narrower design than what hung before.

section item-1 (Gazebo physics-step reality-check fix): ``multi_step_advance``'s
ORIGINAL verification (``while (observed_dt) < (expected_dt - eps): sleep``)
only ever rejected an UNDER-step (never reaching the expected advance
within ``confirm_timeout_sec``) -- an OVER-step (the connected world's real
``<max_step_size>`` is LARGER than ``runtime.gazebo_max_step_size_sec``
declares, so even ONE physics step already jumps past the expected delta)
made the while-condition false on its very first check and returned
immediately, treating a silently WRONG advance as a successful one. It also
never distinguished "``/clock`` never delivered a message" (silently
returned ``nan`` with only a log warning) from a genuine confirmation, and
did not guard against a ``nan``/backward-moving ``/clock`` reading at all.
Fixed with an EXPLICIT, symmetric tolerance check
(``runtime.physics_step_tolerance_sec``): both under- and over-step are
now rejected, missing/nan/regressing ``/clock`` readings fail fast instead
of degrading to an unverified ``nan`` return, and
:func:`GazeboRuntimeMixin.verify_physics_step_calibration` performs this
SAME check once, for real, against the connected Gazebo world (a live
``multi_step[1]`` call) before :func:`GazeboRuntimeMixin.propagate_state`
ever trusts ``runtime.gazebo_max_step_size_sec`` for a real episode --
closing the gap left by ``environment_node.py``'s ``RUNTIME_FIELDS_FIXED_AT_LAUNCH``
mechanism, which only ever compared a requested override's DECLARED value
against this process's own launch-time DECLARED value (two profile-config
numbers), never against the ACTUAL physics step size the connected
Gazebo process is really running. If this calibration cannot succeed (for
any of the same reasons ``multi_step_advance`` itself now fails fast on),
deterministic evaluation never starts at all -- the exception propagates
out of ``propagate_state``, out of ``/reset`` or ``/step``, exactly like
any other Gazebo service failure this module already treats as fatal
rather than silently degrading.
"""

from __future__ import annotations

import time
from typing import Optional

from rclpy.parameter import Parameter
from ros_gz_interfaces.msg import Entity as GzEntity
from ros_gz_interfaces.srv import ControlWorld, SetEntityPose

from hunter_kinodynamic_rl.env.simulation.gazebo_service_wait import (
    GazeboServiceError, bounded_wait_for_service, compute_physics_step_count,
)


class GazeboRuntimeMixin:
    """Mixed into KinodynamicEnvironmentNode. Reads/writes node instance
    state via ``self`` (world_control/set_entity_pose clients, world_name,
    scan_update_count/odom counters, timeout params) set up in __init__."""

    def _await_future(self, future, timeout: float = 3.0, op: str = "service"):
        deadline = time.time() + timeout
        while not future.done():
            if time.time() > deadline:
                self.get_logger().warn(
                    f"[gazebo] {op}: response future timed out after {timeout:.1f}s")
                return None
            time.sleep(0.05)
        return future.result()

    def _wait_for_srv(self, client, name: str, op: str) -> bool:
        ok, elapsed = bounded_wait_for_service(
            lambda step: client.wait_for_service(timeout_sec=step),
            self._gz_wait_timeout_sec, self._gz_wait_poll_sec,
            on_wait=lambda waited: self.get_logger().warn(
                f"[gazebo] {op}: service {name} not available, waiting "
                f"({waited:.1f}/{self._gz_wait_timeout_sec:.1f}s)..."),
        )
        if not ok:
            self.get_logger().error(
                f"[gazebo] {op}: service {name} UNAVAILABLE after {elapsed:.1f}s")
        return ok

    def _call_world_service(self, client, req, srv_name: str, op: str):
        t0 = time.time()
        if not self._wait_for_srv(client, srv_name, op):
            raise GazeboServiceError(f"{srv_name} ({op}): service unavailable after {time.time() - t0:.1f}s wait")
        try:
            future = client.call_async(req)
            result = self._await_future(future, timeout=self._gz_call_timeout_sec, op=op)
        except Exception as e:
            raise GazeboServiceError(f"{srv_name} ({op}): call raised after {time.time() - t0:.1f}s: {e}") from e
        if result is None:
            raise GazeboServiceError(f"{srv_name} ({op}): no response within {self._gz_call_timeout_sec:.1f}s")
        # section P0-2: completion, SUCCESS, and timeout must all be
        # checked -- a response that arrived (completion) but reports
        # success=false (e.g. an invalid pose, an entity name Gazebo
        # doesn't recognize, a world-name mismatch) means the requested
        # operation did NOT actually happen; silently continuing as if it
        # had would let reset/step proceed against a world state that
        # doesn't match what the caller just asked for.
        if not result.success:
            raise GazeboServiceError(f"{srv_name} ({op}): request completed but reported success=false")
        return result

    def pause_world(self, pause: bool) -> None:
        op = "pause" if pause else "unpause"
        srv_name = f"/world/{self.world_name}/control"
        req = ControlWorld.Request()
        req.world_control.pause = bool(pause)
        self._call_world_service(self.world_control_client, req, srv_name, op)

    def reset_world(self) -> None:
        srv_name = f"/world/{self.world_name}/control"
        req = ControlWorld.Request()
        req.world_control.reset.model_only = True
        req.world_control.pause = True
        self._call_world_service(self.world_control_client, req, srv_name, "reset")

    def set_entity_pose_ignition(self, name: str, x: float, y: float, z: float,
                                  qx: float, qy: float, qz: float, qw: float) -> None:
        srv_name = f"/world/{self.world_name}/set_pose"
        req = SetEntityPose.Request()
        req.entity.name = str(name)
        req.entity.type = GzEntity.MODEL
        req.pose.position.x, req.pose.position.y, req.pose.position.z = float(x), float(y), float(z)
        req.pose.orientation.x, req.pose.orientation.y = float(qx), float(qy)
        req.pose.orientation.z, req.pose.orientation.w = float(qz), float(qw)
        self._call_world_service(self.set_entity_pose_client, req, srv_name, f"set_pose[{name}]")

    def multi_step_advance(self, n_steps: int, expected_dt_sec: float, confirm_timeout_sec: float,
                            tolerance_sec: Optional[float] = None) -> float:
        """Deterministically advance the simulation by EXACTLY ``n_steps``
        physics steps via Ignition's ``ControlWorld.multi_step`` field (each
        step is ``runtime.gazebo_max_step_size_sec`` long, so the total
        sim-time advance is ``n_steps * gazebo_max_step_size_sec ==
        expected_dt_sec`` by construction -- see
        ``gazebo_service_wait.compute_physics_step_count``).

        The COMPLETION signal is the SAME bounded
        ``_call_world_service()``/``_await_future()`` machinery every other
        Gazebo call in this mixin already uses (time.sleep()-based polling,
        NEVER spin_once -- see module docstring): Ignition's world-control
        service blocks server-side until the requested step count has
        actually been applied, so by the time that call returns, physics
        has already advanced. The ``/clock`` check below is a DEFENSIVE
        verification on top of that (catching a silent under/over-step a
        bare ``success=True`` wouldn't reveal), not the primary completion
        mechanism -- so this reuses only bounded ``time.sleep()`` polling
        for it too, never ``spin_once``, for the exact same hang-avoidance
        reason (section P0-7; a prior attempt at a STRICTER per-step
        ``/clock``-only wait mechanism, with no service-response completion
        signal at all, is what previously hung -- see
        ``gazebo_service_wait.py``'s module docstring for that history).

        section item-1 (Gazebo physics-step reality-check fix): the
        confirmation below is now a SYMMETRIC ``abs(observed_dt -
        expected_dt_sec) <= tolerance_sec`` check (``tolerance_sec``
        defaults to ``runtime.physics_step_tolerance_sec``) -- rejecting
        BOTH an under-step (never reaches ``expected_dt_sec`` within
        ``confirm_timeout_sec``) and an over-step (the connected world's
        real physics step is LARGER than declared, so the advance jumps
        PAST ``expected_dt_sec + tolerance_sec`` and stays there). Also
        fails fast -- immediately, never waiting out the full
        ``confirm_timeout_sec`` -- on any of: ``/clock`` having never
        delivered a message at all (``before is None``), a ``nan`` sim-time
        reading (before OR during this call), or ``/clock`` moving
        BACKWARD (a negative observed delta beyond ``tolerance_sec``,
        e.g. an external world reset racing this call). None of these are
        ever silently downgraded to a logged warning + an unverified
        ``nan`` return (the pre-fix behavior for the missing-``/clock``
        case) -- every failure mode here raises ``GazeboServiceError``, so
        a caller that does not catch it never proceeds to trust physics
        that was not actually confirmed.

        Returns the REAL observed sim-time delta, for logging."""
        tolerance_sec = self._physics_step_tolerance_sec if tolerance_sec is None else tolerance_sec
        before = self._latest_sim_time_sec
        if before is None:
            raise GazeboServiceError(
                f"multi_step[{n_steps}]: /clock has never delivered a message -- cannot verify a real physics "
                "advance, refusing to proceed as if it happened"
            )
        if before != before:  # NaN check (before is nan) -- see docstring
            raise GazeboServiceError(f"multi_step[{n_steps}]: /clock's sim time was NaN before stepping")

        srv_name = f"/world/{self.world_name}/control"
        req = ControlWorld.Request()
        req.world_control.multi_step = int(n_steps)
        # section item-1 (Gazebo physics-step reality-check fix) -- a REAL
        # bug found via live Gazebo verification, not merely theorized:
        # ros_gz_interfaces/WorldControl.pause defaults to False, and
        # Ignition's ControlWorld handler treats an explicit `pause=False`
        # accompanying `multi_step` as "leave the world UNPAUSED after
        # applying these steps" -- confirmed live (`ros2 service call
        # .../control "{world_control: {multi_step: 1}}"` vs "{world_control:
        # {pause: true, multi_step: 1}}"` against a live drl_arena.world,
        # max_step_size=0.001s): WITHOUT `pause: true`, a single requested
        # step measured ~45-50ms of REAL sim-time advance (the world kept
        # running in real time, real_time_factor=1.0, until this thread's
        # NEXT /clock read); WITH it, the SAME single-step request advanced
        # by EXACTLY 0.001s and stayed there. Every multi_step_advance call
        # before this fix silently left the world free-running after each
        # deterministic step -- this is exactly the kind of undetected
        # over-step this whole verification mechanism exists to catch, and
        # it DID catch it (the tolerance check above raised on a live
        # over-step before this line existed). Explicitly holding
        # `pause=True` alongside `multi_step` is what actually makes
        # "advance by EXACTLY n_steps physics steps, nothing more" true.
        req.world_control.pause = True
        self._call_world_service(self.world_control_client, req, srv_name, f"multi_step[{n_steps}]")

        deadline = time.time() + confirm_timeout_sec
        observed_dt = self._latest_sim_time_sec - before
        while True:
            current = self._latest_sim_time_sec
            if current is None:
                raise GazeboServiceError(f"multi_step[{n_steps}]: /clock stopped publishing mid-step")
            if current != current:  # NaN check
                raise GazeboServiceError(f"multi_step[{n_steps}]: /clock's sim time became NaN mid-step")
            observed_dt = current - before
            if observed_dt < -tolerance_sec:
                raise GazeboServiceError(
                    f"multi_step[{n_steps}]: /clock moved BACKWARD -- observed sim-time delta "
                    f"{observed_dt:.6f}s is negative beyond tolerance {tolerance_sec:.6f}s (before={before:.6f}s, "
                    f"current={current:.6f}s). Refusing to treat a regressing clock as a valid physics advance."
                )
            if observed_dt >= (expected_dt_sec - tolerance_sec):
                break  # reached (at least) the expected advance -- final symmetric check below decides pass/fail
            if time.time() > deadline:
                break  # never reached within budget -- falls through to the tolerance check as an under-step
            time.sleep(0.01)

        if abs(observed_dt - expected_dt_sec) > tolerance_sec:
            raise GazeboServiceError(
                f"multi_step[{n_steps}]: observed sim-time advance {observed_dt:.6f}s does not match expected "
                f"{expected_dt_sec:.6f}s within tolerance {tolerance_sec:.6f}s ({'under' if observed_dt < expected_dt_sec else 'over'}"
                "-step). This usually means runtime.gazebo_max_step_size_sec does not match the connected "
                "Gazebo world's own SDF <max_step_size>, or /clock did not confirm the expected advance within "
                f"confirm_timeout_sec={confirm_timeout_sec:.2f}s."
            )
        return observed_dt

    def verify_physics_step_calibration(self) -> float:
        """section item-1 (Gazebo physics-step reality-check fix): the ONE
        place this package actually confirms the connected Gazebo world's
        REAL physics step size matches what ``runtime.gazebo_max_step_size_sec``
        DECLARES, rather than trusting the declared value outright. Issues a
        real, minimal (``n_steps=1``) :func:`multi_step_advance` call and
        requires the OBSERVED advance to match within
        ``runtime.physics_step_calibration_tolerance_sec`` -- performed
        explicitly ONCE before :func:`propagate_state` ever trusts
        deterministic stepping for a real episode (see that method and this
        module's own docstring). Raises ``GazeboServiceError`` (propagating
        out of ``propagate_state`` -> ``/reset``/``/step``, exactly like any
        other fatal Gazebo failure this module already refuses to silently
        degrade) if verification cannot succeed for ANY reason --
        deterministic evaluation genuinely never starts without this having
        passed.

        code review (physics-step tolerance/contract bug): this probe's
        ``expected_dt_sec`` is a single ``gazebo_max_step_size_sec`` (e.g.
        0.001s) -- judging it against ``runtime.physics_step_tolerance_sec``
        (sized for a full ~0.1s control-period advance, default 0.005s) let
        a REAL step-size mismatch as large as 0.001s vs 0.002s pass
        undetected (``0.001 <= 0.005``). ``_physics_step_calibration_tolerance_sec``
        is a SEPARATE, single-step-scaled tolerance (config-validated to be
        ``< 0.5 * gazebo_max_step_size_sec`` -- see ``RuntimeConfig``'s
        docstring) that always catches a whole-step-scale mismatch like
        that one, by construction."""
        return self.multi_step_advance(
            n_steps=1, expected_dt_sec=self._gazebo_max_step_size_sec,
            confirm_timeout_sec=self._clock_confirm_timeout_sec,
            tolerance_sec=self._physics_step_calibration_tolerance_sec,
        )

    def _publish_physics_step_calibration_status(self, verified: bool, observed_dt_sec: float) -> None:
        """section item-1: exposes calibration's outcome as ROS parameters
        -- mirrors environment_node.py's own ``architecture_fingerprint_sha256``/
        ``evaluation_contract_fingerprint_sha256`` "computed once, exposed
        for remote verification" pattern (see that module's docstring) --
        so a caller building summary.json/fingerprint metadata for a
        deterministic-stepping run can record whether physics was ACTUALLY
        verified, not merely declared. Requires the two parameters to
        already be declared (see environment_node.py's ``__init__``)."""
        self.set_parameters([
            Parameter("physics_step_calibration_verified", Parameter.Type.BOOL, bool(verified)),
            Parameter("physics_step_calibration_observed_dt_sec", Parameter.Type.DOUBLE, float(observed_dt_sec)),
        ])

    def propagate_state(self, duration_sec: float) -> None:
        """Dispatches to the deterministic multi_step path
        (:meth:`multi_step_advance`) when ``runtime.deterministic_stepping``
        is enabled (section P0-7); otherwise the legacy unpause -> wall-clock
        sleep(duration) -> pause path (default, UNCHANGED) -- the actual
        amount of SIMULATED time a wall-clock sleep(duration) advances
        depends on Gazebo's real-time-factor at that moment (system load,
        physics complexity), confirmed live to make two fresh training runs
        with IDENTICAL seeds diverge in episode outcome from this alone.

        section item-1 (Gazebo physics-step reality-check fix): the FIRST
        time this is called with deterministic stepping enabled,
        :meth:`verify_physics_step_calibration` runs first (and its result
        is cached as ``self._physics_step_calibrated``/
        ``self._physics_step_calibration_observed_dt_sec`` -- see
        environment_node.py's ``__init__``, which also publishes these as
        ROS parameters for a remote client to read back) -- deterministic
        evaluation genuinely never advances physics at all until the
        connected Gazebo world's REAL step size has been confirmed to
        match ``runtime.gazebo_max_step_size_sec``, not merely assumed to."""
        if getattr(self, "_deterministic_stepping", False):
            if not getattr(self, "_physics_step_calibrated", False):
                observed = self.verify_physics_step_calibration()
                self._physics_step_calibrated = True
                self._physics_step_calibration_observed_dt_sec = observed
                self._publish_physics_step_calibration_status(True, observed)
            n_steps = compute_physics_step_count(duration_sec, self._gazebo_max_step_size_sec)
            self.multi_step_advance(n_steps, duration_sec, self._clock_confirm_timeout_sec)
            return
        self.pause_world(False)
        time.sleep(duration_sec)
        self.pause_world(True)

    def wait_for_fresh_sensors(self, prev_scan_updates: int, prev_odom_updates: int,
                                timeout_sec: float = 1.5) -> bool:
        """Bounded poll for a NEW scan AND odom update since
        (prev_scan_updates, prev_odom_updates) -- plain time.sleep(), NOT
        spin_once (see module docstring). Returns True iff both refreshed
        within timeout_sec; logs (does not raise) on partial staleness so a
        transient one-tick lag degrades gracefully rather than hanging."""
        t0 = time.time()
        while (
            (self.scan_update_count <= prev_scan_updates or self.odom_update_count <= prev_odom_updates)
            and (time.time() - t0 < timeout_sec)
        ):
            time.sleep(0.02)
        scan_stale = self.scan_update_count <= prev_scan_updates
        odom_stale = self.odom_update_count <= prev_odom_updates
        if scan_stale or odom_stale:
            self.get_logger().warn(
                f"[gazebo] sensor freshness wait timed out after {timeout_sec:.1f}s: "
                f"scan_stale={scan_stale} odom_stale={odom_stale}")
            return False
        return True
