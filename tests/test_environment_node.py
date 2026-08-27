"""Regression coverage for KinodynamicEnvironmentNode's ROS-node-level
behaviour that cannot be exercised through the pure-function modules alone.

Requires a built ROS workspace (rclpy + drl_agent_interfaces, both only
resolvable after ``colcon build`` + ``source install/setup.bash``) -- this
self-skips cleanly on a bare host checkout, mirroring
``test_train_rl_registry.py``'s pattern (see CLAUDE.md's Testing section).
Every Gazebo-network-touching method is monkeypatched to a fast stub so
these tests never need a live Gazebo/Ignition instance -- only the ROS
graph (rclpy context, service/message types) needs to be importable.
"""

import dataclasses
import time

import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("drl_agent_interfaces")

from drl_agent_interfaces.srv import Reset, Step  # noqa: E402

from hunter_kinodynamic_rl.env.simulation import risk_telemetry as rt  # noqa: E402
from hunter_kinodynamic_rl.env.simulation.environment_node import KinodynamicEnvironmentNode  # noqa: E402


def _stub_gazebo(node: KinodynamicEnvironmentNode, reset_world_delay_sec: float = 0.0) -> None:
    """Replace every method that would otherwise block on a real Gazebo
    service response with a fast, deterministic stand-in -- this test suite
    is about the NODE's own reset/step orchestration logic, not Gazebo
    integration (which is covered by live Docker/Gazebo runs instead, see
    docs/ARCHITECTURE.md)."""
    node.pause_world = lambda paused: None
    node.reset_world = lambda: time.sleep(reset_world_delay_sec)
    node.set_entity_pose_ignition = lambda *a, **kw: None
    node._spawn_scenario_obstacles = lambda scenario, is_fixed: None
    node.propagate_state = lambda duration_sec: None
    node.wait_for_fresh_sensors = lambda prev_scan, prev_odom, timeout_sec: True


@pytest.fixture
def node():
    if not rclpy.ok():
        rclpy.init()
    n = KinodynamicEnvironmentNode()
    yield n
    n.destroy_node()


def test_reset_and_step_run_end_to_end_with_stubbed_gazebo(node):
    _stub_gazebo(node)
    reset_resp = node._on_reset(Reset.Request(), Reset.Response())
    assert len(reset_resp.state) == (node.profile.observation.lidar_bins * node._frame_stack.history_len
                                      + node.profile.observation.robot_state_dim)

    step_req = Step.Request()
    step_req.action = [0.0, 0.5, 0.0]
    step_resp = node._on_step(step_req, Step.Response())
    assert len(step_resp.state) == len(reset_resp.state)


def test_first_step_after_a_slow_reset_is_not_a_false_emergency_stop(node):
    """The core P0-3 regression: /reset's own (real, legitimate) Gazebo
    round-trip routinely exceeds SafetyLimits.max_command_age_sec -- here
    simulated by making the stubbed reset_world() call itself sleep past
    runtime.watchdog_command_timeout_sec. Before the fix, _last_command_time
    was stamped ONLY at the top of _on_reset (before this delay), so the
    very first /step's guard() call would see stale elapsed time and force
    an emergency stop regardless of what the policy actually commanded."""
    delay = node.profile.runtime.watchdog_command_timeout_sec + 0.3
    _stub_gazebo(node, reset_world_delay_sec=delay)
    node._on_reset(Reset.Request(), Reset.Response())
    # No real scan message was ever published in this stubbed test (no
    # executor is spinning to deliver one), so _latest_scan_time would
    # otherwise still sit at its __init__ default (0.0) -- an enormous,
    # unrelated "stale sensor" condition that would ALSO force
    # guard()'s STOP_COMMAND path and mask whether the P0-3 fix (command
    # freshness) actually works. Stamp it fresh here so this test isolates
    # the ONE thing it's regression-testing.
    node._latest_scan_time = time.monotonic()

    captured = {}
    node._risk_pub.publish = lambda msg: captured.setdefault("telemetry", msg)

    step_req = Step.Request()
    step_req.action = [0.0, 1.0, 1.0]  # straight ahead, max commanded speed
    step_resp = node._on_step(step_req, Step.Response())

    assert "telemetry" in captured
    telemetry = rt.decode(list(captured["telemetry"].data))
    assert telemetry.emergency_stop is False
    assert telemetry.guarded_speed_mps > 0.0
    assert telemetry.published_speed_mps > 0.0
    assert step_resp.done is False or step_resp.collision is False


def test_risk_telemetry_reports_the_actually_published_command_not_the_guarded_one(node):
    """The core P0-5 regression: with a command-latency queue active
    (command_delay_steps > 0, e.g. a fixed benchmark's command_latency_sec
    override -- see env/randomization/domain_randomizer.py), the command
    ACTUALLY published to /cmd_vel this tick is an OLDER, already-queued
    one (or an explicit hold-at-stop while the queue fills), not the one
    the guard just computed. Telemetry must report BOTH distinctly."""
    _stub_gazebo(node)
    node._on_reset(Reset.Request(), Reset.Response())
    node._latest_scan_time = time.monotonic()
    node._command_delay_steps = 2
    node._command_delay_queue.clear()

    captured = {}
    node._risk_pub.publish = lambda msg: captured.__setitem__("telemetry", msg)

    def _step(v_ref):
        step_req = Step.Request()
        step_req.action = [0.0, v_ref, 1.0]
        node._on_step(step_req, Step.Response())
        return rt.decode(list(captured["telemetry"].data))

    # Queue not yet full (needs 2 entries) -- published must be held at
    # STOP (0.0) even though the guard computed a real forward speed.
    t1 = _step(1.0)
    assert t1.guarded_speed_mps > 0.0
    assert t1.published_speed_mps == pytest.approx(0.0)

    t2 = _step(1.0)
    assert t2.guarded_speed_mps > 0.0
    assert t2.published_speed_mps == pytest.approx(0.0)

    # Queue now full -- step 3 publishes step 1's guarded command. Command
    # this tick with v_ref normalized action -1.0 -> v_min_mps=0.0 (the
    # action space's default floor, no reverse), a DELIBERATELY DIFFERENT
    # guarded value than step 1's, so a match on published proves it's
    # really lagging by command_delay_steps, not coincidentally equal.
    t3 = _step(-1.0)
    assert t3.guarded_speed_mps == pytest.approx(0.0)   # the NEW command, decoded this tick
    assert t3.published_speed_mps > 0.0                  # step 1's DELAYED command, still queued
    assert t3.guarded_speed_mps != pytest.approx(t3.published_speed_mps)


def test_risk_label_reflects_the_nominal_action_even_when_the_guard_forces_a_stop(node):
    """The core research-direction requirement: the risk label must assess
    the ACTOR'S OWN intended (nominal) command, never a version corrected
    by the safety guard -- otherwise a genuinely dangerous action gets
    retroactively graded "safe" purely because the guard happened to stop
    it, and the actor never learns to avoid choosing it in the first
    place. Forces the guard to STOP via stale sensor/command timestamps
    (both left at their __init__ defaults, 0.0 -- see
    test_first_step_after_a_slow_reset_is_not_a_false_emergency_stop for
    the same staleness mechanism) while commanding straight-ahead-at-max-
    speed into a close static obstacle: the guard's own output
    (guarded_speed_mps) must show a full stop, but the risk assessment
    (collision_within_horizon / risk_target, computed from the NOMINAL
    command) must still report the collision the actor's own choice would
    have caused."""
    node.profile.risk.enabled = True
    _stub_gazebo(node)
    node._on_reset(Reset.Request(), Reset.Response())
    # _robot_pose stays at its __init__ default (0,0,0) throughout this
    # stubbed test (no real /odometry message ever arrives) -- place a
    # static obstacle directly ahead, well within reach of a straight,
    # max-speed nominal command's rollout.
    from hunter_kinodynamic_rl.env.scenarios.procedural_generator import StaticObstacle
    node._scenario = dataclasses.replace(
        node._scenario, static_obstacles=[StaticObstacle(x=1.0, y=0.0, radius=0.3)])
    # Deliberately NOT stamping _latest_scan_time/_last_command_time fresh
    # (unlike the sibling tests above) -- both stay at their stale __init__
    # defaults, so guard()'s freshness checks unconditionally force
    # STOP_COMMAND regardless of what this step commands.

    captured = {}
    node._risk_pub.publish = lambda msg: captured.setdefault("telemetry", msg)

    step_req = Step.Request()
    step_req.action = [0.0, 1.0, -1.0]  # straight ahead, max speed, SHORT L (still >= min_safety_horizon_sec)
    node._on_step(step_req, Step.Response())

    telemetry = rt.decode(list(captured["telemetry"].data))
    # The guard genuinely stopped it -- confirms this test actually
    # exercises the guard-intervenes case, not a no-op.
    assert telemetry.guarded_speed_mps == pytest.approx(0.0)
    assert telemetry.published_speed_mps == pytest.approx(0.0)
    # But the risk label reflects the NOMINAL command's real danger.
    assert telemetry.nominal_speed_mps > 0.0
    assert telemetry.valid is True
    assert telemetry.collision_within_horizon is True
    assert telemetry.risk_target == pytest.approx(1.0)


# ------------------------------------------------------- P0-7: deterministic stepping
def test_propagate_state_uses_legacy_wall_clock_path_by_default(node):
    assert node._deterministic_stepping is False  # every profile's default
    calls = []
    node.pause_world = lambda paused: calls.append(paused)
    node.propagate_state(0.001)
    assert calls == [False, True]  # unpause -> (sleep) -> pause, unchanged


def test_propagate_state_dispatches_to_multi_step_when_enabled(node):
    node._deterministic_stepping = True
    node._gazebo_max_step_size_sec = 0.001
    node._clock_confirm_timeout_sec = 1.0
    node._physics_step_tolerance_sec = 0.0005
    node._latest_sim_time_sec = 10.0

    calls = []

    def _fake_call_world_service(client, req, srv_name, op):
        calls.append((req.world_control.multi_step, op))
        # Simulate /clock reflecting the new time by the time this (stubbed,
        # otherwise-blocking) service call returns -- in production this
        # happens concurrently on another executor thread while THIS thread
        # is inside _await_future's own time.sleep() polling loop. Each
        # physics step advances exactly gazebo_max_step_size_sec, matching
        # real Gazebo -- section item-1's own calibration call (n_steps=1)
        # must see a REALISTIC per-step advance too, not a fixed 0.1s.
        node._latest_sim_time_sec += req.world_control.multi_step * node._gazebo_max_step_size_sec
        return type("Result", (), {"success": True})()

    node._call_world_service = _fake_call_world_service
    node.propagate_state(0.1)

    # section item-1: propagate_state's FIRST deterministic-stepping call
    # runs verify_physics_step_calibration() (a real multi_step[1] probe)
    # before the requested 100-step advance -- see that method's docstring.
    assert calls == [(1, "multi_step[1]"), (100, "multi_step[100]")]
    assert node._physics_step_calibrated is True
    assert node._physics_step_calibration_observed_dt_sec == pytest.approx(0.001)


def test_propagate_state_only_calibrates_once_across_multiple_calls(node):
    """Calibration is a ONE-TIME cost per node lifetime, not re-run on
    every single propagate_state() call -- otherwise every step of every
    episode would pay for an extra multi_step round-trip."""
    node._deterministic_stepping = True
    node._gazebo_max_step_size_sec = 0.001
    node._clock_confirm_timeout_sec = 1.0
    node._physics_step_tolerance_sec = 0.0005
    node._latest_sim_time_sec = 10.0

    calls = []

    def _fake_call_world_service(client, req, srv_name, op):
        calls.append((req.world_control.multi_step, op))
        node._latest_sim_time_sec += req.world_control.multi_step * node._gazebo_max_step_size_sec
        return type("Result", (), {"success": True})()

    node._call_world_service = _fake_call_world_service
    node.propagate_state(0.1)
    node.propagate_state(0.1)

    assert calls == [(1, "multi_step[1]"), (100, "multi_step[100]"), (100, "multi_step[100]")]


def test_propagate_state_never_advances_physics_at_all_when_calibration_fails(node):
    """The core item-1 regression: if the real Gazebo world's physics step
    doesn't match what the profile declares, propagate_state must refuse
    to even ATTEMPT the requested advance -- deterministic evaluation
    genuinely never starts, rather than silently running against
    mismatched physics."""
    node._deterministic_stepping = True
    node._gazebo_max_step_size_sec = 0.001  # declared
    node._clock_confirm_timeout_sec = 0.2
    node._physics_step_tolerance_sec = 0.005
    node._latest_sim_time_sec = 10.0

    calls = []

    def _fake_call_world_service(client, req, srv_name, op):
        calls.append((req.world_control.multi_step, op))
        # The REAL world's step size is 0.01s, not the declared 0.001s --
        # a single calibration step (n_steps=1) overshoots by 10x.
        node._latest_sim_time_sec += req.world_control.multi_step * 0.01
        return type("Result", (), {"success": True})()

    node._call_world_service = _fake_call_world_service

    from hunter_kinodynamic_rl.env.simulation.gazebo_service_wait import GazeboServiceError
    with pytest.raises(GazeboServiceError, match="over-step"):
        node.propagate_state(0.1)

    assert calls == [(1, "multi_step[1]")]  # never even attempted the real 100-step advance
    assert node._physics_step_calibrated is False


def test_multi_step_advance_raises_if_clock_never_confirms_the_expected_advance(node):
    """The core P0-7 safety property: a service call that reports
    success=True but /clock never actually reflects the expected sim-time
    delta must raise (bounded), not silently return as if it had worked."""
    node._call_world_service = lambda client, req, srv_name, op: type("Result", (), {"success": True})()
    node._latest_sim_time_sec = 5.0  # never advances -- /clock stuck

    from hunter_kinodynamic_rl.env.simulation.gazebo_service_wait import GazeboServiceError
    with pytest.raises(GazeboServiceError):
        node.multi_step_advance(n_steps=100, expected_dt_sec=0.1, confirm_timeout_sec=0.1)


# --------------------------------------- item-1 (Gazebo physics-step reality-check fix)
def test_multi_step_advance_requests_pause_true_alongside_multi_step(node):
    """The core LIVE-Gazebo-discovered regression: ros_gz_interfaces
    WorldControl.pause defaults to False, and a live drl_arena.world
    (max_step_size=0.001s, real_time_factor=1.0) confirmed Ignition's
    ControlWorld handler treats an unset/False `pause` field accompanying
    `multi_step` as "leave the world running afterward" -- a single
    requested physics step measured ~45-50ms of real sim-time advance
    (the world free-running until the next read) instead of the
    requested 0.001s, WITHOUT this fix; WITH `pause: true` set explicitly,
    the same single-step request advanced by EXACTLY 0.001s and stayed
    there (see gazebo_runtime.py's own comment at this call site for the
    full live A/B evidence). This is a fast, no-Gazebo-needed regression
    lock: the request payload itself must always carry pause=True."""
    node._latest_sim_time_sec = 5.0
    captured = {}

    def _fake_call_world_service(client, req, srv_name, op):
        captured["pause"] = req.world_control.pause
        node._latest_sim_time_sec = 5.1
        return type("Result", (), {"success": True})()

    node._call_world_service = _fake_call_world_service
    node.multi_step_advance(n_steps=100, expected_dt_sec=0.1, confirm_timeout_sec=0.5)

    assert captured["pause"] is True


def test_multi_step_advance_rejects_an_over_step_expected_01_actual_02(node):
    """The EXACT counterexample the governing instruction names: expected
    0.1s, actual 0.2s (e.g. the connected world's real <max_step_size> is
    2x what the profile declares) -- the pre-fix while-loop condition
    `observed_dt < (expected_dt - eps)` is already FALSE the instant
    observed_dt jumps straight to 0.2s, so the old code returned
    immediately as if the advance were correct. Must now raise."""
    from hunter_kinodynamic_rl.env.simulation.gazebo_service_wait import GazeboServiceError

    node._latest_sim_time_sec = 5.0

    def _fake_call_world_service(client, req, srv_name, op):
        node._latest_sim_time_sec = 5.2  # jumps straight to a 0.2s advance, not the requested 0.1s
        return type("Result", (), {"success": True})()

    node._call_world_service = _fake_call_world_service
    with pytest.raises(GazeboServiceError, match="over-step"):
        node.multi_step_advance(n_steps=100, expected_dt_sec=0.1, confirm_timeout_sec=0.5, tolerance_sec=0.01)


def test_multi_step_advance_rejects_an_under_step_that_stalls_short_of_expected(node):
    """A genuine under-step: /clock advances PART way (e.g. the world
    silently applied fewer/smaller steps than requested) and then stalls,
    never reaching within tolerance of expected_dt_sec before the confirm
    budget runs out."""
    from hunter_kinodynamic_rl.env.simulation.gazebo_service_wait import GazeboServiceError

    node._latest_sim_time_sec = 5.0

    def _fake_call_world_service(client, req, srv_name, op):
        node._latest_sim_time_sec = 5.05  # only half the requested 0.1s advance, then stalls
        return type("Result", (), {"success": True})()

    node._call_world_service = _fake_call_world_service
    with pytest.raises(GazeboServiceError, match="under-step"):
        node.multi_step_advance(n_steps=100, expected_dt_sec=0.1, confirm_timeout_sec=0.1, tolerance_sec=0.01)


def test_multi_step_advance_fails_fast_when_clock_was_never_received_at_all(node):
    """section item-1: missing /clock must raise IMMEDIATELY (never a
    logged warning + an unverified nan return, the pre-fix behavior) --
    and never even issue the multi_step service call, since there is no
    real sim-time reference to verify against."""
    from hunter_kinodynamic_rl.env.simulation.gazebo_service_wait import GazeboServiceError

    node._latest_sim_time_sec = None
    calls = []
    node._call_world_service = lambda client, req, srv_name, op: calls.append(1)

    with pytest.raises(GazeboServiceError, match="never delivered"):
        node.multi_step_advance(n_steps=100, expected_dt_sec=0.1, confirm_timeout_sec=0.5)
    assert calls == []  # never even attempted the service call


def test_multi_step_advance_fails_fast_on_a_nan_clock_reading_before_stepping(node):
    from hunter_kinodynamic_rl.env.simulation.gazebo_service_wait import GazeboServiceError

    node._latest_sim_time_sec = float("nan")
    with pytest.raises(GazeboServiceError, match="NaN"):
        node.multi_step_advance(n_steps=100, expected_dt_sec=0.1, confirm_timeout_sec=0.5)


def test_multi_step_advance_fails_fast_on_a_nan_clock_reading_mid_step(node):
    from hunter_kinodynamic_rl.env.simulation.gazebo_service_wait import GazeboServiceError

    node._latest_sim_time_sec = 5.0

    def _fake_call_world_service(client, req, srv_name, op):
        node._latest_sim_time_sec = float("nan")  # a corrupt/malformed /clock message
        return type("Result", (), {"success": True})()

    node._call_world_service = _fake_call_world_service
    with pytest.raises(GazeboServiceError, match="NaN"):
        node.multi_step_advance(n_steps=100, expected_dt_sec=0.1, confirm_timeout_sec=0.5)


def test_multi_step_advance_fails_fast_when_clock_moves_backward(node):
    """A regressing /clock (e.g. a concurrent world reset racing this
    call) must fail IMMEDIATELY, never silently waited out for the full
    confirm_timeout_sec budget."""
    from hunter_kinodynamic_rl.env.simulation.gazebo_service_wait import GazeboServiceError

    node._latest_sim_time_sec = 5.0

    def _fake_call_world_service(client, req, srv_name, op):
        node._latest_sim_time_sec = 4.5  # moved BACKWARD relative to `before`
        return type("Result", (), {"success": True})()

    node._call_world_service = _fake_call_world_service
    with pytest.raises(GazeboServiceError, match="BACKWARD"):
        node.multi_step_advance(n_steps=100, expected_dt_sec=0.1, confirm_timeout_sec=5.0, tolerance_sec=0.01)


def test_multi_step_advance_accepts_a_reading_exactly_at_the_tolerance_boundary(node):
    """Boundary: abs(observed_dt - expected_dt) == tolerance_sec exactly
    must be ACCEPTED (a `<=`-style boundary, matching this package's own
    convention elsewhere -- e.g. build_brake_onset_snapshot's staleness
    check)."""
    node._latest_sim_time_sec = 5.0

    def _fake_call_world_service(client, req, srv_name, op):
        node._latest_sim_time_sec = 5.0 + 0.1 + 0.01  # exactly expected_dt + tolerance
        return type("Result", (), {"success": True})()

    node._call_world_service = _fake_call_world_service
    observed = node.multi_step_advance(n_steps=100, expected_dt_sec=0.1, confirm_timeout_sec=0.5,
                                        tolerance_sec=0.01)
    assert observed == pytest.approx(0.11)


def test_multi_step_advance_rejects_a_reading_just_past_the_tolerance_boundary(node):
    from hunter_kinodynamic_rl.env.simulation.gazebo_service_wait import GazeboServiceError

    node._latest_sim_time_sec = 5.0

    def _fake_call_world_service(client, req, srv_name, op):
        node._latest_sim_time_sec = 5.0 + 0.1 + 0.0101  # just past expected_dt + tolerance
        return type("Result", (), {"success": True})()

    node._call_world_service = _fake_call_world_service
    with pytest.raises(GazeboServiceError, match="over-step"):
        node.multi_step_advance(n_steps=100, expected_dt_sec=0.1, confirm_timeout_sec=0.5, tolerance_sec=0.01)


def test_verify_physics_step_calibration_uses_the_declared_step_size_and_a_single_step(node):
    """verify_physics_step_calibration must probe with EXACTLY n_steps=1
    against runtime.gazebo_max_step_size_sec -- confirms the wiring, not
    just that SOME multi_step call happens."""
    node._gazebo_max_step_size_sec = 0.003
    node._clock_confirm_timeout_sec = 0.5
    node._physics_step_tolerance_sec = 0.0005
    node._latest_sim_time_sec = 1.0

    calls = []

    def _fake_call_world_service(client, req, srv_name, op):
        calls.append(req.world_control.multi_step)
        node._latest_sim_time_sec += 0.003
        return type("Result", (), {"success": True})()

    node._call_world_service = _fake_call_world_service
    observed = node.verify_physics_step_calibration()

    assert calls == [1]
    assert observed == pytest.approx(0.003)


# ------------------------------- code review: physics-step tolerance/contract bug
def test_verify_physics_step_calibration_catches_a_declared_0_001_vs_real_0_002_world_at_shipped_defaults(node):
    """The exact counterexample this fix responds to: at the SHIPPED
    defaults (gazebo_max_step_size_sec=0.001, the pre-fix
    physics_step_tolerance_sec=0.005), a connected world whose real step
    size was 0.002s (double the declared value) passed calibration
    undetected (abs(0.002-0.001)=0.001 <= 0.005). This test deliberately
    does NOT override either tolerance field -- it exercises the node's own
    launch-time defaults -- and must now raise, because
    verify_physics_step_calibration judges its probe against
    physics_step_calibration_tolerance_sec (default 0.0002s), not the
    looser general-purpose physics_step_tolerance_sec."""
    from hunter_kinodynamic_rl.env.simulation.gazebo_service_wait import GazeboServiceError

    assert node._gazebo_max_step_size_sec == pytest.approx(0.001)  # shipped default, unmodified
    node._latest_sim_time_sec = 5.0

    def _fake_call_world_service(client, req, srv_name, op):
        # The real world's step size is 0.002s, not the declared 0.001s.
        node._latest_sim_time_sec += req.world_control.multi_step * 0.002
        return type("Result", (), {"success": True})()

    node._call_world_service = _fake_call_world_service

    with pytest.raises(GazeboServiceError, match="over-step"):
        node.verify_physics_step_calibration()
    assert node._physics_step_calibrated is False


def test_verify_physics_step_calibration_rejects_real_step_at_exactly_half_the_declared_step(node):
    """The schema's strict half-step bound must make a 2x mismatch in the
    opposite direction detectable too: declared 0.001s versus a real
    0.0005s step.  An inclusive 0.0005s tolerance would accept this exact
    boundary because multi_step_advance intentionally accepts error ==
    tolerance for ordinary runtime jitter."""
    from hunter_kinodynamic_rl.env.simulation.gazebo_service_wait import GazeboServiceError

    node._gazebo_max_step_size_sec = 0.001
    node._physics_step_calibration_tolerance_sec = 0.0004999
    node._latest_sim_time_sec = 0.0

    def _fake_call_world_service(client, req, srv_name, op):
        node._latest_sim_time_sec += req.world_control.multi_step * 0.0005
        return type("Result", (), {"success": True})()

    node._call_world_service = _fake_call_world_service

    with pytest.raises(GazeboServiceError, match="under-step"):
        node.verify_physics_step_calibration()


def test_apply_runtime_cfg_invalidates_a_stale_calibration_when_tolerance_changes(node):
    """code review: a completed calibration must not stay cached as
    'verified' once a LATER _apply_runtime_cfg call (as
    _resolve_evaluation_contract_override issues on every genuine override
    change) changes a calibration-relevant field -- a pass under one
    tolerance is not evidence it would still pass under a different one."""
    import dataclasses as dc

    node._physics_step_calibrated = True
    node._physics_step_calibration_observed_dt_sec = 0.001

    changed_rt_cfg = dc.replace(
        node.profile.runtime,
        physics_step_calibration_tolerance_sec=node.profile.runtime.physics_step_calibration_tolerance_sec / 2,
    )
    node._apply_runtime_cfg(changed_rt_cfg)

    assert node._physics_step_calibrated is False
    assert node._physics_step_calibration_observed_dt_sec is None
    assert node.get_parameter("physics_step_calibration_verified").value is False


def test_apply_runtime_cfg_invalidates_a_stale_calibration_when_clock_confirm_timeout_changes(node):
    node._physics_step_calibrated = True
    node._physics_step_calibration_observed_dt_sec = 0.001

    import dataclasses as dc
    changed_rt_cfg = dc.replace(
        node.profile.runtime,
        clock_confirm_timeout_sec=node.profile.runtime.clock_confirm_timeout_sec + 1.0,
    )
    node._apply_runtime_cfg(changed_rt_cfg)

    assert node._physics_step_calibrated is False


def test_apply_runtime_cfg_keeps_a_valid_calibration_cached_when_calibration_fields_are_unchanged(node):
    """The other half of the same guarantee: re-applying an (otherwise
    different) runtime config that leaves EVERY calibration-relevant field
    identical must NOT force a needless re-calibration -- propagate_state's
    whole point is that calibration is a one-time cost per process
    lifetime (see test_propagate_state_only_calibrates_once_across_multiple_calls)."""
    import dataclasses as dc

    node._physics_step_calibrated = True
    node._physics_step_calibration_observed_dt_sec = 0.001

    unrelated_change_rt_cfg = dc.replace(
        node.profile.runtime, time_delta_sec=node.profile.runtime.time_delta_sec * 2,
    )
    node._apply_runtime_cfg(unrelated_change_rt_cfg)

    assert node._physics_step_calibrated is True
    assert node._physics_step_calibration_observed_dt_sec == pytest.approx(0.001)


def test_apply_runtime_cfg_does_not_touch_calibration_state_on_the_very_first_init_time_call(node):
    """_apply_runtime_cfg runs once from __init__ BEFORE
    self._physics_step_calibrated is ever assigned at all -- the
    invalidation logic must not raise (or otherwise misbehave) on that very
    first call. The fixture already exercised this simply by constructing
    `node` successfully; this test asserts the resulting state explicitly."""
    assert node._physics_step_calibrated is False
    assert node._physics_step_calibration_observed_dt_sec is None


def test_evaluation_contract_override_invalidates_a_stale_physics_step_calibration(node, tmp_path):
    """End-to-end version of the same regression, through the real
    evaluation-contract override path _resolve_evaluation_contract_override
    (called from _on_reset) uses in production -- not just a direct
    _apply_runtime_cfg call."""
    import dataclasses as dc

    from hunter_kinodynamic_rl.evaluation.contract_override import write_evaluation_contract_override

    _stub_gazebo(node)
    node._physics_step_calibrated = True
    node._physics_step_calibration_observed_dt_sec = 0.001

    overridden = dc.replace(
        node.profile,
        runtime=dc.replace(
            node.profile.runtime,
            physics_step_calibration_tolerance_sec=node.profile.runtime.physics_step_calibration_tolerance_sec / 2,
        ),
    )
    override_path = str(tmp_path / "contract.yaml")
    write_evaluation_contract_override(override_path, overridden)
    _set_evaluation_contract_override_path(node, override_path)

    node._on_reset(Reset.Request(), Reset.Response())

    assert node._physics_step_calibrated is False
    assert node._physics_step_calibration_observed_dt_sec is None


# ---------------------------------------------- P0-2: reset velocity convergence
def test_reset_raises_if_residual_velocity_never_converges(node):
    """The core P0-2 regression (memory
    hunter_kinodynamic_rl_reset_velocity_carryover.md, confirmed live):
    SetEntityPose only teleports POSE, never velocity, and Ignition's
    WorldReset.model_only reset does not reliably zero reported /odometry
    velocity either -- a reset must FAIL LOUDLY if residual velocity from
    the previous episode never settles below threshold, never silently
    start a new episode with carried-over momentum."""
    _stub_gazebo(node)
    node.profile.runtime.reset_velocity_threshold_mps = 0.05
    node.profile.runtime.reset_velocity_max_retries = 2

    # _on_reset zeroes _robot_twist to (0.0, 0.0) itself early on (its own
    # "sane stationary fallback" -- see the comment at that assignment) --
    # a pre-set value here would just be overwritten before this test's
    # own logic even runs. Simulate a STUCK residual velocity the same way
    # real /odometry would report it: as a side effect of each settle
    # call (propagate_state), which is what actually runs AFTER that
    # internal zeroing.
    settle_calls = []

    def _stuck_settle(duration_sec):
        settle_calls.append(duration_sec)
        node._robot_twist = (0.5, 0.0)  # never decays

    node.propagate_state = _stuck_settle

    with pytest.raises(RuntimeError, match="residual velocity"):
        node._on_reset(Reset.Request(), Reset.Response())

    # 1 initial settle (inside the try/spawn block) + 2 velocity-retry
    # settles = 3 propagate_state calls (sensor-freshness retries are 0
    # here since wait_for_fresh_sensors is stubbed to always succeed).
    assert len(settle_calls) == 3


def test_reset_succeeds_if_a_velocity_retry_recovers_convergence(node):
    """Retrying must actually be ABLE to succeed, not just exist as dead
    code -- residual velocity that decays below threshold on a LATER
    settle attempt (not the first) must let reset complete normally."""
    _stub_gazebo(node)
    node.profile.runtime.reset_velocity_threshold_mps = 0.05
    node.profile.runtime.reset_velocity_max_retries = 3
    node._robot_twist = (0.5, 0.0)
    # Explicit per-call sequence (not a multiplicative decay, to avoid an
    # accidental off-by-one in how many calls it takes to cross the
    # threshold): stays above threshold for 2 settle calls, converges on
    # the 3rd -- exercises genuine multi-retry recovery, not a 1-shot pass.
    velocities = iter([0.3, 0.1, 0.02])

    def _scripted_settle(duration_sec):
        node._robot_twist = (next(velocities), 0.0)

    node.propagate_state = _scripted_settle

    resp = node._on_reset(Reset.Request(), Reset.Response())
    assert len(resp.state) > 0
    assert abs(node._robot_twist[0]) <= node.profile.runtime.reset_velocity_threshold_mps


def test_reset_does_not_retry_when_velocity_already_below_threshold(node):
    """The common case (no residual velocity at all, e.g. the very first
    episode of a run) must not trigger any velocity-convergence retry."""
    _stub_gazebo(node)
    node.profile.runtime.reset_velocity_threshold_mps = 0.05
    # _robot_twist stays at its __init__ default (0.0, 0.0) in this stub.

    settle_calls = []
    node.propagate_state = lambda duration_sec: settle_calls.append(duration_sec)

    node._on_reset(Reset.Request(), Reset.Response())
    assert len(settle_calls) == 1  # only the initial post-teleport settle


# ------------------------------------------------- P0-5: reset_generation / episode_id
def test_reset_generation_increments_every_reset_and_reaches_published_telemetry(node):
    """reset_generation must increment on EVERY /reset call (counting
    ATTEMPTS, so a stale-cross-episode telemetry message is detectable the
    moment a new attempt begins) and reach the actually-published
    risk_telemetry message, alongside this episode's own seed as
    episode_id."""
    _stub_gazebo(node)
    assert node._reset_generation == 0

    # section P1-5 (shared-interface preservation): Reset.srv's response is
    # intentionally UNMODIFIED (`float32[] state` only, byte-identical to
    # drl_agent's own) -- reset_generation is verified via the node's own
    # internal counter and the risk-telemetry RESET_MARKER broadcast below,
    # never via a Reset.srv response field.
    node._on_reset(Reset.Request(), Reset.Response())
    assert node._reset_generation == 1
    first_episode_seed = node._episode_seed

    captured = {}
    node._risk_pub.publish = lambda msg: captured.setdefault("telemetry", msg)
    step_req = Step.Request()
    step_req.action = [0.0, 0.5, 0.5]
    node._on_step(step_req, Step.Response())
    telemetry = rt.decode(list(captured["telemetry"].data))
    assert telemetry.reset_generation == 1
    assert telemetry.episode_id == first_episode_seed

    node._on_reset(Reset.Request(), Reset.Response())
    assert node._reset_generation == 2  # a SECOND reset must bump it again, not reset to 0 or stay put


def test_invalid_reason_is_features_disabled_when_risk_is_off(node):
    """The common case (risk.enabled=False, e.g. ablation rows A-C) must
    report WHY the label is invalid, not just THAT it is."""
    node.profile.risk.enabled = False
    _stub_gazebo(node)
    node._on_reset(Reset.Request(), Reset.Response())
    node._latest_scan_time = __import__("time").monotonic()

    captured = {}
    node._risk_pub.publish = lambda msg: captured.setdefault("telemetry", msg)
    step_req = Step.Request()
    step_req.action = [0.0, 0.5, 0.5]
    node._on_step(step_req, Step.Response())
    telemetry = rt.decode(list(captured["telemetry"].data))
    assert telemetry.valid is False
    assert telemetry.invalid_reason == int(rt.InvalidReason.FEATURES_DISABLED)


# ------------------------------------------ P0-3: dynamic obstacle motion on sim time
def test_random_waypoint_tick_uses_the_real_observed_sim_time_delta_not_nominal_dt(node):
    """RANDOM_WAYPOINT is the one motion pattern that's a stateful
    integration (not reconstructible from spawn spec + elapsed time like
    the other four) -- it must tick on the REAL observed per-tick sim-time
    delta, not the nominal config time_delta_sec, to genuinely run "on a
    simulation-time basis" under the default (non-deterministic,
    wall-clock-sleep) stepping path where a real tick can take
    meaningfully more or less than the nominal duration."""
    from hunter_kinodynamic_rl.env.humans.dynamic_obstacle_motion import MotionPattern, RandomWaypointState
    from hunter_kinodynamic_rl.env.scenarios.procedural_generator import DynamicObstacleSpec

    _stub_gazebo(node)
    node._on_reset(Reset.Request(), Reset.Response())

    waypoint = RandomWaypointState(x=0.0, y=0.0, speed_mps=1.0, half_extent_m=5.0, seed=0)
    tick_calls = []
    real_tick = waypoint.tick

    def _recording_tick(dt_sec):
        tick_calls.append(dt_sec)
        return real_tick(dt_sec)

    waypoint.tick = _recording_tick
    node._dynamic_obstacles = [{
        "spec": DynamicObstacleSpec(x0=0.0, y0=0.0, vx=0.0, vy=0.0, radius=0.3),
        "spec0": DynamicObstacleSpec(x0=0.0, y0=0.0, vx=0.0, vy=0.0, radius=0.3),
        "pattern": MotionPattern.RANDOM_WAYPOINT, "waypoint": waypoint,
    }]
    node._spawned_dynamic_names = ["hkrl_dynamic_0"]  # set_entity_pose_ignition is already stubbed to a no-op

    # Simulate real /clock readings that DIFFER from the nominal
    # time_delta_sec (0.1s default) -- a real tick that took 0.13s of sim
    # time, not the nominal 0.1s.
    node._episode_start_sim_time_sec = 100.0
    node._latest_sim_time_sec = 100.0  # first tick: no PREVIOUS reading yet -> falls back to nominal dt_sec
    node._tick_dynamic_obstacles(node.time_delta)
    node._latest_sim_time_sec = 100.13  # second tick: real observed delta = 0.13s, not nominal 0.1s
    node._tick_dynamic_obstacles(node.time_delta)

    assert len(tick_calls) == 2
    assert tick_calls[0] == pytest.approx(node.time_delta)  # first tick: no prior reading -> nominal fallback
    assert tick_calls[1] == pytest.approx(0.13)  # second tick: REAL observed delta, not nominal 0.1


# --------------------------------------------------- P0-9: sensor-freshness handling
def test_reset_raises_after_exhausting_retries_when_sensors_never_refresh(node):
    """The core P0-9 regression for /reset: a stale-sensor reset must FAIL
    LOUDLY after its bounded retry budget, never silently hand the trainer
    an initial observation that isn't actually confirmed post-reset."""
    _stub_gazebo(node)
    node.wait_for_fresh_sensors = lambda prev_scan, prev_odom, timeout_sec: False
    node.profile.runtime.sensor_freshness_max_reset_retries = 2

    propagate_calls = []
    node.propagate_state = lambda duration_sec: propagate_calls.append(duration_sec)

    with pytest.raises(RuntimeError, match="sensors did not refresh"):
        node._on_reset(Reset.Request(), Reset.Response())

    # 1 initial settle (inside the try/spawn block) + 2 retries = 3 propagate_state calls.
    assert len(propagate_calls) == 3


def test_reset_succeeds_if_a_retry_recovers_freshness(node):
    """Retrying must actually be ABLE to succeed, not just exist as dead
    code -- a transient one-tick staleness right after teleport/spawn is
    the exact case retrying is FOR."""
    _stub_gazebo(node)
    node.profile.runtime.sensor_freshness_max_reset_retries = 2
    attempts = {"n": 0}

    def _flaky_fresh(prev_scan, prev_odom, timeout_sec):
        attempts["n"] += 1
        return attempts["n"] >= 2  # fails once, then recovers

    node.wait_for_fresh_sensors = _flaky_fresh

    resp = node._on_reset(Reset.Request(), Reset.Response())
    assert len(resp.state) > 0
    assert attempts["n"] == 2


def test_step_truncates_and_flags_sensor_stale_when_sensors_never_refresh(node):
    """The core P0-9 regression for /step: a stale-sensor step must
    TRUNCATE the episode (done=True) and flag it in telemetry -- never
    silently return as if the transition were an ordinary continuing one."""
    _stub_gazebo(node)
    node._on_reset(Reset.Request(), Reset.Response())
    node._latest_scan_time = time.monotonic()
    node.wait_for_fresh_sensors = lambda prev_scan, prev_odom, timeout_sec: False

    captured = {}
    node._risk_pub.publish = lambda msg: captured.__setitem__("telemetry", msg)

    step_req = Step.Request()
    step_req.action = [0.0, 0.5, 0.5]
    step_resp = node._on_step(step_req, Step.Response())

    assert step_resp.done is True
    assert step_resp.collision is False
    assert step_resp.target is False
    telemetry = rt.decode(list(captured["telemetry"].data))
    assert telemetry.sensor_stale is True


def test_step_does_not_flag_sensor_stale_when_sensors_refresh_normally(node):
    _stub_gazebo(node)
    node._on_reset(Reset.Request(), Reset.Response())
    node._latest_scan_time = time.monotonic()

    captured = {}
    node._risk_pub.publish = lambda msg: captured.__setitem__("telemetry", msg)

    step_req = Step.Request()
    step_req.action = [0.0, 0.5, 0.5]
    node._on_step(step_req, Step.Response())

    telemetry = rt.decode(list(captured["telemetry"].data))
    assert telemetry.sensor_stale is False


# --------------------------------------------------- P1-10: domain randomization
def _enable_domain_randomization(node, **range_overrides):
    dr = dataclasses.replace(node.profile.domain_randomization, enabled=True, **range_overrides)
    node.profile = dataclasses.replace(node.profile, domain_randomization=dr)


def test_domain_randomization_command_latency_is_no_longer_silently_discarded(node):
    """The core P1-10 regression: command_latency_sec WAS sampled every
    episode by sample_draw() but silently discarded (_command_delay_steps
    hardcoded to 0) on the procedural-randomization path -- only fixed-
    benchmark overrides ever reached the delay queue."""
    _stub_gazebo(node)
    _enable_domain_randomization(node, command_latency_sec_range=[0.3, 0.3])  # fixed at 0.3s
    node._on_reset(Reset.Request(), Reset.Response())
    # time_delta defaults to 0.1s -> 0.3s / 0.1s = exactly 3 delay steps.
    assert node._command_delay_steps == 3


def test_domain_randomization_disabled_keeps_zero_command_delay(node):
    _stub_gazebo(node)
    assert node.profile.domain_randomization.enabled is False
    node._on_reset(Reset.Request(), Reset.Response())
    assert node._command_delay_steps == 0


def test_domain_randomization_never_applies_to_a_fixed_benchmark_scenario(node, tmp_path):
    """TRAIN-ONLY, by construction: a fixed-benchmark /reset must NEVER
    pick up a procedural domain-randomization draw, even when the profile
    has domain_randomization.enabled=True (section P1-10's explicit
    requirement)."""
    _stub_gazebo(node)
    _enable_domain_randomization(node, mass_scale_range=[2.0, 2.0], command_latency_sec_range=[0.5, 0.5])

    scenario_path = tmp_path / "fixed_scenario.yaml"
    scenario_path.write_text(
        "scenario_id: p1_10_train_only_check\nseed: 1\n"
        "start: {x: 0.0, y: 0.0, yaw: 0.0}\ngoal: {x: 2.0, y: 0.0}\n"
        "static_obstacles: []\nmoving_obstacles: []\n"
    )
    node.set_parameters([rclpy.parameter.Parameter(
        "scenario_override_path", rclpy.parameter.Parameter.Type.STRING, str(scenario_path))])

    node._on_reset(Reset.Request(), Reset.Response())

    assert node._active_robot_config.mass_kg == pytest.approx(node.profile.robot.mass_kg)  # NOT scaled by 2.0
    assert node._command_delay_steps == 0  # NOT the 0.5s/0.1s=5 the draw would have set
    assert node._domain_rand_step_rng is None


def test_observation_noise_never_leaks_into_ground_truth_collision_detection(node):
    """Domain-randomization LiDAR noise must perturb ONLY what the agent
    observes -- _current_scan_states() (used for collision/min_obstacle_dist)
    must stay exactly the untouched ground truth regardless."""
    _stub_gazebo(node)
    _enable_domain_randomization(node, lidar_range_noise_std_m_range=[5.0, 5.0])  # huge, guaranteed-visible noise
    node._on_reset(Reset.Request(), Reset.Response())

    ground_truth_before, _ = node._current_scan_states()
    observed = node._observation_obs_state()
    ground_truth_after, _ = node._current_scan_states()

    import numpy as np
    np.testing.assert_array_equal(ground_truth_before, ground_truth_after)
    assert not np.array_equal(observed, ground_truth_after)


# ------------------------------------------- reset frame-stack noise dedup (requirement 3)
def _stacked_frames(node):
    """Splits node._frame_stack.stacked() (current-first, concatenated)
    back into its individual history_len frames, in the same current-first
    order."""
    import numpy as np
    stacked = node._frame_stack.stacked()
    frame_dim = node._frame_stack.frame_dim
    return [stacked[i * frame_dim:(i + 1) * frame_dim] for i in range(node._frame_stack.history_len)]


def test_reset_fills_the_frame_stack_with_a_single_identical_noisy_frame_domain_randomization(node):
    """The requirement-3 regression: a NAIVE reset that samples LiDAR noise
    twice (once to seed the frame stack, once more inside a
    _build_state_vector() call) would leave the NEWEST slot different from
    every older slot even though this is t=0 -- FrameStack's own warm-start
    contract requires every slot to start identical. Domain-randomization
    LiDAR noise is large/guaranteed-visible here so any double-sampling
    would show up as a real difference, not by chance agreement."""
    import numpy as np
    _stub_gazebo(node)
    _enable_domain_randomization(node, lidar_range_noise_std_m_range=[5.0, 5.0])
    node._on_reset(Reset.Request(), Reset.Response())

    frames = _stacked_frames(node)
    for f in frames[1:]:
        np.testing.assert_array_equal(frames[0], f)


def test_reset_fills_the_frame_stack_with_a_single_identical_noisy_frame_sensor_noise(node):
    from hunter_kinodynamic_rl.config.schema import SensorNoiseConfig
    import numpy as np
    _stub_gazebo(node)
    node.profile = dataclasses.replace(node.profile, sensor_noise=SensorNoiseConfig(
        enabled=True, lidar_range_noise_std_m=5.0))
    node._on_reset(Reset.Request(), Reset.Response())

    frames = _stacked_frames(node)
    for f in frames[1:]:
        np.testing.assert_array_equal(frames[0], f)


def test_reset_does_not_advance_the_sensor_noise_rng_a_second_time(node):
    """Direct RNG-consumption proof (rather than inferring it from frame
    equality alone): the sensor_noise RNG stream left over after reset must
    match drawing the LiDAR-noise sample exactly ONCE from a freshly
    reset_state()'d stream with the same episode seed -- never twice."""
    from hunter_kinodynamic_rl.config.schema import SensorNoiseConfig
    from hunter_kinodynamic_rl.env.simulation import sensor_noise as sn
    _stub_gazebo(node)
    node.profile = dataclasses.replace(node.profile, sensor_noise=SensorNoiseConfig(
        enabled=True, lidar_range_noise_std_m=0.05))
    node._on_reset(Reset.Request(), Reset.Response())

    rng_state_after_reset = node._sensor_noise_state.rng.get_state()  # already advanced once by reset
    expected_state = sn.reset_state(node._episode_seed, node.profile.sensor_noise)
    ground_truth, _ = node._current_scan_states()
    sn.apply_lidar_noise(expected_state, node.profile.sensor_noise, ground_truth, node._max_range())
    expected_after_one_draw = expected_state.rng.get_state()

    # numpy RandomState.get_state() returns ('MT19937', keys[624], pos, ...)
    # -- index 2 is the position counter into the Mersenne Twister buffer,
    # the simplest direct signal of "how many numbers has this stream
    # produced so far".
    assert rng_state_after_reset[2] == expected_after_one_draw[2]


# --------------------------------------------------------------- requirement 5
def _capture_diagnostics(node):
    from hunter_kinodynamic_rl.env.simulation import sensor_diagnostics as sd
    captured = {}
    node._sensor_diag_pub.publish = lambda msg: captured.__setitem__("diag", msg)
    return captured, sd


def test_reset_publishes_valid_sensor_diagnostics_at_step_id_zero(node):
    _stub_gazebo(node)
    captured, sd = _capture_diagnostics(node)
    node._on_reset(Reset.Request(), Reset.Response())
    diag = sd.decode(list(captured["diag"].data))
    assert diag.valid is True
    assert diag.step_id == 0
    assert diag.reset_generation == node._reset_generation
    assert diag.episode_id == node._episode_seed


def test_sensor_diagnostics_gt_and_noisy_differ_when_sensor_noise_enabled(node):
    from hunter_kinodynamic_rl.config.schema import SensorNoiseConfig
    _stub_gazebo(node)
    node.profile = dataclasses.replace(node.profile, sensor_noise=SensorNoiseConfig(
        enabled=True, localization_xy_noise_std_m=5.0, lidar_range_noise_std_m=5.0))
    captured, sd = _capture_diagnostics(node)
    node._on_reset(Reset.Request(), Reset.Response())
    diag = sd.decode(list(captured["diag"].data))

    assert diag.gt_x != pytest.approx(diag.noisy_x)  # 5.0m std -- overwhelmingly unlikely to coincide
    assert diag.lidar_perturbation_mean_m > 0.0


def test_sensor_diagnostics_gt_and_noisy_are_identical_when_sensor_noise_disabled(node):
    _stub_gazebo(node)
    assert node.profile.sensor_noise.enabled is False
    captured, sd = _capture_diagnostics(node)
    node._on_reset(Reset.Request(), Reset.Response())
    diag = sd.decode(list(captured["diag"].data))

    assert diag.gt_x == pytest.approx(diag.noisy_x)
    assert diag.gt_y == pytest.approx(diag.noisy_y)
    assert diag.gt_yaw == pytest.approx(diag.noisy_yaw)
    assert diag.gt_v_mps == pytest.approx(diag.noisy_v_mps)
    assert diag.lidar_perturbation_mean_m == pytest.approx(0.0)
    assert diag.lidar_perturbation_max_m == pytest.approx(0.0)
    assert diag.drift_x_m == 0.0 and diag.drift_y_m == 0.0 and diag.drift_yaw_rad == 0.0


def test_sensor_diagnostics_reports_the_current_ou_drift_state(node):
    from hunter_kinodynamic_rl.config.schema import SensorNoiseConfig
    _stub_gazebo(node)
    node.profile = dataclasses.replace(node.profile, sensor_noise=SensorNoiseConfig(
        enabled=True, localization_drift_theta=0.5, localization_drift_xy_sigma_m=1.0))
    captured, sd = _capture_diagnostics(node)
    node._on_reset(Reset.Request(), Reset.Response())
    diag_reset = sd.decode(list(captured["diag"].data))
    assert diag_reset.drift_x_m == 0.0  # drift starts at exactly zero every episode

    step_req = Step.Request()
    step_req.action = [0.0, 0.5, 0.0]
    node._on_step(step_req, Step.Response())
    diag_step = sd.decode(list(captured["diag"].data))
    assert diag_step.drift_x_m == pytest.approx(node._sensor_noise_state.drift_x)


def test_sensor_diagnostics_step_id_and_reset_generation_align_with_risk_telemetry(node):
    """The core requirement-5 sync claim: a consumer must be able to pair a
    risk_telemetry message and a sensor_diagnostics message for the SAME
    tick purely by (reset_generation, step_id)."""
    _stub_gazebo(node)
    risk_captured = {}
    node._risk_pub.publish = lambda msg: risk_captured.__setitem__("t", msg)
    diag_captured, sd = _capture_diagnostics(node)

    node._on_reset(Reset.Request(), Reset.Response())
    step_req = Step.Request()
    step_req.action = [0.0, 0.5, 0.0]
    node._on_step(step_req, Step.Response())

    telemetry = rt.decode(list(risk_captured["t"].data))
    diag = sd.decode(list(diag_captured["diag"].data))
    assert telemetry.step_id == diag.step_id
    assert telemetry.reset_generation == diag.reset_generation
    assert telemetry.episode_id == diag.episode_id


def test_sensor_diagnostics_lidar_beam_count_matches_configured_lidar_bins(node):
    _stub_gazebo(node)
    captured, sd = _capture_diagnostics(node)
    node._on_reset(Reset.Request(), Reset.Response())
    diag = sd.decode(list(captured["diag"].data))
    assert diag.lidar_beam_count == node.profile.observation.lidar_bins


def test_sensor_diagnostics_uses_the_snapshotted_ground_truth_not_a_fresh_scan_query(node):
    """Regression for the GT/noisy race condition: a scan callback firing
    (on another executor thread) between _observation_obs_state()'s ground-
    truth read and _compute_and_publish_sensor_diagnostics's own comparison
    must NOT change what diagnostics reports -- it must always compare
    against self._last_gt_obs_state, the EXACT same frame the noisy
    observation was derived from this tick, never a fresh
    self._current_scan_states() call. Before this fix, diagnostics called
    _current_scan_states() again itself, so a newer scan delivered in
    between would be misattributed to noise perturbation even with
    sensor_noise fully disabled."""
    import numpy as np

    _stub_gazebo(node)
    bins = node.profile.observation.lidar_bins
    node._last_gt_obs_state = np.full(bins, 3.0, dtype=np.float32)
    node._last_obs_state = np.full(bins, 3.0, dtype=np.float32)  # no noise applied -> identical to gt

    # A scan callback delivering a NEW, unrelated scan after the snapshot
    # above was taken but before diagnostics runs -- if diagnostics called
    # _current_scan_states() again, it would see THIS instead.
    sneaky_scan = np.full(bins, 9.0, dtype=np.float32)
    node._current_scan_states = lambda: (sneaky_scan, sneaky_scan)

    captured, sd = _capture_diagnostics(node)
    node._compute_and_publish_sensor_diagnostics(
        gt_x=0.0, gt_y=0.0, gt_yaw=0.0, gt_v=0.0, gt_yaw_rate=0.0, gt_steering=0.0,
        noisy_x=0.0, noisy_y=0.0, noisy_yaw=0.0, noisy_v=0.0, noisy_yaw_rate=0.0, noisy_steering=0.0,
    )
    diag = sd.decode(list(captured["diag"].data))
    assert diag.valid is True
    assert diag.lidar_perturbation_mean_m == pytest.approx(0.0)
    assert diag.lidar_perturbation_max_m == pytest.approx(0.0)
    assert diag.lidar_dropout_count == 0


def test_privileged_obstacle_ground_truth_never_leaks_into_the_policy_observation(node):
    """The core P0-5 leak-prevention property: the STATE VECTOR returned to
    the policy (_build_state_vector) must be COMPLETELY INSENSITIVE to the
    scenario's privileged obstacle ground truth (static_obstacles /
    dynamic obstacle specs) -- only risk_telemetry (a SEPARATE, privileged-
    only channel never fed back into the policy's own observation, see
    ARCHITECTURE.md's "Training vs. inference" section) may use it. Proven
    BEHAVIORALLY, not just by code inspection: two scenarios with
    IDENTICAL robot pose/goal/prev-action/LiDAR reading but WILDLY
    different obstacle configurations (none vs. several placed directly on
    top of the robot) must produce a BYTE-IDENTICAL observation."""
    import dataclasses

    import numpy as np

    from hunter_kinodynamic_rl.env.humans.dynamic_obstacle_motion import MotionPattern
    from hunter_kinodynamic_rl.env.scenarios.procedural_generator import DynamicObstacleSpec, StaticObstacle

    _stub_gazebo(node)
    node._on_reset(Reset.Request(), Reset.Response())
    # Freeze everything the observation legitimately DOES depend on so only
    # the obstacle ground truth varies between the two builds below.
    node._robot_pose = (1.5, -2.0, 0.3)
    node._robot_twist = (0.4, 0.1)
    node._center_steering = 0.05
    node._prev_action = [0.2, -0.1, 0.6]

    node._scenario = dataclasses.replace(node._scenario, static_obstacles=[])
    node._dynamic_obstacles = []
    state_no_obstacles = node._build_state_vector()

    node._scenario = dataclasses.replace(
        node._scenario,
        static_obstacles=[
            StaticObstacle(x=node._robot_pose[0], y=node._robot_pose[1], radius=0.3),
            StaticObstacle(x=node._robot_pose[0] + 0.1, y=node._robot_pose[1] + 0.1, radius=0.5),
        ],
    )
    node._dynamic_obstacles = [{
        "spec": DynamicObstacleSpec(x0=node._robot_pose[0], y0=node._robot_pose[1], vx=5.0, vy=5.0, radius=0.4),
        "spec0": DynamicObstacleSpec(x0=node._robot_pose[0], y0=node._robot_pose[1], vx=5.0, vy=5.0, radius=0.4),
        "pattern": MotionPattern.HEAD_ON, "waypoint": None,
    }]
    state_with_obstacles_on_top_of_robot = node._build_state_vector()

    np.testing.assert_array_equal(state_no_obstacles, state_with_obstacles_on_top_of_robot)


def test_steering_delay_filters_the_real_published_command(node):
    """steering_delay_sec must affect the REAL command sent to Gazebo (a
    dynamics effect, applied to safe_command BEFORE publish), not just the
    observation -- confirmed via the internal lag-filter state moving only
    PARTWAY toward the requested steering on the episode's first step
    (filter state starts at 0.0 every reset)."""
    _stub_gazebo(node)
    _enable_domain_randomization(node, steering_delay_sec_range=[1.0, 1.0])  # long lag relative to time_delta=0.1s
    node._on_reset(Reset.Request(), Reset.Response())
    node._latest_scan_time = time.monotonic()
    assert node._filtered_steering_rad == pytest.approx(0.0)

    step_req = Step.Request()
    step_req.action = [1.0, 0.5, 0.5]  # max steering this tick -> a real, nonzero requested angle
    node._on_step(step_req, Step.Response())

    max_steering = node._active_robot_config.steering_limit_rad
    # alpha = 0.1/1.0 = 0.1 -> filtered_steering only moves ~10% of the way
    # from 0 toward the commanded angle this tick, never reaching it.
    assert 0.0 < abs(node._filtered_steering_rad) < 0.3 * max_steering


def test_steering_delay_zero_is_byte_identical_to_no_filtering(node):
    _stub_gazebo(node)
    assert node.profile.domain_randomization.enabled is False
    node._on_reset(Reset.Request(), Reset.Response())
    node._latest_scan_time = time.monotonic()

    step_req = Step.Request()
    step_req.action = [1.0, 0.5, 0.5]
    node._on_step(step_req, Step.Response())

    # No filtering active -> _filtered_steering_rad is never touched, stays
    # at its reset value (0.0) regardless of what was commanded.
    assert node._filtered_steering_rad == pytest.approx(0.0)


def test_velocity_response_rate_limit_filters_the_real_published_command(node):
    """code review: friction_scale/velocity_response_scale must affect the
    REAL command sent to Gazebo (applied to the NOMINAL command before the
    safety guard), not just the offline dynamics-rollout risk model --
    confirmed via the internal rate-limiter state moving only PARTWAY
    toward the requested speed on the episode's first step (filter state
    starts at 0.0 every reset, and a heavily-scaled-down
    velocity_response_scale makes accel_limit_mps2 tiny relative to a
    single 0.1s tick)."""
    _stub_gazebo(node)
    _enable_domain_randomization(node, velocity_response_scale_range=[0.02, 0.02])
    node._on_reset(Reset.Request(), Reset.Response())
    node._latest_scan_time = time.monotonic()
    node._last_command_time = time.monotonic()
    assert node._filtered_speed_mps == pytest.approx(0.0)

    step_req = Step.Request()
    step_req.action = [0.0, 1.0, 0.5]  # max v_ref this tick -> a real, nonzero requested speed
    node._on_step(step_req, Step.Response())

    max_speed = node._active_robot_config.max_forward_speed_mps
    # accel_limit_mps2 scaled to ~2% of its base value -> at most a tiny
    # fraction of max_speed can be reached in one 0.1s tick.
    assert 0.0 < node._filtered_speed_mps < 0.3 * max_speed


def test_velocity_response_rate_limit_disabled_is_byte_identical(node):
    _stub_gazebo(node)
    assert node.profile.domain_randomization.enabled is False
    node._on_reset(Reset.Request(), Reset.Response())
    node._latest_scan_time = time.monotonic()
    node._last_command_time = time.monotonic()

    step_req = Step.Request()
    step_req.action = [0.0, 1.0, 0.5]
    node._on_step(step_req, Step.Response())

    # No randomization draw active (identity: friction_scale ==
    # velocity_response_scale == 1.0) -> the rate limiter is a passthrough,
    # so the filtered state exactly tracks the raw commanded speed, never
    # lagging behind it.
    max_speed = node._active_robot_config.max_forward_speed_mps
    assert node._filtered_speed_mps == pytest.approx(max_speed, abs=1e-6)


def test_velocity_response_rate_limit_never_softens_a_guard_forced_emergency_stop(node):
    """Safety-critical invariant: the friction/velocity-response rate
    limiter must NEVER delay or soften a guard-forced emergency stop -- it
    runs on the NOMINAL command feeding INTO guard(), so guard() can still
    override with an immediate zero regardless of what the rate limiter
    would otherwise have allowed through. Forces the guard to stop via
    stale sensor/command timestamps (mirrors
    test_risk_label_reflects_the_nominal_action_even_when_the_guard_forces_a_stop)
    while a large velocity_response_scale randomization draw is active."""
    _stub_gazebo(node)
    _enable_domain_randomization(node, velocity_response_scale_range=[0.02, 0.02])
    node._on_reset(Reset.Request(), Reset.Response())
    # Deliberately NOT stamping _latest_scan_time/_last_command_time fresh --
    # both stay at their stale __init__ defaults, so guard()'s freshness
    # checks unconditionally force STOP_COMMAND regardless of this step's
    # nominal (rate-limited) command.

    captured = {}
    node._risk_pub.publish = lambda msg: captured.setdefault("telemetry", msg)

    step_req = Step.Request()
    step_req.action = [0.0, 1.0, 0.5]  # max speed requested
    node._on_step(step_req, Step.Response())

    telemetry = rt.decode(list(captured["telemetry"].data))
    assert telemetry.guarded_speed_mps == pytest.approx(0.0)
    assert telemetry.published_speed_mps == pytest.approx(0.0)
    # code review (emergency_stop metric pollution): the GUARD genuinely
    # intervened here (stale sensors force STOP_COMMAND), so emergency_stop
    # must still fire -- this is the "domain rand enabled AND guard acts"
    # case, distinct from test_emergency_stop_is_not_triggered_by_the_plant_speed_limiter_alone
    # below (domain rand enabled, guard does NOT act).
    assert telemetry.emergency_stop is True
    assert telemetry.guard_intervened is True


def test_emergency_stop_is_not_triggered_by_the_plant_speed_limiter_alone(node):
    """The core P1-10 regression this fix closes: with NO obstacle nearby
    and FRESH sensors/command (the guard has nothing to intervene on), a
    heavily-scaled-down velocity_response_scale draw makes the PLANT speed
    limiter ramp from 0 toward the requested max speed -- on the episode's
    FIRST step this genuinely produces a near-zero plant/guarded/published
    speed, purely from realistic ramp-up, with ZERO guard involvement.
    Before this fix, emergency_stop compared the RAW nominal (pre-plant-
    limiter) speed against the guarded speed, so this exact scenario
    misfired as a false "emergency stop". After the fix it must not."""
    _stub_gazebo(node)
    _enable_domain_randomization(node, velocity_response_scale_range=[0.02, 0.02])
    node._on_reset(Reset.Request(), Reset.Response())
    node._latest_scan_time = time.monotonic()
    node._last_command_time = time.monotonic()

    captured = {}
    node._risk_pub.publish = lambda msg: captured.setdefault("telemetry", msg)

    step_req = Step.Request()
    step_req.action = [0.0, 1.0, 0.5]  # max speed requested -> real nonzero nominal command
    node._on_step(step_req, Step.Response())

    telemetry = rt.decode(list(captured["telemetry"].data))
    # Sanity: the plant limiter actually did something and produced a
    # near-zero guarded/published speed this tick (otherwise this test
    # isn't exercising the scenario it claims to).
    assert telemetry.plant_limited is True
    assert telemetry.nominal_speed_mps > 0.5
    assert telemetry.guarded_speed_mps < 0.3 * telemetry.nominal_speed_mps
    # The actual fix: no guard activity, so neither guard_intervened nor
    # emergency_stop may fire, no matter how small the plant-limited speed is.
    assert telemetry.guard_intervened is False
    assert telemetry.emergency_stop is False


def test_emergency_stop_and_plant_limited_are_both_false_when_domain_randomization_disabled(node):
    """Baseline (no domain randomization): plant_command is always
    identical to vehicle_command (see environment_node.py's
    `plant_command is vehicle_command` no-op branch), so plant_limited must
    always read False, and emergency_stop must depend ONLY on the guard --
    exactly the pre-existing (pre-P1-10) behavior."""
    _stub_gazebo(node)
    assert node.profile.domain_randomization.enabled is False
    node._on_reset(Reset.Request(), Reset.Response())
    node._latest_scan_time = time.monotonic()
    node._last_command_time = time.monotonic()

    captured = {}
    node._risk_pub.publish = lambda msg: captured.setdefault("telemetry", msg)

    step_req = Step.Request()
    step_req.action = [0.0, 1.0, 0.5]
    node._on_step(step_req, Step.Response())

    telemetry = rt.decode(list(captured["telemetry"].data))
    assert telemetry.plant_limited is False
    assert telemetry.guard_intervened is False
    assert telemetry.emergency_stop is False


# --------------------------------------------------- item-1/item-2 (round 2): evaluation contract override
def _set_evaluation_contract_override_path(node, path: str) -> None:
    node.set_parameters([rclpy.parameter.Parameter(
        "evaluation_contract_override_path", rclpy.parameter.Parameter.Type.STRING, path)])


def test_evaluation_contract_override_applies_reward_scenario_runtime_evaluation_sections(node, tmp_path):
    """The core round-2 fairness regression: a REQUESTED evaluation
    profile's own reward/scenario/runtime/evaluation sections must
    actually reach the LIVE environment -- not just this node's own
    launch-time (checkpoint-training) values."""
    from hunter_kinodynamic_rl.evaluation.contract_override import write_evaluation_contract_override

    _stub_gazebo(node)
    overridden = dataclasses.replace(
        node.profile,
        scenario=dataclasses.replace(node.profile.scenario, world_size_m=node.profile.scenario.world_size_m + 7.0),
        reward=dataclasses.replace(node.profile.reward, goal_threshold_m=node.profile.reward.goal_threshold_m + 1.0),
        runtime=dataclasses.replace(node.profile.runtime, time_delta_sec=node.profile.runtime.time_delta_sec * 2),
        evaluation=dataclasses.replace(node.profile.evaluation, max_episode_steps=42),
    )
    override_path = str(tmp_path / "contract.yaml")
    write_evaluation_contract_override(override_path, overridden)
    _set_evaluation_contract_override_path(node, override_path)

    node._on_reset(Reset.Request(), Reset.Response())

    assert node.profile.scenario.world_size_m == pytest.approx(overridden.scenario.world_size_m)
    assert node.profile.reward.goal_threshold_m == pytest.approx(overridden.reward.goal_threshold_m)
    assert node.profile.evaluation.max_episode_steps == overridden.evaluation.max_episode_steps
    # The CACHED runtime attributes (gazebo_runtime.py's hot-path timing
    # fields) must ALSO reflect the override -- not just self.profile.runtime.
    assert node.time_delta == pytest.approx(overridden.runtime.time_delta_sec)
    assert node.profile.runtime.time_delta_sec == pytest.approx(overridden.runtime.time_delta_sec)


def test_evaluation_contract_override_never_touches_architecture_sections(node, tmp_path):
    from hunter_kinodynamic_rl.evaluation.contract_override import write_evaluation_contract_override
    from hunter_kinodynamic_rl.evaluation.fingerprint import architecture_fingerprint

    _stub_gazebo(node)
    original_architecture_fp = node.get_parameter("architecture_fingerprint_sha256").value
    assert original_architecture_fp == architecture_fingerprint(node.profile)

    # scenario.world_size_m deliberately left UNCHANGED here (a large value
    # would make procedural generate_scenario() genuinely slow/retry-heavy
    # -- see the fixed-benchmark tests below for world-size coverage); this
    # test only needs SOME evaluation-contract field to change to prove
    # architecture stays untouched, so reward.goal_threshold_m suffices.
    overridden = dataclasses.replace(
        node.profile, reward=dataclasses.replace(node.profile.reward, goal_threshold_m=1.5))
    override_path = str(tmp_path / "contract.yaml")
    write_evaluation_contract_override(override_path, overridden)
    _set_evaluation_contract_override_path(node, override_path)

    node._on_reset(Reset.Request(), Reset.Response())
    assert node.profile.reward.goal_threshold_m == pytest.approx(1.5)  # sanity: the override DID apply

    # Architecture sections (and the ROS parameter reporting their
    # fingerprint) must be COMPLETELY unaffected by an evaluation-contract
    # override -- only reward/scenario/runtime/evaluation may change.
    assert node.profile.action_space == node._launch_profile.action_space
    assert node.profile.features == node._launch_profile.features
    assert node.profile.observation == node._launch_profile.observation
    assert node.profile.robot == node._launch_profile.robot
    assert node.get_parameter("architecture_fingerprint_sha256").value == original_architecture_fp


def test_evaluation_contract_override_updates_the_live_fingerprint_parameter(node, tmp_path):
    from hunter_kinodynamic_rl.evaluation.contract_override import write_evaluation_contract_override
    from hunter_kinodynamic_rl.evaluation.fingerprint import evaluation_contract_fingerprint

    _stub_gazebo(node)
    launch_time_fp = node.get_parameter("evaluation_contract_fingerprint_sha256").value
    assert launch_time_fp == evaluation_contract_fingerprint(node.profile)

    overridden = dataclasses.replace(
        node.profile, reward=dataclasses.replace(node.profile.reward, goal_threshold_m=1.234))
    override_path = str(tmp_path / "contract.yaml")
    write_evaluation_contract_override(override_path, overridden)
    _set_evaluation_contract_override_path(node, override_path)

    node._on_reset(Reset.Request(), Reset.Response())

    live_fp_after = node.get_parameter("evaluation_contract_fingerprint_sha256").value
    assert live_fp_after != launch_time_fp
    assert live_fp_after == evaluation_contract_fingerprint(node.profile)


def test_evaluation_contract_override_cleared_restores_the_launch_time_contract(node, tmp_path):
    """Uses a FIXED-benchmark scenario override throughout (never
    procedural generation) so an enlarged scenario.world_size_m can be
    exercised without making generate_scenario() genuinely slow/retry-heavy
    -- this test is about contract-section restoration, not procedural
    scenario placement."""
    from hunter_kinodynamic_rl.evaluation.contract_override import write_evaluation_contract_override

    _stub_gazebo(node)
    launch_world_size = node.profile.scenario.world_size_m
    launch_time_delta = node.profile.runtime.time_delta_sec

    scenario_path = tmp_path / "fixed_scenario.yaml"
    scenario_path.write_text(
        "scenario_id: item1_round2_restore_check\nseed: 1\n"
        "start: {x: 0.0, y: 0.0, yaw: 0.0}\ngoal: {x: 2.0, y: 0.0}\n"
        "static_obstacles: []\nmoving_obstacles: []\n"
    )
    node.set_parameters([rclpy.parameter.Parameter(
        "scenario_override_path", rclpy.parameter.Parameter.Type.STRING, str(scenario_path))])

    overridden = dataclasses.replace(
        node.profile, scenario=dataclasses.replace(node.profile.scenario, world_size_m=launch_world_size + 50.0))
    override_path = str(tmp_path / "contract.yaml")
    write_evaluation_contract_override(override_path, overridden)
    _set_evaluation_contract_override_path(node, override_path)
    node._on_reset(Reset.Request(), Reset.Response())
    assert node.profile.scenario.world_size_m == pytest.approx(launch_world_size + 50.0)

    _set_evaluation_contract_override_path(node, "")  # clear -- mirrors run_benchmark's own cleanup
    node._on_reset(Reset.Request(), Reset.Response())

    assert node.profile.scenario.world_size_m == pytest.approx(launch_world_size)
    assert node.profile.runtime.time_delta_sec == pytest.approx(launch_time_delta)
    assert node.time_delta == pytest.approx(launch_time_delta)


def test_fixed_benchmark_episode_timeout_uses_evaluation_max_episode_steps_not_training_episode_length_steps(
        node, tmp_path):
    """The core server-side item-1 (round 2) regression: a fixed-benchmark
    episode must time out at `evaluation.max_episode_steps`, NEVER this
    node's own (much larger) `training.episode_length_steps` -- previously
    the live environment kept using its OWN launch/training profile's
    timeout regardless of what the requested evaluation profile asked for,
    silently giving two checkpoints trained under different profiles
    different server-side timeout budgets on "the same" benchmark."""
    from hunter_kinodynamic_rl.evaluation.contract_override import write_evaluation_contract_override

    _stub_gazebo(node)
    assert node.profile.training.episode_length_steps > 5  # sanity: the training budget is much larger

    scenario_path = tmp_path / "fixed_scenario.yaml"
    scenario_path.write_text(
        "scenario_id: item1_round2_timeout_check\nseed: 1\n"
        "start: {x: 0.0, y: 0.0, yaw: 0.0}\ngoal: {x: 50.0, y: 50.0}\n"
        "static_obstacles: []\nmoving_obstacles: []\n"
    )
    node.set_parameters([rclpy.parameter.Parameter(
        "scenario_override_path", rclpy.parameter.Parameter.Type.STRING, str(scenario_path))])

    overridden = dataclasses.replace(
        node.profile, evaluation=dataclasses.replace(node.profile.evaluation, max_episode_steps=2))
    override_path = str(tmp_path / "contract.yaml")
    write_evaluation_contract_override(override_path, overridden)
    _set_evaluation_contract_override_path(node, override_path)

    node._on_reset(Reset.Request(), Reset.Response())
    assert node._is_fixed_benchmark is True

    step_req = Step.Request()
    step_req.action = [0.0, 0.0, -1.0]  # far from the goal, no collision expected either
    resp1 = node._on_step(step_req, Step.Response())
    assert resp1.done is False  # 1 step in, budget is 2 -- not timed out yet
    resp2 = node._on_step(step_req, Step.Response())
    assert resp2.done is True  # 2 steps in, budget is 2 -- timed out, NOT because of training's own (much larger) budget
    assert resp2.target is False
    assert resp2.collision is False


def test_non_fixed_benchmark_episode_still_uses_training_episode_length_steps(node):
    """The dual of the test above: ordinary (procedural, non-benchmark)
    episodes must be COMPLETELY unaffected by this fix -- they keep using
    training.episode_length_steps exactly as before."""
    _stub_gazebo(node)
    node._on_reset(Reset.Request(), Reset.Response())
    assert node._is_fixed_benchmark is False

    step_req = Step.Request()
    step_req.action = [0.0, 0.0, -1.0]
    for _ in range(3):
        resp = node._on_step(step_req, Step.Response())
    # 3 steps in, training.episode_length_steps is >5 (asserted in the
    # sibling test) -- must NOT have timed out.
    assert resp.done is False


# --------------------------------------------------- item-1 (round 3): runtime field audit
def test_watchdog_period_change_actually_recreates_the_live_ros_timer(node, tmp_path):
    """The core round-3 regression: previously, an evaluation-contract
    override changing runtime.watchdog_period_sec only swapped a cached
    Python attribute -- the ALREADY-RUNNING rclpy Timer's own period never
    changed (a genuine false guarantee: the fingerprint would differ, but
    nothing observable would). Reproduced here by reading the REAL
    rclpy.timer.Timer object's own timer_period_ns before and after."""
    from hunter_kinodynamic_rl.evaluation.contract_override import write_evaluation_contract_override

    _stub_gazebo(node)
    original_timer = node._watchdog_timer
    original_period_ns = original_timer.timer_period_ns
    launch_period_sec = node.profile.runtime.watchdog_period_sec
    new_period_sec = launch_period_sec * 3.0 + 0.5  # deliberately, unambiguously different

    overridden = dataclasses.replace(
        node.profile, runtime=dataclasses.replace(node.profile.runtime, watchdog_period_sec=new_period_sec))
    override_path = str(tmp_path / "contract.yaml")
    write_evaluation_contract_override(override_path, overridden)
    _set_evaluation_contract_override_path(node, override_path)

    node._on_reset(Reset.Request(), Reset.Response())

    assert node._watchdog_period_sec == pytest.approx(new_period_sec)
    # The REAL rclpy Timer object was actually recreated with the new
    # period -- not just a Python-side attribute that nothing reads.
    assert node._watchdog_timer is not original_timer
    assert node._watchdog_timer.timer_period_ns == pytest.approx(int(new_period_sec * 1e9), rel=1e-6)
    assert node._watchdog_timer.timer_period_ns != original_period_ns
    # The OLD timer must actually be DESTROYED (its underlying rcl handle
    # freed via destroy_timer), not merely abandoned/leaked still running
    # in the background -- a destroyed timer raises InvalidHandle on any
    # further use, which is a stronger proof of destruction than a status
    # flag would be (rclpy has no "cancelled but still alive" accessor).
    import rclpy._rclpy_pybind11 as _rclpy_pybind11
    with pytest.raises(_rclpy_pybind11.InvalidHandle):
        original_timer.is_canceled()


def test_watchdog_period_unchanged_does_not_recreate_the_timer(node):
    """No-op change (or no override at all) must NOT tear down/rebuild a
    perfectly good running timer for no reason."""
    _stub_gazebo(node)
    original_timer = node._watchdog_timer
    node._on_reset(Reset.Request(), Reset.Response())  # no override set at all
    assert node._watchdog_timer is original_timer
    assert not original_timer.is_canceled()


def test_gazebo_max_step_size_sec_override_mismatch_fails_fast_before_mutating_profile(node, tmp_path):
    """The core round-3 regression for the OTHER genuinely-unsafe runtime
    field: gazebo_max_step_size_sec describes the ALREADY-RUNNING Gazebo
    world's own physics step size, which this node cannot verify or change
    -- an override requesting a DIFFERENT value must be refused outright,
    never silently accepted into self.profile (which would corrupt
    multi_step_advance's physics-step-count math with no visible symptom
    other than wrong timing)."""
    from hunter_kinodynamic_rl.evaluation.contract_override import write_evaluation_contract_override

    _stub_gazebo(node)
    launch_step_size = node.profile.runtime.gazebo_max_step_size_sec
    launch_world_size = node.profile.scenario.world_size_m

    overridden = dataclasses.replace(
        node.profile,
        runtime=dataclasses.replace(node.profile.runtime, gazebo_max_step_size_sec=launch_step_size * 2.0),
        # ALSO change something legitimately applicable, to prove the
        # REJECTED override leaves self.profile COMPLETELY untouched, not
        # partially applied.
        scenario=dataclasses.replace(node.profile.scenario, world_size_m=launch_world_size + 3.0),
    )
    override_path = str(tmp_path / "contract.yaml")
    write_evaluation_contract_override(override_path, overridden)
    _set_evaluation_contract_override_path(node, override_path)

    with pytest.raises(RuntimeError, match="gazebo_max_step_size_sec"):
        node._on_reset(Reset.Request(), Reset.Response())

    # Atomic reject: NOTHING from the rejected override was applied, not
    # even the legitimately-applicable scenario.world_size_m change.
    assert node.profile.runtime.gazebo_max_step_size_sec == pytest.approx(launch_step_size)
    assert node.profile.scenario.world_size_m == pytest.approx(launch_world_size)


def test_gazebo_max_step_size_sec_override_matching_launch_value_succeeds(node, tmp_path):
    from hunter_kinodynamic_rl.evaluation.contract_override import write_evaluation_contract_override

    _stub_gazebo(node)
    launch_step_size = node.profile.runtime.gazebo_max_step_size_sec

    overridden = dataclasses.replace(
        node.profile,
        runtime=dataclasses.replace(node.profile.runtime, gazebo_max_step_size_sec=launch_step_size),
        reward=dataclasses.replace(node.profile.reward, goal_threshold_m=2.0),
    )
    override_path = str(tmp_path / "contract.yaml")
    write_evaluation_contract_override(override_path, overridden)
    _set_evaluation_contract_override_path(node, override_path)

    node._on_reset(Reset.Request(), Reset.Response())  # must NOT raise

    assert node.profile.reward.goal_threshold_m == pytest.approx(2.0)


def test_evaluation_contract_override_reloads_changed_content_at_the_same_path(node, tmp_path):
    """The core round-3 "same-path caching" regression: writing DIFFERENT
    content to the SAME override path (e.g. a second evaluation_node.py
    invocation reusing an old fixed filename) must still be picked up on
    the NEXT /reset -- tracking only the path string (round-2 behavior)
    would incorrectly skip re-parsing here, silently keeping the FIRST
    content active."""
    from hunter_kinodynamic_rl.evaluation.contract_override import write_evaluation_contract_override

    _stub_gazebo(node)
    override_path = str(tmp_path / "contract.yaml")  # a FIXED, reused path

    first = dataclasses.replace(
        node.profile, reward=dataclasses.replace(node.profile.reward, goal_threshold_m=1.0))
    write_evaluation_contract_override(override_path, first)
    _set_evaluation_contract_override_path(node, override_path)
    node._on_reset(Reset.Request(), Reset.Response())
    assert node.profile.reward.goal_threshold_m == pytest.approx(1.0)

    # Overwrite the SAME path with genuinely different content -- the ROS
    # parameter value (the path string) never changes.
    second = dataclasses.replace(
        node.profile, reward=dataclasses.replace(node.profile.reward, goal_threshold_m=2.0))
    write_evaluation_contract_override(override_path, second)
    node._on_reset(Reset.Request(), Reset.Response())

    assert node.profile.reward.goal_threshold_m == pytest.approx(2.0)  # NOT still 1.0


def test_evaluation_contract_override_missing_file_raises(node, tmp_path):
    _stub_gazebo(node)
    _set_evaluation_contract_override_path(node, str(tmp_path / "does_not_exist.yaml"))
    with pytest.raises(RuntimeError, match="does not exist"):
        node._on_reset(Reset.Request(), Reset.Response())
