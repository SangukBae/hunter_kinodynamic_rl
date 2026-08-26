#!/usr/bin/env python3
"""``ros2 run hunter_kinodynamic_rl system_id_node.py --ros-args \\
    -p trial:=circle -p output:=system_id_results/circle_20deg.csv``

Runs one of the four system-identification trial types (section 16:
velocity step, steering step, constant circle, stop response) by ACTIVELY
COMMANDING the robot through a scripted /cmd_vel sequence (section P1-12 --
this previously only PASSIVELY recorded while a human drove the robot
manually), while recording (t, x, y, yaw, v, steering) from /odometry +
/hunter_se/joint_states. Steering is read from the REAL Ackermann center
steering angle (mean of the two front wheel joints), NEVER
odometry.twist.angular.z (that is YAW RATE, a different physical quantity
-- section P1-12's core correction, mirroring the exact fix already applied
to trainer_base.py's EnvironmentClient and environment_node.py's own
odometry handling).

``--trial:=all`` runs all four trials back-to-back, analyzes each with
dynamics/system_identification.py's analyze_* functions, and writes an
IDENTIFIED robot YAML (config/robot/hunter_se.yaml's own shape) merging the
measured parameters into a base RobotConfig -- see
build_identified_robot_config's docstring for exactly which fields each
trial can/can't identify.

NOT executed against real hardware in this development session -- no real
Hunter SE is available here; only verified in Gazebo (a legitimate sanity
check per this module's original docstring, not a substitute for a real
trial). See docs/SIM2REAL.md.
"""

from __future__ import annotations

import argparse
import math
import time
from typing import List, Optional

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import JointState

from hunter_kinodynamic_rl.config.loader import load_profile
from hunter_kinodynamic_rl.dynamics.system_identification import (
    Sample, _extrapolate_forward_at_constant_velocity, analyze_circle_test, analyze_steering_step_response,
    analyze_stop_test, analyze_velocity_step_response, build_identified_robot_config, samples_to_csv,
    write_identified_robot_yaml, write_results,
)

TRIAL_TYPES = ("velocity_step", "steering_step", "circle", "stop")

# section item-3 (round 3): default bound on how old the LAST-received
# odometry/joint-state message may be, relative to the brake-onset instant,
# before build_brake_onset_snapshot refuses to trust it at all -- mirrors
# analyze_stop_test's own max_sample_gap_sec default/rationale (an
# unbounded extrapolation is exactly as unreliable as an unbounded
# odometry dropout).
DEFAULT_MAX_ONSET_SNAPSHOT_STALENESS_SEC = 0.2


def build_brake_onset_snapshot(
    latest_xy, latest_yaw: float, latest_v_mps: float, latest_odom_monotonic_time: Optional[float],
    latest_center_steering_rad: float, latest_joint_state_monotonic_time: Optional[float],
    onset_monotonic_time: float, t0_monotonic_time: float,
    max_snapshot_staleness_sec: float = DEFAULT_MAX_ONSET_SNAPSHOT_STALENESS_SEC,
):
    """section item-4 (round 2)/item-3 (round 3): builds the PREFERRED
    brake-onset reference `analyze_stop_test` uses, from the recorder's own
    live odometry/joint-state state -- factored out of
    `SystemIdRecorder.run_trial` purely so this is directly unit-testable
    without a live rclpy context (mirrors this package's own convention of
    factoring ROS-node-embedded logic into a plain function -- e.g.
    `real_policy_node.py::_safety_limits_from_profile`).

    section item-3 (round 3) fix: the round-2 version
    (`snapshot_brake_onset_state`) built ``Sample(t_sec=onset_t_sec,
    x=latest_xy[0], y=latest_xy[1], ...)`` UNCONDITIONALLY -- i.e. it
    attached the CURRENT onset timestamp to whatever (possibly STALE)
    odometry/joint-state values happened to be cached, with no freshness
    check at all. If odometry hadn't updated in, say, 300ms before the
    brake command was issued, the reported "state at onset" would silently
    be the state from 300ms EARLIER, mislabeled with the onset's own
    timestamp -- reintroducing, in a more subtle form, exactly the kind of
    position/velocity-vs-time mismatch this whole mechanism exists to
    eliminate.

    Fixed by tracking the MONOTONIC RECEIPT time of the last odometry AND
    the last joint-state message SEPARATELY, and:

    - returning ``(None, "<reason>")`` if EITHER is missing entirely, or is
      older than ``max_snapshot_staleness_sec`` relative to the onset
      instant (the trial's live-snapshot path is then unavailable --
      `analyze_stop_test`'s own OWN, separately-verified,
      `brake_onset_t_sec`-based sample-extrapolation fallback still gets a
      chance to recover a valid result from the discrete sample stream);
    - returning ``(None, "onset_snapshot_receipt_time_inconsistent")`` if a
      message's receipt time is AFTER the onset instant (should be
      impossible given call ordering, but never silently treated as
      "fresh" if it somehow happened);
    - otherwise EXTRAPOLATING the last-known ODOMETRY forward from its OWN
      receipt time to the exact onset instant, at its own (still
      steady-state, not-yet-braking) velocity/heading -- reusing
      `dynamics.system_identification._extrapolate_forward_at_constant_velocity`,
      the SAME physically-justified model `analyze_stop_test`'s own
      sample-based fallback already uses, rather than a second,
      independent implementation of the same physics. Steering is used
      AS-IS (no analogous "steering rate" extrapolation model exists, and
      the stopping-distance/time computation itself never consumes
      steering directly)."""
    if latest_xy is None or latest_odom_monotonic_time is None:
        return None, "no_odometry_received_before_onset"
    if latest_joint_state_monotonic_time is None:
        return None, "no_joint_state_received_before_onset"

    odom_age_sec = onset_monotonic_time - latest_odom_monotonic_time
    joint_state_age_sec = onset_monotonic_time - latest_joint_state_monotonic_time
    if odom_age_sec < 0.0 or joint_state_age_sec < 0.0:
        return None, "onset_snapshot_receipt_time_inconsistent"
    if odom_age_sec > max_snapshot_staleness_sec:
        return None, "odometry_stale_at_brake_onset"
    if joint_state_age_sec > max_snapshot_staleness_sec:
        return None, "joint_state_stale_at_brake_onset"

    onset_t_sec = onset_monotonic_time - t0_monotonic_time
    last_known = Sample(
        t_sec=onset_t_sec - odom_age_sec, x=latest_xy[0], y=latest_xy[1], yaw=latest_yaw,
        v_mps=latest_v_mps, steering_rad=latest_center_steering_rad,
    )
    snapshot = _extrapolate_forward_at_constant_velocity(last_known, onset_t_sec)
    return snapshot, None


class SystemIdRecorder(Node):
    def __init__(self, robot_entity_name: str = "hunter_se",
                 max_onset_snapshot_staleness_sec: float = DEFAULT_MAX_ONSET_SNAPSHOT_STALENESS_SEC):
        super().__init__("hunter_kinodynamic_system_id_recorder")
        self._latest_xy = None
        self._latest_yaw = 0.0
        self._latest_v_mps = 0.0
        self._latest_center_steering_rad = 0.0
        # section item-3 (round 3): MONOTONIC RECEIPT time of the latest
        # odometry/joint-state message, tracked SEPARATELY from the message
        # content itself -- see build_brake_onset_snapshot's own docstring
        # for why the content alone (round-2 behavior) is not enough to
        # tell a genuinely fresh reading apart from a stale one.
        self._latest_odom_monotonic_time: Optional[float] = None
        self._latest_joint_state_monotonic_time: Optional[float] = None
        self._max_onset_snapshot_staleness_sec = max_onset_snapshot_staleness_sec
        self._recording = False
        self._samples: List[Sample] = []
        self._t0 = None
        # section P1-2: set only by the "stop" trial, to the recorder's own
        # elapsed-time timestamp at the moment it switches from commanding
        # target_v_mps to commanding 0 -- lets analyze_stop_test trim away
        # the pre-brake acceleration-to-steady-speed ramp (which itself
        # starts at v=0), instead of treating the trial's very first
        # (v=0, t=0) sample as if it were already the stop event.
        self._stop_trial_brake_onset_t_sec: Optional[float] = None
        # section item-4 (round 2): a REAL Sample snapshotted from the
        # latest live odometry/joint-state readings AT the exact instant
        # the brake command is issued -- the PREFERRED brake-onset
        # reference analyze_stop_test uses (see its own docstring): closes
        # the "some driving between the last recorded pre-onset sample and
        # the true onset instant gets silently counted as braking" gap
        # entirely, since this is captured directly rather than derived
        # from the discrete _samples list at all.
        self._stop_trial_brake_onset_state: Optional[Sample] = None
        # section item-3 (round 3): why the live snapshot above is None (if
        # it is) -- e.g. "odometry_stale_at_brake_onset". Surfaced into the
        # stop-trial result as diagnostic-only metadata (never silently
        # dropped) even when analyze_stop_test's own sample-based fallback
        # still recovers a valid result without it.
        self._stop_trial_brake_onset_invalid_reason: Optional[str] = None

        self._cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_subscription(Odometry, "/odometry", self._on_odom, 10)
        self.create_subscription(JointState, f"/{robot_entity_name}/joint_states", self._on_joint_states, 10)

    def _on_odom(self, msg: Odometry) -> None:
        self._latest_xy = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        self._latest_yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self._latest_v_mps = msg.twist.twist.linear.x
        self._latest_odom_monotonic_time = time.monotonic()
        if self._recording:
            self._record_sample()

    def _on_joint_states(self, msg: JointState) -> None:
        """section P1-12: the REAL Ackermann center steering angle (mean of
        the two front wheel joints) -- NEVER odometry.twist.angular.z,
        which is yaw RATE [rad/s], not a steering ANGLE [rad] at all. This
        was the exact bug found in code review: the original recorder
        logged angular.z into a field literally named steering_rad."""
        try:
            left = float(msg.position[msg.name.index("front_left_steering")])
            right = float(msg.position[msg.name.index("front_right_steering")])
        except (ValueError, IndexError, TypeError):
            return
        self._latest_center_steering_rad = 0.5 * (left + right)
        self._latest_joint_state_monotonic_time = time.monotonic()

    def _record_sample(self) -> None:
        if self._latest_xy is None or self._t0 is None:
            return
        now = time.monotonic()
        self._samples.append(Sample(
            t_sec=now - self._t0, x=self._latest_xy[0], y=self._latest_xy[1], yaw=self._latest_yaw,
            v_mps=self._latest_v_mps, steering_rad=self._latest_center_steering_rad,
        ))

    def _publish(self, speed_mps: float, steering_rad: float) -> None:
        msg = Twist()
        msg.linear.x = speed_mps
        msg.angular.z = steering_rad
        self._cmd_pub.publish(msg)

    def _hold_for(self, speed_mps: float, steering_rad: float, duration_sec: float,
                   republish_period_sec: float = 0.1) -> None:
        """Publish (speed_mps, steering_rad) REPEATEDLY for duration_sec --
        NOT a single one-shot publish. hunter_se_cmd_prefilter.yaml has its
        own command_timeout_sec=2.0 (it zeros the command if not refreshed
        within that window, the same protection environment_node.py's own
        watchdog implements independently -- see _on_watchdog_tick) -- a
        single publish() at the start of an 8s trial was confirmed LIVE to
        silently decay to zero after 2s, making the robot barely move at
        all and producing a meaningless turning-radius fit. Republishing
        well inside that budget (matching the real ~10Hz RL control rate)
        keeps the command alive for the whole trial, exactly like a real
        control loop calling /step repeatedly would."""
        deadline = time.monotonic() + duration_sec
        while time.monotonic() < deadline:
            self._publish(speed_mps, steering_rad)
            rclpy.spin_once(self, timeout_sec=republish_period_sec)

    def run_trial(self, trial: str, target_v_mps: float, target_steering_rad: float,
                  step_duration_sec: float, settle_duration_sec: float = 1.0) -> List[Sample]:
        """Actively commands the robot through ONE trial type (section
        P1-12), recording samples throughout. Returns the collected
        samples (also left in ``self._samples`` for convenience)."""
        if trial not in TRIAL_TYPES:
            raise ValueError(f"unknown trial type {trial!r} -- must be one of {TRIAL_TYPES}")

        self._samples = []
        self._t0 = time.monotonic()
        self._stop_trial_brake_onset_t_sec = None
        self._stop_trial_brake_onset_state = None
        self._stop_trial_brake_onset_invalid_reason = None
        self._recording = True
        self.get_logger().info(f"[system_id] starting trial={trial!r}")

        if trial == "velocity_step":
            self._hold_for(target_v_mps, 0.0, step_duration_sec)
        elif trial == "steering_step":
            # A small constant forward speed so steering has an observable
            # effect on (x, y, yaw) too, not just the joint angle itself.
            self._hold_for(target_v_mps, target_steering_rad, step_duration_sec)
        elif trial == "circle":
            self._hold_for(target_v_mps, target_steering_rad, step_duration_sec)  # several full loops
        elif trial == "stop":
            self._hold_for(target_v_mps, 0.0, settle_duration_sec)  # reach steady speed first
            # section P1-2: stamped BEFORE the brake command is issued, so
            # it marks the exact boundary between the acceleration ramp
            # (which analyze_stop_test must ignore) and the braking segment
            # (which it must analyze).
            onset_monotonic_time = time.monotonic()
            self._stop_trial_brake_onset_t_sec = onset_monotonic_time - self._t0
            # section item-4 (round 2)/item-3 (round 3): snapshot the REAL
            # (x, y, yaw, v, steering) the robot had AT this exact instant,
            # directly from the latest live odometry/joint-state callbacks,
            # WITH a freshness check against each message's own monotonic
            # receipt time -- see build_brake_onset_snapshot's own
            # docstring for why the receipt-time check is necessary (not
            # just deriving from whatever content happens to be cached).
            # (None, reason) when the live snapshot isn't trustworthy --
            # analyze_stop_test's own separate, already-audited
            # brake_onset_t_sec-based sample fallback still gets a chance
            # to recover a valid result.
            self._stop_trial_brake_onset_state, self._stop_trial_brake_onset_invalid_reason = (
                build_brake_onset_snapshot(
                    self._latest_xy, self._latest_yaw, self._latest_v_mps, self._latest_odom_monotonic_time,
                    self._latest_center_steering_rad, self._latest_joint_state_monotonic_time,
                    onset_monotonic_time, self._t0,
                    max_snapshot_staleness_sec=self._max_onset_snapshot_staleness_sec,
                )
            )
            self._hold_for(0.0, 0.0, step_duration_sec)  # record the deceleration

        self._publish(0.0, 0.0)  # always end at a safe stop
        self._recording = False
        self.get_logger().info(f"[system_id] trial={trial!r} collected {len(self._samples)} samples")
        return list(self._samples)


def _stop_trial_result(node: SystemIdRecorder, samples: List[Sample], target_v_mps: float) -> dict:
    """Factored out of ``_run_one_trial`` purely so the stop-trial
    analysis/merge logic is directly unit-testable against a
    ``SystemIdRecorder`` instance without a live ``run_trial()`` call (real
    Gazebo/topics) -- see tests/test_system_id_node.py's node-level
    valid/reason coverage.

    section item-2 (system-ID stale-data fix): ``node._stop_trial_brake_onset_invalid_reason``
    is passed through as ``brake_onset_snapshot_failure_reason`` -- when the
    recorder's own live snapshot attempt explicitly failed (stale/missing/
    inconsistent), ``analyze_stop_test`` now refuses to silently recover
    ``valid=True`` via its own looser sample-extrapolation fallback (see
    that function's own docstring); this call site never opts into
    ``allow_fallback_despite_live_snapshot_failure``, so a live run's own
    stale-data determination always wins."""
    result = analyze_stop_test(
        samples, brake_onset_t_sec=node._stop_trial_brake_onset_t_sec,
        brake_onset_state=node._stop_trial_brake_onset_state,
        # section P1-2: hysteresis_samples=3 on the REAL recorder path
        # (never for the synthetic-data unit tests in
        # test_system_identification.py, which pass the default of 1 to
        # keep their hand-crafted 4-sample traces exact) -- real odometry
        # is noisy enough that a single instantaneous |v| dip below
        # threshold is not reliable evidence the robot has actually
        # stopped.
        brake_onset_snapshot_failure_reason=node._stop_trial_brake_onset_invalid_reason,
        initial_speed_target_mps=target_v_mps, stop_hysteresis_samples=3,
    )
    # section item-3 (round 3): WHY the (preferred) live snapshot wasn't
    # available is real diagnostic information -- never silently dropped,
    # regardless of whether the trial ended up valid or (now, correctly)
    # invalid because of it.
    if node._stop_trial_brake_onset_state is None and node._stop_trial_brake_onset_invalid_reason is not None:
        result = dict(result)
        result["live_onset_snapshot_reason"] = node._stop_trial_brake_onset_invalid_reason
    return result


def _run_one_trial(node: SystemIdRecorder, trial: str, output_dir: str,
                    target_v_mps: float, target_steering_rad: float, step_duration_sec: float) -> tuple:
    samples = node.run_trial(trial, target_v_mps, target_steering_rad, step_duration_sec)
    csv_path = f"{output_dir}/{trial}.csv"
    samples_to_csv(csv_path, samples)
    if trial == "velocity_step":
        result = analyze_velocity_step_response(samples, target_v_mps)
    elif trial == "steering_step":
        result = analyze_steering_step_response(samples, target_steering_rad)
    elif trial == "circle":
        result = analyze_circle_test(samples)
    else:
        result = _stop_trial_result(node, samples, target_v_mps)
    write_results(f"{output_dir}/{trial}_results.json", result)
    return samples, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", "-p", dest="kv", action="append", default=[])
    args, _ = parser.parse_known_args()
    kv = {p.split(":=", 1)[0]: p.split(":=", 1)[1] for p in args.kv if ":=" in p}
    trial = kv.get("trial", "all")
    output_dir = kv.get("output_dir", "system_id_results")
    target_v_mps = float(kv.get("target_v_mps", "1.0"))
    target_steering_rad = float(kv.get("target_steering_rad", str(math.radians(15.0))))
    step_duration_sec = float(kv.get("step_duration_sec", "5.0"))
    robot_profile = kv.get("robot_profile", "kinodynamic_tqc")

    rclpy.init()
    node = SystemIdRecorder()
    try:
        if trial == "all":
            results = {}
            for t in TRIAL_TYPES:
                _, results[t] = _run_one_trial(
                    node, t, output_dir, target_v_mps, target_steering_rad, step_duration_sec)
                # section P1-2: a trial that came back invalid still leaves
                # its corresponding RobotConfig field at the base value
                # (build_identified_robot_config's own .get()-guarded
                # merge below already handles that gracefully), but that
                # must never happen SILENTLY -- surface it loudly here.
                if results[t].get("valid") is False:
                    node.get_logger().warn(
                        f"[system_id] trial={t!r} produced an INVALID result "
                        f"(reason={results[t].get('reason')}) -- its measurement will NOT be merged "
                        "into hunter_se_identified.yaml; the base profile's value is kept instead"
                    )
            base_robot = load_profile(robot_profile).robot
            identified = build_identified_robot_config(
                base_robot, velocity_step_result=results["velocity_step"],
                steering_step_result=results["steering_step"], circle_result=results["circle"],
                stop_result=results["stop"], commanded_steering_rad=target_steering_rad,
            )
            yaml_path = f"{output_dir}/hunter_se_identified.yaml"
            write_identified_robot_yaml(yaml_path, identified)
            node.get_logger().info(f"[system_id] wrote identified robot config to {yaml_path}")
        else:
            _run_one_trial(node, trial, output_dir, target_v_mps, target_steering_rad, step_duration_sec)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
