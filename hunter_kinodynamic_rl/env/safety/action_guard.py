"""Real-robot safety guard (section 43) -- the LAST line of defense between
a policy/trajectory output and an actual Hunter SE motor command.

Every check here is MANDATORY on the real-robot deployment path
(``real_hunter_safe.yaml`` -- see env/safety's module docstring policy: this
is not a set of optional knobs, it is always active in
``nodes/environment_node.py`` when running against real hardware). Pure
logic (no ROS clocks/timers) so it's testable with synthetic timestamps;
the real-robot node supplies wall-clock time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from hunter_kinodynamic_rl.config.schema import RobotConfig
from hunter_kinodynamic_rl.trajectory.pure_pursuit_adapter import VehicleCommand


@dataclass(frozen=True)
class SafetyLimits:
    max_sensor_age_sec: float = 0.5
    max_command_age_sec: float = 0.5
    min_obstacle_stop_distance_m: float = 0.3
    # section P0-3: odometry staleness is a SEPARATE failure mode from LiDAR
    # staleness (e.g. a dropped/dead odometry driver with the scan still
    # publishing fine would otherwise go completely unchecked -- `guard()`
    # only ever checked `max_sensor_age_sec` against the scan timestamp).
    # Defaults to the same budget as `max_sensor_age_sec` since both are
    # "how stale can ANY safety-relevant sensor input be" in spirit.
    max_odom_age_sec: float = 0.5


STOP_COMMAND = VehicleCommand(speed_mps=0.0, steering_rad=0.0)


def sanitize_command(command: VehicleCommand, robot: RobotConfig) -> VehicleCommand:
    """NaN/Inf -> immediate stop (section 43: "Policy가 비정상 값을 내면 즉시
    안전한 stop command로 fallback"). Otherwise clamp to physical bounds
    (belt-and-suspenders -- trajectory_executor already clamps, but this is
    the guard a bad NEW trajectory generator/network must still pass through)."""
    if not (math.isfinite(command.speed_mps) and math.isfinite(command.steering_rad)):
        return STOP_COMMAND
    speed = max(0.0, min(robot.max_forward_speed_mps, command.speed_mps))
    steering = max(-robot.steering_limit_rad, min(robot.steering_limit_rad, command.steering_rad))
    return VehicleCommand(speed_mps=speed, steering_rad=steering)


def check_sensor_freshness(last_sensor_time_sec: float, now_sec: float, limits: SafetyLimits) -> bool:
    return (now_sec - last_sensor_time_sec) <= limits.max_sensor_age_sec


def check_odom_freshness(last_odom_time_sec: float, now_sec: float, limits: SafetyLimits) -> bool:
    """section P0-3: mirrors check_sensor_freshness exactly, against
    ``max_odom_age_sec`` -- a distinct budget/timestamp from the LiDAR scan,
    so a dead odometry source (scan still fresh) is caught independently."""
    return (now_sec - last_odom_time_sec) <= limits.max_odom_age_sec


def check_command_freshness(last_command_time_sec: float, now_sec: float, limits: SafetyLimits) -> bool:
    return (now_sec - last_command_time_sec) <= limits.max_command_age_sec


def apply_collision_proximity_stop(
    command: VehicleCommand, nearest_obstacle_distance_m: Optional[float], limits: SafetyLimits,
) -> VehicleCommand:
    """Regardless of what the policy commanded, never drive FORWARD into an
    obstacle already inside the minimum stop distance (reverse/turn-in-place
    is still allowed -- only forward speed is zeroed)."""
    if nearest_obstacle_distance_m is None:
        return command
    if nearest_obstacle_distance_m < limits.min_obstacle_stop_distance_m and command.speed_mps > 0.0:
        return VehicleCommand(speed_mps=0.0, steering_rad=command.steering_rad)
    return command


def guard(
    command: VehicleCommand, robot: RobotConfig, limits: SafetyLimits,
    last_sensor_time_sec: float, last_command_time_sec: float, now_sec: float,
    nearest_obstacle_distance_m: Optional[float] = None,
    last_odom_time_sec: Optional[float] = None,
) -> VehicleCommand:
    """Full safety pipeline: sanitize -> freshness checks (stale sensor/
    odom/command -> emergency stop) -> collision-proximity stop. Call this
    on EVERY tick before publishing to the real robot's cmd_vel-equivalent.

    ``last_odom_time_sec`` is OPTIONAL (default None -> odom freshness is
    not checked) purely for backward compatibility with existing callers
    that never tracked a separate odometry timestamp (e.g.
    environment_node.py's sim path, where /odometry and /scan come from the
    same synchronized Gazebo process and a scan-only check was never wrong
    in practice) -- section P0-3's real-hardware requirement is satisfied
    by real_policy_node.py, the one caller that actually passes it."""
    safe = sanitize_command(command, robot)
    if not check_sensor_freshness(last_sensor_time_sec, now_sec, limits):
        return STOP_COMMAND
    if last_odom_time_sec is not None and not check_odom_freshness(last_odom_time_sec, now_sec, limits):
        return STOP_COMMAND
    if not check_command_freshness(last_command_time_sec, now_sec, limits):
        return STOP_COMMAND
    return apply_collision_proximity_stop(safe, nearest_obstacle_distance_m, limits)
