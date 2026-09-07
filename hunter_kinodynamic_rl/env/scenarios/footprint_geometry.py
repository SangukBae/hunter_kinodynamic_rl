"""ROS-free 2-D geometry for the v2 Hunter/obstacle footprints.

Legacy code continues to use the conservative circular radius.  The v2
generator additionally uses the measured rectangular Hunter envelope when
checking start/goal clearance, avoiding both the narrow-side false positives
and corner false negatives of a single circle.
"""

from __future__ import annotations

import math
from typing import Iterable, Tuple


def circumscribed_radius(shape: str, radius: float, length_m: float, width_m: float) -> float:
    if shape == "cylinder" or length_m <= 0.0 or width_m <= 0.0:
        return float(radius)
    return max(float(radius), 0.5 * math.hypot(length_m, width_m))


def circle_to_oriented_rectangle_clearance(
    circle_xy: Tuple[float, float],
    circle_radius_m: float,
    rectangle_xy: Tuple[float, float],
    rectangle_yaw_rad: float,
    rectangle_length_m: float,
    rectangle_width_m: float,
    padding_m: float = 0.0,
) -> float:
    """Signed surface clearance; negative means overlap."""
    dx = circle_xy[0] - rectangle_xy[0]
    dy = circle_xy[1] - rectangle_xy[1]
    c = math.cos(rectangle_yaw_rad)
    s = math.sin(rectangle_yaw_rad)
    local_x = c * dx + s * dy
    local_y = -s * dx + c * dy
    half_l = 0.5 * rectangle_length_m + padding_m
    half_w = 0.5 * rectangle_width_m + padding_m
    outside_x = max(abs(local_x) - half_l, 0.0)
    outside_y = max(abs(local_y) - half_w, 0.0)
    outside_distance = math.hypot(outside_x, outside_y)
    if outside_x > 0.0 or outside_y > 0.0:
        return outside_distance - circle_radius_m
    # Circle center is inside the rectangle: report penetration to the
    # nearest face plus the circle's own radius.
    return -min(half_l - abs(local_x), half_w - abs(local_y)) - circle_radius_m


def _rectangle_axes(yaw_rad: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    return (c, s), (-s, c)


def _project_rectangle(
    center: Tuple[float, float], yaw_rad: float, length_m: float, width_m: float,
    axis: Tuple[float, float], padding_m: float,
) -> Tuple[float, float]:
    forward, lateral = _rectangle_axes(yaw_rad)
    center_projection = center[0] * axis[0] + center[1] * axis[1]
    radius = (
        (0.5 * length_m + padding_m) * abs(forward[0] * axis[0] + forward[1] * axis[1])
        + (0.5 * width_m + padding_m) * abs(lateral[0] * axis[0] + lateral[1] * axis[1])
    )
    return center_projection - radius, center_projection + radius


def oriented_rectangles_overlap(
    a_xy: Tuple[float, float], a_yaw_rad: float, a_length_m: float, a_width_m: float,
    b_xy: Tuple[float, float], b_yaw_rad: float, b_length_m: float, b_width_m: float,
    padding_m: float = 0.0,
) -> bool:
    """Separating-axis test for two oriented rectangles."""
    axes: Iterable[Tuple[float, float]] = _rectangle_axes(a_yaw_rad) + _rectangle_axes(b_yaw_rad)
    for axis in axes:
        alo, ahi = _project_rectangle(a_xy, a_yaw_rad, a_length_m, a_width_m, axis, padding_m)
        blo, bhi = _project_rectangle(b_xy, b_yaw_rad, b_length_m, b_width_m, axis, padding_m)
        if ahi < blo or bhi < alo:
            return False
    return True


def oriented_rectangle_boundary_clearance(
    center_xy: Tuple[float, float], yaw_rad: float, length_m: float, width_m: float,
    half_extent_m: float, padding_m: float = 0.0,
) -> float:
    """Minimum clearance from an oriented rectangle to a square world."""
    c, s = abs(math.cos(yaw_rad)), abs(math.sin(yaw_rad))
    half_l = 0.5 * length_m + padding_m
    half_w = 0.5 * width_m + padding_m
    extent_x = half_l * c + half_w * s
    extent_y = half_l * s + half_w * c
    return min(
        half_extent_m - abs(center_xy[0]) - extent_x,
        half_extent_m - abs(center_xy[1]) - extent_y,
    )


def footprint_overlaps_obstacle(
    robot_xy: Tuple[float, float], robot_yaw_rad: float,
    robot_length_m: float, robot_width_m: float, obstacle,
    padding_m: float = 0.0,
) -> bool:
    """Hunter oriented rectangle versus a v1/v2 obstacle geometry."""
    shape = getattr(obstacle, "shape", "cylinder")
    length = float(getattr(obstacle, "length_m", 0.0))
    width = float(getattr(obstacle, "width_m", 0.0))
    obstacle_x = getattr(obstacle, "x", None)
    obstacle_y = getattr(obstacle, "y", None)
    obstacle_xy = (
        float(obstacle_x if obstacle_x is not None else getattr(obstacle, "x0")),
        float(obstacle_y if obstacle_y is not None else getattr(obstacle, "y0")),
    )
    obstacle_yaw = float(getattr(obstacle, "yaw_rad", 0.0))
    if shape == "cylinder" or length <= 0.0 or width <= 0.0:
        return circle_to_oriented_rectangle_clearance(
            obstacle_xy, float(obstacle.radius), robot_xy, robot_yaw_rad,
            robot_length_m, robot_width_m, padding_m,
        ) <= 0.0
    if shape != "l_shape":
        return oriented_rectangles_overlap(
            robot_xy, robot_yaw_rad, robot_length_m, robot_width_m,
            obstacle_xy, obstacle_yaw, length, width, padding_m,
        )

    # Match obstacle_spawner._l_shape_sdf exactly: union of two thin boxes.
    arm = max(0.12, min(length, width) * 0.32)
    local_centers = ((0.0, -0.5 * width + 0.5 * arm, length, arm),
                     (-0.5 * length + 0.5 * arm, 0.0, arm, width))
    c, s = math.cos(obstacle_yaw), math.sin(obstacle_yaw)
    for local_x, local_y, arm_length, arm_width in local_centers:
        arm_xy = (
            obstacle_xy[0] + c * local_x - s * local_y,
            obstacle_xy[1] + s * local_x + c * local_y,
        )
        if oriented_rectangles_overlap(
            robot_xy, robot_yaw_rad, robot_length_m, robot_width_m,
            arm_xy, obstacle_yaw, arm_length, arm_width, padding_m,
        ):
            return True
    return False
