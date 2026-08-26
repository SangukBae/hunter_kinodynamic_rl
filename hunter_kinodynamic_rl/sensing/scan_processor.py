"""Raw LaserScan ranges -> fixed-width angular-bin distance array.

Pure numpy (no ROS message types touched here -- the ROS subscription
callback in env/simulation/environment_node.py converts a ``sensor_msgs/msg/
LaserScan`` to a plain ``ranges`` array + ``angle_min``/``angle_max`` before
calling this), so it is directly unit-testable and swappable for a different
LiDAR's angular layout.

Two independent sectors matter (mirrors drl_agent's obs_state / environment_
state split, see CLAUDE.md's State/Action Space section): a FRONT sector for
the RL observation (policy shouldn't need to look backward) and a FULL 360
sector for collision checking (a robot can be hit from any direction even if
it only steers using the front view).
"""

from __future__ import annotations

import math

import numpy as np


def bin_scan_sector(
    ranges: np.ndarray, angle_min: float, angle_increment: float,
    sector_center: float, sector_width: float, num_bins: int, max_range: float,
) -> np.ndarray:
    """Downsample ``ranges`` into ``num_bins`` angular bins covering
    ``[sector_center - sector_width/2, sector_center + sector_width/2]``.
    Each bin holds the MINIMUM (nearest obstacle) range among the raw scan
    samples that fall inside it -- never the mean (an average would hide a
    thin, dangerous obstacle behind wider free space in the same bin).
    Invalid samples (NaN/Inf/<=0) are treated as ``max_range`` (no
    return -> nothing detected within range, not "obstacle at distance 0")."""
    ranges = np.asarray(ranges, dtype=np.float32)
    n = ranges.shape[0]
    if n == 0:
        return np.full(num_bins, max_range, dtype=np.float32)

    angles = angle_min + angle_increment * np.arange(n, dtype=np.float32)
    # Wrap each sample's angular offset from sector_center into (-pi, pi].
    rel = np.mod(angles - sector_center + math.pi, 2 * math.pi) - math.pi
    half_width = sector_width / 2.0
    in_sector = np.abs(rel) <= half_width

    clean = np.where(np.isfinite(ranges) & (ranges > 0.0), ranges, max_range)
    clean = np.clip(clean, 0.0, max_range)

    out = np.full(num_bins, max_range, dtype=np.float32)
    if not np.any(in_sector):
        return out

    bin_width = sector_width / num_bins
    bin_idx = np.clip(
        ((rel[in_sector] + half_width) / bin_width).astype(np.int64), 0, num_bins - 1,
    )
    sector_ranges = clean[in_sector]
    for b in range(num_bins):
        mask = bin_idx == b
        if np.any(mask):
            out[b] = float(np.min(sector_ranges[mask]))
    return out


def front_and_full_state(
    ranges: np.ndarray, angle_min: float, angle_increment: float,
    num_bins: int, max_range: float, front_width_rad: float = math.pi,
) -> tuple:
    """Returns (obs_state, environment_state): front-``front_width_rad``
    sector (RL observation) and full 360 deg (collision checking), both
    binned into ``num_bins``."""
    obs_state = bin_scan_sector(ranges, angle_min, angle_increment, 0.0, front_width_rad, num_bins, max_range)
    environment_state = bin_scan_sector(ranges, angle_min, angle_increment, 0.0, 2 * math.pi, num_bins, max_range)
    return obs_state, environment_state
