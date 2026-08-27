"""Rolling local-map crop (plan section 5.2/8.5 "Rolling Local Map") -- a
fixed-size window of the mission :class:`PartialMap`, re-centered on the
robot's current mission-frame position every call. Cells outside the mission
map's own bounds are padded as UNKNOWN (never silently treated as free or
occupied) -- the window can legitimately extend past the mission grid near
its edges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from hunter_kinodynamic_rl.navigation.mapping.partial_map import PartialMap


@dataclass(frozen=True)
class RollingCrop:
    occupied: np.ndarray
    free: np.ndarray
    unknown: np.ndarray
    observed_uncertain: np.ndarray
    visited: np.ndarray
    failure: np.ndarray
    inflated: np.ndarray
    resolution_m: float
    origin_x: float
    origin_y: float
    size_cells: int


def crop_rolling(partial_map: PartialMap, center_xy: Tuple[float, float], size_cells: int) -> RollingCrop:
    size_cells = int(size_cells)
    if size_cells <= 0:
        raise ValueError(f"crop_rolling size_cells must be > 0, got {size_cells}")

    res = partial_map.resolution_m
    center_col = int(np.floor((center_xy[0] - partial_map.origin_x) / res))
    center_row = int(np.floor((center_xy[1] - partial_map.origin_y) / res))
    half = size_cells // 2
    row_start = center_row - half
    col_start = center_col - half

    occupied = np.zeros((size_cells, size_cells), dtype=bool)
    free = np.zeros((size_cells, size_cells), dtype=bool)
    unknown = np.ones((size_cells, size_cells), dtype=bool)
    observed_uncertain = np.zeros((size_cells, size_cells), dtype=bool)
    visited = np.zeros((size_cells, size_cells), dtype=np.float32)
    failure = np.zeros((size_cells, size_cells), dtype=np.float32)
    inflated = np.zeros((size_cells, size_cells), dtype=bool)

    src_channels = partial_map.channels()
    n = partial_map.size_cells

    # Overlap between the requested crop window [row_start, row_start+size)
    # and the source grid [0, n) -- copy only that intersection, leave the
    # rest at its UNKNOWN/zero default.
    src_row0, src_row1 = max(0, row_start), min(n, row_start + size_cells)
    src_col0, src_col1 = max(0, col_start), min(n, col_start + size_cells)
    if src_row0 < src_row1 and src_col0 < src_col1:
        dst_row0, dst_col0 = src_row0 - row_start, src_col0 - col_start
        dst_row1, dst_col1 = dst_row0 + (src_row1 - src_row0), dst_col0 + (src_col1 - src_col0)
        occupied[dst_row0:dst_row1, dst_col0:dst_col1] = src_channels.occupied[src_row0:src_row1, src_col0:src_col1]
        free[dst_row0:dst_row1, dst_col0:dst_col1] = src_channels.free[src_row0:src_row1, src_col0:src_col1]
        unknown[dst_row0:dst_row1, dst_col0:dst_col1] = src_channels.unknown[src_row0:src_row1, src_col0:src_col1]
        observed_uncertain[dst_row0:dst_row1, dst_col0:dst_col1] = \
            src_channels.observed_uncertain[src_row0:src_row1, src_col0:src_col1]
        visited[dst_row0:dst_row1, dst_col0:dst_col1] = src_channels.visited[src_row0:src_row1, src_col0:src_col1]
        failure[dst_row0:dst_row1, dst_col0:dst_col1] = src_channels.failure[src_row0:src_row1, src_col0:src_col1]
        inflated[dst_row0:dst_row1, dst_col0:dst_col1] = src_channels.inflated[src_row0:src_row1, src_col0:src_col1]

    crop_origin_x = partial_map.origin_x + col_start * res
    crop_origin_y = partial_map.origin_y + row_start * res
    return RollingCrop(
        occupied=occupied, free=free, unknown=unknown, observed_uncertain=observed_uncertain,
        visited=visited, failure=failure, inflated=inflated,
        resolution_m=res, origin_x=crop_origin_x, origin_y=crop_origin_y, size_cells=size_cells,
    )
