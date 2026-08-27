"""Visited/failure footprint rasterization (plan section 5.6) -- pure grid
math shared by :class:`~hunter_kinodynamic_rl.navigation.mapping.partial_map.PartialMap`'s
``record_visit``/``record_failure``. Kept separate from ``partial_map.py`` so
the rasterization geometry itself is independently unit-testable without a
full :class:`PartialMap` instance.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def rasterize_circle_cells(center_row: int, center_col: int, radius_cells: float) -> List[Tuple[int, int]]:
    """All grid cells whose center lies within ``radius_cells`` of
    ``(center_row, center_col)`` -- NOT clipped to any grid bounds (the
    caller intersects with its own array shape); a nonpositive radius still
    yields the single center cell (a visit/failure mark always covers at
    least its own cell)."""
    radius_cells = max(0.0, float(radius_cells))
    r = int(np.ceil(radius_cells))
    cells: List[Tuple[int, int]] = []
    r2 = radius_cells * radius_cells
    for dr in range(-r, r + 1):
        for dc in range(-r, r + 1):
            if dr * dr + dc * dc <= r2 or (dr == 0 and dc == 0):
                cells.append((center_row + dr, center_col + dc))
    return cells


def normalize_counts(counts: np.ndarray, saturation: int) -> np.ndarray:
    """``counts`` (uint) -> ``float32`` in ``[0, 1]``, saturation-normalized
    (never divides by the observed max, which would make the same raw visit
    count mean a different normalized value across two different maps)."""
    saturation = max(1, int(saturation))
    return np.clip(counts.astype(np.float32) / float(saturation), 0.0, 1.0)


def increment_saturating(array: np.ndarray, cells: List[Tuple[int, int]], max_value: int) -> None:
    """In-place saturating increment of ``array`` at ``cells`` (already
    filtered to in-bounds indices by the caller) -- never wraps past
    ``max_value``/the array dtype's own max."""
    dtype_max = np.iinfo(array.dtype).max
    cap = min(int(max_value), int(dtype_max))
    for r, c in cells:
        if array[r, c] < cap:
            array[r, c] += 1
