"""Bresenham grid ray tracing (plan section 5.5) -- pure integer-cell
geometry, no map/log-odds state here.
"""

from __future__ import annotations

from typing import List, Optional, Tuple


def bresenham_line(r0: int, c0: int, r1: int, c1: int) -> List[Tuple[int, int]]:
    """All grid cells (row, col) on the line from (r0, c0) to (r1, c1),
    inclusive of both endpoints. Standard integer Bresenham -- deterministic,
    symmetric regardless of direction."""
    r0, c0, r1, c1 = int(r0), int(c0), int(r1), int(c1)
    cells: List[Tuple[int, int]] = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    r, c = r0, c0
    while True:
        cells.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
    return cells


def clip_ray_to_bounds(
    r0: int, c0: int, r1: int, c1: int, height: int, width: int,
) -> Optional[Tuple[int, int, int, int]]:
    """Clip the segment (r0,c0)->(r1,c1) to the grid bounds
    ``[0, height) x [0, width)`` using a parametric (Liang-Barsky style)
    clip so a ray whose endpoint lies outside the map is safely shortened to
    the last in-bounds point along its own direction, rather than either
    raising or silently tracing out-of-bounds indices. Returns ``None`` if
    the segment never intersects the grid at all."""
    r0f, c0f, r1f, c1f = float(r0), float(c0), float(r1), float(c1)
    dr = r1f - r0f
    dc = c1f - c0f
    t0, t1 = 0.0, 1.0

    def _clip(p: float, q: float, t0: float, t1: float) -> Optional[Tuple[float, float]]:
        if p == 0.0:
            if q < 0.0:
                return None
            return t0, t1
        t = q / p
        if p < 0.0:
            if t > t1:
                return None
            if t > t0:
                t0 = t
        else:
            if t < t0:
                return None
            if t < t1:
                t1 = t
        return t0, t1

    for p, q in (
        (-dr, r0f - 0.0), (dr, (height - 1) - r0f),
        (-dc, c0f - 0.0), (dc, (width - 1) - c0f),
    ):
        clipped = _clip(p, q, t0, t1)
        if clipped is None:
            return None
        t0, t1 = clipped

    if t0 > t1:
        return None

    cr0 = round(r0f + t0 * dr)
    cc0 = round(c0f + t0 * dc)
    cr1 = round(r0f + t1 * dr)
    cc1 = round(c0f + t1 * dc)
    return int(cr0), int(cc0), int(cr1), int(cc1)


def trace_clipped(r0: int, c0: int, r1: int, c1: int, height: int, width: int) -> List[Tuple[int, int]]:
    """Trace the ray from (r0,c0) toward (r1,c1), clipped to grid bounds
    first -- the safe entry point ``partial_map.py`` uses for every beam.
    Empty list if the ray never touches the grid at all."""
    clipped = clip_ray_to_bounds(r0, c0, r1, c1, height, width)
    if clipped is None:
        return []
    cr0, cc0, cr1, cc1 = clipped
    return bresenham_line(cr0, cc0, cr1, cc1)
