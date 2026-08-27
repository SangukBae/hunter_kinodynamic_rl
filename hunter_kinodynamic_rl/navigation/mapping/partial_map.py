"""Mission-frame online partial occupancy map (plan sections 5.4/5.5/5.6).

Internal state is the minimal array set from the plan: ``observed``,
``log_odds``, ``visited_count``, ``failure_count``, ``last_visit_step``.
Policy-facing channels (:meth:`PartialMap.channels`) are ALWAYS derived from
these, never stored independently, so ``occupied``/``free``/``unknown`` can
never drift out of sync with each other -- ``unknown`` is computed as
``NOT observed``, never as ``1 - occupied`` (that would silently fold
``free`` into ``unknown`` or vice versa, exactly what the spec forbids).

**Exhaustive 4-way partition** (code review finding): the plan's own
two-threshold formula (``occupied = log_odds >= occupied_threshold``,
``free = log_odds <= free_threshold``) leaves an OBSERVED cell whose
log-odds falls strictly between the two thresholds classified as neither
-- that state is real (a genuinely intermediate/still-settling log-odds
estimate) and was previously exported to RViz as an undocumented "occupancy
value 50" with no name anywhere in code or docs. It is now an explicit
fourth channel, :attr:`MapChannels.observed_uncertain`. Every cell belongs
to EXACTLY ONE of ``occupied | free | observed_uncertain | unknown`` --
this is the actual, complete Phase 1 map contract (not the 3-state one
originally documented).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from hunter_kinodynamic_rl.config.schema import MappingConfig
from hunter_kinodynamic_rl.navigation.mapping.raytracing import trace_clipped
from hunter_kinodynamic_rl.navigation.mapping.visited_map import (
    increment_saturating, normalize_counts, rasterize_circle_cells,
)

_MAX_RANGE_EPS = 1e-3


@dataclass(frozen=True)
class MapChannels:
    occupied: np.ndarray
    free: np.ndarray
    unknown: np.ndarray
    # OBSERVED but log-odds strictly between free_threshold and
    # occupied_threshold -- neither confidently free nor confidently
    # occupied. Disjoint from occupied/free/unknown; together the four
    # channels exhaustively partition every cell (see module docstring).
    observed_uncertain: np.ndarray
    visited: np.ndarray
    failure: np.ndarray
    # occupied, dilated by config.inflation_radius_m -- the "known
    # occupied/inflated cell" test a future Global-RL action mask (plan
    # section 8.4) will gate candidate subgoals against; always
    # occupied-inclusive (inflated == occupied when inflation_radius_m == 0).
    inflated: np.ndarray
    resolution_m: float
    origin_x: float
    origin_y: float


class PartialMap:
    def __init__(self, config: MappingConfig, size_cells: Optional[int] = None) -> None:
        self.config = config
        self.size_cells = int(size_cells) if size_cells is not None else int(config.mission_size_cells)
        if self.size_cells <= 0:
            raise ValueError(f"PartialMap size_cells must be > 0, got {self.size_cells}")
        self.resolution_m = float(config.resolution_m)
        # Mission-frame coordinate of the grid's lower-left corner -- the
        # map is centered on the mission origin (0, 0) so a robot starting
        # a mission always begins at the grid's center cell.
        self.origin_x = -(self.size_cells * self.resolution_m) / 2.0
        self.origin_y = -(self.size_cells * self.resolution_m) / 2.0

        h = w = self.size_cells
        self.observed = np.zeros((h, w), dtype=bool)
        self.log_odds = np.zeros((h, w), dtype=np.float32)
        self.visited_count = np.zeros((h, w), dtype=np.uint16)
        self.failure_count = np.zeros((h, w), dtype=np.uint8)
        self.last_visit_step = np.full((h, w), -1, dtype=np.int32)

    # ------------------------------------------------------------------ grid <-> mission-frame
    def world_to_cell(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        col = int(np.floor((x - self.origin_x) / self.resolution_m))
        row = int(np.floor((y - self.origin_y) / self.resolution_m))
        if not self.in_bounds(row, col):
            return None
        return row, col

    def cell_to_world(self, row: int, col: int) -> Tuple[float, float]:
        x = self.origin_x + (col + 0.5) * self.resolution_m
        y = self.origin_y + (row + 0.5) * self.resolution_m
        return x, y

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.size_cells and 0 <= col < self.size_cells

    # ------------------------------------------------------------------ LiDAR integration
    def _apply_log_odds(self, row: int, col: int, delta: float) -> None:
        updated = np.float32(self.log_odds[row, col]) + np.float32(delta)
        self.log_odds[row, col] = np.clip(updated, self.config.log_odds_min, self.config.log_odds_max)
        self.observed[row, col] = True

    def integrate_beam(
        self, sensor_origin_xy: Tuple[float, float], angle_mission: float, range_m: float,
        range_max: float, range_min: float = 0.0,
    ) -> None:
        """Integrate ONE beam (mission-frame sensor origin + absolute
        mission-frame angle). Beam classification (plan section 5.5 item 6,
        tightened by code review):

        - NaN or -Inf: no distance information at all -- skipped entirely,
          not even a free-space update.
        - +Inf: many real LiDAR drivers (and this project's own Gazebo
          bridge) publish a "no return" beam as ``+inf`` rather than
          exactly ``range_max`` -- treated identically to a genuine
          no-hit/max-range beam: FREE ray traced out to ``range_max``, no
          occupied endpoint. Previously incorrectly treated the same as
          NaN (skipped entirely), leaving open space permanently UNKNOWN.
        - ``range_m < range_min``: below the sensor's own minimum valid
          range -- skipped entirely (not trustworthy free-space
          information either).
        - ``range_min <= range_m < range_max``: a genuine hit -- FREE
          passthrough, OCCUPIED endpoint, but ONLY when the physical
          endpoint actually lies inside this map's bounds (see below).
        - ``range_m >= range_max``: FREE ray traced out to ``range_max``,
          no occupied endpoint.

        A hit whose PHYSICAL endpoint (before any grid-boundary clip)
        falls OUTSIDE this map's own bounds never marks an OCCUPIED cell
        (code review finding: clipping the ray to the grid boundary and
        then marking that clipped cell occupied fabricates a wall at the
        map edge for an obstacle that is actually further away, outside
        the map entirely). The boundary-clipped cell still gets marked
        FREE -- the ray was genuinely observed passing through it
        unobstructed -- exactly like the no-endpoint max-range case.
        """
        if math.isnan(range_m) or range_m == -math.inf:
            return
        if math.isinf(range_m):
            effective_r = float(range_max)
            is_hit = False
        else:
            if range_m < range_min:
                return
            effective_r = min(float(range_m), float(range_max))
            is_hit = range_m < (range_max - _MAX_RANGE_EPS)
        if effective_r <= 0.0:
            return

        sx, sy = sensor_origin_xy
        ex = sx + math.cos(angle_mission) * effective_r
        ey = sy + math.sin(angle_mission) * effective_r
        endpoint_in_bounds = self.world_to_cell(ex, ey) is not None

        start_cell = self._clamp_to_grid_index(sx, sy)
        end_cell = self._clamp_to_grid_index(ex, ey)
        cells = trace_clipped(start_cell[0], start_cell[1], end_cell[0], end_cell[1], self.size_cells, self.size_cells)
        if not cells:
            return

        for r, c in cells[:-1]:
            self._apply_log_odds(r, c, self.config.free_log_odds_delta)
        r_end, c_end = cells[-1]
        if is_hit and endpoint_in_bounds:
            self._apply_log_odds(r_end, c_end, self.config.occupied_log_odds_delta)
        else:
            self._apply_log_odds(r_end, c_end, self.config.free_log_odds_delta)

    def integrate_scan(
        self, sensor_origin_xy: Tuple[float, float], beam_angles_mission: np.ndarray,
        ranges: np.ndarray, range_max: float, range_min: float = 0.0,
    ) -> None:
        beam_angles_mission = np.asarray(beam_angles_mission, dtype=np.float64)
        ranges = np.asarray(ranges, dtype=np.float64)
        if beam_angles_mission.shape[0] != ranges.shape[0]:
            raise ValueError("beam_angles_mission and ranges must have the same length")
        for angle, r in zip(beam_angles_mission, ranges):
            self.integrate_beam(sensor_origin_xy, float(angle), float(r), range_max, range_min=range_min)

    def _clamp_to_grid_index(self, x: float, y: float) -> Tuple[int, int]:
        """Un-bounds-checked cell index (may lie outside the grid) --
        ``raytracing.trace_clipped`` performs the actual bounds clip; this
        just converts continuous mission-frame coordinates to the same
        integer cell basis :meth:`world_to_cell` uses."""
        col = int(np.floor((x - self.origin_x) / self.resolution_m))
        row = int(np.floor((y - self.origin_y) / self.resolution_m))
        return row, col

    # ------------------------------------------------------------------ visited / failure
    def record_visit(self, x: float, y: float, step: int, radius_m: Optional[float] = None) -> None:
        center = self.world_to_cell(x, y)
        if center is None:
            return
        radius_cells = (self.config.visit_radius_m if radius_m is None else radius_m) / self.resolution_m
        cells = [c for c in rasterize_circle_cells(center[0], center[1], radius_cells) if self.in_bounds(*c)]
        increment_saturating(self.visited_count, cells, self.config.visited_count_saturation)
        for r, c in cells:
            self.last_visit_step[r, c] = int(step)

    def record_failure(self, x: float, y: float, radius_m: Optional[float] = None) -> None:
        center = self.world_to_cell(x, y)
        if center is None:
            return
        radius_cells = (self.config.visit_radius_m if radius_m is None else radius_m) / self.resolution_m
        cells = [c for c in rasterize_circle_cells(center[0], center[1], radius_cells) if self.in_bounds(*c)]
        increment_saturating(self.failure_count, cells, self.config.failure_count_saturation)

    def _inflate(self, occupied: np.ndarray) -> np.ndarray:
        radius_cells = self.config.inflation_radius_m / self.resolution_m
        if radius_cells <= 0.0:
            return occupied.copy()
        inflated = occupied.copy()
        for r, c in zip(*np.nonzero(occupied)):
            for rr, cc in rasterize_circle_cells(int(r), int(c), radius_cells):
                if self.in_bounds(rr, cc):
                    inflated[rr, cc] = True
        return inflated

    # ------------------------------------------------------------------ policy-facing channels
    def channels(self) -> MapChannels:
        occupied = self.observed & (self.log_odds >= self.config.occupied_threshold)
        free = self.observed & (self.log_odds <= self.config.free_threshold)
        unknown = ~self.observed
        observed_uncertain = self.observed & ~occupied & ~free
        visited = normalize_counts(self.visited_count, self.config.visited_count_saturation)
        failure = normalize_counts(self.failure_count, self.config.failure_count_saturation)
        inflated = self._inflate(occupied)
        return MapChannels(
            occupied=occupied, free=free, unknown=unknown, observed_uncertain=observed_uncertain,
            visited=visited, failure=failure, inflated=inflated,
            resolution_m=self.resolution_m, origin_x=self.origin_x, origin_y=self.origin_y,
        )
