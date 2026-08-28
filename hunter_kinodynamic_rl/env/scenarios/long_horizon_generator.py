"""Long-horizon procedural world generator (plan section 7): a seed-
deterministic room/corridor/junction/loop/dead-end grid-lattice maze, built
on top of a random spanning tree over a coarse lattice graph.

Deliberately a SEPARATE generator from ``env/scenarios/procedural_generator.py``
(the existing small square-arena + circular-obstacle generator, which this
module does not replace or modify) -- a long-horizon maze needs box-wall
geometry and graph topology (dead ends, loops, junctions) that a circular-
obstacle layout cannot represent.

Algorithm (all seed-deterministic via one ``numpy.random.RandomState(seed)``
stream per attempt, mirroring ``procedural_generator.generate_scenario``'s
own "one shared rng, bounded retry loop, redraw the WHOLE attempt on any
infeasibility" structure):

1. Lay out a coarse ``grid_n x grid_n`` lattice of cell centers, spaced by
   ``pitch = size_m / grid_n`` (``grid_n`` floored so ``pitch >=
   corridor_width_max_m + wall_thickness_m`` always holds -- every drawn
   corridor width always fits within one cell pitch with wall clearance on
   both sides).
2. Build a random spanning tree over the lattice's 4-connected grid graph
   (randomized Kruskal) -- this alone guarantees full connectivity and
   naturally produces both dead ends (degree-1 nodes) and junctions
   (degree>=3 nodes).
3. Greedily add extra lattice edges to any current degree-1 node until the
   dead-end count is at or below ``dead_end_count_range``'s upper bound (or
   no more edges are available to add).
4. Draw and add ``loop_count_range``-many additional extra edges from
   whatever lattice edges remain -- each edge added anywhere past a
   spanning tree creates exactly one new cycle, i.e. one alternative route,
   which is why ``alternative_route_min_count`` is enforced purely via
   ``LongHorizonWorldConfig.validate()``'s ``loop_count_range[0] >=
   alternative_route_min_count`` cross-check rather than a separate route
   search here.
5. Steps 3 and 4 share ONE edge-addition mechanism, so a config whose
   ``dead_end_count_range`` forces many reduction edges CAN push the final
   cyclomatic number (== final loop/alternative-route count) above
   ``loop_count_range``'s upper bound for this particular draw -- rather
   than trying to reconcile the two analytically, the final counts are
   checked together and the WHOLE attempt is redrawn (continuing the same
   ``rng`` stream) if either misses its range, exactly like every other
   infeasibility path in ``generate_scenario``. A profile whose ranges are
   fundamentally incompatible for its ``size_m``/lattice size will simply
   exhaust ``generation_attempt_limit`` and raise.
6. A subset of lattice cells (``room_count_range``) is marked as "room"
   cells: any open edge touching a room cell gets the maximum possible
   passage width (``pitch - wall_thickness_m``) instead of a per-edge
   random draw from ``[corridor_width_min_m, corridor_width_max_m]`` -- the
   only effect "room" has on geometry (there is no separate room-square
   carving: since walls are placed only along lattice-cell BOUNDARIES,
   every cell's own interior is free space by construction regardless of
   room/corridor classification).
7. Every lattice-cell boundary edge becomes exactly one CLOSED wall segment
   (length == pitch) when there is no open connection to that neighbor (or
   it is a world-boundary edge), or exactly TWO wall segments flanking a
   centered passage of the chosen width when there is. ``occupancy`` is
   ALWAYS the exact rasterization of the returned ``wall_segments`` list
   (:func:`build_occupancy_from_segments` -- called internally, and
   independently callable by a test/caller to confirm the two never
   drift apart).
8. Start/goal are sampled from the (robot-footprint-inflated) free grid via
   one Dijkstra shortest-path-distance field per candidate start (see
   ``long_horizon_solvability.dijkstra_distances``) -- a goal is only
   accepted if its GEODESIC (not straight-line) distance from the start is
   >= ``start_goal_geodesic_min_m``. ``shortest_path_length_m`` on the
   returned world is exactly this same Dijkstra distance -- there is only
   one shortest-path implementation in this package
   (``long_horizon_solvability.shortest_path_length_m``), so recomputing it
   from the returned ``occupancy`` always matches BY CONSTRUCTION. This is
   REGRESSION protection (the generator and a later recomputation can never
   silently diverge, because they are the same code), not an independent
   check on whether that Dijkstra implementation itself is correct -- see
   ``tests/test_long_horizon_solvability.py``'s hand-authored-grid-fixture
   tests (known geometry, hand-computed expected distance) for genuine
   independent verification of the shortest-path algorithm itself.
9. ``require_ackermann_feasibility`` additionally requires
   ``long_horizon_solvability.is_ackermann_feasible_grid`` to accept the
   sampled start/goal/heading against the inflated occupancy grid.

``topology_metadata`` on the returned world is PRIVILEGED (plan section
3.2) -- generator/evaluation/teacher use only, never a policy observation
input (see ``tests/test_world_information_boundary.py``).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from hunter_kinodynamic_rl.common.geometry import wrap_to_pi
from hunter_kinodynamic_rl.config.schema import LongHorizonWorldConfig
from hunter_kinodynamic_rl.env.scenarios import long_horizon_solvability as solvability
from hunter_kinodynamic_rl.env.scenarios.long_horizon_world import LongHorizonWorld, WallSegment

_MIN_SEGMENT_LEN_M = 1e-3
_START_CANDIDATE_ATTEMPTS = 20


class LongHorizonSeedSplitError(ValueError):
    pass


def long_horizon_seed_split(seed: int, cfg: LongHorizonWorldConfig) -> str:
    """Which of train/validation/test ``seed`` belongs to -- raises if it
    falls in none of ``cfg``'s three (non-overlapping, per
    ``LongHorizonWorldConfig.validate()``) ranges. Independent of
    ``env.scenarios.procedural_generator.seed_split``/its own
    ``ScenarioConfig`` ranges -- the two generators' seed pools are
    unrelated."""
    if cfg.train_seed_range[0] <= seed <= cfg.train_seed_range[1]:
        return "train"
    if cfg.validation_seed_range[0] <= seed <= cfg.validation_seed_range[1]:
        return "validation"
    if cfg.test_seed_range[0] <= seed <= cfg.test_seed_range[1]:
        return "test"
    raise LongHorizonSeedSplitError(
        f"seed {seed} is outside all configured long_horizon_world train/validation/test ranges"
    )


_MODE_RANGE_ATTR = {
    "train": "train_seed_range",
    "validation": "validation_seed_range",
    "test": "test_seed_range",
}


class LongHorizonSeedScheduler:
    """Deterministic, resumable per-mode seed source for
    ``generate_long_horizon_world`` -- mirrors
    ``env.scenarios.seed_scheduler.SeedScheduler`` exactly (same
    ``(run_seed, mode, episode_index)``-seeded draw, same resume-from-
    ``episode_index`` contract), parameterized over
    ``LongHorizonWorldConfig`` instead of ``ScenarioConfig``. A caller that
    wants a real train/validation/test seed STREAM (not just membership
    checking on a seed it already has) should use this rather than drawing
    seeds itself -- every seed it produces already belongs to its own
    ``mode``'s pool by construction, so passing it straight into
    ``generate_long_horizon_world(..., mode=self.mode)`` can never trip the
    seed-pool-isolation check above."""

    def __init__(self, run_seed: int, cfg: LongHorizonWorldConfig, mode: str, episode_index: int = 0):
        if mode not in _MODE_RANGE_ATTR:
            raise ValueError(f"mode must be one of {sorted(_MODE_RANGE_ATTR)}, got {mode!r}")
        self.run_seed = int(run_seed)
        self.cfg = cfg
        self.mode = mode
        self.episode_index = int(episode_index)
        lo, hi = getattr(cfg, _MODE_RANGE_ATTR[mode])
        self._lo, self._hi = int(lo), int(hi)
        if self._hi < self._lo:
            raise ValueError(f"{mode} seed range is empty: [{self._lo}, {self._hi}]")

    def validate_explicit_seed(self, seed: int) -> None:
        try:
            actual = long_horizon_seed_split(seed, self.cfg)
        except LongHorizonSeedSplitError as e:
            raise LongHorizonSeedSplitError(f"seed {seed} is outside ALL configured seed ranges") from e
        if actual != self.mode:
            raise LongHorizonSeedSplitError(
                f"seed {seed} belongs to the {actual!r} pool, but this scheduler is running in "
                f"{self.mode!r} mode -- refusing to cross train/validation/test seed-pool isolation"
            )

    def next_seed(self) -> int:
        range_size = self._hi - self._lo + 1
        mode_salt = {"train": 0, "validation": 1, "test": 2}[self.mode]
        ss = np.random.SeedSequence([self.run_seed & 0xFFFFFFFF, mode_salt, self.episode_index & 0xFFFFFFFF])
        draw = int(ss.generate_state(1, dtype=np.uint32)[0]) % range_size
        seed = self._lo + draw
        self.episode_index += 1
        return seed

    def state_dict(self) -> dict:
        return {"run_seed": self.run_seed, "mode": self.mode, "episode_index": self.episode_index}

    @classmethod
    def from_state_dict(cls, state: dict, cfg: LongHorizonWorldConfig) -> "LongHorizonSeedScheduler":
        return cls(state["run_seed"], cfg, state["mode"], episode_index=state["episode_index"])


def max_possible_pitch_m(cfg: LongHorizonWorldConfig) -> float:
    """Upper bound on a single lattice cell's pitch across every possible
    ``size_m``/``corridor_width_max_m``/``wall_thickness_m`` combination --
    the worst case is the smallest allowed grid (``grid_n=3``, this
    module's own floor), giving ``pitch = size_m / 3``. Shared verbatim by
    ``config/schema.py``'s ``validate_wall_pool_capacity`` so the wall-pool
    capacity bound can never silently drift out of sync with this module's
    own grid-sizing formula."""
    return cfg.size_m / 3.0


def _grid_n(cfg: LongHorizonWorldConfig) -> int:
    """Lattice side length -- driven primarily by the TARGET topology
    counts (room/dead-end/loop ranges), not purely by how many
    ``corridor_width_max_m``-sized cells fit in ``size_m``.

    A random spanning tree over an ``n x n`` grid graph has a leaf
    (dead-end) FRACTION that stays roughly constant (~30-35%) as ``n``
    grows -- so sizing the lattice from ``size_m``/corridor width alone
    (as an earlier version of this function did) produces a lattice whose
    node count can vastly exceed what a small ``dead_end_count_range``/
    ``loop_count_range`` can plausibly reach: reducing e.g. ~25 raw leaves
    down to a configured maximum of 5 costs ~20 extra edges, each of which
    also inflates the cyclomatic number (this module's own ``loop_count``)
    by one, making ``loop_count_range``'s upper bound essentially
    unreachable. Instead, ``grid_n`` targets a node count proportional to
    ``room_count_range[1] + dead_end_count_range[1] + loop_count_range[1] +
    4`` (a small constant margin for ordinary junction nodes) -- large
    enough to plausibly host every configured feature, small enough that
    the natural leaf/cycle statistics are commensurate with the requested
    ranges (:func:`_build_spanning_topology`'s bounded retry loop absorbs
    the remaining per-attempt variance). Never larger than what ``size_m``
    can physically support at ``corridor_width_max_m`` (the floor this
    function used exclusively before) -- whichever bound is smaller wins,
    so ``pitch = size_m / grid_n`` still always satisfies ``pitch >=
    corridor_width_max_m + wall_thickness_m``."""
    size_max = max(3, int(math.floor(cfg.size_m / (cfg.corridor_width_max_m + cfg.wall_thickness_m))))
    target_features = cfg.room_count_range[1] + cfg.dead_end_count_range[1] + cfg.loop_count_range[1] + 4
    target = max(3, int(math.ceil(math.sqrt(target_features))))
    return min(target, size_max)


def max_possible_wall_segment_counts_by_class(
    cfg: LongHorizonWorldConfig, length_classes_m: Sequence[float],
) -> Dict[float, int]:
    """Worst-case count of wall segments that could need EACH of
    ``length_classes_m`` (shared verbatim by ``config/schema.py``'s
    ``validate_wall_pool_capacity`` -- same "computed from the generator's
    own formula, can never silently drift" reasoning as
    :func:`max_possible_pitch_m`).

    Code review finding this addresses: checking only the TOTAL segment
    count a lattice could produce is not enough --
    ``wall_segment_spawner.activate_walls`` requires a free slot in a
    segment's OWN snapped length class (never escalates to a larger one),
    and a real reproduction found a validated
    profile (``max_segments=100`` split evenly across 10 classes -> 10
    slots/class) failing at runtime because one specific class needed 24-33
    segments, far more than its 10-slot share, even though the grand TOTAL
    (100) was never exceeded.

    Derivation, from ``_build_walls``'s own emission rule for an
    ``n x n`` lattice (``n = _grid_n(cfg)``, ``pitch = size_m / n``):

    - ``boundary_and_closed_max = 4n + 2n(n-1)`` -- every world-boundary
      segment (always emitted, length == pitch) PLUS every interior edge
      CLOSED (the worst case for this category: each closed edge emits
      exactly 1 segment, also length == pitch). All of these need
      ``snap_up_to_class(pitch, length_classes_m)`` -- call it the "pitch
      class".
    - ``remainder_max = 4n(n-1)`` -- every interior edge OPEN instead (the
      worst case for THIS category: each open edge emits 2 flanking
      remainder segments). A remainder segment's length is
      ``(pitch - passage_width) / 2``, where ``passage_width`` is either
      ``pitch - wall_thickness_m`` (a room-adjacent edge, giving a FIXED,
      small remainder of ``wall_thickness_m / 2``) or a value drawn
      uniformly from ``[corridor_width_min_m, corridor_width_max_m]`` (an
      ordinary edge) -- see ``_build_walls``'s own ``passage_width()``
      closure. So remainder length ranges over
      ``[min(wall_thickness_m/2, (pitch-corridor_width_max_m)/2), (pitch-corridor_width_min_m)/2]``
      (clamped to >= 0). ANY class whose value falls at or below that
      range's upper bound (``max_remainder``) could be the smallest class
      some remainder draw snaps to, so -- without assuming anything about
      the RNG's actual distribution across attempts/seeds -- EVERY such
      class must independently be able to hold up to ``remainder_max``
      segments; the one class immediately above ``max_remainder`` (if any)
      is also credited defensively, to safely cover the boundary case
      where ``max_remainder`` falls strictly between two configured
      classes.

    A class that is reachable by BOTH categories (only possible with a
    coarse class list, e.g. one class covering both the pitch length AND
    the largest remainder length) is credited the SUM of both worst cases
    -- a deliberately conservative bound (in any ONE generated world,
    boundary-and-closed and remainder counts are complementary, since an
    interior edge is either open or closed, never both; summing avoids
    having to prove that non-overlap here, at the cost of some slack)."""
    n = _grid_n(cfg)
    boundary_and_closed_max = 4 * n + 2 * n * (n - 1)
    remainder_max = 4 * n * (n - 1)
    pitch = cfg.size_m / n

    min_remainder = max(0.0, min(
        cfg.wall_thickness_m / 2.0,
        (pitch - cfg.corridor_width_max_m) / 2.0,
    ))
    max_remainder = max(0.0, (pitch - cfg.corridor_width_min_m) / 2.0)
    # min_remainder is informational only (documents the true lower bound
    # of a remainder segment's length) -- every check below only needs the
    # UPPER bound, since "reachable by some remainder draw" only depends on
    # how large a class is relative to max_remainder.
    del min_remainder

    classes_sorted = sorted(length_classes_m)
    counts: Dict[float, int] = {c: 0 for c in classes_sorted}

    for c in classes_sorted:
        if c >= pitch - 1e-9:
            counts[c] += boundary_and_closed_max
            break

    for c in classes_sorted:
        if c <= max_remainder + 1e-9:
            counts[c] += remainder_max
        else:
            counts[c] += remainder_max
            break

    return counts


def _lattice_edges(grid_n: int) -> List[Tuple[int, int]]:
    """All 4-connected lattice edges as ``(ia, ib)`` node-index pairs with
    ``ia < ib`` always (node index == ``i * grid_n + j``)."""
    edges: List[Tuple[int, int]] = []
    for i in range(grid_n):
        for j in range(grid_n):
            a = i * grid_n + j
            if i + 1 < grid_n:
                edges.append((a, (i + 1) * grid_n + j))
            if j + 1 < grid_n:
                edges.append((a, i * grid_n + j + 1))
    return edges


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[ra] = rb
        return True


def _shuffled(rng: np.random.RandomState, items: list) -> list:
    """Deterministic-given-``rng`` shuffle that never risks numpy's
    array-conversion quirks on a list of tuples (unlike ``rng.shuffle``
    called directly on such a list)."""
    perm = rng.permutation(len(items))
    return [items[int(k)] for k in perm]


def _build_spanning_topology(
    rng: np.random.RandomState, grid_n: int, cfg: LongHorizonWorldConfig,
) -> Optional[Tuple[Set[Tuple[int, int]], int, int]]:
    """One attempt's graph topology: a random spanning tree, greedily
    dead-end-reduced then loop-augmented (see module docstring steps 2-5).
    Returns ``(tree_edges, dead_end_count, cyclomatic_number)`` -- both
    counts already reflect every edge added by every phase -- or ``None``
    if the resulting counts miss their configured ranges."""
    num_nodes = grid_n * grid_n
    all_edges = _lattice_edges(grid_n)
    shuffled_edges = _shuffled(rng, all_edges)

    uf = _UnionFind(num_nodes)
    tree_edges: Set[Tuple[int, int]] = set()
    for ia, ib in shuffled_edges:
        if uf.union(ia, ib):
            tree_edges.add((ia, ib))

    degree = [0] * num_nodes
    for ia, ib in tree_edges:
        degree[ia] += 1
        degree[ib] += 1

    remaining_pool = [e for e in shuffled_edges if e not in tree_edges]
    remaining_pool = _shuffled(rng, remaining_pool)

    dead_lo, dead_hi = cfg.dead_end_count_range

    def leaf_count() -> int:
        return sum(1 for d in degree if d == 1)

    while leaf_count() > dead_hi:
        # Prefer an edge connecting TWO current leaves to each other --
        # removes 2 leaves for the cost of 1 added (cyclomatic-increasing)
        # edge, roughly halving how many extra edges dead-end reduction
        # needs versus always falling back to a leaf-to-non-leaf edge
        # (which only removes 1 leaf per edge) -- directly helps keep the
        # final cyclomatic number inside loop_count_range (see module
        # docstring step 5).
        found_idx = None
        for k, (ia, ib) in enumerate(remaining_pool):
            if degree[ia] == 1 and degree[ib] == 1:
                found_idx = k
                break
        if found_idx is None:
            for k, (ia, ib) in enumerate(remaining_pool):
                if degree[ia] == 1 or degree[ib] == 1:
                    found_idx = k
                    break
        if found_idx is None:
            break
        ia, ib = remaining_pool.pop(found_idx)
        tree_edges.add((ia, ib))
        degree[ia] += 1
        degree[ib] += 1

    loop_lo, loop_hi = cfg.loop_count_range
    loop_target = int(rng.randint(loop_lo, loop_hi + 1))
    added = 0
    for ia, ib in remaining_pool:
        if added >= loop_target:
            break
        tree_edges.add((ia, ib))
        degree[ia] += 1
        degree[ib] += 1
        added += 1

    dead_end_count = leaf_count()
    cyclomatic = len(tree_edges) - num_nodes + 1
    if not (dead_lo <= dead_end_count <= dead_hi):
        return None
    if not (loop_lo <= cyclomatic <= loop_hi):
        return None
    return tree_edges, dead_end_count, cyclomatic


def rasterize_wall_segment(
    seg: WallSegment, occupancy: np.ndarray, resolution_m: float, origin_x: float, origin_y: float,
) -> None:
    """In-place: marks every ``occupancy`` cell whose FOOTPRINT overlaps
    ``seg``'s oriented box as ``True``. General (any ``yaw_rad``), even
    though this module's own generator only ever emits axis-aligned
    (``yaw_rad in {0, pi/2}``) segments -- kept general so the function is
    honest about ``WallSegment``'s own declared contract.

    Deliberately a CONSERVATIVE (cell-footprint, not cell-CENTER) overlap
    test: each half-extent below is inflated by ``resolution_m / 2`` before
    the local-frame box test, i.e. this is a bare point-in-box test against
    the box grown by half a cell in every direction -- equivalent to asking
    "does this cell's own square footprint touch the wall's footprint",
    not "does this cell's center point fall strictly inside the wall".
    Without this, a wall THINNER than ``resolution_m`` (e.g. the default
    ``wall_thickness_m=0.2`` vs. ``resolution_m=0.25``) can rasterize to
    ZERO occupied cells: grid-cell centers are spaced a full
    ``resolution_m`` apart, so a sub-resolution-thickness wall can fall
    entirely BETWEEN two consecutive cell centers with neither one inside
    it. The trade-off is a wall growing by up to one ``resolution_m`` cell
    in every direction versus its geometrically exact footprint -- accepted
    here as conservative (a wall can only become slightly thicker/longer
    than declared, never disappear or open an unintended gap)."""
    half_len = seg.length_m / 2.0 + resolution_m / 2.0
    half_thick = seg.thickness_m / 2.0 + resolution_m / 2.0
    cos_y, sin_y = math.cos(seg.yaw_rad), math.sin(seg.yaw_rad)
    extent = math.hypot(half_len, half_thick)
    h, w = occupancy.shape
    col_lo = max(0, int(math.floor((seg.center_x_m - extent - origin_x) / resolution_m)))
    col_hi = min(w - 1, int(math.ceil((seg.center_x_m + extent - origin_x) / resolution_m)))
    row_lo = max(0, int(math.floor((seg.center_y_m - extent - origin_y) / resolution_m)))
    row_hi = min(h - 1, int(math.ceil((seg.center_y_m + extent - origin_y) / resolution_m)))
    for row in range(row_lo, row_hi + 1):
        wy = origin_y + (row + 0.5) * resolution_m
        dy0 = wy - seg.center_y_m
        for col in range(col_lo, col_hi + 1):
            wx = origin_x + (col + 0.5) * resolution_m
            dx0 = wx - seg.center_x_m
            local_x = dx0 * cos_y + dy0 * sin_y
            local_y = -dx0 * sin_y + dy0 * cos_y
            if abs(local_x) <= half_len and abs(local_y) <= half_thick:
                occupancy[row, col] = True


def build_occupancy_from_segments(
    wall_segments: List[WallSegment], resolution_m: float, origin_xy: Tuple[float, float], n_cells: int,
) -> np.ndarray:
    """Rebuilds an occupancy grid from a ``WallSegment`` list alone -- the
    single source of truth :func:`generate_long_horizon_world` itself uses
    internally, and directly callable by a test to confirm a returned
    world's ``occupancy`` is EXACTLY this rasterization of its own
    ``wall_segments`` (never merely similar)."""
    origin_x, origin_y = origin_xy
    occupancy = np.zeros((n_cells, n_cells), dtype=bool)
    for seg in wall_segments:
        rasterize_wall_segment(seg, occupancy, resolution_m, origin_x, origin_y)
    return occupancy


def _build_walls(
    grid_n: int, pitch: float, half: float, tree_edges: Set[Tuple[int, int]],
    room_indices: Set[int], cfg: LongHorizonWorldConfig, rng: np.random.RandomState,
) -> List[WallSegment]:
    segments: List[WallSegment] = []
    counter = [0]

    def node_xy(i: int, j: int) -> Tuple[float, float]:
        return -half + (i + 0.5) * pitch, -half + (j + 0.5) * pitch

    def idx(i: int, j: int) -> int:
        return i * grid_n + j

    def add(cx: float, cy: float, length: float, yaw: float, semantic: str) -> None:
        if length <= _MIN_SEGMENT_LEN_M:
            return
        segments.append(WallSegment(
            center_x_m=cx, center_y_m=cy, length_m=length, thickness_m=cfg.wall_thickness_m,
            height_m=cfg.wall_height_m, yaw_rad=yaw, semantic=semantic, entity_id=f"wseg_{counter[0]}",
        ))
        counter[0] += 1

    def passage_width(ia: int, ib: int) -> float:
        if ia in room_indices or ib in room_indices:
            return pitch - cfg.wall_thickness_m
        drawn = float(rng.uniform(cfg.corridor_width_min_m, cfg.corridor_width_max_m))
        return min(drawn, pitch - cfg.wall_thickness_m)

    # Interior edges between (i, j) and (i+1, j): shared boundary is a
    # VERTICAL line (yaw = pi/2) at the midpoint x between the two centers.
    for i in range(grid_n - 1):
        for j in range(grid_n):
            ia, ib = idx(i, j), idx(i + 1, j)
            ax, ay = node_xy(i, j)
            edge_x, edge_y = ax + pitch / 2.0, ay
            semantic = "room_boundary" if (ia in room_indices or ib in room_indices) else "corridor"
            if (ia, ib) not in tree_edges:
                add(edge_x, edge_y, pitch, math.pi / 2.0, semantic)
                continue
            passage = passage_width(ia, ib)
            remainder = (pitch - passage) / 2.0
            add(edge_x, edge_y - passage / 2.0 - remainder / 2.0, remainder, math.pi / 2.0, semantic)
            add(edge_x, edge_y + passage / 2.0 + remainder / 2.0, remainder, math.pi / 2.0, semantic)

    # Interior edges between (i, j) and (i, j+1): shared boundary is a
    # HORIZONTAL line (yaw = 0) at the midpoint y between the two centers.
    for i in range(grid_n):
        for j in range(grid_n - 1):
            ia, ib = idx(i, j), idx(i, j + 1)
            ax, ay = node_xy(i, j)
            edge_x, edge_y = ax, ay + pitch / 2.0
            semantic = "room_boundary" if (ia in room_indices or ib in room_indices) else "corridor"
            if (ia, ib) not in tree_edges:
                add(edge_x, edge_y, pitch, 0.0, semantic)
                continue
            passage = passage_width(ia, ib)
            remainder = (pitch - passage) / 2.0
            add(edge_x - passage / 2.0 - remainder / 2.0, edge_y, remainder, 0.0, semantic)
            add(edge_x + passage / 2.0 + remainder / 2.0, edge_y, remainder, 0.0, semantic)

    # World boundary -- always closed, one full-pitch segment per row/column.
    for j in range(grid_n):
        _, ay = node_xy(0, j)
        add(-half, ay, pitch, math.pi / 2.0, "boundary")
        add(half, ay, pitch, math.pi / 2.0, "boundary")
    for i in range(grid_n):
        ax, _ = node_xy(i, 0)
        add(ax, -half, pitch, 0.0, "boundary")
        add(ax, half, pitch, 0.0, "boundary")

    return segments


def _sample_start_yaw(
    rng: np.random.RandomState, cfg: LongHorizonWorldConfig,
    start_x: float, start_y: float, goal_x: float, goal_y: float,
    free: np.ndarray, resolution_m: float, origin_x: float, origin_y: float,
) -> Optional[float]:
    h, w = free.shape

    def is_free(x: float, y: float) -> bool:
        cell = solvability.world_to_cell(x, y, resolution_m, origin_x, origin_y, h, w)
        return cell is not None and bool(free[cell])

    def check(yaw: float) -> bool:
        fx = start_x + cfg.start_yaw_front_safety_distance_m * math.cos(yaw)
        fy = start_y + cfg.start_yaw_front_safety_distance_m * math.sin(yaw)
        return is_free(fx, fy)

    goal_heading = math.atan2(goal_y - start_y, goal_x - start_x)
    for _ in range(cfg.start_yaw_sampling_attempts):
        if rng.uniform(0.0, 1.0) < 0.5:
            candidate = wrap_to_pi(goal_heading + rng.uniform(-0.6, 0.6))
        else:
            candidate = float(rng.uniform(-math.pi, math.pi))
        if check(candidate):
            return candidate

    for k in range(36):
        candidate = wrap_to_pi(-math.pi + 2.0 * math.pi * k / 36)
        if check(candidate):
            return candidate
    return None


def generate_long_horizon_world(
    seed: int, cfg: LongHorizonWorldConfig, robot_radius_m: float,
    min_turning_radius_m: Optional[float] = None, wheelbase_m: Optional[float] = None,
    mode: Optional[str] = None,
) -> LongHorizonWorld:
    """Retries (redrawing the whole attempt from the SAME seeded ``rng``
    stream) until a world satisfying every configured constraint is found,
    or raises ``RuntimeError`` after ``cfg.generation_attempt_limit``
    attempts. ``min_turning_radius_m``/``wheelbase_m`` are required
    (raises immediately otherwise, no silent fallback) whenever
    ``cfg.require_ackermann_feasibility`` -- mirrors
    ``procedural_generator.generate_scenario``'s own
    ``feasibility_check == "ackermann"`` contract.

    ``mode`` (code review: previously the seed-pool separation
    ``long_horizon_seed_split``/``LongHorizonWorldConfig.validate()``
    establish was never actually ENFORCED anywhere a caller could reach --
    ``generate_long_horizon_world(seed=999, cfg=...)`` happily generated a
    world for a seed outside every configured range, or one from the WRONG
    pool for the caller's intended use, with nothing to stop it) --
    ``"train"``/``"validation"``/``"test"``, or ``None`` (default, no
    check, e.g. a manual/ad hoc verification script that doesn't care
    which pool a seed belongs to). When given, ``seed`` MUST belong to
    ``cfg``'s configured range for that mode -- raises
    :class:`LongHorizonSeedSplitError` immediately otherwise, BEFORE any
    generation work happens. A trainer/evaluator that cares about pool
    separation (the whole point of having separate pools at all) should
    always pass its own ``mode`` here rather than relying on the caller
    having drawn the seed correctly upstream -- see
    :class:`LongHorizonSeedScheduler` for a ready-made deterministic,
    resumable per-mode seed source that always satisfies this."""
    if mode is not None:
        actual = long_horizon_seed_split(seed, cfg)
        if actual != mode:
            raise LongHorizonSeedSplitError(
                f"generate_long_horizon_world: seed {seed} belongs to the {actual!r} pool, but mode={mode!r} "
                "was requested -- refusing to cross train/validation/test seed-pool isolation"
            )
    if cfg.size_m < 3.0 * (cfg.corridor_width_max_m + cfg.wall_thickness_m):
        raise RuntimeError(
            f"generate_long_horizon_world: size_m={cfg.size_m} is too small for even a 3x3 lattice at "
            f"corridor_width_max_m={cfg.corridor_width_max_m} + wall_thickness_m={cfg.wall_thickness_m} "
            "-- increase size_m or reduce corridor_width_max_m/wall_thickness_m"
        )
    if cfg.require_ackermann_feasibility and (
        min_turning_radius_m is None or min_turning_radius_m <= 0.0
        or wheelbase_m is None or wheelbase_m <= 0.0
    ):
        raise RuntimeError(
            "generate_long_horizon_world: long_horizon_world.require_ackermann_feasibility=true requires "
            f"the caller to pass a valid min_turning_radius_m/wheelbase_m (got "
            f"min_turning_radius_m={min_turning_radius_m}, wheelbase_m={wheelbase_m}) -- derive both "
            "from the active RobotConfig; refusing to silently skip the feasibility check"
        )

    grid_n = _grid_n(cfg)
    pitch = cfg.size_m / grid_n
    half = cfg.size_m / 2.0
    num_nodes = grid_n * grid_n
    n_cells = max(2, int(round(cfg.size_m / cfg.resolution_m)))
    origin_x = origin_y = -half

    rng = np.random.RandomState(seed)

    for attempt in range(cfg.generation_attempt_limit):
        topology = _build_spanning_topology(rng, grid_n, cfg)
        if topology is None:
            continue
        tree_edges, dead_end_count, cyclomatic = topology

        room_lo, room_hi = cfg.room_count_range
        room_hi = min(room_hi, num_nodes)
        room_lo = min(room_lo, room_hi)
        room_count = int(rng.randint(room_lo, room_hi + 1)) if room_hi > 0 else 0
        room_indices: Set[int] = set(
            int(k) for k in rng.choice(num_nodes, size=room_count, replace=False)
        ) if room_count > 0 else set()

        wall_segments = _build_walls(grid_n, pitch, half, tree_edges, room_indices, cfg, rng)
        occupancy = build_occupancy_from_segments(wall_segments, cfg.resolution_m, (origin_x, origin_y), n_cells)

        free = ~solvability.inflate_occupancy(occupancy, cfg.resolution_m, robot_radius_m)
        free_cells = np.argwhere(free)
        if free_cells.shape[0] < 2:
            continue

        found_pair = False
        start_x = start_y = goal_x = goal_y = 0.0
        path_len = 0.0
        for _try in range(_START_CANDIDATE_ATTEMPTS):
            pick = int(rng.randint(0, free_cells.shape[0]))
            sr, sc = int(free_cells[pick, 0]), int(free_cells[pick, 1])
            dist = solvability.dijkstra_distances(free, cfg.resolution_m, (sr, sc))
            reachable_far = np.argwhere(np.isfinite(dist) & (dist >= cfg.start_goal_geodesic_min_m))
            if reachable_far.shape[0] == 0:
                continue
            gpick = int(rng.randint(0, reachable_far.shape[0]))
            gr, gc = int(reachable_far[gpick, 0]), int(reachable_far[gpick, 1])
            start_x, start_y = solvability.cell_to_world(sr, sc, cfg.resolution_m, origin_x, origin_y)
            goal_x, goal_y = solvability.cell_to_world(gr, gc, cfg.resolution_m, origin_x, origin_y)
            path_len = float(dist[gr, gc])
            found_pair = True
            break
        if not found_pair:
            continue

        yaw = _sample_start_yaw(rng, cfg, start_x, start_y, goal_x, goal_y, free, cfg.resolution_m, origin_x, origin_y)
        if yaw is None:
            continue

        if cfg.require_ackermann_feasibility:
            feasible = solvability.is_ackermann_feasible_grid(
                start_x, start_y, yaw, goal_x, goal_y, cfg.goal_radius_m,
                occupancy, cfg.resolution_m, (origin_x, origin_y), robot_radius_m,
                min_turning_radius_m, wheelbase_m,
            )
            if not feasible:
                continue

        topology_metadata: Dict = {
            "grid_n": grid_n,
            "pitch_m": pitch,
            "room_count": len(room_indices),
            # A plain sorted list here is fine -- LongHorizonWorld.__post_init__'s
            # _deep_freeze recursively converts every nested list/tuple to
            # an immutable tuple regardless of what this dict contains, so
            # this module no longer needs its own tuple-conversion.
            "room_cells": sorted(room_indices),
            "dead_end_count": dead_end_count,
            "loop_count": cyclomatic,
            "alternative_route_count": cyclomatic,
            "num_nodes": num_nodes,
            "num_edges": len(tree_edges),
            "attempt": attempt,
        }
        return LongHorizonWorld(
            seed=seed, occupancy=occupancy, resolution_m=cfg.resolution_m, origin_xy=(origin_x, origin_y),
            wall_segments=wall_segments, start_pose=(start_x, start_y, yaw), goal_pose=(goal_x, goal_y),
            shortest_path_length_m=path_len, topology_metadata=topology_metadata,
        )

    raise RuntimeError(
        f"generate_long_horizon_world: no feasible long-horizon world found for seed={seed} after "
        f"{cfg.generation_attempt_limit} attempts (size_m={cfg.size_m}, "
        f"dead_end_count_range={cfg.dead_end_count_range}, loop_count_range={cfg.loop_count_range} may be "
        "mutually incompatible for this lattice size -- see this module's docstring step 5 -- or "
        f"start_goal_geodesic_min_m={cfg.start_goal_geodesic_min_m}/require_ackermann_feasibility too "
        "demanding for the drawn corridor widths)"
    )
