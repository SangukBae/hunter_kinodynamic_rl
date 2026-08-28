"""Long-horizon difficulty curriculum (plan section 7.5) -- pure config
overlays over ``LongHorizonWorldConfig``, one per named level. No RL, no
learned staging logic: a caller (a future trainer) picks the level, this
module returns the resulting config, ``long_horizon_generator`` does the
rest. Train/validation/test seed pools are UNCHANGED by level (plan section
7.5: "curriculum 여부와 관계없이 train/validation/test seed pool은 완전히
분리한다") -- every level overlay leaves ``train_seed_range``/
``validation_seed_range``/``test_seed_range`` exactly as the base config's.

Level 6 ("dynamic obstacle 추가") is explicitly OUT of this module's scope:
this package's Phase 3 delivers only the STATIC long-horizon world (rooms/
corridors/junctions/loops/dead-ends -- no Global RL, per this phase's own
scope boundary). Level 6's overlay tightens the STATIC geometry to its
hardest setting and leaves dynamic-obstacle integration (this package's
existing ``env/humans`` dynamic-obstacle system, or a future Global-RL-era
combination) to whichever later phase actually wires it in -- attempting to
select it is not an error, but the caller must not assume dynamic obstacles
appear from this module alone.
"""

from __future__ import annotations

import dataclasses
from typing import Dict

from hunter_kinodynamic_rl.config.schema import ConfigError, LongHorizonWorldConfig

MIN_LEVEL = 1
MAX_LEVEL = 6

# Overrides applied ON TOP OF whatever LongHorizonWorldConfig the caller
# already has (dataclasses.replace) -- every field not listed here is left
# untouched, in particular size_m/resolution_m/wall_thickness_m/
# generation_attempt_limit/every seed-range field.
_LEVEL_OVERRIDES: Dict[int, Dict] = {
    # Level 1: wide corridors, short paths, close to no dead ends. A grid-
    # lattice spanning tree always has >= 2 leaves (long_horizon_generator's
    # own docstring step 2), so a literal dead_end_count_range=[0, 0] would
    # need an unbounded number of loop-closing edges to eliminate every
    # last one -- [0, 2] is the smallest range this algorithm can satisfy
    # reliably within generation_attempt_limit while still capturing the
    # plan's "no dead end" intent as "as close to none as this maze
    # topology can plausibly produce".
    1: dict(
        corridor_width_min_m=3.0, corridor_width_max_m=4.0,
        dead_end_count_range=[0, 2], loop_count_range=[0, 3],
        alternative_route_min_count=0,
        start_goal_geodesic_min_m=8.0,
    ),
    # Level 2: junctions plus a small number of dead ends.
    2: dict(
        corridor_width_min_m=2.5, corridor_width_max_m=4.0,
        dead_end_count_range=[1, 3], loop_count_range=[0, 3],
        alternative_route_min_count=0,
        start_goal_geodesic_min_m=12.0,
    ),
    # Level 3: loops and an explicit alternative-route requirement.
    3: dict(
        corridor_width_min_m=2.5, corridor_width_max_m=4.0,
        dead_end_count_range=[1, 4], loop_count_range=[1, 5],
        alternative_route_min_count=1,
        start_goal_geodesic_min_m=16.0,
    ),
    # Level 4: long dead-end/backtracking required.
    4: dict(
        corridor_width_min_m=2.0, corridor_width_max_m=3.5,
        dead_end_count_range=[3, 6], loop_count_range=[1, 3],
        alternative_route_min_count=1,
        start_goal_geodesic_min_m=20.0,
    ),
    # Level 5: narrow, Ackermann-feasibility-binding corridors + rooms.
    5: dict(
        corridor_width_min_m=2.0, corridor_width_max_m=2.5,
        dead_end_count_range=[2, 5], loop_count_range=[1, 4],
        alternative_route_min_count=1,
        room_count_range=[4, 10],
        start_goal_geodesic_min_m=24.0,
        require_ackermann_feasibility=True,
    ),
    # Level 6: hardest static geometry -- dynamic-obstacle integration is
    # OUT of this module's scope (see module docstring).
    6: dict(
        corridor_width_min_m=2.0, corridor_width_max_m=2.5,
        dead_end_count_range=[3, 6], loop_count_range=[2, 5],
        alternative_route_min_count=1,
        room_count_range=[4, 10],
        start_goal_geodesic_min_m=28.0,
        require_ackermann_feasibility=True,
    ),
}


def level_overrides(level: int) -> Dict:
    if level not in _LEVEL_OVERRIDES:
        raise ConfigError(f"long_horizon_curriculum level must be in [{MIN_LEVEL}, {MAX_LEVEL}], got {level}")
    return dict(_LEVEL_OVERRIDES[level])


def apply_level(cfg: LongHorizonWorldConfig, level: int) -> LongHorizonWorldConfig:
    """Returns a NEW ``LongHorizonWorldConfig`` (``cfg`` is never mutated)
    with ``level``'s overlay applied; raises via
    :meth:`LongHorizonWorldConfig.validate` if the resulting config is
    internally inconsistent (e.g. a level's ``loop_count_range`` lower
    bound below ``cfg.alternative_route_min_count`` when the level doesn't
    also override that field)."""
    overrides = level_overrides(level)
    resolved = dataclasses.replace(cfg, **overrides)
    resolved.validate()
    return resolved
