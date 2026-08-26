"""section item-6: exact closest-approach / threshold-crossing math for a
point moving AFFINELY (linearly in a parameter ``s in [0, 1]``) relative to
the origin -- the shared primitive behind every "continuous" (segment-based,
not just sampled-endpoint) risk check in this package (dynamic-obstacle
clearance/TTC in ``future_clearance.py``/``ttc.py``).

Why this is needed: a :class:`~hunter_kinodynamic_rl.dynamics.ackermann_rollout.Rollout`
only ever samples the ego trajectory at discrete instants
(``t in {0, dt, 2*dt, ...}``); a :class:`~hunter_kinodynamic_rl.risk.labels.DynamicObstacle`
moves at an exact constant velocity. Checking ONLY the sampled instants (the
pre-item-6 behavior) can miss a genuine collision that occurs strictly
BETWEEN two consecutive samples where both endpoints individually show
clearance -- e.g. a fast-closing obstacle that grazes the ego's path only at
the segment's midpoint. Treating the EGO's position between two consecutive
samples as linearly interpolated (an honest approximation of the true
curved Ackermann arc, exact in the limit as ``dt -> 0`` and already the
best information available from just two samples) makes the RELATIVE
position between ego and a constant-velocity obstacle an exact affine
(``P + s*Q``) function of ``s`` over each segment -- letting the two
functions below compute the EXACT closest approach / EXACT first
threshold-crossing over the whole continuous segment, not just its two
endpoints.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple


def closest_approach_on_segment(x0: float, y0: float, x1: float, y1: float) -> Tuple[float, float]:
    """For ``P(s) = (x0, y0) + s * ((x1, y1) - (x0, y0))``, ``s in [0, 1]``,
    returns ``(s*, |P(s*)|)`` where ``s*`` minimizes the distance to the
    origin. Clamped to ``[0, 1]`` (the minimum of a convex quadratic outside
    that range is at whichever endpoint is nearer)."""
    dx, dy = x1 - x0, y1 - y0
    denom = dx * dx + dy * dy
    if denom < 1e-12:  # P(s) is (near-)constant over the segment
        s = 0.0
    else:
        s = -(x0 * dx + y0 * dy) / denom
        s = max(0.0, min(1.0, s))
    px, py = x0 + s * dx, y0 + s * dy
    return s, math.hypot(px, py)


def first_crossing_below_radius(x0: float, y0: float, x1: float, y1: float, radius: float) -> Optional[float]:
    """First ``s in [0, 1]`` at which ``|P(s)| <= radius`` (``P(s)`` as in
    :func:`closest_approach_on_segment`), or ``None`` if ``|P(s)|`` never
    drops to ``radius`` anywhere on the segment. Exact via the standard
    moving-point-vs-circle quadratic: ``|P(s)|^2 - radius^2 = a*s^2 + b*s +
    c`` is <= 0 exactly on the interval between its two real roots (if any),
    intersected with ``[0, 1]``."""
    dx, dy = x1 - x0, y1 - y0
    a = dx * dx + dy * dy
    b = 2.0 * (x0 * dx + y0 * dy)
    c = x0 * x0 + y0 * y0 - radius * radius
    if a < 1e-12:  # P(s) is (near-)constant over the segment
        return 0.0 if c <= 0.0 else None
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None
    sqrt_disc = math.sqrt(disc)
    s_lo = (-b - sqrt_disc) / (2.0 * a)
    s_hi = (-b + sqrt_disc) / (2.0 * a)
    if s_hi < 0.0 or s_lo > 1.0:
        return None
    return max(s_lo, 0.0)
