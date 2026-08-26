"""Local trajectory primitives: x(s), y(s), yaw(s), curvature(s) as a
function of ARC LENGTH s along the path (not time -- the shape is a pure
geometric object; dynamics/ackermann_rollout.py separately answers "how long
does it actually take Hunter SE to get there").

Only ``constant_curvature_arc`` is implemented (section 12: "초기 버전은
constant-curvature arc여도 좋다"). ``TrajectoryPrimitive`` is the interface
every future primitive (clothoid, spline, learned) must satisfy, so
trajectory_sampler.py / pure_pursuit_adapter.py / risk/* never need to know
which concrete primitive produced their samples.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Protocol, runtime_checkable


@dataclass(frozen=True)
class PathPoint:
    s_m: float
    x: float
    y: float
    yaw: float
    curvature: float


@runtime_checkable
class TrajectoryPrimitive(Protocol):
    kappa: float
    v_ref: float
    horizon_m: float

    def sample(self, num_samples: int) -> List[PathPoint]: ...
    def point_at(self, s_m: float) -> PathPoint: ...


def _arc_point(kappa: float, s_m: float) -> PathPoint:
    if abs(kappa) < 1e-9:
        return PathPoint(s_m=s_m, x=s_m, y=0.0, yaw=0.0, curvature=0.0)
    radius = 1.0 / kappa
    yaw = s_m * kappa
    x = radius * math.sin(yaw)
    y = radius * (1.0 - math.cos(yaw))
    return PathPoint(s_m=s_m, x=x, y=y, yaw=yaw, curvature=kappa)


@dataclass(frozen=True)
class ConstantCurvatureArc:
    """Robot-local-frame constant-curvature arc: exact circle/line geometry,
    the same closed-form used by dynamics/bicycle_model.step (kept
    consistent on purpose -- a straight-line trajectory shape must agree
    with what the bicycle model actually drives)."""

    kappa: float
    v_ref: float
    horizon_m: float

    def point_at(self, s_m: float) -> PathPoint:
        s = max(0.0, min(float(s_m), self.horizon_m))
        return _arc_point(self.kappa, s)

    def sample(self, num_samples: int) -> List[PathPoint]:
        if num_samples < 2:
            raise ValueError("num_samples must be >= 2")
        return [self.point_at(self.horizon_m * i / (num_samples - 1)) for i in range(num_samples)]

    def path_length_m(self) -> float:
        return self.horizon_m


def make_primitive(primitive_name: str, kappa: float, v_ref: float, horizon_m: float) -> TrajectoryPrimitive:
    if primitive_name == "constant_curvature_arc":
        return ConstantCurvatureArc(kappa=kappa, v_ref=v_ref, horizon_m=horizon_m)
    raise ValueError(f"unknown trajectory primitive: {primitive_name!r}")
