"""Shared per-option safety telemetry (item 8: "OptionTelemetry가 더 이상
write-only 상태가 아니어야 한다").

Extra per-option stats ``SubgoalResult`` doesn't carry (TTC, stopping
margin, inference/command fault counts, collision count/source) --
populated by ANY ``LocalOptionExecutor`` implementation that can compute
them (``LiveGazeboLocalExecutor``/``SimplifiedKinematicLocalExecutor``, see
their own ``run_option``/``last_option_telemetry``), and CONSUMED by
``evaluation.long_horizon_benchmark.run_ablation_mission`` -- accumulated
into the episode dict, then threaded through CSV/``metrics.json``/
``evaluation.global_metrics.aggregate`` (see that module's own docstring for
the ``*_mean``/``*_valid_count`` convention this module's fields follow once
aggregated).

A field an executor cannot compute stays at its dataclass default (``None``
for the optional float fields, ``0`` for the integer counts) -- the
BENCHMARK layer is responsible for keeping ``None`` as ``None`` through
aggregation (never silently treating "this executor doesn't measure TTC" as
"TTC was 0"), while a genuine ``0`` count (e.g. zero inference timeouts
because none occurred) is not disguised either -- these are two different,
both-legitimate meanings integer counts already distinguish naturally
(``0`` vs. simply never being provided at all, checked via ``getattr``
on the executor instance itself).

``collision_signal_source`` documents WHICH kind of answer a given
executor's ``collision_count`` actually is (item 8: "단순 거리 proxy와
구분한다"):

  - ``"ground_truth"``: an authoritative collision truth -- either
    ``SimplifiedKinematicLocalExecutor``'s own exact world-occupancy
    collision check (no sensor noise/approximation at all), or
    ``LiveGazeboLocalExecutor``'s real Gazebo chassis-contact sensor
    (``/hunter_se/chassis_contacts``, ``ros_gz_interfaces/msg/Contacts`` --
    the SAME topic ``drl_agent``'s own ``environment.py`` already
    subscribes to for its own collision termination, reused here rather
    than reimplemented; see ``LiveGazeboLocalExecutor._on_contact``).
  - ``"proxy_clearance_threshold"``: a LiDAR-clearance-distance proxy, for
    an executor with no real contact/collision sensor available at all.
  - ``"unavailable"``: the executor provides no collision signal at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class OptionTelemetry:
    min_time_to_collision_sec: Optional[float] = None
    min_stopping_margin_m: Optional[float] = None
    inference_timeout_count: int = 0
    inference_error_count: int = 0
    command_timeout_count: int = 0
    collision_count: int = 0
    collision_signal_source: str = "unavailable"
