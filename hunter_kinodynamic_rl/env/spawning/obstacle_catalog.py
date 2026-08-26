"""Reads drl_obstacle_assets's ``config/obstacle_catalog.yaml`` (dependency,
not copied -- see docs/SOURCE_MAP.md) and picks a catalog entry whose
``radius`` best matches a scenario obstacle's requested radius, so Gazebo
spawns reuse the SAME model library drl_agent's curriculum uses instead of
a from-scratch asset set (section 3.3 / section 26)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import yaml


@dataclass(frozen=True)
class CatalogEntry:
    key: str
    uri: str
    radius: float
    yaw_random: bool


def _drl_obstacle_assets_share_dir() -> Optional[str]:
    try:
        from ament_index_python.packages import get_package_share_directory
        return get_package_share_directory("drl_obstacle_assets")
    except Exception:
        return None


def load_catalog(path: Optional[str] = None) -> List[CatalogEntry]:
    """Loads and returns only ``motion_type: static`` entries (section 26:
    simple static assets for training; dynamic obstacles use this
    package's own generic moving-obstacle model instead of the catalog's
    now-vestigial dynamic/human entries -- see
    env/humans/dynamic_obstacle_motion.py)."""
    if path is None:
        share_dir = _drl_obstacle_assets_share_dir()
        if share_dir is None:
            return []
        path = os.path.join(share_dir, "config", "obstacle_catalog.yaml")
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    entries = []
    for item in data.get("obstacles", []):
        if item.get("motion_type") != "static":
            continue
        entries.append(CatalogEntry(
            key=item["key"], uri=item["uri"],
            radius=float(item.get("radius", 0.3)), yaw_random=bool(item.get("yaw_random", False)),
        ))
    return entries


def closest_entry(catalog: List[CatalogEntry], target_radius: float) -> Optional[CatalogEntry]:
    if not catalog:
        return None
    return min(catalog, key=lambda e: abs(e.radius - target_radius))
