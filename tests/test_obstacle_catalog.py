import os

import pytest

from hunter_kinodynamic_rl.env.spawning.obstacle_catalog import closest_entry, load_catalog

_REAL_CATALOG_PATH = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "drl_obstacle_assets",
    "config", "obstacle_catalog.yaml",
))


def test_load_catalog_missing_path_returns_empty():
    assert load_catalog("/nonexistent/path.yaml") == []


@pytest.mark.skipif(not os.path.isfile(_REAL_CATALOG_PATH), reason="drl_obstacle_assets not present in this checkout")
def test_load_real_catalog_returns_only_static_entries():
    catalog = load_catalog(_REAL_CATALOG_PATH)
    assert len(catalog) > 0
    for entry in catalog:
        assert entry.radius > 0.0
        assert entry.uri.startswith("model://")


@pytest.mark.skipif(not os.path.isfile(_REAL_CATALOG_PATH), reason="drl_obstacle_assets not present in this checkout")
def test_closest_entry_picks_nearest_radius():
    catalog = load_catalog(_REAL_CATALOG_PATH)
    target = 0.35
    entry = closest_entry(catalog, target)
    assert entry is not None
    best_diff = abs(entry.radius - target)
    for e in catalog:
        assert abs(e.radius - target) >= best_diff


def test_closest_entry_empty_catalog_returns_none():
    assert closest_entry([], 0.3) is None
