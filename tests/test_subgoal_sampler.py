"""Phase 4 Global RL: candidate-subgoal grid (plan section 8.3/9)."""

import math

from hunter_kinodynamic_rl.config.schema import GlobalRLConfig
from hunter_kinodynamic_rl.navigation.global_rl.subgoal_sampler import (
    FALLBACK_LABEL_BACKTRACK, FALLBACK_LABEL_STOP_RECOVERY, build_candidate_set,
    candidate_endpoint_mission, candidate_endpoint_robot_frame,
)
from hunter_kinodynamic_rl.navigation.mission.mission_frame import PoseXYYaw


def test_candidate_count_and_order_is_fixed():
    cfg = GlobalRLConfig(direction_degrees=[-90.0, 0.0, 90.0], distances_m=[2.0, 4.0])
    candidates = build_candidate_set(cfg)
    assert len(candidates) == 3 * 2 + 1 == cfg.n_candidates
    # direction-major, distance-minor order.
    expected = [(-90.0, 2.0), (-90.0, 4.0), (0.0, 2.0), (0.0, 4.0), (90.0, 2.0), (90.0, 4.0)]
    for candidate, (deg, dist) in zip(candidates[:-1], expected):
        assert not candidate.is_fallback
        assert math.isclose(candidate.angle_rad, math.radians(deg), abs_tol=1e-9)
        assert math.isclose(candidate.radius_m, dist)
    assert candidates[-1].is_fallback
    assert candidates[-1].index == cfg.fallback_index


def test_default_candidate_grid_is_16_plus_fallback():
    cfg = GlobalRLConfig()
    candidates = build_candidate_set(cfg)
    assert len(candidates) == 17
    assert cfg.n_direction_distance_candidates == 16


def test_fallback_backtrack_is_behind_robot():
    cfg = GlobalRLConfig(fallback_mode="backtrack", fallback_backtrack_distance_m=2.5)
    candidates = build_candidate_set(cfg)
    fallback = candidates[-1]
    assert fallback.label == FALLBACK_LABEL_BACKTRACK
    assert math.isclose(fallback.radius_m, 2.5)
    assert math.isclose(fallback.angle_rad, math.pi)
    x, y = candidate_endpoint_robot_frame(fallback)
    assert x < 0.0
    assert math.isclose(y, 0.0, abs_tol=1e-9)


def test_fallback_stop_recovery_is_zero_distance():
    cfg = GlobalRLConfig(fallback_mode="stop_recovery")
    fallback = build_candidate_set(cfg)[-1]
    assert fallback.label == FALLBACK_LABEL_STOP_RECOVERY
    assert fallback.radius_m == 0.0


def test_candidate_endpoint_robot_frame_matches_polar_convention():
    cfg = GlobalRLConfig(direction_degrees=[0.0, 90.0], distances_m=[5.0])
    candidates = build_candidate_set(cfg)
    forward = candidates[0]
    left = candidates[1]
    fx, fy = candidate_endpoint_robot_frame(forward)
    assert math.isclose(fx, 5.0, abs_tol=1e-9)
    assert math.isclose(fy, 0.0, abs_tol=1e-9)
    lx, ly = candidate_endpoint_robot_frame(left)
    assert math.isclose(lx, 0.0, abs_tol=1e-9)
    assert math.isclose(ly, 5.0, abs_tol=1e-9)


def test_candidate_endpoint_mission_rotates_and_translates_with_robot_pose():
    cfg = GlobalRLConfig(direction_degrees=[0.0], distances_m=[3.0])
    candidate = build_candidate_set(cfg)[0]
    pose = PoseXYYaw(x=10.0, y=-5.0, yaw=math.pi / 2.0)
    x, y = candidate_endpoint_mission(candidate, pose)
    # Facing +90deg (mission +y), "straight ahead" 3m lands at (10, -5+3).
    assert math.isclose(x, 10.0, abs_tol=1e-9)
    assert math.isclose(y, -2.0, abs_tol=1e-9)


def test_candidate_grid_scales_with_config():
    cfg = GlobalRLConfig(direction_degrees=[0.0, 45.0, 90.0, 135.0], distances_m=[1.0, 2.0, 3.0])
    candidates = build_candidate_set(cfg)
    assert len(candidates) == 4 * 3 + 1
    non_fallback = [c for c in candidates if not c.is_fallback]
    assert len(non_fallback) == 12
    assert all(c.radius_m > 0.0 for c in non_fallback)
