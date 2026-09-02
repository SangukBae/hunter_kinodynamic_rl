"""Phase 4 Global RL: global observation assembly (plan section 8.5)."""

import numpy as np

from hunter_kinodynamic_rl.config.schema import GlobalRLConfig, MappingConfig
from hunter_kinodynamic_rl.navigation.global_rl.action_mask import compute_action_mask
from hunter_kinodynamic_rl.navigation.global_rl.observation import (
    N_CANDIDATE_FEATURES, N_MAP_CHANNELS, N_SCALARS, build_global_observation,
)
from hunter_kinodynamic_rl.navigation.global_rl.subgoal_sampler import build_candidate_set
from hunter_kinodynamic_rl.navigation.mapping.partial_map import PartialMap
from hunter_kinodynamic_rl.navigation.mission.mission_frame import PoseXYYaw

ROBOT_MAX_CURVATURE = 1.0


def _built(cfg=None, pose=None, goal=(5.0, 0.0), size_cells=128):
    cfg = cfg or GlobalRLConfig()
    pose = pose or PoseXYYaw(0.0, 0.0, 0.0)
    partial_map = PartialMap(MappingConfig(mission_size_cells=size_cells), size_cells=size_cells)
    candidates = build_candidate_set(cfg)
    mask = compute_action_mask(candidates, partial_map, pose, cfg, ROBOT_MAX_CURVATURE)
    obs = build_global_observation(
        partial_map, pose, goal, candidates, mask,
        speed_mps=0.3, previous_action_index=2, elapsed_mission_ratio=0.4, config=cfg,
    )
    return cfg, candidates, obs


def test_observation_shapes_are_fixed():
    cfg, candidates, obs = _built()
    assert obs.map_tensor.shape == (N_MAP_CHANNELS, cfg.map_crop_size_cells, cfg.map_crop_size_cells)
    assert obs.scalar_tensor.shape == (N_SCALARS,)
    assert obs.candidate_tensor.shape == (cfg.n_candidates, N_CANDIDATE_FEATURES)
    assert obs.action_mask.shape == (cfg.n_candidates,)
    assert obs.map_tensor.dtype == np.float32
    assert obs.scalar_tensor.dtype == np.float32
    assert obs.candidate_tensor.dtype == np.float32
    assert obs.action_mask.dtype == bool


def test_action_mask_is_embedded_as_first_candidate_feature():
    cfg, candidates, obs = _built()
    np.testing.assert_array_equal(obs.candidate_tensor[:, 0], obs.action_mask.astype(np.float32))


def test_goal_distance_and_bearing_scalars_reflect_true_geometry_even_far_outside_crop():
    cfg = GlobalRLConfig(goal_distance_norm_m=40.0)
    pose = PoseXYYaw(0.0, 0.0, 0.0)
    # Far outside any plausible rolling crop (crop spans ~ map_crop_size_cells * resolution).
    far_goal = (100.0, 0.0)
    _, _, obs = _built(cfg=cfg, pose=pose, goal=far_goal)
    # distance_norm is clipped to [0, 1] -- a goal beyond goal_distance_norm_m saturates at 1.0,
    # not silently zero/undefined.
    assert obs.scalar_tensor[0] == 1.0
    assert obs.scalar_tensor[1] == 0.0  # straight ahead -> bearing 0


def test_previous_action_scalar_varies_with_index():
    cfg = GlobalRLConfig()
    _, candidates, obs_low = _built(cfg=cfg)
    partial_map = PartialMap(MappingConfig(mission_size_cells=128), size_cells=128)
    pose = PoseXYYaw(0.0, 0.0, 0.0)
    mask = compute_action_mask(candidates, partial_map, pose, cfg, ROBOT_MAX_CURVATURE)
    obs_high = build_global_observation(
        partial_map, pose, (5.0, 0.0), candidates, mask,
        speed_mps=0.3, previous_action_index=cfg.n_candidates - 1, elapsed_mission_ratio=0.4, config=cfg,
    )
    assert obs_low.scalar_tensor[3] != obs_high.scalar_tensor[3]


def test_elapsed_mission_ratio_is_clipped_to_unit_interval():
    cfg, candidates, obs = _built()
    partial_map = PartialMap(MappingConfig(mission_size_cells=128), size_cells=128)
    pose = PoseXYYaw(0.0, 0.0, 0.0)
    mask = compute_action_mask(candidates, partial_map, pose, cfg, ROBOT_MAX_CURVATURE)
    obs_over = build_global_observation(
        partial_map, pose, (5.0, 0.0), candidates, mask,
        speed_mps=0.0, previous_action_index=0, elapsed_mission_ratio=5.0, config=cfg,
    )
    assert obs_over.scalar_tensor[4] == 1.0


def test_fallback_candidate_features_are_deterministic():
    cfg, candidates, obs = _built()
    fallback = candidates[-1]
    assert obs.candidate_tensor[fallback.index, 3] == 1.0  # max curvature-difficulty proxy


def test_no_privileged_ground_truth_leaks_into_observation():
    """Regression guard mirroring tests/test_world_information_boundary.py's
    own pattern: GlobalObservation must never carry a LongHorizonWorld
    field name -- it is built purely from PartialMap + mission-frame
    pose/goal (+, Phase 5, the topological-memory node tensor, itself built
    only from the online TopologicalGraph)."""
    import dataclasses
    from hunter_kinodynamic_rl.navigation.global_rl.observation import GlobalObservation
    field_names = {f.name for f in dataclasses.fields(GlobalObservation)}
    assert field_names == {
        "map_tensor", "scalar_tensor", "candidate_tensor", "action_mask", "node_tensor", "node_validity_mask",
    }
