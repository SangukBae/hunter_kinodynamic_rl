"""Phase 4 Global RL: candidate action mask (plan section 8.4)."""

import math

from hunter_kinodynamic_rl.config.schema import GlobalRLConfig, MappingConfig
from hunter_kinodynamic_rl.navigation.global_rl.action_mask import compute_action_mask
from hunter_kinodynamic_rl.navigation.global_rl.subgoal_sampler import build_candidate_set, candidate_endpoint_mission
from hunter_kinodynamic_rl.navigation.mapping.partial_map import PartialMap
from hunter_kinodynamic_rl.navigation.mission.mission_frame import PoseXYYaw


# 1/m == 1 m minimum turning radius -- reasonably permissive but still
# genuinely finite, so a test that wants to exercise curvature clamping can
# do so with a small, easy-to-reason-about radius.
ROBOT_MAX_CURVATURE = 1.0


def _fresh_map(size_cells=128):
    return PartialMap(MappingConfig(mission_size_cells=size_cells), size_cells=size_cells)


def _mark_occupied(partial_map: PartialMap, x: float, y: float):
    cell = partial_map.world_to_cell(x, y)
    assert cell is not None
    partial_map.observed[cell] = True
    partial_map.log_odds[cell] = partial_map.config.occupied_threshold + 1.0


def test_all_reachable_candidates_are_valid_on_a_fresh_unknown_map():
    # Excludes -180 deg on purpose: that direction is geometrically
    # unreachable via any single FORWARD Ackermann arc regardless of map
    # state (see test_directly_behind_candidate_is_unreachable_via_any_forward_arc)
    # -- this test isolates "UNKNOWN must never invalidate on its own" from
    # that unrelated reachability constraint.
    cfg = GlobalRLConfig(direction_degrees=[-135.0, -90.0, -45.0, 0.0, 45.0, 90.0, 135.0], distances_m=[3.0, 6.0])
    candidates = build_candidate_set(cfg)
    partial_map = _fresh_map()
    pose = PoseXYYaw(0.0, 0.0, 0.0)
    mask = compute_action_mask(candidates, partial_map, pose, cfg, ROBOT_MAX_CURVATURE)
    # UNKNOWN endpoints must never be invalidated purely for being unknown.
    assert mask.all()


def test_fallback_always_valid_even_when_everything_else_is_blocked():
    cfg = GlobalRLConfig()
    candidates = build_candidate_set(cfg)
    partial_map = _fresh_map()
    pose = PoseXYYaw(0.0, 0.0, 0.0)
    mask = compute_action_mask(candidates, partial_map, pose, cfg, ROBOT_MAX_CURVATURE, localization_valid=False)
    assert not mask[:-1].any()
    assert mask[-1]


def test_endpoint_in_inflated_cell_is_invalid():
    cfg = GlobalRLConfig(direction_degrees=[0.0], distances_m=[3.0], rollout_sample_count=4)
    candidates = build_candidate_set(cfg)
    partial_map = _fresh_map()
    pose = PoseXYYaw(0.0, 0.0, 0.0)
    forward = candidates[0]
    ex, ey = candidate_endpoint_mission(forward, pose)
    _mark_occupied(partial_map, ex, ey)
    mask = compute_action_mask(candidates, partial_map, pose, cfg, ROBOT_MAX_CURVATURE)
    assert not mask[forward.index]
    assert mask[-1]


def test_short_rollout_collision_invalidates_even_when_endpoint_is_clear():
    cfg = GlobalRLConfig(direction_degrees=[0.0], distances_m=[6.0], rollout_sample_count=8)
    candidates = build_candidate_set(cfg)
    partial_map = _fresh_map()
    pose = PoseXYYaw(0.0, 0.0, 0.0)
    forward = candidates[0]
    # Block a cell halfway along the straight-line rollout, not the endpoint.
    _mark_occupied(partial_map, 3.0, 0.0)
    mask = compute_action_mask(candidates, partial_map, pose, cfg, ROBOT_MAX_CURVATURE)
    assert not mask[forward.index]


def test_robot_footprint_collision_invalidates_all_non_fallback_candidates():
    cfg = GlobalRLConfig()
    candidates = build_candidate_set(cfg)
    partial_map = _fresh_map()
    pose = PoseXYYaw(0.0, 0.0, 0.0)
    _mark_occupied(partial_map, 0.0, 0.0)
    mask = compute_action_mask(candidates, partial_map, pose, cfg, ROBOT_MAX_CURVATURE)
    assert not mask[:-1].any()
    assert mask[-1]


def test_map_invalid_flag_invalidates_all_non_fallback_candidates():
    cfg = GlobalRLConfig()
    candidates = build_candidate_set(cfg)
    partial_map = _fresh_map()
    pose = PoseXYYaw(0.0, 0.0, 0.0)
    mask = compute_action_mask(candidates, partial_map, pose, cfg, ROBOT_MAX_CURVATURE, map_valid=False)
    assert not mask[:-1].any()
    assert mask[-1]


def test_ackermann_rollout_differs_from_straight_line_for_a_lateral_candidate():
    """A lateral candidate whose STRAIGHT chord never touches an obstacle
    can still be invalidated once the check follows the vehicle's actual
    feasible arc instead -- the entire point of "Ackermann short rollout"
    over an idealized straight-line check (plan section 8.4). Places the
    obstacle exactly on the arc's own midpoint, which is off the straight
    chord (x=0 for every t along a pure 90-degree candidate)."""
    max_curvature = 1.0  # 1 m minimum turning radius
    cfg = GlobalRLConfig(direction_degrees=[90.0], distances_m=[6.0], rollout_sample_count=5)
    candidates = build_candidate_set(cfg)
    lateral = candidates[0]
    partial_map = _fresh_map()
    pose = PoseXYYaw(0.0, 0.0, 0.0)

    # kappa = (2/r)*sin(90deg) = 2/6 = 1/3 (feasible, well under 1.0) --
    # the FULL arc length needed to reach (0, 6) is r*phi/sin(phi) =
    # 6*(pi/2)/1 = 3*pi ~= 9.42 m, not the straight chord's 6 m. Marking
    # the arc's own midpoint (s = 3*pi/2 ~= 4.71 m along the arc) occupied
    # is off the straight chord (x=0) but squarely on the real rollout.
    kappa = (2.0 / lateral.radius_m) * math.sin(lateral.angle_rad)
    arc_length = lateral.radius_m * lateral.angle_rad / math.sin(lateral.angle_rad)
    s_mid = arc_length * 0.5
    arc_x = math.sin(kappa * s_mid) / kappa
    arc_y = (1.0 - math.cos(kappa * s_mid)) / kappa
    assert arc_x > 0.5, "test setup invalid: the real arc's midpoint should be well off the straight chord (x=0)"
    _mark_occupied(partial_map, arc_x, arc_y)

    mask = compute_action_mask(candidates, partial_map, pose, cfg, max_curvature)
    assert not mask[lateral.index]


def test_full_rollout_reaches_the_actual_endpoint_not_truncated_at_the_straight_chord_length():
    """Regression guard for the exact bug reported in review: a candidate
    whose required arc is LONGER than its own straight-line ``radius_m``
    must still be swept all the way to its real endpoint, not silently
    truncated at ``radius_m`` -- confirmed by placing a known obstacle
    BEYOND the straight-chord length but still ON the real arc, and
    checking it invalidates the candidate."""
    max_curvature = 1.0
    cfg = GlobalRLConfig(direction_degrees=[90.0], distances_m=[6.0], rollout_sample_count=20)
    candidates = build_candidate_set(cfg)
    lateral = candidates[0]
    partial_map = _fresh_map(size_cells=200)
    pose = PoseXYYaw(0.0, 0.0, 0.0)

    # The reviewer's own repro: the real arc's tail lies in the segment
    # (1.37, 5.67) region -- well past straight-chord distance (6 m from
    # the origin along y) yet still short of the endpoint (0, 6) -- which
    # the OLD (truncated-at-radius_m) rollout never sampled at all.
    _mark_occupied(partial_map, 1.37, 5.67)

    mask = compute_action_mask(candidates, partial_map, pose, cfg, max_curvature)
    assert not mask[lateral.index]


def test_long_arc_rollout_does_not_skip_obstacles_between_sparse_profile_samples():
    """Regression guard for long feasible arcs: the profile's fixed
    ``rollout_sample_count`` is only a minimum, not the full collision
    discretization. A 135-degree/6m candidate has a nearly 20m arc, so the
    Phase 4 default 8 samples would be spaced far enough apart to miss this
    obstacle unless the mask adaptively subdivides and traces between samples."""
    cfg = GlobalRLConfig(direction_degrees=[135.0], distances_m=[6.0], rollout_sample_count=8)
    candidates = build_candidate_set(cfg)
    sharp_backward_left = candidates[0]
    partial_map = _fresh_map()
    pose = PoseXYYaw(0.0, 0.0, 0.0)

    phi = sharp_backward_left.angle_rad
    r = sharp_backward_left.radius_m
    kappa = (2.0 / r) * math.sin(phi)
    # This point lies on the feasible arc between the profile's old sparse
    # samples, more than the inflated radius away from both neighbors.
    s = 10.0
    arc_x = math.sin(kappa * s) / kappa
    arc_y = (1.0 - math.cos(kappa * s)) / kappa
    _mark_occupied(partial_map, arc_x, arc_y)

    mask = compute_action_mask(candidates, partial_map, pose, cfg, robot_max_curvature=1.0)
    assert not mask[sharp_backward_left.index]
    assert mask[-1]


def test_candidate_requiring_curvature_beyond_vehicle_limit_is_invalid():
    """When even the FULL feasible-arc reformulation still needs more
    curvature than the vehicle can produce, the candidate must be rejected
    outright -- never silently clamped to a shorter, different arc that
    doesn't actually reach the declared endpoint (plan section 8.4)."""
    # kappa required = (2/1)*sin(90deg) = 2.0 -- exceeds max_curvature=1.0.
    cfg = GlobalRLConfig(direction_degrees=[90.0], distances_m=[1.0], rollout_sample_count=5)
    candidates = build_candidate_set(cfg)
    lateral = candidates[0]
    partial_map = _fresh_map()
    pose = PoseXYYaw(0.0, 0.0, 0.0)
    mask = compute_action_mask(candidates, partial_map, pose, cfg, robot_max_curvature=1.0)
    assert not mask[lateral.index]
    assert mask[-1]


def test_directly_behind_candidate_is_unreachable_via_any_forward_arc():
    """A candidate at exactly +/-180 degrees requires an infinitely long
    forward arc to reach (the pure-pursuit curvature -> 0 in this limit,
    but the arc length needed diverges) -- must be rejected regardless of
    how generous ``robot_max_curvature`` is, and regardless of map state."""
    cfg = GlobalRLConfig(direction_degrees=[-180.0], distances_m=[3.0], rollout_sample_count=5)
    candidates = build_candidate_set(cfg)
    behind = candidates[0]
    partial_map = _fresh_map()
    pose = PoseXYYaw(0.0, 0.0, 0.0)
    mask = compute_action_mask(candidates, partial_map, pose, cfg, robot_max_curvature=1000.0)
    assert not mask[behind.index]
    assert mask[-1]


def test_straight_ahead_candidate_is_unaffected_by_curvature_clamping():
    """A candidate straight ahead (angle=0) always requires kappa=0
    regardless of the vehicle's turning limit -- curvature clamping must
    never spuriously perturb this common case."""
    cfg = GlobalRLConfig(direction_degrees=[0.0], distances_m=[4.0], rollout_sample_count=8)
    candidates = build_candidate_set(cfg)
    partial_map = _fresh_map()
    pose = PoseXYYaw(0.0, 0.0, 0.0)
    mask = compute_action_mask(candidates, partial_map, pose, cfg, robot_max_curvature=0.1)
    assert mask[candidates[0].index]


def test_mask_shape_is_fixed_to_n_candidates():
    cfg = GlobalRLConfig(direction_degrees=[0.0, 90.0], distances_m=[2.0, 4.0, 6.0])
    candidates = build_candidate_set(cfg)
    partial_map = _fresh_map()
    pose = PoseXYYaw(0.0, 0.0, 0.0)
    mask = compute_action_mask(candidates, partial_map, pose, cfg, ROBOT_MAX_CURVATURE)
    assert mask.shape == (cfg.n_candidates,)
    assert mask.dtype == bool
