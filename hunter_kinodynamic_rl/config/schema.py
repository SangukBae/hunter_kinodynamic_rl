"""Typed config objects for hunter_kinodynamic_rl.

Every physical / research parameter that could plausibly be swept in an
ablation lives here as a dataclass field with NO baked-in numeric default for
anything robot- or research-specific (only structurally-safe Python defaults
like ``field(default_factory=...)``) -- the actual numbers always come from
YAML via :mod:`hunter_kinodynamic_rl.config.loader`. See
``config/robot/hunter_se.yaml`` and ``config/profiles/*.yaml``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional


class ConfigError(ValueError):
    """Raised for any invalid / inconsistent config -- always fail at load
    time (profile validation), never mid-training."""


@dataclass
class RobotConfig:
    """Hunter SE (or a swapped-in Ackermann UGV) physical parameters."""

    name: str = "hunter_se"
    wheelbase_m: float = 0.0
    track_width_m: float = 0.0
    wheel_radius_m: float = 0.0
    length_m: float = 0.0
    width_m: float = 0.0
    height_m: float = 0.0
    mass_kg: float = 0.0
    steering_limit_deg: float = 0.0
    max_forward_speed_mps: float = 0.0
    min_forward_speed_mps: float = 0.0
    accel_limit_mps2: float = 0.0
    brake_decel_mps2: float = 0.0
    steering_rate_deg_s: float = 0.0
    # First-order lag time constant on speed tracking (see
    # hunter_se_cmd_prefilter.yaml speed_lag_tau_sec) -- 0 disables the lag.
    speed_lag_tau_sec: float = 0.0
    # Config-based collision footprint (section 7: "robot footprint 또는
    # 최소한 config 기반 collision radius 사용") -- a single bounding-circle
    # radius, NOT a magic literal scattered through env code.
    collision_radius_m: float = 0.45

    @property
    def steering_limit_rad(self) -> float:
        return math.radians(self.steering_limit_deg)

    @property
    def steering_rate_rad_s(self) -> float:
        return math.radians(self.steering_rate_deg_s)

    @property
    def max_curvature(self) -> float:
        """kappa_max = tan(steering_limit) / wheelbase -- ALWAYS derived, never
        configured directly (section 9 of the research brief)."""
        if self.wheelbase_m <= 0.0:
            return 0.0
        return math.tan(self.steering_limit_rad) / self.wheelbase_m

    def validate(self) -> None:
        if self.wheelbase_m <= 0.0:
            raise ConfigError(f"robot.wheelbase_m must be > 0, got {self.wheelbase_m}")
        if not (0.0 < self.steering_limit_deg < 90.0):
            raise ConfigError(f"robot.steering_limit_deg must be in (0, 90), got {self.steering_limit_deg}")
        if self.max_forward_speed_mps <= 0.0:
            raise ConfigError(f"robot.max_forward_speed_mps must be > 0, got {self.max_forward_speed_mps}")
        if self.min_forward_speed_mps < 0.0:
            raise ConfigError(f"robot.min_forward_speed_mps must be >= 0, got {self.min_forward_speed_mps}")
        if self.min_forward_speed_mps > self.max_forward_speed_mps:
            raise ConfigError("robot.min_forward_speed_mps must be <= max_forward_speed_mps")
        if self.accel_limit_mps2 <= 0.0 or self.brake_decel_mps2 <= 0.0:
            raise ConfigError("robot.accel_limit_mps2 / brake_decel_mps2 must be > 0")
        if self.collision_radius_m <= 0.0:
            raise ConfigError("robot.collision_radius_m must be > 0")


@dataclass
class ActionSpaceConfig:
    """[kappa, v_ref, L] trajectory action, OR legacy [r, theta, yield]."""

    mode: str = "trajectory"  # "trajectory" | "legacy_waypoint"
    # trajectory mode bounds (kappa is clamped to robot.max_curvature at
    # resolve time -- kappa_scale in (0, 1] lets an ablation use less than
    # the full physical range).
    kappa_scale: float = 1.0
    v_min_mps: float = 0.0
    v_max_mps: Optional[float] = None  # None -> robot.max_forward_speed_mps
    horizon_length_min_m: float = 0.5
    horizon_length_max_m: float = 3.0
    # legacy_waypoint mode bounds (mirrors drl_agent's actions_low/high[:2]).
    legacy_r_min_m: float = 0.0
    legacy_r_max_m: float = 2.0
    legacy_theta_max_rad: float = 0.524
    legacy_yield_enabled: bool = True
    # section P0-4 (baseline/reference parity): the REST of drl_agent's own
    # hybrid stop/yield controller contract (CLAUDE.md's Action -- Waypoint
    # commands section / drl_agent/config/environment_curriculum.yaml) --
    # previously hardcoded/approximated ad hoc in trajectory_executor.py
    # using the SUPERSEDED waypoint_to_command() ramp instead of the
    # actually-parity-matching pure_pursuit.hybrid_action_to_command()
    # (verbatim-copied into trajectory/pure_pursuit.py but left UNUSED).
    # Defaults are drl_agent's own numbers exactly.
    legacy_yield_threshold: float = 0.3  # drl_agent's yield_reward.action_threshold
    legacy_lookahead_min_m: float = 0.8  # drl_agent's controller_lookahead_min_m
    legacy_v_move_min_mps: float = 0.35  # drl_agent's controller_v_move_min_mps
    legacy_yield_creep_mps: float = 0.0  # drl_agent's controller_yield_creep_mps
    legacy_speed_steer_factor: float = 0.6  # drl_agent's controller_speed_steer_factor

    def validate(self) -> None:
        if self.mode not in ("trajectory", "legacy_waypoint"):
            raise ConfigError(f"action_space.mode must be trajectory|legacy_waypoint, got {self.mode!r}")
        if not (0.0 < self.kappa_scale <= 1.0):
            raise ConfigError("action_space.kappa_scale must be in (0, 1]")
        if self.horizon_length_min_m <= 0.0 or self.horizon_length_max_m <= self.horizon_length_min_m:
            raise ConfigError("action_space.horizon_length_{min,max}_m must satisfy 0 < min < max")
        if self.v_min_mps < 0.0:
            raise ConfigError("action_space.v_min_mps must be >= 0")
        if not (-1.0 <= self.legacy_yield_threshold <= 1.0):
            raise ConfigError("action_space.legacy_yield_threshold must be in [-1, 1]")
        if self.legacy_lookahead_min_m < 0.0:
            raise ConfigError("action_space.legacy_lookahead_min_m must be >= 0")
        if self.legacy_v_move_min_mps < 0.0:
            raise ConfigError("action_space.legacy_v_move_min_mps must be >= 0")
        if self.legacy_yield_creep_mps < 0.0:
            raise ConfigError("action_space.legacy_yield_creep_mps must be >= 0")
        if not (0.0 <= self.legacy_speed_steer_factor <= 1.0):
            raise ConfigError("action_space.legacy_speed_steer_factor must be in [0, 1]")


@dataclass
class TrajectoryConfig:
    """Local trajectory primitive generation."""

    primitive: str = "constant_curvature_arc"  # future: "clothoid", "spline"
    num_samples: int = 15
    dt_sec: float = 0.1

    def validate(self) -> None:
        if self.primitive not in ("constant_curvature_arc",):
            raise ConfigError(f"trajectory.primitive not implemented: {self.primitive!r}")
        if self.num_samples < 2:
            raise ConfigError("trajectory.num_samples must be >= 2")
        if self.dt_sec <= 0.0:
            raise ConfigError("trajectory.dt_sec must be > 0")


@dataclass
class DynamicsConfig:
    """Ackermann rollout horizon/timestep (kinodynamic feasibility check).

    ``horizon_sec`` is the FIXED fallback used when a trajectory command's
    own L (horizon_m) is not available (e.g. legacy_waypoint mode). In
    trajectory mode, the per-candidate rollout horizon is L-DERIVED --
    ``horizon_from_trajectory()`` in dynamics/ackermann_rollout.py computes
    ``clip(L / v_ref, horizon_min_sec, horizon_max_sec)`` -- see
    docs/ARCHITECTURE.md's "L semantics" section for the full rationale.
    """

    horizon_sec: float = 2.0
    horizon_min_sec: float = 0.5
    horizon_max_sec: float = 3.0
    # L-INDEPENDENT safety floor (section "P0 -- L semantics redesign"): the
    # rollout/risk horizon is NEVER shorter than this, regardless of how
    # small a candidate's L makes horizon_from_trajectory()'s raw L/v_ref
    # value. Without this floor, a policy could pick an arbitrarily small L
    # to make its own rollout too short to ever reach a distant obstacle,
    # always reporting risk=0 for a direction that is actually unsafe a
    # bit further out -- see dynamics/ackermann_rollout.py's
    # horizon_from_trajectory() and docs/ARCHITECTURE.md's "L semantics"
    # section for the full rationale + the regression test that catches a
    # regression of this specific exploit.
    min_safety_horizon_sec: float = 1.5
    dt_sec: float = 0.1
    model_actuator_lag: bool = True

    def validate(self) -> None:
        if not (0.5 <= self.horizon_sec <= 5.0):
            raise ConfigError("dynamics.horizon_sec should be within [0.5, 5.0]s")
        if self.horizon_min_sec <= 0.0 or self.horizon_max_sec <= self.horizon_min_sec:
            raise ConfigError("dynamics.{horizon_min_sec,horizon_max_sec} must satisfy 0 < min < max")
        if not (self.horizon_min_sec <= self.min_safety_horizon_sec <= self.horizon_max_sec):
            raise ConfigError(
                "dynamics.min_safety_horizon_sec must satisfy horizon_min_sec <= "
                "min_safety_horizon_sec <= horizon_max_sec (a floor equal to horizon_max_sec "
                "would make L have NO effect on the horizon at all)"
            )
        if self.dt_sec <= 0.0 or self.dt_sec > self.horizon_min_sec:
            raise ConfigError("dynamics.dt_sec must satisfy 0 < dt_sec <= horizon_min_sec")


@dataclass
class RiskConfig:
    enabled: bool = False
    clearance_horizon_sec: float = 2.0
    ttc_horizon_sec: float = 3.0
    min_safe_clearance_m: float = 0.3
    stopping_margin_safety_factor: float = 1.2
    actor_lambda: float = 0.0  # risk-penalty weight on the actor objective
    # Actor risk penalty stays at ZERO until the risk critic has completed
    # this many SUPERVISED updates -- section 5: "risk critic이 충분한 valid
    # label을 학습하기 전 actor penalty 적용을 warmup/gating할 수 있게 함"
    # (an untrained/near-random critic would otherwise inject noise into the
    # actor gradient from step 1).
    actor_penalty_warmup_updates: int = 500
    # Below this many valid-labeled transitions in a training batch, the risk
    # critic's supervised update for that batch is skipped entirely (too few
    # labels to trust a gradient step on).
    min_valid_labels_per_batch: int = 4
    # code review (risk-telemetry correctness bug): run-level HEALTH GATE --
    # a training run must never finish looking "successful" while the
    # risk-telemetry side channel was systematically broken (e.g. every
    # /step's telemetry timing out, confirmed live as a real failure mode --
    # see trainer_base.py's EnvironmentClient/_await_reset_marker). Checked
    # against ReplayBuffer.valid_ratio() by TrainerBase.run(), ONLY for
    # risk-aware trainers with risk.enabled=true (never for vanilla TQC/SAC,
    # where telemetry validity is diagnostic-only and never affects what
    # gets trained on).
    min_telemetry_valid_ratio: float = 0.3
    # Don't health-check before the buffer has accumulated at least this
    # many transitions -- a tiny early-training sample can have a
    # misleadingly low (or high) ratio purely from noise/warmup.
    telemetry_valid_ratio_grace_steps: int = 200
    telemetry_health_check_interval_steps: int = 200

    def validate(self) -> None:
        if self.clearance_horizon_sec <= 0.0 or self.ttc_horizon_sec <= 0.0:
            raise ConfigError("risk horizons must be > 0")
        if self.min_safe_clearance_m < 0.0:
            raise ConfigError("risk.min_safe_clearance_m must be >= 0")
        if self.actor_lambda < 0.0:
            raise ConfigError("risk.actor_lambda must be >= 0")
        if self.actor_penalty_warmup_updates < 0:
            raise ConfigError("risk.actor_penalty_warmup_updates must be >= 0")
        if self.min_valid_labels_per_batch < 1:
            raise ConfigError("risk.min_valid_labels_per_batch must be >= 1")
        if not (0.0 <= self.min_telemetry_valid_ratio <= 1.0):
            raise ConfigError("risk.min_telemetry_valid_ratio must be in [0, 1]")
        if self.telemetry_valid_ratio_grace_steps < 1:
            raise ConfigError("risk.telemetry_valid_ratio_grace_steps must be >= 1")
        if self.telemetry_health_check_interval_steps < 1:
            raise ConfigError("risk.telemetry_health_check_interval_steps must be >= 1")


@dataclass
class CounterfactualConfig:
    enabled: bool = False
    num_candidates: int = 8
    kappa_offsets_frac: List[float] = field(default_factory=lambda: [-1.0, -0.5, 0.5, 1.0])
    speed_fractions: List[float] = field(default_factory=lambda: [0.5, 1.0])
    horizon_fractions: List[float] = field(default_factory=lambda: [0.67, 1.33])
    include_stop_candidate: bool = True
    # A candidate is a usable counterfactual only when it preserves the
    # actor trajectory's goal progress.  Both constraints are enforced and
    # the stricter threshold wins: candidate_progress must be at least
    # actor_progress*min_progress_ratio and may lose at most
    # max_progress_loss_m.  This prevents the stop candidate from winning
    # merely because it has zero collision risk while making no progress.
    min_progress_ratio: float = 0.8
    max_progress_loss_m: float = 0.25
    # Candidate-augmented risk critic supervision loss weight (relative to
    # the base actor-action supervision, which always has weight 1.0).
    candidate_supervision_weight: float = 0.5
    # Actor risk-penalty reweighting: penalty_weight = 1 + scale*relu(margin)
    # -- transitions where a safer alternative existed push the actor's OWN
    # predicted risk down harder (section 6's margin penalty, applied via the
    # risk critic's differentiable output rather than backprop through the
    # non-differentiable stored margin itself -- see
    # rl/algorithms/kinodynamic_tqc/agent.py's docstring).
    counterfactual_weight_scale: float = 2.0

    def validate(self) -> None:
        if self.enabled and self.num_candidates < 2:
            raise ConfigError("counterfactual.num_candidates must be >= 2 when enabled")
        if self.candidate_supervision_weight < 0.0:
            raise ConfigError("counterfactual.candidate_supervision_weight must be >= 0")
        if self.counterfactual_weight_scale < 0.0:
            raise ConfigError("counterfactual.counterfactual_weight_scale must be >= 0")
        # trajectory_sampler.py's generate_candidates() only ever produces
        # steering/speed alternatives by iterating these two lists -- empty
        # either one and every candidate beyond [actor, stop] silently
        # vanishes, defeating num_candidates>=2's intent without raising
        # anywhere (the resulting list just quietly ends up shorter).
        if self.enabled and not self.kappa_offsets_frac:
            raise ConfigError("counterfactual.kappa_offsets_frac must be non-empty when enabled")
        if self.enabled and not self.speed_fractions:
            raise ConfigError("counterfactual.speed_fractions must be non-empty when enabled")
        if self.enabled and not self.horizon_fractions:
            raise ConfigError("counterfactual.horizon_fractions must be non-empty when enabled")
        if any((not math.isfinite(v) or v <= 0.0) for v in self.horizon_fractions):
            raise ConfigError("counterfactual.horizon_fractions must contain finite values > 0")
        if not (0.0 <= self.min_progress_ratio <= 1.0):
            raise ConfigError("counterfactual.min_progress_ratio must be in [0, 1]")
        if self.max_progress_loss_m < 0.0:
            raise ConfigError("counterfactual.max_progress_loss_m must be >= 0")


@dataclass
class ObservationConfig:
    lidar_bins: int = 80
    frame_stack: int = 4
    # section P0-4 (baseline/reference parity): 7 = drl_agent's OWN 87D
    # contract (goal_dist, heading_err, prev_a0, prev_a1, v, yaw_rate,
    # steering -- REQUIRED for baseline_tqc/legacy_waypoint_tqc parity,
    # and the system-wide DEFAULT so a profile that forgets to think about
    # this gets the parity-safe contract, not the extended one). 8 = the
    # SAME plus a 3rd previous-action component (L / yield memory) -- an
    # explicit, opt-in research extension every kinodynamic_tqc* profile
    # sets deliberately in its own YAML (see docs/SOURCE_MAP.md,
    # tests/test_baseline_observation_parity.py). No other value is valid.
    robot_state_dim: int = 7
    lidar_max_range_m: float = 10.0
    front_sector_width_rad: float = 3.141592653589793  # pi -- front 180 deg
    collision_margin_m: float = 0.05  # added on top of robot.collision_radius_m

    def validate(self) -> None:
        if self.lidar_bins <= 0:
            raise ConfigError("observation.lidar_bins must be > 0")
        if self.frame_stack < 1:
            raise ConfigError("observation.frame_stack must be >= 1")
        if self.lidar_max_range_m <= 0.0:
            raise ConfigError("observation.lidar_max_range_m must be > 0")
        if not (0.0 < self.front_sector_width_rad <= 2 * 3.141592653589793):
            raise ConfigError("observation.front_sector_width_rad must be in (0, 2*pi]")
        if self.collision_margin_m < 0.0:
            raise ConfigError("observation.collision_margin_m must be >= 0")
        # section P0-4: exactly the two contracts observation_builder.py
        # actually implements -- imported locally to avoid config/schema.py
        # (imported nearly everywhere) taking on a module-load-time
        # dependency on the env/ package.
        from hunter_kinodynamic_rl.env.observation.observation_builder import (
            ROBOT_STATE_DIM_LEGACY_PARITY, ROBOT_STATE_DIM_WITH_L_MEMORY,
        )
        valid_dims = (ROBOT_STATE_DIM_LEGACY_PARITY, ROBOT_STATE_DIM_WITH_L_MEMORY)
        if self.robot_state_dim not in valid_dims:
            raise ConfigError(
                f"observation.robot_state_dim={self.robot_state_dim} must be one of {valid_dims} "
                f"({ROBOT_STATE_DIM_LEGACY_PARITY}=drl_agent baseline parity, "
                f"{ROBOT_STATE_DIM_WITH_L_MEMORY}=+ 3rd previous-action/L-memory component)"
            )


@dataclass
class TQCHyperparameters:
    discount: float = 0.99
    batch_size: int = 256
    buffer_size: int = 1_000_000
    n_quantiles: int = 25
    n_critics: int = 2
    top_quantiles_to_drop_per_net: int = 2
    tau: float = 0.005
    target_update_interval: int = 1
    ent_coef: str = "auto_1.0"
    ent_coef_lr: float = 3e-4
    actor_hdim: int = 256
    actor_activ: str = "relu"
    actor_lr: float = 3e-4
    critic_hdim: int = 256
    critic_activ: str = "elu"
    critic_lr: float = 3e-4

    def validate(self) -> None:
        if self.n_quantiles <= 0 or self.n_critics <= 0:
            raise ConfigError("hyperparameters.{n_quantiles,n_critics} must be > 0")
        if not (0 <= self.top_quantiles_to_drop_per_net < self.n_quantiles):
            raise ConfigError("hyperparameters.top_quantiles_to_drop_per_net must be in [0, n_quantiles)")
        if self.batch_size <= 0 or self.buffer_size <= 0:
            raise ConfigError("hyperparameters.{batch_size,buffer_size} must be > 0")


@dataclass
class SACHyperparameters:
    """Standard twin-Q SAC (Haarnoja et al. 2018) -- the "Vanilla SAC"
    algorithm-choice comparison point (section 34/35), sharing this
    package's trajectory action space/observation/reward with the TQC-based
    ablation rows so ONLY the algorithm differs. No quantile fields (unlike
    TQCHyperparameters) -- rl/algorithms/sac/agent.py reuses
    rl/networks/tqc.py's Actor/Critic with Critic(n_quantiles=1), which
    collapses to a plain scalar twin-Q critic pair; there is nothing here
    for a quantile count to mean."""

    discount: float = 0.99
    batch_size: int = 256
    buffer_size: int = 1_000_000
    n_critics: int = 2
    tau: float = 0.005
    target_update_interval: int = 1
    ent_coef: str = "auto_1.0"
    ent_coef_lr: float = 3e-4
    actor_hdim: int = 256
    actor_activ: str = "relu"
    actor_lr: float = 3e-4
    critic_hdim: int = 256
    critic_activ: str = "elu"
    critic_lr: float = 3e-4

    def validate(self) -> None:
        if self.n_critics <= 0:
            raise ConfigError("sac_hyperparameters.n_critics must be > 0")
        if self.batch_size <= 0 or self.buffer_size <= 0:
            raise ConfigError("sac_hyperparameters.{batch_size,buffer_size} must be > 0")
        if self.tau <= 0.0:
            raise ConfigError("sac_hyperparameters.tau must be > 0")
        if self.target_update_interval <= 0:
            raise ConfigError("sac_hyperparameters.target_update_interval must be > 0")


@dataclass
class AlgorithmConfig:
    """Which RL algorithm this profile trains (section 34/35's Vanilla SAC
    comparison point) -- default "tqc" preserves every pre-existing
    profile's behaviour unchanged. nodes/train_node.py dispatches the
    trainer/agent module on this field."""

    name: str = "tqc"

    def validate(self) -> None:
        if self.name not in ("tqc", "sac"):
            raise ConfigError(f"algorithm.name must be 'tqc' or 'sac', got {self.name!r}")


@dataclass
class FeatureFlags:
    """Ablation matrix (section 36) -- config flags only, never comments."""

    temporal_context: bool = False
    ackermann_rollout: bool = True
    risk_critic: bool = False
    counterfactual_risk: bool = False
    curriculum: bool = False
    # L (trajectory horizon) -> actually-published steering commit window
    # (trajectory/pure_pursuit_adapter.py's "STEERING COMMIT WINDOW"); OFF
    # falls back to the legacy instantaneous-steering behavior. Only
    # meaningful under action_space.mode == "trajectory" -- a no-op for
    # legacy_waypoint profiles. Defaults True (mirrors ackermann_rollout)
    # so the primary trajectory-mode training path exercises it.
    trajectory_l_preview_blend: bool = True


@dataclass
class ScenarioConfig:
    world_size_m: float = 12.0
    min_obstacles: int = 0
    max_obstacles: int = 8
    dynamic_obstacle_count: int = 0
    train_seed_range: List[int] = field(default_factory=lambda: [0, 9999])
    validation_seed_range: List[int] = field(default_factory=lambda: [10000, 10999])
    test_seed_range: List[int] = field(default_factory=lambda: [20000, 20999])
    # "grid_bfs" (default, legacy): procedural_generator.is_reachable's
    # 4-connected occupancy-grid flood fill -- topological connectivity only,
    # no heading/turning-radius awareness. "ackermann": additionally requires
    # env/scenarios/ackermann_feasibility.is_ackermann_feasible to succeed --
    # a bounded Hybrid-A*-style search over the robot's OWN bicycle-model
    # kinematics and minimum turning radius. Callers that select "ackermann"
    # MUST pass generate_scenario a valid min_turning_radius_m/wheelbase_m
    # (derived from RobotConfig) -- generate_scenario fails fast rather than
    # silently falling back to grid_bfs if they don't (see its docstring).
    feasibility_check: str = "grid_bfs"
    # section P1-3: a dynamic obstacle's SPAWN position was previously drawn
    # completely independently of start/goal/static obstacles -- initial
    # overlap (e.g. spawning directly on top of the robot's own start pose)
    # was possible. This is the minimum clearance (beyond radius sums) a
    # dynamic obstacle's t=0 position must keep from the robot start, the
    # goal, and every static obstacle; a placement attempt that can't
    # satisfy it retries (bounded, see dynamic_obstacle_placement_attempts)
    # rather than silently overlapping. Default biases toward FEASIBLE
    # scenarios (section P1-3: "기본값은 feasible scenario로 한다").
    dynamic_obstacle_min_clearance_m: float = 0.5
    # Bounded per-obstacle retry budget within one scenario-generation
    # attempt -- exhausting this does NOT silently proceed with an
    # overlapping obstacle; it fails this whole scenario attempt (the outer
    # generate_scenario() retry loop then redraws a fresh layout).
    dynamic_obstacle_placement_attempts: int = 30
    # section item-7: whether generate_scenario's feasibility check
    # (grid_bfs, or ackermann when feasibility_check='ackermann') ALSO
    # treats every dynamic obstacle's t=0 position/radius as a temporary
    # occupied region -- guaranteeing a solvable start->goal path exists
    # at the very INSTANT the episode begins, accounting for where dynamic
    # obstacles actually start out, not just static geometry. Deliberately
    # narrow in scope: this says nothing about the obstacle's SUBSEQUENT
    # motion during the episode -- a dynamic obstacle is still fully free
    # to move into/threaten the robot's path LATER (that is the entire
    # point of a dynamic-obstacle curriculum stage, not a bug to prevent).
    # True (default, section P1-3's same "기본값은 feasible scenario로
    # 한다" bias) rejects and redraws a layout whose t=0 dynamic-obstacle
    # placement already blocks the only path -- a "genuinely infeasible
    # from the start" scenario. Setting this False opts back into the
    # pre-item-7 behavior (dynamic obstacles excluded from the initial
    # feasibility check entirely) for a caller that deliberately WANTS a
    # scenario that starts already partially blocked (e.g. an adversarial
    # stress-test benchmark) -- an explicit, documented choice, never a
    # silent default.
    dynamic_obstacle_initial_feasibility_check: bool = True
    # -- arbitrary short-range subgoal training (hierarchical Phase 2) -----
    # "uniform_world" (default, legacy): goal drawn uniformly over the whole
    # inset world area, exactly the pre-existing behaviour -- BYTE-IDENTICAL
    # RNG draw order to before these fields existed. "robot_relative_band":
    # goal is instead drawn at a distance in goal_distance_range_m and an
    # angle within one of goal_direction_sectors_deg (both relative to the
    # robot's OWN start_yaw), mirroring the actual robot-relative candidate
    # geometry navigation/global_rl/subgoal_sampler.py's Global policy will
    # hand the Local TQC at inference time (see
    # docs/HIERARCHICAL_NAVIGATION_IMPLEMENTATION_PLAN.md section 6.6) --
    # used ONLY by the Local-TQC arbitrary-subgoal training profile, never
    # by any pre-existing profile (all of which stay on "uniform_world").
    # Requires start_pose.heading_mode == "legacy_random" (the only mode
    # whose start_yaw is already known at goal-draw time -- every other
    # heading_mode samples start_yaw AFTER obstacles/goal are placed, which
    # would make the goal geometry depend on a heading not sampled yet);
    # generate_scenario fails fast on this combination rather than silently
    # reordering its own draw sequence.
    goal_sampling_mode: str = "uniform_world"
    goal_distance_range_m: List[float] = field(default_factory=lambda: [2.0, 6.0])
    # Each entry is [lo_deg, hi_deg], an angular sector relative to
    # start_yaw (0 deg == straight ahead, positive == robot's left, matching
    # this package's standard right-handed yaw convention). One sector is
    # drawn uniformly at random per scenario attempt, then an angle uniform
    # within it. Defaults cover front/left/right (never directly behind).
    goal_direction_sectors_deg: List[List[float]] = field(
        default_factory=lambda: [[-60.0, 60.0], [60.0, 150.0], [-150.0, -60.0]]
    )
    # Fraction of "robot_relative_band" scenario attempts that deliberately
    # SKIP the feasibility check (grid_bfs, and ackermann when
    # feasibility_check="ackermann") for the drawn goal -- so the Local TQC
    # training/eval distribution also includes subgoals that are blocked or
    # otherwise unreachable within the episode horizon (plan section 6.6:
    # "즉시 도달 불가능한 subgoal도 학습/평가 분포에 포함"), teaching the
    # policy to time out / signal blocked rather than only ever seeing
    # solvable subgoals. 0.0 (default) never skips the check -- identical to
    # every other feasibility-checked scenario. Only meaningful when
    # goal_sampling_mode="robot_relative_band"; ignored otherwise.
    goal_infeasible_fraction: float = 0.0

    def validate(self) -> None:
        if self.world_size_m <= 0.0:
            raise ConfigError("scenario.world_size_m must be > 0")
        if self.min_obstacles < 0 or self.max_obstacles < self.min_obstacles:
            raise ConfigError("scenario.{min,max}_obstacles invalid")
        if self.dynamic_obstacle_min_clearance_m < 0.0:
            raise ConfigError("scenario.dynamic_obstacle_min_clearance_m must be >= 0")
        if self.dynamic_obstacle_placement_attempts <= 0:
            raise ConfigError("scenario.dynamic_obstacle_placement_attempts must be > 0")
        if self.feasibility_check not in ("grid_bfs", "ackermann"):
            raise ConfigError(
                f"scenario.feasibility_check must be grid_bfs|ackermann, got {self.feasibility_check!r}"
            )
        if self.goal_sampling_mode not in ("uniform_world", "robot_relative_band"):
            raise ConfigError(
                "scenario.goal_sampling_mode must be uniform_world|robot_relative_band, "
                f"got {self.goal_sampling_mode!r}"
            )
        if len(self.goal_distance_range_m) != 2 or not (
            0.0 < self.goal_distance_range_m[0] <= self.goal_distance_range_m[1]
        ):
            raise ConfigError(
                f"scenario.goal_distance_range_m must be [lo, hi] with 0 < lo <= hi, "
                f"got {self.goal_distance_range_m}"
            )
        if not self.goal_direction_sectors_deg:
            raise ConfigError("scenario.goal_direction_sectors_deg must be non-empty")
        for sector in self.goal_direction_sectors_deg:
            if len(sector) != 2 or not (-180.0 <= sector[0] < sector[1] <= 180.0):
                raise ConfigError(
                    f"scenario.goal_direction_sectors_deg entries must be [lo, hi] with "
                    f"-180 <= lo < hi <= 180, got {sector}"
                )
        if not (0.0 <= self.goal_infeasible_fraction <= 1.0):
            raise ConfigError("scenario.goal_infeasible_fraction must be in [0, 1]")
        ranges = [self.train_seed_range, self.validation_seed_range, self.test_seed_range]
        for r in ranges:
            if len(r) != 2 or r[0] > r[1]:
                raise ConfigError(f"seed range must be [lo, hi] with lo<=hi, got {r}")
        lo_hi = sorted(ranges, key=lambda r: r[0])
        for a, b in zip(lo_hi, lo_hi[1:]):
            if a[1] >= b[0]:
                raise ConfigError(
                    "train/validation/test seed ranges must not overlap: "
                    f"{a} overlaps {b}"
                )


@dataclass
class StartPoseConfig:
    """Safe start-position/initial-yaw sampling (curriculum-independent --
    this package has no drl_agent-style curriculum stages; every field here
    applies uniformly for the whole run). Default ``heading_mode="legacy_random"``
    is BYTE-IDENTICAL to the pre-existing behaviour (``start_yaw =
    rng.uniform(-pi, pi)`` drawn immediately after the start position, before
    any obstacle is placed) -- every other mode is an explicit opt-in that
    changes ``generate_scenario``'s internal RNG draw ORDER (obstacles are
    placed BEFORE the heading is sampled, so the sampler can reject a
    heading that points at a wall or into a nearby obstacle), so enabling
    one changes the exact scenario a given seed produces even though the
    *distribution* of feasible layouts stays the same class of problem.

    ``min_wall_clearance_m``/``min_obstacle_clearance_m`` are read
    UNCONDITIONALLY by ``generate_scenario`` (regardless of ``heading_mode``)
    -- they only rename pre-existing magic numbers (the robot-radius-only
    wall inset, and the hardcoded ``0.5`` obstacle-placement rejection
    margin) into config fields, so their DEFAULTS reproduce the old
    numeric behaviour exactly; only a profile that explicitly changes them
    sees a different obstacle/start layout.
    """

    heading_mode: str = "legacy_random"  # legacy_random | random_rejected | goal_biased | free_space_biased
    min_wall_clearance_m: float = 0.0
    min_obstacle_clearance_m: float = 0.5
    front_safety_distance_m: float = 0.6
    front_cone_half_angle_rad: float = 0.5236  # ~30 deg
    max_sampling_attempts: int = 50
    # goal_biased: probability a candidate heading is drawn toward the goal
    # (+/- goal_bias_spread_rad jitter) rather than uniformly at random.
    goal_bias_prob: float = 0.5
    goal_bias_spread_rad: float = 0.3
    # free_space_biased: number of evenly-spaced probe directions scored by
    # obstacle/wall clearance; a candidate is drawn from a softmax over
    # those clearance scores (so it's biased toward open directions, not
    # deterministic argmax, keeping the sampler's own attempts/fallback
    # machinery meaningful).
    free_space_probe_count: int = 12

    def validate(self) -> None:
        valid_modes = ("legacy_random", "random_rejected", "goal_biased", "free_space_biased")
        if self.heading_mode not in valid_modes:
            raise ConfigError(f"start_pose.heading_mode must be one of {valid_modes}, got {self.heading_mode!r}")
        if self.min_wall_clearance_m < 0.0:
            raise ConfigError("start_pose.min_wall_clearance_m must be >= 0")
        if self.min_obstacle_clearance_m < 0.0:
            raise ConfigError("start_pose.min_obstacle_clearance_m must be >= 0")
        if self.front_safety_distance_m <= 0.0:
            raise ConfigError("start_pose.front_safety_distance_m must be > 0")
        if not (0.0 < self.front_cone_half_angle_rad <= math.pi):
            raise ConfigError("start_pose.front_cone_half_angle_rad must be in (0, pi]")
        if self.max_sampling_attempts < 1:
            raise ConfigError("start_pose.max_sampling_attempts must be >= 1")
        if not (0.0 <= self.goal_bias_prob <= 1.0):
            raise ConfigError("start_pose.goal_bias_prob must be in [0, 1]")
        if self.goal_bias_spread_rad < 0.0:
            raise ConfigError("start_pose.goal_bias_spread_rad must be >= 0")
        if self.free_space_probe_count < 1:
            raise ConfigError("start_pose.free_space_probe_count must be >= 1")


@dataclass
class ObstaclePoolConfig:
    """Deterministic Gazebo obstacle ENTITY POOL (opt-in): pre-spawn a fixed
    set of static/dynamic markers once and reuse them (teleport active
    slots to the new episode's positions, park the rest off-arena) instead
    of deleting and re-spawning every ``/reset`` -- avoids the
    delete/create service-call churn (and the resulting Ignition
    "not found, so not removed" log spam when bookkeeping ever drifts) a
    high-reset-rate training loop otherwise produces every single episode.

    Only applies to PROCEDURALLY generated (training) scenarios -- a fixed
    benchmark scenario (``evaluation_node.py``'s exact YAML-authored
    layouts) always uses the legacy spawn/delete path regardless of this
    setting, so evaluation geometry is never approximated by the pool's
    quantized size classes (see env/spawning/obstacle_pool.py's module
    docstring for the full rationale).

    Disabled by default -- byte-identical to pre-existing behaviour.
    """

    enabled: bool = False
    max_static: int = 8
    max_dynamic: int = 4
    # Static markers are plain cylinders (bypassing the drl_obstacle_assets
    # catalog meshes used by the legacy path) whose radius is SNAPPED UP to
    # the nearest class here -- keeps the pool's REAL Gazebo collision
    # geometry exactly consistent with whatever radius
    # generate_scenario's feasibility checks assumed, at the cost of some
    # visual/geometric diversity relative to the legacy catalog-backed path.
    # Must be sorted ascending; the largest class must cover
    # generate_scenario's own max static-obstacle radius (0.5 m).
    static_size_classes_m: List[float] = field(default_factory=lambda: [0.5])
    # Parked slots are placed in a grid this far beyond
    # (scenario.world_size_m/2 + observation.lidar_max_range_m) -- far
    # enough outside the arena that no LiDAR ray from anywhere inside it
    # can ever reach a parked marker.
    parking_margin_m: float = 5.0

    def validate(self) -> None:
        if self.max_static < 0:
            raise ConfigError("obstacle_pool.max_static must be >= 0")
        if self.max_dynamic < 0:
            raise ConfigError("obstacle_pool.max_dynamic must be >= 0")
        if self.enabled and not self.static_size_classes_m:
            raise ConfigError("obstacle_pool.static_size_classes_m must be non-empty when enabled")
        if any(c <= 0.0 for c in self.static_size_classes_m):
            raise ConfigError("obstacle_pool.static_size_classes_m entries must be > 0")
        if list(self.static_size_classes_m) != sorted(self.static_size_classes_m):
            raise ConfigError("obstacle_pool.static_size_classes_m must be sorted ascending")
        if self.parking_margin_m <= 0.0:
            raise ConfigError("obstacle_pool.parking_margin_m must be > 0")


@dataclass
class TrainingConfig:
    max_timesteps: int = 2_000_000
    timesteps_before_training: int = 12_000
    eval_freq: int = 12_000
    eval_episodes: int = 20
    episode_length_steps: int = 600
    seed: int = 0
    # Full observations are large, so this remains opt-in for long runs.
    # Even when False, physical state/goal/reward/risk/trajectory summaries
    # are still logged every step.
    log_full_state: bool = False

    def validate(self) -> None:
        if self.max_timesteps <= 0:
            raise ConfigError("training.max_timesteps must be > 0")
        if self.timesteps_before_training < 0:
            raise ConfigError("training.timesteps_before_training must be >= 0")
        if self.episode_length_steps <= 0:
            raise ConfigError("training.episode_length_steps must be > 0")


def _validate_range(name: str, r: List[float]) -> None:
    if len(r) != 2 or r[0] > r[1]:
        raise ConfigError(f"{name} must be [lo, hi] with lo<=hi, got {r}")


@dataclass
class DomainRandomizationConfig:
    """Per-episode dynamics/sensor randomization ranges (section 30) --
    sim-to-real prep. Every range defaults to a single-point [1.0, 1.0] /
    [0.0, 0.0] (no randomization) so this is opt-in per profile."""

    enabled: bool = False
    mass_scale_range: List[float] = field(default_factory=lambda: [1.0, 1.0])
    friction_scale_range: List[float] = field(default_factory=lambda: [1.0, 1.0])
    wheel_radius_scale_range: List[float] = field(default_factory=lambda: [1.0, 1.0])
    steering_gain_range: List[float] = field(default_factory=lambda: [1.0, 1.0])
    steering_delay_sec_range: List[float] = field(default_factory=lambda: [0.0, 0.0])
    velocity_response_scale_range: List[float] = field(default_factory=lambda: [1.0, 1.0])
    command_latency_sec_range: List[float] = field(default_factory=lambda: [0.0, 0.0])
    lidar_range_noise_std_m_range: List[float] = field(default_factory=lambda: [0.0, 0.0])
    lidar_dropout_prob_range: List[float] = field(default_factory=lambda: [0.0, 0.0])
    odometry_noise_std_range: List[float] = field(default_factory=lambda: [0.0, 0.0])
    sensor_frame_drop_prob_range: List[float] = field(default_factory=lambda: [0.0, 0.0])

    def validate(self) -> None:
        for name in (
            "mass_scale_range", "friction_scale_range", "wheel_radius_scale_range",
            "steering_gain_range", "steering_delay_sec_range", "velocity_response_scale_range",
            "command_latency_sec_range", "lidar_range_noise_std_m_range",
            "lidar_dropout_prob_range", "odometry_noise_std_range", "sensor_frame_drop_prob_range",
        ):
            _validate_range(f"domain_randomization.{name}", getattr(self, name))
        # Delay/noise-std ranges must be non-negative (section P2).
        for name in ("steering_delay_sec_range", "command_latency_sec_range",
                     "lidar_range_noise_std_m_range", "odometry_noise_std_range"):
            r = getattr(self, name)
            if r[0] < 0.0:
                raise ConfigError(f"domain_randomization.{name} must be >= 0, got {r}")
        # Probability-typed ranges must stay within [0, 1].
        for name in ("lidar_dropout_prob_range", "sensor_frame_drop_prob_range"):
            r = getattr(self, name)
            if not (0.0 <= r[0] and r[1] <= 1.0):
                raise ConfigError(f"domain_randomization.{name} must be within [0, 1], got {r}")


@dataclass
class SensorNoiseConfig:
    """Fixed-shape, CURRICULUM-INDEPENDENT sensor/localization error model
    (this package has no drl_agent-style curriculum stages -- every
    magnitude here is a single profile-authored constant applied uniformly
    for the whole run, unlike ``domain_randomization`` above, which draws a
    NEW magnitude every episode from a configured RANGE for sim-to-real
    training-robustness purposes). Applies ON TOP OF (after)
    ``domain_randomization``'s own LiDAR/odometry noise when both are
    enabled -- see ``env/simulation/sensor_noise.py``'s module docstring
    for the exact composition order and the observation/ground-truth
    separation this module enforces.

    Disabled by default -- byte-identical to pre-existing behaviour. Every
    field is independently zero/False-able without touching any other."""

    enabled: bool = False
    lidar_range_noise_std_m: float = 0.0
    lidar_bias_m: float = 0.0
    lidar_dropout_prob: float = 0.0
    localization_xy_noise_std_m: float = 0.0
    localization_yaw_noise_std_rad: float = 0.0
    # OU (Ornstein-Uhlenbeck) drift on the MEASURED (x, y, yaw) -- models a
    # slowly-varying localization bias (e.g. SLAM drift), distinct from the
    # i.i.d.-per-step noise above. Integrated via the EXACT discrete-time OU
    # update (env/simulation/sensor_noise.py::_ou_step), not Euler-Maruyama.
    # theta = mean-reversion rate (1/s, >= 0). sigma = the SDE's own
    # DIFFUSION COEFFICIENT (dX = -theta*X*dt + sigma*dW), units
    # <state-unit>/sqrt(sec) -- NOT a per-step standard deviation; the
    # actual per-tick noise std is derived from sigma, theta, and dt
    # together. A zero sigma disables its own drift axis entirely (theta is
    # then irrelevant) -- byte-identical.
    localization_drift_theta: float = 1.0
    localization_drift_xy_sigma_m: float = 0.0
    localization_drift_yaw_sigma_rad: float = 0.0
    # The observation's localization reading is the (noise+drift-applied)
    # pose from this many CONTROL TICKS ago -- 0 disables latency (the
    # reading is always this tick's own).
    localization_latency_steps: int = 0
    velocity_noise_std_mps: float = 0.0
    yaw_rate_noise_std_radps: float = 0.0
    steering_noise_std_rad: float = 0.0

    def validate(self) -> None:
        for name in (
            "lidar_range_noise_std_m", "lidar_dropout_prob", "localization_xy_noise_std_m",
            "localization_yaw_noise_std_rad", "localization_drift_theta", "localization_drift_xy_sigma_m",
            "localization_drift_yaw_sigma_rad", "velocity_noise_std_mps", "yaw_rate_noise_std_radps",
            "steering_noise_std_rad",
        ):
            if getattr(self, name) < 0.0:
                raise ConfigError(f"sensor_noise.{name} must be >= 0")
        if not (0.0 <= self.lidar_dropout_prob <= 1.0):
            raise ConfigError("sensor_noise.lidar_dropout_prob must be in [0, 1]")
        if self.localization_latency_steps < 0:
            raise ConfigError("sensor_noise.localization_latency_steps must be >= 0")


@dataclass
class RewardConfig:
    """Minimal reward (section 24 -- deliberately NOT the research novelty,
    kept small so it never masks the risk-critic/counterfactual signal)."""

    goal_reached_reward: float = 100.0
    collision_penalty: float = -100.0
    goal_threshold_m: float = 0.42
    progress_weight: float = 1.0
    step_penalty: float = -0.01
    trajectory_smoothness_weight: float = 0.0
    control_smoothness_weight: float = 0.05

    def validate(self) -> None:
        if self.goal_threshold_m <= 0.0:
            raise ConfigError("reward.goal_threshold_m must be > 0")
        for name in ("progress_weight", "trajectory_smoothness_weight", "control_smoothness_weight"):
            if getattr(self, name) < 0.0:
                raise ConfigError(f"reward.{name} must be >= 0")
        # goal_reached_reward/collision_penalty/step_penalty are added to the
        # total DIRECTLY (env/rewards/reward_calculator.py's compute_reward),
        # unlike the weight fields above which scale an already-signed term
        # -- their OWN sign is what makes them a reward vs. a penalty. A
        # flipped sign here would silently reward colliding or punish
        # reaching the goal, with training still running "successfully" and
        # nothing else in the pipeline able to catch it.
        if self.goal_reached_reward < 0.0:
            raise ConfigError(
                "reward.goal_reached_reward must be >= 0 (added directly on success -- "
                "a negative value would punish reaching the goal)"
            )
        if self.collision_penalty > 0.0:
            raise ConfigError(
                "reward.collision_penalty must be <= 0 (added directly on collision -- "
                "a positive value would reward colliding)"
            )
        if self.step_penalty > 0.0:
            raise ConfigError(
                "reward.step_penalty must be <= 0 (added directly every non-terminal step -- "
                "a positive value would reward stalling)"
            )


@dataclass
class RuntimeConfig:
    """ROS/Gazebo runtime timing -- config-driven, not literals scattered
    through environment_node.py (section 3.2's bounded-timeout requirement)."""

    time_delta_sec: float = 0.1
    reset_settle_time_sec: float = 0.2
    gz_service_wait_timeout_sec: float = 5.0
    gz_service_call_timeout_sec: float = 3.0
    gz_service_poll_sec: float = 0.5
    sensor_freshness_timeout_sec: float = 1.5
    # section P0-9: bounded RETRY budget for a stale-sensor /reset before it
    # FAILS LOUDLY (RuntimeError) instead of silently returning a stale
    # initial observation -- each retry re-runs reset_settle_time_sec of
    # physics + another full sensor_freshness_timeout_sec wait, so this is
    # NOT free (worst case: (1+N) * (reset_settle_time_sec +
    # sensor_freshness_timeout_sec) before giving up). 0 disables retrying
    # (fail immediately on the first stale read).
    sensor_freshness_max_reset_retries: int = 2
    # Independent of any single /step call: if no NEW command has been
    # published within this long, a background watchdog timer publishes an
    # explicit zero command -- protects against the TRAINER PROCESS itself
    # dying/hanging (not just one Gazebo service call), so a real robot
    # never coasts on a stale command indefinitely (section 13).
    watchdog_command_timeout_sec: float = 1.0
    watchdog_period_sec: float = 0.2
    # "simulation" (default, trainable) or "real_hardware" (inference-only --
    # nodes/train_node.py refuses to train against a profile marked this
    # way; section P2: "real profile에서 training 금지"). Only
    # real_hunter_safe.yaml sets this to "real_hardware".
    deployment: str = "simulation"
    # section P0-7: propagate_state's default wall-clock time.sleep(duration)
    # -> unpause/sleep/pause makes the ACTUAL amount of simulated time that
    # elapses depend on Gazebo's real-time-factor at that moment (system
    # load, physics complexity, GPU/CPU contention) -- confirmed live: two
    # fresh training runs with IDENTICAL seeds produced DIFFERENT episode
    # lengths/collision timing purely from this. When True, GazeboRuntimeMixin
    # instead uses Ignition's ControlWorld.multi_step field to advance the
    # simulation by an EXACT physics-step count (gazebo_max_step_size_sec
    # must match the running world SDF's own <max_step_size>), confirmed via
    # /clock. Default False (opt-in): this is a NEW mechanism through the
    # same core stepping path every profile depends on, so it stays
    # explicit-only until proven across more world configurations, per
    # every other new/risky feature in this package's config-gated-off
    # convention.
    deterministic_stepping: bool = False
    gazebo_max_step_size_sec: float = 0.001
    clock_confirm_timeout_sec: float = 2.0
    # code review (physics-step tolerance/contract bug): ``physics_step_tolerance_sec``
    # below is the GENERAL runtime tolerance ``multi_step_advance`` is
    # judged against for an ordinary, FULL-DURATION advance (``propagate_state``'s
    # ``expected_dt_sec`` is a whole ``time_delta_sec``/``reset_settle_time_sec``,
    # ~0.1s by default) -- deliberately sized against THAT scale (a few
    # percent of a control period), not against one raw physics step.
    # Reusing it for the ONE-TIME calibration probe
    # (``verify_physics_step_calibration``, which advances by exactly
    # ``n_steps=1`` and expects only ``gazebo_max_step_size_sec`` itself,
    # e.g. 0.001s) was the actual bug: with the shipped defaults
    # (``gazebo_max_step_size_sec=0.001``, this field's old default of
    # ``0.005``), a connected Gazebo world whose REAL ``<max_step_size>``
    # was 0.002s instead of the declared 0.001s advanced the calibration
    # probe by 0.002s -- ``abs(0.002 - 0.001) == 0.001 <= 0.005`` still
    # PASSED, silently certifying a world that was genuinely double the
    # declared step size. ``physics_step_calibration_tolerance_sec`` below
    # is the FIX: its own, separate, single-step-scaled tolerance used
    # ONLY by ``verify_physics_step_calibration`` -- see that field's own
    # docstring for the bound ``validate()`` enforces to make this class of
    # near-miss mismatch structurally undetectable-proof, not just fixed
    # for today's default numbers.
    physics_step_tolerance_sec: float = 0.005
    # code review (physics-step tolerance/contract bug): the calibration-
    # specific tolerance ``GazeboRuntimeMixin.verify_physics_step_calibration``
    # judges its single-step (``n_steps=1``, ``expected_dt_sec==
    # gazebo_max_step_size_sec``) probe against -- INTENTIONALLY separate
    # from ``physics_step_tolerance_sec`` (see that field's docstring for
    # why reusing it masked a real 0.001s-vs-0.002s world mismatch).
    # ``validate()`` enforces ``0 < this < 0.5 * gazebo_max_step_size_sec``:
    # that bound guarantees ANY whole-step-size-scale mismatch (the
    # declared value being off by a factor of >=2 in either direction from
    # the connected world's real step, e.g. 0.001 vs 0.002) always exceeds
    # tolerance and is rejected, while staying far above realistic IEEE-754
    # double-precision /clock accumulation error for these magnitudes
    # (~1e-9s, many orders of magnitude below any reasonable value here) --
    # so ordinary floating-point jitter is never mistaken for a genuine
    # step-size mismatch. Default (0.0002s = 20% of the shipped
    # ``gazebo_max_step_size_sec=0.001``) is deliberately NOT auto-scaled
    # to a differently-configured ``gazebo_max_step_size_sec`` -- a profile
    # that changes the declared step size must explicitly re-choose this
    # value too (``validate()`` fails fast otherwise), rather than silently
    # inheriting a now-disproportionate absolute default.
    physics_step_calibration_tolerance_sec: float = 0.0002
    # section P0-2 (reset velocity carryover): Ignition's SetEntityPose
    # teleports POSE ONLY (ros_gz_interfaces/srv/SetEntityPose has no twist
    # field at all) -- confirmed LIVE that WorldReset.model_only=True
    # (reset_world(), called before the teleport) does NOT reliably leave
    # the robot's reported /odometry velocity at zero either: a robot
    # driven to ~0.5 m/s then reset still reported ~0.15 m/s immediately
    # after the next episode's reset completed, a real residual-momentum
    # carryover (previously flagged but not fixed, memory
    # hunter_kinodynamic_rl_reset_velocity_carryover.md). _on_reset now
    # explicitly WAITS for reported |v| to converge below this threshold
    # (publishing an explicit stop + re-settling, bounded retries) before
    # returning the initial observation -- never silently accepting
    # whatever residual velocity Gazebo reports.
    reset_velocity_threshold_mps: float = 0.05
    reset_velocity_max_retries: int = 5
    # code review (risk-telemetry correctness bug): EnvironmentClient-side
    # (trainer_base.py) bounded waits for the risk-telemetry side channel --
    # previously a hardcoded 0.3s literal, not config-exposed. Per-step wait
    # for the step_id-matching telemetry message; the reset-marker wait gets
    # a separate, more generous default because /reset itself already
    # routinely takes multiple seconds of real Gazebo round-trip time (world
    # reset/respawn/settle/sensor-wait), so a tight budget there would fail
    # spuriously on normal reset latency, not a genuine telemetry problem.
    risk_telemetry_wait_timeout_sec: float = 1.0
    risk_telemetry_reset_marker_timeout_sec: float = 5.0
    # section P0-3: bounds how long real_policy_node.py's control tick waits
    # for ONE agent.select_action() call before giving up on it for THIS
    # tick and publishing a safe stop instead. Python cannot forcibly kill a
    # genuinely hung worker thread -- see real_policy_node.py's
    # ``_infer_with_timeout`` docstring for the exact (documented, honest)
    # guarantee this does and does not provide. Default kept comfortably
    # under a 10 Hz control period (time_delta_sec=0.1) so a slow-but-not-
    # hung CPU inference still usually completes in time.
    policy_inference_timeout_sec: float = 0.4
    # section P0-3: real_policy_node.py's independent watchdog THREAD (never
    # an rclpy timer -- see that module's docstring for why) polls this
    # often and republishes STOP_COMMAND once ``watchdog_command_timeout_sec``
    # has elapsed since the last real publish, regardless of whether the
    # control-tick callback itself is still running at all.
    real_policy_watchdog_period_sec: float = 0.1
    # section item-5: 'thread' (default, the original, most heavily-verified
    # path -- an in-process worker thread per inference call, single-flight
    # guarded) or 'process' (opt-in -- a SEPARATE OS process, restartable
    # via SIGTERM/SIGKILL if it genuinely hangs, since CPython cannot
    # forcibly kill a thread). See real_policy_node.py's module docstring
    # for the documented trade-off between the two.
    inference_worker_mode: str = "thread"
    # section item-5: 'process' mode only -- how many CONSECUTIVE inference
    # timeouts (the worker never responded within policy_inference_timeout_sec)
    # trigger a forced terminate-and-respawn of the worker process, rather
    # than continuing to wait on a possibly-permanently-wedged one forever.
    inference_worker_max_consecutive_timeouts: int = 3
    # section item-5: distinct from policy_inference_timeout_sec (one TICK's
    # budget) -- how long since the LAST actually-successful inference
    # (``_last_successful_inference_time``) before the watchdog thread logs
    # a loud, throttled "policy is not producing successful inferences"
    # diagnostic. Purely a health SIGNAL: the robot is already guaranteed
    # to be safely stopped by the existing command-freshness watchdog +
    # per-tick safe-stop-on-timeout/error logic regardless of this value:
    # this only affects how quickly an operator is told the POLICY itself
    # (not just "no fresh command") is unhealthy. Must be strictly greater
    # than policy_inference_timeout_sec (otherwise it would fire on every
    # single ordinary timeout, not a sustained one).
    policy_health_timeout_sec: float = 5.0

    def validate(self) -> None:
        if self.deployment not in ("simulation", "real_hardware"):
            raise ConfigError(f"runtime.deployment must be 'simulation' or 'real_hardware', got {self.deployment!r}")
        for name in ("time_delta_sec", "reset_settle_time_sec", "gz_service_wait_timeout_sec",
                     "gz_service_call_timeout_sec", "gz_service_poll_sec", "sensor_freshness_timeout_sec",
                     "watchdog_command_timeout_sec", "watchdog_period_sec",
                     "gazebo_max_step_size_sec", "clock_confirm_timeout_sec", "physics_step_tolerance_sec",
                     "physics_step_calibration_tolerance_sec",
                     "risk_telemetry_wait_timeout_sec", "risk_telemetry_reset_marker_timeout_sec",
                     "policy_inference_timeout_sec", "real_policy_watchdog_period_sec",
                     "policy_health_timeout_sec"):
            if getattr(self, name) <= 0.0:
                raise ConfigError(f"runtime.{name} must be > 0")
        # code review (physics-step tolerance/contract bug): reject a
        # calibration tolerance disproportionately large relative to the
        # single physics step it is judged against -- this is the
        # structural fix, not just today's corrected default. Without this
        # bound a profile could still re-introduce the exact masked-mismatch
        # class this field exists to prevent (see its own docstring) by
        # setting it back to something >= half the declared step size.
        _max_calibration_tolerance = 0.5 * self.gazebo_max_step_size_sec
        if self.physics_step_calibration_tolerance_sec >= _max_calibration_tolerance:
            raise ConfigError(
                "runtime.physics_step_calibration_tolerance_sec "
                f"({self.physics_step_calibration_tolerance_sec}) must be < 0.5 * "
                f"runtime.gazebo_max_step_size_sec ({_max_calibration_tolerance}) -- otherwise a connected "
                "Gazebo world whose real physics step is a whole multiple/fraction off from the declared "
                "value (e.g. 0.002s vs a declared 0.001s) could pass calibration undetected. Choose a "
                "tighter calibration tolerance, or correct gazebo_max_step_size_sec to match the connected "
                "world's own SDF <max_step_size>."
            )
        if self.sensor_freshness_max_reset_retries < 0:
            raise ConfigError("runtime.sensor_freshness_max_reset_retries must be >= 0")
        if self.reset_velocity_threshold_mps <= 0.0:
            raise ConfigError("runtime.reset_velocity_threshold_mps must be > 0")
        if self.reset_velocity_max_retries < 0:
            raise ConfigError("runtime.reset_velocity_max_retries must be >= 0")
        if self.inference_worker_mode not in ("thread", "process"):
            raise ConfigError(
                f"runtime.inference_worker_mode must be 'thread' or 'process', got "
                f"{self.inference_worker_mode!r}"
            )
        if self.inference_worker_max_consecutive_timeouts < 1:
            raise ConfigError("runtime.inference_worker_max_consecutive_timeouts must be >= 1")
        if self.policy_health_timeout_sec <= self.policy_inference_timeout_sec:
            raise ConfigError(
                "runtime.policy_health_timeout_sec must be > policy_inference_timeout_sec "
                f"({self.policy_health_timeout_sec} <= {self.policy_inference_timeout_sec})"
            )
        if self.deterministic_stepping:
            for name in ("time_delta_sec", "reset_settle_time_sec"):
                duration = getattr(self, name)
                n = duration / self.gazebo_max_step_size_sec
                if abs(n - round(n)) > 1e-6:
                    raise ConfigError(
                        f"runtime.deterministic_stepping=true requires runtime.{name} ({duration}) to be an "
                        f"exact multiple of runtime.gazebo_max_step_size_sec ({self.gazebo_max_step_size_sec}) "
                        "-- otherwise every step silently rounds to a slightly different duration than requested"
                    )


@dataclass
class EvaluationConfig:
    """Which fixed benchmark (config/benchmarks/<name>/) an evaluation run
    replays. Empty string -- no benchmark selected (training profiles).

    section item-1/item-2 (fixed-benchmark fairness / evaluation-contract
    fingerprint): every field below is the EVALUATION-CONTRACT surface --
    deliberately owned by the requested `--profile` (e.g. `evaluation_id.yaml`),
    NEVER by whatever profile a checkpoint happened to train under, so
    every model benchmarked through the SAME evaluation profile (SAC,
    vanilla TQC, legacy-waypoint, risk-aware, counterfactual, ...) is
    scored under IDENTICAL episode-length/common-metric conditions. See
    `nodes/evaluation_node.py::build_effective_profile` (now also layers
    `evaluation` -- trivially, it always did -- but critically also
    `training` -- NEW -- from the requested profile, never the checkpoint's
    own) and `env/simulation/risk_computation.py::compute_common_evaluation_metrics`
    (the common, architecture-independent clearance/TTC/stopping-margin/
    unrecoverable yardstick every model gets assessed with during a fixed
    benchmark run, built from the REALIZED physical command rather than
    each model's own [kappa,v_ref,L] action -- works even for
    action_space.mode=legacy_waypoint, which has no rollout of its own)."""

    benchmark: str = ""
    episodes_per_scenario: int = 1
    # section item-1: authoritative per-episode step budget for BOTH
    # `evaluation/benchmark_runner.py::run_episode`'s own loop bound and
    # (via `build_effective_profile` overriding the WHOLE `training`
    # section from the requested eval profile) the live environment_node's
    # own timeout detection -- previously silently inherited from whatever
    # `training.episode_length_steps` the CHECKPOINT happened to train
    # under (e.g. `smoke_test`'s 50 vs. a research profile's 600),
    # confounding "did this model actually navigate better" with "did this
    # model just get a longer/shorter episode budget".
    max_episode_steps: int = 600
    # Common-metrics rollout: horizon_min_sec/horizon_max_sec/
    # min_safety_horizon_sec are all pinned to common_metrics_horizon_sec
    # (see compute_common_evaluation_metrics) so the L-derived-horizon
    # machinery collapses to a fixed, non-L-dependent window -- the
    # REALIZED command has no "L" at all for a legacy_waypoint model.
    common_metrics_horizon_sec: float = 2.0
    common_metrics_dt_sec: float = 0.1
    common_metrics_clearance_horizon_sec: float = 2.0
    common_metrics_ttc_horizon_sec: float = 2.0
    common_metrics_min_safe_clearance_m: float = 0.3
    common_metrics_num_candidates: int = 8
    common_metrics_kappa_offsets_frac: List[float] = field(default_factory=lambda: [-1.0, -0.5, 0.5, 1.0])
    common_metrics_speed_fractions: List[float] = field(default_factory=lambda: [0.5, 1.0])
    common_metrics_include_stop_candidate: bool = True

    def validate(self) -> None:
        if self.benchmark and self.benchmark not in ("id", "ood_geometry", "ood_dynamics", "dynamic"):
            raise ConfigError(
                f"evaluation.benchmark must be one of id|ood_geometry|ood_dynamics|dynamic, got {self.benchmark!r}"
            )
        if self.episodes_per_scenario <= 0:
            raise ConfigError("evaluation.episodes_per_scenario must be > 0")
        if self.max_episode_steps <= 0:
            raise ConfigError("evaluation.max_episode_steps must be > 0")
        if self.common_metrics_horizon_sec <= 0.0:
            raise ConfigError("evaluation.common_metrics_horizon_sec must be > 0")
        if self.common_metrics_dt_sec <= 0.0 or self.common_metrics_dt_sec > self.common_metrics_horizon_sec:
            raise ConfigError("evaluation.common_metrics_dt_sec must satisfy 0 < dt_sec <= common_metrics_horizon_sec")
        if self.common_metrics_clearance_horizon_sec <= 0.0 or self.common_metrics_ttc_horizon_sec <= 0.0:
            raise ConfigError("evaluation.common_metrics_{clearance,ttc}_horizon_sec must be > 0")
        if self.common_metrics_min_safe_clearance_m < 0.0:
            raise ConfigError("evaluation.common_metrics_min_safe_clearance_m must be >= 0")
        if self.common_metrics_num_candidates <= 0:
            raise ConfigError("evaluation.common_metrics_num_candidates must be > 0")


@dataclass
class MissionConfig:
    """Hierarchical-navigation mission lifecycle (Phase 1 --
    ``docs/HIERARCHICAL_NAVIGATION_IMPLEMENTATION_PLAN.md`` section 5.7).
    Consumed only by ``navigation/`` -- the existing local-only env/training
    path never reads this section, so it is opt-in by construction (no
    ``enabled`` flag needed)."""

    position_tolerance_m: float = 0.6
    heading_tolerance_rad: float = math.pi
    require_low_speed_on_goal: bool = True
    goal_speed_threshold_mps: float = 0.1
    reset_memory_on_goal_change: bool = True

    def validate(self) -> None:
        if self.position_tolerance_m <= 0.0:
            raise ConfigError("mission.position_tolerance_m must be > 0")
        if not (0.0 <= self.heading_tolerance_rad <= math.pi):
            raise ConfigError("mission.heading_tolerance_rad must be in [0, pi]")
        if self.goal_speed_threshold_mps < 0.0:
            raise ConfigError("mission.goal_speed_threshold_mps must be >= 0")


#: Backend names ``LocalizationConfig.backend``/the real-node backend
#: factory (``nodes/hierarchical_navigation_node.py``) actually know how to
#: construct AND feed from a live topic. ``"lidar_odom"`` is deliberately
#: excluded here (plan section 4/10.5) -- ``navigation/localization/lidar_odom_backend.py``
#: implements the ``LocalizationBackend`` Protocol and is unit-tested, but
#: this repository has no real scan-matching/ICP/NDT algorithm to produce
#: the relative-transform input it needs, so nothing can genuinely feed it
#: from a live topic yet; accepting it as a valid profile value would
#: promise a backend swap this codebase cannot actually perform.
SUPPORTED_LOCALIZATION_BACKENDS = ("odom", "gazebo_odom", "wheel_imu", "lio")


@dataclass
class LocalizationConfig:
    """Localization backend selection + validity gating (Phase 1 section
    5.7, extended Phase 5/6 section 4/10.5). ``backend="odom"`` is the
    ROS-free
    :class:`~hunter_kinodynamic_rl.navigation.localization.odom_backend.OdomLocalizationBackend`
    (tests, replay); ``"gazebo_odom"`` additionally parses
    ``nav_msgs/Odometry`` covariance into a confidence estimate (see
    ``navigation/localization/gazebo_odom_backend.py``); ``"wheel_imu"``
    dead-reckons from the SAME odometry topic's twist (a documented,
    simulation-only stand-in for a real wheel+IMU driver -- see
    ``navigation/localization/wheel_imu_backend.py``'s module docstring);
    ``"lio"`` parses a LIO-SAM-shaped ``nav_msgs/Odometry`` topic (see
    ``navigation/localization/lio_adapter.py``). See
    ``SUPPORTED_LOCALIZATION_BACKENDS`` for why ``"lidar_odom"`` is not yet
    a valid value here."""

    backend: str = "gazebo_odom"
    odom_topic: str = "/odometry"
    pose_timeout_sec: float = 0.5
    minimum_confidence: float = 0.5
    publish_mission_tf: bool = True

    def validate(self) -> None:
        if self.backend not in SUPPORTED_LOCALIZATION_BACKENDS:
            raise ConfigError(
                f"localization.backend must be one of {SUPPORTED_LOCALIZATION_BACKENDS}, got {self.backend!r}"
            )
        if self.pose_timeout_sec <= 0.0:
            raise ConfigError("localization.pose_timeout_sec must be > 0")
        if not (0.0 <= self.minimum_confidence <= 1.0):
            raise ConfigError("localization.minimum_confidence must be in [0, 1]")


@dataclass
class MappingConfig:
    """Online partial-map accumulation (Phase 1 section 5.4/5.5/5.6).
    ``free_threshold`` MUST stay strictly below ``occupied_threshold`` --
    otherwise a single log-odds value could satisfy both the ``occupied``
    and ``free`` channel predicates at once, which would violate the
    spec's explicit UNKNOWN/FREE/OCCUPIED exclusivity requirement.

    A strict ``free_threshold < occupied_threshold`` still leaves a real
    gap between them -- an observed cell whose log-odds falls inside that
    gap is neither confidently free nor confidently occupied. This is not
    a bug: it is the map's actual, documented FOURTH state,
    ``MapChannels.observed_uncertain`` (see
    ``navigation/mapping/partial_map.py``'s module docstring) -- together
    with occupied/free/unknown it exhaustively and disjointly partitions
    every cell."""

    resolution_m: float = 0.2
    mission_size_cells: int = 256
    rolling_size_cells: int = 128
    free_log_odds_delta: float = -0.4
    occupied_log_odds_delta: float = 0.85
    free_threshold: float = -0.2
    occupied_threshold: float = 0.2
    log_odds_min: float = -4.0
    log_odds_max: float = 4.0
    inflation_radius_m: float = 0.45
    visit_radius_m: float = 0.4
    visited_count_saturation: int = 1000
    failure_count_saturation: int = 100

    def validate(self) -> None:
        if self.resolution_m <= 0.0:
            raise ConfigError("mapping.resolution_m must be > 0")
        if self.mission_size_cells <= 0:
            raise ConfigError("mapping.mission_size_cells must be > 0")
        if self.rolling_size_cells <= 0:
            raise ConfigError("mapping.rolling_size_cells must be > 0")
        if self.free_log_odds_delta >= 0.0:
            raise ConfigError("mapping.free_log_odds_delta must be < 0 (a free observation must DECREASE log-odds)")
        if self.occupied_log_odds_delta <= 0.0:
            raise ConfigError(
                "mapping.occupied_log_odds_delta must be > 0 (an occupied observation must INCREASE log-odds)"
            )
        if self.free_threshold >= self.occupied_threshold:
            raise ConfigError(
                "mapping.free_threshold must be strictly < occupied_threshold, otherwise a single log-odds "
                "value could be classified as both free and occupied (channel exclusivity violation)"
            )
        if self.log_odds_min >= self.log_odds_max:
            raise ConfigError("mapping.log_odds_min must be < log_odds_max")
        if not (self.log_odds_min <= self.free_threshold and self.occupied_threshold <= self.log_odds_max):
            raise ConfigError("mapping.{free,occupied}_threshold must lie within [log_odds_min, log_odds_max]")
        if self.inflation_radius_m < 0.0:
            raise ConfigError("mapping.inflation_radius_m must be >= 0")
        if self.visit_radius_m <= 0.0:
            raise ConfigError("mapping.visit_radius_m must be > 0")
        if not (0 < self.visited_count_saturation <= 65535):
            raise ConfigError("mapping.visited_count_saturation must be in (0, 65535] (uint16 storage)")
        if not (0 < self.failure_count_saturation <= 255):
            raise ConfigError("mapping.failure_count_saturation must be in (0, 255] (uint8 storage)")


@dataclass
class HierarchyConfig:
    """Phase 2 hierarchical local/global goal separation
    (``docs/HIERARCHICAL_NAVIGATION_IMPLEMENTATION_PLAN.md`` section 6).
    Consumed only by ``navigation/hierarchy`` + ``navigation/local_rl`` --
    the existing local-only env/training path never reads this section, so
    it is opt-in by construction, matching ``mission``/``localization``/
    ``mapping`` (Phase 1)."""

    subgoal_position_tolerance_m: float = 0.5
    subgoal_heading_tolerance_rad: float = math.pi
    subgoal_require_low_speed_on_reach: bool = False
    subgoal_goal_speed_threshold_mps: float = 0.15
    # replanning.py's local-option-failure/timeout thresholds (plan section 6.5).
    local_option_timeout_steps: int = 200
    no_progress_window_steps: int = 40
    no_progress_min_delta_m: float = 0.1
    consecutive_emergency_stop_limit: int = 5
    local_risk_threshold: float = 0.8
    localization_min_confidence: float = 0.5
    # failure_recovery.py's retry budget before giving up on a subgoal and
    # advancing to the next candidate in the sequence.
    max_retries_per_subgoal: int = 1
    # Mission-level (not per-subgoal) wall clock/step budget -- None disables
    # the check (a coordinator with no mission_timeout configured never sets
    # mission_timed_out on its own).
    mission_timeout_steps: Optional[int] = None
    mission_timeout_sec: Optional[float] = None

    def validate(self) -> None:
        if self.subgoal_position_tolerance_m <= 0.0:
            raise ConfigError("hierarchy.subgoal_position_tolerance_m must be > 0")
        if not (0.0 <= self.subgoal_heading_tolerance_rad <= math.pi):
            raise ConfigError("hierarchy.subgoal_heading_tolerance_rad must be in [0, pi]")
        if self.subgoal_goal_speed_threshold_mps < 0.0:
            raise ConfigError("hierarchy.subgoal_goal_speed_threshold_mps must be >= 0")
        if self.local_option_timeout_steps <= 0:
            raise ConfigError("hierarchy.local_option_timeout_steps must be > 0")
        if self.no_progress_window_steps <= 0:
            raise ConfigError("hierarchy.no_progress_window_steps must be > 0")
        if self.no_progress_min_delta_m < 0.0:
            raise ConfigError("hierarchy.no_progress_min_delta_m must be >= 0")
        if self.consecutive_emergency_stop_limit <= 0:
            raise ConfigError("hierarchy.consecutive_emergency_stop_limit must be > 0")
        if self.local_risk_threshold <= 0.0:
            raise ConfigError("hierarchy.local_risk_threshold must be > 0")
        if not (0.0 <= self.localization_min_confidence <= 1.0):
            raise ConfigError("hierarchy.localization_min_confidence must be in [0, 1]")
        if self.max_retries_per_subgoal < 0:
            raise ConfigError("hierarchy.max_retries_per_subgoal must be >= 0")
        if self.mission_timeout_steps is not None and self.mission_timeout_steps <= 0:
            raise ConfigError("hierarchy.mission_timeout_steps must be > 0 when set")
        if self.mission_timeout_sec is not None and self.mission_timeout_sec <= 0.0:
            raise ConfigError("hierarchy.mission_timeout_sec must be > 0 when set")


@dataclass
class LongHorizonWorldConfig:
    """Phase 3 long-horizon procedural world (``docs/HIERARCHICAL_NAVIGATION_IMPLEMENTATION_PLAN.md``
    section 7) -- room/corridor/junction/loop/dead-end grid-lattice maze,
    seed-deterministic, train/validation/test seed pools kept separate from
    (and independent of) ``scenario``'s own pools (the existing arena/circle
    obstacle procedural generator this section does NOT replace -- see
    ``env/scenarios/long_horizon_generator.py``'s module docstring).
    Consumed only by ``env/scenarios/long_horizon_generator.py`` +
    ``env/scenarios/long_horizon_solvability.py`` -- the existing
    ``scenario``-based procedural arena and the Phase 1/2 navigation stack
    never read this section, so it is opt-in by construction (default
    ``enabled=False``), matching ``mission``/``localization``/``mapping``/
    ``hierarchy``.

    ``alternative_route_min_count`` is guaranteed by construction: every
    edge added on top of the maze's spanning tree (the graph's cyclomatic
    number, i.e. ``loop_count``) creates at least one cycle, hence at least
    one alternative route between any two nodes on that cycle -- so
    ``validate()`` requires ``loop_count_range[0] >= alternative_route_min_count``
    rather than the generator having to separately search for and count
    alternative routes at generation time."""

    enabled: bool = False
    size_m: float = 40.0
    resolution_m: float = 0.25
    wall_thickness_m: float = 0.2
    wall_height_m: float = 1.0
    corridor_width_min_m: float = 2.0
    corridor_width_max_m: float = 4.0
    room_count_range: List[int] = field(default_factory=lambda: [4, 10])
    dead_end_count_range: List[int] = field(default_factory=lambda: [1, 5])
    loop_count_range: List[int] = field(default_factory=lambda: [1, 4])
    alternative_route_min_count: int = 1
    start_goal_geodesic_min_m: float = 20.0
    require_ackermann_feasibility: bool = True
    generation_attempt_limit: int = 100
    train_seed_range: List[int] = field(default_factory=lambda: [0, 9999])
    validation_seed_range: List[int] = field(default_factory=lambda: [10000, 11999])
    test_seed_range: List[int] = field(default_factory=lambda: [12000, 13999])
    goal_radius_m: float = 0.5
    start_yaw_sampling_attempts: int = 50
    start_yaw_front_safety_distance_m: float = 0.6

    def validate(self) -> None:
        if self.size_m <= 0.0:
            raise ConfigError("long_horizon_world.size_m must be > 0")
        if self.resolution_m <= 0.0:
            raise ConfigError("long_horizon_world.resolution_m must be > 0")
        if self.wall_thickness_m <= 0.0:
            raise ConfigError("long_horizon_world.wall_thickness_m must be > 0")
        if self.wall_height_m <= 0.0:
            raise ConfigError("long_horizon_world.wall_height_m must be > 0")
        if self.corridor_width_min_m <= 0.0 or self.corridor_width_max_m < self.corridor_width_min_m:
            raise ConfigError(
                "long_horizon_world.{corridor_width_min_m,corridor_width_max_m} must satisfy "
                "0 < min <= max"
            )
        # env/scenarios/long_horizon_generator.py floors its lattice to a
        # minimum of grid_n=3 cells per side -- a size_m too small to fit
        # even that at the widest configured corridor is a config error,
        # caught here at load time rather than as a generation_attempt_limit
        # exhaustion deep inside a training run.
        if self.size_m < 3.0 * (self.corridor_width_max_m + self.wall_thickness_m):
            raise ConfigError(
                f"long_horizon_world.size_m ({self.size_m}) must be >= 3 * (corridor_width_max_m + "
                f"wall_thickness_m) ({3.0 * (self.corridor_width_max_m + self.wall_thickness_m)}) -- "
                "otherwise even a minimal 3x3 lattice cannot fit the widest configured corridor"
            )
        for name in ("room_count_range", "dead_end_count_range", "loop_count_range"):
            r = getattr(self, name)
            if len(r) != 2 or r[0] < 0 or r[1] < r[0]:
                raise ConfigError(f"long_horizon_world.{name} must be [lo, hi] with 0<=lo<=hi, got {r}")
        if self.alternative_route_min_count < 0:
            raise ConfigError("long_horizon_world.alternative_route_min_count must be >= 0")
        if self.loop_count_range[0] < self.alternative_route_min_count:
            raise ConfigError(
                "long_horizon_world.loop_count_range[0] "
                f"({self.loop_count_range[0]}) must be >= alternative_route_min_count "
                f"({self.alternative_route_min_count}) -- every loop edge guarantees at least one "
                "alternative route, so the minimum drawable loop_count must already cover the "
                "minimum required alternative-route count"
            )
        if self.start_goal_geodesic_min_m < 0.0:
            raise ConfigError("long_horizon_world.start_goal_geodesic_min_m must be >= 0")
        if self.generation_attempt_limit <= 0:
            raise ConfigError("long_horizon_world.generation_attempt_limit must be > 0")
        if self.goal_radius_m <= 0.0:
            raise ConfigError("long_horizon_world.goal_radius_m must be > 0")
        if self.start_yaw_sampling_attempts < 1:
            raise ConfigError("long_horizon_world.start_yaw_sampling_attempts must be >= 1")
        if self.start_yaw_front_safety_distance_m <= 0.0:
            raise ConfigError("long_horizon_world.start_yaw_front_safety_distance_m must be > 0")
        ranges = [self.train_seed_range, self.validation_seed_range, self.test_seed_range]
        for r in ranges:
            if len(r) != 2 or r[0] > r[1]:
                raise ConfigError(f"long_horizon_world seed range must be [lo, hi] with lo<=hi, got {r}")
        lo_hi = sorted(ranges, key=lambda r: r[0])
        for a, b in zip(lo_hi, lo_hi[1:]):
            if a[1] >= b[0]:
                raise ConfigError(
                    "long_horizon_world train/validation/test seed ranges must not overlap: "
                    f"{a} overlaps {b}"
                )


@dataclass
class WallSegmentPoolConfig:
    """Deterministic Gazebo wall-segment ENTITY POOL (opt-in), mirroring
    ``ObstaclePoolConfig``'s "pre-spawn once, teleport per episode" design
    (see ``env/spawning/wall_segment_spawner.py``'s module docstring) --
    only meaningful when ``long_horizon_world.enabled``. Only LENGTH is
    quantized (to the nearest ``length_classes_m`` class, rounded UP, at
    ACTIVATION time -- never at generation time, unlike
    ``ObstaclePoolConfig.static_size_classes_m``): thickness/height are a
    single fixed value per world (``LongHorizonWorldConfig.wall_thickness_m``/
    ``wall_height_m``), so only segment length needs a discrete class set for
    a fixed-geometry SDF pool slot to be reusable across differently-shaped
    mazes without a respawn. Disabled by default -- byte-identical to
    pre-existing behaviour."""

    enabled: bool = False
    max_segments: int = 400
    length_classes_m: List[float] = field(default_factory=lambda: [1.0, 2.0, 3.0, 4.0, 5.0])
    parking_margin_m: float = 5.0

    def validate(self) -> None:
        if self.max_segments < 0:
            raise ConfigError("wall_segment_pool.max_segments must be >= 0")
        # code review: an enabled pool with max_segments=0 previously
        # validated cleanly and only failed loudly much later, at runtime,
        # inside activate_walls (the first time a real world with >= 1 wall
        # segment tried to use it) -- caught here instead, at load time.
        if self.enabled and self.max_segments <= 0:
            raise ConfigError("wall_segment_pool.max_segments must be > 0 when enabled")
        if self.enabled and not self.length_classes_m:
            raise ConfigError("wall_segment_pool.length_classes_m must be non-empty when enabled")
        if any(c <= 0.0 for c in self.length_classes_m):
            raise ConfigError("wall_segment_pool.length_classes_m entries must be > 0")
        if list(self.length_classes_m) != sorted(self.length_classes_m):
            raise ConfigError("wall_segment_pool.length_classes_m must be sorted ascending")
        if self.parking_margin_m <= 0.0:
            raise ConfigError("wall_segment_pool.parking_margin_m must be > 0")


@dataclass
class GlobalRLConfig:
    """Phase 4 Global RL MVP (``docs/HIERARCHICAL_NAVIGATION_IMPLEMENTATION_PLAN.md``
    section 8) -- discrete candidate-subgoal action space, masked-DQN
    network sizing, replay/agent hyperparameters, and the option-level
    Global reward weights (section 8.7/14). Consumed only by
    ``navigation/global_rl`` + ``training/train_hierarchical_dqn.py`` --
    opt-in by construction (default ``enabled=False``), matching
    ``mission``/``localization``/``mapping``/``hierarchy``/``long_horizon_world``.
    Bundles the reward weights alongside the action/network/replay knobs
    (rather than a separate section) -- the same "one flat dataclass per
    opt-in subsystem" shape ``HierarchyConfig`` already uses for its own
    subgoal+replanning+recovery thresholds.

    ``direction_degrees``/``distances_m`` form the robot-relative candidate
    grid (``len(direction_degrees) * len(distances_m)`` candidates); exactly
    one additional fallback candidate (``fallback_mode``) is always appended
    last, so :attr:`n_candidates` is that product plus one. The fallback
    candidate is ALWAYS a valid action regardless of what the action mask
    computes for every other candidate (plan section 8.4)."""

    enabled: bool = False
    direction_degrees: List[float] = field(
        default_factory=lambda: [-180.0, -135.0, -90.0, -45.0, 0.0, 45.0, 90.0, 135.0]
    )
    distances_m: List[float] = field(default_factory=lambda: [3.0, 6.0])
    # "backtrack" -- fallback endpoint is a short step directly behind the
    # robot (robot-relative angle=pi); "stop_recovery" -- fallback endpoint
    # is the robot's OWN current position (zero-distance, i.e. "hold still,
    # let the next Global decision re-evaluate").
    fallback_mode: str = "backtrack"
    fallback_backtrack_distance_m: float = 2.0
    # Number of interpolated points sampled between the robot and a
    # candidate's endpoint for action_mask's short-rollout collision check
    # (plan section 8.4: "endpoint까지의 Ackermann short rollout이 known
    # obstacle과 충돌") -- includes both endpoints.
    rollout_sample_count: int = 8
    robot_footprint_radius_m: float = 0.45

    # -- Global observation (section 8.5) --------------------------------
    map_crop_size_cells: int = 96
    goal_distance_norm_m: float = 40.0
    max_speed_norm_mps: float = 1.5

    # -- Network sizing (section 8.6, masked Dueling Double DQN) ---------
    cnn_channels: List[int] = field(default_factory=lambda: [16, 32])
    map_feature_dim: int = 128
    scalar_feature_dim: int = 32
    candidate_feature_dim: int = 32
    fused_feature_dim: int = 128

    # -- Replay / agent (section 8.8/8.9) --------------------------------
    replay_capacity: int = 50_000
    batch_size: int = 64
    gamma: float = 0.99
    learning_rate: float = 1e-4
    target_update_interval_steps: int = 500
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 20_000
    warmup_options: int = 200

    # -- Global reward weights (section 8.7/14) --------------------------
    goal_reward: float = 100.0
    progress_reward_scale: float = 1.0
    exploration_reward_scale: float = 0.02
    # Caps R_exploration BEFORE it is added into R_global -- plan section
    # 8.7/14: "Exploration reward는 goal reward/final progress를 압도하지
    # 않도록 clip".
    exploration_reward_clip: float = 5.0
    revisit_penalty_scale: float = 0.05
    # Only applied when the caller reports a REPEATED dead-end/branch
    # re-entry (plan section 8.7/15: a FIRST dead-end exploration is normal
    # and must not be penalized identically to repeating one).
    repeated_deadend_penalty: float = 5.0
    local_failure_penalty_timeout: float = 2.0
    local_failure_penalty_no_progress: float = 2.0
    local_failure_penalty_blocked: float = 3.0
    local_failure_penalty_high_risk: float = 4.0
    local_failure_penalty_cancelled_by_replan: float = 1.0
    risk_penalty_scale: float = 1.0
    elapsed_penalty_per_local_step: float = 0.01
    # Only nonzero when global_risk_feedback_enabled -- prices the PREDICTED
    # risk of the candidate actually SELECTED, at decision time (never zero
    # by construction so a caller could pass it anyway; a caller only ever
    # passes a nonzero predicted_risk_at_selection when this ablation flag
    # below is on).
    predicted_risk_penalty_scale: float = 0.0

    # -- Phase 5 ablation flags (docs/HIERARCHICAL_NAVIGATION_IMPLEMENTATION_PLAN.md
    # section 9.9) -- each toggles ONE additional observation/reward input
    # on the SAME code path (never a separate branch), so ablations B-G
    # differ only in which of these are True. All default False, so a
    # profile that sets none of them reproduces Phase 4's exact
    # observation/network/reward shape byte-for-byte. ------------------
    #: Adds PartialMap's per-cell failure-count raster as a 6th map channel
    #: (ablation D's "visited/failure raster" step). Independent of
    #: topology_feedback_enabled/memory.enabled -- this is a cheap raster
    #: already tracked by PartialMap, not the node-graph tensor.
    include_failure_channel: bool = False
    #: Requirement F: the map-tensor "visited" raster channel, ON by
    #: default (so every already-existing profile that never sets this
    #: field explicitly -- Phase 4's own `hierarchical_phase4.yaml` and
    #: ablation profiles B/D/E/F/G, all pre-dating this flag -- stays
    #: byte-identical to before). Ablation B alone sets this False, making
    #: B ("Global partial map, no visited") and C ("+ visited map",
    #: everything else identical) genuinely differ in observation shape
    #: (map_tensor channel count) and therefore in
    #: `hierarchical_architecture_fingerprint` -- previously `visited` was
    #: unconditionally present, so B and C were byte-identical (plan 9.9's
    #: documented, since-fixed limitation).
    include_visited_channel: bool = True
    #: Adds 2 candidate-tensor columns (repeated-dead-end flag, branch
    #: visit-count norm) AND the topological-memory node tensor + validity
    #: mask to the Global observation. Requires memory.enabled=true.
    topology_feedback_enabled: bool = False
    #: Adds hierarchy/feasibility.py's 6 candidate-tensor columns (rollout
    #: collision, steering saturation, clearance, predicted action risk,
    #: progress-preserving, historical success rate). Requires
    #: feasibility.enabled=true.
    feasibility_feedback_enabled: bool = False
    #: Adds 1 candidate-tensor column (the feasibility-computed predicted
    #: action risk, exposed as its own "global risk" feature so the network
    #: can weigh distance against risk per plan section 13) AND enables
    #: predicted_risk_penalty_scale in the reward. Requires
    #: feasibility.enabled=true (reuses the SAME rollout, never a second
    #: risk computation).
    global_risk_feedback_enabled: bool = False

    @property
    def n_direction_distance_candidates(self) -> int:
        return len(self.direction_degrees) * len(self.distances_m)

    @property
    def n_candidates(self) -> int:
        return self.n_direction_distance_candidates + 1

    @property
    def fallback_index(self) -> int:
        return self.n_candidates - 1

    def validate(self) -> None:
        if not self.direction_degrees:
            raise ConfigError("global_rl.direction_degrees must be non-empty")
        for d in self.direction_degrees:
            if not (-180.0 <= d <= 180.0):
                raise ConfigError(f"global_rl.direction_degrees entries must be in [-180, 180], got {d}")
        if not self.distances_m:
            raise ConfigError("global_rl.distances_m must be non-empty")
        for r in self.distances_m:
            if r <= 0.0:
                raise ConfigError("global_rl.distances_m entries must be > 0")
        if self.fallback_mode not in ("backtrack", "stop_recovery"):
            raise ConfigError(
                f"global_rl.fallback_mode must be 'backtrack' or 'stop_recovery', got {self.fallback_mode!r}"
            )
        if self.fallback_backtrack_distance_m <= 0.0:
            raise ConfigError("global_rl.fallback_backtrack_distance_m must be > 0")
        if self.rollout_sample_count < 2:
            raise ConfigError("global_rl.rollout_sample_count must be >= 2 (needs at least both endpoints)")
        if self.robot_footprint_radius_m <= 0.0:
            raise ConfigError("global_rl.robot_footprint_radius_m must be > 0")
        if self.map_crop_size_cells <= 0:
            raise ConfigError("global_rl.map_crop_size_cells must be > 0")
        if self.goal_distance_norm_m <= 0.0:
            raise ConfigError("global_rl.goal_distance_norm_m must be > 0")
        if self.max_speed_norm_mps <= 0.0:
            raise ConfigError("global_rl.max_speed_norm_mps must be > 0")
        if not self.cnn_channels or any(c <= 0 for c in self.cnn_channels):
            raise ConfigError("global_rl.cnn_channels must be a non-empty list of positive ints")
        for name in ("map_feature_dim", "scalar_feature_dim", "candidate_feature_dim", "fused_feature_dim"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"global_rl.{name} must be > 0")
        if self.replay_capacity <= 0:
            raise ConfigError("global_rl.replay_capacity must be > 0")
        if self.batch_size <= 0:
            raise ConfigError("global_rl.batch_size must be > 0")
        if not (0.0 < self.gamma <= 1.0):
            raise ConfigError("global_rl.gamma must be in (0, 1]")
        if self.learning_rate <= 0.0:
            raise ConfigError("global_rl.learning_rate must be > 0")
        if self.target_update_interval_steps <= 0:
            raise ConfigError("global_rl.target_update_interval_steps must be > 0")
        if not (0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0):
            raise ConfigError("global_rl.{epsilon_end,epsilon_start} must satisfy 0 <= epsilon_end <= epsilon_start <= 1")
        if self.epsilon_decay_steps <= 0:
            raise ConfigError("global_rl.epsilon_decay_steps must be > 0")
        if self.warmup_options < 0:
            raise ConfigError("global_rl.warmup_options must be >= 0")
        if self.goal_reward <= 0.0:
            raise ConfigError("global_rl.goal_reward must be > 0")
        if self.exploration_reward_clip < 0.0:
            raise ConfigError("global_rl.exploration_reward_clip must be >= 0")
        for name in (
            "progress_reward_scale", "exploration_reward_scale", "revisit_penalty_scale",
            "repeated_deadend_penalty", "local_failure_penalty_timeout", "local_failure_penalty_no_progress",
            "local_failure_penalty_blocked", "local_failure_penalty_high_risk",
            "local_failure_penalty_cancelled_by_replan", "risk_penalty_scale", "elapsed_penalty_per_local_step",
            "predicted_risk_penalty_scale",
        ):
            if getattr(self, name) < 0.0:
                raise ConfigError(f"global_rl.{name} must be >= 0")


@dataclass
class MemoryConfig:
    """Phase 5 topological memory (``docs/HIERARCHICAL_NAVIGATION_IMPLEMENTATION_PLAN.md``
    section 9.3/9.4) -- flat YAML-facing section that
    ``training/train_hierarchical_dqn.py``'s ``memory_config_from`` splits
    into ``navigation.memory.node_manager.NodeManagerConfig`` and
    ``navigation.memory.dead_end_detector.DeadEndDetectorConfig`` (the same
    "one flat dataclass per opt-in subsystem, translated by a builder
    function" pattern ``hierarchy_config_from`` already uses for
    ``HierarchyConfig``). Opt-in (default ``enabled=False``); consumed only
    by ``navigation/memory`` + the Global RL observation/training path."""

    enabled: bool = False
    node_min_distance_m: float = 2.0
    node_heading_change_rad: float = math.pi / 3.0
    node_merge_radius_m: float = 1.0
    max_nodes: int = 64
    max_nodes_in_observation: int = 16
    node_recency_norm_steps: float = 200.0
    free_direction_threshold: int = 1
    progress_stall_window_steps: int = 20
    progress_stall_min_delta_m: float = 0.2
    repeated_estop_limit: int = 3
    evidence_vote_threshold: int = 2

    def validate(self) -> None:
        if self.node_min_distance_m <= 0.0:
            raise ConfigError("memory.node_min_distance_m must be > 0")
        if not (0.0 < self.node_heading_change_rad <= math.pi):
            raise ConfigError("memory.node_heading_change_rad must be in (0, pi]")
        if self.node_merge_radius_m <= 0.0:
            raise ConfigError("memory.node_merge_radius_m must be > 0")
        if self.node_merge_radius_m >= self.node_min_distance_m:
            raise ConfigError("memory.node_merge_radius_m must be < node_min_distance_m")
        if self.max_nodes <= 0:
            raise ConfigError("memory.max_nodes must be > 0")
        if self.max_nodes_in_observation < 0:
            raise ConfigError("memory.max_nodes_in_observation must be >= 0")
        if self.max_nodes_in_observation > self.max_nodes:
            raise ConfigError("memory.max_nodes_in_observation must be <= memory.max_nodes")
        if self.node_recency_norm_steps <= 0.0:
            raise ConfigError("memory.node_recency_norm_steps must be > 0")
        if self.free_direction_threshold < 0:
            raise ConfigError("memory.free_direction_threshold must be >= 0")
        if self.progress_stall_window_steps <= 0:
            raise ConfigError("memory.progress_stall_window_steps must be > 0")
        if self.progress_stall_min_delta_m < 0.0:
            raise ConfigError("memory.progress_stall_min_delta_m must be >= 0")
        if self.repeated_estop_limit <= 0:
            raise ConfigError("memory.repeated_estop_limit must be > 0")
        if not (1 <= self.evidence_vote_threshold <= 5):
            raise ConfigError("memory.evidence_vote_threshold must be in [1, 5]")


@dataclass
class GlobalFeasibilityConfig:
    """Phase 5 Global-Local feasibility feedback (plan section 9.6) --
    flat YAML-facing section translated by
    ``training/train_hierarchical_dqn.py``'s ``feasibility_config_from``
    into ``navigation.hierarchy.feasibility.FeasibilityConfig``. Opt-in
    (default ``enabled=False``); consumed only by
    ``navigation/hierarchy/feasibility.py`` + the Global RL observation/
    training path. Named ``GlobalFeasibilityConfig`` (not just
    ``FeasibilityConfig``) to avoid colliding with the identically-purposed
    pure-logic dataclass of that name in ``navigation/hierarchy/feasibility.py``
    -- this section IS the YAML-facing translation source for that one."""

    enabled: bool = False
    rollout_sample_count: int = 8
    clearance_search_radius_m: float = 2.0
    clearance_norm_m: float = 2.0
    historical_stats_radius_m: float = 1.0
    # Requirement E (LocalFeasibilityEvaluator wiring): the version of the
    # concrete evaluator's observation/action I/O contract
    # (navigation/hierarchy/local_feasibility_evaluator.py) -- bump whenever
    # that contract changes (e.g. a different sensor-snapshot shape or
    # progress-preserving rollout formula), NEVER when just tuning a
    # threshold below. Part of the `feasibility` section already hashed by
    # `hierarchical_architecture_fingerprint`, so a contract change is
    # guaranteed to invalidate any ablation E/F/G checkpoint trained under
    # the old contract, exactly like every other fingerprinted section.
    local_evaluator_schema_version: int = 1
    # Inference budget for one candidate's LocalFeasibilityEvaluator query
    # (navigation/hierarchy/local_feasibility_evaluator.FrozenLocalFeasibilityEvaluator),
    # bounded via the same SingleFlightThreadWorker discipline
    # live_gazebo_executor.py already uses for its own policy inference --
    # a slow/hung Local policy can never block Global decision-making past
    # this budget.
    local_evaluator_inference_timeout_sec: float = 0.5
    # A caller-supplied sensor snapshot older than this is rejected as
    # stale (evaluate() returns None -- a tracked fallback, never a
    # disguised risk=0/progress=True reading).
    local_evaluator_max_snapshot_age_sec: float = 0.5

    def validate(self) -> None:
        if self.rollout_sample_count < 2:
            raise ConfigError("feasibility.rollout_sample_count must be >= 2")
        if self.clearance_search_radius_m <= 0.0:
            raise ConfigError("feasibility.clearance_search_radius_m must be > 0")
        if self.clearance_norm_m <= 0.0:
            raise ConfigError("feasibility.clearance_norm_m must be > 0")
        if self.historical_stats_radius_m <= 0.0:
            raise ConfigError("feasibility.historical_stats_radius_m must be > 0")
        if self.local_evaluator_schema_version < 1:
            raise ConfigError("feasibility.local_evaluator_schema_version must be >= 1")
        if self.local_evaluator_inference_timeout_sec <= 0.0:
            raise ConfigError("feasibility.local_evaluator_inference_timeout_sec must be > 0")
        if self.local_evaluator_max_snapshot_age_sec <= 0.0:
            raise ConfigError("feasibility.local_evaluator_max_snapshot_age_sec must be > 0")


@dataclass
class HierarchicalTrainingConfig:
    """Phase 4 ROS-free hierarchical training-loop knobs
    (``docs/HIERARCHICAL_NAVIGATION_IMPLEMENTATION_PLAN.md`` section 8.9) --
    separate from :class:`GlobalRLConfig` (network/replay/reward) and
    :class:`HierarchyConfig` (Phase 2 subgoal lifecycle, reused as-is by
    Phase 4). Opt-in (default ``enabled=False``); consumed only by
    ``training/train_hierarchical_dqn.py`` and the Phase 4 node adapters."""

    enabled: bool = False
    # FROZEN local-policy checkpoint (rl.algorithms.kinodynamic_tqc or tqc)
    # to load -- Phase 4 never trains the local policy (plan section 8.9
    # item 7: "Local policy를 joint fine-tuning하지 않는다"). Split into a
    # (directory, tag) pair -- NEVER a single combined file path -- to
    # match ``rl.checkpointing.manager.load_generation``'s own contract
    # (and every other checkpoint-loading entrypoint in this package, e.g.
    # ``real_policy_node.py``'s ``checkpoint_dir``/``checkpoint_name``
    # parameters): ``local_checkpoint_dir`` is the run's ``checkpoints/``
    # directory (containing the ``<tag>`` symlink into
    # ``.generations/<uuid>/``), ``local_checkpoint_name`` is the tag
    # (``"final"``/``"latest"``/``"best"``). The resolved
    # ``<local_checkpoint_dir>/<local_checkpoint_name>/model.pt`` is hashed
    # (sha256) and recorded into the Global checkpoint manifest (see
    # train_hierarchical_dqn.py) so a resumed/evaluated run can verify it
    # was produced against the SAME frozen local checkpoint.
    local_checkpoint_dir: str = ""
    local_checkpoint_name: str = "final"
    # Profile whose scenario section defines the distribution the frozen
    # Local policy is required to have trained against.  A hierarchical
    # profile's own scenario is a long-horizon Global world, so it cannot be
    # used for this comparison.  The live executor loads this named profile
    # and compares its strict local-training-contract fingerprint with the
    # checkpoint manifest before creating any ROS background thread.
    local_training_profile_name: str = ""
    # Local and Global devices are deliberately independent.  In particular,
    # a CUDA-authored Local checkpoint is loadable on CPU via map_location;
    # this gives live=True a supported CPU-only path when an in-process CUDA
    # runtime conflicts with rclpy/DDS.
    local_inference_device: str = "auto"
    global_training_device: str = "auto"
    max_local_steps_per_option: int = 150
    max_global_options_per_mission: int = 40
    # env/scenarios/long_horizon_curriculum.py level (1-6) applied to
    # long_horizon_world before generation -- curriculum-only; leaves
    # long_horizon_world's own train/validation/test seed ranges untouched.
    long_horizon_level: int = 1
    # item 9 (code review, training-update-cadence finding): the Global
    # trainer previously ran EXACTLY ONE optimizer update per MISSION,
    # regardless of how many Global option-transitions that mission
    # actually stored into replay (a mission with max_global_options_per_mission=40
    # options still got only 1 update) -- with a default 1000-mission run,
    # the real update count could be a small fraction of what a
    # comparable flat-RL setup would perform for the same amount of
    # collected experience.
    #
    # "per_mission" (this field's DEFAULT, byte-identical to every
    # pre-existing profile/test that never sets this field): exactly one
    # update per mission once warmup is satisfied -- unchanged behaviour.
    # "per_transition": ``max(1, round(this_mission's stored transitions *
    # updates_per_option_transition))`` updates per mission -- an actual
    # update-to-data ratio scaling with collected experience. The shipped
    # ``hierarchical_phase4.yaml`` profile explicitly opts into this (see
    # that file) -- "per_mission" remains the SCHEMA default so nothing
    # else silently changes cadence.
    training_update_cadence: str = "per_mission"
    updates_per_option_transition: float = 1.0

    def validate(self) -> None:
        if self.max_local_steps_per_option <= 0:
            raise ConfigError("hierarchical_training.max_local_steps_per_option must be > 0")
        if self.max_global_options_per_mission <= 0:
            raise ConfigError("hierarchical_training.max_global_options_per_mission must be > 0")
        if self.training_update_cadence not in ("per_mission", "per_transition"):
            raise ConfigError(
                "hierarchical_training.training_update_cadence must be 'per_mission' or 'per_transition', "
                f"got {self.training_update_cadence!r}"
            )
        if self.updates_per_option_transition <= 0.0:
            raise ConfigError("hierarchical_training.updates_per_option_transition must be > 0")
        from hunter_kinodynamic_rl.env.scenarios.long_horizon_curriculum import MAX_LEVEL, MIN_LEVEL
        if not (MIN_LEVEL <= self.long_horizon_level <= MAX_LEVEL):
            raise ConfigError(
                f"hierarchical_training.long_horizon_level must be in [{MIN_LEVEL}, {MAX_LEVEL}], "
                f"got {self.long_horizon_level}"
            )
        if self.enabled and not self.local_checkpoint_dir:
            raise ConfigError(
                "hierarchical_training.enabled=true requires local_checkpoint_dir to be set "
                "(Phase 4 never trains the local policy -- a frozen checkpoint must be named)"
            )
        if not self.local_checkpoint_name:
            raise ConfigError("hierarchical_training.local_checkpoint_name must be non-empty")
        if self.enabled and not self.local_training_profile_name:
            raise ConfigError(
                "hierarchical_training.enabled=true requires local_training_profile_name so the frozen "
                "Local checkpoint's training distribution can be verified"
            )
        for field_name in ("local_inference_device", "global_training_device"):
            value = getattr(self, field_name)
            if value not in ("auto", "cpu", "cuda"):
                raise ConfigError(
                    f"hierarchical_training.{field_name} must be auto|cpu|cuda, got {value!r}"
                )


@dataclass
class Profile:
    """The fully-resolved config for one run -- what a training/eval node
    actually consumes."""

    name: str
    robot: RobotConfig = field(default_factory=RobotConfig)
    action_space: ActionSpaceConfig = field(default_factory=ActionSpaceConfig)
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    counterfactual: CounterfactualConfig = field(default_factory=CounterfactualConfig)
    hyperparameters: TQCHyperparameters = field(default_factory=TQCHyperparameters)
    sac_hyperparameters: SACHyperparameters = field(default_factory=SACHyperparameters)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    features: FeatureFlags = field(default_factory=FeatureFlags)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    start_pose: StartPoseConfig = field(default_factory=StartPoseConfig)
    obstacle_pool: ObstaclePoolConfig = field(default_factory=ObstaclePoolConfig)
    sensor_noise: SensorNoiseConfig = field(default_factory=SensorNoiseConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    domain_randomization: DomainRandomizationConfig = field(default_factory=DomainRandomizationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    mission: MissionConfig = field(default_factory=MissionConfig)
    localization: LocalizationConfig = field(default_factory=LocalizationConfig)
    mapping: MappingConfig = field(default_factory=MappingConfig)
    hierarchy: HierarchyConfig = field(default_factory=HierarchyConfig)
    long_horizon_world: LongHorizonWorldConfig = field(default_factory=LongHorizonWorldConfig)
    wall_segment_pool: WallSegmentPoolConfig = field(default_factory=WallSegmentPoolConfig)
    global_rl: GlobalRLConfig = field(default_factory=GlobalRLConfig)
    hierarchical_training: HierarchicalTrainingConfig = field(default_factory=HierarchicalTrainingConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    feasibility: GlobalFeasibilityConfig = field(default_factory=GlobalFeasibilityConfig)

    def validate(self) -> None:
        for section in (
            self.robot, self.action_space, self.trajectory, self.dynamics,
            self.observation, self.risk, self.counterfactual,
            self.hyperparameters, self.sac_hyperparameters, self.algorithm, self.scenario, self.start_pose,
            self.obstacle_pool, self.sensor_noise, self.training,
            self.evaluation, self.reward, self.domain_randomization, self.runtime,
            self.mission, self.localization, self.mapping, self.hierarchy,
            self.long_horizon_world, self.wall_segment_pool, self.global_rl, self.hierarchical_training,
            self.memory, self.feasibility,
        ):
            section.validate()
        # Cross-section consistency checks that no single section can do alone.
        if self.features.risk_critic and not self.risk.enabled:
            raise ConfigError("features.risk_critic=true requires risk.enabled=true")
        if self.features.counterfactual_risk and not self.counterfactual.enabled:
            raise ConfigError("features.counterfactual_risk=true requires counterfactual.enabled=true")
        if self.counterfactual.enabled and not self.features.counterfactual_risk:
            raise ConfigError("counterfactual.enabled=true requires features.counterfactual_risk=true")
        if self.counterfactual.enabled and not self.features.risk_critic:
            raise ConfigError("counterfactual.enabled=true requires features.risk_critic=true")
        v_max = self.action_space.v_max_mps if self.action_space.v_max_mps is not None else self.robot.max_forward_speed_mps
        if not (self.action_space.v_min_mps <= v_max <= self.robot.max_forward_speed_mps + 1e-9):
            raise ConfigError(
                f"action_space v_min_mps={self.action_space.v_min_mps} <= v_max({v_max}) <= "
                f"robot.max_forward_speed_mps({self.robot.max_forward_speed_mps}) must hold"
            )
        if self.counterfactual.enabled and not self.risk.enabled:
            raise ConfigError("counterfactual.enabled=true requires risk.enabled=true (nothing to rank candidates by)")
        # environment_node.py's _on_reset branches `if is_fixed_benchmark: ...
        # elif domain_randomization.enabled: ...` -- the fixed-benchmark
        # branch always wins, so domain_randomization is silently NEVER
        # applied whenever evaluation.benchmark is set. A profile enabling
        # both would look like it randomizes benchmark evaluation episodes
        # when it actually never does.
        if self.evaluation.benchmark and self.domain_randomization.enabled:
            raise ConfigError(
                "evaluation.benchmark is set (a fixed benchmark scenario) but "
                "domain_randomization.enabled=true -- the fixed-benchmark reset branch always "
                "takes precedence, so domain_randomization would be silently ignored for the "
                "entire run"
            )
        # features.curriculum has no wired consumer anywhere in this package
        # (curriculum staging lives entirely in the separate drl_agent
        # package) -- reject rather than let it look like a working ablation
        # axis. Every shipped profile leaves this at its default False.
        if self.features.curriculum:
            raise ConfigError(
                "features.curriculum=true is not implemented in hunter_kinodynamic_rl "
                "(no code path reads this flag) -- leave it False"
            )
        # rl/algorithms/sac/agent.py is a standalone "vanilla SAC" comparison
        # point (section 34/35) with no risk-critic/counterfactual extension
        # (unlike rl/algorithms/kinodynamic_tqc/agent.py, which subclasses
        # the vanilla TQC agent specifically to add those hooks) -- nothing
        # reads risk_critic/counterfactual_risk on the SAC training path, so
        # enabling them on an algorithm=sac profile would silently do nothing.
        if self.algorithm.name == "sac" and (self.features.risk_critic or self.features.counterfactual_risk):
            raise ConfigError(
                "algorithm.name=sac has no risk-critic/counterfactual extension -- "
                "features.risk_critic and features.counterfactual_risk must both be False"
            )
        # env/spawning/obstacle_pool.py only ever pools PROCEDURAL scenarios
        # -- a fixed benchmark (self.evaluation.benchmark set) always parks
        # every pool slot via activate_static([])/activate_dynamic([])
        # (see environment_node.py::_spawn_scenario_obstacles's
        # `elif use_pool:` branches) and NEVER draws from
        # self.scenario.max_obstacles/dynamic_obstacle_count at all -- that
        # scenario section instead describes the fixed benchmark's own
        # YAML-authored geometry. Checking pool capacity against it for a
        # benchmark profile is meaningless and previously broke evaluating
        # ANY pool-enabled checkpoint (whose own training-time
        # obstacle_pool sizing has nothing to do with the requested
        # evaluation profile's scenario.max_obstacles) through
        # build_effective_profile (nodes/evaluation_node.py), which layers
        # the checkpoint's obstacle_pool together with the eval profile's
        # scenario section into one Profile before calling validate().
        if self.obstacle_pool.enabled and not self.evaluation.benchmark:
            validate_procedural_pool_capacity(self.obstacle_pool, self.scenario)
        # Phase 3: a wall-segment pool with nothing to pool (no long-horizon
        # world enabled) is a config bug, not a meaningful no-op -- mirrors
        # obstacle_pool's own "opt-in extension of a specific generator"
        # relationship, made an explicit error rather than a silent no-op.
        if self.wall_segment_pool.enabled and not self.long_horizon_world.enabled:
            raise ConfigError(
                "wall_segment_pool.enabled=true requires long_horizon_world.enabled=true "
                "(nothing to pool otherwise)"
            )
        if self.wall_segment_pool.enabled:
            validate_wall_pool_capacity(self.wall_segment_pool, self.long_horizon_world)
        # Phase 4: the hierarchical training loop has nothing to train
        # against without BOTH a Global RL action/network config and a
        # long-horizon world to explore -- mirrors wall_segment_pool's own
        # "opt-in extension requires its base subsystem" relationship.
        if self.hierarchical_training.enabled and not self.global_rl.enabled:
            raise ConfigError(
                "hierarchical_training.enabled=true requires global_rl.enabled=true "
                "(nothing to select subgoals with otherwise)"
            )
        if self.hierarchical_training.enabled and not self.long_horizon_world.enabled:
            raise ConfigError(
                "hierarchical_training.enabled=true requires long_horizon_world.enabled=true "
                "(nothing to explore otherwise)"
            )
        # Phase 5 ablations (B-G): each observation/reward-affecting flag on
        # global_rl requires its own subsystem's section actually enabled --
        # mirrors hierarchical_training's own "opt-in extension requires its
        # base subsystem" relationship above.
        if self.global_rl.topology_feedback_enabled and not self.memory.enabled:
            raise ConfigError(
                "global_rl.topology_feedback_enabled=true requires memory.enabled=true "
                "(nothing to build the node tensor/candidate features from otherwise)"
            )
        if self.global_rl.feasibility_feedback_enabled and not self.feasibility.enabled:
            raise ConfigError(
                "global_rl.feasibility_feedback_enabled=true requires feasibility.enabled=true"
            )
        if self.global_rl.global_risk_feedback_enabled and not self.feasibility.enabled:
            raise ConfigError(
                "global_rl.global_risk_feedback_enabled=true requires feasibility.enabled=true "
                "(the global-risk candidate feature reuses feasibility's own rollout/predicted risk, "
                "never a second independent risk computation)"
            )
        if self.memory.enabled and not self.global_rl.enabled:
            raise ConfigError("memory.enabled=true requires global_rl.enabled=true (nothing to feed into otherwise)")
        if self.feasibility.enabled and not self.global_rl.enabled:
            raise ConfigError(
                "feasibility.enabled=true requires global_rl.enabled=true (nothing to feed into otherwise)"
            )


def validate_procedural_pool_capacity(obstacle_pool: "ObstaclePoolConfig", scenario: "ScenarioConfig") -> None:
    """Capacity must cover the worst case generate_scenario can actually
    draw for a PROCEDURAL (non-fixed-benchmark) profile's own scenario
    section, or activation would fail mid-run on a perfectly ordinary (if
    unlucky) episode instead of at load time (live-Docker-verification
    finding: a 6-static-obstacle episode whose radii happened to cluster in
    the top size bucket exhausted that bucket's slots and crashed the whole
    /reset). requirement 4: activate_static NEVER escalates a radius to a
    LARGER class when its own exact class runs out of free slots (doing so
    would spawn Gazebo collision geometry bigger than ScenarioSpec.radius, a
    silent mismatch against whatever feasibility/risk computation assumed)
    -- it fails fast instead. So EVERY class, independently, must be able to
    hold the worst case of ALL scenario.max_obstacles obstacles drawing a
    radius that lands in that one class. ObstaclePool's own slot allocation
    is a simple round-robin over static_size_classes_m (see
    ObstaclePool.__post_init__), so the smallest number of slots any ONE
    class can end up with is max_static // len(static_size_classes_m) --
    requiring THAT (the worst-populated class) to already cover
    scenario.max_obstacles guarantees EVERY class does too, so
    activate_static can never raise for capacity reasons regardless of how
    the radius draws happen to distribute across classes."""
    min_slots_per_class = obstacle_pool.max_static // len(obstacle_pool.static_size_classes_m)
    if min_slots_per_class < scenario.max_obstacles:
        raise ConfigError(
            f"obstacle_pool.max_static ({obstacle_pool.max_static}) split across "
            f"{len(obstacle_pool.static_size_classes_m)} static_size_classes_m gives only "
            f"{min_slots_per_class} slots for the smallest-allocated class, but "
            f"scenario.max_obstacles ({scenario.max_obstacles}) static obstacles could ALL "
            "draw a radius needing that class in the worst case -- set obstacle_pool.max_static "
            f">= scenario.max_obstacles * len(static_size_classes_m) "
            f"({scenario.max_obstacles * len(obstacle_pool.static_size_classes_m)}), or use "
            "fewer static_size_classes_m"
        )
    if obstacle_pool.max_dynamic < scenario.dynamic_obstacle_count:
        raise ConfigError(
            f"obstacle_pool.max_dynamic ({obstacle_pool.max_dynamic}) must be >= "
            f"scenario.dynamic_obstacle_count ({scenario.dynamic_obstacle_count})"
        )
    # generate_scenario draws static obstacle radii in [0.15, 0.5] --
    # see env/scenarios/procedural_generator.py's STATIC_OBSTACLE_RADIUS_RANGE_M.
    from hunter_kinodynamic_rl.env.scenarios.procedural_generator import (
        STATIC_OBSTACLE_RADIUS_RANGE_M,
    )
    max_drawable_radius = STATIC_OBSTACLE_RADIUS_RANGE_M[1]
    if obstacle_pool.static_size_classes_m[-1] < max_drawable_radius - 1e-9:
        raise ConfigError(
            f"obstacle_pool.static_size_classes_m's largest class "
            f"({obstacle_pool.static_size_classes_m[-1]}) must be >= the largest radius "
            f"generate_scenario can draw ({max_drawable_radius}) -- otherwise some episodes' "
            "static obstacles would have no pool slot large enough to spawn them safely"
        )


def validate_wall_pool_capacity(
    wall_segment_pool: "WallSegmentPoolConfig", long_horizon_world: "LongHorizonWorldConfig",
) -> None:
    """The pool's largest length class must cover the longest wall segment
    ``long_horizon_generator`` can ever produce -- a full, unbroken lattice
    edge, i.e. the maze's own cell pitch. The generator floors
    ``grid_n = size_m // (corridor_width_max_m + wall_thickness_m)`` (see
    that module's docstring), so the worst case (smallest usable grid,
    ``grid_n=3``) puts an upper bound of ``size_m / 3`` on pitch --
    :func:`~hunter_kinodynamic_rl.env.scenarios.long_horizon_generator.max_possible_pitch_m`
    computes this exactly (shared with the generator itself, so this bound
    can never silently drift out of sync with the generator's own grid-size
    formula).

    code review, round 2: also requires EVERY configured length class to
    have enough slots for the worst-case count of segments that could
    specifically need THAT class -- a single-total check (an earlier
    version of this function) is NOT sufficient:
    ``wall_segment_spawner.activate_walls`` requires a free slot in a
    segment's OWN snapped length class (never escalates to a larger one on
    exhaustion), and a real reproduction found a profile whose grand TOTAL
    capacity was never exceeded still failing at runtime because ONE
    specific class ran out long before the pool as a whole did (a
    ``max_segments=100`` pool split evenly across 10 classes gives only 10
    slots/class, but one class alone needed 24-33 for the generated
    world). Uses
    :func:`~hunter_kinodynamic_rl.env.scenarios.long_horizon_generator.max_possible_wall_segment_counts_by_class`
    -- see that function's own docstring for the exact derivation -- and
    ``WallSegmentPool``'s own round-robin slot allocation
    (``classes[i % len(classes)]``, see ``env/spawning/wall_segment_spawner.py``)
    to compute each class's GUARANTEED minimum slot count
    (``max_segments // len(length_classes_m)``, the same
    worst-populated-class reasoning ``validate_procedural_pool_capacity``
    already uses for ``ObstaclePoolConfig``)."""
    from hunter_kinodynamic_rl.env.scenarios.long_horizon_generator import (
        max_possible_pitch_m, max_possible_wall_segment_counts_by_class,
    )

    max_pitch = max_possible_pitch_m(long_horizon_world)
    if wall_segment_pool.length_classes_m[-1] < max_pitch - 1e-9:
        raise ConfigError(
            f"wall_segment_pool.length_classes_m's largest class "
            f"({wall_segment_pool.length_classes_m[-1]}) must be >= the longest wall segment "
            f"long_horizon_generator can produce ({max_pitch:.3f}, the worst-case maze cell pitch) "
            "-- increase the largest length class"
        )
    min_slots_per_class = wall_segment_pool.max_segments // len(wall_segment_pool.length_classes_m)
    counts_by_class = max_possible_wall_segment_counts_by_class(long_horizon_world, wall_segment_pool.length_classes_m)
    worst_class, worst_needed = max(counts_by_class.items(), key=lambda kv: kv[1])
    if min_slots_per_class < worst_needed:
        raise ConfigError(
            f"wall_segment_pool.max_segments ({wall_segment_pool.max_segments}) split across "
            f"{len(wall_segment_pool.length_classes_m)} length_classes_m gives only "
            f"{min_slots_per_class} guaranteed slots per class, but length class {worst_class} could "
            f"need up to {worst_needed} segments in the worst case -- set wall_segment_pool.max_segments "
            f">= {worst_needed * len(wall_segment_pool.length_classes_m)} "
            f"({worst_needed} * len(length_classes_m)), or use fewer length_classes_m"
        )
