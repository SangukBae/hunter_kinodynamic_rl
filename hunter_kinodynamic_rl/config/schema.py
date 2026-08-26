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
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    domain_randomization: DomainRandomizationConfig = field(default_factory=DomainRandomizationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def validate(self) -> None:
        for section in (
            self.robot, self.action_space, self.trajectory, self.dynamics,
            self.observation, self.risk, self.counterfactual,
            self.hyperparameters, self.sac_hyperparameters, self.algorithm, self.scenario, self.training,
            self.evaluation, self.reward, self.domain_randomization, self.runtime,
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
