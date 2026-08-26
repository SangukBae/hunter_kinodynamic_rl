# Architecture

## Data flow

```
Ouster LiDAR (/ouster/points -> pointcloud_to_laserscan -> /scan)
        |
sensing/scan_processor.py        -- 360 raw ranges -> (obs_state front-N, environment_state full-360)
        |
sensing/temporal_stack.py        -- optional frame_stack history (features.temporal_context)
        |
env/observation/observation_builder.py  -- + robot state (goal dist/heading, prev action, v, yaw_rate, steering)
        |
rl/algorithms/{tqc,kinodynamic_tqc}/agent.py   -- Actor -> normalized action in [-1,1]^3
        |
trajectory/action_space.py       -- normalized -> physical [kappa, v_ref, L] (or legacy [r, theta, yield])
        |
trajectory/trajectory_primitive.py      -- constant-curvature arc geometry
        |
dynamics/ackermann_rollout.py    -- Hunter SE actuator+bicycle-model rollout (training: risk scoring;
        |                            inference: not needed, the command is already known)
        |                          [TRAINING ONLY, privileged obstacle GT] --------------------+
        |                                                                                       |
        |                                                              risk/{future_clearance,ttc,        |
        |                                                              stopping_margin,trajectory_risk}.py |
        |                                                              risk/counterfactual_sampler.py      |
        |                                                                        |                         |
        |                                                              rl/networks/risk_critic.py <--------+
        |                                                              (supervised loss + actor penalty)
        |
trajectory/pure_pursuit_adapter.py  -- kappa/v_ref/L -> (speed, steering) via trajectory/pure_pursuit.py
        |
env/safety/action_guard.py       -- NaN/bounds/staleness/collision-proximity guard (mandatory on real robot)
        |
/cmd_vel -> hunter_se_cmd_prefilter (hunter_se_gazebo, unmodified) -> /cmd_vel_filtered -> Gazebo
```

## Training vs. inference

Training-time-only pieces (never touch the real-robot path):

- `risk/labels.DynamicObstacle` -- privileged ground-truth obstacle
  position/velocity, only available from the simulator.
- `risk/counterfactual_sampler.py` -- rolls out N alternative candidate
  trajectories against that privileged obstacle state to compute a
  supervised risk target and the counterfactual "was there a safer
  alternative" margin.
- `env/scenarios/procedural_generator.py` -- procedural training scenario
  generation with train/validation/test seed separation (never used at
  inference).

At inference (sim OR real), the policy only ever sees LiDAR history + robot
state + goal (`env/observation/observation_builder.py`'s output) and its own
risk critic's forward pass (`rl/networks/risk_critic.py`, trained
beforehand) -- never a `DynamicObstacle` list directly. This is enforced by
construction: nothing in `env/simulation/environment_node.py`'s `_on_step`
handler hands obstacle ground truth to `select_action`.

## Module ownership (who calls what)

- `config/` -- typed, validated profile loading. Everything downstream
  receives already-validated dataclasses, never raw YAML dicts.
- `robot/` -- physical bounds + curvature<->steering conversions. The ONLY
  module that would need a sibling implementation to swap Hunter SE for
  another Ackermann UGV (`RobotModel` Protocol in `robot/interface.py`).
- `dynamics/` -- kinematics (`bicycle_model.py`), actuator response
  (`actuator_model.py`), composed rollout (`ackermann_rollout.py`), and
  braking physics (`stopping_model.py`). No obstacle/reward knowledge.
- `trajectory/` -- action encoding/decoding, path-shape primitives, and the
  Pure Pursuit execution adapter. No risk/reward knowledge.
- `risk/` -- everything that needs obstacles: clearance, TTC, stopping
  margin, aggregate risk score, counterfactual candidate ranking,
  approximate Unrecoverable-State check.
- `rl/` -- networks, the vanilla TQC reference agent, its risk-aware
  extension, replay buffer, checkpointing.
- `env/` -- the ROS-facing simulation orchestration: observation assembly,
  reward, procedural/fixed scenarios, domain randomization, obstacle
  spawning, safety guard, and the environment node itself.
- `sensing/` -- raw scan -> fixed-width bins, temporal frame stacking.
- `training/` / `evaluation/` -- the training loop and benchmark runner,
  both talking to a running `environment_node.py` over the
  `drl_agent_interfaces` service contract exactly like `drl_agent`'s own
  trainers do.

## Extension points (section 61)

| Swap | Touch | Don't touch |
|---|---|---|
| New robot (Scout, F1TENTH, ...) | New `robot/<name>.py` + `config/robot/<name>.yaml` | `dynamics/`, `trajectory/`, `risk/`, `rl/` |
| New trajectory primitive (clothoid, spline) | `trajectory/trajectory_primitive.py`'s `make_primitive()` | `action_space.py`'s action CONTRACT, `risk/` |
| New risk model (CVaR, reachability) | `rl/networks/risk_critic.py`'s `forward()` contract stays `(state, action) -> risk`; swap the body | `rl/algorithms/kinodynamic_tqc/agent.py`'s gradient rule |
| New RL algorithm | New `rl/algorithms/<name>/agent.py` implementing `select_action`/`train_step`/`checkpoint_components` | `trajectory/`, `risk/`, `env/` |
| New sensor | New `sensing/<name>_processor.py` producing the same fixed-width bin array | `env/observation/observation_builder.py`'s concatenation contract |

## L semantics (trajectory horizon)

`L` (`action[2]`, physical `horizon_m`) is the trajectory's **planning
horizon**, an arc length in `[horizon_length_min_m, horizon_length_max_m]`.
It has two concrete, causal effects:

1. **Path-shape endpoint**: `trajectory/trajectory_primitive.py`'s
   `ConstantCurvatureArc.point_at(L)`/`sample()` truncate exactly at `L` --
   changing `L` changes where the sampled trajectory ends.
2. **Rollout/risk TIME horizon**: `dynamics/ackermann_rollout.py`'s
   `horizon_from_trajectory(L, v_ref, dynamics_cfg)` computes
   `max(clip(L / v_ref, horizon_min_sec, horizon_max_sec), min_safety_horizon_sec)`
   -- the actual duration `risk/trajectory_risk.py::assess_trajectory_command`
   (and therefore `risk/counterfactual_sampler.py`) rolls out and scores
   this action over. A policy committing to a longer `L` is held
   responsible for a longer future-risk window (the `clip(L/v_ref, ...)`
   term); a short/reactive `L` is scored only down to
   `dynamics.min_safety_horizon_sec` -- **never shorter**. `v_ref` at/below
   a small floor (including an exact stop) maps to `horizon_max_sec` rather
   than dividing by ~0.

   **P0 fix (code review, live-caught before this floor existed)**: without
   the `min_safety_horizon_sec` floor, a policy could pick an arbitrarily
   small `L` purely to shrink its own rollout below the distance to a real
   obstacle -- always reporting `risk_target≈0` for an actually-unsafe
   heading/speed simply by not looking far enough ahead. This is a
   risk-hacking exploit, not a training artifact: the actor's incentive is
   to minimize predicted risk, and `L` was a free knob to do that without
   changing anything about where the robot actually goes. The floor
   guarantees the assessed horizon can never shrink below
   `min_safety_horizon_sec` (default 1.5s, strictly between
   `horizon_min_sec`=0.5s and `horizon_max_sec`=3.0s so `L` still has a
   real effect ABOVE the floor) regardless of `L`.
   `tests/test_l_semantics.py::test_short_L_cannot_hide_a_distant_collision`
   is the regression test: an obstacle placed within the floor's reach but
   outside a naive tiny-`L` rollout's reach must still be detected.

**Known, intentional property, not a bug**: for a *pure constant-curvature*
primitive, Pure Pursuit's recovered STEERING ANGLE from a lookahead point on
that circle is mathematically invariant to which point is chosen -- so two
candidates with the same `kappa`/`v_ref` but different `L` produce the
*same* immediate steering command (verified live: `test_l_semantics.py`'s
`test_different_L_same_kappa_v_produces_different_rollout_endpoint` proves
the ENDPOINT differs; the executed steering does not, by design, for this
primitive). This stops being true the moment a non-circular primitive
(clothoid/spline) is swapped in behind `trajectory_primitive.make_primitive`.

The RL decision period (`runtime.time_delta_sec`, fixed per Gazebo `/step`)
is INDEPENDENT of `L` -- the environment always advances physics by exactly
`time_delta_sec` regardless of `L`, so `L` never changes how often the
policy is re-queried or the reward's discount semantics.

## Reset/step synchronization (fixed-step Gazebo control)

`env/simulation/gazebo_runtime.py`'s `GazeboRuntimeMixin` (adapted from
drl_agent, see SOURCE_MAP.md) gives every `/reset`/`/step` call:

- **Bounded Gazebo service calls** -- `pause_world`/`reset_world`/
  `set_entity_pose_ignition` never block forever; a timeout raises
  `GazeboServiceError`, which propagates out of the service callback
  (the node exits with a clear diagnostic rather than continuing with a
  half-reset world -- confirmed live: see docs/TROUBLESHOOTING.md's
  callback-group-deadlock entry for the bug this surfaced and its fix).
- **Fixed-step physics advance** -- `propagate_state(duration)` is
  unpause -> `time.sleep(duration)` -> pause, called with
  `runtime.reset_settle_time_sec` on reset and `runtime.time_delta_sec` on
  every step.
- **Sensor-freshness confirmation** -- `wait_for_fresh_sensors()` bounded-polls
  (`time.sleep`, NEVER `spin_once`) until BOTH `/scan` and `/odometry` have
  published at least one NEW message since the physics advance began, so a
  step never returns a cached, pre-advance observation.

## Privileged risk label transport (env -> replay -> risk critic)

`drl_agent_interfaces`'s `Step.srv` is NEVER extended with privileged data
(section 5's explicit requirement). Instead, `environment_node.py` publishes
a package-owned side channel after every step:
`/hunter_kinodynamic_rl/risk_telemetry` (`std_msgs/Float32MultiArray`,
schema documented and unit-tested in `env/simulation/risk_telemetry.py`),
keyed by a `step_id` counter that resets to 0 on `/reset` and increments
once per `/step` on BOTH ends. `training/trainer_base.py`'s
`EnvironmentClient` subscribes and bounded-polls (`rclpy.spin_once` -- safe
here because this client node drives its own spinning, unlike a service
callback already running under an executor) its cache until the received
`step_id` matches its own local counter, so a transition is never paired
with a stale or mismatched label. A poll that times out returns an
EXPLICIT `invalid` telemetry (`valid=False`), never a silently-stale one or
a bare NaN. `ReplayBuffer` (schema v2) stores `valid` as a first-class
boolean mask -- `rl/algorithms/kinodynamic_tqc/agent.py` filters on it, not
on `isnan(risk_target)`, and skips the risk critic's supervised update
entirely when a batch has fewer than `risk.min_valid_labels_per_batch`
valid rows. The actor-side risk penalty additionally stays at exactly zero
until the risk critic has completed `risk.actor_penalty_warmup_updates`
supervised updates (verified live: a 200-step smoke run reports
`risk/actor_penalty_active: 0.0` throughout, since it never reaches the
default 500-update warmup in that short a run).

## Counterfactual objective (math)

Two independent, disableable pieces (`counterfactual.enabled`):

1. **Candidate-augmented risk critic supervision** -- for every VALID
   counterfactual candidate `(kappa, v_ref, L)` in a batch, converted back
   to a normalized action via `trajectory.action_space.
   trajectory_command_to_normalized`'s inverse-lerp, the risk critic is
   ALSO trained (MSE) to predict that candidate's own `risk_score` from
   `(state, candidate_action)` -- not just the actor's stored action --
   weighted by `counterfactual.candidate_supervision_weight` and summed
   into the SAME backward pass as the base supervision.
2. **Margin-reweighted actor penalty** -- `safer_alternative_margin`
   (`actor_risk - best_candidate_risk`, ground truth, precomputed by the
   environment's own candidate search) is NOT backpropagated through
   directly (it is a stored scalar, not a function of the CURRENT actor's
   live output). Instead it reweights the existing risk-critic-mediated
   actor penalty: `weight = 1 + counterfactual_weight_scale * relu(margin)`,
   so transitions where a meaningfully safer alternative existed push the
   actor's own predicted risk down harder. `predicted_risk = risk_critic(
   state, actions_pi)` is computed with the risk critic's OWN parameters
   FROZEN (not detached) for that forward pass, so `d(risk)/d(action)`
   still reaches the actor without any gradient updating the risk critic's
   weights from the actor loss.

`tests/test_kinodynamic_tqc.py::test_counterfactual_margin_reweights_actor_penalty`
verifies a large stored margin produces a larger penalty magnitude than a
zero margin, all else equal.

## Exact fixed-benchmark replay

`environment_node.py` declares a `scenario_override_path` ROS parameter
(default `""`). `evaluation/benchmark_runner.py` calls
`EnvironmentClient.set_scenario_override(<yaml path>)` (via
`rcl_interfaces/SetParameters`) immediately before each `/reset` for a
benchmark episode; `_on_reset` checks this parameter FIRST and, if set,
loads the EXACT scenario file (`env/scenarios/benchmark_loader.
load_scenario_file`) instead of procedurally generating one from a seed --
confirmed live: `[reset] episode seed=20001 fixed_benchmark=True
static_obstacles=3` matches `config/benchmarks/id/scenario_001.yaml`
exactly. Set back to `""` after a benchmark run to return the node to
normal (procedural/seeded) operation.

## Domain randomization: where it actually applies

`env/randomization/domain_randomizer.py`'s `RandomizationDraw` fields split
into three honest categories (see that module's `MODEL_ONLY_FIELDS`/
`GAZEBO_APPLIED_FIELDS`/`OBSERVATION_ONLY_FIELDS` constants and
`classify_draw_fields()`, logged per-episode by `environment_node.py`'s
`/reset` handler as `domain_rand_classification` -- code review: no silent
no-op randomization flag, this boundary must be visible in a run's logs,
not just here):

- **MODEL-ONLY** (`mass_scale`, `wheel_radius_scale`): applied via
  `apply_to_robot_config` to the `RobotConfig` used for that episode's
  action decoding / offline dynamics-rollout risk model ONLY -- genuinely
  not runtime-settable in this environment. Ignition Fortress's public
  `ros_gz_interfaces` service surface this package actually uses
  (SpawnEntity/DeleteEntity/SetEntityPose/ControlWorld -- see
  `env/simulation/gazebo_runtime.py`) has no service for per-link mass/
  inertia or wheel geometry; achieving that would need either a custom
  world/model plugin or a full delete+respawn with an edited SDF, both out
  of scope here. No plugin-parameter or respawn path exists yet for these
  two.
- **GAZEBO-APPLIED** (`friction_scale`, `velocity_response_scale`,
  `steering_gain`, `steering_delay_sec`, `command_latency_sec`): each has a
  real consumer on the command-PUBLISH path, so each genuinely changes
  simulated motion, not just the risk model:
  - `friction_scale`/`velocity_response_scale` scale `accel_limit_mps2`/
    `brake_decel_mps2` (via `apply_to_robot_config`) and are now ALSO read
    by `domain_randomizer.rate_limit_speed`, applied to the NOMINAL command
    in `environment_node.py`'s `/step` handler BEFORE the safety guard runs
    (never after -- a guard-forced emergency stop must stay immediate, never
    softened by this "realism" ramp). A non-randomized draw (both == 1.0,
    always true when `domain_randomization.enabled` is false) is a
    byte-identical no-op.
  - `steering_gain` scales `steering_rate_deg_s`; whenever
    `features.trajectory_l_preview_blend` is also enabled (default True for
    the trajectory-mode profiles that use domain randomization --
    `kinodynamic_tqc_domain_rand.yaml`), `trajectory/pure_pursuit_adapter.py`'s
    L-derived steering commit window (see its module docstring, and the "L
    semantics" section above) uses it as a REAL actuator rate limit on the
    published steering command. With that feature OFF, `steering_gain`
    reverts to model-only.
  - `steering_delay_sec`/`command_latency_sec`: unchanged from before --
    `steering_lag_alpha`/the command-delay queue, both always active,
    applied on the real command-publish path.
- **OBSERVATION-ONLY, by design** (`lidar_range_noise_std_m`,
  `lidar_dropout_prob`, `odometry_noise_std`, `sensor_frame_drop_prob`):
  `apply_lidar_noise`/`apply_odometry_noise`/`should_drop_sensor_frame`
  explicitly perturb only the AGENT'S OBSERVATION, never ground truth or the
  published command -- this is the intended boundary (evaluation/collision/
  risk-label ground truth must stay noise-free), not a gap.

## Ackermann-aware scenario feasibility: guarantee level (code review)

`env/scenarios/ackermann_feasibility.is_ackermann_feasible` (wired into
`procedural_generator.generate_scenario` via `scenario.feasibility_check:
ackermann`) is a bounded, discretized Hybrid-A*-style SEARCH used as a
scenario REJECTION FILTER -- it is **not** a formal/complete kinodynamic
motion planner and carries no soundness or completeness proof. An earlier
version of its own module docstring claimed "never a false positive",
which was not accurate; that claim has been corrected there and is
restated honestly here:

- **False negatives** (rejecting an actually-feasible layout) are CHEAP:
  the coarse motion-primitive set (3 fixed steering choices, one fixed
  speed/duration) or the coarse xy/yaw dedup grid can miss a real path --
  costs one extra retry in `generate_scenario`'s existing loop.
- **False positives** (accepting an actually-infeasible layout) are the
  real risk: each primitive's collision check only samples a FIXED number
  of discrete points along its arc (`num_path_samples`, default 5), not a
  continuous sweep -- an obstacle small enough, or precisely enough
  placed, to fall entirely between two consecutive samples can be missed
  ("tunneling"). See `ackermann_feasibility.py`'s module docstring for the
  exact sample-spacing arithmetic this trades off, and
  `tests/test_ackermann_feasibility.py`'s resolution-sensitivity tests for
  a concrete case a coarse resolution misses and a finer one catches.

This does **not** compromise runtime safety: the gate only decides which
PROCEDURAL SCENARIOS are worth attempting before an episode starts -- the
real collision detector, safety guard, and episode-truncation-on-collision
all still run in full against the actual simulated episode regardless of
what this gate decided. Its only job is reducing (not eliminating) how
often the trainer/evaluator is handed a scenario that turns out to be
geometrically unsolvable for an Ackermann vehicle.

## Not yet done (see docs/DELIVERY_REPORT.md for the full list)

- Real Hunter SE hardware trials (no hardware available in this
  environment) -- `nodes/real_policy_node.py` and
  `launch/record_real_trial.launch.py` exist, are live-verified against
  Gazebo, but never against real hardware.
- Nav2-MPPI classical baseline (`evaluation/nav2_mppi_runner.py`) is
  code-complete and only PARTIALLY live-verified -- see that module's
  docstring for the exact status (no live attempt reached a full
  goal-reaching episode; a low-effective-velocity issue is unresolved).
