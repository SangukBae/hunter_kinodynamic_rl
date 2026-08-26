# Source Map

Every piece of `drl_agent` code reused in `hunter_kinodynamic_rl`, how it was
reused, and why. Per the project brief: `drl_agent` is treated as a read-only
reference implementation -- nothing in this file corresponds to an edit made
to `drl_agent` itself; `git status` on `ros2_ws/src/drl_agent/` stays clean
throughout this package's development.

`hunter_kinodynamic_rl` does **not** declare a runtime/package.xml dependency
on `drl_agent` -- everything below is either (a) a verbatim copy, hash-pinned
against drift, or (b) an independent implementation informed by reading
`drl_agent`'s code. `hunter_se_gazebo`, `drl_agent_interfaces`,
`pointcloud_to_laserscan`, and `drl_obstacle_assets` ARE runtime dependencies
(declared in `package.xml`) -- those are shared infrastructure, not
`drl_agent`-internal code.

## Verbatim copies (byte-identical, hash-pinned)

| New file | Source file | SHA-256 (source == copy) | Notes |
|---|---|---|---|
| `hunter_kinodynamic_rl/common/seed.py` | `drl_agent/drl_agent/common/seed_utils.py` | `a05f44d2...2b0` | Centralised RNG seeding (`seed_all`, `enable_torch_determinism`, `make_substream_rngs`, `derive_resume_seed`) -- reproducibility contract (section 40) reused unchanged. |
| `hunter_kinodynamic_rl/trajectory/pure_pursuit.py` | `drl_agent/drl_agent/common/pure_pursuit.py` | `996432b9...944` | The controller Hunter SE actually runs through (`waypoint_to_command`, `hybrid_action_to_command`, `ackermann_swept_path`). `trajectory/pure_pursuit_adapter.py` and `trajectory/trajectory_executor.py` build on top of this UNCHANGED file — see section 12 / section 6. |
| `hunter_kinodynamic_rl/rl/networks/tqc.py` | `drl_agent/drl_agent/rl/networks/tqc.py` | `6e6b94db...4ef` | `Actor`, `Critic`, `quantile_huber_loss` — the actual TQC math. Verified byte-identical at both weight-init and forward-pass level by `tests/test_tqc_parity.py` (run inside the Docker container: 4/4 parity checks pass against the live `drl_agent` source). |
| `hunter_kinodynamic_rl/env/simulation/gazebo_service_wait.py` | `drl_agent/drl_agent/env/simulation/gazebo_service_wait.py` | `997a0e71...4d5` | Pure, ROS-free bounded-wait helper (`GazeboServiceError`, `bounded_wait_for_service`). Byte-identical copy — no reason to reimplement a hard-won "never block forever" primitive. |

`hunter_kinodynamic_rl/common/geometry.py` was ALSO seeded as a verbatim
copy of `drl_agent/drl_agent/common/geometry_utils.py` originally, but is
**no longer byte-identical** (code review caught this table claiming
"verbatim" after the fact had drifted) -- it gained one added function,
`to_world_frame()`, that `geometry_utils.py` does not have. It has been
moved to the "Adapted" table below with an accurate description. This is
exactly the kind of drift `tests/test_source_map.py` (see below) now
catches automatically instead of relying on this table being kept
up to date by hand.

Re-verify with `sha256sum <file>` any time, or run
`python3 -m pytest tests/test_source_map.py -v` (self-skips if the
`drl_agent` sibling package isn't checked out) -- that test is the
AUTHORITATIVE check (it hashes both files directly on every run); this
table is a human-readable summary of its assertions and can drift if
edited by hand without also re-running the test. A hash mismatch on a row
still listed here as "verbatim" means one side drifted and both
`tests/test_source_map.py` and (for `tqc.py` specifically)
`tests/test_tqc_parity.py` will fail on the next run.

## Adapted (informed by drl_agent, not a verbatim copy)

| New file | Reference | What differs and why |
|---|---|---|
| `hunter_kinodynamic_rl/common/geometry.py` | `drl_agent/drl_agent/common/geometry_utils.py` | Started as a verbatim copy; every function `geometry_utils.py` has (`wrap_to_pi`, `angle_to`, `heading_error`, `euclidean_distance`, `goal_distance_and_heading`, `to_robot_frame`) is still byte-for-byte unchanged. ONE function was added on top, `to_world_frame()` (the inverse rotation of `to_robot_frame`), needed by `risk/boundary.py` to convert `dynamics/ackermann_rollout.py`'s robot-LOCAL rollout points back to world coordinates for the world-boundary check -- `geometry_utils.py` has no equivalent (nothing in `drl_agent` needed a local->world inverse transform). No longer hash-pinned as a whole file; see `tests/test_source_map.py`'s `test_geometry_shared_functions_stay_byte_identical_source_lines` for the finer-grained "shared functions unchanged, `to_world_frame` is new" check this split implies. |
| `dynamics/bicycle_model.py`, `dynamics/actuator_model.py`, `dynamics/ackermann_rollout.py` | `drl_agent/common/pure_pursuit.py`'s `ackermann_rollout()` / `ackermann_swept_path()` | Same closed-form bicycle-model integration and rate-limited actuator idea (move-toward-target-by-max-delta, accel/brake-by-sign selection, midpoint/trapezoidal integration), but restructured around an explicit `VehicleState`/`ActuatorState` object and a general per-substep target profile instead of one fixed start/end target pair — needed so `risk/counterfactual_sampler.py` can roll out N candidate trajectories through the same code path `trajectory/pure_pursuit_adapter.py` uses for execution. |
| `env/simulation/gazebo_runtime.py` | `drl_agent/env/simulation/gazebo_runtime.py` + `gazebo_entity_manager.py::_await_future` | Same ALGORITHM reused deliberately unchanged (bounded pause/reset/set_pose, `time.sleep()`-polled future waits, never `spin_once` inside a service callback — see that module's docstring for the exact deadlock this avoids), restructured for this package's own attribute names/mixin shape. Not a byte-identical copy because it's tightly coupled to each package's own Environment class layout. |
| `env/spawning/obstacle_spawner.py`, `env/spawning/obstacle_catalog.py` | `drl_agent/env/spawning/obstacle_catalog_spawner.py` | Same idea (bounded SpawnEntity/DeleteEntity calls, `drl_obstacle_assets` catalog lookup by closest radius), independent implementation matching this package's own scenario/spec types. |
| `rl/algorithms/tqc/agent.py` | `drl_agent/rl/algorithms/tqc/agent.py` + `update.py` | **NOT** a copy. Clean-room reimplementation of the vanilla TQC update rule (target-quantile truncation formula, entropy auto-tuning, actor loss form) with none of `drl_agent`'s aux_prediction / action_risk_head / temporal-fusion machinery — that machinery is `drl_agent`'s OWN research feature set (a risk-map-sector aux head), not what this package's risk framework builds on (dynamics-grounded clearance/TTC/counterfactual instead). The underlying `Critic`/`Actor` networks it calls into are the verbatim copy above, so the vanilla numerics (target formula, quantile huber loss) are checked directly by `tests/test_tqc_parity.py`; the training-loop control flow is checked by `tests/test_tqc_networks.py`. |
| `rl/algorithms/kinodynamic_tqc/agent.py` | `drl_agent`'s Action-Risk Head (`rl/networks/action_risk_head.py`) gradient rule | New network (`rl/networks/risk_critic.py`), but reuses the same "freeze critic params, don't detach the output" gradient pattern documented in `action_risk_head.py`'s docstring, so `d(risk)/d(action)` reaches the actor without an extra optimizer step on the risk critic's own weights. |
| `rl/replay/buffer.py`, `rl/replay/schema.py` | `drl_agent/rl/replay/buffer.py` (`LAP`) | Deliberately SIMPLER: uniform sampling, no LAP/PER, no risk-balanced stratified sampling. `drl_agent`'s buffer solves a different problem (priority replay for an already-mature training loop); this package's contribution is the training OBJECTIVE (risk-aware actor/critic), not the sampling scheme, so a plain, fully-tested buffer with `risk_target` as a native v1 field is the right complexity level. A risk-balanced sampler can be layered on later behind the same `sample()` signature. |
| `rl/checkpointing/manager.py` | `drl_agent/rl/checkpointing/{manager.py,tqc_io.py}` | Simplified to a generic `{name: component}` dict contract instead of `drl_agent`'s TQC-specific field list, so it works unchanged for both the vanilla and risk-aware agent (which adds one extra component). Checkpoint COMPLETENESS goal (actor/critic/target/optimizers/entropy coef/step/seed/config snapshot) is the same as `drl_agent`'s (section 39). |
| `config/schema.py`, `config/loader.py`, `config/validation.py` | `drl_agent/config/{loader.py,validation.py}` + `drl_experiments/profiles/*/profile.yaml` layering | Same LAYERED-YAML-with-validation idea (robot/training defaults -> profile overrides -> fail-fast dataclass validation), reimplemented with Python dataclasses instead of `drl_agent`'s dict-based validator, because this package's config surface (kinodynamic action space, risk, counterfactual) has no `drl_agent` equivalent to copy from. |

## Reused as a dependency, never copied (package.xml `<depend>`/`<exec_depend>`)

`hunter_se_gazebo` (robot URDF/mesh/Gazebo launch), `drl_agent_interfaces`
(`Reset`/`Step`/`GetDimensions`/`Seed` service contracts, used unmodified),
`pointcloud_to_laserscan`, `drl_obstacle_assets`, plus standard ROS2/Gazebo
packages (`ros_gz_interfaces`, `tf2_ros`, `geometry_msgs`, `nav_msgs`,
`sensor_msgs`, `rcl_interfaces`).

## Independent (no drl_agent equivalent)

`trajectory/action_space.py`, `trajectory/trajectory_primitive.py`,
`trajectory/trajectory_sampler.py`, `dynamics/stopping_model.py`,
`dynamics/system_identification.py`, the entire `risk/` package
(`future_clearance.py`, `ttc.py`, `stopping_margin.py`, `trajectory_risk.py`,
`counterfactual_sampler.py`, `unrecoverable_state.py`, `labels.py`),
`robot/{interface,limits,hunter_se}.py`, `env/scenarios/` (procedural
generator + fixed benchmarks), `env/randomization/` (domain randomization).
These implement the three core research contributions (section 67/68) and
have no `drl_agent` precedent to draw from.
