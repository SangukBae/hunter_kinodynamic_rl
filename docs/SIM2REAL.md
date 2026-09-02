# Sim-to-Real

## Status: not yet attempted on real hardware

No real AgileX Hunter SE was available in this development environment.
Everything below is the STRUCTURE this package provides for that transition,
not a claim that it has been validated on real hardware.

The authoritative package-wide readiness assessment is `CURRENT_STATUS.md`.
The audited hierarchical Local-observation and formal-evaluation code defects
are closed, but a newly trained/promoted Local checkpoint, corrected-release
live smoke, formal sim results, residual/risk calibration and hardware evidence
are still required before autonomous deployment.

## What's already in place

1. **Identical policy interface in sim and real** -- the observation vector
   (`env/observation/observation_builder.py`) and action decoding
   (`trajectory/action_space.py` + `trajectory/trajectory_executor.py`) are
   pure functions with no simulator dependency; `nodes/real_policy_node.py`
   calls the same decode/guard path used by a Gazebo episode.
2. **Mandatory safety guard** -- `env/safety/action_guard.py`'s `guard()`
   sanitizes NaN/Inf, clamps to physical bounds, enforces sensor/command
   freshness timeouts, and forces a stop when an obstacle is inside the
   minimum stop distance. `real_hunter_safe.yaml` additionally caps
   `robot.max_forward_speed_mps` to 0.8 m/s and shortens the max trajectory
   horizon for the first trials.
3. **System identification tooling** -- `dynamics/system_identification.py`'s
   `analyze_{velocity_step_response,steering_step_response,circle_test,
   stop_test}` functions are unit-tested against synthetic data
   (`tests/test_system_identification.py`) and ready to run against real
   logged trials. `nodes/system_id_node.py` actively publishes scripted
   `/cmd_vel` excitations and records `/odometry` plus steering joint state;
   it is not a passive/manual logger.
4. **Domain randomization** -- `env/randomization/domain_randomizer.py`
   randomizes applied command-path, observation and nominal-model fields.
   Mass and wheel-radius draws remain model-only in the current Gazebo stack,
   so they must not be reported as physically randomized simulator dynamics;
   see `ARCHITECTURE.md` for the exact field classification.

### Running the implemented system-ID sequence

Run only in a controlled, cleared test area with propulsion containment,
hardware E-stop and an operator ready to intervene. The node commands motion.

```bash
ros2 run hunter_kinodynamic_rl system_id_node.py --ros-args \
  -p trial:=all -p output_dir:=system_id_results \
  -p target_v_mps:=1.0 -p target_steering_rad:=0.261799 \
  -p step_duration_sec:=5.0
```

`trial` may be `velocity_step`, `steering_step`, `circle`, `stop`, or `all`.
Each trial writes `<output_dir>/<trial>.csv` and
`<output_dir>/<trial>_results.json`; `all` additionally writes
`<output_dir>/hunter_se_identified.yaml`. Invalid analyses stay explicit in
the JSON and must be repeated rather than silently copied into a robot config.

## Target sim-to-real model stack

The research roadmap does not replace nominal Hunter physics with a black-box
world model. It adds a learned response residual:

$$
x_{t+1}=f_{Hunter}(x_t,u_t)+g_{\phi_k}(x_t,u_t,h_t).
$$

The initial residual outputs are `[delta_v, delta_yaw_rate, delta_steering]`.
Inputs include measured vehicle response, current/previous commands and command
age/history. A small ensemble supplies multiple rollout hypotheses. Before the
model may affect action selection, validate:

- one-step velocity/yaw-rate/steering error;
- multi-step pose, heading and endpoint error versus horizon;
- induced clearance, TTC and stopping-margin error;
- ensemble spread versus actual rollout error under ID and OOD trials;
- inference latency at the deployed control rate.

The nominal-only rollout remains a supported fallback. A residual checkpoint,
normalization, source-data hashes and calibrator identity must be part of the
policy generation/fingerprint; silently loading a missing or mismatched model
is forbidden for formal or hardware runs.

### System-ID data collection matrix

Collect complete, timestamped trials rather than isolated shuffled samples:

- forward/reverse velocity steps and controlled stops;
- left/right steering steps and steady circles at several speeds;
- combined speed/steering transients;
- command latency/dead-zone trials;
- repeated payload, surface/friction and battery conditions.

Record commanded, guarded and actually published commands separately from
odometry/IMU/joint-state response. Split train/validation/test by entire trial
or rosbag so adjacent time samples cannot leak. The locked real test runs are
used only once for model and navigation evaluation, never residual fitting.

### Risk and uncertainty use on hardware

At deployment, the learned risk model consumes only available sensors and
state. Simulator ground-truth risk factors remain training/evaluation labels.
Use the conservative statistic `risk_mean + beta * risk_uncertainty` only after
held-out calibration. If the counterfactual target or rollout has uncertainty
above its registered threshold, do not train/imitate or blindly execute it;
slow down/stop, retain the mandatory guard and record an explicit abstention
reason.

## Section P2-15 (this session): real-robot inference node + rosbag2

- **`nodes/real_policy_node.py`** -- the dedicated real-robot inference
  node: subscribes `/scan`/`/odometry`/`/hunter_se/joint_states` (the SAME
  topic names Gazebo bridges to, so this node's ROS interface is identical
  whether the publisher is Gazebo or the real robot's own driver stack),
  loads a trained checkpoint, and publishes `/cmd_vel` at
  `runtime.time_delta_sec`. Refuses to run against any profile that isn't
  `runtime.deployment: real_hardware`. Composes ONLY already-tested
  ROS-free pure functions (`observation_builder`, `trajectory_executor`,
  `action_guard.guard` -- MANDATORY here, never optional). Live-verified
  against Gazebo (the node has no way to distinguish a Gazebo-bridged topic
  from a real one, so this exercises its full plumbing; see the delivery
  report for what was and wasn't observed).
  One deliberate asymmetry: it never computes/publishes risk_telemetry
  (that needs privileged ground-truth obstacle positions that don't exist
  on real hardware) -- see the node's own module docstring.
- **`launch/record_real_trial.launch.py`** -- `ros2 bag record` wrapping
  the section-42 topic list (`/ouster/points`, `/scan`, `/odometry`,
  `/hunter_se/joint_states`, `/cmd_vel`, `/cmd_vel_filtered`,
  `/tf`/`/tf_static`/`/clock`), plus a new
  `/hunter_kinodynamic_rl/real_policy_diagnostics` topic (raw policy
  action + goal + nearest-obstacle-distance + emergency-stop flag) as the
  real-path replacement for risk_telemetry's "policy action / collision-
  event" role in that topic list.

### Section P2-11 (this session): safety/operational interface

`real_policy_node.py` gained four operational parameters (see the node's
own module docstring for the full detail), all covered by
`tests/test_real_policy_node.py`:

- `-p dry_run:=true` -- runs the FULL pipeline (observation build,
  inference, `action_guard.guard`, diagnostics) every tick but never
  reaches `cmd_vel_topic`. `_publish()` is the single call site dry_run
  gates; nothing bypasses it.
- `-p replay_mode:=true` -- requires `dry_run:=true` (raises at startup
  otherwise). Feed pre-recorded data via `ros2 bag play <bag> --clock`
  alongside it -- the node's plain topic-subscription interface already
  works against replayed messages with no code of its own to read bags.
- `-p estop_topic:=...` (default
  `/hunter_kinodynamic_rl/real_policy/estop`, `std_msgs/Bool`) -- a
  software E-stop kill switch, independent of and in addition to
  `action_guard.guard`'s own sensor/command-freshness and
  collision-proximity checks. Once latched True it short-circuits
  `_on_control_tick` BEFORE the sensor-freshness check, publishing
  `STOP_COMMAND` every tick until cleared.
- `-p sensor_qos_reliability:=best_effort|reliable` -- scan/odom
  subscription QoS (default `best_effort`, matching this package's
  Gazebo bridge convention); set to `reliable` for a real driver stack
  that publishes RELIABLE instead (a mismatched QoS makes a subscription
  silently receive nothing, no error -- the exact live bug
  `evaluation/nav2_mppi_runner.py` already found and documents for its
  own `/scan` subscription, section P1-9).

## What real-robot deployment still needs

- Measured `config/robot/hunter_se_identified.yaml` (the nominal
  `hunter_se.yaml` values are from the URDF/prefilter config, not a real
  trial) -- `dynamics/system_identification.py`'s
  `build_identified_robot_config`/`write_identified_robot_yaml` and
  `nodes/system_id_node.py`'s real Ackermann-steering excitation sequences
  exist and are tested/live-verified against Gazebo, but never run
  against real hardware.
- An actual real-Hunter-SE trial run through `system_id_node.py` +
  `analyze_*` to validate the nominal dynamics parameters this package
  trains against, and through `real_policy_node.py` itself.
- Residual-dynamics collection/training/calibration tooling and immutable model
  artifacts; only the nominal model and system-ID analysis tools exist today.
- A calibrated multi-task risk ensemble and uncertainty-aware inference path;
  today's `RiskCritic` is scalar and has no ensemble uncertainty.

## Hierarchical real-robot claim boundary

The physical robot does not have simulator ground-truth obstacles, privileged
risk labels or a ground-truth map. Deployment may use only live LiDAR,
localization estimates, online partial/visited/topological memory, the frozen
learned risk model and the mandatory command guard. A paper must state how the
relative final goal is supplied, which frame owns it, how drift changes the
goal vector and whether loop closure is active.

Calling the system GPS-denied requires more than omitting a GPS topic. Before
that claim, evaluate the same missions under ideal, noisy and drifting
localization and report final-goal error/success versus drift magnitude. If the
system only stops when confidence falls, report that as safe degradation rather
than autonomous recovery.

The target method propagates pose covariance, not only scalar confidence. For
$x\sim\mathcal N(\hat x,\Sigma_x)$, evaluate each candidate across pose samples
and residual-dynamics members, then compare mean/quantile/CVaR risk. The current
`WheelImuLocalizationBackend` confidence growth without injected pose error is
not sufficient evidence for this claim; a backend or simulator that produces
measurable pose error is required.

Minimum real-Hunter evaluation:

- at least three environments covering open space, cluttered corridor and a
  dead-end/loop layout;
- repeated runs of identical start-relative-goal pairs;
- wheel odometry + IMU and LiDAR-odometry conditions where available;
- steering delay, low-friction and LiDAR-dropout stress conditions;
- payload/mass, steering offset/dead-zone, acceleration response, command
  latency, yaw bias and accumulated localization-drift sweeps, with test-only
  OOD ranges separated from training randomization;
- raw policy risk, guard intervention, command latency/control rate and
  emergency-stop recovery;
- risk mean/uncertainty, uncertainty abstention and actual risk-model error;
- collision, timeout, stuck, localization loss and operator intervention as
  separate outcomes;
- rosbag recording of mission/subgoal/map/risk/status and command stages.

## Rosbag replay acceptance

A replay is valid only when required sensor topics are present, at least one
decision is produced, replay finishes without error and actuator publishing is
independently measured as zero. A mock publisher's `.published` list is unit-
test scaffolding, not real-node evidence. The current schema-v2 dry-run helper
implements these gates with an instrumented publisher and streamed bag input,
but it has not been executed against an actual mission bag in this session.
The helper's unit tests are implementation evidence, not real-robot validation.

Hardware trials begin with wheels raised or propulsion disabled, then a bounded
low-speed corridor, and only then autonomous junction/dead-end missions. A
software E-stop and downstream command-timeout watchdog remain mandatory; RL
and the Global planner are not safety certificates.
