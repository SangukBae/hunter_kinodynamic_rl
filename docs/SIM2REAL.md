# Sim-to-Real

## Status: not yet attempted on real hardware

No real AgileX Hunter SE was available in this development environment.
Everything below is the STRUCTURE this package provides for that transition,
not a claim that it has been validated on real hardware.

## What's already in place

1. **Identical policy interface in sim and real** -- the observation vector
   (`env/observation/observation_builder.py`) and action decoding
   (`trajectory/action_space.py` + `trajectory_executor.py`) are pure
   functions with no simulator dependency; a real-robot inference node would
   call the exact same code a Gazebo episode does.
2. **Mandatory safety guard** -- `env/safety/action_guard.py`'s `guard()`
   sanitizes NaN/Inf, clamps to physical bounds, enforces sensor/command
   freshness timeouts, and forces a stop when an obstacle is inside the
   minimum stop distance. `real_hunter_safe.yaml` additionally caps
   `robot.max_forward_speed_mps` to 0.8 m/s and shortens the max trajectory
   horizon for the first trials.
3. **System identification tooling** -- `dynamics/system_identification.py`'s
   `analyze_{velocity_step_response,steering_step_response,circle_test,
   stop_test}` functions are unit-tested against synthetic data
   (`tests/test_system_identification.py`, 5/5 passing) and ready to run
   against real logged trials; `nodes/system_id_node.py` records
   `/odometry` to CSV during a manual trial.
4. **Domain randomization** -- trains policies against randomized dynamics
   (`env/randomization/domain_randomizer.py`) so the sim-trained policy is
   not brittle to the exact nominal Gazebo parameters.

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
