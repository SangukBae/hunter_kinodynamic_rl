# Research Protocol

## Seeds

Configured per-profile via `scenario.{train,validation,test}_seed_range`
(default: train `[0, 9999]`, validation `[10000, 10999]`, test `[20000,
20999]`) -- see `config/training/defaults.yaml`. Ranges are validated
non-overlapping at profile-load time
(`config/schema.py::ScenarioConfig.validate`), and
`env/scenarios/procedural_generator.seed_split()` raises rather than
silently misclassifying a seed outside all three ranges. **Test seeds never
enter the replay buffer** -- the training loop (`training/trainer_base.py`)
only ever calls `env.seed()` with seeds from
`training.seed`-derived training draws; benchmark evaluation
(`evaluation/benchmark_runner.py`) only ever seeds from a loaded benchmark
scenario's own `seed` field (`config/benchmarks/*/*.yaml`, all in the
`[20000, 20999]` test range).

## Scenarios

- **Training**: procedural (`env/scenarios/procedural_generator.py`),
  regenerated fresh every episode from `env.seed()`.
- **Evaluation**: FIXED, hand-authored YAML files under
  `config/benchmarks/{id,ood_geometry,ood_dynamics,dynamic}/` -- every
  baseline runs the identical scenario set (section 32's "모든 baseline이
  정확히 같은 scenario에서 평가되어야 한다").

## Baselines / ablation matrix

| Letter | Profile | Trajectory action | Temporal | Risk prediction | Risk-aware critic/actor | Counterfactual |
|---|---|---|---|---|---|---|
| A | `baseline_tqc` / `legacy_waypoint_tqc` | no (legacy waypoint) | no | no | no | no |
| B | `kinodynamic_tqc` | yes | no | no | no | no |
| C | `kinodynamic_tqc_temporal` | yes | yes | no | no | no |
| D | `kinodynamic_tqc_risk_supervised_only` | yes | yes | yes | no (`actor_lambda=0.0`) | no |
| E | `kinodynamic_tqc_risk` | yes | yes | yes | yes (`actor_lambda=0.1`) | no |
| F | `kinodynamic_tqc_counterfactual` | yes | yes | yes | yes | yes |

Every row is the SAME code path (`training/train_tqc.py` for A/B/C,
`training/train_kinodynamic_tqc.py` for D-F, selected automatically by
`nodes/train_node.py` from `features.risk_critic`) -- ablations are config
flags (`config/schema.py::FeatureFlags`), never a forked implementation
(section 36).

Section 34/35's remaining two comparison points are now implemented:
- **Vanilla SAC** (`rl/algorithms/sac/agent.py`, `training/train_sac.py`,
  `sac_baseline` profile, selected via `algorithm.name=sac`): standard
  twin-Q SAC sharing this package's trajectory action space/observation/
  reward with ablation row B, so only the algorithm differs. Fully
  live-verified (a real training run against Gazebo produced real
  checkpoints and a real periodic-validation event).
- **Nav2-MPPI classical baseline** (`evaluation/nav2_mppi_runner.py`,
  `config/nav2_mppi/`, `launch/nav2_mppi.launch.py`,
  `nodes/nav2_mppi_eval_node.py`): map-free (no AMCL/map_server -- see that
  config file's header comment), Ackermann motion model, adapted from
  scout_nav2's verified MPPI controller block. CODE-COMPLETE and
  PARTIALLY live-verified only -- see `evaluation/nav2_mppi_runner.py`'s
  module docstring for the exact live-verification status (the stack
  configures/activates/accepts goals and its collision/telemetry pipeline
  produces real measurements, but no live attempt achieved a full
  goal-reaching episode; an unresolved low-effective-velocity issue would
  need further live iteration to root-cause).

## Metrics

Computed by `evaluation/metrics.py::aggregate()` from per-episode result
dicts: success rate, collision rate, timeout rate, Unrecoverable-State rate,
SPL, navigation time, path length, average velocity, minimum clearance, TTC
statistics, steering saturation rate, steering smoothness, control
smoothness, emergency-stop count (section 37's full list).

## Checkpoint policy

`rl/checkpointing/manager.py` saves every `nn.Module`/`Optimizer` component
an agent exposes via `checkpoint_components()` (actor, critic, critic
target, optimizers, risk critic + its optimizer when present) plus a JSON
manifest recording which components were present, training step, and seed.
Loading a checkpoint saved WITHOUT the risk critic into a risk-aware agent
degrades gracefully (the risk critic stays freshly-initialised, reported in
the manifest's `skipped` list) rather than erroring -- see
`tests/test_checkpointing.py::test_checkpoint_load_reports_skipped_component_when_absent`.

## Domain randomization

Opt-in (`domain_randomization.enabled`, default false) -- see
`config/domain_randomization/default.yaml` for the default ranges and
`env/randomization/domain_randomizer.py` for the sampler. A `RandomizationDraw`
is deterministic given `(seed, config)`
(`tests/test_env_modules.py::test_randomization_is_deterministic_per_seed`).

## Real-robot trials

Not run in this development session -- no real Hunter SE is available here.
See `docs/SIM2REAL.md`.
