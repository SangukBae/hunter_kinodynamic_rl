# Experiment Card

Copy this template before a run. Empty fields mean the run is not registered evidence.

## Identity

```text
experiment_id:
owner:
created_at_utc:
parent_or_warm_start:
source_commit:
dirty_state_digest:
container_image_digest:
hardware:
output_root:
```

## Question and contract

```text
research_question:
hypothesis_and_expected_mechanism:
method_or_ablation_id:
single_changed_axis:
protocol_version:
scenario/split/seed manifest hashes:
architecture/data/training/robot fingerprints:
train and evaluation budget:
preregistered primary metrics, thresholds and exclusion rules:
```

## Preflight

- [ ] P0 prerequisites and dataset validation passed
- [ ] output root is fresh and identity manifest written
- [ ] checkpoint/load and fallback probes passed
- [ ] source/config/container/GPU information captured
- [ ] locked test has not been used for tuning

## Completion

```text
status: REGISTERED | PREFLIGHT_PASSED | RUNNING | TRAIN_COMPLETE |
        BENCHMARK_COMPLETE | AGGREGATED | ACCEPTED | REJECTED |
        FAILED | INTERRUPTED | INVALID
final checkpoint generation/root hash:
training counters and wall-clock/GPU-hours:
benchmark artifact/root hash:
missing or excluded rows with reason:
protocol deviations:
```

## Results and claim

Record per-seed metrics, event counts, intervals, failure attribution, timing/memory and artifact
links. End with `unsupported`, `exploratory` or `accepted within <exact scope>`—never just “passed.”
