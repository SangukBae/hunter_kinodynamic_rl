#!/usr/bin/env python3
"""Requirements F/G: Phase 5 A-G ablation suite -- combining runner +
acceptance report over ONE shared scenario manifest.

:func:`run_ablation_suite` runs every requested ablation label against the
SAME :class:`~hunter_kinodynamic_rl.evaluation.long_horizon_benchmark.BenchmarkScenarioSpec`
manifest (never regenerating/reshuffling it per label) and NEVER silently
substitutes a missing Global checkpoint with a heuristic or another
ablation's checkpoint -- ``agents`` must supply a real agent object for
every label in ``B``-``G`` or the label is skipped with an explicit
``skipped_reason``, never run against something else.

:func:`evaluate_acceptance_report` is pure aggregation (works over
synthetic/already-collected episode-dict lists, no live execution) and
answers the three comparisons plan section 9.11/15.9 defines:

  - D vs C: repeated dead-end entries / revisit ratio should DECREASE
  - E vs D: local planner failure rate should DECREASE
  - F/G vs E: collision rate / risk should not INCREASE while success is
    maintained (not necessarily improved)

Every verdict is qualified by ``valid_sample_count`` on both sides of the
comparison -- ``insufficient_data`` (never "improved"/"regressed") when
either side falls below ``min_valid_samples``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from hunter_kinodynamic_rl.evaluation.global_metrics import aggregate
from hunter_kinodynamic_rl.evaluation.long_horizon_benchmark import (
    ABLATION_LABELS, BenchmarkScenarioSpec, run_ablation_benchmark,
)

ABLATION_SUITE_SCHEMA_VERSION = 1

#: (label_with_improvement_expected, baseline_label, metric_key, direction)
#: direction: "lower_is_better" | "higher_is_better" | "non_regression"
#: (F/G vs E is a non-regression check: must not get WORSE, improvement not required)
ACCEPTANCE_COMPARISONS: Sequence[Dict[str, str]] = (
    {"candidate": "D", "baseline": "C", "metric": "repeated_dead_end_entries_mean", "direction": "lower_is_better",
     "description": "D adds topological memory over C -- repeated dead-end entries should decrease"},
    {"candidate": "D", "baseline": "C", "metric": "revisit_ratio_mean", "direction": "lower_is_better",
     "description": "D adds topological memory over C -- revisit ratio should decrease"},
    {"candidate": "E", "baseline": "D", "metric": "local_planner_failure_count_mean", "direction": "lower_is_better",
     "description": "E adds local feasibility feedback over D -- local planner failure rate should decrease"},
    {"candidate": "F", "baseline": "E", "metric": "collision_count_mean", "direction": "non_regression",
     "description": "F adds global risk feedback over E -- collision rate must not regress"},
    {"candidate": "G", "baseline": "E", "metric": "collision_count_mean", "direction": "non_regression",
     "description": "G (full system) vs E -- collision rate must not regress"},
    {"candidate": "F", "baseline": "E", "metric": "final_goal_success_rate", "direction": "non_regression_higher",
     "description": "F vs E -- final-goal success rate must be maintained (not regress)"},
    {"candidate": "G", "baseline": "E", "metric": "final_goal_success_rate", "direction": "non_regression_higher",
     "description": "G vs E -- final-goal success rate must be maintained (not regress)"},
)

#: Metrics whose *_valid_count field name differs from f"{metric}_valid_count"
#: (some metrics -- counts/rates always defined per episode -- have no
#: separate valid_count field at all; num_episodes is the right sample size
#: for those).
_METRIC_TOTAL_COUNT_KEY: Dict[str, str] = {}


def _sample_count_for_metric(summary: Dict[str, Any], metric: str) -> int:
    valid_key = metric.replace("_mean", "_valid_count")
    if valid_key in summary:
        return int(summary[valid_key])
    return int(summary.get("num_episodes", 0))


@dataclass
class ComparisonVerdict:
    candidate: str
    baseline: str
    metric: str
    direction: str
    description: str
    verdict: str  # "improved" | "regressed" | "maintained" | "insufficient_data"
    candidate_value: Optional[float]
    baseline_value: Optional[float]
    candidate_valid_count: int
    baseline_valid_count: int


@dataclass
class AcceptanceReport:
    schema_version: int
    benchmark_kind: str
    labels_present: List[str]
    comparisons: List[ComparisonVerdict]
    created_at_unix: float = field(default_factory=time.time)


def _verdict_for(direction: str, candidate_value: float, baseline_value: float) -> str:
    eps = 1e-9
    if direction == "lower_is_better":
        if candidate_value < baseline_value - eps:
            return "improved"
        if candidate_value > baseline_value + eps:
            return "regressed"
        return "maintained"
    if direction == "higher_is_better":
        if candidate_value > baseline_value + eps:
            return "improved"
        if candidate_value < baseline_value - eps:
            return "regressed"
        return "maintained"
    if direction == "non_regression":
        # lower is still better/neutral; only a strict INCREASE counts as regression.
        return "regressed" if candidate_value > baseline_value + eps else "maintained"
    if direction == "non_regression_higher":
        # higher is still better/neutral; only a strict DECREASE counts as regression.
        return "regressed" if candidate_value < baseline_value - eps else "maintained"
    raise ValueError(f"unknown comparison direction {direction!r}")


def evaluate_acceptance_report(
    per_ablation_episodes: Dict[str, List[dict]], *, benchmark_kind: str = "smoke",
    min_valid_samples: int = 20, comparisons: Sequence[Dict[str, str]] = ACCEPTANCE_COMPARISONS,
) -> AcceptanceReport:
    """``per_ablation_episodes``: ``{label: [episode_dict, ...]}`` for
    whichever labels were actually run (missing labels are simply skipped
    in the comparisons that need them, never fabricated). ``benchmark_kind``
    MUST be ``"smoke"`` or ``"formal"`` -- callers must not claim "formal"
    for a suite that was not actually run at formal scale/checkpoints."""
    if benchmark_kind not in ("smoke", "formal"):
        raise ValueError(f"benchmark_kind must be 'smoke' or 'formal', got {benchmark_kind!r}")

    summaries: Dict[str, Dict[str, Any]] = {}
    for label, episodes in per_ablation_episodes.items():
        if episodes:
            summaries[label] = aggregate(episodes)

    verdicts: List[ComparisonVerdict] = []
    for comp in comparisons:
        candidate, baseline, metric, direction, description = (
            comp["candidate"], comp["baseline"], comp["metric"], comp["direction"], comp["description"]
        )
        cand_summary = summaries.get(candidate)
        base_summary = summaries.get(baseline)
        if cand_summary is None or base_summary is None:
            verdicts.append(ComparisonVerdict(
                candidate=candidate, baseline=baseline, metric=metric, direction=direction,
                description=description, verdict="insufficient_data", candidate_value=None, baseline_value=None,
                candidate_valid_count=0, baseline_valid_count=0,
            ))
            continue
        cand_value = cand_summary.get(metric)
        base_value = base_summary.get(metric)
        cand_n = _sample_count_for_metric(cand_summary, metric)
        base_n = _sample_count_for_metric(base_summary, metric)
        if cand_value is None or base_value is None or cand_n < min_valid_samples or base_n < min_valid_samples:
            verdicts.append(ComparisonVerdict(
                candidate=candidate, baseline=baseline, metric=metric, direction=direction,
                description=description, verdict="insufficient_data", candidate_value=cand_value,
                baseline_value=base_value, candidate_valid_count=cand_n, baseline_valid_count=base_n,
            ))
            continue
        verdicts.append(ComparisonVerdict(
            candidate=candidate, baseline=baseline, metric=metric, direction=direction, description=description,
            verdict=_verdict_for(direction, cand_value, base_value), candidate_value=cand_value,
            baseline_value=base_value, candidate_valid_count=cand_n, baseline_valid_count=base_n,
        ))

    return AcceptanceReport(
        schema_version=ABLATION_SUITE_SCHEMA_VERSION, benchmark_kind=benchmark_kind,
        labels_present=sorted(summaries.keys()), comparisons=verdicts,
    )


def run_ablation_suite(
    scenarios: Sequence[BenchmarkScenarioSpec], *, labels: Sequence[str] = ABLATION_LABELS,
    agents: Optional[Dict[str, Any]] = None, local_evaluators: Optional[Dict[str, Any]] = None,
    seed: int = 0, benchmark_kind: str = "smoke", min_valid_samples: int = 20, **mission_kwargs,
) -> Dict[str, Any]:
    """Runs every label in ``labels`` against the SAME ``scenarios``
    manifest. ``agents``/``local_evaluators`` are ``{label: object}`` dicts
    -- a Global-RL-enabled label (B-G) missing from ``agents`` is SKIPPED
    (recorded in the returned ``skipped`` dict with an explicit reason),
    NEVER silently run against a heuristic or another label's agent. An
    E/F/G-tier label missing from ``local_evaluators`` is likewise skipped
    (never run with a silently zero-filled feasibility evaluator) --
    matches :func:`~hunter_kinodynamic_rl.evaluation.long_horizon_benchmark.run_ablation_mission`'s
    own fail-fast contract, just converted into a skip here so one missing
    checkpoint doesn't abort the whole suite."""
    agents = agents or {}
    local_evaluators = local_evaluators or {}
    per_ablation_episodes: Dict[str, List[dict]] = {}
    skipped: Dict[str, str] = {}

    for label in labels:
        needs_agent = label != "A"
        needs_evaluator = label in ("E", "F", "G")
        if needs_agent and label not in agents:
            reason = f"no agent supplied for ablation {label!r} (Global-RL-enabled labels require one)"
            if benchmark_kind == "formal":
                # Defect-fix item 10: "요청한 B-G checkpoint가 하나라도
                # 없으면 formal 실행 전체 실패 -- label skip을 허용하는
                # 것은 smoke 모드뿐" -- a formal suite must never silently
                # narrow itself to whichever labels happened to have a
                # usable checkpoint.
                raise RuntimeError(f"run_ablation_suite: formal benchmark cannot skip label {label!r}: {reason}")
            skipped[label] = reason
            continue
        if needs_evaluator and label not in local_evaluators:
            reason = (
                f"no local_evaluator supplied for ablation {label!r} (feasibility/global-risk-feedback tiers "
                "require a real LocalFeasibilityEvaluator)"
            )
            if benchmark_kind == "formal":
                raise RuntimeError(f"run_ablation_suite: formal benchmark cannot skip label {label!r}: {reason}")
            skipped[label] = reason
            continue
        agent = agents.get(label)
        local_evaluator = local_evaluators.get(label)
        episodes = run_ablation_benchmark(
            label, scenarios, agent=agent, seed=seed, local_evaluator=local_evaluator, **mission_kwargs,
        )
        per_ablation_episodes[label] = episodes

    report = evaluate_acceptance_report(
        per_ablation_episodes, benchmark_kind=benchmark_kind, min_valid_samples=min_valid_samples,
    )
    # Requirement L: every new ablation/sweep artifact carries package
    # provenance (git root/SHA/dirty/diff-hash) -- never just the outer
    # DRL_Robot_Path_Planning repo's SHA (this package is its OWN nested
    # git repo, see evaluation.provenance's own module docstring).
    from hunter_kinodynamic_rl.evaluation.provenance import collect_package_provenance
    provenance = collect_package_provenance()
    return {
        "schema_version": ABLATION_SUITE_SCHEMA_VERSION, "benchmark_kind": benchmark_kind,
        "num_scenarios": len(scenarios), "labels_run": sorted(per_ablation_episodes.keys()),
        "labels_skipped": skipped, "episodes": per_ablation_episodes,
        "summaries": {label: aggregate(eps) for label, eps in per_ablation_episodes.items()},
        "acceptance_report": report, "provenance": provenance,
    }
