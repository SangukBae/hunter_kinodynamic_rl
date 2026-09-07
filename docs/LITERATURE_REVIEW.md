# Literature and Novelty Review

Snapshot: **2026-09-06; refresh before submission**

This document records the comparison logic, not a claim that every recent paper has been exhausted.
Final manuscript work must rerun database/venue searches and verify bibliographic metadata from primary
sources.

## 1. Relevant research lines

| Line | What it contributes | Why it does not by itself answer TRACTOR's question |
|---|---|---|
| TQC / distributional RL | truncated return quantiles and tail control | no explicit geometric candidate-scene interaction |
| recurrent/POMDP policies | temporal state under partial observation | latent memory need not expose occupancy or collision mechanism |
| Dreamer/TD-MPC latent world models | learned prediction and planning/value learning | may entangle scene, action and model bias; physical footprint interaction is implicit |
| safe model-based RL | constraints, imagined safety objectives | simulator/domain assumptions and safety definitions differ |
| occupancy/flow forecasting | spatial future scene prediction | usually not coupled to Ackermann candidate tubes and RL return |
| trajectory collision prediction | candidate-conditioned collision/risk | often endpoint/scalar risk without calibrated cause-time survival |
| LiDAR dynamic-obstacle navigation | deployable sensing/control evidence | action, robot, map and evaluation contracts vary widely |
| residual vehicle dynamics | correct nominal model mismatch | established support technique, not core novelty |
| MPPI/classical local planning | explicit sampled trajectory evaluation | different optimization/controller contract; valuable external baseline |

These lines are complementary. “Uses attention/world model/risk” is not a sufficient novelty claim.

## 2. Nearest-comparison questions

For each candidate paper or baseline, extract from the primary source:

1. observation modality/history and localization assumptions;
2. action and vehicle dynamics contract;
3. whether scene future is action-independent or candidate-branched;
4. whether robot footprint occupancy is explicit and differentiable;
5. value/risk output: scalar, quantile, event time, cause, severity and calibration;
6. whether imagined transitions enter Bellman/value learning;
7. simulation/real robot, ODD, seeds, event counts and compute;
8. source/checkpoint availability and reproducibility.

Methods are direct architecture baselines only when observation/action/robot/data budgets are alignable.
Otherwise they are conceptual or external references and contract differences are reported.

## 3. Defensible novelty boundary

Potentially defensible, after experiments:

> A causal operator that evaluates Ackermann candidate swept tubes against one factorized,
> action-independent future occupancy belief while preserving explicit time/cause interaction for
> return-quantile and calibrated hazard prediction.

This statement requires all of the following evidence:

- action-invariance and candidate-permutation property tests;
- explicit-product vs implicit concat/generic-attention ablations;
- same-contract flat, recurrent and latent-world-model baselines;
- risk structure and calibration ablations;
- equal-parameter/FLOP and fixed-candidate/horizon controls;
- multi-seed ID/OOD navigation and target-device timing.

Do not claim novelty for TQC, a ConvGRU/attention block, residual bicycle dynamics, an ensemble/UCB,
reward weights, the fixed guard, or merely combining known modules.

## 4. Literature-derived design constraints

- Preserve physical action parity so improvement can be attributed to representation/operator design.
- Keep observed posterior and dense future read-only across candidates.
- Keep simulator ground truth in labels, never deployment inputs.
- Use real executed transitions for the primary Bellman target to isolate model bias.
- Report prediction, ranking, calibration and navigation together; one proxy is insufficient.
- Separate aleatoric event risk from epistemic/OOD disagreement.
- Compare Local control with the same localization input and report localization failures separately.
- Keep a classical controller as an external systems reference without calling it a clean ablation.

## 5. Required baseline families

Minimum paper set: current/same-action TQC, parameter-matched flat TQC, recurrent TQC, factorized BEV
with implicit fusion, generic candidate cross-attention, a compact latent world-model method, a
risk-sensitive distributional baseline and a classical trajectory optimizer. Exact registered IDs and
fairness rules are in [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md).

## 6. Reviewer risks

| Objection | Evidence needed |
|---|---|
| “incremental module stack” | operator invariants plus factorial isolation |
| “only a larger model/planner” | equal parameter/FLOP/K/H comparisons |
| “hazard is renamed collision classifier” | cause/time/censoring/calibration ablations |
| “physics prior is wrong” | system-ID, nominal/residual comparison, fail-safe fallback |
| “world model is just auxiliary loss” | action-conditioned gradient and candidate-ranking evidence |
| “safety claims exceed evidence” | conservative language, event counts, staged real ODD |

## 7. Search/update checklist

Before protocol freeze and again before submission, search IEEE Xplore, ACM DL, Web of Science/Scopus,
Google Scholar, arXiv and venue proceedings for combinations of LiDAR, dynamic obstacle, occupancy
forecast, trajectory-conditioned risk, Ackermann, distributional RL, world-model planning and
real-robot navigation. Record query/date, deduplicate versions, prefer peer-reviewed primary sources,
separate author claims from our inference and add any closer method to the frozen comparison plan or
explain why comparison is infeasible.
