# Historical Verification Index

The former dated Markdown files were short compatibility notes and are consolidated here. Tracked
historical wording may be recovered from Git history; uncommitted duplicate notes are represented only
by this index. Exact numbers still require their original immutable artifact.

| Date / former record | Reusable scope | Not evidence for |
|---|---|---|
| [2026-09-08 training readiness smoke](2026-09-08_training_readiness_smoke.md) | current build/regression, L0 data path and L4 dynamic-scene wiring | formal comparison or navigation performance |
| 2026-08-26 item1-4 fixes | fail-fast and targeted regression patterns | current model performance |
| 2026-08-26 residual defects round2 | edge-case re-audit discipline | learned residual dynamics |
| 2026-08-27 start-pose/noise/pool/OU | reset, sensor-noise, obstacle-pool provenance | TRACTOR data validity without rerun |
| 2026-08-28 hierarchy phase1 | localization/mission/Local interface checks | completed hierarchy research |
| 2026-08-28 phase1 review fixes | stale input, frame, subgoal and shutdown regression cases | formal navigation success |
| 2026-08-28 phase2 | Local belief vs Global partial-map responsibility | TRACTOR belief implementation |
| 2026-08-28 phase3 | action/guard/option telemetry boundary | deployed TRACTOR path |
| 2026-08-28 phase4 | Global DDQN/action-mask prerequisites | accepted-Local Global result |
| 2026-08-29 phase5 | end-to-end identity/timing/fallback trace requirements | formal benchmark |
| 2026-08-31 live verification | Gazebo simulation-only boundary | real robot behavior |
| 2026-09-02 Stage-1 freeze | source/config/checkpoint freeze practice | current protocol completion |
| 2026-09-02 Stage-2 Local baselines | L0–L5 setup and 0/30 runtime snapshot | trained baseline results |
| 2026-09-03 CAD-informed A/B | geometry/dynamics prior provenance | measured Hunter parameters |
| 2026-09-03 drive control | Gazebo command-following checks | real-fidelity MAE |
| 2026-09-03 hub geometry | footprint inconsistency discovery | resolved footprint |
| 2026-09-03 ros2_control A/B | controller-specific command semantics | cross-controller equivalence |
| 2026-09-03 ros2_control Docker | reproducible simulator runtime practice | target-device timing |

Interpretation rules:

1. Current facts come from `CURRENT_STATUS`, the experiment registry and immutable artifacts.
2. Historical smoke/regression does not become training, benchmark or real evidence.
3. Reuse a historical check as a new regression only after rerunning it against the current
   fingerprints and recording complete provenance.
4. Cite an exact historical result only with its original Git revision and raw artifact.
