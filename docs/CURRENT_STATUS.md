# Current Status

기준일: **2026-09-07 KST**
대상: `main`의 `ffb5fcc28e00e9668cde8305c2300f9d4627f8dd` 기반 working tree와
본 문서-코드 정합성 수정, 사용자 소유 untracked media

## 결론

현재 패키지는 기존 328D TQC Local infrastructure에 더해 **TRACTOR-TQC 학습 실행 경로**를
구현했다.

- 4×80 front-LiDAR + 8D tail의 **328D** Local observation
- tanh-Gaussian actor와 `[kappa, v_ref, L]` trajectory action
- profile-matched differentiable actuator/bicycle rollout, Pure-Pursuit adapter, fixed ActionGuard
- two critics × 25 quantiles TQC와 scalar supervised `RiskCritic`
- 최대 8개 counterfactual trajectory candidate 저장·평가 기반
- localization interface, partial/visited map, hierarchical Global infrastructure
- typed `encode → propose → score` TRACTOR-TQC와 A7/A8/A9 variant
- ego-warped four-class BEV, action-independent future, nominal/residual Ackermann rollout
- probabilistic swept tube, candidate-private explicit occupancy product, temporal aggregation
- twin return quantiles, cause-time hazard, ordered severity heads, conservative batched selector
- immutable/checksummed sequence episode store, reset-prefix index/sampler와 exact sampler resume
- Environment v2 episode collector와 `value → risk → actor/entropy → EMA` 전용 sequence trainer
- disjoint optimizer/EMA target과 value·risk·actor loss, atomic file publication 기반 checkpoint/resume와 inference-only bundle
- current direct 328D TQC인 B1과 B2–B8 executable comparison architecture
- privileged pre-action simulator snapshot에서 actual future obstacle track을 맞추는 formal counterfactual relabeler
- calibration-only Platt fitting, locked-test common evaluator와 complete-matrix campaign orchestrator
- frozen acceptance/scenario plan, dataset/preflight/latency, calibration/ranking/seed-level paired-statistics 및 formal matrix 검사 도구
- opt-in `tractor_env_v2`: 5단계 curriculum, TTC/DCPA conflict, 4개 shape, 10개 motion,
  3개 interaction mode, Hunter 지향 footprint, 5× physics substep와 48개 checksum 고정 ID/OOD scenario

이는 위에 열거한 **개발용 실행 경로**의 `IMPLEMENTED + REGRESSION-TESTED` code evidence다.
Stage 3/4 목적함수, Stage-5 value+risk 원자적 transaction, 전체 supervision schema,
H1–H3 evaluator, semantic promotion lineage, system-axis frozen scenario와 rollout source-parity는
아직 구현되지 않아 formal campaign은 코드에서 fail-closed된다. 학습, held-out calibration,
formal benchmark, target-hardware timing 또는 실차 성능 증거는 아니다.

## 증거 상태

| 항목 | 상태 | 해석 |
|---|---|---|
| Docker full regression | prior report `2,138 passed` | exact command/image/commit provenance가 불완전한 code-health snapshot |
| current combined ROS+Torch regression | **`2,224 passed`** (`258.55 s`, 정합성 수정 후) | sourced ROS workspace overlay + Torch Docker, CPU-only code-health; 성능 증거 아님 |
| Stage-2 L0–L5 training | **0/30** | five seeds × six baselines의 formal completion 없음 |
| Stage-2 benchmark | **0/30** | locked test aggregate 없음 |
| accepted/promoted Local | 없음 | Global formal campaign 선행조건 미충족 |
| paper comparison code | development runner와 navigation evaluator `IMPLEMENTED`; full-paper formal stages `BLOCKED` | readiness gap이 남아 collect/train/calibrate/evaluate/aggregate가 fail-closed |
| TRACTOR synthetic execution | A7/A9 default forward 완료; A9 1회 CPU E2E `142.24 ms`로 100 ms deadline 초과 | 실행성 smoke일 뿐 target hardware/p99/ROS-load gate가 아님 |
| paper comparison formal data | **0/616** | realized-track corpus를 아직 수집하지 않음 |
| paper comparison training | **0/55** | B1–B8/A7–A9 × five seeds; checkpoint 없음 |
| paper comparison calibration | **0/20** | A7/A8/A9/B8 × five seeds; calibration artifact 없음 |
| paper comparison locked eval | **0/55 runs, 0/9,680 episodes** | complete matrix와 aggregate 없음 |
| TRACTOR protocol/data plan | acceptance/scenario `tractor_protocol_v1` frozen; formal implementation readiness `NOT READY` | 616 geometry 계약은 존재하지만 문서 전체 연구축 구현 증거가 아님 |
| TRACTOR simulation env v2 | implemented/configured; 6 suites×8 = 48 fixed scenarios | 학습·navigation rollout·성능·sim-to-real 증거 아님 |
| v2 calibration | `engineering_prior`; application classification 검증 | measured Hunter system-ID가 아니며 mass/wheel은 model-only |
| Global formal result | 없음 | infrastructure 존재만 확인 |
| Gazebo Hunter drive | historical smoke/command tracking | simulator wiring 증거이며 실차 fidelity가 아님 |
| Hunter SE real navigation | 없음 | real-robot 또는 GPS-denied robustness 주장 불가 |

현재 audit와 claim 제한은 [evidence/README.md](evidence/README.md), 개별 run 상태는
[EXPERIMENT_REGISTRY.md](EXPERIMENT_REGISTRY.md)가 소유한다.

## Current Local 계약

```text
scan history       4 × 80 = 320
robot/goal tail              8
observation                 328
```

8D tail은 goal distance, heading error, 이전 normalized action 3개, measured speed,
yaw rate, centre steering 순서다. FrameStack은 current-first raw concatenate이며 learned
temporal state, timestamp, ego warp가 없다. policy는 front 180°를 보고 environment의
collision/proximity path는 full scan을 사용할 수 있다.

Actor는 `328→256→256→256→mean/log_std(3)`이고 normalized action을
`[kappa,v_ref,L]`로 decode한다. `L`은 Global option duration이 아니라 Local arc preview와
commit에 쓰인다. 실행은 Pure-Pursuit와 guard를 거쳐 `[speed,centre_steering]`을 publish한다.

TQC는 2×25 quantile을 사용하고 Bellman target에서 정렬된 50개 중 상위 4개를 제거한다.
기존 scalar risk baseline은 collision cause/time/severity와 calibrated epistemic uncertainty를
보존하지 않는다. 기존 replay도 IID `done` transition이다. 새 TRACTOR 경로만
`tractor_sequence_v2`, timestamp-derived discount, censoring, `terminated`/`truncated` 및
reset-prefix recurrence를 별도 계약으로 사용하며 legacy replay를 묵시적으로 변환하지 않는다.

## TRACTOR 구현 inventory

| 영역 | 현재 상태 | 핵심 경로 |
|---|---|---|
| A7/A8/A9 model | implemented/tested | `rl/networks/tractor/*`, `config/tractor/model*.yaml` |
| B1–B8 comparison model | B1 current direct 328D TQC로 교정; B2–B8 implemented/tested, 모두 untrained | `rl/algorithms/comparison_baselines/*`, `config/tractor/baseline_models.yaml` |
| learning math/runner | Bellman value→risk-head→actor/entropy→EMA 개발 경로 구현; Stage 3/4 및 Stage-5 원자 transaction 미구현 | `rl/algorithms/tractor_tqc/*`, `training/train_tractor_tqc.py` |
| sequence replay/data plan | core transition/candidate/privileged-snapshot schema와 relabeler 구현; dense BEV/flow 및 전체 lineage schema 미구현 | `rl/replay/sequence_*`, `training/realized_counterfactual.py` |
| checkpoint/export | payload hash, role, model fingerprint, sampler/RNG resume와 inference-only export 구현; canonical superset/lineage/startup probe 미구현 | `rl/checkpointing/tractor.py` |
| runtime policy boundary | implemented/unit-tested, not wired to a live ROS node | `navigation/local_rl/tractor_policy.py` |
| protocol/evaluation/statistics | navigation common evaluator와 matrix/statistics 구현; H1–H3 prediction/ranking/risk record 미구현, formal stages 차단 | `evaluation/evaluate_paper_comparison.py`, `evaluation/run_paper_comparison_campaign.py` |
| dynamic label basis | realized displacement/actual-dt helper implemented/tested | `env/humans/dynamic_obstacle_motion.py` |
| simulation environment v2 | implemented/configured, untrained | `env/scenarios/tractor_environment_v2.py`, `config/environment_v2/*` |

## Localization과 Global 경계

Localization interface는 pose, timestamp, covariance, confidence, validity를 제공한다.
기본 Gazebo path는 simulator odometry이며 `lio`는 외부 odometry adapter다. `lidar_odom`은
실제 ICP/NDT 구현이 아니고 `wheel_imu`는 단순 dead reckoning이다. FAST-LIO2/LIO-SAM을
연결하더라도 localization 성능과 Local policy 성능을 분리해 측정한다.

Global layer에는 robot-relative 후보, partial/visited map, topology, action mask와
Dueling Double-DQN infrastructure가 있다. accepted Local, calibrated capability distribution,
formal Global run과 real mission evidence는 없다. Local promotion 전에는 공동 학습하지 않는다.

## Formal run 전 P0 상태

| ID | 결함 | 완료 조건 |
|---|---|---|
| P0-01 | **implemented/tested** | hierarchy footprint를 `0.58 m`로 통일하고 config equality fail-fast 추가 |
| P0-02 | **partial** | config/revision fingerprint는 있으나 rollout source hash와 full semantic fingerprint는 미구현 |
| P0-03 | **implemented/tested basis** | static/dynamic manifest 분리, waypoint velocity를 realized displacement/dt로 생성 |
| P0-04 | **core relabeler implemented; full formal schema blocked; corpus 없음** | privileged pre-action snapshot과 realized cause/time/censor/severity는 구현, complete lineage는 미구현 |
| P0-05 | **implemented for TRACTOR only** | sequence schema와 Bellman control flow가 terminal/time-limit/invalid cut을 분리 |
| P0-06 | **partial; formal-blocked** | path/role/payload hash와 atomic publication은 구현, full lineage/promotion/startup probe는 미구현 |
| P0-07 | **TRACTOR rule implemented/tested** | front-only 미관측 셀을 정확히 unknown으로 고정하고 forward-only action 유지 |

## 다음 순서

1. **완료:** 물리 rollout/profile equality와 current-TQC B1 identity를 fail-fast test로 고정한다.
2. **완료:** `prepare`가 모든 method contract hash와 formal readiness gap을 immutable manifest에 기록한다.
3. **선행 구현:** Stage 3/4 목적함수와 Stage-5 원자적 value+risk/RNG transaction을 완성한다.
4. **선행 구현:** dense supervision/lineage schema, H1–H3 evaluator, semantic checkpoint/promotion 검증을 완성한다.
5. **선행 구현:** vehicle/sensor/localization frozen scenario axes와 rollout parity/source fingerprint를 완성한다.
6. 위 gap을 제거하는 code+test 변경 뒤에만 새 output root에서 formal `collect→train→calibrate→evaluate→aggregate`를 실행한다.
7. target-hardware timing/HIL/contained Hunter trial을 통과한 Local만 승격하고 그 후 Global 연구를 시작한다.

B0/B12와 legacy L0–L5는 행동·제어 또는 online-training 계약이 달라 위 11-method architecture
matrix에 섞지 않고 별도 참고 결과로 보고한다.

## 현재 금지 주장

다음은 아직 말할 수 없다: TRACTOR가 학습되어 성능이 입증됐다, SOTA다, risk가 calibrated/formally
safe하다, Hunter에서 real-time이다, GPS-denied/Global navigation이 검증됐다, Gazebo MAE가
real fidelity를 입증한다, regression이 navigation 성능을 입증한다.
