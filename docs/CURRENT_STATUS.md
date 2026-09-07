# Current Status

기준일: **2026-09-07 KST**
대상: `phase1-hierarchical-navigation`, paper-comparison implementation through
`584025d`, 사용자 소유 untracked media가 있는 working tree

## 결론

현재 패키지는 기존 328D TQC Local infrastructure에 더해 **TRACTOR-TQC 학습 실행 경로**를
구현했다.

- 4×80 front-LiDAR + 8D tail의 **328D** Local observation
- tanh-Gaussian actor와 `[kappa, v_ref, L]` trajectory action
- nominal actuator/bicycle rollout, Pure-Pursuit adapter, fixed ActionGuard
- two critics × 25 quantiles TQC와 scalar supervised `RiskCritic`
- 최대 8개 counterfactual trajectory candidate 저장·평가 기반
- localization interface, partial/visited map, hierarchical Global infrastructure
- typed `encode → propose → score` TRACTOR-TQC와 A7/A8/A9 variant
- ego-warped four-class BEV, action-independent future, nominal/residual Ackermann rollout
- probabilistic swept tube, candidate-private explicit occupancy product, temporal aggregation
- twin return quantiles, cause-time hazard, ordered severity heads, conservative batched selector
- immutable/checksummed sequence episode store, reset-prefix index/sampler와 exact sampler resume
- Environment v2 episode collector와 `value → risk → actor/entropy → EMA` 전용 sequence trainer
- disjoint optimizer/EMA target/value·risk loss, atomic training checkpoint/resume와 inference-only bundle
- B1–B8 executable architecture/optimizer/checkpoint와 A7–A9가 공유하는 immutable sequence-data training contract
- privileged pre-action simulator snapshot에서 actual future obstacle track을 맞추는 formal counterfactual relabeler
- calibration-only Platt fitting, locked-test common evaluator와 complete-matrix campaign orchestrator
- frozen acceptance/scenario plan, dataset/preflight/latency, calibration/ranking/seed-level paired-statistics 및 formal matrix 검사 도구
- opt-in `tractor_env_v2`: 5단계 curriculum, TTC/DCPA conflict, 4개 shape, 10개 motion,
  3개 interaction mode, Hunter 지향 footprint, 5× physics substep와 48개 checksum 고정 ID/OOD scenario

이는 `IMPLEMENTED + REGRESSION-TESTED` code evidence다. 학습, held-out calibration, formal benchmark,
target-hardware timing 또는 실차 성능 증거는 아니다.

## 증거 상태

| 항목 | 상태 | 해석 |
|---|---|---|
| Docker full regression | prior report `2,138 passed` | exact command/image/commit provenance가 불완전한 code-health snapshot |
| current combined ROS+Torch regression | **`2,220 passed`** (`274.50 s`) | sourced ROS workspace overlay + Torch Docker, CPU-only code-health; 성능 증거 아님 |
| Stage-2 L0–L5 training | **0/30** | five seeds × six baselines의 formal completion 없음 |
| Stage-2 benchmark | **0/30** | locked test aggregate 없음 |
| accepted/promoted Local | 없음 | Global formal campaign 선행조건 미충족 |
| paper comparison code | B1–B8/A7–A9 train, formal label, calibration, common eval, campaign `IMPLEMENTED` | runner 존재만 확인; 학습 weights와 성능 artifact 없음 |
| TRACTOR synthetic execution | A7/A9 default forward 완료; A9 1회 CPU E2E `142.24 ms`로 100 ms deadline 초과 | 실행성 smoke일 뿐 target hardware/p99/ROS-load gate가 아님 |
| paper comparison formal data | **0/616** | realized-track corpus를 아직 수집하지 않음 |
| paper comparison training | **0/55** | B1–B8/A7–A9 × five seeds; checkpoint 없음 |
| paper comparison calibration | **0/20** | A7/A8/A9/B8 × five seeds; calibration artifact 없음 |
| paper comparison locked eval | **0/55 runs, 0/9,680 episodes** | complete matrix와 aggregate 없음 |
| TRACTOR protocol/data plan | `tractor_protocol_v1` frozen; 616/616 geometry가 bounded Ackermann filter 통과 | 구성·분리 계약 증거이며 rollout/성능 증거 아님 |
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
| B1–B8 comparison model | implemented/tested, untrained | `rl/algorithms/comparison_baselines/*`, `config/tractor/baseline_models.yaml` |
| learning math/runner | value·risk·actor·EMA transaction implemented/unit-tested, not trained | `rl/algorithms/tractor_tqc/*`, `training/train_tractor_tqc.py` |
| sequence replay/data plan | formal 616-scenario collector/relabeler + shared comparison replay implemented/unit-tested, no collected rollouts | `rl/replay/sequence_*`, `training/collect_formal_comparison_data.py`, `training/realized_counterfactual.py` |
| checkpoint/export | implemented/unit-tested, no promoted model | `rl/checkpointing/tractor.py` |
| runtime policy boundary | implemented/unit-tested, not wired to a live ROS node | `navigation/local_rl/tractor_policy.py` |
| protocol/evaluation/statistics | common locked evaluator, split-isolated calibration and 55-run campaign implemented/unit-tested, no formal records | `evaluation/evaluate_paper_comparison.py`, `evaluation/fit_comparison_calibration.py`, `evaluation/run_paper_comparison_campaign.py` |
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
| P0-02 | **implemented for TRACTOR; legacy coverage improved** | trajectory/replay revision과 전체 TRACTOR structural config fingerprint |
| P0-03 | **implemented/tested basis** | static/dynamic manifest 분리, waypoint velocity를 realized displacement/dt로 생성 |
| P0-04 | **formal relabeler/schema/collector implemented and tested; corpus 없음** | privileged pre-action snapshot은 저장 전용이며 realized timestamp-aligned cause/time/censor/severity를 생성 |
| P0-05 | **implemented for TRACTOR only** | sequence schema와 Bellman control flow가 terminal/time-limit/invalid cut을 분리 |
| P0-06 | **checkpoint/export implemented; live ROS load 미연결** | path/role/hash 선검증, atomic generation, inference-only calibrated bundle |
| P0-07 | **TRACTOR rule implemented/tested** | front-only 미관측 셀을 정확히 unknown으로 고정하고 forward-only action 유지 |

## 다음 순서

1. **완료:** current checkout full regression `2,220 passed`를 기록했다.
2. **완료:** `tractor_protocol_v1` 합격 기준과 616개 split/scenario plan을 digest-lock하고 freeze했다.
3. **구현 완료/실행 대기:** `run_paper_comparison_campaign.py prepare`로 protocol/scenario/data/config digest를 고정한다.
4. simulator를 실행하고 `collect`로 352 development + 88 calibration + 176 locked-test realized-track corpus를 한 번 수집한다. 정책 입력에는 privileged snapshot이 들어가지 않는다.
5. `train`으로 B1–B8/A7–A9 × five seeds의 55 checkpoint를 동일 development sequence와 update budget으로 만든다.
6. `calibrate`는 calibration split만 사용해 A7/A8/A9/B8의 20 calibrator를 만든다.
7. `evaluate`로 모든 checkpoint를 같은 176 locked scenario에서 평가하고 `aggregate`의 55-run/9,680-record 완전성 gate와 paired seed CI를 통과시킨다.
8. target-hardware timing/HIL/contained Hunter trial을 통과한 Local만 승격하고 그 후 Global 연구를 시작한다.

B0/B12와 legacy L0–L5는 행동·제어 또는 online-training 계약이 달라 위 11-method architecture
matrix에 섞지 않고 별도 참고 결과로 보고한다.

## 현재 금지 주장

다음은 아직 말할 수 없다: TRACTOR가 학습되어 성능이 입증됐다, SOTA다, risk가 calibrated/formally
safe하다, Hunter에서 real-time이다, GPS-denied/Global navigation이 검증됐다, Gazebo MAE가
real fidelity를 입증한다, regression이 navigation 성능을 입증한다.
