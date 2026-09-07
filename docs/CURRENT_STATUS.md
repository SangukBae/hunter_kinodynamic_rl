# Current Status

기준일: **2026-09-07 KST**
대상: `phase1-hierarchical-navigation`, audit 당시 commit
`42e192f2f038c8cc9f214ff15ed6f8e57c00c5f6`, dirty working tree

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
- frozen acceptance/scenario plan, dataset/preflight/latency, calibration/ranking/seed-level paired-statistics 및 formal matrix 검사 도구
- opt-in `tractor_env_v2`: 5단계 curriculum, TTC/DCPA conflict, 4개 shape, 10개 motion,
  3개 interaction mode, Hunter 지향 footprint, 5× physics substep와 48개 checksum 고정 ID/OOD scenario

이는 `IMPLEMENTED + REGRESSION-TESTED` code evidence다. 학습, held-out calibration, formal benchmark,
target-hardware timing 또는 실차 성능 증거는 아니다.

## 증거 상태

| 항목 | 상태 | 해석 |
|---|---|---|
| Docker full regression | prior report `2,138 passed` | exact command/image/commit provenance가 불완전한 code-health snapshot |
| current combined ROS+Torch regression | **`2,207 passed`** (`258.41 s`) | sourced ROS + Torch Docker, CPU-only code-health; 성능 증거 아님 |
| Stage-2 L0–L5 training | **0/30** | five seeds × six baselines의 formal completion 없음 |
| Stage-2 benchmark | **0/30** | locked test aggregate 없음 |
| accepted/promoted Local | 없음 | Global formal campaign 선행조건 미충족 |
| TRACTOR-TQC code | `IMPLEMENTED`, 새 학습 경로 포함 targeted `71 passed` | 학습 실행은 안 했으며 weights는 무작위 초기화; 성능 주장 불가 |
| TRACTOR synthetic execution | A7/A9 default forward 완료; A9 1회 CPU E2E `142.24 ms`로 100 ms deadline 초과 | 실행성 smoke일 뿐 target hardware/p99/ROS-load gate가 아님 |
| TRACTOR training/benchmark | **0 / 0** | 학습·calibration·locked test artifact 없음 |
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
| learning math/runner | value·risk·actor·EMA transaction implemented/unit-tested, not trained | `rl/algorithms/tractor_tqc/*`, `training/train_tractor_tqc.py` |
| sequence replay/data plan | Environment v2 collector + replay adapter implemented/unit-tested, 616 fixed scenarios planned, no collected rollouts | `rl/replay/sequence_*`, `training/tractor_episode_collector.py`, `training/tractor_sequence_training.py` |
| checkpoint/export | implemented/unit-tested, no promoted model | `rl/checkpointing/tractor.py` |
| runtime policy boundary | implemented/unit-tested, not wired to a live ROS node | `navigation/local_rl/tractor_policy.py` |
| protocol/evaluation/statistics | frozen v1 + implemented/unit-tested, no formal records | `config/tractor/protocol.yaml`, `evaluation/tractor_*` |
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
| P0-04 | **generator/loss/schema implemented and scripted-tested; corpus 없음** | realized tracks로 cause/time/censor, closest-time speed severity 및 tie priority 생성 |
| P0-05 | **implemented for TRACTOR only** | sequence schema와 Bellman control flow가 terminal/time-limit/invalid cut을 분리 |
| P0-06 | **checkpoint/export implemented; live ROS load 미연결** | path/role/hash 선검증, atomic generation, inference-only calibrated bundle |
| P0-07 | **TRACTOR rule implemented/tested** | front-only 미관측 셀을 정확히 unknown으로 고정하고 forward-only action 유지 |

## 다음 순서

1. **완료:** current checkout full regression `2,207 passed`를 기록했다.
2. **완료:** `tractor_protocol_v1` 합격 기준과 616개 split/scenario plan을 digest-lock하고 freeze했다.
3. **구현 완료/실행 대기:** `train_tractor_tqc.py`로 Environment v2 development episode를 수집하고 sequence replay 학습을 실행한다. 현재 collector의 `nominal_preaction_rollout_summary_v1` 라벨은 개발용이며 formal validator가 논문 증거로 거부한다.
4. current L0–L5 baseline의 30 training과 30 benchmark를 별도 immutable output root에서 끝낸다.
5. 사용자 승인 시 새 전용 runner로 A7 representation/value→risk→joint actor 학습을 실행한다.
6. A7 core를 B1–B8 same-contract/equal-compute baseline과 최소 five seeds로 비교한다.
7. `evaluation_v2_*` 48개 고정 scenario, held-out calibration, target-hardware timing을 통과한 모델만 bundle로 export한다.
8. HIL과 contained Hunter trial 후에만 Global 연구를 시작한다.

## 현재 금지 주장

다음은 아직 말할 수 없다: TRACTOR가 학습되어 성능이 입증됐다, SOTA다, risk가 calibrated/formally
safe하다, Hunter에서 real-time이다, GPS-denied/Global navigation이 검증됐다, Gazebo MAE가
real fidelity를 입증한다, regression이 navigation 성능을 입증한다.
