# Current Status

기준일: **2026-09-08 KST**
대상: `main`의 `b4de2e4e3523e8210ffe078c887bbd38f4d97bde` 기반 working tree

## 결론

논문용 Local 비교 연구를 실행하는 데 필요했던 **코드 수준의 차단 항목은 모두 구현됐다**.
`formal_research_implementation_readiness()`는 현재 `ready=true`이며 formal
`collect → train → calibrate → evaluate → aggregate`의 각 public entry point가 이 상태를 직접
검사한다. 그러나 정식 데이터, 학습 checkpoint, calibration, locked-test 결과와 실차 증거는
아직 없으므로 연구 결과가 완성된 것은 아니다.

현재 구현은 다음을 포함한다.

- 328D Hunter Local observation과 `[kappa,v_ref,L]` trajectory action;
- A7/A8/A9 TRACTOR-TQC와 같은 계약의 B1–B8 비교 모델;
- ego-warped BEV, action-independent future, Ackermann rollout, swept-tube interaction;
- twin TQC return, cause-time risk, ordered severity와 conservative selector;
- Stage 3 representation, Stage 4 risk/ranking, Stage 5 joint RL의 분리된 학습 및 warm-start;
- Stage-5 value+risk 원자 transaction, actor+entropy rollback과 모든 성공 후 EMA;
- realized timestamp-aligned candidate labels, dense BEV/flow/vehicle-response/tube supervision;
- immutable episode/checksum/index, reset-prefix replay와 formal dataset root manifest;
- semantic component hash, parent/root/stage lineage, exact resume, inference-only export와 startup finite probe;
- H1 prediction, H2 ranking, H3 risk/calibration denominator를 포함한 locked evaluator;
- vehicle/sensor/localization 축이 고정된 616-scenario protocol v2 plan;
- shape-aware Environment v2 48-scenario ID/OOD suite와 Ackermann start-goal feasibility 검사.

## 증거 상태

| 항목 | 상태 | 해석 |
|---|---|---|
| implementation readiness | **READY** | 등록된 코드 capability gap이 비어 있음 |
| targeted regression | **47 passed** | 최근 data/stage/checkpoint/protocol 경로의 Docker 회귀 |
| full Docker regression | **2,253 passed** (`273.95 s`) | sourced ROS overlay + Torch Docker, CPU-only code-health |
| ROS build | **PASS** | `colcon build --packages-select hunter_kinodynamic_rl --symlink-install` |
| live Gazebo collection smoke | **PASS** | curriculum L0 3 episodes/120 steps + L4 static 4/dynamic 8 scene 20 steps; development evidence only |
| formal source preflight | **container identity 입력 전까지 의도적으로 실패** | delivered commit과 exact `HUNTER_CONTAINER_IMAGE_DIGEST`가 필요 |
| legacy Stage-2 L0–L5 | training **0/30**, benchmark **0/30** | 별도 historical matrix, 미실행 |
| formal realized-track corpus | **0/616** | 아직 수집하지 않음 |
| same-contract training | **0/55 final models** | B1–B8/A7–A9 × 5 seeds; A 계열은 내부 Stage 3→4→5 lineage 사용 |
| risk calibration | **0/20** | A7/A8/A9/B8 × 5 seeds |
| locked evaluation | **0/55 runs, 0/9,680 episodes** | 11 methods × 5 seeds × 176 scenarios |
| promoted Local / formal Global | 없음 | Local formal gate와 deployment gate 미통과 |
| target-hardware/HIL/real robot | 없음 | 실시간·sim-to-real·GPS-denied 성능 주장 불가 |

회귀와 Gazebo smoke는 실행 경로의 code-health 증거일 뿐 학습 성능 증거가 아니다. 빈 formal
수치는 실패율 0이 아니라 **아직 실행하지 않음**을 뜻한다.

## 실행 계약

정식 corpus는 `tractor_formal_dataset_manifest_v1`로 다음을 한 번에 고정한다.

- 616개 episode 파일의 byte SHA-256, row/split counts와 sequence-index hash;
- scenario manifest, protocol, resolved contract와 behavior seed;
- committed source commit/content manifest, 실행 module hash와 container image digest.

수집 뒤 episode가 추가·삭제·변조되거나 runtime source/container가 달라지면 training,
calibration 또는 evaluation이 fail closed된다. Formal source는 tracked diff와 untracked source
file을 허용하지 않는다. 사용자 소유 `docs/figures/`, `docs/media/`, `temp/` 같은 제외된
비-source asset은 검사 범위 밖이다. Formal 실행 시에는 delivered commit을 checkout하고 그
실행 container의 content digest를 명시해야 한다.

훈련 budget은 protocol에 고정되어 있다.

| 대상 | total updates | 단계 |
|---|---:|---|
| A7/A8/A9 | 150,000 | Stage 3: 30,000 → Stage 4: 30,000 → Stage 5: 90,000 |
| B1–B8 | 150,000 | 단일 baseline training lineage |

Formal CLI에서 update, batch size(`64`) 또는 checkpoint interval(`5,000`)을 바꾸면 거부한다.
Stage 전환은 online compatible weights만 가져오며 optimizer, sampler, RNG, counter는 새로
시작하고 full online value path를 target에 복사한다. Exact resume은 반대로 모든 학습 상태를
복원한다.

## 시뮬레이션·시나리오 계약

`tractor_scenario_plan_v2`는 development 352, calibration 88, locked-test 176개로 총 616개다.
geometry와 seed는 split 간 중복될 수 없고, 모든 의도된 feasible scenario는 static과 dynamic
장애물의 `t=0` 상태를 함께 고려한 grid/Ackermann feasibility 검사를 통과한다. 목적상 goal을
막는 negative fixture는 별도 reason으로 명시된다.

고정 system axes는 다음과 같다.

- vehicle: nominal, low friction, steering response, command latency;
- sensor: nominal, LiDAR range noise, LiDAR dropout, frame dropout;
- localization: nominal, odometry noise.

Environment v2는 robot-relative start/goal, topology-conditioned obstacle, shape-aware collision,
substep motion과 bounded retry를 사용한다. L-shape SDF collision/visual pose는 6-DoF pose 형식으로
검증된다. `multi_step`은 요청 전후 `/clock` queue를 안정화하고 한 physics tick 미만의 허용치로
검사한다. 2026-09-08 live smoke는 L0 3개 episode/120 step의 dataset(75 windows, validator error 0)과
L4 static 4/dynamic 8 scene의 20 step과 다음 cleanup reset에서 통과했다. 상세 기록은
[`verification/2026-09-08_training_readiness_smoke.md`](verification/2026-09-08_training_readiness_smoke.md)에
있다. 이는 특정 policy가 목표에 도달한다는 성능 증거가 아니다.

## 모델·평가 상태

| 영역 | 구현 상태 | 아직 없는 증거 |
|---|---|---|
| A7/A8/A9 | forward, loss, Stage 3–5 training, checkpoint lineage 구현/테스트 | trained weights와 비교 성능 |
| B1–B8 | 같은 data/action/evaluation 계약 구현/테스트 | trained weights와 equal-compute 결과 |
| H1 | occupancy NLL/IoU/F1, flow EPE, tube coverage/count | held-out multi-seed 수치 |
| H2 | ranking regret, NDCG, unsafe top-1, candidate coverage | paired CI와 ablation 결과 |
| H3 | Brier/NLL/ECE, cause-time NLL/accuracy, selective-risk curve | fitted calibrator와 event support |
| navigation/H4 | success/collision/SPL/time/clearance/operations | complete locked matrix |
| deployment/H5 | semantic Stage-5 binding, calibration lineage와 latency gate | target-device 10k decisions/HIL/real trial |

지원하지 않는 metric은 0으로 기록하지 않고 `available=false`와 명시적 denominator로 남긴다.
집계는 seed를 replication unit으로 사용하고 vehicle/sensor/localization/system-domain 별 paired
effect를 함께 출력한다.

## Localization과 Global 경계

Localization interface는 pose, timestamp, covariance, confidence, validity를 제공한다. 기본 Gazebo
path는 simulator odometry이고, `lio`는 외부 odometry adapter다. `lidar_odom`은 완전한 ICP/NDT
구현이 아니며 `wheel_imu`는 단순 dead reckoning이다. TRACTOR-TQC는 localization drift 자체를
해결하지 않는다.

Global infrastructure는 존재하지만 accepted Local과 formal Global artifact는 없다. Local formal
simulation, target timing/HIL과 승인된 deployment bundle 이전에 Global/실차 성능을 주장하지 않는다.

## 다음 실행 순서

1. delivered commit에 exact container digest를 결합해 formal preflight를 통과시킨다.
2. 새 immutable output root에서 formal corpus 616개를 수집하고 root manifest를 검증한다.
3. 모든 method/seed를 고정 budget으로 학습하고 A 계열 Stage lineage를 검증한다.
4. calibration split만으로 A7/A8/A9/B8 calibrator를 fitting한다.
5. locked 9,680 episodes를 평가·집계하고 H1–H5 gate를 판정한다.
6. 통과한 Local만 target timing, HIL, contained Hunter trial과 Global follow-up으로 승격한다.

## 현재 금지 주장

아직 다음을 말할 수 없다: TRACTOR가 baseline보다 우수하다, calibrated/formally safe하다,
Hunter에서 real-time이다, 실차나 GPS-denied navigation이 검증됐다, regression/Gazebo smoke가
navigation 성능 또는 real fidelity를 입증한다.
