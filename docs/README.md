# TRACTOR-TQC 문서 안내

기준일: **2026-09-07 KST**

이 폴더는 Hunter SE용 **Trajectory-Risk via Action-Conditioned Tube-Occupancy Reasoning with
Truncated Quantile Critics (TRACTOR-TQC)** 연구의 설계, 구현, 평가 및 실차 전환 계약을 담는다.
문서는 역할이 겹치지 않도록 제한했다. 기존 328D flat-MLP TQC Local baseline과 별도로
TRACTOR-TQC의 모델부터 formal data 수집, sequence replay, B1–B8/A7–A9 학습, calibration,
locked evaluation과 campaign 집계까지 전용 code path가 구현돼 있다. 코드의 존재와
회귀 통과는 성능 증거가 아니며, 학습·정식 benchmark·calibration·target-hardware 지연·실차
결과는 별도 immutable artifact가 생기기 전까지 `UNMEASURED`다.

## 가장 짧은 읽기 순서

1. [CURRENT_STATUS.md](CURRENT_STATUS.md): 실제 구현, 증거, 차단 항목
2. [TRACTOR_TQC_MODEL_SPEC.md](TRACTOR_TQC_MODEL_SPEC.md): 모델·수식·추론 계약
3. [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md): 가설, 학습, 비교, 판정 기준
4. [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md): 코드 변경 순서와 운영법

## 문서별 역할

| 문서 | 한 가지 책임 |
|---|---|
| [CURRENT_STATUS.md](CURRENT_STATUS.md) | 구현/설정/시험/학습/실차 상태의 유일한 정본 |
| [TRACTOR_TQC_MODEL_SPEC.md](TRACTOR_TQC_MODEL_SPEC.md) | architecture, tensor, 수식, gradient, inference 계약 |
| [DATA_AND_REPLAY.md](DATA_AND_REPLAY.md) | sequence dataset, label, split, replay 계약 |
| [CHECKPOINT_AND_COMPATIBILITY.md](CHECKPOINT_AND_COMPATIBILITY.md) | 저장, resume, fingerprint, 배포 호환성 |
| [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md) | 연구 질문, baseline, 학습, benchmark, ablation, 통계, 출판 gate |
| [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) | source ownership, 단계별 구현, 통합, 실행·장애 대응 |
| [TEST_PLAN.md](TEST_PLAN.md) | unit부터 실차까지 검증 항목과 통과 조건 |
| [SIMULATION_ENVIRONMENT_V2.md](SIMULATION_ENVIRONMENT_V2.md) | v2 curriculum, scene/motion/shape, calibration, ID/OOD suite 계약 |
| [SIM2REAL.md](SIM2REAL.md) | robot contract, safety case, HIL·실차 승격 절차 |
| [LITERATURE_REVIEW.md](LITERATURE_REVIEW.md) | 선행연구, novelty 경계, 필수 비교군 |
| [EXPERIMENT_REGISTRY.md](EXPERIMENT_REGISTRY.md) | 실험 ID, 불변 기록, 결정 및 결과 상태 |
| [GLOSSARY.md](GLOSSARY.md) | 반복 정의를 대신하는 공통 용어 |
| [evidence/README.md](evidence/README.md) | current audit, claim ledger, historical evidence index |
| [verification/README.md](verification/README.md) | TRACTOR 이전 검증 기록의 해석 범위 |
| [templates/EXPERIMENT_CARD.md](templates/EXPERIMENT_CARD.md) | run 시작 전 기록 양식 |
| [templates/MODEL_CARD.md](templates/MODEL_CARD.md) | checkpoint 승격·배포 양식 |
| [templates/REAL_ROBOT_TRIAL_CARD.md](templates/REAL_ROBOT_TRIAL_CARD.md) | 실차 승인·실행 양식 |

`README.md` 자체는 위 문서의 위치와 권한만 설명하며 연구 사실의 정본이 아니다.

## 정본 우선순위

| 질문 | 우선 문서 |
|---|---|
| 지금 무엇이 존재하고 검증됐는가? | `CURRENT_STATUS` → `EXPERIMENT_REGISTRY` → artifact |
| 모델/API/shape가 무엇인가? | `TRACTOR_TQC_MODEL_SPEC` |
| 데이터와 resume가 호환되는가? | `DATA_AND_REPLAY` + `CHECKPOINT_AND_COMPATIBILITY` |
| 성공 또는 논문 주장을 할 수 있는가? | `RESEARCH_PROTOCOL` + `evidence/README` |
| 다음 코딩 작업은 무엇인가? | `IMPLEMENTATION_PLAN` |
| 실차를 시작해도 되는가? | `SIM2REAL` + `TEST_PLAN` |

충돌 시 최신 immutable run artifact가 개별 실험 사실을 소유하고,
`CURRENT_STATUS`가 프로젝트 전체 상태를 소유한다. 계획 문서의 목표값을 결과로 인용하지
않는다.

## 상태 표기

| Label | 의미 |
|---|---|
| `PROPOSED` | 문서 설계만 존재 |
| `IMPLEMENTED` | 실행 가능한 code path 존재 |
| `CONFIGURED` | profile/runner/interface 준비 |
| `REGRESSION-TESTED` | 현재 checkout 회귀 통과 |
| `SMOKE-VALIDATED` | 제한된 실행 경로 확인 |
| `FORMALLY-TRAINED` | frozen training contract 완료 |
| `FORMALLY-BENCHMARKED` | locked evaluation과 aggregate 완료 |
| `SIM-VALIDATED` | 사전 정의 simulator 성능 gate 통과 |
| `REAL-ROBOT-VALIDATED` | 사전 정의 실차 protocol 통과 |

하위 label은 상위 label을 뜻하지 않는다. 특히 regression, smoke, Gazebo command tracking은
학습 성능이나 실차 안전성의 증거가 아니다.

## 문서 유지 규칙

1. 현재 사실은 `CURRENT_STATUS`와 `EXPERIMENT_REGISTRY`에 먼저 반영한다.
2. tensor/action/gradient 변경은 model fingerprint와 model spec을 함께 바꾼다.
3. data/reward/termination/split 변경은 새 schema/protocol/output root를 만든다.
4. 실험 진행 중 frozen protocol을 소급 수정하지 않는다.
5. 빈 metric을 0으로 채우지 않고 `missing` 또는 `not run`으로 남긴다.
6. simulator와 real-robot evidence를 한 label로 합치지 않는다.
7. 새 문서는 기존 문서와 책임이 겹치지 않을 때만 추가하고 이 표에 등록한다.
8. 삭제된 tracked 문서는 Git history와 artifact manifest로 추적한다. 과거 uncommitted 중복본은
   통합 정본과 원시 evidence만 보존하며 문장 단위 복원을 보장하지 않는다.

## 현재 핵심 경계

- prior audit의 Docker `2,138 passed`는 provenance가 불완전한 code-health snapshot이다.
- Stage-2 L0–L5 formal matrix는 training **0/30**, benchmark **0/30**이다.
- TRACTOR-TQC와 matched B1–B8 implementation, formal realized-track collector, 공통
  training/calibration/locked-evaluation campaign, frozen protocol v1과 616-instance split-safe
  scenario plan은 존재하지만 dataset/training/benchmark/calibration/target timing 결과는 없다.
- 별도 `tractor_env_v2` curriculum과 48개 fixed ID/OOD scenario가 구현·checksum 고정됐지만,
  아직 navigation rollout 또는 비교 성능 증거는 아니다.
- accepted Local을 고정한 formal Global 결과와 Hunter SE 실차 navigation 증거도 없다.

따라서 현재 허용되는 요약은 다음과 같다.

> The package contains a TQC-based Local baseline, hierarchical infrastructure, and an
> untrained regression-tested TRACTOR-TQC and matched-baseline implementation with a formal comparison
> runner. Formal training, comparative
> benchmarking, calibration, target-hardware timing, and real-robot validation remain open.
