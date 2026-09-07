# Glossary

문서 전체에서 반복 정의하지 않고 아래 의미를 사용한다.

## 상태와 증거

| 용어 | 의미 |
|---|---|
| `PROPOSED` | 설계만 존재하고 실행 코드 증거가 없음 |
| `IMPLEMENTED` | production/test code path가 존재함 |
| `REGRESSION-TESTED` | 코드 회귀 통과; 학습 성능을 뜻하지 않음 |
| `SMOKE-VALIDATED` | 제한된 wiring/reset/step/save 경로 확인 |
| `FORMALLY-TRAINED` | frozen training contract와 completion artifact 존재 |
| `FORMALLY-BENCHMARKED` | locked episode matrix와 aggregate 존재 |
| `SIM-VALIDATED` | predefined simulator performance gate 통과 |
| `REAL-ROBOT-VALIDATED` | predefined real protocol과 raw evidence 통과 |
| locked test | tuning에 사용하지 않는 고정 scenario/seed manifest |
| historical evidence | 과거 checkout/환경에서 생성되어 현재 결과로 자동 승계할 수 없는 기록 |
| promotion | 정의된 gate를 통과한 immutable model bundle만 다음 단계에 허용하는 절차 |

## 시스템과 모델

| 용어 | 의미 |
|---|---|
| Local | LiDAR와 robot-relative subgoal에서 단기 Ackermann trajectory를 고르는 계층 |
| Global | partial map과 frozen Local capability로 장기 subgoal을 고르는 계층 |
| TRACTOR-TQC | **Trajectory-Risk via Action-Conditioned Tube-Occupancy Reasoning with Truncated Quantile Critics**; swept tube와 future occupancy를 명시적으로 결합하는 TQC Local model |
| `[kappa,v_ref,L]` | curvature, reference speed, arc length의 physical trajectory action |
| nominal rollout | attested Hunter actuator/bicycle equations으로 만든 deterministic prior |
| vehicle residual | obstacle나 localization이 아닌 vehicle response 오차만 보정하는 learned term |
| factorized belief | free/static/dynamic/unknown occupancy, dynamic flow, scene/ego/plant latent의 분리 표현 |
| swept tube | 시간별 robot footprint occupancy distribution |
| causal interaction | action-independent scene future를 읽고 candidate-private tube feature만 갱신하는 연산 |
| return quantile | discounted return distribution의 분위수 |
| hazard | 아직 사건이 없다는 조건에서 다음 step에 원인별 사건이 날 확률 |
| survival | 해당 step까지 사건 없이 지속될 확률 |
| censoring | 관측 종료가 사건 부재를 의미하지 않는 표본 처리 |
| calibration | 예측 확률과 관측 빈도의 일치도를 held-out data에서 조정·평가하는 절차 |

## 데이터·학습·배포

| 용어 | 의미 |
|---|---|
| sequence replay | recurrent burn-in과 loss window를 episode 경계 안에서 샘플링하는 replay |
| actual `delta_t` | nominal 0.1 s로 대체하지 않은 timestamp 차이 |
| privileged label | simulator ground truth에서 생성하지만 deployment input으로 금지된 target |
| counterfactual candidate | 실행하지 않은 trajectory의 simulator-derived auxiliary target |
| real-transition-only TQC | Bellman update에는 실제 실행 transition만 사용하는 원칙 |
| stop-gradient | 특정 loss가 upstream module을 갱신하지 못하도록 graph를 차단함 |
| fingerprint | architecture/data/training/physical semantics의 canonical hash |
| strict incompatibility | 의미가 다르면 shape가 같아도 load를 거부함 |
| warm start | 다른 experiment에서 일부 online weights만 가져오는 새 lineage |
| resume | 같은 experiment의 weights, target, optimizer, replay, RNG, counters를 정확히 복구함 |
| deployment bundle | optimizer/replay 없이 검증된 inference weights, calibrator, manifest를 담은 artifact |
| fail closed | identity, freshness, validity 또는 deadline 실패 시 실행 대신 보수적 fallback |
| HIL | 실제 또는 동등 controller/compute를 포함하되 자유주행 위험은 제한한 시험 |

## 표기

`B`: batch, `K`: candidate 수, `H`: future horizon, `M`: sparse tube cell 수,
`Q_m`: vehicle residual member 수, `Q_r`: risk member 수, `C`: risk cause 수.
단위는 metre, second, radian, metre/second, curvature `m^-1`을 기본으로 한다.
