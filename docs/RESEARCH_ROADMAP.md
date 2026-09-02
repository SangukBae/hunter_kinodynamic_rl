# Research Roadmap

이 문서는 `hunter_kinodynamic_rl`의 **향후 연구개발 방향**을 정의하는 정본이다.
현재 구현·검증 상태는 `CURRENT_STATUS.md`, 실행 절차는
`RUNBOOK_HIERARCHICAL_NAVIGATION.md`, 알고리즘의 현재/목표 데이터 계약은
`ARCHITECTURE.md`, 실험 계약은 `RESEARCH_PROTOCOL.md`와 `BENCHMARK.md`를 따른다.
과거 delivery/completion/verification 문서는 작성 당시의 증거이며 이 로드맵을
대체하지 않는다.

기준일: 2026-09-02

## 1. 연구 North Star

기존 구조를 교체하지 않고 다음 두 연구축을 순서대로 깊게 만든다.

### Paper 1 — Local navigation

현재 기반:

```text
TQC -> [kappa, v_ref, L] -> nominal dynamics rollout
    -> scalar risk -> candidate supervision / margin-weighted actor penalty
```

목표:

```text
Kinodynamic Policy
    + Uncertainty-Calibrated Multi-Task Risk
    + Physics + Learned Residual Dynamics Ensemble
    + Progress-Preserving Counterfactual Policy Improvement
```

가제: **Risk-Calibrated Counterfactual Policy Improvement for Kinodynamic
Ackermann Navigation**

핵심 질문:

> Can a robot improve its policy using physically grounded safe alternatives
> while preserving task progress, rather than merely penalizing unsafe actions?

TQC는 안정적인 구현 기반이지 논문의 독창성 주장이 아니다. 핵심 기여는
구조화된 trajectory action, 물리적으로 grounded된 미래 위험, 위험/동역학
불확실성, 그리고 progress를 보존하는 counterfactual policy improvement다.

### Paper 2 — Global navigation

목표:

```text
Online Partial Map
    + Experience-Aware Topological Memory
    + Frozen-Local Capability Distribution
    + Localization-Uncertainty Propagation
```

가제: **Capability-Aware Hierarchical Navigation with Experience Memory under
Localization and Dynamics Uncertainty**

핵심 질문:

> Can a long-horizon planner make better decisions by reasoning about what its
> own local controller can reliably accomplish?

Global 연구는 Local policy와 calibration을 먼저 완료하고 checkpoint를 동결한 뒤
시작한다. Global의 DDQN 자체는 기여가 아니라 candidate-conditioned value
estimator다.

## 2. 현재 구현과 목표 구현의 경계

| 구성요소 | 현재 구현 | 목표 구현 |
|---|---|---|
| Local action | `[kappa, v_ref, L]` constant-curvature primitive | 계약 유지; spline/Bezier는 비교 baseline |
| Dynamics | nominal Hunter actuator + bicycle rollout | nominal physics + learned residual ensemble |
| Risk | scalar supervised risk critic | multi-task risk factors + calibrated ensemble uncertainty |
| Counterfactual | candidate risk supervision + safer-margin actor weighting | progress/feasibility 제약과 uncertainty gate를 갖는 직접 policy-improvement target |
| Global feedback | geometry, learned feasibility/risk feature | Local capability distribution `P_success`, risk mean/uncertainty, progress/time/stop/unrecoverable |
| Memory | edge별 traversal/success/failure, path length, elapsed time, mean/max risk 등을 저장하나 Global 입력은 일부 요약만 소비 | uncertainty/recency posterior와 edge-conditioned Global 입력 |
| Localization | confidence/covariance interface와 sweep 기반 | pose covariance를 candidate rollout risk와 tail risk에 직접 전파 |

목표 열은 연구 계획이며 현재 코드가 이미 제공한다고 주장하지 않는다. 기능을
추가할 때마다 config flag, fingerprint, replay/checkpoint schema, telemetry, unit test와
독립 ablation을 함께 추가해야 한다.

## 3. Local 목표 알고리즘

### 3.1 Multi-task risk와 uncertainty

scalar risk 하나로 조기에 압축하지 않고, 각 action의 해석 가능한 실행 위험을
다음처럼 예측한다.

$$
f_\psi(s,a)=
[P_{collision}, P_{unrecoverable}, \hat d_{min}, \widehat{TTC},
\hat m_{stop}]
$$

`K=3~5`개의 독립 head 또는 ensemble을 사용해 각 risk factor의 평균과 epistemic
spread를 추정한다.

$$
\mu_R(s,a)=\frac{1}{K}\sum_{k=1}^{K}R_k(s,a), \qquad
\sigma_R(s,a)=\sqrt{\frac{1}{K}\sum_{k=1}^{K}
(R_k(s,a)-\mu_R(s,a))^2}
$$

선택과 actor penalty에는 보수적 위험을 사용한다.

$$
R^+(s,a)=\mu_R(s,a)+\beta\sigma_R(s,a)
$$

`sigma`를 곧바로 calibrated probability로 부르지 않는다. held-out ID/OOD 데이터에서
ensemble disagreement와 실제 오차의 관계, ECE/Brier/FNR을 확인한 뒤에만 uncertainty
주장을 한다. collision, unrecoverable, clearance, TTC, stopping margin은 각자의
label/censoring/단위를 유지하고, 최종 scalar 조합 규칙도 artifact에 기록한다.

### 3.2 Physics + learned residual dynamics

전체 pose dynamics를 black-box로 대체하지 않고 nominal Hunter 모델의 오차만
학습한다.

$$
x_{t+1}=f_{Hunter}(x_t,u_t)+g_{\phi_k}(x_t,u_t,h_t),
\quad k=1,\ldots,K
$$

우선 residual target은 command와 실제 응답 사이의
`[delta_v, delta_yaw_rate, delta_steering]`으로 제한한다. 입력 history에는 현재/이전
command, measured velocity/yaw rate/steering, command age를 포함할 수 있다. steering
lag, tire slip, dead zone, acceleration lag, friction, payload와 command latency가 주요
오차원이다.

동일 action을 모든 residual member로 rollout해 trajectory/risk distribution을 얻는다.
ensemble 간 미래 궤적이 크게 벌어지는 영역은 dynamics OOD로 간주한다. 비교는
반드시 `nominal rollout` 대 `single residual` 대 `residual ensemble`로 분리한다.

### 3.3 Progress-preserving counterfactual policy improvement

actor action을 $a=\pi_\theta(s)$, 주변 후보를 다음처럼 정의한다.

$$
\mathcal C(s,a)=\{a^{\kappa+},a^{\kappa-},a^{v-},a^{L-},a^{stop},\ldots\}
$$

후보별 progress $P$, conservative risk $R^+$, feasibility $F$와 uncertainty
$U$를 평가한다. 목표 counterfactual은 단순 최저-risk 또는 정지가 아니라 현재
progress의 일부를 보존하는 안전한 대안이다.

$$
a_{cf}=\arg\min_{a'\in\mathcal C(s,a)}R^+(s,a')
$$

subject to

$$
P(s,a')\ge\rho P(s,a),\qquad F(s,a')=1,
\qquad U(s,a')\le\epsilon_u
$$

활성 조건은 최소한 다음을 모두 만족해야 한다.

$$
R^+(s,a)-R^+(s,a_{cf})>\Delta_R,
\qquad P(s,a_{cf})\ge\rho P(s,a)
$$

목표 actor loss는 다음 형태다.

$$
\mathcal L_\pi = \mathcal L_{TQC}
+\lambda_r R^+(s,\pi_\theta(s))
+\lambda_{cf}w(s)\|\pi_\theta(s)-\operatorname{sg}(a_{cf})\|^2
$$

현재 구현은 마지막 imitation 항을 아직 제공하지 않으며, candidate-supervised risk
learning과 safer-margin reweighting까지만 제공한다. 새 objective는 기존 방식과 별도
flag/profile로 추가하고, `risk penalty only`, `random candidates`, `structured
candidate supervision`, `direct counterfactual target`을 분리 비교한다.

### 3.4 High-uncertainty 행동

counterfactual target의 uncertainty가 높으면 actor를 그 target으로 학습시키지 않는다.
이때의 동작은 다음 중 명시적으로 설정하고 telemetry에 남긴다.

- conservative slowdown 또는 stop;
- mandatory safety-guard intervention;
- uncertainty buffer에 transition 저장;
- residual/risk model용 추가 데이터 수집.

unknown을 `risk=0`으로 해석하거나 불확실한 후보를 pseudo-label로 사용하지 않는다.

## 4. Global 목표 알고리즘

### 4.1 Local capability distribution

각 Global candidate $c_i$에 대해 단일 risk가 아니라 다음 capability vector를
계산한다.

$$
C_i=[P_{success}, E[R], U[R], E[progress], E[T_{execute}],
P_{stop}, P_{unrecoverable}]
$$

모든 candidate는 하나의 immutable Local temporal/context snapshot을 공유해야 한다.
timeout, error, non-finite prediction은 별도 validity/reason으로 기록하고 unknown으로
마스킹한다.

현재 evaluator artifact는 validity/reason을 기록하지만 candidate tensor에는 별도
validity 열이 없고 실패한 `predicted_action_risk`/`progress_preserving`을 0으로
채운다. 따라서 위 문장은 **목표 계약**이다. 스키마 변경 전 formal 결과는 raw
action/risk fallback count가 모두 0인 gate를 요구한다. 현재 `fallback_rate`는
decision/candidate 단위가 섞여 있으므로 acceptance에 사용하지 않으며, 그 0을
저위험 관측으로 해석하지 않는다.

Global value는 다음처럼 해석한다.

$$
z_m=Encoder_{map}(M),\quad z_h=Encoder_{memory}(H),\quad
z_c=Encoder_{candidate}(C_i)
$$

$$
Q(s,c_i)=MLP([z_m,z_h,z_c,z_{state}])
$$

현재 masked Dueling Double DQN을 유지한다. 알고리즘 교체보다 capability feature의
정확성, calibration과 원인 분리 ablation을 우선한다.

### 4.2 Experience-Aware Topological Memory

현재 `TopologicalGraph`는 이미 edge별 traversal/success/failure count, path length,
elapsed time, mean/max risk, last direction과 blocked 상태를 저장한다. 다음 연구
단계는 이 정보를 새로 발명하는 것이 아니라 Global observation이 직접 소비할 수
있게 노출하고, uncertainty/recency와 smoothed posterior를 추가하는 것이다. 목표
확장 예시는 다음과 같다.

$$
e_{ij}=[N_{visit},N_{success},N_{fail},\bar R,\bar T,\bar U,t_{last}]
$$

성공 확률은 작은 표본에서 0/1로 붕괴하지 않도록 예를 들어 Beta posterior mean을
사용할 수 있다.

$$
P_{success}(e)=\frac{N_{success}+\alpha}
{N_{visit}+\alpha+\beta}
$$

단순 방문 여부가 아니라 “이 edge에서 frozen Local이 얼마나 자주, 얼마나
위험하게 실패했는가”를 Global observation에 전달한다. 시간 decay, 환경 변화,
동일 edge 정의와 loop-closure 시 통계 병합 규칙을 먼저 정해야 한다.

### 4.3 Localization uncertainty propagation

scalar confidence만 전달하지 않고 pose distribution을 사용한다.

$$
x\sim\mathcal N(\hat x,\Sigma_x)
$$

동일 candidate를 pose sample과 residual-dynamics ensemble 전체에서 평가해 collision,
clearance와 risk distribution을 얻는다. 좁은 corridor에서 covariance가 커질수록
자연스럽게 안전 여유가 줄어야 한다. 평균 risk 외에 quantile 또는
$CVaR_\alpha$를 비교하되, 실제 pose-error injection이 없는 confidence-only sweep을
localization-robustness 증거로 사용하지 않는다.

## 5. 우선순위

| 우선순위 | 개발 내용 | 논문 기여도 | 현재 판정 |
|---|---|---:|---|
| P0 | 기존 feasibility/evaluator/formal gate 연결과 baseline freeze | 매우 높음 | release·Local live smoke 완료; promoted Local 부재로 Global smoke만 보류 |
| P1 | Counterfactual learning 정식화 | 매우 높음 | 목표 objective 미구현 |
| P1 | Risk -> multi-task Risk + calibrated uncertainty | 매우 높음 | 미구현 |
| P1 | Physics + learned residual dynamics ensemble | 매우 높음 | system-ID 도구만 존재, residual 미구현 |
| P1 | dynamics/sensor/localization OOD 평가 | 매우 높음 | 일부 config/driver 존재, formal 실행 미수행 |
| P2 | Global Local-capability distribution | 매우 높음 | scalar/feasibility 기반에서 확장 필요 |
| P2 | localization covariance의 risk 전파 | 높음 | covariance interface는 있으나 rollout 전파 미구현 |
| P2 | topological memory의 experience graph 확장 | 높음 | 기본 edge 통계 저장됨; observation 노출·uncertainty/recency posterior 미구현 |
| P3 | dynamic-obstacle 연구 확장 | 높음 | 핵심 Local/Global 주장 이후 수행 |

## 6. 단계별 개발 순서

### Phase 0 — 현재 시스템 동결

- [x] 전체 회귀 2,093 pytest / 2,098 colcon checks와 29/29 profile validation.
- [x] 실제 Gazebo Local reset→step/update→save→resume bounded smoke.
- [x] legacy/failed checkpoint formal 배제 및 Global preflight fail-closed 확인.
- [x] corrected baseline source를 annotated tag
  `hunter-kinodynamic-rl-r0-20260902`로 동결.
- [ ] accepted/promoted Local이 생긴 후 Global mission→option/update→save→resume
  bounded smoke를 마저 수행.

Global smoke를 위해 legacy/smoke Local을 강제 승격시키지 않는다. 남은
R0 live gate는 Phase 1의 formal Local benchmark/promotion 후 닫힌다.

### Phase 1 — 현재 Local baseline 완성

현 구현의 `L0~L5`를 여러 training seed로 학습·평가해 이후 모든 확장의 기준점을
만든다. risk calibration과 raw/guarded action 분리도 이 단계부터 기록한다.

### Phase 2 — Risk uncertainty

multi-task risk ensemble, calibration artifact, conservative risk와 uncertainty gate를
추가하고 ID/OOD 성능을 평가한다.

### Phase 3 — Residual dynamics

Gazebo mismatch와 실제 Hunter system-ID 데이터로 residual을 학습한다. nominal,
single residual, ensemble rollout의 one-step/multi-step/risk 오차를 비교한다.

### Phase 4 — Local paper completion

progress-preserving counterfactual target과 uncertainty-aware abstention을 결합한다.
Local ablation, calibration/OOD, safety attribution과 실제 Hunter SE 결과를 완료한다.

### Phase 5 — Local checkpoint freeze

사전 등록된 promotion 기준을 통과한 최선의 Local checkpoint를 immutable generation으로
고정한다. 이후 Global 실험 중 Local weight나 normalization을 변경하지 않는다.

### Phase 6 — Capability-aware Global navigation

capability distribution과 experience-aware edge memory를 추가하고 `G0~G8`을 독립
학습한다.

### Phase 7 — GPS-denied uncertainty

실제 pose-error를 갖는 localization backend/simulator를 준비하고 covariance를 candidate
risk에 전파한다. 동일 mission에서 ideal/noisy/drifting/LiDAR-odometry 조건을 비교한다.

## 7. Ablation ladder

### Local

| Label | 구성 |
|---|---|
| L0 | direct-control SAC/TQC baseline |
| L1 | `[kappa, v_ref, L]` trajectory action |
| L2 | L1 + temporal observation |
| L3 | L2 + supervised scalar/multi-task risk (actor penalty 없음) |
| L4 | L3 + risk actor penalty |
| L5 | L4 + 현재 structured counterfactual candidate supervision/weighting |
| L6 | L5 + risk ensemble uncertainty |
| L7 | L6 + learned residual dynamics |
| L8 | L7 + residual ensemble uncertainty + uncertainty-gated direct counterfactual target |

기존 A–F profile은 대략 L0~L5의 engineering lineage를 제공하지만, direct-control,
fixed-L, path representation 비교까지 자동으로 충족하지는 않는다. 새 label을 기존
profile 이름에 소급 적용하지 않는다.

### Global

| Label | 구성 |
|---|---|
| G0 | frozen Local only |
| G1 | online partial map |
| G2 | G1 + visited channel |
| G3 | G2 + topology/dead-end memory |
| G4 | G3 + Local `P_success` |
| G5 | G4 + Local expected risk |
| G6 | G5 + Local risk uncertainty |
| G7 | G6 + localization-aware candidate risk |
| G8 | G7 + full experience-aware capability model |

각 learned row는 고유 fingerprint와 독립 학습 checkpoint를 가져야 한다. oracle
feasibility/risk와 learned version, classical exploration/planning baseline을 함께 둔다.

## 8. OOD와 안전 평가 원칙

training randomization과 test-only OOD 범위를 분리한다. 최소 축은 다음과 같다.

- tire friction, robot mass/payload;
- steering delay/rate/offset/dead zone;
- acceleration/braking response와 command latency;
- LiDAR range noise/dropout/frame dropout;
- odometry noise, yaw bias, pose covariance와 accumulated drift;
- held-out topology/layout family와 real floor plan.

안전 결과는 다음 네 조건을 같은 manifest에서 분리한다.

1. Raw Actor;
2. Raw Actor + Counterfactual learning;
3. Actor + Guard;
4. Counterfactual Actor + Guard.

collision rate뿐 아니라 unsafe proposal rate, guard intervention rate, stop/failure,
recovery와 progress를 보고한다. Risk model은 AUROC, AUPRC, Brier, ECE, reliability,
false-negative rate, uncertainty-error correlation과 OOD calibration을 별도 평가한다.

## 9. 이번 연구 범위에서 보류할 것

- TQC를 최신 RL 알고리즘으로 교체하는 작업;
- diffusion policy;
- VLM/VLA 또는 RGB semantic navigation;
- topology가 있다는 이유만으로 GNN을 도입하는 작업;
- risk를 다시 reward shaping 항으로 다수 섞는 작업.

이 항목들은 핵심 가설을 더 잘 검증한다는 선행 증거가 있을 때 후속 연구로 다룬다.

## 10. 문서별 책임

| 문서 | 책임 |
|---|---|
| `README.md` | 패키지 개요, 현재 상태, 연구 방향 진입점 |
| `CURRENT_STATUS.md` | 구현/검증의 최신 사실과 다음 게이트 |
| `ARCHITECTURE.md` | 현재 및 목표 알고리즘·데이터 계약 |
| `RESEARCH_PROTOCOL.md` | 가설, ablation, 통계와 claim 규칙 |
| `BENCHMARK.md` | dataset, artifact, metric과 OOD 실행 계약 |
| `SIM2REAL.md` | system-ID, residual dynamics, 실차 안전/검증 순서 |
| `HIERARCHICAL_NAVIGATION_IMPLEMENTATION_PLAN.md` | Global 구현 단계와 의존성 |
| `RUNBOOK_HIERARCHICAL_NAVIGATION.md` | 현재 구현에서 실제 실행 가능한 명령과 판정법 |
