# Current Status

이 문서는 `hunter_kinodynamic_rl`의 최신 구현·검증 상태를 요약하는 정본이다.
설계 목표는 `hunter_se_unknown_gps_denied_hierarchical_navigation_detailed_spec.txt`,
구현 순서와 완료 기준은 `HIERARCHICAL_NAVIGATION_IMPLEMENTATION_PLAN.md`, 실제 실행
명령은 `RUNBOOK_HIERARCHICAL_NAVIGATION.md`, 향후 연구개발 방향은
`RESEARCH_ROADMAP.md`를 따른다. `docs/verification/`과 과거
delivery/completion report는 작성 당시의 증거를 보존하는 historical artifact이며,
현재 상태 판정에는 이 문서를 우선한다.

기준일: 2026-09-02 (`hunter-kinodynamic-rl-r0-20260902` release freeze)

## 한 줄 판정

코드 차단 결함을 닫고 전체 회귀·프로필 검증과 실제 Gazebo Local
reset→step/update→save→resume를 통과한 corrected baseline을 release로
동결했다. Global live smoke는 현재 `local_frozen`이 legacy/unpromoted임을
preflight가 정상적으로 차단해 실행하지 않았다. 이를 우회하지 말고
Local L0–L5 정식 학습·benchmark·promotion 후 Global smoke를 수행한다.
E/F/G candidate validity 채널, 정식 학습/A–G, 실차 증거는 여전히 open이다.

## 단계별 상태

| Phase | 현재 판정 | 정식 완료에 필요한 핵심 |
|---|---|---|
| 1 | 최신 release 회귀·Local live save/resume 완료 | 없음 |
| 2 | 학습 코드 구현 + 좌표계/benchmark/promotion 결함 수정 완료, 연구 검증(수렴/성능) 미착수 | 새 Local 학습·formal benchmark·promotion 실행 |
| 3 | world/reset/sensor 구성요소 구현 + Local live reset/step 재검증 | long-horizon evidence runner 실행, 원시 증거 |
| 4 | Global 학습 인프라 구현 + formal A/B 게이트(mode/budget/promoted-Local/total_optimizer_updates) 강화 완료, 정책 미검증 | 유효한 promoted Local, 새 Global 정식 학습, formal A/B 실행 |
| 5 | B/C 분리, evaluator 배선, temporal-context 오염/backtrack 하드코딩 수정 완료; candidate validity 입력 채널은 없음 | validity schema 또는 엄격한 fallback gate, A–G 각 label 정식 학습·formal suite |
| 6 | localization sweep live driver 구현 완료(HONEST LIMITATION 있음), rosbag dry-run 재설계 완료 | live sweep/rosbag dry-run/실차 실제 실행 |

## 연구개발 목표와 현재 gap

Local의 최종 연구 목표는 다음 결합이다.

```text
Kinodynamic Policy + Uncertainty-Calibrated Multi-Task Risk
                   + Physics/Residual Dynamics Ensemble
                   + Progress-Preserving Counterfactual Policy Improvement
```

Global의 후속 목표는 다음 결합이다.

```text
Online Partial Map + Experience-Aware Topological Memory
                   + Frozen-Local Capability Distribution
                   + Localization-Uncertainty Propagation
```

현재 패키지는 두 연구의 뼈대(`[kappa, v_ref, L]`, nominal rollout, scalar
risk critic, structured candidate evaluation, partial/visited map, topology,
Global DDQN)를 갖췄다. 다음 요소는 **아직 구현 완료가 아니다**.

| 목표 요소 | 현재 가능한 것 | 남은 핵심 |
|---|---|---|
| Risk uncertainty | scalar `RiskCritic`, rollout label | 3~5 head/ensemble, multi-task factor, calibration, conservative risk |
| Residual dynamics | nominal Hunter model, system-ID 도구 | residual dataset/schema/model/ensemble과 multi-step 검증 |
| 정식 counterfactual PI | candidate supervision, safer-margin actor weighting | progress/feasibility/uncertainty 제약, 직접 actor target, abstention |
| Local capability distribution | geometry/feasibility/risk feature | success/risk uncertainty/progress/time/stop/unrecoverable distribution |
| Experience graph | edge별 traversal/success/failure, path length, elapsed time, mean/max risk, direction/blocked 저장; 관측에는 일부 요약만 사용 | uncertainty/recency posterior와 edge-conditioned Global 입력 |
| Localization risk propagation | covariance/confidence backend와 sweep driver | 실제 pose-error model과 candidate rollout/CVaR 전파 |

목표 수식, ablation label과 구현 순서는 `RESEARCH_ROADMAP.md`가 정본이다.

## 장시간 학습 전 차단 항목 — 상태: 전부 CLOSED (코드+회귀 테스트)

1. **[CLOSED]** 계층 경로의 Local observation 좌표계 통일 —
   `LocalPolicyController.robot_relative_state()`가 유일한 생성 경로가 되어
   robot-relative subgoal에는 항상 origin `RobotState`만 짝지어진다
   (`nodes/hierarchical_navigation_node.py`, `nodes/hierarchical_environment_node.py`,
   `navigation/local_rl/live_gazebo_executor.py`,
   `navigation/hierarchy/local_feasibility_evaluator.py` 4개 호출부 전부 수정).
   회귀: `tests/test_hierarchical_local_observation_frame_contract.py`,
   `tests/test_hierarchical_navigation_node.py::test_control_tick_builds_robot_relative_observation_at_nonorigin_pose`.
2. **[CLOSED]** Local benchmark infeasible 지표 재설계 — `infeasible_goal_rejection`/
   `infeasible_goal_rejection_rate` 제거, `feasible_subgoal_success_rate`(feasible
   조건부) + `infeasible_scenario_count`/`infeasible_false_success_rate`/
   `infeasible_collision_rate`/`infeasible_high_risk_rate`/
   `infeasible_safe_termination_rate`(전부 infeasible 조건부, 0표본이면 `None`)로
   교체. schema_version 1→2, promotion이 구버전 스키마를 거부.
   회귀: `tests/test_local_subgoal_benchmark.py`, `tests/test_local_promotion.py`.
3. **[CLOSED]** benchmark artifact/promotion 신뢰 체인 — manifest/episode 1:1
   pairing·중복 scenario_id·seed 불일치·non-test mode를 `build_local_benchmark_artifact`가
   거부; `EnvironmentClient.get_remote_parameter`로 서버의
   `architecture_fingerprint_sha256`/`local_training_contract_fingerprint_sha256`을
   확인하는 `verify_environment_server_identity` 추가(`environment_node.py`에 후자
   파라미터 신규 노출 — 재빌드 필요); promotion은 caller-supplied manifest를
   신뢰하지 않고 source의 `manifest.json`/`model.pt` SHA를 직접 재검증하며,
   resolved_config로부터 architecture/training-contract fingerprint를 재계산해
   자기 일관성을 확인; 빈 `{}`/tampered promotion marker 거부
   (`validate_promoted_local_checkpoint`); 복사→temp write→atomic rename,
   충돌 시 기존 promoted checkpoint 보존.
   회귀: `tests/test_local_subgoal_benchmark.py`, `tests/test_local_promotion.py`.
4. **[CLOSED]** Phase 5 evaluator temporal 오염 — `FrozenLocalFeasibilityEvaluator`가
   자체 `LocalPolicyController`(및 그 frame stack)를 더 이상 갖지 않고, 실제 제어용
   `LocalPolicyController.snapshot_temporal_context()`가 반환한 frozen
   (lidar_frame, prev_action) 스냅샷을 `capture_context()`로 "한 Global decision당
   1회"만 얻어 모든 candidate가 공유한다 (`evaluate(candidate, pose, context)` 시그니처
   변경). 후보 순서 무관, 4-frame 이력 불변, 실제 prev_action 반영, 실제 제어
   controller는 절대 mutate하지 않음을 모두 테스트로 고정.
   회귀: `tests/test_local_feasibility_evaluator.py`, `tests/test_global_local_feasibility.py`.
5. **[CLOSED: artifact visibility]** evaluator fallback 가시성 — `predict_risk`도 `select_action`과
   동일한 bounded single-flight timeout으로 보호되며, timeout/error/NaN·Inf를
   `risk_timeout`/`risk_error`/`risk_non_finite`로 구분 기록한다. 중복 구현이던
   `navigation/local_rl/feasibility_evaluator.py`
   (미사용 dead code, 동일 좌표계 결함 보유)는 삭제하고
   `navigation/hierarchy/local_feasibility_evaluator.py`를 유일한 production
   구현으로 통합.
   단, 현재 `compute_feasibility_features()`는 실패 시
   `predicted_action_risk=0.0`, `progress_preserving=0.0`으로 채우며 candidate
   tensor에 validity 열이 없다. 즉 artifact 해석에서는 unknown이지만 Global
   network 입력만 보면 알려진 0과 구분되지 않는다. 이는 아래 별도
   research-validity gap으로 남긴다.
   회귀: `tests/test_local_feasibility_evaluator.py`.
6. **[CLOSED]** 전진 전용 Local action vs 후방 BACKTRACK — geometry-only 4개
   feature(rollout_collision/steering_saturation_ratio/min_clearance_norm/
   historical_success_rate)는 endpoint 직접 clearance/occupancy 조회로(단일 arc
   fit 불가능한 radius=0·angle=π를 우회) 계산하고, policy-conditioned 2개
   feature(predicted_action_risk/progress_preserving)는 다른 candidate와 동일하게
   `local_evaluator.evaluate()`를 통과시켜 더 이상 `risk=0`/`progress=True`로
   하드코딩하지 않는다 — 단일 tick 기준 정직한 값(대개 progress_preserving=False)을
   보고한다. Multi-tick U-turn 인식 평가는 향후 별도 확장으로 명시.
   회귀: `tests/test_global_local_feasibility.py::test_fallback_candidate_reports_honest_geometry_never_fabricated_progress`.
7. **[CLOSED]** Local promotion tag 통일 — `hierarchical_phase4.yaml`의
   `local_checkpoint_name`을 `"best"`→`"final"`로 변경, `evaluation.local_promotion.DEFAULT_PROMOTION_TAG`
   상수를 promotion 기본값으로 사용. 모든 phase4/phase5_{a..g} profile이 동일 tag를
   가리키는지 테스트로 고정.
   회귀: `tests/test_hierarchical_ablation_profiles.py::test_every_profile_uses_the_canonical_local_checkpoint_tag_including_phase4`.

## 평가 인프라 — 이전 "남은 조건" 상태

- **[CLOSED]** `formal`은 이제 `mode='test'`, `max_options`/`max_local_steps`
  프로필 기본값 유지(override 시 거부), promoted Local(`validate_promoted_local_checkpoint`),
  `total_optimizer_updates >= 1`, 완전한 scenario 결과(불완전 시 `benchmark_kind="incomplete"`)를
  모두 강제한다 — A/B(`run_live_hierarchical_benchmark.py`)와 A–G
  (`run_live_ablation_suite.py`)가 `evaluation/global_checkpoint_validation.py`의
  동일한 strict validator를 공유해 검증 규칙이 갈라지지 않는다.
- **[CLOSED]** A–G formal suite는 요청한 label 중 checkpoint가 없거나 strict
  검증에 실패하면 (`ablation_suite.run_ablation_suite`가) 전체 실행을 실패시킨다 —
  skip은 `benchmark_kind="smoke"`에서만 허용. evaluator fallback telemetry가
  결과 JSON에 label별로 기록된다.
- **[OPEN: formal-result acceptance]** E/F/G feature tensor에는 evaluator
  validity/unknown 열이 아직 없다. 스키마를 확장하기 전에는 formal protocol이
  fallback reason/raw count를 전부 기록하고 하나라도 0이 아니면 artifact를
  불완전으로 판정해야 한다. 현재 `fallback_rate`는 query는 decision 단위인데
  일부 failure count는 candidate 단위라 1을 넘을 수 있으므로 acceptance metric으로
  사용하지 않는다. 단순히 telemetry가 있다는 이유로 입력의 0을 안전·저위험으로
  해석하면 안 된다.
- **[CLOSED]** rosbag dry-run(schema v2)은 필수 topic 존재 + 최소 message 수,
  실제 사용된 topic, decision ≥ 1, `dry_run`/`replay_mode` 활성, actuator publish
  attempt 0건을 모두 성공 조건으로 요구한다. 실제 `rclpy.Publisher`에는 없는
  `.published` 속성을 읽는 대신, `node._cmd_pub`를 계측 wrapper로 교체해 publish
  시도 자체를 카운트한다(실제 publish는 절대 forward하지 않음). bag identity
  hash가 metadata뿐 아니라 실제 파일 바이트도 반영하고, 메시지는 스트리밍으로
  처리한다(전체 bag을 메모리에 올리지 않음).
- **[CLOSED]** live evidence runner(`evaluation/live_evidence_runner.py`)는
  기본 profile을 존재하지 않던 `hierarchical_phase3`에서 `hierarchical_phase4`로
  변경했고, Gazebo를 건드리기 전에 `preflight_live_evidence`(long_horizon_world
  enabled, Local checkpoint promoted+architecture 일치, wall pool capacity는
  config-load 시 이미 검증됨, timeout budget 보고)를 실행한다. wall activation
  완료/robot teleport 완료/sensor freshness wait 시작·완료(경과시간)/각 Local
  control tick(step/time/pose/subgoal/action/command/risk/guard/collision)을
  `LiveGazeboLocalExecutor.bind_mission`의 `on_event`와 `run_option`의 `on_tick`
  hook을 통해 JSONL 이벤트로 기록한다. leftover-process 검사는 이 실행이 직접
  생성한 Gazebo 프로세스 그룹(`pgrep -g <pgid>`)만 검사하며, `launch_gazebo=False`일
  때는 "not_applicable"로 명시한다. stdout/stderr 파일 핸들을 명시적으로 닫는다.
- **[CLOSED]** localization sweep live driver
  (`evaluation/localization_sweep_live_driver.py`) 신규 구현 — 조건마다 독립된
  `LiveGazeboLocalExecutor`를 그 조건 전용 localization backend로 구성해 동일
  fixed scenario manifest/ablation/checkpoint를 실행하고, 결과를 기존
  `aggregate_localization_sweep`에 전달한다. **HONEST LIMITATION**: 현재
  `WheelImuLocalizationBackend`는 실제 twist에 random noise를 주입하지 않고
  covariance/confidence만 성장시키므로(그 모듈 자체의 기존 문서화된 제약),
  이 sweep은 confidence-gated 행동 차이는 비교하지만 실제 위치 drift에 따른
  성능 저하를 정량화하지는 못한다 — noise model 자체를 확장하는 것은 이번
  세션의 범위 밖으로 남겨둔다. `LidarOdomLocalizationBackend`는 실제
  scan-matcher가 이 저장소에 없어 사용하지 않는다.
- **[미실행, 인프라만 준비]** Global preflight에 E/F/G-tier(`feasibility_feedback_enabled`/
  `global_risk_feedback_enabled`) profile이 `live=False`이면 즉시 거부하는 체크,
  structured promotion validator 기반 이유 노출, resume 시 Global checkpoint
  architecture/replay-schema 일치 확인, `ros_gz_interfaces` import 확인을 추가했다.
  Local preflight는 `ros_gz_interfaces` import 확인, `sample_count`/confidence-z
  검증, 고정 tolerance 대신 binomial confidence interval(과거 target=0.15/
  measured=0.0 boundary bug 수정), resume 시 기존 checkpoint의
  training-contract/architecture fingerprint 일치 확인을 추가했다. 실 학습
  시작 경로(`train_kinodynamic_tqc.py::main`)가 `dry_run=False`에서도 자동으로
  preflight를 실행하도록 변경(명시적 `skip_preflight=True`로만 우회 가능, 우회 시
  로그 남김).

## 확인된 구현 개선 (이전 버전에서 이미 확인됨, 유지)

- Phase 5 B/C는 `include_visited_channel`로 구분되고 architecture fingerprint도 다르다.
- E/F/G에서 evaluator 객체 자체가 없으면 학습·평가 경로가 fail-fast한다.
- Global checkpoint의 Local identity 비교, CPU `map_location`, success-only
  time-to-goal, nested Git provenance 경로가 구현돼 있다.

## 2026-09-02 release 검증 결과

- Docker `colcon build --packages-select hunter_kinodynamic_rl --symlink-install`: 성공.
- direct pytest: **2,093 passed**; `colcon test-result`: **2,098 tests,
  0 errors, 0 failures, 0 skipped**.
- `config/profiles/*.yaml`: **29/29 valid**.
- 신규 `smoke_test_arbitrary_subgoal` preflight: full-circle, target infeasible
  fraction 0.15, measured 0.17, `state_dim=328` 통과.
- 실제 Gazebo Local smoke: fresh 60 step에서 replay/critic·actor·risk update,
  checkpoint save 완료. `best` step 53에서 resume해 step 60 final을 새
  generation으로 저장했고 telemetry 7/7 match, timeout 0을 확인했다.
- Global live preflight: **expected fail-closed**. `local_frozen/checkpoints/final`에
  `promotion_manifest.json`이 없어 live Global을 시작하지 않았다.
- smoke 종료 후 Gazebo/ROS process·node 0개를 확인했다.
- 심볼릭 설치에서 `ros2 run` node가 누락되던 실행 권한 결함을 수정하고
  CMake `install(PROGRAMS)` 전체를 검사하는 회귀 테스트를 추가했다.
- release source provenance: annotated tag `hunter-kinodynamic-rl-r0-20260902`.
  실행 세부 사항은 `verification/2026-09-02_stage1_release_freeze.md`.

이 smoke는 배선·저장·재개 증거이며 성능·수렴 증거가 아니다.

## 아직 수행하지 않은 연구 검증

- 현재 full-circle/infeasible 분포로 새 Local TQC 정식 학습
- Local formal benchmark와 유효한 promotion
- promoted Local을 고정한 Global DQN 정식 학습
- trained Global을 사용한 formal A/B
- Phase 5 A–G 독립 학습·formal ablation
- ideal/noisy/drifting localization live sweep (드라이버는 구현됨, 실제 실행은 안 함)
- 실제 mission rosbag dry-run (드라이버는 구현됨, 실제 bag 없음)
- Hunter SE 실차 시험

또한 다음 연구 확장은 구현과 실험 모두 미완료다.

- multi-task risk ensemble의 calibration/OOD 평가;
- physics + learned residual dynamics 및 residual ensemble rollout;
- uncertainty-gated, progress-preserving direct counterfactual actor update;
- frozen Local capability distribution을 사용한 Global 학습;
- 저장된 topological edge 통계를 소비하는 uncertainty/recency posterior;
- pose covariance를 candidate collision/tail risk에 전파하는 평가.

## 다음 실행 순서

1. 현재 구현의 Local `L0~L5`를 각각 5개 이상 seed로 학습하고
   동일 formal benchmark로 평가한다.
2. acceptance gate를 통과한 Local `final`만 `local_frozen`으로 promotion한다.
3. 이 promoted Local로 bounded Global mission/update/save/resume를 수행해
   R0의 남은 live 게이트를 닫는다.
4. risk multi-task/ensemble·calibration/OOD(`L6`) 후 residual dynamics·정식
   counterfactual target(`L7~L8`)을 추가한다.
5. 최종 Local을 immutable하게 동결한 후에만 Global
   capability/memory/localization `G0~G8`을 독립 학습한다.

장시간 Global 학습을 먼저 실행하거나 현재 smoke checkpoint를 연구 baseline으로
승격하지 않는다.

## 권장 연구 범위

첫 논문은 Local 방법에 초점을 둔다. 핵심 주장은 물리적으로 해석 가능한
`[kappa, v_ref, L]` trajectory action, physics+residual ensemble rollout,
calibrated risk uncertainty, progress를 보존하고 uncertainty가 높으면 abstain하는
structured model-based counterfactual policy improvement다. TQC는 기반
알고리즘이며 그 자체를 독창성으로 주장하지 않는다.

Global 연구는 frozen Local의 capability distribution과 experience-aware topological
memory가 반복 dead end와 실행 불가능한 subgoal 선택을 줄이는지를 묻는 후속 연구로
분리한다. localization covariance를 candidate risk에 전파해야 GPS-denied를 단순한
센서 부재가 아니라 uncertainty-aware navigation 기여로 주장할 수 있다.
전체 시스템 논문으로 결합하려면 두 계층 각각의 강한 baseline, 원인 분리 ablation,
큰 환경, localization drift 및 Hunter SE 증거가 먼저 필요하다.

## 용어

- `mapless` 대신 `prior-map-free with online partial mapping`을 사용한다.
- `GPS-denied`는 GPS를 쓰지 않는다는 뜻만이 아니라 localization drift 조건에서의
  성능 곡선이 있을 때 연구 주장으로 사용한다.
- 현재 counterfactual은 causal intervention이 아니라 dynamics rollout으로 후보 action을
  비교하는 `model-based counterfactual action evaluation`이다.
- `L`은 option duration이 아니다. Local policy는 고정 주기로 재계획하며, `L`은
  trajectory preview 길이와 rollout/risk horizon에 영향을 준다.
