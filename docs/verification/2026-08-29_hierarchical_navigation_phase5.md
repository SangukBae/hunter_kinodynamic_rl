# Hierarchical Navigation Phase 5 검증 기록 (2026-08-29)

> **Historical verification artifact.** 당시 source/명령/결과를 보존한다. 최신
> 상태는 [`../CURRENT_STATUS.md`](../CURRENT_STATUS.md)를 따른다.

`docs/HIERARCHICAL_NAVIGATION_IMPLEMENTATION_PLAN.md` 섹션 9(Phase 5: Topological
memory와 Global-Local feedback)와, 사용자 요청이 함께 포함시킨 계획서 섹션
10(Long-Horizon Benchmark / Localization drift / 실차 배포) 항목 일부를 구현했다.
원본 요구사항 기준은 `hunter_se_unknown_gps_denied_hierarchical_navigation_detailed_spec.txt`.

## 0. Round 2 (코드 리뷰 수정, 같은 날짜)

외부 코드 리뷰에서 실제 결함 6건을 발견해 모두 수정했다:

1. **(High) real/dry-run node의 final goal frame 버그**: `hierarchical_navigation_node.py`/
   `hierarchical_environment_node.py`가 `goal_x`/`goal_y`(spec 3.1 기준 "시작 시
   robot frame 상대 목표", 즉 정의상 이미 mission frame 좌표)에
   `mission_frame.odom_to_mission()`을 한 번 더 적용하고 있었다. start pose가
   `(0,0,0)`일 때만 우연히 값이 맞고, 비영점 start pose/yaw에서는 최종 목표가
   회전/이동되어 틀어졌다(`navigation/mission/goal_manager.py`/
   `navigation/ros/mission_map_node.py`의 기존 관례와 명백히 불일치). 두 노드
   모두 `self._final_goal_mission = (goal_x, goal_y)`로 직접 대입하도록 수정.
2. **(High) safety guard clock domain 혼합**: `_on_control_tick`이
   `now=time.monotonic()`을 `action_guard.guard()`에 넘기면서
   `last_sensor_time_sec`/`last_odom_time_sec`에는 ROS 메시지 header stamp
   (`_latest_scan_time`/`_latest_odom_time`, Gazebo 브릿지 sim time 도메인)를
   그대로 넘기고 있었다 -- `is_pose_usable()`/`action_guard.py`가 명시적으로
   요구하는 "같은 clock domain" 계약 위반(`navigation/ros/mission_map_node.py`
   문서화된 과거 실제 버그, `evaluation/nav2_mppi_runner.py` 참조, 와 동일한
   클래스). `_latest_scan_receipt_time`/`_latest_odom_receipt_time`(둘 다
   수신 시점 `time.monotonic()`)을 신설해 safety guard 호출에는 이 값만
   사용하도록 분리했다 -- 기존 메시지 stamp 변수는 `pose_at()`/동기화 용도로
   그대로 유지(mission_map_node.py의 "메시지 stamp만 쓰고 wall/monotonic과
   섞지 않는다" 규칙과 real_policy_node.py의 "watchdog/freshness는
   monotonic receipt time" 규칙을 각각의 올바른 용도에 적용).
3. **(Medium) localization backend가 항상 GazeboOdomLocalizationBackend로
   고정**: `hierarchical_navigation_node.py`가 `profile.localization.backend`
   값과 무관하게 항상 같은 backend를 생성했다. `_build_localization_backend()`
   factory(순수 함수, 단위 테스트 가능)를 추가해 `gazebo_odom`/`odom`/
   `wheel_imu`/`lio`를 실제로 구분 생성하고, `_on_odom()`이 backend 타입에 맞는
   피딩 API(`on_odometry_msg`/`integrate()`/`on_lio_odometry_msg`)로 분기하도록
   수정했다. `wheel_imu`는 이 저장소에 실제 wheel encoder/IMU 드라이버가 없어
   브릿지된 `/odometry`의 twist를 정직하게 문서화된 대체 입력으로 사용하고,
   `lio`는 별도 `lio_odom_topic` 파라미터(LIO-SAM 자체 출력 topic)를 구독한다.
   `lidar_odom`은 실제 scan-matcher가 없어 `LocalizationConfig.backend`의 유효값
   (`SUPPORTED_LOCALIZATION_BACKENDS`)에 여전히 포함하지 않았다 -- 정직한 범위
   제한으로 문서화.
4. **(Medium) BenchmarkScenarioSpec.relative_goal이 world 좌표계 그대로**:
   `long_horizon_benchmark.py`의 `build_fixed_benchmark_manifest()`가
   `world.goal_pose`(generator의 world/odom 좌표계)를 `relative_goal` 필드에
   그대로 넣고 있었다 -- 필드 의미(mission-frame 상대 목표)와 불일치. 임시
   `MissionFrame`을 `world.start_pose`로 초기화해 `odom_to_mission()`으로
   변환한 값을 저장하도록 수정(`run_ablation_mission()`이 world를 재생성할 때
   쓰는 것과 동일한 변환).
5. **(Medium) ablation label과 effective config 불일치**: `run_ablation_mission()`이
   `_ablation_flags(ablation)`으로 계산한 로컬 boolean은 candidate feature
   계산 여부만 결정하고, `build_global_observation()`에는 caller가 넘긴
   `global_cfg`를 그대로 전달했다 -- caller의 config가 label과 다른 flag를
   가지면 (a) label "B"에 Phase5-flag가 켜진 config를 넣으면
   `build_global_observation`이 feature 배열을 요구하며 예외를 던지고, (b)
   label "D"에 flag가 꺼진 config를 넣으면 topology feature가 조용히
   누락되는 두 방향 모두 실패할 수 있었다. `effective_global_rl_config()`/
   `effective_memory_config()`/`effective_feasibility_config()`를 추가해
   `run_ablation_mission()` 진입 시 ablation label로 `global_cfg`/`memory_cfg`/
   `feasibility_cfg`의 flag를 강제 재작성하도록 했다 -- label이 유일한
   source of truth가 됨.
6. **(Low/Medium) 새 topological node의 visit_count가 0으로 시작**:
   `node_manager.py`의 `maybe_create_or_update_node()`가 기존 node에 merge할
   때만 `record_visit()`을 호출하고, 새로 생성한 node(방금 도착해서 만들어진
   node)에는 호출하지 않아 `visit_count=0`으로 남았다 -- merge 케이스와
   비일관. 생성 분기에도 `record_visit()`을 호출하도록 수정(기존 테스트의
   `visit_count == 1` 기대값을 생성+merge 후 `== 2`로 갱신).

신규/갱신 테스트: `test_final_goal_fixed_correctly_for_nonzero_start_pose_and_yaw`,
`test_final_goal_is_a_noop_transform_only_at_zero_start_pose`,
`test_scan_and_odom_receipt_time_tracked_separately_from_message_stamp`,
`test_control_tick_guard_call_uses_monotonic_receipt_time_not_message_stamp`,
`test_localization_backend_factory_selects_matching_type`,
`test_localization_backend_factory_rejects_unsupported_backend`,
`test_wheel_imu_backend_dispatch_dead_reckons_from_odometry_twist`
(`tests/test_hierarchical_navigation_node.py`); `test_effective_global_rl_config_forces_flags_from_label_not_caller`,
`test_effective_memory_and_feasibility_config_follow_the_same_label`,
`test_run_ablation_mission_b_never_raises_even_with_phase5_flagged_base_config`,
`test_run_ablation_mission_d_builds_topology_features_even_with_phase4_shaped_base_config`,
`test_relative_goal_is_mission_frame_relative_not_raw_world_pose`
(`tests/test_long_horizon_benchmark.py`).

Round 2 이후 Docker 전체 회귀 테스트(`python3 -m pytest -q`) **1785개**, `colcon test`
**1790개**가 오류·실패·스킵 없이 통과했고, 26/26 profile이 유효하다.

## 0.1 Round 3 (코드 리뷰 수정, 같은 날짜)

Round 2 이후 다시 리뷰에서 실제 결함 2건을 발견해 모두 수정했다:

1. **(Medium) `_localization_valid()`가 여전히 stale odom을 감지하지 못함**:
   Round 2에서 safety guard(물리 command)는 receipt time으로 고쳤지만,
   `_localization_valid()`는 여전히 `now_sec=self._latest_odom_time`으로
   `is_pose_usable()`을 호출하고 있었다 -- pose의 message stamp를 자기 자신과
   비교하는 것과 같아서, odom 수신이 완전히 끊긴 뒤에도 "valid"가 영원히
   True로 남는다. 이 값은 Global action mask(`compute_action_mask`'s
   `localization_valid`)와 `HierarchyCoordinator.activate_next_subgoal`/
   `record_local_tick`의 `localization_confidence`에도 쓰이므로, stale
   localization이 물리 command는 막혀도 **Global subgoal 선택/activation은
   막지 못하는** 문제로 이어졌다(spec의 "stale/invalid localization이면
   invalid subgoal/replanning/safe stop" 계약 위반). `_localization_valid()`에
   `_latest_odom_receipt_time`(monotonic) 기반의 별도 freshness 체크를
   추가하고(message-stamp 도메인은 `is_pose_usable`의 valid/finite/confidence
   체크에만 계속 사용, 도메인을 섞지 않음), 새 `_effective_localization_confidence()`
   헬퍼(stale이면 무조건 `0.0`)를 만들어 `activate_next_subgoal`/
   `record_local_tick`에 넘기는 confidence 값 자체를 고쳤다 -- Phase 2에서
   이미 검증된 `HierarchyCoordinator`의 degraded-localization state
   machine(활성 subgoal 취소, `stop_required`, 회복 전까지 어떤 subgoal도
   activate 거부)을 그대로 재사용하는 방식이라 새 병렬 게이트를 만들지
   않았다. 동일한 결함이 `hierarchical_environment_node.py`에도 있어 같은
   방식으로 함께 수정했다(receipt-time 추적 자체가 없었으므로 새로 추가).
2. **(Low/Medium) benchmark manifest hash가 실제로 검증되지 않음**:
   `BenchmarkScenarioSpec.content_hash`/`occupancy_hash`가 "regeneration
   검증용"이라고 문서화돼 있었지만 `run_ablation_mission()`은 world를
   재생성한 뒤 이 값들을 실제로 비교하지 않았다 -- generator나
   `long_horizon_world` config가 바뀐 뒤 오래된 manifest로 평가해도 조용히
   다른 world를 돌릴 수 있었다. `_verify_scenario_regeneration()`을 추가해
   `run_ablation_mission()`이 world를 재생성한 직후 두 hash를 모두
   재계산·비교하고, 하나라도 불일치하면 즉시 `ValueError`로 fail-fast하도록
   했다(`build_fixed_benchmark_manifest()`의 content hash 계산도 공용
   `_content_hash()` 헬퍼로 통일해 두 곳이 절대 다른 공식을 쓰지 않도록 함).

신규 테스트: `test_localization_valid_detects_staleness_after_odom_receipt_stops`,
`test_localization_valid_true_for_fresh_receipt`,
`test_localization_valid_false_when_receipt_never_recorded`,
`test_effective_localization_confidence_forced_to_zero_when_stale`,
`test_effective_localization_confidence_passes_through_when_fresh`,
`test_maybe_select_next_subgoal_blocks_global_decision_when_odom_receipt_stale`,
`test_maybe_select_next_subgoal_activates_normally_when_localization_fresh`
(`tests/test_hierarchical_navigation_node.py`);
`test_run_ablation_mission_rejects_a_manifest_whose_hash_no_longer_matches`,
`test_run_ablation_mission_rejects_a_manifest_from_a_different_world_config`
(`tests/test_long_horizon_benchmark.py`). 기존
`test_mission_starts_and_first_subgoal_activates`도 receipt time을 명시적으로
fresh하게 설정하도록 갱신(이제 receipt-time을 전혀 설정하지 않은 테스트는
"never received odom"과 동일하게 stale로 취급되는 것이 의도된 동작).

Round 3 이후 Docker 전체 회귀 테스트(`python3 -m pytest -q`) **1794개**, `colcon test`
**1799개**가 오류·실패·스킵 없이 통과했고, 26/26 profile이 유효하다.

## 0.2 Round 4 (코드 리뷰 수정, 같은 날짜)

Round 3에서 `_localization_valid()`/`_effective_localization_confidence()`를
고쳤지만, 리뷰에서 behavioral gap 1건을 추가로 발견했다:

- **(Medium) stale localization이 "activation"은 막아도 "enqueue"는 막지 못함**:
  `_maybe_select_next_subgoal()`이 `localization_confidence`를 0으로
  계산한 뒤에도 여전히 Global observation을 만들고 candidate를 선택해
  `coordinator.enqueue_subgoal()`로 큐에 넣고 있었다. 그 다음
  `activate_next_subgoal()`이 degraded 상태라 activation은 거부하지만,
  이 함수는 degraded일 때 `while self._queue:` 루프에 진입하지 않으므로
  큐를 pop하지 않는다 -- 즉 odom이 끊긴 동안 매 tick stale pose 기준
  subgoal이 큐에 계속 쌓이고, localization이 회복되면 `activate_next_subgoal`이
  큐를 FIFO(`popleft`)로 소비하므로 **가장 오래된(가장 stale한) candidate가
  먼저 activate**될 수 있었다. `hierarchical_environment_node.py`도 동일
  구조였다. `_maybe_select_next_subgoal()` 시작부에 `_localization_valid()`
  체크를 추가해, stale이면 observation 생성/action 선택/enqueue를 전부
  건너뛰고 `coordinator.activate_next_subgoal(..., localization_confidence=<forced 0>)`
  만 호출(이 시점엔 큐가 항상 비어 있으므로 pop되는 것 없이 degraded gate만
  갱신)한 뒤 `False`를 반환하도록 수정했다 -- localization이 살아있을
  때만 정상적으로 observation을 만들고 candidate를 큐에 넣는다. 두 노드
  모두 동일하게 수정.

신규 테스트: `test_maybe_select_next_subgoal_never_enqueues_while_stale_across_repeated_ticks`
(반복된 stale tick 동안 `pending_subgoal_count == 0` 유지 확인),
`test_maybe_select_next_subgoal_activates_a_fresh_candidate_after_recovery_never_a_stale_one`
(stale 기간 이후 회복 시 activate되는 subgoal이 항상 CURRENT pose 기준으로
새로 계산된 것이며 큐에 남아있던 stale candidate가 아님을 확인)
(둘 다 `tests/test_hierarchical_navigation_node.py`).

Round 4 이후 Docker 전체 회귀 테스트(`python3 -m pytest -q`) **1796개**, `colcon test`
**1801개**가 오류·실패·스킵 없이 통과했고, 26/26 profile이 유효하다.

## 1. 구현 범위 (Round 1)

### 1.1 Topological memory (`navigation/memory/`)
- `topological_graph.py`: `TopologicalGraph`/`TopoNode`/`TopoEdge`. Node id와 pose를
  분리한 `update_node_pose()`(loop-closure 대비), undirected edge(반복 traversal의
  path_length/risk를 running mean/max로 누적), `find_nearest_node()`(dedup),
  `is_connected()`(loop connectivity 검증), `to_fixed_tensor()`(fixed-length node
  tensor + validity mask, 최근 방문 N개), `to_dict()/from_dict()`(체크포인트/기록용
  직렬화).
- `route_history.py`: ENTER/REVISIT/BACKTRACK 분류, `branch_since()`(junction 이후
  탐색한 branch 추출) — junction→dead end→junction 복귀→새 branch 흐름을 그대로
  보존.
- `dead_end_detector.py`: 5개 evidence(낮은 free-direction degree, frontier 부재,
  progress 정체, 반복 emergency stop, 좁아진 candidate mask) 투표제 판정.
  **caller 계약**: `verdict()`는 `graph.mark_dead_end()` 호출 **이전**에 실행해야
  first-vs-repeated가 올바르게 분류된다(그래프의 사전 상태로 판정).
- `node_manager.py`: 거리/heading/junction/subgoal-reached/subgoal-failed/dead-end
  이벤트 기반 node 생성, `node_merge_radius_m` 내 기존 node로 dedup(중복 생성 방지),
  route_history와 edge 갱신을 함께 처리. `compute_candidate_topology_features()`로
  candidate별 repeated-deadend flag/branch visit count를 계산.

### 1.2 Global-Local feasibility feedback (`navigation/hierarchy/feasibility.py`)
- Geometry-only feature(rollout collision, steering saturation ratio, clearance,
  historical success rate)는 `action_mask.py`의 Ackermann arc 계산을 재사용해
  PartialMap만으로 계산.
- Policy-conditioned feature(predicted action risk, progress-preserving)는
  `LocalFeasibilityEvaluator` Protocol을 통해 주입 — candidate를 먼저 local subgoal로
  conditioning한 뒤 그 action을 평가하는 계약(`[radius, angle]`을 risk critic에 직접
  넣지 않음). 이 세션의 ROS-free 학습 loop는 실제 local policy가 없으므로
  `local_evaluator=None`으로 0-fill(정직하게 문서화, Phase 4의 동일한 한계 계승).
- `global_rl/feasibility_predictor.py`: plan 9.7의 learned predictor(선택적,
  기본 미사용) — local map crop + candidate + vehicle state → success
  probability/expected steps/expected risk. torch-gated.

### 1.3 Global observation/network/replay/reward 확장 (opt-in, config-gated)
- `global_rl/observation.py`: `resolve_map_channel_names()`/`resolve_candidate_feature_names()`가
  `GlobalRLConfig`의 새 ablation flag(`include_failure_channel`,
  `topology_feedback_enabled`, `feasibility_feedback_enabled`,
  `global_risk_feedback_enabled`)에 따라 채널/후보 feature 폭을 결정 — 모두 기본
  False이므로 Phase 4 프로파일은 byte-identical. `GlobalObservation`에
  `node_tensor`/`node_validity_mask` 필드 추가(비활성 시 `(0, N_NODE_FEATURES)`/`(0,)`).
- `global_rl/networks.py`: `MaskedDuelingDQN`이 `max_nodes>0`일 때만 node encoder를
  구성(masked-mean pooling, GNN은 도입하지 않음 — plan 9.5) — `max_nodes=0`(기본)이면
  Phase 4와 파라미터 수/shape가 완전히 동일함을
  `tests/test_hierarchical_checkpoint_compatibility.py`로 검증.
- `global_rl/replay.py`/`replay_schema.py`: SCHEMA_VERSION 2→3. 채널/후보 feature
  이름을 버퍼 생성자 인자로 받도록 변경(고정 상수 대신 profile별 resolved 값), node
  tensor 저장 추가. `load()`는 caller가 전달한 expected 이름과 저장된 이름을 비교 —
  Phase 4 기본값이 기본 expected 값이라 기존 파일 검증 의미는 그대로 유지.
- `global_rl/reward.py`: `R_predicted_risk`(신규, `predicted_risk_penalty_scale` 기본
  0.0) — 선택된 candidate의 predicted risk를 결정 시점에 price. 기존 항은 변경 없음.

### 1.4 학습 루프/노드 배선
- `training/train_hierarchical_dqn.py`: `memory_config_from()`/`feasibility_config_from()`
  builder 추가, `HierarchicalTrainingLoop`에 `memory_cfg`/`feasibility_cfg` optional
  파라미터, mission마다 `TopologicalGraph`/`TopologicalNodeManager`/`DeadEndDetector`
  생성, dead-end verdict(first-vs-repeated)를 `repeated_deadend` reward 입력으로
  연결(memory 비활성 시 Phase 4의 기존 endpoint-failure-count 휴리스틱으로 fallback),
  `checkpoint_meta()`에 topology/feasibility/risk feature schema + local checkpoint
  hash 기록.
- `nodes/hierarchical_train_node.py`: resolved map/candidate 차원과 `max_nodes`로
  `GlobalDQNAgent` 구성, manifest에 `hierarchical_architecture_fingerprint` 기록.
- `nodes/hierarchical_environment_node.py`: Phase 5 ablation flag가 켜진 프로파일은
  명시적으로 거부(이 노드는 Phase 4 관측 계약만 계산 — Phase 5는
  `hierarchical_navigation_node.py` 사용).
- `evaluation/fingerprint.py`: `hierarchical_architecture_fingerprint()`(local-only
  `architecture_fingerprint()`와 완전히 분리된 섹션 집합 — 기존 local checkpoint
  fingerprint 의미를 절대 바꾸지 않음).

### 1.5 Long-Horizon Benchmark + Global metrics (plan 10.3/10.4/9.9)
- `evaluation/global_metrics.py`: Final Goal Success Rate, obstacle-aware Global
  SPL(`long_horizon_solvability.shortest_path_length_m`을 L*로 사용), Excess Path
  Ratio, Time to Goal, Subgoal Success Rate, Revisit Ratio, Dead-End/Repeated
  Dead-End Entries, Backtracking Distance, Explored Area, Unnecessary Exploration
  Ratio(observed vs visited cell 비율), Local Planner Failure Count, Global Replan
  Count, Localization Drift Sensitivity — 16개 전부 구현, `*_valid_count` 쌍으로
  undefined/0 구분(`evaluation/metrics.py`의 기존 관례 그대로). 기존 local-only
  `metrics.py`는 변경 없이 유지.
- `evaluation/long_horizon_benchmark.py`: `BenchmarkScenarioSpec`(scenario id,
  seed, content/occupancy hash, GT shortest path, dead-end/loop count, difficulty
  class) + `build_fixed_benchmark_manifest()`(test seed pool에서 결정론적 추출) +
  `run_ablation_mission()`/`run_ablation_benchmark()`(A-G 동일 manifest로 실행).
  ROS-free(`SimplifiedKinematicLocalExecutor` 재사용) — 라이브 Gazebo 통합은
  Phase 4와 동일하게 범위 밖.

### 1.6 Localization backend 확장 (plan 4/10.5)
- `navigation/localization/wheel_imu_backend.py`: dead-reckoning + 누적
  covariance-trace 기반 confidence 감쇠(정직한 단순화 — 실제 EKF/UKF 아님, docstring에
  명시).
- `navigation/localization/lidar_odom_backend.py`: 외부 scan-matcher의 relative
  transform + match_quality를 받아 pose 합성/confidence 매핑(scan matching 알고리즘
  자체는 미구현 — 통합 지점만 제공).
- `navigation/localization/lio_adapter.py`: LIO-SAM류 `nav_msgs/Odometry` 파서
  (gazebo_odom_backend.py와 동일한 파싱 로직 재사용).
- 셋 다 `LocalizationBackend` Protocol만 구현 — navigation core는 backend 종류를
  전혀 알지 못함(`tests/test_localization_drift_evaluation.py`의
  `test_pose_usable_gate_rejects_low_confidence_regardless_of_backend`로 확인).

### 1.7 실차/rosbag dry-run 노드 (plan 10.6/10.7)
- `nodes/hierarchical_navigation_node.py`: `hierarchical_environment_node.py`의
  Global/Local/mapping/memory 배선 위에 `real_policy_node.py`와 동일한 안전 계층을
  추가 — E-stop 구독/latch, 별도 watchdog thread(명령 staleness마다 STOP
  재발행), local 추론 timeout-boxed 단일 실행(single-flight thread), `dry_run`(전체
  파이프라인 실행하되 `/cmd_vel` 미발행)과 `replay_mode`(dry_run 필수, 미충족 시
  `SystemExit`). 모든 물리 명령은 `env.safety.action_guard.guard()`를 그대로
  거친다(재구현 없음 — `test_hierarchical_navigation_node.py`의
  `test_hierarchical_navigation_node_reuses_action_guard_never_reimplements_it`로
  소스 레벨 검증).

### 1.8 신규 프로파일 (ablation A/B/D/E/F/G)
`config/profiles/hierarchical_phase5_{a,b,d,e,f,g}.yaml`. **B와 C는 이 구현에서
동일하다** — Phase 4의 `visited` map channel이 애초에 별도 토글 뒤에 있지 않았기
때문(재작업 범위가 이 세션에서는 정당화되지 않는다고 판단, 정직하게 문서화). A는
`global_rl.enabled=false`이면서 `long_horizon_world`/`mapping`/`mission`/`hierarchy`는
켜둬 동일 benchmark manifest로 local-only baseline을 비교할 수 있게 했다(final goal을
매 option마다 유일한 subgoal로 재주입).

## 2. 테스트

신규 13개 테스트 파일 전부 작성·통과:

```
test_topological_graph.py, test_topological_node_manager.py, test_route_history.py,
test_dead_end_detector.py, test_global_local_feasibility.py, test_feasibility_predictor.py,
test_global_metrics.py, test_long_horizon_benchmark.py, test_localization_drift_evaluation.py,
test_hierarchical_navigation_node.py, test_hierarchical_real_safety.py,
test_hierarchical_ablation_profiles.py, test_hierarchical_checkpoint_compatibility.py
```

필수 테스트 항목(node/edge 중복 생성 방지, junction→dead end→junction 복귀 history
보존, 첫/반복 dead-end 구분, loop connectivity, candidate feature/mask 정렬, risk
critic에 candidate-conditioned action 입력, graph tensor padding/validity mask,
ablation flag의 observation/network/checkpoint fingerprint 반영, obstacle-aware
SPL, localization noise-free pose 누출 금지, inference timeout 즉시 stop, dry-run에서
cmd 미발행, 기존 local-only 회귀 없음) 전부 커버.

기존 `test_global_observation.py`/`test_global_replay.py`는 `GlobalObservation`/
`GlobalTransition`이 실제로 확장된 계약을 반영하도록 갱신(신규 필드는 기본값을 가지므로
기존 호출부는 무변경으로 통과).

## 3. Docker 검증

컨테이너 `7a2702b311a1`(`DRL_Robot_Path_Planning`), `/root/DRL_Robot_Path_Planning/ros2_ws`.

```
source /opt/ros/humble/setup.bash
colcon build --packages-select hunter_kinodynamic_rl   # 성공 (0.66s, 캐시 히트)
source install/setup.bash
cd src/hunter_kinodynamic_rl && python3 -m pytest -q   # 1773 passed, 0 failed, 0 skipped
colcon test --packages-select hunter_kinodynamic_rl
colcon test-result --test-result-base build/hunter_kinodynamic_rl --verbose
  # Summary: 1778 tests, 0 errors, 0 failures, 0 skipped
```

프로파일 검증: `config/profiles/*.yaml` 26개(기존 20 + 신규 6) 전부
`load_profile()` 통과.

> 컨테이너 안에서 `python3 -m pytest tests` (workspace 미source 상태)를 바로
> 실행하면 ament_flake8/ament_pep257/launch_testing 플러그인 조합이
> `pytest.importorskip`으로 인한 모듈 skip을 만나는 즉시 전체 세션을
> "collected 0 items / 1 skipped"로 잘못 종료하는 pytest 6.2.5 환경 이슈를
> 재현했다(workspace가 아직 build/source되지 않아 `drl_agent_interfaces` import가
> 실제로 실패하면서 처음 발현). `PYTHONPATH`/`install/setup.bash`를 정상 source하면
> (즉 실제 검증 절차대로 build 이후 실행하면) 문제없이 전체 스위트가 수집·실행된다 —
> 위에 기록한 “1773 passed”가 그 결과다. 참고용으로만 기록.

### 3.1 추가 end-to-end smoke test (라이브 Gazebo 없이)

- `HierarchicalTrainingLoop`(ablation G, memory+feasibility+global-risk 모두 활성)로
  3개 mission을 실제로 실행 — replay에 13-column candidate tensor/16-node tensor가
  정확한 shape로 저장되고, `GlobalDQNAgent.train_step()`이 실제 gradient step을
  수행함을 확인.
- `nodes/hierarchical_train_node.train_hierarchical_dqn()`을 ablation D 프로파일로
  실행 — checkpoint(`model.pt`/`manifest.json`/`replay.npz`) 저장 및 manifest에
  `hierarchical_architecture_fingerprint`/`node_feature_names`/`memory_enabled` 등이
  정확히 기록됨을 확인.
- `evaluation.long_horizon_benchmark.run_ablation_benchmark()`를 실제 torch
  `GlobalDQNAgent`(ablation D)와 `agent=None`(ablation A)로 각각 실행 — 두 경우 모두
  `global_metrics.aggregate()`가 유효한(NaN/Inf 없는) 결과를 산출함을 확인.
- `rclpy.init()` 상태에서 `hierarchical_navigation_node` 모듈 import까지 확인. 실제
  `HierarchicalNavigationNode(...)` 전체 생성은 로컬 checkpoint 차원이 맞는 frozen
  local policy가 저장소에 없어(Phase 4와 동일한 기존 한계) 이번 세션에서 실행하지
  못했다 — 이는 Phase 4의 `hierarchical_environment_node.py`도 동일하게 겪는 제한이며,
  `_on_control_tick`/`_maybe_select_next_subgoal`/safety 로직 자체는
  `tests/test_hierarchical_navigation_node.py`/`test_hierarchical_real_safety.py`의
  bare-instance 테스트(`real_policy_node.py` 테스트와 동일한 기법)로 개별 검증했다.

## 4. 남은 제한 (정직하게 기록)

1. **B == C ablation**: Phase 4의 `visited` map channel이 원래 토글 뒤에 있지 않아
   이번 구현에서 두 ablation이 동일한 코드 경로/결과를 만든다. 진짜 구분이 필요하면
   `GlobalRLConfig`에 `include_visited_channel` 같은 flag를 추가하고
   `MAP_CHANNEL_NAMES` 베이스 구성 자체를 config-의존적으로 만드는 후속 작업이
   필요하다(Phase 4의 고정 5채널 계약을 건드리는 작업이라 이번 세션 범위 밖으로 남김).
2. **실제 로컬 정책 없이 feasibility 계산**: `hierarchy/feasibility.py`의
   policy-conditioned feature(predicted action risk, progress-preserving)는
   `LocalFeasibilityEvaluator`가 주입될 때만 실제 값을 가진다. ROS-free 학습
   loop(`SimplifiedKinematicLocalExecutor`)에는 실제 local TQC가 없으므로 이 두
   feature는 0으로 채워진다 — Phase 4가 이미 남긴 "실제 kinodynamic TQC + 라이브
   Gazebo로 이 Global 정책을 검증하지 않았다"는 제한을 그대로 계승한다.
3. **라이브 Gazebo 미검증**: 이번 세션도 Phase 3/4와 동일하게 실행 중인 Gazebo
   인스턴스가 없어 `hierarchical_navigation_node.py`/`hierarchical_environment_node.py`
   를 라이브 시뮬레이션에 대해 실행하지 못했다. rosbag dry-run(plan 10.8 순서의 2단계)
   도 이번 세션에서는 수행하지 않았다(재생할 실제 rosbag이 없음) — `dry_run`/
   `replay_mode` 파라미터 계약과 안전 가드 로직 자체는 bare-instance 테스트로
   검증했다.
4. **Localization drift 평가는 합성 데이터 기준**: `wheel_imu_backend.py`/
   `lidar_odom_backend.py`는 실제 wheel encoder/IMU/scan-matcher 드라이버가 아니라
   dead-reckoning + 성장하는 covariance 모델이다(문서화된 의도적 단순화). 실제
   drift 저하 곡선(plan 10.5)은 이 backend들과 `evaluation.long_horizon_benchmark`를
   연결해 얻을 수 있지만, 이번 세션은 그 연결 지점(각 backend가
   `LocalizationBackend` Protocol을 만족하고, drift sensitivity 계산 함수가
   올바르게 동작함)만 단위 테스트로 확인했다 — 전체 sweep을 실행하지는 않았다.
5. **learned feasibility predictor 미사용**: `feasibility_predictor.py`는 구현·
   테스트되었지만 어떤 프로파일도 기본으로 활성화하지 않는다(plan 9.7의 "heuristic이
   안정화된 뒤" 원칙을 따름).
6. **`hunter_kinodynamic_rl`는 저장소에서 git 추적되지 않는다**(`.gitignore`) —
   이 문서와 모든 코드 변경은 디스크에는 실재하지만 `git log`/`git diff`로는 보이지
   않는다. 이전 Phase 1-4 verification 문서들과 동일한 상황이다.

## 5. 완료 기준 대비 확인

- 기존 local-only test 회귀 없음: 1773/1773(host+Docker) 전부 통과, 실패/스킵 0.
- Global replay/checkpoint manifest에 topology/feasibility/risk feature schema +
  local checkpoint hash 기록: `checkpoint_meta()`/실제 저장된 `hierarchical_metadata.json`으로
  확인.
- ablation profile이 서로 다른 architecture fingerprint로 구분(B/C 제외):
  `test_hierarchical_ablation_profiles.py`로 확인.
- obstacle-aware Global SPL 구현 및 hand-computed 값 일치: `test_global_metrics.py`.
- localization backend 교체 가능(Protocol만 의존): `test_localization_drift_evaluation.py`.
- dry-run/replay mode에서 actuation 완전 차단: `test_hierarchical_real_safety.py`.
