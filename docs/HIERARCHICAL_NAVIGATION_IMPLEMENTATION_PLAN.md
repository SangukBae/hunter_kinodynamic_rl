# Unknown GPS-Denied Hierarchical Navigation 구현 계획

## 1. 문서 목적

이 문서는 `hunter_se_unknown_gps_denied_hierarchical_navigation_detailed_spec.txt`의 요구사항을
현재 `hunter_kinodynamic_rl` 패키지에 구현하기 위한 단계별 작업 계획이다.

최종 시스템의 목표는 다음과 같다.

- 사전 global map과 GPS를 사용하지 않는다.
- 사용자는 출발 시점 로봇 좌표계 기준 상대 목표 `(x, y)`만 지정한다.
- 로봇은 LiDAR로 관측한 영역만 online partial map에 누적한다.
- `FREE`, `OCCUPIED`, `UNKNOWN`을 명시적으로 구분한다.
- visited map과 topological memory로 지나간 경로와 실패한 branch를 기억한다.
- High-Level RL은 장거리 탐색 subgoal을 선택한다.
- 기존 Low-Level kinodynamic risk-aware TQC는 선택된 subgoal을 안전하게 추종한다.
- Hunter SE의 Ackermann 제약과 localization drift를 고려해 unseen environment의 최종 목표까지 이동한다.

이 문서의 범위는 총 6개 구현 단계다.

1. Mission frame, localization, mapping 기반
2. Local policy와 hierarchy 분리
3. Long-horizon procedural world
4. Global RL MVP
5. Topological memory와 global-local feedback
6. Long-horizon 평가와 실차 배포

## 2. 현재 패키지에서 유지할 부분

다음 구현은 Low-Level navigation 기반으로 유지한다.

- `sensing/scan_processor.py`
- `sensing/temporal_stack.py`
- `env/observation/observation_builder.py`의 local observation 계약
- `rl/algorithms/kinodynamic_tqc/`
- `trajectory/action_space.py`의 `[kappa, v_ref, L]` 계약
- `trajectory/trajectory_primitive.py`
- `trajectory/pure_pursuit*.py`
- `dynamics/ackermann_rollout.py`
- `risk/`의 future risk, stopping margin, TTC, counterfactual risk
- 기존 Local replay/checkpoint/trainer
- domain randomization과 real-hardware safety guard

기존 모듈을 삭제하거나 현재 checkpoint 계약을 묵시적으로 변경하지 않는다. Hierarchical 기능은
별도 모듈과 별도 profile로 opt-in하며, 기존 local-only profile은 동일하게 동작해야 한다.

## 3. 공통 설계 원칙

### 3.1 좌표계

최소 좌표계 계약은 다음과 같다.

```text
odom/localization frame
        |
        | 출발 시 pose 캡처
        v
mission_start frame (mission 동안 고정)
        |
        +-- final goal: mission_start 기준 고정 좌표
        +-- partial map: mission_start 기준
        +-- topological nodes: mission_start 기준
        +-- route history: mission_start 기준

base_link/robot frame
        |
        +-- Global action: robot-relative subgoal
        +-- Local observation: robot-relative subgoal 거리/방위
```

사용자가 입력한 상대 목표를 현재 body frame에 계속 붙여 두지 않는다. 출발 pose가
`(x0, y0, yaw0)`이고 사용자 목표가 `(gx, gy)`라면 odom상의 목표는 다음과 같다.

```text
goal_odom.x = x0 + cos(yaw0) * gx - sin(yaw0) * gy
goal_odom.y = y0 + sin(yaw0) * gx + cos(yaw0) * gy
```

### 3.2 정보 경계

Simulator의 full ground-truth world는 다음 용도에만 사용한다.

- scenario solvability 검증
- ground-truth shortest path 계산
- benchmark 난이도와 SPL 계산
- 선택적인 teacher/auxiliary label 생성
- 충돌과 privileged risk training label 계산

다음 정보는 policy observation에 직접 포함하면 안 된다.

- 전체 장애물 목록
- 미관측 영역의 occupancy
- ground-truth 최단 경로
- ground-truth dead-end/정답 branch
- noisy localization profile에서의 noise-free pose

### 3.3 시간척도

```text
Global decision: 0.5~2 Hz 또는 event-driven
Local decision: 10~20 Hz
Safety guard: 모든 physical command publish 직전
```

Global transition은 여러 local step을 포함하는 option/SMDP transition으로 취급한다.

### 3.4 호환성

- 기존 `drl_agent_interfaces/Reset.srv`, `Step.srv`, `GetDimensions.srv`는 수정하지 않는다.
- Local-only trainer와 benchmark는 계속 기존 service를 사용한다.
- Hierarchical trainer에는 package-owned interface 또는 별도
  `hunter_kinodynamic_rl_interfaces` 패키지를 사용한다.
- BEV channel 수, resolution, subgoal action 의미가 변경되면 checkpoint fingerprint에도 반영한다.

## 4. 전체 구현 순서와 의존성

```text
Phase 1: Mission / Localization / Mapping
                    |
                    v
Phase 2: Local Goal 분리 / Hierarchy Coordinator
                    |
                    v
Phase 3: Long-Horizon Procedural World
                    |
                    v
Phase 4: Global RL MVP
                    |
                    v
Phase 5: Topology / Feasibility / Global Risk
                    |
                    v
Phase 6: Benchmark / Drift Evaluation / Real Hunter
```

Phase 1~3은 Global RL 없이도 독립적으로 검증해야 한다. Phase 4에서 최소 hierarchical system을
완성하고, Phase 5는 연구 기여를 확장하며, Phase 6은 성능과 실차 적용 가능성을 검증한다.

---

## 5. Phase 1: Mission frame, localization, mapping 기반

> **당시 구성요소 구현 기록 (2026-08-28; 최신 판정은 section 18).** Mission frame, relative final goal, timestamp-synchronized
> localization, online partial/rolling/visited/inflated map과 ROS/RViz adapter를 구현했다.
> Docker 전체 회귀 테스트 1,379개가 오류·실패·skip 없이 통과했고, Gazebo에서
> 기본 start pose와 비영점 start pose/yaw 두 경우의 scan 수신, map 누적, goal 고정,
> `odom -> mission` TF를 확인했다. 상세 근거는
> `docs/verification/2026-08-28_hierarchical_navigation_phase1_review_fixes.md`를 참조한다.

### 5.1 목표

- 출발 pose 기준의 고정 `mission_start` frame을 만든다.
- 사용자 상대 목표를 mission frame에 고정한다.
- localization backend를 navigation 코드에서 분리한다.
- LiDAR 관측으로 partial map과 visited map을 누적한다.
- `UNKNOWN`, `FREE`, `OCCUPIED`가 서로 배타적인 channel이 되도록 보장한다.

### 5.2 신규 모듈

```text
hunter_kinodynamic_rl/navigation/
├── __init__.py
├── mission/
│   ├── __init__.py
│   ├── mission_frame.py
│   └── goal_manager.py
├── localization/
│   ├── __init__.py
│   ├── interface.py
│   ├── odom_backend.py
│   └── gazebo_odom_backend.py
└── mapping/
    ├── __init__.py
    ├── raytracing.py
    ├── partial_map.py
    ├── rolling_map.py
    └── visited_map.py
```

### 5.3 주요 데이터 계약

```python
@dataclass(frozen=True)
class PoseEstimate:
    x: float
    y: float
    yaw: float
    stamp_sec: float
    covariance: np.ndarray
    confidence: float
    valid: bool


class LocalizationBackend(Protocol):
    def latest_pose(self) -> PoseEstimate: ...
    def pose_at(self, stamp_sec: float, max_dt_sec: float) -> PoseEstimate: ...
```

> **구현 갱신 (2026-08-28, 코드 리뷰 반영)**: `pose_at()`은 실제
> `navigation/localization/interface.py`의 `LocalizationBackend` protocol에
> 포함된 정식 계약이다 (scan timestamp에 대응하는 pose synchronization에
> 필요 — `mission_map_node.py`가 직접 의존). 위 예시는 최초 설계 당시
> `latest_pose()`만 있던 버전을 갱신한 것이다.

`MissionFrame`은 다음 API를 제공한다.

- `initialize(start_pose)`
- `mission_to_odom(x, y, yaw)`
- `odom_to_mission(x, y, yaw)`
- `mission_to_robot(point, robot_pose)`
- `robot_to_mission(point, robot_pose)`
- initialization 이전 호출에 대한 명시적 오류

`GoalManager`는 다음 상태를 소유한다.

- 사용자 입력 relative final goal
- mission-frame final goal
- position/heading tolerance
- mission active/cancelled/reached 상태
- goal 변경 시 map/memory를 초기화할지 여부

### 5.4 Partial map 표현

내부 map 상태는 최소 다음 배열을 유지한다.

```text
observed:        bool[H, W]
log_odds:        float32[H, W]
visited_count:   uint16[H, W]
failure_count:   uint8[H, W]
last_visit_step: int32[H, W]
```

Policy용 channel은 다음과 같이 생성한다.

```text
occupied = observed AND log_odds >= occupied_threshold
free     = observed AND log_odds <= free_threshold
unknown  = NOT observed
visited  = normalized visit count 또는 binary mask
```

UNKNOWN을 `1 - occupied`로 만들면 free와 unknown이 섞이므로 금지한다.

> **구현 결정 (2026-08-28, 코드 리뷰 반영)**: 위 공식대로 `free_threshold <
> occupied_threshold`인 두 개의 서로 다른 threshold를 쓰면, observed이면서
> log-odds가 두 threshold 사이인 셀은 occupied도 free도 아닌 상태가 실제로
> 발생한다 (`PartialMap.channels()`가 이 상태를 무시하지 않고
> `observed_uncertain`이라는 명시적 4번째 채널로 노출함 — 내부/RViz 전용,
> `occupied | free | observed_uncertain | unknown`이 모든 셀을 정확히 한 번씩
> 덮는 완전 분할). 본 섹션과 `hunter_se_unknown_gps_denied_hierarchical_navigation_detailed_spec.txt`의
> Global RL 입력 설명은 3채널(occupied/free/unknown)만 언급하므로, Phase 4에서
> Global observation tensor를 설계할 때 다음 중 하나를 명시적으로 정해야 한다:
> (a) `observed_uncertain`을 별도 입력 채널로 사용, (b) policy 입력에서는
> unknown 또는 free/occupied 중 하나로 병합하고 RViz에만 유지, (c) threshold
> 설계 자체를 단일 경계로 바꿔 observed 셀이 항상 free/occupied 중 하나가
> 되도록 함. Phase 1은 (a)/(b)/(c) 중 어느 쪽도 아직 선택하지 않은, 내부
> 표현이 정직하게 4-state라는 사실만 확정한 상태다.

### 5.5 LiDAR integration

각 beam에 대해 다음 순서를 적용한다.

1. scan timestamp와 대응하는 localization pose를 선택한다.
2. LiDAR extrinsic을 적용해 sensor origin을 mission frame으로 변환한다.
3. finite range이고 `range < range_max`이면 hit beam으로 처리한다.
4. sensor origin부터 endpoint 이전까지 ray-traced cell을 FREE로 갱신한다.
5. hit endpoint만 OCCUPIED로 갱신한다.
6. `inf`, `NaN`, max-range return은 occupied endpoint를 생성하지 않는다.
7. map boundary를 벗어나는 ray는 안전하게 clip한다.

초기 구현은 Bresenham ray tracing으로 충분하다.

### 5.6 Visited map

- 로봇 중심점 하나가 아니라 collision footprint 또는 설정된 visit radius를 rasterize한다.
- binary와 visit count를 모두 내부적으로 유지한다.
- policy observation에서는 `[0, 1]`로 normalize한다.
- local failure 발생 시 별도의 failure map을 갱신할 수 있게 API를 준비한다.
- odometry가 invalid/stale인 동안에는 잘못된 위치에 visited mark를 쓰지 않는다.

### 5.7 Config 변경

`config/schema.py`, `config/loader.py`, `config/training/defaults.yaml`에 다음 section을 추가한다.

```yaml
mission:
  position_tolerance_m: 0.6
  heading_tolerance_rad: 3.141592653589793
  require_low_speed_on_goal: true
  goal_speed_threshold_mps: 0.1

localization:
  backend: gazebo_odom
  odom_topic: /odometry
  pose_timeout_sec: 0.5
  minimum_confidence: 0.5
  publish_mission_tf: true

mapping:
  resolution_m: 0.2
  mission_size_cells: 256
  rolling_size_cells: 128
  free_log_odds_delta: -0.4
  occupied_log_odds_delta: 0.85
  free_threshold: -0.2
  occupied_threshold: 0.2
  log_odds_min: -4.0
  log_odds_max: 4.0
  inflation_radius_m: 0.45
  visit_radius_m: 0.4
```

숫자는 초기값이며 Hunter footprint와 LiDAR 특성에 맞춰 조정한다.

### 5.8 테스트

신규 테스트 예시는 다음과 같다.

```text
tests/test_mission_frame.py
tests/test_goal_manager.py
tests/test_localization_interface.py
tests/test_mapping_raytracing.py
tests/test_partial_map.py
tests/test_visited_map.py
```

필수 테스트 항목:

- arbitrary start yaw에서 relative goal 변환
- mission↔odom round-trip 오차
- 회전 후에도 final goal이 body frame을 따라 움직이지 않음
- 한 beam에서 free cells와 occupied endpoint가 올바름
- max-range beam이 occupied endpoint를 만들지 않음
- unknown/free/occupied channel이 배타적임
- map boundary clip과 잘못된 range 처리
- localization invalid/stale 시 map update 거부
- visited count saturation/normalization

### 5.9 완료 기준

- RL 없이 prerecorded scan/odom 또는 synthetic scan으로 map을 누적할 수 있다.
- RViz에서 mission pose, final goal, occupancy, unknown, visited를 확인할 수 있다.
- Gazebo start pose와 yaw를 바꿔도 동일한 relative goal 의미가 유지된다.
- 모든 Phase 1 단위 테스트가 통과한다.
- 기존 local-only test가 회귀 없이 통과한다.

---

## 6. Phase 2: Local policy와 hierarchy 분리

> **당시 구성요소 구현 기록 (2026-08-28; 최신 판정은 section 18).** `LocalPolicyController`(local_rl),
> `SubgoalManager`/`replanning`/`failure_recovery`/`HierarchyCoordinator`
> (hierarchy), opt-in `HierarchyConfig`를 구현했고 `real_policy_node.py`의
> observation/decode/guard 파이프라인을 `LocalPolicyController` 위임으로
> 리팩터링했다(동작 불변, 기존 테스트로 검증). 코드 리뷰 4라운드를 거쳐
> start_mission()의 stale subgoal 정리, degraded localization에서의
> success 오판 방지, degraded localization 중 physical motion 완전 차단
> (queue에 다음 candidate가 있어도 즉시 activate하지 않음), non-finite/malformed
> localization confidence의 안전한 degraded 처리, 최초 subgoal activation의
> confidence gate, validate_action()의 "never raises" 계약을 모두 코드로
> 고정했다. Round 3 기준 Docker 전체 회귀 테스트 1,475개가 오류·실패·skip
> 없이 통과했고, Round 4 후에는 영향 범위 targeted test 59개와 18개 profile
> config validation이 통과했다. 상세 근거는
> `docs/verification/2026-08-28_hierarchical_navigation_phase2.md`를 참조한다.
> Global RL(Phase 4)이 아직 없으므로 subgoal sequence는 외부 큐/heuristic으로
> 주입하고, `subgoal_endpoint_blocked`/`is_valid`는 caller가 공급하는
> boolean/callback으로 남겨 두었다 (PartialMap 연동은 Phase 3/4에서 실제 map이
> 생기는 시점에 연결).
>
> **2026-08-31 갱신 (6.6/6.9 arbitrary subgoal checkpoint, 진행 중).**
> `config/schema.py`의 `ScenarioConfig`에 `goal_sampling_mode="robot_relative_band"`
> (기본값 `"uniform_world"`, 다른 모든 profile은 byte-identical 유지),
> `goal_distance_range_m`(기본 2-6m), `goal_direction_sectors_deg`(전방/좌/우),
> `goal_infeasible_fraction`(blocked/unreachable subgoal 비율)을 추가하고
> `env/scenarios/procedural_generator.generate_scenario`에 구현했다(RNG 순서는
> band mode에서만 분기, `start_pose.heading_mode="legacy_random"` 아니면 fail-fast).
> `kinodynamic_tqc_counterfactual.yaml`(Ablation F: 전체 기능)과 동일한 architecture/
> action/risk 계약을 사용하는 `kinodynamic_tqc_arbitrary_subgoal.yaml` profile을
> 추가했다(`world_size_m=16`, `training.max_timesteps=150000` -- 세션 1회 학습에
> 맞춘 예산, 기존 2,000,000 baseline보다 적음). `tests/test_arbitrary_subgoal_sampling.py`
> (23 tests: 거리/섹터 범위, seed 결정성, 기본 모드 byte-identical, heading_mode
> fail-fast, infeasible_fraction 효과)를 추가했다. 실제 live Gazebo 학습을
> `runtime/experiments/20260831_005836_kinodynamic_tqc_arbitrary_subgoal_seed0/`에서
> 시작해 live Gazebo에서 실제로 학습을 진행했다. **최종 상태: global_step=10274,
> training_steps=7274 (gradient update 수) -- 150,000 목표의 약 6.8%에서 Phase 3/4
> live wiring 검증을 위해 의도적으로 중단했다 (Gazebo를 다른 검증 단계에 넘겨주기
> 위함, 세션 시간 제약).** "best"/"latest" checkpoint를
> `runtime/experiments/local_frozen`(symlink) 경유로 저장했고
> `hierarchical_phase4.yaml`의 `local_checkpoint_dir`/`name`이 이를 가리킨다.
> **6.9의 마지막 완료 기준("arbitrary subgoal checkpoint가 local benchmark를
> 통과한다")은 이 checkpoint 수준에서 NOT MET이다** -- live 20-scenario unseen
> benchmark(ablation A, 이 checkpoint가 final goal을 직접 추종)에서
> subgoal_success_rate=0.0, local_planner_failure_count 평균 3.45/mission으로,
> 아직 subgoal을 안정적으로 달성하지 못한다(예상된 결과: 7274 gradient step은
> TQC치고 극히 초기 단계). 학습/배선 자체는 실제 Gazebo에서 검증됐고 episode reward가
> 개선 추세(예: -100대에서 -34.7까지, 600-step 무충돌 episode 발생)를 보였으므로
> "코드가 작동하지 않는다"가 아니라 "더 많은 학습이 필요하다"는 결론이다 -- 세션
> 최종 보고서 참조.

### 6.1 목표

- 최종 mission goal과 현재 local subgoal을 분리한다.
- 기존 Local TQC가 final goal이 아니라 active subgoal을 추종하도록 만든다.
- Global policy 없이도 heuristic 또는 외부 입력 subgoal로 hierarchical control loop를 실행한다.
- subgoal success/failure와 mission success/failure를 분리한다.

### 6.2 신규 모듈

```text
hunter_kinodynamic_rl/navigation/
├── local_rl/
│   ├── __init__.py
│   └── controller.py
└── hierarchy/
    ├── __init__.py
    ├── coordinator.py
    ├── subgoal_manager.py
    ├── replanning.py
    └── failure_recovery.py
```

현재 `real_policy_node.py`의 observation→inference→action decode 부분은 가능한 범위에서
ROS-independent `LocalPolicyController`로 분리한다. Simulation과 real node가 동일한 local
controller 계약을 사용해야 한다.

### 6.3 상태 분리

현재 하나의 goal과 `done`에 섞인 의미를 다음처럼 분리한다.

```text
final_goal_mission
active_subgoal_mission
previous_final_goal_distance
previous_subgoal_distance

subgoal_reached
subgoal_failed
mission_reached
mission_failed
mission_timed_out
```

Local observation의 goal distance/bearing은 `active_subgoal` 기준이다. Global reward와 mission 종료는
`final_goal` 기준이다.

### 6.4 Subgoal lifecycle

```text
CREATED
  -> ACTIVE
  -> REACHED
  -> FAILED_BLOCKED
  -> FAILED_TIMEOUT
  -> FAILED_NO_PROGRESS
  -> FAILED_HIGH_RISK
  -> CANCELLED_BY_REPLAN
```

모든 종료에는 reason code와 다음 통계를 남긴다.

- local step 수
- elapsed simulation time
- path length
- 시작/종료 final-goal distance
- 시작/종료 subgoal distance
- minimum clearance
- maximum/mean predicted risk
- emergency stop count
- steering saturation count
- newly explored cells

### 6.5 Replanning 조건

- subgoal tolerance 도달
- local option timeout
- 설정 시간 동안 subgoal progress 없음
- known occupied/inflated cell로 subgoal이 변경됨
- emergency stop 연속 발생
- local risk threshold 초과
- localization confidence 저하
- 새로운 junction 발견 이벤트

Replanning 중에는 이전 command가 남지 않도록 stop command를 명시적으로 publish한다.

### 6.6 Local TQC 재학습

현재 local policy가 다양한 short-range subgoal을 경험하도록 scenario sampler를 확장한다.

- subgoal 거리 분포: 예를 들어 2~6 m
- 방향 분포: 전방뿐 아니라 좌우 방향 포함
- Ackermann으로 즉시 도달 불가능한 후보도 일부 포함
- local timeout과 blocked 상황 포함
- final mission goal 정보는 local policy에 주지 않음

기존 checkpoint는 먼저 inference compatibility를 확인하되, 최종 hierarchical 실험에는 arbitrary
subgoal 분포로 새 checkpoint를 학습하는 것을 원칙으로 한다.

### 6.7 Hierarchical interface

기존 shared service는 유지하고 다음 package-owned interface를 정의한다.

```text
SetRelativeGoal.srv
ResetMission.srv
StepGlobal.srv 또는 ExecuteSubgoal.action
Subgoal.msg
SubgoalResult.msg
NavigationStatus.msg
```

Global trainer가 사용할 최소 `StepGlobal` 응답은 다음 정보를 포함해야 한다.

```text
next_global_observation
next_action_mask
global_reward 또는 reward components
mission_done
mission_success
collision
subgoal_success
subgoal_failure_reason
local_steps
```

### 6.8 테스트

```text
tests/test_subgoal_manager.py
tests/test_replanning.py
tests/test_hierarchy_coordinator.py
tests/test_local_controller_contract.py
tests/test_hierarchical_interface.py
```

필수 검증:

- subgoal 도달이 mission 종료로 오인되지 않음
- final goal 도달만 mission success가 됨
- timeout/blocked/risk/replan reason 분리
- stale localization 또는 invalid subgoal 시 physical motion 금지
- replanning 동안 stop command 보장
- local observation에 final goal이 누출되지 않음
- 기존 local-only service와 checkpoint test 회귀 없음

### 6.9 완료 기준

- Global RL 대신 정해진 subgoal sequence를 주어 여러 subgoal을 연속 수행할 수 있다.
- subgoal 실패 후 다른 subgoal로 recovery할 수 있다.
- mission과 option의 reward/termination/log가 분리된다.
- arbitrary short-range subgoal checkpoint가 local benchmark를 통과한다.

---

## 7. Phase 3: Long-horizon procedural world

> **당시 구성요소 구현 기록 (2026-08-28; 최신 판정은 section 18).** `long_horizon_world`/`long_horizon_generator`/
> `long_horizon_solvability`/`long_horizon_curriculum`/`wall_segment_spawner`를
> 구현했다. Seed 기반 deterministic room/corridor/junction/loop/dead-end
> lattice maze generator, occupancy와 항상 일치하는 wall segment 목록,
> Dijkstra 기반 shortest path/geodesic 검증, grid 기반 Ackermann feasibility
> 검사, train/validation/test seed pool 분리, opt-in `LongHorizonWorldConfig`/
> `WallSegmentPoolConfig`를 포함한다. Global RL은 추가하지 않았다(범위 밖).
> `environment_node.py`에는 아직 연결하지 않았으며 (Phase 4/5에서 실제
> long-horizon episode가 필요할 때 연결 예정), live Gazebo 검증은
> 수행하지 않았다(세션 시작 시 실행 중인 Gazebo 인스턴스가 없었음).
>
> **Round 2 (code review, same date).** 리뷰에서 실제 결함 5건을 발견해
> 모두 수정했다: (1) `wall_segment_spawner.py`가 module scope에서
> `ros_gz_interfaces`까지 끌고 들어와 ROS 없는 bare host에서 import 자체가
> 실패하던 문제 (순수 `_snap_length_up_to_class` 재구현 + `ensure_spawned()`
> 내부 lazy import로 수정), (2) `generate_long_horizon_world()`가
> train/validation/test seed pool 분리를 강제하지 않던 문제 (`mode` 파라미터 +
> `LongHorizonSeedScheduler` 추가), (3) `LongHorizonWorld`가 `frozen=True`임에도
> `occupancy`/`wall_segments`/`topology_metadata`를 실제로는 in-place 수정 가능하던
> 문제 (`__post_init__`에서 read-only copy/tuple/`MappingProxyType`로 정규화), (4)
> `wall_segment_pool.max_segments=0`이 validate를 통과하고 worst-case segment 수를
> 검증하지 않던 문제 (`max_possible_wall_segment_count` 추가 + 스키마 검증 강화),
> (5) "독립 검증" 문서 표현 과장 (hand-computed 정확값 fixture 테스트 추가).
> Round 2 이후 Docker 전체 회귀 테스트 **1590개가 오류·실패·skip 없이 통과**했고,
> 19/19 profile이 유효하다.
>
> **Round 3 (code review, same date).** Round 2의 wall pool capacity 수정이
> 여전히 불충분함을 재현: `validate_wall_pool_capacity`는 총 segment 수만
> 검증했지만 `activate_walls()`는 class별 정확히 맞는 free slot을 요구해,
> validate를 통과한 profile이 실제로는 `no free slot in the length class`로
> 런타임에 실패했다. `max_possible_wall_segment_counts_by_class`를 추가해
> class별 worst-case 수요를 계산하고, `max_segments // len(length_classes_m)`
> (round-robin 배분의 class별 최소 보장 슬롯 수)가 이를 감당하는지 검증하도록
> 수정했다 (`hierarchical_phase3.yaml`도 재조정: `length_classes_m: [3.0, 4.0,
> 13.4]`, `max_segments: 240`). 또한 `LongHorizonWorld.__post_init__`의 nested
> immutability가 top-level `MappingProxyType`만 적용해 실제로는 generator가
> `room_cells`를 tuple로 넣어줄 때만 보호되던 문제를 `_deep_freeze` 재귀
> helper로 수정했다 (외부 caller가 nested list/dict를 넘겨도 동일하게 보호됨).
> Round 3 이후 Docker 전체 회귀 테스트 **1597개가 오류·실패·skip 없이 통과**했고,
> 19/19 profile이 유효하다. 상세 근거는
> `docs/verification/2026-08-28_hierarchical_navigation_phase3.md`를 참조한다.
>
> **2026-08-31 갱신 (live Gazebo 연결, 진행 중).** 이전까지 이 phase의 유일한
> gap이었던 "`long_horizon_world`/`wall_segment_pool`이 live Gazebo reset에
> 연결되지 않음"을
> `navigation/local_rl/live_gazebo_executor.py`의 `LiveGazeboLocalExecutor.bind_mission()`으로
> 닫았다: `pause_world` → `reset_world` → `activate_walls(wall_pool, world.wall_segments)`
> → robot을 `world.start_pose`로 `set_entity_pose_ignition` → `propagate_state`(settle)
> → `wait_for_fresh_sensors` 순서로, `env/simulation/gazebo_runtime.GazeboRuntimeMixin`
> (기존 `environment_node.py`가 쓰는 hang-회피 `time.sleep()` polling 규율 재사용)과
> `env/spawning/wall_segment_spawner`(기존 `ensure_spawned`/`activate_walls`)를
> 그대로 재사용했다.
>
> **Live Gazebo 검증 완료 (같은 세션, Phase 2 학습을 중단해 Gazebo를 확보한 뒤).**
> `runtime/smoke_test_live_hierarchical.py`로 seed=0 world(57 wall segment)를
> 생성해 `bind_mission()`을 실행: 150개 wall pool slot이 실제로 spawn됐고, robot이
> 요청한 `start_pose`로 정확히(오차 0.000m) teleport됐으며, `wait_for_fresh_sensors`가
> 실제 새 `/scan`/`/odometry` 메시지 도착을 확인했다(80-91 scan, 202-230 odom
> update). 이 과정에서 THREE개의 실제 결함을 발견해 수정했다: (1) 백그라운드
> `MultiThreadedExecutor` spin thread가 wall-pool spawn 서비스 호출보다 늦게
> 시작해 모든 Gazebo 서비스 호출이 타임아웃 -- thread 시작 순서를 fail-fast
> checkpoint 검증 이후·Gazebo 의존 호출 이전으로 재배치, (2) `/scan`/`/odometry`
> 구독이 기본 RELIABLE QoS를 써서 실제 BEST_EFFORT publisher와 호환되지 않아
> 메시지가 전혀 도착하지 않음 -- `hierarchical_environment_node.py`와 동일한
> `SENSOR_QOS`(BEST_EFFORT)로 수정, (3) `pose_world` 프로퍼티가 아직 odom을
> 받지 못한 상태(`INVALID_POSE`, x=y=0.0 -- **finite**)를 유효한 (0,0,0) 판독으로
> 오인 -- `pose.valid` 플래그 확인을 추가. 세 번째는 라이브 실행 전 코드 리뷰에서
> 발견, 앞의 둘은 라이브 실행에서 직접 확인. 이후 `run_option()`을 실제로 호출해
> partial map이 0 -> 5000+ observed cell로 확장되고 robot이 물리적으로 1m 이상
> 이동하는 것을 확인했다(Phase 4 절 참조). 자세한 수치는 세션 최종 보고서 참조.

### 7.1 목표

현재 사각 arena와 원형 장애물 배치를 room/corridor/junction/loop/dead-end를 포함하는
장거리 unknown world generator로 확장한다.

### 7.2 World representation

generator 내부에서는 full occupancy/semantic 정보 사용이 가능하다.

```python
@dataclass(frozen=True)
class LongHorizonWorld:
    seed: int
    occupancy: np.ndarray
    resolution_m: float
    origin_xy: tuple[float, float]
    wall_segments: list[WallSegment]
    start_pose: tuple[float, float, float]
    goal_pose: tuple[float, float]
    shortest_path_length_m: float
    topology_metadata: dict
```

`topology_metadata`는 evaluation/teacher용 privileged 데이터이며 policy observation으로 전달하지 않는다.

### 7.3 생성 순서

1. seed 기반 random graph 또는 maze skeleton 생성
2. room과 corridor를 metric grid로 rasterize
3. dead-end branch를 최소 개수 이상 포함
4. loop와 alternative route를 최소 개수 이상 포함
5. start/goal 사이 최소 geodesic distance 조건 적용
6. collision footprint를 반영해 free space inflation
7. grid connectivity 검사
8. Hunter turning radius를 반영한 Ackermann feasibility 검사
9. shortest feasible path와 난이도 metadata 계산
10. wall segment를 Gazebo spawn spec으로 변환

### 7.4 Gazebo geometry

현재 virtual world boundary만으로는 LiDAR가 벽을 관측할 수 없다. 실제 collision/visual geometry를 가진
box wall을 spawn해야 한다.

성능을 위해 다음 방식을 우선 고려한다.

- 고정 길이 wall segment를 미리 pool로 spawn
- episode reset 때 pose/yaw만 변경
- 사용하지 않는 segment는 LiDAR range 밖 parking 위치로 이동
- corridor 폭과 wall thickness는 고정 size class 사용

가변 크기 room wall이 반드시 필요하면 여러 fixed segment를 조합한다. 매 episode마다 SDF world 전체를
재시작하는 방식은 reset 비용이 크므로 후순위로 둔다.

### 7.5 난이도 curriculum

```text
Level 1: 넓은 corridor, 짧은 경로, dead end 없음
Level 2: junction과 1개 dead end
Level 3: loop와 alternative route
Level 4: 긴 dead-end/backtracking 필요
Level 5: 좁은 Ackermann-feasible corridor와 복합 room
Level 6: dynamic obstacle 추가
```

curriculum 여부와 관계없이 train/validation/test seed pool은 완전히 분리한다.

### 7.6 Config

```yaml
long_horizon_world:
  enabled: true
  size_m: 40.0
  resolution_m: 0.25
  corridor_width_min_m: 2.0
  corridor_width_max_m: 4.0
  room_count_range: [4, 10]
  dead_end_count_range: [1, 5]
  loop_count_range: [1, 4]
  alternative_route_min_count: 1
  start_goal_geodesic_min_m: 20.0
  require_ackermann_feasibility: true
  generation_attempt_limit: 100
```

### 7.7 테스트

```text
tests/test_long_horizon_generator.py
tests/test_long_horizon_solvability.py
tests/test_wall_segment_spawner.py
tests/test_world_information_boundary.py
```

필수 검증:

- 동일 seed에서 동일 world 생성
- train/validation/test seed 분리
- start/goal이 occupied 또는 inflated wall에 있지 않음
- 설정된 dead end, loop, alternative route 조건 충족
- shortest-path metadata 재계산 일치
- wall spawn geometry와 generator occupancy 일치
- policy observation에 GT occupancy/metadata가 들어가지 않음
- 실제 LiDAR scan으로 wall이 관측되고 partial map에 누적됨

### 7.8 완료 기준

- 최소 100개 이상의 seed를 연속 생성해 bounded attempt 안에 성공한다.
- Gazebo에서 room/corridor/dead-end 구조가 물리적으로 spawn된다.
- Hunter가 oracle path 또는 검증 controller로 selected worlds를 완주할 수 있다.
- partial map은 시작 시 대부분 UNKNOWN이며 주행에 따라 확장된다.

---

## 8. Phase 4: Global RL MVP

> **당시 구성요소 구현 기록 (2026-08-28; 최신 판정은 section 18).** `navigation/global_rl/`(observation,
> subgoal_sampler, action_mask, networks, agent, replay, replay_schema,
> reward), opt-in `GlobalRLConfig`/`HierarchicalTrainingConfig`,
> `training/train_hierarchical_dqn.py`(ROS-free `HierarchicalTrainingLoop`),
> `nodes/hierarchical_environment_node.py`/`nodes/hierarchical_train_node.py`,
> `config/profiles/hierarchical_phase4.yaml`를 구현했다. 8방향x2거리
> discrete candidate + backtrack fallback, action mask(occupied/inflated
> endpoint, **Ackermann arc** short rollout collision -- 직선이 아니라
> endpoint까지 실제로 도달하는 데 필요한 전체 arc length(`s=r*phi/sin(phi)`,
> 직선거리 `radius_m`로 잘라내지 않음)를 vehicle의 실제 turning radius로
> sweep, 필요 curvature가 한계를 넘으면 clamp 대신 해당 후보를 invalid
> 처리, 즉시 footprint collision, localization/map invalid, UNKNOWN은
> valid 유지, fallback 항상 valid), map/scalar/candidate 고정 shape 관측,
> masked Dueling Double DQN (`Q(invalid)=-inf`), `option_reward +
> gamma^local_steps * (1-mission_done) * max_valid Q_target` SMDP target,
> uint8 압축 Global replay(schema version + 채널/scalar/candidate-feature
> 이름 metadata 저장 및 load 시 불일치 검증, RNG 상태 저장, `mode="train"`
> 외 저장 거부 guard), reason별 local-failure penalty + exploration reward
> clip + repeated dead-end 전용 penalty를 포함한 Global reward를 구현했다.
> Local checkpoint는 `local_checkpoint_dir`/`local_checkpoint_name`
> (directory+tag) pair로 config화해 `ckpt_manager.load_generation`과
> checkpoint hash 계산이 정확히 같은 파일을 가리키도록 했다. 리뷰 라운드 1에서
> 발견된 3건(checkpoint dir/path 불일치, 직선 rollout, replay 채널 metadata
> 누락)을 모두 수정했고, 리뷰 라운드 2에서 라운드 1의 Ackermann arc 수정이
> endpoint까지의 전체 arc length가 아니라 여전히 직선거리(radius_m)만큼만
> sweep해 실제 arc의 뒷부분(예: 90도/6m 후보의 6m~9.42m 구간)에 있는
> obstacle을 놓치는 결함을 재발견해 수정했다(clamp 대신 curvature 초과 시
> invalid 처리로 변경). 이후 최종 리뷰에서 긴 feasible arc가 고정
> `rollout_sample_count` 사이의 known obstacle을 건너뛸 수 있던 문제를
> 추가로 수정했다: `rollout_sample_count`는 최소 샘플 수로만 사용하고,
> 실제 arc length를 `partial_map.resolution_m * 0.5` 이하 간격으로
> adaptive subdivision하며, 인접 샘플 사이 grid cell도 `trace_clipped()`로
> 검사한다. 135도/6m 후보의 약 20m 원호에서 기존 sparse sample 사이에
> 놓인 obstacle 회귀 테스트를 추가했다. 두 라운드 수정 후 Docker 전체
> 회귀 테스트 1660개, `colcon test` 1665개가 오류·실패·skip 없이
> 통과했고, 최종 adaptive sampling 수정 후 Phase 4 관련 로컬 회귀 테스트
> 55개가 통과했다(2개 skip은 기존 torch-gated 테스트). 20/20 profile이
> 유효하다(19개 기존 + `hierarchical_phase4`). 상세 근거는
> `docs/verification/2026-08-28_hierarchical_navigation_phase4.md`를
> 참조한다.
>
> **MVP 범위상 제한 (다음 세션/Phase 5-6에서 마저 다룰 것):**
> `HierarchicalTrainingLoop`의 local-option 실행은 실제 frozen kinodynamic
> TQC + live Gazebo가 아니라 `SimplifiedKinematicLocalExecutor`(단순
> point-robot steer-then-drive 모델 + 시뮬레이션 LiDAR ray-cast를 Phase
> 1의 `PartialMap`/`raytracing`에 실제로 적분하는 lightweight stand-in)로
> 대체했다 -- Global RL/map/hierarchy 배선 자체는 실제 Phase 1-3 코드를
> 그대로 재사용해 검증했지만, "실제 Hunter kinodynamic 동역학으로 이
> Global 정책이 안전하게 subgoal을 추종하는지"는 아직 검증하지 않았다.
> `nodes/hierarchical_environment_node.py`/`hierarchical_train_node.py`는
> 구조적으로 완성된 ROS adapter(프로파일 로드, 센서 구독,
> `HierarchyCoordinator`/`LocalPolicyController`/`GlobalDQNAgent` 배선,
> Docker에서 `rclpy.init()`으로 생성 자체는 확인함)이지만 **살아있는
> Gazebo 인스턴스로 실행/검증하지 않았다**(Phase 3와 동일한 제한). Local
> checkpoint hash는 Global manifest에 기록되도록 구현·테스트했지만, 실제
> frozen local TQC checkpoint를 만들어 Phase 4 MVP를 그 위에서 학습시키는
> 실험 자체는 이번 세션 범위 밖이다.
>
> **2026-08-31 갱신 (live Local TQC executor, 진행 중).** 위 MVP 범위상 제한의
> 핵심 gap이었던 `SimplifiedKinematicLocalExecutor` 대체를
> `navigation/local_rl/live_gazebo_executor.LiveGazeboLocalExecutor`로 구현했다:
> `LocalOptionExecutor` protocol(`run_option(coordinator, partial_map, rng,
> max_local_steps)`)과 `run_mission`이 요구하는 `pose_world` 프로퍼티를 그대로
> 만족하고, `HierarchyCoordinator.record_local_tick`(기존 Phase 1/2 termination
> 로직 재사용, 재구현 없음)을 매 tick 호출한다. `training/train_hierarchical_dqn.py`의
> `HierarchicalTrainingLoop`와 `evaluation/long_horizon_benchmark.py`의
> `run_ablation_mission` 양쪽에 `local_executor_factory` 주입점을 추가했다
> (기본값 `None` = 기존 `SimplifiedKinematicLocalExecutor` 경로와 byte-identical --
> 기존 hierarchical 테스트 전체가 무수정 통과로 회귀 없음을 확인). 누락/architecture
> 불일치 checkpoint는 `LocalCheckpointError`로 즉시 fail-fast(4개 신규 테스트로
> 검증, `ckpt_manager.load_generation`의 생성/해시 검증에 더해 `state_dim`/
> `action_dim`/`architecture_fingerprint` 비교 추가). `nodes/hierarchical_train_node.py`에
> `live` 파라미터(및 `launch/hierarchical_train.launch.py`)를, 새 CLI
> `evaluation/run_live_hierarchical_benchmark.py`(및
> `launch/hierarchical_environment.launch.py`)를 추가해 A(local-only baseline)
> vs B(Phase 4 Global DQN) 비교를 live Gazebo + 실제 frozen Local TQC로 실행할
> 수 있게 했다.
>
> **Live Gazebo 검증 완료 (같은 세션).** `hierarchical_phase4.yaml`을 실제 frozen
> checkpoint(Phase 2, global_step=10274)에 물렸을 때 즉시 실제 결함을 하나 더
> 발견했다: 이 profile은 `action_space`/`features`/`observation`/`risk`/
> `counterfactual`을 `defaults.yaml` 상속(robot_state_dim=7, temporal_context=false
> 등, state_dim=88)에 맡기고 있었는데, checkpoint는 328-wide 입력(frame_stack=4,
> robot_state_dim=8, temporal_context=true)로 학습됐다 -- `LiveGazeboLocalExecutor`가
> 이 profile로 `RiskAgent`를 구성하자 `load_state_dict` shape mismatch로 즉시
> fail-fast했다(설계대로 동작 -- 조용히 진행하지 않음). `kinodynamic_tqc_arbitrary_subgoal.yaml`과
> 동일한 5개 section을 `hierarchical_phase4.yaml`에 명시적으로 복사해 수정했다.
> 이후 `LiveGazeboLocalExecutor.run_option()`을 live Gazebo에서 직접 실행해 확인:
> subgoal 활성화 -> 44 local step -> `FAILED_NO_PROGRESS`(실제
> `HierarchyCoordinator` 판정, 재구현 아님) -> final goal distance 9.96m -> 8.93m로
> 실제 감소, partial map 0 -> 5000+ observed cell, robot 1.03m 실제 이동. 이 과정에서
> 스레드 경합 버그(`_on_scan`이 `self._active_partial_map`을 두 번 재읽어 main
> thread의 `None` 대입과 경합, 백그라운드 스레드에서 조용히 예외 발생)도 발견해
> local 변수로 한 번만 읽도록 수정했다.
>
> `nodes/hierarchical_train_node.py --live`로 5-mission end-to-end 학습(Global
> decision -> live Local TQC option -> SMDP replay -> `GlobalDQNAgent.train_step`
> -> checkpoint 저장)을 실행: global_step=200, 실제 gradient update 발생(mission 4
> loss=9.4096), checkpoint 저장 확인. `resume_run_dir`로 1개 추가 mission을 재개
> 실행해 "resumed ... at global_step=200" 로그와 함께 정확히 이어서 실행되고
> (다시 실제 gradient update, loss=8.0420) global_step=240으로 진행하는 것을
> live Gazebo에서 확인했다 -- replay/optimizer/seed scheduler/checkpoint resume이
> 실제로 동작한다.
>
> **Live 20-scenario unseen benchmark (test seed pool, `run_live_hierarchical_benchmark.py`,
> `max_options=5`/`max_local_steps=35`로 wall-clock 축소 -- profile 기본값
> 40/150 그대로면 비현실적으로 느림)**: ablation A(local-only baseline)와
> B(Phase 4 Global DQN, 이번 세션 5-mission/200-step 학습 checkpoint) 모두
> **final_goal_success_rate=0.0, subgoal_success_rate=0.0** -- Phase 4 완료
> 기준("hierarchical MVP가 baseline보다 final-goal success가 높다")은 **이 학습
> 수준에서 NOT MET이다**. Local TQC(7274 gradient step)와 Global DQN(1 gradient
> step)이 모두 극히 초기 단계인 것이 원인으로, "wiring이 작동하지 않는다"가
> 아니라 "성능에 필요한 학습량에 크게 못 미친다"는 뜻이다 -- 정확한 수치와
> 해석은 세션 최종 보고서 참조.

### 8.1 목표

- partial map, visited map, final-goal vector로 다음 subgoal을 선택한다.
- 첫 구현은 `8방향 x 2거리` discrete candidate를 사용한다.
- Local TQC는 고정된 checkpoint로 사용해 High-Level 학습의 non-stationarity를 줄인다.

### 8.2 신규 모듈

```text
hunter_kinodynamic_rl/navigation/global_rl/
├── __init__.py
├── observation.py
├── subgoal_sampler.py
├── action_mask.py
├── networks.py
├── agent.py
├── replay.py
├── replay_schema.py
└── reward.py

hunter_kinodynamic_rl/training/
└── train_hierarchical_dqn.py

hunter_kinodynamic_rl/nodes/
├── hierarchical_environment_node.py
└── hierarchical_train_node.py
```

### 8.3 Global action

기본 action set:

```text
directions = [-180, -135, -90, -45, 0, 45, 90, 135] deg
distances  = [3.0, 6.0] m
candidate count = 16
```

모든 후보가 invalid일 때를 위해 `STOP_RECOVERY` 또는 `BACKTRACK` fallback action을 추가하는 것을
권장한다.

Global action은 robot-relative로 생성하고 선택 즉시 mission-frame subgoal로 변환해 저장한다.

### 8.4 Action mask

다음 후보를 invalid로 처리한다.

- endpoint가 known occupied/inflated cell 안에 있음
- endpoint까지의 Ackermann short rollout이 known obstacle과 충돌
- 수치가 NaN/Inf이거나 허용 radius 밖
- localization/map 상태가 invalid
- 현재 robot footprint와 즉시 충돌하는 경로

UNKNOWN endpoint는 exploration 대상이므로 UNKNOWN이라는 이유만으로 invalid 처리하지 않는다.
모든 non-fallback action이 invalid여도 fallback action은 항상 valid여야 한다.

### 8.5 Global observation

MVP observation:

```text
Map tensor:
  occupied
  free
  unknown
  visited
  final-goal heatmap/direction

Scalar tensor:
  final-goal distance
  final-goal bearing
  current speed
  previous global action
  elapsed mission ratio

Candidate tensor:
  action mask
  known-free ratio along candidate
  unknown gain estimate
  approximate curvature difficulty
```

Goal이 rolling map 밖에 있으면 heatmap 가장자리에 억지로 clamp한 값만 사용하지 않는다. 실제 거리와
bearing scalar를 항상 함께 제공한다.

### 8.6 Network

```text
BEV channels -> CNN encoder -----------+
scalar state -> MLP encoder -----------+-> fused MLP -> Q value per action
candidate features -> shared MLP ------+
action mask ------------------------------> invalid Q = -inf at selection
```

초기 알고리즘은 masked Double DQN 또는 masked Dueling Double DQN으로 한다. Distributional DQN은
MVP 안정화 후 확장한다. 현재 continuous TQC Actor/Critic을 Global policy에 재사용하지 않는다.

### 8.7 Global reward

Global reward는 subgoal option 종료 시 한 번 계산한다.

```text
R_global = R_goal
         + R_final_progress
         + R_exploration
         - R_repeated_revisit
         - R_repeated_deadend
         - R_local_failure
         - R_risk
         - R_elapsed
```

구현 규칙:

- `R_goal`: final goal 도달 시 큰 terminal reward
- `R_final_progress`: option 시작/종료의 final-goal 거리 차이
- `R_exploration`: 새로 observed가 된 cell 수 또는 면적
- `R_repeated_revisit`: visit count가 높은 영역을 불필요하게 반복한 거리
- `R_repeated_deadend`: 이미 실패로 기록된 branch 재진입
- `R_local_failure`: timeout/blocked/high-risk 등 reason별 penalty
- `R_risk`: option 동안의 maximum 또는 integral risk
- `R_elapsed`: local step 수 또는 simulation time 기반 penalty

첫 dead-end 탐색과 junction으로의 정상 backtracking은 동일한 실패를 반복한 것으로 처리하지 않는다.
Exploration reward는 goal reward 또는 장기 progress를 압도하지 않도록 clip한다.

### 8.8 Global replay와 SMDP target

```python
GlobalTransition(
    map_state,
    scalar_state,
    candidate_features,
    action_mask,
    action,
    option_reward,
    next_map_state,
    next_scalar_state,
    next_candidate_features,
    next_action_mask,
    mission_done,
    subgoal_success,
    failure_reason,
    local_steps,
    exploration_gain,
    risk_integral,
)
```

Global option 길이가 서로 다르므로 target은 다음 형태를 사용한다.

```text
target = option_reward
       + gamma^local_steps * (1 - mission_done) * max_valid Q_target(next_state)
```

Replay memory 규칙:

- Local replay와 완전히 분리
- map은 `uint8` 또는 bit-packed representation으로 저장
- sampling 시 float tensor로 변환
- 초기 capacity는 20k~100k 수준에서 memory usage 측정 후 결정
- replay schema version과 map/channel metadata 저장
- test/benchmark transition은 Global replay에 저장하지 않음

### 8.9 학습 순서

1. 검증된 Local checkpoint를 freeze한다.
2. Phase 3의 쉬운 curriculum world부터 Global replay를 수집한다.
3. random masked action warmup을 수행한다.
4. masked Double DQN을 학습한다.
5. 별도 validation seed로 success/revisit/collision을 평가한다.
6. best checkpoint는 global success 중심 metric으로 선택한다.
7. MVP가 안정화되기 전에는 Local policy를 joint fine-tuning하지 않는다.

### 8.10 테스트

```text
tests/test_global_observation.py
tests/test_subgoal_sampler.py
tests/test_global_action_mask.py
tests/test_global_reward.py
tests/test_global_replay.py
tests/test_global_smdp_target.py
tests/test_masked_dqn.py
tests/test_hierarchical_training_loop.py
```

필수 검증:

- invalid action이 random/greedy selection에서 선택되지 않음
- all-invalid 상황에서 fallback action 선택
- UNKNOWN 후보가 exploration 가능 상태로 유지됨
- map channel과 scalar/candidate shape 고정
- `gamma^local_steps` 적용
- `mission_done`과 `subgoal_done` bootstrapping 차이
- first dead end/backtracking에 과도한 반복 penalty가 없음
- replay save/resume 후 sampling RNG와 schema 유지
- Local checkpoint hash가 Global manifest에 기록됨

### 8.11 완료 기준

- unseen validation seed에서 local-only/no-memory baseline보다 높은 final-goal success를 보인다.
- 동일 dead-end 반복 진입과 무한 loop가 감소한다.
- Global/local decision rate가 분리되어 로그로 확인된다.
- Global checkpoint와 replay가 중단 후 정상 resume된다.

---

## 9. Phase 5: Topological memory와 Global-Local feedback

> **당시 구성요소 구현 기록 (2026-08-29; 최신 판정은 section 18).** `navigation/memory/`(topological_graph,
> node_manager, route_history, dead_end_detector), `navigation/hierarchy/feasibility.py`,
> `navigation/global_rl/feasibility_predictor.py`를 구현했다. Global
> observation/network/replay/reward는 `GlobalRLConfig`의 신규 ablation flag
> (`include_failure_channel`/`topology_feedback_enabled`/
> `feasibility_feedback_enabled`/`global_risk_feedback_enabled`, 모두 기본
> False)로 확장했고, 모든 flag가 꺼진 기본 상태는 Phase 4와 byte-identical
> 관측/네트워크 shape을 유지한다(`test_hierarchical_checkpoint_compatibility.py`).
> Replay schema를 v3로 올리고(node tensor 저장), `hierarchical_architecture_fingerprint`를
> 신설했다(local-only `architecture_fingerprint`와 완전히 분리). 사용자 요청이
> 함께 포함시킨 Phase 6 항목(`evaluation/global_metrics.py`,
> `evaluation/long_horizon_benchmark.py`, localization backend 확장
> `wheel_imu_backend.py`/`lidar_odom_backend.py`/`lio_adapter.py`,
> `nodes/hierarchical_navigation_node.py`의 실차/dry-run 안전 계층)도 함께
> 구현했다. Ablation A/B/D/E/F/G 프로파일(`hierarchical_phase5_{a,b,d,e,f,g}.yaml`)을
> 추가했다 -- **B와 C는 이 구현에서 동일**하다(Phase 4의 `visited` 채널이
> 원래 별도 토글 뒤에 있지 않았음, 정직하게 문서화). 신규 테스트 13개
> 전부 통과, Docker 전체 회귀 테스트(`python3 -m pytest -q`) 1773개와
> `colcon test` 1778개가 오류·실패·스킵 없이 통과했고, 26/26 profile이
> 유효하다. `HierarchicalTrainingLoop`/`train_hierarchical_dqn()`/
> `run_ablation_benchmark()`를 실제 torch Global agent로 end-to-end
> smoke-run해 replay/checkpoint/metrics 파이프라인 전체를 확인했다. 라이브
> Gazebo 검증과 rosbag dry-run은 Phase 3/4와 동일하게 이번 세션 범위 밖이다
> (실행 중인 Gazebo 인스턴스 없음). 상세 근거는
> `docs/verification/2026-08-29_hierarchical_navigation_phase5.md`를 참조한다.
>
> **Round 2 (code review, same date).** 외부 리뷰에서 실제 결함 6건을 발견해
> 모두 수정했다: (1) real/dry-run node가 `goal_x`/`goal_y`(이미 mission-frame
> 좌표)에 `mission_frame.odom_to_mission()`을 한 번 더 적용해 비영점 start
> pose/yaw에서 final goal이 틀어지던 문제, (2) safety guard 호출에서
> `now_sec`(monotonic)과 `last_sensor/odom_time_sec`(ROS 메시지 stamp)의
> clock domain이 섞여 있던 문제(신규 `_latest_{scan,odom}_receipt_time` 분리로
> 수정), (3) localization backend가 profile 설정과 무관하게 항상
> `GazeboOdomLocalizationBackend`로 고정되던 문제(`_build_localization_backend()`
> factory 신설, `wheel_imu`/`lio` 실제 배선), (4) benchmark manifest의
> `relative_goal`이 mission-frame이 아닌 world 좌표 그대로였던 문제, (5)
> `run_ablation_mission()`이 ablation label과 무관하게 caller의 config
> flag를 그대로 써서 label-config 불일치 시 예외 또는 feature 누락이
> 가능했던 문제(`effective_*_config()` 강제 재작성으로 수정), (6) 새로
> 생성된 topological node의 `visit_count`가 0으로 시작해 merge된 node와
> 비일관이었던 문제. Round 2 이후 Docker 전체 회귀 테스트 1785개,
> `colcon test` 1790개가 오류·실패·스킵 없이 통과했고, 26/26 profile이
> 유효하다. 상세 근거는 검증 문서의 "Round 2" 섹션을 참조한다.
>
> **Round 3 (code review, same date).** 다시 리뷰에서 결함 2건을 발견해
> 수정했다: (1) `_localization_valid()`가 여전히 `now_sec=self._latest_odom_time`
> (pose의 message stamp를 자기 자신과 비교)으로 검사해 odom 수신이 끊긴
> 뒤에도 계속 valid로 남던 문제 -- Global action mask/subgoal activation이
> stale localization으로도 계속 동작할 수 있었다(물리 command는 Round 2에서
> 이미 막혔지만 Global decision은 막지 못함). `_latest_odom_receipt_time`
> 기반 별도 freshness 체크와 `_effective_localization_confidence()`(stale이면
> 0.0)를 추가해 기존 `HierarchyCoordinator`의 degraded-localization state
> machine에 올바른 confidence를 흘려보내도록 수정(`hierarchical_environment_node.py`도
> 동일 수정). (2) `long_horizon_benchmark.py`의 manifest hash(`content_hash`/
> `occupancy_hash`)가 문서상 "재현성 검증용"이라면서 실제로는 비교되지
> 않던 문제 -- `_verify_scenario_regeneration()`을 추가해 world 재생성
> 직후 hash mismatch 시 fail-fast하도록 수정. Round 3 이후 Docker 전체
> 회귀 테스트 1794개, `colcon test` 1799개가 오류·실패·스킵 없이
> 통과했고, 26/26 profile이 유효하다.
>
> **Round 4 (code review, same date).** stale localization이 activation은
> 막아도 `_maybe_select_next_subgoal()`의 observation 생성/action 선택/
> `coordinator.enqueue_subgoal()`은 막지 못하던 behavioral gap을 발견해
> 수정했다 -- `activate_next_subgoal()`은 degraded 상태에서 큐를 pop하지
> 않으므로, odom이 끊긴 매 tick 새 stale candidate가 큐에 계속 쌓이고
> localization 회복 시 가장 오래된(가장 stale한) candidate가 FIFO로 먼저
> activate될 수 있었다. `_maybe_select_next_subgoal()` 시작부에
> `_localization_valid()` 체크를 추가해 stale이면 전체 Global 결정을
> 건너뛰고 degraded gate만 갱신하도록 수정(두 노드 모두). Round 4 이후
> Docker 전체 회귀 테스트 1796개, `colcon test` 1801개가 오류·실패·스킵
> 없이 통과했고, 26/26 profile이 유효하다.

### 9.1 목표

- 장거리 route history를 compact graph로 유지한다.
- dead-end와 실패 branch를 명시적으로 기억한다.
- Local controller의 실제 가능성과 위험을 Global candidate 선택에 반영한다.

### 9.2 신규 모듈

```text
hunter_kinodynamic_rl/navigation/memory/
├── __init__.py
├── topological_graph.py
├── node_manager.py
├── route_history.py
└── dead_end_detector.py

hunter_kinodynamic_rl/navigation/hierarchy/
└── feasibility.py

hunter_kinodynamic_rl/navigation/global_rl/
└── feasibility_predictor.py
```

### 9.3 Topological graph

Node 생성 이벤트:

- 마지막 node에서 설정 거리 이상 이동
- 큰 heading 변화
- 새로운 junction 발견
- room/corridor entrance 감지
- subgoal 도달
- local failure
- dead-end 감지

Node feature:

```text
mission pose
visit count
available/free direction descriptor
dead-end flag
failure count/reason
local risk statistics
final-goal distance/bearing
local visual/LiDAR descriptor
last-visited time
```

Edge feature:

```text
traversal count
path length
elapsed time
mean/max risk
success/failure count
last traversal direction
blocked status
```

Loop closure 또는 localization pose correction을 고려해 node ID와 pose를 분리하고, pose update API를
제공한다.

### 9.4 Dead-end 처리

dead-end 판정은 하나의 heuristic에 의존하지 않고 다음 evidence를 조합한다.

- known-free topology degree
- frontier 부재
- local progress 정체
- 반복 emergency stop
- 선택 가능한 subgoal mask 감소
- 직전 junction으로 되돌아가는 route history

첫 dead-end 확인은 정상 exploration event로 저장한다. 같은 branch 재진입만 repeated-dead-end로
계산한다.

### 9.5 Global observation 연결 순서

1. visited/failure raster channel만 사용
2. 최근 또는 인접 N개 node를 fixed-length tensor로 추가
3. node validity mask 추가
4. 성능 이득이 확인된 후 GNN encoder 도입

초기부터 GNN을 도입하면 map encoder, hierarchy, graph 생성 오류를 동시에 디버깅해야 하므로 피한다.

### 9.6 Local feasibility feedback

각 Global candidate에 다음 feature를 계산한다.

- short Ackermann rollout collision 여부
- steering saturation ratio
- known occupied까지의 minimum clearance
- local policy가 생성한 초기 action의 predicted risk
- candidate 방향의 progress-preserving action 존재 여부
- 과거 동일 영역의 local success/failure 통계

Local risk critic은 `[local state, local trajectory action]`의 위험 모델이므로 Global
`[radius, angle]`에 직접 적용하지 않는다. candidate subgoal로 conditioned된 local action을 먼저 생성한
후 그 action을 평가한다.

### 9.7 Learned feasibility predictor

Heuristic feature가 안정화된 뒤 실제 option 결과를 label로 다음 predictor를 학습할 수 있다.

```text
input: local map crop + candidate subgoal + vehicle state + topology context
output: success probability + expected local steps + expected risk
```

Predictor label은 실제 Local option 결과에서 생성하며, simulator GT success 가능성을 inference input으로
사용하지 않는다.

### 9.8 Joint fine-tuning 정책

기본 실험은 다음 세 단계로 분리한다.

1. frozen Local + Global training
2. frozen Global + Local adaptation 검증
3. 낮은 learning rate의 alternating update

Global과 Local을 처음부터 동시에 업데이트하지 않는다. Local policy 변화가 Global transition dynamics를
계속 바꾸면 Global replay가 비정상적인 off-policy 데이터가 되기 때문이다.

Local checkpoint가 변경되면 Global replay/checkpoint manifest에 local policy generation/hash를 기록한다.

### 9.9 Ablation

```text
A. Local only / no memory
B. Global partial map
C. + visited map
D. + topological memory
E. + local feasibility feedback
F. + global risk feedback
G. full hierarchical system
```

각 ablation은 가능한 한 동일한 code path와 config flag를 사용한다.

### 9.10 테스트

```text
tests/test_topological_graph.py
tests/test_topological_node_manager.py
tests/test_dead_end_detector.py
tests/test_route_history.py
tests/test_global_local_feasibility.py
tests/test_feasibility_predictor.py
tests/test_hierarchical_ablation_profiles.py
```

필수 검증:

- node/edge 중복 생성 방지
- junction→dead end→junction 복귀 history 보존
- 첫 dead-end와 repeated dead-end 구분
- loop traversal 시 graph connectivity 유지
- candidate별 feature/action mask 정렬
- risk critic에 올바른 local action/state 입력
- graph tensor padding과 validity mask 처리
- profile flag가 실제 observation/network/checkpoint에 반영됨

### 9.11 완료 기준

- visited-only 대비 repeated dead-end와 revisit ratio가 감소한다.
- feasibility feedback으로 local failure rate가 감소한다.
- global risk feedback으로 success를 심각하게 떨어뜨리지 않으면서 collision/risk가 감소한다.
- ablation checkpoint와 결과가 서로 다른 architecture fingerprint로 구분된다.

---

## 10. Phase 6: Long-horizon 평가와 실차 배포

### 10.1 목표

- 장거리 navigation에 맞는 fixed benchmark와 metric을 구현한다.
- localization noise/drift에 대한 성능 민감도를 측정한다.
- Simulation coordinator와 같은 계약으로 실제 Hunter mission을 실행한다.

### 10.2 신규 모듈

```text
hunter_kinodynamic_rl/evaluation/
├── global_metrics.py
└── long_horizon_benchmark.py

hunter_kinodynamic_rl/nodes/
└── hierarchical_navigation_node.py

hunter_kinodynamic_rl/navigation/localization/
├── wheel_imu_backend.py
├── lidar_odom_backend.py
└── lio_adapter.py
```

### 10.3 Benchmark dataset

고정 benchmark world는 최소 다음 split을 가진다.

```text
ID geometry
OOD geometry
long dead-end
multiple loops
narrow Ackermann-feasible corridors
localization noise
localization drift
dynamic obstacles
```

각 scenario는 다음 metadata를 저장한다.

- immutable scenario ID와 seed
- world/config content hash
- full GT occupancy hash
- start pose와 relative final goal
- ground-truth shortest feasible path length
- dead-end/loop 수
- expected difficulty class
- sensor/dynamics/localization override

### 10.4 Global metric

기존 직선거리 기반 SPL과 별도로 obstacle-aware Global SPL을 구현한다.

```text
SPL_i = success_i * L*_i / max(L_i, L*_i)

L*_i: GT map에서 계산한 shortest feasible path
L_i : 실제 odometry 기반 route length
```

필수 Global metric:

- Final Goal Success Rate
- Global SPL
- Total Route Length
- Excess Path Ratio
- Time to Goal
- Number of Global Subgoals
- Subgoal Success Rate
- Revisit Ratio
- Dead-End Entries
- Repeated Dead-End Entries
- Backtracking Distance
- Explored Area
- Unnecessary Exploration Ratio
- Local Planner Failure Count
- Global Replan Count
- Localization Drift Sensitivity

기존 Local metric은 함께 유지한다.

- collision rate
- minimum clearance
- uncensored TTC와 collision-free rate
- steering saturation/rate
- stopping margin
- local success rate
- unrecoverable state rate
- counterfactual safety gain

### 10.5 Localization 실험 순서

```text
Phase A: Gazebo ideal odometry
Phase B: i.i.d. pose noise
Phase C: OU drift와 latency
Phase D: wheel odom + IMU
Phase E: LiDAR odometry 또는 LIO
```

각 단계에서 navigation architecture는 동일하게 유지하고 backend/config만 교체한다.

### 10.6 실차 hierarchical node

`hierarchical_navigation_node.py`의 책임:

- localization backend 시작 및 health 확인
- 첫 유효 pose로 mission frame 초기화
- relative goal service/parameter 수신
- partial/visited/topological memory 초기화
- Global inference와 action mask
- Local inference와 physical command 생성
- safety guard 적용
- mission/subgoal 상태 publish
- localization confidence 저하 시 stop/recovery
- policy timeout, invalid subgoal, stale scan/odom 처리
- E-stop latch 호환

실차 node가 처리하지 않아야 할 책임:

- full GT map 사용
- simulator obstacle list 사용
- benchmark 전용 privileged label 사용
- localization backend 내부 알고리즘 구현

### 10.7 실차 안전 조건

- maximum speed clamp
- steering bound/rate limit
- NaN/Inf action rejection
- stale LiDAR/odometry/localization timeout
- Global/Local policy inference timeout
- command watchdog
- invalid/all-masked subgoal fallback
- proximity emergency stop
- localization confidence 기반 stop/recovery
- E-stop compatibility
- dry-run/replay mode에서 actuation 완전 차단

### 10.8 실차 적용 순서

1. Gazebo hierarchical end-to-end
2. recorded rosbag dry-run
3. 실제 센서 연결 + wheel off-ground test
4. 저속 제한 구역 시험
5. 단순 corridor relative-goal mission
6. junction/dead-end mission
7. 장거리 mission
8. 필요 시 LiDAR odometry/LIO backend 교체

### 10.9 테스트

```text
tests/test_global_metrics.py
tests/test_long_horizon_benchmark.py
tests/test_localization_drift_evaluation.py
tests/test_hierarchical_navigation_node.py
tests/test_hierarchical_real_safety.py
tests/test_hierarchical_checkpoint_compatibility.py
```

Integration/E2E 검증:

- 동일 benchmark manifest에서 모든 baseline 실행
- GT shortest path와 실제 route length 단위 검증
- localization noise-free/noisy pose 정보 누출 검사
- inference timeout 시 즉시 stop
- localization backend failure 시 stop/recovery
- dry-run/replay mode에서 `/cmd_vel` 미발행
- rosbag에 mission/subgoal/map/risk/command/status 기록

### 10.10 완료 기준

- fixed unseen benchmark에서 최종 목표 success와 Global SPL을 재현 가능하게 측정한다.
- 모든 high-level/low-level ablation이 동일 scenario set을 사용한다.
- drift magnitude별 성능 저하 curve를 생성할 수 있다.
- rosbag dry-run과 제한된 실차 시험에서 safety condition이 검증된다.
- 최종 checkpoint manifest만으로 Global/Local policy와 observation/action 계약을 재구성할 수 있다.

---

## 11. 단계별 산출물 요약

| 단계 | 핵심 산출물 | 다음 단계 진입 조건 |
|---|---|---|
| Phase 1 | mission frame, localization API, partial/visited map | synthetic/Gazebo map 누적 및 좌표 변환 검증 |
| Phase 2 | local subgoal controller, hierarchy coordinator | Global RL 없이 subgoal sequence 수행 |
| Phase 3 | long-horizon procedural world와 GT metadata | solvable room/corridor/dead-end world 반복 생성 |
| Phase 4 | masked DQN, Global replay/reward/trainer | unseen seed에서 hierarchical MVP goal success |
| Phase 5 | topology, feasibility/risk feedback, ablation | revisit/local failure/risk 개선 확인 |
| Phase 6 | global benchmark, drift 평가, real node | 동일 benchmark 재현 및 안전한 실차 dry-run |

## 12. 권장 작업 단위

각 Phase는 하나의 큰 변경으로 합치지 않고 다음 단위로 나눈다.

- 순수 Python data structure/algorithm
- config schema와 validation
- ROS adapter/node
- unit tests
- Gazebo integration test
- documentation/profile/launch

예를 들어 Phase 1은 다음 순서로 나눈다.

1. mission-frame pure geometry와 tests
2. localization protocol과 odom adapter
3. ray tracing과 partial-map pure implementation
4. visited map과 failure map
5. ROS subscriptions/TF/RViz publish
6. Gazebo integration 검증

이렇게 하면 ROS/Gazebo 문제와 좌표·map 알고리즘 문제를 분리해서 진단할 수 있다.

## 13. 전체 품질 게이트

모든 단계에서 다음 조건을 지킨다.

- 기존 local-only unit/integration test 회귀 없음
- 모든 config field는 실제 consumer를 가지며 no-op flag를 허용하지 않음
- observation/action/replay/checkpoint schema versioning
- train/validation/test seed 완전 분리
- policy input과 privileged simulator data 경계 테스트
- physical 단위와 frame ID를 로그/메시지에 명시
- stale sensor/localization/policy failure는 silent fallback이 아니라 명시적 상태와 safe stop
- fixed benchmark 결과에 scenario/config/checkpoint hash 기록
- 실제 command와 nominal/guarded command를 구분해 기록

## 14. 최종 완료 정의

다음 조건을 모두 만족하면 상세 명세의 구현이 완료된 것으로 본다.

1. 임의의 출발 pose에서 사용자 relative goal이 mission frame에 고정된다.
2. policy가 full map을 보지 않고 LiDAR 기반 partial map만 사용한다.
3. UNKNOWN과 FREE가 명시적으로 구분된다.
4. Global RL이 event-driven robot-relative subgoal을 선택한다.
5. Local TQC가 `[kappa, v_ref, L]`로 subgoal을 안전하게 추종한다.
6. visited/topological memory가 실패한 branch의 반복 진입을 줄인다.
7. Local feasibility/risk가 Global candidate 선택에 반영된다.
8. room/corridor/junction/loop/dead-end unseen world에서 평가된다.
9. obstacle-aware Global SPL과 장거리 탐색 metric이 기록된다.
10. localization backend를 교체해도 navigation core를 수정하지 않는다.
11. Gazebo ideal/noisy/drifting localization 실험을 재현할 수 있다.
12. 실차 또는 실차 rosbag dry-run에서 safety guard와 mission lifecycle이 검증된다.

## 15. 구현 작업 프롬프트 작성 규칙

각 Phase 구현을 요청하는 프롬프트는 다음 규칙을 따른다.

- 해당 Phase의 목표, 필수 산출물, 완료 조건만 적고 전체 명세를 반복하지 않는다.
- 동일 요구를 표현만 바꿔 반복하지 않는다.
- 한 문단에 가능한 내용을 지나치게 많은 하위 목록으로 분리하지 않는다.
- section 사이에는 빈 줄 하나만 사용하고 불필요한 공백·장식·장문의 배경 설명을 생략한다.
- 작업 범위 안의 파일 탐색, 편집, build, test, profile validation, Docker 명령은 사용자에게 실행 여부를 묻지 말고 직접 수행하도록 지시한다.
- 구현 중 발견한 결함도 해당 Phase 완료에 필요한 범위라면 직접 수정하고 회귀 테스트를 추가하도록 한다.
- 기존 사용자 변경을 보존하고 shared interface, unrelated package, destructive command는 임의로 변경하지 않도록 한다.
- 사용자 판단이 반드시 필요한 요구 충돌이나 복구하기 어려운 외부 변경만 질문하도록 한다.
- 중간 결과만 제출하지 말고 코드, config, test, 문서, 설치 규칙까지 완료하도록 한다.
- 마지막에는 반드시 활성 Docker 환경에서 build와 전체 package test를 실행하도록 한다.
- 최종 응답은 변경 파일, 핵심 동작, Docker 검증 명령과 결과, 남은 제한만 짧게 보고하도록 한다.

Docker 검증은 저장소 `CLAUDE.md`의 Active Docker Environment를 기준으로 한다. 컨테이너 ID가 바뀔 수 있으므로 먼저
`docker ps`로 `DRL_Robot_Path_Planning` 컨테이너를 확인하고, 컨테이너 내부
`/root/DRL_Robot_Path_Planning/ros2_ws`에서 ROS Humble을 source한 후 실행한다.

## 16. Phase 1 구현 요청 프롬프트

아래 블록을 그대로 복사해 Phase 1 구현 작업에 사용한다.

```text
`hunter_kinodynamic_rl/docs/HIERARCHICAL_NAVIGATION_IMPLEMENTATION_PLAN.md`의 Phase 1을 완전히 구현해줘. 원본 요구사항은 `hunter_se_unknown_gps_denied_hierarchical_navigation_detailed_spec.txt`를 기준으로 하고, 기존 local-only 학습·평가·checkpoint 동작은 보존해.

구현 범위:
- `navigation/mission`: 시작 pose에 고정되는 `MissionFrame`과 relative final-goal `GoalManager`
- `navigation/localization`: timestamp, covariance, confidence, valid 상태를 가진 backend protocol과 odom/Gazebo adapter
- `navigation/mapping`: Bresenham LiDAR ray tracing, mission partial map, rolling crop, visited/failure map
- 명시적인 UNKNOWN/FREE/OCCUPIED channel과 OccupancyGrid/RViz 확인용 ROS adapter
- `mission`, `localization`, `mapping` config dataclass·loader·validation·defaults 및 Phase 1 확인 profile
- 필요한 `__init__.py`, node entrypoint, CMake/package dependency, 문서와 테스트

필수 계약:
- 사용자 `(gx, gy)`는 출발 순간 robot frame 기준이며 mission 동안 고정한다.
- map과 goal 계산은 raw world 좌표가 아니라 mission frame 계약을 사용한다.
- valid hit beam은 통과 cell을 FREE, endpoint를 OCCUPIED로 갱신한다. NaN/Inf/max-range beam은 occupied endpoint를 만들지 않는다.
- UNKNOWN은 `not observed`로 계산하고 FREE와 섞지 않는다.
- scan timestamp에 대응하는 유효 pose만 map에 반영하고 stale/invalid localization이면 update를 거부한다.
- visited는 robot footprint/설정 radius를 반영하며 count, failure, last-visit 상태를 보존한다.
- simulator GT obstacle/map은 policy map 생성에 사용하지 않는다.
- 새 기능은 opt-in으로 두고 기존 profile, service, observation/action 차원과 `drl_agent_interfaces`를 변경하지 않는다.

검증:
- mission transform/round-trip/arbitrary yaw, goal 고정, ray clipping, invalid range, channel 배타성, stale pose 거부, rolling crop, visited/failure map에 대한 unit test를 작성해.
- synthetic scan/odom으로 ROS 없이 mapping pipeline을 검증하고, 가능한 Docker 환경에서는 ROS node import·profile validation도 검증해.
- 기존 `hunter_kinodynamic_rl` 전체 테스트를 회귀 실행해.

작업에 필요한 안전한 범위 내 탐색·편집·build·test·Docker 명령은 나에게 묻지 말고 직접 실행해. 기존 사용자 변경은 보존하고 destructive command, 실차 actuation, unrelated package 변경은 하지 마. 막히면 먼저 안전한 대안을 모두 확인하고, Phase 1 완료에 반드시 필요한 사용자 결정이 있을 때만 질문해. 불필요하거나 반복되는 설명은 생략하고 section 사이 공백도 최소화해.

작업 마지막에는 반드시 `CLAUDE.md`와 `docker ps`로 활성 `DRL_Robot_Path_Planning` 컨테이너를 확인한 뒤 컨테이너 내부 `/root/DRL_Robot_Path_Planning/ros2_ws`에서 다음 순서로 검증해:
1. `source /opt/ros/humble/setup.bash`
2. `colcon build --packages-select hunter_kinodynamic_rl`
3. `source install/setup.bash`
4. 신규 Phase 1 test와 config/profile validation
5. `colcon test --packages-select hunter_kinodynamic_rl`
6. `colcon test-result --test-result-base build/hunter_kinodynamic_rl --verbose`

실패하면 원인을 수정하고 동일 검증을 다시 실행해. 최종 응답에는 구현 결과, 주요 변경 파일, Docker 검증 결과와 실제로 남은 제한만 간결하게 정리해.
```

## 17. 2026-09-01 세션: 정식 학습 준비 인프라 (요구사항 A-O)

> **Historical implementation report.** 당시 후속 audit에서 training-ready
> 판정이 철회됐고, 그 결함들은 이후 defect-fix pass에서 닫혔다. 최신 판정은
> section 19와 `docs/CURRENT_STATUS.md`를 따른다.
> 이 세션은 "정식 Local/Global 학습과 A-G/A-B benchmark를 사용자가 직접 실행할
> 수 있도록" 준비하는 것이 목적이었다 (연구 성능 검증이 아님). 아래는 요구사항
> A-O 각각의 당시 구현 보고다. 2026-09-01 후속 감사에서 좌표계, benchmark/
> promotion, feasibility snapshot, dry-run/formal gate 결함이 확인됐으므로 이 절의
> "완료" 표현을 현재 판정으로 사용하지 않는다. 최신 상태는
> `docs/CURRENT_STATUS.md`, 수정 전 실행 제한은
> `docs/RUNBOOK_HIERARCHICAL_NAVIGATION.md`를 참조한다.

- **A (Local preflight)**: `training/preflight.py` (`run_local_preflight`) —
  full-circle 검사, 실측 infeasible fraction 샘플링, resume/fresh 일관성,
  device/ROS 의존성, identity(아키텍처/training-contract fingerprint) 리포트.
  기존 checkpoint 자동 resume 방지와 distribution-mismatch strict 거부는
  이미 `training/trainer_base.py`에 구현돼 있었음을 확인 (수정 없이 유지).
- **B (Local benchmark + promotion)**: `evaluation/local_subgoal_benchmark.py`
  (manifest/metrics/artifact, subgoal_success/collision/timeout/
  infeasible_goal_rejection/high_risk_failure/termination_reason,
  성공 전용 time_to_goal vs 전체 time_to_termination), `evaluation/
  local_promotion.py` (`LocalAcceptanceConfig`/`config/local_acceptance.yaml`,
  `promote_local_checkpoint` — 실패/legacy checkpoint는 항상 거부).
- **C (Global 학습 준비)**: `training/hierarchical_preflight.py`,
  `nodes/hierarchical_train_node.py`의 `require_promoted_local`(기본 True,
  fresh 실행만 적용) 게이트, `hierarchical_navigation_node.py`/
  `hierarchical_environment_node.py` checkpoint 로드에 누락돼 있던
  `map_location`(CUDA-저장 checkpoint의 CPU 로드) 수정.
- **D (A/B formal benchmark)**: `evaluation/run_live_hierarchical_benchmark.py`의
  `formal=True` — heuristic Global 대체 금지, 20 scenario 미만 즉시 거부,
  `benchmark_kind="formal"` 강제. `evaluation/global_metrics.py`에 `loop_count`/
  `global_decision_rate` 지표 추가.
- **E (LocalFeasibilityEvaluator)**: `navigation/hierarchy/
  local_feasibility_evaluator.py` (`FrozenLocalFeasibilityEvaluator`) —
  3개 production 경로(`nodes/hierarchical_navigation_node.py`,
  `training/train_hierarchical_dqn.py`의 `HierarchicalTrainingLoop`,
  `evaluation/long_horizon_benchmark.py`의 `run_ablation_mission`)에 연결.
  evaluator 없이 E/F/G-tier ablation을 실행하면 즉시 fail-fast(이전에는
  `local_evaluator=None`으로 조용히 zero-fill). fallback은 별도 telemetry
  카운터로 기록된다. 단, 2026-09-02 재감사 결과 candidate tensor 자체는
  policy-conditioned 두 열을 0으로 채우고 validity 열이 없으므로, "risk=0으로
  위장되지 않음"은 artifact 해석에만 해당하며 network 입력에는 해당하지 않는다.
- **F (B/C 분리)**: `config.schema.GlobalRLConfig.include_visited_channel`
  (B=False, C=True) — `hierarchical_phase5_c.yaml` 신설, B/C가 이제 서로
  다른 `hierarchical_architecture_fingerprint`를 가짐 (이전에는 동일했음).
- **G (A-G aggregation)**: `evaluation/ablation_suite.py`
  (`run_ablation_suite`/`evaluate_acceptance_report`, D-vs-C/E-vs-D/
  F,G-vs-E 비교, `insufficient_data` 가드), `evaluation/
  run_live_ablation_suite.py`(live 드라이버).
- **H (Phase 3 live evidence)**: `evaluation/live_evidence_runner.py` —
  구조화 JSONL 이벤트 로그 + raw launch stdout/stderr 보존 + leftover
  process 검사.
- **I (localization sweep)**: `evaluation/localization_sweep.py` —
  pairing 검증(`ScenarioPairingError`) + drift-sensitivity aggregation +
  drift curve. **live 드라이버(노이즈 backend를 실제로 스윕하며 episode를
  생성하는 스크립트)는 아직 없음** — noise model(`WheelImuNoiseModel`/
  `LidarOdomNoiseModel`)은 이미 존재.
- **J (rosbag dry-run)**: `evaluation/rosbag_dry_run.py` —
  `inspect_bag`/`verify_required_topics`(실제 `rosbag2_py`) +
  `replay_message_stream`(순수 로직, mock 가능) + actuator-command-count=0
  성공 조건. `HierarchicalNavigationNode._publish()`의 dry_run 차단은
  이미 존재했음을 확인(수정 없음), 그 보장을 증명하는 테스트를 추가.
- **K (thread safety)**: `navigation/mapping/partial_map.py`의 락은 이미
  올바르게 모든 public method를 감싸고 있었음(수정 없음) — 실제 동시
  접근 테스트(`tests/test_partial_map_concurrency.py`)만 추가.
- **L (provenance)**: `evaluation/provenance.py`는 이미 nested-git-root를
  올바르게 처리(수정 없음). `ablation_suite`/`run_live_ablation_suite`
  결과에 `collect_package_provenance()` 출력을 추가.
- **M (runbook)**: `docs/RUNBOOK_HIERARCHICAL_NAVIGATION.md` 신설 — 1~12
  단계 전체, 실제 명령/입력/산출물/성공조건/실패 로그/resume/formal-vs-smoke.
- **N (테스트)**: 이 세션에서 신규/수정 테스트 파일 다수 추가 (아래
  "주요 변경 파일" 참조) — Docker 전체 회귀 `colcon test` 2002개
  (0 오류/실패/스킵, pytest 1997 + lint 5).
- **O (문서/legacy 표시)**: 이 섹션. `runtime/experiments/
  20260831_005836_kinodynamic_tqc_arbitrary_subgoal_seed0/LEGACY_INVALID.md`
  및 Global smoke checkpoint 4곳에 legacy 마커 추가(기존 파일은 삭제/이동
  없이 보존).

**의도적으로 실행하지 않은 것**: 정식 Local 학습(150k step), 정식 Local
benchmark(20+ scenario 실제 실행), 정식 Global 학습, 정식 A/B benchmark,
Phase 5 A-G 정식 학습/benchmark, localization sweep 실제 실행, 실제
rosbag dry-run(bag 없음). `docs/RUNBOOK_HIERARCHICAL_NAVIGATION.md`의
"요약" 표에 상태별로 정리했다.

## 18. 2026-09-01 후속 감사: 정식 학습 전 필수 수정

> **Historical audit.** 이 절이 지적한 결함은 이후 defect-fix pass에서 코드와
> 회귀 테스트로 닫혔다. 항목 자체는 결함 발견 이력으로 보존하며 현재 상태는
> section 19와 `docs/CURRENT_STATUS.md`가 대체한다.

이 절은 section 17의 구현 보고를 검토한 당시 판정이었다. 저장된 회귀 결과는 Docker `colcon
test-result` 기준 2,002 tests, 0 errors/failures/skips이고 관련 신규 테스트 87개도
별도로 통과했다. 그러나 이 수치는 pure-Python/ROS 회귀 증거이며 live policy 성능
또는 formal 연구 결과가 아니다.

### 18.1 차단 결함

1. **Local observation frame**: 계층 경로가 robot-relative subgoal과 odom/world
   `RobotState`를 함께 `build_robot_state_vector()`에 전달한다. 동일 frame 계약으로
   통일하고 비원점·회전 pose 회귀 테스트를 추가해야 한다.
2. **Local benchmark infeasible metric**: 전체 episode를 분모로 한
   `infeasible_goal_rejection_rate`와 0.5 acceptance는 목표 infeasible fraction 0.15와
   양립하지 않는다. timeout/collision을 rejection으로 부르지 말고 infeasible 조건부
   false-success/collision/high-risk 지표와 valid count로 바꿔야 한다.
3. **Promotion trust chain**: 빈 `promotion_manifest.json`도 promoted로 인정되며
   caller-supplied manifest와 identity를 신뢰한다. 실제 source generation/SHA,
   architecture/training-contract, supported artifact schema, manifest/episode 일치,
   provenance를 독립 검증하고 atomic promotion을 구현해야 한다.
4. **Feasibility temporal context**: 후보마다 동일 scan을 frame stack에 push해
   candidate order가 Local observation을 바꾼다. 한 Global decision의 모든 후보가
   immutable LiDAR-history/vehicle-state/previous-action snapshot을 공유해야 한다.

### 18.2 높은 우선순위 결함

- `predict_risk()`도 bounded timeout/error handling에 포함하고 evaluator validity와
  fallback telemetry를 replay/episode/formal artifact에 기록한다.
- 전진 전용 Local action과 항상 valid인 후방 BACKTRACK의 계약을 reachability mask,
  실제 recovery, reverse 또는 multi-arc 중 하나로 닫는다.
- rosbag dry-run의 성공 조건에 필수 topic, 실제 decision, replay 완료, 실제 actuator
  publish 0건을 모두 포함한다. real rclpy publisher를 mock `.published` 속성으로
  검증하지 않는다.
- live evidence runner가 유효한 Local checkpoint/profile로 시작하고 wall activation,
  teleport, sensor readiness, control tick과 소유 process teardown을 실제로 기록하게 한다.
- formal A/B는 test mode, strict trained checkpoint, promoted Local, profile 기본 budget을
  강제한다. formal A-G는 요청 label 누락을 skip하지 않고 실패한다.
- Phase 4의 Local tag `best`와 Phase 5/promotion의 `final`을 하나의 canonical tag로
  통일한다.
- localization sweep은 pairing/aggregation뿐 아니라 동일 manifest를 실제 backend별로
  실행하는 live driver와 condition provenance를 제공해야 한다.

### 18.3 수정 후 검증 순서

> **2026-09-02 R0 실행:** 1–4단계와 Local save/resume, teardown,
> release tag를 완료했다. 5단계 Global live smoke는 legacy/unpromoted
> `local_frozen`을 preflight가 차단해 보류했다. acceptance/promotion 없이
> smoke checkpoint를 강제 승격시키지 않았다.

1. frame/benchmark/promotion/evaluator/dry-run/formal gate negative regression tests
2. 관련 targeted pytest
3. 전체 pytest와 package build/colcon test
4. Local 1 episode 또는 수십 step 이하의 bounded smoke
5. Global 1 mission, option 1~2개, optimizer update 1회, save/resume 1회의 bounded smoke
6. leftover process와 raw/structured evidence 확인

정식 Local 150k 학습, Global 1000+ mission, 20+ scenario formal A/B, A–G,
localization sweep와 실차 시험은 위 차단 결함을 닫기 전 실행하지 않는다.

### 18.4 연구 검증 순서

코드 게이트를 닫은 뒤 Local을 먼저 검증한다.

```text
Local direct-control/fixed-L/path baseline
    -> rollout-risk calibration
    -> counterfactual contribution
    -> raw/guarded safety attribution
    -> Hunter SE Local trial
    -> frozen promoted Local
    -> Global map/memory/feasibility/risk ablation
    -> large-layout/drift/real-Hunter hierarchy
```

Phase 완료 판정과 연구 주장에 필요한 통계·baseline은
`docs/RESEARCH_PROTOCOL.md`, artifact/metric 계약은 `docs/BENCHMARK.md`, 현재 한 줄
판정은 `docs/CURRENT_STATUS.md`를 따른다.

## 19. 2026-09-01 연구 로드맵 반영: Local-first, capability-aware Global

이 절은 기존 Phase 1~6 구현 계획을 폐기하지 않고 연구 우선순위를 재배치한다.
알고리즘 수식과 전체 ablation은 `docs/RESEARCH_ROADMAP.md`, 실험/통계 계약은
`docs/RESEARCH_PROTOCOL.md`가 정본이다.

### 19.1 의존성 변경

기존 구현 Phase 1~6은 hierarchy 기능의 engineering 단계다. 앞으로의 연구 실행은
다음 순서를 강제한다.

```text
Phase R0  corrected baseline release freeze
    |
Phase R1  current Local L0-L5 formal baselines
    |
Phase R2  multi-task risk ensemble + calibration/OOD
    |
Phase R3  physics + residual dynamics ensemble
    |
Phase R4  progress-preserving uncertainty-gated counterfactual Local
    |
Phase R5  promoted Local checkpoint immutable freeze
    |
Phase R6  capability distribution + experience-aware Global
    |
Phase R7  localization covariance propagation + GPS-denied evaluation
```

Global R6/R7은 R5의 immutable Local generation 없이 시작하지 않는다. Global 학습
중 Local actor/risk/residual weight, normalization 또는 calibrator를 변경하면 새로운
Local generation으로 간주하고 Global 전체를 다시 학습한다.

### 19.2 Global candidate capability 계약

현재 geometry/feasibility/risk feature를 다음 candidate vector로 확장한다.

$$
C_i=[P_{success},E[R],U[R],E[progress],E[T_{execute}],
P_{stop},P_{unrecoverable}].
$$

추가 규칙:

- 한 Global decision의 모든 candidate는 동일한 immutable Local temporal snapshot을
  사용한다.
- risk/residual ensemble member와 calibrator generation을 Global artifact에 기록한다.
- timeout/error/non-finite는 `valid=false`와 reason을 갖는 unknown이며 0으로 채우지 않는다.
- `P_success`, risk mean과 uncertainty는 held-out Local benchmark에서 calibration한다.
- BACKTRACK/recovery candidate는 single forward arc로 허위 progress를 만들지 않고
  실제 recovery execution model 또는 명시적 unknown capability를 사용한다.

Network는 masked Dueling Double DQN을 유지하고 candidate-conditioned value
estimator로 해석한다.

위 validity 규칙은 목표 schema다. 현재 구현은 fallback reason을 artifact에는
기록하지만 candidate tensor에는 validity 열이 없고 두 policy-conditioned 값을
0으로 채운다. schema 확장 전 formal E/F/G는 raw action/risk fallback count가
모두 0인 gate를 적용한다. 기존 `fallback_rate`는 decision/candidate 단위가 섞인
분모이므로 formal acceptance에 사용하지 않는다.

$$
Q(s,c_i)=MLP([Encoder_{map}(M),Encoder_{memory}(H),
Encoder_{candidate}(C_i),z_{state}]).
$$

DDQN 교체는 우선순위가 아니다. 비교 대상은 geometry-only, oracle capability,
learned mean capability, learned mean+uncertainty다.

### 19.3 Experience-Aware Topological Memory

현재 `TopologicalGraph`는 이미 traversal/success/failure count, path length,
elapsed time, mean/max risk, last direction과 blocked 상태를 edge별로 보존한다.
다음 목표는 이 저장값을 Global observation에 직접 노출하고 uncertainty/recency
posterior까지 확장하는 것이다. 목표 vector 예시는 다음과 같다.

$$
e_{ij}=[N_{visit},N_{success},N_{fail},\bar R,\bar T,\bar U,t_{last}].
$$

성공률은 작은 표본에서 과신하지 않도록 Beta prior 등의 smoothed posterior를
사용한다. 구현 전 다음 계약을 확정한다.

- node/edge merge 시 count와 moment 병합법;
- loop closure 또는 map correction 시 edge identity;
- 환경 변화에 대한 time decay와 `t_last` 의미;
- failure reason별 분리와 Local generation이 바뀔 때 경험 통계의 유효성;
- replay/checkpoint schema migration과 architecture fingerprint.

### 19.4 Localization-aware risk

scalar localization confidence만 candidate feature로 사용하지 않고
$x\sim\mathcal N(\hat x,\Sigma_x)$에서 pose sample을 생성해 residual-dynamics
ensemble과 함께 rollout한다. Global에는 expected risk, uncertainty와 선택적인
quantile/CVaR를 전달한다.

현재 `WheelImuLocalizationBackend`의 confidence/covariance 성장만으로 실제 drift
robustness를 주장하지 않는다. R7 전에 실제 pose error가 발생하는 injection/backend,
GT 대비 localization error logging과 covariance calibration을 구현해야 한다.

### 19.5 Global ablation

| Label | 구성 |
|---|---|
| G0 | frozen Local only |
| G1 | partial map |
| G2 | G1 + visited |
| G3 | G2 + topology/dead-end memory |
| G4 | G3 + Local success probability |
| G5 | G4 + Local expected risk |
| G6 | G5 + Local risk uncertainty |
| G7 | G6 + localization-aware risk |
| G8 | G7 + complete capability/experience model |

이 label은 기존 Phase-5 A–G를 소급해 이름만 바꾸는 표가 아니다. 각 learned row는
고유 config/fingerprint와 독립 training checkpoint를 가져야 하며, 동일 immutable
test manifest에서 classical frontier/A*/D* Lite 또는 Hybrid-A* 비교도 수행한다.

### 19.6 단계별 완료 게이트

| 단계 | 완료 게이트 |
|---|---|
| R0 | 전체 회귀 + bounded live Local/Global save-resume + frozen benchmark + release provenance (2026-09-02: release/Local 완료, Global은 promoted Local 대기) |
| R1 | L0–L5 multi-seed baseline, calibration/guard attribution 원시 artifact |
| R2 | risk factor별 calibration, uncertainty-error/coverage, ID/OOD 성능 |
| R3 | nominal/single residual/ensemble의 one-step·multi-step·risk error 비교 |
| R4 | progress/feasibility/uncertainty constraint와 abstention ablation, Local 실차 |
| R5 | 사전 등록 promotion 기준, immutable Local/risk/residual/calibrator generation |
| R6 | G0–G8 독립 학습, capability와 experience memory 원인 분리 |
| R7 | 실제 pose-error drift curve, covariance calibration, long-route/실차 hierarchy |

### 19.7 의도적으로 보류

TQC 교체, diffusion policy, VLM/VLA/RGB, 무조건적인 GNN 도입, 추가 reward shaping은
현재 핵심 가설의 선행 조건이 아니다. Local risk/residual/counterfactual과 frozen-
Local capability 연구를 완료한 뒤 필요성을 재평가한다.
