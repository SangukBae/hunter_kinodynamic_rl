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

> **구현 상태: 완료 (2026-08-28).** Mission frame, relative final goal, timestamp-synchronized
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

> **구현 상태: 완료 (2026-08-28).** `LocalPolicyController`(local_rl),
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

> **구현 상태: 완료 확정 (2026-08-28).** `long_horizon_world`/`long_horizon_generator`/
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
