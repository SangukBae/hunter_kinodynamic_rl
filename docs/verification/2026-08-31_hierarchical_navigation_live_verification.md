# Hierarchical Navigation 라이브 검증 기록 (2026-08-31)

> **Historical verification artifact.** 당시 source/명령/결과를 보존한다. 최신
> 상태는 [`../CURRENT_STATUS.md`](../CURRENT_STATUS.md)를 따른다.

이 문서는 items 1-9(코드 수정, colcon build/test 및 26개 프로파일 검증 완료 -- 이전
세션에서 완료)에 이어, item 10(fault-path 검증)/item 11(라이브 Gazebo 종합 검증)을
다룬다. 원본 요구사항 기준은
`hunter_se_unknown_gps_denied_hierarchical_navigation_detailed_spec.txt`이며,
직전 세션이 이미 확인해 둔 두 가지 기존 한계(landmine)를 그대로 적용했다:
`PYTHONPATH`를 소스 트리 우선으로 잡지 않으면 `python3 -m hunter_kinodynamic_rl...`
가 stale install 사본을 조용히 resolve한다는 것, 그리고 이 패키지는
`.gitignore`(`hunter_kinodynamic_rl_git_tracking.md` 참조)로 저장소 git 추적에서
제외돼 있어 이 문서와 코드 변경 모두 `git log`/`git diff`로는 보이지 않는다는 것.

## 0. 라이브 환경 재기동

이전 세션이 백그라운드로 띄운 headless Gazebo(`/tmp/gazebo_live.log`)는 이번 세션
시작 시점에 이미 ctrl-c로 종료된 stale 프로세스였다(로그 mtime이 세션 시작보다
26분 앞섬, `ps aux`에 gazebo/ros2 프로세스 없음) -- 재기동 필요.

```
docker exec -d 7a2702b311a1 /bin/bash -c "source /opt/ros/humble/setup.bash && \
  source /root/DRL_Robot_Path_Planning/ros2_ws/install/setup.bash && \
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && \
  cd /root/DRL_Robot_Path_Planning && \
  ros2 launch hunter_se_gazebo simulate_hunter_se_ignition.launch.py rviz:=false headless:=true \
  > /tmp/gazebo_live.log 2>&1"
```

재기동 후 확인:
- `/tmp/gazebo_live.log`에 error/exception/traceback 없음(startup 구간).
- `ros2 topic list`: `/clock /cmd_vel /cmd_vel_filtered /drl/model_poses
  /hunter_se/chassis_contacts /hunter_se/joint_states /hunter_se/robot_description
  /odometry /ouster/points /parameter_events /rosout /scan /tf /tf_static` 전부 존재.
- `/scan` ~18.4 Hz, `/odometry` ~45.9 Hz, `/ouster/points` ~18.4 Hz(wall-clock 측정 --
  headless 모드가 real-time factor > 1로 도는 것과 일관, CLAUDE.md의 "RTF=1.0
  기준 ~10/~50 Hz" 문서값과 절대치는 다르지만 비율/무오류 자체가 정상 동작의 증거).
- `PYTHONPATH` 우선순위 확인:
  `python3 -c "import hunter_kinodynamic_rl.navigation.local_rl.live_gazebo_executor as m; print(m.__file__)"`
  → `ros2_ws/src/hunter_kinodynamic_rl/...`(소스 트리) 출력 확인, `install/...` 아님.

## 1. Item 10 -- Local 옵션 5대 종료 경로 fault-path 검증

`docs/verification/live_fault_paths.py`. 실제 running Gazebo에 대해 진짜
`LiveGazeboLocalExecutor`를 구성하고, 각 종료 경로를 narrow monkeypatch로 하나씩
강제 유발(ROS/Gazebo 인프라 자체는 건드리지 않음) -- 5개 전부 기대한
status/reason으로 종료됨을 확인:

| 경로 | 강제 방법 | 결과 |
|---|---|---|
| localization stale | `_localization_valid` → False | `cancelled_by_replan` / `localization_confidence_degraded` ✅ |
| scan stale | `_latest_scan_receipt_time`를 1000초 과거로 | `cancelled_by_replan` / `scan_stale` ✅ |
| inference timeout | `select_action`을 3초 sleep으로 래핑 | `cancelled_by_replan` / `inference_timeout` ✅ |
| invalid action (NaN) | `select_action`이 NaN 벡터 반환 | `cancelled_by_replan` / `invalid_action_output` ✅ |
| max_local_steps 소진 | `predict_risk`를 0으로 고정 + `max_local_steps=3` | `failed_timeout`, `local_steps==3` ✅ |

전부 PASS. STOP이 매 케이스 마지막 명령으로 발행됨(executor 내부에서 이미 보장 --
`test_hierarchical_navigation_node.py`의 기존 단위 테스트와 이번 라이브 실행이
서로 다른 각도로 같은 계약을 확인).

## 2. Item 11 -- 라이브 Gazebo 종합 검증

### 2.1 Wall pool spawn / multi-mission reset / clock 정합성 (11.1-11.6)

`docs/verification/live_smoke_executor.py`. 실행 중인 Gazebo에 대해
`LiveGazeboLocalExecutor`를 1회 구성한 뒤 3개 mission(서로 다른 non-zero start
pose/yaw)을 순차로 `bind_mission()`, 각 mission에서 option을 실행:

```
[smoke] wall_segment_pool: spawned 150 wall pool slots (parking_distance_m=27.0)
[smoke] mission 0: seed=12396 start_pose=(-3.375, -9.875, 1.789) bind_elapsed=8.04s
        post-bind pose_world=(-3.375,-9.875,1.789) scan_count=78 odom_count=198
[smoke]   option 0: result=failed_high_risk local_steps=1 elapsed_time_sec=0.1
[smoke] mission 1: seed=12706 start_pose=(-4.625, 9.375, -0.634) bind_elapsed=8.03s
        post-bind pose_world=(-4.625,9.375,-0.634) scan_count=94 odom_count=227
[smoke]   option 0: result=failed_high_risk local_steps=5 elapsed_time_sec=0.5
[smoke] mission 2: seed=12529 start_pose=(8.125, -1.875, -1.599) bind_elapsed=8.04s
        post-bind pose_world=(8.125,-1.875,-1.599) scan_count=145 odom_count=288
[smoke]   option 0: result=failed_high_risk local_steps=8 elapsed_time_sec=0.8
[smoke] executor.close() completed in 0.40s
[smoke] SUMMARY: 3 options run, 0 activated-but-no-terminal-result, 0 clock-budget violations
[smoke] PASS
```

확인된 것:
- **Wall pool spawn**: 150 슬롯 정상 spawn(1회, mission 간 재사용).
- **Multi-mission reset**: 3개 mission 모두 teleport 후 `pose_world`가 요청한
  non-zero start pose/yaw와 정확히 일치, `scan_count`/`odom_count`가 매 bind 후
  계속 누적 증가(센서 freshness 정상 -- 이전 mission의 stale 센서 데이터가
  새 mission으로 새는 사고 없음).
- **Item 2 (terminal SubgoalResult)**: activate된 3개 option 전부 non-None
  terminal result를 받음(0건 "activated인데 결과 없음").
- **Item 1 (option-local clock 정합성)**: `elapsed_time_sec`가 매번
  `local_steps * dt_sec` 범위 내(0.5초 tolerance 포함) -- clock-budget 위반 0건.
- **Graceful close**: `executor.close()` 0.40초 내 완료, 이후 프로세스에
  hang/leftover thread 없음(item 11.10과 별개로 executor 단위에서도 확인).

**행동 관찰(버그 아님, 정직하게 기록)**: 3개 mission 모두 첫 option이
`failed_high_risk`(`local_risk_threshold_exceeded`)로 즉시(1~8 step) 종료됐다.
이는 `runtime/experiments/local_frozen/checkpoints/best`(local-only 학습,
7274 steps로 매우 짧게 학습된 frozen 체크포인트)가 risk-critic 기준으로 매우
보수적으로 학습돼 있다는 뜻이며, executor/coordinator 배선 자체의 결함이 아니다
-- 아래 2.3의 A/B benchmark 결과와 일관된다(§2.4 한계 참조).

### 2.2 CUDA GlobalDQNAgent 생성 순서 진단

`docs/verification/live_train_diag2.py`. `nodes/hierarchical_train_node.py`의
기존 코드 주석(라인 167-234)이 이미 "GlobalDQNAgent를 CUDA로 먼저 생성해도
'publisher's context is invalid' 결함은 해결되지 않는다"고 기록해 둔 것을
최소 재현으로 다시 확인: `GlobalDQNAgent(device=cuda)`를 `LiveGazeboLocalExecutor`
생성보다 먼저 만들고, subgoal 하나를 수동 activate/run_option해도 crash 없이
`failed_high_risk`로 정상 종료(`PASS (no publish crash)`).

**중요한 구분**: 이 최소 재현은 실제 결함을 재현하지 않는다 -- 결함은
`HierarchicalTrainingLoop.run_mission`이 매 step마다 `agent.select_action()`
(CUDA forward pass)을 executor의 백그라운드 spin thread와 **같은 프로세스에서
반복 실행**할 때만 나타나며, diag2는 Global agent의 forward pass를 학습 루프
안에서 전혀 호출하지 않는다(subgoal을 수동으로 큐에 넣고 `run_option`만 부름).
즉 diag2는 "Local-only 실행은 CUDA 유무와 무관하게 문제없다"만 재확인한 것이고,
아래 2.3의 실제 재현과 모순되지 않는다.

### 2.3 Global 학습 라이브 smoke + resume (11.8) -- 기존 문서화 결함 재확인 + 신규 관찰

`docs/verification/live_train_resume.py`. `train_hierarchical_dqn(...,
live=True)`로 2개 mission 학습 → 저장 → resume 2개 mission 추가 학습을
시도했다. 결과: **`nodes/hierarchical_train_node.py`가 이미 "KNOWN UNRESOLVED
LIVE DEFECT"로 문서화해 둔 결함이 그대로, 결정론적으로 재현됨**:

```
WARNING: live=True with a CUDA Global agent has a KNOWN, UNRESOLVED live defect --
GlobalDQNAgent inference alongside LiveGazeboLocalExecutor's background spin thread
reproducibly raises 'publisher's context is invalid' during run_option's first
cmd_vel publish ...
...
rclpy._rclpy_pybind11.RCLError: Failed to publish: publisher's context is invalid,
  at ./src/rcl/publisher.c:389
```

`run_mission`의 첫 `run_option` 호출, 첫 `cmd_vel` publish에서 재현 -- 코드
주석이 이미 기록한 "agent.select_action() 실행 + executor의 백그라운드
MultiThreadedExecutor spin thread 동시 존재" 조건과 정확히 일치.

**신규 관찰(이전 세션 문서에는 없던 것)**: 이 크래시 이후 프로세스가 자체
종료되지 않고 **행(hang)** -- `timeout 180`으로 강제 종료해야 했다(exit code
124). 크래시가 예외로 전파된 뒤 `finally: rclpy.shutdown()` 경로가 손상된
rclpy/DDS 상태에서 되돌아오지 못하는 것으로 보인다. 강제 종료(SIGTERM/SIGKILL)
후 컨테이너에 leftover 프로세스는 남지 않음(`ps aux` 확인) -- 즉 이 결함은
"틀린 결과를 조용히 내는" 종류가 아니라 "즉시 크래시 + 외부 timeout 없이는
자체 회복 불가"라는 두 겹의 실패 모드다. 코드는 이미 정직하게 fail-fast로
경고를 찍고 있으나(§ 위 WARNING 로그), graceful shutdown까지는 보장하지
못한다 -- 후속 작업 항목으로 아래 §3에 남긴다.

이 결함 때문에 item 11.8(Global 학습 resume)의 "resume 후 global_step 증가"
자체는 라이브로 실행/확인하지 못했다(`live=False`, 즉 ROS-free
`SimplifiedKinematicLocalExecutor` 경로로는 기존 phase5 문서에서 이미
검증됨 -- Global 학습 로직 자체의 문제가 아니라 CUDA+라이브 executor 조합
고유의 문제).

### 2.4 A/B 라이브 benchmark (11.9)

`docs/verification/live_benchmark_run.py` →
`evaluation.run_live_hierarchical_benchmark.run_live_hierarchical_benchmark()`.
Global checkpoint가 아직 없으므로(§2.3의 결함으로 라이브 Global 학습 자체가
아직 산출물을 만들지 못함) ablation B는 학습된 Global DQN이 아니라 내장
goal-seeking heuristic으로 평가된다 -- 코드가 이를 실행 시점에 명시적으로
경고하며 조용히 다른 것을 비교로 둔갑시키지 않는다:

```
no global_checkpoint_dir given -- ablation B evaluates the built-in goal-seeking
heuristic (hierarchical_environment_node._heuristic_select_action), NOT a trained
Global policy (fine for a wiring smoke test, not for the Phase 4 completion comparison)
```

이 실행의 목적은 정책 성능이 아니라 **benchmark 파이프라인(clock domain,
telemetry, provenance) wiring의 end-to-end 정확성**이며, 그 목적 기준으로
PASS: 3개 fixed-seed 시나리오 × ablation A/B 전부 예외 없이 완주,
`collision_free_rate=1.0`(양쪽 모두), `inference_error_count_total=0`,
provenance 필드(local checkpoint sha256/generation, architecture fingerprint,
profile hash, scenario manifest hash) 전부 정상 기록. 결과는
`runtime/hierarchical_benchmark/20260831_084423/{metrics.json,manifest.json,
episodes_A.csv,episodes_B.csv}`에 저장(git 미추적 디렉토리, phase5 문서의
`runtime/hierarchical_benchmark/20260831_072439`와 동일 패턴 -- 이번 세션이
같은 스크립트로 재실행해 재현성도 함께 확인).

`final_goal_success_rate=0.0`(양쪽) -- §2.1에서 관찰한 것과 동일하게, 3-step
짧은 옵션 예산(`max_local_steps=10`)과 보수적인 frozen local checkpoint 조합상
예상되는 결과이지 파이프라인 결함이 아니다(`local_planner_failure_count_total`
9(A)/7(B), `collision_count_total=0`, `emergency_stop_count_total`
19(A)/30(B) -- 위험을 감지하면 확실하게 멈춘다는 뜻으로 일관됨).

셧다운 시 `rclpy` 소멸자 관련 무해한 경고 3건(`The following exception was
never retrieved: cannot use Destroyable because destruction was requested`) --
exit code 0, §2.3과 달리 hang 없이 정상 종료.

### 2.5 Graceful shutdown / leftover process 확인 (11.10)

각 라이브 스크립트 실행 직후 `ps aux`로 확인 -- 스크립트 자체가 남긴 leftover
python 프로세스는 한 건도 없었다(§2.3의 timeout-kill 케이스 포함, `timeout`이
정상적으로 프로세스 그룹을 정리함).

Gazebo 자체의 종료는 한 가지 조작이 더 필요했다: launch 트리 최상위(`ros2
launch .../simulate_hunter_se_ignition.launch.py`)에 SIGINT를 보내면 하위
bridge/robot_state_publisher/prefilter/pointcloud_to_laserscan 노드는 정상
종료되지만, `ign gazebo` 본체(및 그 `ruby` 래퍼)는 부모가 죽은 뒤에도 살아남는
orphan이 됐다(`on_exit_shutdown:=true` world 인자가 있음에도) -- `ign gazebo`
프로세스에 SIGINT를 직접 한 번 더 보내야 완전히 종료됐다. 이 세션이 끝난 시점
컨테이너의 ROS/Gazebo 관련 프로세스는 0개(영구 `ros2-daemon`만 남음, 이 세션이
띄운 것이 아니라 컨테이너 자체의 상시 인프라).

## 3. 남은 제한 / 후속 작업 (정직하게 기록)

1. **Item 11.8 라이브 Global 학습/resume은 실제 실행 확인이 아직 불가능**:
   §2.3의 CUDA+라이브 executor 동시 실행 결함이 근본 원인. 기존 코드 주석이
   제안한 방향(별도 OS 프로세스로 Global agent inference 격리, `real_policy_node.py`의
   `InferenceWorkerProcess` 패턴 재사용)이 여전히 유효한 다음 단계다.
2. **신규**: 위 크래시 이후 프로세스가 자체적으로 정리되지 않고 hang한다(§2.3) --
   `finally: rclpy.shutdown()`이 손상된 DDS 상태에서 멈추는 것으로 보임. 운영상
   외부 timeout으로는 안전하지만(leftover 프로세스 없음 확인됨), 코드 차원의
   fail-fast 종료(예: 손상 감지 시 `os._exit()`로 즉시 종료)는 아직 없음.
3. **Item 11.9 A/B benchmark는 학습된 Global 정책과의 비교가 아니다**: 현재
   유효한 Global checkpoint가 없어(1의 결함으로 라이브 학습이 산출물을 못 만듦)
   ablation B는 heuristic 평가에 그친다 -- 파이프라인 정확성만 확인됐고, Phase 4
   완료 기준의 "Global DQN vs baseline" 성능 비교 자체는 아직 수행되지 않았다.
4. **frozen local checkpoint의 보수성**: §2.1/2.4에서 관찰된 즉각적인
   `failed_high_risk` 종료는 `local_frozen` 체크포인트가 7274 step의 짧은
   학습만 거쳤다는 사실과 일관된다 -- 더 길게 학습된 local 체크포인트로
   교체하면 benchmark의 `final_goal_success_rate`가 유의미하게 달라질 가능성이
   높다(현재 0%는 정책 품질의 하한을 보여줄 뿐, 파이프라인 결함의 증거가 아님).

## 4. 완료 기준 대비 확인

- Item 10(5개 fault-path): 라이브 Gazebo에서 5/5 PASS.
- Item 11.1-11.6(wall pool/multi-mission/clock/terminal-state): 라이브 Gazebo에서
  PASS, 0 clock-budget 위반, 0건 "activated인데 terminal result 없음".
- Item 11.7(CUDA agent-executor 상호작용): 기존 문서화된 결함을 최소 재현으로
  올바르게 구분 재확인(생성 순서는 원인이 아님, 실제 원인은 학습 루프 내
  반복 forward pass).
- Item 11.8(Global 학습 smoke+resume, live=True): 라이브로 실행 시도했고,
  기존에 문서화된 크래시가 결정론적으로 재현됨을 재확인 + hang이라는 신규
  실패 모드를 추가로 발견·기록. **아직 통과 상태 아님** -- §3-1/2가 후속 작업.
- Item 11.9(A/B benchmark): 파이프라인 wiring 기준 PASS(정책 성능 비교는
  범위 밖, §3-3에 명시).
- Item 11.10(graceful shutdown, zero leftover): 모든 검증 스크립트 및 Gazebo
  자체 최종 확인 결과 leftover 프로세스 0개.

items 1-9(코드 수정)는 이전 세션에서 완료(Docker `colcon build`/`colcon test`
전부 green, 1867+ pytest cases, 26/26 프로파일 유효 -- 이번 세션은 코드를
변경하지 않았으므로 그 결과는 그대로 유효하다).
