# TRACTOR Simulation Environment v2

상태: **IMPLEMENTED / CONFIGURED / NOT TRAINED / NOT FORMALLY BENCHMARKED**
계약 ID: `tractor_env_v2`
기준일: **2026-09-08 KST**

## 1. 목적과 호환성

`tractor_env_v2`는 Hunter SE Local policy가 단순 원통·등속 장애물에 과적합되지 않도록 만든
opt-in 학습 환경이다. 현재 formal comparison은 `tractor_protocol_v2`와 별도
`tractor_scenario_plan_v2`의 616개 scenario/system-axis 계약을 사용한다. `environment_v2.enabled=false`일 때 기존 seed→scenario,
seed→motion 경로와 training/evaluation fingerprint는 이전 형식과 동일하게 유지된다.

새 학습은 `tractor_local_dynamic_v2`, 새 고정 평가는 `evaluation_v2_*` profile을 사용한다.
이전 protocol 또는 Environment v1 결과와 v2 결과는 환경 계약이 다르므로 같은 표에서
무조건 합치지 않는다.

## 2. 실제 구현

한 episode reset은 다음 순서로 실행된다.

```text
split-safe seed + restored trainer episode index
  → curriculum level
  → robot-relative start/goal
  → topology-conditioned static layout
  → TTC/DCPA-conditioned moving conflicts
  → static + dynamic t=0 grid/Ackermann reachability 검사와 bounded layout retry
  → shape-aware Gazebo spawn
  → acceleration/turn-rate-limited motion
  → 5 substep pose/physics integration per 0.1 s control tick
  → oriented Hunter footprint collision/termination
```

주요 source는 다음과 같다.

- 설정과 fail-fast validation: `config/schema.py`, `config/loader.py`
- curriculum과 scenario 생성: `env/scenarios/tractor_environment_v2.py`
- footprint geometry: `env/scenarios/footprint_geometry.py`
- obstacle motion: `env/humans/dynamic_obstacle_motion.py`
- shape SDF: `env/spawning/obstacle_spawner.py`
- reset/step 연결: `env/simulation/environment_node.py`
- 고정 suite 생성: `evaluation/materialize_environment_v2.py`

## 3. Curriculum

단계는 **학습 episode index**로 결정된다. Trainer가 checkpoint에서 복원한 index를 reset 전에
환경에 전달한다. v2 training seed를 전달하면서 index가 빠지면 환경이 seed를 거부한다.
validation/test는 항상 level 4를 사용한다.

| Level | 시작 episode | 최대 static | dynamic | dynamic/Hunter 최대속도 비 |
|---:|---:|---:|---:|---:|
| 0 | 0 | 2 | 0 | 0.0 |
| 1 | 250 | 4 | 1 | 0.5 |
| 2 | 1,000 | 6 | 2 | 0.75 |
| 3 | 3,000 | 8 | 4 | 1.0 |
| 4 | 6,000 | 10 | 8 | 1.0 |

기본 Local world는 16 m이고 goal 거리는 2–6 m다. 정식 generalization 평가는 12 m/16 m/24 m를
분리된 고정 suite로 측정한다. Local 학습 world를 Global 24 m world와 혼동하지 않는다.

## 4. 장면, 형상, 운동

Topology taxonomy는 open, corridor, doorway, intersection, S-curve, warehouse aisle, clutter다.
Static/dynamic object는 cylinder, box, cart, L-shape를 지원한다. L-shape SDF의 두 primitive는
collision/visual 모두 유효한 6-DoF pose로 생성된다. legacy risk/feasibility 계산이
shape를 아직 직접 처리하지 못하는 경로에서는 각 shape의 circumscribed radius를 보수적으로
사용하고, Gazebo spawn과 episode collision termination은 실제 primitive shape를 사용한다.

Motion taxonomy는 crossing, head-on, cut-in, parallel, random waypoint에 더해 stop-go,
acceleration/deceleration, sine swerve, boundary bounce와 U-turn을 포함한다. `nonreactive`,
`yielding`, `reciprocal` interaction mode를 분리한다. 매 tick obstacle state에는 위치, 속도,
가속도, yaw와 motion time이 유지된다.

기본 dynamic scene의 65%는 nominal straight-path ego motion에 대해 TTC 1–4 s, DCPA 0–0.25 m를
목표로 역산한다. 생성된 obstacle에는 재계산한 실제 initial TTC/DCPA를 저장한다. 배치가 불가능한
경우 무한 재시도하지 않고 bounded retry 후 일반 obstacle로 대체하며, 실제 conflict 수를 scenario
metadata에 기록한다.

## 5. Hunter footprint와 시간 동기화

v2 collision termination은 원형 하나가 아니라 yaw가 반영된 Hunter 직사각형과 obstacle
primitive의 교차를 계산한다. 폭은 robot config의 실제 폭을 쓰고, 길이는 URDF에서 검증한
collision radius를 잃지 않도록 명목 차체 길이보다 필요한 경우 확장한다. Safety guard와 기존
risk rollout의 원형 footprint는 기존 checkpoint 의미를 보존하기 위한 보수적 fallback이다.

v1 moving obstacle은 0.1 s당 한 번 이동하는 기존 zero-order hold를 그대로 쓴다. v2는 기본
5개 0.02 s substep마다 obstacle pose update와 Gazebo physics advance를 교차 실행한다. 이는
Gazebo model plugin이 만드는 진짜 동적 rigid-body actuator는 아니며, service-driven substep
integration이다. 따라서 접촉 dynamics 연구를 주장하려면 별도 model plugin 검증이 필요하다.

Deterministic stepping은 각 `ControlWorld.multi_step` 요청 전후에 값이 변하는 `/clock` queue가
안정됐는지 확인한 뒤, 관측된 시간 증가를 예상값과 대조한다. 일반 허용치와 1-step calibration
허용치는 모두 기본 0.0002 s이며 0.001 s Gazebo physics tick의 절반보다 작아야 한다. 따라서
완전한 한 tick 누락/초과는 허용 오차로 숨길 수 없다.

## 6. Randomization과 calibration 경계

`config/environment_v2/hunter_se_calibration.yaml`이 범위와 적용 위치를 고정한다. 현재 상태는
`engineering_prior`이고 실차 측정값이 아니다.

| 분류 | 축 |
|---|---|
| Gazebo command path에 적용 | friction/velocity response, steering gain/delay, command latency |
| observation only | LiDAR noise/dropout, odometry noise, frame drop |
| model only | mass scale, wheel-radius scale |

Profile과 manifest 범위 또는 code의 적용 분류가 다르면 profile load가 실패한다. Formal sim-to-real
전에는 system-ID artifact를 `source_artifacts`에 넣고 manifest status를 `measured`로 바꾸며,
`require_measured_calibration=true`로 승격해야 한다. 현재 mass/wheel-radius는 실제 Gazebo plant에
적용되지 않으므로 plant-level randomization이라고 주장할 수 없다.

## 7. 고정 평가 suite

각 suite는 겹치지 않는 test seed 8개와 파일 SHA-256을
`config/benchmarks/environment_v2_manifest.json`에 기록한다. 생성 시 start/goal과 모든 `t=0`
static/dynamic footprint를 함께 검사하며, 의도된 feasible scene은 grid connectivity와
Ackermann-reachable heading을 만족할 때만 고정된다. Goal-blocked negative fixture는 reason으로
명시되어 feasible count에 섞이지 않는다. 현재 manifest의 분류는 intended-feasible **41/41**,
deliberate `goal_blocked` negative fixture **7/7**이며, 이 수치는 policy 성공률이 아니라 생성기
계약 검사 결과다.

| Profile | World | 목적 |
|---|---:|---|
| `evaluation_v2_id` | 16 m | training-support와 같은 ID taxonomy |
| `evaluation_v2_ood_motion` | 16 m | 1.0–1.5× 속도와 비선형 운동 |
| `evaluation_v2_ood_density` | 16 m | static 14, dynamic 12 |
| `evaluation_v2_ood_geometry_12m` | 12 m | 작은 unseen topology 조합 |
| `evaluation_v2_ood_geometry_24m` | 24 m | 큰 unseen geometry/density |
| `evaluation_v2_ood_system` | 16 m | command-path system edge cases |

`v2_ood_system`의 mass/wheel-radius 항목은 위 표대로 model-only다. sim-to-sim 또는 실차 결과를
대신하지 않는다.

## 8. 학습 전 확인 명령

```bash
python3 -m hunter_kinodynamic_rl.evaluation.materialize_environment_v2
python3 -m hunter_kinodynamic_rl.training.preflight \
  --profile tractor_local_dynamic_v2 --no-require-ros
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python3 -m pytest -q tests/test_environment_v2.py
```

첫 명령은 같은 plan이면 동일 YAML/checksum을 다시 만든다. 두 번째 명령의 calibration warning은
현재 measured artifact가 없다는 의도된 경고다. 학습·성능·실시간성·실차 증거는 이 명령들로
생기지 않는다.

## 9. 남은 실증 작업

1. 사용자가 승인한 뒤 v2 profile로 학습한다.
2. 동일 checkpoint를 여섯 고정 suite에서 평가하고 seed-level CI를 계산한다.
3. conflict 생성률, realized TTC/DCPA 분포, topology/shape별 실패율을 보고한다.
4. Hunter 로그로 calibration manifest를 측정값으로 교체한다.
5. model-only mass/wheel 축이 중요하면 Gazebo model plugin 또는 SDF respawn path를 별도 구현한다.
6. sim-to-sim, HIL과 contained real-robot trial을 순서대로 수행한다.

현재 허용되는 결론은 “향상된 환경 code와 고정 평가 계약이 구현됐다”까지다. TRACTOR-TQC의
성능 향상 또는 ICRA/IROS급 실증은 학습과 위 평가 artifact가 생긴 뒤에만 판단한다.
