# 2026-09-02 Stage 1 corrected-baseline release freeze

## 판정

| 항목 | 결과 |
|---|---|
| source build | PASS |
| direct pytest | PASS: 2,093 |
| colcon aggregate | PASS: 2,098, error/failure/skip 0 |
| profile validation | PASS: 29/29 |
| Local Gazebo save/resume | PASS |
| Global Gazebo save/resume | BLOCKED AS DESIGNED: promoted Local 부재 |
| teardown | PASS: 잔류 Gazebo/ROS process/node 0 |
| provenance | annotated tag `hunter-kinodynamic-rl-r0-20260902` |

이 검증은 wiring, update, checkpoint와 resume의 bounded engineering smoke다.
학습 수렴, benchmark 성능 또는 논문 결과를 증명하지 않는다.

## 수정한 release 차단 결함

1. 기존 `smoke_test`는 `uniform_world`/infeasible 0이라 arbitrary-subgoal
   preflight를 통과할 수 없다. 게이트를 완화하지 않고 동일한 full-circle,
   target infeasible fraction 0.15 계약을 갖는 60-step
   `smoke_test_arbitrary_subgoal.yaml`을 추가했다. 측정값 0.17,
   `state_dim=328`로 preflight를 통과했다.
2. `CMakeLists.txt`의 `install(PROGRAMS)`에 등록된 여러 node가 실행 비트 없이
   symlink-install되어 `ros2 run ... environment_node.py`가 `No executable found`로
   실패했다. 설치 대상 node의 실행 권한을 바로잡고 모든 PROGRAMS 대상의 존재,
   shebang과 실행 비트를 검사하는 `test_ros_entrypoint_permissions.py`를 추가했다.

## 실행 증거

Docker에서 ROS Humble과 workspace를 source한 뒤 다음을 실행했다.

```bash
colcon build --packages-select hunter_kinodynamic_rl --symlink-install
colcon test --packages-select hunter_kinodynamic_rl --event-handlers console_direct+
colcon test-result --test-result-base build/hunter_kinodynamic_rl --all --verbose
for profile_path in src/hunter_kinodynamic_rl/config/profiles/*.yaml; do
  profile=${profile_path##*/}
  python3 -m hunter_kinodynamic_rl.config.validation "${profile%.yaml}" || exit 1
done
```

최종 결과는 pytest 2,093개, colcon aggregate 2,098개, profile 29/29개 모두
통과다. `ros2 pkg executables hunter_kinodynamic_rl`에서도 CMake에 등록된 15개
실행 파일이 노출됐다.

Local은 실제 headless Gazebo/Hunter SE와 환경 node를 띄우고 다음 bounded
profile로 실행했다.

```bash
ros2 run hunter_kinodynamic_rl environment_node.py --ros-args \
  -p profile:=smoke_test_arbitrary_subgoal
ros2 run hunter_kinodynamic_rl train_node.py --ros-args \
  -p profile:=smoke_test_arbitrary_subgoal
```

fresh run은 60 env step, 6 episode, replay/update와 `best`/`latest`/`final`
generation 저장을 완료했다. 마지막 health event는 telemetry valid ratio 1.0,
matched 61, telemetry/reset-marker timeout 0, risk-supervised update 48이었다.

동일 run의 `best`(step 53)에서 다음처럼 재개했다.

```bash
ros2 run hunter_kinodynamic_rl train_node.py --ros-args \
  -p profile:=smoke_test_arbitrary_subgoal \
  -p resume_run_dir:=runtime/experiments/20260902_082233_smoke_test_arbitrary_subgoal_seed0 \
  -p resume_checkpoint_tag:=best
```

재개 실행은 7개 전이를 수행해 step 60에 도달하고 새 `final` generation을
원자적으로 저장했다. resume 구간 telemetry는 7/7 match, timeout 0이었다.
해당 `runtime/` run은 재현 가능한 smoke 산출물로 Git에는 포함하지 않았다.

Global은 다음 production preflight부터 실행했다.

```bash
ros2 run hunter_kinodynamic_rl hierarchical_preflight.py \
  --profile hierarchical_phase4 --live
```

exit code 1과 함께 `local_frozen/checkpoints/final`의
`promotion_manifest.json` 부재를 보고했다. 현재 symlink가 가리키는 2026-08-31
Local은 `LEGACY_INVALID`이므로 이 차단이 올바른 결과다. preflight를 우회하거나
bounded smoke checkpoint를 승격하지 않았고, 따라서 Global mission/update/save/
resume는 실행하지 않았다.

종료 후 smoke 전용 process group을 정리하고 Gazebo, environment/trainer,
bridge, robot-state publisher, pointcloud converter와 ROS node가 남지 않았음을
확인했다.

## 다음 순서

1. L0–L5를 multi-seed로 학습하고 동일 formal Local benchmark로 평가한다.
2. acceptance를 통과한 `final`만 `local_frozen`으로 promotion한다.
3. promoted Local로 bounded Global mission/update/save/resume를 완료한다.
