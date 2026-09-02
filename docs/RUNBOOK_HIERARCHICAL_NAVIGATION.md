# Hierarchical Navigation Runbook

> **2026-09-01 defect-fix pass 완료:** 이전 audit(`CURRENT_STATUS.md` 참조)이
> 지목한 7개 장시간-학습-전 차단 결함(Local observation 좌표계, infeasible
> benchmark/promotion 신뢰 체인, feasibility temporal snapshot 오염, evaluator
> fallback 가시성, BACKTRACK 하드코딩, canonical Local tag 불일치)과 평가
> 인프라 gap(rosbag dry-run false positive, live evidence 기본 profile 부재,
> formal A/B·A–G 검증 누락, preflight 경계 버그, localization sweep live
> driver 부재)을 모두 코드와 회귀 테스트로 닫았다 (`colcon test-result`: 2,096
> tests, 0 errors/failures/skips). 아래 절차는 이 수정을 반영해 갱신했다 —
> **`environment_node.py`를 재빌드해야** 3단계가 요구하는
> `local_training_contract_fingerprint_sha256` 파라미터가 노출된다. 정식
> 장시간 학습·정식 A/B·A–G의 실제 실행 자체는 이 세션에서 수행하지 않았다
> (아래 각 단계의 "예상 시간"이 긴 명령들) — 코드 계약이 정식 실행을 시작해도
> 되는 상태가 됐다는 뜻이지, 학습된 정책의 성능을 증명했다는 뜻이 아니다.
>
> **2026-09-02 R0 release 검증:** Docker direct `pytest` 2,093개,
> `colcon test-result` 2,098개, profile validation 29/29개가 통과했다.
> `smoke_test_arbitrary_subgoal`로 실제 Gazebo Local 60-step save와
> `best` step 53→60 resume를 완료했다. Global live preflight는 기존
> `local_frozen`의 promotion manifest 부재를 정상적으로 차단했으며,
> smoke checkpoint를 승격시키지 않았다. evaluator
> fallback reason은 artifact에 남지만 현재 Global candidate tensor에는 명시적
> validity 열이 없고 실패한 policy-conditioned feature가 0으로 채워진다. E/F/G
> formal 결과는 아래 raw fallback-count 판정 규칙 없이 수용하지 않는다.

이 문서는 새 Local TQC 학습부터 Phase 6 평가까지, 사용자가 직접 실행할 명령을
순서대로 정리한다. 모든 명령은 실제 존재하는 entry point/launch/config 이름을
사용한다 (`hunter_kinodynamic_rl/CMakeLists.txt`의 `install(PROGRAMS ...)`
목록이 실제 `ros2 run` 대상의 정본이다). 모든 단계는 `docker exec -it
<container> /bin/bash` 로 컨테이너에 들어간 뒤 `/root/DRL_Robot_Path_Planning/
ros2_ws`에서 ROS Humble을 source하고 실행한다 (`CLAUDE.md`의 Active Docker
Environment 참조). 각 명령 앞에 `source /opt/ros/humble/setup.bash &&
source install/setup.bash`가 이미 실행됐다고 가정한다.

여기 적힌 "정식(formal)" 명령의 코드 경로는 이제 강제 게이트(mode=test,
promoted Local, profile 기본 budget, strict checkpoint 검증 등)로 보호되지만,
실제로 장시간 실행된 적은 없다. 명령이 존재하고 게이트를 통과한다는 사실을
연구 성능 증거로 해석하지 않는다.

이 runbook은 **현재 구현된 scalar-risk/nominal-dynamics Local과 기존 Phase-5
A–G 경로**의 실행법만 다룬다. `RESEARCH_ROADMAP.md`의 L6–L8 risk/residual
uncertainty, direct counterfactual target, capability distribution, experience
posterior와 localization-risk propagation은 아직 실행 entry point/profile이 없는
목표 기능이다. 구현 전까지 아래 명령의 결과에 그 이름을 붙이지 않는다.

## 0. 사전 확인

```bash
docker ps  # DRL_Robot_Path_Planning 컨테이너 ID 확인
docker exec -it <container_id> /bin/bash
cd /root/DRL_Robot_Path_Planning/ros2_ws
source /opt/ros/humble/setup.bash && source install/setup.bash
colcon build --packages-select hunter_kinodynamic_rl --symlink-install
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  src/hunter_kinodynamic_rl/tests/
for profile_path in src/hunter_kinodynamic_rl/config/profiles/*.yaml; do
  profile=${profile_path##*/}
  python3 -m hunter_kinodynamic_rl.config.validation "${profile%.yaml}" || exit 1
done
```

2026-09-02 R0 기준 결과는 `2,093 passed`, profile `29/29 valid`다. profile
수가 바뀌면 고정 숫자보다 loop의 실패 유무와 실제 파일 수를 기준으로 판정한다.

새 프로필/설정 파일을 추가했다면 (본 세션에서 여러 개 추가함:
`hierarchical_phase5_c.yaml`, `config/local_acceptance.yaml` 등)
`--symlink-install` 빌드를 한 번 다시 해야 심볼릭 링크가 install 트리에
생긴다 (colcon symlink-install은 새 파일을 자동으로 추적하지 않는다).

---

## 1. Local preflight

**명령:**
```bash
ros2 run hunter_kinodynamic_rl preflight.py \
  --profile kinodynamic_tqc_arbitrary_subgoal --run-root runtime/experiments
```
(또는 코드에서 직접: `hunter_kinodynamic_rl.training.preflight.run_local_preflight(...)`)

**필요 입력:** 없음 (프로필만 있으면 됨). ROS/Gazebo가 아직 없어도 대부분의
체크가 동작하며, `rclpy`와 `ros_gz_interfaces` import 여부를 모두 검사한다
(`--no-require-ros`로 둘 다 완화 가능).

**생성물:** 없음 (stdout에 리포트만 출력). exit code 0 = 통과, 1 = 실패.

**성공 조건:** `full_circle_goal_direction=True`, `sample_count > 0`, 측정된
`infeasible_fraction`이 설정값의 binomial confidence interval
(`infeasible_fraction_confidence_z`, 기본 4-sigma) 이내, `resume=False`인데
`resume_run_dir`가 비어 있음, `local_training_contract_fingerprint`가 출력됨,
(resume 시) 기존 `checkpoints/`의 `local_training_contract_fingerprint`/
architecture fingerprint가 현재 profile과 일치.

**실패 시 확인할 로그:** stdout의 `ERROR:` 줄 — profile 스키마 오류,
full-circle 미달, resume/fresh 불일치, `ros_gz_interfaces` import 실패, 또는
resume checkpoint의 distribution/architecture 불일치 중 하나다.

**정식 vs smoke 차이:** 이 단계 자체가 게이트이므로 구분 없음.

**bounded Local wiring 검사:** 정식 프로필과 동일한 full-circle/
infeasible 분포 게이트를 유지하면서 60 step만 실행할 때는
`--profile smoke_test_arbitrary_subgoal`을 사용한다. 이 프로필과 checkpoint는
promotion/formal 입력으로 사용하지 않는다.

**자동 호출:** `train_kinodynamic_tqc.py::main`은 `dry_run=False`(실제 학습
시작)에서도 이 preflight를 자동으로 먼저 실행하고 실패 시 학습을 시작하지
않는다 — `--skip-preflight`로만 명시적으로 우회 가능(우회 시 경고 로그).

---

## 2. Fresh Local TQC 학습

**명령:**
```bash
# 터미널 1
ros2 launch hunter_se_gazebo simulate_hunter_se_ignition.launch.py rviz:=false headless:=true

# 터미널 2
ros2 run hunter_kinodynamic_rl environment_node.py --ros-args -p profile:=kinodynamic_tqc_arbitrary_subgoal

# 터미널 3 (fresh -- resume_run_dir를 주지 않으면 항상 새 timestamp 디렉터리)
ros2 run hunter_kinodynamic_rl train_node.py --ros-args -p profile:=kinodynamic_tqc_arbitrary_subgoal
```

**resume 재개:**
```bash
ros2 run hunter_kinodynamic_rl train_node.py --ros-args \
  -p profile:=kinodynamic_tqc_arbitrary_subgoal \
  -p resume_run_dir:=runtime/experiments/<timestamp>_kinodynamic_tqc_arbitrary_subgoal_seed0
```
`resume_run_dir`를 주지 않으면 무조건 새 디렉터리가 생성된다
(`training/run_logger.run_directory`) — 기존 checkpoint를 묵시적으로
재사용하지 않는다.

**생성물:** `runtime/experiments/<timestamp>_kinodynamic_tqc_arbitrary_subgoal_seed<seed>/
{configs,checkpoints,logs,evaluation,benchmark,tensorboard}/`.
`checkpoints/`의 매니페스트에는 `local_training_contract_fingerprint`,
`architecture_fingerprint`, `generation`, `pt_sha256`가 기록된다
(다른 분포로 학습된 checkpoint는 resume 시 strict하게 거부됨 —
`training/trainer_base.py::_resume_from`).

**성공 조건:** `training.max_timesteps=150000`(profile 기본값)까지
episode reward가 개선 추세를 보이고 collision-free episode가 늘어난다.
정식 완료 판정은 3단계(Local benchmark)에서 내린다 — 학습 자체의
"수렴"을 이 단계에서 자동 판정하지 않는다.

**실패 시 확인할 로그:** `runtime/experiments/<run>/logs/steps.jsonl`,
`episodes.csv`; Gazebo 연결 실패는 터미널 2의 서비스 타임아웃 메시지.

**예상 시간:** 150,000 timestep 학습은 실시간 Gazebo 기준 수 시간~하루
단위로 길 수 있다.

---

## 3. Local 정식 benchmark

**명령 (코드에서 조립, 아직 단일 CLI 스크립트는 없음 — 아래는 실제 사용 가능한
Python 조합):**
```bash
python3 -c "
import rclpy
from hunter_kinodynamic_rl.config.loader import load_profile
from hunter_kinodynamic_rl.training.trainer_base import EnvironmentClient
from hunter_kinodynamic_rl.rl.algorithms.kinodynamic_tqc.agent import Agent
from hunter_kinodynamic_rl.rl.checkpointing import manager as ckpt_manager
from hunter_kinodynamic_rl.evaluation.local_subgoal_benchmark import (
    build_local_benchmark_manifest, save_local_benchmark_manifest,
    run_local_subgoal_benchmark,
)
import json

profile = load_profile('kinodynamic_tqc_arbitrary_subgoal')
manifest = build_local_benchmark_manifest(profile, num_scenarios=20, run_seed=0)
save_local_benchmark_manifest(manifest, 'runtime/local_benchmark/manifest.json')

rclpy.init()
env = EnvironmentClient()
history_len = profile.observation.frame_stack if profile.features.temporal_context else 1
state_dim = profile.observation.lidar_bins * history_len + profile.observation.robot_state_dim
agent = Agent(state_dim, 3, 1.0, profile.hyperparameters, profile.risk, profile.counterfactual)
ckpt_dir = 'runtime/experiments/<timestamp>_kinodynamic_tqc_arbitrary_subgoal_seed0/checkpoints'
result = ckpt_manager.load_generation(ckpt_dir, 'final', agent.checkpoint_components(), map_location=str(agent.device))
manifest_dict = result['manifest']

artifact = run_local_subgoal_benchmark(
    profile, agent, manifest=manifest, checkpoint_generation=manifest_dict['generation'],
    checkpoint_sha256=manifest_dict['pt_sha256'], env=env, benchmark_kind='formal',
)
with open('runtime/local_benchmark/artifact.json', 'w') as f:
    json.dump(artifact, f, indent=2, default=str)
env.destroy_node()
rclpy.shutdown()
"
```
(20개 미만 scenario로는 실행하지 말 것 — 다음 단계 promotion이 `min_scenarios`
gate로 거부한다. 위 예시의 `env`는 `environment_node.py`가 실제 로드한 profile과
architecture/training-contract fingerprint가 일치해야 한다 —
`run_local_subgoal_benchmark`가 시작 전에
`verify_environment_server_identity`로 `env.get_remote_parameter(...)`를 통해
자동 확인하며, 불일치 시 Gazebo에 아무 scenario도 실행하지 않고 즉시
`BenchmarkArtifactError`를 낸다. 이 파라미터는 `environment_node.py`가
새로 노출하므로 **재빌드가 필요하다**.)

**필요 입력:** 1개 이상 학습된 Local checkpoint (2단계 산출물), 실행 중인
Gazebo + `environment_node.py`(같은 profile, 재빌드된 버전).

**생성물:** `runtime/local_benchmark/{manifest.json,artifact.json}` —
`artifact.json`은 `evaluation.local_subgoal_benchmark.build_local_benchmark_artifact`
스키마 v2 (schema_version=2, benchmark_kind, checkpoint identity, fingerprint,
summary: `feasible_subgoal_success_rate`(feasible episode 조건부)/전체
collision·timeout·high_risk_failure rate/`infeasible_scenario_count`/
`infeasible_valid_count`/`infeasible_false_success_rate`/`infeasible_collision_rate`/
`infeasible_high_risk_rate`/`infeasible_safe_termination_rate`(전부 infeasible
episode 조건부, 표본 0이면 `None`), termination_reason별 count(`collision`/
`goal_reached`/`timeout`/`unknown` — `infeasible_goal_rejected`는 더 이상 없음),
time_to_goal(성공만)/time_to_termination(전체)). 저장 전 manifest/episode
1:1 pairing(중복 scenario_id/seed 불일치/non-test mode)을 자동 검증한다.

**성공 조건:** `artifact.json`의 `summary`가 `config/local_acceptance.yaml`의
`min_feasible_subgoal_success_rate`/`max_collision_rate`/`max_timeout_rate`/
`max_high_risk_failure_rate`/`min_infeasible_scenarios`(및 표본이 충분할 때만
평가되는 4개 infeasible-conditional threshold)를 모두 만족해야 한다 — 4단계
(promotion)가 정확히 어떤 항목이 실패했는지 보고한다.

**실패 시 확인할 로그:** artifact의 `summary` 필드 자체가 진단 — 어떤
지표가 threshold를 넘었는지 4단계(promotion)가 구체적으로 보고한다.

**정식 vs smoke:** `benchmark_kind='formal'`을 명시적으로 지정해야 한다
(기본값은 `'formal'`이므로 smoke로 쓸 때는 반드시 `benchmark_kind='smoke'`를
명시). schema_version이 `evaluation.local_subgoal_benchmark.MIN_SUPPORTED_BENCHMARK_SCHEMA_VERSION`
(현재 2) 미만인 구버전 artifact는 promotion이 즉시 거부한다.

---

## 4. Local checkpoint promotion

**명령 (`checkpoint_manifest`는 이제 선택 인자다 — 생략하면
`source_checkpoint_dir/source_checkpoint_tag`의 실제 `manifest.json`/`model.pt`를
직접 읽고 SHA를 재계산해 사용한다; 넘기면 그 값과 on-disk 값이 다를 때 즉시
거부하는 cross-check로만 쓰인다):**
```bash
python3 -c "
import json
from hunter_kinodynamic_rl.evaluation.local_promotion import (
    promote_local_checkpoint, load_local_acceptance_config, DEFAULT_PROMOTION_TAG,
)

with open('runtime/local_benchmark/artifact.json') as f:
    artifact = json.load(f)

result = promote_local_checkpoint(
    artifact, None,
    source_checkpoint_dir='runtime/experiments/<timestamp>_kinodynamic_tqc_arbitrary_subgoal_seed0/checkpoints',
    source_checkpoint_tag='final',
    target_dir='runtime/experiments/local_frozen/checkpoints', target_tag=DEFAULT_PROMOTION_TAG,
)
print(result)
"
```

**필요 입력:** 3단계 `artifact.json` (formal, schema_version>=2, 20개 이상
scenario), 원본 checkpoint 디렉터리 (`manifest.json`/`model.pt`가 실제로
있어야 함 — 없으면 promotion이 즉시 거부한다).

**생성물:** 통과 시 `runtime/experiments/local_frozen/checkpoints/final`
(=`DEFAULT_PROMOTION_TAG`, Phase 4/5 profile 전부가 읽는 canonical tag) 심볼릭
링크가 **원자적으로**(temp 디렉터리에 복사 → `promotion_manifest.json` 기록 →
`os.rename` 발행 → tag symlink를 temp-symlink+`os.replace`로 교체) 새
checkpoint를 가리키도록 갱신된다. 동일 checkpoint의 재승격은 idempotent
no-op이고, 다른 checkpoint가 이미 그 목적지를 차지하고 있으면 기존 promoted
checkpoint를 보존한 채 거부한다. 실패 시 아무 파일도 변경되지 않는다.

**성공 조건:** `result.accepted == True`.

**실패 시 확인할 로그:** `result.reasons` 리스트 — schema_version 미달,
acceptance 미달(어떤 threshold인지 구체적으로), checkpoint identity 불일치
(artifact vs 실제 checkpoint의 generation/SHA/local_training_contract_fingerprint/
architecture_fingerprint), source의 `model.pt` 실제 SHA가 `manifest.json`
기록값과 다름(파일 변조/손상), 또는 checkpoint manifest에 필수 필드 누락
(레거시 checkpoint) 중 하나를 정확히 명시한다.

**promoted 상태 확인:** `evaluation.local_promotion.validate_promoted_local_checkpoint(dir, tag)`
가 구조화된 결과(`.promoted`, `.reasons`, `.promotion_manifest`)를 반환한다 —
`is_local_checkpoint_promoted(dir, tag)`는 이를 감싼 boolean 하위호환
wrapper이며, 빈 `{}` marker나 checkpoint 파일과 어긋나는 stale/tampered
marker는 둘 다 `False`를 반환한다.

**현재 `local_frozen`의 상태:** 기존 `local_frozen`은 실패한(7274-step,
subgoal_success_rate=0.0) 학습을 가리키며, `runtime/experiments/
20260831_005836_kinodynamic_tqc_arbitrary_subgoal_seed0/LEGACY_INVALID.md`로
표시해 두었다 — promotion을 거치지 않으면 이 상태 그대로 남는다.

---

## 5. Global preflight

Phase 4/5 profile의 `local_checkpoint_name`은 4단계 canonical promoted tag
(`final`)와 동일하다 (모든 profile에서 테스트로 고정됨). preflight는 marker
파일 존재가 아니라 `evaluation.local_promotion.validate_promoted_local_checkpoint`
구조화 검증기를 사용하고, 그 결과의 `architecture_fingerprint`를 현재 profile의
것과 추가로 대조한다.

**명령:**
```bash
ros2 run hunter_kinodynamic_rl hierarchical_preflight.py \
  --profile hierarchical_phase4 --live --no-require-ros   # Gazebo 시작 전 config-only 확인

# Gazebo가 이미 떠 있으면 --no-require-ros 없이:
ros2 run hunter_kinodynamic_rl hierarchical_preflight.py --profile hierarchical_phase4 --live

# resume 시 기존 Global checkpoint 신원도 확인:
ros2 run hunter_kinodynamic_rl hierarchical_preflight.py \
  --profile hierarchical_phase4 --live --resume \
  --resume-run-dir runtime/hierarchical_experiments/<run>
```

**필요 입력:** 4단계에서 promotion된 checkpoint (`local_frozen`).

**생성물:** 없음, stdout 리포트만.

**성공 조건:** `local_checkpoint_promoted=True`이고 그 promotion manifest의
`architecture_fingerprint`가 현재 profile과 일치, device 설정이 실제 사용
가능한 것과 일치, (live 모드면) `ros_available=True`이고
`ros_gz_interfaces_available=True`, (resume 시) 기존 Global checkpoint의
`hierarchical_architecture_fingerprint`/`global_replay_schema_version`이
현재 profile과 일치. E/F/G-tier profile(`feasibility_feedback_enabled`/
`global_risk_feedback_enabled`)은 `--live` 없이는 항상 실패한다 — ROS-free
stand-in에 실제 `LocalFeasibilityEvaluator`가 없기 때문이다.

**실패 시 확인할 로그:** `ERROR:` 줄 — "requires a PROMOTED Local
checkpoint"(뒤에 `validate_promoted_local_checkpoint`의 구체적 이유가
따라옴), architecture_fingerprint 불일치, CUDA 미가용, `ros_gz_interfaces`
import 실패, resume checkpoint 불일치, 또는 "feasibility/global-risk
feedback ... but live=False" 중 하나다.

---

## 6. Global smoke (아주 짧은 검증)

> **R0 상태:** 2026-09-02 live preflight는 expected fail-closed로 종료됐다.
> `runtime/experiments/local_frozen/checkpoints/final` 소스에
> `promotion_manifest.json`이 없다. 2–4단계로 유효한 Local을 만들고
> promotion하기 전에는 아래 명령을 실행하지 않는다.

**명령:**
```bash
ros2 launch hunter_se_gazebo simulate_hunter_se_ignition.launch.py rviz:=false headless:=true

ros2 run hunter_kinodynamic_rl hierarchical_train_node.py --ros-args \
  -p profile:=hierarchical_phase4 -p live:=true -p num_missions:=5
```

**범위 제한 (매우 중요):** `num_missions`을 반드시 1~5 정도로 작게 유지한다.
mission마다 실시간 Gazebo로 최대
`hierarchical_training.max_global_options_per_mission * max_local_steps_per_option`
스텝이 걸릴 수 있어 몇 분 이상 걸릴 수 있으면 즉시 중단하고 dry-run으로
대체한다 (5단계).

**resume 확인 (smoke의 일부):**
```bash
ros2 run hunter_kinodynamic_rl hierarchical_train_node.py --ros-args \
  -p profile:=hierarchical_phase4 -p live:=true -p num_missions:=1 \
  -p resume:=true -p resume_run_dir:=runtime/hierarchical_experiments/<run>
```

**생성물:** `runtime/hierarchical_experiments/<timestamp>_hierarchical_phase4_seed0/
{checkpoints,hierarchical_metadata.json}`.

**성공 조건:** mission 초기화 → option 생성/종료 → replay 삽입 → optimizer
update 1회 이상 (`global_step` 증가) → checkpoint 저장 → RNG/replay 포함
resume 성공 → teardown 정상 (프로세스가 걸리지 않고 종료).

**실패 시 확인할 로그:** stdout의 `hierarchical training finished: {...}`
직전 예외; `LocalCheckpointError`가 뜨면 4/5단계를 다시 확인.

---

## 7. Global 정식 학습 (실행하지 않음 — 명령만 기록)

**명령:**
```bash
ros2 launch hunter_se_gazebo simulate_hunter_se_ignition.launch.py rviz:=false headless:=true

ros2 launch hunter_kinodynamic_rl hierarchical_train.launch.py \
  profile:=hierarchical_phase4 live:=true num_missions:=1000
```

`require_promoted_local`은 기본 True이므로 4단계를 건너뛰면 즉시 거부된다.
fresh 학습은 `resume_run_dir`를 주지 않는 것으로 충분히 구분된다 — 별도
`--fresh` 플래그는 없다 (준 것과 안 준 것이 유일한 구분자).

**생성물:** 6단계와 동일 레이아웃, 더 큰 run.

**성공 조건:** `hierarchical_architecture_fingerprint`, `global_replay_schema_version`,
`total_optimizer_updates`, `epsilon_schedule_basis`, RNG sidecar
path/SHA/schema, Local checkpoint generation/SHA/training-contract
fingerprint가 모두 checkpoint manifest에 기록됨 (이미 코드로 강제됨,
`nodes/hierarchical_train_node.py::_check_resume_compatibility`).

**예상 시간:** 매우 김 (Gazebo 실시간 기준 mission당 수십 초~분, 1000
mission이면 하루 이상 가능) — 이 세션에서 실행하지 않았다.

---

## 8. A/B 정식 benchmark (formal)

**명령:**
```bash
ros2 run hunter_kinodynamic_rl run_live_hierarchical_benchmark.py --ros-args \
  -p profile:=hierarchical_phase4 -p num_scenarios:=20 \
  -p global_checkpoint_dir:=runtime/hierarchical_experiments/<run>/checkpoints \
  -p global_checkpoint_name:=latest -p output_dir:=runtime/hierarchical_benchmark \
  -p formal:=true
```

`formal:=true`면 `global_checkpoint_dir`가 비어 있거나 `num_scenarios<20`이면
Gazebo를 건드리기 전에 즉시 실패한다 (heuristic Global로 대체되지 않음).
`formal:=false`(기본값)는 기존 wiring-smoke 동작 그대로다.

추가 formal 조건(모두 Gazebo를 건드리기 전에 검사됨,
`evaluation/global_checkpoint_validation.py` 공유 검증기): `mode='test'`,
`max_options`/`max_local_steps`를 프로필 기본값에서 override하지 않음(둘 중
하나라도 override하면 즉시 실패 — smoke 전용), Local checkpoint가 실제
promoted 상태(`validate_promoted_local_checkpoint`), Global checkpoint가
`total_optimizer_updates >= 1`을 포함해 strict 검증을 통과. 모든 scenario
결과가 존재하지 않으면(A/B 어느 한쪽이라도 개수가 부족하면) `benchmark_kind`를
`"formal"`이 아니라 `"incomplete"`로 기록한다.

**생성물:** `runtime/hierarchical_benchmark/<timestamp>/{episodes_A.csv,
episodes_B.csv,manifest.json,metrics.json}`. `metrics.json`의
`benchmark_kind`는 `formal:=true`이고 위 조건을 모두 만족할 때 정확히
`"formal"`(불완전하면 `"incomplete"`).

**성공 조건:** 산출물 자체가 목적 (성공/실패 threshold는 이 단계에 없음 —
final_goal_success_rate, Global SPL, collision, revisit, repeated dead-end,
loop_count, Local failure, global_decision_rate, time_to_goal(성공만)/
time_to_termination(전체)이 A/B 양쪽에 기록된다).

**실패 시 확인할 로그:** stdout의 `RuntimeError` — formal 게이트(mode/budget/
promoted-Local) 또는
`evaluation.global_checkpoint_validation.validate_global_checkpoint_manifest`의
architecture/resolved-config/replay-schema/RNG-basis/Local identity/
total_optimizer_updates 불일치 메시지.

**예상 시간:** 20 scenario x 2 ablation, 각 mission 실시간 Gazebo 기준
수십 분~수 시간. **이 세션에서 실행하지 않았다.**

---

## 9. Phase 5 A-G 학습/benchmark

각 ablation은 자기 profile을 쓴다: `hierarchical_phase5_{a,b,c,d,e,f,g}`.
B/C는 이제 `include_visited_channel`로 실제로 구분된다.

**개별 ablation 학습 (예: E):**
```bash
ros2 run hunter_kinodynamic_rl hierarchical_train_node.py --ros-args \
  -p profile:=hierarchical_phase5_e -p live:=true -p num_missions:=1000
```
E/F/G는 `feasibility_feedback_enabled`/`global_risk_feedback_enabled`가
켜져 있어 실제 `LocalFeasibilityEvaluator`가 필요하다 — `live:=true`가
아니면 (ROS-free stand-in에는 실제 Local 정책이 없으므로) 시작 시점에
즉시 거부된다 (Global preflight의 5단계 체크와 동일한 조건).

**A-G 통합 benchmark (같은 manifest 공유):**
```bash
ros2 run hunter_kinodynamic_rl run_live_ablation_suite.py --ros-args \
  -p labels:=A,B,C,D,E,F,G -p num_scenarios:=20 \
  -p global_checkpoint_root:=runtime/ablation_checkpoints \
  -p global_checkpoint_name:=final -p formal:=true \
  -p output_dir:=runtime/ablation_suite
```
`global_checkpoint_root/<label>/`에 각 ablation의 checkpoint가 있어야 하고,
각 label은 A/B와 동일한 `evaluation/global_checkpoint_validation.py` strict
검증(architecture/resolved-config/replay-schema/RNG-basis/Local identity/
total_optimizer_updates)을 통과해야 한다. `formal:=true`에서는 요청한 B–G
중 하나라도 checkpoint가 없거나 strict 검증에 실패하면 **전체 실행이 즉시
실패한다** (label skip은 `formal:=false`/smoke에서만 허용 — 이제
`labels_skipped`가 비어 있지 않은 상태로 `benchmark_kind="formal"`이 저장되는
경로가 없다).

**생성물:** `runtime/ablation_suite/<timestamp>/ablation_suite_result.json`
— `manifest.json`(scenario manifest 원문), `seed`/`mode`/`scenario_manifest_hash`,
label별 `episodes`(원시 per-scenario dict), `evaluator_telemetry`(E/F/G
label별 query/fallback/risk-timeout/risk-error 카운터), `acceptance_report`에
D-vs-C(repeated dead-end/revisit), E-vs-D(local failure), F/G-vs-E(collision/
risk 유지하며 success 유지) 비교가 `verdict: improved|regressed|maintained|insufficient_data`로
기록됨. `insufficient_data`는 표본이 `min_valid_samples`(기본 20) 미만일 때만
나오며 "개선"으로 절대 판정되지 않는다.

**E/F/G 결과 수용 전 추가 확인:** `evaluator_telemetry`의 fallback은 unknown
evaluation이다. 현재 network 입력에는 per-candidate validity 열이 없으므로 0을
known low risk로 해석하면 안 된다. 현재 schema에서는 label/condition별 raw
action/risk fallback count가 모두 0이어야 한다. 기존 `fallback_rate`는 query는
decision 단위, 일부 failure는 candidate 단위라 1을 넘을 수 있으므로 gate에 쓰지
않는다. 하나라도 nonzero이면 코드가 실행을 끝냈더라도 formal artifact는
incomplete로 판정하고, affected decision 제외 분석은 진단용으로만 남긴다.

**예상 시간:** 매우 김 (7개 ablation x 20 scenario 학습+평가) — 이
세션에서 실행하지 않았다.

---

## 10. Localization sweep

노이즈/드리프트 주입 자체는 이미 구현되어 있다
(`navigation/localization/wheel_imu_backend.WheelImuNoiseModel`,
`navigation/localization/lidar_odom_backend.LidarOdomNoiseModel`).
**Live driver 구현 완료** (`evaluation/localization_sweep_live_driver.py`,
defect-fix item 12) — ideal/noisy/drifting 세 condition을 실제로 실행해 결과를
얻는다:

```python
import rclpy
from hunter_kinodynamic_rl.config.loader import load_profile
from hunter_kinodynamic_rl.evaluation.localization_sweep_live_driver import run_live_localization_sweep

rclpy.init()
profile = load_profile('hierarchical_phase4')
agent = ...  # trained/heuristic Global agent, 같은 계약을 8/9단계와 공유
sweep = run_live_localization_sweep(profile, agent, num_scenarios=20, seed=20000)
print(sweep.summaries, sweep.drift_sensitivity)
rclpy.shutdown()
```

내부적으로 조건마다 독립된 `LiveGazeboLocalExecutor`를 그 조건 전용
localization backend(`ideal`→`GazeboOdomLocalizationBackend`, 그 외→
`WheelImuLocalizationBackend` + 조건별 noise model)로 구성하고, 하나의 fixed
scenario manifest를 공유한 뒤 기존 `aggregate_localization_sweep`으로 넘긴다.
scenario_id 집합이 세 condition 간에 다르면 `ScenarioPairingError`로 즉시
거부된다 — 절대 조용히 잘못된 비교를 만들지 않는다.

**HONEST LIMITATION:** `WheelImuLocalizationBackend`는 실제 twist에 random
noise를 주입하지 않고 covariance/confidence만 성장시킨다(그 모듈 자체의
기존 문서화된 제약) — 이 sweep은 confidence-gated 동작 차이는 비교하지만
실제 위치 drift에 따른 성능 저하를 정량화하지는 못한다. `LidarOdomLocalizationBackend`는
실제 scan-matcher가 이 저장소에 없어 조건으로 쓰지 않는다.

**이 세션에서 실제 실행은 하지 않았다** (live Gazebo 인스턴스 없음) — pure
단위 테스트(`tests/test_localization_sweep_live_driver.py`)만 통과했다.

---

## 11. rosbag dry-run

**실제 bag으로 실행:**
```bash
python3 -c "
from hunter_kinodynamic_rl.evaluation.rosbag_dry_run import run_rosbag_dry_run
# node: dry_run=True/replay_mode=True로 구성된 HierarchicalNavigationNode
#       (또는 동일 콜백 표면을 갖는 테스트 더블)
report = run_rosbag_dry_run(
    '/path/to/mission.bag', node, scan_topic='/scan', odom_topic='/odometry',
    joint_states_topic='/hunter_se/joint_states', output_dir='runtime/rosbag_dry_run',
)
print(report)
"
```

**생성물:** `runtime/rosbag_dry_run/rosbag_dry_run_report.json` (schema v2) —
bag identity(metadata + 실제 파일 바이트를 반영한 sha256), 사용된/누락된/
message-수-부족 topic, 생성된 decision 수, `actuator_publish_attempt_count`
(계측된 `.publish()` 호출 시도 횟수 — 실제 `rclpy.Publisher`에는 없는 mock
전용 `.published` 속성을 읽지 않는다), `actuator_topic_message_count`(항상
0 — wrapper가 절대 forward하지 않는 구조적 보장), `dry_run_active`/
`replay_mode_active`.

**성공 조건:** `report.success == True`, 즉 다음을 모두 만족: 필수 topic
전부 존재 + 최소 message 수 만족, 실제 사용된 topic이 1개 이상, decision
1개 이상, `dry_run`/(정의돼 있다면) `replay_mode`가 모두 True,
`actuator_publish_attempt_count == 0`. `run_rosbag_dry_run`은 이제
`node._cmd_pub`를 계측 wrapper로 교체해 publish 시도 자체를 세므로, 실제
`rclpy.Publisher`를 쓰는 진짜 노드에 대해서도 이 카운트가 정확하다. bag은
전체를 메모리에 올리지 않고 스트리밍으로 읽는다.

**실제 bag 없이 안전성만 확인:** `evaluation.rosbag_dry_run.replay_message_stream`을
mock 메시지 리스트 + mock node로 직접 호출 (`tests/test_rosbag_dry_run.py` 참조).

**남은 작업 (외부 입력 필요):** 실제 mission rosbag이 없다 — 위 명령은
사용자가 실차/실 Gazebo에서 `ros2 bag record`로 만든 bag이 있어야 실행
가능하다.

---

## 12. 결과 aggregation

- Local benchmark: `evaluation.local_subgoal_benchmark.aggregate_local_benchmark_episodes`
- A/B formal: `run_live_hierarchical_benchmark.py`가 쓰는 `metrics.json` 자체가 이미 A/B aggregate 결과
- A-G: `evaluation.ablation_suite.run_ablation_suite`/`run_live_ablation_suite.py`의
  `acceptance_report`
- Localization sweep: `evaluation.localization_sweep.aggregate_localization_sweep` +
  `drift_curve`

모든 aggregation 함수는 표본 수가 부족하면 `insufficient_data`/`None`을
반환하고 "개선"으로 판정하지 않는다 (`min_valid_samples`, `*_valid_count`
필드 참조).

---

## 요약: 무엇이 준비되어 있고, 무엇이 남았는가

| 상태 | 항목 |
|---|---|
| 코드/설정/검증기/회귀 테스트 완료 (2026-09-01 defect-fix pass) | Local observation 좌표계 통일, Local benchmark infeasible 지표 재설계, benchmark/promotion 신뢰 체인(atomic promotion 포함), Phase 5 evaluator temporal 오염/fallback 가시성/BACKTRACK 하드코딩, canonical Local tag(`final`) 통일, formal A/B·A-G 게이트, rosbag dry-run 계측 publisher, live evidence runner 기본 profile/preflight/이벤트, Local/Global preflight 강화, localization sweep live driver |
| 이전부터 구현 및 회귀 확인됨 | B/C visited 분리, missing-evaluator fail-fast, CPU map-location, success-only time-to-goal, provenance |
| 알려진 남은 한계 (코드로 명시됨, 별도 확장 필요) | evaluator fallback은 artifact에서 구분되지만 Global candidate tensor에 validity 열이 없어 두 policy-conditioned feature가 0으로 채워짐; `WheelImuLocalizationBackend`가 실제 twist에 random noise를 주입하지 않음(confidence만 성장) — localization sweep이 실제 위치 drift 성능은 정량화 못함; `LidarOdomLocalizationBackend`용 실제 scan-matcher 없음 |
| 연구 로드맵상 미구현 | multi-task risk ensemble/calibration, learned residual dynamics ensemble, progress-preserving uncertainty-gated direct counterfactual actor target, Local capability distribution, 저장된 topological edge 통계의 uncertainty/recency posterior와 Global 입력 노출, pose covariance의 candidate risk 전파 |
| R0 release 검증 완료 | 2,093 pytest, 2,098 colcon checks, 29/29 profiles, Local Gazebo 60-step save + step 53→60 resume, clean teardown, annotated release tag |
| fail-closed로 보류 | Global live save/resume: 유효한 promoted Local이 없어 preflight가 시작 전 차단 |
| 정식 실행 미수행 | 새 Local/Global 정식 학습, Local/A-B/A-G formal benchmark, live evidence, localization sweep |
| 외부 입력 필요 | 실제 rosbag (11단계), 실차 Hunter SE, 정식 학습에 필요한 GPU/Gazebo 시간 |

최신 상세 판정은 `CURRENT_STATUS.md`를 따른다. 코드 계약 수정과 회귀 테스트는
이번 pass로 완료됐지만, 정식 학습·benchmark의 수렴·성능은 이 세션에서
검증되지 않았다 — "정식 학습 실행 준비 완료"는 코드/테스트가 갖춰졌다는
뜻이지, 학습된 정책의 연구 성능을 증명했다는 뜻이 아니다.

다음 연구 단계는 이 runbook으로 현재 L0–L5 baseline을 동결한 뒤 진행한다.
L6–L8 구현 시에는 risk/residual/calibrator identity와 uncertainty/abstention telemetry를
checkpoint·benchmark 계약에 추가한 별도 runbook revision이 필요하다. Global G0–G8은
그 revision에서 승격된 immutable Local generation만 입력으로 사용한다.
