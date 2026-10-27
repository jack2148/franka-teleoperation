# OMY–FR3 Cartesian 텔레오퍼레이션 개발 문서

> **Historical reference.** The text and commands below describe the earlier OMY system. Its original scene/model bundle, including the front wall, is not included or reconstructed in this source/reference release. Present-tense asset/installation descriptions are historical, not current readiness claims. See [reproduction limits](HISTORICAL_OMY_LIMITATIONS.md).

> Current architecture and implementation reference

## 1. 문서 목적과 구현 범위

이 문서는 실제 OMY-L100 leader의 joint state를 MuJoCo OMY FK와 Cartesian
retargeting을 거쳐 MuJoCo FR3 command로 변환하는 현재 architecture를 설명한다.

현재 bridge는 `position_only`, `orientation_only`, `full_pose`를 지원하며,
MuJoCo free-space에서 unilateral Cartesian teleoperation을 검증하는 baseline이다.

설치·실행·CLI 사용법은 [README.md](../README.md), 원인과 증거 중심의 debugging
기록은 [position_mode_debugging.md](position_mode_debugging.md)에서 다룬다.
실제 FR3, joystick, hardware assistance, precision/contact/haptic은 후속 범위다.

## 2. 시스템 아키텍처

현재 end-to-end data flow는 다음과 같다.

```text
OMY-L100 /leader/joint_states
  → joint name 기반 state extraction
  → OMY MuJoCo qpos synchronization
  → mujoco.mj_forward()
  → OMY EE current pose
  → clutch-relative Cartesian increment
  → position/orientation frame mapping
  → cumulative logical command pose
  → rate-limited conditioned target pose
  → Cartesian task velocity
  → velocity-based DLS IK
  → optional null-space posture velocity
  → joint-speed limiting
  → qdot * dt integration
  → FR3 actuator position command
  → MuJoCo FR3 actual state

```

| Component | Responsibility | Source |
|---|---|---|
| OMY state interface | `/leader/joint_states` 수신, joint name 확인 및 최신 state 저장 | `OmyPose`, `FR3_omy_bridge.py` |
| OMY FK | 수신 joint를 OMY MuJoCo `qpos`에 반영하고 EE pose 계산 | `main()`, `read_site_pose()` |
| Clutch mapper | runtime anchor 기준 relative motion과 cumulative command 생성 | `make_desired_target()`, main loop |
| Target conditioner | command pose를 speed/acceleration 제한 target으로 변환 | `condition_target()` |
| DLS controller | Cartesian task를 FR3 joint velocity로 변환 | `compute_joint_target()`, `dls_pseudoinverse()` |
| Null-space controller | optional `q_home` posture objective 계산 | `compute_joint_target()` 내부 SVD 기반 projector |
| FR3 simulator interface | actuator command 적용, MuJoCo step, actual EE pose 조회 | main loop |
| Logger | pose, target dynamics, IK diagnostics를 실행별 CSV로 저장 | `write_log()`, main loop |

현재 별도의 `ClutchMapper` 또는 `TargetConditioner` class는 없으며, 해당
책임은 함수와 main loop state가 담당한다.

## 3. 주요 상태와 데이터 소유권

### Leader state

`OmyPose`가 `/leader/joint_states`에서 필요한 이름을 확인한 뒤 최신 OMY
joint state, trigger 값, 수신 시각을 state lock 아래에 저장한다.

- OMY current joint state: `joint_positions`
- OMY current EE pose: `omy_current_position`, `omy_current_rotation`
- OMY clutch anchor pose: `omy_anchor_position`, `omy_anchor_rotation`
- clutch ID: `clutch_id`

### Logical command

`fr3_command_position`, `fr3_command_rotation`은 OMY motion을 누적한
unconditioned command pose다. speed/acceleration limit을 적용하기 전의
logical reference이며 다음 clutch의 FR3 anchor로 사용된다.

### Conditioned target

`fr3_target_position`, `fr3_target_rotation`은 logical command를 향해
`condition_target()`이 제한된 속도와 가속도로 이동시키는 current target이다.
함께 유지되는 state는 `target_linear_velocity`와 `target_angular_velocity`다.

### Actual follower state

FR3 actual q와 EE pose는 MuJoCo `data.qpos` 및 site state에서 읽는다.
actual pose는 command anchor를 대체하지 않으며, Cartesian task error와
tracking diagnostics 계산에 사용된다.

### Joint command state

| State | 갱신 주체 | Trigger ON | Trigger OFF | ROS timeout |
|---|---|---|---|---|
| `omy_anchor_*` | main loop | 현재 OMY pose로 새로 저장 | 유지 | 마지막 값 유지 |
| `fr3_command_*` | main loop | 현재 logical command를 anchor로 사용 후 active 중 갱신 | 마지막 command 유지 | 현재 conditioned target로 freeze |
| `fr3_target_*` | `condition_target()` | command를 향해 갱신 | command에 계속 수렴 | stale branch에서 command와 함께 target 기준으로 freeze |
| target velocity | `condition_target()` | zero로 시작 후 갱신 | 수렴 과정 동안 유지 | zero로 reset |
| `hold_q_target` | main loop/IK | qdot 적분 결과로 갱신 | target 수렴 후 hold | 마지막 actuator command hold |

Trigger rising edge에서 `fr3_anchor_position/rotation`은 actual FR3 pose가
아니라 현재 `fr3_command_position/rotation`에서 캡처된다. Trigger OFF는
leader 입력만 중단하고 마지막 logical command는 유지한다.

## 4. OMY 상태 수신과 Forward Kinematics

`OmyPose.joint_state_callback()`은 `/leader/joint_states` 메시지에서 다음
이름을 요구한다.

```text
joint1, joint2, joint3, joint4, joint5, joint6, rh_r1_joint

```

여섯 개 arm joint는 `OMY_ROS_JOINTS` 순서로 저장하고 trigger는
`TRIGGER_JOINT = "rh_r1_joint"`에서 읽는다. 필요한 이름이 하나라도 없으면
해당 메시지를 무시한다.

main loop는 저장된 state를 OMY MuJoCo model의 다음 joint 주소에 반영한다.

```python
for address, position in zip(omy_qpos_addresses, omy_target):
    omy_data.qpos[address] = position
mujoco.mj_forward(omy_model, omy_data)

```

OMY model은 FK 용도다. 현재 source의 frame identifiers는 다음과 같다.

| Item | Current value |
|---|---|
| OMY model | `robotis_mujoco_menagerie/robotis_omy/scene.xml` |
| OMY base body | `base_unit` |
| OMY EE site | `omy_ee_site` |
| OMY MuJoCo joints | `Joint1` … `Joint6` |

EE pose는 `data.site_xpos`와 `data.site_xmat`에서 읽는다. base-frame
inspection이 필요할 때는 `read_base_ee_transform()`이
`inverse(T_world_base) @ T_world_ee`를 계산한다.

수신 state가 `ROS_TIMEOUT_S = 0.20`보다 오래되면 stale branch로 들어간다.
이 branch에서는 teleoperation을 비활성화하고 target dynamics를 zero로
설정하며, logical command를 현재 conditioned target로 freeze한다.

## 5. 좌표계와 Cartesian mapping

현재 source는 OMY와 FR3의 MuJoCo base body/site를 별도로 조회하고 startup
시점에 world-frame base와 base-relative EE transform을 출력한다.

| Robot | Base body | EE site |
|---|---|---|
| OMY | `base_unit` | `omy_ee_site` |
| FR3 | `base` | `attachment_site` |

`read_body_transform()`, `read_site_transform()`, `read_base_ee_transform()`이
homogeneous transform을 구성한다. rotation은 determinant와
orthonormality error도 출력한다. model base alignment는 실제 hardware
설치 frame과 동일하다는 뜻이 아니다.

Position과 orientation mapping은 독립적으로 정의된다.

```python
POSITION_AXIS_MAP = np.array([
    [0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
])

ORIENTATION_AXIS_MAP = np.array([
    [0.0, -1.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
])

```

현재 position mapping의 의미는 다음과 같다.

```text
FR3 X =  OMY Y
FR3 Y = -OMY X
FR3 Z =  OMY Z

```

`make_desired_target()`은 position에 `POSITION_SCALE`과
`POSITION_AXIS_MAP`을 적용한다. orientation은 Euler subtraction이 아니라
relative rotation matrix와 rotation vector를 사용한다.

```python
R_omy_rel = R_omy_anchor.T @ R_omy_current
r_omy_rel = matrix_to_rotvec(R_omy_rel)
r_fr3 = ORIENTATION_SCALE * (ORIENTATION_AXIS_MAP @ r_omy_rel)
R_desired = R_fr3_anchor @ rotvec_to_matrix(r_fr3)

```

Position mapping 검증과 orientation sign 검증은 별개다. 현재 orientation
validation은 upward/downward와 right/left의 반대 sign을 확인한 제한된 범위며,
hardware installation frame calibration 또는 전체 roll/pitch/yaw 검증 완료를
의미하지 않는다. 상세 evidence는
[position mode debugging](position_mode_debugging.md#position-axis-mapping-error)을
참조한다.

## 6. Clutch 기반 command 생성

Trigger threshold는 `TRIGGER_ON_THRESHOLD = -0.90`,
`TRIGGER_OFF_THRESHOLD = -0.70`이다.

| Event | OMY anchor | Logical command | Conditioned target | Target velocity |
|---|---|---|---|---|
| Trigger ON | 현재 OMY EE pose 저장 | 현재 command를 FR3 anchor로 사용 | 기존 target에서 계속 상태 유지 | linear/angular zero로 시작 |
| Trigger active | anchor-relative OMY motion 사용 | mode에 따라 position/rotation 갱신 | command를 향해 제한 갱신 | conditioner state 유지 |
| Trigger OFF | 마지막 anchor 유지 | 마지막 command hold | 마지막 command로 계속 수렴 | 수렴 전까지 유지 |
| ROS timeout | 마지막 anchor 유지 | 현재 target로 freeze | stale branch에서 추가 갱신 안 함 | zero로 reset |

active stroke에서 `make_desired_target()`은 OMY anchor-relative position과
rotation을 FR3 anchor에 적용한다.

```text
OMY current pose - OMY anchor pose
  → independent position/orientation mapping
  → FR3 command anchor + mapped increment
  → cumulative logical command

```

`position_only`에서는 FR3 rotation을 `fr3_anchor_rotation`에 hold하고,
`orientation_only`에서는 FR3 position을 `fr3_anchor_position`에 hold한다.
`full_pose`에서는 두 command를 모두 갱신한다. command가 누적되므로 leader가
한 stroke에서 사용할 수 있는 workspace보다 넓은 Cartesian motion을 여러
clutch operation으로 표현할 수 있다.

## 7. Command와 conditioned target

controller 입력의 상태 관계는 다음과 같다.

```text
OMY leader input
  → cumulative logical command
  → conditioned target
  → FR3 actual pose

```

logical command에는 직접 target speed/acceleration limit을 적용하지 않는다.
`condition_target()`은 current target에서 command로 이동하며 linear와 angular
velocity state를 각각 유지한다.

### Linear conditioner

position error norm을 `e_p`라고 하면 current source는 다음 속도를 계산한다.

```python
stopping_speed = sqrt(2 * MAX_TARGET_LINEAR_ACCEL * error_norm)
proportional_speed = POSITION_KP * error_norm
desired_speed = min(
    proportional_speed,
    MAX_TARGET_LINEAR_SPEED,
    stopping_speed,
)

```

이후 error 방향으로 desired velocity를 만들고, velocity 변화량을
`MAX_TARGET_LINEAR_ACCEL * dt`까지 제한한 뒤 position을 velocity로 적분한다.
snap은 위치 error와 이전/현재 velocity가 모두 작은 경우에만 허용한다.

### Angular conditioner

angular error는 다음 relative rotation에서 rotation vector로 계산한다.

```python
R_error = R_desired @ R_target.T
r_error = matrix_to_rotvec(R_error)

```

error angle에서 stopping speed를 계산하고, `TARGET_ROTATION_KP`,
`MAX_TARGET_ANGULAR_SPEED`, `MAX_TARGET_ANGULAR_ACCEL`을 사용해 angular
velocity를 제한한다. target rotation은 제한된 rotation step을 current target
앞에 곱해 갱신한다. `ROTATION_SNAP_ERROR`보다 작으면 rotation을 desired pose에
맞추고 angular velocity를 zero로 만든다.

주요 conditioning parameter는 다음과 같다.

- `MAX_TARGET_LINEAR_SPEED`, `MAX_TARGET_LINEAR_ACCEL`
- `MAX_TARGET_ANGULAR_SPEED`, `MAX_TARGET_ANGULAR_ACCEL`
- `TARGET_ROTATION_KP`
- `POSITION_SNAP_ERROR`, `POSITION_SNAP_SPEED`
- `ROTATION_SNAP_ERROR`

Trigger OFF에서는 command만 hold하고 conditioner가 마지막 command까지
수렴한다. ROS timeout은 stale branch로 전환되어 target dynamics를 zero로
reset하고 logical command를 current target에 freeze한다.

## 8. Teleoperation mode

| Mode | Position command | Orientation command | Main use |
|---|---|---|---|
| `position_only` | Update | `fr3_anchor_rotation`에 hold | translation mapping/IK validation |
| `orientation_only` | `fr3_anchor_position`에 hold | Update | orientation mapping validation |
| `full_pose` | Update | Update | normal teleoperation |

현재 default는 `TELEOP_MODE = "full_pose"`다. `full`은 지원 mode가 아니며,
mode validation은 `SUPPORTED_TELEOP_MODES`와 main 시작부에서 수행한다.

mode isolation에서는 signed position XYZ와 signed rotation-vector XYZ를
분리해 확인한다. orientation 검증은 제한된 단일 방향 조작의 sign 확인이며,
전체 orientation calibration 완료를 의미하지 않는다.

## 9. Velocity-based DLS IK

`compute_joint_target()`은 FR3 actual EE pose와 conditioned target의 차이를
task velocity로 변환한다.

```text
position_error = p_target - p_current
rotation_error = log(R_target R_currentᵀ)
v_position = POSITION_KP * position_error
v_rotation = ORIENTATION_KP * rotation_error

```

MuJoCo `mj_jacSite()`에서 position Jacobian `J_position`과 rotational
Jacobian `J_rotation`을 계산한다. mode별 task는 다음과 같다.

| Mode | Task Jacobian | Task velocity |
|---|---|---|
| `position_only` | `J_position` | `v_position` |
| `orientation_only` | `ROTATION_LENGTH_SCALE * J_rotation` | `ROTATION_LENGTH_SCALE * v_rotation` |
| `full_pose` | `vstack(J_position, ROTATION_LENGTH_SCALE * J_rotation)` | concatenate position/rotation task |

`full_pose` task Jacobian은 6×7이다. DLS pseudoinverse는 current source에서
다음 normal-equation 형태로 계산한다.

```python
J_dls = J.T @ solve(J @ J.T + damping**2 * I, I)
qdot_task = J_dls @ task_velocity

```

`ROTATION_LENGTH_SCALE`은 rotational Jacobian과 angular task velocity 양쪽에
적용되어 position과 rotation의 metric unit을 맞춘다.

joint command 순서는 다음과 같다.

```text
qdot_task
  → optional qdot_null addition
  → [-MAX_JOINT_SPEED, MAX_JOINT_SPEED] clipping
  → q_target = hold_q_target + qdot_command * dt
  → joint range clipping
  → data.ctrl

```

초기 per-cycle `dq` 제한은 frequency-dependent하여 현재는 velocity-level
command와 `qdot * dt` 적분으로 대체되었다. `CONTROL_HZ`는 configured nominal
rate이며 Python loop의 measured hard real-time 보장을 의미하지 않는다.
상세 causal debugging은
[velocity IK 전환 기록](position_mode_debugging.md#per-cycle-dq-velocity-ik)을
참조한다.

## 10. Null-space posture control

FR3는 7DoF이므로 Cartesian task 외 posture 방향이 존재한다. 현재 optional
posture objective는 `q_home`을 reference로 사용한다.

```python
qdot_posture = NULLSPACE_GAIN * (q_home - q_current)
qdot_null = N @ qdot_posture
qdot_total = qdot_task + qdot_null

```

`N`은 current task Jacobian의 SVD 기반 null-space projector다. position-only와
full-pose는 task row 수가 다르므로 redundancy도 다르다.

현재 baseline에는 OMY axis를 FR3 특정 q6/q7에 직접 매핑하는 로직이 없다.
`ENABLE_NULLSPACE_POSTURE=True`, `NULLSPACE_GAIN=0.1`은 current experimental
configuration이며 optimal gain을 의미하지 않는다. workspace posture benefit,
joint-limit avoidance, task interference는 정량 검증되지 않았다.

## 11. MuJoCo FR3 command 적용

FR3 model은 IK와 actuator dynamics 용도로 사용한다.

| Variable | Meaning | Used for |
|---|---|---|
| `q_current` | MuJoCo FR3 current joint position | IK state/error 계산 |
| `hold_q_target` | held/integrated actuator command | 다음 command의 integration reference |
| `qdot_task` | Cartesian task velocity 결과 | primary task motion |
| `qdot_null` | null-space posture velocity | optional redundancy objective |
| `qdot_command` | joint speed clipping 후 velocity | actuator target 적분 |
| `q_target` | joint range clipping 후 position target | `data.ctrl` 입력 |

현재 source는 `fr3_joint1`부터 `fr3_joint7`까지의 position actuator를 사용한다.
main loop는 다음 순서로 실행한다.

```python
fr3_data.ctrl[fr3_actuator_indices] = q_target
mujoco.mj_step(fr3_model, fr3_data)
fr3_actual_position, fr3_actual_rotation = read_site_pose(
    fr3_data,
    fr3_ee_site_id,
)

```

MuJoCo position actuator 구조를 real FR3 torque/position controller와 동일한
제어기라고 해석하지 않는다.

## 12. Logging 및 diagnostics

`ENABLE_LOGGING=True`이면 실행 종료 시 rows를
`logs/refactored_teleop_YYYYMMDD_HHMMSS.csv`에 저장한다.

| Group | Contents |
|---|---|
| Teleoperation state | `teleop_mode`, active state, `clutch_id`, `wall_time`, `sim_time`, `control_dt` |
| Cartesian state | command/target/actual position과 session-relative rotation vector |
| Tracking | position/orientation error, command-target rotation gap |
| Target dynamics | linear/angular speed 및 acceleration norm |
| IK/controller | task velocity norm, raw/commanded qdot, qdot task norm |
| Solver | task residual, Jacobian condition, joint speed saturation |
| Joint state | `fr3_joint_1`부터 `fr3_joint_7`까지 |

조건부 diagnostics flag는 다음과 같다.

| Flag | Purpose |
|---|---|
| `ENABLE_MAPPING_DEBUG_LOGS` | OMY raw delta와 mapped position/rotation signed components |
| `ENABLE_OMY_INTERNAL_DEBUG_LOGS` | OMY base-relative pose와 clutch-relative base diagnostics |
| `ENABLE_NULLSPACE_DEBUG_LOGS` | 선언된 null-space diagnostics option; 현재 main loop의 CSV 분기는 없음 |

기본 mapping/OMY internal debug logging은 `False`다. plot 생성 방법과 CSV
파일 지정법은 [README의 CSV/plot section](../README.md#5-csv-확인과-plot-생성)을
사용한다.

## 13. 현재 설정

현재 `launch/FR3_omy_bridge.py`에서 확인한 MuJoCo baseline은 다음과 같다.

| Parameter | Current value |
|---|---:|
| `TELEOP_MODE` | `"full_pose"` |
| `POSITION_SCALE` | `0.4` |
| `ORIENTATION_SCALE` | `0.3` |
| `POSITION_MAP_CANDIDATE` | `"current_90"` |
| `MAX_TARGET_LINEAR_SPEED` | `0.09 m/s` |
| `MAX_TARGET_LINEAR_ACCEL` | `2.0 m/s²` |
| `MAX_TARGET_ANGULAR_SPEED` | `2.0 rad/s` |
| `MAX_TARGET_ANGULAR_ACCEL` | `4.0 rad/s²` |
| `TARGET_ROTATION_KP` | `4.0` |
| `POSITION_KP` | `8.0` |
| `ORIENTATION_KP` | `4.0` |
| `IK_DAMPING` | `0.05` |
| `ROTATION_LENGTH_SCALE` | `0.10 m/rad` |
| `MAX_JOINT_SPEED` | `0.80 rad/s` |
| `ENABLE_NULLSPACE_POSTURE` | `True` |
| `NULLSPACE_GAIN` | `0.1` |
| `CONTROL_HZ` | `1000 Hz` |
| `LOG_HZ` | `100 Hz` |
| `ENABLE_LOGGING` | `True` |
| `ENABLE_MAPPING_DEBUG_LOGS` | `False` |
| `ENABLE_OMY_INTERNAL_DEBUG_LOGS` | `False` |
| `ENABLE_NULLSPACE_DEBUG_LOGS` | `False` |

`POSITION_AXIS_MAP`은 `current_90` 후보에서 선택되고,
`ORIENTATION_AXIS_MAP`은 별도 matrix로 정의된다. threshold는
`TRIGGER_ON_THRESHOLD=-0.90`, `TRIGGER_OFF_THRESHOLD=-0.70`,
`POSITION_SNAP_ERROR=1e-6`, `POSITION_SNAP_SPEED=1e-3`,
`ROTATION_SNAP_ERROR=1e-4`, `ROS_TIMEOUT_S=0.20`이다.

이 값들은 MuJoCo free-space baseline이다. `CONTROL_HZ`는 nominal configured
rate이며 measured hard real-time rate가 아니다. real FR3 controller parameter로
직접 사용할 수 없다.

## 14. 구현 상태와 검증 범위

| Feature | Implementation | Validation scope |
|---|---|---|
| OMY ROS joint-state synchronization | 구현 및 확인 | `/leader/joint_states` state path |
| OMY MuJoCo FK | 구현 및 확인 | OMY EE site pose |
| Runtime clutch anchor | 구현 및 확인 | tested MuJoCo clutch runs |
| Cumulative logical command | 구현 및 확인 | repeated clutch behavior |
| Position mapping | 구현 및 확인 | tested one-axis directions |
| Orientation mapping | 구현 | upward/downward/right/left sign 범위 |
| Three teleoperation modes | 구현 및 확인 | mode isolation |
| Command-target separation | 구현 및 확인 | target convergence behavior |
| Linear/angular conditioner | 구현 | tested MuJoCo configuration |
| Velocity-based DLS IK | 구현 및 확인 | representative free-space runs |
| Full-pose 6D DLS | 구현 | representative free-space runs |
| qdot saturation logging | 구현 및 확인 | raw/commanded qdot fields |
| Null-space posture term | 구현 완료, 정량 미검증 | workspace/posture effect 미검증 |
| Joystick | 미구현 | 후속 범위 |
| Passive gravity compensation | 미구현 | hardware 후속 범위 |
| Precision/contact | 미구현 | 후속 범위 |
| Haptic feedback | 미구현 | 후속 범위 |
| Real FR3 | 보류 | safety validation 전 |
| PushT dataset pipeline | 미구현 | 별도 환경 작업 |

정량 validation의 제한은 manual trajectory, scripted replay 부재,
workspace-wide test 부재, contact/hardware test 부재, real robot 부재다.

## 15. 확장 구조

### 15.1 Joystick mode

```text
joystick input layer
  → incremental Cartesian command source
  → existing logical command
  → target conditioner
  → existing DLS stack

```

새 IK controller를 만드는 대신 command-generation front end를 추가하는
구조로 확장한다.

### 15.2 Hardware assistance

```text
OMY physical leader
  → passive gravity compensation
  → same ROS joint-state / FK pipeline

```

러버밴드나 스프링은 Cartesian mapping code와 분리된 physical layer로 다룬다.

### 15.3 Precision and haptic control

```text
FR3 contact/wrench state
  → force filtering/scaling
  → compliance or leader feedback path

```

현재 unilateral kinematic pipeline 위에 contact-aware feedback/controller를
추가하는 후속 단계다. 세 확장 기능의 상세 설계는 아직 구현하지 않는다.

## 16. 관련 문서

| Document | Responsibility |
|---|---|
| [README.md](../README.md) | 설치, 실행, 사용법, CSV/plot 확인 |
| [position_mode_debugging.md](position_mode_debugging.md) | 주요 증상, 원인, 수정, 검증 제한 |
| `joystick_mode_debugging.md` | joystick mode 후속 문서 예정 |
| `hardware.md` | hardware, passive gravity compensation, safety 후속 문서 예정 |
| `precision.md` | precision, contact, compliance, 정량 평가 후속 문서 예정 |
| [archived development log](archive/development_log_20260723_20260730.md) | 날짜별 원본 개발 기록 |
