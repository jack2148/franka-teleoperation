# OMY-L100 → MuJoCo 텔레옵

> **Historical reference.** The text and commands below describe the earlier OMY system. Its original scene/model bundle, including the front wall, is not included or reconstructed in this source/reference release. Present-tense asset/installation descriptions are historical, not current readiness claims. See [reproduction limits](HISTORICAL_OMY_LIMITATIONS.md).

## 목표

실제 ROBOTIS OMY-L100 리더암의 관절 상태를 ROS 2로 받아 MuJoCo OMY 모델이 따라 움직이도록 한다.

```text
실제 OMY-L100 → /leader/joint_states → omy_sim_bridge.py → MuJoCo data.ctrl
```

현재 구조는 실제 로봇에서 시뮬레이터로 전달하는 단방향 동기화이다.

## 폴더 구조

```text
omy_franka_teleop/
├── config/omy_sim_sync.yaml
├── docs/omy_l100_teleop.md
├── launch/omy_sim_sync.py
├── scripts/omy_sim_bridge.py
├── open_manipulator_omy/           # ROBOTIS ROS 2 clone
├── robotis_mujoco_menagerie/       # ROBOTIS MuJoCo clone
└── mujoco_menagerie/               # FR3 모델 저장소
```

## 1. 환경 적용

```bash
source /opt/ros/humble/setup.bash
source $TELEOP_ROOT/open_manipulator_omy/install/setup.bash
```

## 2. ROS workspace 빌드

```bash
cd $TELEOP_ROOT/open_manipulator_omy
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## 3. 실제 OMY-L100 leader 실행

```bash
ros2 launch open_manipulator_bringup omy_l100_leader_ai.launch.py \
  port_name:=/dev/ttyUSB0 \
  use_sim:=false \
  use_mock_hardware:=false
```

정상 실행 후 별도 터미널에서 상태와 주기를 확인한다.

```bash
ros2 topic echo /leader/joint_states
ros2 topic hz /leader/joint_states
```

정상 joint 이름:

```text
joint1, joint2, joint3, joint4, joint5, joint6, rh_r1_joint
```

## 4. MuJoCo bridge 실행

실제 leader가 실행 중인 별도 터미널에서 실행한다.

```bash
cd $TELEOP_ROOT
source /opt/ros/humble/setup.bash
source open_manipulator_omy/install/setup.bash
/usr/bin/python3 scripts/omy_sim_bridge.py
```

## 5. 통합 launch 실행

상단 `launch/omy_sim_sync.py`는 실제 leader와 MuJoCo bridge를 함께 실행한다.

```bash
cd $TELEOP_ROOT
source /opt/ros/humble/setup.bash
source open_manipulator_omy/install/setup.bash
/usr/bin/python3 launch/omy_sim_sync.py port_name:=/dev/ttyUSB0
```

상단 launch는 ROS 패키지 내부가 아니므로 현재는 `ros2 launch`가 아니라 `/usr/bin/python3`로 실행한다.

## 관절 매핑

| ROS joint | MuJoCo actuator |
|---|---|
| `joint1` | `Joint1` |
| `joint2` | `Joint2` |
| `joint3` | `Joint3` |
| `joint4` | `Joint4` |
| `joint5` | `Joint5` |
| `joint6` | `Joint6` |
| `rh_r1_joint` | `Gripper` |

## MuJoCo 제어 흐름

```python
data.ctrl[actuator_id] = target_position
mujoco.mj_step(model, data)
viewer.sync()
```

- `data.qpos`: MuJoCo 현재 관절 위치
- `data.qvel`: MuJoCo 현재 관절 속도
- `data.ctrl`: actuator 목표값
- `mj_step()`: 물리 시뮬레이션 진행

실제 리더의 중력 보상 토크가 MuJoCo로 전달되는 것은 아니다. MuJoCo 자체 중력과 position actuator가 별도로 동작한다.

## 부드러운 움직임 설정

중력 보상만 사용하고 스프링은 끈다.

```yaml
enable_spring_effect: false
```

2·3번 관절이 뻑뻑하면 먼저 마찰 보상을 낮춰 확인한다.

```yaml
kinetic_friction_scalars:
  - 0.0
  - 0.0
  - 0.0
  - 0.1
  - 0.1
  - 0.1
```

MuJoCo actuator의 `kp`는 목표 위치를 따라가는 강도이다. `80 → 110 → 150`처럼 단계적으로 조정한다. 너무 높이면 진동하거나 불안정해질 수 있다.

### 텔레옵 동작 영상

![OMY-L100 MuJoCo 텔레옵](omy_l100_mujoco_teleop.gif)

## 다음 개발 단계

- 현재 `config/omy_sim_sync.yaml`은 아직 bridge에서 읽지 않는다.
- 현재는 실제 관절값을 MuJoCo position actuator 목표값으로 1:1 전달한다.
- 최종 목표는 증분 position control 또는 joystick velocity control이다.
- velocity 입력은 `q_target += velocity × dt` 방식으로 position actuator에 넣는 것이 간단하다.
- FR3로 확장할 때는 OMY 6축과 FR3 7축의 매핑을 별도로 정의해야 한다.
