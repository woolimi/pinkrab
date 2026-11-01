---
title: 리포 구성
description: Pinkrab 저장소의 디렉터리별 역할. Python 패키지, ROS 2 노드, LeRobot plugin, 문서 사이트, 저장소 밖의 작업 폴더.
---

Pinkrab 저장소는 ROS 2 workspace 이고, `src/` 아래 패키지를 `colcon build` 로 함께 빌드합니다. 설치 방법은 [설치](../../start/install/)에 있습니다.

## 디렉터리

| 경로 | 내용 |
|---|---|
| `src/pinkrab_core` | rig 설정, 포트, 서보 bus, 각도, URDF, kinematics, LeRobot robot class |
| `src/pinkrab_cli` | `pinkrab` 명령: 포트, 서보 점검, calibration, 카메라, 렌즈 측정 |
| `src/pinkrab_description` | URDF/xacro, 메시, MJCF scene, 출력 부품과 3MF 프로젝트 |
| `src/pinkrab_umi` | Hermit: 녹화, 마커 추적, LeRobot dataset 변환, 출력 파일 |
| `src/pinkrab_studio` | 브라우저 작업대, 풋 페달, 원격 학습 |
| `src/pinkrab_policy` | rollout, 학습 실행기, 실행 기록 |
| `src/pinkrab_sim` | MuJoCo 속 UMI 흐름([시뮬레이션](../../use/simulation/)) |
| `src/pinkrab_mujoco` | headless 로 돌리고 접촉을 묻는 rig 의 MuJoCo 모델 |
| `src/pinkrab_driver` · `pinkrab_safety` · `pinkrab_teleop` · `pinkrab_bridge` | ROS 2 노드: 서보 driver, safety gate, Hermit teleoperation, studio 브리지 |
| `src/pinkrab_bringup` · `pinkrab_interfaces` | launch 파일, message 와 service |
| `lerobot-plugin` | LeRobot robot type `bi_pinkrab_follower` |
| `docs` | 이 문서 사이트(Starlight) |

## 저장소 밖의 작업 폴더

calibration, 포트 지정, 카메라 렌즈 측정, 녹화한 에피소드, dataset, 학습 결과는 모두 저장소 밖의 `~/pinkrab/` 에 저장됩니다. 이 값들은 각자의 rig 을 나타내므로 저장소에 커밋하지 않습니다.
