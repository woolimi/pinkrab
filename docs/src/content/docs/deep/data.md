---
title: 시연 데이터와 학습
description: Hermit 시연이 LeRobot dataset 이 되고, 학습한 policy 가 Pinkrab 팔을 움직이기까지의 계산. 원시 에피소드, dataset 변환의 개요, 학습, 실행.
---

Hermit 시연에는 **팔의 관절각이 없습니다**. 기록되는 것은 두 그리퍼 카메라의 영상, 탑 카메라로 잰 두 Hermit 의 pose, 두 jaw 의 열림입니다. policy 는 그리퍼 카메라 영상을 보고 tool 이 다음에 어떻게 움직일지를 tool 자신의 frame 에서 배우며, 실행할 때는 그 움직임을 팔의 IK 로 관절각으로 바꿉니다.

## 원시 에피소드

녹화 하나는 `~/pinkrab/episodes/<이름>/` 에 그대로 저장됩니다. LeRobot dataset 은 나중에 이것으로 다시 만들 수 있지만, 영상과 프레임마다의 숫자는 **나중에 되살릴 수 없기** 때문입니다.

| 파일 | 내용 |
|---|---|
| `gripper_left.mp4`, `gripper_right.mp4` | 두 그리퍼 카메라의 영상. 탑 카메라 영상은 저장하지 않습니다 |
| `frames.jsonl` | 카메라 프레임마다 어느 카메라의 몇 번째 프레임인지와 시각. 탑 카메라도 영상 없이 프레임 시각을 남깁니다 |
| `poses.jsonl` | 탑 카메라 프레임마다 찾은 마커 다면체의 pose(카메라 frame 과 테이블 마커 frame), 그 순간 두 jaw 의 열림, 그 프레임에서 읽은 모든 마커의 네 모서리 픽셀 좌표(`markers`) |
| `robot.jsonl` | 실행 기록에만 있습니다. 팔의 관절각, gate 의 mode, 그 순간 누가 팔을 움직였는지(`controller`), 몇 번째 교정인지 |
| `events.jsonl` | 시작과 끝, 실행 기록에서는 다시 한 교정 |
| `meta.json` | 시계, 탑 카메라 calibration, 테이블 마커의 크기와 방향, 에피소드 종류, 마커 다면체마다 모든 마커의 네 모서리(다면체 frame, m)와 ArUco dictionary, 그리퍼 카메라마다 렌즈 fit 전체(모델, 계수, 프레임 크기, image circle), 실행 기록이면 판정(`outcome`) |

모든 줄에는 그 데이터가 이 컴퓨터에 도착한 시각(`host_ts`)이 있고, 세 카메라와 jaw 는 나중에 이 시계로 맞춥니다.

`meta.json` 에 마커 모서리와 렌즈 fit 이 함께 들어 있어서, 에피소드 폴더 하나만 있으면 이 코드 없이도 `poses.jsonl` 의 마커 모서리 좌표로 마커 pose 를 다시 풀고 그리퍼 영상의 왜곡을 펼 수 있습니다. 탑 카메라 렌즈를 다시 측정하면 `pinkrab-umi resolve` 가 이 좌표로 지난 시연의 pose 를 새 렌즈로 다시 풉니다. 탑 카메라 영상은 방과 시연자를 찍을 뿐 policy 가 보지 않아서 저장하지 않고, 그 대신 남기는 모서리 좌표는 영상을 압축하며 생기는 오차도 없습니다. 다른 pose 추정 방법을 쓰려면 그 결과를 `poses.jsonl` 과 같은 형식으로 쓰면 Convert 가 그대로 받습니다.

## dataset 변환

Convert 는 원시 에피소드를 LeRobot dataset 으로 바꿉니다. 마커 pose 로 tool 궤적을 만들고, dataset 의 fps 에 맞춰 프레임을 놓고, 손마다 tool frame 의 상대 동작을 계산합니다. 단계별 계산과 dataset 구성, fps, 상대 동작을 쓰는 이유는 [dataset 변환](../convert/)에 있습니다.

## 학습

[학습](../../use/train/)은 LeRobot 의 `lerobot-train` 이 합니다. state 와 동작은 dataset 통계로 정규화되고, 학습 중 영상에는 밝기·대비 등의 무작위 변형(image augmentation)이 걸립니다. policy 는 한 번에 동작 하나가 아니라 앞으로의 동작 여러 step(chunk)을 내놓도록 학습합니다. 구조는 [policy 개요](../policies/)와 [ACT](../policies/act/) 에 있습니다.

## 실행

1. **시작 자세**: 두 팔이 Ready 자세를 거쳐 시연들의 첫 TCP pose 평균으로 갑니다. 평균은 `pose.tcp_<hand>` 를 테이블 마커 규약으로 로봇 좌표에 옮겨 구하므로, 두 팔은 시연에서 두 손이 있던 간격과 방향으로 시작합니다. **다른 자리에서 시작하면** policy 는 학습에서 본 적 없는 영상을 보게 됩니다.
2. **관찰**: 팔의 그리퍼 카메라 영상을 dataset 과 같은 방식으로 자르고 줄입니다. state 는 두 팔 jaw 의 열림과, 두 팔의 forward kinematics 로 계산한 상대 pose 입니다.
3. **추론**: policy 는 dataset 의 fps 로 한 step 씩 실행합니다. chunk 를 내면 그 앞의 **몇 step 만 실행하고** 다시 카메라를 봅니다(기본으로 0.1 초 분량, 기본 50 fps 에서 5 step). 남은 step 을 버리고 새로 묻는 것은 물체가 예상과 다르게 움직였을 때 바로 반응하기 위해서입니다.
4. **동작 적용**: 각 동작의 변화를 그 팔에 마지막으로 보낸 tool pose 에 곱하고(왼팔은 tool frame 의 반 바퀴만큼 돌려서), teleoperation 과 같은 `track()` 으로 관절각을 구합니다. jaw 는 열림 비율을 팔의 jaw 범위로 옮깁니다.
5. **멈춤**: policy 가 숫자가 아닌 값을 내거나, IK 가 요청한 자리에 크게 못 미치거나, 관절이 한계에 닿은 채 policy 가 계속 그 너머로 가라고 해서 요청과 실제의 차이가 쌓이면 **실행을 멈춥니다**. 실행 중 관절 속도는 gate 가 **초당 60°** 로 제한합니다.

## 교정 데이터

[실행](../../use/run/) 기록도 같은 Convert 로 dataset 이 되지만, 영상과 동작을 Hermit 이 아니라 팔에서 계산합니다. 녹화 방식과 원리는 [교정 데이터](../corrections/)에, 변환은 [dataset 변환](../convert/#실행-기록의-변환)에 있습니다.
