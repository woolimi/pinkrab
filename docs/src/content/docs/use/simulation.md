---
title: 시뮬레이션
description: 하드웨어 없이 MuJoCo 에서 시연 생성, dataset 변환, 학습, 평가까지 돌리는 pinkrab-sim. 장면, 실행 순서, 시연 생성, 평가.
---

`pinkrab-sim` 은 시연 수집부터 policy 실행까지의 흐름을 하드웨어 없이 MuJoCo 에서 돌립니다. 시뮬레이션 속 Hermit 이 시연을 녹화하면 실물과 같은 변환 코드가 dataset 을 만들고, 시뮬레이션 속 팔이 학습한 policy 를 실행합니다. 손의 움직임이 팔의 움직임이 되기까지의 코드는 실물에서 도는 그대로이고, 시뮬레이션이 대신하는 것은 렌즈, 책상, 물체, 사람입니다.

렌더링은 화면 없이(headless, EGL) 하므로 모니터가 없는 GPU 컴퓨터에서도 돌아갑니다. 시연은 `~/pinkrab/sim/episodes/` 에, 평가 결과는 `~/pinkrab/sim/evals/` 에 저장됩니다.

## 장면

| 장면 | 작업 |
|---|---|
| `bowls` | 오른손이 40 mm 블록을 집어 100 mm 그릇에 넣고, 왼손은 Ready 자세에서 jaw 를 벌린 채 기다립니다. 실물에서 검증한 작업과 같습니다 |
| `blocks` | 왼팔이 흰 cap 을 집어 초록 받침 위에 올립니다. `--scene` 을 주지 않으면 이 장면입니다 |

## 실행 순서

```bash
pinkrab-sim demos --scene bowls --count 200 --batch b --workers 6   # 시연 200개
pinkrab-sim convert b --out sim-b                                    # LeRobot dataset
```

`sim-b` 는 실물 dataset 과 같은 형식이라 [학습](../train/)에 그대로 씁니다. 학습이 끝나면 checkpoint 를 점검하고 시뮬레이션 팔로 실행합니다.

```bash
pinkrab-sim swap <checkpoint>                     # policy 가 영상을 보고 움직이는지
pinkrab-sim evaluate <checkpoint> --scene bowls   # 시뮬레이션 팔로 실행
```

`<checkpoint>` 는 `~/pinkrab/train/<run>/checkpoints/last/pretrained_model` 같은 `pretrained_model` 디렉터리입니다.

## 시연 생성

`demos` 는 사람이 손으로 하듯 동작을 계획해 Hermit 을 움직이고, 팔이 매 프레임 실행 때와 같은 IK 로 Hermit 을 따라갈 수 있으며 책상이나 다른 팔과 부딪히지 않고 물체가 목표 자리에서 끝나는 시연만 남깁니다. 녹화 형식은 실물 Recorder 와 같고, 실제 손 위치를 담은 `truth.jsonl` 이 더 붙습니다.

| 인자 | 뜻 |
|---|---|
| `--count` | 남길 시연 수 |
| `--batch` | 시연 이름의 접두어. 시연은 `<batch>-000`, `<batch>-001`, … 로 저장됩니다 |
| `--scene` | `bowls` 또는 `blocks` |
| `--workers` | 시연을 나란히 렌더링할 프로세스 수 |
| `--seed` | 배치와 동작을 뽑는 seed. 기본 0 |

## 평가

`evaluate` 는 시연을 만들 수 있었던 배치로 물체를 놓고, 실물에서 policy 를 실행하기 전과 같은 시작 자세에 팔을 둔 뒤, 에피소드마다 물체를 집었는지, 책상에서 들었는지, 끝에 목표 자리에 있는지를 셉니다. 기본 판정은 jaw 가 물체를 닫으면 claw 에 붙여 옮기는 방식이고, `--physics` 를 주면 MuJoCo 의 접촉만으로 판정합니다. `--videos` 는 에피소드마다 영상을 결과 옆에 남깁니다.

`oracle` 은 policy 없이 팔이 계획한 시연을 그대로 따라가게 해, 시뮬레이션의 판정이 허용하는 성공률의 상한을 잽니다. policy 의 성공률이 이 상한보다 낮은 만큼만 policy 의 실패입니다.

`swap` 은 실물의 [실행 전 점검](../run/#실행-전-점검)과 같은 검사입니다.

장면 구성, 어안 렌즈 렌더링, 검사 스크립트는 [src/pinkrab_sim/README.md](https://github.com/woolimi/pinkrab/blob/main/src/pinkrab_sim/README.md) 에 있습니다.
