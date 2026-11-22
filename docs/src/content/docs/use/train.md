---
title: 학습
description: 수집한 dataset 으로 policy 를 학습하는 방법. 학습할 컴퓨터, studio 의 Train 페이지, 학습 설정, 진행 상태와 checkpoint.
---

[Curate](../curate/) 페이지에서 만든 LeRobot dataset 으로 [LeRobot](https://github.com/huggingface/lerobot) 에서 policy 를 학습합니다. 학습은 studio 의 Train 페이지에서 시작하고 지켜보며, studio 를 실행한 컴퓨터나 ssh 로 연결한 GPU 컴퓨터에서 돌립니다.

## Policy

policy 는 **[ACT](../../deep/policies/act/)** (Action Chunking with Transformers)입니다. LeRobot 에 들어 있는 policy 로, 두 그리퍼 카메라의 영상과 두 jaw 의 열림, 두 tool 의 상대 위치를 받아 앞으로 할 동작 여러 step 을 한 묶음(chunk)으로 내놓습니다. ResNet-18 이 읽은 영상을 transformer 가 받아 chunk 를 만들고, 학습할 때 CVAE 를 함께 써서 같은 장면에 맞는 동작이 여러 가지일 때 그 평균이 아니라 하나를 고르게 합니다. 기본값인 **batch 8** 로 학습할 때 GPU 메모리는 최대 약 5.7 GB 입니다. chunk 의 각 칸은 그 chunk 를 시작한 tool pose 에서 본 앞으로의 tool pose 이고, 학습을 맡는 `pinkrab-train` 이 dataset 의 step 들을 이렇게 바꿔 가르칩니다([chunk 의 표현](../../deep/convert/#chunk-의-표현)).

## 학습할 컴퓨터

| 항목 | 조건 |
|---|---|
| 시스템 | x86_64 Linux, `python3`·`curl`, 처음 설정할 때 인터넷 |
| GPU | NVIDIA, compute capability 7.5 이상(T4, RTX 20–50 시리즈, A10, A100, L4, L40S, H100 등) |
| 드라이버 | 580 이상(CUDA 13.0). CUDA toolkit 은 따로 설치하지 않아도 됩니다 |
| GPU 메모리 | 8 GB 이상 |
| CPU | 8 코어 이상. batch 마다 영상을 디코딩하는 일이 CPU 몫입니다 |
| 디스크 | 약 15 GB 와 dataset, checkpoint 를 둘 자리 |

기본 batch 8 로는 **8 GB GPU 한 장**에서 학습합니다. 학습할 컴퓨터에 필요한 Python 과 패키지는 처음 학습을 시작할 때 studio 가 그 컴퓨터에 설치합니다.

## Train 페이지

studio 첫 화면의 Train 카드를 누르면 http://localhost:8730/train 이 열립니다. 왼쪽에는 학습(run) 목록이 최신순으로, 오른쪽에는 loss 같은 지표마다 차트가 하나씩 있습니다. 카드를 누르면 그 run 하나를, 체크하면 여러 run 을 한 차트에 겹쳐 그립니다.

학습은 **studio 를 닫아도 계속되고**, studio 를 다시 열면 그동안 기록된 값을 읽어 와 이어서 그립니다.

## 원격 GPU 컴퓨터

studio 를 실행한 컴퓨터에 GPU 가 없으면, ssh 로 접속할 수 있는 다른 GPU 컴퓨터에서 학습할 수 있습니다. 집에 있는 다른 컴퓨터든 클라우드에서 빌린 서버든 [학습할 컴퓨터](#학습할-컴퓨터)의 조건만 맞으면 됩니다. Train 페이지의 **Sources** 에 그 컴퓨터를 추가합니다.

| 칸 | 내용 |
|---|---|
| name | 아무 이름, 예: `gpu-box` |
| ssh 줄 | `ssh -i ~/.ssh/id_ed25519 -p 22 user@gpu.example.com` 같은 접속 명령 |
| key | 개인 키 파일. `~/pinkrab/keys/` 에 source 별 파일로 복사해 이 사용자만 읽게 해 두고 쓰며, source 를 지우면 함께 지웁니다. 비밀번호 로그인은 쓰지 않습니다 |
| runs are under | 그 컴퓨터에서 dataset, Python 환경, 학습 결과를 둘 폴더, 기본 `~/pinkrab-train` |

저장한 뒤 **test** 를 누르면 그 컴퓨터의 GPU, 메모리, 드라이버, CPU, 남은 디스크를 확인하고, 학습에 모자란 것이 있으면 무엇인지 알려 줍니다.

ssh 는 처음 접속하는 컴퓨터의 host key 를 받아 `~/pinkrab/keys/` 의 source 별 `known_hosts` 파일에 적어 두고, 이후에는 그 key 와 같을 때만 접속합니다. 클라우드 세션은 같은 주소와 포트로 새 컴퓨터가 올 수 있어 host key 가 바뀌면 접속이 거절되고 오류 줄에 source 를 다시 저장하라고 나옵니다. Sources 에서 그 source 를 다시 저장하면 적어 둔 key 를 비우고 새 key 를 받습니다.

원격 컴퓨터에서 학습해도 다루는 방법은 이 컴퓨터에서 학습할 때와 같습니다.

- **보내기**: 학습을 시작하면 studio 가 dataset 을 그 컴퓨터로 보내고, 필요한 Python 환경을 설치한 뒤 학습을 띄웁니다.
- **지표**: studio 가 30 초마다 그 컴퓨터의 학습 기록을 읽어 와 이 컴퓨터에 쌓으므로, loss 같은 지표가 Train 페이지에 이 컴퓨터의 run 과 똑같이 그려집니다. 연결이 끊겼다 다시 이어지면 그동안의 기록도 이어서 가져옵니다.
- **checkpoint**: checkpoint 는 저장될 때마다 이 컴퓨터로 복사됩니다. studio 가 꺼져 있던 동안 저장된 것은 다시 켜질 때 가져옵니다. 복사된 checkpoint 는 `~/pinkrab/train/<run>@<source>/checkpoints/` 에 source 별 폴더로 놓이므로 source 가 다른 같은 이름의 run 은 서로 덮어쓰지 않습니다. 바로 Policy 페이지에서 [실행](../run/)할 수 있습니다.

## 새 학습

**New run** 에서 다음을 고르고 Start 를 누릅니다.

| 칸 | 내용 |
|---|---|
| machine | 학습할 컴퓨터 |
| dataset | `~/pinkrab/datasets/` 의 dataset. 원격 컴퓨터에는 처음 한 번 보내고, 다시 변환한 dataset 은 새로 보냅니다 |
| policy | ACT |
| batch | 비워 두면 기본값 8. 늘리면 GPU 메모리를 그만큼 더 씁니다. batch 16 이면 ACT 는 약 10.3 GB 입니다 |
| steps | 기본 100 000 |
| log every, checkpoint every | 기본 50, 20 000. log 간격을 줄이면 차트가 촘촘해질 뿐 학습 속도는 그대로입니다 |
| name | 비워 두면 `policy-dataset-날짜` |
| more arguments | `lerobot-train` 이 받는 다른 인자, 예: `--policy.optimizer_lr=5e-5` |

화면 위 노란 줄에 단계가 차례로 표시됩니다. 원격 컴퓨터는 처음 학습할 때 Python 환경을 만드느라 몇 분 걸리고, 그다음부터는 몇 초면 시작합니다. 학습은 접속과 상관없이 그 컴퓨터에서 따로 실행되므로 **브라우저를 닫거나 연결이 끊겨도 계속됩니다**.

## 진행 상태

| 상태 | 뜻 |
|---|---|
| running | 최근 5분 안에 기록이 늘었습니다 |
| done | 마지막 step 까지 학습했습니다 |
| stopped | 여기서 멈췄거나 중단됐습니다 |
| failed | 오류로 끝났고, 오류 내용이 카드 아래에 보입니다 |
| quiet | 5분 넘게 기록이 없고 끝나지도 않았습니다. 프로세스나 컴퓨터가 사라진 경우입니다 |

## 중지, 재개, checkpoint

run 의 **details** 에서 학습을 관리합니다.

- **Stop**: **두 번 누르면** 학습을 멈춥니다. dataset 을 보내거나 환경을 설치하는 중인 시작 작업과 별개로 바로 실행됩니다.
- **Resume**: **마지막 checkpoint 부터** 같은 컴퓨터에서 이어서 학습합니다. 컴퓨터가 꺼져 quiet 이 된 run 에 씁니다.
- **delete**: checkpoint 하나를 학습한 컴퓨터와 이 컴퓨터 양쪽에서 지웁니다. 두 번 눌러야 지워지고, Policy 페이지에서 실행 중인 checkpoint 는 지우지 않습니다. 마지막 checkpoint 를 지우면 Resume 은 남은 것 중 가장 최근 checkpoint 부터 이어 갑니다.

이 컴퓨터로 복사된 checkpoint 는 studio 의 Policy 페이지에 나타나고, 거기서 팔에 [실행](../run/)합니다.

## 명령으로 학습

Train 페이지 없이 터미널에서 학습할 때는 `lerobot-train` 대신 `pinkrab-train` 을 **같은 인자로 실행합니다**. `pinkrab-train` 은 `lerobot-train` 과 같은 학습을 하면서 지표를 `metrics.jsonl` 로 남기기 때문에, 이렇게 시작한 run 도 Train 페이지에 그려집니다.
