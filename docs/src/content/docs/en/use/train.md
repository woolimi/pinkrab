---
title: Training
description: How to train a policy on a collected dataset. Training computer, the Train page in studio, training settings, progress status, and checkpoints.
---

Train a policy in [LeRobot](https://github.com/huggingface/lerobot) on the LeRobot dataset made on the [Curate](../curate/) page. Training is started and watched from the Train page in studio, and runs on the computer where studio runs or on a GPU computer connected over ssh.

## Policy

The policy is **[ACT](../../deep/policies/act/)** (Action Chunking with Transformers). It is a policy included in LeRobot; it takes the images of the two gripper cameras, the openings of the two jaws, and the relative position of the two tools, and outputs several upcoming steps of motion as one bundle (chunk). A transformer takes the images read by ResNet-18 and produces the chunk, and a CVAE is used during training so that when several motions fit the same scene, it picks one rather than their average. Training with the default **batch 8** uses up to about 5.7 GB of GPU memory. Each entry of a chunk is a future tool pose seen from the tool pose where that chunk began, and `pinkrab-train`, which runs the training, teaches the dataset's steps converted this way ([chunk representation](../../deep/convert/#chunk-representation)).

## Training computer

| Item | Requirement |
|---|---|
| System | x86_64 Linux, `python3` and `curl`, internet for the first setup |
| GPU | NVIDIA, compute capability 7.5 or higher (T4, RTX 20–50 series, A10, A100, L4, L40S, H100, etc.) |
| Driver | 580 or higher (CUDA 13.0). The CUDA toolkit does not need to be installed separately |
| GPU memory | 8 GB or more |
| CPU | 8 cores or more. Decoding video for every batch is the CPU's job |
| Disk | About 15 GB plus room for the dataset and checkpoints |

With the default batch 8, training fits on **a single 8 GB GPU**. The Python and packages needed on the training computer are installed on it by studio when training is first started.

## Train page

Press the Train card on the studio home screen to open http://localhost:8730/train. On the left is the list of training runs, newest first, and on the right is one chart for each metric such as loss. Press a card to draw that one run, or check several to overlay runs on one chart.

Training **continues even if you close studio**, and when you reopen studio it reads the values recorded in the meantime and continues drawing.

## Remote GPU computer

If the computer running studio has no GPU, you can train on another GPU computer reachable over ssh. Whether it is another computer at home or a server rented in the cloud, it only has to meet the conditions in [Training computer](#training-computer). Add that computer to **Sources** on the Train page.

| Field | Content |
|---|---|
| name | Any name, e.g. `gpu-box` |
| ssh line | A connection command such as `ssh -i ~/.ssh/id_ed25519 -p 22 user@gpu.example.com` |
| key | The private key file. It is copied to `~/pinkrab/keys/` as a file of the source's own and made readable by this user only, and it is deleted with the source. Password login is not used |
| runs are under | The folder on that computer for the dataset, the Python environment, and training results; default `~/pinkrab-train` |

After saving, press **test** to check that computer's GPU, memory, driver, CPU, and free disk, and it tells you what is missing for training, if anything.

On the first connection to a computer, ssh takes its host key and writes it to the source's own `known_hosts` file in `~/pinkrab/keys/`, and after that connects only when the key is the same. A cloud session can bring a new computer to the same address and port, and a changed host key makes the connection refused with an error line that says so. Saving that source again in Sources clears the kept key and takes the new one.

Training on a remote computer is handled the same way as training on this computer.

- **Sending**: When you start training, studio sends the dataset to that computer, installs the needed Python environment, and launches the training.
- **Metrics**: Studio reads that computer's training log every 30 seconds and accumulates it on this computer, so metrics such as loss are drawn on the Train page just like runs on this computer. If the connection drops and comes back, the records from the gap are fetched as well.
- **checkpoint**: Each checkpoint is copied to this computer when saved. Ones saved while studio was off are brought over when it comes back. A copied checkpoint is placed in a folder of its own for the source, `~/pinkrab/train/<run>@<source>/checkpoints/`, so runs of the same name from different sources never overwrite each other. It can be [run](../run/) right away on the Policy page.

## New run

In **New run**, choose the following and press Start.

| Field | Content |
|---|---|
| machine | The training computer |
| dataset | A dataset in `~/pinkrab/datasets/`. It is sent to a remote computer once, and a re-converted dataset is sent again |
| policy | ACT |
| batch | Default 8 if left empty. Raising it uses that much more GPU memory. With batch 16, ACT uses about 10.3 GB |
| steps | Default 100 000 |
| log every, checkpoint every | Default 50, 20 000. A shorter log interval only makes the chart denser; training speed stays the same |
| name | If left empty, `policy-dataset-date` |
| more arguments | Other arguments `lerobot-train` accepts, e.g. `--policy.optimizer_lr=5e-5` |

The yellow line at the top of the screen shows the steps in turn. A remote computer takes a few minutes the first time to build the Python environment, and after that starts in a few seconds. Training runs separately on that computer regardless of the connection, so it **continues even if you close the browser or the connection drops**.

## Progress status

| Status | Meaning |
|---|---|
| running | Records grew within the last 5 minutes |
| done | Trained to the last step |
| stopped | Stopped here or interrupted |
| failed | Ended with an error, and the error is shown under the card |
| quiet | No records for over 5 minutes and not finished. The process or the computer is gone |

## Stop, resume, checkpoint

Manage training from a run's **details**.

- **Stop**: **Press twice** to stop the training. It runs at once, apart from a start that is still sending the dataset or installing the environment.
- **Resume**: Continues training on the same computer **from the last checkpoint**. Use it on a run that became quiet because the computer shut down.
- **delete**: Deletes one checkpoint on the training computer and on this one. It takes two presses, and a checkpoint running on the Policy page is not deleted. After the newest checkpoint is deleted, Resume continues from the newest one left.

A checkpoint copied to this computer appears on the Policy page in studio, and from there you [run](../run/) it on the arms.

## Training from the command line

To train in a terminal without the Train page, **run** `pinkrab-train` instead of `lerobot-train` **with the same arguments**. `pinkrab-train` does the same training as `lerobot-train` and also writes metrics to `metrics.jsonl`, so a run started this way is also drawn on the Train page.
