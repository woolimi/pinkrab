---
title: Simulation
description: pinkrab-sim runs demonstration generation, dataset conversion, training and evaluation in MuJoCo with no hardware. Scenes, the order of steps, demonstration generation and evaluation.
---

`pinkrab-sim` runs the whole loop from collecting demonstrations to running a policy in MuJoCo, with no hardware. A simulated Hermit records demonstrations, the same conversion code as on the real rig builds the dataset, and the simulated arms run the trained policy. All the code between a hand moving and an arm moving is the code the real rig runs; what the simulation replaces is the lens, the desk, the objects and the person.

Rendering is headless (EGL), so it runs on a GPU computer with no monitor. Demonstrations are saved to `~/pinkrab/sim/episodes/` and evaluation results to `~/pinkrab/sim/evals/`.

## Scenes

| Scene | Task |
|---|---|
| `bowls` | The right hand picks up a 40 mm block and puts it in a 100 mm bowl, while the left hand waits at the Ready pose with its jaw open. The same task the real rig is proven on |
| `blocks` | The left arm picks up a white cap and puts it on a green base. This is the scene when `--scene` is not given |

## Order of steps

```bash
pinkrab-sim demos --scene bowls --count 200 --batch b --workers 6   # 200 demonstrations
pinkrab-sim convert b --out sim-b                                    # a LeRobot dataset
```

`sim-b` has the same format as a real dataset, so it goes straight into [training](../train/). When training is done, check the checkpoint and run it on the simulated arms.

```bash
pinkrab-sim swap <checkpoint>                     # does the policy act on the images
pinkrab-sim evaluate <checkpoint> --scene bowls   # run on the simulated arms
```

`<checkpoint>` is a `pretrained_model` directory such as `~/pinkrab/train/<run>/checkpoints/last/pretrained_model`.

## Demonstration generation

`demos` plans motions the way a hand would make them and moves Hermit along them. It keeps only demonstrations where the arm can follow Hermit in every frame with the same IK used when running, touches neither the desk nor the other arm, and leaves the object in its target place. The recording format is the same as the real Recorder's, plus `truth.jsonl` with the true hand pose.

| Argument | Meaning |
|---|---|
| `--count` | Number of demonstrations to keep |
| `--batch` | Name prefix. Demonstrations are saved as `<batch>-000`, `<batch>-001`, … |
| `--scene` | `bowls` or `blocks` |
| `--workers` | Number of processes rendering demonstrations side by side |
| `--seed` | Seed for layouts and motions. Default 0 |

## Evaluation

`evaluate` lays out the objects as a demonstration could have been made on them, puts the arms in the same start pose as before running a policy on the real rig, and counts per episode whether the object was picked, lifted off the desk, and in its target place at the end. By default the object is attached to the claw once the jaw closes on it; with `--physics` the grasp is left to MuJoCo contact alone. `--videos` writes each episode's video beside the results.

`oracle` makes the arms follow planned demonstrations exactly, with no policy, and measures the highest success rate the simulation's judging allows. Only the gap below that ceiling is the policy's own failure.

`swap` is the same check as the real rig's [pre-run check](../run/#pre-run-check).

Scene construction, fisheye rendering and the check script are described in [src/pinkrab_sim/README.md](https://github.com/woolimi/pinkrab/blob/main/src/pinkrab_sim/README.md).
