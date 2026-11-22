---
title: Run
description: How to run a trained policy on Pinkrab. Pre-run check, the Policy page in studio, Ask and Drive, inference settings, correction with Hermit, run records and outcome marking.
---

Run a [trained](../train/) checkpoint on Pinkrab from the Policy page in studio. If the policy makes a mistake during a run, you can take over both arms with Hermit and correct it, and the corrected segments become data for the next training.

## Preparation

- Bring up the robot with `ros2 launch pinkrab_bringup robot.launch.py`, as in [First run](../first-run/).
- Place the objects at the workspace. They must be in **the same place and of the same kind** as in the demonstrations so that the policy sees the scene it was trained on.
- The checkpoint to run must be **copied to this computer** (**copy here** on the Train page).

## Pre-run check

Before driving a new checkpoint on the real robot, use `pinkrab-picture-swap` to check that the policy acts on the gripper camera images.

```bash
pinkrab-picture-swap ~/pinkrab/train/<run>/checkpoints/last/pretrained_model
```

It picks frames from the dataset used for training and, for each hand, measures how much the policy's answer changes when the images of both hands are replaced with those of another frame and when the state is replaced. If the direction of the answer stays almost the same after the images are swapped (cos close to +1), the policy is acting on the state rather than the images, and on the real robot it keeps drifting in the direction it first moved. If the answer with swapped images has cos of 0.55 or less for both hands, the check passes with "looks at its pictures". Otherwise it prints "does NOT look at its pictures" and exits with code 1. The threshold is `swap_max_picture_cos` in the `pinkrab_core` config, and the check requires the dataset used for training to be on this computer.

## Policy page

Press the Policy card on the studio home screen. The top bar has the connection status, the correction pedals, and **Connect**. Below it, the left side has the two steps and the list of run records, which stay in place as the page scrolls, and the right side shows the policy's answer and, large and one under the other, the wrist camera views of both arms. Buttons are enabled only when they can be pressed.

**Connect** opens the real robot, the wrist cameras of both arms, and the pedals. The top camera is turned on only for correction. If required setup is not finished, the page shows what is left.

1. **Checkpoint**: Choose the checkpoint to run. For a policy that takes a task command, a **command** field appears below, initially filled with the dataset's task description.
2. **Run**: Press **Ask** or **Drive**. Once connected, buttons that move both arms to the Ready and Home poses also appear here.

| Button | Action |
|---|---|
| Ask | Moves both arms through the Ready pose to the pose where the demonstrations started, then asks the policy and only draws its answer on the screen. The arms do not move further |
| Drive | Moves both arms as the policy answers, starting right from the pose they are in. The arms are not moved anywhere first |
| Stop | Stops in place. The servos hold the stopped position. While the arms are walking to the start pose or to Ready or Home, the walk also ends where they are |

While the arms walk to the start pose or to Ready or Home, only Stop is enabled and the other buttons are disabled. They are enabled again once the walk ends, and pressing Stop or Drive also ends the wait that Ask had on the walk, so the policy is not asked after the walk is over.

The start pose is the average of the first poses of the demonstrations, so the two arms start as far apart as the two hands were in the demonstrations, facing each other. Pressing Drive after stopping Ask starts the arms from this start pose, and pressing the Ready button first starts them from the Ready pose. For a checkpoint you run for the first time, it is safer to **look at the answer with Ask first** and then move on to Drive.

**The run stops** if the policy answers with a non-number (NaN), or if a joint keeps moving away from its target while at its limit. Commands sent to the arms still pass the safety gate's joint range, collision, and speed checks during a run.

## Stop when the connection is lost

While a policy drives the real arms (a correction included), studio watches its connection to the Policy page. studio pings the Policy page's WebSocket every 0.5 seconds, and if no page answers for more than 2 seconds, whether the browser was closed, the network dropped or the laptop went to sleep, it stops the run the way Stop does. The gate goes idle so the arms hold where they are, the correction in progress and the run record are saved, and the cameras and the bridge stay up. The browser answers a ping for the page, so a tab behind another tab does not stop the run. The time is `policy_page_lost_s` in the `pinkrab_core` config.

If the rollout node exits while driving, the run ends the same way, and a correction in progress is saved. The reason is left in the alert in the corner of the screen, and when the page is reopened or the connection returns, the status line shows stopped and the reason.

## Inference settings

The gear button next to Checkpoint opens the settings window. At the top is a field for choosing which camera belongs to which arm; below it are the inference settings for the chosen checkpoint's policy type, shown next to the values used in training. These are values you can change without retraining, for example how many steps of a chunk to run before looking at the camera again (`n_action_steps`) and whether to blend overlapping chunks at each step (`temporal_ensemble_coeff`). Values common to all policies (device, joint speed, criteria for stopping a run) are folded under "every policy".

**Save** writes them to `~/pinkrab/inference.yaml`, and runs use those values **from the next Ask and Drive**.

## Motion shaping

The safety gate does not send the policy's commands straight to the servos; it smooths them (shaping). A command is used only as the place a joint is heading to, and the gate computes and sends a setpoint approaching it for each joint 60 times per second. Joint speed, acceleration, and the change in acceleration (jerk) each stay under their limit, so even if a command suddenly bends where one chunk joins the next, the arm **changes direction over about 0.1 seconds**. The smoothed pose goes through the same joint range and collision checks.

| Setting (every policy) | Default | Meaning |
|---|---|---|
| `smoothing` | on | Whether to use shaping. When off, commands are sent as they arrive, keeping only `max_joint_dps` |
| `max_joint_dps` | 60 | Joint speed limit, °/s |
| `max_joint_accel_dps2` | 400 | Joint acceleration limit, °/s². 0.15 s to reach 60 °/s |
| `max_joint_jerk_dps3` | 8000 | Limit on the change in acceleration, °/s³. The lower it is, the rounder starts and stops become |
| `action_latency_s` | 0.085 | Time until a commanded motion becomes the arm's motion, seconds. Each chunk is run this far ahead |

Shaping makes the arm follow commands slightly late. For a command moving at 30 °/s, the arm follows about 2° behind, and for a command that stops it does not overshoot. The policy adds its next motion to the last position sent, not to where the arm actually is, so this lag does not accumulate. Until a commanded motion becomes the actual motion, this shaping lag (35 to 50 ms at 30 °/s, longer near the speed limit) adds to the time the servo takes to answer its setpoint (about 40 ms). So each chunk is run `action_latency_s` ahead, sending now the entry for the time the arm will actually move. The execution recording keeps the commands and the setpoints the gate sent beside the joint readings, so `pinkrab-umi latency` measures this lag for real and shows it. If no command arrives for more than 0.1 seconds, or a command itself is rejected (for example a non-number), the gate does not carry the arm to the last command and instead **decelerates it to a stop in place**. If joint readings are interrupted or the smoothed pose fails the collision check, it stops right at the last pose sent. The gate rereads the settings each time the policy takes over the arms, so saved values apply **from the next Drive**. Following Hermit, correction, and jog and Ready·Home moves in studio are smoothed the same way, but their limits are the teleop values in the `pinkrab_core` config (speed 300 °/s, acceleration 900 °/s², jerk 15000 °/s³), not inference.yaml, and the jaw keeps only the speed limit.

## Correction with Hermit

If the policy makes a mistake during a Drive run, take over both arms with the foot pedals and correct it. Each takeover starts one correction, and **each segment** in which Hermit moved the arms during a correction becomes **one episode of the dataset**. While the arms are held with pedal 1, segments are split.

| Pedal | While the policy is moving | While both arms are taken over |
|---|---|---|
| 2 | Takes over both arms | Gives them back to the policy |
| 1 | | Makes the arms follow Hermit, or holds them in place |
| 3 | Stops the run and saves the record | Discards this correction and holds the arms in place |

1. When the arm starts to go wrong, **press pedal 2 right away** to take over. Both arms wait where they stand, and the pose at the moment of takeover is remembered.
2. Hold **each Hermit in a pose similar to the arm's tool**, then press pedal 1. As in [First run](../first-run/#following-hermit), the arms follow as far as Hermit moves. Each jaw keeps the opening the policy last sent, and follows the finger once Hermit's jaw opening reaches that value.
3. After correcting, press **pedal 2** and the policy continues from that pose.

If the correction went wrong, press **pedal 3** while taken over. The arms stop following Hermit and stay in place, and that correction is marked as discarded in the record and **does not go into the dataset**. After you put an object that moved on the desk back by hand and press pedal 2, the arms slowly return to the pose they were taken over in and wait there held again, and from then you do step 2 again. Pressing pedal 3 after a correction is discarded or while the arms are returning stops the run. In no case does pressing pedal 3 start the arms moving.

It is better to **keep a correction short** and return the arms to a pose the policy can pick up from before handing back. A correction made after the arms have already strayed far covers a scene the policy would have difficulty meeting again.

## Run record

A run made with Drive is **recorded entirely**, from the start until it stops. The images from both wrist cameras, the joints of both arms, and who was moving the arms at each moment (policy, person, hold) are kept together. The top camera turns on at the moment of takeover and off when you let go, so the Hermit markers are recorded only during correction. If recording is not possible, **the run does not start**.

Every kept record, past runs included, appears newest first in the list under the pedals, each row naming the checkpoint that run used, and you can press it to replay the video. The Collect page list shows not the whole run but only its kept corrections, one row each, alongside the demonstrations. Pressing **delete** twice on such a row leaves only that correction out of the dataset, and the run's record stays on the Policy page. When you Convert, without redoing anything, **only the segments where Hermit moved the arms** go into the dataset from the kept corrections, one episode per segment, with the arm-side camera video and the motion computed from the arm's joints. How recording works and why is in [Correction data](../../deep/corrections/).

## Latency check

[The latency set from demonstrations](../collect/#camera-latency-measurement) is the offset between the two cameras; the two latencies a run uses can only be measured from run records. After a few first Drives, run `pinkrab-umi latency` again and it prints two more lines from the run records. The joint side is also measured from turning, so the runs must include turning the wrists.

| Line | Value to change |
|---|---|
| `against the joints: +N ms later than stamped` | How much later the wrist camera video is than the joint readings. Add N **equally** to `latency_s` in both the `gripper` and `top` blocks of `camera.yaml`, so the offset between the two cameras set from demonstrations stays as it is |
| `commands: a step is made N ms after it is asked` | Time until a commanded step becomes the arm's motion. Set `action_latency_s` in the [inference settings](#inference-settings) to this value. It is in seconds, so 85 ms is 0.085 |

If you changed `camera.yaml`, convert the datasets that hold corrections again.

## Outcome marking

Each record in the list has **succeeded** and **failed**, where the person who watched writes down whether the run accomplished the task. The video and joint records alone cannot tell whether an object went into place. Pressing the selected one again clears the mark, and the mark is kept in that record's `meta.json` as `outcome`.

A run taken over with Hermit was not accomplished by the policy alone, so it shows **taken over** instead of the mark buttons and counts as a failure. Above the list is how many runs of the currently chosen checkpoint succeeded; runs not yet marked are not counted. When you run several times under the same conditions to measure a success rate, mark each run as soon as it ends, and keep failed runs instead of deleting them.
