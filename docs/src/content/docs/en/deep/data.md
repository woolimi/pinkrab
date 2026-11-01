---
title: Demonstration data and training
description: The computation from a Hermit demonstration to a LeRobot dataset and from a trained policy to Pinkrab arm motion. Raw episodes, an overview of dataset conversion, training, and execution.
---

A Hermit demonstration has **no arm joint angles**. What is recorded is the video of the two gripper cameras, the poses of the two Hermits measured by the top camera, and the opening of the two jaws. The policy looks at the gripper camera video and learns how the tool should move next in the tool's own frame, and at execution that motion is turned into joint angles by the arm's IK.

## Raw episodes

One recording is saved as is in `~/pinkrab/episodes/<name>/`. A LeRobot dataset can be rebuilt from it later, but the video and the per-frame numbers **cannot be recovered later**.

| File | Contents |
|---|---|
| `gripper_left.mp4`, `gripper_right.mp4` | Video of the two gripper cameras. The top camera video is not saved |
| `frames.jsonl` | For each camera frame, which camera and which frame number it is, and its time. The top camera also leaves frame times without video |
| `poses.jsonl` | For each top camera frame, the pose of the marker solid found (in the camera frame and the table marker frame), the opening of the two jaws at that moment, and the pixel coordinates of the four corners of every marker read in that frame (`markers`) |
| `robot.jsonl` | Only in rollout recordings. The arm's joint angles, the gate mode, who was moving the arm at that moment (`controller`), and which correction it was |
| `events.jsonl` | Start and end, and in a rollout recording, a correction that was redone |
| `meta.json` | Clock, top camera calibration, size and orientation of the table marker, episode kind, the four corners of every marker on each marker solid (solid frame, m) and the ArUco dictionary, the full lens fit of each gripper camera (model, coefficients, frame size, image circle), and for a rollout recording, the verdict (`outcome`) |

Every line has the time the data arrived at this computer (`host_ts`), and the three cameras and the jaws are aligned later with this clock.

`meta.json` contains the marker corners and the lens fit together, so with only one episode folder you can re-solve the marker poses from the marker corner coordinates in `poses.jsonl` and undistort the gripper video without this code. When the top camera lens is measured again, `pinkrab-umi resolve` uses these coordinates to re-solve the poses of past demonstrations with the new lens. The top camera video only films the room and the demonstrator and the policy does not see it, so it is not saved, and the corner coordinates kept instead also carry no error from video compression. To use a different pose estimation method, write its result in the same format as `poses.jsonl` and Convert accepts it as is.

## Dataset conversion

Convert turns raw episodes into a LeRobot dataset. It builds the tool trajectory from the marker poses, places frames at the dataset fps, and computes the relative action in the tool frame for each hand. The step-by-step computation, the dataset layout, fps, and why relative actions are used are in [Dataset conversion](../convert/).

## Training

[Training](../../use/train/) is done by LeRobot's `lerobot-train`. State and actions are normalized with dataset statistics, and during training the video gets random changes such as brightness and contrast (image augmentation). The policy is trained to output not one action at a time but several future action steps (a chunk). The structure is in the [Policy overview](../policies/) and [ACT](../policies/act/).

## Execution

1. **Start pose**: The two arms go through the Ready pose to the average of the demonstrations' first TCP poses. The average is computed by moving `pose.tcp_<hand>` into robot coordinates with the table marker convention, so the two arms start with the spacing and orientation that the two hands had in the demonstrations. **Starting from a different place** makes the policy see video it never saw in training.
2. **Observation**: The video of the arm's gripper cameras is cropped and scaled the same way as in the dataset. The state is the opening of the two arm jaws and the relative pose computed by the forward kinematics of the two arms.
3. **Inference**: The policy runs one step at a time at the dataset fps. When it outputs a chunk, it **executes only the first few steps** and then looks at the camera again (by default 0.1 second worth, 5 steps at the default 50 fps). The remaining steps are discarded and a new query is made so that it can react right away if an object moves differently than expected.
4. **Applying the action**: Each action's change is multiplied onto the tool pose last sent to that arm (with the left arm rotated by a half turn of the tool frame), and the joint angles are obtained with the same `track()` as in teleoperation. The jaw opening ratio is converted to the arm's jaw range.
5. **Stopping**: Execution **stops** if the policy outputs a non-numeric value, if IK falls far short of the requested position, or if a joint is at its limit while the policy keeps asking to go beyond it and the gap between the request and the actual position builds up. During execution the gate limits joint speed to **60° per second**.

## Correction data

A [rollout](../../use/run/) recording also becomes a dataset through the same Convert, but its video and actions are computed from the arm rather than from Hermit. The recording method and principle are in [Correction data](../corrections/), and the conversion is in [Dataset conversion](../convert/#conversion-of-rollout-recordings).
