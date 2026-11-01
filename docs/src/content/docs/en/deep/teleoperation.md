---
title: Hermit and teleoperation
description: Hermit's components, how the top camera measures Hermit's pose, the computation of relative teleoperation, and the role of the table marker.
---

Hermit is a handheld with the same claw, jaw, and camera as the Pinkrab arm's gripper, mounted on a handle. The camera sits **in the same place relative to the claw as on the arm**, so video shot while working with Hermit is the same as the video the arm would shoot in the same pose. That is why a policy can be trained on **demonstrations collected by human hands without a robot**, which is the approach of [UMI](https://umi-gripper.github.io/) (Universal Manipulation Interface). The same Hermit is also used for teleoperation, where you move the arm directly, and for corrections during execution.

## Hermit components

| Part | Role |
|---|---|
| claw, jaw | The same shape as the arm's gripper. The jaw is opened and closed with the index finger |
| Gripper camera | The same AR0234 global shutter fisheye camera (200°) as on the arm, which films the video the policy sees |
| Jaw servo | An STS3215 used as an angle sensor rather than a motor. During calibration the ticks at jaw closed and jaw open are recorded, and positions in between are read as an opening of 0–1 |
| UMI marker | A polyhedron (rhombicuboctahedron) with 18 square faces of 30 mm, 16 of which hold ArUco marker tiles. The left hand uses ids 0–15 and the right hand 16–31, so the marker determines which hand it is |

## Pose measured from markers

The top camera (global shutter AR0234, 1920×1200, 60 fps requested for about 50 fps actual, exposure fixed at 10 ms) finds the corners of the ArUco markers in every frame. Since it is known which face of the polyhedron each marker is on, PnP over all visible corners gives the pose of the polyhedron relative to the camera in one solve. **Solutions with a reprojection error above 3.5 px are discarded.**

A single planar marker tilted by +θ or −θ produces two poses whose images are almost identical (twofold ambiguity). For a 25 mm marker tilted 30° at a distance of 70 cm, the two poses differ by about 0.6 px, which is buried in corner detection noise, so the pose can flip from frame to frame. On a polyhedron, two or more faces angled 45° from each other are visible at once, so **the pose is uniquely determined**.

Multiplying the polyhedron's pose by the transform fixed by the handle design (`solid_T_tcp`) gives the pose of the Hermit tool center point (TCP). The top camera's exposure is **fixed at 10 ms**. This is because automatic exposure stretches the exposure to 20–33 ms in a dark room, which blurs a moving hand, and keeps changing the brightness, which makes the detection of edge faces flicker.

## Relative teleoperation

The arm follows **how far Hermit has moved**, not where Hermit is. Let `T_start` be the arm's tool pose at the moment following begins and `H_0` the Hermit tool pose at that moment; the arm target at every later frame is then:

```text
Δ        = H_0⁻¹ · H_t          How far Hermit has moved, in the tool frame of H_0
T_target = T_start · Δ          The same amount applied to the arm tool, in its tool frame
```

`Δ` is a change measured in Hermit's own frame, so when Hermit goes 5 cm forward, the arm's tool also goes 5 cm forward along its own front. If two poses are measured in the same frame, `Δ` is the same wherever that frame is, but if the frame moves while following, that motion appears as Hermit having moved. The top camera is worn on the demonstrator's head and moves with it, so the Hermit pose is measured not in the camera frame but in the **table marker frame** of the marker placed on the desk. That is why turning the head does not move the arm, and why the table marker can be placed anywhere on the desk, but **if it disappears from the view, that arm stops even if Hermit is visible**.

- **Re-establishing the reference**: if Hermit is **not visible for more than 0.5 seconds**, that arm stops, and the place where it reappears becomes the new `H_0` and the arm's pose at that time the new `T_start`. This is the behavior used to send the arm somewhere your hand cannot reach: hide Hermit, then bring it back into view at a comfortable position.
- **Left arm**: the description is a mirror image of the right arm, so the left arm's tool frame is rotated half a turn about the approach axis. For the left arm, `Δ` is applied rotated by that amount.
- **Jitter**: even when the hand is held still there is tracking noise of about 1 mm and 0.5°, and IK amplifies this to several degrees at the joints. The pose is smoothed with a 0.1 second time constant before it is followed.
- **jaw**: Hermit's jaw opening ratio is mapped directly onto the arm jaw's range of motion.

## IK and the safety gate

`T_target` becomes joint angles through one step of differential IK (`track()`). Because every frame continues from the previous joint angles, solving the same target twice never jumps to a different configuration. At many poses this wrist cannot match both position and orientation exactly, so **position takes priority**. The orientation weight is 0.1; in a motion that moves 80 mm from the ready pose while swinging 15°, orientation gives up to 5° and position stays within 14 mm.

The joint commands IK produces pass through the safety gate of [first run](../../use/first-run/#robot-bringup), which checks the joint range, collisions with the desk and board, and joint speed (300° per second or less), and **only then go to the servos**. The gate shapes these commands just like the policy's commands, so the pose solved 30 times per second by the top camera does not come out in steps, and even when a marker is lost and found again the arm stops and starts **ramping its acceleration down and up**. The limits are 900 °/s² acceleration and 15000 °/s³ jerk, and for the jaw only the speed limit applies so it follows the instant the hand grasps and releases.

## Differences in correction

Correction (DAgger), where you take over from the policy during [execution](../../use/run/#correction-with-hermit), uses the same relative method. There are two differences.

- Right after the takeover the arm does not follow Hermit and stands still. You must match the Hermit in your hand to a shape similar to how the arm holds the tool and then **start following with pedal 1**, so that the axes of `Δ` match the arm's axes.
- The jaw keeps the opening the policy last sent, and starts following only after Hermit's opening reaches that value. This is so that **the jaw does not open** at the moment you take over an arm that is holding an object.

## Role of the table marker

In teleoperation and correction, the table marker is only the reference for measuring Hermit's motion, so it can be placed anywhere. Where the table marker's position matters is when placing a demonstration in robot coordinates.

During a demonstration Pinkrab is not there, so there is no way to measure the Hermit's position in the demonstration directly in the robot frame. Instead, **the demonstrator's chest is taken as the position of the robot board**, and the table marker placed right in front of the demonstrator is computed as if it were in front of the board between the two arms. Multiplying the Hermit's pose relative to the table marker by this transform gives the pose in the robot base frame.

| Used for | What it does |
|---|---|
| Demonstration verdict | Follows the two hands' trajectories, moved into robot coordinates, with the IK of the two arms, and judges the demonstration by the fraction of frames where the arm falls short of the position, jumps, or touches the desk. Frames that follow while sitting at a joint limit are shown only as warnings |
| Convert | For every frame, locates the top camera from that frame's table marker and expresses the Hermit pose in the table marker frame. The actions used for training are relative changes between frames, so this frame cancels out, and the absolute pose is kept separately only as `pose.tcp_<hand>` |
| Starting pose for execution | Averages the first poses of the demonstrations, moves them into robot coordinates, and sends both arms there before running the policy |

The demonstration verdict and the starting pose both rely on this convention, so a demonstration with the table marker placed far from the demonstrator becomes a demonstration in which **the arm works that much farther from the board**.
