<p align="center">
  <img src="docs/src/assets/photos/arms.jpg" alt="Two pink Pinkrab arms mounted on a pegboard above a desk" width="720">
</p>

<h1 align="center">Pinkrab</h1>

<p align="center">
  <b>A wall-mounted, bimanual 6-DoF robot and Hermit, the handheld gripper that teaches it.</b><br>
  Collect demonstrations by hand, train with LeRobot, run the policy on the arms.
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-e8318a"></a>
  <img alt="ROS 2 Jazzy" src="https://img.shields.io/badge/ROS%202-Jazzy-22314e">
  <img alt="Python 3.12+" src="https://img.shields.io/badge/python-3.12%2B-3776ab">
  <a href="https://github.com/huggingface/lerobot"><img alt="LeRobot 0.6.1" src="https://img.shields.io/badge/LeRobot-0.6.1-ffcc4d"></a>
  <img alt="MuJoCo" src="https://img.shields.io/badge/sim-MuJoCo-4b6cb7">
</p>

<p align="center">
  <a href="https://woolimi.github.io/pinkrab/"><b>Documentation</b></a> ·
  <a href="https://woolimi.github.io/pinkrab/hardware/bom/">Bill of materials</a> ·
  <a href="https://woolimi.github.io/pinkrab/hardware/print/">Print files</a> ·
  <a href="https://woolimi.github.io/pinkrab/start/install/">Install</a>
</p>

---

Pinkrab (Pink + Crab) is a pair of 6-DoF arms bolted to an IKEA SKÅDIS pegboard
on the wall behind a desk. Each arm is an [SO-ARM 101] with an extra printed
wrist yaw joint and a new L-shaped base. A crab keeps its body still and works
with its claws, and so do these arms.

**Hermit** is the claw of the arm with a handle on it, a [UMI]-style handheld
gripper. It has the same claw, jaw and wrist camera as the arm, so a policy
trained on video recorded with Hermit sees the same picture when it runs on the
robot. Demonstrations are collected without the robot in the loop.

> [!NOTE]
> The documentation site is written in Korean.

[SO-ARM 101]: https://github.com/TheRobotStudio/SO-ARM100
[UMI]: https://umi-gripper.github.io

## Highlights

- **Six joints per arm.** The added wrist yaw gives every arm a full 6-DoF tool
  pose, so all motion is solved with inverse kinematics from the URDF. No leader arm.
- **Hermit for data and teleoperation.** A head-worn top camera tracks the UMI
  marker on each Hermit against a table marker, giving each tool's pose in the
  desk frame every frame. The same tracking drives the arms live.
- **Pinkrab Studio.** One browser workbench over a MuJoCo model of the rig:
  teleoperation, Collect, Curate, Train and Policy pages, driven by a
  three-pedal foot switch.
- **ROS 2 safety gate.** Every command to a servo passes one gate that checks
  joint bands, contact with the desk, board and the other arm, and speed,
  acceleration and jerk limits.
- **LeRobot native.** The pair is registered as the robot type
  `bi_pinkrab_follower`, and policies train and run as LeRobot's ACT.
- **Corrections while running.** Take over a running policy with Hermit, fix
  the motion and add it to the dataset (HG-DAgger).

## Workflow

```
 Collect  ───►  Curate  ─────►  Train  ───►  Run
 Hermit +       verdicts,       ACT          Pinkrab,
 top camera     SAM 3 random                 safety gate
                backgrounds                       │
    ▲                                             │
    └────────────── corrections with Hermit ◄─────┘
```

| Step | Docs |
|---|---|
| Buy, print and assemble | [BOM](https://woolimi.github.io/pinkrab/hardware/bom/) · [Print](https://woolimi.github.io/pinkrab/hardware/print/) · [Arm](https://woolimi.github.io/pinkrab/hardware/assemble-arm/) · [Hermit](https://woolimi.github.io/pinkrab/hardware/assemble-hermit/) |
| Set up and calibrate | [Setup](https://woolimi.github.io/pinkrab/use/setup/) · [First run](https://woolimi.github.io/pinkrab/use/first-run/) |
| Collect demonstrations | [Collect](https://woolimi.github.io/pinkrab/use/collect/) |
| Build a dataset | [Curate](https://woolimi.github.io/pinkrab/use/curate/) |
| Train a policy | [Train](https://woolimi.github.io/pinkrab/use/train/) · [Policies](https://woolimi.github.io/pinkrab/deep/policies/) |
| Run and correct | [Run](https://woolimi.github.io/pinkrab/use/run/) · [Corrections](https://woolimi.github.io/pinkrab/deep/corrections/) |

## Hardware

<table>
  <tr>
    <td width="50%"><img src="docs/src/assets/photos/arm-claw.jpg" alt="A Pinkrab arm with its claw, wrist camera and L-shaped base on the pegboard"></td>
    <td width="50%"><img src="docs/src/assets/photos/hermit-held.jpg" alt="Hermit held in one hand, with the UMI marker on the handle and the camera above the claw"></td>
  </tr>
  <tr>
    <td align="center">Pinkrab arm</td>
    <td align="center">Hermit</td>
  </tr>
</table>

| | |
|---|---|
| Joints | 6 per arm + gripper |
| Servos | Feetech STS3215 (12 V), 7 per arm |
| Reach | about 570 mm from the board to the tool tip |
| Payload | about 0.45 kg, arm fully stretched (calculated) |
| Pose accuracy | joint error ≤ 0.8°, tool tip 1 mm on average (measured) |
| Cameras | ELP AR0234 global-shutter fisheye on each wrist and each Hermit, one AR0234 top camera |
| Power and bus | one 12 V 6 A adapter for both arms, one Seeed XIAO bus servo board per arm over USB |
| Printer | every part is printed; Bambu Studio projects for the Bambu Lab P2S are included |

The full list, with screws and quantities, is the
[bill of materials](https://woolimi.github.io/pinkrab/hardware/bom/).

## Quick start

macOS uses a RoboStack conda environment, Ubuntu 24.04 uses apt ROS 2 Jazzy
with a venv. The [install page](https://woolimi.github.io/pinkrab/start/install/)
has the full steps for both.

```bash
git clone https://github.com/woolimi/pinkrab.git
cd pinkrab

# macOS
mamba env create -f requirements/environment.yml
mamba activate pinkrab
pip install -e src/pinkrab_core -e src/pinkrab_cli -e src/pinkrab_mujoco \
  -e src/pinkrab_studio -e src/pinkrab_umi -e src/pinkrab_policy \
  -e lerobot-plugin

# Ubuntu 24.04
./scripts/setup.sh && source /opt/ros/jazzy/setup.bash && source .venv/bin/activate
rosdep install --from-paths src --ignore-src -r -y

colcon build --symlink-install && source install/setup.bash
pinkrab doctor          # checks the environment, boards and description without moving anything
```

On macOS, `colcon build` needs the Python and NumPy paths of the environment;
see the install page.

With the arms built and plugged in:

```bash
pinkrab ports detect                         # tell the two identical boards apart, once
pinkrab check                                # every servo answers
pinkrab calibrate left                       # one arm at a time
ros2 launch pinkrab_bringup robot.launch.py  # servo driver and safety gate
pinkrab-studio                               # http://localhost:8730/
```

Calibration, port assignment and datasets live in `~/pinkrab`, outside the
checkout. They describe your rig, so they are never committed.

### Simulation

`pinkrab-sim` runs the whole loop in MuJoCo with no hardware: a simulated
Hermit records takes, the same converter builds the dataset, and the simulated
arms run the trained policy. Rendering is headless (EGL).

```bash
pinkrab-sim demos --scene bowls --count 200 --batch b
pinkrab-sim convert b --out sim-b
pinkrab-sim evaluate <checkpoint> --scene bowls
```

See [src/pinkrab_sim/README.md](src/pinkrab_sim/README.md).

## Repository layout

| Path | Contents |
|---|---|
| `src/pinkrab_core` | Rig config, ports, servo bus, angles, URDF, kinematics, LeRobot robot classes |
| `src/pinkrab_cli` | The `pinkrab` CLI: ports, servo check, calibration, cameras, lens fit |
| `src/pinkrab_description` | URDF/xacro, meshes, MJCF scene, printable parts and 3MF projects |
| `src/pinkrab_umi` | Hermit: recording, marker tracking, conversion to LeRobot datasets, print files |
| `src/pinkrab_studio` | The browser workbench, foot switch and remote training |
| `src/pinkrab_policy` | Rollout, training launcher, run log |
| `src/pinkrab_sim` | The UMI loop in simulation |
| `src/pinkrab_mujoco` | The rig in MuJoCo, stepped headless, asked about contact |
| `src/pinkrab_driver` · `pinkrab_safety` · `pinkrab_teleop` · `pinkrab_bridge` | ROS 2 nodes: servo driver, safety gate, Hermit teleoperation, studio bridge |
| `src/pinkrab_bringup` · `pinkrab_interfaces` | Launch files, messages and services |
| `lerobot-plugin` | `bi_pinkrab_follower` as a LeRobot robot type |
| `docs` | The documentation site (Starlight) |

## Acknowledgements

Pinkrab builds on the ideas, designs and code of these projects:

- [Qubit](https://github.com/0xaiwhisperer/qubit) by The A.I. Whisperer, the idea of an SO-ARM 101 mounted on the wall above a desk
- [SO-ARM 100 / 101](https://github.com/TheRobotStudio/SO-ARM100) by TheRobotStudio, the base arm, URDF and most parts (Apache-2.0)
- [SO-ARM100 additional wrist joint](https://makerworld.com/en/models/2845098) by ahb, the wrist yaw bracket (CC BY-SA)
- [UMI, the Universal Manipulation Interface](https://umi-gripper.github.io) by Chi et al., collecting demonstrations with a handheld copy of the robot's gripper
- [LeRobot](https://github.com/huggingface/lerobot) by Hugging Face, the dataset format, training code and policies

## License

Apache-2.0, with one exception: `wrist_yaw_bracket_left.stl` and
`wrist_yaw_bracket_right.stl` are derived from ahb's
[SO-ARM100 additional wrist joint](https://makerworld.com/en/models/2845098) and
are **CC BY-SA**, so anything derived from them carries the same license.
[SAM 3](https://github.com/facebookresearch/sam3), used for background
randomisation, is installed separately under Meta's SAM License and is not
part of this repository. See [NOTICE](NOTICE).
