---
title: Repository layout
description: What each directory of the Pinkrab repository holds. Python packages, ROS 2 nodes, the LeRobot plugin, the documentation site, and the work folder outside the repository.
---

The Pinkrab repository is a ROS 2 workspace, and the packages under `src/` are built together with `colcon build`. How to install is described in [Install](../../start/install/).

## Directories

| Path | Contents |
|---|---|
| `src/pinkrab_core` | Rig config, ports, servo bus, angles, URDF, kinematics, LeRobot robot classes |
| `src/pinkrab_cli` | The `pinkrab` CLI: ports, servo check, calibration, cameras, lens fit |
| `src/pinkrab_description` | URDF/xacro, meshes, MJCF scene, printable parts and 3MF projects |
| `src/pinkrab_umi` | Hermit: recording, marker tracking, conversion to LeRobot datasets, print files |
| `src/pinkrab_studio` | The browser workbench, foot switch and remote training |
| `src/pinkrab_policy` | Rollout, training launcher, run log |
| `src/pinkrab_sim` | The UMI loop in MuJoCo ([Simulation](../../use/simulation/)) |
| `src/pinkrab_mujoco` | The rig in MuJoCo, stepped headless, asked about contact |
| `src/pinkrab_driver` · `pinkrab_safety` · `pinkrab_teleop` · `pinkrab_bridge` | ROS 2 nodes: servo driver, safety gate, Hermit teleoperation, studio bridge |
| `src/pinkrab_bringup` · `pinkrab_interfaces` | Launch files, messages and services |
| `lerobot-plugin` | `bi_pinkrab_follower` as a LeRobot robot type |
| `docs` | This documentation site (Starlight) |

## Work folder outside the repository

Calibration, port assignment, camera lens fits, recorded episodes, datasets and training results all live in `~/pinkrab/`, outside the checkout. They describe your own rig, so they are never committed.
