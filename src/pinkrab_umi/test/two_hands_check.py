#!/usr/bin/env python3
"""Offline check of a two-handed take through convert and back to a start.

A take is made up from two tool paths against the desk solid - each hand
moving its own way, its jaw opening at its own pace - and written as the
recorder writes one: the solids in a still top camera's frame and against
the solid, the jaws, and a short gripper video per hand, the left red and the
right blue. Convert has to hand back each hand's pose and steps under its own
name, each hand's picture under its own key, and in the state the right tool
where it truly stands in the left's frame; and the start a policy run is
walked to has to be where the hands began, carried into the robot's world,
and the arms put there have to give the relative pose the dataset holds.
A top camera that moves through the take, as one worn on a head does, is
followed frame by frame. No camera, no robot.

    python src/pinkrab_umi/test/two_hands_check.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from pinkrab_core.config import SETTINGS as CORE
from pinkrab_core.dataset import image_key, pose_key, relative_pose
from pinkrab_core.demo_start import demo_start
from pinkrab_core.follow import tool_absolute_target
from pinkrab_umi import convert
from pinkrab_umi.config import SETTINGS as UMI

FAILURES: list[str] = []
FPS = 30.0
FRAMES = 90
VIDEO = (320, 200)
COLOUR = {"left": (0, 0, 255), "right": (255, 0, 0)}   # BGR: red, blue


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}" + (f" - {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


def tool_path(hand: str) -> list[np.ndarray]:
    """The hand's tool against the desk solid, frame by frame: straight
    lines, at a steady turn, each hand its own way."""
    sign = -1.0 if hand == "left" else 1.0
    out = []
    for i in range(FRAMES):
        s = i / FPS
        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler("xyz", [180.0, 20.0 * s, sign * 15.0 * s], degrees=True).as_matrix()
        T[:3, 3] = [sign * 0.12 + 0.03 * s, 0.25 + sign * 0.02 * s, 0.10 + 0.01 * s]
        out.append(T)
    return out


def camera_at(i: int, moving: bool) -> np.ndarray:
    """desk_T_cam at frame i: looking down from 0.8 m, still - or, `moving`,
    turning 20 degrees and travelling 100 mm across the take, as a head does."""
    desk_T_cam = np.eye(4)
    desk_T_cam[:3, :3] = np.diag([1.0, -1.0, -1.0])   # looking down
    desk_T_cam[:3, 3] = [0.0, 0.1, 0.8]
    if moving:
        turn = np.eye(4)
        turn[:3, :3] = Rotation.from_euler("xz", [10.0 * i / FRAMES, 20.0 * i / FRAMES], degrees=True).as_matrix()
        turn[:3, 3] = [0.1 * i / FRAMES, 0.0, 0.0]
        desk_T_cam = turn @ desk_T_cam
    return desk_T_cam


def write_take(root: Path, tools: dict[str, list[np.ndarray]], moving: bool = False,
               unseen: range = range(0)) -> None:
    """A take as the recorder writes one; the desk solid out of view in the
    frames `unseen`."""
    root.mkdir(parents=True)
    poses, frames = [], []
    for i in range(FRAMES):
        desk_T_cam = camera_at(i, moving)
        stamp = 100.0 + i / FPS
        solids, jaws = {}, {}
        for hand, along in tools.items():
            desk_T_solid = along[i] @ np.linalg.inv(UMI.for_solid(f"handheld_{hand}").solid_T_tcp)
            solids[f"handheld_{hand}"] = {"solved": True, "pose": (np.linalg.inv(desk_T_cam) @ desk_T_solid).tolist(),
                                          "world_pose": None if i in unseen else desk_T_solid.tolist()}
            jaws[hand] = (0.2 if hand == "left" else 0.8) + (0.005 if hand == "left" else -0.005) * i
        # Each reading lands on its exposure once its latency is taken off.
        poses.append({"host_ts": stamp + UMI.jaw_latency_s, "stamp": stamp, "solids": solids, "jaw": jaws})
        frames.append({"stream": "head", "index": i, "host_ts": stamp})
        for hand in tools:
            frames.append({"stream": f"gripper_{hand}", "index": i, "host_ts": stamp + UMI.gripper_camera_latency_s})
    for hand in tools:
        writer = cv2.VideoWriter(str(root / f"gripper_{hand}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, VIDEO)
        picture = np.zeros((VIDEO[1], VIDEO[0], 3), np.uint8)
        picture[:] = COLOUR[hand]
        for _ in range(FRAMES):
            writer.write(picture)
        writer.release()
    desk = {"solid": UMI.desk_solid, "placement": "chest", "yaw_deg": -90.0, "origin_above_table_mm": 15.0,
            "face_mm": UMI.for_solid(UMI.desk_solid).face_size_mm}
    streams = {"head": {"fps": FPS}}
    streams.update({f"gripper_{hand}": {"fps": FPS, "size": list(VIDEO), "centre": [0.5, 0.5]} for hand in tools})
    meta = {"name": root.name, "kind": "demo", "desk": desk, "streams": streams, "seconds": FRAMES / FPS,
            "hands": {f"handheld_{h}": UMI.for_solid(f"handheld_{h}").face_size_mm for h in tools},
            "solid_T_tcp": {f"handheld_{h}": UMI.for_solid(f"handheld_{h}").solid_T_tcp.tolist() for h in tools}}
    (root / "meta.json").write_text(json.dumps(meta))
    (root / "poses.jsonl").write_text("".join(json.dumps(p) + "\n" for p in poses))
    (root / "frames.jsonl").write_text("".join(json.dumps(f) + "\n" for f in frames))


def matrix(pose) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
    T[:3, 3] = pose[:3]
    return T


if __name__ == "__main__":
    import pyarrow.parquet as pq
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    tools = {hand: tool_path(hand) for hand in ("left", "right")}
    out = "check-two-hands"
    root = CORE.workspace / "datasets" / out
    shutil.rmtree(root, ignore_errors=True)
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp)
        # One take whose right gripper camera gave nothing, ahead of a whole one.
        write_take(source / "partial", tools)
        meta = json.loads((source / "partial" / "meta.json").read_text())
        del meta["streams"]["gripper_right"]
        (source / "partial" / "meta.json").write_text(json.dumps(meta))
        frames = [line for line in (source / "partial" / "frames.jsonl").read_text().splitlines()
                  if json.loads(line)["stream"] != "gripper_right"]
        (source / "partial" / "frames.jsonl").write_text("\n".join(frames) + "\n")
        write_take(source / "take", tools)
        try:
            # At the takes' own rate, frame for frame against what was written.
            made = convert.convert(["partial", "take"], out, "hand the block over", source=source,
                                   say=lambda *_: None, fps=int(FPS))
            check("a take missing a hand's camera is skipped, not the dataset",
                  "right gripper camera" in made["episodes"][0].get("skipped", ""), str(made["episodes"][0]))
            segment = made["episodes"][1]["segments"][0]
            account = json.loads((root / "meta" / "pinkrab_convert.json").read_text())
            check("the dataset is of both hands", account["hands"] == ["left", "right"]
                  and set(account["episodes"][1]["centre"]) == {"left", "right"}, str(account["hands"]))
            info = json.loads((root / "meta" / "info.json").read_text())["features"]
            names = info["observation.state"]["names"]
            check("each feature carries its hand's name", all(k in info for k in (
                image_key("left"), image_key("right"), pose_key("left"), pose_key("right")))
                and names[0] == "left_gripper" and names[1] == "right_gripper" and names[2] == "rx"
                and len(names) == 11 and info["action"]["names"][10] == "right_dx", str(sorted(info)))

            table = pq.read_table(next(root.glob("data/*/*.parquet")))
            state = np.array(table.column("observation.state").to_pylist())
            action = np.array(table.column("action").to_pylist())
            truth = {hand: along[segment["start"]:segment["end"]] for hand, along in tools.items()}
            for i, hand in enumerate(("left", "right")):
                got = [matrix(p) for p in table.column(pose_key(hand)).to_pylist()]
                err = max(np.linalg.norm(a[:3, 3] - b[:3, 3]) for a, b in zip(got, truth[hand])) * 1000
                check(f"{hand}: its pose is its own hand's", err < 0.5, f"worst {err:.3f} mm")
                T = truth[hand]
                dp = np.array([T[k][:3, :3].T @ (T[k + 1][:3, 3] - T[k][:3, 3]) for k in range(len(T) - 1)])
                step_err = np.abs(action[:-1, 10 * i:10 * i + 3] - dp).max() * 1000
                check(f"{hand}: its steps are its own hand's, in its own place in the action", step_err < 0.1,
                      f"worst {step_err:.4f} mm")
            relative = np.array([relative_pose(a, b) for a, b in zip(truth["left"], truth["right"])])
            rel_err = np.abs(state[:, 2:] - relative).max()
            check("the state's last nine are the right tool in the left's frame", rel_err < 1e-3,
                  f"worst {rel_err:.5f}")
            check("each hand's opening is its own", np.allclose(state[:, 0], 0.2 + 0.005 * np.arange(
                segment["start"], segment["end"]), atol=1e-3) and np.allclose(state[:, 1], 0.8 - 0.005 * np.arange(
                    segment["start"], segment["end"]), atol=1e-3))

            dataset = LeRobotDataset(f"pinkrab/{out}", root=root)
            frame = dataset[0]
            left = frame[image_key("left")].mean(dim=(1, 2)).numpy()
            right = frame[image_key("right")].mean(dim=(1, 2)).numpy()
            check("each hand's picture is under its own key", left[0] > 0.8 > left[2] and right[2] > 0.8 > right[0],
                  f"left rgb {np.round(left, 2)}, right rgb {np.round(right, 2)}")

            # The start: where each hand began, carried into the robot's world
            # by the solid it stood against.
            desk = np.eye(4)
            desk[:3, 3] = [0.3, 0.0, 0.0]
            start = demo_start(root, ("left", "right"), lambda _: desk)
            for hand in ("left", "right"):
                want = tool_absolute_target(hand, desk, truth[hand][0])
                off = np.linalg.norm(start["targets"][hand][:3, 3] - want[:3, 3]) * 1000
                turn = Rotation.from_matrix(start["targets"][hand][:3, :3].T @ want[:3, :3]).magnitude()
                check(f"{hand}: the start is where the hand began, in the robot's world",
                      off < 0.5 and np.degrees(turn) < 0.1, f"{off:.3f} mm, {np.degrees(turn):.3f} deg")
            # At rollout the relative pose is read off the arms: each arm's
            # tool, put where its hand's pose is carried, has to give the
            # numbers the dataset was made with.
            from pinkrab_core import inference
            from pinkrab_policy.rollout import _Arm

            arms = {hand: _Arm(hand, inference.common(), FPS) for hand in ("left", "right")}
            for hand, arm in arms.items():
                arm.T_cmd = tool_absolute_target(hand, desk, truth[hand][0])
            at_rollout = relative_pose(arms["left"].tool(), arms["right"].tool())
            check("rollout reads the arms' relative pose as convert wrote the hands'",
                  np.allclose(at_rollout, state[0, 2:], atol=1e-5), f"{np.round(at_rollout - state[0, 2:], 6)}")
            try:
                demo_start(root, ("left", "right"), lambda _: None)
                check("a dataset with no take placed in the world has no start", False)
            except RuntimeError as exc:
                check("a dataset with no take placed in the world has no start", True, str(exc))
        finally:
            shutil.rmtree(root, ignore_errors=True)

    # The top camera moving through the take, as a head turns: each frame
    # placed by its own desk solve, the hands' poses the same as from a still
    # camera.
    out = "check-two-hands-head"
    root = CORE.workspace / "datasets" / out
    shutil.rmtree(root, ignore_errors=True)
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp)
        write_take(source / "worn", tools, moving=True)
        write_take(source / "glanced", tools, moving=True, unseen=range(40, 46))
        write_take(source / "blinked", tools, moving=True, unseen=range(40, 42))
        try:
            made = convert.convert(["worn", "glanced", "blinked"], out, "hand the block over", source=source,
                                   say=lambda *_: None, fps=int(FPS))
            worn = made["episodes"][0]
            check("a camera that moves is followed", bool(worn["segments"]), str(worn))
            cut = [(s["start"], s["end"]) for s in made["episodes"][1]["segments"]]
            check("frames without the desk solid are a hole", len(cut) == 2 and cut[0][1] <= 40 <= 46 <= cut[1][0],
                  str(cut))
            cut = [(s["start"], s["end"]) for s in made["episodes"][2]["segments"]]
            check("two of them are bridged, not cut", len(cut) == 1 and cut[0][0] < 40 and cut[0][1] > 42, str(cut))
            segment = worn["segments"][0]
            table = pq.read_table(next(root.glob("data/*/*.parquet")))
            for hand in ("left", "right"):
                got = [matrix(p) for p in table.column(pose_key(hand)).to_pylist()]
                truth = tools[hand][segment["start"]:segment["end"]]
                err = max(np.linalg.norm(a[:3, 3] - b[:3, 3]) for a, b in zip(got, truth)) * 1000
                check(f"{hand}: its pose under a moving head is its own hand's", err < 0.5, f"worst {err:.3f} mm")
        finally:
            shutil.rmtree(root, ignore_errors=True)

    print("\n" + ("all checks passed" if not FAILURES else f"{len(FAILURES)} failed: {', '.join(FAILURES)}"))
    sys.exit(1 if FAILURES else 0)
