"""Offline check of the shelf's replay judging a take absolutely: two-handed
takes made up from tool poses in the robot's world, carried back through the
desk solid and a still top camera to the solid poses a recording would hold.
Hands resting on the ready pose's tools leave the arms there, the take is
played on the dataset's clock in its own time even where the top camera
skipped frames, a hand that moves within reach is followed clean - either
arm, the other resting - one carried out of reach is flagged short and fails
that hand, one that starts out of reach is followed again once within it, a take recorded
before the chest placing is unjudged and not failed, a take made on a desk
solid of another size is carried in from that solid, and the cached answer
goes stale when the solid's recorded yaw changes. No camera, no robot.

    python src/pinkrab_studio/test/replay_check.py
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from pinkrab_core.config import SETTINGS as CORE
from pinkrab_core.follow import TOOL_TURN
from pinkrab_core.kinematics import ArmKinematics
from pinkrab_core.origin import ready_joints
from pinkrab_studio import replay, verdict
from pinkrab_umi.config import SETTINGS as UMI
from pinkrab_umi.desk_solid import world_T_desk

FAILURES: list[str] = []
FPS = 30.0
DESK = UMI.for_solid(UMI.desk_solid)
PLACED = (-90.0, DESK.cut_mm)   # the standard yaw, and the solid's origin above the desk as printed


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}" + (f" - {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


def solid_for(arm: str, tool_world: np.ndarray, desk: np.ndarray) -> np.ndarray:
    """The solid pose against the solid that puts the hand's tool at `tool_world`."""
    hand_tool = tool_world.copy()
    hand_tool[:3, :3] = hand_tool[:3, :3] @ TOOL_TURN[arm].T
    return np.linalg.inv(desk) @ hand_tool @ np.linalg.inv(UMI.for_solid(f"handheld_{arm}").solid_T_tcp)


def placed_at(face_mm: float) -> tuple[float, float]:
    """The standard yaw, and the origin of a desk solid of `face_mm` faces
    above the desk, standing on its cut as printed."""
    return PLACED[0], UMI.at_face(UMI.desk_solid, face_mm).cut_mm


def write_take(root: Path, tools: dict[str, list[np.ndarray]], placement: str | None, sizes: bool = True,
               desk_face: float | None = DESK.face_size_mm, skipped: int = 0) -> None:
    """A take whose hands hold their tools at each of `tools[arm]` in turn,
    seen by a top camera standing still 0.8 m above the solid - and
    recording the hand solids' sizes, unless `sizes` is False, and the desk
    solid's, `desk_face`, unless that is None. A camera that sends one frame
    and skips `skipped` keeps every (skipped + 1)th of them."""
    root.mkdir(parents=True)
    placed = placed_at(desk_face or DESK.face_size_mm)
    desk = world_T_desk(placed, desk_face or DESK.face_size_mm)
    desk_T_cam = np.eye(4)
    desk_T_cam[:3, :3] = np.diag([1.0, -1.0, -1.0])   # looking down
    desk_T_cam[:3, 3] = [0.0, 0.1, 0.8]
    poses, frames = [], []
    for i in range(0, len(next(iter(tools.values()))), skipped + 1):
        t = 100.0 + i / FPS
        solids, jaws = {}, {}
        for arm, along in tools.items():
            desk_T_solid = solid_for(arm, along[i], desk)
            solids[f"handheld_{arm}"] = {"solved": True, "pose": (np.linalg.inv(desk_T_cam) @ desk_T_solid).tolist(),
                                         "world_pose": desk_T_solid.tolist()}
            jaws[arm] = 0.5 + 0.4 * math.sin(i / 10.0)
        poses.append({"host_ts": t, "stamp": t - 0.03, "solids": solids, "jaw": jaws})
        frames.append({"stream": "head", "index": len(frames), "host_ts": t})
    desk_meta = {"solid": UMI.desk_solid, "yaw_deg": placed[0], "origin_above_table_mm": placed[1]}
    if desk_face is not None:
        desk_meta["face_mm"] = desk_face
    if placement:
        desk_meta["placement"] = placement
    meta = {"name": root.name, "kind": "demo", "desk": desk_meta, "streams": {"head": {"fps": FPS}}}
    if sizes:
        meta["hands"] = {f"handheld_{h}": UMI.for_solid(f"handheld_{h}").face_size_mm for h in ("left", "right")}
        meta["solid_T_tcp"] = {f"handheld_{h}": UMI.for_solid(f"handheld_{h}").solid_T_tcp.tolist() for h in ("left", "right")}
    (root / "meta.json").write_text(json.dumps(meta))
    (root / "poses.jsonl").write_text("".join(json.dumps(p) + "\n" for p in poses))
    (root / "frames.jsonl").write_text("".join(json.dumps(f) + "\n" for f in frames))


def moved(T: np.ndarray, dx: float) -> np.ndarray:
    out = T.copy()
    out[0, 3] += dx
    return out


def pose_of(rows: list[list[float]]) -> np.ndarray:
    """A 4x4 from its top three rows, the turn made a rotation again after rounding."""
    out = np.eye(4)
    out[:3, :3] = Rotation.from_matrix(np.array(rows)[:, :3]).as_matrix()
    out[:3, 3] = np.array(rows)[:, 3]
    return out


def between(a: np.ndarray, b: np.ndarray, w: float) -> np.ndarray:
    """`w` of the way from pose `a` to pose `b`: the place straight, the turn the shortest way."""
    out = np.eye(4)
    out[:3, 3] = a[:3, 3] + w * (b[:3, 3] - a[:3, 3])
    out[:3, :3] = Slerp([0.0, 1.0], Rotation.from_matrix([a[:3, :3], b[:3, :3]]))(w).as_matrix()
    return out


def chain_at_ready(arm: str) -> tuple[ArmKinematics, np.ndarray, np.ndarray]:
    chain = ArmKinematics(arm)
    ready = np.array(ready_joints(arm, chain.joint_names), dtype=float)
    return chain, ready, chain.forward(ready)


def resting(n: int, **moving: list[np.ndarray]) -> dict[str, list[np.ndarray]]:
    """Both hands' tools over n frames: each at its arm's ready tool, but for
    the ones given."""
    return {arm: moving.get(arm) or [chain_at_ready(arm)[2]] * n for arm in ("left", "right")}


if __name__ == "__main__":
    # These takes wave their jaws to be read, not to let go of anything:
    # the retract after a release is retract_check.py's.
    object.__setattr__(UMI, "keep_min_retract_mm", 0.0)
    # ...and are made of poses, with no video to judge: recording_check.py's.
    verdict.recording = lambda root: []
    n = int(4 * FPS)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        write_take(tmp / "still", resting(n), "chest")
        answer = replay.solved(tmp / "still")
        check("resting hands are both followed", answer["hands"] == ["left", "right"] and not answer["unjudged"],
              f"hands {answer['hands']}, unjudged {answer['unjudged']!r}")
        for arm in ("right", "left"):
            chain, ready, tool = chain_at_ready(arm)
            last = np.array([math.degrees(answer["trajectory"][-1]["joints"][j]) for j in chain.joint_names])
            off = float(np.max(np.abs(last - ready)))
            check(f"{arm}: a hand on the ready pose's tool leaves the arm at ready", off < 1.0,
                  f"{off:.2f} deg at most")
            check(f"{arm}: the start is reached", answer["start"][arm]["reached"], str(answer["start"][arm]))
        times = np.array([f["t"] for f in answer["trajectory"]])
        seconds = (n - 1) / FPS
        check("the take is played on the dataset's clock", answer["fps"] == CORE.dataset_fps
              and np.allclose(np.diff(times), 1 / CORE.dataset_fps)
              and len(times) == int(seconds * CORE.dataset_fps + 1e-3) + 1,
              f"{len(times)} moments at {answer['fps']} fps over {times[-1]:.2f} s")

        # A top camera that sent a third of its frames: the take still lasts as long.
        write_take(tmp / "skipping", resting(n), "chest", skipped=2)
        answer = replay.solved(tmp / "skipping")
        last = answer["trajectory"][-1]
        check("a take whose camera skipped frames is played in its own time",
              abs(last["t"] - (n - 1) // 3 * 3 / FPS) < 1e-6 and last["head"] == (n - 1) // 3,
              f"ends at {last['t']:.2f} s on head frame {last['head']}")

        # The same take not saying how big its hand solids were: no tool centre to place.
        write_take(tmp / "unsized", resting(n), "chest", sizes=False)
        answer = replay.solved(tmp / "unsized")
        check("a take without its solids' sizes places no hand",
              answer["hands"] == [] and set(answer["unplaced"]) == {"left", "right"},
              f"hands {answer['hands']}, unplaced {answer['unplaced']}")

        for arm in ("right", "left"):
            chain, ready, tool = chain_at_ready(arm)
            # Out 60 mm and back, over the take: inside the room the ready pose was picked for.
            sweep = [moved(tool, 0.06 * math.sin(math.pi * i / (n - 1))) for i in range(n)]
            write_take(tmp / f"sweep-{arm}", resting(n, **{arm: sweep}), "chest")
            answer = replay.solved(tmp / f"sweep-{arm}")
            clean = answer["summary"]["clean"]
            check(f"{arm}: a hand moving within reach, the other resting, is followed clean",
                  clean == answer["frames"],
                  f"{clean} of {answer['frames']} clean, flags {answer['summary']['flags']}")
            judged = verdict.judge(tmp / f"sweep-{arm}", answer)
            check(f"{arm}: it passes", not judged["fails"] and judged["feasible"] == 1.0,
                  f"fails {judged['fails']}, feasible {judged['feasible']}")

        chain, ready, tool = chain_at_ready("right")
        # Resting hands over a desk solid of another size than today's: its
        # side was against the chest, so its centre stood elsewhere, and the
        # hands are carried in from there - from today's solid they would land
        # the difference in their half-widths away.
        other = DESK.face_size_mm + 6.0
        write_take(tmp / "other-desk", resting(n), "chest", desk_face=other)
        answer = replay.solved(tmp / "other-desk")
        apart = np.linalg.norm(world_T_desk(placed_at(other), other)[:3, 3] - world_T_desk(placed_at(other))[:3, 3])
        check("a take on another size of desk solid is carried in from that solid",
              np.allclose(answer["desk"], world_T_desk(placed_at(other), other), atol=1e-6),
              f"today's solid would put it {apart * 1000:.1f} mm away")
        last = np.array([math.degrees(answer["trajectory"][-1]["joints"][j]) for j in chain.joint_names])
        off = float(np.max(np.abs(last - ready)))
        check("and its hand on the ready pose's tool leaves the arm at ready", off < 0.1, f"{off:.3f} deg at most")
        write_take(tmp / "unsized-desk", resting(n), "chest", desk_face=None)
        answer = replay.solved(tmp / "unsized-desk")
        check("a take that does not record its desk solid's size is unjudged",
              answer["unjudged"] == "no desk solid size recorded" and not answer["hands"],
              f"unjudged {answer['unjudged']!r}, hands {answer['hands']}")

        far = [moved(tool, 0.45 * i / (n - 1)) for i in range(n)]
        write_take(tmp / "far", resting(n, right=far), "chest")
        answer = replay.solved(tmp / "far")
        short = answer["summary"]["flags"]["right"].get("short", 0)
        check("a hand carried out of reach is flagged short", short > answer["frames"] // 4,
              f"{short} of {answer['frames']} short")
        judged = verdict.judge(tmp / "far", answer)
        check("it fails on that hand's arm", any(f.startswith("right: arm follows") for f in judged["fails"])
              and not any(f.startswith("left:") for f in judged["fails"]), str(judged["fails"]))

        # A hand held near the board, low and flat, out of the arm's reach,
        # for a second, then carried over a second to a pose within reach and
        # held there (both tool poses in the robot's world, off a recorded
        # take). The arm folded against its bands at the start is followed
        # again once the hand is within reach.
        rest = pose_of([[-0.55, 0.517, 0.656, 0.187], [0.811, 0.517, 0.273, -0.139], [-0.198, 0.682, -0.704, 0.111]])
        reach = pose_of([[-0.725, 0.596, 0.344, 0.324], [0.688, 0.611, 0.392, -0.069], [0.024, 0.522, -0.853, 0.038]])
        second = int(FPS)
        carry = [rest] * second + [between(rest, reach, k / second) for k in range(second)]
        carry += [reach] * (n - len(carry))
        write_take(tmp / "carried", resting(n, right=carry), "chest")
        answer = replay.solved(tmp / "carried")
        tail = answer["trajectory"][-len(answer["trajectory"]) // 3:]
        check("a hand that starts out of reach is followed again once within it",
              not answer["start"]["right"]["reached"] and all(not f["flags"]["right"] for f in tail),
              f"start {answer['start']['right']}, last third flags "
              f"{sorted({k for f in tail for k in f['flags']['right']})}, short {tail[-1]['short']['right']}")

        write_take(tmp / "before", resting(n, right=far), None)
        answer = replay.solved(tmp / "before")
        check("a take from before the chest placing is unjudged", answer["unjudged"] and not answer["hands"],
              f"unjudged {answer['unjudged']!r}, hands {answer['hands']}")
        judged = verdict.judge(tmp / "before", answer)
        check("it warns and does not fail", not judged["fails"] and judged["feasible"] is None
              and any("arm not judged" in w for w in judged["warns"]),
              f"fails {judged['fails']}, warns {judged['warns']}")
        write_take(tmp / "before-unsized", resting(n, right=far), None, sizes=False)
        judged = verdict.judge(tmp / "before-unsized", replay.solved(tmp / "before-unsized"))
        check("so is one that does not record its solids' sizes", not judged["fails"]
              and any("arm not judged" in w for w in judged["warns"]),
              f"fails {judged['fails']}, warns {judged['warns']}")

        model = replay.model_mtime()
        check("an answer is kept", replay.cached(tmp / "sweep-right", model) is not None)
        meta = json.loads((tmp / "unsized" / "meta.json").read_text())
        meta["hands"] = {f"handheld_{h}": UMI.for_solid(f"handheld_{h}").face_size_mm for h in ("left", "right")}
        meta["solid_T_tcp"] = {f"handheld_{h}": UMI.for_solid(f"handheld_{h}").solid_T_tcp.tolist()
                               for h in ("left", "right")}
        (tmp / "unsized" / "meta.json").write_text(json.dumps(meta))
        check("a take given its solids' sizes afterwards is solved again",
              replay.cached(tmp / "unsized", model) is None
              and replay.solved(tmp / "unsized")["hands"] == ["left", "right"])
        meta = json.loads((tmp / "sweep-right" / "meta.json").read_text())
        meta["desk"]["yaw_deg"] = -80.0
        (tmp / "sweep-right" / "meta.json").write_text(json.dumps(meta))
        check("a solid recorded at another yaw is solved again", replay.cached(tmp / "sweep-right", model) is None)

    print("\n" + ("all checks passed" if not FAILURES else f"{len(FAILURES)} failed: {', '.join(FAILURES)}"))
    sys.exit(1 if FAILURES else 0)
