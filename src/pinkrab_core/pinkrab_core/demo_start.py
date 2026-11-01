"""Where the demonstrations began, for the arms to begin there too.

A policy trained on camera-local deltas is shown, at every step, what the
gripper cameras see - what the HANDHELDS' cameras saw while the takes were
made - and, with two hands, where one tool stands from the other. Shown a
start from somewhere else, the policy is shown pictures and an arrangement
of the hands it has never seen, and what comes back is noise. So before a
policy drives, each arm is taken to where its hand began: the mean of the
demonstrations' first tool poses, carried into the robot's world as the
hands were while the takes were made - the desk solid at the operator's
chest stands for the robot's board (pinkrab_umi.desk_solid.world_T_desk), so
a pose against it is a pose in the robot's frame.
Taken whole - place, heading and all - so the two arms begin as far apart,
and turned to each other, as the hands did.

The dataset carries each frame's absolute tool pose (`pose.tcp_<hand>`, the
desk solid's frame), so the start is read off the first frame of every
episode - every demonstration's: a correction (DAgger) begins wherever a
policy went wrong, not where a take begins, and its pose is the arm's, not
the desk's. How the solid stood for each take - which way it faced, how far
above the desk its origin was, its size - is in the dataset's account of its
sources, and `carry` turns that into where the solid stood in the robot's
world (or None for a take that has no place there).
"""
from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .config import SETTINGS as CORE
from .dataset import POSE_NAMES, pose_key
from .follow import tool_absolute_target
from .origin import ready_joints

# Steps of the differential IK from the ready pose to the start: far more
# than a tick's worth, so the solution has settled.
IK_STEPS = 300
REACH_MM, REACH_DEG = 5.0, 5.0


class DemoStartError(RuntimeError):
    pass


def mean_rotation(Rs) -> np.ndarray:
    """The mean of rotations: the orthogonal matrix nearest their sum (SVD).
    The one a policy run's start is taken from, and so the one a take's start
    is set beside."""
    u, _, vt = np.linalg.svd(np.sum(Rs, axis=0))
    R = u @ vt
    if np.linalg.det(R) < 0:
        u[:, -1] *= -1
        R = u @ vt
    return R


def _turn_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _heading(R: np.ndarray) -> float:
    """Which way the tool's approach (its z) points about the vertical."""
    return math.atan2(R[1, 2], R[0, 2])


def level(R: np.ndarray) -> np.ndarray:
    """A tool attitude with its heading taken out: the approach turned to
    look along +x, how far down it looks and its roll kept - what gravity
    fixes, and so what a take's start says whichever way the desk solid faced."""
    return _turn_z(-_heading(R)) @ R


class DatasetChanged(DemoStartError):
    """The dataset of the checkpoint's name is not the one it learned from.
    Not a missing start: driving from ready instead is what this module is
    here to prevent, so whoever drives must stop. `root` is the dataset as it
    is now."""

    def __init__(self, message: str, root: Path) -> None:
        super().__init__(message)
        self.root = root


def _matrix(pose) -> np.ndarray:
    """A pose.tcp row - xyz and a quaternion x y z w - as a 4x4."""
    x, y, z, qx, qy, qz, qw = pose
    T = np.eye(4)
    T[:3, :3] = [
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ]
    T[:3, 3] = [x, y, z]
    return T


def demo_start(dataset_root: Path, hands, carry: Callable[[dict], np.ndarray | None]) -> dict:
    """The mean first tool pose of each hand over the dataset's
    demonstrations, in the robot's world, as each arm's tool is asked for it.

    `carry(desk)` is where the desk solid stood in the robot's world for an
    episode whose solid the account describes as `desk`, or None when it has
    no place there. Returns `targets` ({hand: 4x4}), `episodes` (how many
    the mean is over) and `unplaced` (demonstrations left out for no place).
    """
    import json

    import pyarrow.parquet as pq

    root = Path(dataset_root)
    files = sorted(root.glob("data/*/*.parquet"))
    if not files:
        raise DemoStartError(f"no data under {root}")
    info = json.loads((root / "meta" / "info.json").read_text())
    for hand in hands:
        names = list((info["features"].get(pose_key(hand)) or {}).get("names") or [])
        if names != POSE_NAMES:
            raise DemoStartError(f"{root} has no {pose_key(hand)} with x y z and a quaternion")
    corrections = correction_episodes(root)
    sources = _sources(root)
    firsts: dict[int, dict[str, list[float]]] = {}
    for file in files:
        table = pq.read_table(file, columns=["episode_index", *(pose_key(hand) for hand in hands)])
        rows = zip(table.column("episode_index").to_pylist(),
                   *(table.column(pose_key(hand)).to_pylist() for hand in hands))
        for e, *poses in rows:
            if int(e) not in corrections:
                firsts.setdefault(int(e), dict(zip(hands, poses)))
    if not firsts:
        raise DemoStartError(f"no demonstration in {root}" if corrections else f"no episodes in {root}")
    tools: dict[str, list[np.ndarray]] = {hand: [] for hand in hands}
    unplaced = 0
    for e, poses in firsts.items():
        desk = carry((sources[e] if e < len(sources) else {}).get("desk") or {})
        if desk is None:
            unplaced += 1
            continue
        for hand in hands:
            tools[hand].append(tool_absolute_target(hand, desk, _matrix(poses[hand])))
    if not tools[hands[0]]:
        raise DemoStartError(f"none of the {len(firsts)} demonstrations in {root} has a place in the robot's world")
    targets = {}
    for hand in hands:
        T = np.eye(4)
        T[:3, :3] = mean_rotation([T[:3, :3] for T in tools[hand]])
        T[:3, 3] = np.mean([T[:3, 3] for T in tools[hand]], axis=0)
        targets[hand] = T
    return {"targets": targets, "episodes": len(tools[hands[0]]), "unplaced": unplaced}


def _sources(dataset_root: Path) -> list[dict]:
    """What convert wrote of each of the dataset's episodes' source
    (meta/pinkrab_convert.json), by episode index: each segment it kept is
    one episode, in order, and stands for the raw episode it was cut from. A
    dataset with no such account has none."""
    import json

    try:
        report = json.loads((dataset_root / "meta" / "pinkrab_convert.json").read_text())
    except (OSError, ValueError):
        return []
    return [entry for entry in report.get("episodes") or [] for _ in entry.get("segments") or []]


def correction_episodes(dataset_root: Path) -> set[int]:
    """The dataset's episodes that are corrections - kept from a rollout, not a
    demonstration - by episode index, from what convert wrote of its sources.
    A dataset with no such account has none."""
    return {index for index, entry in enumerate(_sources(dataset_root)) if entry.get("kind", "demo") != "demo"}


def start_joints(dataset_root: Path, chains: dict, carry: Callable[[dict], np.ndarray | None]
                 ) -> tuple[dict[str, np.ndarray], dict]:
    """The joints (degrees, each chain's order) that put each arm's tool where
    its hand began, by arm (the keys of `chains`).

    Solved from the ready pose by the differential IK a policy's solve uses
    (`policy_orientation_weight`), and the reach error is returned beside the
    answer - a start an arm cannot take is worth knowing before the policy is
    handed arms somewhere else.
    """
    start = demo_start(dataset_root, list(chains), carry)
    joints, found = {}, {"episodes": start["episodes"], "unplaced": start["unplaced"], "arms": {}}
    for arm, chain in chains.items():
        target = start["targets"][arm]
        q = np.array(ready_joints(arm, chain.joint_names), dtype=float)
        for _ in range(IK_STEPS):
            q = chain.track(target, q, orientation_weight=CORE.policy_orientation_weight)
        mm, deg = chain.reach_error(target, q)
        lo, hi = chain.limits
        found["arms"][arm] = {"reach_mm": float(mm * 1000.0), "reach_deg": float(deg), "target": target,
                              "height_m": float(target[2, 3]),
                              "pitch_deg": float(math.degrees(math.asin(-max(-1.0, min(1.0, target[2, 2]))))),
                              "margin_deg": float(np.min(np.minimum(q - lo, hi - q)))}
        if mm * 1000.0 > REACH_MM or deg > REACH_DEG:
            raise DemoStartError(
                f"no start the {arm} arm can take: it lands {mm * 1000.0:.0f} mm and {deg:.0f} deg off where "
                f"the {arm} hand began ({target[2, 3] * 1000:.0f} mm above the desk, "
                f"over {start['episodes']} demonstrations)")
        joints[arm] = q
    return joints, found


def dataset_of_checkpoint(checkpoint: Path) -> Path:
    """The dataset a checkpoint was trained on, from its train_config.json.

    The config holds the path on the machine that trained it. A checkpoint
    fetched from elsewhere names a directory this machine does not have, so
    the dataset of the same name in this workspace stands in for it. That name
    is the repo id's (`pinkrab/<name>`): a training machine keeps its copy as
    `<name>--<fingerprint>`, which no directory here is called - and that
    fingerprint has to be this dataset's, or it is not the one trained on.
    """
    import json
    import re

    from .config import SETTINGS
    from .dataset import fingerprint

    config = json.loads((Path(checkpoint) / "train_config.json").read_text())
    dataset = config.get("dataset") or {}
    root = dataset.get("root")
    if not root:
        raise DemoStartError(f"{checkpoint} does not say which dataset it was trained on")
    sent = re.search(r"--([0-9a-f]{12})$", Path(root).name)
    if Path(root).is_dir():
        if sent and fingerprint(Path(root)) != sent.group(1):
            raise DatasetChanged(f"trained on {root} as it was ({sent.group(1)}); it has changed since "
                                 f"({fingerprint(Path(root))})", Path(root))
        return Path(root)
    names = list(dict.fromkeys(filter(None, [str(dataset.get("repo_id") or "").rpartition("/")[2],
                                             Path(root).name[:sent.start()] if sent else Path(root).name])))
    for name in names:
        here = SETTINGS.workspace / "datasets" / name
        if not here.is_dir():
            continue
        # The fingerprint the copy was sent under is the dataset it learned
        # from; one converted again under the same name since is another.
        if sent and fingerprint(here) != sent.group(1):
            raise DatasetChanged(f"trained on {name} as it was ({sent.group(1)}); {here} has changed since "
                                 f"({fingerprint(here)})", here)
        return here
    raise DemoStartError(f"trained on {root}, and no dataset named {' or '.join(names)} is here")


def checkpoint_fps(checkpoint: Path) -> int:
    """The rate a checkpoint is asked at: its dataset's. Each of its steps is
    one of that dataset's frames apart, so asked at any other rate the arms
    move slower or faster than the takes. DemoStartError when the dataset is
    not here to say. A dataset changed since answers as it is now: nothing
    drives on it (`DatasetChanged` stops the start), and the policy can still
    be loaded to say so."""
    from .dataset import dataset_fps

    try:
        root = dataset_of_checkpoint(checkpoint)
    except DatasetChanged as exc:
        root = exc.root
    return dataset_fps(root)


def checkpoint_tasks(checkpoint: Path) -> list[str]:
    """The commands a checkpoint's dataset labels its frames with - one for a
    dataset of one task - or none when that dataset is not on this machine."""
    try:
        import pandas as pd
        return [str(t) for t in pd.read_parquet(dataset_of_checkpoint(checkpoint) / "meta" / "tasks.parquet").index]
    except (DemoStartError, OSError, KeyError, ValueError):
        return []
