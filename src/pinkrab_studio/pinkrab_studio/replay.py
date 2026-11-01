"""An episode played back on the arms: what the recorded hand would have made
the robot do, frame by frame, and where it could not have.

The recorder keeps poses, not joints - the arm is a deployment detail - so
this is where the joints come from: the hand's tool pose against the desk
solid, carried into the robot's world where the solid is reckoned to stand (pinkrab_umi.desk_solid.world_T_desk, the solid
stood as the take recorded it, at the size it recorded), asked of the arm as
it is - on the clock a dataset made of the take runs on (dataset_fps), each
hand's pose splined through the head frames as convert splines it, so the
replay plays at the rate a policy trained on it is asked and in the take's
own time, whatever rate the top camera kept. The first moment the hand is
placed is solved from the ready pose, and the arm follows the take from
there - absolute poses in the robot's own frame, as UMI and Pollen filter
takes - solved afresh now and then where following leaves it short. Only a take made with
the solid set in front of the operator's chest (meta.json's desk `placement`)
is judged so: one from
before is played with both arms at ready and named unjudged, its hand placed
nowhere. Beside each frame's joints go the reasons a frame would be
refused: the solver falling short of the target, a joint at its band, two
bodies of the robot meeting, a joint jumping between frames. Both hands are
followed, each by its own arm, as a policy of the two drives them. A hand is
named when it cannot be put on the desk - its solid never seen beside the
desk solid, never solved with its jaw read, or where the take does not
record it on the handheld.

Each hand's start is also set beside the mean start of that hand over the
takes in the same place, this one among them - near the mean a
policy run begins from - so a take that began far from the rest - looking
down more steeply, rolled, higher - is pointed out (not removed: varied
starts are wanted, a few far ones are worth a look). The
answer is cached beside the episode as replay.json and remade when the
episode, the arm's description or MuJoCo model, or where the solid carries the
hand is no longer what it was solved from - the dates of the first two and the
third itself are kept in it; the comparison with the others is made fresh.

A rollout is not solved: the arm did it, under the policy or under teleop,
which carried the hand's motion relative to where following began - where
the hand's solid stood on the desk is not where the arm went. Its joints are
the robot's own readings (robot.jsonl). A frame is flagged with who had the
arm when not the hand - `policy`, `held` where the clutch held it, `idle`
for neither - all of which convert leaves out, and `lost` where the hand
drove and no reading lies close either side. One recorded on the arm as it was built before is not
played: its angles are not this arm's.
"""
from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path

import numpy as np

from pinkrab_core.config import ARMS
from pinkrab_core.config import SETTINGS as CORE
from pinkrab_core.demo_start import level, mean_rotation
from pinkrab_core.follow import ORIENTATION_WEIGHT, tool_absolute_target
from pinkrab_core.jaw import shut_wide
from pinkrab_core.kinematics import ArmKinematics, arm_fingerprint
from pinkrab_core.origin import READY_DEG, ready_joints
from pinkrab_mujoco.scene import Scene, pressing
from pinkrab_umi.config import SETTINGS as UMI
from pinkrab_umi.convert import (
    OTHER_BUILD,
    ConvertError,
    cameras_on_desk,
    exposures,
    grid,
    latest,
    rate_of,
    resample,
    smooth,
)
from pinkrab_umi.convert import placed as has_hand
from pinkrab_umi.desk_solid import PlacementError, carried
from pinkrab_umi.episode import ROBOT_GAP_S, STREAMS

# What makes a frame red.
SHORT_MM = 15.0          # the solver landed this far from the target
SHORT_DEG = 20.0
JUMP_DEG = 15.0          # a joint moved this much between two frames
AT_LIMIT_DEG = 0.5
# An arm held short is solved afresh from ready at most this often, with this
# many starts: a take whose hand rests out of reach is not solved whole at
# every moment.
RESOLVE_S = 0.5
RESOLVE_RESTARTS = 8
# A start this far from the mean start of that hand over the takes is pointed out.
FAR_START_DEG = 20.0
FAR_START_MM = 50.0
# Bumped when what this writes changes, so a cached answer from an older
# solver is remade rather than trusted.
VERSION = 16


def _rules() -> list[float]:
    """What a touch is judged by, kept in the answer: fingertips on the desk
    to this depth are not one, at this reach of the tool centre."""
    return [CORE.fingertip_desk_mm, CORE.fingertip_reach_mm]


def _ready() -> dict[str, dict[str, float]]:
    """The ready pose a take is replayed from - the first frame is solved from
    it, and the idle arm waits in it - kept in the answer."""
    return {arm: dict(READY_DEG[arm]) for arm in ARMS}


def _desk(meta: dict) -> tuple[np.ndarray | None, str]:
    """Where a take's hand is carried into the robot's world from the desk
    solid (world_T_desk, the solid stood as the take recorded it and of the
    size it recorded), or None and why the take cannot be judged there. A
    rollout carries none."""
    if meta.get("kind") == "rollout":
        return None, ""
    return carried(meta.get("desk") or {})


def _placing(meta: dict) -> list:
    """What the take records of where its hand solids were on the handhelds,
    which places its hand: as kept in its answer."""
    return [meta.get("hands"), meta.get("solid_T_tcp")]


def _video_fps(meta: dict) -> dict[str, float]:
    """The rate each video was written at: a frame's index over it is where
    the frame sits in its file, whatever the camera kept. A stream kept
    without its pictures (the top camera's, its file None) has no video."""
    streams = meta.get("streams") or {}
    return {name: float((streams.get(name) or {}).get("fps") or UMI.fps) for name in STREAMS
            if (streams.get(name) or {}).get("file", f"{name}.mp4") is not None}


def _desk_key(meta: dict) -> list | None:
    """The take's world_T_desk as kept in its answer: what the replay depends
    on of where the solid stood and how the robot reckons it."""
    desk, _ = _desk(meta)
    return None if desk is None else np.round(desk, 6).tolist()

_chains: dict[str, ArmKinematics] = {}
_scene: Scene | None = None
# The model_mtime() the two above were built at: built again for a newer arm.
_built_at: float | None = None
# One solve at a time: the chains and the scene above hold the state of the
# take being walked, and the shelf solves on a thread while a page may ask.
_solving = threading.Lock()


def _chain(arm: str) -> ArmKinematics:
    if arm not in _chains:
        _chains[arm] = ArmKinematics(arm)
    return _chains[arm]


def _the_scene() -> Scene:
    global _scene
    if _scene is None:
        _scene = Scene(CORE.scene_file)
    return _scene


def replay_file(root: Path) -> Path:
    return root / "replay.json"


def load_or_solve(root: Path) -> dict:
    """The cached answer, or a fresh one when the episode or the arm is no longer
    what it was solved from, with its starts set beside the mean start of that hand over the takes that
    worked it."""
    answer = solved(root)
    answer["start_vs_others"] = start_vs_others(root)
    return answer


def model_mtime() -> float:
    """When the arm a replay is solved on last changed: its description (the
    xacro and the mount it is expanded with) and the MuJoCo model it is checked
    for contact in - its XML, the part meshes and the collision pieces. Read
    off the sources, so asking writes nothing: the workspace's expansion is
    brought up to date by the solver that reads it."""
    mjcf = CORE.scene_file.parent
    sources = [CORE.xacro_file, CORE.mount_file, *mjcf.glob("*.xml"), *mjcf.rglob("*.stl"),
               *CORE.mesh_dir.glob("*.stl")]
    return max(p.stat().st_mtime for p in sources if p.is_file())


def cached(root: Path, model: float) -> dict | None:
    """The answer kept beside the episode, if it stands for the episode as it
    is now, for the arm as it is now (`model`, model_mtime()'s), for this
    solver, for the touch rules it was judged by, for the ready pose it
    started from, for where the desk solid carries the hand and for where the
    take records its hand solids on the handhelds; else None.
    Judged by the dates the answer was solved from, kept in it, not by when it
    was written: an arm changed while it was being solved leaves it standing
    for the arm before. A place for the solid that cannot be read is no
    answer either: solving says why."""
    out = replay_file(root)
    if not out.is_file():
        return None
    try:
        answer = json.loads(out.read_text())
        meta = json.loads((root / "meta.json").read_text())
        desk = _desk_key(meta)
    except (OSError, ValueError, PlacementError):
        return None
    if (answer.get("version") != VERSION or answer.get("model_mtime") != model
            or answer.get("rules") != _rules() or answer.get("ready") != _ready()
            or answer.get("desk") != desk or answer.get("placing") != _placing(meta)
            or answer.get("poses_mtime") != (root / "poses.jsonl").stat().st_mtime
            or (answer.get("kind") != "rollout" and answer.get("fps") != float(CORE.dataset_fps))):
        return None
    return answer


def solved(root: Path) -> dict:
    """The kept answer, or a fresh one written beside the episode."""
    global _scene, _built_at
    # A kept answer is read without waiting: the lock is for solving, and the
    # shelf may be solving other takes for a while.
    answer = cached(root, model_mtime())
    if answer is not None:
        return answer
    with _solving:
        model = model_mtime()
        answer = cached(root, model)   # solved meanwhile by whoever held the lock
        if answer is None:
            # A newer arm than the solver and the scene were read from: both
            # are read again, or the answer would be the old arm's under a
            # new date.
            if _built_at != model:
                _chains.clear()
                _scene = None
                _built_at = model
            poses = (root / "poses.jsonl").stat().st_mtime
            answer = solve(root)
            answer["poses_mtime"], answer["model_mtime"] = poses, model
            replay_file(root).write_text(json.dumps(answer))
    return answer


def _hand_start(root: Path, arm: str) -> tuple[np.ndarray, float] | None:
    """The hand's first placed tool pose in a take, placed as convert places
    it: its attitude with the heading out, and its height above the desk
    surface (m) - over the desk solid's origin, which stood as high above the
    desk as the take records. None if the hand was never placed."""
    if _kind(root) == "rollout":
        return None   # the hand's start in it is not where the arm began
    poses = _lines(root / "poses.jsonl")
    try:
        world_T_cams = cameras_on_desk(poses, ARMS)
    except ConvertError:
        return None
    try:
        meta = json.loads((root / "meta.json").read_text())
        solid_T_tcp = UMI.recorded_solid_T_tcp(f"handheld_{arm}", meta)
        above = float(meta["desk"]["origin_above_table_mm"]) / 1000.0
    except (OSError, ValueError, TypeError, KeyError):
        return None   # a take that does not record its solid's size, or its solid's height, places no hand
    for p, world_T_cam in zip(poses, world_T_cams):
        if world_T_cam is not None and has_hand(p, arm):
            T = world_T_cam @ np.array(p["solids"][f"handheld_{arm}"]["pose"], dtype=float) @ solid_T_tcp
            return level(T[:3, :3]), float(T[2, 3]) + above
    return None


def start_vs_others(root: Path) -> dict:
    """Per hand in this take: how far its start lies from the mean start of
    that hand over every take in its folder that convert would take it from,
    this one included - near the mean a policy run begins from, which is
    taken over the converted episodes' first frames (attitude in degrees,
    height in mm) - and whether that is far enough to point out."""
    out = {}
    takes = [p for p in root.parent.iterdir() if (p / "meta.json").is_file()]
    for arm in ARMS:
        mine = _hand_start(root, arm)
        if mine is None:
            continue
        starts = [s for s in (_hand_start(p, arm) for p in takes) if s is not None]
        if len(starts) < 3:
            out[arm] = {"takes": len(starts), "note": "too few takes to compare with"}
            continue
        out[arm] = {"takes": len(starts), **start_apart(mine, starts)}
    return out


def start_apart(mine: tuple[np.ndarray, float], starts: list[tuple[np.ndarray, float]]) -> dict:
    """How far one start (attitude, height) lies from the mean of `starts`:
    degrees, millimetres, and whether that is far enough to point out."""
    mean_R = mean_rotation([R for R, _ in starts])
    mean_h = float(np.mean([h for _, h in starts]))
    cos = (np.trace(mean_R.T @ mine[0]) - 1.0) / 2.0
    deg = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
    mm = (mine[1] - mean_h) * 1000.0
    return {"deg": round(deg, 1), "mm": round(mm, 1), "far": deg > FAR_START_DEG or abs(mm) > FAR_START_MM}


def _kind(root: Path) -> str:
    try:
        return json.loads((root / "meta.json").read_text()).get("kind") or "demo"
    except (OSError, ValueError):
        return "demo"


def _lines(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _nearest_index(stamps: np.ndarray, t: float) -> int | None:
    if len(stamps) == 0:
        return None
    i = int(np.searchsorted(stamps, t))
    candidates = [j for j in (i - 1, i) if 0 <= j < len(stamps)]
    return min(candidates, key=lambda j: abs(stamps[j] - t))


# Who had a rollout's arm, when not the hand, as the frame's flag.
NOT_THE_HAND = {"policy": "policy", "hold": "held", "none": "idle"}


def _rollout(root: Path, meta: dict) -> dict:
    """A rollout played back from the robot's readings, by the top
    camera's clock (the arm's camera's, with no head video); the other arm
    waits at ready."""
    started = time.monotonic()
    arms = [arm for arm in ARMS if arm in (meta.get("arms") or [])]
    if not arms:
        raise ValueError(f"a rollout of no arm ({meta.get('arms')!r})")
    if any((meta.get("description") or {}).get(arm) != arm_fingerprint(arm) for arm in arms):
        raise ValueError(OTHER_BUILD)
    frames = _lines(root / "frames.jsonl")
    readings = sorted(_lines(root / "robot.jsonl"), key=lambda r: r["host_ts"])
    by_stream = {name: np.array([f["host_ts"] for f in frames if f["stream"] == name]) for name in STREAMS}
    clock = "head" if len(by_stream["head"]) else f"gripper_{arms[0]}"
    fps = float((meta.get("streams") or {}).get(clock, {}).get("fps") or UMI.fps)
    still: dict[str, float] = {}
    for other in ARMS:
        chain = _chain(other)
        still.update(zip(chain.joint_names, (math.radians(v) for v in ready_joints(other, chain.joint_names))))
    names = [n for arm in arms for n in (*_chain(arm).joint_names, f"{arm}_gripper")]
    t = np.array([r["host_ts"] for r in readings], dtype=float) - UMI.robot_state_latency_s
    table = np.array([[r["joints"].get(n, np.nan) for n in names] for r in readings], dtype=float)
    controllers = [r.get("controller", "none") for r in readings]
    out_frames = []
    for i, ts in enumerate(by_stream[clock]):
        flags = []
        k = int(np.searchsorted(t, ts))
        here = None
        if 0 < k < len(t) and t[k] - t[k - 1] <= ROBOT_GAP_S:
            w = (ts - t[k - 1]) / (t[k] - t[k - 1]) if t[k] > t[k - 1] else 0.0
            here = table[k - 1] + w * (table[k] - table[k - 1])
        if here is None or not np.isfinite(here).all():
            # The arm left where it was last read. A hole matters where the
            # hand drove - convert keeps only that; elsewhere the frame is the
            # policy's or nobody's whether it was read or not.
            near = [controllers[j] for j in (k - 1, k) if 0 <= j < len(t)]
            other = next((c for c in near if c != "human"), None)
            flags.append("lost" if not near or "human" in near else NOT_THE_HAND.get(other, "idle"))
        else:
            still.update({n: float(v) for n, v in zip(names, here)})
            # Convert keeps an exposure only between two readings of the hand.
            other = next((c for c in (controllers[k - 1], controllers[k]) if c != "human"), None)
            if other is not None:
                flags.append(NOT_THE_HAND.get(other, "idle"))
        # One controller has both arms: a reading's flags are each arm's.
        frame = {"i": i, "t": round(float(ts - by_stream[clock][0]), 4), "host_ts": float(ts),
                 "joints": {n: round(v, 5) for n, v in still.items()},
                 "flags": {a: (list(flags) if a in arms else []) for a in ARMS}}
        frame.update({name: _nearest_index(by_stream[name], ts) for name in STREAMS})
        out_frames.append(frame)
    counts = {a: {} for a in ARMS}
    for arm in arms:
        for flag in (flag for f in out_frames for flag in f["flags"][arm]):
            counts[arm][flag] = counts[arm].get(flag, 0) + 1
    return {
        "version": VERSION, "rules": _rules(), "ready": _ready(), "desk": None, "kind": "rollout",
        "placing": _placing(meta),
        "name": root.name, "fps": fps, "frames": len(out_frames), "videos": _video_fps(meta),
        "solved_in_s": round(time.monotonic() - started, 2),
        "attitude": "the robot's own joints",
        "hands": arms, "start": {}, "unplaced": {},
        "summary": {"clean": sum(1 for f in out_frames if not f["flags"][arms[0]]), "flags": counts},
        "trajectory": out_frames,
    }


def _exposed(poses: list[dict]) -> np.ndarray:
    """When each head frame was exposed, on this host's clock - as convert
    reads them - or when it was solved, for a take whose stamps are not on
    this host's clock."""
    try:
        return exposures(poses)
    except ConvertError:
        return np.array([p["host_ts"] for p in poses], dtype=float)


def _on_grid(tools: list[np.ndarray | None], times: np.ndarray, at: np.ndarray, rate: float) -> list:
    """A hand's tool poses at the head frames' `times` laid on the dataset's
    clock `at`, as convert lays them: each run of frames the hand was placed
    in smoothed and splined through, and read at the moments inside it. A
    moment outside every run - the hand lost, or a run of a single frame - is
    None."""
    out: list[np.ndarray | None] = [None] * len(at)
    start = None
    for i in range(len(tools) + 1):
        usable = i < len(tools) and tools[i] is not None
        if start is not None and (not usable or times[i] <= times[i - 1]):
            if i - start >= 2:
                t = times[start:i]
                inside = np.nonzero((at >= t[0]) & (at <= t[-1]))[0]
                if len(inside):
                    # Splined from the run's start: absolute stamps lose digits.
                    for k, T in zip(inside, resample(smooth(np.array(tools[start:i]), rate), t - t[0],
                                                     at[inside] - t[0])):
                        out[k] = T
            start = None
        if usable and start is None:
            start = i
    return out


def solve(root: Path) -> dict:
    meta = json.loads((root / "meta.json").read_text())
    kind = meta.get("kind", "demo")
    if kind == "rollout":
        return _rollout(root, meta)
    if kind != "demo":
        raise ValueError(f"a {kind} is not replayed")
    poses = _lines(root / "poses.jsonl")
    if not poses:
        raise ValueError("a take with no poses")
    frames = _lines(root / "frames.jsonl")
    scene = _the_scene()
    desk, unjudged = _desk(meta)

    # The take on the clock a dataset made of it runs on - a moment every
    # 1/dataset_fps s from its first head frame's exposure to its last - which
    # is the clock a policy trained on it is asked on: each hand's pose
    # splined through the head frames as convert splines it, whatever rate
    # the top camera kept, and each video's last frame by then shown.
    fps = float(CORE.dataset_fps)
    times = _exposed(poses)
    rate = rate_of(times) if len(times) > 1 else UMI.fps
    at = grid(float(times[0]), float(times[-1]), int(fps))
    head = np.clip(latest(times, at), 0, max(0, sum(1 for f in frames if f["stream"] == "head") - 1))
    shots = {f"gripper_{arm}": latest(np.array([f["host_ts"] for f in frames if f["stream"] == f"gripper_{arm}"])
                                      - UMI.gripper_camera_latency_s, at) for arm in ARMS}

    arms = {}
    starts = {}
    unplaced: dict[str, str] = {}
    # The hands against the desk solid as convert has them: the camera,
    # worn on a head, placed on the desk by each frame's own desk solve, and
    # a frame without one is not placed.
    try:
        world_T_cams = cameras_on_desk(poses, ARMS)
    except ConvertError as err:
        world_T_cams, unseen = None, str(err)
    for arm in ARMS:
        chain = _chain(arm)
        ready = np.array(ready_joints(arm, chain.joint_names), dtype=float)
        arms[arm] = {"chain": chain, "ready": ready, "joints": ready, "targets": None, "present": False,
                     "solved_at": -math.inf}
        arms[arm]["lo"], arms[arm]["hi"] = chain.limits
        readings = [(p["host_ts"] - UMI.jaw_latency_s, p["jaw"][arm])
                    for p in poses if (p.get("jaw") or {}).get(arm) is not None]
        arms[arm]["jaw"] = (np.interp(at, *(np.array(v, dtype=float) for v in zip(*readings)))
                            if readings else None)
        if world_T_cams is None:
            unplaced[arm] = unseen
            continue
        placed = [world_T_cam @ np.array(p["solids"][f"handheld_{arm}"]["pose"], dtype=float)
                  if world_T_cam is not None and has_hand(p, arm) else None
                  for p, world_T_cam in zip(poses, world_T_cams)]
        if not any(T is not None for T in placed):
            unplaced[arm] = "its solid was never solved with its jaw read"
            continue
        if desk is None:
            continue   # unjudged: no place in the robot's world to put the hand
        try:
            solid_T_tcp = UMI.recorded_solid_T_tcp(f"handheld_{arm}", meta)
        except KeyError:
            unplaced[arm] = "the take does not record where its solid was on the handheld"
            continue
        tools = _on_grid([None if T is None else T @ solid_T_tcp for T in placed], times, at, rate)
        targets = [None if T is None else tool_absolute_target(arm, desk, T) for T in tools]
        first = next((T for T in targets if T is not None), None)
        if first is None:
            unplaced[arm] = "it was never placed for two frames running"
            continue
        # The first target solved from the ready pose alone, as a policy run
        # walks the arm there from ready: a restart from elsewhere could land
        # on a shape of the arm it never passes through on the way. The
        # attitude is weighed against the position as teleop weighs it.
        joints = chain.solve(first, seed=ready, restarts=1, orientation_weight=ORIENTATION_WEIGHT)
        mm, deg = chain.reach_error(first, joints)
        a = arms[arm]
        a["present"] = True
        a["joints"] = joints
        a["targets"] = targets
        starts[arm] = {"reached": bool(mm * 1000.0 <= SHORT_MM and deg <= SHORT_DEG),
                       "reach_mm": round(mm * 1000.0, 1), "reach_deg": round(deg, 1),
                       "height_mm": round(float(first[2, 3]) * 1000.0)}

    gripper_band = {arm: scene.limits.get(f"{arm}_gripper") for arm in ARMS}
    gripper_band = {arm: band and shut_wide(*band) for arm, band in gripper_band.items()}
    out_frames = []
    previous: dict[str, np.ndarray] = {}
    started = time.monotonic()
    for k, t in enumerate(at):
        t = float(t)
        entry = {
            "i": k, "t": round(t - float(at[0]), 4), "host_ts": t, "head": int(head[k]),
            **{name: (int(index[k]) if index[k] >= 0 else None) for name, index in shots.items()},
            "joints": {}, "flags": {}, "short": {},
        }
        pose_all: dict[str, float] = {}
        for arm, a in arms.items():
            chain = a["chain"]
            flags: list[str] = []
            if not a["present"]:
                # Not in this take: held at the ready pose, judged on nothing.
                for n, v in zip(chain.joint_names, a["joints"]):
                    pose_all[n] = math.radians(float(v))
                entry["flags"][arm] = flags
                entry["short"][arm] = None
                continue
            target = a["targets"][k]
            if target is None:
                flags.append("lost")   # the hand had no place in the world this moment: the arm holds
                short = None
            else:
                joints = chain.track(target, a["joints"], orientation_weight=ORIENTATION_WEIGHT, rate_hz=fps)
                mm, deg = chain.reach_error(target, joints)
                # Followed from where it was, an arm sent to a pose beyond it
                # folds against its bands and stays there when the hand comes
                # back within reach: a pose of another shape of the arm is
                # never walked to. So one held short is solved afresh from
                # ready now and then, and the closer answer kept - the step
                # across shows as a jump.
                if (mm * 1000.0 > SHORT_MM or deg > SHORT_DEG) and t - a["solved_at"] >= RESOLVE_S:
                    a["solved_at"] = t
                    fresh = chain.solve(target, seed=a["ready"], restarts=RESOLVE_RESTARTS,
                                        orientation_weight=ORIENTATION_WEIGHT)
                    fresh_mm, fresh_deg = chain.reach_error(target, fresh)
                    if fresh_mm * 1000.0 + fresh_deg < mm * 1000.0 + deg:
                        joints, mm, deg = fresh, fresh_mm, fresh_deg
                a["joints"] = joints
                short = {"mm": round(mm * 1000.0, 1), "deg": round(deg, 1)}
                if mm * 1000.0 > SHORT_MM or deg > SHORT_DEG:
                    flags.append("short")
            joints = a["joints"]
            at_limit = [n for n, v, lo, hi in zip(chain.joint_names, joints, a["lo"], a["hi"])
                        if v - lo < AT_LIMIT_DEG or hi - v < AT_LIMIT_DEG]
            if at_limit:
                flags.append("limit")
                entry.setdefault("at_limit", {})[arm] = at_limit
            if arm in previous:
                jump = float(np.max(np.abs(joints - previous[arm])))
                if jump > JUMP_DEG:
                    flags.append("jump")
            previous[arm] = joints.copy()
            for n, v in zip(chain.joint_names, joints):
                pose_all[n] = math.radians(float(v))
            # The jaw: the hand's opening spent across the arm's own band.
            band = gripper_band[arm]
            if a["jaw"] is not None and band is not None:
                pose_all[f"{arm}_gripper"] = band[0] + float(a["jaw"][k]) * (band[1] - band[0])
            entry["flags"][arm] = flags
            entry["short"][arm] = short
        # Fingertips on the desk within the allowance are not a touch: a thin
        # object is picked off the desk so (pinkrab_mujoco.scene.pressing).
        touching = [c for c in scene.touching(pose_all) if pressing(c)]
        if touching:
            entry["touching"] = [f"{c['a']} / {c['b']}" for c in touching[:3]]
            # Each arm is flagged for what its own bodies touch: a resting
            # arm on the desk says nothing about the one doing the work.
            for arm in ARMS:
                if arms[arm]["present"] and any(label.startswith(f"{arm}_")
                                                for c in touching for label in (c["a"], c["b"])):
                    entry["flags"][arm].append("touch")
        entry["joints"] = {n: round(v, 5) for n, v in pose_all.items()}
        out_frames.append(entry)

    counts = {arm: {} for arm in ARMS}
    for f in out_frames:
        for arm in ARMS:
            for flag in f["flags"][arm]:
                counts[arm][flag] = counts[arm].get(flag, 0) + 1
    # Clean: the arm followed the hand and nothing held it back - none of a
    # take whose hand it never followed.
    hands = [arm for arm in ARMS if arms[arm]["present"]]
    clean = sum(1 for f in out_frames if hands and not any(f["flags"][arm] for arm in hands))
    return {
        "version": VERSION, "rules": _rules(), "ready": _ready(),
        "desk": None if desk is None else np.round(desk, 6).tolist(),
        "placing": _placing(meta),
        "name": root.name, "fps": fps, "frames": len(out_frames), "videos": _video_fps(meta),
        "solved_in_s": round(time.monotonic() - started, 2),
        "attitude": "as held against the desk solid",
        "hands": hands,
        "start": starts,
        "unplaced": unplaced,
        "unjudged": unjudged,
        "summary": {"clean": clean, "flags": counts},
        "trajectory": out_frames,
    }
