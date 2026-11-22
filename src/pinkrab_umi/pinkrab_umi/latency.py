"""`pinkrab-umi latency` - how late each gripper camera's pictures arrive.

A gripper camera is bolted to the tool it looks out of, so how fast its
pictures turn is how fast the tool turns. The same turning is read off a
stream that does not wait on the camera - on a handheld, the top camera's
poses of its solid; on an arm, the forward kinematics of the joints the
driver reads - and the two are slid against each other until they agree
best (cross-correlation of the turning speeds). Where they agree is how much
later the pictures arrive than the latencies they are stamped with say:
camera.yaml's `latency_s` is right when it is zero.

A picture's turning is read off features followed from one frame to the
next (Lucas-Kanade), undistorted through the unit's lens fit to rays, and
the rotation that best carries one frame's rays onto the next (Kabsch, with
RANSAC: what moves with the tool - its own jaws - and what moves on its own
are left out). Only the upper part of the picture is followed - the jaws
fill the lower - and only rays within 60 degrees of the lens axis. A hand
that hardly turned in a take says nothing, and is left out.

A rollout recorded with its commands (robot.jsonl's `asked`, what the policy
or the hands asked for, and `sent`, what the gate let through) also says how
long a step takes to become the arm's motion, the inference setting
action_latency_s: each joint's speed as asked, as shaped and as read, slid
against each other the same way.

Convert lays a dataset's demonstrations with the gripper latency their own
hands measure (demo_latency): camera.yaml's, moved by the median of the
takes' lags against the top camera, one value for every take - a lag
measured off one take is a frame or so either way, and the latency itself
is the cameras' and the computer's, not a take's. Each sitting's median is
kept beside it, to tell when the rig changed between them.

    pinkrab-umi latency                 # every take on the shelf
    pinkrab-umi latency 20261102-000717 20261102-084501
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from pinkrab_core.config import ARMS
from pinkrab_core.config import SETTINGS as CORE
from pinkrab_core.dataset import gripper_region, lens_fit

from .config import SETTINGS
from .convert import ConvertError, cameras_on_desk, exposures, placed, rate_of, segments, smooth
from .episode import _lines

# The slides tried, seconds: the pictures from 0.1 s ahead of their stamps to
# 0.3 s behind them, a millisecond apart.
LAGS_S = np.arange(-0.1, 0.3001, 0.001)
# A take says something of a camera when the turnings agree at least this
# well, and the tool turned at least this fast for a tenth of the take.
MIN_AGREEMENT = 0.6
MIN_TURN_DPS = 5.0
# Rays this far from the lens axis or nearer it are followed (cos 60 deg),
# and features on the frame's upper part only.
MIN_RAY_Z = 0.5
UPPER = 0.55
FEATURES = 400
RANSAC_TRIES = 60
RANSAC_RAD = 0.004
# A turn this fast between two frames is a follow gone wrong.
MAX_TURN_RAD_S = 20.0
# Convert moves the gripper latency to what the demonstrations measure when
# this many hands say; fewer, and camera.yaml's stands.
MIN_HANDS = 10
# A take started this long after the one before is of another sitting: the
# cameras may have been plugged in again, or the computer started again.
SITTING_GAP_S = 3600.0


def _rays(points: np.ndarray, region, scale, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Pixels of a recorded frame as unit rays through the lens."""
    x0, y0, _, _ = region
    sensor = np.stack([x0 + points[:, 0] * scale[0], y0 + points[:, 1] * scale[1]], axis=1)
    flat = cv2.fisheye.undistortPoints(sensor.reshape(-1, 1, 2).astype(np.float64), K, D).reshape(-1, 2)
    rays = np.concatenate([flat, np.ones((len(flat), 1))], axis=1)
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def _kabsch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(b.T @ a)
    return U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt


def picture_turning(video: Path, lens: dict, times: np.ndarray) -> np.ndarray:
    """How fast a gripper video turns, frame to frame: (n, 2) of [moment,
    rad/s], each between the two frames' stamps `times`."""
    K = np.array([[lens["fx"], 0.0, lens["cx"]], [0.0, lens["fy"], lens["cy"]], [0.0, 0.0, 1.0]])
    D = np.array(lens["k"], dtype=float)
    cap = cv2.VideoCapture(str(video))
    rng = np.random.default_rng(0)
    out, before, i = [], None, 0
    region = scale = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if region is None:
            h, w = grey.shape
            region = gripper_region(w, h)
            scale = (region[2] / w, region[3] / h)
        if before is not None and i < len(times):
            p0 = cv2.goodFeaturesToTrack(before, FEATURES, 0.01, 8)
            if p0 is not None and len(p0) >= 12:
                p1, status, _ = cv2.calcOpticalFlowPyrLK(before, grey, p0, None, winSize=(21, 21), maxLevel=3)
                kept = status.ravel() == 1
                a, b = p0[kept].reshape(-1, 2), p1[kept].reshape(-1, 2)
                A, B = _rays(a, region, scale, K, D), _rays(b, region, scale, K, D)
                use = (A[:, 2] > MIN_RAY_Z) & (B[:, 2] > MIN_RAY_Z) & (a[:, 1] < UPPER * grey.shape[0])
                A, B = A[use], B[use]
                if len(A) >= 12:
                    best = None
                    for _ in range(RANSAC_TRIES):
                        pick = rng.choice(len(A), 3, replace=False)
                        inliers = np.linalg.norm(B - A @ _kabsch(A[pick], B[pick]).T, axis=1) < RANSAC_RAD
                        if best is None or inliers.sum() > best.sum():
                            best = inliers
                    dt = times[i] - times[i - 1]
                    if best.sum() >= 10 and dt > 0:
                        speed = np.linalg.norm(Rotation.from_matrix(_kabsch(A[best], B[best])).as_rotvec()) / dt
                        if speed < MAX_TURN_RAD_S:
                            out.append(((times[i] + times[i - 1]) / 2, speed))
        before = grey
        i += 1
    cap.release()
    return np.array(out).reshape(-1, 2)


def _turning(times: np.ndarray, rotations: np.ndarray) -> np.ndarray:
    dt = np.diff(times)
    turn = Rotation.from_matrix(np.einsum("nji,njk->nik", rotations[:-1], rotations[1:])).magnitude()
    return np.stack([(times[:-1] + times[1:]) / 2, turn / np.where(dt > 0, dt, np.inf)], axis=1)


def hand_turning(root: Path, meta: dict, hand: str) -> np.ndarray:
    """How fast a handheld's tool turns, off the top camera's poses as
    convert places it: (n, 2) of [moment, rad/s]."""
    poses = _lines(root / "poses.jsonl")
    times = exposures(poses, meta)
    world_T_cams = cameras_on_desk(poses, ARMS)
    tools = [W @ np.array(p["solids"][f"handheld_{hand}"]["pose"]) if W is not None and placed(p, hand) else None
             for p, W in zip(poses, world_T_cams)]
    runs = []
    for start, end in segments(np.array([T is not None for T in tools]), times, 0.5):
        T = smooth(np.array(tools[start:end]), rate_of(times[start:end]))
        runs.append(_turning(times[start:end], T[:, :3, :3]))
    return np.concatenate(runs) if runs else np.zeros((0, 2))


def arm_turning(root: Path, hand: str) -> np.ndarray:
    """How fast an arm's tool turns, off the joints read in a rollout, at
    60 a second: (n, 2) of [moment, rad/s]."""
    from pinkrab_core.kinematics import ArmKinematics

    readings = sorted(_lines(root / "robot.jsonl"), key=lambda r: r["host_ts"])
    chain = ArmKinematics(hand)
    t = np.array([r["host_ts"] for r in readings], dtype=float) - SETTINGS.robot_state_latency_s
    J = np.array([[r["joints"].get(n, np.nan) for n in chain.joint_names] for r in readings], dtype=float)
    ok = np.isfinite(J).all(axis=1)
    t, J = t[ok], J[ok]
    if len(t) < 2:
        return np.zeros((0, 2))
    at = np.arange(t[0], t[-1], 1 / 60)
    joints = np.stack([np.interp(at, t, J[:, k]) for k in range(J.shape[1])], axis=1)
    return _turning(at, np.array([chain.forward(np.degrees(j))[:3, :3] for j in joints]))


def lag(pictures: np.ndarray, tool: np.ndarray) -> tuple[float, float] | None:
    """How much later the pictures turn than the tool, seconds, and how well
    they agree there (correlation) - None where they overlap too little."""
    at = np.arange(max(pictures[0, 0], tool[0, 0]) - LAGS_S[0], min(pictures[-1, 0], tool[-1, 0]) - LAGS_S[-1],
                   0.001)
    if len(at) < 1000:
        return None
    reference = np.interp(at, tool[:, 0], tool[:, 1])
    scores = np.array([np.corrcoef(reference, np.interp(at + s, pictures[:, 0], pictures[:, 1]))[0, 1]
                       for s in LAGS_S])
    best = int(np.nanargmax(scores))
    return float(LAGS_S[best]), float(scores[best])


def _speeds(readings: list[dict], key: str, name: str, at: np.ndarray, less: float) -> np.ndarray | None:
    """One joint's speed under `key` of the readings that have it, at `at`."""
    rows = [(r["host_ts"] - less, r[key][name]) for r in readings if name in (r.get(key) or {})]
    if len(rows) < 30:
        return None
    t, q = np.array(rows).T
    # The speed at each reading, then laid on `at`: speeds taken after
    # laying the angles would be a staircase of the readings' rate, and two
    # staircases agree best a whole reading apart.
    return np.interp(at, t, np.gradient(q, t))


def command_lags(root: Path) -> dict[str, float] | None:
    """A rollout's lags from command to motion, seconds - `gate` (asked to
    sent), `servo` (sent to read) and `total` (asked to read), each the median
    over the joints that moved - or None for a run recorded without its
    commands."""
    readings = sorted((r for r in _lines(root / "robot.jsonl") if r.get("asked")), key=lambda r: r["host_ts"])
    if len(readings) < 60:
        return None
    at = np.arange(readings[0]["host_ts"] + 0.5, readings[-1]["host_ts"] - 0.5, 0.001)
    if len(at) < 1000:
        return None
    found: dict[str, list[float]] = {"gate": [], "servo": [], "total": []}
    for name in sorted(readings[-1]["asked"]):
        asked = _speeds(readings, "asked", name, at, 0.0)
        sent = _speeds(readings, "sent", name, at, 0.0)
        read = _speeds(readings, "joints", name, at, SETTINGS.robot_state_latency_s)
        if asked is None or read is None or np.degrees(np.percentile(np.abs(read), 90)) < MIN_TURN_DPS:
            continue
        for what, a, b in (("gate", asked, sent), ("servo", sent, read), ("total", asked, read)):
            if a is None or b is None:
                continue
            found_lag = lag(np.stack([at, b], axis=1), np.stack([at, a], axis=1))
            if found_lag is not None and found_lag[1] >= MIN_AGREEMENT:
                found[what].append(found_lag[0])
    return {what: float(np.median(lags)) for what, lags in found.items() if lags} or None


def measure(root: Path) -> dict[str, dict]:
    """One take's lag for each hand that turned enough: {hand: {"lag_s",
    "agreement", "against"}} - against the top camera for a demonstration,
    against the joints for a rollout."""
    meta = json.loads((root / "meta.json").read_text())
    rollout = meta.get("kind") == "rollout"
    frames = _lines(root / "frames.jsonl")
    out = {}
    for hand in ARMS:
        stream = (meta.get("streams") or {}).get(f"gripper_{hand}") or {}
        lens = stream.get("lens") or lens_fit(f"follower_{hand}" if rollout else hand)
        shots = sorted((f for f in frames if f["stream"] == f"gripper_{hand}"), key=lambda f: f["index"])
        if lens is None or len(shots) < 30:
            continue
        times = np.array([f["host_ts"] for f in shots], dtype=float) - SETTINGS.gripper_camera_latency_s
        try:
            tool = arm_turning(root, hand) if rollout else hand_turning(root, meta, hand)
        except ConvertError:
            continue
        if len(tool) < 30 or np.degrees(np.percentile(tool[:, 1], 90)) < MIN_TURN_DPS:
            continue
        pictures = picture_turning(root / f"gripper_{hand}.mp4", lens, times)
        if len(pictures) < 30:
            continue
        found = lag(pictures, tool)
        if found is not None and found[1] >= MIN_AGREEMENT:
            out[hand] = {"lag_s": round(found[0], 3), "agreement": round(found[1], 2),
                         "against": "joints" if rollout else "top camera"}
    return out


def demo_latency(source: Path, names: list[str], fps: int | None = None, say=print) -> dict:
    """The gripper latency for convert to lay these demonstrations' pictures
    with, as `settled` reads what each one's hands measure against the top
    camera. Each take is said, a rollout too, which is not measured here -
    its pictures are laid on the joints, by camera.yaml's latency."""
    takes = {}
    for name in names:
        root = source / name
        try:
            meta = json.loads((root / "meta.json").read_text())
            if meta.get("kind", "demo") != "demo":
                say(f"  {name}: a {meta.get('kind')} - laid by camera.yaml's latency")
                continue
            found = measure(root)
        except (OSError, ValueError, KeyError, ConvertError, cv2.error) as exc:
            say(f"  {name}: latency not measured - {exc}")
            continue
        said = ", ".join(f"{hand} {h['lag_s'] * 1000:+.0f} ms" for hand, h in found.items())
        say(f"  {name}: {said or 'no hand turned enough to measure'}")
        if found:
            takes[name] = (meta.get("started_wall"), [h["lag_s"] for h in found.values()])
    return settled(takes, SETTINGS.gripper_camera_latency_s, CORE.dataset_fps if fps is None else fps)


def settled(takes: dict[str, tuple[float | None, list[float]]], stamped: float, fps: int) -> dict:
    """What `takes` ({name: (started_wall, each hand's lag)}) say of the
    gripper latency: `gripper_camera_latency_s`, `stamped` (camera.yaml's)
    moved by the median of every hand's lag - or `stamped` as it is, with
    `measured` False, when fewer than MIN_HANDS hands say - and each sitting's
    own (takes SITTING_GAP_S or more apart begin another; a take that does
    not say when it started is in none), with `sittings_differ` when one is
    more than a dataset frame off the latency used."""
    lags = [lag_s for _, hand_lags in takes.values() for lag_s in hand_lags]
    measured = len(lags) >= MIN_HANDS
    used = stamped + float(np.median(lags)) if measured else stamped
    sittings: list[dict] = []
    last = None
    for name, (when, hand_lags) in sorted(((n, t) for n, t in takes.items() if t[0] is not None),
                                          key=lambda item: item[1][0]):
        if last is None or when - last >= SITTING_GAP_S:
            sittings.append({"first": name, "takes": 0, "lags": []})
        sittings[-1]["takes"] += 1
        sittings[-1]["lags"].extend(hand_lags)
        last = when
    for sitting in sittings:
        hand_lags = sitting.pop("lags")
        sitting["hands"] = len(hand_lags)
        sitting["latency_s"] = round(stamped + float(np.median(hand_lags)), 4)
    return {"gripper_camera_latency_s": round(used, 4), "measured": measured, "stamped_s": stamped,
            "hands": len(lags), "min_hands": MIN_HANDS, "against": "top camera", "sittings": sittings,
            "sittings_differ": measured and any(abs(s["latency_s"] - used) > 1.0 / fps for s in sittings)}


def said(latency: dict) -> str:
    """One line on the gripper latency `settled` found."""
    used, stamped = latency["gripper_camera_latency_s"], latency["stamped_s"]
    if not latency["measured"]:
        return (f"  gripper latency: camera.yaml's {stamped * 1000:.0f} ms - only {latency['hands']} hand(s) "
                f"turned enough to measure it, {latency['min_hands']} wanted")
    line = (f"  gripper latency: {used * 1000:.0f} ms, measured over {latency['hands']} hands "
            f"(camera.yaml {stamped * 1000:.0f} ms)")
    if latency["sittings_differ"]:
        line += " - sittings differ by more than a frame: " + ", ".join(
            f"{s['first']} {s['latency_s'] * 1000:.0f} ms" for s in latency["sittings"])
    return line


def main(names: list[str] | None = None) -> int:
    root = SETTINGS.episodes_dir
    names = names or sorted(p.name for p in root.iterdir() if (p / "meta.json").is_file())
    found: dict[str, list[float]] = {"top camera": [], "joints": []}
    commands: dict[str, list[float]] = {"gate": [], "servo": [], "total": []}
    for name in names:
        try:
            answer = measure(root / name)
            lags = command_lags(root / name) if (root / name / "robot.jsonl").is_file() else None
        except (OSError, ValueError) as exc:
            print(f"  {name}: skipped - {exc}")
            continue
        if lags:
            for what, value in lags.items():
                commands[what].append(value)
            print(f"  {name}: a step is made {lags.get('total', float('nan')) * 1000:.0f} ms after it is asked "
                  f"(gate {lags.get('gate', float('nan')) * 1000:.0f} ms, servo {lags.get('servo', float('nan')) * 1000:.0f} ms)")
        for hand, said in answer.items():
            found[said["against"]].append(said["lag_s"])
            print(f"  {name} {hand}: {said['lag_s'] * 1000:+.0f} ms against the {said['against']} "
                  f"(agreement {said['agreement']:.2f})")
    print()
    for against, lags in found.items():
        if lags:
            q1, mid, q3 = np.percentile(lags, [25, 50, 75]) * 1000
            print(f"  against the {against}: {mid:+.0f} ms later than stamped (middle half {q1:+.0f} to {q3:+.0f}, "
                  f"{len(lags)} hand(s)) - gripper latency_s {SETTINGS.gripper_camera_latency_s * 1000:.0f} ms now")
    if not any(found.values()):
        print("  no take turned a hand enough to say")
    if commands["total"]:
        from pinkrab_core import inference

        print(f"  commands: a step is made {np.median(commands['total']) * 1000:.0f} ms after it is asked "
              f"({len(commands['total'])} run(s)) - action_latency_s {inference.common()['action_latency_s'] * 1000:.0f} ms now")
    else:
        print("  no run recorded its commands")
    return 0
