"""What the shelf says of each take: whether it is worth training on, and why not.

UMI's dataset plan drops a demonstration on its tracking - too many frames
lost, too few held - and its paper filters what the target robot cannot
follow; Grabette checks each trajectory for jumps and lost tracking and leaves
the choice to a person. A take is judged the same way here, from what was
recorded and from what the replay on the arms found, against thresholds that
are UMI's where it has them (pinkrab_umi.config, keep_*) - and an arm swung
far round shoulder_pan from the ready pose to follow its hand. A take that fails
says which test it failed, and a warning is shown and not counted: the
deleting is the person's. Both hands are judged, and a take fails on either. A take recorded before the desk solid stood at the
chest has no place in the robot's world for the arm to be judged in: its
tracking is judged, and the arm is named unjudged - a warning, not a fail,
so tick failing does not offer every take from before for deletion.

The replay behind each verdict is solved on a thread and kept beside the
episode, so asking is quick once the shelf has been through it.
"""
from __future__ import annotations

import collections
import json
import threading
from pathlib import Path

import cv2
import numpy as np

from pinkrab_core.config import ARMS
from pinkrab_core.config import SETTINGS as CORE
from pinkrab_core.jaw import aperture_of, gripper_band
from pinkrab_core.follow import desk_tool
from pinkrab_core.origin import READY_DEG, ready_joints
from pinkrab_umi.config import SETTINGS as UMI
from pinkrab_umi.convert import ConvertError, cameras_on_desk, placed, rate_of
from pinkrab_umi.ready_guide import apart
from pinkrab_umi.episode import STREAMS, list_episodes

from . import replay


def judge(root: Path, answer: dict) -> dict:
    """One take's verdict from its poses and its replay: the measures, what
    failed and what is only worth a look - each hand judged, and the take
    failing on either."""
    if answer.get("kind") == "rollout":
        out = _rollout(answer)
        out["fails"] += recording(root)
        return out
    poses = replay._lines(root / "poses.jsonl")
    # Tracking is counted over the head frames, at the rate the top camera
    # kept: the replay's own rate is the dataset's.
    fps = rate_of(np.array([p["host_ts"] for p in poses])) if len(poses) > 1 else float(answer["fps"])
    per, fails, warns = {}, recording(root), []
    for hand in ARMS:
        measures, failed, warned = _hand(poses, answer, hand, fps)
        per[hand] = measures
        fails += [f"{hand}: {f}" for f in failed]
        warns += [f"{hand}: {w}" for w in warned]
    # A jaw that never moved is pointed out, never failed: a hand may only
    # steady, push or wait, and a jaw read short of its travel is no reason to
    # throw the take away.
    still = [hand for hand in ARMS if per[hand]["jaw_spread"] < UMI.keep_min_jaw_spread]
    if len(still) == len(ARMS):
        warns.append("no jaw moved")
    else:
        warns += [f"{hand}: jaw did not move" for hand in still]
    # Convert's own test that the desk solid was seen with the hands, their
    # poses being read against it: a take it leaves out is none to keep. A
    # hand never put on the desk has failed already.
    if not answer.get("unplaced"):
        try:
            cameras_on_desk(poses, ARMS)
        except ConvertError:
            fails.append("desk never seen with the hands")
    if UMI.keep_min_retract_mm > 0:
        meta = _meta(root)
        for hand in ARMS:
            moved = retract(poses, meta, hand)
            per[hand]["retract"] = moved
            if moved and moved["opened"] is False and UMI.keep_release_opens:
                fails.append(f"{hand}: let go without opening the jaw - it only shut")
            if moved and min(moved["lift_mm"], moved["back_mm"]) < UMI.keep_min_retract_mm:
                # A hand that went back to the ready pose has drawn off what it
                # put down however it got there: back to where takes start.
                home = end_from_ready(poses, meta, hand, answer)
                per[hand]["end_from_ready"] = home
                if home is None or home["mm"] > UMI.ready_guide_mm or home["deg"] > UMI.ready_guide_deg:
                    fails.append(f"{hand}: no retract after letting go - lift {moved['lift_mm']:.0f} mm, "
                                 f"back {moved['back_mm']:.0f} mm, and not back at ready")
    feasible = [per[hand]["feasible"] for hand in ARMS]
    return {"hands": per, "lost_s": max(per[h]["lost_s"] for h in ARMS),
            "held_s": min(per[h]["held_s"] for h in ARMS),
            "feasible": None if None in feasible else min(feasible),
            "flags": dict(sum((collections.Counter(per[h]["flags"]) for h in ARMS), collections.Counter())),
            "jaw_read": min(per[h]["jaw_read"] for h in ARMS),
            "jaw_spread": {h: per[h]["jaw_spread"] for h in ARMS}, "fails": fails, "warns": warns}


def _hand(poses: list[dict], answer: dict, hand: str, fps: float) -> tuple[dict, list[str], list[str]]:
    """One hand of a take: its measures, what failed, what is worth a look."""
    frames = len(poses)
    # Tracking, as UMI counts it: the hand's solid solved. The jaw read is
    # counted apart, and only warned about.
    held = sum(1 for p in poses if (p.get("solids") or {}).get(f"handheld_{hand}", {}).get("solved")
               and p["solids"][f"handheld_{hand}"].get("pose") is not None)
    jaws = [p["jaw"][hand] for p in poses if (p.get("jaw") or {}).get(hand) is not None]
    lost_s = (frames - held) / fps
    held_s = held / fps
    # The hand's frames the replay placed it in. One it could not place
    # ("lost": solid unsolved, or only the jaw unread) is left out: the
    # tracking judges the first (lost_s), the jaw's reading warns of the
    # second (jaw_read); neither is the arm's.
    # A joint at its band is not a frame lost: the arm still lands on the
    # target, at the edge of its reach, and when that edge does harm - a
    # stretched elbow's singularity throwing a joint, the target left behind -
    # it shows as a jump or a miss of its own. So a limit alone is followed,
    # and pointed out.
    frames_of = [f for f in answer["trajectory"] if "lost" not in (f["flags"].get(hand) or [])]
    judged = [f["flags"].get(hand) or [] for f in frames_of]
    unjudged = answer.get("unjudged")
    if unjudged:
        feasible = None
    elif hand in answer.get("hands", []):
        feasible = sum(1 for flags in judged if not set(flags) - {"limit"}) / max(1, len(judged))
    else:
        feasible = 0.0   # a hand the replay never placed followed nothing
    jaw_read = len(jaws) / max(1, frames)
    jaw_spread = (max(jaws) - min(jaws)) if jaws else 0.0
    # What held the arm back, over the same frames.
    flags = dict(collections.Counter(flag for frame in judged for flag in frame))
    fails, warns = [], []
    if hand in (answer.get("unplaced") or {}):
        fails.append(f"hand not on the desk - {answer['unplaced'][hand]}")
    if lost_s > UMI.keep_max_lost_s:
        fails.append(f"lost {lost_s:.2f} s")
    # How far round the arm swings from ready to follow its hand, over the
    # frames it followed.
    pan = [abs(np.degrees(f["joints"][f"{hand}_shoulder_pan"]) - READY_DEG[hand]["shoulder_pan"])
           for f in frames_of if f"{hand}_shoulder_pan" in f["joints"]]
    swung = max(pan, default=0.0)
    if hand in answer.get("hands", []) and swung > UMI.keep_warn_pan_from_ready_deg:
        warns.append(f"shoulder_pan {swung:.0f} deg from ready")
    # The fastest any joint turns between two moments the arm followed: past
    # what the servos turn, the arm cannot do what the take asks there - a
    # flip to another shape of the arm, as a reach runs out - and the gate
    # smears it into something else.
    # The replay follows at that speed at most, so an arm only keeping up
    # with a quick hand turns at it exactly; past it is a jump the follower
    # did not make itself.
    fastest = fastest_joint(answer["trajectory"], hand) if hand in answer.get("hands", []) else None
    if fastest is not None and fastest[1] > CORE.max_joint_speed_dps * SPEED_SLACK:
        fails.append(f"{fastest[0].removeprefix(hand + '_')} {fastest[1]:.0f} deg/s, "
                     f"past the arm's {CORE.max_joint_speed_dps:.0f}")
    if held_s < UMI.keep_min_held_s:
        fails.append(f"held {held_s:.1f} s")
    if unjudged:
        warns.append(f"arm not judged - {unjudged}")
    elif feasible < UMI.keep_min_feasible:
        held_back = {k: n for k, n in flags.items() if k != "limit"}
        worst = max(held_back, key=held_back.get) if held_back else ""
        fails.append(f"arm follows {feasible:.0%}" + (f" ({worst})" if worst else ""))
    at_edge = collections.Counter(j for f in frames_of for j in (f.get("at_limit") or {}).get(hand, []))
    if at_edge:
        joint, _ = at_edge.most_common(1)[0]
        warns.append(f"reach edge {flags.get('limit', 0) / max(1, len(judged)):.0%} "
                     f"({joint.removeprefix(hand + '_')} at its limit)")
    if jaw_read < UMI.keep_warn_jaw_read:
        warns.append(f"jaw read {jaw_read:.0%}")
    return ({"lost_s": round(lost_s, 2), "held_s": round(held_s, 1), "pan_from_ready_deg": round(swung, 1),
             "fastest_dps": None if fastest is None else round(fastest[1]),
             "feasible": None if feasible is None else round(feasible, 3), "flags": flags,
             "jaw_read": round(jaw_read, 3), "jaw_spread": round(jaw_spread, 3)}, fails, warns)


def end_from_ready(poses: list[dict], meta: dict, hand: str, answer: dict) -> dict | None:
    """How far a hand ended from the ready pose - its last placed frame's tool
    against the ready pose's, carried onto the desk solid as the take's replay
    carries it (ready_guide) - in mm and degrees, or None for a take with no
    place in the robot's world or a hand never placed."""
    if answer.get("desk") is None:
        return None
    try:
        world_T_cams = cameras_on_desk(poses, ARMS)
        solid_T_tcp = np.array(UMI.recorded_solid_T_tcp(f"handheld_{hand}", meta), dtype=float)
    except (ConvertError, KeyError):
        return None
    last = next(((world_T_cam, p) for p, world_T_cam in zip(reversed(poses), reversed(world_T_cams))
                 if world_T_cam is not None and placed(p, hand)), None)
    if last is None:
        return None
    world_T_cam, p = last
    chain = replay._chain(hand)
    ready = desk_tool(hand, np.array(answer["desk"], dtype=float),
                      chain.forward(np.array(ready_joints(hand, chain.joint_names), dtype=float)))
    mm, deg = apart(world_T_cam @ np.array(p["solids"][f"handheld_{hand}"]["pose"], dtype=float) @ solid_T_tcp, ready)
    return {"mm": round(mm, 1), "deg": round(deg, 1)}


def fastest_joint(frames: list[dict], hand: str) -> tuple[str, float] | None:
    """The joint of `hand`'s arm that turns fastest between two moments of a
    replay the arm followed both of, and how fast (deg/s), or None for a
    replay with no two such moments."""
    joints = None
    best = None
    for a, b in zip(frames, frames[1:]):
        if "lost" in (a["flags"].get(hand) or []) or "lost" in (b["flags"].get(hand) or []) or b["t"] <= a["t"]:
            continue
        joints = joints or [n for n in a["joints"] if n.startswith(f"{hand}_") and not n.endswith("_gripper")]
        name, turn = max(((n, abs(b["joints"][n] - a["joints"][n])) for n in joints), key=lambda nt: nt[1])
        speed = float(np.degrees(turn)) / (b["t"] - a["t"])
        if best is None or speed > best[1]:
            best = (name, speed)
    return best


# How far past the servos' speed a replay's joint has to turn to be more
# than an arm following at full speed: a hair, for rounding.
SPEED_SLACK = 1.01
# Every this many frames a video's picture is looked at for change.
PICTURE_EVERY = 10


def picture_change(path: Path) -> float | None:
    """The most a video's picture moves between pictures PICTURE_EVERY frames
    apart - the mean of the grey levels' change, shrunk to 64 x 40 - or None
    for a video with no picture in it."""
    cap = cv2.VideoCapture(str(path))
    try:
        most, last, n = 0.0, None, 0
        while cap.grab():
            if n % PICTURE_EVERY == 0:
                ok, picture = cap.retrieve()
                if ok:
                    grey = cv2.resize(cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY), (64, 40)).astype(float)
                    if last is not None:
                        most = max(most, float(np.abs(grey - last).mean()))
                    last = grey
            n += 1
        return None if last is None else most
    finally:
        cap.release()


def recording(root: Path) -> list[str]:
    """What failed of a take's cameras: a video missing or with no picture,
    one kept at too little of its rate or with a hole in it, one whose
    picture never changed. A demonstration has all three; a rollout, the
    ones it recorded. The top camera's pictures are not kept (its stream has
    no file): its frames are judged by their holes alone - asked for 60 a
    second it sends about 50, and what its poses lose is judged as lost
    tracking."""
    meta = _meta(root)
    streams = meta.get("streams") or {}
    frames = replay._lines(root / "frames.jsonl")
    wanted = STREAMS if meta.get("kind", "demo") == "demo" else tuple(streams)
    fails = []
    for name in wanted:
        stream = streams.get(name) or {}
        stamps = np.array(sorted(f["host_ts"] for f in frames if f["stream"] == name))
        if "file" in stream and stream["file"] is None:
            if len(stamps) < 2:
                fails.append(f"{name}: no frames")
            elif float(np.diff(stamps).max()) > UMI.keep_max_frame_gap_s:
                fails.append(f"{name}: no frame for {float(np.diff(stamps).max()):.2f} s")
            continue
        path = root / str(stream.get("file") or f"{name}.mp4")
        change = picture_change(path) if path.is_file() else None
        if len(stamps) < 2 or change is None:
            fails.append(f"{name}: no video")
            continue
        fps = float(stream.get("fps") or UMI.fps)
        rate = (len(stamps) - 1) / float(stamps[-1] - stamps[0])
        if rate < UMI.keep_min_stream_share * fps:
            fails.append(f"{name}: {rate:.0f} fps of {fps:.0f}")
        gap = float(np.diff(stamps).max())
        if gap > UMI.keep_max_frame_gap_s:
            fails.append(f"{name}: no frame for {gap:.2f} s")
        if change < UMI.keep_min_picture_change:
            fails.append(f"{name}: picture never changed")
    return fails


def _meta(root: Path) -> dict:
    try:
        return json.loads((root / "meta.json").read_text())
    except (OSError, ValueError):
        return {}


# A jaw holding something holds still: within this share of its travel over
# this many head frames - and carries it for longer than a jaw pausing on its
# way shut: this many at least.
HOLD_SPREAD = 0.1
HOLD_FRAMES = 6
CARRY_FRAMES = 15


def carries(jaws: np.ndarray) -> list[tuple[int, int, float]]:
    """Each stretch a jaw held something: (first frame, last frame, width).
    A carry is a width the jaw settles at after opening - shut onto an object
    from wider, and short of shut - held to the frame it leaves it, opening
    wider or shutting; the next one is shut onto from wider again. One held
    to the last frame ends there."""
    span = float(jaws.max() - jaws.min())
    half = (jaws.max() + jaws.min()) / 2
    tol = HOLD_SPREAD * span
    out = []
    widest = -np.inf
    i = int(np.argmax(jaws >= half))
    while i <= len(jaws) - HOLD_FRAMES:
        widest = max(widest, float(jaws[i]))
        held = jaws[i:i + HOLD_FRAMES]
        if np.ptp(held) <= tol and widest - 2 * tol > held[0] > jaws.min() + 2 * tol:
            j = i
            while j + 1 < len(jaws) and abs(jaws[j + 1] - jaws[i]) <= tol:
                j += 1
            if j - i + 1 >= CARRY_FRAMES:
                out.append((i, j, float(jaws[i])))
                widest = -np.inf
            i = j + 1
        else:
            i += 1
    return out


def let_go(jaws: np.ndarray) -> int | None:
    """The frame a hand last let go of what it carried, or None for none:
    where the jaw left the width of its last carry (carries) - unless it held
    on to the stop, which let nothing go. A jaw seen to carry nothing lets go
    at its last opening, the last time it crosses half way up its range - so
    a jaw shut again before the stop does not hide it."""
    held = carries(jaws)
    if held:
        last = held[-1][1]
        return last if last + 1 < len(jaws) else None
    half = (jaws.max() + jaws.min()) / 2
    opened = np.nonzero((jaws[:-1] < half) & (jaws[1:] >= half))[0]
    return int(opened[-1]) if len(opened) else None


def retract(poses: list[dict], meta: dict, hand: str) -> dict | None:
    """How far a hand rose and drew back after it let go - from the release
    (let_go) to the take's end, in the desk's frame - or None for a hand
    that let go of nothing (its jaw hardly moved) or that the take cannot
    place on the desk."""
    try:
        world_T_cams = cameras_on_desk(poses, ARMS)
        solid_T_tcp = np.array(UMI.recorded_solid_T_tcp(f"handheld_{hand}", meta), dtype=float)
    except (ConvertError, KeyError):
        return None
    track = [(world_T_cam @ np.array(p["solids"][f"handheld_{hand}"]["pose"], dtype=float) @ solid_T_tcp,
              float(p["jaw"][hand])) for p, world_T_cam in zip(poses, world_T_cams)
             if world_T_cam is not None and placed(p, hand)]
    jaws = np.array([jaw for _, jaw in track])
    if len(jaws) < 2 or jaws.max() - jaws.min() < UMI.keep_min_jaw_spread:
        return None
    at = let_go(jaws)
    if at is None:
        return None
    release = track[at][0]
    # Let go of by opening: the jaw goes wider than it carried at, as an arm
    # has to open its own to let go - one that only shuts, what it carried
    # having slipped from a hand gone slack, teaches an arm to carry it off
    # again. Not judged where no carry was seen (let_go fell back to the last
    # opening).
    held = [c for c in carries(jaws) if c[1] == at]
    opened = None if not held else bool(
        jaws[at + 1:].max(initial=-np.inf) > held[0][2] + HOLD_SPREAD * float(np.ptp(jaws)))
    after = np.array([T[:3, 3] for T, _ in track[at:]]) - release[:3, 3]
    lift = float(np.max(after[:, 2]))
    # The approach, laid flat on the desk; a tool pointing straight down has
    # none, and then any way off the desk is back.
    approach = release[:2, 2]
    if np.linalg.norm(approach) < 0.2:
        back = float(np.max(np.linalg.norm(after[:, :2], axis=1)))
    else:
        back = float(np.max(-(after[:, :2] @ (approach / np.linalg.norm(approach)))))
    return {"lift_mm": round(lift * 1000.0, 1), "back_mm": round(back * 1000.0, 1), "opened": opened}


def _rollout(answer: dict) -> dict:
    """A rollout's verdict, from the robot's readings its replay is made of:
    what convert learns from is the arms' joints, so the hands' solids going
    unseen is no hole in it - the arms only held still. It fails on readings
    missing. A run with too short a correction in it is only pointed out -
    convert takes nothing from it, and it is kept for what the policy did -
    and so is a correction whose jaws did not move, as one may put the reach
    right alone."""
    arms = answer["hands"]
    fps = float(answer["fps"])
    frames = answer["trajectory"]
    flags = dict(collections.Counter(flag for f in frames for flag in f["flags"].get(arms[0], [])))
    corrected = [f for f in frames if not f["flags"].get(arms[0])]
    held_s, lost_s = len(corrected) / fps, flags.get("lost", 0) / fps
    spread = {}
    for arm in arms:
        band = gripper_band(arm)
        jaws = [aperture_of(arm, f["joints"][f"{arm}_gripper"], band) for f in corrected]
        spread[arm] = round((max(jaws) - min(jaws)) if jaws else 0.0, 3)
    fails, warns = [], []
    if lost_s > UMI.keep_max_lost_s:
        fails.append(f"robot unread {lost_s:.2f} s")
    if held_s < UMI.keep_min_held_s:
        warns.append(f"corrected {held_s:.1f} s")
    elif all(s < UMI.keep_min_jaw_spread for s in spread.values()):
        warns.append("no jaw moved")
    return {"kind": "rollout", "lost_s": round(lost_s, 2), "held_s": round(held_s, 1),
            "feasible": 1.0, "flags": flags, "jaw_read": 1.0, "jaw_spread": spread,
            "fails": fails, "warns": warns}


class Verdicts:
    """Every complete take's verdict, the replays behind them solved on a
    thread as takes arrive, and each take's start set beside the others'."""

    def __init__(self, root: Path | None = None) -> None:
        self._root = root or UMI.episodes_dir
        self._lock = threading.Lock()
        # name -> (the take's poses', its meta.json's and the arm's dates it was judged at, verdict, start)
        self._kept: dict[str, tuple[tuple[float, float, float], dict, tuple | None]] = {}
        # name -> (the same, why it could not be replayed or judged): not tried
        # again until the take or the arm changes.
        self._failed: dict[str, tuple[tuple[float, float, float], str]] = {}
        self._solving: threading.Thread | None = None

    def _solve_rest(self, names: list[tuple[str, tuple[float, float, float]]]) -> None:
        for name, stamp in names:
            try:
                replay.solved(self._root / name)
            except Exception as exc:   # noqa: BLE001 - one bad take must not stop the rest
                print(f"  verdict: {name} could not be replayed - {type(exc).__name__}: {exc}")
                with self._lock:
                    self._failed[name] = (stamp, f"replay failed: {type(exc).__name__}: {exc}")

    def get(self) -> dict:
        """{"verdicts": {name: verdict}, "pending": [names whose replay is still being solved]}."""
        names = [e["name"] for e in list_episodes(self._root) if e.get("complete")]
        pending = []
        # A take is judged again when it changes, and all of them when the arm
        # does; only what stands for each take as it is now is shown.
        model = replay.model_mtime()
        # A take's stamp is the dates its replay was solved from - the take's
        # poses, its meta.json (where the desk solid stood, the hand sizes)
        # and the arm's - compared as they are, as replay.cached does.
        current: dict[str, tuple[float, float, float]] = {}
        for name in names:
            root = self._root / name
            try:
                stamp = ((root / "poses.jsonl").stat().st_mtime, (root / "meta.json").stat().st_mtime, model)
            except OSError:
                continue   # deleted meanwhile, or written without poses: nothing to judge
            current[name] = stamp
            with self._lock:
                kept = self._kept.get(name)
                failed = self._failed.get(name)
            if (kept is not None and kept[0] == stamp) or (failed is not None and failed[0] == stamp):
                continue
            try:
                answer = replay.cached(root, model)
            except OSError:
                continue
            if answer is None:
                pending.append((name, stamp))
                continue
            try:
                verdict = judge(root, answer)
            except Exception as exc:   # noqa: BLE001 - one bad take must not empty the shelf's column
                with self._lock:
                    self._failed[name] = (stamp, f"could not be judged: {type(exc).__name__}: {exc}")
                continue
            # The start is only ever pointed out: one that cannot be had leaves
            # the take out of the comparison, not failed.
            starts = {}
            for hand in ARMS:
                try:
                    starts[hand] = replay._hand_start(root, hand)
                except Exception:   # noqa: BLE001
                    starts[hand] = None
            with self._lock:
                self._kept[name] = (stamp, verdict, starts)
                self._failed.pop(name, None)
        with self._lock:
            for gone in set(self._kept) - set(names):
                del self._kept[gone]
            for gone in set(self._failed) - set(names):
                del self._failed[gone]
            if pending and (self._solving is None or not self._solving.is_alive()):
                self._solving = threading.Thread(target=self._solve_rest, args=(pending,),
                                                 name="verdict-replay", daemon=True)
                self._solving.start()
            kept = {name: k for name, k in self._kept.items() if current.get(name) == k[0]}
            failed = {name: why for name, (stamp, why) in self._failed.items() if current.get(name) == stamp}
        verdicts = self._with_starts(kept)
        # A take that could not be replayed or judged says why, as an error
        # and not a fail: a broken description or solver is no fault of the
        # take, and tick failing must not offer it for deletion.
        for name, why in failed.items():
            verdicts[name] = {"lost_s": 0.0, "held_s": 0.0, "feasible": None, "flags": {},
                              "jaw_read": 0.0, "jaw_spread": 0.0, "fails": [], "warns": [], "error": why}
        return {"verdicts": verdicts, "pending": [name for name, _ in pending]}

    @staticmethod
    def _with_starts(kept: dict) -> dict:
        """Each verdict, with each hand's start set beside the mean start of
        that hand over the takes - pointed out, never failed on: varied
        starts are wanted."""
        out = {}
        by_hand: dict[str, list] = {}
        for name, (_, verdict, starts) in kept.items():
            for hand, start in starts.items():
                if start is not None:
                    by_hand.setdefault(hand, []).append(start)
        for name, (_, verdict, starts) in kept.items():
            verdict = dict(verdict, warns=list(verdict["warns"]))
            for hand, start in starts.items():
                others = by_hand.get(hand, [])
                if start is not None and len(others) >= 3:
                    apart = replay.start_apart(start, others)
                    if apart["far"]:
                        verdict["warns"].append(f"{hand}: start {apart['deg']:.0f} deg / {apart['mm']:+.0f} mm "
                                                "from the others")
            out[name] = verdict
        return out
