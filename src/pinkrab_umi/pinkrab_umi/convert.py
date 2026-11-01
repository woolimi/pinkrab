"""`pinkrab-umi convert` - raw episodes into a LeRobot dataset.

What the policy learns from is not what was recorded. The recorder keeps each
hand's pose in the desk's frame, frame by frame; the policy is trained on
what the grippers SAW and how they MOVED from there - each a delta in its own
frame, which is the same whichever corner of the desk the block was on - and,
with two hands, where one hand's tool is in the other's (UMI's
inter-gripper proprioception), which is what lets one hand act knowing where
the other is. A dataset is of the hands its takes were made with, both on
the rig; every frame has them all. So each episode goes through these steps:

  1. the camera on the desk: the top camera is worn on the operator's head
     and moves with it, so each frame is placed by its own desk solve, and a
     frame that has none has no hand;
  2. the source clock: the top camera's exposures - its sensor stamps,
     already on this host's clock - are when each pose was. Each gripper
     video and each jaw are laid on them by their own host stamps less the
     latency each reading is stamped with (a jaw's, the time it was read);
  3. the cut: a frame where any hand's solid was not solved, or with no
     picture from any hand's gripper taken within `GRIP_MATCH_FRAMES` of a
     frame period before it, is a hole. One no longer than
     `convert_max_hole_s` is bridged - each pose interpolated across it from
     the frames either side, each picture the last one taken - and a longer
     one ends the segment; a segment shorter than `min_segment_s` is
     dropped;
  4. the tool centres: each hand's pose, the solid's and solid_T_tcp composed,
     then smoothed without lag - Savitzky-Golay over `SMOOTH_S`, on the
     position and on the unit quaternion - because the tracker's noise from
     one frame to the next is about half of a step's motion;
  4c. the dataset's clock: a frame every 1/fps s from the segment's first
     exposure, whatever the cameras' own rates (`fps`, up to the gate's
     policy_shaping_hz). Each tool's place is read off a cubic spline through
     its smoothed poses and its turn off a rotation spline, each jaw is
     interpolated, and each gripper's picture is the last one taken by then -
     as a policy at rollout sees the last picture there is. Faster than the
     cameras, a picture stands for more than one frame; the poses never do,
     so no step is a still one;
  4b. the still ends: frames before any hand first moves - `STILL_MM` from
     where it began, or its jaw by `STILL_APERTURE` - and after the last one
     last moves are left out. A take begins with the hands at rest while the
     pedal goes down; a policy shown that rest at the start learns to rest
     at the start, and at rollout never leaves it;
  5. camera-local deltas, per hand: R_t^T (p_{t+1} - p_t) and R_t^T R_{t+1},
     the latter as the first two columns of the matrix (6D, continuous);
  6. despike: a delta faster than `DESPIKE_MM_S` or `filter_max_rot_speed_dps` over its
     frame is a tracking glitch, not a hand, and becomes zero;
  7. each gripper's action is the NEXT frame's measured opening, 0..1: the
     width is replayed as the hand made it, never projected to shut - a jaw
     driven all the way closed crushes what a hand holds gently;
  8. the state: each gripper's opening and, with two hands, the right
     tool's pose in the left tool's frame - never an absolute position, and
     never the last step: a policy shown its own last step answers with it
     again and stops looking at the picture;
  9. the absolute poses ride along as extra features (`pose.tcp_<hand>`:
     tool centre in the desk frame, xyz + quaternion, off the same smoothed
     trajectory) - named outside `observation.` on purpose, so no policy
     takes the desk's coordinates as an input; it is what a replay or a
     start-pose search needs, and LeRobot has no hook to add it later. Its
     z is over the solid's origin, which stood as high above the desk as
     the take records - the dataset's account keeps that, the solid's size
     and which way it faced, per episode;
 10. the state's spread in the dataset's statistics, which a policy is
     normalised by - its std and the width between its min and max and between
     each pair of quantiles - is no narrower than `STATE_STD_FLOOR`.

A rollout (a policy's run, the handhelds taking the arms over to put them
right - DAgger) goes through the same steps from the robot's side instead:
the arms' cameras - the pictures the policy is shown - are the image, the
first hand's camera the clock, each tool's pose the arm's forward kinematics
at each exposure (its joints, read off the robot, interpolated), the
openings the arms' own jaws, and only the corrections - the readings whose
`controller` is `human`, the hands driving the arms - are kept, but for a
correction thrown away and made again (a `redo` in events.jsonl). One recorded
on an arm as it was built before - its meta.json names the description its
angles are of - is skipped. The poses are turned into the handheld's tool
convention (TOOL_TURN, which rollout turns back), so a correction's deltas
mean what a demonstration's do; its `pose.tcp_<hand>` is the arm's tool in
the robot's base frame, there being no desk-frame pose of an arm.

Every episode's task is the command given with --task.

    pinkrab-umi convert 20260920-161346 20260920-161415 --out blocks-pilot --task "pick and place the block"
    pinkrab-umi convert 20260920-161346 --out blocks-50hz --task "pick and place the block" --fps 50
"""
from __future__ import annotations

import itertools
import json
import math
import shutil
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, RotationSpline

from pinkrab_core.config import ARMS
from pinkrab_core.config import SETTINGS as CORE
from pinkrab_core.dataset import (GRIPPER_WIDTH, NO_STEP, POSE_NAMES, STEP_NAMES,
                                  LensMissing, action_names, check_fps, delta_of, delta_to_matrix, gripper_crop,
                                  gripper_shape, gripper_view, image_key, pose_key, relative_pose, require_lens,
                                  state_names, state_vector)

from .backgrounds import painted, pattern, seed
from .backgrounds import read as read_labels
from .config import SETTINGS
from .episode import ROBOT_GAP_S
from .trajectory import matrix_to_quat, quat_to_matrix

# About a second: thirty frames at 30 fps span 0.97 s.
MIN_SEGMENT_S = 0.95
# 80 mm in a frame at 30 fps. A turn is a glitch past the speed the stream
# filter takes for a flip (filter_max_rot_speed_dps), the one limit on how fast
# a hand may turn.
DESPIKE_MM_S = 2400.0
# A hand holding still still wanders a millimetre or two; past this it has
# set off.
STILL_MM = 4.0
STILL_APERTURE = 0.03
# Seven frames at 30 fps: a quarter second, short against a reach and long
# enough to bring the tracker's noise down about threefold.
SMOOTH_S = 7 / 30
SMOOTH_ORDER = 2
# A gripper picture taken more than this many of its frame periods before the
# moment it stands for is a hole in the video, not a late frame: the segment
# ends there.
GRIP_MATCH_FRAMES = 1.5
# Stamps this close are one moment: a picture stamped a hair after a frame on
# the dataset's clock - the same instant, added up another way - is its own.
SAME_INSTANT_S = 1e-4
DATASET_GOP = 30
# The narrowest spread a state number is normalised by: a hundredth of a jaw's
# band, 10 mm of the relative pose. A number the takes never vary - the jaw of
# a hand at rest - has a spread of nearly nothing, and LeRobot divides by it:
# an arm's reading a hair off that value is a state hundreds of spreads out,
# and the policy answers it with the same step whatever it sees.
STATE_STD_FLOOR = 0.01
# A rollout's joints are a pose only on the arm they were read from.
OTHER_BUILD = ("recorded on the arm built another way - its joint angles are not "
               "this description's")


class ConvertError(RuntimeError):
    pass


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def learned_from(readings: list[dict], events: list[dict]) -> tuple[np.ndarray, int]:
    """Which of a rollout's readings its dataset learns from: the hands driving
    the arms (`controller` human), in a correction not thrown away and made
    again (a `redo` event). And how many were redone."""
    redone = {e.get("correction") for e in events if e.get("event") == "redo"}
    return np.array([r.get("controller") == "human" and r.get("correction") not in redone
                     for r in readings], dtype=bool), len(redone)


def latest(stamps: np.ndarray, times: np.ndarray) -> np.ndarray:
    """For each of `times`, the index of the last of the sorted `stamps` at or
    before it - -1 where none is."""
    return np.searchsorted(stamps, np.asarray(times) + SAME_INSTANT_S, side="right") - 1


def fresh(stamps: np.ndarray, times: np.ndarray, period: float) -> tuple[np.ndarray, np.ndarray]:
    """`latest`, and whether each is a picture of that moment: taken no more
    than GRIP_MATCH_FRAMES of its camera's `period` before it."""
    index = latest(stamps, times)
    ok = index >= 0
    ok[ok] = np.asarray(times)[ok] - stamps[index[ok]] <= GRIP_MATCH_FRAMES * period
    return index, ok


def grid(start: float, end: float, fps: int) -> np.ndarray:
    """The dataset's clock over [start, end]: a frame every 1/fps s from start."""
    # A thousandth of a frame's slack: stamps a day's worth of seconds from
    # zero carry rounding of that order, and a frame at the very end is one.
    return start + np.arange(math.floor((end - start) * fps + 1e-3) + 1) / fps


class _Video:
    """Frames of one mp4 by index, read in order, held one at a time, each as
    the dataset keeps it: `gripper_view` at `size` about `centre` - and with
    `labels` (keep_masks, frame by frame) and a `backdrop`, everything they
    do not keep painted over with it (backgrounds.painted)."""

    def __init__(self, path: Path, size: tuple[int, int], centre: tuple[float, float],
                 labels: np.ndarray | None = None, backdrop: np.ndarray | None = None) -> None:
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise ConvertError(f"cannot open {path}")
        self.size = size
        self.centre = centre
        self.labels = labels
        self.backdrop = backdrop
        self.next_index = 0
        self.last = None

    def frame(self, index: int) -> np.ndarray:
        if index < self.next_index - 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            self.next_index = index
        while self.next_index <= index:
            ok, picture = self.cap.read()
            if not ok:
                raise ConvertError(f"the video has {self.next_index} frames, and its frame list names frame {index}")
            self.last = picture
            self.next_index += 1
        view = gripper_view(self.last, self.size, self.centre)
        if self.labels is not None:
            view = painted(view, self.labels[index], self.backdrop)
        return cv2.cvtColor(view, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        self.cap.release()


def open_videos(src: Path, hands, size: tuple[int, int], centre: dict, labels: dict | None = None,
                backdrop: np.ndarray | None = None) -> dict[str, _Video]:
    """Each hand's gripper video of the take at `src`, opened - or ConvertError
    for the first that will not open, with the others let go."""
    videos: dict[str, _Video] = {}
    try:
        for hand in hands:
            videos[hand] = _Video(src / f"gripper_{hand}.mp4", size, tuple(centre[hand]),
                                  (labels or {}).get(hand), backdrop)
    except ConvertError:
        for video in videos.values():
            video.close()
        raise
    return videos


def placed(pose: dict, hand: str) -> bool:
    """Whether this frame has the hand: its solid solved, and its jaw read."""
    solid = pose["solids"].get(f"handheld_{hand}") or {}
    return bool(solid.get("solved")) and solid.get("pose") is not None \
        and (pose.get("jaw") or {}).get(hand) is not None


def jaw_readings(poses: list[dict], hand: str) -> tuple[np.ndarray, np.ndarray]:
    """One hand's jaw readings over a take - when each was taken, on this
    host's clock less the latency it is stamped with, and its opening - each
    reading once, however many frames carried it. A line that names no read
    time (`jaw_ts`) is read at its own."""
    seen = {}
    for p in poses:
        opening = (p.get("jaw") or {}).get(hand)
        if opening is not None:
            taken = (p.get("jaw_ts") or {}).get(hand)
            seen[p["host_ts"] if taken is None else taken] = float(opening)
    stamps = sorted(seen)
    return (np.array(stamps, dtype=float) - SETTINGS.jaw_latency_s,
            np.array([seen[t] for t in stamps], dtype=float))


def bridged(present: np.ndarray, times: np.ndarray, max_s: float) -> np.ndarray:
    """`present` with each run of frames missing from it filled in where the
    frames either side are present and no more than `max_s` plus a frame
    period apart, the later stamped after the earlier: a hole of `max_s` at
    most."""
    out = present.copy()
    period = 1.0 / rate_of(times) if len(times) > 1 else 0.0
    have = np.nonzero(present)[0]
    for a, b in itertools.pairwise(have):
        if b - a > 1 and 0.0 < times[b] - times[a] <= max_s + period:
            out[a + 1:b] = True
    return out


def fill_holes(tools: list[np.ndarray | None], times: np.ndarray, max_s: float) -> list[np.ndarray | None]:
    """A hand's tool poses with each hole of `max_s` at most (bridged) filled
    in: the place straight and the turn the shortest way between the frames
    either side, by time."""
    present = np.array([T is not None for T in tools])
    out = list(tools)
    filled = bridged(present, times, max_s) & ~present
    for i in np.nonzero(filled)[0]:
        a = max(j for j in range(i) if present[j])
        b = min(j for j in range(i + 1, len(tools)) if present[j])
        w = min(max((times[i] - times[a]) / (times[b] - times[a]), 0.0), 1.0)
        T = np.eye(4)
        T[:3, 3] = tools[a][:3, 3] + w * (tools[b][:3, 3] - tools[a][:3, 3])
        turn = Rotation.from_matrix([tools[a][:3, :3], tools[b][:3, :3]])
        T[:3, :3] = (turn[0] * Rotation.from_rotvec(w * (turn[0].inv() * turn[1]).as_rotvec())).as_matrix()
        out[i] = T
    return out


def segments(usable: np.ndarray, times: np.ndarray, min_s: float) -> list[tuple[int, int]]:
    """Runs of `usable` frames [start, end) - every hand there to be had and
    the frame ready otherwise - `min_s` long at least. A frame stamped no
    later than the one before it starts another run."""
    out = []
    start = None
    for i in range(len(usable) + 1):
        ok = i < len(usable) and bool(usable[i])
        if start is not None and (not ok or times[i] <= times[i - 1]):
            if times[i - 1] - times[start] >= min_s:
                out.append((start, i))
            start = None
        if ok and start is None:
            start = i
    return out


def on_grid(tools: list[np.ndarray | None], times: np.ndarray, at: np.ndarray, rate: float,
            max_hole_s: float) -> list[np.ndarray | None]:
    """A hand's tool poses at the head frames' `times` read at the moments `at`
    of the dataset's clock, as a segment of a take is: each hole of `max_hole_s`
    at most bridged (fill_holes), each run of frames the hand was placed in
    smoothed (smooth) and splined through (resample), and read at the moments
    inside it. A moment outside every run - the hand lost for longer, or a run
    of a single frame - is None."""
    filled = fill_holes(tools, times, max_hole_s)
    out: list[np.ndarray | None] = [None] * len(at)
    for start, end in segments(np.array([T is not None for T in filled]), times, 0.0):
        if end - start < 2:
            continue
        t = times[start:end]
        inside = np.nonzero((at >= t[0]) & (at <= t[-1]))[0]
        if len(inside):
            # Splined from the run's start: absolute stamps lose digits.
            for k, T in zip(inside, resample(smooth(np.array(filled[start:end]), rate), t - t[0],
                                             at[inside] - t[0])):
                out[k] = T
    return out


def cameras_on_desk(poses: list[dict], hands) -> list[np.ndarray | None]:
    """world_T_cam for each frame of the episode, off that frame's desk
    solve - None where the desk solid was not seen with a hand."""
    out = [next((np.array(c["world_pose"]) @ np.linalg.inv(np.array(c["pose"]))
                 for c in ((p["solids"].get(f"handheld_{hand}") or {}) for hand in hands)
                 if c.get("world_pose") is not None and c.get("pose") is not None), None)
           for p in poses]
    if all(W is None for W in out):
        raise ConvertError("the desk solid was never seen with a hand")
    return out


def exposures(poses: list[dict]) -> np.ndarray:
    """When each head frame was exposed, on this host's clock."""
    stamps = np.array([p["stamp"] for p in poses], dtype=float)
    lag = float(np.median(np.array([p["host_ts"] for p in poses]) - stamps))
    if not 0.0 <= lag < 1.0:
        raise ConvertError(f"the top camera's stamps are not on this host's clock ({lag:+.2f} s)")
    return stamps


def smooth_frames(rate: float) -> int:
    """SMOOTH_S at `rate` frames a second, as the odd window the filter takes."""
    n = max(SMOOTH_ORDER + 1, round(SMOOTH_S * rate))
    return n if n % 2 else n + 1


def continuous_quats(R: np.ndarray) -> np.ndarray:
    """Rotations (n,3,3) as unit quaternions (n,4), each on the side of the
    sphere nearest the one before: q and -q are one rotation, and kept on one
    side neighbours average and interpolate as turns."""
    q = np.array([matrix_to_quat(r) for r in R])
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    return q


def smooth(T: np.ndarray, rate: float) -> np.ndarray:
    """A segment of tool poses (n,4,4) `rate` frames a second, smoothed without lag."""
    frames = smooth_frames(rate)
    if len(T) < frames:
        return T
    out = T.copy()
    out[:, :3, 3] = savgol_filter(T[:, :3, 3], frames, SMOOTH_ORDER, axis=0)
    q = savgol_filter(continuous_quats(T[:, :3, :3]), frames, SMOOTH_ORDER, axis=0)
    out[:, :3, :3] = np.array([quat_to_matrix(x) for x in q])
    return out


def resample(T: np.ndarray, times: np.ndarray, at: np.ndarray) -> np.ndarray:
    """Tool poses (n,4,4) at `times`, read at the times `at`: the place off a
    cubic spline through them, the turn off a rotation spline - smooth through
    every source frame, so a step on the dataset's clock is the hand's own
    motion over it, never a stair of the cameras' frames."""
    out = np.tile(np.eye(4), (len(at), 1, 1))
    out[:, :3, 3] = CubicSpline(times, T[:, :3, 3], axis=0)(at)
    out[:, :3, :3] = RotationSpline(times, Rotation.from_matrix(T[:, :3, :3]))(at).as_matrix()
    return out


def rate_of(times: np.ndarray) -> float:
    """How many frames a second `times` were taken at."""
    return 1.0 / float(np.median(np.diff(times)))


def moving(Ts: dict[str, np.ndarray], apertures: dict[str, np.ndarray]) -> tuple[int, int]:
    """The part of a segment [first, last) from the frame before any hand
    first moves to the frame after the last one last moves."""
    n = len(next(iter(Ts.values())))

    def away(i: int, j: int) -> bool:
        return any(np.linalg.norm(T[i, :3, 3] - T[j, :3, 3]) * 1000.0 > STILL_MM
                   or abs(float(apertures[hand][i]) - float(apertures[hand][j])) > STILL_APERTURE
                   for hand, T in Ts.items())

    first = next((i for i in range(n) if away(i, 0)), n)
    last = next((i for i in range(n - 1, -1, -1) if away(i, n - 1)), -1)
    return max(0, first - 1), min(n, last + 2)


def deltas(T: np.ndarray, aperture: np.ndarray, fps: int):
    """One hand's actions for a segment of its tool poses (n,4,4) `fps` frames
    a second: (n,10), plus how many were despiked."""
    n = len(T)
    action = np.zeros((n, 10), dtype=np.float32)
    despiked = 0
    for t in range(n - 1):
        step = delta_of(T[t], T[t + 1])
        dR = delta_to_matrix(step)[:3, :3]
        angle = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(dR) - 1.0) / 2.0))))
        if np.linalg.norm(step[:3]) * 1000.0 * fps > DESPIKE_MM_S or angle * fps > SETTINGS.filter_max_rot_speed_dps:
            step = NO_STEP
            despiked += 1
        action[t, :9] = step
        action[t, 9] = aperture[t + 1]
    action[n - 1, :9] = NO_STEP
    action[n - 1, 9] = aperture[n - 1]
    return action, despiked


def gripper_size(episodes: list[str], source: Path, hands) -> tuple[int, int]:
    """The dataset's gripper picture (height, width), from the first episode
    that recorded every hand's gripper video: every picture in a dataset has
    one shape, every hand's and every episode's. A take missing a hand's
    video says nothing of it, and is skipped when it is converted."""
    for name in episodes:
        meta_file = source / name / "meta.json"
        if not meta_file.is_file():
            continue
        streams = json.loads(meta_file.read_text()).get("streams") or {}
        sizes = [tuple((streams.get(f"gripper_{hand}") or {}).get("size") or ()) for hand in hands]
        if not all(sizes):
            continue
        if len(set(sizes)) > 1:
            raise ConvertError(f"{name}'s gripper cameras record frames of different sizes: {sorted(set(sizes))}")
        return gripper_shape(*sizes[0])
    raise ConvertError("no episode recorded every hand's gripper video")


def _frames(segment: dict[str, dict], state_names: list[str], task: str, pictures, fps: int):
    """The dataset's frames for one segment, one at a time: `segment` is each
    hand's smoothed poses (`T`) and openings (`aperture`) `fps` frames a
    second; `pictures(hand, k)` its k-th gripper picture, read as its frame
    is. Each hand's despiked count is left in `segment`."""
    hands = list(segment)
    actions, quats = {}, {}
    for hand in hands:
        actions[hand], segment[hand]["despiked"] = deltas(segment[hand]["T"], segment[hand]["aperture"], fps)
        quats[hand] = continuous_quats(segment[hand]["T"][:, :3, :3])
    for k in range(len(actions[hands[0]])):
        relative = (relative_pose(segment[hands[0]]["T"][k], segment[hands[1]]["T"][k])
                    if len(hands) == 2 else None)
        frame = {
            "observation.state": state_vector(state_names, {h: float(segment[h]["aperture"][k]) for h in hands},
                                              relative),
            "action": np.concatenate([actions[hand][k] for hand in hands]),
            "task": task,
        }
        for hand in hands:
            T = segment[hand]["T"][k]
            frame[image_key(hand)] = pictures(hand, k)
            frame[pose_key(hand)] = np.concatenate([T[:3, 3], quats[hand][k]]).astype(np.float32)
        yield frame


def convert(episodes: list[str], out: str, task: str, hands: tuple[str, ...] = ARMS,
            min_segment_s: float = MIN_SEGMENT_S, say=print,
            source: Path | None = None, gop: int = DATASET_GOP,
            fps: int | None = None, randomised: bool = False) -> dict:
    """`hands` are the hands every take worked with - both on the rig; the
    simulation's one. `source` is where the raw episodes are, the
    workspace's episodes/ by default. `gop` is the dataset video's keyframe
    spacing: training reads frames at random, and each one costs a decode
    from the keyframe before it - 2 trains several times faster than 30 at a
    few times the size. `fps` is the dataset's rate, the core settings'
    dataset_fps unless said - and so the rate a policy trained on it is
    asked at. `randomised` paints everything a take's keep labels do not keep
    (keep_masks) over with a pattern of the take's own (backgrounds): a take
    without labels is left out."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = CORE.workspace / "datasets" / out
    if root.exists():
        raise ConvertError(f"{root} exists - pick another --out or remove it")
    hands = tuple(hand for hand in ARMS if hand in hands)
    if not hands:
        raise ConvertError(f"hands are some of {', '.join(ARMS)}")
    source = source or SETTINGS.episodes_dir
    try:
        fps = check_fps(CORE.dataset_fps if fps is None else fps)
    except ValueError as exc:
        raise ConvertError(str(exc)) from None
    size = gripper_size(episodes, source, hands)
    names = state_names(hands)
    # The grippers' pictures are the only ones the policy is shown. The top
    # camera is the clock and the source of the labels - it is where the
    # solids' poses, and so every delta, come from - and it is not on the
    # arms: at rollout there is no top camera to ask, only the desk with no
    # markers on it. An observation it cannot have is not an observation.
    features = {image_key(hand): {"dtype": "video", "shape": (*size, 3), "names": ["height", "width", "channels"]}
                for hand in hands}
    features.update({
        "observation.state": {"dtype": "float32", "shape": (len(names),), "names": names},
        **{pose_key(hand): {"dtype": "float32", "shape": (7,), "names": POSE_NAMES} for hand in hands},
        "action": {"dtype": "float32", "shape": (10 * len(hands),), "names": action_names(hands)},
    })
    from lerobot.configs.video import RGBEncoderConfig

    # LeRobot's default puts a keyframe every second frame, which is most of
    # the size: a keyframe a second is a tenth of the bytes and still seeks
    # well enough for training batches. Frames go straight to the encoder:
    # by default each is written out as a PNG, read back to encode and read
    # again for the episode's statistics, and that round trip was nine
    # tenths of a conversion (three takes, 653 frames: 13.8 s, streamed 2.5 s,
    # the same video). The encoder's queue is unbounded (maxsize 0): bounded,
    # a frame that waits on a full queue for 0.1 s is dropped with a warning,
    # which while recording keeps a camera live and here would leave a video
    # a frame short of its rows.
    dataset = LeRobotDataset.create(f"pinkrab/{out}", fps=fps, features=features, root=root,
                                    robot_type="pinkrab_umi", use_videos=True,
                                    rgb_encoder=RGBEncoderConfig(g=gop), streaming_encoding=True,
                                    encoder_queue_maxsize=0)
    report = {"dataset": str(root), "task": task, "fps": fps, "source": str(source), "hands": list(hands),
              "gop": gop, "randomised": randomised, "episodes": []}
    started = time.monotonic()
    try:
        for name in episodes:
            report["episodes"].append(_episode(dataset, source / name, task, hands, fps, min_segment_s,
                                               names, size, say, background=seed(name) if randomised else None))
        if not any(e["segments"] for e in report["episodes"]):
            raise ConvertError("no episode had a segment long enough to keep")
        dataset.finalize()
        _floor_state_std(root)
    except BaseException:
        # Half a dataset is worse than none: it would list as one, and its
        # name would stand in the way of the run that makes it whole.
        shutil.rmtree(root, ignore_errors=True)
        raise
    report["seconds"] = round(time.monotonic() - started, 1)
    (root / "meta" / "pinkrab_convert.json").write_text(json.dumps({
        **report,
        "desk": {"note": "pose.tcp_<hand> is the tool centre in the desk solid's frame, z up - not an observation, so no "
                         "policy takes it as input; a rollout's is the arm's tool in the robot's base frame. Each "
                         "demonstration's own desk is the solid its take recorded: its face size, how far above the "
                         "desk its origin stood and which way it faced",
                 "camera": "each frame's own world_T_cam, off its desk solve; a frame without one is a hole"},
        "clock": {"grid": f"a frame every 1/{fps} s from each segment's first exposure",
                  "source": "top camera exposure (sensor stamp, host clock)",
                  "pose": "a cubic spline (place) and a rotation spline (turn) through the smoothed poses",
                  "picture": "the last one taken by the frame's time",
                  "gripper_camera_latency_s": SETTINGS.gripper_camera_latency_s,
                  "jaw_latency_s": SETTINGS.jaw_latency_s},
        "gripper": {"crop": list(gripper_crop()), "width": GRIPPER_WIDTH, "size": list(size),
                    "note": "a window of each frame about its camera's lens axis (each episode's centre, by hand), "
                            "then scaled"},
        "smoothing": {"savgol_s": round(SMOOTH_S, 4), "order": SMOOTH_ORDER,
                      "note": "on the source poses, before they are laid on the dataset's clock"},
        "still": {"mm": STILL_MM, "aperture": STILL_APERTURE,
                  "note": "frames before any hand first moves and after the last one last moves are left out"},
        "rollouts": {"clock": f"the {hands[0]} arm's camera exposures (host clock)",
                     "pose": "forward kinematics of the robot's readings, turned into the handheld's tool convention",
                     "robot_state_latency_s": SETTINGS.robot_state_latency_s, "gap_s": ROBOT_GAP_S,
                     "kept": "the corrections: frames whose readings say the hands drove the arms (controller human)"},
        "action": {"names": STEP_NAMES, "frame": "tool (camera-local): R_t^T (p_{t+1}-p_t), R_t^T R_{t+1} as two columns",
                   "despike_mm_s": DESPIKE_MM_S, "despike_dps": SETTINGS.filter_max_rot_speed_dps,
                   "gripper": "next frame's measured opening 0..1"},
        "relative": {"note": "with two hands: the right tool in the left tool's frame, as a step is written"},
        "state_std_floor": STATE_STD_FLOOR,
    }, indent=2))
    return report


# The statistics that bound a number from either side, widest pair first: a
# policy normalised by min/max or by quantiles divides by the width of one.
STATE_BOUNDS = (("min", "max"), ("q01", "q99"), ("q05", "q95"), ("q10", "q90"))


def floored(stats: dict, floor: float) -> dict:
    """A feature's statistics (as LeRobot writes them, a list per statistic)
    with no number's spread under `floor`, whichever normalisation reads them:
    the std at least `floor`, and each bounding pair at least `floor` apart,
    widened evenly about its middle."""
    out = dict(stats)
    out["std"] = [max(float(v), floor) for v in stats["std"]]
    for low, high in STATE_BOUNDS:
        if low in stats and high in stats:
            lows, highs = [], []
            for lo, hi in zip(stats[low], stats[high]):
                middle, half = (float(lo) + float(hi)) / 2, max(float(hi) - float(lo), floor) / 2
                lows.append(middle - half)
                highs.append(middle + half)
            out[low], out[high] = lows, highs
    return out


def _floor_state_std(root: Path) -> None:
    """The dataset's statistics with no state number's spread under
    `STATE_STD_FLOOR`."""
    path = root / "meta" / "stats.json"
    stats = json.loads(path.read_text())
    stats["observation.state"] = floored(stats["observation.state"], STATE_STD_FLOOR)
    path.write_text(json.dumps(stats, indent=4))


def _add_segments(dataset, entry: dict, runs, times: np.ndarray, poses_of, apertures_of, pictures_at,
                  state_names: list[str], task: str, fps: int, min_segment_s: float, say) -> None:
    """Each run [start, end) of source frames - taken at `times` - into the
    dataset as an episode of its own, its still ends cut off, laid on the
    dataset's clock of `fps`: `poses_of(hand, start, end)` the hand's smoothed
    tool poses, `apertures_of(hand, start, end)` its openings,
    `pictures_at(hand, t)` its picture for the moment t."""
    hands = entry["hands"]
    for start, end in runs:
        Ts = {hand: poses_of(hand, start, end) for hand in hands}
        apertures = {hand: apertures_of(hand, start, end) for hand in hands}
        first, last = moving(Ts, apertures)
        t = times[start:end][first:last]
        if len(t) < 2 or t[-1] - t[0] < min_segment_s:
            entry.setdefault("still", []).append({"start": start, "end": end})
            continue
        at = grid(t[0], t[-1], fps)
        # Splined from the segment's start: absolute stamps lose digits.
        segment = {hand: {"T": resample(Ts[hand][first:last], t - t[0], at - t[0]),
                          "aperture": np.interp(at, t, apertures[hand][first:last]).astype(np.float32)}
                   for hand in hands}
        trimmed = (first, end - start - last)
        start, end = start + first, start + last
        k = 0
        try:
            for k, frame in enumerate(_frames(segment, state_names, task,
                                              lambda hand, k, at=at: pictures_at(hand, at[k]), fps)):
                dataset.add_frame(frame)
        except ConvertError as exc:
            # A video that ends early: the frames this segment has added so
            # far go, and so does the rest of the episode.
            dataset.clear_episode_buffer()
            entry["stopped"] = {"frame": start, "step": k, "reason": str(exc)}
            say(f"  {entry['episode']}: stopped {k} frames into the segment from frame {start} - {exc}")
            return
        dataset.save_episode()
        entry["segments"].append({"start": start, "end": end, "frames": len(at),
                                  "seconds": round(float(at[-1] - at[0]), 3),
                                  "despiked": {hand: segment[hand]["despiked"] for hand in hands},
                                  "still_trimmed": list(trimmed)})


def _episode(dataset, src: Path, task: str, hands: tuple[str, ...], fps: int, min_segment_s: float,
             state_names: list[str], size: tuple[int, int], say, background: int | None = None) -> dict:
    """One raw episode into the dataset, as many LeRobot episodes as it has
    segments - with `background`, its pictures painted over but for what its
    keep labels keep, with the pattern of that seed."""
    name = src.name
    if not (src / "meta.json").is_file():
        raise ConvertError(f"no episode {name!r} at {src}")
    try:
        meta = json.loads((src / "meta.json").read_text())
        frames = _lines(src / "frames.jsonl")
        poses = _lines(src / "poses.jsonl")
    except ValueError as exc:
        say(f"  {name}: skipped - its files do not read ({exc})")
        return {"episode": name, "kind": "demo", "frames": 0, "segments": [], "skipped": f"its files do not read ({exc})"}
    kind = meta.get("kind", "demo")
    if kind == "rollout" and background is not None:
        say(f"  {name}: skipped - a rollout has no keep labels, so no randomised background")
        return {"episode": name, "kind": kind, "frames": 0, "segments": [],
                "skipped": "a rollout has no keep labels"}
    if kind == "rollout":
        return _rollout(dataset, src, meta, frames, task, hands, fps, min_segment_s, state_names, size, say)
    if kind != "demo":
        say(f"  {name}: skipped - a {kind} is not converted")
        return {"episode": name, "kind": kind, "frames": 0, "segments": [], "skipped": f"a {kind} is not converted"}
    entry = {"episode": name, "kind": "demo", "task": task, "hands": list(hands), "frames": len(poses), "segments": [],
             "seconds": meta.get("seconds")}
    # The solid the poses are in, as the take recorded it: pose.tcp's z is
    # over its origin, and which way it faced carries the poses into the
    # robot's world (demo_start).
    desk = meta.get("desk") or {}
    entry["desk"] = {k: desk[k] for k in ("solid", "face_mm", "origin_above_table_mm", "placement", "yaw_deg")
                     if desk.get(k) is not None}
    streams = meta.get("streams") or {}
    labels = {}
    if background is not None:
        entry["background"] = background
    try:
        if background is not None:
            for hand in hands:
                try:
                    labels[hand] = read_labels(src, f"gripper_{hand}")
                except (OSError, ValueError, zipfile.BadZipFile) as exc:
                    raise ConvertError(f"the {hand} gripper video's keep labels do not read ({exc}) - "
                                       "make them again on the Curate page") from exc
                if labels[hand] is None:
                    raise ConvertError(f"the {hand} gripper video has no keep labels - make them on the Curate page")
        world_T_cams = cameras_on_desk(poses, hands)
        exposed = exposures(poses)
        solid_T_tcp, opening, grip_t, centre = {}, {}, {}, {}
        for hand in hands:
            key = f"handheld_{hand}"
            try:
                solid_T_tcp[hand] = SETTINGS.recorded_solid_T_tcp(key, meta)
            except KeyError:
                raise ConvertError(f"the take does not record where {key} was on the handheld") from None
            jaw_t, jaw_v = jaw_readings(poses, hand)
            if not len(jaw_t):
                raise ConvertError(f"no reading from the {hand} jaw")
            opening[hand] = np.interp(exposed, jaw_t, jaw_v).astype(np.float32)
            grip_t[hand] = np.array([f["host_ts"] for f in frames if f["stream"] == f"gripper_{hand}"]) \
                - SETTINGS.gripper_camera_latency_s
            if not len(grip_t[hand]):
                raise ConvertError(f"no frame from the {hand} gripper camera")
            if hand in labels and len(labels[hand]) != len(grip_t[hand]):
                raise ConvertError(f"the {hand} keep labels are for {len(labels[hand])} frames, the video has "
                                   f"{len(grip_t[hand])} - make them again")
            # Where the lens axis lands in this video: as the recorder wrote
            # it, or, for a take recorded before the lens was measured, as
            # this rig's lens file has that hand now - and with neither, no
            # dataset.
            centre[hand] = (streams.get(f"gripper_{hand}") or {}).get("centre")
            if not (isinstance(centre[hand], list) and len(centre[hand]) == 2):
                try:
                    centre[hand] = list(require_lens(hand))
                except LensMissing as exc:
                    raise ConvertError(str(exc)) from exc
    except ConvertError as exc:
        entry["skipped"] = str(exc)
        say(f"  {name}: skipped - {exc}")
        return entry
    entry["centre"] = {hand: [round(float(c), 4) for c in centre[hand]] for hand in hands}
    # Each hand's tool where its camera was placed and its hand had, and
    # each frame's pictures fresh - every hole short enough bridged.
    tools = {hand: fill_holes([world_T_cam @ np.array(p["solids"][f"handheld_{hand}"]["pose"]) @ solid_T_tcp[hand]
                               if world_T_cam is not None and placed(p, hand) else None
                               for p, world_T_cam in zip(poses, world_T_cams)], exposed, SETTINGS.convert_max_hole_s)
             for hand in hands}
    pictured = np.ones(len(poses), dtype=bool)
    for hand in hands:
        period = 1.0 / ((streams.get(f"gripper_{hand}") or {}).get("fps") or 30)
        pictured &= fresh(grip_t[hand], exposed, period)[1]
    usable = bridged(pictured, exposed, SETTINGS.convert_max_hole_s) \
        & np.array([all(tools[hand][i] is not None for hand in hands) for i in range(len(poses))])
    rate = rate_of(exposed)
    backdrop = None if background is None else pattern(background, size)
    try:
        videos = open_videos(src, hands, size, centre, labels, backdrop)
    except ConvertError as exc:
        entry["skipped"] = str(exc)
        say(f"  {name}: skipped - {exc}")
        return entry
    try:
        _add_segments(
            dataset, entry, segments(usable, exposed, min_segment_s), exposed,
            lambda hand, start, end: smooth(np.array(tools[hand][start:end]), rate),
            lambda hand, start, end: opening[hand][start:end],
            lambda hand, t: videos[hand].frame(int(latest(grip_t[hand], [t])[0])),
            state_names, task, fps, min_segment_s, say)
    finally:
        for video in videos.values():
            video.close()
    kept = sum(s["frames"] for s in entry["segments"])
    if entry["segments"]:
        say(f"  {name}: {len(entry['segments'])} segment(s), {kept}/{len(poses)} frames kept, "
            f"{sum(sum(s['despiked'].values()) for s in entry['segments'])} despiked")
    elif "stopped" not in entry:
        say(f"  {name}: no segment long enough - skipped")
    return entry


def _rollout(dataset, src: Path, meta: dict, frames: list[dict], task: str, hands: tuple[str, ...],
             fps: int, min_segment_s: float, state_names: list[str], size: tuple[int, int], say) -> dict:
    """One rollout's corrections into the dataset: the arms' cameras are the
    pictures - the first hand's the clock - their forward kinematics the
    tools' poses, their jaws the openings - and only what the hands drove is
    kept, not what the policy drove nor where the clutch held."""
    from pinkrab_core.follow import TOOL_TURN
    from pinkrab_core.jaw import aperture_of, gripper_band
    from pinkrab_core.kinematics import ArmKinematics, arm_fingerprint

    name = src.name
    readings = sorted(_lines(src / "robot.jsonl"), key=lambda r: r["host_ts"])
    shots = {hand: sorted((f for f in frames if f["stream"] == f"gripper_{hand}"), key=lambda f: f["index"])
             for hand in hands}
    clock = hands[0]
    entry = {"episode": name, "kind": "rollout", "hands": list(hands), "frames": len(shots[clock]),
             "segments": [], "seconds": meta.get("seconds")}
    streams = meta.get("streams") or {}
    centre = {hand: (streams.get(f"gripper_{hand}") or {}).get("centre") for hand in hands}
    try:
        if sorted(meta.get("arms") or []) != sorted(hands):
            raise ConvertError(f"a rollout of the {' and '.join(meta.get('arms') or ['no'])} arm(s), "
                               f"not of the dataset's {' and '.join(hands)}")
        if any((meta.get("description") or {}).get(hand) != arm_fingerprint(hand) for hand in hands):
            raise ConvertError(OTHER_BUILD)
        if len(readings) < 2:
            raise ConvertError("no readings of the robot")
        for hand in hands:
            if not shots[hand]:
                raise ConvertError(f"no frame from the {hand} arm's camera")
            if not (isinstance(centre[hand], list) and len(centre[hand]) == 2):
                centre[hand] = list(require_lens(f"follower_{hand}"))
    except (ConvertError, LensMissing) as exc:
        entry["skipped"] = str(exc)
        say(f"  {name}: skipped - {exc}")
        return entry
    entry["centre"] = {hand: [round(float(c), 4) for c in centre[hand]] for hand in hands}
    chains = {hand: ArmKinematics(hand) for hand in hands}
    bands = {hand: gripper_band(hand) for hand in hands}
    turns = {}
    for hand in hands:
        turns[hand] = np.eye(4)
        turns[hand][:3, :3] = TOOL_TURN[hand]
    names = [n for hand in hands for n in (*chains[hand].joint_names, f"{hand}_gripper")]
    robot_t = np.array([r["host_ts"] for r in readings], dtype=float) - SETTINGS.robot_state_latency_s
    table = np.array([[r["joints"].get(n, np.nan) for n in names] for r in readings], dtype=float)
    driven, entry["redone"] = learned_from(readings, _lines(src / "events.jsonl"))
    exposed = np.array([f["host_ts"] for f in shots[clock]], dtype=float) - SETTINGS.gripper_camera_latency_s
    period = 1.0 / ((streams.get(f"gripper_{clock}") or {}).get("fps") or 30)
    # Each camera's last frame by each exposure of the clock's - none taken
    # more than a frame period and a half before it.
    stamps = {hand: np.array([f["host_ts"] for f in shots[hand]], dtype=float) - SETTINGS.gripper_camera_latency_s
              for hand in hands}
    pictured = np.ones(len(exposed), dtype=bool)
    for hand in hands[1:]:
        pictured &= fresh(stamps[hand], exposed, period)[1]

    # Each exposure's joints, between the readings either side of it - none
    # where those are far apart, or where the hands were not driving.
    joints = np.full((len(exposed), len(names)), np.nan)
    after = np.searchsorted(robot_t, exposed)
    for k, (i, t) in enumerate(zip(after, exposed)):
        if i == 0 or i >= len(robot_t):
            continue
        lo, hi = i - 1, i
        span = robot_t[hi] - robot_t[lo]
        if span > ROBOT_GAP_S or not (driven[lo] and driven[hi]):
            continue
        w = (t - robot_t[lo]) / span if span > 0 else 0.0
        joints[k] = table[lo] + w * (table[hi] - table[lo])
    usable = np.isfinite(joints).all(axis=1) & pictured
    # A hole in the video ends a segment as a hole in the readings does, and
    # so does a frame stamped no later than the one before it.
    gaps = np.diff(exposed)
    broken = np.concatenate([[False], (gaps > GRIP_MATCH_FRAMES * period) | (gaps <= 0)])

    runs, start = [], None
    for k in range(len(exposed) + 1):
        ok = k < len(exposed) and usable[k] and not (start is not None and broken[k])
        if ok and start is None:
            start = k
        elif not ok and start is not None:
            if exposed[k - 1] - exposed[start] >= min_segment_s:
                runs.append((start, k))
            start = k if k < len(exposed) and usable[k] else None
    columns = {}
    at = 0
    for hand in hands:
        n = len(chains[hand].joint_names)
        columns[hand] = (at, at + n)
        at += n + 1
    try:
        videos = open_videos(src, hands, size, centre)
    except ConvertError as exc:
        entry["skipped"] = str(exc)
        say(f"  {name}: skipped - {exc}")
        return entry
    rate = rate_of(exposed)

    def poses_of(hand: str, start: int, end: int) -> np.ndarray:
        a, b = columns[hand]
        return smooth(np.array([chains[hand].forward(np.degrees(joints[k, a:b])) @ turns[hand]
                                for k in range(start, end)]), rate)

    def apertures_of(hand: str, start: int, end: int) -> np.ndarray:
        _, b = columns[hand]
        return np.array([aperture_of(hand, float(joints[k, b]), bands[hand]) for k in range(start, end)],
                        dtype=np.float32)

    try:
        _add_segments(dataset, entry, runs, exposed, poses_of, apertures_of,
                      lambda hand, t: videos[hand].frame(int(shots[hand][int(latest(stamps[hand], [t])[0])]["index"])),
                      state_names, task, fps, min_segment_s, say)
    finally:
        for video in videos.values():
            video.close()
    kept = sum(s["frames"] for s in entry["segments"])
    if entry["segments"]:
        say(f"  {name}: rollout, {len(entry['segments'])} correction(s), {kept}/{len(exposed)} frames kept, "
            f"{sum(sum(s['despiked'].values()) for s in entry['segments'])} despiked")
    elif "stopped" not in entry:
        say(f"  {name}: rollout with no correction long enough - skipped")
    return entry
