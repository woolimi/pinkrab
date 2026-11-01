"""The solid being solved, while you hold it.

Two measurements only work with the solid in your hand and an answer on the screen,
and neither of them can be made from a recorded session.

The first is a freshly glued solid. A patch turned a quarter of the way round is
geometrically consistent all by itself, so a frame that sees only that face solves
cleanly and lies; it is the moment a second face joins the solve that the two
disagree and the reprojection error jumps. Finding that means turning the solid
until every pair of faces has been seen together, and watching the number while
you do.

The second is how noisy the pose is at a given distance. Hold the solid still,
watch the spread of the samples settle, write it down, move half a metre back and
do it again. What comes out is the curve that says how far from the camera the
gripper may be before the measurement stops being worth recording.

The browser is a window and nothing more. Python opens the camera, solves with its
lens - the board fit, lens.yaml top - and sends pictures and numbers; a page that ran its own
detector would be a second answer to the same question, computed from intrinsics
it had to be told, and the two would drift apart on the day the camera was
swapped.
"""

from __future__ import annotations

import errno
import json
import re
import sys
import threading
import time
import traceback
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np

from . import bench, ready_guide, record
from .episode import (EPISODE_NAME, EpisodeError, EpisodeWriter, default_name, delete_episode, list_episodes,
                      set_outcome)
from .gripper_cams import GripperCameras
from .jaw import JawReaders
from .config import SETTINGS
from .filtering import PoseFilter
from .markers import face_label
from .pnp import SolidPose, read_markers, solve_frame
from .trajectory import compose, matrix_to_quat
from .uvc import ControlError

STATIC = Path(__file__).parent / "static"
# What the top camera says while the arms' view keeps it closed (put_down).
HEAD_OFF = "off until a correction"

# The separator between JPEGs in the multipart stream. Any string the payload
# cannot contain will do, and JPEG is binary, so this one is only ever seen here.
BOUNDARY = "pinkrabsolid"

# Green for a face the pose was solved from, red for one that was seen and then
# thrown away - by the area gate, by belonging to some other board, or because the
# whole frame failed the reprojection gate. Which of the two a face is showing is
# the entire point of the picture, so they are the two colours furthest apart.
SOLVED_COLOUR = (0, 220, 0)      # BGR, as OpenCV draws
DESK_COLOUR = (255, 160, 0)       # the desk solid, blue-ish: the world, not a hand
DROPPED_COLOUR = (0, 0, 220)

# The frames the reported rate is averaged over. A second of them: shorter and the
# number flickers with USB jitter, longer and it takes a while to admit the
# pipeline has slowed down.
RATE_WINDOW = SETTINGS.fps

# What the viewer's own clock is worth. The page polls at this rate, and an MJPEG
# reader that is waiting on a frame gives up this often to notice the server going
# down rather than blocking on a socket forever.
POLL_TIMEOUT_S = 1.0

@dataclass(frozen=True)
class Frame:
    """What one tick of the source hands the loop: the picture, and when it
    was exposed, on this host's clock. A take can only be recorded from a
    source that fills all of it in.
    """

    picture: np.ndarray
    sensor_ts: float | None = None
    frame_number: int = 0


FrameSource = Callable[[], "Frame | None"]

# A session directory is named by this, straight off the network. Anything
# outside it - a slash above all - would name a directory somewhere else.
SESSION_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class LiveError(RuntimeError):
    """A request the view cannot honour right now - not a broken view."""


def rotation_spread_deg(quats: np.ndarray) -> tuple[float, float]:
    """How far a window of rotations sits from its own mean, in degrees.

    q and -q are the same rotation at opposite ends of the sphere, so a window that
    happened to straddle both would average to something near zero and report a
    spread that is an artefact of the sign. Every sample is flipped onto the first
    one's hemisphere first; after that, at the spreads worth measuring here - a
    solid resting on a table, a degree or two - the normalised mean quaternion is
    the geodesic mean to well inside the noise being measured.

    Both the mean and the worst sample are reported, and the worst is the one to
    watch: a single visible face is a planar target whose pose has a twofold
    ambiguity, and when noise flips it the sample lands tens of degrees away while
    the mean barely moves.
    """
    aligned = quats * np.where(quats @ quats[0] < 0.0, -1.0, 1.0)[:, None]
    mean = aligned.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm == 0.0:
        return float("nan"), float("nan")
    # The angle between two rotations is twice the angle between their unit
    # quaternions, and the absolute value takes the short way round.
    dot = np.clip(np.abs(aligned @ (mean / norm)), 0.0, 1.0)
    deviations = np.degrees(2.0 * np.arccos(dot))
    return float(deviations.mean()), float(deviations.max())


@dataclass
class _Take:
    """One recording in progress: the writer, and the same frames kept for the verdict.

    The results list is what bench would recompute from the PNGs - the arrays go
    to disk lossless and the solver is the same one, so keeping the answers as
    they happen buys the verdict at the moment the take ends instead of a minute
    of re-solving, and buys it identical.
    """

    name: str
    writer: record.SessionWriter
    started: float                                     # monotonic
    stamps: list[float] = field(default_factory=list)
    results: list[SolidPose] = field(default_factory=list)
    desks: list[SolidPose] = field(default_factory=list)   # the desk solid, per frame, unfiltered


class Live:
    """One thread pulling frames, solving them, and keeping what the page asks for.

    The frames come from a callable rather than from the camera directly, so the
    same loop runs over anything that can produce the frames the settings ask for
    and the server can be exercised on a machine with no camera on it.

    Everything the handlers read is behind one condition variable: the picture, the
    last frame's answer, and the window of samples. The condition is also how a
    stream waits for the next frame instead of spinning on the capture rate.
    """

    def __init__(self, source: FrameSource, calibration: dict, window: int | None = None,
                 cameras: str = "handheld"):
        # Whose gripper cameras are opened: the handhelds' (recorded from) or
        # the follower arms' (shown to a policy, never recorded as a demo).
        self.cameras = cameras
        self.grippers: GripperCameras | None = None
        self.jaws: JawReaders | None = None
        # The top camera as a thing that can be asked to open again; set by run().
        self.head_camera = None
        # The episode being recorded, under the io lock like the take.
        self._episode: EpisodeWriter | None = None
        # Which solids are looked for in every frame. Both handhelds when the
        # settings name one of them, so the handheld that is in frame is the one
        # that gets solved and nobody has to say which hand they picked up: the
        # left solid's ids are 0-15 and the right's 16-31, and a solid's own face
        # table simply has no entry for the other's ids.
        #
        # The domes are the exception, and they are why this is not just "always
        # both". Their id ranges overlap the handhelds' while their faces sit at
        # different distances from the centre, so solving a dome's ids against a
        # handheld's geometry would return a pose rather than an error - a wrong
        # answer that looks like a right one. A dome is therefore solved alone.
        self._solids: dict[str, object] = (
            {name: SETTINGS.for_solid(name) for name in ("handheld_left", "handheld_right")}
            if SETTINGS.solid.startswith("handheld_") else {SETTINGS.solid: SETTINGS}
        )
        # ...and the desk solid, always: it is the world every hand pose is
        # reported in, solved from the same picture at the same instant.
        self.desk = SETTINGS.desk_solid
        self._solids[self.desk] = SETTINGS.for_solid(self.desk)
        self._desk_ids = set(SETTINGS.for_solid(self.desk).face_ids.values())
        # The lens's numbers, as fitted against the board: the scale of every
        # pose. None when there is no top camera: the view then serves the
        # gripper cameras and solves nothing, and says so.
        self.head_reason = ""
        if calibration is None:
            self.intrinsics = None
        else:
            try:
                self.intrinsics = record.camera_of(calibration)
            except record.RecorderError as exc:
                raise LiveError(str(exc)) from exc
        self.window = window if window is not None else SETTINGS.live_window_frames
        # The calibration.json content - the lens's numbers it is solved with
        # and which fit they are (camera_source) - which a session on disk
        # needs a copy of.
        self.calibration = calibration
        # Set when the view is coming down, so an open stream lets go rather than
        # writing into a socket nobody is reading.
        self.stopping = threading.Event()
        self._source = source
        self._thread: threading.Thread | None = None
        self._state = threading.Condition()
        self._jpeg: bytes | None = None
        # Counted rather than timestamped: a stream asks for anything newer than
        # what it last sent, and two frames can share a millisecond.
        self._seq = 0
        self._primary: str | None = None   # the hand the single-solid fields last spoke for
        self._drawn: dict[int, tuple[np.ndarray, tuple, int]] = {}   # id -> (quad, colour, frames left)
        self._last: dict = {"solved": False, "faces": [], "detected": [],
                            "reproj_px": None, "distance_mm": None, "pose": None,
                            "reason": "", "world": None, "world_pose": None,
                            "solid": None, "solids": {}}
        # One stream filter per solid. The filter is causal - it judges a frame
        # against the pose it last accepted - so two solids sharing one would
        # judge each against where the other was. Warm from the whole time the
        # view has been up; a take's verdict does not use it, but runs a fresh
        # one over the take's frames, cold, exactly as bench will.
        self._filters = {name: PoseFilter() for name in self._solids}
        self._positions: deque[np.ndarray] = deque(maxlen=self.window)
        self._quats: deque[np.ndarray] = deque(maxlen=self.window)
        self._counts: Counter[int] = Counter()
        self._stamps: deque[float] = deque(maxlen=RATE_WINDOW)
        # The take in progress and the last verdict. The writer is only ever fed
        # by the capture thread and only ever swapped under this lock, so a stop
        # arriving mid-frame waits for the pair being written rather than closing
        # the files under it.
        self._io = threading.Lock()
        self._take: _Take | None = None
        self._verdict: dict | None = None
        # What the last frame died of, if it died. The loop does not stop for a
        # frame that would not solve - a picture that freezes on the page is a
        # camera nobody can tell from a dead one - so the failure is kept here
        # and shown beside the picture instead.
        self._trouble: str | None = None
        self._trouble_said: str | None = None
        # Each hand's tool at the robot's ready pose against the desk solid,
        # and the desk under the solid: drawn into the picture once worked out
        # (ready_guide); None until then.
        self._ready_guide: ready_guide.Guide | None = None

    # --- the capture side -------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="umi-live", daemon=True)
        self._thread.start()
        # Where the hands start a take, for the picture: the arms' model is
        # read for it, which takes a moment the picture does not wait for.
        if self.cameras == "handheld" and self.intrinsics is not None:
            threading.Thread(target=self._load_ready_guide, name="ready-guide", daemon=True).start()
        # The gripper cameras, beside the top camera: absent sides are
        # reported as absent, so nothing here waits for them.
        self.grippers = GripperCameras(self.cameras)
        for cam in self.grippers.sides.values():
            cam.sink = self._feed_episode_gripper
        self.grippers.start()
        # ...and the jaw servos, so every sensor on the handheld is read here.
        self.jaws = JawReaders()
        self.jaws.start()

    def _load_ready_guide(self) -> None:
        try:
            self._ready_guide = ready_guide.load()
        except Exception as exc:  # noqa: BLE001 - the picture goes on without the guide
            print(f"  ready guide not drawn: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)

    def stop(self) -> None:
        self.stopping.set()
        if self.grippers is not None:
            self.grippers.stop()
        if self.jaws is not None:
            self.jaws.stop()
        with self._io:
            if self._episode is not None:
                self._episode.close()
                self._episode = None
        with self._state:
            # Wake the streams so they see the flag instead of waiting out their
            # timeout on a server that has already gone.
            self._state.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        # A take left open when the view goes down still lands on disk - without
        # frames.jsonl it is a directory of orphan PNGs nothing can read.
        with self._io:
            if self._take is not None:
                self._take.writer.close()
                self._take = None

    def _run(self) -> None:
        while not self.stopping.is_set():
            frame = None
            try:
                frame = self._source()
                if frame is None:
                    continue
                self.consume(frame)
            except Exception as exc:  # noqa: BLE001 - one bad frame is not the end of the camera
                where = traceback.extract_tb(exc.__traceback__)[-1]
                self._stumble(f"{type(exc).__name__}: {exc} at {Path(where.filename).name}:{where.lineno}")
                # The picture still moves. A frame that would not solve is
                # shown undrawn rather than not at all, so the stream never
                # stands still on the last good one.
                if frame is not None:
                    self._show_raw(frame)
                else:
                    time.sleep(0.05)
                continue
            if self._trouble is not None:
                with self._state:
                    self._trouble = None

    def _show_raw(self, frame: Frame) -> None:
        picture = frame.picture
        if picture is None:
            return
        encoded, jpeg = cv2.imencode(".jpg", picture, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not encoded:
            return
        with self._state:
            self._jpeg = jpeg.tobytes()
            self._seq += 1
            self._stamps.append(time.monotonic())
            self._state.notify_all()

    def _stumble(self, what: str) -> None:
        """Note a frame that failed and keep going. Said aloud once per cause."""
        with self._state:
            self._trouble = what
            self._state.notify_all()
        if what != self._trouble_said:
            self._trouble_said = what
            print(f"  frame not solved, carrying on: {what}", file=sys.stderr, flush=True)

    def consume(self, frame: Frame) -> None:
        """Solve one frame and fold it into everything the page can ask about."""
        picture = frame.picture
        grey = picture if picture.ndim == 2 else cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
        # Decoded once, solved per solid. The detector is the expensive half
        # and it does not care which solid is being asked about - it reads
        # every marker in the picture, and each solid then claims its own.
        seen = read_markers(grey)
        raws = {name: solve_frame(grey, self.intrinsics, cfg, seen)
                for name, cfg in self._solids.items()}
        self._fold(frame, raws)

    def consume_markers(self, stamp: float | None, markers: dict) -> None:
        """Solve one frame again from the markers its pose line kept (`markers`
        - each id's four corners in the picture, as read and recovered) and
        fold it in as `consume` does a picture: what solving a take again with
        another lens fit takes, no picture needed."""
        detections = (tuple(int(k) for k in markers),
                      tuple(np.asarray(v, dtype=np.float64).reshape(4, 2) for v in markers.values()), ())
        raws = {name: solve_frame(None, self.intrinsics, cfg, detections) for name, cfg in self._solids.items()}
        self._fold(Frame(picture=None, sensor_ts=stamp), raws)

    def _fold(self, frame: Frame, raws: dict) -> None:
        """Each solid's answer for one frame filtered, and folded into
        everything the page can ask about - and the take, and the episode."""
        picture = frame.picture
        stamp = frame.sensor_ts if frame.sensor_ts is not None else time.time()

        answers = {name: self._filters[name].apply(raw, stamp) for name, raw in raws.items()}
        # Which solid the single-solid fields speak for. The one that spoke last
        # keeps the floor while it solves - two hands in view at once would
        # otherwise trade places every time one showed a face more, and the
        # numbers on the page with them. When it drops out, the one that solved
        # on more faces takes over.
        def standing(item) -> tuple:
            name, answer = item
            return (answer.ok and not answer.reason, name == self._primary,
                    len(answer.faces), name == SETTINGS.solid)
        hands = {name: answer for name, answer in answers.items() if name != self.desk}
        primary, result = max(hands.items(), key=standing)
        solved = result.ok and not result.reason
        if solved:
            self._primary = primary

        # The head. The desk solid, read off the same picture the hand's was:
        # same frame, same instant, nothing to line up in time. Every hand that
        # solved is carried into the desk's frame, whose z is up.
        desk = answers[self.desk]
        placed = desk.ok and not desk.reason and desk.pose is not None
        world = {"tracking": bool(placed), "faces": len(desk.faces),
                 "reproj_px": None if desk.pose is None else float(desk.reproj_px),
                 "reason": desk.reason or ("" if placed else "desk solid not in view")}
        world_T_cam = (np.linalg.inv(desk.pose)) if placed else None
        worlds = ({name: compose(world_T_cam, answer.pose)
                   for name, answer in hands.items()
                   if answer.ok and not answer.reason and answer.pose is not None}
                  if placed else {})
        world_pose = worlds.get(primary)
        # What the recording keeps: the same composition of the poses as the
        # filter chose them, not averaged with the frames before.
        held_cam = np.linalg.inv(desk.held) if placed else None
        held_worlds = ({name: compose(held_cam, answer.held)
                        for name, answer in hands.items()
                        if answer.ok and not answer.reason and answer.pose is not None}
                       if placed else {})
        encoded = False
        if picture is not None:
            encoded, jpeg = cv2.imencode(".jpg", self._draw(picture, result, answers),
                                         [cv2.IMWRITE_JPEG_QUALITY, 80])
        now = time.monotonic()

        with self._state:
            if encoded:
                self._jpeg = jpeg.tobytes()
            self._seq += 1
            self._stamps.append(now)
            for marker_id in result.detected:
                self._counts[marker_id] += 1
            self._last = {
                # Which handheld is in frame, and so which arm follows it. The
                # fields beside this one are that solid's; `solids` holds every one
                # that was looked for, solved or not.
                "solid": primary,
                "solids": {
                    name: {
                        "solved": answer.ok and not answer.reason,
                        "faces": list(answer.faces),
                        "reproj_px": None if answer.pose is None else float(answer.reproj_px),
                        "reason": answer.reason,
                        "distance_mm": None if answer.pose is None else
                                       float(np.linalg.norm(answer.pose[:3, 3]) * 1000.0),
                        "pose": None if answer.pose is None else
                                [[float(v) for v in row] for row in answer.pose],
                        "world_pose": None if name not in worlds else
                                      [[float(v) for v in row] for row in worlds[name]],
                        "held_pose": None if answer.pose is None else
                                     [[float(v) for v in row] for row in answer.held],
                        "held_world_pose": None if name not in held_worlds else
                                           [[float(v) for v in row] for row in held_worlds[name]],
                    }
                    for name, answer in hands.items()
                },
                "solved": solved,
                # When the picture was exposed, on the sensor clock in seconds, and
                # when it was solved, on this host's clock: the two ages a reader
                # downstream can measure its own lateness against.
                "stamp": float(stamp),
                "solved_at": time.time(),
                "faces": list(result.faces),
                "detected": list(result.detected),
                "reproj_px": None if result.pose is None else float(result.reproj_px),
                "distance_mm": None,
                "reason": result.reason,
                # Whether this frame could place the camera on the desk solid, and how well.
                "world": world,
                # cam_T_solid carried into the desk's frame, when it can be, for
                # the recording; teleop follows each solid's world_pose (solids, above).
                "world_pose": None if world_pose is None else
                              [[float(v) for v in row] for row in world_pose],
            }
            if result.pose is not None:
                translation = result.pose[:3, 3]
                self._last["distance_mm"] = float(np.linalg.norm(translation) * 1000.0)
                # The pose itself, for anything downstream of the page: this is
                # cam_T_solid, metres, and composing it with solid_T_tcp is what
                # turns a solid in a picture into a tool centre in a room.
                self._last["pose"] = [[float(v) for v in row] for row in result.pose]
                if solved:
                    self._positions.append(translation.copy())
                    self._quats.append(matrix_to_quat(result.pose[:3, :3]))
            self._state.notify_all()

        # The take keeps the unfiltered answer for the solid being carried: its
        # verdict refilters from the first frame so it matches what bench
        # recomputes from the disk.
        if picture is not None:
            self._feed_take(frame, raws[primary], stamp, raws[self.desk])
            self._feed_episode_head(frame, raws)

    # --- the episode ------------------------------------------------------------
    def _feed_episode_head(self, frame: Frame, raws: dict) -> None:
        """One head frame into the episode, if one is recording: its clocks,
        and everything this frame said about the solids, the desk and the jaws,
        with every marker read in it - the corners each solid's answer was
        solved from, recovered ones and all."""
        with self._io:
            episode = self._episode
        if episode is None:
            return
        host_ts = time.time()
        line = {"host_ts": host_ts, **self.pose_line()}
        readings = self.jaws.readings() if self.jaws is not None else {}
        # Each jaw's opening with the host time it was read, which is not this
        # frame's: a reading is held for up to the reader's stale limit.
        line["jaw"] = {side: None if r is None else r[0] for side, r in readings.items()}
        line["jaw_ts"] = {side: None if r is None else r[1] for side, r in readings.items()}
        line["markers"] = {str(marker_id): [round(float(v), 2) for v in np.asarray(quad).reshape(-1)]
                           for answer in raws.values()
                           for marker_id, quad in zip(answer.detected, answer.corners)}
        episode.add_poses(line)
        episode.add_frame("head", frame.picture, host_ts, frame.sensor_ts, frame.frame_number)

    def sizes(self) -> dict[str, float]:
        """Every solid this view looks for - the hand solids and the desk solid
        (`desk`) - by face size (mm): what a pose of one is scaled by."""
        return {name: cfg.face_size_mm for name, cfg in self._solids.items()}

    def pose_line(self) -> dict:
        """What the last frame said about the solids and the desk, as an
        episode's poses.jsonl keeps it - the poses as the stream filter chose
        them, not averaged (`SolidPose.held`) - less the host's clock and the jaws,
        which are the recorder's (`pinkrab-umi resolve` makes these again)."""
        with self._state:
            last = self._last
            return {
                "stamp": last.get("stamp"),
                "primary": last.get("solid"),
                "solids": {name: {"solved": c.get("solved"), "faces": c.get("faces"),
                                  "reproj_px": c.get("reproj_px"), "pose": c.get("held_pose"),
                                  "world_pose": c.get("held_world_pose")}
                           for name, c in (last.get("solids") or {}).items()},
                "desk": last.get("world"),
            }

    def _feed_episode_gripper(self, side: str, picture, host_ts: float) -> None:
        with self._io:
            episode = self._episode
        if episode is not None:
            episode.add_frame(f"gripper_{side}", picture, host_ts)

    def start_episode(self, name: str | None = None, kind: str = "demo", arms: list[str] | None = None,
                      policy: str | None = None, description: dict[str, str] | None = None) -> str:
        """A demonstration is recorded with the handheld's cameras, a rollout
        with the arms': the same sockets may hold either set, and a take filed
        under the other kind would be one set's pictures labelled as the
        other's."""
        wanted = {"demo": "handheld", "rollout": "follower"}.get(kind)
        if wanted is None:
            raise LiveError(f"an episode is a demo or a rollout, not {kind!r}")
        if self.cameras != wanted:
            raise LiveError(f"this live view has the {self.cameras} cameras - "
                            f"a {kind} is recorded with the {wanted} ones")
        if kind == "rollout" and not (isinstance(arms, list) and arms and set(arms) <= {"left", "right"}
                                      and isinstance(description, dict) and set(description) == set(arms)):
            raise LiveError(f"a rollout names the arms the policy drives and each one's build, not {arms!r}")
        name = name or default_name()
        if not EPISODE_NAME.match(name):
            raise LiveError(f"{name!r} is not an episode name (letters, digits, _.-)")
        root = SETTINGS.episodes_dir / name
        if root.exists():
            raise LiveError(f"an episode named {name!r} exists")
        head_fps = SETTINGS.fps
        # The lens the top camera is solved with, kept whether it is open now
        # or not: the arms' view opens it only for a correction, and a
        # correction's poses are solved again against it.
        from .head_cam import top_calibration
        calibration = self.calibration or top_calibration()
        with self._io:
            if self._episode is not None:
                raise LiveError(f"already recording {self._episode.name!r}")
            self._episode = EpisodeWriter(name, root, calibration, head_fps,
                                          SETTINGS.gripper_camera_fps,
                                          hands={name: mm for name, mm in self.sizes().items() if name != self.desk},
                                          kind=kind, cameras=self.cameras, arms=arms, policy=policy,
                                          description=description)
        return name

    def add_robot(self, samples: list[dict]) -> int:
        """The robot's readings into the rollout being recorded."""
        with self._io:
            episode = self._episode
        if episode is None:
            raise LiveError("no episode is recording")
        return episode.add_robot(samples)

    def redo(self, correction: int) -> None:
        """A correction of the rollout being recorded thrown away, to be made
        again: its readings stay in the file, and convert leaves them out."""
        with self._io:
            episode = self._episode
        if episode is None:
            raise LiveError("no episode is recording")
        episode.event("redo", correction=int(correction))

    def stop_episode(self) -> dict:
        with self._io:
            episode, self._episode = self._episode, None
        if episode is None:
            raise LiveError("no episode is recording")
        return episode.close()

    def episode_status(self) -> dict:
        with self._io:
            episode = self._episode
        return {"active": episode is not None,
                "kind": episode.kind if episode is not None else None,
                **(episode.status() if episode is not None else {"name": None, "seconds": None, "frames": None})}

    def _feed_take(self, frame: Frame, result: SolidPose, stamp: float,
                   desk: SolidPose) -> None:
        """The recording side of one frame, if a take is on.

        Outside the state lock: two PNG writes take milliseconds, and the page
        polling /stats should not wait on the disk. The io lock is enough - only
        this thread feeds the writer, and a stop taking the lock first simply
        wins, with add() refusing the frame that was in flight.
        """
        with self._io:
            take = self._take
            if take is None:
                return
            take.results.append(result)
            take.stamps.append(stamp)
            take.desks.append(desk)
            if frame.sensor_ts is not None:
                take.writer.add(frame.picture, frame.sensor_ts, frame.frame_number, time.time())
        if time.monotonic() - take.started >= SETTINGS.take_max_seconds:
            try:
                self.stop_take()
            except LiveError:
                pass  # somebody's stop beat this one to it - theirs counts

    def _draw(self, picture: np.ndarray, result: SolidPose,
              answers: dict[str, SolidPose]) -> np.ndarray:
        """The frame with the detection drawn back onto it, and where each hand
        starts a take.

        Every solid that solved has its faces drawn, not only the one being
        followed: two handhelds in one picture are two solved solids, and the one
        that is not primary is not a thrown-away face. The desk's faces are in
        their own colour, so what is world and what is hand is told apart at a
        glance. Over them, once the desk solid is placed, each hand's solid
        where the robot's ready pose would hold it, and how far the hand is
        from it (ready_guide). The picture is copied, so the frame going to
        disk is not drawn on.
        """
        canvas = picture.copy() if picture.ndim == 3 else cv2.cvtColor(picture, cv2.COLOR_GRAY2BGR)
        used = {
            face for answer in answers.values() if answer.ok and not answer.reason
            for face in answer.faces
        }
        # Every square read this frame, from every solid: the hands' answers
        # carry their own detections, and the desk's are told apart by id.
        seen: dict[int, np.ndarray] = {}
        for answer in answers.values():
            for marker_id, quad in zip(answer.detected, answer.corners):
                seen[marker_id] = quad
        # One weight for every outline. A face that contributed and one that
        # was set aside differ in colour, not in thickness: a line that changes
        # weight from frame to frame reads as a face that blinks.
        for marker_id, quad in seen.items():
            if marker_id in self._desk_ids:
                colour = DESK_COLOUR
            else:
                colour = SOLVED_COLOUR if marker_id in used else DROPPED_COLOUR
            self._drawn[marker_id] = (quad, colour, SETTINGS.draw_hold_frames)
            self._paint(canvas, marker_id, quad, colour, 2)
        # ...and the ones read a moment ago and not now, held where they were,
        # exactly as they were, until they are read again or the hold runs out.
        for marker_id in list(self._drawn):
            if marker_id in seen:
                continue
            quad, colour, left = self._drawn[marker_id]
            if left <= 0:
                del self._drawn[marker_id]
                continue
            self._drawn[marker_id] = (quad, colour, left - 1)
            self._paint(canvas, marker_id, quad, colour, 2)
        guide, desk = self._ready_guide, answers.get(self.desk)
        if guide is not None and desk is not None and desk.ok and not desk.reason and desk.pose is not None:
            hands = {name.removeprefix("handheld_"): (answer.pose if answer.ok and not answer.reason else None)
                     for name, answer in answers.items() if name.startswith("handheld_")}
            ready_guide.draw(canvas, self.intrinsics, desk.pose, hands, guide)
        return canvas

    @staticmethod
    def _paint(canvas, marker_id: int, quad: np.ndarray, colour, thick: int) -> None:
        cv2.polylines(canvas, [quad.astype(np.int32)], True, colour, thick)
        centre = quad.mean(axis=0).astype(int)
        cv2.putText(canvas, str(marker_id), (int(centre[0]) - 6, int(centre[1]) + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)

    # --- what the page reads ----------------------------------------------

    def frame_after(self, seen: int) -> tuple[bytes | None, int]:
        """The current picture, waiting first if it is the one the caller already has."""
        with self._state:
            if self._seq == seen and not self.stopping.is_set():
                self._state.wait(POLL_TIMEOUT_S)
            return self._jpeg, self._seq

    def stats(self) -> dict:
        with self._state:
            last = dict(self._last)
            positions = np.array(self._positions) if self._positions else None
            quats = np.array(self._quats) if self._quats else None
            counts = dict(self._counts)
            stamps = list(self._stamps)

        samples = 0 if positions is None else len(positions)
        std_mm = {"x": None, "y": None, "z": None}
        rotation = {"mean_deg": None, "max_deg": None}
        if samples >= 2:
            # Population standard deviation, not the sample one: this is the spread
            # of the measurements that were taken, not an estimate of a wider
            # population's, and there is no smaller set they were drawn from.
            spread = positions.std(axis=0) * 1000.0
            std_mm = {"x": float(spread[0]), "y": float(spread[1]), "z": float(spread[2])}
            mean_deg, max_deg = rotation_spread_deg(quats)
            rotation = {"mean_deg": mean_deg, "max_deg": max_deg}

        # Measured over the frames that actually arrived rather than assumed from
        # the mode the camera was asked for: a solve that cannot keep up shows here
        # and nowhere else.
        fps = None
        if len(stamps) >= 2 and stamps[-1] > stamps[0]:
            fps = (len(stamps) - 1) / (stamps[-1] - stamps[0])

        return {
            "last": last,
            "trouble": self._trouble,
            "window": {
                "size": self.window,
                "samples": samples,
                "std_mm": std_mm,
                "rotation": rotation,
            },
            "counts": {str(i): counts.get(i, 0) for i in sorted(SETTINGS.id_faces)},
            "labels": {str(i): face_label(name) for i, name in SETTINGS.id_faces.items()},
            "fps": fps,
            "max_reproj_px": SETTINGS.max_reproj_px,
            "take": self.take_status(),
            "verdict": self._verdict,
            "thresholds": {
                "translation_mm": SETTINGS.closure_translation_mm,
                "rotation_deg": SETTINGS.closure_rotation_deg,
                "coverage": SETTINGS.min_coverage,
            },
        }

    def reset(self) -> None:
        """Start the measurement again, which is what moving the solid means."""
        with self._state:
            self._positions.clear()
            self._quats.clear()
            self._counts.clear()

    # --- closure takes ----------------------------------------------------

    @property
    def head(self) -> bool:
        return self.intrinsics is not None

    def attach_head(self, calibration: dict) -> None:
        """The top camera came up (or came back): take its intrinsics."""
        intrinsics = record.camera_of(calibration)
        with self._state:
            self.intrinsics = intrinsics
            self.calibration = calibration
            self.head_reason = ""

    def detach_head(self, reason: str) -> None:
        """The top camera went away: nothing to solve with until it returns."""
        with self._state:
            self.intrinsics = None
            self.calibration = None
            self.head_reason = reason
        self.reset()

    def start_take(self, name: str | None = None) -> str:
        if not self.head:
            raise LiveError(f"no top camera: {self.head_reason or 'not connected'}")
        """Open a session on disk and start feeding it. Returns the name used."""
        if self.calibration is None:
            raise LiveError("this source cannot back a session - no calibration")
        name = name or record.default_session_name()
        if not SESSION_NAME.match(name):
            raise LiveError(f"{name!r} is not a session name (letters, digits, _.-)")
        with self._io:
            if self._take is not None:
                raise LiveError(f"already recording {self._take.name!r}")
            writer = record.SessionWriter(record.session_paths(name), self.calibration)
            self._take = _Take(name=name, writer=writer, started=time.monotonic())
        return name

    def stop_take(self) -> dict:
        """Close the session and judge it, with bench's own arithmetic.

        The frames went to disk lossless and were solved by the same solver on the
        way, so the closure and coverage computed here from the kept results are
        the ones `pinkrab-umi bench` recomputes from the PNGs - the page shows the
        verdict now, and the session stays re-benchable later.
        """
        with self._io:
            take, self._take = self._take, None
            if take is None:
                raise LiveError("no take is recording")
            root = take.writer.close()

        posefilter = PoseFilter()
        results = [posefilter.apply(r, t) for r, t in zip(take.results, take.stamps)]
        deskfilter = PoseFilter()
        desks = [deskfilter.apply(d, t) for d, t in zip(take.desks, take.stamps)]
        poses = [compose(np.linalg.inv(d.pose), r.pose)
                 for r, d in zip(results, desks) if r.ok and d.ok and not d.reason]
        closure = bench.closure_error(poses)
        cover = bench.coverage(results, composed=len(poses))

        def num(value: float) -> float | None:
            return None if np.isnan(value) else float(value)

        verdict = {
            "name": take.name,
            "root": str(root),
            "seconds": round(time.monotonic() - take.started, 1),
            "closure": {
                "translation_mm": num(closure.translation_mm),
                "rotation_deg": num(closure.rotation_deg),
                "passes": bool(closure.passes),
            },
            "coverage": {
                "frames": cover.frames,
                "solved": cover.solved,
                "composed": cover.composed,
                "fraction": cover.fraction,
                "mean_faces": cover.mean_faces,
                "passes": bool(cover.passes),
            },
            "passes": bool(closure.passes and cover.passes),
        }
        with self._state:
            self._verdict = verdict
        return verdict

    def take_status(self) -> dict:
        with self._io:
            take = self._take
            active = take is not None
            return {
                "available": self.calibration is not None,
                "active": active,
                "name": take.name if active else None,
                "seconds": round(time.monotonic() - take.started, 1) if active else None,
                "frames": take.writer.count if active else None,
            }


def _name_of(body: bytes) -> str | None:
    """The session name in a start request, if the body carries one at all."""
    try:
        name = json.loads(body.decode() or "{}").get("name")
    except (ValueError, AttributeError):
        return None
    return str(name) if name else None


def make_handler(live: Live):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quieter than the default
            pass

        def _send(self, body: bytes, content_type: str, code: int = 200) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, obj, code: int = 200) -> None:
            self._send(json.dumps(obj).encode(), "application/json", code)

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path

            if path == "/":
                self._send((STATIC / "live.html").read_bytes(), "text/html; charset=utf-8")
                return

            if path == "/stream":
                self._stream()
                return

            if path == "/stats":
                stats = live.stats()
                stats["gripper"] = live.grippers.stats() if live.grippers else {}
                stats["head"] = live.head
                stats["head_reason"] = (live.head_camera.reason if live.head_camera is not None
                                        else live.head_reason)
                stats["head_opening"] = bool(live.head_camera is not None and live.head_camera.opening)
                stats["head_off"] = bool(live.head_camera is not None and live.head_camera.off)
                stats["head_light"] = live.head_camera.light() if live.head_camera is not None else None
                stats["jaw"] = live.jaws.stats() if live.jaws else {}
                stats["episode"] = live.episode_status()
                # Whose gripper cameras these are: a page switching the set
                # can tell the view it asked for from the one going down.
                stats["cameras"] = live.cameras
                self._send_json(stats)
                return

            if path == "/episodes":
                self._send_json({"episodes": list_episodes(), "dir": str(SETTINGS.episodes_dir)})
                return

            if path.startswith("/gripper/"):
                side = path.split("/")[2] if len(path.split("/")) > 2 else ""
                cam = live.grippers.sides.get(side) if live.grippers else None
                if cam is None:
                    self._send_json({"error": "not found"}, 404)
                    return
                self._stream(lambda seen: cam.jpeg_after(seen, POLL_TIMEOUT_S))
                return

            self._send_json({"error": "not found"}, 404)

        def _stream(self, source=None) -> None:
            """The left imager, as fast as it is being solved.

            MJPEG rather than anything cleverer because the browser decodes it in an
            <img> with no script at all, and because a viewer that falls behind
            wants the newest frame rather than the next one in a queue - which is
            what a stream of whole pictures gives for free.
            """
            self.send_response(200)
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
            self.send_header("Cache-Control", "no-cache")
            # This connection is held open, so it cannot carry anything else and the
            # browser should not wait to find that out.
            self.send_header("Connection", "close")
            self.end_headers()

            seen = -1
            after = source or live.frame_after
            try:
                while not live.stopping.is_set():
                    jpeg, seen = after(seen)
                    if jpeg is None:
                        continue
                    self.wfile.write(
                        f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                        f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                    )
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return  # the tab was closed; nothing here to report

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            # The body is read whether the route wants it or not: bytes left in
            # the socket become the front of the next request.
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(min(length, 64_000)) if length > 0 else b""

            if path == "/episode/start":
                try:
                    asked = json.loads(body.decode() or "{}")
                    if not isinstance(asked, dict):
                        raise ValueError
                except (ValueError, UnicodeDecodeError):
                    asked = {}
                try:
                    name = live.start_episode(_name_of(body), str(asked.get("kind") or "demo"),
                                              asked.get("arms"), asked.get("policy"),
                                              description=asked.get("description"))
                except (LiveError, EpisodeError) as exc:
                    self._send_json({"error": str(exc)}, 409)
                    return
                self._send_json({"ok": True, "name": name})
                return

            if path == "/episode/stop":
                try:
                    status = live.stop_episode()
                except LiveError as exc:
                    self._send_json({"error": str(exc)}, 409)
                    return
                self._send_json({"ok": True, "episode": status})
                return

            if path == "/episode/robot":
                # The robot's readings for a rollout, in batches from the
                # studio, which hears the bridge.
                try:
                    samples = json.loads(body.decode() or "{}").get("samples") or []
                    kept = live.add_robot(list(samples))
                except (ValueError, AttributeError, TypeError, UnicodeDecodeError) as exc:
                    self._send_json({"error": f"not a batch of readings: {exc}"}, 400)
                    return
                except LiveError as exc:
                    self._send_json({"error": str(exc)}, 409)
                    return
                self._send_json({"ok": True, "kept": kept})
                return

            if path == "/episode/redo":
                # A correction thrown away and made again (the policy page's
                # pedal 3 while the hands have the arms) - see dagger.py.
                try:
                    live.redo(int(json.loads(body.decode() or "{}")["correction"]))
                except (ValueError, KeyError, TypeError, UnicodeDecodeError) as exc:
                    self._send_json({"error": f"not a correction to redo: {exc}"}, 400)
                    return
                except LiveError as exc:
                    self._send_json({"error": str(exc)}, 409)
                    return
                self._send_json({"ok": True})
                return

            if path == "/episode/delete":
                try:
                    delete_episode(_name_of(body) or "")
                except EpisodeError as exc:
                    self._send_json({"error": str(exc)}, 404)
                    return
                self._send_json({"ok": True})
                return

            if path == "/episode/outcome":
                # What the person watching said of a rollout (the policy
                # page's succeeded / failed), into its meta.json.
                try:
                    asked = json.loads(body.decode() or "{}")
                    set_outcome(str(asked.get("name") or ""), asked.get("outcome"))
                except (ValueError, AttributeError, UnicodeDecodeError) as exc:
                    self._send_json({"error": f"not an outcome: {exc}"}, 400)
                    return
                except (EpisodeError, OSError) as exc:
                    self._send_json({"error": str(exc)}, 409)
                    return
                self._send_json({"ok": True})
                return

            if path == "/head/retry":
                # The Retry button: open the top camera again, now. Answers
                # at once; whether it came up is in the stats that follow.
                if live.head_camera is None:
                    self._send_json({"error": "no top camera in this view"}, 409)
                    return
                self._send_json({"ok": True, "started": live.head_camera.retry()})
                return

            if path == "/head/off":
                # The top camera put down until /head/retry asks for it again:
                # the arms' view keeps it closed but for a correction.
                if live.head_camera is None:
                    self._send_json({"error": "no top camera in this view"}, 409)
                    return
                live.head_camera.put_down(HEAD_OFF)
                self._send_json({"ok": True})
                return

            if path == "/head/light":
                # The picture's light, set from the page while it runs: the
                # exposure in ms and the bright end the gain follows.
                if live.head_camera is None:
                    self._send_json({"error": "no top camera in this view"}, 409)
                    return
                try:
                    asked = json.loads(body.decode() or "{}")
                    exposure = asked.get("exposure_ms")
                    target = asked.get("target")
                    light = live.head_camera.set_light(
                        None if exposure is None else float(exposure), None if target is None else float(target))
                except (ValueError, TypeError, AttributeError, UnicodeDecodeError) as exc:
                    self._send_json({"error": f"not a light setting: {exc}"}, 400)
                    return
                except ControlError as exc:
                    self._send_json({"error": str(exc)}, 502)
                    return
                if light is None:
                    self._send_json({"error": "the top camera is not open"}, 409)
                    return
                self._send_json({"ok": True, "light": light})
                return

            parts = path.strip("/").split("/")
            if len(parts) == 3 and parts[0] in ("gripper", "jaw") and parts[2] == "retry":
                # A Refresh on the page's connection list: one camera or one
                # jaw found and opened again. Answers at once; the stats that
                # follow say how it went.
                readers = live.grippers if parts[0] == "gripper" else live.jaws
                if readers is None or parts[1] not in readers.sides:
                    self._send_json({"error": f"no {parts[0]} {parts[1]} in this view"}, 404)
                    return
                readers.retry(parts[1])
                self._send_json({"ok": True})
                return

            if path == "/reset":
                live.reset()
                self._send_json({"ok": True})
                return

            if path == "/take/start":
                try:
                    name = live.start_take(_name_of(body))
                except LiveError as exc:
                    self._send_json({"error": str(exc)}, 409)
                    return
                self._send_json({"ok": True, "name": name})
                return

            if path == "/take/stop":
                try:
                    verdict = live.stop_take()
                except LiveError as exc:
                    self._send_json({"error": str(exc)}, 409)
                    return
                self._send_json({"ok": True, "verdict": verdict})
                return

            self._send_json({"error": "not found"}, 404)

    return Handler


def make_server(live: Live, host: str, port: int) -> ThreadingHTTPServer:
    try:
        return ThreadingHTTPServer((host, port), make_handler(live))
    except OSError as exc:
        if exc.errno != errno.EADDRINUSE:
            raise
        # Say which failure this is. A silent bind failure leaves an older server
        # answering, and an older server is showing an older camera.
        raise SystemExit(
            f"\n  port {port} is already taken - another live view is running.\n"
            f"  Find it with:  lsof -i :{port}\n"
            f"  then kill it, or start this one with --port.\n"
        ) from exc


def serve(live: Live, host: str, port: int) -> None:
    httpd = make_server(live, host, port)
    print(f"\n  Pinkrab UMI live  ->  http://localhost:{port}/")
    cam = live.intrinsics
    if live.head_camera is not None and live.head_camera.off:
        print(f"  top camera: {live.head_camera.reason} - opened for each, closed after")
    elif cam is None:
        why = live.head_camera.reason if live.head_camera is not None else live.head_reason
        print(f"  top camera: none - {why}. The gripper cameras are served;")
        print("           nothing is solved. Plug the top camera in and press Retry on the page.")
    else:
        print(f"  top camera: {cam.width}x{cam.height}, fx {cam.fx:.1f} px - "
              f"{(live.calibration or {}).get('camera_source')}")
        ids = sorted(SETTINGS.for_solid(live.desk).face_ids.values())
        print(f"  world: the desk solid, ids {ids[0]}-{ids[-1]} - poses in its frame, z up")
        print(f"  window: {live.window} solved frames")
    print(f"  gripper cameras: the {live.cameras} set, left and right, as found on the USB")
    print("  Ctrl+C to stop.\n")
    live.start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped\n")
    finally:
        live.stop()
        httpd.server_close()


def run(port: int | None = None, cameras: str = "handheld") -> None:
    """Open the camera, put the marker's picture on a page, and serve until Ctrl-C.

    Bound to the loopback address. The page carries a picture of whatever the
    camera is pointed at, which on a head-worn rig is the room somebody is sitting
    in, and there is no authentication anywhere in this server.

    The camera's exposure and gain are set before its first frame is handed
    on, so there is no settling to discard.
    """
    from .head_cam import HeadCamera

    head = HeadCamera()
    live = Live(head.next_frame, None, cameras=cameras)
    live.head_camera = head
    head.on_open = live.attach_head
    head.on_close = live.detach_head
    if cameras == "follower":
        # A policy is shown the arms' gripper cameras alone; the top camera
        # follows the handhelds, which only a correction needs (/head/retry).
        head.put_down(HEAD_OFF)
        print(f"top camera: {HEAD_OFF}")
    else:
        print("opening the top camera ...")
        head.start()
        # Given a few seconds to come up before the banner, so the usual start
        # prints the camera's numbers; a camera that is not there does not hold
        # the server back.
        for _ in range(40):
            if head.open or not head.opening:
                break
            time.sleep(0.25)
    try:
        serve(live, "127.0.0.1", SETTINGS.live_port if port is None else port)
    finally:
        head.stop()
