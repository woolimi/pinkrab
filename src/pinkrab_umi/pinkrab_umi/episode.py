"""One recorded episode: two videos and the numbers that go with them.

Raw, on purpose. The LeRobot dataset is made from these later and can be
remade with a different transform; the videos and the per-frame numbers are
what cannot be recovered afterwards. One directory per episode:

    gripper_left.mp4, gripper_right.mp4   the pictures
    frames.jsonl     one line per camera frame: stream, index in its video,
                     clocks - the top camera's too, which is kept as what was
                     read off it rather than as a video
    poses.jsonl      one line per head frame: every solid the solver looked for,
                     in the camera and in the desk's frame, the desk itself,
                     the jaws' openings at that instant, and every marker read
                     in the frame, its four corners in the picture (`markers`)
                     - what the poses are solved again from
    robot.jsonl      a rollout's only: the driven arms' joints as the robot
                     read them, the gate's mode, who had the arms
                     (`controller`) and which correction it was
                     (`correction`), a line per reading (meta.json's
                     `description` says which build of each arm the angles
                     are of)
    events.jsonl     start and stop, and in a rollout each correction redone
    meta.json        clocks, calibration, where the desk solid stands, counts,
                     what kind of episode it is, what each camera was opened
                     as, every marker's corners on its solid and each gripper
                     camera's lens fit - enough to
                     solve the poses again and undistort the pictures from the
                     episode alone - and of a rollout, once the person watching
                     has said so, its `outcome`

Two kinds. A demonstration (`demo`) is the handhelds': their cameras, their
solids' poses, their jaws - the hands did the task. A rollout (`rollout`) is
the robot's: a policy's run from its start to its stop, the handhelds taking
the arms over to put them right (DAgger) when it went wrong. What the policy
sees and does is the ARMS' cameras and the arms' own motion - their joints,
read off the robot - and every reading says who had the arms then:

    policy   the policy driving them
    human    the handhelds driving them - a correction
    hold     the clutch holding them still
    none     neither - the walk to ready, a run stopped at an arm's reach

The videos run through all of it; which stretches are learned from is
convert's choice, made from `controller`.

Every line carries `host_ts`, this machine's clock at the moment the frame
or reading arrived - the master clock the three cameras and the jaws are
lined up on later. The top camera's lines also carry `sensor_ts`, their
arrival less the module's latency, on the same clock.

Encoding runs on a thread per video so the camera threads never wait on the
disk; a frame that finds its queue full is dropped and counted.
"""
from __future__ import annotations

import json
import queue
import re
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from pinkrab_core.config import CAMERA_ROLES
from pinkrab_core.config import SETTINGS as CORE
from pinkrab_core.dataset import gripper_region, lens_centre, lens_fit

from .config import SETTINGS

EPISODE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
STREAMS = ("head", "gripper_left", "gripper_right")


class EpisodeError(RuntimeError):
    pass


def default_name() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def kept_size(width: int, height: int) -> tuple[int, int]:
    """The size a width x height gripper frame is kept at: the part of the
    sensor it shows at episode_sensor_scale, even for the encoder - or as
    read, where that is no bigger or the frame is no gripper camera's."""
    try:
        _, _, region_w, region_h = gripper_region(width, height)
    except ValueError:
        return width, height
    kept = (2 * round(region_w * SETTINGS.episode_sensor_scale / 2),
            2 * round(region_h * SETTINGS.episode_sensor_scale / 2))
    return kept if kept[0] < width else (width, height)


class Encoder:
    """One video on disk, fed BGR frames. PyAV with the configured codec, or
    OpenCV's writer when that cannot be had here."""

    def __init__(self, path: Path, fps: float, width: int, height: int) -> None:
        self.codec = ""
        self._av = None
        self._stream = None
        self._cv = None
        try:
            import av
            container = av.open(str(path), "w")
            stream = container.add_stream(SETTINGS.episode_codec, rate=int(round(fps)))
            stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
            stream.options = {"crf": str(SETTINGS.episode_crf), "preset": SETTINGS.episode_preset,
                              "g": str(SETTINGS.episode_gop)}
            self._av, self._stream = container, stream
            self.codec = SETTINGS.episode_codec
        except Exception:  # noqa: BLE001 - no PyAV, or no such encoder: OpenCV then
            for codec in SETTINGS.episode_codecs:
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
                if writer.isOpened():
                    self._cv, self.codec = writer, codec
                    break
                writer.release()

    @property
    def ok(self) -> bool:
        return self._av is not None or self._cv is not None

    def write(self, picture: np.ndarray) -> None:
        if self._av is not None:
            import av
            frame = av.VideoFrame.from_ndarray(picture, format="bgr24")
            for packet in self._stream.encode(frame):
                self._av.mux(packet)
        elif self._cv is not None:
            self._cv.write(picture)

    def close(self) -> None:
        if self._av is not None:
            for packet in self._stream.encode():
                self._av.mux(packet)
            self._av.close()
        elif self._cv is not None:
            self._cv.release()


@dataclass
class _Video:
    stream: str
    path: Path | None    # None: clocked, its pictures not kept
    fps: float
    queue: queue.Queue
    thread: threading.Thread | None = None
    writer: Encoder | None = None
    written: int = 0
    dropped: int = 0
    size: tuple[int, int] | None = None
    codec: str = ""
    # What stopped the encoder, if it stopped: the frames after it are counted
    # as dropped, and the video is not whole.
    failure: str | None = None


KINDS = ("demo", "rollout")
CONTROLLERS = ("policy", "human", "hold", "none")
# Two robot readings further apart than this are a hole in the readings: the
# time between them is nobody's, and a pose between them is not interpolated.
ROBOT_GAP_S = 0.1


class EpisodeWriter:
    def __init__(self, name: str, root: Path, calibration: dict | None, head_fps: float,
                 gripper_fps: float, hands: dict[str, float], kind: str = "demo",
                 cameras: str = "handheld", arms: list[str] | None = None, policy: str | None = None,
                 description: dict[str, str] | None = None) -> None:
        if kind not in KINDS:
            raise EpisodeError(f"an episode is one of {', '.join(KINDS)}, not {kind!r}")
        self.name = name
        self.kind = kind
        self.cameras = cameras
        self.root = root
        self.started_wall = time.time()
        self.started = time.monotonic()
        self.ended: float | None = None
        self._lock = threading.Lock()
        self._closed = False
        # A rollout's seconds by who had the arm, summed between readings,
        # and the reading they are summed from.
        self._controlled = {name: 0.0 for name in CONTROLLERS}
        self._last_reading: tuple[float, str] | None = None
        self.root.mkdir(parents=True, exist_ok=False)
        self._frames = open(self.root / "frames.jsonl", "w")
        self._poses = open(self.root / "poses.jsonl", "w")
        self._events = open(self.root / "events.jsonl", "w")
        self._robot = open(self.root / "robot.jsonl", "w") if kind == "rollout" else None
        # The top camera's pictures are not kept: they are of the room and the
        # person, never shown to a policy, and the poses are solved again from
        # the corners each frame's line keeps. Its frames are still counted
        # and clocked.
        self._videos = {
            "head": _Video("head", None, head_fps, queue.Queue(1)),
            "gripper_left": _Video("gripper_left", self.root / "gripper_left.mp4", gripper_fps,
                                   queue.Queue(SETTINGS.episode_queue_frames)),
            "gripper_right": _Video("gripper_right", self.root / "gripper_right.mp4", gripper_fps,
                                    queue.Queue(SETTINGS.episode_queue_frames)),
        }
        for video in self._videos.values():
            if video.path is None:
                continue
            video.thread = threading.Thread(target=self._encode, args=(video,),
                                            name=f"episode-{video.stream}", daemon=True)
            video.thread.start()
        self.meta = {
            "name": name,
            "started_wall": self.started_wall,
            "started_monotonic": self.started,
            "calibration": calibration,
            "desk": _desk_meta(kind),
            # The hand solids looked for, by their face size: what a pose of
            # one is scaled by, and what it is solved again against.
            "hands": hands,
            # where each sat on its handheld: the tool centre in its frame
            "solid_T_tcp": {name: SETTINGS.for_solid(name).solid_T_tcp.tolist() for name in hands},
            "kind": kind,
            "cameras": cameras,
            "gripper_cameras": _camera_meta(cameras),
            # Every marker the top camera was looking for: what solving the
            # poses again from the corners kept takes, with or without this code.
            "markers": _marker_meta(hands),
            # What each camera was opened as (camera.yaml's profiles).
            "camera_profiles": {role: CORE.camera_profile(role).as_dict() for role in CAMERA_ROLES},
            "streams": {},
        }
        if kind == "rollout":
            # The arms the policy drove, the policy, and what each arm's joint
            # angles mean (pinkrab_core's arm_fingerprint), by arm.
            self.meta["arms"] = arms
            self.meta["policy"] = policy
            self.meta["description"] = description
        self.event("start")

    # --- feeding ------------------------------------------------------------------
    def add_frame(self, stream: str, picture: np.ndarray, host_ts: float,
                  sensor_ts: float | None = None, frame_number: int | None = None) -> None:
        video = self._videos[stream]
        if video.path is None:
            # Kept as read, not as a picture: its line now, its size once.
            with self._lock:
                if video.size is None:
                    video.size = (picture.shape[1], picture.shape[0])
                if not self._closed:
                    self._frames.write(json.dumps(self._frame_line(video, host_ts, sensor_ts, frame_number)) + "\n")
                video.written += 1
            return
        try:
            video.queue.put_nowait((picture, host_ts, sensor_ts, frame_number))
        except queue.Full:
            with self._lock:
                video.dropped += 1

    def add_poses(self, line: dict) -> None:
        with self._lock:
            if not self._closed:
                self._poses.write(json.dumps(line) + "\n")

    def add_robot(self, samples: list[dict]) -> int:
        """Readings of the robot, for a rollout: each `{host_ts, joints, mode,
        following, controller, correction}` - `controller` one of CONTROLLERS,
        who had the arms, and `correction` the number of the take-over the
        reading is in, None outside one. Returns how many were kept - none for
        a demonstration."""
        kept = 0
        with self._lock:
            if self._closed or self._robot is None:
                return 0
            for sample in samples:
                if not isinstance(sample, dict) or not isinstance(sample.get("joints"), dict):
                    continue
                controller = sample.get("controller")
                if controller not in CONTROLLERS:
                    continue
                host_ts = float(sample["host_ts"])
                correction = sample.get("correction")
                self._robot.write(json.dumps({"host_ts": host_ts,
                                              "joints": sample["joints"],
                                              "mode": sample.get("mode"),
                                              "following": bool(sample.get("following")),
                                              "controller": controller,
                                              "correction": correction if isinstance(correction, int) else None})
                                  + "\n")
                # The time to this reading is the previous one's controller's.
                if self._last_reading is not None:
                    gap = host_ts - self._last_reading[0]
                    if 0.0 < gap <= ROBOT_GAP_S:
                        self._controlled[self._last_reading[1]] += gap
                self._last_reading = (host_ts, controller)
                kept += 1
        return kept

    def event(self, what: str, **fields) -> None:
        with self._lock:
            if not self._closed:
                self._events.write(json.dumps({"host_ts": time.time(), "monotonic": time.monotonic(),
                                               "event": what, **fields}) + "\n")

    # --- the encoder threads ---------------------------------------------------------
    def _encode(self, video: _Video) -> None:
        """Drain the video's queue into its encoder until the closing None. An
        encoder that fails (a full disk) stops writing but keeps draining, so
        the queue never fills behind a thread that is gone and the close that
        follows is never left waiting."""
        while True:
            item = video.queue.get()
            if item is None:
                break
            if video.failure is not None:
                with self._lock:
                    video.dropped += 1
                continue
            try:
                self._write(video, *item)
            except Exception as exc:  # noqa: BLE001 - the recording goes on without this video
                with self._lock:
                    video.failure = f"{type(exc).__name__}: {exc}"
                    video.dropped += 1
                print(f"  {video.stream}: video stopped - {video.failure}", flush=True)

    def _write(self, video: _Video, picture: np.ndarray, host_ts: float, sensor_ts: float | None,
               frame_number: int | None) -> None:
        if picture.ndim == 2:
            picture = cv2.cvtColor(picture, cv2.COLOR_GRAY2BGR)
        if video.writer is None:
            w, h = kept_size(picture.shape[1], picture.shape[0])
            video.size = (w, h)
            encoder = Encoder(video.path, video.fps, w, h)
            if not encoder.ok:
                with self._lock:
                    video.dropped += 1
                return
            video.writer, video.codec = encoder, encoder.codec
        if picture.shape[1] != video.size[0]:
            picture = cv2.resize(picture, video.size, interpolation=cv2.INTER_AREA)
        video.writer.write(picture)
        line = self._frame_line(video, host_ts, sensor_ts, frame_number)
        with self._lock:
            video.written += 1
            if not self._closed:
                self._frames.write(json.dumps(line) + "\n")

    @staticmethod
    def _frame_line(video: _Video, host_ts: float, sensor_ts: float | None, frame_number: int | None) -> dict:
        line = {"stream": video.stream, "index": video.written, "host_ts": host_ts}
        if sensor_ts is not None:
            line["sensor_ts"] = sensor_ts
        if frame_number is not None:
            line["frame_number"] = frame_number
        return line

    # --- status and closing -----------------------------------------------------------
    def status(self) -> dict:
        with self._lock:
            end = self.ended or time.monotonic()
            out = {
                "name": self.name,
                "seconds": round(end - self.started, 1),
                "frames": {name: v.written for name, v in self._videos.items()},
                "dropped": {name: v.dropped for name, v in self._videos.items()},
            }
            if self.kind == "rollout":
                out["controlled"] = {k: round(v, 1) for k, v in self._controlled.items()}
            return out

    def close(self) -> dict:
        self.event("stop")
        self.ended = time.monotonic()
        for video in self._videos.values():
            if video.thread is not None:
                try:
                    video.queue.put(None, timeout=30.0)
                except queue.Full:
                    video.failure = "the encoder did not finish in time"
        for video in self._videos.values():
            running = False
            if video.thread is not None:
                video.thread.join(timeout=30.0)
                running = video.thread.is_alive()
                if running:
                    video.failure = "the encoder did not finish in time"
            # An encoder still writing is not closed under its thread.
            if video.writer is not None and not running:
                try:
                    video.writer.close()
                except Exception as exc:  # noqa: BLE001 - the meta.json below is still written
                    video.failure = video.failure or f"{type(exc).__name__}: {exc}"
        with self._lock:
            self._closed = True
            self._frames.close()
            self._poses.close()
            self._events.close()
            if self._robot is not None:
                self._robot.close()
        self.meta["ended_wall"] = time.time()
        self.meta["seconds"] = round(self.ended - self.started, 3)
        if self.kind == "rollout":
            self.meta["controlled"] = {k: round(v, 3) for k, v in self._controlled.items()}
        self.meta["streams"] = {
            name: {"file": None if v.path is None else v.path.name, "fps": v.fps, "codec": v.codec, "size": v.size,
                   "frames": v.written, "dropped": v.dropped,
                   **({"failure": v.failure} if v.failure else {})}
            for name, v in self._videos.items()
        }
        # A gripper video also says where its lens axis lands in the frame,
        # which is where convert cuts it - once the lens has been measured;
        # until then convert asks the lens file when it runs.
        for name, entry in self.meta["streams"].items():
            side = name.removeprefix("gripper_")
            lens = side if self.cameras == "handheld" else f"follower_{side}"
            centre = lens_centre(lens) if name.startswith("gripper_") else None
            if centre is not None:
                entry["centre"] = list(centre)
            # And the whole fit, so the picture can be undistorted from the
            # episode alone.
            fit = lens_fit(lens) if name.startswith("gripper_") else None
            if fit is not None:
                entry["lens"] = {"camera": lens, **fit}
        (self.root / "meta.json").write_text(json.dumps(self.meta, indent=2))
        return self.status()


def _desk_meta(kind: str) -> dict:
    """Where the desk solid stands: its yaw against the robot and how far its
    origin sits above the table, as teleop places it (desk_solid.placing) -
    no yaw where desk.yaml will not read. A demonstration is marked as made
    with the solid set in front of the operator's chest, the placing
    desk_solid reckons in the robot's world, so the shelf can judge it
    there; a rollout is not - its solid may lie anywhere."""
    from .desk_solid import PlacementError, placing

    desk = SETTINGS.for_solid(SETTINGS.desk_solid)
    try:
        yaw, height = placing()
    except PlacementError:
        yaw, height = None, desk.cut_mm
    out = {"solid": SETTINGS.desk_solid, "yaw_deg": yaw, "origin_above_table_mm": height,
           "face_mm": desk.face_size_mm}
    if kind == "demo":
        out["placement"] = "chest"
    return out


def _marker_meta(hands: dict[str, float]) -> dict:
    """The ArUco dictionary and, for each solid looked for - the hands at the
    face size they were recorded at, and the desk - every marker's four
    corners in the solid's frame (metres, ArUco's corner order)."""
    from .markers import solid_object_points

    solids = {name: SETTINGS.at_face(name, mm) for name, mm in hands.items()}
    solids[SETTINGS.desk_solid] = SETTINGS.for_solid(SETTINGS.desk_solid)
    return {"dictionary": SETTINGS.aruco_dict,
            "solids": {name: {"face_mm": cfg.solid_spec.face_mm,
                              "corners": {str(i): np.round(c, 7).tolist()
                                          for i, c in solid_object_points(cfg).items()}}
                       for name, cfg in solids.items()}}


def _camera_meta(cameras: str = "handheld") -> dict:
    try:
        from .gripper_cams import mapping
        return mapping(cameras)
    except Exception:  # noqa: BLE001
        return {}


# --- the shelf ----------------------------------------------------------------------
def list_episodes(root: Path | None = None) -> list[dict]:
    root = root or SETTINGS.episodes_dir
    out = []
    if not root.is_dir():
        return out
    for path in sorted(root.iterdir(), reverse=True):
        if not path.is_dir():
            continue
        meta_file = path / "meta.json"
        entry = {"name": path.name, "complete": meta_file.is_file(), "bytes": _size_of(path)}
        if meta_file.is_file():
            try:
                meta = json.loads(meta_file.read_text())
                entry["seconds"] = meta.get("seconds")
                entry["frames"] = {k: v.get("frames") for k, v in (meta.get("streams") or {}).items()}
                entry["dropped"] = sum(v.get("dropped", 0) for v in (meta.get("streams") or {}).values())
                entry["started_wall"] = meta.get("started_wall")
                entry["kind"] = meta.get("kind", "demo")
                if entry["kind"] == "rollout":
                    entry["arms"] = meta.get("arms")
                    entry["controlled"] = meta.get("controlled")
                    entry["policy"] = meta.get("policy")
                    entry["outcome"] = meta.get("outcome")
            except (OSError, ValueError):
                entry["complete"] = False
        out.append(entry)
    return out


def delete_episode(name: str, root: Path | None = None) -> None:
    root = root or SETTINGS.episodes_dir
    if not EPISODE_NAME.match(name):
        raise EpisodeError(f"{name!r} is not an episode name")
    path = root / name
    if not path.is_dir():
        raise EpisodeError(f"no episode {name!r}")
    shutil.rmtree(path)


# What the person watching a rollout says of it. Nothing the recording holds
# tells whether the blocks went in, so it is said, not measured.
OUTCOMES = ("succeeded", "failed")


def set_outcome(name: str, outcome: str | None, root: Path | None = None) -> None:
    """A rollout's outcome written into its meta.json, or taken back (None).
    Only a closed rollout has one: its meta.json is written as it closes."""
    root = root or SETTINGS.episodes_dir
    if not EPISODE_NAME.match(name):
        raise EpisodeError(f"{name!r} is not an episode name")
    if outcome is not None and outcome not in OUTCOMES:
        raise EpisodeError(f"{outcome!r} is not an outcome ({', '.join(OUTCOMES)})")
    meta_file = root / name / "meta.json"
    if not meta_file.is_file():
        raise EpisodeError(f"no closed episode {name!r}")
    meta = json.loads(meta_file.read_text())
    if meta.get("kind") != "rollout":
        raise EpisodeError(f"{name!r} is a demonstration - only a rollout has an outcome")
    if outcome is None:
        meta.pop("outcome", None)
    else:
        meta["outcome"] = outcome
    # Written beside and moved over, so a reader never meets half a file.
    partial = meta_file.with_suffix(".json.partial")
    partial.write_text(json.dumps(meta, indent=2))
    partial.replace(meta_file)


def _size_of(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
