"""What of each gripper picture is kept when its background is randomised.

A policy learns from the gripper cameras' pictures, and everything in them
but the grippers and the task's own things - the room, the person, the
handheld's body and its marker solid - is what it should not learn from. So
each frame of each gripper video gets a label picture, at the size the
dataset keeps it (gripper_view's window about the lens axis, scaled), saying
what is kept:

  OWN    this camera's own gripper: its claw, jaw, servo and camera bracket;
  OTHER  the other hand's gripper, as the robot's other arm will show it;
  3..    the task's things, one label each (from SAM 3);
  BORDER the black outside the fisheye's picture circle, which the robot's
         camera has the same.

The grippers are drawn from geometry, not looked for: the robot's gripper
off the description (gripper_model), the camera where camera.yaml puts it
on the gripper (gripper_camera), the lens as the take recorded it (a
Kannala-Brandt fit), each jaw turned to its hand's opening at that moment,
and the other hand where the top camera placed it against this one - each
hand's tool off its solid, as convert places it. A camera sees its own
gripper the same way in every frame but for its jaw, so its own is drawn
from the jaw alone; the other is drawn only where both hands were placed.

The labels go beside the raw take, in masks/, one compressed array per
gripper video: frame by frame of the video, as convert picks its frames.
"""
from __future__ import annotations

import base64
import functools
import json
import os
import shlex
import subprocess
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from pinkrab_core.config import ARMS
from pinkrab_core.dataset import GRIPPER_WIDTH, gripper_window
from pinkrab_core.follow import TOOL_TURN
from pinkrab_core.gripper_camera import camera_spec, mirrored, tcp_T_camera
from pinkrab_core.jaw import gripper_band, shut_wide

from . import sam_track
from .backgrounds import MASKS, write
from .config import SETTINGS
from .convert import cameras_on_desk, exposures, fill_holes, placed
from .gripper_model import Gripper, gripper

OWN, OTHER = 1, 2
FIRST_THING = 3
BORDER = 255
# The lens border is outside the picture circle: a circle fitted to where,
# looking out from the middle of the picture, it turns darker than
# BORDER_DARK (0..255) for good - in every one of BORDER_SAMPLES frames
# spread over the take - one ray every BORDER_RAY_DEG. A ray that turns dark
# short of the circle, on something dark in the room, is left out of the
# fit when it misses the circle by more than BORDER_FIT_PX.
BORDER_DARK = 40
BORDER_SAMPLES = 40
BORDER_RAY_DEG = 2
BORDER_FIT_PX = 4.0
# A triangle with a corner nearer the lens than this, or behind it, is not
# drawn: the fisheye model folds what is behind back into the picture.
NEAREST_M = 0.002
# The meshes are drawn with corners this near each other made one: the
# servo and the claw are tens of thousands of triangles, the picture needs
# a few thousand.
CELL_M = 0.0015
# A thing whose box comes within this of the picture's edge may have left the
# view; one gone while clear of it was lost by the tracker.
EDGE_PX = 8
# A thing's area grown or shrunk more than this from one frame to the next is
# worth a look: another thing taken for it, or half of it dropped.
JUMP = 3.0
# A thing gone within this of where a gripper is drawn went behind it.
BEHIND_PX = 12
# A gripper frame this far outside the head camera's frames takes the hands
# from the nearest one: about a frame at 30 fps.
EDGE_S = 0.04


def decimated(tris: np.ndarray, cell: float = CELL_M) -> np.ndarray:
    """Triangles (n,3,3) with corners within `cell` of each other made one,
    the triangles that leaves flat dropped."""
    corners = tris.reshape(-1, 3)
    keys, which = np.unique(np.floor(corners / cell).astype(np.int64), axis=0, return_inverse=True)
    which = which.ravel()
    verts = np.zeros((len(keys), 3))
    np.add.at(verts, which, corners)
    verts /= np.bincount(which)[:, None]
    faces = which.reshape(-1, 3)
    faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])]
    faces = np.unique(np.sort(faces, axis=1), axis=0)
    return verts[faces]


class Lens:
    """A gripper camera as the take recorded it, and the dataset's window
    of its pictures: points in the camera's frame to the window's pixels."""

    def __init__(self, stream: dict) -> None:
        lens = stream["lens"]
        if lens.get("model") != "kannala_brandt":
            raise ValueError(f"a {lens.get('model')} lens is not drawn here")
        self.K = np.array([[lens["fx"], 0.0, lens["cx"]], [0.0, lens["fy"], lens["cy"]], [0.0, 0.0, 1.0]])
        self.D = np.array(lens["k"], dtype=float).reshape(4, 1)
        cols, rows = lens["size"]
        self.x, self.y, w, h = gripper_window(cols, rows, tuple(stream["centre"]))
        self.scale = GRIPPER_WIDTH / w
        self.size = (round(h * self.scale), GRIPPER_WIDTH)   # (height, width), as gripper_view

    def draw(self, canvas: np.ndarray, tris_cam: np.ndarray, label: int) -> None:
        """Fill `tris_cam` (n,3,3, the camera's frame) on `canvas` as `label`."""
        tris = tris_cam[(tris_cam[:, :, 2] > NEAREST_M).all(axis=1)]
        if not len(tris):
            return
        px, _ = cv2.fisheye.projectPoints(tris.reshape(-1, 1, 3), np.zeros((3, 1)), np.zeros((3, 1)), self.K, self.D)
        px = (px.reshape(-1, 3, 2) - [self.x, self.y]) * self.scale
        cv2.fillPoly(canvas, list(np.round(px).astype(np.int32)), label)


def _moved(tris: np.ndarray, T: np.ndarray) -> np.ndarray:
    return tris @ T[:3, :3].T + T[:3, 3]


def _jaw_angle(side: str, opening: float | None) -> float:
    shut, wide = shut_wide(*gripper_band(side))
    return shut + (0.0 if opening is None else float(opening)) * (wide - shut)


@functools.lru_cache(maxsize=1)
def grippers() -> tuple[dict[str, Gripper], dict[str, np.ndarray]]:
    """Each side's robot gripper, its meshes thinned, and each side's camera
    in its gripper link's frame - read once a process: the description does
    not change under a running studio."""
    from pinkrab_core.kinematics import fresh_urdf

    urdf = ET.parse(fresh_urdf()).getroot()
    out = {}
    for side in ARMS:
        g = gripper(urdf, side)
        g.fixed = {name: decimated(tris) for name, tris in g.fixed.items()}
        g.jaw = decimated(g.jaw)
        out[side] = g
    spec = camera_spec()
    right = out["right"].link_T_tcp @ tcp_T_camera(spec["offset_out"], spec["offset_back"], spec["pitch"], spec["yaw"])
    return out, {"right": right, "left": mirrored(right)}


def _lerp_pose(a: np.ndarray, b: np.ndarray, w: float) -> np.ndarray:
    out = np.eye(4)
    out[:3, 3] = a[:3, 3] + w * (b[:3, 3] - a[:3, 3])
    turn = Rotation.from_matrix([a[:3, :3], b[:3, :3]])
    out[:3, :3] = (turn[0] * Rotation.from_rotvec(w * (turn[0].inv() * turn[1]).as_rotvec())).as_matrix()
    return out


def _at(tools: list[np.ndarray | None], times: np.ndarray, t: float) -> np.ndarray | None:
    """A hand's tool at moment `t`, between the head frames either side of
    it, or None where either has no place for the hand. A moment up to
    EDGE_S before the first head frame or after the last - a gripper frame
    taken as the head camera started or stopped - has the hand where that
    frame put it."""
    i = int(np.searchsorted(times, t))
    if i == 0:
        return tools[0] if len(times) and times[0] - t <= EDGE_S else None
    if i >= len(times):
        return tools[-1] if t - times[-1] <= EDGE_S else None
    a, b = tools[i - 1], tools[i]
    if a is None or b is None or times[i] <= times[i - 1]:
        return None
    return _lerp_pose(a, b, (t - times[i - 1]) / (times[i] - times[i - 1]))


class Scene:
    """A take's grippers, ready to be drawn into any frame of its gripper
    videos: each hand's tool, the robot's tool frame, in the desk's at each
    head frame - as convert places it - and each jaw's opening at any moment."""

    def __init__(self, root: Path) -> None:
        meta = json.loads((root / "meta.json").read_text())
        poses = [json.loads(line) for line in (root / "poses.jsonl").read_text().splitlines() if line.strip()]
        frames = [json.loads(line) for line in (root / "frames.jsonl").read_text().splitlines() if line.strip()]
        streams = meta.get("streams") or {}
        self.sides = [side for side in ARMS if f"gripper_{side}" in streams]
        self.models, self.link_T_cam = grippers()
        self.times = exposures(poses)
        world_T_cams = cameras_on_desk(poses, ARMS)
        self.tools, self.openings = {}, {}
        for side in ARMS:
            solid_T_tcp = np.array(SETTINGS.recorded_solid_T_tcp(f"handheld_{side}", meta), dtype=float)
            hand = [W @ np.array(p["solids"][f"handheld_{side}"]["pose"]) @ solid_T_tcp
                    if W is not None and placed(p, side) else None for p, W in zip(poses, world_T_cams)]
            turn = np.eye(4)
            turn[:3, :3] = TOOL_TURN[side]
            self.tools[side] = [None if T is None else T @ turn
                                for T in fill_holes(hand, self.times, SETTINGS.convert_max_hole_s)]
            readings = [(p["host_ts"] - SETTINGS.jaw_latency_s, p["jaw"][side])
                        for p in poses if (p.get("jaw") or {}).get(side) is not None]
            self.openings[side] = tuple(np.array(v, dtype=float) for v in zip(*readings)) if readings else None
        self.lenses = {side: Lens(streams[f"gripper_{side}"]) for side in self.sides}
        # Each gripper frame's moment, on the head camera's clock.
        self.stamps = {side: [f["host_ts"] - SETTINGS.gripper_camera_latency_s
                              for f in frames if f["stream"] == f"gripper_{side}"] for side in self.sides}

    def _opening(self, side: str, t: float) -> float | None:
        o = self.openings[side]
        return None if o is None else float(np.interp(t, *o))

    def draw(self, side: str, k: int) -> np.ndarray:
        """Frame `k` of `side`'s gripper video's labels, (height, width)
        uint8: OWN and OTHER where the grippers are, 0 elsewhere. A camera
        sees its own gripper the same way in every frame but for its jaw; the
        other is drawn only where both hands were placed."""
        models, link_T_cam = self.models, self.link_T_cam
        other = next(s for s in ARMS if s != side)
        lens = self.lenses[side]
        t = self.stamps[side][k]
        labels = np.zeros(lens.size, dtype=np.uint8)
        mine = _at(self.tools[side], self.times, t)
        theirs = _at(self.tools[other], self.times, t)
        if mine is not None and theirs is not None:
            desk_T_cam = mine @ np.linalg.inv(models[side].link_T_tcp) @ link_T_cam[side]
            cam_T_other = np.linalg.inv(desk_T_cam) @ theirs @ np.linalg.inv(models[other].link_T_tcp)
            lens.draw(labels, _moved(models[other].triangles(_jaw_angle(other, self._opening(other, t))),
                                     cam_T_other), OTHER)
        lens.draw(labels, _moved(models[side].triangles(_jaw_angle(side, self._opening(side, t))),
                                 np.linalg.inv(link_T_cam[side])), OWN)
        return labels


def geometry(root: Path) -> dict[str, np.ndarray]:
    """Each gripper video's labels for its frames, (n, height, width) uint8:
    OWN and OTHER where the grippers are, 0 elsewhere."""
    scene = Scene(root)
    return {f"gripper_{side}": np.stack([scene.draw(side, k) for k in range(len(scene.stamps[side]))])
            for side in scene.sides}


def picture(root: Path, name: str, k: int, size: tuple[int, int]) -> np.ndarray:
    """Frame `k` of a gripper video as the dataset keeps it, BGR."""
    from pinkrab_core.dataset import gripper_view

    stream = json.loads((root / "meta.json").read_text())["streams"][name]
    cap = cv2.VideoCapture(str(root / f"{name}.mp4"))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        raise ValueError(f"{name} has no frame {k}")
    return gripper_view(frame, size, tuple(stream["centre"]))


@dataclass(frozen=True)
class Remote:
    """SAM 3 on another machine, reached over ssh: `ssh` is the command up to
    and including user@host, `root` the directory there that holds SAM 3's
    environment (sam3_env), sam_track.py and the weights (hf-cache)."""
    name: str
    ssh: tuple[str, ...]
    root: str

    def path(self, *parts: str) -> str:
        """A path under root, for the remote shell: a root under ~ left to
        that shell to expand."""
        rest = "/".join((self.root, *parts))
        if rest == "~" or rest.startswith("~/"):
            return '"$HOME"' + (shlex.quote(rest[1:]) if len(rest) > 1 else "")
        return shlex.quote(rest)

    def env(self, offline: bool = True) -> str:
        """What SAM 3 runs with there: nothing of the login's PYTHONPATH, and
        the weights kept under root, where training keeps its own (a cloud
        box's home goes with its session). Offline once they are there: the
        weights are gated, and asked for again online they would want a
        Hugging Face login the machine may no longer have."""
        return (f"env -u PYTHONPATH HF_HUB_CACHE={self.path('hf-cache', 'hub')}"
                + (" HF_HUB_OFFLINE=1" if offline else ""))


class Sam:
    """SAM 3 in its own environment (sam3_python), kept running with its
    model loaded for one request after another: sam_track --serve. Loading
    the model is ten seconds or so; a job over many videos, and a page
    looking at frame after frame, load it once. A request that is given up
    on - its asker stopped, or it went unanswered for `timeout_s` - stops
    the process with it, and the next starts it again.

    With `remote`, it runs on that machine over ssh: this checkout's
    sam_track.py is put there each time it starts, and the files a request
    names go in its line and come back in the answer's."""

    def __init__(self, timeout_s: float = 180.0, remote: Remote | None = None) -> None:
        self.timeout_s = timeout_s
        self.remote = remote
        self._run: subprocess.Popen | None = None
        self._err = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        if self._run is not None:
            if self._run.poll() is None:
                self._run.kill()
            self._run.wait()
            self._run = None
        if self._err is not None:
            self._err.close()
            self._err = None

    def _argv(self) -> list[str]:
        script = Path(sam_track.__file__)
        if self.remote is None:
            python = SETTINGS.sam3_python
            if not python.is_file():
                raise FileNotFoundError(f"no SAM 3 environment at {python} - see the docs' install page")
            return [str(python), "-I", str(script), "--serve"]
        r = self.remote
        sent = subprocess.run([*r.ssh, f"mkdir -p {r.path()} && cat > {r.path('sam_track.py')}"],
                              input=script.read_bytes(), capture_output=True, timeout=60, check=False)
        if sent.returncode != 0:
            said = sent.stderr.decode("utf-8", "replace").strip().splitlines()
            raise RuntimeError(f"{r.name}: {said[-1] if said else f'ssh exited {sent.returncode}'}")
        # exec: the ssh session is SAM 3 itself, and ends with it.
        command = (f"exec {r.env()} {r.path(SETTINGS.sam3_env, 'bin', 'python')} -I "
                   f"{r.path('sam_track.py')} --serve")
        return [*r.ssh, command]

    def _started(self) -> subprocess.Popen:
        if self._run is not None and self._run.poll() is None:
            return self._run
        self.close()
        argv = self._argv()
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        # Its warnings and progress bars go to a file: a pipe nobody reads
        # while stdout is being read fills and stops it.
        self._err = tempfile.TemporaryFile("w+")  # noqa: SIM115 - it lives as long as the process
        self._run = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._err,
                                     text=True, env=env, bufsize=1)
        return self._run

    def _line(self, run: subprocess.Popen) -> dict:
        """The next thing it says, waited for no longer than timeout_s."""
        answer: list[str] = []
        reader = threading.Thread(target=lambda: answer.append(run.stdout.readline()), daemon=True)
        reader.start()
        reader.join(self.timeout_s)
        if not answer or not answer[0]:
            self._err.seek(0)
            last = self._err.read().strip().splitlines()
            where = f" on {self.remote.name}" if self.remote else ""
            raise RuntimeError(f"SAM 3{where} stopped: {last[-1] if last else 'no answer'}")
        try:
            return json.loads(answer[0])
        except ValueError:
            return {}

    def _carried(self, request: dict) -> tuple[dict, str | None]:
        """`request` as sent to another machine: the files it names carried
        in it, and its out left for the answer to bring back."""
        if self.remote is None:
            return request, None
        request = dict(request)
        for key in ("video", "image"):
            if key in request:
                request[f"{key}_data"] = base64.b64encode(Path(request.pop(key)).read_bytes()).decode()
        return request, request.pop("out", None)

    def ask(self, request: dict, say=None) -> dict:
        """`request` answered: every line it says on the way handed to
        `say`, and its last - done, ok or error - returned."""
        run = self._started()
        request, out = self._carried(request)
        try:
            run.stdin.write(json.dumps(request) + "\n")
            run.stdin.flush()
            while True:
                line = self._line(run)
                if "error" in line or "ok" in line or (line.get("done") and "thing" not in line):
                    if out is not None and "data" in line:
                        Path(out).write_bytes(base64.b64decode(line.pop("data")))
                    return line
                if say is not None and line:
                    say(**line)
        except BaseException:
            # Halfway through a request, the process cannot be asked anything else.
            self.close()
            raise


def tracked(root: Path, name: str, things: list[str], sam: Sam, say=lambda **_: None) -> np.ndarray:
    """A gripper video's labels for the task's `things`, by words - each the
    label FIRST_THING + its place in the list where SAM 3 found it - from
    `sam`. `say` is told each line of its progress; an exception out of it
    stops SAM 3 too."""
    meta = json.loads((root / "meta.json").read_text())
    stream = meta["streams"][name]
    lens = Lens(stream)
    # The window in the video's own frame, which may be a scaled part of the
    # sensor the lens was fitted on.
    window = list(gripper_window(*stream["size"], tuple(stream["centre"])))
    with tempfile.TemporaryDirectory() as tmp:
        spec = {"video": str(root / f"{name}.mp4"), "window": window, "size": list(lens.size),
                "resolution": SETTINGS.sam3_resolution, "new_thing_score": SETTINGS.sam3_new_thing_score,
                "thing_score": SETTINGS.sam3_thing_score, "out": str(Path(tmp) / "labels.npz"),
                "things": [{"label": FIRST_THING + k, "text": text} for k, text in enumerate(things)]}
        answer = sam.ask(spec, say)
        if "error" in answer:
            raise RuntimeError(f"SAM 3 stopped on {name}: {answer['error']}")
        return np.load(Path(tmp) / "labels.npz")["labels"]


def border(root: Path, name: str, size: tuple[int, int]) -> np.ndarray:
    """Where a gripper video, kept at `size` as the dataset keeps it, never
    sees anything: the black outside the fisheye's picture circle. Found
    from the take itself, as a circle, so something dark in the room at the
    picture's edge is not taken for it."""
    from pinkrab_core.dataset import gripper_view

    stream = json.loads((root / "meta.json").read_text())["streams"][name]
    cap = cv2.VideoCapture(str(root / f"{name}.mp4"))
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        brightest = np.zeros(size, dtype=np.uint8)
        for i in np.unique(np.linspace(0, max(0, n - 1), BORDER_SAMPLES).astype(int)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, picture = cap.read()
            if ok:
                view = gripper_view(picture, size, tuple(stream["centre"]))
                np.maximum(brightest, cv2.cvtColor(view, cv2.COLOR_BGR2GRAY), out=brightest)
    finally:
        cap.release()
    dark = brightest < BORDER_DARK
    height, width = size
    middle = np.array([(width - 1) / 2, (height - 1) / 2])
    # Along each ray out of the middle, where the picture turns dark and
    # stays dark to the edge of the window.
    points = []
    for angle in np.radians(np.arange(0, 360, BORDER_RAY_DEG)):
        step = np.array([np.cos(angle), np.sin(angle)])
        reach = np.arange(0, int(np.hypot(width, height)))
        xy = np.round(middle + reach[:, None] * step).astype(int)
        inside = (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
        xy = xy[inside]
        if not len(xy):
            continue
        along = dark[xy[:, 1], xy[:, 0]]
        if not along[-1]:
            continue   # the picture reaches the window's edge here: no border on this ray
        lit = np.nonzero(~along)[0]
        first = lit[-1] + 1 if len(lit) else 0
        points.append(xy[first])
    if len(points) < 8:
        return np.zeros(size, dtype=bool)
    points = np.array(points, dtype=float)
    keep = np.ones(len(points), dtype=bool)
    for _ in range(3):
        # A circle through the points (x² + y² = 2ax + 2by + c), least squares.
        x, y = points[keep, 0], points[keep, 1]
        a, b, c = np.linalg.lstsq(np.stack([2 * x, 2 * y, np.ones_like(x)], axis=1), x * x + y * y, rcond=None)[0]
        radius = np.sqrt(c + a * a + b * b)
        miss = np.abs(np.hypot(points[:, 0] - a, points[:, 1] - b) - radius)
        keep = miss <= max(BORDER_FIT_PX, float(np.median(miss)))
    ys, xs = np.mgrid[0:height, 0:width]
    return np.hypot(xs - a, ys - b) > radius


def flags(labels: np.ndarray, things: list[str], grippers: np.ndarray | None = None) -> list[dict]:
    """Where a video's tracking is worth a look, as spans of frames: a thing
    gone from one frame to the next while it was clear of the picture's
    edge - it did not leave the view, it was lost - or its area grown or
    shrunk more than JUMP times from one frame to the next. With `grippers`
    (the drawn labels, frame by frame), a thing gone or shrunk where a
    gripper now covers it went behind the gripper, and one grown where a
    gripper covered it came out from behind it: neither is flagged."""
    near = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * BEHIND_PX + 1, 2 * BEHIND_PX + 1))
    out = []
    height, width = labels.shape[1:]
    for k, text in enumerate(things):
        label = FIRST_THING + k
        area = np.zeros(len(labels))
        inside = np.zeros(len(labels), dtype=bool)
        for i, frame in enumerate(labels):
            ys, xs = np.nonzero(frame == label)
            area[i] = len(ys)
            if len(ys):
                inside[i] = (ys.min() > EDGE_PX and xs.min() > EDGE_PX
                             and ys.max() < height - 1 - EDGE_PX and xs.max() < width - 1 - EDGE_PX)
        bad = np.zeros(len(labels), dtype=bool)
        why = [""] * len(labels)

        def behind(i: int, label: int = label) -> bool:
            """Whether what of the thing went from frame i-1 to i went behind a gripper."""
            if grippers is None:
                return False
            was = cv2.dilate((labels[i - 1] == label).astype(np.uint8), near).astype(bool)
            return bool((grippers[i][was] != 0).any())

        def out_from_behind(i: int, label: int = label) -> bool:
            """Whether what of the thing came into frame i came out from behind a gripper."""
            if grippers is None:
                return False
            now = cv2.dilate((labels[i] == label).astype(np.uint8), near).astype(bool)
            return bool((grippers[i - 1][now] != 0).any())

        for i in range(1, len(labels)):
            if area[i] == 0 and area[i - 1] > 0 and inside[i - 1] and not behind(i):
                bad[i], why[i] = True, "lost"
            elif (area[i] > 0 and area[i - 1] > 0 and max(area[i] / area[i - 1], area[i - 1] / area[i]) > JUMP
                  and not (behind(i) if area[i] < area[i - 1] else out_from_behind(i))):
                bad[i], why[i] = True, "jumped"
        # A thing lost stays lost until it is found again.
        for i in range(1, len(labels)):
            if not bad[i] and why[i - 1] == "lost" and area[i] == 0:
                bad[i], why[i] = True, "lost"
        i = 0
        while i < len(labels):
            if bad[i]:
                j = i
                while j + 1 < len(labels) and bad[j + 1] and why[j + 1] == why[i]:
                    j += 1
                out.append({"thing": text, "label": label, "start": i, "end": j + 1, "why": why[i]})
                i = j + 1
            else:
                i += 1
    return out


def make(root: Path, things: list[str], say=lambda **_: None, sam: Sam | None = None) -> dict:
    """A take's labels for every gripper video - the task's `things` tracked,
    the grippers drawn over them - written beside it, with what each label
    is, how many frames each thing was found in and where to look again
    (masks.json). Returns that account. `sam` is SAM 3 already running, for
    a job over many takes; without it, one is started for this take."""
    if sam is None:
        with Sam() as own:
            return make(root, things, say, own)
    drawn = geometry(root)
    found, flagged = {}, {}
    for name, labels in drawn.items():
        tracks = tracked(root, name, things, sam, lambda video=name, **line: say(video=video, **line)) if things \
            else np.zeros_like(labels)
        if tracks.shape != labels.shape:
            raise ValueError(f"{name}: SAM 3 read {len(tracks)} frames, the take lists {len(labels)}")
        flagged[name] = flags(tracks, things, labels)
        found[name] = {text: int((tracks == FIRST_THING + k).any(axis=(1, 2)).sum()) for k, text in enumerate(things)}
        tracks[labels != 0] = labels[labels != 0]
        tracks[border(root, name, tracks.shape[1:])[None] & (tracks == 0)] = BORDER
        drawn[name] = tracks
    folder = write(root, drawn)
    account = {"labels": {"0": "background", str(OWN): "own gripper", str(OTHER): "other gripper",
                          str(BORDER): "lens border",
                          **{str(FIRST_THING + k): text for k, text in enumerate(things)}},
               "things": things, "frames": {name: len(labels) for name, labels in drawn.items()},
               "found": found, "flags": flagged, "resolution": SETTINGS.sam3_resolution, "made": time.time()}
    (folder / "masks.json").write_text(json.dumps(account, indent=2) + "\n")
    return account


def account(root: Path) -> dict | None:
    """What a take's labels are (masks.json), or None where it has none."""
    path = root / MASKS / "masks.json"
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


# What is painted over shows this dark in a review, and what is kept is
# ringed in the studio's pink (BGR).
DIM = 0.3
RING_BGR = (142, 72, 226)


def review(root: Path, name: str) -> Path:
    """A gripper video as its labels leave it, for looking over: what is kept
    ringed and the rest darkened. Made on first ask and kept beside the
    labels until they are made again."""
    from pinkrab_core.dataset import gripper_view

    from .backgrounds import kept, read
    from .episode import Encoder

    labels_path = root / MASKS / f"{name}.npz"
    if not labels_path.is_file():
        raise FileNotFoundError(f"{root.name} has no labels for {name}")
    out = root / MASKS / f"review_{name}.mp4"
    if out.is_file() and out.stat().st_mtime >= labels_path.stat().st_mtime:
        return out
    labels = read(root, name)
    stream = json.loads((root / "meta.json").read_text())["streams"][name]
    height, width = labels.shape[1:]
    cap = cv2.VideoCapture(str(root / f"{name}.mp4"))
    partial = out.with_suffix(".part.mp4")
    encoder = Encoder(partial, float(stream.get("fps") or 30), width, height)
    try:
        for frame in labels:
            ok, picture = cap.read()
            if not ok:
                break
            view = gripper_view(picture, (height, width), tuple(stream["centre"]))
            keep = kept(frame)
            shown = (view * DIM).astype(np.uint8)
            shown[keep] = view[keep]
            rings, _ = cv2.findContours(keep.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(shown, rings, -1, RING_BGR, 2)
            encoder.write(shown)
    finally:
        cap.release()
        encoder.close()
    partial.replace(out)
    return out
