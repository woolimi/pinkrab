"""Serve the Pinkrab studio: a one-page three.js view of the robot.

Why a browser and not RViz: the arms are attached to a laptop that has no ROS, and
the rig itself needs no display - the page opens from whatever machine is nearby.

What it does today is show the robot: the description, posed by sliders, by dragging
a handle, or by whatever the servos currently report.

Under it is the MuJoCo scene - stepped headless in a thread here, with the page as
its only window. The sliders ask the simulated servos for a pose rather than placing
the arm, so the desk, the board, the other arm and the force limit all get a say in
where it ends up.

MuJoCo is therefore required, not optional: a page that tested contact itself would
give a move two meanings, and would know nothing about the arms meeting each other.

Switched to the real robot on the page, the simulated arms are stood
where the real ones are, reading by reading, so the picture is the arms on the wall.
A command has nothing to move while that is so - the next reading puts the arm back
where the metal is. The page still carries a third number per joint: where the
metal is, beside what the model made of it.

Only the standard library. three.js is vendored beside this file rather than pulled
from a CDN, so a rig without internet still comes up and nothing here depends on
another host staying reachable.
"""

from __future__ import annotations

import errno
import json
import math
import signal
import threading
import time
import urllib.error
import urllib.request
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from pinkrab_core import inference, origin, workspace
from pinkrab_core.config import ARMS, SETTINGS
from pinkrab_core.desk import STANDARD_YAW_DEG
from pinkrab_core.follow import TOOL_TURN
from pinkrab_core.jaw import gripper_band
from pinkrab_core.urdf import DescriptionError, expand_for_meshes
from pinkrab_mujoco import physics as physics_module
from pinkrab_umi.config import SETTINGS as UMI
from pinkrab_umi.episode import EPISODE_NAME, EpisodeError, delete_episode
from pinkrab_umi.desk_solid import PlacementError, chest_x, solid_in_world, world_T_desk
from pinkrab_umi.markers import solid_stl, face_poses, marker_modules

from . import replay, wsock
from .clutch import Clutch
from .dagger import STOP_TIMEOUT_S, Dagger
from .converting import Converter
from .masking import Looker, Machines, Masker
from .feeds import Feeds
from .feeds import serve as serve_page
from .policy import PolicyError, PolicyRun, checkpoint_hands, checkpoints
from .real import MODE_HANDHELD, MODE_IDLE, MODE_POLICY, MODE_STUDIO, RealRobot
from .rig import Rig, RigError, live_request, live_stats, live_url, node_already_up
from .launching import DEFAULTS as LAUNCH_DEFAULTS
from .launching import RECIPES, Jobs, LaunchError, image_shape, save_key
from .training import TrainError, TrainWatch
from .sim_handheld import SimHandheld
from .verdict import Verdicts

STATIC = Path(__file__).parent / "static"

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json",
    ".svg": "image/svg+xml",
    ".stl": "model/stl",
}


# The pages behind the front page, by path.
PAGES = {
    "/teleop": "index.html",
    "/collect": "handheld.html",
    "/curate": "curate.html",
    "/handheld": "handheld.html",
    "/policy": "policy.html",
    "/train": "train.html",
}
# The ones that drive an arm, opened only once sim or real is chosen. Collecting
# is a hand and cameras, training is files, and the policy page chooses real
# itself with its Connect: none of them needs the choice made first.
ARM_PAGES = {"/teleop"}


# What the live view can be asked to open again, one thing at a time.
RETRIES = {f"/rig/live/{thing}/retry" for thing in
           ("head", "gripper/left", "gripper/right", "jaw/left", "jaw/right")}


class StudioState:
    """What the page needs, gathered once and refreshed on request.

    The URDF is expanded here rather than in the browser: xacro is a python tool and
    the page should not have to know that the description is a template.
    """

    def __init__(
        self,
        xacro_path: Path,
        mesh_dir: Path,
        scene: Path,
        mount: Path | None = None,
    ):
        self.xacro_path = xacro_path
        self.mesh_dir = mesh_dir
        self.mount = mount
        # What the bridge says about the metal, or an empty holder when nobody has
        # pointed this studio at a robot. The page is told which, because a picture
        # of a robot and a picture of a robot that is also moving on a desk beside
        # you must not look the same. The bus is not opened here either way - one
        # process can hold a serial port, and that one is the driver.
        self.real = RealRobot(expected=False)
        # Whether sim or real has been chosen since this server started - on
        # the front page, or by the policy page's Connect or Disconnect. Until
        # it has, the teleop page has no robot to show.
        self.mode_chosen = False
        # The walk just asked for - its request id and last pose - waiting for
        # the bridge to start it.
        self._walk_target: tuple[int, dict[str, float]] | None = None
        self._urdf: str | None = None
        self._stamp: tuple[float, ...] | None = None
        self._lock = threading.Lock()
        # The walk target alone: the main lock is held through a URDF expansion.
        self._walk_lock = threading.Lock()
        # Which camera pair the live view shows, and who reads it: held from
        # a check to the act it allows - a pair switched, a socket chosen, a
        # policy or an episode started - so another cannot slip between.
        self.cameras_lock = threading.Lock()
        # Built here rather than on the first request, so a scene that will not
        # compile is reported to the terminal that started the server rather than to
        # whoever opens the page next.
        self.physics = physics_module.Physics(scene)
        # Where the tool can grasp on the desk: a minute's solve, started now on
        # a thread so the switch finds it ready, and kept on disk after.
        self.reach = workspace.DeskReach()
        desk_z, board_x = self._desk_and_board()
        self.reach.warm(desk_z, board_x)
        # The top camera's live view and the teleop loop, as processes the page
        # switches on and off. See rig.py.
        self.rig = Rig()
        # The clutch: while armed, pedal 2 held is an episode recording. See clutch.py.
        self.clutch = Clutch(self.cameras_lock)
        self.verdicts = Verdicts()
        # Conversions run here, one at a time, on a thread: the page asks and watches.
        self.converter = Converter()
        # A trained checkpoint, asked about the live pictures. Nothing is driven
        # from here - see policy.py.
        self.policy = PolicyRun()
        # A policy's run recorded whole, and its arm taken over by the
        # handheld to put it right - see dagger.py.
        self.dagger = Dagger(self.real, self.policy,
                             teleop_up=lambda: node_already_up("pinkrab_teleop", "teleop"),
                             halt=self.halt_policy, walk=self.walk)
        # The last stop's close of its recording, which the live view waits on.
        self._halting: threading.Thread | None = None
        # Training runs, read from this machine and from any ssh source into one
        # sqlite file on a thread of its own - see training.py.
        self.training = TrainWatch()
        # Starting, stopping and resuming runs on those machines, and copying
        # their checkpoints home - see launching.py.
        self.train_jobs = Jobs(self.training.store, on_change=self.training.now)
        self.training.after_sync.append(self.train_jobs.fetch_due)
        # Takes' keep labels, made here one job at a time, for the Curate page,
        # with SAM 3 on this machine or one of the training sources.
        self.machines = Machines(self.training.store)
        self.masker = Masker(self.machines)
        self.looker = Looker(self.machines)
        # Set when the server is coming down, so the open event streams let go
        # instead of writing into a socket nobody is reading.
        self.stopping = threading.Event()
        # The handheld following the simulation, the teleop node's work when
        # there is no metal and so no node - see sim_handheld.py.
        self.sim_hand = SimHandheld(self.physics, self.stopping)
        # What the pages watch, each topic asked of the above at its own pace
        # and sent down every page's socket when it changes.
        self.feeds = Feeds(self)

    def halt_policy(self) -> dict:
        """Stop, not limp: idle at the gate, so nothing more passes and the
        servos keep holding where they are; then the node and the asking loop
        go, and the run's recording is kept - on a thread, since closing its
        videos takes seconds and the stop must not wait on it."""
        halt = self.dagger.halt_begun()
        queued = self.real.request("mode", mode=MODE_IDLE)
        self.policy.stop_quietly()
        self.policy.end_driven()
        node = self.rig.stop_policy()
        self._halting = threading.Thread(target=self._after_halt, args=(halt,), name="dagger-stop", daemon=True)
        self._halting.start()
        return {"ok": True, "queued": queued, "node": node, "policy": self.policy.state()}

    def halted(self) -> None:
        """Once the last stop's recording is closed: the live view it records
        from is not put down under it, or the episode is left without its
        meta.json - the page's Disconnect stops the run and then the view."""
        if self._halting is not None:
            self._halting.join(STOP_TIMEOUT_S + 1.0)

    def _after_halt(self, halt: tuple[int, str | None]) -> None:
        """Once no order of a take-over is on its way - one may have asked the
        gate for a mode around the stop's idle - idle again, unless a new run
        has been started since; then the run's recording closed and kept."""
        def idle() -> None:
            if not self.policy.state()["driving"]:
                self.real.request("mode", mode=MODE_IDLE)
        self.dagger.stop(halt, settle=idle)

    def walk(self, poses: list[dict[str, float]]) -> int | None:
        """Ask the bridge to walk the arms through `poses`; the last one becomes the page's target.

        The walk is the bridge's, but what the page shows as asked for is this
        server's target; left alone it would still show whatever was asked
        before the walk. It is set once the bridge reports the walk under way,
        not now: the bridge reads requests a few times a second but the target
        continuously, and a target that reached it before the walk would be
        taken as an order to go straight there at the gate's speed - past any
        pose the walk goes through. And it waits for THIS walk: a walk already
        under way when this one was asked reports walking too. Once walking,
        the bridge takes a target equal to the pose it walks to as the same
        order, not a new one. A walk not queued - the page is on the
        simulation - leaves no target to set: None is also what the bridge
        reports when it walks nothing.
        """
        queued = self.real.request("goto", poses=poses, speed_dps=float(SETTINGS.park_speed_dps))
        if queued is not None:
            with self._walk_lock:
                self._walk_target = (queued, dict(poses[-1]))
        return queued

    def walk_reported(self, walk_id) -> None:
        """The bridge says which walk it is on: once it is the one asked last, set the target."""
        # Compared and cleared together: a walk asked meanwhile, on another
        # thread, puts its own target in and must not lose it.
        with self._walk_lock:
            target = self._walk_target
            if walk_id is None or not target or walk_id != target[0]:
                return
            self._walk_target = None
        self.physics.command(target[1])

    def set_real(self, on: bool) -> None:
        """Point the page at the metal, or back at the simulation.

        Chosen on the page, not on the command line: the same studio serves a
        desk with no robot and a desk with one, and which it is can change
        without restarting the server. With the metal, each reading places the
        model and runs the kinematics; stepping on top of that would only add
        a tremor, so the physics is placed and not stepped.
        """
        if on != self.real.expected:
            # A walk waited for was asked of the other mode. (Its queued
            # requests need no clearing: the simulation queues only stops, and
            # forget() below keeps only those on the way back.)
            with self._walk_lock:
                self._walk_target = None
        self.real.expected = on
        self.mode_chosen = True
        if on:
            self.sim_hand.set(MODE_IDLE)
        self.real.listener = self._arms_moved if on else None
        self.physics.placed_only = on
        if not on:
            self.real.forget()

    def start_live(self, options: dict) -> str | None:
        """Start the live view with these options (its pair named), or say
        why not. A policy is shown the follower pair and nothing else, whether
        the view is up now or comes up after it died; other options put a
        running view down first, which `live_held` may forbid."""
        with self.cameras_lock:
            policy = self.policy.state()
            if (policy["running"] or policy["driving"]) and options["cameras"] != "follower":
                return "a policy is reading the live view's cameras - halt it first"
            live = self.rig.state()["live"]
            if live["running"] and live["options"] != options:
                held = self.live_held()
                if held:
                    return held
            self.rig.start("live", options)
            return None

    def live_held(self) -> str | None:
        """Why the running live view must not be put down on the way to
        something else - a switch to the other camera pair, a socket chosen -
        or None. A policy reads its pictures and an episode is written from
        them; a view that does not answer is not known to be doing neither.
        A Disconnect is not asked here: it is the operator's own. Asked under
        `cameras_lock`."""
        policy = self.policy.state()
        if policy["running"] or policy["driving"]:
            return "a policy is reading the live view's cameras - halt it first"
        stats = live_stats()
        if not stats:
            return "the live view is not answering - try again"
        if stats["episode"].get("active"):
            return "an episode is recording - stop it first"
        return None

    def policy_unseeing(self) -> str | None:
        """Why a policy may not be started now, or None: it is shown the
        follower arms' cameras - the arm it drives seeing through its own -
        and the live view is not up on them. The page's Connect starts it so.
        Asked under `cameras_lock`, held until the policy has started."""
        live = self.rig.state()["live"]
        if not live["running"]:
            return "the live view is not running - Connect on the policy page first"
        if live["options"].get("cameras") != "follower":
            return "the live view has the handhelds' cameras - Connect on the policy page first"
        return None

    def _arms_moved(self, angles: dict[str, float]) -> None:
        """Stand the simulated arms where the driver says the real ones are.

        Placed, not commanded: a servo reading is a fact about the rig, and driving
        the model toward it would show the trip rather than the arm.
        """
        self.physics.place(angles)

    def _sources_stamp(self) -> tuple[float, ...]:
        """Modification times of everything the expansion depends on."""
        files = [self.xacro_path, self.mount] if self.mount else [self.xacro_path]
        return tuple(f.stat().st_mtime if f.is_file() else 0.0 for f in files)

    def urdf(self) -> str:
        # Re-expand when a source file changes. Caching for the life of the process
        # would mean restarting the server after every edit to the description or
        # the mount - and quietly serving a stale robot until someone noticed.
        with self._lock:
            stamp = self._sources_stamp()
            if self._urdf is None or stamp != self._stamp:
                self._urdf = self._expand()
                self._stamp = stamp
            return self._urdf

    def _expand(self) -> str:
        # The page fetches meshes from this server, so they are rewritten to a path
        # it can request. Expansion itself lives in pinkrab_core, which already had
        # to solve the same `$(find ...)` problem for machines without ROS.
        try:
            return expand_for_meshes("meshes", self.mount)
        except DescriptionError as exc:
            raise RuntimeError(str(exc)) from exc

    def _desk_and_board(self) -> tuple[float, float]:
        """The desk's top and the board's face, as the scene has them, metres."""
        scenery = self.physics.scenery()
        board = scenery.get("pegboard")
        desk = scenery.get("desk")
        return (desk["pos"][2] if desk else 0.0), ((board["pos"][0] + board["half"][0]) if board else 0.0)

    def servo_state(self) -> dict:
        """Where the arms are, as last pushed in by the bridge.

        Read rather than measured. The studio held the bus itself until the driver
        existed; two processes cannot, and the one with the serial port is the one
        on the robot's own launch file.
        """
        return self.real.snapshot()


def marker_faces(cfg) -> dict:
    """A marker solid's marked faces, for a page to draw them: each face's pose
    in the solid's frame and its marker's modules (black 1, border included),
    with the paper's and the marker's sizes, metres."""
    poses = face_poses(cfg)
    return {"faces": [{"id": marker_id, "face_T": poses[name].tolist(),
                       "modules": marker_modules(marker_id).astype(int).tolist()}
                      for name, marker_id in cfg.face_ids.items()],
            "patch_m": cfg.patch_mm / 1000.0, "marker_m": cfg.marker_mm / 1000.0}


def hermit_parts() -> dict:
    """The handheld, for the front page and the hands view to draw, by hand:
    the parts it has that the arm's claw does not - its body and jaw, and the
    marker solid with where that hangs in the arm's tool frame. The rest, the
    servo and the camera bracket, is the arm's own."""
    hands = {}
    for side in ARMS:
        hand = replace(UMI, solid=f"handheld_{side}")
        parts = {"body": f"handheld_body_{side}.stl", "jaw": f"handheld_jaw_{side}.stl",
                 "solid": f"marker_solid_{hand.solid}.stl"}
        if all((UMI.mesh_dir / name).is_file() for name in parts.values()):
            # Where the solid hangs in this arm's tool frame - the handheld's
            # tool frame (solid_T_tcp) turned into the arm's as teleop turns it -
            # so a Hermit drawn on an arm's gripper holds its solid where that
            # arm's hand does.
            turn = np.eye(4)
            turn[:3, :3] = TOOL_TURN[side]
            hands[side] = {**parts, "tool_T_solid": (turn.T @ np.linalg.inv(hand.solid_T_tcp)).tolist(),
                           "markers": marker_faces(hand)}
            # The arm's jaw band, (shut, wide) in radians: the hands view
            # spends the handheld's opening across it.
            try:
                hands[side]["jaw_band"] = list(gripper_band(side))
            except (OSError, KeyError, TypeError, ValueError):
                hands[side]["jaw_band"] = None
    return hands


def make_handler(state: StudioState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quieter than the default
            pass

        def _send(self, body: bytes, content_type: str, code: int = 200) -> None:
            # A client that has gone - a poll the page abandoned, a tab closed
            # mid-answer - is not an error here, and the stream path already
            # treats it so. Without this every such poll is a traceback in the
            # terminal that reads like the studio failing.
            try:
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                return

        def _send_json(self, obj, code: int = 200) -> None:
            self._send(json.dumps(obj).encode(), "application/json", code)

        def _send_file(self, target: Path, content_type: str) -> None:
            """A file, whole or the byte range asked for - a video is seeked
            by the browser asking for pieces of it."""
            size = target.stat().st_size
            header = self.headers.get("Range")
            start, end = 0, size - 1
            code = 200
            if header and header.startswith("bytes="):
                first, _, last = header[6:].partition("-")
                try:
                    start = int(first) if first else max(0, size - int(last))
                    end = int(last) if last and first else size - 1
                except ValueError:
                    start, end = 0, size - 1
                end = min(end, size - 1)
                if start > end:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                code = 206
            length = end - start + 1
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            if code == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with open(target, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(1 << 20, remaining))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    remaining -= len(chunk)

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path

            if path == "/ws":
                # A page's socket: what it watches, sent as it changes, and
                # what it asks of the simulation. See feeds.py.
                ws = wsock.accept(self)
                if ws is not None:
                    serve_page(ws, state.feeds)
                return

            if path == "/":
                self._send((STATIC / "home.html").read_bytes(), CONTENT_TYPES[".html"])
                return

            # The pages in ARM_PAGES (teleop), only through the front page: a sim
            # or real chosen since this server started, and with real, the driver
            # and the gate up. A tab reopened after a restart, or left open while
            # the robot came down, is sent back to choose - and on to the page it
            # asked for. The others open as they are - the policy page drives an
            # arm too, but chooses real itself with its Connect.
            if path in PAGES:
                from .rig import robot_already_up
                up = robot_already_up() if path in ARM_PAGES and state.real.expected else {}
                if path in ARM_PAGES and (not state.mode_chosen
                                          or (state.real.expected and not all(up.values()))):
                    # Why, for the front page to say: nothing chosen yet, or
                    # real chosen and the robot's side gone since.
                    why = "down" if state.mode_chosen else "choose"
                    self.send_response(302)
                    self.send_header("Location", f"/?next={path}&why={why}")
                    self.end_headers()
                    return
                if path in ARM_PAGES and state.real.expected:
                    # Real chosen by the policy page's Connect brought up the
                    # bridge alone; this page also needs its handheld loop.
                    # Started here, the missing parts only - not a trip back to
                    # a choice whose other answer would take the bridge away.
                    state.rig.start_robot(state.rig.studio_port, only_if_asked=True)
                self._send((STATIC / PAGES[path]).read_bytes(), CONTENT_TYPES[".html"])
                return


            if path == "/train/guide":
                self._send((STATIC / "train-guide.html").read_bytes(), CONTENT_TYPES[".html"])
                return

            if path == "/train/series":
                query = parse_qs(urlparse(self.path).query)
                try:
                    ids = [int(i) for i in (query.get("ids", [""])[0]).split(",") if i]
                except ValueError:
                    self._send_json({"error": "ids is a comma-separated list of run ids"}, 400)
                    return
                self._send_json({"series": state.training.store.series(ids)})
                return

            if path == "/train/recipes":
                datasets = []
                for d in Converter.list_datasets():
                    root = Converter.datasets_dir() / d["name"]
                    try:
                        shape = image_shape(root)
                    except (OSError, ValueError):
                        shape = None
                    datasets.append({**d, "image": shape})
                self._send_json({"recipes": {k: {"label": r["label"], "batch_size": r["batch_size"]}
                                             for k, r in RECIPES.items()},
                                 "defaults": LAUNCH_DEFAULTS, "datasets": datasets})
                return

            if path == "/train/run":
                try:
                    run_id = int(parse_qs(urlparse(self.path).query).get("id", [""])[0])
                    self._send_json(state.training.store.run(run_id))
                except (ValueError, TrainError) as exc:
                    self._send_json({"error": str(exc)}, 404)
                return

            if path.startswith("/datasets/"):
                # A dataset's episodes for the Curate page to play them as
                # stored - /datasets/<name>/episodes - and its video files,
                # which the browser seeks with Range requests -
                # /datasets/<name>/videos/....mp4.
                parts = path.split("/", 3)
                try:
                    if len(parts) == 4 and parts[3] == "episodes":
                        self._send_json(Converter.episodes(parts[2]))
                    elif len(parts) == 4 and parts[3].startswith("videos/"):
                        self._send_file(Converter.video(parts[2], parts[3]), "video/mp4")
                    else:
                        self._send_json({"error": "not found"}, 404)
                except (OSError, ValueError, KeyError) as exc:
                    self._send_json({"error": str(exc)}, 404)
                return

            if path == "/policy/checkpoints":
                self._send_json({"checkpoints": checkpoints()})
                return

            if path == "/policy/settings":
                # How a checkpoint would be asked: its type's settings from
                # the rig's inference.yaml, beside what it was trained with.
                checkpoint = parse_qs(urlparse(self.path).query).get("checkpoint", [""])[0]
                try:
                    self._send_json(inference.describe(inference.trained(Path(checkpoint))))
                except inference.InferenceError as exc:
                    self._send_json({"error": str(exc)}, 400)
                return

            if path == "/model":
                try:
                    self._send(state.urdf().encode(), "application/xml")
                except RuntimeError as exc:
                    self._send_json({"error": str(exc)}, 500)
                return

            if path == "/workspace":
                # The patch of the desk each arm can grasp on, for the page to
                # lay on the desk. Solved once, at start: asked while that is
                # still going, the answer waits for it rather than being asked again.
                desk_z, board_x = state._desk_and_board()
                try:
                    arms = state.reach.get(desk_z=desk_z, board_x=board_x)
                except RuntimeError as exc:
                    self._send_json({"error": str(exc)}, 500)
                    return
                except OSError as exc:
                    self._send_json({"error": f"the description could not be read: {exc}"}, 500)
                    return
                self._send_json({"cell_m": workspace.CELL_M, "desk_z": desk_z, "arms": arms})
                return

            if path == "/ready":
                # What the page's ready button puts the arms at. Served rather
                # than written into the page, so it and a policy run's walk
                # cannot end up being two different quarter turns. Home needs no
                # serving: it is every joint at zero.
                self._send_json(origin.ready_pose())
                return

            if path == "/physics":
                described = state.physics.describe()
                described["real"] = state.real.expected
                self._send_json(described)
                return

            if path == "/physics/stream":
                self._stream()
                return

            if path == "/real/requests":
                # The bridge's side of a button. What the page asked for is
                # handed over once and made into a service call over there,
                # because a page cannot call one and should not pretend to.
                self._send_json({"requests": state.real.take_requests()})
                return

            if path == "/hermit":
                self._send_json(hermit_parts())
                return

            if path == "/desk_solid":
                # The desk solid, for the hands view to draw: its markers, and
                # where it stands in the robot's world, as teleop takes it
                # (desk_solid.world_T_desk).
                try:
                    placed, carried, chest = solid_in_world(), world_T_desk(), chest_x()
                except PlacementError as exc:
                    self._send_json({"error": f"the desk solid cannot be placed: {exc}"}, 500)
                    return
                note = ("" if SETTINGS.desk_file.is_file()
                        else f"no desk.yaml - the solid drawn at the standard yaw, {STANDARD_YAW_DEG:g}")
                # The solid where it is reckoned, the frame a hand against it is
                # carried into the robot's world by, the operator's chest where
                # it is reckoned, and teleop's own low-pass, so a hand is drawn
                # as steady as the arm is asked to follow it.
                self._send_json({"markers": marker_faces(UMI.for_solid("desk")),
                                 "solid_in_world": placed.tolist(), "world_T_desk": carried.tolist(),
                                 "chest_x": chest, "note": note,
                                 "smoothing_s": SETTINGS.teleop_pose_smoothing_s})
                return

            if path == "/desk_solid.stl":
                self._send(solid_stl(UMI.for_solid("desk"), scale=0.001), CONTENT_TYPES[".stl"])
                return

            if path.startswith("/hermit/meshes/"):
                name = Path(path).name
                mesh = UMI.mesh_dir / name
                if mesh.name != name or not mesh.is_file():
                    self._send_json({"error": f"no such mesh: {name}"}, 404)
                    return
                self._send(mesh.read_bytes(), CONTENT_TYPES[".stl"])
                return

            if path == "/checklist":
                # Everything done once per machine or rig, with whether it has
                # been: the front page's Before Starting card. The same list
                # `pinkrab doctor` prints. Read-only - the driver may be up.
                from pinkrab_core.checklist import GROUPS, checks
                self._send_json({"items": checks(), "groups": GROUPS})
                return

            if path == "/rig/needs":
                # What the rig has to have on disk before the driver will hold
                # the arms: the port assignment, a calibration per arm, and the
                # description. Each missing one is said with the command that
                # writes it, so the page can put the list in front of whoever
                # pressed Connect rather than let the driver exit at once with
                # the reason in a terminal they are not looking at. The same
                # checks `pinkrab doctor` prints, read from the same settings.
                needs = []
                if not SETTINGS.ports_file.is_file():
                    needs.append({"what": "port assignment",
                                  "where": str(SETTINGS.ports_file),
                                  "why": "which board is which arm - the driver will not open a bus it cannot name",
                                  "run": "pinkrab ports detect"})
                # In the driver's own order: it stops at the first of these.
                if not SETTINGS.offsets_file.is_file():
                    needs.append({"what": "tick-to-angle offsets",
                                  "where": str(SETTINGS.offsets_file),
                                  "why": "what a servo tick means in joint angle - written when the home pose is recorded",
                                  "run": "pinkrab calibrate left && pinkrab calibrate right"})
                for arm in ARMS:
                    if not SETTINGS.calibration_file(arm).is_file():
                        needs.append({"what": f"{arm} arm calibration",
                                      "where": str(SETTINGS.calibration_file(arm)),
                                      "why": "the travel of these servos - without it a command lands somewhere else",
                                      "run": f"pinkrab calibrate {arm}"})
                # The URDF itself is not on this list: the first solver to
                # need it expands it from the description and the mount. What
                # can be missing is the mount - a mount.yaml still holding the
                # placeholder geometry gives an expansion that looks fine and
                # puts the arms in the wrong place.
                mount = SETTINGS.mount_file
                if not mount.is_file() or "PLACEHOLDER" in mount.read_text():
                    needs.append({"what": "wall mount",
                                  "where": str(mount),
                                  "why": "how the arms are bolted to the wall - every solve goes through it",
                                  "run": "pinkrab mount"})
                self._send_json({"needs": needs})
                return

            if path.startswith("/curate/review/"):
                # A take's gripper video as its labels leave it, made on first
                # ask: /curate/review/<take>/<gripper_left|gripper_right>.mp4
                parts = path.split("/")
                if (len(parts) != 5 or not EPISODE_NAME.match(parts[3])
                        or parts[4] not in ("gripper_left.mp4", "gripper_right.mp4")):
                    self._send_json({"error": "not found"}, 404)
                    return
                from pinkrab_umi import keep_masks
                try:
                    target = keep_masks.review(UMI.episodes_dir / parts[3], parts[4][:-4])
                except (OSError, ValueError, KeyError) as exc:
                    self._send_json({"error": str(exc)}, 404)
                    return
                self._send_file(target, "video/mp4")
                return

            if path.startswith("/episodes/"):
                # An episode's files, for the replay: the videos, which the
                # browser seeks with Range requests, and replay.json, solved on
                # first ask and cached beside them.
                parts = path.split("/")
                if len(parts) != 4 or not EPISODE_NAME.match(parts[2]):
                    self._send_json({"error": "not found"}, 404)
                    return
                root = UMI.episodes_dir / parts[2]
                if not root.is_dir():
                    self._send_json({"error": f"no episode {parts[2]!r}"}, 404)
                    return
                if parts[3] == "replay.json":
                    try:
                        self._send_json(replay.load_or_solve(root))
                    except Exception as exc:  # noqa: BLE001 - a bad episode is a message, not a crash
                        self._send_json({"error": f"could not solve the episode: {exc}"}, 500)
                    return
                target = root / parts[3]
                if target.name != parts[3] or not target.is_file():
                    self._send_json({"error": "not found"}, 404)
                    return
                self._send_file(target, {".mp4": "video/mp4", ".json": "application/json",
                                         ".jsonl": "application/x-ndjson"}.get(target.suffix, "application/octet-stream"))
                return

            if path.startswith("/meshes/"):
                name = Path(path).name
                mesh = state.mesh_dir / name
                # Serve only what is in the mesh directory, by name, so a crafted
                # path cannot walk out of it.
                if mesh.name != name or not mesh.is_file():
                    self._send_json({"error": f"no such mesh: {name}"}, 404)
                    return
                self._send(mesh.read_bytes(), CONTENT_TYPES[".stl"])
                return

            if path.startswith("/static/"):
                name = Path(path).name
                asset = STATIC / name
                if asset.name != name or not asset.is_file():
                    self._send_json({"error": f"no such asset: {name}"}, 404)
                    return
                self._send(asset.read_bytes(), CONTENT_TYPES.get(asset.suffix, "text/plain"))
                return

            # A page nobody serves, typed into the bar: back to the front door.
            # Anything that looks like an API path stays a JSON 404.
            if "." not in path and not path.startswith(
                ("/rig", "/real", "/physics", "/state", "/robot", "/policy/", "/train/")
            ):
                self.send_response(302)
                self.send_header("Location", "/")
                self.end_headers()
                return
            self._send_json({"error": "not found"}, 404)

        # The bridge's reading of the simulation: the same frames the pages get
        # over their sockets, as server-sent events - the bridge is a ROS node
        # reading one way with urllib, and has no use for a socket it would
        # only listen on.
        def _stream(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            # This connection is held open, so it cannot carry anything else and
            # the client should not wait to find that out.
            self.send_header("Connection", "close")
            self.end_headers()

            topic = state.feeds.topics["physics"]
            previous: str | None = None
            spoke_at = 0.0
            try:
                while not state.stopping.is_set():
                    answer = topic.read()
                    now = time.monotonic()
                    # Send what changed, and send something once a second whatever
                    # happens: a settled robot has nothing to report, but a stream
                    # that goes quiet is indistinguishable from one that has died.
                    if answer is not None and (answer[0] != previous or now - spoke_at >= 1.0):
                        basis, frame = answer
                        self.wfile.write(f"data: {json.dumps(frame, separators=(',', ':'))}\n\n".encode())
                        self.wfile.flush()
                        previous, spoke_at = basis, now
                    time.sleep(topic.every)
            except (BrokenPipeError, ConnectionResetError):
                return  # the bridge went; nothing here to report

        def _body(self) -> dict | None:
            length = int(self.headers.get("Content-Length") or 0)
            # A pose is fourteen numbers. Anything larger is not one, and reading it
            # into memory on the say-so of the sender is how a server falls over.
            if length <= 0 or length > 64_000:
                self._send_json({"error": "expected a JSON body"}, 400)
                return None
            try:
                return json.loads(self.rfile.read(length))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                self._send_json({"error": f"bad JSON: {exc}"}, 400)
                return None

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path

            if path == "/rig/footswitch/start":
                # The pedal's Refresh: its reader started again if it is not
                # running. A running one is root and cannot be stopped from
                # here; it tries the switch again every two seconds by itself.
                # Ahead of the /rig/<which>/<verb> branch below, which would
                # take it for a part it does not know.
                self._body()
                state.rig.start_footswitch(self.server.server_address[1])
                self._send_json({"ok": True, "rig": state.rig.state()})
                return

            if path == "/rig/live/restart":
                # The live row's Refresh: the view put down, waited for, and
                # brought up again as it was started. Refused while an episode
                # records from it or a policy reads it; a view that does not
                # answer is let go, since that is what the Refresh is for.
                self._body()
                with state.cameras_lock:
                    policy = state.policy.state()
                    if policy["running"] or policy["driving"]:
                        self._send_json({"error": "a policy is reading the live view's cameras - halt it first"}, 409)
                        return
                    if (live_stats().get("episode") or {}).get("active"):
                        self._send_json({"error": "an episode is recording - stop it first"}, 409)
                        return
                    try:
                        state.rig.restart("live")
                    except RigError as exc:
                        self._send_json({"error": f"the live view did not restart: {exc}",
                                         "rig": state.rig.state()}, 500)
                        return
                self._send_json({"ok": True, "rig": state.rig.state()})
                return

            if path.startswith("/rig/") and len(path.split("/")) == 4:
                # /rig/<live|teleop>/<start|stop>. Start takes options in the body;
                # stop takes nothing, but the socket is drained either way.
                parts = path.split("/")
                which, verb = (parts[2], parts[3]) if len(parts) == 4 else ("", "")
                if which not in ("live", "teleop") or verb not in ("start", "stop"):
                    self._send_json({"error": "not found"}, 404)
                    return
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(min(length, 64_000)) if length > 0 else b""
                try:
                    options = json.loads(raw) if raw.strip() else {}
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    self._send_json({"error": f"bad JSON: {exc}"}, 400)
                    return
                options = options if isinstance(options, dict) else {}
                if which == "live":
                    # Whose gripper cameras, said either way: a view started
                    # without it and one asked for the handhelds' are the same.
                    from pinkrab_umi.gripper_cams import CAMERA_SETS
                    options = {"cameras": "handheld", **options}
                    if options["cameras"] not in CAMERA_SETS:
                        self._send_json({"error": f"cameras must be one of {', '.join(CAMERA_SETS)}"}, 400)
                        return
                try:
                    if verb == "start" and which == "live":
                        refused = state.start_live(options)
                        if refused:
                            self._send_json({"error": refused}, 409)
                            return
                    elif verb == "start":
                        state.rig.start(which, options)
                    else:
                        if which == "live":
                            state.halted()
                        state.rig.stop(which)
                except RigError as exc:
                    self._send_json({"error": str(exc)}, 409)
                    return
                state.feeds.touch("rig")
                self._send_json({"ok": True, "rig": state.rig.state()})
                return

            if path == "/real/state":
                # The bridge pushing what ROS says: where the arms are, and what
                # the gate believes about arming and mode.
                payload = self._body()
                if payload is None:
                    return
                angles = payload.get("angles_rad")
                state.real.push(
                    {str(k): float(v) for k, v in angles.items()} if isinstance(angles, dict) else {},
                    payload.get("teleop") if isinstance(payload.get("teleop"), dict) else None,
                    payload.get("trouble") if isinstance(payload.get("trouble"), str) else None,
                    torque=payload.get("torque") if isinstance(payload.get("torque"), dict) else None,
                    walking=bool(payload.get("walking")),
                    sent={str(k): float(v) for k, v in payload["sent_rad"].items()}
                    if isinstance(payload.get("sent_rad"), dict) else None,
                    asked={str(k): float(v) for k, v in payload["asked_rad"].items()}
                    if isinstance(payload.get("asked_rad"), dict) else None,
                    following=payload["following"] if isinstance(payload.get("following"), bool) else None,
                )
                state.walk_reported(payload.get("walk_id"))
                if isinstance(angles, dict):
                    state.dagger.robot({str(k): float(v) for k, v in angles.items()},
                                       payload.get("teleop") if isinstance(payload.get("teleop"), dict) else None,
                                       payload.get("following") if isinstance(payload.get("following"), bool) else None)
                self._send_json({"ok": True})
                return

            if path == "/policy/settings":
                # The settings for checkpoints of this one's type, and the
                # common ones, as the page has them: the knobs named are set,
                # the rest left to Pinkrab's own or the checkpoint's. A running
                # policy keeps what it started with; the next Ask or Drive takes
                # these, and the gate its speed when the robot's launch starts.
                payload = self._body()
                if payload is None:
                    return
                values = payload.get("values")
                shared = payload.get("common")
                if not isinstance(values, dict) or not isinstance(shared, (dict, type(None))):
                    self._send_json({"error": "expected values: {setting: value}, and common: {setting: value}"}, 400)
                    return
                try:
                    cfg = inference.trained(Path(str(payload.get("checkpoint") or "")))
                    inference.save(cfg, values, shared)
                    self._send_json(inference.describe(cfg))
                except inference.InferenceError as exc:
                    self._send_json({"error": str(exc)}, 400)
                return

            if path == "/policy/ask":
                # Ready first, then ask: the arms walk to the ready pose through
                # the gate - the studio mode, the same walk the teleop page's
                # ready button orders - and the page starts the asking loop once
                # they have arrived. Nothing else is published meanwhile, so
                # the arms hold at ready while the policy is only being asked.
                payload = self._body()
                if payload is None:
                    return
                checkpoint = str(payload.get("checkpoint") or "")
                # Settings that do not hold - the common ones, and the
                # checkpoint's kind's against it - are refused before the
                # gate is asked anything, not after the arm has walked.
                try:
                    tol_deg = inference.common()["ready_tol_deg"]
                    wrong = inference.describe(inference.trained(Path(checkpoint)))["problems"] if checkpoint else []
                except inference.InferenceError as exc:
                    wrong = [str(exc)]
                if wrong:
                    self._send_json({"error": "the inference settings do not fit this checkpoint: "
                                              + "; ".join(wrong)}, 409)
                    return
                # Then on to where the dataset's takes began - each tool
                # where its hand began - so the policy is shown pictures, and
                # the hands apart, like the ones it was trained on. Without a
                # checkpoint to read it from, ready is the start.
                legs = [origin.ready_pose()]
                start = origin.ready_pose()
                note = ""
                if checkpoint:
                    try:
                        from pinkrab_core.demo_start import DatasetChanged, dataset_of_checkpoint, start_joints
                        from pinkrab_core.kinematics import ArmKinematics
                        from pinkrab_policy.rollout import carry
                        chains = {hand: ArmKinematics(hand) for hand in checkpoint_hands(checkpoint)}
                        joints, info = start_joints(dataset_of_checkpoint(Path(checkpoint)), chains, carry)
                        start = {**origin.ready_pose(),
                                 **{n: math.radians(float(v)) for hand, chain in chains.items()
                                    for n, v in zip(chain.joint_names, joints[hand])}}
                        legs = legs + [start]
                        note = "start: " + ", ".join(
                            f"{hand} tool {found['pitch_deg']:.0f} deg down, {found['height_m'] * 1000:.0f} mm "
                            "above the desk" for hand, found in info["arms"].items())
                    except DatasetChanged as exc:
                        # Not a missing start: the policy would be shown takes
                        # it never saw. Nothing is walked.
                        self._send_json({"error": str(exc)}, 409)
                        return
                    except Exception as exc:  # noqa: BLE001 - the start is a convenience; ready is the fallback
                        note = f"no demo start, ready instead: {exc}"
                state.real.drive(MODE_STUDIO)
                queued = state.walk(legs)
                self._send_json({"queued": queued, "ready": start, "ready_tol_deg": tol_deg, "note": note})
                return

            if path == "/policy/goto":
                # The ready pose or home, walked through the gate in the studio
                # mode - the same walk the teleop page's pose buttons order -
                # so a session can be set up or put away from this page too.
                # Not while a policy is asking or driving: those own the arm.
                payload = self._body()
                if payload is None:
                    return
                which = str(payload.get("pose") or "")
                if which == "ready":
                    goal = origin.ready_pose()
                elif which == "home":
                    goal = {f"{arm}_{name}": 0.0 for arm in ARMS for name in SETTINGS.joint_names}
                else:
                    self._send_json({"error": f"no such pose: {which!r} - ready or home"}, 400)
                    return
                if state.policy.state()["running"] or state.policy.state()["driving"]:
                    self._send_json({"error": "a policy has the arm - stop it first"}, 409)
                    return
                state.real.drive(MODE_STUDIO)
                queued = state.walk([goal])
                self._send_json({"queued": queued, "pose": which})
                return

            if path == "/policy/drive":
                # The rollout node, with the checkpoint the page chose - it
                # drives the arms the checkpoint was trained for - and the
                # gate handed to it. The node walks to ready on
                # its own once it sees policy mode, then drives; each step it
                # takes comes back to /policy/step. The run is recorded from
                # here to its stop (dagger.py), and not driven unrecorded.
                payload = self._body()
                if payload is None:
                    return
                checkpoint = str(payload.get("checkpoint") or "")
                if not (Path(checkpoint) / "config.json").is_file():
                    self._send_json({"error": f"{checkpoint} is not a checkpoint"}, 400)
                    return
                # Settings the node would refuse to load with, said here
                # rather than as a node that exits after it was started.
                try:
                    wrong = inference.describe(inference.trained(Path(checkpoint)))["problems"]
                except inference.InferenceError as exc:
                    wrong = [str(exc)]
                if wrong:
                    self._send_json({"error": "the inference settings do not fit this checkpoint: "
                                              + "; ".join(wrong)}, 409)
                    return
                with state.cameras_lock:
                    unseeing = state.policy_unseeing()
                    if unseeing:
                        self._send_json({"error": unseeing}, 409)
                        return
                    try:
                        hands = checkpoint_hands(checkpoint)
                    except PolicyError as exc:
                        self._send_json({"error": str(exc)}, 409)
                        return
                    recording = state.dagger.begin(list(hands), checkpoint)
                    if "error" in recording:
                        self._send_json({"error": f"not recorded, so not driven: {recording['error']}"}, 409)
                        return
                    task = str(payload.get("task") or "").strip()
                    try:
                        state.policy.begin_driven(checkpoint, task)
                        node = state.rig.start_policy(checkpoint, task)
                    except (PolicyError, RigError) as exc:
                        state.policy.end_driven()
                        state.dagger.abandon()
                        self._send_json({"error": str(exc)}, 409)
                        return
                queued = state.real.drive(MODE_POLICY)
                self._send_json({"ok": True, "node": node, "queued": queued, "policy": state.policy.state()})
                return

            if path == "/policy/halt":
                self._body()
                self._send_json(state.halt_policy())
                return

            if path in ("/dagger/takeover", "/dagger/handback", "/dagger/follow", "/dagger/throw", "/dagger/redo",
                        "/dagger/delete", "/dagger/outcome"):
                # The page's buttons for what the pedals do on the policy
                # page - see dagger.py.
                payload = self._body()
                if payload is None:
                    return
                order = path.rsplit("/", 1)[1]
                if order == "delete":
                    answer = state.dagger.delete(str(payload.get("name") or ""))
                elif order == "outcome":
                    answer = state.dagger.outcome(str(payload.get("name") or ""), payload.get("outcome"))
                else:
                    answer = {"takeover": state.dagger.take_over, "handback": state.dagger.hand_back,
                              "follow": state.dagger.follow_toggle, "throw": state.dagger.stop_or_redo,
                              "redo": state.dagger.redo}[order]()
                state.feeds.touch("dagger", "episodes")
                self._send_json(answer, 409 if "error" in answer else 200)
                return

            if path == "/policy/step":
                # The rollout node, reporting one step. Kept, not answered.
                payload = self._body()
                if payload is None:
                    return
                state.policy.record(payload)
                self._send_json({"ok": True})
                return

            if path in ("/policy/start", "/policy/stop"):
                # A checkpoint, asked about the live pictures. It drives nothing:
                # the answer goes to the page and no further. See policy.py.
                payload = self._body()
                if payload is None:
                    return
                try:
                    if path.endswith("start"):
                        with state.cameras_lock:
                            unseeing = state.policy_unseeing()
                            if unseeing:
                                self._send_json({"error": unseeing}, 409)
                                return
                            state.policy.start(str(payload.get("checkpoint") or ""), state.real.angles,
                                               str(payload.get("task") or "").strip())
                    else:
                        state.policy.stop()
                except PolicyError as exc:
                    self._send_json({"error": str(exc)}, 409)
                    return
                self._send_json({"ok": True, "policy": state.policy.state()})
                return

            if path == "/mode":
                # Simulation or the real robot, chosen here. Real starts what
                # the page asks for beside this page - the teleop page the
                # operator's side (bridge, handheld loop), the policy page the
                # same, the handheld loop for corrections; no page asks for the
                # rig's own launch - and
                # simulation stops them and forgets what they said.
                payload = self._body()
                if payload is None:
                    return
                on = bool(payload.get("real"))
                parts = payload.get("parts")
                if parts is not None:
                    if (not isinstance(parts, list)
                            or any(p not in state.rig.STARTERS for p in parts)):
                        self._send_json({"error": f"parts must be from {sorted(state.rig.STARTERS)}"}, 400)
                        return
                    parts = tuple(parts)
                if on:
                    state.set_real(True)
                    state.rig.start_robot(state.rig.studio_port, parts)
                else:
                    # Not under a policy: its node would go on driving the arm
                    # with no bridge left to carry a stop to the gate.
                    policy = state.policy.state()
                    if policy["running"] or policy["driving"]:
                        self._send_json({"error": "a policy has the arm - halt it first"}, 409)
                        return
                    state.rig.stop_robot()
                    state.set_real(False)
                state.feeds.touch("rig")
                self._send_json({"real": state.real.expected, "rig": state.rig.state()})
                return

            if path in RETRIES:
                # A Retry or Refresh for one thing the live view opens - the
                # top camera, a gripper camera, a jaw: asked of the live
                # server, which answers at once; the stats that follow say how
                # it went.
                from .rig import live_url
                self._body()
                # A gripper camera or a jaw reopened drops what an episode or a
                # policy is reading from it meanwhile.
                if "/head/" not in path:
                    with state.cameras_lock:
                        held = state.live_held()
                    if held:
                        self._send_json({"error": held}, 409)
                        return
                request = urllib.request.Request(live_url() + path.removeprefix("/rig/live/"), data=b"{}",
                                                 headers={"Content-Type": "application/json"})
                try:
                    with urllib.request.urlopen(request, timeout=2.0) as answer:
                        self._send_json(json.loads(answer.read().decode()))
                except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
                    self._send_json({"error": f"the live view did not answer: {exc}"}, 502)
                return

            if path == "/cameras":
                # One side's camera of one set, chosen on a page: written to
                # that set's file, and the live view started again so it opens
                # that socket - when it is showing that set.
                from pinkrab_umi.cameras import assign
                payload = self._body()
                if payload is None:
                    return
                which = str(payload.get("which") or "handheld")
                with state.cameras_lock:
                    # Asked before the file is written: a socket the view would
                    # not open until later is a select that says one thing and a
                    # picture that shows another.
                    live = state.rig.state()["live"]
                    shown = live["running"] and live["options"].get("cameras", "handheld") == which
                    held = state.live_held() if shown else None
                    if held:
                        self._send_json({"error": held}, 409)
                        return
                    try:
                        chosen = assign(str(payload.get("side") or ""), payload.get("socket") or None, which)
                    except ValueError as exc:
                        self._send_json({"error": str(exc)}, 400)
                        return
                    state.feeds.touch(f"cameras/{which}")
                    restarted = None
                    try:
                        if shown:
                            restarted = state.rig.restart("live")
                    except RigError as exc:
                        self._send_json({"error": f"written, but the live view did not restart: {exc}",
                                         "mapping": chosen}, 500)
                        return
                self._send_json({"ok": True, "mapping": chosen, "live": restarted})
                return

            if path == "/real/pedal":
                # A foot on the floor. Remembered so the page can flash the
                # matching pedal; what it asked for arrives as its own request.
                payload = self._body()
                if payload is None:
                    return
                name = payload.get("name")
                if name not in ("jog", "handheld", "stop"):
                    self._send_json({"error": f"no such pedal: {name!r}"}, 400)
                    return
                down = bool(payload.get("down", True))
                state.real.pedal(name, down)
                state.clutch.pedal(name, down)
                state.dagger.pedal(name, down)
                self._send_json({"ok": True})
                return

            if path == "/rig/live/episode/delete":
                # One name, or a list of them: the shelf deletes what was ticked.
                payload = self._body()
                if payload is None:
                    return
                names = payload.get("names") or [payload.get("name") or ""]
                gone, errors = [], []
                for name in names:
                    try:
                        delete_episode(str(name))
                        gone.append(name)
                    except EpisodeError as exc:
                        errors.append(str(exc))
                state.feeds.touch("episodes", "verdicts", "home", "dagger")
                self._send_json({"ok": not errors, "deleted": gone, "errors": errors}, 200 if not errors else 404)
                return

            if path in ("/train/sources", "/train/sources/delete"):
                payload = self._body()
                if payload is None:
                    return
                store = state.training.store
                try:
                    if path == "/train/sources":
                        found = store.save_source(payload)
                        state.training.now()
                        self._send_json(found)
                    else:
                        store.delete_source(str(payload.get("name") or ""))
                        self._send_json({"ok": True})
                except (TrainError, KeyError, TypeError, ValueError) as exc:
                    self._send_json({"error": str(exc)}, 400)
                state.feeds.touch("train")
                return

            if path == "/train/sync":
                state.training.now()
                self._send_json({"ok": True})
                return

            if path in ("/train/sources/key", "/train/sources/test", "/train/launch", "/train/stop",
                        "/train/resume", "/train/fetch", "/train/autofetch"):
                payload = self._body()
                if payload is None:
                    return
                jobs, store = state.train_jobs, state.training.store
                try:
                    if path == "/train/sources/key":
                        save_key(store, str(payload.get("name") or ""), str(payload.get("key") or ""))
                        answer = {"ok": True}
                    elif path == "/train/sources/test":
                        answer = jobs.probe(str(payload.get("name") or ""))
                    elif path == "/train/launch":
                        answer = {"queued": jobs.launch(payload)["run"]}
                    elif path == "/train/stop":
                        jobs.stop(int(payload["id"]))
                        answer = {"ok": True}
                    elif path == "/train/resume":
                        jobs.resume(int(payload["id"]))
                        answer = {"ok": True}
                    elif path == "/train/fetch":
                        jobs.fetch(int(payload["id"]), int(payload["step"]))
                        answer = {"ok": True}
                    else:
                        store.set_auto_fetch(int(payload["id"]), bool(payload.get("on", True)))
                        jobs.fetch_due()
                        answer = {"ok": True}
                except (LaunchError, TrainError, KeyError, TypeError, ValueError) as exc:
                    self._send_json({"error": str(exc)}, 400)
                    return
                state.feeds.touch("train_jobs", "train")
                self._send_json(answer)
                return

            if path == "/convert":
                payload = self._body()
                if payload is None:
                    return
                try:
                    state.converter.start(payload)
                except ValueError as exc:
                    self._send_json({"error": str(exc)}, 409)
                    return
                state.feeds.touch("convert")
                self._send_json(state.converter.state())
                return

            if path == "/masks/look":
                # One frame of a take with its grippers drawn and the things
                # named looked for in it - the Curate page's preview.
                payload = self._body()
                if payload is None:
                    return
                try:
                    self._send_json(state.looker.look(str(payload.get("take") or ""), str(payload.get("side") or ""),
                                                      int(payload.get("frame") or 0),
                                                      [str(t).strip() for t in payload.get("things") or []
                                                       if str(t).strip()]))
                except (OSError, ValueError, KeyError, RuntimeError) as exc:
                    self._send_json({"error": str(exc)}, 409)
                return

            if path == "/masks/machine":
                # Where SAM 3 runs: chosen, or the chosen one tried again.
                payload = self._body()
                if payload is None:
                    return
                try:
                    if payload.get("name"):
                        state.machines.choose(str(payload["name"]))
                    else:
                        state.machines.retry()
                except ValueError as exc:
                    self._send_json({"error": str(exc)}, 409)
                    return
                state.feeds.touch("masks")
                self._send_json(state.machines.state())
                return

            if path in ("/masks", "/masks/stop"):
                payload = self._body()
                if payload is None:
                    return
                try:
                    if path == "/masks":
                        state.masker.start(payload)
                    else:
                        state.masker.stop()
                except ValueError as exc:
                    self._send_json({"error": str(exc)}, 409)
                    return
                state.feeds.touch("masks")
                self._send_json(state.masker.state())
                return

            if path == "/datasets/delete":
                payload = self._body()
                if payload is None:
                    return
                try:
                    Converter.delete_dataset(str(payload.get("name") or ""))
                except ValueError as exc:
                    self._send_json({"error": str(exc)}, 404)
                    return
                state.feeds.touch("datasets", "home")
                self._send_json({"ok": True})
                return

            if path == "/rig/live/head/light":
                # The top camera's exposure and brightness, set from a page.
                payload = self._body()
                if payload is None:
                    return
                self._send_json(*live_request("/head/light", payload))
                return

            if path in ("/rig/live/episode/start", "/rig/live/episode/stop"):
                payload = self._body()
                if payload is None:
                    return
                if path.endswith("start"):
                    # Not while the pair is being switched under it.
                    with state.cameras_lock:
                        self._send_json(*live_request(path.replace("/rig/live", ""), payload))
                    return
                self._send_json(*live_request(path.replace("/rig/live", ""), payload))
                return

            if path == "/real/request":
                # A button. Queued for the bridge rather than acted on here, and
                # the page is not told it worked - what it draws is the next
                # TeleopState that comes back, which is the only thing that knows.
                payload = self._body()
                if payload is None:
                    return
                kind = str(payload.get("kind", ""))
                if kind not in ("mode", "estop", "torque", "goto", "drive"):
                    self._send_json({"error": f"no such request: {kind!r}"}, 400)
                    return
                fields = {k: v for k, v in payload.items() if k != "kind"}
                # Pedal 3 while an episode records is the recording's stop, not
                # the arms': the clutch took the press already.
                if (kind == "mode" and payload.get("pedal") and state.clutch.armed
                        and state.clutch.recording is not None):
                    self._send_json({"queued": None, "note": "the clutch has this pedal"})
                    return
                if kind == "drive":
                    # A pedal: arms live, and handed to this driver.
                    try:
                        mode = int(fields["mode"])
                    except (KeyError, TypeError, ValueError):
                        self._send_json({"error": "drive needs a mode"}, 400)
                        return
                    # On the policy page pedals 1 and 2 are the correction's
                    # (dagger.py took the press); the stop is never taken.
                    if state.dagger.armed and payload.get("pedal"):
                        self._send_json({"queued": None, "note": "the correction has this pedal"})
                        return
                    # While the clutch is armed, pedal 2 records rather than
                    # drives: the same press must not also arm the arms.
                    if state.clutch.armed and mode == MODE_HANDHELD and payload.get("pedal"):
                        self._send_json({"queued": None, "note": "the clutch has this pedal"})
                        return
                    if not state.real.expected:
                        # The simulation: the hands followed here, or the page jogging.
                        self._send_json({"queued": None, "sim": state.sim_hand.set(
                            mode if mode in (MODE_STUDIO, MODE_HANDHELD) else MODE_IDLE)})
                        return
                    self._send_json({"queued": state.real.drive(mode)})
                    return
                if kind == "goto":
                    # A pose to walk to, at the parking speed: nobody is
                    # steering, and the arm may be anywhere when this is asked.
                    pose = fields.get("pose")
                    if not isinstance(pose, dict) or not all(
                        isinstance(k, str) and isinstance(v, (int, float)) for k, v in pose.items()
                    ):
                        self._send_json({"error": "goto needs a pose: {joint: radians}"}, 400)
                        return
                    goal = {k: float(v) for k, v in pose.items()}
                    self._send_json({"queued": state.walk([goal])})
                    return
                if kind == "mode" and not state.real.expected:
                    # A stop on the simulation: nothing drives the arms.
                    state.sim_hand.set(MODE_IDLE)
                self._send_json({"queued": state.real.request(kind, **fields)})
                return

            self._send_json({"error": "not found"}, 404)

    return Handler


def serve(state: StudioState, host: str, port: int) -> None:
    try:
        httpd = ThreadingHTTPServer((host, port), make_handler(state))
    except OSError as exc:
        if exc.errno != errno.EADDRINUSE:
            raise
        # Say WHICH failure this is, loudly. A silent bind failure leaves an older
        # server answering on the port, and everything that talks to it sees that
        # server's stale robot - a phantom bug this project has now chased twice.
        raise SystemExit(
            f"\n  port {port} is already taken - another studio is running.\n"
            f"  Find it with:  lsof -i :{port}\n"
            f"  then kill it, or start this one with --port.\n"
        ) from exc
    where = "this machine only" if host == "127.0.0.1" else "any machine on this network"
    print(f"\n  Pinkrab studio  ->  http://localhost:{port}/")
    print(f"  reachable from: {where}")
    model = state.physics.model
    print(f"  physics: {state.physics.scene.name} - {model.njnt} joints, "
          f"{model.nu} actuators, {1 / model.opt.timestep:.0f} Hz")
    print("  arms:    simulation until the page says otherwise - the robot switch")
    print("           on the page starts the bridge and the handheld loop beside")
    print("           it (the pedal starts with this server); the rig's own side")
    print("           is not started here:")
    print("           ros2 launch pinkrab_bringup robot.launch.py")
    if host != "127.0.0.1":
        print("  Anyone who can reach this port can see the robot. There is no auth.")
    print("  Ctrl+C to stop.\n")
    state.physics.start()
    state.rig.studio_port = port
    state.clutch.live_url = live_url().rstrip("/")
    # The pedals, for as long as the page is up: they record on the collect
    # page with no robot, so they are not the robot switch's to start.
    state.rig.start_footswitch(port)
    # The training page's reader: runs keep training whether the page is open
    # or not, and a page opened later should find them already read.
    state.training.start()
    # A kill, or the terminal closing, is Ctrl+C too. Python's default for
    # SIGTERM is to die on the spot, and a studio that dies on the spot leaves
    # the bridge, the loop and the camera it started running with nobody to
    # stop them - the next start then finds its own leftovers.
    def interrupt(signum, _frame):
        raise KeyboardInterrupt(signal.Signals(signum).name)
    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupt)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped\n")
    finally:
        # The event first: it is what lets the open streams return, and a stream
        # still writing while the socket closes is a traceback per open tab.
        state.stopping.set()
        # The processes the page started go down with the page's server: a live
        # view left holding the camera is the next start's "failed to set power
        # state".
        state.policy.stop_quietly()
        state.training.stop()
        state.rig.stop_all()
        state.physics.stop()
        httpd.server_close()
