"""Making takes' keep labels, from the Curate page.

One job at a time, on a thread of its own: SAM 3 takes minutes a take, and
the page watches it go - which take, which video, which thing, how far - and
can stop it. The page asks with the takes and the task's things in words;
each take's labels go beside it (keep_masks.make), and the page reads what
each take has from its masks.json (keep_masks.account).

SAM 3 runs on this machine or on one of the training page's ssh sources
(Machines): a GPU there tracks a take in a fraction of a Mac's time. The
takes stay here; only each gripper video goes there, and its labels come
back.
"""
from __future__ import annotations

import base64
import json
import shlex
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from pinkrab_umi.config import SETTINGS as UMI
from pinkrab_umi.episode import EPISODE_NAME, list_episodes

from .alerts import ALERTS

if TYPE_CHECKING:
    from pinkrab_umi.keep_masks import Remote
from .launching import LaunchError, _run, _stream
from .training import LOCAL, TrainError, TrainStore, ssh_base

SAM3_SETUP_SH = r"""#!/bin/bash
# SAM 3's environment, $1 beside this file, from the packages after it. uv's
# Python and cache are the training kit's, under the same directory.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV="$ROOT/$1"
shift
export UV_PYTHON_INSTALL_DIR="$ROOT/uv-python" UV_CACHE_DIR="$ROOT/uv-cache" UV_LINK_MODE=copy
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
uv python install 3.12
[ -x "$ENV/bin/python" ] || uv venv --python 3.12 "$ENV"
VIRTUAL_ENV="$ENV" uv pip install "$@"
"$ENV/bin/python" -c "import torch, sam3; print('torch', torch.__version__, '- cuda', torch.cuda.is_available())"
printf '%s\n' "$*" > "$ENV/.packages"
"""
# SAM 3 given up on when it says nothing for this long. Its first answer,
# loading the model, can take a minute; a frame takes well under a second.
SAM_TIMEOUT_S = 180.0
# SAM 3 left unused this long is stopped, its model out of the GPU, so a
# training run on the same machine has it: a look a minute after the last
# one finds it still loaded, and one after that waits some ten seconds.
SAM_IDLE_S = 60.0
# Setting SAM 3 up the first time: some 3 GB of torch, then its 3.4 GB of weights.
SETUP_TIMEOUT_S = 40 * 60


class Machines:
    """Where SAM 3 runs for the Curate page: this machine, or one of the
    training page's ssh sources. The choice is kept (curate_file); a source
    chosen is made ready on a thread of its own - SAM 3's environment set up
    under its root, beside the training kit, and its weights fetched there -
    and nothing is asked of it until it is.

    The labels job and the preview share one SAM 3 (sam()): two of them, a
    model each, do not fit an 8 GB GPU side by side. It is stopped once
    nobody has used it for SAM_IDLE_S."""

    def __init__(self, store: TrainStore) -> None:
        self.store = store
        self._lock = threading.Lock()
        try:
            chosen = json.loads(UMI.curate_file.read_text()).get("machine") or LOCAL
        except (OSError, ValueError):
            chosen = LOCAL
        self._chosen = chosen
        self._status: dict = {}
        self._thread: threading.Thread | None = None
        self._sam = None
        self._sam_lock = threading.Lock()
        self._idle: threading.Timer | None = None
        self._prepare()

    def choose(self, name: str) -> None:
        try:
            self.store.source(name)
        except TrainError as exc:
            raise ValueError(str(exc)) from exc
        if not self._sam_lock.acquire(blocking=False):
            raise ValueError("SAM 3 is busy - choose again when it is done")
        try:
            with self._lock:
                if self._thread is not None and self._thread.is_alive():
                    raise ValueError(f"SAM 3 is still being set up on {self._chosen}")
                # Kept first: a choice that cannot be kept changes nothing.
                try:
                    UMI.curate_file.write_text(json.dumps({"machine": name}) + "\n")
                except OSError as exc:
                    raise ValueError(f"the choice could not be kept: {exc}") from exc
                self._chosen = name
            if self._sam is not None:
                self._sam.close()
                self._sam = None
        finally:
            self._sam_lock.release()
        self._prepare()

    def retry(self) -> None:
        """The machine chosen made ready again, after it was not."""
        with self._lock:
            if self._status.get("busy") or self._status.get("ready"):
                return
        self._prepare()

    def _prepare(self) -> None:
        with self._lock:
            name = self._chosen
            self._status = {"name": name, "ready": False, "busy": True, "lines": [], "error": None}
            self._thread = threading.Thread(target=self._ready, args=(name,), name="sam3-machine", daemon=True)
            self._thread.start()

    def _say(self, line: str) -> None:
        with self._lock:
            self._status["lines"] = [*self._status["lines"][-4:], line]

    def _ready(self, name: str) -> None:
        try:
            if name == LOCAL:
                if not UMI.sam3_python.is_file():
                    raise LaunchError(f"no SAM 3 environment at {UMI.sam3_python} - see the docs' install page")
            else:
                self._set_up(self.store.source(name))
            error = None
        except (LaunchError, TrainError, OSError) as exc:
            error = str(exc)
        with self._lock:
            if self._status.get("name") == name:
                self._status.update(ready=error is None, busy=False, error=error)

    def _set_up(self, source: dict) -> None:
        """SAM 3's environment on `source`, as this checkout asks for, and its
        weights - set up there where either is missing."""
        remote = self.remote_for(source)
        env, python = remote.path(UMI.sam3_env), remote.path(UMI.sam3_env, "bin", "python")
        self._say(f"asking {source['name']} what it has")
        packages = " ".join(UMI.sam3_packages)
        has = _run(source, f"cat {env}/.packages 2>/dev/null; true", timeout=60).strip()
        if has != packages:
            self._say("setting SAM 3 up there - the first time takes several minutes")
            _run(source, f"mkdir -p {remote.path()} && cat > {remote.path('sam3-setup.sh')}",
                 stdin=SAM3_SETUP_SH.encode(), timeout=60)
            _stream(source, f"bash {remote.path('sam3-setup.sh')} {shlex.quote(UMI.sam3_env)} "
                            + " ".join(shlex.quote(p) for p in UMI.sam3_packages), self._say, SETUP_TIMEOUT_S)
        # The files SAM 3's model builder asks the hub for, asked without
        # loading torch: offline first, to see they are there.
        fetch = shlex.quote("from huggingface_hub import hf_hub_download as get; "
                            "[get('facebook/sam3', f) for f in ('config.json', 'sam3.pt')]")
        try:
            _run(source, f"{remote.env()} {python} -c {fetch}", timeout=120)
        except LaunchError:
            self._say("fetching SAM 3's weights there (3.4 GB)")
            try:
                _stream(source, f"{remote.env(offline=False)} {python} -c {fetch}", self._say, SETUP_TIMEOUT_S)
            except LaunchError as exc:
                raise LaunchError(f"{source['name']} cannot fetch SAM 3's weights ({exc}) - they are gated: once "
                                  f"facebook/sam3 grants you access, log in there with "
                                  f"{source['root']}/{UMI.sam3_env}/bin/hf auth login") from exc
        self._say(f"SAM 3 is ready on {source['name']}")

    @staticmethod
    def remote_for(source: dict) -> Remote:
        from pinkrab_umi.keep_masks import Remote

        return Remote(source["name"], tuple(ssh_base(source)), source["root"])

    def remote(self) -> Remote | None:
        """Where SAM 3 is asked now: None for this machine, else a Remote.
        Raises ValueError while the machine chosen is not ready."""
        with self._lock:
            status = dict(self._status)
        if status.get("busy"):
            raise ValueError(f"SAM 3 is being set up on {status['name']}")
        if not status.get("ready"):
            raise ValueError(status.get("error") or f"SAM 3 is not ready on {status.get('name')}")
        if status["name"] == LOCAL:
            return None
        return self.remote_for(self.store.source(status["name"]))

    @contextmanager
    def sam(self, wait: bool = True) -> Iterator:
        """SAM 3 on the machine chosen, started there on first use and kept
        running until it has been idle SAM_IDLE_S, for one user at a time.
        Raises ValueError while the machine is not ready, and, without
        `wait`, while another is using it."""
        from pinkrab_umi import keep_masks

        remote = self.remote()
        if not self._sam_lock.acquire(blocking=wait):
            raise ValueError("SAM 3 is making labels")
        try:
            if self._idle is not None:
                self._idle.cancel()
            if self._sam is not None and self._sam.remote != remote:
                self._sam.close()
                self._sam = None
            if self._sam is None:
                self._sam = keep_masks.Sam(SAM_TIMEOUT_S, remote)
            yield self._sam
        finally:
            self._idle = threading.Timer(SAM_IDLE_S, self._close_idle)
            self._idle.daemon = True
            self._idle.start()
            self._sam_lock.release()

    def _close_idle(self) -> None:
        # In use again: its end starts the wait over.
        if not self._sam_lock.acquire(blocking=False):
            return
        try:
            if self._sam is not None:
                self._sam.close()
                self._sam = None
        finally:
            self._sam_lock.release()

    def state(self) -> dict:
        with self._lock:
            status = {**self._status, "lines": list(self._status.get("lines") or [])}
        names = sorted((s["name"] for s in self.store.sources()), key=lambda n: (n != LOCAL, n))
        return {**status, "machines": names}


class Stopped(Exception):
    pass


class Masker:
    def __init__(self, machines: Machines) -> None:
        self.machines = machines
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._asked: dict | None = None
        self._at: dict = {}
        self._done: list[str] = []
        self._failed: dict[str, str] = {}
        self._started = 0.0
        self._stopped = False

    def start(self, payload: dict) -> None:
        things = [str(t).strip() for t in payload.get("things") or [] if str(t).strip()]
        takes = [str(t) for t in payload.get("takes") or []]
        if not things:
            raise ValueError("name the task's things, in words - e.g. black cube, white bowl")
        if not takes:
            raise ValueError("no take to make labels for")
        if len(things) > 8:
            raise ValueError("at most 8 things")
        complete = {e["name"] for e in list_episodes() if e.get("complete")}
        for name in takes:
            if not EPISODE_NAME.match(name) or name not in complete:
                raise ValueError(f"no complete take {name!r} on the shelf")
        remote = self.machines.remote()
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise ValueError("labels are already being made")
            self._asked = {"takes": takes, "things": things, "machine": remote.name if remote else LOCAL}
            self._at, self._done, self._failed, self._stopped = {}, [], {}, False
            self._stop.clear()
            self._started = time.monotonic()
            self._thread = threading.Thread(target=self._run, name="masks", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        asked = self._asked or {}
        # SAM 3 held for the whole job: its model loaded once, not once a
        # video, and the preview left without it until the job is done.
        try:
            with self.machines.sam() as sam:
                self._each(asked, sam)
        except ValueError as exc:
            ALERTS.say("masks", str(exc))
        with self._lock:
            self._stopped = self._stop.is_set()
            self._at = {}

    def _each(self, asked: dict, sam) -> None:
        from pinkrab_umi import keep_masks

        for i, name in enumerate(asked["takes"]):
            if self._stop.is_set():
                break
            with self._lock:
                self._at = {"take": name, "index": i, "stage": "grippers"}

            def say(name=name, i=i, **line) -> None:
                if self._stop.is_set():
                    raise Stopped
                if "frame" in line:
                    with self._lock:
                        # Forward, every thing side by side; back, one thing at a time.
                        self._at = {"take": name, "index": i, "stage": line.get("stage"), "video": line.get("video"),
                                    "thing": line.get("thing"), "frame": line.get("frame"),
                                    "frames": line.get("frames")}
            try:
                keep_masks.make(UMI.episodes_dir / name, asked["things"], say, sam)
            except Stopped:
                break
            except Exception as exc:  # noqa: BLE001 - one take that fails is said, and the rest go on
                with self._lock:
                    self._failed[name] = f"{type(exc).__name__}: {exc}"
                ALERTS.say("masks", f"{name}: {exc}")
                continue
            with self._lock:
                self._done.append(name)

    def state(self) -> dict:
        with self._lock:
            running = self._thread is not None and self._thread.is_alive()
            return {"running": running, "asked": self._asked, "at": self._at, "done": list(self._done),
                    "failed": dict(self._failed), "stopped": self._stopped,
                    "seconds": round(time.monotonic() - self._started, 1) if self._asked else None}

    @staticmethod
    def takes() -> dict[str, dict]:
        """Each complete take's labels as its masks.json has them: the
        things, how many frames each was found in, the spans to look at."""
        from pinkrab_umi import keep_masks

        out = {}
        for e in list_episodes():
            if not e.get("complete"):
                continue
            account = keep_masks.account(UMI.episodes_dir / e["name"])
            if account is not None:
                out[e["name"]] = {k: account.get(k) for k in ("things", "frames", "found", "flags", "made")}
        return out


# Outlines are sent as polygons this close to the mask's edge, in pixels.
OUTLINE_PX = 1.5


def outlines(mask: np.ndarray) -> list[list[int]]:
    """A mask's outer edges as flat [x, y, x, y, ...] polygons."""
    rings, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.approxPolyDP(r, OUTLINE_PX, True).reshape(-1).tolist() for r in rings if len(r) >= 3]


class Looker:
    """The Curate page's preview: one frame of a take, the grippers drawn
    into it and the task's things looked for in it, as the labels would
    find them there, by the SAM 3 the labels job uses (Machines.sam) - kept
    running between looks, and not waited for while the job has it."""

    def __init__(self, machines: Machines) -> None:
        self.machines = machines
        self._lock = threading.Lock()
        self._scene: tuple[str, object] | None = None

    def look(self, take: str, side: str, frame: int, things: list[str]) -> dict:
        from pinkrab_umi import keep_masks

        if not EPISODE_NAME.match(take) or side not in ("left", "right"):
            raise ValueError("no such take or side")
        root = UMI.episodes_dir / take
        with self._lock:
            if self._scene is None or self._scene[0] != take:
                self._scene = (take, keep_masks.Scene(root))
            scene = self._scene[1]
            if side not in scene.sides:
                raise ValueError(f"{take} has no {side} gripper video")
            frame = max(0, min(int(frame), len(scene.stamps[side]) - 1))
            lens = scene.lenses[side]
            picture = keep_masks.picture(root, f"gripper_{side}", frame, lens.size)
            drawn = scene.draw(side, frame)
            answer = {"take": take, "side": side, "frame": frame, "frames": len(scene.stamps[side]),
                      "size": list(lens.size),
                      "grippers": outlines(drawn != 0), "things": []}
            _, jpeg = cv2.imencode(".jpg", picture, [cv2.IMWRITE_JPEG_QUALITY, 88])
            answer["picture"] = "data:image/jpeg;base64," + base64.b64encode(jpeg.tobytes()).decode()
            if things:
                try:
                    with self.machines.sam(wait=False) as sam:
                        self._found(answer, picture, things, sam)
                except ValueError as exc:
                    # Busy making labels: the frame and its grippers, and why no things.
                    answer["note"] = f"{exc} - the things are looked for again when it is done"
            return answer

    @staticmethod
    def _found(answer: dict, picture: np.ndarray, things: list[str], sam) -> None:
        """The things looked for in `picture` by `sam`, added to `answer`."""
        with tempfile.TemporaryDirectory() as tmp:
            cv2.imwrite(str(Path(tmp) / "picture.jpg"), picture, [cv2.IMWRITE_JPEG_QUALITY, 95])
            said = sam.ask({"image": str(Path(tmp) / "picture.jpg"), "texts": things,
                            "out": str(Path(tmp) / "found.npz"), "resolution": UMI.sam3_resolution,
                            "new_thing_score": UMI.sam3_new_thing_score, "thing_score": UMI.sam3_thing_score})
            if "error" in said:
                raise RuntimeError(said["error"])
            found = np.load(Path(tmp) / "found.npz")
            for k, text in enumerate(things):
                answer["things"].append({"text": text, "scores": said["scores"][k],
                                         "outlines": [p for m in found[f"t{k}"] for p in outlines(m)]})
