"""What the pages watch: the studio's state as topics, sent down each page's
WebSocket once per change.

A page names the topics it shows. A topic is a function of the studio's state
and how often it is worth asking, asked on a thread of its own while anyone
looks - so a slow one (a process table, a camera list, the live view's HTTP)
holds up no page's socket - and every tab watching it shares that answer. A
page is sent a topic's value when it starts watching and again each time the
value changes - nothing while it stays the same.

The page talks back on the same socket: which topics, which page it is and
whether it is in front, where its policy trace has got to, and the poses its
sliders ask of the simulation. Anything that is an action with an answer -
start this, delete that - stays an ordinary POST.
"""
from __future__ import annotations

import json
import math
import threading
import time
from typing import Callable

from pinkrab_core.config import SETTINGS as CORE
from pinkrab_mujoco.physics import STREAM_HZ
from pinkrab_umi.config import SETTINGS as UMI
from pinkrab_umi.episode import list_episodes

from .alerts import ALERTS
from .converting import Converter
from .policy import checkpoints
from .rig import bridge_already_up, live_hands, live_stats, live_url, node_already_up
from .wsock import Closed, WebSocket

# The policy page's trace: sent as steps arrive, from where that page has got
# to, rather than as one value.
HISTORY = "policy_history"
# Longest a page's socket waits between looks when it watches nothing faster.
TICK_S = 0.25
# A page that has had nothing new for this long is pinged: a socket that has
# gone quiet and one that has died look the same until something is sent.
PING_S = 15.0
# The page whose being in front makes the pedals record.
COLLECT_PAGE = "collect"
POLICY_PAGE = "policy"
# A topic nobody has read for this long stops being asked until somebody does.
IDLE_S = 5.0


class Topic:
    def __init__(self, name: str, every: float, produce: Callable[[], object], stop: threading.Event,
                 volatile: tuple[str, ...] = ()):
        self.name = name
        self.every = every
        self._produce = produce
        self._stop = stop
        # Keys that change on every ask and say nothing by themselves - a
        # clock - are left out of whether the value changed, and go along
        # with it when something else has.
        self._volatile = volatile
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._read_at = -math.inf
        self._thread: threading.Thread | None = None
        self._last: tuple[str, object] | None = None
        self._failed: str | None = None

    def touch(self) -> None:
        """Asked again now: something just changed it."""
        self._wake.set()

    def read(self) -> tuple[str, object] | None:
        """The last answer - what decides whether it changed, and the value -
        or None before the first. Reading is what keeps it being asked."""
        with self._lock:
            now = time.monotonic()
            if now - self._read_at > IDLE_S:
                # It was resting, and what it said last may be long out of
                # date - a wait on it would be over before it began. Nothing
                # is said until it has been asked again, now rather than a
                # period from now.
                self._last = None
                self._wake.set()
            self._read_at = now
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name=f"topic {self.name}", daemon=True)
                self._thread.start()
            return self._last

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                looked_at = time.monotonic() - self._read_at <= IDLE_S
            if looked_at:
                answer = self._ask()
                with self._lock:
                    self._last = answer
            self._wake.wait(self.every if looked_at else IDLE_S)
            self._wake.clear()

    def _ask(self) -> tuple[str, object]:
        try:
            value = self._produce()
            basis = value
            if self._volatile and isinstance(value, dict):
                basis = {k: v for k, v in value.items() if k not in self._volatile}
            # Made into text here, where a value that cannot be is one more
            # failure to say, not the end of this topic's thread.
            answer = json.dumps(basis, separators=(",", ":")), value
            self._failed = None
            return answer
        except Exception as exc:  # noqa: BLE001 - a topic failing must not stop it being asked again
            error = {"error": f"{type(exc).__name__}: {exc}"}
            if error["error"] != self._failed:
                print(f"  {self.name}: {error['error']}")
                ALERTS.say(self.name, error["error"])
                self._failed = error["error"]
            return json.dumps(error), error


class Feeds:
    """Every topic a page can watch, built on the studio's state."""

    def __init__(self, state) -> None:
        self.state = state
        self._datasets: tuple[tuple, list[dict]] = ((None,), [])
        self._datasets_lock = threading.Lock()
        rate = 1.0 / STREAM_HZ

        def topic(name: str, every: float, produce: Callable[[], object], volatile: tuple[str, ...] = ()) -> Topic:
            return Topic(name, every, produce, state.stopping, volatile)

        topics = [
            # The simulation and, beside it, the processes the page started
            # and, with the metal, what the bridge says of it.
            topic("physics", rate, self._physics, volatile=("time",)),
            topic("rig", 0.5, self._rig),
            # The live view's numbers, while it runs. Not asked otherwise: the
            # studio would wait on a port nobody answers.
            topic("stats", 0.5, lambda: live_stats() if state.rig.state()["live"]["running"] else None),
            # The handhelds against the desk solid, at the camera's pace, for
            # the hands view. The live server has no push, so it is asked.
            topic("hands", 1.0 / 30.0,
                  lambda: live_hands() if state.rig.state()["live"]["running"] else None),
            topic("pedal", 0.1, lambda: state.servo_state().get("pedal")),
            topic("servo", 0.25, self._servo),
            topic("robot", 2.0, lambda: {"driver": node_already_up("pinkrab_driver", "driver"),
                                         "gate": node_already_up("pinkrab_safety", "gate")}),
            topic("clutch", 0.5, state.clutch.state),
            topic("convert", 0.5, state.converter.state),
            # Takes' keep labels: the job making them, and what each take has.
            topic("masks", 1.0, lambda: {"job": state.masker.state(), "takes": state.masker.takes(),
                                         "machine": state.machines.state()}),
            topic("episodes", 1.0, lambda: {"episodes": list_episodes(), "dir": str(UMI.episodes_dir)}),
            # Takes whose replay is still being solved are named as pending,
            # and asking is what sets their solving going.
            topic("verdicts", 1.0, state.verdicts.get),
            topic("datasets", 1.0, lambda: {"datasets": self.datasets(), "dir": str(Converter.datasets_dir())}),
            topic("policy", 0.25, state.policy.state),
            topic("dagger", 0.25, state.dagger.state),
            topic("home", 5.0, self._home),
            topic("train", 2.0, self._train, volatile=("now",)),
            topic("train_jobs", 1.0, state.train_jobs.state),
            # What went wrong, for whichever page is open - see alerts.py.
            topic("alerts", 0.5, ALERTS.state, volatile=("now",)),
        ]
        # The gripper cameras on the bus and which side each is in a set's
        # file, a set per topic - see gripper_cams.
        from pinkrab_umi.cameras import present
        from pinkrab_umi.gripper_cams import CAMERA_SETS, mapping
        for which in CAMERA_SETS:
            topics.append(topic(f"cameras/{which}", 3.0,
                                lambda which=which: {"which": which, "present": present(which), "mapping": mapping(which)}))
        self.topics = {t.name: t for t in topics}

    def touch(self, *names: str) -> None:
        for name in names:
            self.topics[name].touch()

    # --- the topics that are more than one call ----------------------------
    def _rig(self) -> dict:
        return {**self.state.rig.state(), "live_url": live_url()}

    def _physics(self) -> dict:
        state = self.state
        frame = state.physics.snapshot()
        frame["rig"] = self._rig()
        if state.real.expected:
            # Composed here rather than inside Physics: the simulation has no
            # business knowing whether there is metal.
            frame["measured"] = state.real.snapshot()
        else:
            frame["sim_hands"] = state.sim_hand.state()
        return frame

    def _servo(self) -> dict:
        state = self.state
        answer = state.servo_state()
        if state.real.expected and not answer.get("heard"):
            # Nothing has come. Say which of the two usual reasons it is
            # rather than leaving the page to repeat "waiting" forever.
            answer["quiet"] = state.real.why_quiet(CORE.shell_ros_domain(), bridge_already_up())
        return answer

    def datasets(self) -> list[dict]:
        """Every dataset, read again only when one has come, gone or been
        finished: summing a dataset's files is the slow part, and the
        listing is asked every second."""
        root = Converter.datasets_dir()
        stamp: tuple = ()
        if root.is_dir():
            def mtime(p):
                try:
                    return p.stat().st_mtime
                except OSError:
                    return None
            # pinkrab_convert.json is the last a conversion writes.
            stamp = tuple(sorted((p.name, mtime(p), mtime(p / "meta" / "info.json"),
                                  mtime(p / "meta" / "pinkrab_convert.json")) for p in root.iterdir()))
        with self._datasets_lock:
            if stamp != self._datasets[0]:
                self._datasets = (stamp, Converter.list_datasets())
            return self._datasets[1]

    def _home(self) -> dict:
        """The front page's numbers: what has been recorded, what is
        training, what can be run."""
        state = self.state
        episodes = list_episodes()
        store = state.training.store
        runs = store.runs()
        return {
            "episodes": len(episodes),
            "episode_seconds": round(sum(e.get("seconds") or 0 for e in episodes)),
            "datasets": len(self.datasets()),
            "labelled": len(state.masker.takes()),
            "runs": len(runs),
            "training": [{"name": r["name"], "step": r["last_step"], "steps": r["steps"],
                          "eta_s": r["eta_s"], "source": r["source"]}
                         for r in runs if r["status"] == "running"],
            # On a machine's GPUs and no recorded run's: seen, not charted.
            "unrecorded": [{**u, "source": src["name"]} for src in store.sources()
                           for u in src["unrecorded"] or []],
            "checkpoints": len(checkpoints()),
            "real": bool(state.real.expected),
        }

    def _train(self) -> dict:
        store = self.state.training.store
        return {"runs": store.runs(), "sources": store.sources(), "now": time.time()}


def serve(ws: WebSocket, feeds: Feeds) -> None:
    """One page's socket, until the page goes or the studio stops."""
    state = feeds.state
    watching: list[str] = []
    said: dict[str, str] = {}
    page = ""
    since = 0            # the last policy step this page has
    # The trace is sent whole, and drawn afresh, on a new socket and when the
    # page starts it over: a page that reconnects still holds the steps it had.
    restart = True
    spoke = time.monotonic()
    try:
        while not state.stopping.is_set():
            tick = min([feeds.topics[n].every for n in watching if n in feeds.topics] + [TICK_S])
            for text in ws.receive(tick):
                try:
                    message = json.loads(text)
                except ValueError:
                    continue
                if not isinstance(message, dict):
                    continue
                if isinstance(message.get("watch"), list):
                    watching = [n for n in message["watch"] if n in feeds.topics or n == HISTORY]
                    said = {n: b for n, b in said.items() if n in watching}
                if "page" in message:
                    page = str(message["page"])
                # In front, as the page says once a second while it is visible:
                # what makes the floor's pedals record rather than drive. Armed
                # by what is heard, so a page gone quiet - closed, frozen, off
                # the network - lapses.
                if message.get("visible") is True and page == COLLECT_PAGE:
                    state.clutch.seen()
                # And on the policy page they take the policy's arm over (dagger.py).
                if message.get("visible") is True and page == POLICY_PAGE:
                    state.dagger.seen()
                if isinstance(message.get("since"), int):
                    since, restart = message["since"], True
                for verb in ("ctrl", "place"):
                    if verb in message:
                        _drive(state, verb, message[verb])
            for name in watching:
                if name == HISTORY:
                    since = _send_history(ws, state, since, restart)
                    restart = False
                    continue
                answer = feeds.topics[name].read()
                if answer is None:
                    continue        # not asked yet: sent once it has been
                basis, value = answer
                if said.get(name) != basis:
                    ws.send(json.dumps({"topic": name, "data": value}, separators=(",", ":")))
                    said[name] = basis
                    spoke = time.monotonic()
            if time.monotonic() - spoke >= PING_S:
                ws.ping()
                spoke = time.monotonic()
        ws.close(1001)   # the studio is coming down
    except Closed:
        return


def _send_history(ws: WebSocket, state, since: int, restart: bool) -> int:
    """The policy steps the page has not drawn, and where it has got to now.
    The page is told to start its trace over when asked to (`restart`) and
    when a run started since numbers its steps from one again."""
    rows = state.policy.history(since)
    reset = restart or rows["latest"] < since
    if reset:
        rows = state.policy.history(0)
    if rows["steps"] or reset:
        ws.send(json.dumps({"topic": HISTORY, "data": {"steps": rows["steps"], "reset": reset}},
                           separators=(",", ":")))
    return rows["steps"][-1]["step"] if rows["steps"] else (0 if reset else since)


def _drive(state, verb: str, pose) -> None:
    """A pose from the page's sliders: asked of the simulated servos (ctrl),
    or put outright (place) - the difference that is the whole reason physics
    is here, see Physics.command and Physics.place. Anything that is not
    {joint: number} is dropped: the next pose of the gesture carries the same
    intent."""
    if not isinstance(pose, dict):
        return
    try:
        values = {str(name): float(v) for name, v in pose.items()}
    except (TypeError, ValueError):
        return
    if verb == "ctrl":
        state.physics.command(values)
    else:
        state.physics.place(values)
