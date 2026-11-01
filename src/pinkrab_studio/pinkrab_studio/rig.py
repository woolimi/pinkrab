"""What runs beside the studio: the top camera, the foot switch, and - once the
page is switched to the real robot - the two ROS nodes the page cannot mean
anything without.

The camera view is started and stopped from the page. The foot switch starts
with the server and stops with it. The bridge and the handheld
loop are started when a page switches to the real robot - the teleop page both,
the policy page the bridge alone - and stopped when it switches back -
`ros2 run`, as a terminal would - because a page with a "follow the handheld"
button on it has to be able to answer for the loop that button needs, and a
launch file the page cannot see was a robot armed and standing still. Each is
skipped when one is already running.

The live view stays a separate process. Folding it into the studio would make
the studio need the cameras along with it. So the studio launches the same
console script a terminal would, from the environment it runs in itself, and
stops it the way ctrl-c would.

Root, on macOS, for the foot switch alone. The studio runs as whoever opened
it, and a web server cannot type a password; a keyboard's HID interface opens
for root on macOS, so `pinkrab-footswitch` is launched through `sudo -n`,
which either runs it or refuses at once, and a refusal is reported on the page
with the line of sudoers that would allow it. The line names that one file, and
sudoers then lets it run with any arguments. The file sits in the environment
the user installed into, so whoever can write it runs as root through the
rule; a copy installed where only root can write closes that. The reader posts
the presses to this studio over localhost.
"""
from __future__ import annotations

import collections
import getpass
import json
import os
import signal
import subprocess
import sys
import sysconfig
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

from pinkrab_core.config import SETTINGS as CORE
from pinkrab_umi.config import SETTINGS as UMI

from .alerts import ALERTS

# The console script lives beside this interpreter: the same install that put
# pinkrab-studio on the path put it there, and a studio launched from one
# environment must not reach into another.
BIN = Path(sys.executable).parent
LIVE = BIN / "pinkrab-umi"
FOOTSWITCH = BIN / "pinkrab-footswitch"

# How long a process gets to leave after ctrl-c, then after a term, before it is
# killed. The live view has a camera to close and the loop a servo bus; both do
# so in well under a second when they are answering at all.
GRACE_S = 3.0
TERM_S = 2.0
# How long a stopped process is waited for before one is started in its place.
GONE_S = 15.0

# How long the bridge gets to fall over before it is called started. Long enough
# for `ros2 run` to fail to find a package and say so, short enough that nobody
# watches the page wait for it.
BRIDGE_START_S = 2.5   # for the teleop loop too

TAIL_LINES = 20


class RigError(RuntimeError):
    """A start or stop that cannot be done as asked - not a process that failed."""


def bridge_already_up() -> int | None:
    return node_already_up("pinkrab_bridge", "bridge")


def node_already_up(package: str, executable: str) -> int | None:
    """The pid of this node already running, or None.

    Asked of the process table rather than of the ROS graph, because the studio
    has no rclpy in it and is not getting any. Two bridges would both push
    readings into this page and both republish its target, which is two of
    everything for no gain.

    Matched on the two shapes a bridge actually has - the installed executable,
    and the `ros2 run` that launches it - rather than on the words anywhere in a
    command line. A looser pattern matches this page's own diagnostics, a grep,
    or a test script, and then the studio quietly declines to start the bridge
    nobody else was running.
    """
    try:
        listing = subprocess.run(
            ["ps", "-axo", "pid=,command="],
            capture_output=True, text=True, check=False, timeout=3.0,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    mine = os.getpid()
    for line in listing.splitlines():
        pid, _, command = line.strip().partition(" ")
        if not pid.isdigit() or int(pid) == mine:
            continue
        if not _looks_like_node(command, package, executable):
            continue
        # Confirmed against the process itself before it is believed. A listing
        # is a snapshot, and the thing that matched may have been a command that
        # merely mentioned the bridge and has already exited - in which case the
        # studio would decline to start the bridge nobody is running.
        alive = subprocess.run(["ps", "-p", pid, "-o", "command="],
                               capture_output=True, text=True, check=False)
        if alive.returncode == 0 and _looks_like_node(alive.stdout.strip(), package, executable):
            return int(pid)
    return None


def _looks_like_bridge(command: str) -> bool:
    return _looks_like_node(command, "pinkrab_bridge", "bridge")


def robot_already_up() -> dict[str, int | None]:
    """The rig's own side - the driver and the gate - by pid, each or None.

    robot.launch.py starts both; they are looked for as nodes rather than as
    the launch, because the launch is a launcher and what matters is whether
    the two it launches are alive. The gate carries on alone when there are
    no arms plugged in, which is why they are reported one by one.
    """
    return {
        "driver": node_already_up("pinkrab_driver", "driver"),
        "gate": node_already_up("pinkrab_safety", "gate"),
    }


def _looks_like_node(command: str, package: str, executable: str) -> bool:
    """A node is the installed executable, or the ros2 run that launches it.

    Everything that merely MENTIONS it is not one, and the things that mention
    it are common: a grep for the word, a ps piped through one, a script being
    edited. Measured while writing this - the loose version reported a different
    live pid on almost every call, all of them tooling. A studio that believes
    one of those declines to start the bridge nobody is running, and the page
    then waits forever for readings.
    """
    words = command.split()
    if not words:
        return False
    # The searchers, whatever they are searching for.
    if Path(words[0]).name in {"grep", "egrep", "fgrep", "ugrep", "rg", "ps", "pgrep", "sh",
                               "bash", "zsh", "awk", "sed"}:
        return False
    installed = f"lib/{package}/{executable}"
    if any(Path(word).as_posix().endswith(installed) for word in words):
        return True
    # `ros2 run <package> <executable> [args]`, as the three words in a row.
    for i in range(len(words) - 3):
        if (Path(words[i]).name == "ros2" and words[i + 1] == "run"
                and words[i + 2] == package and words[i + 3] == executable):
            return True
    return False


def start_bridge(port: int) -> subprocess.Popen | None:
    """Run the bridge beside this studio, or say why not.

    The page is useless on its own with the real robot: the bridge is what carries
    readings in and the page's target back out. Starting it here means one
    command brings up the operator's side, and `ros2 run` is what a terminal
    would have typed.

    The port is passed because the bridge has to fetch from THIS page. Left to
    its own default it would poll 8730 whatever port this studio actually got,
    and a studio moved with --port would sit there reading nothing while a
    bridge talked to a page that is not running.
    """
    return start_node("bridge", "pinkrab_bridge", "bridge",
                      ["-p", f"studio:=http://127.0.0.1:{port}"])


def start_teleop() -> subprocess.Popen | None:
    """Run the handheld loop beside this studio, or say why not.

    The page has a button that asks for the handheld mode, and a mode with
    nobody following is a robot armed and standing still - which is what
    happened when the loop was left to a launch file the page could not see.
    The loop stays up and drives only in its mode, so it costs nothing while
    the arms are being jogged. It reads the solid from the live server on this
    machine, which is where this page starts it.
    """
    return start_node("teleop", "pinkrab_teleop", "teleop",
                      ["-p", f"live:={live_url().rstrip('/')}"])


def start_policy_node(checkpoint: str, studio_port: int, task: str = "") -> subprocess.Popen | None:
    """Run the rollout node beside this studio: the checkpoint the page chose,
    the command it was given (empty: the checkpoint's dataset's), and this
    page as the place each step is reported to.

    Started when the page asks to drive, not with the rest of the robot's
    side: the weights are chosen on the page, and a node started earlier
    would have to be started again with them anyway.
    """
    params = [
        "-p", f"checkpoint:={checkpoint}",
        "-p", f"studio:=http://127.0.0.1:{studio_port}",
    ]
    if task:
        # Quoted: a parameter is read as YAML, and a command of "no" or "1"
        # would arrive as a bool or a number where the node declares a string.
        params += ["-p", f"task:={json.dumps(task)}"]
    return start_node("policy", "pinkrab_policy", "rollout", params)


def start_node(label: str, package: str, executable: str,
               params: list[str]) -> subprocess.Popen | None:
    """`ros2 run` one of the studio's own nodes, unless one is already up.

    One that is up but belongs to a studio that is gone - the node outlived
    the page that started it, which `parent:=` is there to prevent and does
    not when the studio was killed outright - is put down first, group and
    all; a running one that names a living studio, or none, is left alone.
    Every node is told this studio's pid, so it can leave on its own when the
    studio does.
    """
    running = node_already_up(package, executable)
    if running is not None:
        if _orphan_group(running, label):
            running = None
        else:
            print(f"  {label}:  already running (pid {running}) - leaving it alone")
            return None
    return _spawn(label, ["ros2", "run", package, executable, "--ros-args",
                          "-p", f"parent:={os.getpid()}", *params],
                  f"ros2 run {package} {executable}")


def _orphan_group(pid: int, label: str) -> bool:
    """Put down the process group of `pid` if its studio is gone; say whether it was."""
    from pinkrab_core.parent import is_orphan

    alive = subprocess.run(["ps", "-p", str(pid), "-o", "command="],
                           capture_output=True, text=True, check=False)
    command = alive.stdout.strip()
    if alive.returncode != 0 or not command or not is_orphan(command):
        return False
    try:
        group = os.getpgid(pid)
    except ProcessLookupError:
        return True
    print(f"  {label}:  pid {pid} was left behind by a studio that is gone - putting it down")
    _reap_group(group)
    return True


def _child_env() -> dict[str, str]:
    """What a ROS node started from here runs in.

    On a venv beside an apt ROS the node runs on /usr/bin/python3 - colcon
    wrote that shebang - and that interpreter does not see the venv's pip
    packages, so the gate reports "mujoco is not installed" on a machine that
    has it. The two are the same interpreter, 3.12 for 3.12 (that is the whole
    reason setup.sh insists on it), so the venv's site-packages are simply put
    on the path ahead of whatever was there. Not on macOS, where there is no
    ROS to start, and not when the studio itself is not in a venv.
    """
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if sys.prefix != sys.base_prefix:
        packages = sysconfig.get_paths()["purelib"]
        # cmeel's own prefix beside it: placo lives there, reached in the venv
        # by a .pth that only a site directory runs.
        cmeel = str(Path(packages) / "cmeel.prefix" / "lib" / "python3.12" / "site-packages")
        env["PYTHONPATH"] = os.pathsep.join(p for p in (packages, cmeel, env.get("PYTHONPATH", "")) if p)
        # And the venv's shared libraries ahead of ROS's: placo carries its own
        # libeigenpy, and a sourced ROS's LD_LIBRARY_PATH would hand it the
        # older one. The same line setup.sh puts in activate, for a node that
        # does not go through activate.
        libs = str(Path(packages) / "cmeel.prefix" / "lib")
        env["LD_LIBRARY_PATH"] = os.pathsep.join(p for p in (libs, env.get("LD_LIBRARY_PATH", "")) if p)
    return env


def _spawn(label: str, command: list[str], typed: str) -> subprocess.Popen | None:
    """Start one of the rig's processes beside this page, and see it stay up.

    `typed` is the line to give the reader when it does not: what they would
    have typed in a terminal to get the same thing.
    """
    try:
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            env=_child_env(),
            # Its own session, so it can be stopped as a group. `ros2 run` is a
            # launcher that execs the node as a CHILD: signal the launcher alone
            # and it exits while the node carries on, orphaned, still holding the
            # graph - measured, a bridge left running after its studio was gone.
            start_new_session=True,
        )
    except OSError as exc:
        print(f"  {label}:  could not start it ({exc}) - run it yourself:")
        print(f"           {typed}")
        ALERTS.say(label, f"could not start it: {exc}", [typed])
        return None
    # Drained, or a node that talks fills its pipe and stops dead.
    tail: collections.deque = collections.deque(maxlen=TAIL_LINES)
    threading.Thread(target=_drain_quietly, args=(proc, tail, label), name=f"{label}-log",
                     daemon=True).start()
    # Given a moment to fail. `ros2 run` returns a process whether or not it
    # found the package, and a node that died at once is worse than one that
    # never started: the page would sit waiting while the terminal said it had
    # been started.
    try:
        proc.wait(timeout=BRIDGE_START_S)
    except subprocess.TimeoutExpired:
        proc.started = True   # past its start: a death from now on is the drain's to say
        print(f"  {label}:  started beside this page (pid {proc.pid})")
        return proc
    print(f"  {label}:  it stopped at once (exit {proc.returncode}). Start it yourself:")
    print(f"           {typed}")
    for line in list(tail)[-4:]:
        print(f"           {line}")
    ALERTS.say(label, f"stopped at once (exit {proc.returncode})", [*tail, f"start it yourself: {typed}"])
    return None


def stop_bridge(proc: subprocess.Popen | None) -> None:
    """Take the whole group down the way ctrl-c would, and wait for it.

    The GROUP, not the process: `ros2 run` execs the node as a child of its own,
    and signalling the launcher leaves that child running with nobody to stop
    it. A terminal's ctrl-c reaches both because it goes to the foreground
    group, and this is the same thing said explicitly.

    Waited on rather than signalled and forgotten: a child nobody reaps is a
    zombie, and a studio restarted in a hurry would find its own leftovers in
    the process table and decline to start a new bridge.
    """
    if proc is None:
        return
    proc.left = True    # asked to stop: its ending is not news (see _ended_well)
    if proc.poll() is not None:
        proc.wait(timeout=1.0)
        return
    try:
        group = os.getpgid(proc.pid)
    except ProcessLookupError:
        proc.wait(timeout=1.0)
        return
    for sig, wait in ((signal.SIGINT, GRACE_S), (signal.SIGTERM, TERM_S)):
        try:
            os.killpg(group, sig)
        except ProcessLookupError:
            break
        except PermissionError:
            # Root, through sudo: it watches this process and leaves on its own.
            return
        try:
            proc.wait(timeout=wait)
            break
        except subprocess.TimeoutExpired:
            continue
    else:
        try:
            os.killpg(group, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    try:
        proc.wait(timeout=TERM_S)
    except subprocess.TimeoutExpired:
        pass
    # The launcher is down; anything of its group still standing is the node it
    # exec'd, and it is not allowed to outlive the page that started it.
    _reap_group(group)


def _reap_group(group: int) -> None:
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(group, 0)
        except ProcessLookupError:
            return
        try:
            os.killpg(group, sig)
        except ProcessLookupError:
            return
        for _ in range(20):
            time.sleep(0.1)
            try:
                os.killpg(group, 0)
            except ProcessLookupError:
                return


def _drain_quietly(proc: subprocess.Popen, tail: collections.deque, label: str = "bridge") -> None:
    """Read it so the pipe cannot fill, keep the tail, print what went wrong.

    Everything is kept rather than filtered: the line that explains a bridge
    that would not start is often not shaped like an error - "Package
    'pinkrab_bridge' not found" says nothing an error filter would catch - and
    that line is the whole answer to why the page has no readings.
    """
    assert proc.stdout is not None
    # Everything it says also goes to a file in the workspace, because the
    # terminal shows only what looks wrong and the question afterwards is
    # always "what did it say before that".
    log = None
    try:
        CORE.workspace.mkdir(parents=True, exist_ok=True)
        log = (CORE.workspace / f"{label}.log").open("a")
        log.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} pid {proc.pid}\n")
    except OSError:
        log = None
    inside = False   # a traceback, being relayed whole
    for line in proc.stdout:
        text = line.rstrip()
        tail.append(text)
        if log is not None:
            log.write(text + "\n")
            log.flush()
        if text.startswith("Traceback"):
            inside = True
        elif inside and text and not text.startswith(" "):
            inside = False   # the exception line ends it - printed below
        if inside or any(word in text for word in ("error", "Error", "Traceback", "unreachable")):
            print(f"  {label}:  {text}")
    proc.stdout.close()
    code = proc.wait()
    if log is not None:
        log.write(f"--- exited with {code}\n")
        log.close()
    if not _ended_well(proc, code):
        print(f"  {label}:  exited with {code} - the page has lost it")
        # One that died within its start is said by whoever started it.
        if getattr(proc, "started", False):
            ALERTS.say(label, f"exited with {code}", tail)


def _ended_well(proc: subprocess.Popen, code: int | None) -> bool:
    """Whether a process ended well, so its ending is not news: it was asked
    to stop, or says it finished, or ctrl-c or a TERM ended it - by itself, or
    relayed by a sudo in front (128 + the signal)."""
    return getattr(proc, "left", False) or code in (
        0, None, -signal.SIGINT, -signal.SIGTERM, 128 + signal.SIGINT, 128 + signal.SIGTERM)


def _last_said(proc: subprocess.Popen | None) -> str:
    """The last line a sidecar printed, without its clock: what it is doing now."""
    # Copied first: the drain thread appends to it while this reads.
    lines = [line for line in list(getattr(proc, "tail", ())) if line.strip()]
    if not lines:
        return ""
    words = lines[-1].strip().split(maxsplit=1)
    # "01:21:38  handheld" is a pedal pressed, said with the time.
    return words[1] if len(words) == 2 and words[0].count(":") == 2 else lines[-1].strip()


def footswitch_sudoers_line() -> str:
    return f"{getpass.getuser()} ALL=(root) NOPASSWD: {FOOTSWITCH}"


def start_footswitch(port: int) -> tuple[subprocess.Popen | None, str]:
    """Run the pedal reader beside this studio, or say why not.

    Root on macOS, for the keyboard interface; through `sudo -n`, which runs it
    or refuses at once, and a refusal is one line to add to sudoers. Nothing to
    do with ROS, so this is not `ros2 run` and does not care about the domain.
    Returns the process, or None and why - the reason the page shows.
    """
    if not FOOTSWITCH.is_file():
        print(f"  pedal:  {FOOTSWITCH} is not there - pinkrab_studio is not installed here")
        return None, f"{FOOTSWITCH.name} is not installed here"
    command = [str(FOOTSWITCH), "--port", str(port), "--parent", str(os.getpid())]
    if sys.platform == "darwin" and os.geteuid() != 0:
        command = ["sudo", "-n", *command]
    try:
        proc = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"}, start_new_session=True,
        )
    except OSError as exc:
        print(f"  pedal:  could not start it ({exc})")
        ALERTS.say("pedal", f"could not start it: {exc}")
        return None, f"could not start it: {exc}"
    tail: collections.deque = collections.deque(maxlen=TAIL_LINES)
    threading.Thread(target=_drain_quietly, args=(proc, tail, "pedal"), name="pedal-log",
                     daemon=True).start()
    # What it said last, for the page: "cannot be opened" says more than "running".
    proc.tail = tail
    try:
        proc.wait(timeout=BRIDGE_START_S)
    except subprocess.TimeoutExpired:
        proc.started = True   # past its start: a death from now on is the drain's to say
        print(f"  pedal:  reading the foot switch (pid {proc.pid})")
        return proc, ""
    said = " ".join(tail)
    if "password is required" in said or "sudo:" in said:
        print("  pedal:  the foot switch needs root here and the studio cannot type a password.")
        print(f"          Allow this one command, in sudoers: {footswitch_sudoers_line()}")
        ALERTS.say("pedal", "needs root, and sudo asks a password", [f"in sudoers: {footswitch_sudoers_line()}"])
        return None, f"needs root, and sudo asks a password - in sudoers: {footswitch_sudoers_line()}"
    print(f"  pedal:  it stopped at once (exit {proc.returncode}):")
    for line in list(tail)[-4:]:
        print(f"          {line}")
    ALERTS.say("pedal", f"stopped at once (exit {proc.returncode})", tail)
    return None, _last_said(proc) or f"stopped at once (exit {proc.returncode})"


def live_url() -> str:
    return f"http://localhost:{UMI.live_port}/"


def live_request(path: str, payload: dict | None, timeout: float = 5.0) -> tuple[dict, int]:
    """One request through to the live view, its answer and status back."""
    url = live_url().rstrip("/") + path
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as answer:
            return json.loads(answer.read().decode()), 200
    except urllib.error.HTTPError as exc:
        try:
            return json.loads(exc.read().decode()), exc.code
        except ValueError:
            return {"error": f"the live view answered {exc.code}"}, exc.code
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
        return {"error": f"the live view did not answer: {exc}"}, 502


def live_hands() -> dict:
    """Where each handheld is, as the live view last solved it - each solid's
    world_T_solid against the desk solid (metres, the solid's frame), for the
    hands view and for following its motion; None where it was not - and
    each jaw's opening. Empty when the
    live server is not answering."""
    try:
        with urllib.request.urlopen(live_url() + "stats", timeout=0.8) as answer:
            stats = json.loads(answer.read().decode())
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return {}
    last = stats.get("last") or {}
    return {
        "solids": {name: c.get("world_pose") for name, c in (last.get("solids") or {}).items()},
        "jaw": {arm: (j or {}).get("aperture") for arm, j in (stats.get("jaw") or {}).items()},
        "desk": bool((last.get("world") or {}).get("tracking")),
    }


def live_stats() -> dict:
    """What the page shows beside the picture: how far the solid is and how fast
    the camera is being read - the two numbers that say whether a take is
    worth starting. Empty when the live server is not answering, which is a
    state rather than an error."""
    try:
        with urllib.request.urlopen(live_url() + "stats", timeout=0.8) as answer:
            stats = json.loads(answer.read().decode())
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return {}
    last = stats.get("last") or {}
    return {
        "fps": stats.get("fps"),
        "solved": bool(last.get("solved")),
        "distance_mm": last.get("distance_mm"),
        "faces": len(last.get("faces") or ()),
        "reason": last.get("reason") or "",
        "world": last.get("world"),
        "trouble": stats.get("trouble"),
        # Per hand, for the handheld page: each solid, and each gripper camera.
        "solids": {name: {"solved": bool(c.get("solved")), "faces": len(c.get("faces") or ()),
                          "reproj_px": c.get("reproj_px"), "distance_mm": c.get("distance_mm"),
                          "world": c.get("world_pose") is not None}
                  for name, c in (last.get("solids") or {}).items()},
        "gripper": stats.get("gripper") or {},
        "head": stats.get("head", True),
        "head_reason": stats.get("head_reason") or "",
        "head_opening": bool(stats.get("head_opening")),
        # The top camera's exposure, gain and bright end, for the page's light controls.
        "head_light": stats.get("head_light"),
        "jaw": stats.get("jaw") or {},
        "episode": stats.get("episode") or {"active": False},
        "cameras": stats.get("cameras"),
    }


class Rig:
    """Two processes by name, and what the page needs to know about them."""

    def __init__(self) -> None:
        # Set by serve(), which is the first to know which port the studio got.
        self.studio_port: int | None = None
        self._lock = threading.Lock()
        self._procs: dict[str, subprocess.Popen] = {}
        self._tails: dict[str, collections.deque] = {}
        self._options: dict[str, dict] = {}
        self._robot: dict[str, subprocess.Popen | None] = {}
        self._robot_asked = False
        self._robot_parts: tuple[str, ...] = self.ROBOT_PARTS
        self._robot_steps: dict[str, str] = {}
        self._robot_thread: threading.Thread | None = None
        # Parts asked for and not started yet, taken in order by that thread.
        self._robot_pending: list[str] = []
        self._footswitch: subprocess.Popen | None = None
        # Why the last start of it came to nothing, for the page.
        self._footswitch_failed = ""
        self._footswitch_starting = False
        # The rollout node, while the page is driving with a policy.
        self._policy: subprocess.Popen | None = None
        self._policy_options: dict = {}
        # One start or stop of it at a time. Waiting for the node to go or to
        # stay up takes seconds, and `_lock` is not held for that: every page's
        # poll of `state()` would wait with it.
        self._policy_turn = threading.Lock()

    # --- what the page reads ----------------------------------------------

    def state(self) -> dict:
        with self._lock:
            proc = self._footswitch
            alive = proc is not None and proc.poll() is None
            policy = self._policy
            policy_alive = policy is not None and policy.poll() is None
            return {"live": self._one("live"), "robot": self._robot_state(),
                    "footswitch": {"running": alive, "pid": proc.pid if alive else None,
                                   "exit": None if proc is None or alive else proc.returncode,
                                   "note": _last_said(proc), "failed": self._footswitch_failed},
                    "policy": {"running": policy_alive, "pid": policy.pid if policy_alive else None,
                               "exit": None if policy is None or policy_alive else policy.returncode,
                               **self._policy_options}}

    # --- the policy node --------------------------------------------------

    def start_policy(self, checkpoint: str, task: str = "") -> dict:
        """The rollout node with this checkpoint and command; a running one is
        stopped first, because the page may have chosen differently."""
        with self._policy_turn:
            with self._lock:
                old = self._policy
            if old is not None and old.poll() is None:
                stop_bridge(old)
            proc = start_policy_node(checkpoint, self.studio_port or CORE.studio_port, task)
            with self._lock:
                self._policy = proc
                self._policy_options = {"checkpoint": checkpoint, "task": task}
            if proc is None:
                raise RigError("the rollout node did not start - see the studio's terminal")
        return self.state()["policy"]

    def stop_policy(self) -> dict:
        with self._policy_turn:
            with self._lock:
                proc = self._policy
            stop_bridge(proc)
            with self._lock:
                self._policy = None
                self._policy_options = {}
        return self.state()["policy"]

    # --- the foot switch: the studio's own, up for as long as the page is ------
    # Not the robot's: the pedals record on the collect page with no robot at
    # all, so this starts with the server and stops with it.

    def start_footswitch(self, port: int) -> None:
        # Started outside the lock: it waits a few seconds to see the reader
        # stay up, and every page's poll of `state()` would wait with it.
        with self._lock:
            if self._footswitch_starting or (self._footswitch is not None and self._footswitch.poll() is None):
                return
            self._footswitch_starting = True
        try:
            proc, failed = start_footswitch(port)
        finally:
            with self._lock:
                self._footswitch_starting = False
        with self._lock:
            self._footswitch, self._footswitch_failed = proc, failed

    # --- the robot's side: bridge and teleop loop -------------------------------
    # Started as a page switches to the real robot - the parts that page asks
    # for that are not up yet - and stopped together when it switches back.
    # Each is skipped when one is already up.

    # What the teleop page brings up: the operator's side. The policy page asks
    # for the bridge alone, which carries what the rig says into the page, and
    # nothing that drives by hand. The rig's own launch is not among them: its
    # terminal is where its output belongs, and it is looked for, not started.
    ROBOT_PARTS = ("bridge", "teleop")
    STARTERS: ClassVar[dict[str, tuple[Callable, Callable]]] = {
        "bridge": (lambda port: start_bridge(port),
                   lambda: node_already_up("pinkrab_bridge", "bridge")),
        "teleop": (lambda port: start_teleop(),
                   lambda: node_already_up("pinkrab_teleop", "teleop")),
    }

    def start_robot(self, port: int, parts: tuple[str, ...] | None = None,
                    only_if_asked: bool = False) -> dict:
        """Start the parts asked for, one after another, saying which is being started.

        Returns at once; the work runs on a thread and `state()` carries its
        progress, because each start waits a few seconds to see the process
        stay up and a page that blocked for all of them showed nothing meanwhile.
        Parts already up are kept and only the missing ones started: the policy
        page asks for the bridge alone, and the teleop page after it still
        needs its handheld loop. `only_if_asked` is that page's load: it tops
        up a robot side asked for and not since stopped, and starts nothing
        once `stop_robot` has begun - the check is under the same lock.
        """
        # Once each: a name twice would start it, then find it and drop its handle.
        parts = tuple(dict.fromkeys(parts or self.ROBOT_PARTS))
        with self._lock:
            if only_if_asked and not self._robot_asked:
                return self._robot_state()
            # The starting thread clears its own handle under this lock as it
            # finishes, so a part queued here while it runs is never left.
            busy = self._robot_thread is not None
            # Up: started here and still running. One found running elsewhere,
            # or one that failed, is looked for again - `already()` - each time.
            up = {name for name, proc in self._robot.items() if proc is not None and proc.poll() is None}
            waiting = set(self._robot_pending) | {name for name, step in self._robot_steps.items()
                                                  if busy and step == "starting"}
            missing = tuple(name for name in parts if name not in up and name not in waiting)
            if not missing:
                return self._robot_state()
            if not busy:
                self._robot = {name: proc for name, proc in self._robot.items() if name in up}
                self._robot_steps = {name: self._robot_steps.get(name, "already") for name in self._robot}
                self._robot_parts = tuple(self._robot)
            self._robot_parts = tuple(dict.fromkeys(self._robot_parts + parts))
            for name in missing:
                self._robot_steps[name] = "pending"
            self._robot_pending.extend(missing)
            self._robot_asked = True
            if not busy:
                self._robot_thread = threading.Thread(target=self._start_robot, args=(port,),
                                                      name="rig-robot", daemon=True)
                self._robot_thread.start()
            return self._robot_state()

    def _start_robot(self, port: int) -> None:
        while True:
            with self._lock:
                if not self._robot_pending:
                    # Not a later thread's: one outlived stop_robot's wait.
                    if self._robot_thread is threading.current_thread():
                        self._robot_thread = None
                    return
                name = self._robot_pending.pop(0)
                self._robot_steps[name] = "starting"
            start, already = self.STARTERS[name]
            try:
                pid = already()
                proc = None if pid is not None else start(port)
            except Exception as exc:  # noqa: BLE001 - one part, not the thread that clears the handle
                print(f"  {name}:  could not start it ({exc})")
                with self._lock:
                    self._robot_steps[name] = "failed"
                continue
            if pid is not None:
                with self._lock:
                    self._robot[name] = None
                    self._robot_steps[name] = "already"
                continue
            with self._lock:
                self._robot[name] = proc
                self._robot_steps[name] = "running" if proc is not None else "failed"

    def stop_robot(self) -> dict:
        with self._lock:
            thread = self._robot_thread
            self._robot_pending.clear()   # what has not started yet does not
            # ...and a page load from here on tops nothing up.
            self._robot_asked = False
        if thread is not None and thread.is_alive():
            thread.join(timeout=15.0)
        with self._lock:
            going = list(self._robot.values())
            self._robot = {}
            # The foot switch is the server's, not the robot side's: it stays,
            # and the page goes on hearing its presses. A stop pressed now
            # reaches the gate only through a bridge still up elsewhere or
            # started within REQUEST_TTL_S - the one stopped here takes nothing.
            self._robot_steps = {}
            self._robot_thread = None
        # Waited for with the lock free, as in `stop`.
        for proc in going:
            stop_bridge(proc)
        with self._lock:
            return self._robot_state()

    def _robot_state(self) -> dict:
        parts = {}
        busy = self._robot_thread is not None and self._robot_thread.is_alive()
        for name in self._robot_parts:
            proc = self._robot.get(name)
            alive = proc is not None and proc.poll() is None
            step = self._robot_steps.get(name, "")
            # A step that was running and has since died reads as exited.
            if step == "running" and not alive:
                step = "exited"
            parts[name] = {"running": alive, "pid": proc.pid if alive else None,
                           "exit": None if proc is None or alive else proc.returncode,
                           "step": step}
        return {"asked": self._robot_asked, "starting": busy, "parts": parts}


    def _one(self, which: str) -> dict:
        proc = self._procs.get(which)
        running = proc is not None and proc.poll() is None
        exit_code = None if proc is None or running else proc.returncode
        tail = [line for line in list(self._tails.get(which, ())) if line.strip()]
        note = ""
        if exit_code not in (None, 0):
            # The last line that says something went wrong, else the last
            # line at all; the page shows the whole tail beside it.
            said = [line for line in tail if not line.startswith("log -> ")]
            wrong = [line for line in said
                     if any(word in line.lower() for word in ("error", "failed", "not ", "traceback"))]
            note = (wrong or said)[-1] if said else f"exited with {exit_code}"
        return {
            "running": running,
            "pid": proc.pid if running else None,
            "exit": exit_code,
            "note": note,
            "options": dict(self._options.get(which, {})),
            "tail": tail[-8:],
        }

    # --- start and stop ---------------------------------------------------

    def start(self, which: str, options: dict | None = None) -> dict:
        """Start it with these options. Running with these already is an
        error; running with others - the live view on the other set of
        cameras - it is put down and started again with these."""
        options = dict(options or {})
        with self._lock:
            proc = self._procs.get(which)
            switch = (proc is not None and proc.poll() is None
                      and self._options.get(which, {}) != options)
        if switch:
            self._stop_and_wait(which)
        with self._lock:
            proc = self._procs.get(which)
            if proc is not None and proc.poll() is None:
                raise RigError(f"{which} is already running (pid {proc.pid})")
            command = self._command(which, options)
            tail: collections.deque = collections.deque(maxlen=TAIL_LINES)
            # Unbuffered, so what the process prints arrives in the order it
            # was printed: piped, python holds stdout back and lets stderr
            # through, and the error then reads as if it came before the line
            # that led up to it.
            env = _child_env()
            try:
                proc = subprocess.Popen(
                    command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1, cwd=str(Path.home()), env=env,
                )
            except OSError as exc:
                raise RigError(f"could not start {which}: {exc}") from exc
            self._procs[which] = proc
            self._tails[which] = tail
            self._options[which] = options
            threading.Thread(target=self._drain, args=(proc, tail, which),
                             name=f"rig-{which}", daemon=True).start()
            return self._one(which)

    def stop(self, which: str) -> dict:
        with self._lock:
            proc = self._procs.get(which)
            if proc is None or proc.poll() is not None:
                raise RigError(f"{which} is not running")
        # Left with the lock free: the process takes up to GRACE_S + TERM_S to
        # go, and every poll of `state()` would wait for it.
        self._leave(proc)
        with self._lock:
            return self._one(which)

    def restart(self, which: str) -> dict | None:
        """Stop and start again with the same options, if it is running; else nothing."""
        with self._lock:
            proc = self._procs.get(which)
            running = proc is not None and proc.poll() is None
            options = dict(self._options.get(which, {}))
        if not running:
            return None
        self._stop_and_wait(which)
        return self.start(which, options)

    def _stop_and_wait(self, which: str) -> None:
        """Stop it and wait until it is gone. The live view takes a moment to
        let go of its port and its cameras; one started meanwhile finds both
        taken."""
        with self._lock:
            proc = self._procs.get(which)
            if proc is None or proc.poll() is not None:
                return   # gone on its own, or stopped by a Disconnect meanwhile
        self._leave(proc)
        try:
            proc.wait(timeout=GONE_S)
        except subprocess.TimeoutExpired as exc:
            raise RigError(f"{which} did not go down within {GONE_S:g} s") from exc

    def stop_all(self) -> None:
        with self._lock:
            procs = list(self._procs.values())
            bridges = [*self._robot.values(), self._policy, self._footswitch]
            self._robot = {}
            self._policy = None
            self._footswitch = None
        for proc in procs:
            if proc.poll() is None:
                self._leave(proc)
        for proc in bridges:
            stop_bridge(proc)

    @staticmethod
    def _leave(proc: subprocess.Popen) -> None:
        proc.left = True    # asked to stop: its ending is not news (see _ended_well)
        # ctrl-c first: the live view catches it and puts its cameras down.
        for sig, wait in ((signal.SIGINT, GRACE_S), (signal.SIGTERM, TERM_S)):
            try:
                proc.send_signal(sig)
                proc.wait(timeout=wait)
                return
            except subprocess.TimeoutExpired:
                continue
            except ProcessLookupError:
                return
        proc.kill()
        proc.wait(timeout=TERM_S)

    @staticmethod
    def _drain(proc: subprocess.Popen, tail: collections.deque, which: str) -> None:
        # Read what the process says so it never blocks on a full pipe, and keep
        # the last few lines: they are the answer to "why did it stop".
        assert proc.stdout is not None
        for line in proc.stdout:
            tail.append(line.rstrip("\n"))
        proc.stdout.close()
        code = proc.wait()
        if not _ended_well(proc, code):
            ALERTS.say(which, f"exited with {code}", tail)

    def _command(self, which: str, options: dict) -> list[str]:
        if which == "live":
            if not LIVE.is_file():
                raise RigError(f"{LIVE} is not there - pinkrab_umi is not installed in "
                               "this environment")
            command = [str(LIVE), "live", "--parent", str(os.getpid())]
            if options.get("solid"):
                command += ["--solid", str(options["solid"])]
            if options.get("cameras"):
                command += ["--cameras", str(options["cameras"])]
            return command
        raise RigError(f"no such process: {which!r}")
