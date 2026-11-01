"""Leave when the process that started this one is gone.

The studio starts the pedal reader through sudo on macOS, and a process
running as root cannot be signalled by the user who started it - a studio
stopping its own sidecars gets EPERM and leaves them holding the pedal; and a
studio that dies leaves every sidecar it started holding its camera or its
port. So each sidecar watches the studio instead: told its
pid, it checks once a second that the pid is still there and sends itself
a TERM, which `on_term_interrupt` below turns into the interrupt the terminal
would have sent, so its own shutdown runs as usual. TERM rather than INT
because a process started in the background has INT ignored, and so does
everything it starts.
"""
from __future__ import annotations

import os
import signal
import threading


def watch_parent(pid: int | None, period_s: float = 1.0) -> None:
    """Start the watch; nothing happens for a pid of None or 0."""
    if not pid:
        return

    def run() -> None:
        # The first look comes after a period, so the caller is inside its
        # own try by the time the interrupt can arrive.
        threading.Event().wait(period_s)
        while True:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                os.kill(os.getpid(), signal.SIGTERM)
                return
            except PermissionError:
                pass  # alive, and not ours to signal - which is all we asked
            threading.Event().wait(period_s)

    threading.Thread(target=run, name="parent-watch", daemon=True).start()


def on_term_interrupt() -> None:
    """Make TERM (and HUP) raise KeyboardInterrupt in the main thread, so the
    shutdown written for ctrl-c runs for a kill and for the parent watch too."""
    def interrupt(signum, _frame):
        raise KeyboardInterrupt(signal.Signals(signum).name)
    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupt)


def declared_parent(command: str) -> int | None:
    """The studio pid a command line says it belongs to, or None.

    Two spellings: `--parent N` on the sidecars started as programs, and
    `parent:=N` on the nodes started through `ros2 run ... --ros-args -p`.
    """
    words = command.split()
    if "--parent" in words:
        try:
            return int(words[words.index("--parent") + 1])
        except (ValueError, IndexError):
            return None
    for word in words:
        if word.startswith("parent:="):
            try:
                return int(word[len("parent:="):])
            except ValueError:
                return None
    return None


def is_orphan(command: str) -> bool:
    """Whether this command line names a studio that is gone.

    A command that names no parent was started by hand or by a test and is
    somebody's: False. A parent of 1 is init, never a studio: an orphan.
    """
    parent = declared_parent(command)
    if parent is None:
        return False
    if parent <= 1:
        return True
    try:
        os.kill(parent, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def reap_orphans(marker: str, say=print) -> int:
    """Stop earlier copies of this program whose studio is gone.

    A sidecar the studio could not stop (root, see above) is still holding
    the camera or the pedal when the next studio starts one - and the next
    one, running as root too, is the one process on the machine that can put
    it down. So each sidecar looks for others named like it before it opens
    anything: one whose --parent pid is dead, or that was started without a
    parent at all, is TERMed, then KILLed if it does not leave. A copy whose
    studio is alive is left alone; it is somebody's.
    """
    import subprocess
    import time

    me = os.getpid()
    try:
        listing = subprocess.run(["ps", "-axo", "pid=,ppid=,command="],
                                 capture_output=True, text=True, check=False, timeout=3.0).stdout
    except (OSError, subprocess.TimeoutExpired):
        return 0
    family = {me, os.getppid()}
    stale: list[int] = []
    for line in listing.splitlines():
        fields = line.split(None, 2)
        if len(fields) < 3 or not fields[0].isdigit():
            continue
        pid, ppid, command = int(fields[0]), int(fields[1]), fields[2]
        if pid in family or ppid == me or marker not in command:
            continue
        if any(word in command.split()[0] for word in ("grep", "ps")):
            continue
        # A copy that names no parent, or names 1 - launchd, never a studio -
        # was started by hand or by a test, and the pedal is not its to keep.
        if declared_parent(command) is None or is_orphan(command):
            stale.append(pid)
    for pid in stale:
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            continue
    if stale:
        time.sleep(2.0)
        for pid in stale:
            try:
                os.kill(pid, 0)
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        say(f"  put down {len(stale)} earlier {marker} left behind by a studio that is gone")
    return len(stale)
