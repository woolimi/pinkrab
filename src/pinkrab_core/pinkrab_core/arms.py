"""The metal, when there is any: both arms on their buses, read and driven.

Reading is a thread of its own rather than part of a request. Fourteen servos over a
serial bus take milliseconds to answer, and a caller asking for them at frame rate
would spend its frames waiting on a UART. So the bus is polled here and whoever
wants the numbers takes the latest ones, or asks to be handed each reading as it
lands.

One transaction on a port at a time. The SDK under the bus keeps a busy flag on
the port and answers `Port is in use!` to a second caller rather than queueing
it, so a write arriving while the poll is mid-read was a write that never
happened - and a jog that lost every other command was an arm moving in fits.
Each bus has a lock, held for exactly one transaction: a write waits the few
milliseconds a read takes instead of failing. The poll reads the whole arm in
one sync read for the same reason - seven round trips held the port seven
times as long as one.

Writing is deliberate in three ways, and each is a way this has been got wrong
before. Torque is off until somebody asks for it, because a process that opens a
port should not change what the metal is doing. Enabling torque writes the present
position as the goal FIRST - a servo remembers the last goal it was given, and
enabling torque on a stale one is an arm snapping to a pose nobody asked for just
now. And a pose for an arm whose torque is off is dropped rather than stored, for
the same reason: the write would sit in the servo waiting to be obeyed later.

What this does NOT do is decide whether a pose is a good idea. Joint bands,
contact and the ramp belong to pinkrab_safety, and the driver above this takes
only what has been through it. The one exception is `park`, which walks the
arms to a pose on the way down: by then the gate may already be gone, so the
walk is ramped here and checked against nothing but the encoder's range.
"""

from __future__ import annotations

import math
import threading
import time

from . import ports, servos
from .angles import load_offsets, radians_to_ticks, ticks_to_radians
from .bus import make_bus
from .config import SETTINGS


class ArmsError(RuntimeError):
    """Raised when the arms asked for are not there."""


class Arms:
    """Both arms on their buses, polled for where they are.

    Built eagerly: `--real` is a statement that there is hardware, so hardware that
    is not there is an error at startup rather than an empty column on a page
    somebody is trying to trust.
    """

    def __init__(self) -> None:
        self._settings = SETTINGS
        try:
            assigned = ports.load()
        except Exception as exc:  # whatever ports says, say it
            raise ArmsError(str(exc).splitlines()[0]) from exc

        arms = assigned.assigned()
        if not arms:
            raise ArmsError("no arms are assigned to a port - run `pinkrab ports detect`")

        # Ticks are what a servo reports. Turning one into a joint angle needs the
        # pairing this rig recorded, and without it there is no honest number to put
        # beside the simulation - so this is required rather than worked around.
        try:
            self._offsets = load_offsets()
        except Exception as exc:
            raise ArmsError(str(exc).splitlines()[0]) from exc

        self._buses: dict[str, object] = {}
        for arm in arms:
            try:
                bus = make_bus(assigned[arm])
                bus.connect(handshake=False)
                bus.set_baudrate(SETTINGS.baudrate)
                # The same register values LeRobot's commands write, with torque
                # left as it was found.
                servos.configure(bus)
            except Exception as exc:
                self.close()
                raise ArmsError(f"{arm} arm: {type(exc).__name__}: {exc}") from exc
            self._buses[arm] = bus

        self.arms = tuple(self._buses)
        # Which arms are holding torque. Nothing is written to an arm that is not
        # in here, and nothing is in here until somebody asked.
        self._torque: dict[str, bool] = dict.fromkeys(self._buses, False)
        self._latest: dict[str, float] = {}
        self._trouble: str | None = None
        # Reads that failed since `failed_reads` was last asked, per arm: how
        # many, and what the last of them said.
        self._failed: dict[str, tuple[int, str]] = {}
        # Called with each fresh set of angles, outside the lock. One listener, set
        # by whoever owns the reader; a reading nobody listens for is still kept.
        self.listener = None
        self._lock = threading.Lock()
        # One per port. Everything that talks to a bus takes its lock for the
        # length of one transaction and no longer; see the module docstring.
        self._bus_locks: dict[str, threading.Lock] = {
            name: threading.Lock() for name in self._buses
        }
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # --- what the page asks for ------------------------------------------
    def latest(self) -> dict:
        """The most recent reading, and whether it is current.

        A bus that has stopped answering leaves the last angles in place and says so
        rather than blanking the column: an arm that was somewhere a moment ago is
        still there, and a reading that vanished is a fact about the cable.
        """
        with self._lock:
            return {"angles_rad": dict(self._latest), "trouble": self._trouble}

    def failed_reads(self) -> dict[str, tuple[int, str]]:
        """Per arm, how many reads failed since this was last asked, and what
        the last of them said. A failed read leaves the arm's angles as they
        were and publishes nothing for it, so this is the only trace of one."""
        with self._lock:
            failed, self._failed = self._failed, {}
        return failed

    def torque_held(self) -> dict[str, bool]:
        """Which arms are holding torque, as this process last set it."""
        return dict(self._torque)

    # --- driving ------------------------------------------------------------
    def torque(self, on: bool, arm: str | None = None) -> list[str]:
        """Switch torque on or off, one arm or both. Returns the arms changed.

        Turning it ON puts the servo in position mode and writes where the arm
        already is as the goal, before the torque is enabled. A servo holds the
        last goal it was given - from this session or the one before it - and
        enabling torque against that is the arm leaving for somewhere nobody
        asked for, at whatever speed it has.
        """
        from lerobot.motors.feetech import OperatingMode

        changed: list[str] = []
        for name, bus in self._buses.items():
            if arm not in (None, name):
                continue
            try:
                with self._bus_locks[name]:
                    if on:
                        for motor in self._settings.joint_names:
                            bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
                        bus.write("Torque_Limit", "gripper",
                                  self._settings.gripper_torque_limit, normalize=False)
                        here = {
                            motor: int(value) for motor, value in bus.sync_read(
                                "Present_Position", normalize=False, num_retry=2
                            ).items()
                        }
                        bus.sync_write("Goal_Position", here, normalize=False)
                        bus.enable_torque()
                    else:
                        bus.disable_torque()
            except Exception as exc:
                raise ArmsError(f"{name} arm: {type(exc).__name__}: {exc}") from exc
            self._torque[name] = on
            changed.append(name)
        return changed

    def write(self, pose: dict[str, float]) -> list[str]:
        """Ask the servos for a pose: prefixed joint names, radians. Returns what was sent.

        Joints belonging to an arm with torque off are dropped, not queued. So are
        joints with no recorded offset and angles off the end of the encoder - see
        `angles.degrees_to_ticks`.
        """
        sent: list[str] = []
        for name, bus in self._buses.items():
            if not self._torque.get(name):
                continue
            wanted = {
                joint.removeprefix(f"{name}_"): value
                for joint, value in pose.items()
                if joint.startswith(f"{name}_")
            }
            if not wanted:
                continue
            ticks = radians_to_ticks(name, wanted, self._offsets)
            if not ticks:
                continue
            try:
                with self._bus_locks[name]:
                    bus.sync_write("Goal_Position", ticks, normalize=False)
            except Exception as exc:
                raise ArmsError(f"{name} arm: {type(exc).__name__}: {exc}") from exc
            sent.extend(f"{name}_{joint}" for joint in ticks)
        return sent

    def park(self, pose: dict[str, float], speed_dps: float, timeout_s: float,
             tolerance_deg: float | None = None, ease_s: float | None = None,
             slow_deg: float | None = None, progress=None,
             settle_deg: float | None = None, settle_s: float | None = None) -> list[str]:
        """Walk every arm that is holding torque to `pose`, ramped, and wait.

        Returns the arms that did not get there in time. The walk is `Walk`'s
        - the same one the handheld mode and the page's pose buttons use -
        from the latest reading rather than from the last goal, so an arm that
        settled below where it was sent is not first snapped back up.
        `progress`, if given, is called about once a second with how far the
        furthest joint still has to go, in degrees. Needs the reader running.

        The walk ends within `tolerance_deg` of the pose, and the pose
        is then held as sent - no ramp, no lead - for up to `settle_s`, until
        every joint reads within `settle_deg` of it. The last few degrees are
        got that way: the walk alone, stepping from readings that lag a little
        under the arm's weight, stops a joint short of where it was sent.
        """
        from .walk import Approach, Walk

        held = [arm for arm in self.arms if self._torque.get(arm)]
        if not held:
            return []
        settings = self._settings
        tolerance_deg = settings.park_tolerance_deg if tolerance_deg is None else tolerance_deg
        settle_deg = settings.park_settle_deg if settle_deg is None else settle_deg
        settle_s = settings.park_settle_s if settle_s is None else settle_s
        final = {joint: value for joint, value in pose.items()
                 if any(joint.startswith(f"{arm}_") for arm in held)}
        approach = Approach(final, tolerance_deg, speed_dps, ease_s, slow_deg)
        period = 1.0 / settings.arm_poll_hz
        started = time.monotonic()
        deadline = started + timeout_s
        spoke = started
        settling_since: float | None = None
        while True:
            now = time.monotonic()
            with self._lock:
                at = dict(self._latest)
            if settling_since is None:
                command = approach.step(at, now)
                if approach.arrived:
                    settling_since = now
                    command = dict(final)
                elif not command:
                    return held   # nothing read yet: nowhere to walk from
            else:
                if Walk.arrived(final, at, settle_deg) or now - settling_since >= settle_s:
                    return []
                command = dict(final)
            try:
                self.write(command)
            except ArmsError:
                pass  # a bus that will not take it is reported by the reading
            if progress is not None and now - spoke >= 1.0:
                spoke = now
                progress(approach.to_go_deg(at) if settling_since is None
                         else max((math.degrees(abs(final[j] - at[j])) for j in final if j in at), default=0.0))
            late = [
                arm for arm in held
                if not Walk.arrived({j: t for j, t in final.items() if j.startswith(f"{arm}_")},
                                    at, tolerance_deg)
            ]
            if now >= deadline:
                return late
            time.sleep(period)

    # --- the loop ---------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="arms", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.close()

    def close(self) -> None:
        """Let the ports go, leaving the servos as they are.

        Torque is not touched here. Whether to let go on the way out is a decision
        about the robot, and it is made by whoever owns this object - the driver
        does, before it calls this.
        """
        for bus in getattr(self, "_buses", {}).values():
            try:
                bus.disconnect(disable_torque=False)
            except Exception:  # noqa: BLE001, S110 - coming down; a port that
                pass           # will not close is not the caller's to act on
        self._buses = {}

    def _run(self) -> None:
        period = 1.0 / self._settings.arm_poll_hz
        due = time.monotonic()
        while not self._stop.is_set():
            angles: dict[str, float] = {}
            trouble: str | None = None
            failed: dict[str, str] = {}
            for arm, bus in self._buses.items():
                try:
                    with self._bus_locks[arm]:
                        raw = bus.sync_read("Present_Position", normalize=False, num_retry=2)
                    ticks = {name: int(value) for name, value in raw.items()}
                    # Prefixed, because that is what the joints are called
                    # everywhere else - the description, the MJCF and the page all
                    # say `left_shoulder_pan`, and an unprefixed name here would
                    # have to be translated by whoever received it.
                    for joint, value in ticks_to_radians(arm, ticks, self._offsets).items():
                        angles[f"{arm}_{joint}"] = round(value, 5)
                except Exception as exc:  # noqa: BLE001 - a bus is allowed to fail
                    trouble = f"{arm} arm: {type(exc).__name__}"
                    words = str(exc).strip().splitlines()
                    failed[arm] = f"{type(exc).__name__}: {words[0] if words else ''}"[:200]
            with self._lock:
                if angles:
                    self._latest.update(angles)
                self._trouble = trouble
                for arm, why in failed.items():
                    self._failed[arm] = (self._failed.get(arm, (0, ""))[0] + 1, why)
            listener = self.listener
            if angles and listener is not None:
                try:
                    listener(dict(angles))
                except Exception:  # noqa: BLE001, S110 - a listener that fails does
                    pass           # not get to stop the arms being read
            # Paced from when the last poll was due, not from when it finished:
            # the read itself takes a slice of the period, and sleeping a whole
            # period after it would poll slower than the rate says.
            due += period
            ahead = due - time.monotonic()
            if ahead > 0:
                self._stop.wait(ahead)
            else:
                due = time.monotonic()
