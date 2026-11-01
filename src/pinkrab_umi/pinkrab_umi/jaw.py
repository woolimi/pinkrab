"""The handhelds' jaw servos, read for how far each hand is pinching.

One servo per handheld, on a board of its own, read on a thread of its own:
a serial read is a round trip with a timeout on the end, and a handheld whose
cable has worked loose answers with that timeout every time. The live view
owns the reading, beside the cameras, so one process holds every sensor on
the handheld and the recorder and the teleop loop both take the latest
number from /stats. A reading older than `stale_s` counts as no reading.
"""
from __future__ import annotations

import threading
import time

from pinkrab_core import angles, bus, ports
from pinkrab_core.config import SETTINGS as CORE


class JawReader:
    stale_s = 0.5
    silent_after = 5
    retry_s = 3.0

    def __init__(self, side: str, say=print) -> None:
        self.side = side
        self.device = f"umi_{side}"
        self.say = say
        self._lock = threading.Lock()
        self._latest: tuple[float, int, float] | None = None   # (aperture, ticks, monotonic)
        self._note = "not opened yet"
        self._misses = 0
        self._silent = False
        self.stopping = threading.Event()
        # Set by `retry` (and by `stop`): the wait between attempts ends at
        # once, and an open bus is closed and opened again - ports and
        # calibration read afresh.
        self._wake = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"jaw-{side}", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self.stopping.set()
        self._wake.set()
        self._thread.join(timeout=2.0)

    def retry(self) -> None:
        """Open the jaw's bus again, now - the Refresh on the page."""
        self._wake.set()

    def aperture(self) -> float | None:
        with self._lock:
            latest = self._latest
        if latest is None or time.monotonic() - latest[2] > self.stale_s:
            return None
        return latest[0]

    def stats(self) -> dict:
        with self._lock:
            latest = self._latest
            note = self._note
        fresh = latest is not None and time.monotonic() - latest[2] < self.stale_s
        return {"aperture": latest[0] if fresh else None, "ticks": latest[1] if fresh else None,
                "note": "" if fresh else note}

    def _open(self):
        """The bus, or the reason there is none. Each reason costs the jaw, not the view."""
        try:
            port = ports.load(require=(self.device,))[self.device]
        except ports.PortError as exc:
            return None, None, str(exc).splitlines()[0]
        try:
            calibration = angles.load_handheld()
            angles.handheld_aperture(self.device, 0, calibration)
        except (angles.HandheldCalibrationMissing, KeyError, ValueError) as exc:
            return None, None, str(exc).splitlines()[0]
        try:
            link = bus.make_handheld_bus(port)
            link.connect(handshake=False)
            link.set_baudrate(CORE.baudrate)
        except Exception as exc:  # noqa: BLE001 - a missing board is a state, not a crash
            return None, None, f"cannot open {port.name}: {str(exc).splitlines()[0]}"
        return link, calibration, ""

    def _run(self) -> None:
        period = 1.0 / CORE.teleop_fps
        while not self.stopping.is_set():
            link, calibration, why = self._open()
            if link is None:
                with self._lock:
                    self._note = why
                self._wake.wait(self.retry_s)
                self._wake.clear()
                continue
            self.say(f"  jaw {self.side}: reading {self.device}")
            try:
                while not self.stopping.is_set() and not self._wake.is_set():
                    started = time.monotonic()
                    ticks = bus.read_raw(link, CORE.handheld_joint, retries=0)
                    if ticks is None:
                        self._misses += 1
                        if self._misses == self.silent_after and not self._silent:
                            self._silent = True
                            with self._lock:
                                self._note = "the jaw servo has gone silent - check its cable and power"
                            self.say(f"  jaw {self.side}: {self._note}")
                    else:
                        opening = angles.handheld_aperture(self.device, int(ticks), calibration)
                        with self._lock:
                            self._latest = (float(opening), int(ticks), time.monotonic())
                        if self._silent:
                            self._silent = False
                            self.say(f"  jaw {self.side}: answers again")
                        self._misses = 0
                    self.stopping.wait(max(0.0, period - (time.monotonic() - started)))
            finally:
                try:
                    link.disconnect(disable_torque=False)
                except Exception:  # noqa: BLE001
                    pass
            self._wake.clear()


class JawReaders:
    def __init__(self, say=print) -> None:
        self.sides = {side: JawReader(side, say) for side in ("left", "right")}

    def start(self) -> None:
        for reader in self.sides.values():
            reader.start()

    def stop(self) -> None:
        for reader in self.sides.values():
            reader.stop()

    def apertures(self) -> dict[str, float | None]:
        return {side: reader.aperture() for side, reader in self.sides.items()}

    def stats(self) -> dict:
        return {side: reader.stats() for side, reader in self.sides.items()}

    def retry(self, side: str) -> None:
        self.sides[side].retry()
