"""The clutch: pedal 2 starts an episode, pedal 3 stops and saves it, and
pedal 1 cancels it - stopped and deleted.

Which page is in front decides what the pedals mean. The collect page says so
over its socket once a second while it is visible; for as long as that is recent, the
"follow the handheld" pedal means record, the stop pedal also stops the
recording (and the arms, as on every page), and the jog pedal throws the
recording away - a press each, hands free in between. On the
teleop page the pedals are the arms' again. The studio relays the two calls
and keeps nothing else; what was recorded is the live view's to say.
"""
from __future__ import annotations

import threading
import time

from .rig import live_request

START_PEDAL = "handheld"
STOP_PEDAL = "stop"
CANCEL_PEDAL = "jog"


# How long the collect page's last word keeps the pedals; it speaks every second.
IN_FRONT_S = 3.0


class Clutch:
    def __init__(self, start_lock: threading.Lock) -> None:
        self._lock = threading.Lock()
        # The studio's camera lock: an episode starts only while no switch of
        # the live view's camera pair is between its check and its act.
        self._start_lock = start_lock
        self._seen = 0.0                   # when the handheld page last said it was in front
        self.recording: str | None = None
        self.last: dict | None = None      # the last episode's status, for the page
        self.note = ""

    def seen(self) -> None:
        """The collect page, visible, once a second."""
        with self._lock:
            self._seen = time.monotonic()

    @property
    def armed(self) -> bool:
        with self._lock:
            return time.monotonic() - self._seen < IN_FRONT_S

    def pedal(self, name: str, down: bool) -> None:
        """A press (not a release) of any of the three, while armed."""
        if not down or name not in (START_PEDAL, STOP_PEDAL, CANCEL_PEDAL):
            return
        if not self.armed:
            return
        # Off the request thread: the live view takes a moment to close three
        # videos, and the next press must not wait on it. Whether an episode
        # is recording is asked of the live view there, not remembered here:
        # remembered, it went stale the moment the page's own button stopped
        # the episode or the live view was started again, and every press of
        # the record pedal after that was a press "while recording" - ignored.
        order = {START_PEDAL: self._start, STOP_PEDAL: self._stop, CANCEL_PEDAL: self._cancel}[name]
        threading.Thread(target=order, name="clutch", daemon=True).start()

    def _recording_now(self) -> bool | None:
        """Whether the live view has an episode open, or None when it does not answer."""
        answer, code = live_request("/stats", None, timeout=2.0)
        if code != 200:
            return None
        episode = answer.get("episode") or {}
        self.sync(episode.get("name") if episode.get("active") else None)
        return bool(episode.get("active"))

    def sync(self, name: str | None) -> None:
        """What the live view says is recording: its episode's name, or None."""
        with self._lock:
            self.recording = name

    def _start(self) -> None:
        with self._start_lock:
            # A second press while recording changes nothing.
            if self._recording_now():
                return
            answer, code = live_request("/episode/start", {}, timeout=10.0)
        with self._lock:
            if code == 200:
                self.recording = answer.get("name")
                self.note = ""
            else:
                self.note = answer.get("error", "could not start")

    def _stop(self, cancel: bool = False) -> None:
        # The stop or cancel pedal with nothing recording changes nothing.
        if self._recording_now() is False:
            return
        answer, code = live_request("/episode/cancel" if cancel else "/episode/stop", {}, timeout=10.0)
        with self._lock:
            self.recording = None
            if code == 200:
                if not cancel:
                    self.last = {**(answer.get("episode") or {}), "at": time.time()}
                self.note = ""
            else:
                self.note = answer.get("error", "could not stop")

    def _cancel(self) -> None:
        self._stop(cancel=True)

    def state(self) -> dict:
        armed = self.armed
        with self._lock:
            return {"armed": armed, "recording": self.recording, "last": self.last,
                    "note": self.note, "start_pedal": START_PEDAL, "stop_pedal": STOP_PEDAL}
