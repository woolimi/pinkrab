"""The clutch: pedal 2 starts an episode, pedal 3 stops and saves it.

Which page is in front decides what the pedals mean. The collect page says so
over its socket once a second while it is visible; for as long as that is recent, the
"follow the handheld" pedal means record and the stop pedal stops the
recording rather than the arms - a press each, hands free in between. On the
teleop page the pedals are the arms' again. The studio relays the two calls
and keeps nothing else; what was recorded is the live view's to say.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request

START_PEDAL = "handheld"
STOP_PEDAL = "stop"


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
        self.live_url = "http://127.0.0.1:8731"

    def seen(self) -> None:
        """The collect page, visible, once a second."""
        with self._lock:
            self._seen = time.monotonic()

    @property
    def armed(self) -> bool:
        with self._lock:
            return time.monotonic() - self._seen < IN_FRONT_S

    def pedal(self, name: str, down: bool) -> None:
        """A press (not a release) of either pedal, while armed."""
        if not down or name not in (START_PEDAL, STOP_PEDAL):
            return
        if not self.armed:
            return
        # Off the request thread: the live view takes a moment to close three
        # videos, and the next press must not wait on it. Whether an episode
        # is recording is asked of the live view there, not remembered here:
        # remembered, it went stale the moment the page's own button stopped
        # the episode or the live view was started again, and every press of
        # the record pedal after that was a press "while recording" - ignored.
        threading.Thread(target=self._start if name == START_PEDAL else self._stop,
                         name="clutch", daemon=True).start()

    def _recording_now(self) -> bool | None:
        """Whether the live view has an episode open, or None when it does not answer."""
        try:
            with urllib.request.urlopen(self.live_url + "/stats", timeout=2.0) as answer:
                episode = (json.loads(answer.read().decode()).get("episode") or {})
        except (urllib.error.URLError, TimeoutError, ValueError, OSError):
            return None
        with self._lock:
            self.recording = episode.get("name") if episode.get("active") else None
        return bool(episode.get("active"))

    def _post(self, path: str, body: dict) -> tuple[dict, int]:
        request = urllib.request.Request(self.live_url + path, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=10.0) as answer:
                return json.loads(answer.read().decode()), 200
        except urllib.error.HTTPError as exc:
            try:
                return json.loads(exc.read().decode()), exc.code
            except ValueError:
                return {"error": f"{exc.code}"}, exc.code
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
            return {"error": f"the live view did not answer: {exc}"}, 502

    def _start(self) -> None:
        with self._start_lock:
            # A second press while recording changes nothing.
            if self._recording_now():
                return
            answer, code = self._post("/episode/start", {})
        with self._lock:
            if code == 200:
                self.recording = answer.get("name")
                self.note = ""
            else:
                self.note = answer.get("error", "could not start")

    def _stop(self) -> None:
        # The stop pedal with nothing recording changes nothing.
        if self._recording_now() is False:
            return
        answer, code = self._post("/episode/stop", {})
        with self._lock:
            self.recording = None
            if code == 200:
                self.last = {**(answer.get("episode") or {}), "at": time.time()}
                self.note = ""
            else:
                self.note = answer.get("error", "could not stop")

    def state(self) -> dict:
        armed = self.armed
        with self._lock:
            return {"armed": armed, "recording": self.recording, "last": self.last,
                    "note": self.note, "start_pedal": START_PEDAL, "stop_pedal": STOP_PEDAL}
