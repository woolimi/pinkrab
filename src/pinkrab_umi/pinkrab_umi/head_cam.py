"""The top camera, opened on demand.

The camera on this desk comes and goes: unplugged for the handheld's cables,
replugged, or opened and then silent until it is replugged. The live view used
to need a restart for each of those. This owns the capture instead: it opens
the camera once at start, hands each frame on, puts the capture down when the
camera stops answering, and opens it again when asked - the Retry button on
the page - so a replug is followed by one click, not a restart.

The camera is a UVC camera read with OpenCV, in the mode camera.yaml's `top`
profile names; its exposure
is pinned and its gain fitted to the room through the module's own controls
(`uvc`), since a marker read over a long exposure is a marker that vanishes
when the hand moves. It stamps nothing of its own, so a frame is stamped on
arrival, less the module's latency.

The capture is read on a thread of its own: a read waits for a frame with no
limit, and a camera whose USB link dropped and came back can leave it waiting
for good. The view waits on the thread's frames instead, and a camera silent
for STALL_S is put down. Every frame the thread reads is also written to
shared memory, for any other process (frame_share).

A camera can also be put down on purpose and left closed until it is asked
for again (`put_down`, then `retry`): the arms' live view opens it only for a
correction, where it follows the handhelds - a policy is shown the grippers
alone, and the top camera's 60 frames a second, decoded and solved, are
nothing to it.
"""
from __future__ import annotations

import math
import sys
import threading
import time

import cv2
import numpy as np

from . import lens, record
from .config import SETTINGS
from .frame_share import FrameWriter
from .gripper_cams import TOP, TOP_COMMAND, find_top
from .live import Frame
from .uvc import ControlError, Controls, open_controls

# Consecutive reads that came back empty, or seconds without a frame, before
# the camera is called gone.
SILENT_LIMIT = 30
STALL_S = 2.0
# After a change from the page, how long the gain steps at once, and how often.
HURRY_S = 3.0
HURRY_STEP_S = 0.25
# The exposure the page may set, in ms: below this the picture is black in any
# room, and above the frame period the module stretches the frame.
EXPOSURE_MIN_MS = 1.0


class GainFollower:
    """Keep the picture's bright end near the target as the room changes.

    The bring-up fit puts the gain right for the room as it was; this keeps
    it right as the lamp goes on or the sun comes round, at the frame rate's
    own pace rather than the module's auto exposure - which would also lengthen
    the exposure, and a longer exposure is a marker that vanishes when the
    hand moves. One step per period, proportional, with a deadband so a steady
    room is left alone and a hand crossing the lens does not swing it.
    """

    def __init__(self, controls: Controls, target: float, period_s: float, deadband: float):
        self._controls = controls
        self.target = target
        self._period = period_s
        self._deadband = deadband
        self._next = 0.0
        # Until when it steps at once rather than every period: a target or an
        # exposure just set from the page, which wants its answer in a second.
        self._hurry = 0.0
        self._on = period_s > 0
        self.limits = controls.gain_range()
        # The gain as last read or set, and the picture's bright end as last
        # measured: what the page is shown, without a USB request per look.
        self.gain = controls.gain()
        self.bright: float | None = None

    def hurry(self) -> None:
        """Step toward the target on the next frames, not the next period."""
        self._next = 0.0
        self._hurry = time.monotonic() + HURRY_S

    def feed(self, bgr: np.ndarray) -> None:
        if not self._on:
            return
        now = time.monotonic()
        if now < self._next:
            return
        self._next = now + (HURRY_STEP_S if now < self._hurry else self._period)
        bright = _bright_end(bgr)
        self.bright = bright
        if abs(bright - self.target) <= self._deadband * self.target:
            return
        try:
            gain = self.gain = self._controls.gain()
            # From 1 at least: a gain of 0 scaled by any ratio stays 0.
            wanted = round(min(max(max(gain, 1) * self.target / bright, self.limits[0]), self.limits[1]))
            if wanted != gain:
                self._controls.set_gain(wanted)
                self.gain = wanted
        except ControlError as exc:
            # The gain is a nicety once the exposure is pinned: said once, then left.
            self._on = False
            print(f"  top camera: the gain no longer follows the room - {exc}", flush=True)


def _bright_end(picture: np.ndarray) -> float:
    grey = picture if picture.ndim == 2 else cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
    return max(1.0, float(np.percentile(grey, 99)))


def _open_capture(device: int | str) -> cv2.VideoCapture:
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_V4L2
    cap = cv2.VideoCapture(device, backend)
    profile = SETTINGS.top_camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, profile.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, profile.height)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*profile.fourcc))
    cap.set(cv2.CAP_PROP_FPS, profile.fps)
    return cap


def pin_exposure(cap: cv2.VideoCapture, controls: Controls, exposure_us: float,
                 target: float) -> tuple[int, bool, str]:
    """Pin the exposure, and fit the gain to the room in three steps of one
    measurement toward the bright end wanted: the exposure it holds, in the
    module's count, whether the camera has a gain to set, and what was set, in
    a line. A camera whose exposure cannot be held is refused - its markers
    smear as the hands move - and one without a gain is left at the exposure,
    its brightness the room's."""
    unit = SETTINGS.top_exposure_unit_us
    try:
        held = controls.manual_exposure(round(exposure_us / unit))
    except ControlError as exc:
        raise record.RecorderError(
            f"the top camera takes no manual exposure ({exc}) - a camera whose exposure cannot be held "
            "cannot be the top camera: see the docs' camera replacement page") from exc

    def settled(frames: int = 6) -> float:
        picture = None
        for _ in range(frames):
            ok, got = cap.read()
            picture = got if ok else picture
        if picture is None:
            raise record.RecorderError("the top camera opened but sent no frames")
        return _bright_end(picture)

    try:
        low, high = controls.gain_range()
        gain = min(max(controls.gain(), low, 1), high)
        for _ in range(3):
            gain = round(min(max(gain * target / settled(), low), high))
            controls.set_gain(gain)
    except ControlError as exc:
        return held, False, (f"exposure {held * unit / 1000:.1f} ms, no gain to set ({exc}) -> bright end "
                             f"{settled():.0f} of 255: the room's light and the exposure set the brightness")
    reached = settled()
    return held, True, (f"exposure {held * unit / 1000:.1f} ms, gain {gain} of {low}..{high} "
                        f"-> bright end {reached:.0f} of 255 (wanted {target:.0f})")


def top_calibration() -> dict | None:
    """The top camera's lens as fitted, as a recording keeps it (calibration.json's
    content) - read off lens.yaml, the camera open or not; None while unfitted."""
    found = lens.top_camera()
    if found is None:
        return None
    camera, source, unit = found
    return record.calibration_dict(camera, source, unit)


class HeadCamera:
    """The capture, and whether it is delivering.

    `solve` False opens the camera without its lens - for keeping views of the
    board the lens is fitted from; the live view asks for the lens, and a
    camera without a fit does not open for it.
    """

    def __init__(self, solve: bool = True) -> None:
        self._solve = solve
        self._lock = threading.Condition()
        # The exposure and the bright end the gain follows: the settings', until
        # the page sets them (set_light), and kept when the camera opens again.
        self._exposure_us = float(SETTINGS.top_exposure_us)
        self._target = float(SETTINGS.top_target_p99)
        self._cap: cv2.VideoCapture | None = None
        self._reader: threading.Thread | None = None
        self._controls: Controls | None = None
        self._follower: GainFollower | None = None
        self._calibration: dict | None = None
        self._reason = "not opened yet"
        self._opening = False
        # Put down on purpose (put_down): closed until asked again, and not
        # opened by an opening already on its way.
        self._off = False
        # The newest frame the reader has, when it arrived, how many it has
        # read, and how many the view has been handed.
        self._latest: np.ndarray | None = None
        self._arrival = 0.0
        self._read = 0
        self._seen = 0
        # Called with the calibration each time the camera comes up, and with
        # the reason each time it goes: the view takes its intrinsics from it.
        self.on_open = None
        self.on_close = None
        # Every frame, for other processes: kept across a close and an open
        # again, so a reader finds the camera back where it was.
        self.share = FrameWriter(TOP)

    @property
    def reason(self) -> str:
        return self._reason

    @property
    def open(self) -> bool:
        return self._cap is not None

    @property
    def opening(self) -> bool:
        return self._opening

    @property
    def off(self) -> bool:
        return self._off

    def start(self) -> None:
        """The first try, on its own thread: the server should listen meanwhile."""
        self.retry()

    def stop(self) -> None:
        self._close("stopped", say=False)
        self.share.close()

    # --- opening and closing ------------------------------------------------------
    def retry(self) -> bool:
        """Open the camera again, unless it is open or already being opened."""
        with self._lock:
            self._off = False
            if self._cap is not None or self._opening:
                return False
            self._opening = True
            self._reason = "opening"
        threading.Thread(target=self._try_open, name="head-cam-open", daemon=True).start()
        return True

    def put_down(self, reason: str) -> None:
        """Closed, said as `reason`, until retry asks for it again."""
        with self._lock:
            self._off = True
        self._close(reason)

    def _try_open(self) -> None:
        # Whatever stops the camera opening is said and the opening ended: a
        # thread that died opening would leave Retry refused for good, and the
        # capture it had started running.
        calibration = None
        if self._solve:
            calibration = top_calibration()
            if calibration is None:
                self._not_opened(record.RecorderError(
                    f"the top camera's lens is not fitted - {lens.lens_command(lens.TOP, 'capture')}, "
                    f"then {lens.lens_command(lens.TOP, 'fit')}"))
                return
        where = find_top()
        if where is None:
            self._not_opened(record.RecorderError(
                f"no top camera on the bus - is it plugged in where `{TOP_COMMAND}` named it?"))
            return
        device, socket = where
        cap = _open_capture(device)
        controls = None
        try:
            if not cap.isOpened():
                raise record.RecorderError(f"the top camera at {device} would not open")
            controls = open_controls(device, socket, SETTINGS.top_camera.usb)
            held, has_gain, said = pin_exposure(cap, controls, self._exposure_us, self._target)
            print(f"  top camera: {said}", flush=True)
            follower = (GainFollower(controls, self._target, SETTINGS.top_gain_follow_s, SETTINGS.top_gain_deadband)
                        if has_gain else None)
        except Exception as exc:  # noqa: BLE001
            cap.release()
            if controls is not None:
                controls.close()
            self._not_opened(exc)
            return
        with self._lock:
            # Put down while it was opening: not opened after all.
            put_down = self._off
            if put_down:
                self._opening = False
            else:
                self._cap, self._controls, self._follower = cap, controls, follower
                self._exposure_us = held * SETTINGS.top_exposure_unit_us
                self._calibration = calibration
                self._latest, self._arrival, self._read, self._seen = None, time.time(), 0, 0
                self._reason = ""
                self._opening = False
                self._reader = threading.Thread(target=self._read_loop, args=(cap,), name="head-cam-read",
                                                daemon=True)
                self._reader.start()
        if put_down:
            cap.release()
            controls.close()
            return
        print("  top camera: up", flush=True)
        if self.on_open is not None:
            self.on_open(calibration)

    def _not_opened(self, exc: Exception) -> None:
        lines = str(exc).splitlines()
        why = lines[0] if lines else ""
        if not isinstance(exc, (record.RecorderError, ControlError)):
            why = f"{type(exc).__name__}: {why}"
        with self._lock:
            self._reason = why
            self._opening = False
        print(f"  top camera: {why}", flush=True)

    def _close(self, reason: str, say: bool = True) -> None:
        with self._lock:
            cap, self._cap, self._follower = self._cap, None, None
            controls, self._controls = self._controls, None
            reader, self._reader = self._reader, None
            self._reason = reason
            self._lock.notify_all()
        if cap is not None:
            # A reader stuck in a read that will never return is left to the
            # process's exit; releasing the capture under it would hang here too.
            if reader is not None:
                reader.join(timeout=1.0)
            if reader is None or not reader.is_alive():
                cap.release()
            if controls is not None:
                controls.close()
            if say:
                print(f"  top camera: gone - {reason}", flush=True)
            if self.on_close is not None:
                self.on_close(reason)

    # --- the picture's light, from the page ------------------------------------
    def light(self) -> dict | None:
        """The exposure, the gain and the bright end, and what the gain
        follows, as last read - no request to the camera; None while it is
        not open. A camera with no gain to set has the exposure alone."""
        with self._lock:
            follower = self._follower
            exposure_us, target = self._exposure_us, self._target
            if self._controls is None:
                return None
            out = {"exposure_ms": round(exposure_us / 1000, 2),
                   "exposure_max_ms": round(1000 / SETTINGS.fps, 1), "exposure_min_ms": EXPOSURE_MIN_MS,
                   "gain_control": follower is not None}
            if follower is None:
                return {**out, "target": None, "bright": None, "gain": None, "gain_min": None, "gain_max": None}
            gain, (low, high), bright = follower.gain, follower.limits, follower.bright
        return {**out, "target": round(target), "bright": None if bright is None else round(bright),
                "gain": gain, "gain_min": low, "gain_max": high}

    def set_light(self, exposure_ms: float | None = None, target: float | None = None) -> dict | None:
        """Set the exposure (ms, within the frame period) and the bright end
        the gain follows (of 255) while the camera runs; what it holds now.

        For the room as it is today. The longer the exposure the brighter, and
        the more a moving marker smears; the higher the target, the more gain,
        and the noisier the picture."""
        for value in (exposure_ms, target):
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{value} is not a number to set")
        with self._lock:
            controls, follower = self._controls, self._follower
            if controls is None:
                return None
            if exposure_ms is not None:
                ms = min(max(float(exposure_ms), EXPOSURE_MIN_MS), 1000 / SETTINGS.fps)
                held = controls.manual_exposure(round(ms * 1000 / SETTINGS.top_exposure_unit_us))
                self._exposure_us = held * SETTINGS.top_exposure_unit_us
            if target is not None:
                self._target = min(max(float(target), 20.0), 250.0)
                if follower is not None:
                    follower.target = self._target
            if follower is not None:
                follower.hurry()
        return self.light()

    # --- the reader, and what the view reads ----------------------------------
    def _read_loop(self, cap: cv2.VideoCapture) -> None:
        misses = 0
        while self._cap is cap:
            ok, picture = cap.read()
            arrival = time.time()
            if not ok:
                # One empty read is a hiccup; a run of them is a camera that
                # has stopped, which the view finds silent and puts down.
                misses += 1
                if misses >= SILENT_LIMIT:
                    return
                continue
            misses = 0
            self.share.write(picture, time.monotonic())
            with self._lock:
                self._latest, self._arrival = picture, arrival
                self._read += 1
                self._lock.notify_all()

    def next_frame(self) -> Frame | None:
        with self._lock:
            if self._cap is None:
                self._lock.wait(0.2)
                return None
            if self._read == self._seen:
                self._lock.wait(1.0 / max(SETTINGS.fps, 1) * 3)
            silent = time.time() - self._arrival > STALL_S
            fresh = self._read != self._seen and self._cap is not None
            picture, arrival = self._latest, self._arrival
            self._seen = self._read
            number = self._read
            # Under the lock, so a close cannot put the controls down while
            # the gain is being set through them; a step is one request every
            # few seconds, or every HURRY_STEP_S for HURRY_S after the page
            # set the light.
            if fresh and picture is not None and not silent and self._follower is not None:
                self._follower.feed(picture)
        if silent:
            self._close("the camera stopped sending frames")
            return None
        if not fresh or picture is None:
            return None
        return Frame(picture=picture, sensor_ts=arrival - SETTINGS.top_camera_latency_s, frame_number=number)
