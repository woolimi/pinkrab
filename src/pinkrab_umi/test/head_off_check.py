"""Check of the top camera put down until asked again (HeadCamera.put_down,
then retry), with no camera: the capture, its controls and the exposure are
stand-ins, and an opening can be held mid-way.

- put down from the start, it says why and opens nothing;
- asked again, it opens; put down, it closes and the view is told;
- put down while an opening is on its way, that opening opens nothing;
- the arms' live view starts with it put down, and /head/off and
  /head/retry put it down and open it, the stats saying which.

    python src/pinkrab_umi/test/head_off_check.py
"""
from __future__ import annotations

import json
import threading
import time
import urllib.request

from pinkrab_umi import head_cam, live

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}" + (f" - {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


class Capture:
    """A capture that opens, hands out one dark frame a tick, and is released."""

    def __init__(self) -> None:
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802 - OpenCV's name
        return True

    def read(self):
        time.sleep(0.02)
        import numpy as np
        return (not self.released), np.zeros((8, 8, 3), np.uint8)

    def release(self) -> None:
        self.released = True


class Controls:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


GATE = threading.Event()     # an opening waits here when held
OPENED: list[tuple[Capture, Controls]] = []


class Share:
    """Frames shared nowhere: the real top camera's block may be in use by a live view on this machine."""

    def __init__(self, camera: str) -> None:
        pass

    def write(self, picture, stamp: float) -> None:
        pass

    def close(self) -> None:
        pass


def stand_ins() -> None:
    head_cam.FrameWriter = Share
    def open_capture(device):
        GATE.wait()
        cap = Capture()
        OPENED.append((cap, None))
        return cap

    def open_controls(device, socket, usb):
        controls = Controls()
        OPENED[-1] = (OPENED[-1][0], controls)
        return controls

    head_cam.find_top = lambda: (0, "0x0000")
    head_cam._open_capture = open_capture
    head_cam.open_controls = open_controls
    head_cam.pin_exposure = lambda cap, controls, exposure_us, target: (10_000 // 10, False, "stand-in")


def wait(test, seconds: float = 3.0) -> bool:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if test():
            return True
        time.sleep(0.02)
    return False


def main() -> int:
    stand_ins()
    GATE.set()
    camera = head_cam.HeadCamera(solve=False)
    closed: list[str] = []
    camera.on_close = closed.append

    camera.put_down("off until a correction")
    check("put down from the start opens nothing", not camera.open and not camera.opening and not OPENED)
    check("and says why", camera.off and camera.reason == "off until a correction", camera.reason)

    camera.retry()
    check("asked again, it opens", wait(lambda: camera.open) and not camera.off)
    camera.put_down("off until a correction")
    cap, controls = OPENED[-1]
    check("put down, it closes", not camera.open and cap.released and controls.closed)
    check("and the view is told", closed == ["off until a correction"], str(closed))

    GATE.clear()
    camera.retry()
    wait(lambda: camera.opening, 1.0)
    camera.put_down("off until a correction")
    GATE.set()
    check("put down mid-opening, that opening opens nothing",
          wait(lambda: not camera.opening) and not camera.open and OPENED[-1][0].released and OPENED[-1][1].closed)
    camera.stop()

    # The arms' live view, served on a port of its own: put down from its start.
    original = head_cam.HeadCamera
    made: list[head_cam.HeadCamera] = []

    def recorded(*args, **kwargs):
        made.append(original(*args, **kwargs))
        return made[-1]
    head_cam.HeadCamera = recorded
    live.GripperCameras.start = lambda self: None
    live.JawReaders.start = lambda self: None
    port = 18799
    threading.Thread(target=live.run, args=(port, "follower"), daemon=True).start()

    def stats() -> dict:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/stats", timeout=2.0) as answer:
            return json.loads(answer.read())

    def post(path: str) -> None:
        request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=b"{}",
                                         headers={"Content-Type": "application/json"})
        urllib.request.urlopen(request, timeout=2.0).close()

    up = wait(lambda: bool(made) and _answers(stats), 10.0)
    check("the arms' view answers", up)
    if up:
        first = stats()
        check("it starts with the top camera put down", first["head_off"] and not made[0].open,
              f"head_off {first['head_off']}, reason {first['head_reason']!r}")
        post("/head/retry")
        check("/head/retry opens it", wait(lambda: made[0].open) and not stats()["head_off"])
        post("/head/off")
        check("/head/off puts it down", wait(lambda: not made[0].open) and stats()["head_off"])
    print("head off:", "all ok" if not FAILURES else f"{len(FAILURES)} failed: {', '.join(FAILURES)}")
    return 1 if FAILURES else 0


def _answers(stats) -> bool:
    try:
        stats()
        return True
    except OSError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
