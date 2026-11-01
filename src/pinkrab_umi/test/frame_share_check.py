"""Check of the cameras' frames shared with other processes (frame_share),
with no camera: a writer process stands in for the live view's camera
thread, 30 frames a second of the gripper cameras' 1280x960 mode, and this
process reads at 50 Hz as a policy does.

- every frame read is one frame, never half of one and half of the next;
- a read, and a read with the cut a 780-wide checkpoint is shown, are fast
  next to a 20 ms tick;
- a writer that stops is said within camera_share_stale_s, not answered
  with its last frame;
- a writer started again, at another size, is found on the next read;
- one killed outright leaves nothing that stops the next;
- no writer is said as no picture;
- a frame that cannot be shared is dropped, not raised into the camera's thread;
- a reader let go of gives its lock file back.

    python src/pinkrab_umi/test/frame_share_check.py
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

import numpy as np

from pinkrab_core.dataset import gripper_shape, gripper_view
from pinkrab_umi.config import SETTINGS
from pinkrab_umi.frame_share import FrameReader, FrameWriter

CAMERA = "check"
FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}" + (f" - {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


def write_frames(shape: tuple[int, ...], seconds: float, fps: float) -> None:
    """Frames of one value each - the frame's number - so a torn read shows as two values."""
    writer = FrameWriter(CAMERA)
    frame = np.empty(shape, np.uint8)
    n, started = 0, time.monotonic()
    try:
        while time.monotonic() - started < seconds:
            n += 1
            frame.fill(n % 251)
            writer.write(frame, time.monotonic())
            time.sleep(max(0.0, started + n / fps - time.monotonic()))
    finally:
        writer.close()


def writer(shape: tuple[int, ...], seconds: float, fps: float = 30.0) -> subprocess.Popen:
    """A process of its own, as the live view is: nothing shared with this one but the block."""
    return subprocess.Popen([sys.executable, __file__, "--write", ",".join(map(str, shape)), str(seconds), str(fps)])


def wait_for(reader: FrameReader, timeout: float = 10.0) -> np.ndarray | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            return reader.read()
        except RuntimeError:
            time.sleep(0.02)
    return None


def main() -> int:
    reader = FrameReader(CAMERA)
    try:
        reader.read()
        check("no writer is no picture", False, "a frame was read with nothing writing")
    except RuntimeError as exc:
        check("no writer is no picture", "nothing is sharing" in str(exc), str(exc))

    mode = (SETTINGS.gripper_camera.height, SETTINGS.gripper_camera.width, 3)
    proc = writer(mode, seconds=4.0)
    first = wait_for(reader)
    check("a writer's frame is read", first is not None and first.shape == mode,
          None if first is None else f"{first.shape[1]}x{first.shape[0]}")

    size = gripper_shape(mode[1], mode[0])
    centre = (0.5, 0.5)
    reads, cuts, torn, values = [], [], 0, set()
    period, next_tick = 1 / 50, time.monotonic()
    for _ in range(100):
        began = time.monotonic()
        picture = reader.read()
        read_s = time.monotonic() - began
        gripper_view(picture, size, centre)
        cuts.append(time.monotonic() - began)
        reads.append(read_s)
        lo, hi = int(picture.min()), int(picture.max())
        torn += lo != hi
        values.add(lo)
        next_tick += period
        time.sleep(max(0.0, next_tick - time.monotonic()))
    check("every frame read is whole", torn == 0, f"{torn} of {len(reads)} reads held two frames")
    check("reads follow the writer", len(values) > 30, f"{len(values)} distinct frames in 100 ticks at 50 Hz")
    ms = 1000 * np.array(reads)
    cut_ms = 1000 * np.array(cuts)
    check("a read is fast", np.percentile(ms, 99) < 5.0,
          f"median {np.median(ms):.2f} ms, p99 {np.percentile(ms, 99):.2f} ms")
    check("a read and the cut are well inside a 20 ms tick", np.percentile(cut_ms, 99) < 10.0,
          f"median {np.median(cut_ms):.2f} ms, p99 {np.percentile(cut_ms, 99):.2f} ms, to {size[1]}x{size[0]}")

    proc.wait()
    time.sleep(SETTINGS.camera_share_stale_s + 0.1)
    try:
        reader.read()
        check("a stopped writer is said", False, "its last frame was still handed out")
    except RuntimeError as exc:
        check("a stopped writer is said", "nothing is sharing" in str(exc) or "old" in str(exc), str(exc))

    bigger = (1200, 1600, 3)
    proc = writer(bigger, seconds=2.0)
    again = wait_for(reader)
    check("a writer started again at another size is found",
          again is not None and again.shape == bigger, None if again is None else f"{again.shape}")
    proc.wait()

    proc = writer(mode, seconds=30.0)
    wait_for(reader)
    os.kill(proc.pid, signal.SIGKILL)
    proc.wait()
    time.sleep(SETTINGS.camera_share_stale_s + 0.1)
    proc = writer(mode, seconds=2.0)
    after = wait_for(reader)
    check("a writer killed outright does not stop the next", after is not None and after.shape == mode)
    proc.wait()

    lost = FrameWriter(CAMERA)
    try:
        lost.write(np.zeros(4, np.uint8), time.monotonic())
        lost.write(np.zeros(mode, np.uint8), time.monotonic())
        shared = wait_for(reader, 1.0)
        check("a frame that cannot be shared is dropped, and the next shared",
              shared is not None and shared.shape == mode)
    except Exception as exc:  # noqa: BLE001 - what is being checked
        check("a frame that cannot be shared is dropped, and the next shared", False, f"{type(exc).__name__}: {exc}")
    finally:
        lost.close()

    reader.close()
    before = len(os.listdir("/dev/fd"))
    for _ in range(300):
        FrameReader(CAMERA)
    after = len(os.listdir("/dev/fd"))
    check("a reader let go of gives its lock file back", after <= before + 1, f"{before} open files, then {after}")
    SETTINGS.camera_share_lock(CAMERA).unlink(missing_ok=True)
    print("frame share:", "all ok" if not FAILURES else f"{len(FAILURES)} failed: {', '.join(FAILURES)}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["--write"]:
        write_frames(tuple(int(v) for v in sys.argv[2].split(",")), float(sys.argv[3]), float(sys.argv[4]))
        sys.exit(0)
    sys.exit(main())
