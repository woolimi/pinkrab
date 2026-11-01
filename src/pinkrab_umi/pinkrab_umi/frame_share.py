"""Every camera's newest frame, handed to other processes on this machine.

A camera opens in one process - the live view, for the top camera and a set
of gripper cameras - and whatever else wants its pictures reads them from
here: each frame as it arrives, whole and unencoded, in a block of shared
memory, written by the camera's own reading thread. No other way out of the
process is kept for a camera's pictures. Fetched over HTTP instead, encoded
on asking and a connection each, a policy's two gripper frames took 21 ms of
a 50 Hz tick's 20; from here a frame is one copy each way and no wait. A
browser is the one reader that cannot map memory: the live view's page
streams are served from inside the process that holds the cameras.

A block per camera, named by camera_share_name - `top`, or a gripper set and
side (`follower_left`, `handheld_right`): a header - what it is, the frame's
height, width and channels, how many frames have been written, when the last
arrived - and the frame's pixels. The copy in and the copy out take a lock on
a file of the block's own (camera_share_lock), shared to read and whole to
write: the lock is a system call, which orders memory on any CPU, so a reader
never sees half of one frame and half of the next, nor a stamp ahead of its
pixels. Each holds it for one copy, about a millisecond.

The stamp is time.monotonic(), one clock for every process on the machine.
A frame older than camera_share_stale_s is no picture: a camera that has
stopped, or a live view that is gone, is said, not answered with its last
frame. A reader lets go of a block that has gone stale, so a block made anew
- the live view started again, the camera opened at another size - is found
on the next read.
"""
from __future__ import annotations

import fcntl
import os
import struct
import sys
import time
import weakref
from multiprocessing import resource_tracker, shared_memory

import numpy as np

from .config import SETTINGS

MAGIC = b"PKFRAME1"
# magic, height, width, channels, frames written, stamp (time.monotonic()).
HEADER = struct.Struct("<8sIIIQd")


class _Lock:
    """A lock on the block's own file: shared to read, whole to write. Its
    file is let go when closed, or when nothing holds the lock any more - a
    policy page that asks a checkpoint again and again makes a reader each time."""

    def __init__(self, camera: str) -> None:
        path = SETTINGS.camera_share_lock(camera)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        self.close = weakref.finalize(self, os.close, self._fd)

    def hold(self, mode: int) -> _Held:
        return _Held(self._fd, mode)


class _Held:
    def __init__(self, fd: int, mode: int) -> None:
        self._fd, self._mode = fd, mode

    def __enter__(self) -> None:
        fcntl.flock(self._fd, self._mode)

    def __exit__(self, *_) -> None:
        fcntl.flock(self._fd, fcntl.LOCK_UN)


class FrameWriter:
    """One camera's block, written by the thread that reads the camera."""

    def __init__(self, camera: str) -> None:
        self.name = SETTINGS.camera_share_name.format(camera=camera)
        self._lock = _Lock(camera)
        self._shm: shared_memory.SharedMemory | None = None
        self._pixels: np.ndarray | None = None
        self._count = 0
        self._closed = False
        self._failing = False

    def _make(self, shape: tuple[int, int, int]) -> None:
        self._drop()
        size = HEADER.size + int(np.prod(shape))
        try:
            self._shm = shared_memory.SharedMemory(self.name, create=True, size=size)
        except FileExistsError:
            # Left by a process that did not close it - killed outright.
            # Readers holding it find it stale and come back to this one.
            old = shared_memory.SharedMemory(self.name)
            old.close()
            old.unlink()
            self._shm = shared_memory.SharedMemory(self.name, create=True, size=size)
        self._pixels = np.ndarray(shape, np.uint8, self._shm.buf, offset=HEADER.size)
        self._count = 0

    def write(self, picture: np.ndarray, stamp: float) -> None:
        """One frame as OpenCV reads it - height x width x channels of uint8 -
        that arrived at `stamp` (time.monotonic()).

        A frame that cannot be shared is said, once, and dropped; the camera
        goes on - the thread that writes it also feeds the page and the
        recorder, and dies with whatever this raises. Nothing once closed: a
        reading thread left stuck in a read can still come back after its
        camera was stopped."""
        if self._closed:
            return
        try:
            if self._pixels is None or self._pixels.shape != picture.shape:
                self._make(picture.shape)
            self._count += 1
            with self._lock.hold(fcntl.LOCK_EX):
                np.copyto(self._pixels, picture)
                HEADER.pack_into(self._shm.buf, 0, MAGIC, *picture.shape, self._count, stamp)
        except Exception as exc:  # noqa: BLE001 - sharing a frame never stops the camera
            if not self._failing:
                print(f"  {self.name}: frames not shared - {type(exc).__name__}: {exc}", flush=True)
            self._failing = True
            self._drop()
            return
        if self._failing:
            print(f"  {self.name}: frames shared again", flush=True)
            self._failing = False

    def _drop(self) -> None:
        if self._shm is None:
            return
        # The view into the block goes first: a block with one open will not close.
        self._pixels = None
        self._shm.close()
        try:
            self._shm.unlink()
        except FileNotFoundError:
            pass
        self._shm = None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._drop()
        self._lock.close()


class FrameReader:
    """One camera's block, read from another process: the newest frame, or why there is none."""

    def __init__(self, camera: str) -> None:
        self.camera = camera
        self.name = SETTINGS.camera_share_name.format(camera=camera)
        self._lock = _Lock(camera)
        self._shm: shared_memory.SharedMemory | None = None

    def _attach(self) -> shared_memory.SharedMemory:
        if self._shm is None:
            try:
                if sys.version_info >= (3, 13):
                    shm = shared_memory.SharedMemory(self.name, track=False)
                else:
                    shm = shared_memory.SharedMemory(self.name)
                    # Before 3.13 a block opened is a block this process is
                    # taken to own, and unlinked when it ends - the live
                    # view's, from under it.
                    resource_tracker.unregister(shm._name, "shared_memory")
            except FileNotFoundError:
                raise RuntimeError(f"no picture from the {self.camera} camera: nothing is sharing it - "
                                   "is the live view up on it?") from None
            self._shm = shm
        return self._shm

    def read(self) -> np.ndarray:
        """The newest frame as the camera gave it, height x width x channels - a copy, the caller's to keep."""
        shm = self._attach()
        with self._lock.hold(fcntl.LOCK_SH):
            magic, height, width, channels, count, stamp = HEADER.unpack_from(shm.buf, 0)
            picture = None
            if magic == MAGIC and count and HEADER.size + height * width * channels <= shm.size:
                picture = np.ndarray((height, width, channels), np.uint8, shm.buf, offset=HEADER.size).copy()
        if picture is None:
            raise RuntimeError(f"no picture from the {self.camera} camera yet")
        age = time.monotonic() - stamp
        if age > SETTINGS.camera_share_stale_s:
            self._let_go()
            raise RuntimeError(f"no picture from the {self.camera} camera: its newest frame is {age:.1f} s old")
        return picture

    def _let_go(self) -> None:
        """The block given up, to be opened afresh on the next read."""
        if self._shm is not None:
            self._shm.close()
            self._shm = None

    def close(self) -> None:
        self._let_go()
        self._lock.close()
