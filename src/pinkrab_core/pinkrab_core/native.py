"""Native libraries loaded quietly, once, before a process starts its threads.

OpenCV's and PyAV's wheels each carry their own copy of FFmpeg's
libavdevice. On macOS the second to load has the Objective-C runtime print a
line per class the two share (AVFFrameReceiver, AVFAudioReceiver) to fd 2,
under Python. The copies are the same release; the lines only say there are
two. Once both are loaded nothing more is printed, so a process that will
reach for both - the studio, the rollout node - loads them here first, while
it has one thread and shutting fd 2 for a moment loses nobody else's output.
"""
from __future__ import annotations

import os
import sys


def quiet_ffmpeg_twins() -> None:
    """Import OpenCV and PyAV with fd 2 shut, on macOS; a no-op elsewhere or
    once both are in."""
    wanted = [name for name in ("cv2", "av") if name not in sys.modules]
    if sys.platform != "darwin" or not wanted:
        return
    sys.stderr.flush()
    kept = os.dup(2)
    null = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null, 2)
        for name in wanted:
            __import__(name)
    finally:
        os.dup2(kept, 2)
        os.close(kept)
        os.close(null)
