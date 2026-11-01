"""What went wrong, for the pages to show: a process that died, a checkpoint
that would not load, a conversion or a job that failed.

Each was already said somewhere - the terminal, a note on one page - and
missed by whoever was looking at another. Said here as well, it goes down
every page's socket (the `alerts` topic) and stands in a corner of whichever
page is open until it is closed.
"""
from __future__ import annotations

import collections
import itertools
import threading
import time

# How many are kept for a page that opens late.
KEEP = 30
# The last lines of what a process said that go with its alert.
DETAIL_LINES = 15


class Alerts:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ids = itertools.count(1)
        self._kept: collections.deque = collections.deque(maxlen=KEEP)

    def say(self, source: str, message: str, detail=()) -> None:
        """One thing gone wrong: where (`source`), in a line, and what was said before it."""
        with self._lock:
            self._kept.append({"id": next(self._ids), "at": time.time(), "source": source,
                               "message": message, "detail": list(detail)[-DETAIL_LINES:]})

    def state(self) -> dict:
        with self._lock:
            return {"alerts": list(self._kept), "now": time.time()}


ALERTS = Alerts()
