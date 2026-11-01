"""Offline check of who a sidecar says it belongs to, and whether that studio is gone.

    python src/pinkrab_core/test/parent_check.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pinkrab_core.parent import declared_parent, is_orphan  # noqa: E402

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}" + (f" - {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


if __name__ == "__main__":
    me = os.getpid()
    dead = 2 ** 22 - 7   # above pid_max on every default kernel: nobody
    print("declared_parent")
    check("--parent N", declared_parent(f"/x/pinkrab-footswitch --port 8730 --parent {me}") == me)
    check("parent:=N", declared_parent(f"python3 /x/bridge --ros-args -p studio:=http://a -p parent:={me}") == me)
    check("none declared", declared_parent("python3 /x/bridge --ros-args -p studio:=http://a") is None)
    check("garbage is None", declared_parent("/x/live --parent soon") is None)
    print("is_orphan")
    check("a living studio: not an orphan", not is_orphan(f"/x/bridge --ros-args -p parent:={me}"))
    check("a dead studio: an orphan", is_orphan(f"/x/bridge --ros-args -p parent:={dead}"))
    check("parent 1 (init): an orphan", is_orphan("/x/live --parent 1"))
    check("no parent named: somebody's, left alone", not is_orphan("/x/bridge --ros-args"))
    print()
    if FAILURES:
        print(f"{len(FAILURES)} failure(s): {FAILURES}")
        sys.exit(1)
    print("all ok")
