"""How late a step becomes the arm's motion (latency.command_lags), on a run made here.

    python src/pinkrab_umi/test/latency_check.py
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

from pinkrab_umi import latency
from pinkrab_umi.config import SETTINGS

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}" + (f" - {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


def angle(t: float) -> float:
    """A joint swung back and forth, 30 deg/s at its fastest, and again more slowly."""
    return math.radians(10.0) * math.sin(3.0 * t) + math.radians(4.0) * math.sin(1.1 * t)


def main() -> int:
    root = Path(tempfile.mkdtemp()) / "run"
    root.mkdir(parents=True)
    gate, servo = 0.040, 0.050
    lines = []
    for k in range(600):
        t = 100.0 + k / 60
        lines.append({"host_ts": t, "controller": "policy",
                      "asked": {"right_elbow_flex": angle(t)},
                      "sent": {"right_elbow_flex": angle(t - gate)},
                      # Read off the bus robot_state_latency_s before it is stamped.
                      "joints": {"right_elbow_flex": angle(t - SETTINGS.robot_state_latency_s - gate - servo)}})
    (root / "robot.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))
    lags = latency.command_lags(root) or {}
    check("the gate's shaping is the lag from asked to sent", abs(lags.get("gate", 1) - gate) <= 0.002, str(lags))
    check("the servo's is the lag from sent to read", abs(lags.get("servo", 1) - servo) <= 0.002, str(lags))
    check("a step is made the two after it is asked", abs(lags.get("total", 1) - gate - servo) <= 0.002, str(lags))
    (root / "robot.jsonl").write_text("".join(json.dumps({k: v for k, v in line.items() if k not in ("asked", "sent")})
                                              + "\n" for line in lines))
    check("a run recorded without its commands says nothing", latency.command_lags(root) is None)

    print("\n" + ("all checks passed" if not FAILURES else f"{len(FAILURES)} failed: {', '.join(FAILURES)}"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
