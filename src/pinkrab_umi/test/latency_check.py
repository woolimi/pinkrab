"""How late a step becomes the arm's motion (latency.command_lags), on a run
made here, and the gripper latency convert takes from its demonstrations'
measures (latency.settled).

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

    print("\nthe demonstrations' gripper latency")
    stamped, fps = 0.084, 50
    # Two sittings a day apart: the first's hands 10 ms late, the second's 40.
    takes = {f"a{k}": (1000.0 + 60 * k, [0.009, 0.011]) for k in range(4)}
    takes.update({f"b{k}": (90000.0 + 60 * k, [0.039, 0.041]) for k in range(4)})
    found = latency.settled(takes, stamped, fps)
    check("camera.yaml's latency moved by the median of every hand's lag",
          found["measured"] and abs(found["gripper_camera_latency_s"] - (stamped + 0.025)) < 1e-6, str(found))
    check("each sitting with its own", [(s["first"], s["takes"], s["hands"]) for s in found["sittings"]]
          == [("a0", 4, 8), ("b0", 4, 8)] and abs(found["sittings"][1]["latency_s"] - (stamped + 0.04)) < 1e-6,
          str(found["sittings"]))
    check("sittings a frame or less off the latency used do not differ", not found["sittings_differ"])
    found = latency.settled(takes, stamped, 100)
    check("and more than a frame off do", found["sittings_differ"])
    few = latency.settled({"a0": (1000.0, [0.03] * (latency.MIN_HANDS - 1))}, stamped, fps)
    check("too few hands leave camera.yaml's as it is",
          not few["measured"] and few["gripper_camera_latency_s"] == stamped, str(few))
    check("a take that does not say when it started is in no sitting",
          latency.settled({"a0": (None, [0.01] * 10)}, stamped, fps)["sittings"] == [])

    print("\n" + ("all checks passed" if not FAILURES else f"{len(FAILURES)} failed: {', '.join(FAILURES)}"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
