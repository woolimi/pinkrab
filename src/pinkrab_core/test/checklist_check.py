"""Offline check of the before-starting list: an empty workspace wants everything
that lives in it, and the list is shaped for a page. No servo, no ROS.

    python src/pinkrab_core/test/checklist_check.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pinkrab_core.checklist import checks, with_workspace  # noqa: E402

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}" + (f" - {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        items = checks(with_workspace(Path(tmp)))
    keys = [i["key"] for i in items]
    check("every item has the page's fields",
          all({"key", "label", "ok", "detail", "run", "optional"} <= set(i) for i in items))
    check("keys are unique", len(keys) == len(set(keys)))
    wanted = ["ports", "calibration_left", "calibration_right", "offsets", "board_umi_left", "board_umi_right",
              "jaw_umi_left", "jaw_umi_right", "cameras", "lens_umi_left", "lens_umi_right", "follower_cameras",
              "lens_follower_left", "lens_follower_right"]
    for key in wanted:
        item = next((i for i in items if i["key"] == key), None)
        check(f"{key}: listed, not ok in an empty workspace, and says what to run",
              item is not None and not item["ok"] and bool(item["run"]), item["detail"] if item else "absent")
    check("both handhelds and both arms' lenses are asked for",
          all(not i["optional"] for i in items if i["key"].endswith(("umi_left", "umi_right"))
              or i["key"].startswith("lens_follower_")))
    check("python passes here", next(i for i in items if i["key"] == "python")["ok"])
    # One hand's camera named, the other not: the cameras are not named yet.
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "cameras.yaml").write_text('right:\n  mac_id: "0x1200"\n  device: /dev/gripper_right\n')
        right = {i["key"]: i for i in checks(with_workspace(Path(tmp)))}
    check("a right camera alone does not name the cameras", not right["cameras"]["ok"], right["cameras"]["detail"])
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "cameras.yaml").write_text('left:\n  mac_id: "0x1100"\nright:\n  mac_id: "0x1200"\n')
        both = {i["key"]: i for i in checks(with_workspace(Path(tmp)))}
    check("both do", both["cameras"]["ok"], both["cameras"]["detail"])
    print()
    if FAILURES:
        print(f"{len(FAILURES)} failure(s): {FAILURES}")
        sys.exit(1)
    print("all ok")
