"""Offline check of the tick-to-angle record, of the sweep judged against the
description's axes, and of the hash that says which arm a joint angle is of.
No servo, no ROS.

    python src/pinkrab_core/test/angles_check.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pinkrab_core import angles  # noqa: E402
from pinkrab_core.config import SETTINGS  # noqa: E402

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}" + (f" - {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


def record_is_an_offset() -> None:
    print("record_zero_pose")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "joint_offsets.yaml"
        home = {"left": {n: 1000 + 200 * i for i, n in enumerate(SETTINGS.joint_names)},
                "right": {n: 3000 - 200 * i for i, n in enumerate(SETTINGS.joint_names)}}
        angles.record_zero_pose("left", home["left"], path)
        angles.record_zero_pose("right", home["right"], path)
        data = angles.load_offsets(path)
        check("a fresh file carries no signs", "signs" not in data)
        for arm in ("left", "right"):
            deg = angles.ticks_to_degrees(arm, home[arm], data)
            check(f"{arm}: home ticks read as zero", max(abs(v) for v in deg.values()) < 1e-3)
            back = angles.degrees_to_ticks(arm, {n: 0.0 for n in home[arm]}, data)
            check(f"{arm}: zero degrees write the home ticks", back == home[arm])
            plus = angles.degrees_to_ticks(arm, {n: 10.0 for n in home[arm]}, data)
            check(f"{arm}: +10 deg moves every servo up its ticks",
                  all(plus[n] > home[arm][n] for n in home[arm]))
        angles.record_zero_pose("left", home["left"], path)
        check("recalibrating one arm keeps the other", "right" in angles.load_offsets(path)["offsets_deg"])

        # A file that reverses a servo means other angles: refused, and a rewrite
        # drops the other arm's offsets.
        path.write_text(path.read_text() + "signs:\n  right:\n    elbow_flex: -1\n")
        try:
            angles.load_offsets(path)
            check("a file with a reversed servo is refused", False)
        except angles.OffsetsMissing as exc:
            check("a file with a reversed servo is refused", "reverses some servos" in str(exc))
        angles.record_zero_pose("left", home["left"], path)
        data = angles.load_offsets(path)
        check("rewriting it drops the other arm's stale offsets",
              list(data["offsets_deg"]) == ["left"] and "signs" not in data)

        # Offsets are zeros of the home pose they were read at. A file from
        # another home - or from before the home was written down - is refused,
        # and calibrating one arm keeps the other only if its home still stands.
        import yaml
        angles.record_zero_pose("right", home["right"], path)
        check("each arm's offsets carry the home they were measured at",
              angles.load_offsets(path)["home_deg"] == angles.description_home())
        whole = yaml.safe_load(path.read_text())
        unmarked = {k: v for k, v in whole.items() if k != "home_deg"}
        path.write_text(yaml.safe_dump(unmarked))
        try:
            angles.load_offsets(path)
            check("a file that names no home is refused", False)
        except angles.OffsetsMissing as exc:
            check("a file that names no home is refused",
                  "another home pose" in str(exc) and "calibrate left" in str(exc)
                  and "calibrate right" in str(exc), str(exc))
        elsewhere = dict(whole, home_deg=dict(whole["home_deg"], right=dict(
            whole["home_deg"]["right"], elbow_flex=whole["home_deg"]["right"]["elbow_flex"] + 5.0)))
        path.write_text(yaml.safe_dump(elsewhere))
        try:
            angles.load_offsets(path)
            check("an arm measured at another home is refused, and named", False)
        except angles.OffsetsMissing as exc:
            check("an arm measured at another home is refused, and named",
                  "(right)" in str(exc).splitlines()[0], str(exc))
        angles.record_zero_pose("left", home["left"], path)
        data = angles.load_offsets(path)
        check("calibrating the other arm drops the one whose home moved",
              list(data["offsets_deg"]) == ["left"] and list(data["home_deg"]) == ["left"])


def sweep_judged_against_axes() -> None:
    print("sweep_direction_disagreements")
    # Bands as an arm's can be: an elbow and a jaw that count toward negative,
    # a wrist_yaw from one stop up, and a wrist_roll that turns 175 either way.
    bands = {"elbow_flex": (-185.0, 0.0), "shoulder_pan": (-119.0, 115.0),
             "wrist_yaw": (0.0, 189.0), "wrist_roll": (-175.0, 175.0), "gripper": (-80.0, 10.0)}
    k = 4096 / 360.0
    sweeps = {"elbow_flex": (0, round(-180 * k), 10),
              "shoulder_pan": (0, round(-110 * k), round(110 * k)),
              "wrist_yaw": (0, -10, round(185 * k)),
              "wrist_roll": (0, round(-170 * k), round(170 * k)),
              "gripper": (0, round(-78 * k), 20)}
    wrong = angles.sweep_direction_disagreements("left", sweeps, bands, resolution=4096)
    check("agreeing sweep passes", wrong == [], "; ".join(wrong))
    backwards = dict(sweeps, elbow_flex=(0, -10, round(180 * k)))
    wrong = angles.sweep_direction_disagreements("left", backwards, bands, resolution=4096)
    check("an elbow fitted the other way round is named",
          len(wrong) == 1 and wrong[0].startswith("elbow_flex"), "; ".join(wrong))
    turned = dict(sweeps, shoulder_pan=(0, round(-100 * k), round(120 * k)))
    wrong = angles.sweep_direction_disagreements("left", turned, bands, resolution=4096)
    check("a symmetric joint is not judged", wrong == [], "; ".join(wrong))
    short = dict(sweeps, wrist_yaw=(0, round(-30 * k), round(20 * k)))
    wrong = angles.sweep_direction_disagreements("left", short, bands, resolution=4096)
    check("a sweep too short to lean is not judged", wrong == [], "; ".join(wrong))
    check("the joints a sweep cannot judge are named",
          angles.unjudged(bands) == ["shoulder_pan", "wrist_roll"], str(angles.unjudged(bands)))


def fingerprint_follows_the_joints() -> None:
    print("arm_fingerprint")
    import xml.etree.ElementTree as ET

    from pinkrab_core.kinematics import arm_fingerprint, fresh_urdf

    left, right = arm_fingerprint("left"), arm_fingerprint("right")
    check("the same arm reads the same twice", left == arm_fingerprint("left"))
    check("each arm has its own", left != right, left)

    def moved(joint_name: str, by_m: float, into: Path) -> Path:
        """The description with one joint's origin shifted along x."""
        robot = ET.fromstring(fresh_urdf().read_text())
        origin = next(j for j in robot.findall("joint") if j.get("name") == joint_name).find("origin")
        x, y, z = (float(v) for v in origin.get("xyz").split())
        origin.set("xyz", f"{x + by_m} {y} {z}")
        into.write_text(ET.tostring(robot, encoding="unicode"))
        return into

    with tempfile.TemporaryDirectory() as tmp:
        longer = moved("left_elbow_flex", 0.01, Path(tmp) / "longer.urdf")
        check("a link a centimetre longer is another arm",
              arm_fingerprint("left", longer) != left and arm_fingerprint("right", longer) == right)
        remounted = moved("left_base_joint", 0.05, Path(tmp) / "remounted.urdf")
        check("the same arm bolted somewhere else is the same arm",
              arm_fingerprint("left", remounted) == left)


if __name__ == "__main__":
    record_is_an_offset()
    sweep_judged_against_axes()
    fingerprint_follows_the_joints()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} failure(s): {FAILURES}")
        sys.exit(1)
    print("all ok")
