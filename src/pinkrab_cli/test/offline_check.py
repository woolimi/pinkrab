#!/usr/bin/env python3
"""Verify everything that does not need the robot.

Port assignment and calibration id derivation are places where a mistake does not
raise - it drives the wrong arm or silently reads someone else's calibration. Those
are exactly the parts worth checking without hardware, so this runs against a
temporary workspace and fake device files.

    python src/pinkrab_cli/test/offline_check.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Both packages, because the CLI is the surface and the library under it is what
# these checks are actually about.
SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC / "pinkrab_cli"))
sys.path.insert(0, str(SRC / "pinkrab_core"))

from pinkrab_core import ports  # noqa: E402
from pinkrab_core.config import ARMS, SETTINGS  # noqa: E402

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}" + (f" - {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(label)


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        object.__setattr__(SETTINGS, "workspace", root / "workspace")
        object.__setattr__(SETTINGS, "ports_file", root / "workspace" / "ports.json")
        SETTINGS.__class__.serial_dir = property(lambda self, _root=root: _root)  # type: ignore[assignment]

        devices = {}
        for i, arm in enumerate(ARMS):
            path = root / f"tty.usbmodemFAKE{i}"
            path.write_text("")
            devices[arm] = path

        print("\nmodel")
        check("two arms, both followers", ARMS == ("left", "right"), str(ARMS))
        check("seven servos per arm", len(SETTINGS.joint_names) == 7)
        # Six articulated joints means an end-effector target can pin a full pose -
        # position and all three orientation angles. That is what the wrist yaw was
        # added for; if this reads 5 again, the yaw fell out of the joint list and
        # every pose recorded off a handheld gripper is quietly unreachable.
        check("six articulated joints (6-DOF arm)", SETTINGS.arm_dof == 6, str(SETTINGS.arm_dof))
        check("gripper is the last motor", SETTINGS.joint_names[-1] == "gripper")
        check("wrist_yaw sits between flex and roll",
              SETTINGS.joint_names.index("wrist_yaw")
              == SETTINGS.joint_names.index("wrist_flex") + 1
              and SETTINGS.joint_names.index("wrist_roll")
              == SETTINGS.joint_names.index("wrist_yaw") + 1)
        check("motor ids run 1..7", list(SETTINGS.motor_ids.values()) == [1, 2, 3, 4, 5, 6, 7])

        print("\nports")
        saved = ports.save(devices)
        check("save writes every arm", len(saved.assigned()) == len(ARMS))
        check("stored by device name, not full path", "/" not in SETTINGS.ports_file.read_text().split('"')[3])

        loaded = ports.load(require=ARMS)
        check("round-trips", loaded.mapping == devices)

        swapped = ports.swap()
        check(
            "swap flips the pair",
            swapped.mapping["left"] == devices["right"] and swapped.mapping["right"] == devices["left"],
        )
        ports.swap()
        check("swapping twice is a no-op", ports.load().mapping == devices)

        partial = ports.save({"left": devices["left"]})
        check("partial assignment is allowed", len(partial.assigned()) == 1)
        try:
            ports.load(require=("right",))
        except ports.PortError:
            missing_raises = True
        else:
            missing_raises = False
        check("load(require=) rejects an unassigned arm", missing_raises)

        try:
            ports.save({"left": devices["left"], "right": devices["left"]})
        except ports.PortError:
            duplicate_raises = True
        else:
            duplicate_raises = False
        check("two arms cannot share one device", duplicate_raises)
        ports.save(devices)

        print("\ncalibration ids")
        check("per-arm id", SETTINGS.arm_id("left") == "pinkrab_left", SETTINGS.arm_id("left"))
        # The bimanual wrapper appends _left/_right to the pair id, so a pair driven
        # through it must land on the files the single-arm path wrote.
        check(
            "pair id derives the per-arm ids",
            f"{SETTINGS.pair_id}_left" == SETTINGS.arm_id("left")
            and f"{SETTINGS.pair_id}_right" == SETTINGS.arm_id("right"),
        )
        check(
            "calibration stays inside the workspace",
            SETTINGS.workspace in SETTINGS.calibration_file("left").parents,
        )
        # LeRobot names the calibration directory after the robot class, so the
        # class name and the setting have to agree or calibration is written in one
        # place and read from another.
        check(
            "calibration lands in the class-named directory",
            SETTINGS.calibration_file("left").parent.name == SETTINGS.follower_type,
        )
        check(
            "the two arms do not share a calibration file",
            SETTINGS.calibration_file("left") != SETTINGS.calibration_file("right"),
        )

        print("\nthe robot class")
        try:
            from pinkrab_core.robot import BiPinkrabFollower, PinkrabFollower

            check("its name is the one calibration is filed under",
                  PinkrabFollower.name == SETTINGS.follower_type)
            check("the pair's name matches too",
                  BiPinkrabFollower.name == SETTINGS.bi_follower_type)
            bus = PinkrabFollower(
                PinkrabFollower.config_class(port="/dev/null", id=SETTINGS.arm_id("left"))
            ).bus
            check(f"{len(SETTINGS.joint_names)} servos, ids in bus order",
                  [m.id for m in bus.motors.values()] == list(range(1, len(SETTINGS.joint_names) + 1)))
            check("the joints are the ones the description declares",
                  list(bus.motors) == list(SETTINGS.joint_names))
            check("only the gripper is normalised 0..100",
                  [n for n, m in bus.motors.items() if m.norm_mode.name == "RANGE_0_100"] == ["gripper"])

            # Calibration opens the arm with connect(calibrate=False) and then asks
            # for a hand on it, so that connect must not switch torque on.
            from pinkrab_core.robot import pinkrab_follower

            class Calibrated(PinkrabFollower):
                is_calibrated = True

            torque = []
            real_configure = pinkrab_follower.servos.configure
            pinkrab_follower.servos.configure = lambda bus, **kw: torque.append(kw["torque_after"])
            try:
                for calibrate in (False, True):
                    arm = Calibrated(
                        PinkrabFollower.config_class(port="/dev/null", id=SETTINGS.arm_id("left"))
                    )
                    arm.bus.connect = lambda: None
                    arm.connect(calibrate=calibrate)
            finally:
                pinkrab_follower.servos.configure = real_configure
            check("connecting to calibrate leaves torque off, connecting to drive switches it on",
                  torque == [False, True], str(torque))
        except ImportError as exc:
            print(f"  skip  LeRobot not importable ({exc})")

        print("\nenvironment handed to lerobot")
        env = SETTINGS.subprocess_env()
        check(
            "HF_LEROBOT_HOME points at the workspace",
            env["HF_LEROBOT_HOME"] == str(SETTINGS.workspace),
            env.get("HF_LEROBOT_HOME", "unset"),
        )
        check("urdf path sits under the workspace", SETTINGS.workspace in SETTINGS.urdf_file.parents)

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)}")
        for name in FAILURES:
            print(f"  - {name}")
        print()
        return 1
    print("All offline checks passed.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
