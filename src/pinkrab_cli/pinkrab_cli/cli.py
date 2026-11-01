"""`pinkrab` - bring up the wall-mounted dual-arm SO-ARM 101."""

from __future__ import annotations

import argparse
import sys

from pinkrab_core.config import ARMS, SETTINGS

from . import commands


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pinkrab",
        description="Pinkrab - wall-mounted dual-arm SO-ARM 101",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Order:\n"
            "  1. pinkrab doctor           environment, boards, description\n"
            "  2. pinkrab ports detect     pin each board to an arm (once)\n"
            "  3. pinkrab check            every servo answers\n"
            "  4. pinkrab calibrate left   one arm at a time\n"
            "  5. pinkrab cameras          which gripper camera is which arm\n"
            "  6. pinkrab lens capture left  an arm's gripper lens, then lens fit left\n"
            "\n"
            "setup-motors writes servo ids and is not part of that order. Run it only\n"
            "when check reports servos missing or colliding on the factory id 1.\n"
            "\n"
            "The handheld UMI grippers plug into the desk, not the robot: their\n"
            "boards, jaws and cameras are pinkrab-umi's (detect, calibrate, cameras).\n"
            "The arms' cameras are the same module, named in a file of their own;\n"
            "the two sets may share the rig's camera sockets.\n"
            "\n"
            "  7. pinkrab-studio           look at it in a browser\n"
            "\n"
            "There is no teleop command yet. Both arms are followers - a leader arm\n"
            "would have to be clamped down and the board is taken - so driving them\n"
            "means inverse kinematics. The studio is where that will land, and it is\n"
            "its own command now: it comes from pinkrab_studio and needs MuJoCo,\n"
            "which nothing in this CLI does.\n"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="check environment, boards and description (read-only)")

    p_ports = sub.add_parser("ports", help="assign serial devices to arms")
    ports_sub = p_ports.add_subparsers(dest="ports_command", required=True)
    ports_sub.add_parser("show", help="print the current assignment")
    p_detect = ports_sub.add_parser("detect", help="assign by plug order (once)")
    p_detect.add_argument(
        "--arms", nargs="+", choices=ARMS, metavar="ARM",
        help=f"only these arms, in plug order (default: {' '.join(ARMS)})",
    )
    ports_sub.add_parser("swap", help="flip left and right when the arms are mirrored")
    ports_sub.add_parser("env", help='shell exports - eval "$(pinkrab ports env)"')
    ports_sub.add_parser("clear", help="forget the assignment (board replaced)")

    p_check = sub.add_parser("check", help="ping every servo (read-only)")
    p_check.add_argument(
        "--arms", nargs="+", choices=ARMS, metavar="ARM",
        help="only these arms (default: every assigned arm)",
    )

    p_setup = sub.add_parser(
        "setup-motors", help="write servo ids into EEPROM (new or renumbered servos only)"
    )
    p_setup.add_argument(
        "arm", choices=[*ARMS, "both"], nargs="?", default="both",
        help="which arm to number (default: both)",
    )
    p_setup.add_argument(
        "--joints", nargs="+", metavar="JOINT", choices=SETTINGS.joint_names,
        help=f"only these joints (default: all six, {' '.join(SETTINGS.joint_names)})",
    )
    p_setup.add_argument(
        "--port", help="write to this device instead of the stored assignment (needs one arm)"
    )
    p_setup.add_argument("-y", "--yes", action="store_true", help="skip the confirmation")

    p_cal = sub.add_parser("calibrate", help="calibrate one arm")
    p_cal.add_argument("arm", choices=ARMS)

    sub.add_parser("mount", help="measure the rig: base spacing and height above the desk")

    p_cameras = sub.add_parser(
        "cameras", help="which follower arm's gripper camera is which - plug them in one at a time"
    )
    p_cameras.add_argument("--show", action="store_true", help="what is recorded and what is on the bus")

    p_lens = sub.add_parser("lens", help="measure a follower arm's gripper camera against the ChArUco board")
    lens_sub = p_lens.add_subparsers(dest="lens_command", required=True)
    p_lens_capture = lens_sub.add_parser("capture", help="show the camera the board; views are kept as they come")
    p_lens_capture.add_argument("arm", choices=ARMS)
    p_lens_capture.add_argument("--views", type=int, default=None, help="stop at this many kept views")
    p_lens_fit = lens_sub.add_parser("fit", help="Kannala-Brandt from the kept views, into the workspace's lens.yaml")
    p_lens_fit.add_argument("arm", choices=ARMS)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "doctor":
        return commands.doctor()
    if args.command == "ports":
        if args.ports_command == "show":
            return commands.ports_show()
        if args.ports_command == "detect":
            return commands.ports_detect(tuple(args.arms) if args.arms else ARMS)
        if args.ports_command == "swap":
            return commands.ports_swap()
        if args.ports_command == "env":
            return commands.ports_env()
        if args.ports_command == "clear":
            return commands.ports_clear()
    if args.command == "check":
        return commands.check(tuple(args.arms) if args.arms else None)
    if args.command == "setup-motors":
        return commands.setup_motors(
            arm=args.arm, joints=args.joints, port=args.port, assume_yes=args.yes
        )
    if args.command == "calibrate":
        return commands.calibrate(args.arm)
    if args.command == "mount":
        return commands.mount()
    if args.command == "cameras":
        return commands.cameras(args.show)
    if args.command == "lens":
        return commands.lens(args.lens_command, args.arm, args.views if args.lens_command == "capture" else None)
    return 2


if __name__ == "__main__":
    sys.exit(main())
