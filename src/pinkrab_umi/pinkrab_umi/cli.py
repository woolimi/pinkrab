"""`pinkrab-umi` - print the marker solid, record the top camera, judge a take."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pinkrab_core.config import HANDHELDS

from .config import SETTINGS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pinkrab-umi",
        description="Top-camera 6-DoF pose of the handheld UMI gripper, against the desk solid",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Order:\n"
            "  1. pinkrab-umi marker ./print    sheet and solid - print, cut, glue\n"
            "  2. pinkrab-umi cameras --top     which socket the top camera is in\n"
            "  3. pinkrab-umi lens capture top  the top camera's lens (live view down); then lens fit top\n"
            "  4. pinkrab-umi live              watch it solve; record takes from the page\n"
            "  5. pinkrab-umi bench <session>   re-judge a recorded take\n"
            "\n"
            "Takes are recorded from the live page, which shows the closure verdict\n"
            "the moment one ends. bench recomputes the same verdict from the session\n"
            "on disk. Every pose is in the frame of the desk solid (marker --solid desk),\n"
            "read off the same picture as the hand's.\n"
            "\n"
            "live needs the top camera, fitted; marker and bench run on any machine.\n"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_marker = sub.add_parser("marker", help="write the printable sheet and the solid STL")
    p_marker.add_argument("out", nargs="?", default=".", type=Path,
                          help="directory to write into (default: here)")
    p_marker.add_argument("--solid", default=SETTINGS.solid,
                          choices=[c.name for c in SETTINGS.solids],
                          help="which solid to print (default: %(default)s)")

    p_live = sub.add_parser("live", help="watch the solid solve in a browser, live")
    p_live.add_argument("--port", type=int, default=SETTINGS.live_port,
                        help=f"port to serve on (default: {SETTINGS.live_port})")
    p_live.add_argument("--parent", type=int, default=0,
                        help="the studio's pid: leave when it is gone")
    p_live.add_argument("--solid", default=SETTINGS.solid,
                        choices=[c.name for c in SETTINGS.solids],
                        help="which solid is being held (default: %(default)s)")
    p_live.add_argument("--cameras", default="handheld", choices=("handheld", "follower"),
                        help="whose gripper cameras to open: the handhelds' to record, "
                             "the follower arms' for a policy (default: %(default)s)")
    _detector_args(p_live)

    # The handheld boards - one servo each. See boards.py.
    for name, helptext in (
        ("detect", "assign the handhelds' boards by plug order (the arms keep theirs)"),
        ("setup-motor", "write the gripper id into a handheld's servo (new or renumbered only)"),
        ("check", "ping a handheld's servo (read-only)"),
        ("calibrate", "pair a handheld's servo ticks with the two ends of its jaw's travel"),
        ("read", "print where a handheld's servo is (read-only)"),
    ):
        sp = sub.add_parser(name, help=helptext)
        if name == "detect":
            sp.add_argument("devices", nargs="*", choices=HANDHELDS, metavar="HANDHELD",
                            help=f"only these handhelds, in plug order (default: {' '.join(HANDHELDS)})")
        else:
            sp.add_argument("device", nargs="?", default=HANDHELDS[0], choices=HANDHELDS,
                            help=f"which handheld (default: {HANDHELDS[0]})")
            sp.add_argument("--port", help="use this device instead of the stored assignment")
        if name == "setup-motor":
            sp.add_argument("-y", "--yes", action="store_true", help="skip the confirmation")
        if name == "read":
            sp.add_argument("--seconds", type=float, default=0.0,
                            help="keep printing for this long (default: once)")
            sp.add_argument("--aperture", action="store_true",
                            help="also show how far open the jaw is, 0 shut to 1 wide")

    p_cameras = sub.add_parser("cameras", help="which gripper camera is which hand - plug them in one at a time")
    p_cameras.add_argument("--show", action="store_true", help="what is recorded and what is on the bus")
    p_cameras.add_argument("--top", action="store_true", help="which socket the top camera is in - one plug")

    p_lens = sub.add_parser("lens", help="measure a camera's lens against a ChArUco board - a gripper's, or the top camera's")
    lens_sub = p_lens.add_subparsers(dest="lens_command", required=True)
    p_board = lens_sub.add_parser("board", help="write the board, one A4 page, as PDF and an SVG to see it by")
    p_board.add_argument("out_dir", nargs="?", default="print", help="where to write it (default: ./print)")
    p_capture = lens_sub.add_parser("capture", help="show the camera the board; views are kept as they come "
                                                    "(top: with the live view down)")
    p_capture.add_argument("side", choices=("left", "right", "top"))
    p_capture.add_argument("--views", type=int, default=None,
                           help=f"stop at this many kept views (default: {SETTINGS.lens_views}, "
                                f"top {SETTINGS.top_lens_views})")
    p_fit = lens_sub.add_parser("fit", help="the lens from the kept views, into the workspace's lens.yaml")
    p_fit.add_argument("side", choices=("left", "right", "top"))

    p_resolve = sub.add_parser("resolve", help="solve recorded takes again from their top-camera video, "
                                                "with the top camera's fitted lens")
    p_resolve.add_argument("episodes", nargs="*",
                           help="episode names under the workspace's episodes/ (default: every take of the fitted unit "
                                "not yet solved with its lens)")

    p_convert = sub.add_parser("convert", help="raw episodes into a LeRobot dataset")
    p_convert.add_argument("episodes", nargs="+", help="episode names under the workspace's episodes/")
    p_convert.add_argument("--out", required=True, help="dataset name, under the workspace's datasets/")
    p_convert.add_argument("--task", required=True, help="the task, in words, e.g. 'pick and place the block'")
    p_convert.add_argument("--from", dest="source", default=None,
                           help="where the episodes are (default: the workspace's episodes/)")
    p_convert.add_argument("--gop", type=int, default=30,
                           help="keyframe spacing of the dataset video: 2 for training speed, 30 for size")
    p_convert.add_argument("--fps", type=int, default=None,
                           help="the dataset's frames a second, whatever the cameras' own rates - and the rate a "
                                "policy trained on it is asked at (default: the core settings' dataset_fps)")
    p_convert.add_argument("--randomised", action="store_true",
                           help="paint everything but each take's keep labels over with a pattern of the take's own "
                                "(default: the pictures as recorded)")

    p_bench = sub.add_parser("bench", help="closure error and coverage for a session")
    p_bench.add_argument("session", help="session name, or a path to one")
    p_bench.add_argument("--solid", default=SETTINGS.solid,
                         choices=[c.name for c in SETTINGS.solids],
                         help="which solid the take was made with (default: %(default)s)")
    _detector_args(p_bench)

    return parser


def _detector_args(sub) -> None:
    """Which library reads the squares, and which family it reads.

    Two readers of the same pictures, so a session recorded once can be solved
    both ways and compared. The dictionary has to be one the reader can read -
    the AprilTag library reads the AprilTag families only - and it has to be the
    one the paper on the solid was printed from, which is the real constraint.
    """
    sub.add_argument("--detector", default=SETTINGS.detector,
                     choices=("aruco", "apriltag"),
                     help="which library reads the markers (default: %(default)s)")
    sub.add_argument("--dict", dest="aruco_dict", default=SETTINGS.aruco_dict,
                     help=f"marker family (default: {SETTINGS.aruco_dict})")


def _choose_reader(args) -> None:
    """Point the package at a reader and a family, the way _choose_solid does."""
    from dataclasses import replace  # noqa: PLC0415

    from . import config  # noqa: PLC0415

    if args.detector == config.SETTINGS.detector and args.aruco_dict == config.SETTINGS.aruco_dict:
        return
    config.SETTINGS = replace(config.SETTINGS, detector=args.detector,
                              aruco_dict=args.aruco_dict)
    print(f"reader: {args.detector} on {args.aruco_dict}")


def _choose_solid(name: str) -> None:
    """Point the package at one solid, before anything that reads it is imported.

    The solid's size is what solvePnP is given, so solving for the wrong one is a
    pose that is confidently the wrong distance away. Every module that needs it
    binds `SETTINGS` at import, and the ones that do are imported inside the
    branches below and nowhere earlier - so replacing it here reaches all of
    them, and doing it any later would reach none.
    """
    from . import config  # noqa: PLC0415

    config.SETTINGS = SETTINGS.for_solid(name)
    if name != SETTINGS.solid:
        print(f"solid: {name}")


def _session_root(session: str) -> Path:
    """A name in the workspace, or a path - whichever the argument turns out to be."""
    candidate = Path(session)
    if candidate.is_dir():
        return candidate
    return SETTINGS.session_dir(session)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "marker":
        from . import markers  # noqa: PLC0415  - cv2 is not needed to print --help

        cfg = SETTINGS.for_solid(args.solid)
        written = markers.write_all(args.out, cfg)
        print(f"marker solid {cfg.solid} - {cfg.face_size_mm:g} mm faces, "
              f"{cfg.marker_mm:g} mm {cfg.aruco_dict} markers, "
              f"ids {', '.join(f'{n}={i}' for n, i in cfg.face_ids.items())}")
        for path in written:
            print(f"  {path}  ({path.stat().st_size:,} bytes)")
        print("\nPrint the sheet at 100% and measure the ruler bar before cutting.")
        return 0

    if args.command == "live":
        _choose_solid(args.solid)
        _choose_reader(args)
        from pinkrab_core.parent import on_term_interrupt, reap_orphans, watch_parent  # noqa: PLC0415

        from . import live, record  # noqa: PLC0415

        on_term_interrupt()
        # An earlier view nobody could stop is holding the camera: put it down first.
        reap_orphans("pinkrab-umi live")
        watch_parent(args.parent)

        try:
            live.run(args.port, args.cameras)
        except record.RecorderError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.command in ("detect", "setup-motor", "check", "calibrate", "read"):
        from . import boards  # noqa: PLC0415

        if args.command == "detect":
            return boards.umi_detect(tuple(args.devices) or HANDHELDS)
        if args.command == "setup-motor":
            return boards.umi_setup_motor(args.device, port=args.port, assume_yes=args.yes)
        if args.command == "check":
            return boards.umi_check(args.device, port=args.port)
        if args.command == "calibrate":
            return boards.umi_calibrate(args.device, port=args.port)
        return boards.umi_read(args.device, port=args.port, seconds=args.seconds,
                               aperture=args.aperture)

    if args.command == "cameras":
        from . import cameras  # noqa: PLC0415

        return cameras.show() if args.show else cameras.pair_top() if args.top else cameras.pair()

    if args.command == "lens":
        from . import lens
        if args.lens_command == "board":
            for path in lens.write_board(Path(args.out_dir) / "lens_charuco_a4.pdf"):
                print(f"  written: {path}")
            return 0
        if args.lens_command == "capture":
            return lens.capture(args.side, args.views)
        if args.side == "top":
            return lens.fit_top()
        return lens.fit(args.side)

    if args.command == "resolve":
        from . import resolve  # noqa: PLC0415

        return resolve.resolve(args.episodes or None)

    if args.command == "convert":
        from . import convert  # noqa: PLC0415

        try:
            report = convert.convert(args.episodes, args.out, args.task,
                                     source=Path(args.source).expanduser() if args.source else None,
                                     gop=args.gop, fps=args.fps,
                                     randomised=args.randomised)
        except convert.ConvertError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"\n  dataset: {report['dataset']}  ({report['seconds']} s)")
        return 0

    if args.command == "bench":
        _choose_solid(args.solid)
        _choose_reader(args)
        from . import bench, record  # noqa: PLC0415

        root = _session_root(args.session)
        try:
            return bench.evaluate(root)
        except (record.RecorderError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    return 2


if __name__ == "__main__":
    sys.exit(main())
