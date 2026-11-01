"""`pinkrab-studio` - the browser view of the robot.

One purpose: serve the page. The MuJoCo scene under it is stepped headless beside
the server, so the page is the only window there is.

Its own command rather than a subcommand of `pinkrab`, because it needs MuJoCo and
the CLI does not. The rig the arms are bolted to installs the CLI to bring the
servos up; a machine that also wants a picture installs this package on top.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pinkrab_core.config import SETTINGS

BAD = "\033[31mx\033[0m"
WARN = "\033[33m!\033[0m"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pinkrab-studio", description="Look at the Pinkrab robot."
    )
    parser.add_argument("--port", type=int, help=f"default {SETTINGS.studio_port}")
    parser.add_argument(
        "--bind", default="127.0.0.1",
        help="0.0.0.0 to reach it from the network - there is no auth "
             "(default: this machine only)",
    )
    parser.add_argument("--mount", help="a different rig's mount.yaml")
    parser.add_argument("--scene", help=f"a different scene (default: {SETTINGS.scene_file.name})")
    parser.add_argument("--export", metavar="DIR",
                        help="write the front page's robot as static files into DIR and exit "
                             "(the documentation site's copy)")
    return parser


def studio(
    port: int | None = None,
    bind: str = "127.0.0.1",
    mount: str | None = None,
    scene: str | None = None,
) -> int:
    """Serve the browser view of the robot.

    The page drives the MuJoCo scene rather than posing the description: the scene
    is stepped headless beside the server. MuJoCo is required - see the studio
    section of the README for why there is no reduced version of this page.

    Simulation or the real robot is chosen on the page. Real means the arms are
    drawn where ROS reports them: the bus is NOT opened here - one process can
    hold a serial port and that one is the driver, on the robot's own launch
    file - so the readings arrive by way of the bridge, which the page starts
    beside this server along with the handheld loop and the foot switch.
    """
    from pinkrab_core.native import quiet_ffmpeg_twins

    from .server import StudioState, serve

    # Before the server's threads: a policy loaded later reaches for both.
    quiet_ffmpeg_twins()

    xacro_file = SETTINGS.xacro_file
    if not xacro_file.is_file():
        print(f"\n{BAD} No description at {xacro_file}\n")
        return 1

    mount_path = Path(mount) if mount else SETTINGS.mount_file
    if not mount_path.is_file():
        print(f"\n{BAD} No mount file at {mount_path}\n")
        return 1

    # Say it before the browser does. The geometry being provisional is the whole
    # reason to look at this page, so it should not be a surprise found later.
    if "PLACEHOLDER" in mount_path.read_text():
        print(f"\n{WARN} {mount_path.name} still holds placeholder geometry -")
        print("      the arms will render in the wrong place until it is resolved.")

    scene_path = Path(scene) if scene else SETTINGS.scene_file
    if not scene_path.is_file():
        print(f"\n{BAD} No scene at {scene_path}")
        print("    The model is generated - see the Simulation section of the README.\n")
        return 1

    # Built before the server binds, so a missing MuJoCo or a scene that will not
    # compile is said here, against its own name, rather than surfacing as a blank
    # page to whoever opens it.
    from pinkrab_mujoco.physics import PhysicsError

    try:
        state = StudioState(xacro_file, SETTINGS.mesh_dir, scene_path, mount_path)
    except PhysicsError as exc:
        print(f"\n{BAD} {exc}\n")
        return 1

    serve(state, bind, port or SETTINGS.studio_port)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.export:
        from .front_page import export

        for path in export(Path(args.export), Path(args.mount) if args.mount else None,
                           Path(args.scene) if args.scene else None):
            print(path)
        return 0
    return studio(port=args.port, bind=args.bind, mount=args.mount, scene=args.scene)


if __name__ == "__main__":
    sys.exit(main())
