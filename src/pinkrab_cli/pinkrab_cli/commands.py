"""What the `pinkrab` CLI does: doctor, ports, check, setup-motors, calibrate, mount, cameras, lens.

Every command exists to supply resolved ports and derived calibration ids so a pair
can never be typed in the wrong order — not to reimplement anything LeRobot already
does.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from pinkrab_core import assign, bus, ports
from pinkrab_core import mount as mount_cfg
from pinkrab_core.config import ARMS, HANDHELDS, SETTINGS

OK = "\033[32m/\033[0m"
BAD = "\033[31mx\033[0m"
WARN = "\033[33m!\033[0m"


# --------------------------------------------------------------------------
# doctor
# --------------------------------------------------------------------------
def doctor() -> int:
    """Check the environment before touching hardware. Read-only."""
    problems = 0

    def report(ok: bool, label: str, detail: str = "", fix: str = "") -> None:
        nonlocal problems
        print(f"  {OK if ok else BAD} {label}" + (f" - {detail}" if detail else ""))
        if not ok:
            problems += 1
            if fix:
                print(f"      {fix}")

    print("\nEnvironment")
    report(
        sys.version_info >= (3, 10),
        "python",
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "LeRobot needs 3.10 or newer.",
    )

    lerobot_version = _module_version("lerobot")
    report(
        lerobot_version is not None,
        "lerobot",
        lerobot_version or "not importable",
        "pip install -e '.[feetech]' in your LeRobot checkout.",
    )
    report(
        _module_version("scservo_sdk") is not None,
        "scservo_sdk",
        _module_version("scservo_sdk") or "not importable",
        "The Feetech driver: pip install -e '.[feetech]'.",
    )

    for cli in ("lerobot-find-port", "lerobot-teleoperate"):
        path = shutil.which(cli)
        report(path is not None, cli, path or "not on PATH", "Activate the LeRobot environment.")

    print("\nWorkspace")
    report(
        SETTINGS.workspace.is_dir(),
        "workspace",
        str(SETTINGS.workspace),
        f"Created on first write; make it now with: mkdir -p {SETTINGS.workspace}",
    )
    print(f"    HF_LEROBOT_HOME is exported as {SETTINGS.workspace} for every lerobot-* call.")

    print("\nSerial")
    found = ports.candidates()
    report(
        bool(found),
        "servo boards",
        f"{len(found)} device(s)" if found else "none detected",
        "Check power and USB. On Linux you may also need: sudo usermod -aG dialout $USER",
    )
    for device in found:
        print(f"      {device}")

    assigned = None
    try:
        assigned = ports.load()
        print(f"  {OK} port assignment - {len(assigned.assigned())}/{len(ARMS)} arms")
        for arm, device in assigned.items():
            print(f"      {arm:<8} {device.name}")
    except ports.PortError as exc:
        report(False, "port assignment", str(exc).splitlines()[0], "Run: pinkrab ports detect")

    # One line per arm. The full per-joint table belongs to `check`; repeating it
    # here would make doctor something you scroll instead of something you read.
    print("\nServos")
    if assigned is None:
        print(f"  {WARN} not scanned - no port assignment")
    else:
        expected = len(SETTINGS.joint_names)
        for arm in assigned.assigned():
            try:
                scan = bus.scan_arm(arm, assigned[arm])
            except Exception as exc:  # noqa: BLE001 - a closed bus is a finding, not a crash
                report(False, f"{arm:<8}", f"cannot open the bus: {exc}")
                continue

            found = scan.found_count
            detail = f"{found}/{expected} responded"
            if scan.motors and all(m.found for m in scan.motors):
                temps = [m.temperature for m in scan.motors if m.temperature is not None]
                if temps:
                    detail += f", {min(temps)}-{max(temps)}C"
            if scan.collision_ids:
                detail += f", {len(scan.collision_ids)} id collision(s)"
            if scan.unexpected_ids:
                detail += f", {len(scan.unexpected_ids)} unexpected id(s)"
            report(scan.ok, f"{arm:<8}", detail, "Detail: pinkrab check")

    print("\nDescription")
    urdf = SETTINGS.urdf_file
    print(f"  {OK if urdf.is_file() else WARN} urdf - {urdf if urdf.is_file() else 'not written yet'}")
    if not urdf.is_file():
        print("      There is no leader arm, so every way to drive these arms goes")
        print("      through inverse kinematics, which needs this file. Measure the")
        print("      wall mount and write it.")

    # The same list the studio's front page shows as Before Starting.
    print("\nBefore starting")
    from pinkrab_core.checklist import GROUPS, checks

    group = None
    for item in checks():
        if item["group"] != group:
            group = item["group"]
            print(f"  {GROUPS[group]}")
        mark = OK if item["ok"] else (WARN if item["optional"] else BAD)
        print(f"    {mark} {item['label']}" + (f" - {item['detail']}" if item["detail"] else ""))
        if not item["ok"] and item["run"]:
            print(f"        {item['run']}")
        if not item["ok"] and not item["optional"]:
            problems += 1

    print()
    if problems:
        print(f"{BAD} {problems} problem(s) above.\n")
        return 1
    print(f"{OK} Environment looks good.\n")
    return 0


def _module_version(module: str) -> str | None:
    try:
        import importlib.metadata as md

        return md.version(module.replace("_", "-"))
    except Exception:  # noqa: BLE001
        try:
            mod = __import__(module)
        except Exception:  # noqa: BLE001
            return None
        return getattr(mod, "__version__", "installed")


# --------------------------------------------------------------------------
# ports
# --------------------------------------------------------------------------
def ports_show() -> int:
    found = {p.name for p in ports.candidates()}
    try:
        assigned = ports.load()
    except ports.PortError as exc:
        print(f"\n{BAD} {exc}\n")
        return 1

    print("\nPort assignment")
    for name in ARMS + HANDHELDS:
        device = assigned.mapping.get(name)
        if device is None:
            print(f"  {WARN} {name:<9} unassigned")
        elif device.name in found:
            print(f"  {OK} {name:<9} {device.name}")
        elif name in HANDHELDS:
            # Unplugged is where a handheld lives; it is only carried to record.
            print(f"  {WARN} {name:<9} {device.name} (not plugged in)")
        else:
            print(f"  {BAD} {name:<9} {device.name}")
    print(f"\n  stored in {SETTINGS.ports_file}\n")
    return 0


def ports_detect(arms: tuple[str, ...] = ARMS) -> int:
    print("\nPort detection")
    print("  Plug order is the only thing that tells the identical boards apart.")
    print("  You will unplug these boards, then plug them in one at a time:")
    print("    " + " -> ".join(arms))
    try:
        assigned = ports.detect_interactive(arms)
    except ports.PortError as exc:
        print(f"\n{BAD} {exc}\n")
        return 1
    except (EOFError, KeyboardInterrupt):
        print("\n  Cancelled. Nothing was written.\n")
        return 1

    print(f"\n{OK} Saved to {SETTINGS.ports_file}")
    for arm, device in assigned.items():
        print(f"    {arm:<8} {device.name}")
    print()
    return 0


def ports_swap() -> int:
    try:
        assigned = ports.swap()
    except ports.PortError as exc:
        print(f"\n{BAD} {exc}\n")
        return 1
    print(f"\n{OK} Swapped left and right")
    for arm, device in assigned.items():
        print(f"    {arm:<8} {device.name}")
    print()
    return 0


def ports_env() -> int:
    """Shell exports, for calling lerobot-* directly without retyping the ports."""
    try:
        assigned = ports.load()
    except ports.PortError as exc:
        print(f"{BAD} {exc}", file=sys.stderr)
        return 1

    print(f"export HF_LEROBOT_HOME={SETTINGS.workspace}")
    for arm, device in assigned.items():
        print(f"export PINKRAB_{arm.upper()}={device}")
    return 0


def ports_clear() -> int:
    removed = ports.clear()
    print(
        f"\n{OK} Cleared {SETTINGS.ports_file}\n"
        if removed
        else f"\n{WARN} Nothing to clear ({SETTINGS.ports_file} does not exist)\n"
    )
    return 0


# --------------------------------------------------------------------------
# check
# --------------------------------------------------------------------------
def check(arms: tuple[str, ...] | None = None) -> int:
    """Ping every servo on the selected arms and print what answered. Read-only."""
    try:
        assigned = ports.load()
    except ports.PortError as exc:
        print(f"\n{BAD} {exc}\n")
        return 1

    targets = [a for a in (arms or assigned.assigned()) if a in assigned.mapping]
    if not targets:
        print(f"\n{BAD} No assigned arms to check. Run: pinkrab ports detect\n")
        return 1

    print("\nServo check (read-only)")
    failed = False
    for arm in targets:
        port = assigned[arm]
        try:
            scan = bus.scan_arm(arm, port)
        except Exception as exc:  # noqa: BLE001 - report, do not traceback at the user
            print(f"\n  {BAD} {arm} - cannot open the bus: {exc}")
            failed = True
            continue

        print(f"\n  {arm}  {port.name}")
        print(f"    {'joint':<14} {'id':>2}  ok  {'position':>9}  {'temp':>5}")
        for m in scan.motors:
            mark = OK if m.ok else BAD
            pos = "-" if m.position is None else str(m.position)
            temp = "-" if m.temperature is None else f"{m.temperature}C"
            print(f"    {m.name:<14} {m.expected_id:>2}  {mark}   {pos:>9}  {temp:>5}")

        if scan.unexpected_ids:
            extra = ", ".join(f"id {i} (model {mdl})" for i, mdl in scan.unexpected_ids.items())
            print(f"    {WARN} Unexpected ids answered: {extra}")
            print(f"       Renumber: pinkrab setup-motors {arm}")

        if not scan.ok:
            failed = True
            missing = [m for m in scan.motors if not m.found]
            colliding = [m for m in missing if m.expected_id in scan.collision_ids]
            silent = [m for m in missing if m.expected_id not in scan.collision_ids]

            if colliding:
                names = ", ".join(f"{m.name}(id {m.expected_id})" for m in colliding)
                print(f"    {BAD} Several servos answer at once on: {names}")
                print("       Servos that were never numbered all sit at the factory id 1.")
                print(f"       Number them: pinkrab setup-motors {arm}")
            if silent:
                names = ", ".join(f"{m.name}(id {m.expected_id})" for m in silent)
                print(f"    {BAD} No answer from {len(silent)}: {names}")
                if colliding:
                    # Do not send anyone looking for a broken connector. Servos
                    # crowded onto the factory id are also missing from their own
                    # ids, and that is the whole explanation.
                    print("       Expected, given the collision above: servos sitting on the")
                    print("       factory id are absent from the ids they should have.")
                    print("       Number them first, then run this again.")
                else:
                    first = scan.first_silent()
                    if first is not None:
                        print(
                            f"       The chain runs {SETTINGS.joint_names[0]} -> "
                            f"{SETTINGS.joint_names[-1]}, so check the connector just before "
                            f"{first.name} first - a break there silences everything after it."
                        )

    print()
    if failed:
        print(f"{BAD} Check failed. Fix the items above and run it again.\n")
        return 1
    print(f"{OK} All servos responded on {len(targets)} arm(s).\n")
    return 0


# --------------------------------------------------------------------------
# calibrate
# --------------------------------------------------------------------------
def calibrate(arm: str) -> int:
    """Calibrate one arm.

    Run in this process rather than through `lerobot-calibrate`, because the arm it
    calibrates is this project's seven-servo one and that command only knows the
    robot types LeRobot ships. The work is LeRobot's either way - the class is a
    subclass of its follower.
    """
    try:
        assigned = ports.load(require=(arm,))
    except ports.PortError as exc:
        print(f"\n{BAD} {exc}\n")
        return 1

    path = SETTINGS.calibration_file(arm)

    print(f"\nCalibrating {arm}")
    print("  This arm is wall-mounted - clear its swing path before starting.")
    print("  Steps: zero (ENTER), then sweep every joint through its full range.")
    print(f"  Every joint is swept, {SETTINGS.joint_names[-2]} included: it has a printed")
    print("  hard stop rather than a free turn, and the servo's own limit is the only")
    print("  thing between a command and the wall.")
    if path.is_file():
        print(f"  {WARN} A calibration already exists. Press 'c' + ENTER to redo it;")
        print("      a bare ENTER reuses the existing file.")

    try:
        from pinkrab_core.robot import PinkrabFollower, PinkrabFollowerConfig
    except ImportError as exc:
        print(f"\n{BAD} LeRobot is not importable: {exc}")
        print("    Activate the environment LeRobot is installed in: pinkrab doctor\n")
        return 1

    robot = PinkrabFollower(
        PinkrabFollowerConfig(
            port=str(assigned[arm]),
            id=SETTINGS.arm_id(arm),
            calibration_dir=SETTINGS.calibration_dir,
        )
    )
    # The same steps as lerobot-calibrate: connect without calibrating (which
    # writes the servo registers), calibrate, disconnect.
    try:
        robot.connect(calibrate=False)
        robot.calibrate()
    except KeyboardInterrupt:
        print(f"\n{WARN} Interrupted - nothing was written.\n")
        return 130
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        print(f"\n{BAD} {type(exc).__name__}: {exc}\n")
        return 1
    finally:
        if robot.bus.is_connected:
            robot.disconnect()

    if path.is_file():
        print(f"\n{OK} Wrote {path}\n")
        return 0
    print(f"\n{WARN} Calibration finished but no file appeared at {path}\n")
    return 1


# --------------------------------------------------------------------------
# setup-motors  (writes servo EEPROM)
# --------------------------------------------------------------------------
def _setup_targets(arm: str, port_override: str | None) -> list[tuple[str, Path]]:
    """Which (arm, port) pairs to write, from the stored assignment or an override."""
    if port_override:
        if arm == "both":
            raise ports.PortError("--port applies to one arm: pass left or right, not both.")
        path = Path(port_override)
        if not path.exists():
            raise ports.PortError(f"No such device: {path}")
        return [(arm, path)]

    arms = ARMS if arm == "both" else (arm,)
    try:
        assigned = ports.load(require=arms)
        return [(a, assigned[a]) for a in arms]
    except ports.PortError:
        # Numbering only needs to know which serial port to write to - both arms use
        # ids 1..6 on their own bus. So when one arm is on the bench with a single
        # board plugged in, there is no need to have detected the pair first.
        found = ports.candidates()
        if arm != "both" and len(found) == 1:
            print(f"  {WARN} No assignment stored; using the only connected board "
                  f"as {arm}: {found[0].name}")
            return [(arm, found[0])]
        raise


def setup_motors(
    arm: str = "both",
    joints: list[str] | None = None,
    port: str | None = None,
    assume_yes: bool = False,
) -> int:
    """Write servo ids into EEPROM. Destructive - only for new or renumbered servos."""
    try:
        targets = _setup_targets(arm, port)
        order = assign.assign_order(joints)
    except (ports.PortError, ValueError) as exc:
        print(f"\n{BAD} {exc}\n")
        return 1

    print("\nMotor numbering - this writes new ids into servo EEPROM.")
    print("  Do not run it on servos that already work. It is for a servo that was")
    print("  replaced, or an arm whose numbering is wrong or unfinished.")
    for target_arm, path in targets:
        print(f"  target: {target_arm:<6} {path.name}")
    print(f"  order:  {' -> '.join(order)}")
    print("  Connect only the servo named on screen, then press ENTER.")
    print("  With more than one connected, they all take the id, or the wrong one does.")

    if not assume_yes:
        try:
            answer = input("\nProceed? Type 'y' to continue: ")
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.\n")
            return 130
        if answer.strip().lower() != "y":
            print("Cancelled.\n")
            return 130

    for target_arm, path in targets:
        print(f"\n=== {target_arm} ===")
        try:
            assign.assign_ids(target_arm, path, order)
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled. Ids already written stay written - check with: pinkrab check\n")
            return 130
        except Exception as exc:  # noqa: BLE001 - a bad servo must not traceback at the user
            print(f"\n{BAD} {target_arm}: {exc}")
            print("       Check that exactly one servo is connected and powered.\n")
            return 1

    print(f"\n{OK} Numbering done. Verify with: pinkrab check\n")
    return 0


# --------------------------------------------------------------------------
# mount
# --------------------------------------------------------------------------


def mount() -> int:
    """The two tape-measure numbers about this rig, asked for one at a time."""
    print("\nWall mount")
    print("  Two measurements say where the arms are; everything else in mount.yaml")
    print("  is derived. Millimetres, Enter keeps the current value.")
    print("    1. between the two arm bases, centre to centre")
    print("    2. from the desk surface up to the top of a base - the flat face")
    print("       its plate, its beam and the cap over the first servo share\n")
    try:
        measured = mount_cfg.measure_interactive()
    except mount_cfg.MountError as exc:
        print(f"\n{BAD} {exc}\n")
        return 1
    except (EOFError, KeyboardInterrupt):
        print("\n  Cancelled. Nothing was written.\n")
        return 1
    try:
        path = mount_cfg.write(measured["spacing"], measured["top_height"])
    except mount_cfg.MountError as exc:
        print(f"\n{BAD} {exc}\n")
        return 1
    print(f"\n{OK} Written to {path}")
    print(f"    arms      {measured['spacing'] * 1000:.0f} mm apart, at y = "
          f"{measured['spacing'] * 500:+.1f} / {-measured['spacing'] * 500:+.1f} mm")
    print(f"    base top  {measured['top_height'] * 1000:.0f} mm above the desk")
    print("    The workspace URDF is expanded again by whatever next needs it.\n")
    return 0


# --------------------------------------------------------------------------
# cameras
# --------------------------------------------------------------------------


def cameras(show: bool) -> int:
    """Name the follower arms' gripper cameras by socket, as `pinkrab-umi
    cameras` names the handhelds'. The same module and the same pairing; the
    file is the arms' own (see SETTINGS.follower_cameras_file)."""
    try:
        from pinkrab_umi import cameras as umi_cameras
    except ImportError as exc:
        print(f"{BAD} the camera tools come with pinkrab_umi, which is not importable here: {exc}")
        return 1
    return umi_cameras.show("follower") if show else umi_cameras.pair("follower")


def lens(step: str, arm: str, views: int | None) -> int:
    """Measure an arm's gripper camera as `pinkrab-umi lens` measures a
    handheld's: the same board and fit, kept in lens.yaml as follower_<arm>."""
    try:
        from pinkrab_umi import lens as umi_lens
    except ImportError as exc:
        print(f"{BAD} the lens tools come with pinkrab_umi, which is not importable here: {exc}")
        return 1
    camera = f"follower_{arm}"
    return umi_lens.capture(camera, views) if step == "capture" else umi_lens.fit(camera)
