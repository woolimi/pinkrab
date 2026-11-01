"""The handheld boards: one servo each, on a board of its own.

Read for the angle the hand is pinching at. Numbering a board is the same job
the arms have, so it is the same steps - assign the board by plug order, write
the id, ping it - and the same code underneath, in pinkrab_core. Here rather
than in `pinkrab`, because the handhelds plug into the desk with the cameras,
not into the robot's host, and `pinkrab-umi` is the desk's command:

    pinkrab-umi detect                     pin both boards (the arms keep theirs)
    pinkrab-umi setup-motor umi_left       write the gripper id into its servo
    pinkrab-umi check umi_left             it answers
    pinkrab-umi calibrate umi_left         shut and wide, once per handheld
    pinkrab-umi read umi_left --aperture --seconds 5
"""
from __future__ import annotations

from pathlib import Path

from pinkrab_core import angles, assign, bus, ports
from pinkrab_core.config import SETTINGS

OK = "\033[32m/\033[0m"
BAD = "\033[31mx\033[0m"
WARN = "\033[33m!\033[0m"

def umi_detect(devices: tuple[str, ...]) -> int:
    """Pin the handhelds' boards in plug order, leaving the arms' assignments alone."""
    try:
        assigned = ports.detect_interactive(devices)
    except (ports.PortError, EOFError, KeyboardInterrupt) as exc:
        print(f"\n{BAD} {exc}\n" if isinstance(exc, ports.PortError) else "\nCancelled.\n")
        return 1 if isinstance(exc, ports.PortError) else 130
    print()
    for device in devices:
        print(f"{OK} {device:<9} -> {assigned[device].name}")
    print(f"  stored in {SETTINGS.ports_file}\n")
    return 0


def _umi_port(device: str, port_override: str | None) -> Path:
    if port_override:
        path = Path(port_override)
        if not path.exists():
            raise ports.PortError(f"No such device: {path}")
        return path
    try:
        return ports.load(require=(device,))[device]
    except ports.PortError as exc:
        # Numbering one servo only needs to know which board to write to, and a
        # handheld on the bench is usually the only thing plugged in.
        found = ports.candidates()
        if len(found) == 1:
            print(f"  {WARN} No assignment stored for {device}; using the only connected "
                  f"board ({found[0].name}).")
            return found[0]
        first = str(exc).splitlines()[0]
        raise ports.PortError(
            f"{first}\n"
            f"  Run: pinkrab-umi detect {device}   (or pass --port)"
        ) from exc


def umi_setup_motor(device: str, port: str | None = None, assume_yes: bool = False) -> int:
    """Write the gripper id into the handheld's servo. Destructive."""
    try:
        path = _umi_port(device, port)
    except ports.PortError as exc:
        print(f"\n{BAD} {exc}\n")
        return 1

    name = SETTINGS.handheld_joint
    motor_id = SETTINGS.motor_ids[name]
    print("\nHandheld numbering - this writes a new id into servo EEPROM.")
    print(f"  target: {device:<9} {path.name}   -> '{name}' id {motor_id}")
    print("  Connect ONLY the handheld's servo. An arm on the same board takes the id too.")

    if not assume_yes:
        try:
            answer = input("\nProceed? Type 'y' to continue: ")
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.\n")
            return 130
        if answer.strip().lower() != "y":
            print("Cancelled.\n")
            return 130

    try:
        assign.assign_handheld_id(device, path)
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.\n")
        return 130
    except Exception as exc:  # noqa: BLE001 - a bad servo must not traceback at the user
        print(f"\n{BAD} {device}: {exc}")
        print("       Check that exactly one servo is connected and powered.\n")
        return 1

    print(f"\n{OK} Numbered. Verify with: pinkrab-umi check {device}\n")
    return 0


def umi_check(device: str, port: str | None = None) -> int:
    """Ping the handheld's servo and print what answered. Read-only."""
    try:
        path = _umi_port(device, port)
    except ports.PortError as exc:
        print(f"\n{BAD} {exc}\n")
        return 1

    try:
        scan = bus.scan_handheld(device, path)
    except Exception as exc:  # noqa: BLE001 - report, do not traceback at the user
        print(f"\n{BAD} {device} - cannot open the bus: {exc}\n")
        return 1

    print(f"\nHandheld check (read-only)\n\n  {device}  {path.name}")
    print(f"    {'joint':<14} {'id':>2}  ok  {'position':>9}  {'temp':>5}")
    for m in scan.motors:
        mark = OK if m.ok else BAD
        pos = "-" if m.position is None else str(m.position)
        temp = "-" if m.temperature is None else f"{m.temperature}C"
        print(f"    {m.name:<14} {m.expected_id:>2}  {mark}   {pos:>9}  {temp:>5}")

    if scan.unexpected_ids:
        extra = ", ".join(f"id {i} (model {mdl})" for i, mdl in scan.unexpected_ids.items())
        print(f"    {WARN} Unexpected ids answered: {extra}")
        print(f"       Renumber: pinkrab-umi setup-motor {device}")
    if not scan.ok:
        print(f"\n{BAD} The handheld's servo did not answer as expected.\n")
        return 1
    print(f"\n{OK} {device} answers.\n")
    return 0


def umi_read(device: str, port: str | None = None, seconds: float = 0.0,
             aperture: bool = False) -> int:
    """Print where the handheld's servo is, once or until interrupted."""
    import time

    try:
        path = _umi_port(device, port)
    except ports.PortError as exc:
        print(f"\n{BAD} {exc}\n")
        return 1

    calibration = None
    if aperture:
        try:
            calibration = angles.load_handheld()
            angles.handheld_aperture(device, 0, calibration)
        except angles.HandheldCalibrationMissing as exc:
            print(f"\n{BAD} {exc}\n")
            return 1

    def say(ticks):
        if ticks is None:
            return "-"
        if calibration is None:
            return str(ticks)
        return f"{ticks:>5}  {angles.handheld_aperture(device, ticks, calibration):5.2f}"

    try:
        if seconds <= 0:
            print(say(bus.read_handheld(path)))
            return 0
        head = "ticks  open" if calibration else "ticks"
        print(f"\n{device}  {path.name}   {head}, ctrl-c to stop\n")
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            print(f"  {say(bus.read_handheld(path))}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print()
    except Exception as exc:  # noqa: BLE001
        print(f"\n{BAD} {device}: {exc}\n")
        return 1
    return 0


MIN_SPAN_TICKS = 200


def umi_calibrate(device: str, port: str | None = None) -> int:
    """Pair the handheld's servo ticks with the two ends of its jaw's travel.

    Two readings and nothing else. The jaw is a printed pair of blades on a horn
    with a hard stop at each end, so what the hand can reach IS the calibration -
    there is no zero to find and no range to sweep, and a wizard that swept one
    would only be asking the same two questions slowly.
    """
    try:
        path = _umi_port(device, port)
    except ports.PortError as exc:
        print(f"\n{BAD} {exc}\n")
        return 1

    print(f"\nCalibrating {device}'s jaw   {path.name}")
    print("  Hold each end firmly against its stop - not near it. The two numbers")
    print("  are what every later reading is measured between.")

    readings: dict[str, int] = {}
    try:
        for key, prompt_text in (("closed_ticks", "Squeeze the jaw fully SHUT"),
                                 ("open_ticks", "Let the jaw open fully WIDE")):
            input(f"\n  {prompt_text}, hold it, then press ENTER...")
            value = bus.read_handheld(path)
            if value is None:
                print(f"\n{BAD} The servo did not answer. Check power and the cable.\n")
                return 1
            # Within one turn: the servo counts on from wherever it woke up, and
            # a number from one session must mean the same jaw in the next.
            readings[key] = int(value) % angles.HANDHELD_TICKS_PER_TURN
            print(f"    {key.split('_')[0]:<7} {readings[key]}")
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.\n")
        return 130
    except Exception as exc:  # noqa: BLE001 - a bad servo must not traceback at the user
        print(f"\n{BAD} {device}: {exc}\n")
        return 1

    span = abs(int(angles.handheld_span(readings["closed_ticks"], readings["open_ticks"])))
    if span < MIN_SPAN_TICKS:
        print(f"\n{BAD} Only {span} ticks between the two ends "
              f"(expected at least {MIN_SPAN_TICKS}).")
        print("    Either the jaw did not move between the readings, or the servo")
        print("    is not the one the jaw is on. Nothing was written.\n")
        return 1

    import yaml

    fpath = SETTINGS.handheld_file
    data = {}
    if fpath.is_file():
        loaded = yaml.safe_load(fpath.read_text())
        if isinstance(loaded, dict):
            data = loaded
    data[device] = readings
    fpath.parent.mkdir(parents=True, exist_ok=True)
    fpath.write_text(yaml.safe_dump(data, sort_keys=True))

    print(f"\n{OK} {device}: {span} ticks of travel -> {fpath}")
    print(f"    Check it: pinkrab-umi read {device} --aperture --seconds 5\n")
    return 0
