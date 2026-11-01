"""Resolve which serial device is which arm, permanently.

Pinkrab has two identical servo boards, one per wall-mounted arm. Nothing on a
board says which arm it belongs to, and the kernel hands out device names in plug
order, so the mapping has to be established once and stored.

Getting it wrong is not a crash. A swapped pair simply drives the wrong arm, which
on a wall-mounted robot means a claw swinging toward the wall instead of away from
it, so this is pinned down rather than passed on the command line.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from .config import ARMS, DEVICES, HANDHELDS, SETTINGS


class PortError(RuntimeError):
    """Raised when the arm ports cannot be resolved."""


@dataclass(frozen=True)
class ArmPorts:
    """Resolved arm -> device path for every arm that has been assigned."""

    mapping: dict[str, Path]

    def __getitem__(self, arm: str) -> Path:
        try:
            return self.mapping[arm]
        except KeyError:
            raise PortError(
                f"{arm} has no port assigned yet.\n"
                "  Run: pinkrab ports detect"
            ) from None

    def assigned(self) -> tuple[str, ...]:
        """Arms that have a port, in canonical order."""
        return tuple(arm for arm in ARMS if arm in self.mapping)

    def items(self):
        return ((arm, self.mapping[arm]) for arm in self.assigned())


def candidates() -> list[Path]:
    """Every serial device that could be a servo board, sorted and stable.

    On macOS the device name embeds the board's USB serial number; on Linux the
    same stability comes from the /dev/serial/by-id symlink. Either way the name
    survives a replug, which /dev/ttyACMx does not.
    """
    directory = SETTINGS.serial_dir
    if not directory.is_dir():
        return []
    return sorted(directory.glob(SETTINGS.port_glob))


def load(require: tuple[str, ...] = ()) -> ArmPorts:
    """Read the stored assignment.

    Raises if the file is missing or corrupt, if an arm named in ``require`` has no
    assignment, or if an assigned device is not currently connected.
    """
    fpath = SETTINGS.ports_file
    if not fpath.is_file():
        raise PortError(
            f"No port assignment file ({fpath}).\n"
            "  Run: pinkrab ports detect"
        )

    try:
        raw = json.loads(fpath.read_text())
        mapping = {dev: SETTINGS.serial_dir / name for dev, name in raw.items() if dev in DEVICES}
    except (json.JSONDecodeError, AttributeError, TypeError) as exc:
        raise PortError(
            f"Port assignment file is corrupt ({fpath}): {exc}\n"
            "  Run: pinkrab ports detect"
        ) from exc

    missing_arms = [dev for dev in require if dev not in mapping]
    if missing_arms:
        raise PortError(
            "No port assigned for: " + ", ".join(missing_arms) + "\n"
            "  Run: pinkrab ports detect"
        )

    # A handheld is plugged in while somebody is holding it and not otherwise, so
    # its absence is the normal state and only an error when it was asked for.
    watched = require or tuple(d for d in mapping if d not in HANDHELDS)
    absent = [f"{d}={mapping[d].name}" for d in watched
              if d in mapping and not mapping[d].exists()]
    if absent:
        raise PortError(
            "Assigned devices are not connected right now: " + ", ".join(absent) + "\n"
            "  Check power and USB cabling. If a board was replaced: pinkrab ports detect"
        )

    return ArmPorts(mapping=mapping)


def save(mapping: dict[str, Path]) -> ArmPorts:
    """Persist the assignment by device name, never by a full path we might move."""
    names = [p.name for p in mapping.values()]
    if len(set(names)) != len(names):
        raise PortError(
            "Two arms resolved to the same device. Check that every board is plugged in."
        )

    SETTINGS.ports_file.parent.mkdir(parents=True, exist_ok=True)
    ordered = {dev: mapping[dev].name for dev in DEVICES if dev in mapping}
    SETTINGS.ports_file.write_text(json.dumps(ordered, indent=2) + "\n")
    return ArmPorts(mapping=dict(mapping))


def clear() -> bool:
    """Forget the stored assignment. True if a file was actually removed."""
    fpath = SETTINGS.ports_file
    if not fpath.is_file():
        return False
    fpath.unlink()
    return True


def swap() -> ArmPorts:
    """Flip left and right — the fix when the arms turn out to be mirrored."""
    current = load(require=ARMS)
    mapping = dict(current.mapping)
    mapping["left"], mapping["right"] = mapping["right"], mapping["left"]
    return save(mapping)


def stored() -> dict[str, Path]:
    """Whatever is on file, with no check that any of it is plugged in."""
    try:
        raw = json.loads(SETTINGS.ports_file.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return {dev: SETTINGS.serial_dir / name for dev, name in raw.items() if dev in DEVICES}


def detect_interactive(devices: tuple[str, ...] = ARMS, prompt=input,
                       settle_s: float = 3.0) -> ArmPorts:
    """Guided detection: unplug the boards being detected, then plug them in one at a time.

    Plug order is the only thing that distinguishes the boards. Each device is
    the one that appears at its own plug step, diffed against whatever was
    connected before it - so boards not being detected can stay plugged in, and
    a board that already has an assignment can be pinned again.

    Devices not being detected keep the assignment they had. Detecting the
    handhelds is a thing done on its own, long after the arms were pinned down,
    and it must not cost the arms their boards. A board that moves to a new name
    loses its old one, so no two names ever share a device.
    """
    labels = [d.replace("_", " ") for d in devices]
    which = " and ".join(labels) if len(labels) <= 2 else ", ".join(labels[:-1]) + f", and {labels[-1]}"
    plural = "s" if len(devices) > 1 else ""
    prompt(f"\n[1] Unplug the USB cable{plural} of the {which} board{plural} "
           f"(others may stay), then press ENTER...")
    seen = set(candidates())

    mapping: dict[str, Path] = {d: p for d, p in stored().items() if d not in devices}
    for step, (device, label) in enumerate(zip(devices, labels), start=2):
        prompt(f"[{step}] Plug in the {label} board, then press ENTER...")
        new = _wait_for_new(seen, settle_s)
        if len(new) != 1:
            raise PortError(
                f"Expected exactly one new device for {device}, saw {len(new)}.\n"
                "  Check the board LED and that the USB cable is seated at both ends."
            )
        path = new.pop()
        for other in [d for d, p in mapping.items() if p.name == path.name]:
            del mapping[other]
        mapping[device] = path
        seen.add(path)

    return save(mapping)


def _wait_for_new(before: set[Path], settle_s: float) -> set[Path]:
    """Devices that appear after `before`, polling while the OS enumerates the board."""
    deadline = time.monotonic() + settle_s
    while True:
        new = set(candidates()) - before
        if new or time.monotonic() >= deadline:
            return new
        time.sleep(0.2)
