"""`pinkrab-umi cameras` and `pinkrab cameras` - which gripper camera is which side.

The modules are the same part and report the same serial number, so nothing
about them says which side they are on: only the socket they are plugged into
does. So the sockets are named the way the handheld boards were: with both
unplugged, plug the left one in and it is the left, then the right. The
answer goes beside the rig's other calibrations, and on Linux a udev rule is
printed that gives each socket a fixed name, so the host sees
/dev/gripper_left and /dev/gripper_right whatever order the kernel found them.

Two sets are named this way, each in a file of its own: the handhelds'
cameras, by `pinkrab-umi cameras`, and the follower arms', by `pinkrab
cameras` (/dev/follower_left, /dev/follower_right). The two may share the
rig's two sockets - the handhelds' plugged in to record, the arms' to run a
policy - and nothing here holds one file against the other.

The top camera is the same module once more, so its socket is named the same
way and kept in the handhelds' file (/dev/top_camera): `--top`, one plug.
Every other naming leaves it out - it stays plugged in while the hands' are
named.

    pinkrab-umi cameras            # unplug both, plug left, plug right - Enter after each
    pinkrab-umi cameras --top      # unplug the top camera, plug it back - Enter after each
    pinkrab-umi cameras --show     # what is recorded, and what is on the bus
    pinkrab cameras [--show]       # the same, for the follower arms' cameras
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from .gripper_cams import (CAMERA_SETS, TOP, TOP_COMMAND, camera_file, candidates, mapping, same_socket,
                           top_socket)

SETTLE_S = 1.0
# Whose cameras a set is, in the prompts; the device name udev gives them on Linux.
WHOSE = {"handheld": "handheld", "follower": "follower arm"}
DEVICE = {"handheld": "gripper", "follower": "follower"}
TOP_DEVICE = "top_camera"


def _ids(cams: list[dict]) -> dict[str, dict]:
    return {str(cam["id"]): cam for cam in cams}


def _ask(prompt: str) -> None:
    """Wait for Enter. The bus is read after, not watched during: a plug going
    in re-enumerates for a moment, and a poll that caught it half-way would
    record the wrong thing."""
    try:
        input(f"  {prompt} - then press Enter ")
    except EOFError:
        print()
        raise SystemExit(1)
    time.sleep(SETTLE_S)


def pair(which: str = "handheld") -> int:
    path = camera_file(which)
    whose = WHOSE[which]
    print(f"\n  Naming the {whose} cameras by their USB sockets, one plug at a time.\n")
    while True:
        _ask(f"UNPLUG both {whose} cameras")
        present = _ids(candidates(fresh=True))
        if not present:
            break
        print(f"  {len(present)} still on the bus: " + ", ".join(c["port"] for c in present.values())
              + ("" if top_socket() else f" - if one is the top camera, the same part, name it first: `{TOP_COMMAND}`"))
    print("  the bus is clear.\n")

    chosen: dict[str, dict] = {}
    for side in ("left", "right"):
        while True:
            _ask(f"plug in the {side.upper()} {whose}'s camera")
            seen = {str(c["id"]) for c in chosen.values()}
            new = [cam for k, cam in _ids(candidates(fresh=True)).items() if k not in seen]
            if len(new) == 1:
                break
            if not new:
                print("  nothing new on the bus yet - give it a second and try again")
            else:
                print(f"  {len(new)} new cameras at once - plug in one at a time")
        cam = new[0]
        chosen[side] = cam
        print(f"  {side}: {cam['port']}\n")

    _write(which, {side: _entry(str(cam["id"]), f"/dev/{DEVICE[which]}_{side}") for side, cam in chosen.items()},
           f"written by `{CAMERA_SETS[which]}`. Move a plug and run it again.", keep=(TOP,))
    print(f"\n  written: {path}")
    for side, cam in chosen.items():
        print(f"    {side:5s} {cam['port']}")
    _say_udev(which)
    return 0


def pair_top() -> int:
    """Name the top camera's socket: the one module that leaves the bus when it
    is unplugged and comes back when it is plugged in again."""
    print("\n  Naming the top camera by its USB socket - the gripper cameras may stay plugged in.\n")
    while True:
        _ask("UNPLUG the top camera")
        before = _ids(candidates(fresh=True, top=True))
        _ask("plug the top camera in again")
        new = [cam for k, cam in _ids(candidates(fresh=True, top=True)).items() if k not in before]
        if len(new) == 1:
            break
        print("  nothing new on the bus - give it a second and try again" if not new
              else f"  {len(new)} new cameras at once - plug in only the top one")
    cam = new[0]
    _write("handheld", {TOP: _entry(str(cam["id"]), f"/dev/{TOP_DEVICE}")},
           f"written by `{TOP_COMMAND}`. Move a plug and run it again.", keep=("left", "right"))
    print(f"  top: {cam['port']}\n\n  written: {camera_file('handheld')}")
    _say_udev("handheld")
    return 0


def _entry(socket: str, device: str) -> dict[str, str]:
    """One camera's lines in a camera file: its socket, and on Linux the fixed
    name the udev rule gives it."""
    return {"mac_id": socket} if sys.platform == "darwin" else {"device": device, "by_path": socket}


def _write(which: str, named: dict[str, dict[str, str]], said: str, keep: tuple[str, ...]) -> None:
    """A set's camera file naming `named`, and of what it named before, the
    cameras in `keep`: the hands' naming never touches the top camera's
    socket, nor the top camera's the hands'."""
    path = camera_file(which)
    try:
        import yaml
        kept = yaml.safe_load(path.read_text()) if path.is_file() else {}
    except Exception:  # noqa: BLE001 - unreadable: written afresh
        kept = {}
    kept = {k: v for k, v in (kept if isinstance(kept, dict) else {}).items()
            if k in keep and isinstance(v, dict)}
    kept.update(named)
    whose = f"{WHOSE[which]} camera" + (", and the top camera," if TOP in kept else "")
    lines = [f"# Which USB socket each {whose} is in - {said}"]
    for name in ("left", "right", TOP):
        if name in kept:
            lines.append(f"{name}:")
            lines += [f"  {key}: \"{value}\"" for key, value in kept[name].items()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _say_udev(which: str) -> None:
    """The udev rule for a set's file as it now is, on Linux; on macOS, that
    there is none."""
    if sys.platform == "darwin":
        print("\n  on macOS the USB location is the name; nothing to install. Restart the live view.")
        return
    path = camera_file(which)
    try:
        import yaml
        data = yaml.safe_load(path.read_text()) or {}
    except Exception:  # noqa: BLE001
        data = {}
    named = {name: entry["by_path"] for name, entry in data.items()
             if isinstance(entry, dict) and entry.get("by_path")}
    rules = "99-pinkrab-cameras.rules" if which == "handheld" else f"99-pinkrab-{which}-cameras.rules"
    print("\n  the udev rule that names the sockets (needs root once):")
    print(f"    sudo tee /etc/udev/rules.d/{rules} <<'EOF'\n{udev_rule(named, which)}EOF")
    print("    sudo udevadm control --reload && sudo udevadm trigger")


def udev_rule(chosen: dict[str, str], which: str = "handheld") -> str:
    """One line per camera: the USB port path this by-path link stands for,
    matched by KERNELS, given a fixed name by SYMLINK."""
    out = []
    for side, by_path in chosen.items():
        name = TOP_DEVICE if side == TOP else f"{DEVICE[which]}_{side}"
        # /dev/v4l/by-path/pci-0000:00:14.0-usb-0:1.2:1.0-video-index0 -> the
        # kernel path is what udev sees; ask it rather than parse the name.
        kernels = _kernels_of(by_path)
        match = f'KERNELS=="{kernels}", ' if kernels else f'ENV{{ID_PATH}}=="{Path(by_path).name.rsplit("-video", 1)[0]}", '
        out.append(f'SUBSYSTEM=="video4linux", ATTR{{index}}=="0", {match}SYMLINK+="{name}"\n')
    return "".join(out)


def _kernels_of(by_path: str) -> str:
    try:
        target = Path(by_path).resolve()
        info = subprocess.run(["udevadm", "info", "-a", "-n", str(target)],
                              capture_output=True, text=True, timeout=5.0).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""
    # The first KERNELS that looks like a USB port (e.g. "1-1.2"), walking up.
    for line in info.splitlines():
        line = line.strip()
        if line.startswith("KERNELS==") and ":" not in line and "-" in line:
            return line.split('"')[1]
    return ""


def show(which: str = "handheld") -> int:
    path = camera_file(which)
    print(f"  recorded ({path}{'' if path.is_file() else ' - not written yet'}):")
    for side in ("left", "right"):
        print(f"    {side:5s} {mapping(which).get(side, f'not named - run `{CAMERA_SETS[which]}`')}")
    print(f"    {TOP:5s} {top_socket() or f'not named - run `{TOP_COMMAND}`'}")
    print("  on the bus now:")
    cams = candidates(fresh=True, top=True)
    if not cams:
        print("    no camera of the part")
    above = top_socket()
    for cam in cams:
        print(f"    {cam['port']}" + ("  (top)" if above and same_socket(above, str(cam["id"])) else ""))
    return 0


def present(which: str = "handheld") -> list[dict]:
    """Every gripper camera on the bus now, with the side the set's file
    assigns it to, if any."""
    known = mapping(which)
    out = []
    for cam in candidates(fresh=True):
        side = next((h for h, dev in known.items() if same_socket(str(dev), str(cam["id"]))), None)
        out.append({**cam, "side": side})
    return out


def assign(side: str, socket: str | None, which: str = "handheld") -> dict[str, str]:
    """Put one side's camera at `socket` (a candidate's `id`; None for none)
    and write the set's file.

    The other side keeps its socket - unless it held this very one, in which
    case it is left unassigned rather than have one camera be both sides.
    The other set's file is not looked at: it may name the same sockets.
    What the pairing command writes by plug order, this writes from a page's
    select; the file is the same.
    """
    if side not in ("left", "right"):
        raise ValueError(f"no such side: {side!r}")
    current = dict(mapping(which))
    if socket:
        socket = str(socket)
        if socket not in {str(c["id"]) for c in candidates(fresh=True)}:
            raise ValueError(f"no camera at {socket} now")
        current[side] = socket
        for other in ("left", "right"):
            if other != side and same_socket(current.get(other, ""), socket):
                current.pop(other, None)
    else:
        current.pop(side, None)
    _write(which, {hand: _entry(dev, dev) for hand, dev in current.items() if dev},
           f"written from the studio's camera select, or by `{CAMERA_SETS[which]}`. Move a plug and set it again.",
           keep=(TOP,))
    return {h: d for h, d in current.items() if d}
