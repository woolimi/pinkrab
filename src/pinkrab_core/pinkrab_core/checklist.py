"""What a rig has to have before anything on it is driven, and whether it has it.

Each item is something done once per machine or per rig - a rule installed,
a board named, a servo calibrated, a solid measured - and forgotten about
until a fresh machine or a rebuilt arm is missing it, when the symptom is a
driver that exits at once or an arm that moves the wrong way. Listed here so
the front page and `pinkrab doctor` can put the same list in front of
whoever is starting, with the command that settles each one.

Read-only, and nothing here opens a bus or a camera: it is asked while the
driver may be holding the arms.
"""
from __future__ import annotations

import dataclasses
import grp
import os
import sys
from pathlib import Path

from . import top_lens
from .config import ARMS, HANDHELDS, SETTINGS

UDEV_RULE = Path("/etc/udev/rules.d/60-pinkrab.rules")
UDEV_INSTALL = ("sudo cp scripts/udev/60-pinkrab.rules /etc/udev/rules.d/ && "
                "sudo udevadm control --reload && sudo udevadm trigger")


def _item(key: str, label: str, ok: bool, detail: str = "", run: str = "",
          optional: bool = False) -> dict:
    return {"key": key, "label": label, "ok": bool(ok), "detail": detail, "run": run,
            "optional": optional}


# What each item belongs to, in the order `pinkrab doctor` prints the groups.
# The studio shows a group with something left ahead of the finished ones.
GROUPS = {"machine": "This machine", "arms": "Arms", "handheld": "Handheld (UMI)"}


def _tag(items: list[dict], group: str) -> None:
    for item in items:
        item.setdefault("group", group)


def _importable(module: str) -> str | None:
    try:
        import importlib.metadata as md

        return md.version(module.replace("_", "-"))
    except Exception:  # noqa: BLE001 - not installed, or not in this environment
        try:
            __import__(module)
            return "present"
        except Exception:  # noqa: BLE001
            return None


def _named_sides(path: Path, names: tuple[str, ...] = ("left", "right")) -> list[str]:
    """The cameras a camera file names a socket for, of `names`; none for a
    missing or unreadable file."""
    try:
        import yaml

        data = yaml.safe_load(path.read_text()) if path.is_file() else {}
    except Exception:  # noqa: BLE001
        return []
    return [name for name in names if isinstance(data, dict) and name in data]


def _measured_lenses(path: Path) -> set[str]:
    """The cameras lens.yaml has a fit for - the gripper cameras and the top
    camera; none for a missing or unreadable file."""
    try:
        import yaml

        data = yaml.safe_load(path.read_text()) if path.is_file() else {}
    except Exception:  # noqa: BLE001
        return set()
    if not isinstance(data, dict):
        return set()
    return {name for name, fit in data.items() if isinstance(fit, dict) and {"cx", "cy", "size"} <= set(fit)}


def _lens_item(key: str, label: str, camera: str, measured: set[str]) -> dict:
    from .dataset import lens_command

    done = camera in measured
    return _item(key, label, done, "measured" if done else "not measured",
                 f"{lens_command(camera, 'capture')} && {lens_command(camera, 'fit')}")


def _in_group(name: str) -> bool | None:
    try:
        gid = grp.getgrnam(name).gr_gid
    except KeyError:
        return None
    return gid in os.getgroups()


def checks(settings=SETTINGS) -> list[dict]:
    """Every item, in the order it is done, each with `ok`, what to run and
    the group it belongs to (a key of `GROUPS`)."""
    from . import angles, ports

    out: list[dict] = []

    # --- the machine ----------------------------------------------------
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    out.append(_item("python", "python 3.10 or newer", sys.version_info >= (3, 10), version))
    for module, why in (("lerobot", "the servo driver and the datasets"),
                        ("mujoco", "the scene the gate checks contact against"),
                        ("placo", "inverse kinematics"),
                        ("av", "H.264 for the recordings")):
        have = _importable(module)
        out.append(_item(f"module_{module}", f"{module} - {why}", have is not None,
                         have or "not importable", "bash scripts/setup.sh"))
    # The nodes set it themselves; a `ros2 service call` typed in a shell on
    # another one reaches them with its fields zeroed. Optional: nothing the
    # robot runs depends on it.
    shell = settings.shell_ros_middleware()
    out.append(_item("ros_middleware", f"ros2 typed in this shell runs on {settings.ros_middleware}",
                     shell == settings.ros_middleware, shell or "not set - the distribution's default",
                     f"export RMW_IMPLEMENTATION={settings.ros_middleware}", optional=True))
    if sys.platform.startswith("linux"):
        dialout = _in_group("dialout")
        out.append(_item("dialout", "serial ports open for this user (dialout group)",
                         dialout is not False,
                         "in the group" if dialout else "not in the group - log out and in after",
                         "sudo usermod -aG dialout $USER"))
        out.append(_item("udev", "udev rule for the foot switch", UDEV_RULE.is_file(),
                         str(UDEV_RULE) if UDEV_RULE.is_file() else "not installed",
                         UDEV_INSTALL + ", then plug it in again"))

    _tag(out, "machine")

    # --- the arms -------------------------------------------------------
    stored = {}
    try:
        stored = ports.stored() if settings.ports_file.is_file() else {}
    except Exception:  # noqa: BLE001 - an unreadable file is a missing one here
        stored = {}
    have = [arm for arm in ARMS if arm in stored]
    out.append(_item("ports", "arm boards named (which USB board is which arm)", len(have) == len(ARMS),
                     f"{', '.join(have) or 'none'} assigned", "pinkrab ports detect"))
    mount = settings.mount_file
    measured = mount.is_file() and "PLACEHOLDER" not in mount.read_text()
    out.append(_item("mount", "wall mount measured (base spacing, height above the desk)", measured,
                     mount.name if measured else "still the placeholder", "pinkrab mount"))
    for arm in ARMS:
        path = settings.calibration_file(arm)
        out.append(_item(f"calibration_{arm}", f"{arm} arm calibrated (home pose held, every joint swept)",
                         path.is_file(), path.name if path.is_file() else "missing",
                         f"pinkrab calibrate {arm}"))
    try:
        angles.load_offsets(settings.offsets_file)
        out.append(_item("offsets", "joint offsets recorded", True,
                         settings.offsets_file.name))
    except angles.OffsetsMissing as exc:
        out.append(_item("offsets", "joint offsets recorded", False,
                         str(exc).splitlines()[0], "pinkrab calibrate left && pinkrab calibrate right"))
    # What a policy is shown: both arms' pictures - it drives both.
    named = _named_sides(settings.follower_cameras_file)
    out.append(_item("follower_cameras", "follower cameras named (which socket is which arm)",
                     set(named) == set(ARMS), ", ".join(named) or "none", "pinkrab cameras"))
    # A policy is shown each arm's picture cut about that camera's own lens
    # axis, and does not start on an arm without one.
    lenses = _measured_lenses(settings.lens_file)
    for arm in ARMS:
        out.append(_lens_item(f"lens_follower_{arm}", f"{arm} arm gripper lens measured (ChArUco board)",
                              f"follower_{arm}", lenses))

    _tag(out, "arms")

    # --- the handheld ---------------------------------------------------
    jaws = {}
    try:
        jaws = angles.load_handheld(settings.handheld_file) if settings.handheld_file.is_file() else {}
    except Exception:  # noqa: BLE001
        jaws = {}
    # Takes are made with both hands.
    named = _named_sides(settings.cameras_file)
    for device in HANDHELDS:
        hand = device.removeprefix("umi_")
        out.append(_item(f"board_{device}", f"{hand} handheld board named", device in stored,
                         "assigned" if device in stored else "not assigned",
                         f"pinkrab-umi detect {device}"))
        entry = jaws.get(device) or {}
        done = "closed_ticks" in entry and "open_ticks" in entry
        out.append(_item(f"jaw_{device}", f"{hand} handheld jaw calibrated (shut and wide)", done,
                         "recorded" if done else "not recorded",
                         f"pinkrab-umi calibrate {device}"))
    out.append(_item("cameras", "gripper cameras named (which socket is which hand)", set(named) == set(ARMS),
                     ", ".join(named) or "none", "pinkrab-umi cameras"))
    # The dataset's pictures are cut about each hand's lens axis, and a take
    # does not convert without one.
    for device in HANDHELDS:
        hand = device.removeprefix("umi_")
        out.append(_lens_item(f"lens_{device}", f"{hand} handheld gripper lens measured (ChArUco board)",
                              hand, lenses))

    # The top camera is the gripper cameras' module, told from them by its socket.
    top = bool(_named_sides(settings.cameras_file, ("top",)))
    out.append(_item("top_camera", "top camera named (which socket it is in)", top,
                     "named" if top else "not named", "pinkrab-umi cameras --top"))
    # Every marker pose is solved through the top camera's lens, which it
    # carries no numbers of: nothing is solved until it is fitted against the
    # board. The capture opens the camera itself, so the live view is down for it.
    fit = top_lens.fit_entry(settings.lens_file)
    wrong = top_lens.unusable(fit) if fit is not None else None
    if fit is None:
        detail = "not fitted - nothing is solved until it is; close the live view first"
    elif wrong:
        detail = f"{wrong} - capture again and fit"
    else:
        detail = f"fitted {fit.get('fitted')}, {fit.get('views')} views, rms {fit.get('rms_px')} px"
    out.append(_item("lens_top", "top camera lens measured (ChArUco board)", fit is not None and not wrong,
                     detail, "pinkrab-umi lens capture top && pinkrab-umi lens fit top"))
    # A take is solved with the lens the live view had when it was recorded;
    # one recorded before the fit, or before a new one, is solved again - and
    # one resolve could not solve again is said, not hidden.
    if fit is not None and not wrong:
        behind = top_lens.stale(settings.episodes_dir, fit)
        left = top_lens.left_by_resolve(settings.episodes_dir, fit)
        detail = f"{len(behind)} take(s) solved with an older lens" if behind else "all of them"
        if left:
            detail += f"; {len(left)} left by resolve (resolve_skipped in each take says why)"
        out.append(_item("takes_top", "recorded takes solved with the top camera's fitted lens", not behind,
                         detail, "pinkrab-umi resolve"))
    _tag(out, "handheld")
    return out


def with_workspace(path: Path):
    """Settings pointed at another workspace, for a check of the checks."""
    return dataclasses.replace(SETTINGS, workspace=path)
