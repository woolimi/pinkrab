"""Servo ticks and URDF joint angles, in both directions.

A tick carries no angle on its own. `pinkrab calibrate` records the tick each joint
read at the home pose, which is the description's zero, and this turns ticks into
angles from there. It refuses to guess when that record is missing.

The way back is what a write path needs, and it is stricter: a reading nobody can
place is a number nobody acts on, while an angle turned into the wrong tick moves
metal. So joints with no offset recorded, and angles that fall outside the
encoder's own range, are dropped rather than approximated.
"""

from __future__ import annotations

import math
from pathlib import Path

from .config import SETTINGS


class OffsetsMissing(RuntimeError):
    """Raised when tick-to-angle offsets have not been recorded for this rig."""


def load_offsets(path: Path | None = None) -> dict:
    import yaml

    fpath = path or SETTINGS.offsets_file
    if not fpath.is_file():
        raise OffsetsMissing(
            f"No tick-to-angle offsets at {fpath}.\n"
            "  A servo tick has no joint angle until the home pose is recorded:\n"
            "  run `pinkrab calibrate left` and `pinkrab calibrate right`."
        )
    data = yaml.safe_load(fpath.read_text())
    if not isinstance(data, dict) or "offsets_deg" not in data:
        raise OffsetsMissing(f"{fpath} has no offsets_deg section.")
    # A file that carries a reversed sign was measured against a description whose
    # axes did not all follow the servos. Its offsets place those joints at the
    # opposite angle, and a wrong angle on the write path moves metal the wrong way.
    if _carries_reversed_sign(data):
        raise OffsetsMissing(
            f"{fpath} reverses some servos against the description's axes, which now\n"
            "  follow the servos, so its offsets mean other angles. Recalibrate:\n"
            "  run `pinkrab calibrate left` and `pinkrab calibrate right`."
        )
    # An offset is the tick read at the home pose, and it means the description's
    # zero only while that is the pose the description calls zero. Measured at
    # another home - the arm rebuilt, a zero moved in home.yaml - it places every
    # joint at an angle the arm is not at, and the gate checks a pose that is not
    # the arm's.
    moved = _measured_at_another_home(data)
    if moved:
        raise OffsetsMissing(
            f"{fpath} was measured at another home pose than the description's "
            f"({' and '.join(moved)}): recalibrate.\n"
            + "".join(f"  run `pinkrab calibrate {arm}`\n" for arm in moved).rstrip("\n")
        )
    return data


def _carries_reversed_sign(data: dict) -> bool:
    return any(int(sign) < 0 for per_arm in (data.get("signs") or {}).values()
               for sign in (per_arm or {}).values())


def description_home() -> dict[str, dict[str, float]]:
    """The pose the description calls zero, per arm, as config/home.yaml states it."""
    import yaml

    data = yaml.safe_load((SETTINGS.description_pkg / "config" / "home.yaml").read_text())
    return {arm: {name: float(v) for name, v in joints.items()} for arm, joints in data.items()}


def _measured_at_another_home(data: dict) -> list[str]:
    """The arms whose offsets were not recorded at the description's home pose."""
    home = description_home()
    recorded = data.get("home_deg") or {}
    return [arm for arm in (data.get("offsets_deg") or {}) if recorded.get(arm) != home.get(arm)]


OFFSETS_HEADER = """\
# What a servo tick means, in URDF joint angles.
#
#     angle_deg = (ticks - neutral_ticks) * 360 / resolution + offset_deg
#
# Written by `pinkrab calibrate`. Each offset is the angle that puts the tick read at
# the home pose - held when calibration asked for it - at zero, which is the pose
# config/home.yaml bakes into the description. Recalibrating an arm rewrites that
# arm's offsets; the other arm's section is kept while its home_deg is still the
# description's, and dropped once it is not.
#
# home_deg is that pose as home.yaml stated it when the arm was calibrated. An arm
# whose home_deg is no longer the description's has to be calibrated again.
#
# Every joint axis in the description points the way its servo counts up, so there
# is no sign to apply.
"""


def record_zero_pose(
    arm: str, zero_ticks: dict[str, int], path: Path | None = None,
    neutral: int = 2048, resolution: int = 4095,
) -> Path:
    """Write the offsets that make the ticks read at the home pose angle zero."""
    import yaml

    fpath = path or SETTINGS.offsets_file
    data = {}
    if fpath.is_file():
        loaded = yaml.safe_load(fpath.read_text())
        if isinstance(loaded, dict):
            data = loaded
    if data.get("neutral_ticks", neutral) != neutral or data.get("resolution", resolution) != resolution:
        data = {}
    # The other arm's offsets are kept only if they mean the same angles: no
    # servo reversed, and measured at the home pose the description has now.
    home = description_home()
    moved = _measured_at_another_home(data)
    offsets = {} if _carries_reversed_sign(data) else {
        other: values for other, values in (data.get("offsets_deg") or {}).items() if other not in moved}
    offsets[arm] = {
        name: round((neutral - tick) * 360.0 / resolution, 3)
        for name, tick in zero_ticks.items()
    }
    body = {"neutral_ticks": neutral, "resolution": resolution, "offsets_deg": offsets,
            "home_deg": {each: home[each] for each in offsets}}
    fpath.parent.mkdir(parents=True, exist_ok=True)
    fpath.write_text(OFFSETS_HEADER + "\n" + yaml.safe_dump(body, sort_keys=False))
    return fpath


# A joint's band has to lean this far to one side of zero before a sweep can say
# which way the servo counts: past it, one side of the travel is plainly the long
# one. shoulder_pan, wrist_flex and wrist_roll (175 degrees either way of home)
# straddle zero almost evenly and are not judged; the rest lean by 70 degrees or
# more.
SWEEP_LEAN_DEG = 40.0


def unjudged(bands: dict[str, tuple[float, float]]) -> list[str]:
    """The joints a sweep cannot check the servo's direction on: their bands
    straddle zero too evenly (SWEEP_LEAN_DEG)."""
    return [name for name, (lo, hi) in bands.items() if abs(lo + hi) < SWEEP_LEAN_DEG]


def joint_bands(arm: str) -> dict[str, tuple[float, float]]:
    """Each joint's band in degrees, as the description declares it, by bare name."""
    import xml.etree.ElementTree as ET

    from .kinematics import fresh_urdf

    tree = ET.fromstring(fresh_urdf().read_text())
    prefix = f"{arm}_"
    out = {}
    for joint in tree.findall("joint"):
        name, band = joint.get("name", ""), joint.find("limit")
        if name.startswith(prefix) and band is not None:
            out[name[len(prefix):]] = (
                math.degrees(float(band.get("lower"))), math.degrees(float(band.get("upper")))
            )
    return out


def sweep_direction_disagreements(
    arm: str, sweeps: dict[str, tuple[int, int, int]], bands: dict[str, tuple[float, float]],
    resolution: int = 4095,
) -> list[str]:
    """Joints whose sweep says the servo counts against the description's axis.

    `sweeps` is per joint the tick the home pose read and the lowest and highest
    ticks the sweep reached, all in one frame. The description's band says which
    side of zero the long way of the travel lies; the sweep has to put its long way
    on the same side. A joint whose band leans less than SWEEP_LEAN_DEG is not
    judged - a sweep cannot tell on it.

    Returns one line per disagreement, ready to print. Empty means agreement, or
    nothing a sweep could see.
    """
    k = 360.0 / resolution
    out = []
    for name, (zero, low, high) in sweeps.items():
        band = bands.get(name)
        if band is None or name in unjudged({name: band}):
            continue
        # The two ends of the sweep as angles: one at or above zero, one at or below it.
        ends = sorted(((low - zero) * k, (high - zero) * k))
        down, up = ends
        band_leans_up = band[0] + band[1] > 0
        sweep_leans_up = up + down > 0
        if abs(up + down) < SWEEP_LEAN_DEG or band_leans_up == sweep_leans_up:
            continue
        out.append(
            f"{name}: the sweep ran {up:.0f} deg one way and {down:.0f} the other, but the "
            f"description's band is {band[0]:.0f}..{band[1]:.0f} - either this servo is fitted "
            f"the other way round from the description, or the sweep did not reach both stops"
        )
    return out


def ticks_to_degrees(arm: str, ticks: dict[str, int], offsets: dict | None = None) -> dict[str, float]:
    """Joint angles in degrees for one arm."""
    data = offsets or load_offsets()
    neutral = data.get("neutral_ticks", 2048)
    resolution = data.get("resolution", 4095)
    per_arm = data["offsets_deg"].get(arm)
    if per_arm is None:
        raise OffsetsMissing(f"No offsets recorded for the {arm} arm.")

    out = {}
    for name, value in ticks.items():
        if name not in per_arm or not isinstance(value, (int, float)):
            continue
        out[name] = (value - neutral) * 360.0 / resolution + per_arm[name]
    return out


def ticks_to_radians(arm: str, ticks: dict[str, int], offsets: dict | None = None) -> dict[str, float]:
    return {k: math.radians(v) for k, v in ticks_to_degrees(arm, ticks, offsets).items()}


def degrees_to_ticks(arm: str, degrees: dict[str, float], offsets: dict | None = None) -> dict[str, int]:
    """The tick that puts a joint at this angle - `ticks_to_degrees` turned around.

    A joint with no offset recorded is left out rather than guessed at. Reading
    an unknown joint gives a number nobody acts on; writing one moves metal, so
    the two directions cannot be equally forgiving.

    Ticks outside the encoder's own range are dropped for the same reason. They
    mean the angle asked for is not on this servo's dial at all, and the nearest
    tick that is would be a quiet substitution of some other pose for the one
    that was asked for.
    """
    data = offsets or load_offsets()
    neutral = data.get("neutral_ticks", 2048)
    resolution = data.get("resolution", 4095)
    per_arm = data["offsets_deg"].get(arm)
    if per_arm is None:
        raise OffsetsMissing(f"No offsets recorded for the {arm} arm.")

    out: dict[str, int] = {}
    for name, angle in degrees.items():
        if name not in per_arm or not isinstance(angle, (int, float)):
            continue
        tick = round((angle - per_arm[name]) * resolution / 360.0 + neutral)
        if 0 <= tick <= resolution:
            out[name] = int(tick)
    return out


def radians_to_ticks(arm: str, radians: dict[str, float], offsets: dict | None = None) -> dict[str, int]:
    return degrees_to_ticks(arm, {k: math.degrees(v) for k, v in radians.items()}, offsets)


class HandheldCalibrationMissing(RuntimeError):
    """Raised when the handheld's ticks have not been paired with its travel."""


def load_handheld(path: Path | None = None) -> dict:
    import yaml

    fpath = path or SETTINGS.handheld_file
    if not fpath.is_file():
        raise HandheldCalibrationMissing(
            f"No handheld calibration at {fpath}.\n"
            "  A servo tick has no jaw opening until the two ends are recorded:\n"
            "  run `pinkrab-umi calibrate umi_left`."
        )
    data = yaml.safe_load(fpath.read_text())
    if not isinstance(data, dict):
        raise HandheldCalibrationMissing(f"{fpath} is not a mapping.")
    return data


# One turn of the handheld's servo, in ticks. The jaw's travel is a fraction of
# a turn, but the servo counts on past 4095 and below 0 from wherever it woke up,
# so the same shut jaw has read 325 in one session and 4422 in the next. Every
# comparison of two readings goes through `handheld_wrap`, which takes the short
# way round the circle, and nothing depends on which turn the servo thinks it is on.
HANDHELD_TICKS_PER_TURN = 4096


def handheld_wrap(delta: float) -> float:
    """The shortest signed way round the circle, in (-half turn, +half turn]."""
    half = HANDHELD_TICKS_PER_TURN / 2
    return (delta + half) % HANDHELD_TICKS_PER_TURN - half


def handheld_span(closed_ticks: float, open_ticks: float) -> float:
    """Signed travel from shut to wide, ticks, the short way round the circle."""
    return handheld_wrap(float(open_ticks) - float(closed_ticks))


def handheld_aperture(device: str, ticks: int, data: dict | None = None) -> float:
    """How far open the handheld's jaw is, 0 shut to 1 wide, from raw ticks.

    A fraction rather than an angle. What the servo can say is where it is
    between the two ends the hand can actually reach, and turning that into a
    joint angle would be tying the reading to the follower's picture of its own
    joint, not to the thing reading a servo. Values outside the recorded ends
    are clamped: a hand squeezing past `closed` is still a shut jaw.
    """
    entry = (data or load_handheld()).get(device)
    if not entry:
        raise HandheldCalibrationMissing(
            f"No calibration recorded for {device}.\n"
            f"  Run: pinkrab-umi calibrate {device}"
        )
    shut, wide = float(entry["closed_ticks"]), float(entry["open_ticks"])
    span = handheld_span(shut, wide)
    if span == 0:
        raise HandheldCalibrationMissing(f"{device}: closed and open ticks are the same.")
    return max(0.0, min(1.0, handheld_wrap(float(ticks) - shut) / span))

