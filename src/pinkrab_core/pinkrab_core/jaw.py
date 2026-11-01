"""The arm's jaw band: what a handheld's opening, 0 shut to 1 wide, is spent across."""
from __future__ import annotations


def shut_wide(lower: float, upper: float) -> tuple[float, float]:
    """A jaw's limits as (shut, wide): shut is the end nearer zero.

    Zero is the closed jaw on both arms, but the left jaw's servo is turned round
    to fit the mirrored arm and opens toward negative, so which limit is shut
    depends on the arm. Everything that spends an opening across a band takes it
    in this order, `shut + opening * (wide - shut)`.
    """
    return (lower, upper) if abs(lower) <= abs(upper) else (upper, lower)


def gripper_band(arm: str) -> tuple[float, float]:
    """The jaw's own band in radians, (shut, wide), from the description.

    The kinematic chain stops at the wrist - the gripper opens the hand, it does
    not move it - so the band comes from the URDF directly. What the handheld
    reports is a fraction of ITS travel, and a fraction has to be spent across
    the arm's travel to mean anything to a servo.
    """
    import xml.etree.ElementTree as ET

    from .kinematics import fresh_urdf

    tree = ET.fromstring(fresh_urdf().read_text())
    for joint in tree.findall("joint"):
        if joint.get("name") == f"{arm}_gripper":
            band = joint.find("limit")
            return shut_wide(float(band.get("lower")), float(band.get("upper")))
    raise KeyError(f"the description has no {arm}_gripper")


def aperture_of(arm: str, radians: float, band: tuple[float, float] | None = None) -> float:
    """The arm's jaw as a fraction of its band, the way the handheld reports its own."""
    shut, wide = band or gripper_band(arm)
    if wide == shut:
        return 0.0
    return min(1.0, max(0.0, (radians - shut) / (wide - shut)))
