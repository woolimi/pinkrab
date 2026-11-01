"""Write servo ids into EEPROM, one arm - or one joint - at a time.

LeRobot's `lerobot-setup-motors` walks every servo on the device in a fixed order,
with no way to stop or to redo a single joint. Replacing one servo, or finishing an
arm that was interrupted, then means sitting through the whole wizard again.

This does the same work through the same ``FeetechMotorsBus.setup_motor`` call, but
lets the caller pick the arm and the joints. Both arms use the same numbering
(shoulder_pan..gripper = 1..6) because they sit on separate buses, so the arm only
selects which serial port is written to.
"""

from __future__ import annotations

from pathlib import Path

from .bus import make_bus, make_handheld_bus
from .config import SETTINGS


def assign_order(joints: list[str] | None = None) -> list[str]:
    """Gripper first, then back toward the base.

    Working from the free end of the harness inward means the servo being written is
    always the one that can be detached and reattached without disturbing the ones
    already done.
    """
    if joints is None:
        return list(reversed(SETTINGS.joint_names))

    unknown = [j for j in joints if j not in SETTINGS.motor_ids]
    if unknown:
        raise ValueError(
            f"Unknown joint name(s): {', '.join(unknown)}\n"
            f"  Valid: {', '.join(SETTINGS.joint_names)}"
        )
    # Keep the caller's subset, but force the safe order.
    return [j for j in reversed(SETTINGS.joint_names) if j in joints]


def assign_ids(arm: str, port: Path, joints: list[str], prompt=input, echo=print) -> list[str]:
    """Write ids for `joints` on one arm. Returns the joints actually written.

    Each step needs that servo - and only that servo - connected to the board. The
    id is written to whatever answers, so a second servo on the bus either takes the
    number too or takes it instead.
    """
    bus = make_bus(port)
    bus.connect(handshake=False)
    done: list[str] = []
    try:
        for name in joints:
            motor_id = SETTINGS.motor_ids[name]
            prompt(
                f"\n  Connect ONLY the {arm} '{name}' servo to the board, "
                f"then press ENTER (-> id {motor_id})..."
            )
            bus.setup_motor(name)
            done.append(name)
            echo(f"    ok  {arm} '{name}' -> id {motor_id}")
    finally:
        bus.disconnect(disable_torque=False)
    return done


def assign_handheld_id(device: str, port: Path, prompt=input, echo=print) -> str:
    """Write the gripper's id into the handheld's one servo.

    Same call and same number as the arm's gripper: the handheld reads the joint
    the arm's gripper servo reads, so giving it a second identity would only mean
    a second thing to look up. Its bus carries nothing else, so there is no order
    to keep and nothing to detach first - but the servo IS written to whatever
    answers, so a board with an arm still on it must not be the one plugged in.
    """
    name = SETTINGS.handheld_joint
    motor_id = SETTINGS.motor_ids[name]
    bus = make_handheld_bus(port)
    bus.connect(handshake=False)
    try:
        prompt(
            f"\n  Connect ONLY the {device} handheld's servo to the board, "
            f"then press ENTER (-> id {motor_id})..."
        )
        bus.setup_motor(name)
        echo(f"    ok  {device} '{name}' -> id {motor_id}")
    finally:
        bus.disconnect(disable_torque=False)
    return name
