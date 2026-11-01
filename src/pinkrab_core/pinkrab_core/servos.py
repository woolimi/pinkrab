"""The register values every arm servo is kept at, whichever tool opens the bus.

LeRobot configures an SO-101's servos each time it connects - the position loop's
gains, the fastest response and acceleration, the gripper's burnout protection -
and the arms here were first run that way. Pinkrab's own tools did not, so the
values stayed as the SO-101 left them, on the SO-101's numbering: the gripper's
protection ended up on the servo that became the wrist yaw, and a servo added or
renumbered since sat at its factory values. This writes the same set from every
path - the CLI, the driver, and LeRobot's own commands through the follower class.

Torque is left as it was found. Writing these needs it off, and switching it
back on is done the way the driver does it: the present position is written as
the goal first, so a servo holding an old goal does not leave for it.
"""

from __future__ import annotations

from .config import SETTINGS


def configure(bus, p: int | None = None, i: int | None = None, d: int | None = None,
              torque_after: bool | None = None, settings=SETTINGS) -> None:
    """Write the servo registers on every motor of `bus`.

    `p`, `i`, `d` default to the settings. `torque_after` None leaves each motor's
    torque as it was; True switches every motor on, False leaves them all off.
    """
    from lerobot.motors.feetech import OperatingMode

    p = settings.servo_p_coefficient if p is None else p
    i = settings.servo_i_coefficient if i is None else i
    d = settings.servo_d_coefficient if d is None else d
    was_on = [m for m in bus.motors
              if int(bus.read("Torque_Enable", m, normalize=False, num_retry=2))]

    bus.disable_torque()
    # LeRobot's own part: no response delay, full acceleration, and the angle
    # feedback that keeps readings inside one turn.
    bus.configure_motors()
    for motor in bus.motors:
        bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
        bus.write("P_Coefficient", motor, p)
        bus.write("I_Coefficient", motor, i)
        bus.write("D_Coefficient", motor, d)
        if motor == "gripper":
            bus.write("Max_Torque_Limit", motor, settings.gripper_max_torque_limit)
            bus.write("Protection_Current", motor, settings.gripper_protection_current)
            bus.write("Overload_Torque", motor, settings.gripper_overload_torque)
            bus.write("Torque_Limit", motor, settings.gripper_torque_limit)
        else:
            bus.write("Max_Torque_Limit", motor, settings.joint_max_torque_limit)
            bus.write("Protection_Current", motor, settings.joint_protection_current)
            bus.write("Overload_Torque", motor, settings.joint_overload_torque)
            # Torque_Limit is RAM, copied from Max_Torque_Limit at power-up; a
            # servo that booted with the gripper's 500 keeps it until told.
            bus.write("Torque_Limit", motor, settings.joint_max_torque_limit)

    on = was_on if torque_after is None else (list(bus.motors) if torque_after else [])
    if on:
        here = bus.sync_read("Present_Position", on, normalize=False, num_retry=2)
        bus.sync_write("Goal_Position", {m: int(v) for m, v in here.items()}, normalize=False)
        bus.enable_torque(on)
