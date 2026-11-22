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

    `p`, `i`, `d` default to the settings - `i` per joint, 0 on the joints that
    carry no weight (`servo_i_unloaded`); one given is written to every motor.
    `torque_after` None leaves each motor's torque as it was; True switches
    every motor on, False leaves them all off.
    """
    p = settings.servo_p_coefficient if p is None else p
    d = settings.servo_d_coefficient if d is None else d
    wanted = {m: _wanted(m, p, i, d, settings) for m in bus.motors}
    was_on = [m for m in bus.motors
              if int(bus.read("Torque_Enable", m, normalize=False, num_retry=2))]
    if all(_holds(bus, m, regs) for m, regs in wanted.items()):
        # Already as wanted: the torque is not touched except to reach `torque_after`.
        if torque_after is False:
            bus.disable_torque(was_on)
        elif torque_after:
            _enable(bus, [m for m in bus.motors if m not in was_on])
        return

    # A write that fails half way must not leave a motor that was holding limp.
    try:
        bus.disable_torque()
        # LeRobot's own part: no response delay, full acceleration, and the angle
        # feedback that keeps readings inside one turn.
        bus.configure_motors()
        for motor, regs in wanted.items():
            for name, value in regs.items():
                bus.write(name, motor, value)
    except BaseException:
        _enable(bus, was_on)
        raise
    _enable(bus, was_on if torque_after is None else list(bus.motors) if torque_after else [])


def _wanted(motor: str, p: int, i: int | None, d: int, settings) -> dict[str, int]:
    """The registers `configure` writes on one motor, beyond LeRobot's own part."""
    from lerobot.motors.feetech import OperatingMode

    if motor == "gripper":
        limits = {
            "Max_Torque_Limit": settings.gripper_max_torque_limit,
            "Protection_Current": settings.gripper_protection_current,
            "Overload_Torque": settings.gripper_overload_torque,
            "Torque_Limit": settings.gripper_torque_limit,
        }
    else:
        limits = {
            "Max_Torque_Limit": settings.joint_max_torque_limit,
            "Protection_Current": settings.joint_protection_current,
            "Overload_Torque": settings.joint_overload_torque,
            # Torque_Limit is RAM, copied from Max_Torque_Limit at power-up; a
            # servo that booted with the gripper's 500 keeps it until told.
            "Torque_Limit": settings.joint_max_torque_limit,
        }
    if i is None:
        i = 0 if motor in settings.servo_i_unloaded else settings.servo_i_coefficient
    return {"Operating_Mode": OperatingMode.POSITION.value, "P_Coefficient": p,
            "I_Coefficient": i, "D_Coefficient": d, **limits}


def _holds(bus, motor: str, regs: dict[str, int]) -> bool:
    """Whether `motor` already has every register `configure` would write."""
    held = {"Return_Delay_Time": 0, "Maximum_Acceleration": 254, "Acceleration": 254, **regs}
    if any(int(bus.read(name, motor, normalize=False, num_retry=2)) != value
           for name, value in held.items()):
        return False
    return not (bus.motors[motor].model == "sts3215"
                and int(bus.read("Phase", motor, normalize=False, num_retry=2)) & 0x10)


def _enable(bus, motors: list[str]) -> None:
    """Torque on at the present position, so a motor holding an old goal stays put."""
    if not motors:
        return
    here = bus.sync_read("Present_Position", motors, normalize=False, num_retry=2)
    bus.sync_write("Goal_Position", {m: int(v) for m, v in here.items()}, normalize=False)
    bus.enable_torque(motors)
