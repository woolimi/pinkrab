"""Read-only servo inspection over the Feetech bus.

LeRobot has no command that checks servos on their own, so a broken daisy chain or a
mis-assigned id only surfaces as a joint that never moves during calibration. This
module pings the bus directly and reports what answered.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import SETTINGS

# Model number every STS3215 reports. Anything else on an expected id is a servo we
# do not know how to drive, which is worth saying out loud rather than ignoring.
EXPECTED_MODEL_NUMBER = 777


@dataclass
class MotorStatus:
    name: str
    expected_id: int
    found: bool
    model_number: int | None = None
    position: int | None = None
    temperature: int | None = None

    @property
    def ok(self) -> bool:
        return self.found and self.model_number == EXPECTED_MODEL_NUMBER


@dataclass
class ArmScan:
    arm: str
    port: Path
    motors: list[MotorStatus]
    unexpected_ids: dict[int, int]  # id -> model number, ids nobody asked for
    # Ids where several servos answer at once, garbling the frame. A Feetech servo
    # leaves the factory as id 1, so every servo that never went through
    # `setup-motors` sits there and collides with its siblings.
    collision_ids: list[int] = field(default_factory=list)

    @property
    def found_count(self) -> int:
        return sum(1 for m in self.motors if m.found)

    @property
    def ok(self) -> bool:
        return all(m.ok for m in self.motors) and not self.unexpected_ids

    def first_silent(self) -> MotorStatus | None:
        """The earliest joint in the chain that did not answer.

        The servos are daisy-chained from the base outward, so a break takes out
        everything after it. Naming the first one points at the connector to check.
        """
        for motor in self.motors:
            if not motor.found:
                return motor
        return None


def make_bus(port: Path):
    """Build a bus for the expected servos. Imported lazily — LeRobot is heavy."""
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    motors = {
        name: Motor(motor_id, SETTINGS.motor_model, MotorNormMode.RANGE_M100_100)
        for name, motor_id in SETTINGS.motor_ids.items()
    }
    return FeetechMotorsBus(port=str(port), motors=motors)


def make_handheld_bus(port: Path):
    """A bus for the handheld's one servo, under the joint name it stands for.

    The handheld carries the gripper servo and nothing else, so a bus built for
    seven would report six silences every time it is opened.
    """
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    name = SETTINGS.handheld_joint
    motors = {name: Motor(SETTINGS.motor_ids[name], SETTINGS.motor_model,
                          MotorNormMode.RANGE_M100_100)}
    return FeetechMotorsBus(port=str(port), motors=motors)


def scan_handheld(device: str, port: Path) -> ArmScan:
    """Ping the handheld's bus and report what is on it. Read-only.

    Broadcast rather than a ping at the expected id, because the answer that
    matters before numbering is "a servo is there, under the wrong number" -
    a servo out of its box answers to 1, and one pulled off an arm answers to
    whatever that joint was.
    """
    name = SETTINGS.handheld_joint
    motor_id = SETTINGS.motor_ids[name]
    bus = make_handheld_bus(port)
    bus.connect(handshake=False)
    try:
        bus.set_baudrate(SETTINGS.baudrate)
        found = bus.broadcast_ping() or {}
        status = MotorStatus(name=name, expected_id=motor_id, found=motor_id in found)
        if status.found:
            status.model_number = found[motor_id]
            status.position = _try_read(bus, "Present_Position", name)
            status.temperature = _try_read(bus, "Present_Temperature", name)
        unexpected = {i: m for i, m in found.items() if i != motor_id}
        collisions = [] if status.found else [i for i in (motor_id,) if _is_collision(bus, i)]
    finally:
        bus.disconnect(disable_torque=False)
    return ArmScan(arm=device, port=port, motors=[status],
                   unexpected_ids=unexpected, collision_ids=collisions)


def read_handheld(port: Path) -> int | None:
    """Where the handheld's servo is, raw ticks. Read-only, torque untouched."""
    bus = make_handheld_bus(port)
    bus.connect(handshake=False)
    try:
        bus.set_baudrate(SETTINGS.baudrate)
        return _try_read(bus, "Present_Position", SETTINGS.handheld_joint)
    finally:
        bus.disconnect(disable_torque=False)


def scan_arm(arm: str, port: Path, retries: int = 1, settle_s: float = 0.5) -> ArmScan:
    """Ping every id on one arm's bus and read raw position/temperature.

    Retries once on a failure to open. A board that was just replugged, or one
    opened immediately after another arm's port was closed, can report readiness and
    then return nothing for a moment. Reporting that as a hardware fault sends
    someone looking for a broken servo that is not broken.
    """
    last: Exception | None = None
    for attempt in range(retries + 1):
        if attempt:
            time.sleep(settle_s)
        try:
            return _scan_once(arm, port)
        except Exception as exc:  # noqa: BLE001 - retried, then re-raised as-is
            last = exc
    assert last is not None
    raise last


def _scan_once(arm: str, port: Path) -> ArmScan:
    bus = make_bus(port)
    # handshake=False: the whole point is to report which servos are missing rather
    # than to fail on the first one that is.
    bus.connect(handshake=False)
    try:
        bus.set_baudrate(SETTINGS.baudrate)
        found = bus.broadcast_ping() or {}

        motors: list[MotorStatus] = []
        for name, motor_id in SETTINGS.motor_ids.items():
            status = MotorStatus(name=name, expected_id=motor_id, found=motor_id in found)
            if status.found:
                status.model_number = found[motor_id]
                status.position = _try_read(bus, "Present_Position", name)
                status.temperature = _try_read(bus, "Present_Temperature", name)
            motors.append(status)

        expected_ids = set(SETTINGS.motor_ids.values())
        unexpected = {i: m for i, m in found.items() if i not in expected_ids}
        # A silent id has two very different causes. Ask each one directly: a clean
        # timeout means nobody is there, a corrupt frame means several servos are.
        collisions = [i for i in sorted(expected_ids - set(found)) if _is_collision(bus, i)]
    finally:
        # Never disable torque here — this command is read-only and may be run while
        # an arm is being held in a pose.
        bus.disconnect(disable_torque=False)

    return ArmScan(
        arm=arm, port=port, motors=motors, unexpected_ids=unexpected, collision_ids=collisions
    )


def read_raw(bus, motor: str, register: str = "Present_Position", retries: int = 2) -> int | None:
    """One register off an open bus, unnormalised, None if the servo is silent.

    Public because a control loop cannot afford to open and close the port on
    every tick, and because normalising wants a LeRobot calibration that a bus
    built for one handheld servo has no business carrying.
    """
    return _try_read(bus, register, motor, retries)


def _try_read(bus, register: str, motor: str, retries: int = 2) -> int | None:
    try:
        return int(bus.read(register, motor, normalize=False, num_retry=retries))
    except Exception:  # noqa: BLE001 — a silent servo must not abort the whole scan
        return None


def _is_collision(bus, motor_id: int, attempts: int = 6) -> bool:
    """True if several servos answer on this id at once.

    A corrupt reply frame means more than one servo transmitted; a timeout means the
    id is genuinely empty. Repeated because a single garbled read could be noise.
    """
    from scservo_sdk import COMM_RX_CORRUPT

    corrupt = 0
    for _ in range(attempts):
        _model, comm, _err = bus.packet_handler.ping(bus.port_handler, motor_id)
        if comm == COMM_RX_CORRUPT:
            corrupt += 1
    return corrupt >= max(2, attempts // 2)
