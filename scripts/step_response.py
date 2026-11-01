"""Step one joint on the real arm and the same joint in MuJoCo, and compare the two.

Whether the simulated STS3215 behaves like the servos on this rig is a measurement,
not a property of the model file: the same step goes to both, from the same pose,
and the two traces are written side by side for scripts/fit_servo.py.

    python scripts/step_response.py left elbow_flex --step 20

    python scripts/step_response.py left shoulder_lift --step -15 --at 30

THE ARM MOVES. Every joint of the chosen arm but the wrist's roll and the jaw is
held where it stands; the one joint walks to `--at` if given, steps by `--step`
degrees, then walks back, and torque is left as it was found.
The traces go to ~/pinkrab/step/<arm>_<joint>_<time>.json.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from pinkrab_core import ports, servos
from pinkrab_core.angles import degrees_to_ticks, joint_bands, load_offsets, ticks_to_degrees
from pinkrab_core.bus import make_bus
from pinkrab_core.config import SETTINGS

SCENE = Path(__file__).resolve().parents[1] / "src/pinkrab_description/mjcf/scene.xml"
HOLD_S = 1.0          # both: held at the start pose this long before the step
WALK_DPS = 60.0       # how fast --at walks the joint out to its start and back
SETTLE_BAND_DEG = 1.0  # settled once the joint stays within this of its goal


def walk(bus, joint: str, frm: int, to: int) -> None:
    """Move one held joint between two ticks at WALK_DPS, in small goals."""
    n = max(1, round(abs(to - frm) * 360 / 4096 / WALK_DPS / 0.02))
    for k in range(1, n + 1):
        bus.write("Goal_Position", joint, round(frm + (to - frm) * k / n), normalize=False)
        time.sleep(0.02)


def real_step(arm: str, joint: str, step: float, seconds: float,
              at: float | None = None, pid: tuple | None = None) -> tuple[dict, list]:
    lo, hi = joint_bands(arm)[joint]
    bus = make_bus(ports.load()[arm])
    bus.connect(handshake=False)
    bus.set_baudrate(SETTINGS.baudrate)
    offsets = load_offsets()
    try:
        servos.configure(bus, *(pid or (None, None, None)))
        was_on = [m for m in bus.motors
                  if int(bus.read("Torque_Enable", m, normalize=False, num_retry=2))]
        here = bus.sync_read("Present_Position", normalize=False, num_retry=2)
        begin = ticks_to_degrees(arm, here, offsets)[joint]
        base = begin if at is None else at
        if not (lo <= base <= hi and lo <= base + step <= hi):
            raise SystemExit(f"{joint}: {base:.1f} and {base + step:.1f} are not both inside {lo:.0f}..{hi:.0f}")

        # The wrist's roll and the jaw stay limp: they carry nothing the step
        # loads, and a servo powered up holds a goal of 0 until told otherwise, so
        # each goal written here is read back before any torque goes on.
        held = [m for m in bus.motors if m not in ("wrist_roll", "gripper")]
        for m in held:
            bus.write("Goal_Position", m, int(here[m]), normalize=False)
            if int(bus.read("Goal_Position", m, normalize=False, num_retry=2)) != int(here[m]):
                raise SystemExit(f"{m}: goal did not take, nothing switched on")
        bus.enable_torque(held)
        if at is not None:
            walk(bus, joint, int(here[joint]), degrees_to_ticks(arm, {joint: at}, offsets)[joint])
        time.sleep(HOLD_S)
        pose = bus.sync_read("Present_Position", normalize=False, num_retry=2)
        start = ticks_to_degrees(arm, pose, offsets)
        goal = degrees_to_ticks(arm, {joint: start[joint] + step}, offsets)[joint]

        trace = []
        t0 = time.monotonic()
        bus.write("Goal_Position", joint, goal, normalize=False)
        while (t := time.monotonic() - t0) < seconds:
            tick = int(bus.read("Present_Position", joint, normalize=False, num_retry=2))
            trace.append((t, ticks_to_degrees(arm, {joint: tick}, offsets)[joint]))

        walk(bus, joint, goal, int(here[joint]))
        time.sleep(HOLD_S)
        off = [m for m in held if m not in was_on]
        if off:
            bus.disable_torque(off)
        after = bus.sync_read("Present_Position", normalize=False, num_retry=2)
        first = ticks_to_degrees(arm, here, offsets)
        moved = {m: round(d - first[m], 1) for m, d in ticks_to_degrees(arm, after, offsets).items()}
        print("back at the start, moved by (deg):", moved)
    finally:
        bus.disconnect(disable_torque=False)
    return start, trace


def sim_step(arm: str, joint: str, start: dict, step: float, seconds: float) -> list:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    for name, deg in start.items():
        data.qpos[model.jnt_qposadr[model.joint(f"{arm}_{name}").id]] = math.radians(deg)
    mujoco.mj_forward(model, data)
    for a in range(model.nu):
        data.ctrl[a] = data.qpos[model.jnt_qposadr[model.actuator_trnid[a, 0]]]
    target = model.actuator(f"{arm}_{joint}").id
    adr = model.jnt_qposadr[model.joint(f"{arm}_{joint}").id]

    for _ in range(round(HOLD_S / model.opt.timestep)):
        mujoco.mj_step(model, data)
    data.ctrl[target] = math.radians(start[joint] + step)
    trace = []
    for k in range(round(seconds / model.opt.timestep)):
        trace.append((k * model.opt.timestep, math.degrees(data.qpos[adr])))
        mujoco.mj_step(model, data)
    return trace


def metrics(trace: list, start: float, step: float) -> dict:
    t = np.array([p[0] for p in trace])
    y = (np.array([p[1] for p in trace]) - start) * np.sign(step)
    size = abs(step)

    def reached(frac):
        return float(t[np.argmax(y >= frac * size)]) if (y >= frac * size).any() else None

    outside = np.nonzero(np.abs(y - size) > SETTLE_BAND_DEG)[0]
    settled = float(t[outside[-1] + 1]) if len(outside) and outside[-1] + 1 < len(t) else (0.0 if not len(outside) else None)
    tail = y[t >= t[-1] - 0.2]
    late = y[t >= t[-1] - 0.8]
    return {
        "samples": len(t),
        "rate_hz": round(float(len(t) / t[-1]), 1) if t[-1] else None,
        "rise_10_90_s": (lambda a, b: round(b - a, 3) if a is not None and b is not None else None)(reached(0.1), reached(0.9)),
        "overshoot_deg": round(float(y.max() - size), 2),
        "settle_s": None if settled is None else round(settled, 3),
        "final_error_deg": round(float(size - tail.mean()), 2),
        # How much the joint still moves once it should have settled: a gain
        # that makes it hunt shows here before it shows anywhere else.
        "late_ripple_deg": round(float(late.max() - late.min()), 2),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("arm", choices=("left", "right"))
    p.add_argument("joint", choices=[n for n in SETTINGS.joint_names if n != "gripper"])
    p.add_argument("--step", type=float, default=20.0, help="degrees, signed")
    p.add_argument("--seconds", type=float, default=2.0)
    p.add_argument("--at", type=float, help="walk the joint here first, degrees")
    p.add_argument("--p", type=int, help="P coefficient for this run (default: the config's)")
    p.add_argument("--i", type=int, help="I coefficient for this run")
    p.add_argument("--d", type=int, help="D coefficient for this run")
    args = p.parse_args()

    pid = (args.p, args.i, args.d)
    start, real = real_step(args.arm, args.joint, args.step, args.seconds, args.at,
                            pid if any(v is not None for v in pid) else None)
    sim = sim_step(args.arm, args.joint, start, args.step, args.seconds)

    s0 = start[args.joint]
    out = {
        "arm": args.arm, "joint": args.joint, "step_deg": args.step, "start_deg": start,
        "real": {"metrics": metrics(real, s0, args.step), "trace": real},
        "sim": {"metrics": metrics(sim, s0, args.step), "trace": sim},
    }
    path = SETTINGS.workspace / "step" / f"{args.arm}_{args.joint}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out["at_deg"] = args.at
    out["pid"] = pid
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    for name in ("real", "sim"):
        print(f"{name:5s}", out[name]["metrics"])
    print("written", path)


if __name__ == "__main__":
    main()
