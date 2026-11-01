"""Fit the simulated STS3215 to step responses recorded on the real arm.

Reads every trace `step_response.py` wrote under ~/pinkrab/step/, replays each step
in the MuJoCo scene from the pose it started at, and searches one set of servo
values - the position actuator's kp and kv, the joint's friction loss and
armature, and a command delay - for the least squared difference from the real
angles over the first second. Every joint is the same servo, so one set serves
all of them; the damping and force range stay as the model has them.

    python scripts/fit_servo.py            # fit, print, write step/fit.json
    python scripts/fit_servo.py --check    # the model as it runs (delay included), no search
    python scripts/fit_servo.py --integral 8   # the model against the runs at I 8

MuJoCo's position actuator is P and D only; the servos run an integral term as
well (I 8 in config.py). --integral replays the steps recorded at that I with
an integral term of each gain added on top of the actuator, and prints where
each run ends. Measured 2026-09-26: the model with none already ends within
0.3 deg, where the real arm at I 8 ends within 0.5 - its settling short at I 0
was never in the model - so the physics loop adds none.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from pinkrab_core.config import SETTINGS

SCENE = Path(__file__).resolve().parents[1] / "src/pinkrab_description/mjcf/scene.xml"
HOLD_S = 0.5
WINDOW_S = 1.0
INTEGRAL_WINDOW_S = 2.0     # the integral term's work shows over seconds, not the first one
INTEGRAL_LIMIT = math.radians(10.0)   # the most the integral term may add to a command
NAMES = ("kp", "kv", "frictionloss", "armature", "delay_s")


def load_traces(i_gain: int | None = None, window: float = WINDOW_S) -> list[dict]:
    """The recorded steps: those at the default gains, or with i_gain those at that I only."""
    out = []
    for path in sorted((SETTINGS.workspace / "step").glob("*_*_*.json")):
        d = json.loads(path.read_text())
        p, i, dd = (list(d.get("pid") or ()) + [None] * 3)[:3]
        if i_gain is None and any(v is not None for v in (p, i, dd)):
            continue    # a run at other gains than the arm runs is not this servo
        if i_gain is not None and (i != i_gain or p is not None or dd is not None):
            continue
        t, v = np.array(d["real"]["trace"]).T
        keep = t <= window
        out.append({"name": path.stem, "arm": d["arm"], "joint": d["joint"], "step": d["step_deg"],
                    "start": d["start_deg"], "t": t[keep], "real": v[keep] - d["start_deg"][d["joint"]]})
    return out


class Replay:
    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.arm_dofs = [m.jnt_dofadr[j] for j in range(m.njnt)
                         if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
                         and m.joint(j).name.split("_", 1)[0] in ("left", "right")]

    def current(self) -> dict:
        m = self.model
        return {"kp": float(m.actuator_gainprm[0, 0]), "kv": float(-m.actuator_biasprm[0, 2]),
                "frictionloss": float(m.dof_frictionloss[self.arm_dofs[0]]),
                "armature": float(m.dof_armature[self.arm_dofs[0]]), "delay_s": SETTINGS.servo_delay_s}

    def set(self, p: dict) -> None:
        m = self.model
        m.actuator_gainprm[:, 0] = p["kp"]
        m.actuator_biasprm[:, 1] = -p["kp"]
        m.actuator_biasprm[:, 2] = -p["kv"]
        m.dof_frictionloss[self.arm_dofs] = p["frictionloss"]
        m.dof_armature[self.arm_dofs] = p["armature"]

    def run(self, tr: dict, delay: float, integral: float = 0.0,
            window: float = WINDOW_S) -> tuple[np.ndarray, np.ndarray]:
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        for name, deg in tr["start"].items():
            d.qpos[m.jnt_qposadr[m.joint(f"{tr['arm']}_{name}").id]] = math.radians(deg)
        mujoco.mj_forward(m, d)
        adrs = np.array([m.jnt_qposadr[m.actuator_trnid[a, 0]] for a in range(m.nu)])
        cmd = d.qpos[adrs].copy()
        summed = np.zeros(m.nu)
        act = m.actuator(f"{tr['arm']}_{tr['joint']}").id
        adr = m.jnt_qposadr[m.joint(f"{tr['arm']}_{tr['joint']}").id]
        dt = m.opt.timestep

        def step():
            # The servo's integral term, as physics.py adds it: the error summed
            # over time, times the gain, capped, on top of the command.
            if integral > 0:
                summed[:] += (cmd - d.qpos[adrs]) * dt
                d.ctrl[:] = cmd + np.clip(integral * summed, -INTEGRAL_LIMIT, INTEGRAL_LIMIT)
            else:
                d.ctrl[:] = cmd
            mujoco.mj_step(m, d)

        for _ in range(round(HOLD_S / dt)):
            step()
        q0 = d.qpos[adr]
        goal = math.radians(tr["start"][tr["joint"]] + tr["step"])
        ts, qs = [], []
        for k in range(round(window / dt) + 1):
            if k * dt >= delay:
                cmd[act] = goal
            ts.append(k * dt)
            qs.append(math.degrees(d.qpos[adr] - q0))
            step()
        return np.array(ts), np.array(qs)

    def errors(self, traces: list[dict], p: dict, integral: float = 0.0,
               window: float = WINDOW_S) -> list[float]:
        self.set(p)
        out = []
        for tr in traces:
            ts, qs = self.run(tr, p["delay_s"], integral, window)
            sim = np.interp(tr["t"], ts, qs)
            out.append(float(np.sqrt(np.mean((sim - tr["real"]) ** 2))))
        return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="score the model's own values only")
    ap.add_argument("--integral", type=int, metavar="I",
                    help="fit the integral term's gain to the runs recorded at this I")
    args = ap.parse_args()

    if args.integral is not None:
        fit_integral(args.integral)
        return

    traces = load_traces()
    replay = Replay()
    base = replay.current()
    if not args.check:
        from scipy.optimize import minimize

        def cost(x):
            p = dict(zip(NAMES, [*np.exp(x[:4]), x[4]]))
            if not 0.0 <= p["delay_s"] <= 0.15:
                return 1e3
            return float(np.mean(replay.errors(traces, p)))

        x0 = [*np.log([base[n] for n in NAMES[:4]]), base["delay_s"]]
        res = minimize(cost, x0, method="Nelder-Mead",
                       options={"maxiter": 400, "xatol": 1e-3, "fatol": 1e-3, "disp": True})
        best = dict(zip(NAMES, [*np.exp(res.x[:4]), res.x[4]]))
    else:
        best = base

    before = replay.errors(traces, base)
    after = replay.errors(traces, best)
    print(f"{'trace':42s} {'model':>8s} {'fit':>8s}   RMS error, degrees")
    for tr, b, a in zip(traces, before, after):
        print(f"{tr['name']:42s} {b:8.2f} {a:8.2f}")
    print(f"{'mean':42s} {np.mean(before):8.2f} {np.mean(after):8.2f}")
    print("model:", {k: round(v, 4) for k, v in base.items()})
    print("fit:  ", {k: round(v, 4) for k, v in best.items()})
    if not args.check:
        print("delay_s is config.py's servo_delay_s; kp, kv, frictionloss and armature are the MJCF's")
    if not args.check:
        (SETTINGS.workspace / "step" / "fit.json").write_text(json.dumps(
            {"model": base, "fit": best, "rms_before": before, "rms_after": after,
             "traces": [t["name"] for t in traces]}, indent=1))


def fit_integral(i_gain: int) -> None:
    traces = load_traces(i_gain, INTEGRAL_WINDOW_S)
    if not traces:
        raise SystemExit(f"no step runs recorded at I {i_gain} - step_response.py --i {i_gain}")
    replay = Replay()
    p = replay.current()

    def settled(t, v):
        return float(np.mean(v[t >= INTEGRAL_WINDOW_S - 0.5]))

    print("short of the step once settled, degrees")
    print("  real      " + "  ".join(f"{tr['step'] - settled(tr['t'], tr['real']):+.2f}" for tr in traces))
    for ki in (0.0, 2.0, 4.0, 8.0, 16.0):
        replay.set(p)
        ends = []
        for tr in traces:
            ts, qs = replay.run(tr, p["delay_s"], ki, INTEGRAL_WINDOW_S)
            ends.append(tr["step"] - settled(ts, qs))
        rms = np.mean(replay.errors(traces, p, ki, INTEGRAL_WINDOW_S))
        print(f"  sim {ki:4.1f}  " + "  ".join(f"{e:+.2f}" for e in ends) + f"   RMS {rms:.2f}")


if __name__ == "__main__":
    main()
