"""Walk one arm through a set of poses over ROS and read how far short of each it stops.

Each pose is asked for twice, once arriving from above it and once from below
(every joint a few degrees past it first), and held until the arm is still. What
the encoders read then, against what was asked, is the error the servos leave;
the gravity torque the model expects at the pose sits beside it, and per joint
the two are fitted against each other:

    error = offset + compliance * gravity torque

A joint whose error follows its load (high R², a compliance well off zero)
could have it taken out by asking for the pose minus (offset + compliance *
torque); a
joint whose two arrivals disagree has backlash or a dead band, which no
feed-forward takes out; a joint well inside half a degree both ways has
nothing left to give.

    python scripts/pose_error.py right --dry     the poses and their clearance, nothing moves
    python scripts/pose_error.py right           the measurement, about three minutes

THE ARM MOVES, over the robot's own path: commands go to /joint_commands and
cross the gate like every other client's, so `ros2 launch pinkrab_bringup
robot.launch.py` has to be up, and nothing else may be publishing commands
(the studio disconnected). The poses are the ready pose's turn with the
shoulder, elbow and wrist bent across a grid; each one, its approaches, every
walk the run takes between them and the straight walk from each back to ready
are kept CLEARANCE_MM off the desk, the board and the other arm in the model.
Out and home the arm goes the way the ready and home buttons do, straight
between home and ready. An approach or a pose the arm does not reach within
FOLLOW_MAX_DEG - a command the gate refused, a joint held - stops the run;
what was measured is written first, then the arm walks home - back to ready
first when it got that far - and is left holding there, torque on, the gate
idle. A stop the gate was told about (/estop) is left as it is.

The run goes to ~/pinkrab/pose_error/<arm>_<time>.json.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import threading
import time
from collections import deque

import numpy as np

from pinkrab_core.angles import joint_bands
from pinkrab_core.config import SETTINGS
from pinkrab_core.origin import READY_DEG, mirrored

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll")
# The grid, for the right arm; the left is its mirror. Pan, yaw and roll stay
# at ready's.
LIFT_DEG = (50.0, 65.0, 80.0, 95.0, 110.0)
ELBOW_DEG = (-35.0, -55.0, -75.0, -95.0)
FLEX_DEG = (0.0, 30.0, 60.0, 90.0, 110.0)
CLEARANCE_MM = 30.0
APPROACH_DEG = 4.0    # how far past the pose each joint starts its last stretch
WALK_DPS = 30.0
RATE_HZ = 50.0
STILL_DEG = 0.1       # still: no joint moved more than this over STILL_S
STILL_S = 0.3
SETTLE_MAX_S = 3.0
READ_S = 0.5          # readings averaged once still
FOLLOW_MAX_DEG = 2.0  # a joint further off than this once still has not followed
PATH_STEP_DEG = 3.0   # a walk between poses is checked this often
OWN_BASE_M = 0.15     # world-body pieces this near the arm's shoulder are its own base


# --- the model: clearance and gravity ---------------------------------------
class Model:
    def __init__(self, arm: str) -> None:
        import mujoco

        self.mj = mujoco
        self.arm = arm
        self.m = mujoco.MjModel.from_xml_path(str(SETTINGS.scene_file))
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.qadr = {}
        self.dof = {}
        for j in range(m.njnt):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
            if name:
                self.qadr[name] = int(m.jnt_qposadr[j])
                self.dof[name] = int(m.jnt_dofadr[j])
        # The bodies the shoulder lift carries move with the grid; the arm's
        # base and pan bracket sit on the board whatever the pose, so they are
        # not asked for clearance from it.
        lift_body = int(m.jnt_bodyid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{arm}_shoulder_lift")])
        carried = {lift_body}
        for b in range(m.nbody):
            p = b
            while p > 0:
                if p in carried:
                    carried.add(b)
                    break
                p = int(m.body_parentid[p])
        colliding = [g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
        self.moving = [g for g in colliding if int(m.geom_bodyid[g]) in carried]
        # Everything the arm is not: the desk, the board and the other arm. The
        # arms' bases are fused into the world body with the scenery; this
        # arm's own base - the pieces around its shoulder - is the joint
        # bands' to keep clear of, not a distance's.
        mujoco.mj_forward(m, self.d)
        shoulder = self.d.xanchor[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{arm}_shoulder_pan")]
        own = f"{arm}_"
        self.others = []
        for g in colliding:
            body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or "world"
            if body.startswith(own):
                continue
            if int(m.geom_bodyid[g]) == 0 and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH \
                    and np.linalg.norm(self.d.geom_xpos[g] - shoulder) < OWN_BASE_M:
                continue
            self.others.append(g)

    def _set(self, pose: dict[str, float], rest: dict[str, float]) -> None:
        self.d.qpos[:] = 0
        self.d.qvel[:] = 0
        for name, value in {**rest, **pose}.items():
            if name in self.qadr:
                self.d.qpos[self.qadr[name]] = value
        self.mj.mj_forward(self.m, self.d)

    def clearance_mm(self, pose: dict[str, float], rest: dict[str, float]) -> float:
        """The nearest the carried part of the arm comes to anything that is not
        the arm, in mm - negative when it is inside it - or -inf when the arm
        presses into itself."""
        self._set(pose, rest)
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            if c.dist < 0 and int(c.geom1) in self.moving and int(c.geom2) not in self.others \
                    or c.dist < 0 and int(c.geom2) in self.moving and int(c.geom1) not in self.others:
                return -math.inf
        far = 0.5
        nearest = far
        for g1 in self.moving:
            for g2 in self.others:
                nearest = min(nearest, self.mj.mj_geomDistance(self.m, self.d, g1, g2, far, None))
        return nearest * 1000.0

    def gravity(self, pose: dict[str, float], rest: dict[str, float]) -> dict[str, float]:
        """The torque each joint of the arm holds against gravity here, N·m."""
        self._set(pose, rest)
        return {j: float(self.d.qfrc_bias[self.dof[f"{self.arm}_{j}"]]) for j in JOINTS}


def full(arm: str, pose_deg: dict[str, float]) -> dict[str, float]:
    return {f"{arm}_{j}": math.radians(v) for j, v in pose_deg.items()}


def plan(arm: str, model: Model, rest: dict[str, float], count: int) -> list[dict[str, float]]:
    """Up to `count` grid poses, each clear with its approaches, in an order whose
    every walk is clear too, starting from ready - or none when ready is not
    clear."""
    bands = joint_bands(arm)
    ready = dict(READY_DEG[arm])

    def inside(p: dict[str, float]) -> bool:
        return all(bands[j][0] + 1.0 <= v <= bands[j][1] - 1.0 for j, v in p.items())

    def approaches(p: dict[str, float]) -> list[dict[str, float]]:
        return [{j: v + s * APPROACH_DEG for j, v in p.items()} for s in (1.0, -1.0)]

    def clear(p: dict[str, float]) -> bool:
        return model.clearance_mm(full(arm, p), rest) >= CLEARANCE_MM

    grid = []
    for lift, elbow, flex in itertools.product(LIFT_DEG, ELBOW_DEG, FLEX_DEG):
        right = {"shoulder_lift": lift, "elbow_flex": elbow, "wrist_flex": flex}
        bent = mirrored(right) if arm == "left" else right
        grid.append({**ready, **bent})

    def walk_clear(a: dict[str, float], b: dict[str, float]) -> bool:
        n = max(1, math.ceil(max(abs(b[j] - a[j]) for j in JOINTS) / PATH_STEP_DEG))
        return all(clear({j: a[j] + (b[j] - a[j]) * k / n for j in JOINTS}) for k in range(1, n))

    # A pose is used when it, its approaches and the walks between them are
    # clear, and so is the straight walk back to ready a stopped run takes.
    def usable(p: dict[str, float]) -> bool:
        return (all(inside(q) and clear(q) for q in [p, *approaches(p)])
                and all(walk_clear(p, q) for q in approaches(p)) and walk_clear(p, ready))

    # Spread over the loads first - each next pose the one whose gravity
    # torques, each joint's scaled by its span over the grid, are farthest
    # from those picked, since the fit is against torque - then walk them
    # nearest-first so the arm does not cross the grid back and forth.
    def gap(a, b):
        return max(abs(a[j] - b[j]) for j in JOINTS)

    fits = [p for p in grid if usable(p)]
    load = {id(p): model.gravity(full(arm, p), rest) for p in [ready, *fits]}
    span = {j: (max(t[j] for t in load.values()) - min(t[j] for t in load.values())) or 1.0 for j in JOINTS}

    def apart(a, b):
        return math.dist([load[id(a)][j] / span[j] for j in JOINTS], [load[id(b)][j] / span[j] for j in JOINTS])

    # Ready is where the run goes out to and comes back through, so a run
    # without it clear is no run.
    if not usable(ready):
        return []
    picked = [ready]
    pool = [p for p in fits if gap(p, ready) > 1e-6]
    while pool and len(picked) < count:
        best = max(pool, key=lambda p: min((apart(p, q) for q in picked), default=0.0))
        picked.append(best)
        pool.remove(best)
    order, here, left = [picked[0]], picked[0], picked[1:]
    while left:
        # The walk taken is to the pose's approach from above, not to the pose.
        reachable = [p for p in left if walk_clear(here, approaches(p)[0])]
        if not reachable:
            break
        here = min(reachable, key=lambda p: gap(here, p))
        order.append(here)
        left.remove(here)
    return order


# --- the robot, over ROS ------------------------------------------------------
class Robot:
    def __init__(self, arm: str) -> None:
        import rclpy
        from pinkrab_interfaces.msg import TeleopState, TorqueState
        from pinkrab_interfaces.srv import SetTeleopMode, SetTorque
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, QoSProfile
        from rclpy.signals import SignalHandlerOptions
        from sensor_msgs.msg import JointState

        SETTINGS.pin_ros_middleware()
        # Ctrl-C stays a KeyboardInterrupt alone: rclpy's own handler would shut
        # the context down with it, and the walk home after a stop could not
        # publish a command.
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        self.rclpy, self.JointState, self.TeleopState = rclpy, JointState, TeleopState
        self.SetTeleopMode, self.SetTorque = SetTeleopMode, SetTorque
        self.arm = arm
        self.node = Node("pinkrab_pose_error")
        self.commands = self.node.create_publisher(JointState, "joint_commands", 10)
        self.node.create_subscription(JointState, "joint_states", self._on_state, 10)
        self.node.create_subscription(TeleopState, "teleop_state", self._on_teleop, 10)
        self.node.create_subscription(
            TorqueState, "torque_state", self._on_torque,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.held: dict[str, bool] = {}
        self.torque_client = self.node.create_client(SetTorque, "set_torque")
        self.mode_client = self.node.create_client(SetTeleopMode, "set_teleop_mode")
        self.lock = threading.Lock()
        self.angles: dict[str, float] = {}
        self.history: deque = deque(maxlen=600)
        self.teleop = None
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.spinner = threading.Thread(target=self.executor.spin, daemon=True)
        self.spinner.start()

    def _on_state(self, message) -> None:
        pose = dict(zip(message.name, message.position))
        with self.lock:
            self.angles.update(pose)
            self.history.append((time.monotonic(), dict(self.angles)))

    def _on_teleop(self, message) -> None:
        with self.lock:
            self.teleop = message

    def _on_torque(self, message) -> None:
        with self.lock:
            self.held = dict(zip(message.arms, message.enabled))

    def reading(self) -> dict[str, float]:
        with self.lock:
            return {j: self.angles[f"{self.arm}_{j}"] for j in JOINTS if f"{self.arm}_{j}" in self.angles}

    def everything(self) -> dict[str, float]:
        with self.lock:
            return dict(self.angles)

    def wait_ready(self) -> None:
        end = time.monotonic() + 5.0
        while time.monotonic() < end:
            with self.lock:
                have = len(self.reading_names()) == len(JOINTS) and self.teleop is not None
            if have:
                return
            time.sleep(0.1)
        raise SystemExit("no joint_states for this arm or no teleop_state - is robot.launch.py up?")

    def reading_names(self) -> list[str]:
        return [j for j in JOINTS if f"{self.arm}_{j}" in self.angles]

    def others_commanding(self) -> int:
        return self.node.count_publishers("joint_commands") - 1

    def call(self, client, request, what: str):
        if not client.wait_for_service(timeout_sec=3.0):
            raise SystemExit(f"{what}: the service is not there")
        future = client.call_async(request)
        end = time.monotonic() + 5.0
        while not future.done() and time.monotonic() < end:
            time.sleep(0.02)
        if not future.done():
            raise SystemExit(f"{what}: no answer")
        return future.result()

    def torque_on(self) -> None:
        r = self.SetTorque.Request()
        r.arm, r.enable = self.arm, True
        # Not successful when the torque was on already - nothing changed then -
        # so what the driver says it holds is what is believed.
        answer = self.call(self.torque_client, r, "set_torque")
        end = time.monotonic() + 2.0
        while time.monotonic() < end:
            with self.lock:
                if self.held.get(self.arm):
                    return
            time.sleep(0.05)
        raise SystemExit(f"the {self.arm} arm's torque did not come on: {answer.message}")

    def mode(self, mode: int) -> None:
        r = self.SetTeleopMode.Request()
        r.mode = mode
        answer = self.call(self.mode_client, r, "set_teleop_mode")
        if not answer.success:
            raise SystemExit(f"set_teleop_mode: {answer.message}")

    def check_gate(self) -> None:
        with self.lock:
            t = self.teleop
        if t is None or not t.armed or t.mode != self.TeleopState.MODE_STUDIO:
            raise Abort(f"the gate left studio mode ({t.gate_reason if t else 'no state'})")

    def send(self, pose_rad: dict[str, float]) -> None:
        message = self.JointState()
        message.header.stamp = self.node.get_clock().now().to_msg()
        message.name = [f"{self.arm}_{j}" for j in pose_rad]
        message.position = [float(v) for v in pose_rad.values()]
        self.commands.publish(message)

    def walk(self, goal_deg: dict[str, float]) -> None:
        """To a pose at WALK_DPS, eased, then the pose itself."""
        start = {j: math.degrees(v) for j, v in self.reading().items()}
        span = max(abs(goal_deg[j] - start[j]) for j in JOINTS)
        steps = max(1, math.ceil(span / WALK_DPS * RATE_HZ))
        for k in range(1, steps + 1):
            s = k / steps
            s = s * s * (3 - 2 * s)
            self.send({j: math.radians(start[j] + (goal_deg[j] - start[j]) * s) for j in JOINTS})
            self.check_gate()
            time.sleep(1.0 / RATE_HZ)

    def settle(self, goal_deg: dict[str, float]) -> tuple[float | None, float]:
        """Keep asking for the pose until the arm is still, and make sure it got
        there. Returns when it first had every joint within a degree, and when
        it was still, in seconds."""
        goal = {j: math.radians(v) for j, v in goal_deg.items()}
        t0 = time.monotonic()
        within = None
        while True:
            self.send(goal)
            self.check_gate()
            time.sleep(1.0 / RATE_HZ)
            now = time.monotonic()
            here = self.reading()
            if within is None and all(abs(math.degrees(here[j] - goal[j])) <= 1.0 for j in JOINTS):
                within = now - t0
            with self.lock:
                recent = [a for t, a in self.history if t >= now - STILL_S]
            moved = max((abs(math.degrees(max(a[f"{self.arm}_{j}"] for a in recent)
                                          - min(a[f"{self.arm}_{j}"] for a in recent)))
                         for j in JOINTS), default=math.inf) if len(recent) > 3 else math.inf
            if (now - t0 >= STILL_S and moved <= STILL_DEG) or now - t0 >= SETTLE_MAX_S:
                break
        here = self.reading()
        off = {j: round(math.degrees(here[j]) - goal_deg[j], 1) for j in JOINTS
               if abs(math.degrees(here[j]) - goal_deg[j]) > FOLLOW_MAX_DEG}
        if off:
            raise Abort(f"the arm did not follow - off by {off} deg")
        return within, time.monotonic() - t0

    def measure(self, goal_deg: dict[str, float]) -> dict:
        """Settle on the pose, then read it."""
        within, settled = self.settle(goal_deg)
        goal = {j: math.radians(v) for j, v in goal_deg.items()}
        samples = []
        end = time.monotonic() + READ_S
        while time.monotonic() < end:
            self.send(goal)
            self.check_gate()
            time.sleep(1.0 / RATE_HZ)
            samples.append(self.reading())
        mean = {j: float(np.mean([math.degrees(s[j]) for s in samples])) for j in JOINTS}
        return {
            "error_deg": {j: round(mean[j] - goal_deg[j], 3) for j in JOINTS},
            "noise_deg": {j: round(float(np.std([math.degrees(s[j]) for s in samples])), 3) for j in JOINTS},
            "within_1deg_s": None if within is None else round(within, 2),
            "settled_s": round(settled, 2),
        }

    def driving(self) -> bool:
        """Whether the gate is still in the studio mode this run asked for."""
        with self.lock:
            t = self.teleop
        return t is not None and t.armed and t.mode == self.TeleopState.MODE_STUDIO

    def close(self) -> None:
        self.executor.shutdown()
        self.node.destroy_node()
        self.rclpy.try_shutdown()


class Abort(RuntimeError):
    pass


# --- the fit -------------------------------------------------------------------
def fit(records: list[dict]) -> dict:
    out = {}
    for j in JOINTS:
        tau = np.array([r["gravity_nm"][j] for r in records])
        up = np.array([r["from_above"]["error_deg"][j] for r in records])
        down = np.array([r["from_below"]["error_deg"][j] for r in records])
        mid = (up + down) / 2
        row = {
            "max_abs_deg": round(float(np.max(np.abs(np.concatenate([up, down])))), 2),
            "rms_deg": round(float(np.sqrt(np.mean(np.concatenate([up, down]) ** 2))), 2),
            "hysteresis_deg": round(float(np.mean(np.abs(up - down))), 2),
            "torque_span_nm": round(float(tau.max() - tau.min()), 3),
        }
        if len(records) >= 3 and tau.max() - tau.min() > 1e-3:
            a = np.vstack([np.ones_like(tau), tau]).T
            (offset, compliance), *_ = np.linalg.lstsq(a, mid, rcond=None)
            predicted = a @ np.array([offset, compliance])
            total = float(np.sum((mid - mid.mean()) ** 2))
            row.update({
                "offset_deg": round(float(offset), 3),
                "compliance_deg_per_nm": round(float(compliance), 3),
                "r2": round(1 - float(np.sum((mid - predicted) ** 2)) / total, 2) if total > 1e-12 else None,
                "residual_rms_deg": round(float(np.sqrt(np.mean((mid - predicted) ** 2))), 2),
            })
        out[j] = row
    return out


def print_fit(result: dict) -> None:
    head = f"{'joint':14s} {'max':>6s} {'rms':>6s} {'hyst':>6s} {'offset':>7s} {'deg/Nm':>7s} {'R2':>5s} {'resid':>6s}"
    print(head)
    for j, r in result.items():
        def f(k, w, fmt, r=r):
            return (format(r[k], fmt) if r.get(k) is not None else "-").rjust(w)
        print(f"{j:14s} {f('max_abs_deg', 6, '.2f')} {f('rms_deg', 6, '.2f')} {f('hysteresis_deg', 6, '.2f')} "
              f"{f('offset_deg', 7, '.2f')} {f('compliance_deg_per_nm', 7, '.2f')} {f('r2', 5, '.2f')} "
              f"{f('residual_rms_deg', 6, '.2f')}")


# --- the run -------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("arm", choices=("left", "right"))
    p.add_argument("--poses", type=int, default=24, help="how many poses, ready included")
    p.add_argument("--dry", action="store_true", help="plan and print the poses, move nothing")
    args = p.parse_args()

    model = Model(args.arm)
    robot = None if args.dry else Robot(args.arm)
    if robot is not None:
        robot.wait_ready()
    rest = robot.everything() if robot else {}
    poses = plan(args.arm, model, rest, args.poses)
    if not poses:
        raise SystemExit("ready is not clear in the model - nothing moves")
    if len(poses) < 3:
        raise SystemExit(f"only {len(poses)} pose(s) clear in the model, too few to fit - nothing moves")
    print(f"{len(poses)} poses, {CLEARANCE_MM:g} mm clear:")
    for k, pose in enumerate(poses):
        clear = model.clearance_mm(full(args.arm, pose), rest)
        bent = "  ".join(f"{j} {pose[j]:6.1f}" for j in ("shoulder_lift", "elbow_flex", "wrist_flex"))
        print(f"  {k:2d}  {bent}   clear {clear:5.0f} mm")
    if args.dry:
        return

    if robot.others_commanding() > 0:
        raise SystemExit("something else publishes /joint_commands - disconnect the studio first")
    with robot.lock:
        mode = robot.teleop.mode
    if mode != robot.TeleopState.MODE_IDLE:
        raise SystemExit("the gate is not idle - something is driving the arms")

    records = []
    started = time.strftime("%Y%m%d-%H%M%S")
    ready = poses[0]
    out_at_ready = False
    try:
        # The mode first: a gate with an /estop latched refuses it before any
        # torque has been switched on.
        robot.mode(robot.TeleopState.MODE_STUDIO)
        robot.torque_on()
        time.sleep(0.3)
        here = robot.everything()
        try:
            robot.walk(ready)
            robot.settle(ready)
            out_at_ready = True
            t_run = time.monotonic()
            for k, pose in enumerate(poses):
                record = {"pose_deg": pose, "gravity_nm": model.gravity(full(args.arm, pose), here)}
                for name, side in (("from_above", 1.0), ("from_below", -1.0)):
                    # Settled on its side first, which is also where a refused
                    # approach shows: the arm is not there.
                    approach = {j: v + side * APPROACH_DEG for j, v in pose.items()}
                    robot.walk(approach)
                    robot.settle(approach)
                    robot.walk(pose)
                    record[name] = robot.measure(pose)
                records.append(record)
                worst = max(abs(v) for side in ("from_above", "from_below")
                            for v in record[side]["error_deg"].values())
                print(f"  {k + 1:2d}/{len(poses)}  worst {worst:.2f} deg  "
                      f"({time.monotonic() - t_run:.0f} s)")
        except (Abort, KeyboardInterrupt) as exc:
            print(f"stopped: {str(exc) or 'interrupted'}")
        # Written before the walk home, which can itself be cut short.
        if records:
            write(args.arm, started, records)
        try:
            if robot.driving():
                home = {j: 0.0 for j in JOINTS}
                if out_at_ready:
                    # Every pose of the run has a clear straight walk back to ready,
                    # and from ready home is the home button's way.
                    robot.walk(ready)
                    robot.settle(ready)
                robot.walk(home)
                robot.settle(home)
        except (Abort, KeyboardInterrupt) as exc:
            print(f"the walk home stopped: {str(exc) or 'interrupted'} - the arm holds where it is")
    finally:
        # Idle is asked for only while the mode is still this run's: an /estop
        # is left latched for whoever pressed it.
        if robot.driving():
            robot.mode(robot.TeleopState.MODE_IDLE)
        robot.close()


def write(arm: str, started: str, records: list[dict]) -> None:
    result = fit(records)
    print_fit(result)
    out = {"arm": arm, "started": started, "approach_deg": APPROACH_DEG,
           "gains": {"p": SETTINGS.servo_p_coefficient, "i": SETTINGS.servo_i_coefficient,
                     "d": SETTINGS.servo_d_coefficient},
           "records": records, "fit": result}
    path = SETTINGS.workspace / "pose_error" / f"{arm}_{started}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print("written", path)

if __name__ == "__main__":
    main()
