"""A trained policy driving the simulated arm, or both arms, and whether what each carried ended up where it goes.

The arm is driven by the rollout's own `Driver` - the checkpoint's pre- and
post-processors, the camera-local delta composed onto where the tool was put,
`track()`, the step guard and the stop when the arm cannot follow - so what is
measured here is the code that drives the metal. Only the picture and the
servos are MuJoCo's: the gripper camera is the fisheye drawn from the arm's own
camera, and the joints are position servos walked toward what is asked at the
speed the gate walks a policy's commands at.

Each episode lays the blocks out as the demonstrations did, puts the arm
where the takes began, and lets the policy run. Three things are counted:
the cap picked, lifted clear of the desk, and on the green piece at the end.
With both hands (`bowls`) each working arm's block is counted the same way,
and an episode is placed when every working arm's block is in its bowl; an
arm at rest is watched for how far it wanders. The arms start where the rig
starts them, from the takes' desk solid at the chest.

The grasp is judged two ways. MuJoCo's contact alone cannot hold this cap: a
pyramid shut in a jaw that swings about a hinge is squeezed out of it, even
by the arm replaying a take exactly (`oracle`), so contact alone mostly
measures the simulation. By default the cap is therefore carried once the
jaw has shut on it where the takes held it - between the claw and the jaw,
against the claw - and let go when the jaw opens again; what is measured is
whether the policy went there, shut, carried and let go where it should.
`physics=True` leaves it all to contact.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np

from pinkrab_core import inference
from pinkrab_core.config import SETTINGS as CORE
from pinkrab_core.dataset import gripper_view, image_key
from pinkrab_core.demo_start import DemoStartError, dataset_of_checkpoint, start_joints

from .config import SETTINGS
from .demos import DemoError, Planner, Trajectory, placed, two_hand_take
from .demos import faces as block_faces
from .world import CARRIED_BIT, Layout, World, desk_pose, hands_of

LIFTED_M = 0.02
# The cap is taken as held when its grasp point is this near where the takes
# held it, in the TCP's frame, and the jaw has shut past SHUT; it is let go
# once the jaw opens past OPEN. A block bound for a bowl, met wherever its
# shape has the jaw meet it, is let go once the jaw opens LET_GO past where it
# shut.
HOLD_BOX_M = (0.008, 0.012, 0.010)
SHUT = 0.32
OPEN = 0.40
LET_GO = 0.06
# A thing counts as put down once it is let go and has come to rest: it moves
# slower than this after the arms have had half a second to settle.
SETTLED_MPS = 0.01


def physics_steps(k: int, rate: float, timestep: float) -> int:
    """How many physics steps tick `k` of a clock at `rate` runs: the ticks
    together span exactly 1/rate seconds each, whatever `rate` is, however
    many steps of the model's timestep a tick comes to."""
    return round((k + 1) / (rate * timestep)) - round(k / (rate * timestep))


class Carry:
    """What the jaw has shut on, welded to the claw, and let go when it opens.

    The cap is taken as held when its grasp point is where the takes held it
    and the jaw has shut past SHUT; a block bound for a bowl when the claw and
    the jaw both touch it.
    """

    def __init__(self, world: World, arm: str) -> None:
        from pinkrab_core.jaw import shut_wide

        from .world import quat_of

        self.world = world
        self.arm = arm
        # Held by touch on both sides: a block the jaw meets wherever its
        # shape has it meet - not the cap, which only a weld keeps in a jaw.
        self.by_touch = world.scene == "bowls"
        m = world.model
        self.band = shut_wide(*(float(v) for v in m.jnt_range[m.joint(f"{arm}_gripper").id]))
        self.quat_of = quat_of
        names = ([n for n, t in world.things.items() if t.kind == "block"] if self.by_touch
                 else [SETTINGS.carried_block])
        self.eqs = {n: m.equality(f"carry_{arm}_{n}").id for n in names}
        self.bits = {g: (int(m.geom_contype[g]), int(m.geom_conaffinity[g]))
                     for n in names for g in world.things[n].geoms}
        cap = world.things[SETTINGS.carried_block] if not self.by_touch else None
        self.point = None if cap is None else np.array([0.0, 0.0, SETTINGS.grasp_height * cap.height, 1.0])
        self.want = np.array(SETTINGS.grasp_in_tcp)
        self._claw = {g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(f"{arm}_gripper_link").id}
        self._jaw = {g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(f"{arm}_jaw_link").id}
        self.held: str | None = None
        self.first: str | None = None
        self._shut_at = 0.0
        self._jaw_q = m.jnt_qposadr[m.joint(f"{arm}_gripper").id]
        self.reset()

    @property
    def ever(self) -> bool:
        return self.first is not None

    def aperture(self) -> float:
        shut, wide = self.band
        return (float(self.world.data.qpos[self._jaw_q]) - shut) / (wide - shut)

    def _shut_on(self, opening: float) -> str | None:
        w = self.world
        if not self.by_touch:
            if opening > SHUT:
                return None
            at = (np.linalg.inv(w.tcp(self.arm)) @ w.block_pose(SETTINGS.carried_block) @ self.point)[:3]
            return SETTINGS.carried_block if np.all(np.abs(at - self.want) < HOLD_BOX_M) else None
        touched: dict[str, set[str]] = {}
        d = w.data
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist >= 0:
                continue
            for mine, other in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                side = "claw" if other in self._claw else "jaw" if other in self._jaw else None
                if side is None:
                    continue
                for name in self.eqs:
                    if mine in w.things[name].geoms:
                        touched.setdefault(name, set()).add(side)
        return next((n for n, sides in touched.items() if sides == {"claw", "jaw"}), None)

    def update(self) -> None:
        w = self.world
        opening = self.aperture()
        if self.held is None:
            name = self._shut_on(opening)
            if name is None:
                return
            rel = np.linalg.inv(w.body_pose(f"{self.arm}_gripper_link")) @ w.block_pose(name)
            eq = self.eqs[name]
            data = w.model.eq_data[eq]
            data[0:3] = 0.0
            data[3:6] = rel[:3, 3]
            data[6:10] = self.quat_of(rel[:3, :3])
            data[10] = 1.0
            w.data.eq_active[eq] = 1
            for g in w.things[name].geoms:
                w.model.geom_contype[g] = w.model.geom_conaffinity[g] = CARRIED_BIT
            self.held = name
            self.first = self.first or name
            self._shut_at = opening
        elif opening > (self._shut_at + LET_GO if self.by_touch else OPEN):
            self._let_go()

    def _restore(self, geoms) -> None:
        m = self.world.model
        for g in geoms:
            m.geom_contype[g], m.geom_conaffinity[g] = self.bits[g]

    def _let_go(self) -> None:
        # Only what this arm holds: another arm's weld keeps its thing out of
        # the claw's collisions.
        if self.held is not None:
            self.world.data.eq_active[self.eqs[self.held]] = 0
            self._restore(self.world.things[self.held].geoms)
        self.held = None

    def reset(self) -> None:
        for eq in self.eqs.values():
            self.world.data.eq_active[eq] = 0
        self._restore(self.bits)
        self.held = None
        self.first = None


class EvaluateError(RuntimeError):
    pass


def recorded_starts(dataset_root: Path) -> list[np.ndarray]:
    """The first tool pose of every demonstration in the dataset, in the
    robot's frame. Corrections are left out: they begin wherever a policy went
    wrong, and their pose is not the desk's."""
    import pyarrow.parquet as pq
    from scipy.spatial.transform import Rotation

    from pinkrab_core.dataset import pose_key
    from pinkrab_core.demo_start import correction_episodes

    key = pose_key(SETTINGS.arm)
    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    names = list(info["features"][key]["names"])
    at = {n: i for i, n in enumerate(names)}
    corrections = correction_episodes(dataset_root)
    firsts: dict[int, list[float]] = {}
    for file in sorted(dataset_root.glob("data/*/*.parquet")):
        table = pq.read_table(file, columns=["episode_index", key])
        for e, p in zip(table.column("episode_index").to_pylist(), table.column(key).to_pylist()):
            if int(e) not in corrections:
                firsts.setdefault(int(e), p)
    desk = desk_pose()
    out = []
    for p in firsts.values():
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat([p[at[n]] for n in ("qx", "qy", "qz", "qw")]).as_matrix()
        T[:3, 3] = [p[at[n]] for n in ("x", "y", "z")]
        out.append(desk @ T)
    return out


# A grasp this far, or further, from square to any face of the cap was made
# between two faces.
BETWEEN_FACES_DEG = 25.0


def heading_of(T: np.ndarray) -> float:
    """Which way a tool points round the vertical: its axis, seen from above."""
    return math.atan2(T[1, 2], T[0, 2])


def face_taken(heading: float, block_yaw: float, nearest: int, second: int) -> tuple[str, float]:
    """Which face of the cap a tool heading meets - "nearest" to the hand's
    line, the "second" nearest, "between" two, or an "other" - and how far
    off square to it, in degrees."""
    turns = (heading - block_yaw) / (math.pi / 2)
    k = round(turns)
    off = math.degrees((turns - k) * math.pi / 2)
    if abs(off) >= BETWEEN_FACES_DEG:
        return "between", off
    face = "nearest" if (k - nearest) % 4 == 0 else "second" if (k - second) % 4 == 0 else "other"
    return face, off


class Evaluation:
    def __init__(self, checkpoint: Path, device: str | None = None, start: str = "demo",
                 rate: float | None = None, seconds: float = 12.0, physics: bool = False,
                 scene: str = "blocks") -> None:
        from pinkrab_policy.rollout import Driver
        from pinkrab_policy.rollout import carry as rig_carry

        self.hands = hands_of(scene)
        self.checkpoint = Path(checkpoint)
        self.scene = scene
        self.world = World(arms=self.hands, hermits=(), scene=scene)
        self.planners = {h: Planner(self.world, arm=h) for h in self.hands}
        self.planner = self.planners[self.hands[0]]
        self.driver = Driver(str(checkpoint), device)
        if self.driver.hands != self.hands:
            raise EvaluateError(f"the checkpoint drives the {' and '.join(self.driver.hands)} arm(s); "
                                f"the {scene} scene runs the {' and '.join(self.hands)}")
        self.chains = {h: self.driver.arms[h].chain for h in self.hands}
        # A step a tick of the simulation's clock: the checkpoint's dataset's
        # rate unless another is asked.
        self.rate = rate or self.driver.brain.fps
        if self.rate * self.world.model.opt.timestep > 1.0:
            raise EvaluateError(f"{self.rate} steps a second is faster than the model's "
                                f"{1.0 / self.world.model.opt.timestep:.0f} physics steps a second")
        self.seconds = seconds
        self.start = start
        self.physics = physics
        self.dataset = dataset_of_checkpoint(self.checkpoint)
        if start == "demo":
            try:
                # Both hands' takes were made against a solid where the rig
                # reckons one at the chest, and start where the rig starts
                # them; one hand's against the simulation's own desk solid.
                carry = rig_carry if scene == "bowls" else (lambda desk: desk_pose())
                self._demo_joints, self._demo_info = start_joints(self.dataset, self.chains, carry)
            except DemoStartError as exc:
                raise EvaluateError(str(exc)) from exc
        elif start == "episode":
            if len(self.hands) > 1:
                raise EvaluateError("both hands start where the rig starts them (--start demo)")
            self._starts = recorded_starts(self.dataset)
        else:
            raise EvaluateError(f"start is demo or episode, not {start!r}")
        m = self.world.model
        self._claw = {h: {i for i in range(m.ngeom) if m.geom_bodyid[i] == m.body(f"{h}_gripper_link").id}
                      for h in self.hands}
        self._jaw = {h: {i for i in range(m.ngeom) if m.geom_bodyid[i] == m.body(f"{h}_jaw_link").id}
                     for h in self.hands}
        self._all = [f"{a}_{j}" for a in ("left", "right") for j in CORE.joint_names]

    def close(self) -> None:
        self.world.close()

    def _start_joints(self, rng: np.random.Generator) -> dict[str, np.ndarray]:
        if self.start == "demo":
            return {h: np.array(q, dtype=float) for h, q in self._demo_joints.items()}
        # A recorded start is the tracker's pose of a hand at rest, noise and
        # all; now and then one lands just past what the solver takes. Another
        # take's start stands in for it.
        for k in rng.permutation(len(self._starts)):
            q = self.planner.reach(self._starts[int(k)])
            if q is not None:
                return {self.planner.arm: q}
        raise EvaluateError("no recorded start the arm can reach")

    def _held(self, hand: str, thing: str) -> bool:
        d = self.world.data
        mine = set(self.world.things[thing].geoms)
        claw = jaw = False
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist >= 0 or not ({c.geom1, c.geom2} & mine):
                continue
            other = c.geom2 if c.geom1 in mine else c.geom1
            claw |= other in self._claw[hand]
            jaw |= other in self._jaw[hand]
        return claw and jaw

    def layout(self, rng: np.random.Generator) -> Layout:
        """A layout the takes could have been made on: one a demonstration was
        planned for, whose every frame the arm could follow - both arms, and
        clear of each other, for two hands. Laid out from the whole desk
        instead, a share of the episodes would ask for what no take showed and
        no arm reaches."""
        for _ in range(500):
            try:
                if len(self.hands) == 1:
                    plan = self.planner.plan(rng)
                    self.planner.walk(Trajectory(plan, None), SETTINGS.fps)
                    return plan.layout
                return two_hand_take(self.world, self.planners, rng, tremor=False)[0]
            except DemoError:
                continue
        raise EvaluateError("no layout a take could have been made on")

    def episode(self, rng: np.random.Generator, video: Path | None = None) -> dict:
        w = self.world
        layout = self.layout(rng)
        # The arms at work and what each carries; with both hands, an arm the
        # layout gives nothing rests.
        mine = {h: layout.of(h) for h in layout.hands} if layout.hands else {self.hands[0]: layout}
        working = tuple(mine)
        resting = tuple(h for h in self.hands if h not in mine)
        w.data.qvel[:] = 0
        w.lay_out(layout)
        joints = self._start_joints(rng)
        opening = self.planner.open
        pose = {n: 0.0 for n in self._all}
        for h in self.hands:
            pose.update({n: math.radians(v) for n, v in zip(self.chains[h].joint_names, joints[h])})
            band = w.bands[h]
            pose[f"{h}_gripper"] = band[0] + opening * (band[1] - band[0])
        w.set_joints(pose)
        w.step(0.3)
        self.driver.start(joints)
        target = dict(pose)
        thing0 = {h: w.block_pose(mine[h].carried)[2, 3] for h in working}
        rest0 = {h: w.tcp(h)[:3, 3].copy() for h in resting}
        wander = {h: 0.0 for h in resting}
        # The gate's speed under a policy.
        ceiling = math.radians(inference.policy_dps(CORE.max_joint_speed_dps)) * w.model.opt.timestep
        from pinkrab_policy.rollout import DriveStopped

        held = {h: False for h in working}
        lifted = {h: False for h in working}
        first_held: dict[str, float | None] = {h: None for h in working}
        highest = {h: 0.0 for h in working}
        held_steps = {h: 0 for h in working}
        # Which face the cap is taken by, for one hand on the blocks: the
        # faces a take would have used, from where the hand comes; the heading
        # read at the grasp, or where the tool came nearest the cap when there
        # was none.
        faces = None
        arm = working[0]
        if self.scene == "blocks":
            cx, cy, cyaw = layout.things[layout.carried]
            faces = block_faces(cyaw, cx, cy, self.planner.origin)
        nearest_mm, approach = float("inf"), None
        stopped = None
        writer = None
        carries = {h: (None if self.physics else Carry(w, h)) for h in working}
        lens = SETTINGS.lens_spec
        centre = (lens.cx / lens.width, lens.cy / lens.height)
        steps = int(self.seconds * self.rate)
        try:
            for k in range(steps):
                pictures = {image_key(h): gripper_view(w.gripper_image(h), self.driver.sizes[image_key(h)], centre)
                            for h in self.hands}
                grippers = {h: float(w.data.qpos[w.model.jnt_qposadr[w.model.joint(f"{h}_gripper").id]])
                            for h in self.hands}
                try:
                    wanted, info = self.driver.step(pictures, grippers)
                except DriveStopped as exc:
                    # As on the metal: the run ends there, and is scored as it stands.
                    stopped = str(exc)
                    break
                target.update(wanted)
                ids = [w.model.actuator(n).id for n in target]
                goal = np.array([target[n] for n in target])
                for _ in range(physics_steps(k, self.rate, w.model.opt.timestep)):
                    now = w.data.ctrl[ids]
                    w.data.ctrl[ids] = now + np.clip(goal - now, -ceiling, ceiling)
                    for carry in carries.values():
                        if carry is not None:
                            carry.update()
                    mujoco.mj_step(w.model, w.data)
                if faces is not None and first_held[arm] is None:
                    tcp, cap = w.site_pose(f"{arm}_tcp"), w.block_pose(layout.carried)
                    gap = float(np.linalg.norm(tcp[:3, 3] - cap[:3, 3])) * 1000
                    if gap < nearest_mm:
                        nearest_mm, approach = gap, (heading_of(tcp), math.atan2(cap[1, 0], cap[0, 0]))
                for h in resting:
                    wander[h] = max(wander[h], float(np.linalg.norm(w.tcp(h)[:3, 3] - rest0[h])))
                for h in working:
                    thing = mine[h].carried
                    carry = carries[h]
                    if self._held(h, thing) or (carry is not None and carry.held == thing):
                        held_steps[h] += 1
                        if first_held[h] is None:
                            first_held[h] = k / self.rate
                            if faces is not None and h == arm:
                                tcp, cap = w.site_pose(f"{arm}_tcp"), w.block_pose(thing)
                                approach = (heading_of(tcp), math.atan2(cap[1, 0], cap[0, 0]))
                        held[h] = True
                    rise = w.block_pose(thing)[2, 3] - thing0[h]
                    highest[h] = max(highest[h], rise)
                    lifted[h] |= bool(rise > LIFTED_M)
                if video is not None:
                    frame = self._frame([pictures[image_key(h)] for h in self.hands])
                    if writer is None:
                        video.parent.mkdir(parents=True, exist_ok=True)
                        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), self.rate,
                                                 (frame.shape[1], frame.shape[0]))
                    writer.write(frame)
        finally:
            if writer is not None:
                writer.release()
        w.step(0.5)
        outcomes = {h: placed(w, mine[h]) for h in working}
        # Put down is let go and at rest, not held in the air over the target.
        for h in working:
            carry = carries[h]
            let_go = (not self._held(h, mine[h].carried)) if carry is None else carry.held is None
            outcomes[h]["released"] = let_go
            outcomes[h]["placed"] = bool(outcomes[h]["placed"] and let_go
                                         and w.speed(mine[h].carried) < SETTLED_MPS)
        for carry in carries.values():
            if carry is not None:
                # A thing still welded at the end would stay out of the claw's
                # collisions, and the next episode's layout would be planned
                # around a thing the jaw cannot meet.
                carry.reset()
        per = {h: {"held": bool(held[h]), "lifted": bool(lifted[h]), "placed": bool(outcomes[h]["placed"]),
                   "held_s": first_held[h], "highest_mm": round(float(highest[h]) * 1000, 1),
                   "held_frames": held_steps[h], "held_steps_refused": self.driver.arms[h].held,
                   **{k: v for k, v in outcomes[h].items() if k != "placed"}}
               for h in working}
        per.update({h: {"resting": True, "wander_mm": round(wander[h] * 1000, 1)} for h in resting})
        result = {"held": all(per[h]["held"] for h in working), "lifted": all(per[h]["lifted"] for h in working),
                  "placed": all(per[h]["placed"] for h in working), "stopped": stopped,
                  "layout": {k: [round(x, 4) for x in v] for k, v in layout.things.items()}}
        if len(self.hands) > 1:
            return {**result, "hands": per, "off_mm": max(per[h]["off_mm"] for h in working),
                    "highest_mm": min(per[h]["highest_mm"] for h in working)}
        one = per[arm]
        return {**result, **{k: v for k, v in one.items() if k not in ("held", "lifted", "placed")},
                "carried": carries[arm] is not None and carries[arm].ever,
                **({"face": face_taken(*approach, *faces)[0],
                    "face_off_deg": round(face_taken(*approach, *faces)[1], 1),
                    "face_at": "grasp" if first_held[arm] is not None else f"nearest {nearest_mm:.0f} mm"}
                   if faces is not None and approach is not None else {})}

    def _frame(self, grippers_rgb: list[np.ndarray]) -> np.ndarray:
        side = self.world.view("side", (480, 360))
        small = [cv2.resize(g, (576, 360)) for g in grippers_rgb]
        return cv2.cvtColor(np.concatenate([side, *small], axis=1), cv2.COLOR_RGB2BGR)

    def run(self, episodes: int, seed: int = 0, videos: Path | None = None, say=print) -> dict:
        rng = np.random.default_rng(seed)
        results = []
        started = time.monotonic()
        for i in range(episodes):
            for h in self.hands:
                self.driver.arms[h].held = 0
            result = self.episode(rng, None if videos is None else videos / f"{i:03d}.mp4")
            results.append(result)
            if len(self.hands) > 1:
                say(f"  {i:3d}: placed {'yes' if result['placed'] else 'no '}  " + "  ".join(
                    f"{h}: at rest, wandered {r['wander_mm']:.0f} mm" if r.get("resting") else
                    f"{h}: held {'yes' if r['held'] else 'no '} lifted {'yes' if r['lifted'] else 'no '} "
                    f"placed {'yes' if r['placed'] else 'no '} {r['off_mm']:.0f} mm off"
                    for h, r in result["hands"].items()))
                continue
            say(f"  {i:3d}: held {'yes' if result['held'] else 'no '}  lifted {'yes' if result['lifted'] else 'no '}  "
                f"placed {'yes' if result['placed'] else 'no '}  {result['off_mm']:.0f} mm off, "
                f"highest {result['highest_mm']:.0f} mm"
                + (f"  face {result['face']} ({result['face_off_deg']:+.0f} deg, {result['face_at']})"
                   if "face" in result else ""))
        n = max(1, len(results))
        summary = {
            "checkpoint": str(self.checkpoint), "dataset": str(self.dataset), "start": self.start,
            "scene": self.scene,
            "grasp": "physics" if self.physics else "carried once shut on it",
            "episodes": len(results), "seed": seed,
            "held": sum(r["held"] for r in results) / n,
            "lifted": sum(r["lifted"] for r in results) / n,
            "placed": sum(r["placed"] for r in results) / n,
            "seconds": round(time.monotonic() - started, 1),
            "results": results,
        }
        if len(self.hands) > 1:
            # Each hand on its own, beside the episodes both made.
            summary["hands"] = {
                h: ({"resting": True, "wander_mm": max(r["hands"][h]["wander_mm"] for r in results)}
                    if results and results[0]["hands"][h].get("resting") else
                    {k: sum(r["hands"][h][k] for r in results) / n for k in ("held", "lifted", "placed")})
                for h in self.hands}
        if any("face" in r for r in results):
            # The faces it took the cap by, of the grasps, and of every episode
            # by where it came nearest: one face only, both, or between them.
            names = ("nearest", "second", "between", "other")
            summary["faces_held"] = {f: sum(r.get("face") == f for r in results if r["held"]) for f in names}
            summary["faces_all"] = {f: sum(r.get("face") == f for r in results) for f in names}
        return summary


def oracle(episodes: int = 20, seed: int = 2000, videos: Path | None = None, say=print,
           physics: bool = False, scene: str = "blocks") -> dict:
    """The arm, or both arms, made to follow planned takes exactly, with no
    policy: can the simulation's grasp do what the takes ask?

    Each take is one the demonstrations could have been: its tool pose frame
    by frame through `track()`, its opening across the arm's band, the joints
    walked at the speed the gate walks a policy at. What is counted is what
    `evaluate` counts. A policy can do no better than this, so a policy's
    failures are its own only below it.
    """
    hands = hands_of(scene)
    world = World(arms=hands, hermits=(), scene=scene)
    planners = {h: Planner(world, arm=h) for h in hands}
    names = [f"{a}_{j}" for a in ("left", "right") for j in CORE.joint_names]
    m = world.model
    claw = {h: {i for i in range(m.ngeom) if m.geom_bodyid[i] == m.body(f"{h}_gripper_link").id} for h in hands}
    jaw = {h: {i for i in range(m.ngeom) if m.geom_bodyid[i] == m.body(f"{h}_jaw_link").id} for h in hands}
    rng = np.random.default_rng(seed)
    results = []
    # The speed a policy is walked at, so this stays the ceiling of one.
    ceiling = math.radians(inference.policy_dps(CORE.max_joint_speed_dps)) * m.opt.timestep
    started = time.monotonic()
    try:
        for i in range(episodes):
            for _ in range(500):
                try:
                    if len(hands) == 1:
                        plans = {hands[0]: planners[hands[0]].plan(rng)}
                        trajectories = {hands[0]: Trajectory(plans[hands[0]], None)}
                        planners[hands[0]].walk(trajectories[hands[0]], SETTINGS.fps)
                    else:
                        _, plans, trajectories, _ = two_hand_take(world, planners, rng, tremor=False)
                    seconds = max(p.seconds for p in plans.values() if not p.info.get("rest"))
                    break
                except DemoError:
                    continue
            else:
                raise EvaluateError("no take to follow")
            world.data.qvel[:] = 0
            world.lay_out(plans[hands[0]].layout)
            pose = {n: 0.0 for n in names}
            q = {}
            for h in hands:
                T0, a0 = trajectories[h].at(0.0)
                q[h] = planners[h].reach(T0)
                pose.update({n: math.radians(v) for n, v in zip(planners[h].chain.joint_names, q[h])})
                band = world.bands[h]
                pose[f"{h}_gripper"] = band[0] + a0 * (band[1] - band[0])
            world.set_joints(pose)
            world.step(0.3)
            working = [h for h in hands if not plans[h].info.get("rest")]
            carried = {h: plans[h].layout.carried for h in working}
            thing_geoms = {h: set(world.things[carried[h]].geoms) for h in working}
            thing0 = {h: world.block_pose(carried[h])[2, 3] for h in working}
            held = {h: False for h in working}
            lifted = {h: False for h in working}
            writer = None
            carries = {h: (None if physics else Carry(world, h)) for h in working}
            n = int(math.floor(seconds * SETTINGS.fps)) + 1
            ids = [m.actuator(nm).id for nm in pose]
            try:
                for k in range(1, n):
                    for h in hands:
                        T, aperture = trajectories[h].at(k / SETTINGS.fps)
                        q[h], _, _ = planners[h].track(T, q[h])
                        pose.update({nm: math.radians(v) for nm, v in zip(planners[h].chain.joint_names, q[h])})
                        band = world.bands[h]
                        pose[f"{h}_gripper"] = band[0] + aperture * (band[1] - band[0])
                    goal = np.array([pose[nm] for nm in pose])
                    for _ in range(physics_steps(k - 1, SETTINGS.fps, m.opt.timestep)):
                        now = world.data.ctrl[ids]
                        world.data.ctrl[ids] = now + np.clip(goal - now, -ceiling, ceiling)
                        for carry in carries.values():
                            if carry is not None:
                                carry.update()
                        mujoco.mj_step(m, world.data)
                    d = world.data
                    for h in working:
                        touching = {int(c.geom2 if c.geom1 in thing_geoms[h] else c.geom1)
                                    for c in (d.contact[j] for j in range(d.ncon))
                                    if c.dist < 0 and {c.geom1, c.geom2} & thing_geoms[h]}
                        held[h] |= ((bool(touching & claw[h]) and bool(touching & jaw[h]))
                                    or (carries[h] is not None and carries[h].held == carried[h]))
                        lifted[h] |= bool(world.block_pose(carried[h])[2, 3] - thing0[h] > LIFTED_M)
                    if videos is not None:
                        frame = cv2.cvtColor(np.concatenate(
                            [world.view("side", (480, 360)),
                             *(cv2.resize(world.gripper_image(h), (576, 360)) for h in hands)], axis=1),
                            cv2.COLOR_RGB2BGR)
                        if writer is None:
                            videos.mkdir(parents=True, exist_ok=True)
                            writer = cv2.VideoWriter(str(videos / f"oracle-{i:03d}.mp4"),
                                                     cv2.VideoWriter_fourcc(*"mp4v"), SETTINGS.fps,
                                                     (frame.shape[1], frame.shape[0]))
                        writer.write(frame)
            finally:
                if writer is not None:
                    writer.release()
            world.step(0.5)
            outcomes = {h: placed(world, plans[h].layout) for h in working}
            for h in working:
                carry = carries[h]
                touching = {int(c.geom2 if c.geom1 in thing_geoms[h] else c.geom1)
                            for c in (world.data.contact[j] for j in range(world.data.ncon))
                            if c.dist < 0 and {c.geom1, c.geom2} & thing_geoms[h]}
                let_go = (not (touching & claw[h] and touching & jaw[h])) if carry is None else carry.held is None
                outcomes[h]["released"] = let_go
                outcomes[h]["placed"] = bool(outcomes[h]["placed"] and let_go
                                             and world.speed(carried[h]) < SETTLED_MPS)
            for carry in carries.values():
                if carry is not None:
                    carry.reset()
            per = {h: {"held": held[h], "lifted": lifted[h], "placed": bool(outcomes[h]["placed"]),
                       **{k: v for k, v in outcomes[h].items() if k != "placed"}, "plan": plans[h].info}
                   for h in working}
            if len(hands) == 1:
                result = per[hands[0]]
            else:
                result = {k: all(r[k] for r in per.values()) for k in ("held", "lifted", "placed")}
                result.update({"hands": per, "off_mm": max(r["off_mm"] for r in per.values())})
            results.append(result)
            say(f"  {i:3d}: held {'yes' if result['held'] else 'no '}  lifted {'yes' if result['lifted'] else 'no '}  "
                f"placed {'yes' if result['placed'] else 'no '}  "
                + ", ".join(f"{carried[h]} {per[h]['off_mm']:.0f} mm off" for h in working))
    finally:
        world.close()
    count = max(1, len(results))
    return {"episodes": len(results), "seed": seed, "scene": scene,
            "grasp": "physics" if physics else "carried once shut on it",
            "held": sum(r["held"] for r in results) / count,
            "lifted": sum(r["lifted"] for r in results) / count,
            "placed": sum(r["placed"] for r in results) / count,
            "seconds": round(time.monotonic() - started, 1), "results": results}
