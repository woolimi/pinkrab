"""The scene as something asked questions, rather than something running.

`physics.py` steps this model in real time for the studio. The safety gate wants
the same geometry and none of the stepping: given a pose an operator is asking
for, does the arm meet the desk, the board, itself or the other arm - and is the
pose inside the joint bands at all. Both questions are answered by putting the
model at that pose and running the kinematics once, with no time passing.

Kept apart from `Physics` because a gate that had to own a running simulation to
ask about a pose would either fight the studio's for the model or keep a second
one stepping for nothing. This holds `MjData` of its own and touches it under a
lock, so one instance answers for as many callers as ask.

What it cannot see is worth saying twice, because the gate's whole promise rests
on it. MuJoCo skips contact between a body and its own parent, so the pairs a
joint's own bracket makes - forearm against the truss - never appear here and are
the joint limits' business alone. And the scene holds a desk, a board and two
arms: a cable, a tripod, a coffee cup and a hand are not in it, so a pose that
passes is a pose clear of the ROBOT, not of the room.

The camera bracket is inside that exclusion and it is the tightest pair on the
robot. The bracket is a geom of the gripper link and the jaw is that link's
child, so jaw against bracket is skipped: measured here, a jaw shut hard and a
jaw wide open both report the scene clear. config/camera.yaml states 1.9 mm
between them at the closed end, and that 2.1 mm is held by the gripper's own
band and by the bracket's shape, never by a contact test. Move the camera pose
or the jaw's limits and nothing in this file will object.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from pinkrab_core.config import SETTINGS

# The scene's own surfaces, as opposed to the robot standing on them. The page
# draws these itself - they are not in the description - so it has to be told how
# big they are, and told by the scene rather than by a second set of numbers kept
# beside the drawing code. A board drawn wider than the one that stops the arm
# shows the hand halting in mid air.
SCENERY = ("desk", "pegboard")

# A hand's two fingers: the bodies whose tips may meet the desk on purpose
# (see `fingertip_desk_mm`). Each hand's tool centre is the site `<arm>_tcp`.
FINGERS = ("gripper_link", "jaw_link")


def pressing(contact: dict) -> bool:
    """Whether a contact is one to refuse: anything but a hand's fingertips on
    the desk no deeper than `fingertip_desk_mm`."""
    return not (contact.get("fingertip") and contact["depth"] * 1000.0 <= SETTINGS.fingertip_desk_mm)


class SceneError(RuntimeError):
    """Raised when the scene cannot be loaded or MuJoCo is not installed."""


def load_spec(path: Path):
    """The scene as an MjSpec: the rig's own scene, to be added to and compiled.

    For a scene that is the rig plus what a task needs on top of it - blocks, a
    handheld, cameras - built in memory. The file is never written back: it is
    what the gate checks a pose against.
    """
    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise SceneError("mujoco is not installed - pip install -r requirements/sim.in") from exc
    try:
        return mujoco.MjSpec.from_file(str(Path(path).resolve()))
    except Exception as exc:  # whatever the parser says, say it
        raise SceneError(f"{path} did not load: {exc}") from exc


class Scene:
    """One compiled MuJoCo scene, and what can be read off it without stepping."""

    def __init__(self, path: Path) -> None:
        try:
            import mujoco
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise SceneError(
                "mujoco is not installed - pip install -r requirements/sim.in"
            ) from exc

        self.mujoco = mujoco
        self.path = path
        try:
            self.model = mujoco.MjModel.from_xml_path(str(path.resolve()))
        except Exception as exc:  # whatever the compiler says, say it
            raise SceneError(f"{path} did not compile: {exc}") from exc

        m = self.model
        # Hinges only, in model order. A free joint would carry seven numbers
        # rather than one and there is none here - the robot is bolted to a wall.
        self.joints: dict[str, int] = {}
        for i in range(m.njnt):
            if m.jnt_type[i] != mujoco.mjtJoint.mjJNT_HINGE:
                continue
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name:
                self.joints[name] = int(m.jnt_qposadr[i])

        # Which actuator drives which joint, so a command can be addressed by
        # joint name. Position actuators are one per joint here, but the mapping
        # is read off the model rather than assumed from the order.
        self.actuators: dict[str, int] = {}
        for a in range(m.nu):
            if m.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT:
                continue
            joint = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[a, 0]))
            if joint:
                self.actuators[joint] = a

        # Joint limits. The URDF carries these too, but whatever drives this model
        # should be bounded by what the model will accept, not by a second copy
        # that might have drifted.
        self.limits: dict[str, tuple[float, float]] = {}
        for name in self.joints:
            a = self.actuators.get(name)
            if a is not None and m.actuator_ctrllimited[a]:
                lo, hi = m.actuator_ctrlrange[a]
            else:
                jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
                lo, hi = m.jnt_range[jid] if m.jnt_limited[jid] else (-3.14159, 3.14159)
            self.limits[name] = (float(lo), float(hi))

        # A geom belongs to a body; the body is what a person looking at the page
        # recognises. Scenery hangs off the world body, where the body name says
        # nothing, so those are reported by their own geom name instead - which is
        # how `desk` and `pegboard` come out of a contact rather than `world`.
        self.geom_label: list[str] = []
        for g in range(m.ngeom):
            body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g]))
            geom = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
            scenery = body in (None, "world")
            self.geom_label.append((geom or f"geom{g}") if scenery else body)

        # Each hand's tool centre, which a fingertip contact lands near.
        self._tips: dict[str, int] = {}
        for s in range(m.nsite):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, s) or ""
            if name.endswith("_tcp"):
                self._tips[name.removesuffix("_tcp")] = s

        # For asking about a pose. Separate from any data a caller is stepping,
        # so a question never moves a robot somebody is watching.
        self._probe = mujoco.MjData(self.model)
        self._lock = threading.Lock()

    @property
    def joint_names(self) -> list[str]:
        return list(self.joints)

    def clamp(self, pose: dict[str, float]) -> tuple[dict[str, float], list[str]]:
        """The pose inside the joint bands, and the joints that had to be pulled in."""
        out: dict[str, float] = {}
        hit: list[str] = []
        for name, value in pose.items():
            if name not in self.limits:
                continue
            lo, hi = self.limits[name]
            clamped = min(hi, max(lo, float(value)))
            if abs(clamped - float(value)) > 1e-9:
                hit.append(name)
            out[name] = clamped
        return out, hit

    def touching(self, pose: dict[str, float]) -> list[dict]:
        """What the robot presses against at this pose, deepest first.

        The kinematics are run, not the dynamics: the question is where the bodies
        are when the joints are at these angles, and a step would answer a
        different question - where they end up after the model has reacted.

        Only negative distance counts. MuJoCo reports a pair inside the solver's
        margin as a contact before the surfaces meet, and a gate that refused
        those would refuse an arm resting a centimetre above the desk.

        A contact of a hand's fingers with the desk says whose (`hand`), and
        whose fingertips when it lands near the tool centre (`fingertip`):
        whether it may stand is `pressing`'s to say.
        """
        mujoco = self.mujoco
        with self._lock:
            self._probe.qpos[:] = 0
            for name, adr in self.joints.items():
                if name in pose:
                    self._probe.qpos[adr] = float(pose[name])
            self._probe.qvel[:] = 0
            mujoco.mj_forward(self.model, self._probe)
            found = []
            for i in range(self._probe.ncon):
                contact = self._probe.contact[i]
                depth = -float(contact.dist)
                if depth <= 0:
                    continue
                a, b = self.geom_label[int(contact.geom1)], self.geom_label[int(contact.geom2)]
                found.append({"a": a, "b": b, "depth": round(depth, 5)})
                hand = self._hand(a, b)
                if hand is not None:
                    found[-1]["hand"] = hand
                    reach = float(np.linalg.norm(contact.pos - self._probe.site_xpos[self._tips[hand]]))
                    if reach * 1000.0 <= SETTINGS.fingertip_reach_mm:
                        found[-1]["fingertip"] = hand
        found.sort(key=lambda c: -c["depth"])
        return found

    def _hand(self, a: str, b: str) -> str | None:
        """The arm whose fingers - anywhere on them - a contact of `a` and `b`
        puts on the desk, or None."""
        if "desk" not in (a, b):
            return None
        part = b if a == "desk" else a
        for arm in self._tips:
            if part in {f"{arm}_{finger}" for finger in FINGERS}:
                return arm
        return None

    def scenery(self) -> dict:
        """The desk and the board, as the scene has them.

        Sizes are MuJoCo's: half extents for a box. A plane is a half space - its
        size is what the simulator draws, not what it collides, so the page may
        draw a desk of any extent as long as it puts it at this height.
        """
        mujoco = self.mujoco
        out: dict[str, dict] = {}
        for name in SCENERY:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid < 0:
                continue
            out[name] = {
                "type": "plane" if self.model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_PLANE
                        else "box",
                "pos": [round(float(v), 6) for v in self.model.geom_pos[gid]],
                "half": [round(float(v), 6) for v in self.model.geom_size[gid]],
            }
        return out
