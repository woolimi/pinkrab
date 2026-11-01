"""Where the hand is, and how to put it somewhere.

There is no leader arm on this rig, so nothing here is driven by copying joints:
every motion starts as a place to put the hand and becomes joint angles here. The
solver is LeRobot's, over placo, reading the same description the simulation and
RViz read - so a pose aimed at in MuJoCo is the same pose aimed at on the bench.

    k = ArmKinematics("left")
    pose = k.forward(k.home)          # 4x4, metres, in the world the mount defines
    joints = k.inverse(pose, seed=k.home)

**These are description angles, in degrees - not servo readings.** A servo's zero is
wherever the arm was held when calibration asked for it, and turning ticks into
these is `angles.ticks_to_degrees`. Feeding raw servo degrees to `inverse` gives an
answer that is wrong without being obviously wrong, which is the failure this
distinction exists to prevent.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

from .config import ARMS, SETTINGS
from .urdf import expand_for_meshes


class KinematicsError(RuntimeError):
    """Raised when the solver cannot be built or a target cannot be reached."""


def write_urdf(path: Path | None = None) -> Path:
    """Expand the description to a file, because placo reads a path, not a string.

    Written into the workspace rather than the checkout: it is an expansion of one
    rig's mount.yaml, so it belongs where the calibration for that rig lives.

    Mesh references are turned into filesystem paths on the way out. `package://`
    resolves through an ament index, and the solver has to run on a machine with no
    ROS on it - the same reason the studio rewrites them for the browser.
    """
    target = path or SETTINGS.urdf_file
    target.parent.mkdir(parents=True, exist_ok=True)
    text = expand_for_meshes(str(SETTINGS.description_pkg / "meshes"))
    # Written beside it and swapped in, so another process reading it meanwhile
    # gets the old file or the new one whole, never one cut short.
    with tempfile.NamedTemporaryFile("w", dir=target.parent, suffix=".urdf", delete=False) as handle:
        handle.write(text)
    os.chmod(handle.name, 0o644)  # a temporary file is born private
    os.replace(handle.name, target)
    return target


def fresh_urdf() -> Path:
    """The workspace's expansion, written again whenever the description is newer.

    The expansion is made once and kept, so without this an edit to the xacro or
    the mount would go on being served from the old file: inverse kinematics aimed
    through links of the old length, and bands the parts no longer allow.
    """
    target = SETTINGS.urdf_file
    sources = [p for p in (SETTINGS.xacro_file, SETTINGS.mount_file) if p.is_file()]
    if not target.is_file() or any(p.stat().st_mtime > target.stat().st_mtime for p in sources):
        write_urdf(target)
    return target


def arm_fingerprint(arm: str, urdf: Path | None = None) -> str:
    """What an arm's joint angles mean, as a short hash: where the description
    (the workspace's expansion, or `urdf`) puts each of its joints and which
    way each turns.

    Joint angles kept on disk are an arm's pose only on the arm they were read
    from. Rebuilt - a link another length, a zero moved - the same angles put
    the tool somewhere else, and nothing in the numbers says so. Kept beside
    them, this does. Where the arm is bolted is the rig's and is left out.
    """
    import hashlib
    import xml.etree.ElementTree as ET

    prefix = f"{arm}_"
    rows = []
    for joint in ET.fromstring((urdf or fresh_urdf()).read_text()).findall("joint"):
        name = joint.get("name", "")
        if not name.startswith(prefix) or name == f"{prefix}base_joint":
            continue
        origin, axis = joint.find("origin"), joint.find("axis")
        places = [(origin, "xyz"), (origin, "rpy"), (axis, "xyz")]
        # Rounded to a micron and a microradian, and -0 written as 0.
        numbers = [round(float(v), 6) + 0.0
                   for element, key in places
                   for v in (element.get(key, "0 0 0") if element is not None else "0 0 0").split()]
        rows.append((name, joint.get("type"), joint.find("parent").get("link"),
                     joint.find("child").get("link"), *numbers))
    return hashlib.sha256(repr(sorted(rows)).encode()).hexdigest()[:16]


class ArmKinematics:
    """One arm's forward and inverse kinematics, in description angles.

    The gripper is not a kinematic joint - it opens the hand, it does not move it -
    so it is absent from every vector here.
    """

    def __init__(self, arm: str, urdf: Path | None = None) -> None:
        if arm not in ARMS:
            raise KinematicsError(f"no such arm: {arm!r}, expected one of {ARMS}")
        self.arm = arm
        self.joint_names = tuple(f"{arm}_{name}" for name in SETTINGS.joint_names[:-1])
        self.frame = f"{arm}_gripper_frame_link"

        path = urdf or fresh_urdf()
        if not path.is_file():
            path = write_urdf(path)
        try:
            import placo
        except ImportError as exc:  # pragma: no cover - environment, not logic
            raise KinematicsError(f"placo is not importable: {exc}") from exc
        # placo directly rather than through LeRobot's wrapper around it. The
        # wrapper puts one frame task on the solver and nothing else, and a QP
        # with nothing else in it is an unregularised least squares: wherever the
        # Jacobian is close to singular - and for this wrist that is most of the
        # workspace, condition numbers in the hundreds - the step it takes is
        # enormous. Measured: a target moved 3.7 mm, and the first step landed
        # 304 mm away. Three things the wrapper leaves off make it a solver that
        # follows a hand:
        #   - a regularisation task, so a step costs something and the near-null
        #     direction is walked, not leapt;
        #   - the joint limits, so the answer is one the arm can be in;
        #   - a velocity limit, so no tick asks for more than the servos have.
        self._robot = placo.RobotWrapper(str(path))
        self._solver = placo.KinematicsSolver(self._robot)
        self._solver.mask_fbase(True)
        self._task = self._solver.add_frame_task(self.frame, np.eye(4))
        self._solver.add_regularization_task(SETTINGS.ik_regularization)
        self._solver.enable_joint_limits(True)
        self._robot.set_velocity_limits(float(np.radians(SETTINGS.max_joint_speed_dps)))
        self._solver.enable_velocity_limits(True)

    @property
    def dof(self) -> int:
        return len(self.joint_names)

    @property
    def home(self) -> np.ndarray:
        """The pose the description calls zero - what `pinkrab-studio` opens on."""
        return np.zeros(self.dof)

    def forward(self, joints_deg) -> np.ndarray:
        """Where the tool centre point is, as a 4x4 in the mount's world frame."""
        joints = np.asarray(joints_deg, dtype=float)
        if joints.shape != (self.dof,):
            raise KinematicsError(
                f"{self.arm} takes {self.dof} joint angles "
                f"({', '.join(self.joint_names)}), got {joints.shape}"
            )
        self._set(np.radians(joints))
        return np.asarray(self._robot.get_T_world_frame(self.frame))

    def _set(self, joints_rad: np.ndarray) -> None:
        for name, value in zip(self.joint_names, joints_rad):
            self._robot.set_joint(name, float(value))
        self._robot.update_kinematics()

    def _get(self) -> np.ndarray:
        return np.array([self._robot.get_joint(name) for name in self.joint_names])

    def track(self, pose, current, steps: int = 3,
              orientation_weight: float = 0.01, rate_hz: float | None = None) -> np.ndarray:
        """Step the arm toward `pose` from where it is. The control-loop primitive.

        placo takes ONE step of a differential IK per solve - it is built to follow
        a target that moves a little each tick, not to jump to one. Three steps
        are taken per call. What a tick can close is bounded by the velocity limit
        and, near a singular pose, by the regularisation: there the residual is
        walked down over the ticks that follow rather than closed in this one.

        This is what a dragged handle, a replayed trajectory, or a servo loop wants.
        `rate_hz` is how often that loop calls - ik_rate_hz, teleop's, unless
        said: a policy's is its dataset's.
        """
        target = self._as_pose(pose)
        joints = np.asarray(current, dtype=float)
        if joints.shape != (self.dof,):
            raise KinematicsError(f"{self.arm} takes {self.dof} angles, got {joints.shape}")
        self._set(np.radians(joints))
        self._task.T_world_frame = target
        self._task.configure(self.frame, "soft", 1.0, orientation_weight)
        # The velocity limit is spent per solve, so the tick is shared out over
        # the steps taken in it: the call as a whole asks no joint for more than
        # one tick's worth of turning.
        self._solver.dt = 1.0 / ((rate_hz or SETTINGS.ik_rate_hz) * max(1, steps))
        for _ in range(max(1, steps)):
            self._solver.solve(True)
            self._robot.update_kinematics()
        return np.degrees(self._get())

    def solve(self, pose, seed=None, restarts: int = 12,
              orientation_weight: float = 0.01, tol_m: float = 1e-4) -> np.ndarray:
        """Best joint angles for `pose`, found from scratch.

        A differential IK asked to jump somewhere far settles into whatever local
        minimum it walks into, and which one depends entirely on where it started -
        `home` is a particularly poor start, because shoulder_lift's range begins
        AT zero and a solver that starts on a limit has one fewer way to go. So
        several starts are tried and the closest is kept.

        It is best-effort by nature. A target beyond the arm's 583 mm reach comes
        back as the nearest pose it could hold, which is a fine answer to a
        different question - always check with `reach_error`.
        """
        target = self._as_pose(pose)
        rng = np.random.default_rng(0)
        starts = [self.home if seed is None else np.asarray(seed, dtype=float)]
        lo, hi = self.limits
        starts += [rng.uniform(lo, hi) for _ in range(max(0, restarts - 1))]

        # No velocity limit here: this is a pose to jump to, not a tick to
        # follow, and forty steps of one tick's turning each would not get
        # across the workspace.
        self._solver.enable_velocity_limits(False)
        try:
            best, best_err = starts[0], float("inf")
            for start in starts:
                joints = self.track(target, start, steps=40,
                                    orientation_weight=orientation_weight)
                err, _ = self.reach_error(target, joints)
                if err < best_err:
                    best, best_err = joints, err
                if best_err < tol_m:
                    break
        finally:
            self._solver.enable_velocity_limits(True)
        return best

    @property
    def limits(self) -> tuple[np.ndarray, np.ndarray]:
        """Each joint's band, in degrees, as the description declares it."""
        import xml.etree.ElementTree as ET

        tree = ET.fromstring(fresh_urdf().read_text())
        band = {j.get("name"): j.find("limit") for j in tree.findall("joint")}
        lo = [np.degrees(float(band[n].get("lower"))) for n in self.joint_names]
        hi = [np.degrees(float(band[n].get("upper"))) for n in self.joint_names]
        return np.array(lo), np.array(hi)

    def reach_ball(self) -> tuple[np.ndarray, float]:
        """A ball the tool centre never leaves: centred on the shoulder_pan joint,
        in the mount's world frame, its radius the distances from each moving
        joint to the next and on to the tool centre, added up. Each distance is
        a joint origin in the description and holds at any angles, so no shape
        of the arm puts the tool further out than all of them laid in one line."""
        import xml.etree.ElementTree as ET

        tree = ET.fromstring(fresh_urdf().read_text())
        child = {j.get("name"): j.find("child").get("link") for j in tree.findall("joint")}
        self._set(np.zeros(self.dof))
        points = [np.asarray(self._robot.get_T_world_frame(link))[:3, 3]
                  for link in [child[n] for n in self.joint_names] + [self.frame]]
        return points[0], float(sum(np.linalg.norm(b - a) for a, b in zip(points, points[1:])))

    def keep_off_stops(self, margin_deg: float) -> None:
        """Solve with every joint held `margin_deg` inside its band. `limits`
        still answers the band the description declares."""
        lo, hi = self.limits
        for name, low, high in zip(self.joint_names, lo, hi):
            self._robot.set_joint_limits(name, float(np.radians(low + margin_deg)),
                                         float(np.radians(high - margin_deg)))

    def _as_pose(self, pose) -> np.ndarray:
        target = np.asarray(pose, dtype=float)
        if target.shape != (4, 4):
            raise KinematicsError(f"pose must be a 4x4 transform, got {target.shape}")
        return target

    def reach_error(self, pose, joints_deg) -> tuple[float, float]:
        """How far a solution lands from what was asked: metres, and degrees.

        The solver returns the best it found, not a promise, so whatever calls it
        has to look. A target outside the arm's reach comes back as the nearest pose
        it could hold, which is a perfectly good answer to a different question.
        """
        got = self.forward(joints_deg)
        want = np.asarray(pose, dtype=float)
        offset = float(np.linalg.norm(got[:3, 3] - want[:3, 3]))
        turn = got[:3, :3].T @ want[:3, :3]
        angle = float(np.degrees(np.arccos(np.clip((np.trace(turn) - 1) / 2, -1, 1))))
        return offset, angle
