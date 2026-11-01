"""Single source of truth for the UMI loop in MuJoCo.

The simulation stands in for three things the rig has: a lens, a desk with
things on it - two blocks, or a block and a bowl for each hand - and a
person holding a Hermit, or one in each hand. Each is described here once. The
robot, the camera's place on the hand and the scene around it are not - they
are the description's (camera.yaml, scene.xml), read where they are, so the
simulation cannot hold a second copy of the rig that drifts from the first.

Lengths are metres, angles radians unless a name says otherwise.
"""
from __future__ import annotations

import dataclasses
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from pinkrab_core.config import SETTINGS as CORE
from pinkrab_umi.config import SETTINGS as UMI

# Every picture here is rendered off screen. MuJoCo picks its GL backend when it
# is first imported, from this variable; on Linux that is EGL, with no display.
# macOS has no EGL and renders off screen through its own default.
if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")


@dataclass(frozen=True)
class LensSpec:
    """A fisheye as the recorder sees it, Kannala-Brandt: a ray `theta` off the
    optical axis lands `f * theta * (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 +
    k4 theta^8)` pixels from the centre. With every k zero it is equidistant.

    `f` is pixels per radian near the axis and (`cx`, `cy`) where the axis
    lands. Nothing is drawn past `fov / 2` off the axis - the image circle ends
    there - nor outside the sensor.
    """

    name: str
    width: int
    height: int
    f: float
    cx: float
    cy: float
    fov: float
    k: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    def radius(self, theta):
        """Pixels from the centre for a ray `theta` radians off the axis."""
        t2 = np.square(theta)
        k1, k2, k3, k4 = self.k
        return self.f * theta * (1.0 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))

    def _growing(self) -> tuple[np.ndarray, np.ndarray]:
        """(theta, radius) over the stretch where the polynomial still grows -
        a fit holds only out to where the board was seen, and past its turn it
        says nothing."""
        grid = np.linspace(0.0, math.pi, 4096)
        r = self.radius(grid)
        falls = np.flatnonzero(np.diff(r) <= 0)
        end = int(falls[0]) + 1 if len(falls) else len(r)
        return grid[:end], r[:end]

    @property
    def reach(self) -> float:
        """The widest ray the model can answer for, radians off the axis."""
        return float(self._growing()[0][-1]) if any(self.k) else math.pi

    def theta_at(self, radius):
        """The ray off the axis that lands `radius` pixels from the centre;
        infinite past what the model can answer for."""
        if not any(self.k):
            return np.asarray(radius) / self.f
        grid, r = self._growing()
        return np.interp(radius, r, grid, right=np.inf)


LENSES = {
    # The OV2710 module camera.yaml's pitch and yaw are solved for: measured off
    # its own footage, equidistant, the image circle 1541 px across with its
    # centre on row 516.
    "ov2710": LensSpec("ov2710", 1920, 1080, 490.8, 960.0, 516.0, 2 * 770.5 / 490.8),
    # Its replacement, ELP AR0234 global shutter with the 200 degree lens, taken
    # as an equidistant circle filling the sensor's height - the default
    # (`lens` below). Each unit measured by `pinkrab-umi lens fit` is listed
    # beside it as ar0234_<hand>, for `lens` to name.
    "ar0234_200": LensSpec("ar0234_200", 1920, 1200, 600.0 / math.radians(100.0), 960.0, 600.0,
                           math.radians(200.0)),
}


def measured_lenses() -> dict[str, LensSpec]:
    """The gripper lenses this rig has measured, from the workspace's lens.yaml."""
    try:
        data = yaml.safe_load(CORE.lens_file.read_text()) or {}
    except (OSError, yaml.YAMLError):
        return {}
    out = {}
    for hand, fit in data.items():
        if not isinstance(fit, dict) or fit.get("model") != "kannala_brandt":
            continue
        name = f"ar0234_{hand}"
        w, h = fit["size"]
        spec = LensSpec(name, int(w), int(h), (fit["fx"] + fit["fy"]) / 2.0, fit["cx"], fit["cy"],
                        math.pi, tuple(float(x) for x in fit["k"]))
        # The image circle bounds the field; where the fit turns before it, the
        # turn does.
        circle = fit.get("circle") or {}
        edge = float(spec.theta_at(circle["diameter"] / 2.0)) if circle.get("diameter") else math.inf
        out[name] = dataclasses.replace(spec, fov=2.0 * min(edge, spec.reach))
    return out


@dataclass(frozen=True)
class BlockSpec:
    """One block: a square frustum, bottom side, top side and height.

    A top of zero is a pyramid. Solid, and convex, which is how MuJoCo collides
    it whatever the moulding has inside.
    """

    name: str
    bottom: float
    top: float
    height: float
    mass: float
    rgba: tuple[float, float, float, float]
    friction: float = 0.5


@dataclass(frozen=True)
class BowlSpec:
    """A bowl built of plain shapes - a floor and a ring of wall slabs - so
    MuJoCo collides with its inside as well as its outside, which a mesh's
    convex hull would fill in."""

    name: str
    diameter: float        # outside, across the rim
    height: float          # floor's underside to the rim
    wall: float            # thickness of the wall and the floor
    segments: int
    mass: float
    rgba: tuple[float, float, float, float]


@dataclass(frozen=True)
class Settings:
    # --- the lens ---------------------------------------------------------------
    lens: str = "ar0234_200"
    # The gripper video is written at this fraction of the sensor: the dataset
    # is made at 960 wide, and rendering the full sensor only to scale it down
    # costs four times the rendering.
    video_scale: float = 0.5
    # Each cube-map face is rendered square at this many pixels per pixel of the
    # fisheye's own resolution at its centre - a pinhole spends its pixels on
    # its edges, the fisheye on its middle.
    face_oversample: float = 1.2

    # --- the top camera -----------------------------------------------------------
    # A narrow 16:9 view, 43 degrees vertical, to watch a take by. Rendered
    # small; nothing learns from it.
    head_size: tuple[int, int] = (640, 360)
    head_fovy_deg: float = 43.3
    # Standing beside the left arm, looking down at where it works: the person
    # demonstrates from the robot's side, facing the way the arm reaches.
    head_pos: tuple[float, float, float] = (0.30, 0.78, 0.62)
    head_target: tuple[float, float, float] = (0.36, 0.18, 0.0)

    # --- the desk solid ---------------------------------------------------------------
    # Where the world of every recorded pose is: the solid's centre, standing on
    # its cut half a face up, and turned the standard way (its +Y the way the
    # arms reach, desk.yaml's yaw -90). Off to the side of where the blocks go.
    desk_origin: tuple[float, float, float] = (0.56, 0.40, UMI.for_solid(UMI.desk_solid).cut_mm / 1000.0)
    desk_yaw_deg: float = -90.0

    # --- the blocks ---------------------------------------------------------------------
    # The top two pieces of the stacking pyramid: the cap is carried and put
    # down on the green piece.
    blocks: tuple[BlockSpec, ...] = (
        BlockSpec("cap", bottom=0.030, top=0.0, height=0.024, mass=0.006, rgba=(0.93, 0.93, 0.91, 1.0)),
        BlockSpec("base", bottom=0.052, top=0.030, height=0.016, mass=0.014, rgba=(0.10, 0.62, 0.24, 1.0)),
    )
    carried_block: str = "cap"
    target_block: str = "base"
    # Where blocks are put down, in the robot's frame: the part of the desk the
    # left arm reaches with its tool pitched down (x from the board, y toward
    # the left arm). Two blocks at least this far apart, centre to centre.
    block_x: tuple[float, float] = (0.29, 0.40)
    block_y: tuple[float, float] = (0.10, 0.30)
    block_yaw_deg: float = 20.0
    block_gap: float = 0.085

    # --- the bowls: two hands, one or both at work ----------------------------------------------
    # A hand at work picks up its own block and puts it in its own bowl, on its
    # own side of the desk; a hand not at work is held still where its arm
    # stands ready. The rig is proven with the right hand at work and the left
    # at rest (`two_hand_working`); both at once is ("left", "right").
    # Written for the left hand, in the robot's frame; the right hand's is the
    # same across y = 0.
    two_hand_working: tuple[str, ...] = ("right",)
    # The opening the hand waits at over the 40 mm block: wide enough that
    # the jaw shutting on it travels as far as a hand's does - the shelf calls
    # a jaw that hardly moved a take that grasped nothing.
    two_hand_open_aperture: float = 0.80
    # The printed ones: a 40 mm black cube, the grasp cube of the arm cells
    # before this one, and a 100 mm bowl 30 mm high.
    bowl_blocks: tuple[BlockSpec, ...] = (
        BlockSpec("block_left", bottom=0.040, top=0.040, height=0.040, mass=0.030, rgba=(0.06, 0.06, 0.07, 1.0)),
        BlockSpec("block_right", bottom=0.040, top=0.040, height=0.040, mass=0.030, rgba=(0.06, 0.06, 0.07, 1.0)),
    )
    bowls: tuple[BowlSpec, ...] = (
        BowlSpec("bowl_left", diameter=0.100, height=0.030, wall=0.0024, segments=16, mass=0.04,
                 rgba=(0.88, 0.29, 0.56, 1.0)),
        BowlSpec("bowl_right", diameter=0.100, height=0.030, wall=0.0024, segments=16, mass=0.04,
                 rgba=(0.88, 0.29, 0.56, 1.0)),
    )
    two_hand_block_x: tuple[float, float] = (0.29, 0.40)
    two_hand_block_y: tuple[float, float] = (0.12, 0.30)
    two_hand_bowl_x: tuple[float, float] = (0.24, 0.40)
    two_hand_bowl_y: tuple[float, float] = (0.06, 0.32)
    # Centre to centre: a hand's block and its bowl; anything of one hand and
    # anything of the other.
    two_hand_block_bowl_gap: float = 0.09
    two_hand_apart: float = 0.12
    # Where each hand starts, as for one: out from its shoulder, clear of what
    # is on the desk.
    two_hand_start_x: tuple[float, float] = (0.32, 0.44)
    two_hand_start_y: tuple[float, float] = (0.06, 0.34)
    # The top camera, wide and in the middle, both hands in its view.
    two_hand_head_pos: tuple[float, float, float] = (0.78, 0.0, 0.66)
    two_hand_head_target: tuple[float, float, float] = (0.32, 0.0, 0.0)
    two_hand_head_fovy_deg: float = 62.0
    # A block bound for a bowl is held this far up its height, the jaw shut
    # this far past where it meets it, carried this high over the desk to clear
    # a bowl's rim, and let go with its bottom this far over it.
    thing_grasp_height: float = 0.45
    thing_squeeze: float = 0.06
    thing_carry_height: float = 0.075
    bowl_drop: float = 0.012
    # The claw's inner face near its tips, along the arm's tool x: a block
    # bound for a bowl is held with its side against it.
    claw_face_in_tcp: float = -0.003

    # --- the demonstrations ---------------------------------------------------------------
    fps: int = 30
    # How far below level the hand holds the tool, and how far its heading
    # wanders off the line from the shoulder to the block.
    pitch_deg: tuple[float, float] = (55.0, 72.0)
    heading_deg: float = 12.0
    # Where the take starts: anywhere over the part of the desk the arm works
    # in, this high, the tool this far below level and heading the way the
    # arm would hold it there - out from the shoulder, not at the cap; a tool
    # turned back toward the board is one the arm cannot hold. A hand does
    # not begin every take lined up behind the cap, and a policy shown only
    # takes that do has never seen the cap anywhere but dead ahead.
    start_x: tuple[float, float] = (0.32, 0.44)
    start_y: tuple[float, float] = (0.03, 0.35)
    start_height: tuple[float, float] = (0.08, 0.14)
    start_pitch_deg: tuple[float, float] = (40.0, 65.0)
    start_clear: float = 0.04
    # How high the cap is carried.
    carry_height: float = 0.05
    # The approach: the tool comes in along its own axis from this far out.
    approach: float = 0.05
    # After letting go the hand draws back, level, the way it came, and rises:
    # the shelf wants 20 mm of each (pinkrab_umi keep_min_retract_mm).
    retreat_back: float = 0.04
    retreat_up: float = 0.05
    # Seconds for each part of the take, drawn from these ranges.
    reach_s: tuple[float, float] = (1.2, 1.8)
    descend_s: tuple[float, float] = (0.6, 0.9)
    close_s: tuple[float, float] = (0.35, 0.5)
    carry_s: tuple[float, float] = (1.3, 1.9)
    lower_s: tuple[float, float] = (0.5, 0.8)
    open_s: tuple[float, float] = (0.3, 0.45)
    retreat_s: tuple[float, float] = (0.6, 0.9)
    # A held hand is not still: a slow wander of this size rides on the plan.
    tremor_m: float = 0.0015
    tremor_rad: float = math.radians(0.8)
    # The opening the Hermit waits at, open, as a fraction of its travel.
    open_aperture: float = 0.55
    # Where the cap is held, in the arm's tool frame: against the claw's inner
    # face (the claw is the fixed finger, on the tool's +x) and a centimetre in
    # from the tips. Measured with the arm's own collision pieces: resting there,
    # the jaw meets the cap's sides at 0.27 to 0.30 of its travel, whatever
    # the pitch; the hand shuts a little past that.
    grasp_in_tcp: tuple[float, float, float] = (-0.018, 0.0, -0.010)
    grasp_height: float = 0.4          # of the cap's height, from its bottom
    grasp_aperture: float = 0.24
    # The cap is let go this far above the green piece's top.
    drop: float = 0.004
    # What the recorder would have seen of it: the tracker's noise on the
    # hand's pose, frame by frame.
    pose_noise_m: float = 0.0005
    pose_noise_rad: float = math.radians(0.3)

    # --- the arm --------------------------------------------------------------------------
    arm: str = "left"
    # The servos as the bench has them, in the task scene only: scene.xml's
    # position gain of 15 N m/rad lets an arm reaching over the desk sag 15 mm
    # under its own weight, where an STS3215 - stalling near 3 N m and giving
    # all of it within a couple of degrees - holds to a few. And the gripper
    # squeezes with the quarter of its stall torque that Torque_Limit leaves it.
    servo_kp: float = 80.0
    servo_stall_nm: float = 3.0
    # The motor's rotor as the joint feels it through the 1:345 gearbox, kg m^2.
    # scene.xml has none, and the jaw alone - 12 g - under that gain would
    # ring faster than the 2 ms step can follow.
    servo_armature: float = 0.006
    # Contact friction as MuJoCo recommends it for grasping: an elliptic cone,
    # and the friction constraint weighted over the normal one.
    impratio: float = 10.0

    @property
    def lens_spec(self) -> LensSpec:
        return {**LENSES, **measured_lenses()}[self.lens]

    @property
    def root(self) -> Path:
        """Everything simulated lands here, apart from what the rig records."""
        return CORE.workspace / "sim"

    @property
    def episodes_dir(self) -> Path:
        return self.root / "episodes"

    @property
    def camera_file(self) -> Path:
        return CORE.description_pkg / "config" / "camera.yaml"


SETTINGS = Settings()
