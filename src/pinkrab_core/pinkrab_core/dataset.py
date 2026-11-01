"""The dataset's contract: what each feature holds, and how a step becomes numbers and back.

Written by `pinkrab-umi convert`, read by whatever asks a policy trained on it
- the rollout node, the studio's policy page, the simulation. Kept here, below
all of them, so a node on the robot's side can know the layout without taking
in the tools that record and convert, and so the two directions of a step -
two poses into an action, an action into a motion - are one rule in one file.

A dataset is of the hands its takes worked with - both on the rig, the
simulation's one - and every feature of a hand carries its name:

    action            per hand: dp (3, tool frame) + the step's rotation as
                      its first two columns (6, continuous) + the jaw's next
                      opening (1) - `left_dx` ... `right_gripper`
    observation.state per hand, by name: the opening; with two hands, then
                      where the right hand's tool is in the left's
                      (`relative_pose`), UMI's inter-gripper
                      proprioception: what lets one hand act knowing where the
                      other is. A lone number rides with a pad.
    pose.tcp_<hand>   the tool centre in the desk solid's frame, xyz + quaternion -
                      for a correction (DAgger, from a rollout - its kind in meta/pinkrab_convert.json)
                      the arm's tool in the robot's base frame instead, so whatever
                      reads starts off it leaves corrections out
                      (`demo_start.correction_episodes`)
    observation.images.gripper_<hand>
                      the hand's gripper camera frame cut to a window about its
                      lens axis, then scaled - `gripper_view`
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .config import ARMS, SETTINGS


def fps_limit() -> int:
    """The fastest a dataset may be: one step a tick of the gate's shaping."""
    return int(SETTINGS.policy_shaping_hz)


def check_fps(fps) -> int:
    """`fps` as a dataset's rate, or ValueError for one that cannot be."""
    if isinstance(fps, bool) or not isinstance(fps, int) or not 1 <= fps <= fps_limit():
        raise ValueError(f"a dataset's rate is a whole number of frames a second from 1 to {fps_limit()} "
                         f"(the gate's policy_shaping_hz), not {fps!r}")
    return fps


def steps_in(seconds: float, fps: float) -> int:
    """How many of a dataset's steps last `seconds` at `fps` - at least one."""
    return max(1, round(seconds * fps))


def dataset_fps(root: Path) -> int:
    """A dataset's rate, from LeRobot's meta/info.json."""
    import json

    return int(json.loads((root / "meta" / "info.json").read_text())["fps"])

STEP_NAMES = ["dx", "dy", "dz", "r11", "r21", "r31", "r12", "r22", "r32", "gripper"]
# The right hand's tool in the left's frame, as a step is written: position,
# then the first two columns of the rotation.
RELATIVE_NAMES = ["rx", "ry", "rz", "rr11", "rr21", "rr31", "rr12", "rr22", "rr32"]
# The quaternion x y z w, the order the trajectory code writes it.
POSE_NAMES = ["x", "y", "z", "qx", "qy", "qz", "qw"]
# The step that goes nowhere, as the action's first nine numbers.
NO_STEP = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float32)


def image_key(hand: str) -> str:
    return f"observation.images.gripper_{hand}"


def pose_key(hand: str) -> str:
    return f"pose.tcp_{hand}"


def hands_of(keys) -> tuple[str, ...]:
    """The hands a policy or dataset is of, by its gripper pictures, in ARMS order."""
    keys = set(keys)
    return tuple(hand for hand in ARMS if image_key(hand) in keys)


def action_names(hands) -> list[str]:
    return [f"{hand}_{n}" for hand in hands for n in STEP_NAMES]


def state_names(hands) -> list[str]:
    """observation.state's names for these hands: each hand's opening in turn,
    then - with two - the right hand's tool in the left's. A policy shown its
    own last step answers with it again and stops looking at the picture, so
    the state carries no step. LeRobot stores a one-number feature as a
    scalar, which ACT cannot take, so a lone opening rides with a pad."""
    names = [f"{hand}_gripper" for hand in hands]
    if len(hands) == 2:
        names += RELATIVE_NAMES
    return names + ["pad"] if len(names) == 1 else names


def relative_pose(T_left: np.ndarray, T_right: np.ndarray) -> np.ndarray:
    """Where the right hand's tool is in the left's frame: nine numbers."""
    return delta_of(T_left, T_right)


def state_vector(names: list[str], apertures: dict[str, float],
                 relative: np.ndarray | None = None) -> np.ndarray:
    """observation.state by its names: each hand's opening, the hands'
    relative pose, a pad."""
    values: dict[str, float] = {"pad": 0.0}
    for hand, aperture in apertures.items():
        values[f"{hand}_gripper"] = aperture
    if relative is not None:
        values.update(zip(RELATIVE_NAMES, relative))
    return np.array([values[n] for n in names], dtype=np.float32)


def delta_of(T_from: np.ndarray, T_to: np.ndarray) -> np.ndarray:
    """The step between two tool poses (4x4), in the first one's frame: nine numbers."""
    R = T_from[:3, :3]
    dp = R.T @ (T_to[:3, 3] - T_from[:3, 3])
    dR = R.T @ T_to[:3, :3]
    return np.concatenate([dp, dR[:, 0], dR[:, 1]])


def delta_to_matrix(action) -> np.ndarray:
    """An action's step - dp (3) and two rotation columns (6) - as a 4x4."""
    a = np.asarray(action, dtype=float)
    c1 = a[3:6] / max(np.linalg.norm(a[3:6]), 1e-9)
    c2 = a[6:9] - np.dot(c1, a[6:9]) * c1
    c2 = c2 / max(np.linalg.norm(c2), 1e-9)
    c3 = np.cross(c1, c2)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([c1, c2, c3])
    T[:3, 3] = a[:3]
    return T


# The dataset keeps a window of each gripper camera's sensor - camera.yaml's
# `crop`, as fractions of the sensor's width and height (1560 x 1140 of the
# AR0234's 1920 x 1200, whose image circle leaves the corners black) - centred
# on where that unit's lens axis lands, scaled to GRIPPER_WIDTH. A frame is the
# whole sensor or the part of it the mode opened shows (the profile's
# `region`), at some scale - told apart by their shape (gripper_region) - and
# the window is found in it from the sensor's. Two units' axes land up to some
# 30 px apart (26 px in height, the measured left pair), so centred on each
# one's own the same ray falls within a few pixels in both pictures. An axis
# further off the middle than the window's margin pulls the window against the
# frame's edge, and `lens fit` says so.
def gripper_crop() -> tuple[float, float]:
    from .config import SETTINGS

    return SETTINGS.camera_profile("gripper").crop


GRIPPER_WIDTH = 780


class LensMissing(RuntimeError):
    """A gripper camera whose lens has not been measured."""


def _window_aspect() -> float:
    """The window's width over its height: the crop of the sensor."""
    from .config import SETTINGS

    crop = gripper_crop()
    sensor_w, sensor_h = SETTINGS.camera_profile("gripper").sensor_size
    return crop[0] * sensor_w / (crop[1] * sensor_h)


def gripper_region(width: int, height: int) -> tuple[int, int, int, int]:
    """The part of the sensor (x, y, w, h, in the sensor's pixels) a width x
    height gripper frame shows: the whole sensor, or the profile's region,
    whichever has its shape - at whatever scale. ValueError for neither."""
    from .config import SETTINGS

    profile = SETTINGS.camera_profile("gripper")
    whole = (0, 0, *profile.sensor_size)
    for region in (whole, profile.sensor_region):
        if abs(width / height - region[2] / region[3]) <= 0.01:
            return region
    raise ValueError(f"a {width}x{height} gripper frame is neither the {whole[2]}x{whole[3]} sensor nor its "
                     f"{profile.sensor_region[2]}x{profile.sensor_region[3]} region at any scale")


def gripper_shape(width: int, height: int) -> tuple[int, int]:
    """The dataset's gripper picture (height, width) for a camera frame of this
    size: the window at GRIPPER_WIDTH, the height even for the video encoder.
    The same for every frame: the window is the sensor's."""
    window_h = GRIPPER_WIDTH / _window_aspect()
    return 2 * round(window_h / 2), GRIPPER_WIDTH


def gripper_window(width: int, height: int, centre: tuple[float, float]) -> tuple[int, int, int, int]:
    """The window (x, y, w, h) of a width x height frame the dataset keeps,
    about `centre` (x, y as fractions of the sensor - as a lens fit of its
    whole frame has it), moved to stay inside the part of the sensor the frame
    shows."""
    from .config import SETTINGS

    crop = gripper_crop()
    sensor_w, sensor_h = SETTINGS.camera_profile("gripper").sensor_size
    rx, ry, rw, rh = gripper_region(width, height)
    w, h = sensor_w * crop[0], sensor_h * crop[1]
    x = min(max(centre[0] * sensor_w - w / 2, rx), rx + rw - w)
    y = min(max(centre[1] * sensor_h - h / 2, ry), ry + rh - h)
    scale = width / rw
    return round((x - rx) * scale), round((y - ry) * scale), round(w * scale), round(h * scale)


def gripper_cut(size: tuple[int, int], cols: int, rows: int) -> bool:
    """Whether a policy of picture `size` (height, width) is shown the window
    of a cols x rows frame (True) or the whole frame (False) - a checkpoint
    trained before there was a window. ValueError for a shape that is
    neither: nothing made here matches the pictures it learned from."""
    h, w = size
    if abs(w / h - cols / rows) <= 0.01:
        return False
    if abs(w / h - _window_aspect()) <= 0.01:
        return True
    raise ValueError(f"a {w}x{h} gripper picture is neither the window of a {cols}x{rows} frame nor the "
                     "whole of it - the checkpoint learned from pictures cut another way; convert its takes "
                     "again and train anew")


def gripper_view(picture: np.ndarray, size: tuple[int, int], centre: tuple[float, float]) -> np.ndarray:
    """A gripper camera frame as a policy of picture `size` (height, width) is
    shown it: the window about `centre`, scaled - one rule for a frame at any
    scale and of either part of the sensor (see `gripper_cut`, `gripper_region`)."""
    import cv2

    h, w = size
    rows, cols = picture.shape[:2]
    if gripper_cut(size, cols, rows):
        x, y, cw, ch = gripper_window(cols, rows, centre)
        picture = picture[y:y + ch, x:x + cw]
    if picture.shape[:2] != (h, w):
        picture = cv2.resize(picture, (w, h), interpolation=cv2.INTER_AREA)
    return picture


def lens_command(camera: str, step: str) -> str:
    """How `camera`'s lens is measured: `pinkrab-umi lens fit left` for a
    handheld's, `pinkrab lens fit left` for an arm's (`follower_left`)."""
    if camera.startswith("follower_"):
        return f"pinkrab lens {step} {camera.removeprefix('follower_')}"
    return f"pinkrab-umi lens {step} {camera}"


def lens_fit(camera: str) -> dict | None:
    """`camera`'s lens fit as the workspace's lens.yaml holds it - the model,
    its coefficients, the frame size and the image circle - or None if it
    has not been measured. Named as for `lens_centre`."""
    import yaml

    from .config import SETTINGS

    try:
        lenses = yaml.safe_load(SETTINGS.lens_file.read_text())
    except (OSError, yaml.YAMLError):
        return None
    lens = lenses.get(camera) if isinstance(lenses, dict) else None
    return lens if isinstance(lens, dict) else None


def lens_centre(camera: str) -> tuple[float, float] | None:
    """Where `camera`'s lens axis lands in its frame, (x, y) as fractions, as
    its lens fit measured it - `left`/`right` for the handhelds,
    `follower_left`/`follower_right` for the arms; None if it has not been."""
    import yaml

    from .config import SETTINGS

    try:
        lenses = yaml.safe_load(SETTINGS.lens_file.read_text())
        lens = lenses[camera] if isinstance(lenses, dict) else {}
        return float(lens["cx"]) / float(lens["size"][0]), float(lens["cy"]) / float(lens["size"][1])
    except (OSError, yaml.YAMLError, KeyError, TypeError, ValueError, IndexError):
        return None


def require_lens(camera: str) -> tuple[float, float]:
    """`lens_centre`, or LensMissing saying how to measure it: nothing that
    makes or is shown a gripper picture goes on a guess."""
    centre = lens_centre(camera)
    if centre is None:
        raise LensMissing(f"the {camera.replace('_', ' ')} gripper camera's lens is not measured - "
                          f"run `{lens_command(camera, 'capture')}`, then `{lens_command(camera, 'fit')}`")
    return centre


def require_cut(root: Path) -> None:
    """LensMissing unless the dataset at `root` was made with its gripper
    pictures cut about each camera's measured lens axis - by a convert that
    had the lens fit, which keeps every kept episode's centre by hand. One
    made before has pictures no arm camera is cut to match."""
    import json

    try:
        made = json.loads((root / "meta" / "pinkrab_convert.json").read_text())
    except (OSError, ValueError):
        made = {}
    crop = (made.get("gripper") or {}).get("crop")
    hands = made.get("hands") or []
    kept = [e for e in made.get("episodes") or [] if e.get("segments")]

    def cut(entry: dict) -> bool:
        centre = entry.get("centre")
        return isinstance(centre, dict) and all(isinstance(centre.get(hand), list) for hand in hands)
    if not (isinstance(crop, list) and len(crop) == 2 and hands and kept and all(cut(e) for e in kept)):
        raise LensMissing(f"{root.name} was converted without the gripper lens fit - convert it again, "
                          "under a new name, now that the lens is measured")


# A training machine keeps its copy of a dataset as <name>--<fingerprint>, with
# the fingerprint also in this file inside it.
FINGERPRINT = ".pinkrab_fingerprint"


def fingerprint(root: Path) -> str:
    """A dataset's identity: its files by name and size, and its info.json."""
    h = hashlib.sha256()
    for f in sorted(p for p in root.rglob("*") if p.is_file() and p.name != FINGERPRINT):
        h.update(f"{f.relative_to(root)}\0{f.stat().st_size}\n".encode())
    h.update((root / "meta" / "info.json").read_bytes())
    return h.hexdigest()[:12]

