"""`pinkrab-umi lens` - measure a camera's lens against a ChArUco board.

    pinkrab-umi lens board ./print        the board, one A4 page - print at 100 %
    pinkrab-umi lens capture left         show it the board; views are kept as they come
    pinkrab-umi lens fit left             Kannala-Brandt from the kept views -> lens.yaml
    pinkrab-umi lens capture top          the top camera, the live view down
    pinkrab-umi lens fit top              its pinhole from the kept views -> lens.yaml

The arms' gripper cameras are measured by `pinkrab lens capture|fit
left|right`, which is this, kept in lens.yaml as `follower_left` and
`follower_right`. They are measured apart from the handhelds' because a lens
is a unit: the policy is trained on what the handheld's camera saw and shown
what the arm's sees, and each picture is cut about its own camera's lens
axis (`pinkrab_core.dataset.gripper_view`).

A gripper camera's model is OpenCV's fisheye one, Kannala-Brandt with four coefficients:
a ray theta off the axis lands at theta (1 + k1 theta^2 + k2 theta^4 +
k3 theta^6 + k4 theta^8) times f from the centre. A flat board can only be
shown in front of the lens, so the fit holds out to where the board was seen -
toward 90 degrees with the board held close and tilted - and past that it is
the polynomial carried on. The image circle is measured as well, off the
views' bright region, as the edge of what the lens takes in. The top
camera's lens is a wide rectilinear one, fitted as a pinhole (see its section).
"""
from __future__ import annotations

import datetime
import json
import math
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import yaml

from pinkrab_core.config import SETTINGS as CORE
from pinkrab_core.dataset import gripper_crop, lens_command
from pinkrab_core.top_lens import COEFFS as TOP_COEFFS
from pinkrab_core.top_lens import fit_entry, unusable

from .config import SETTINGS, Settings
from .markers import write_pdf, write_svg
from .pnp import Intrinsics

PX_PER_MM = 2
# The top camera's name in lens.yaml and under lens/: the camera over the desk,
# whose picture every marker pose is solved in.
TOP = "top"


def _set_and_side(camera: str) -> tuple[str, str]:
    """`follower_left` -> ("follower", "left"); `left` -> ("handheld", "left")."""
    if camera.startswith("follower_"):
        return "follower", camera.removeprefix("follower_")
    return "handheld", camera


def _counterpart(camera: str) -> str:
    """The other set's camera on the same side: the one this is compared with."""
    which, side = _set_and_side(camera)
    return side if which == "follower" else f"follower_{side}"


def board(cfg: Settings = SETTINGS) -> cv2.aruco.CharucoBoard:
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg.lens_board_dict))
    return cv2.aruco.CharucoBoard(cfg.lens_board_squares, cfg.lens_board_square_mm,
                                  cfg.lens_board_marker_mm, dictionary)


def detector(cfg: Settings = SETTINGS) -> cv2.aruco.CharucoDetector:
    return cv2.aruco.CharucoDetector(board(cfg))


def write_board(path: Path, cfg: Settings = SETTINGS) -> tuple[Path, Path]:
    """The board as vector PDF, centred on the page, with its size printed
    under it to check the print against with a ruler; and the same page as SVG
    beside it, to see it on a screen. Returns both."""
    cols, rows = cfg.lens_board_squares
    w_mm, h_mm = cols * cfg.lens_board_square_mm, rows * cfg.lens_board_square_mm
    picture = board(cfg).generateImage((int(w_mm * PX_PER_MM), int(h_mm * PX_PER_MM)),
                                       marginSize=0, borderBits=1)
    page_w, page_h = cfg.lens_page_mm
    x0, y0 = (page_w - w_mm) / 2, (page_h - h_mm) / 2
    # Black runs of each pixel row, merged down the rows while they repeat.
    open_runs: dict[tuple[int, int], int] = {}
    filled = []
    for row in range(picture.shape[0] + 1):
        runs = set()
        if row < picture.shape[0]:
            dark = np.concatenate([[False], picture[row] < 128, [False]])
            edges = np.flatnonzero(dark[1:] != dark[:-1])
            runs = {(int(a), int(b - a)) for a, b in zip(edges[::2], edges[1::2])}
        for run in [r for r in open_runs if r not in runs]:
            top = open_runs.pop(run)
            x, w = run
            # PDF y runs up from the page's bottom; the picture's rows run down.
            filled.append((x0 + x / PX_PER_MM, y0 + h_mm - row / PX_PER_MM,
                           w / PX_PER_MM, (row - top) / PX_PER_MM))
        for run in runs:
            open_runs.setdefault(run, row)
    label = (f"ChArUco {cols} x {rows}, squares {cfg.lens_board_square_mm:g} mm, markers "
             f"{cfg.lens_board_marker_mm:g} mm, {cfg.lens_board_dict} - print at 100 %, "
             f"the board is {w_mm:g} x {h_mm:g} mm")
    texts = [(x0, y0 - 8.0, 3.2, label)]
    path.parent.mkdir(parents=True, exist_ok=True)
    return (write_pdf(path, (page_w, page_h), [(filled, [], texts)]),
            write_svg(path.with_suffix(".svg"), (page_w, page_h), (filled, [], texts)))


def find_corners(gray: np.ndarray, cfg: Settings = SETTINGS,
                 found: cv2.aruco.CharucoDetector | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(corners (n, 2), ids (n,)) of the board's chessboard corners in a picture."""
    corners, ids, _, _ = (found or detector(cfg)).detectBoard(gray)
    if ids is None or len(ids) == 0:
        return np.empty((0, 2)), np.empty((0,), dtype=int)
    return corners.reshape(-1, 2), ids.reshape(-1)


class Coverage:
    """Where in the frame kept views have put corners, on a coarse grid."""

    def __init__(self, width: int, height: int, cfg: Settings = SETTINGS) -> None:
        self.cols, self.rows = cfg.lens_grid
        self.width, self.height = width, height
        self.counts = np.zeros((self.rows, self.cols), dtype=int)
        self.full = cfg.lens_cell_full

    def cells(self, corners: np.ndarray) -> np.ndarray:
        cx = np.clip((corners[:, 0] * self.cols / self.width).astype(int), 0, self.cols - 1)
        cy = np.clip((corners[:, 1] * self.rows / self.height).astype(int), 0, self.rows - 1)
        return np.column_stack([cy, cx])

    def new_fraction(self, corners: np.ndarray) -> float:
        cells = self.cells(corners)
        return float(np.mean(self.counts[cells[:, 0], cells[:, 1]] < self.full))

    def add(self, corners: np.ndarray) -> None:
        for r, c in {tuple(rc) for rc in self.cells(corners)}:
            self.counts[r, c] += 1

    def draw(self, picture: np.ndarray) -> None:
        h, w = picture.shape[:2]
        for r in range(self.rows):
            for c in range(self.cols):
                n = self.counts[r, c]
                if not n:
                    continue
                x0, y0 = c * w // self.cols, r * h // self.rows
                x1, y1 = (c + 1) * w // self.cols, (r + 1) * h // self.rows
                tint = picture[y0:y1, x0:x1].astype(np.float32)
                green = np.array([60, 200, 60] if n >= self.full else [220, 160, 60], np.float32)
                picture[y0:y1, x0:x1] = (0.75 * tint + 0.25 * green).astype(np.uint8)


def _open_gripper(camera: str, cfg: Settings):
    """A gripper camera, opened here: (read a frame, release, what to say when
    it stops sending), or None."""
    from .gripper_cams import CAMERA_SETS, find_devices

    which, side = _set_and_side(camera)
    device = find_devices(which).get(side)
    if device is None:
        print(f"  no {camera} gripper camera - run `{CAMERA_SETS[which]}` and check it is plugged in")
        return None
    backend = cv2.CAP_AVFOUNDATION if isinstance(device, int) else cv2.CAP_V4L2
    cap = cv2.VideoCapture(device, backend)
    # The sensor's whole frame: a fit is of the whole lens, whatever part of
    # it the cameras are opened at.
    sensor_w, sensor_h = cfg.gripper_camera.sensor_size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, sensor_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, sensor_h)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cfg.gripper_camera.fourcc))
    if not cap.isOpened():
        print(f"  the {camera} camera at {device} would not open")
        return None

    def read() -> np.ndarray | None:
        ok, frame = cap.read()
        return frame if ok else None

    stopped = (f"the {camera} camera stopped sending frames - its USB link dropped. Check the cable, "
               f"run `{CAMERA_SETS[which]}` if the socket changed, and run this again: the views kept "
               "so far stay.")
    return read, cap.release, stopped


def _open_top(cfg: Settings):
    """The top camera, opened here the way the live view opens it - the same
    mode and exposure: (read a frame, release, what to say when it stops), or
    None. It is one camera to one process, so the live view must be
    down."""
    from .head_cam import HeadCamera

    camera = HeadCamera(solve=False)
    camera.retry()
    while camera.opening:
        time.sleep(0.1)
    if not camera.open:
        print(f"  the top camera would not open: {camera.reason}.\n  If the live view is running - the "
              "studio's Collect page, or `pinkrab-umi live` - stop it first: it holds the camera.")
        return None

    def read() -> np.ndarray | None:
        frame = camera.next_frame()
        return None if frame is None else frame.picture

    stopped = ("the top camera stopped sending frames - check its cable and run this again: the "
               "views kept so far stay.")
    return read, camera.stop, stopped


def nominal_top(cfg: Settings = SETTINGS) -> Intrinsics:
    """The top lens as its maker states it: a pinhole of the nominal field of
    view, centred, without distortion. What a fit starts from and the board's
    tilt is shown with while views are kept - never a pose."""
    w, h = cfg.top_camera_width, cfg.top_camera_height
    f = (w / 2) / math.tan(math.radians(cfg.top_lens_nominal_hfov_deg) / 2)
    return Intrinsics(width=w, height=h, fx=f, fy=f, cx=w / 2, cy=h / 2, coeffs=(0.0,) * TOP_COEFFS)


def _top_unit(out: Path, kept: bool) -> str | None:
    """The lens the views in `out` are of, as its camera.json names it - named
    afresh when there are no views yet; None, said, when views are there that
    name none, kept of another camera."""
    stamp = out / "camera.json"
    try:
        unit = json.loads(stamp.read_text()).get("unit") if stamp.is_file() else None
    except (OSError, ValueError, AttributeError):
        unit = None
    if unit:
        return str(unit)
    if kept:
        print(f"  the views in {out} name no lens of this camera - kept of another one. Empty {out} "
              "and capture again")
        return None
    unit = uuid.uuid4().hex[:12]
    stamp.write_text(json.dumps({"unit": unit}, indent=2) + "\n")
    return unit


def _top_guess(unit: str, cfg: Settings) -> Intrinsics:
    """The lens the board's tilt is shown with, and a fit starts from: this
    lens's earlier fit, else the nominal one."""
    found = top_fit()
    if found is not None and found[1].get("unit") == unit:
        return found[0]
    return nominal_top(cfg)


def _facing(corners: np.ndarray, ids: np.ndarray, guess: Intrinsics | None,
            cfg: Settings, object_all: np.ndarray) -> np.ndarray | None:
    """Which way the board faces in the camera's frame - its normal - from
    the lens the camera is solved with now; None without one."""
    if guess is None or len(ids) < cfg.lens_min_corners:
        return None
    ok, rvec, _ = cv2.solvePnP(object_all[ids].astype(np.float64), corners.astype(np.float64),
                               guess.matrix.astype(np.float64), np.asarray(guess.coeffs, np.float64))
    return cv2.Rodrigues(rvec)[0][:, 2] if ok else None


def _new_facing(facing: np.ndarray, facings: list[np.ndarray], cfg: Settings) -> bool:
    """Whether the board faces a way none of `facings` does, by
    top_lens_new_tilt_deg or more."""
    return all(math.degrees(math.acos(float(np.clip(facing @ other, -1.0, 1.0)))) >= cfg.top_lens_new_tilt_deg
               for other in facings)


def capture(camera: str, views: int | None = None, cfg: Settings = SETTINGS) -> int:
    """Keep views of the board from one camera until there are enough.

    A window shows the camera with the corners found and the grid of where kept
    views have put corners - blue once, green when the cell is covered. A view
    is kept on its own when the board is still and lands somewhere new; space
    keeps the current one anyway, q stops.

    The top camera's lens is a rectilinear one, known from how the board is
    tilted as much as from where it lands: a view tilted a new way is kept
    too, and the tilt is shown.
    """
    top = camera == TOP
    opened = _open_top(cfg) if top else _open_gripper(camera, cfg)
    if opened is None:
        return 1
    read_frame, release, stopped = opened
    views = views or (cfg.top_lens_views if top else cfg.lens_views)
    out = cfg.lens_dir / camera
    out.mkdir(parents=True, exist_ok=True)
    guess = None
    if top:
        # The views kept so far are of the lens camera.json names: another
        # lens's views added to them would make a fit of neither.
        unit = _top_unit(out, any(out.glob("*.png")))
        if unit is None:
            release()
            return 1
        guess = _top_guess(unit, cfg)
    kept = sorted(out.glob("*.png"))
    found = detector(cfg)
    object_all = board(cfg).getChessboardCorners()
    # facings: one board normal per way the board has faced, the views kept
    # so far; last: the corners of the last view kept, which a view must move
    # away from to be worth keeping.
    coverage, facings, last = None, [], {}
    print(f"  {camera}: {len(kept)} views already in {out}; keeping up to {views}. "
          "Hold the board still in each place"
          + (f", lifted off the desk and tilted toward each side - {cfg.top_lens_min_facings} ways "
             "at least" if top else "")
          + "; space keeps a view, q stops.")
    title = f"lens {camera}"
    last_kept, previous = 0.0, {}
    # The camera is read on a thread of its own: a read waits for a frame with
    # no limit, and a camera that stops sending - a USB link dropped and
    # renegotiated - would otherwise freeze the window with it.
    latest = {"frame": (None, time.monotonic())}
    stop = threading.Event()

    def read() -> None:
        while not stop.is_set():
            frame = read_frame()
            if frame is not None:
                latest["frame"] = (frame, time.monotonic())

    reader = threading.Thread(target=read, name=f"lens-{camera}", daemon=True)
    reader.start()
    stalled = False
    shown_at = 0.0
    try:
        while len(kept) < views:
            picture, at = latest["frame"]
            if time.monotonic() - at > cfg.lens_stall_s:
                stalled = True
                print(f"  {stopped}")
                break
            if picture is None or at == shown_at:
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break
                continue
            shown_at = at
            if coverage is None:
                # The grid is the camera's picture, and what is already kept
                # is on it from the start.
                coverage = Coverage(picture.shape[1], picture.shape[0], cfg)
                for path in kept:
                    corners, ids = find_corners(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), cfg, found)
                    if len(corners):
                        coverage.add(corners)
                        last = dict(zip(ids.tolist(), corners))
                    facing = _facing(corners, ids, guess, cfg, object_all)
                    if facing is not None and _new_facing(facing, facings, cfg):
                        facings.append(facing)
            gray = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
            corners, ids = find_corners(gray, cfg, found)
            here = dict(zip(ids.tolist(), corners))
            moved = [np.linalg.norm(here[i] - previous[i]) for i in here if i in previous]
            still = len(moved) >= cfg.lens_min_corners and float(np.median(moved)) < cfg.lens_still_px
            previous = here
            enough = len(ids) >= cfg.lens_min_corners
            facing = _facing(corners, ids, guess, cfg, object_all) if enough else None
            turned = facing is not None and _new_facing(facing, facings, cfg)
            # The board held where the last kept view had it is that view again.
            shift = [np.linalg.norm(here[i] - last[i]) for i in here if i in last]
            again = len(shift) >= cfg.lens_min_corners and float(np.median(shift)) < cfg.lens_moved_px
            fresh = enough and not again and (coverage.new_fraction(corners) >= cfg.lens_new_fraction or turned)
            shown = picture.copy()
            coverage.draw(shown)
            for x, y in corners:
                cv2.circle(shown, (int(x), int(y)), 6, (0, 255, 255) if enough else (0, 0, 255), 2)
            tilt = ("" if facing is None
                    else f"  tilt {math.degrees(math.acos(min(1.0, abs(float(facing[2]))))):.0f}")
            cv2.putText(shown, f"{len(kept)}/{views} kept  corners {len(ids)}" + tilt
                        + ("  still" if still else "") + ("  new" if fresh else ""),
                        (24, 56), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)
            if guess is not None:
                short = len(facings) < cfg.top_lens_min_facings
                cv2.putText(shown, f"facings {len(facings)}/{cfg.top_lens_min_facings}"
                            + ("  - lift the board and tilt it a new way" if short else ""),
                            (24, 112), cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                            (80, 160, 255) if short else (120, 230, 120), 3)
            scale = 960 / shown.shape[1]
            cv2.imshow(title, cv2.resize(shown, (960, round(shown.shape[0] * scale))))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            keep = key == ord(" ") and enough
            keep = keep or (still and fresh and time.monotonic() - last_kept > cfg.lens_gap_s)
            if keep:
                path = out / f"{len(kept):03d}.png"
                while path.exists():
                    path = path.with_name(f"{int(path.stem) + 1:03d}.png")
                cv2.imwrite(str(path), picture)
                kept.append(path)
                coverage.add(corners)
                last = here
                if turned:
                    facings.append(facing)
                last_kept = time.monotonic()
                print(f"  kept {path.name}: {len(ids)} corners" + tilt.replace("  tilt", ", tilt"))
    finally:
        stop.set()
        # A reader stuck in a read that will never return is left to the
        # process's exit; releasing the capture under it would hang here too.
        reader.join(timeout=1.0)
        if not reader.is_alive():
            release()
        cv2.destroyAllWindows()
    print(f"  {len(kept)} views in {out}. Next: {lens_command(camera, 'fit')}")
    return 1 if stalled else 0


def _write_lens(data: dict) -> None:
    """lens.yaml, every camera's fit in it."""
    CORE.lens_file.write_text(
        "# Each camera's lens, fitted against a ChArUco board by `pinkrab-umi lens fit` (the\n"
        "# handhelds' and the top camera) and `pinkrab lens fit` (the arms'). A gripper\n"
        "# camera's is OpenCV's Kannala-Brandt: theta -> f theta (1 + k1 theta^2 + k2 theta^4\n"
        "# + k3 theta^6 + k4 theta^8). The top camera's is OpenCV's pinhole with the rational\n"
        "# distortion, k1, k2, p1, p2, k3, k4, k5, k6.\n"
        + yaml.safe_dump(data, sort_keys=False))


def image_circle(pictures: list[np.ndarray], threshold: int = 24) -> dict | None:
    """The edge of what the lens takes in: the brightest each pixel ever was,
    over the views, thresholded; a circle fitted to that region's boundary
    where it is not cut off by the frame's own edge."""
    brightest = np.max(np.stack(pictures), axis=0)
    mask = (brightest > threshold).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    edge = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float)
    h, w = mask.shape
    inside = (edge[:, 0] > 2) & (edge[:, 0] < w - 3) & (edge[:, 1] > 2) & (edge[:, 1] < h - 3)
    edge = edge[inside]
    if len(edge) < 20:
        return None
    # x^2 + y^2 + a x + b y + c = 0, least squares.
    A = np.column_stack([edge[:, 0], edge[:, 1], np.ones(len(edge))])
    a, b, c = np.linalg.lstsq(A, -(edge ** 2).sum(axis=1), rcond=None)[0]
    cx, cy = -a / 2, -b / 2
    return {"cx": round(float(cx), 1), "cy": round(float(cy), 1),
            "diameter": round(float(2 * math.sqrt(cx * cx + cy * cy - c)), 1)}


def fit(camera: str, cfg: Settings = SETTINGS) -> int:
    """Kannala-Brandt for one gripper camera from its kept views, into lens.yaml."""
    paths = sorted((cfg.lens_dir / camera).glob("*.png"))
    if not paths:
        print(f"  no views in {cfg.lens_dir / camera} - run `{lens_command(camera, 'capture')}` first")
        return 1
    found = detector(cfg)
    object_all = board(cfg).getChessboardCorners()
    grays, names, objects, images = [], [], [], []
    for path in paths:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        grays.append(gray)
        corners, ids = find_corners(gray, cfg, found)
        if len(ids) < cfg.lens_min_corners:
            continue
        names.append(path.name)
        objects.append(object_all[ids].reshape(-1, 1, 3).astype(np.float64))
        images.append(corners.reshape(-1, 1, 2).astype(np.float64))
    if len(objects) < 5:
        print(f"  only {len(objects)} views with {cfg.lens_min_corners}+ corners - keep more")
        return 1
    h, w = grays[0].shape
    circle = image_circle(grays)

    def calibrate(objs, imgs):
        # Started from an equidistant lens whose measured image circle is the
        # vendor's 200 degrees - or, with no circle measured, the frame's height.
        radius = (circle or {}).get("diameter", h) / 2
        f0 = radius / math.radians(100.0)
        K = np.array([[f0, 0.0, (circle or {}).get("cx", w / 2)],
                      [0.0, f0, (circle or {}).get("cy", h / 2)], [0.0, 0.0, 1.0]])
        D = np.zeros((4, 1))
        flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
                 | cv2.fisheye.CALIB_USE_INTRINSIC_GUESS)
        criteria = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 200, 1e-10)
        rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(objs, imgs, (w, h), K, D, flags=flags,
                                                       criteria=criteria)
        errors = []
        for obj, img, rvec, tvec in zip(objs, imgs, rvecs, tvecs):
            projected, _ = cv2.fisheye.projectPoints(obj, rvec, tvec, K, D)
            errors.append(float(np.sqrt(np.mean(np.sum((projected - img) ** 2, axis=2)))))
        return rms, K, D, np.array(errors)

    rms, K, D, errors = calibrate(objects, images)
    # A view the rest disagree with - a blurred one, a misread corner - is
    # dropped once and the fit made again without it.
    worst = errors > max(3.0 * float(np.median(errors)), 1.0)
    if worst.any() and (~worst).sum() >= 5:
        print("  dropped: " + ", ".join(f"{n} ({e:.2f} px)" for n, e, bad
                                        in zip(names, errors, worst) if bad))
        objects = [o for o, bad in zip(objects, worst) if not bad]
        images = [i for i, bad in zip(images, worst) if not bad]
        rms, K, D, errors = calibrate(objects, images)

    # How far off axis the board was seen: the widest corner ray, undistorted.
    rays = cv2.fisheye.undistortPoints(np.concatenate(images), K, D)
    theta_seen = math.degrees(float(np.max(np.arctan(np.linalg.norm(rays.reshape(-1, 2), axis=1)))))
    entry = {
        "model": "kannala_brandt",
        "size": [int(w), int(h)],
        "fx": round(float(K[0, 0]), 3), "fy": round(float(K[1, 1]), 3),
        "cx": round(float(K[0, 2]), 3), "cy": round(float(K[1, 2]), 3),
        "k": [round(float(x), 6) for x in D.reshape(-1)],
        "rms_px": round(float(rms), 3),
        "views": len(objects),
        "seen_to_deg": round(theta_seen, 1),
        "circle": circle,
        "fitted": datetime.datetime.now().astimezone().date().isoformat(),
    }
    data = {}
    if CORE.lens_file.is_file():
        data = yaml.safe_load(CORE.lens_file.read_text()) or {}
    data[camera] = entry
    _write_lens(data)
    print(f"  {camera}: rms {rms:.3f} px over {len(objects)} views (worst {errors.max():.2f} px), "
          f"seen out to {theta_seen:.0f} deg off axis")
    print(f"    f {K[0, 0]:.1f} / {K[1, 1]:.1f} px/rad, centre ({K[0, 2]:.1f}, {K[1, 2]:.1f}), "
          f"k {', '.join(f'{x:+.4f}' for x in D.reshape(-1))}")
    if circle:
        print(f"    image circle {circle['diameter']:.0f} px across, centre ({circle['cx']:.0f}, {circle['cy']:.0f})")
    other = data.get(_counterpart(camera))
    if isinstance(other, dict) and other.get("model") == "kannala_brandt":
        # What a policy trained on one and shown the other meets: the two
        # lens axes this far apart - which the window, cut about each one's
        # own, takes out - and the focal lengths, which it does not.
        f, f_other = (entry["fx"] + entry["fy"]) / 2, (other["fx"] + other["fy"]) / 2
        print(f"    against {_counterpart(camera)}: lens axis {entry['cx'] - other['cx']:+.1f}, "
              f"{entry['cy'] - other['cy']:+.1f} px, f {f - f_other:+.1f} px/rad ({(f / f_other - 1) * 100:+.1f} %)")
    crop = gripper_crop()
    room = ((1 - crop[0]) * w / 2, (1 - crop[1]) * h / 2)
    off = (float(K[0, 2]) - w / 2, float(K[1, 2]) - h / 2)
    if abs(off[0]) > room[0] or abs(off[1]) > room[1]:
        print(f"  ! the lens axis is {off[0]:+.0f}, {off[1]:+.0f} px off the frame's middle, past the window's "
              f"room of {room[0]:.0f}, {room[1]:.0f}: this camera's pictures are cut off its axis - "
              f"widen the margin in {CORE.camera_file.name}'s gripper crop")
    print(f"  written: {CORE.lens_file}")
    return 0


# --- the top camera ---------------------------------------------------------------
# Its lens is a rectilinear one, wide, so the model is OpenCV's pinhole with
# the rational distortion - k1, k2, p1, p2, k3, k4, k5, k6: the five of an
# ordinary lens fall short this far off the axis. The views are kept by
# `capture top`, and the fit starts from this lens's earlier fit, else the
# nominal one; camera.json beside the views names the lens they are of.


def fit_top(cfg: Settings = SETTINGS) -> int:
    """The top camera's lens from its kept views, into lens.yaml as `top`."""
    out = cfg.lens_dir / TOP
    paths = sorted(out.glob("*.png"))
    if not paths:
        print(f"  no views in {out} - run `{lens_command(TOP, 'capture')}` first")
        return 1
    unit = _top_unit(out, True)
    if unit is None:
        return 1
    guess = _top_guess(unit, cfg)
    found = detector(cfg)
    object_all = board(cfg).getChessboardCorners()
    names, objects, images, facings = [], [], [], []
    size = None
    for path in paths:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if size is None:
            size = gray.shape[::-1]
        corners, ids = find_corners(gray, cfg, found)
        if len(ids) < cfg.lens_min_corners:
            continue
        facing = _facing(corners, ids, guess, cfg, object_all)
        if facing is not None and _new_facing(facing, facings, cfg):
            facings.append(facing)
        names.append(path.name)
        objects.append(object_all[ids].reshape(-1, 1, 3).astype(np.float32))
        images.append(corners.reshape(-1, 1, 2).astype(np.float32))
    w, h = size
    if (w, h) != (guess.width, guess.height):
        print(f"  the views are {w}x{h}, the camera's picture {guess.width}x{guess.height} - "
              f"empty {out} and capture again")
        return 1
    if len(objects) < 8:
        print(f"  only {len(objects)} views with {cfg.lens_min_corners}+ corners - keep more")
        return 1
    if len(facings) < cfg.top_lens_min_facings:
        # Every view at one tilt - the board lying on the desk, slid about -
        # leaves the focal length to be traded against the distance.
        print(f"  the board faced {len(facings)} way(s) across the views, {cfg.top_lens_min_facings} at "
              f"least are needed - run `{lens_command(TOP, 'capture')}` again, lift the board off the "
              "desk and tilt it toward each side; the window counts the facings")
        return 1

    def calibrate(objs, imgs):
        D = np.zeros(TOP_COEFFS)
        D[:len(guess.coeffs)] = guess.coeffs[:TOP_COEFFS]
        rms, K, D, _, _, spread, _, errors = cv2.calibrateCameraExtended(
            objs, imgs, (w, h), guess.matrix.astype(np.float64), D,
            flags=cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_RATIONAL_MODEL)
        return rms, K, D.reshape(-1), errors.reshape(-1), spread.reshape(-1)

    rms, K, D, errors, spread = calibrate(objects, images)
    # A view the rest disagree with - a blurred one, a misread corner - is
    # dropped once and the fit made again without it.
    worst = errors > max(3.0 * float(np.median(errors)), 1.0)
    if worst.any() and (~worst).sum() >= 8:
        print("  dropped: " + ", ".join(f"{n} ({e:.2f} px)" for n, e, bad
                                        in zip(names, errors, worst) if bad))
        objects = [o for o, bad in zip(objects, worst) if not bad]
        images = [i for i, bad in zip(images, worst) if not bad]
        rms, K, D, errors, spread = calibrate(objects, images)
    f_sd = float(max(spread[0] / K[0, 0], spread[1] / K[1, 1]))
    if f_sd > cfg.top_lens_max_f_sd:
        print(f"  the views pin the focal length to +-{f_sd:.1%} only (f {K[0, 0]:.0f} px) - capture "
              "more with the board lifted off the desk and tilted 30-45 deg this way and that, "
              "near the camera and far")
        return 1
    entry = {
        "model": "rational",
        "size": [int(w), int(h)],
        "fx": round(float(K[0, 0]), 3), "fy": round(float(K[1, 1]), 3),
        "cx": round(float(K[0, 2]), 3), "cy": round(float(K[1, 2]), 3),
        "coeffs": [round(float(x), 6) for x in D[:TOP_COEFFS]],
        "rms_px": round(float(rms), 3),
        "views": len(objects),
        # The lens the views are of: what tells a take recorded through it
        # from one recorded through another (`pinkrab-umi resolve`).
        "unit": unit,
        "fitted": datetime.datetime.now().astimezone().date().isoformat(),
    }
    data = yaml.safe_load(CORE.lens_file.read_text()) if CORE.lens_file.is_file() else {}
    data = data or {}
    data[TOP] = entry
    _write_lens(data)
    print(f"  top camera: rms {rms:.3f} px over {len(objects)} views (worst {errors.max():.2f} px)")
    print(f"    f {K[0, 0]:.1f} / {K[1, 1]:.1f} px (+-{f_sd:.2%}), centre ({K[0, 2]:.1f}, {K[1, 2]:.1f}), "
          f"k {', '.join(f'{x:+.4f}' for x in D[:TOP_COEFFS])}")
    print(f"    it started from f {guess.fx:.1f} / {guess.fy:.1f}, centre ({guess.cx:.1f}, {guess.cy:.1f}): "
          f"f {(K[0, 0] / guess.fx - 1) * 100:+.2f} %, centre {K[0, 2] - guess.cx:+.1f}, {K[1, 2] - guess.cy:+.1f} px")
    print(f"  written: {CORE.lens_file} - the live view solves with it from its next start")
    return 0


def top_fit() -> tuple[Intrinsics, dict] | None:
    """lens.yaml's fit of the top camera - its lens, and the entry it was
    read from (the lens it is of, when) - or None when there is none or it
    cannot be used."""
    fit = fit_entry(CORE.lens_file)
    if fit is None:
        return None
    wrong = unusable(fit)
    if wrong:
        print(f"  lens.yaml top: {wrong} - {lens_command(TOP, 'capture')}, then {lens_command(TOP, 'fit')}",
              flush=True)
        return None
    width, height = fit["size"]
    measured = Intrinsics(width=int(width), height=int(height), fx=float(fit["fx"]),
                          fy=float(fit["fy"]), cx=float(fit["cx"]), cy=float(fit["cy"]),
                          coeffs=tuple(float(c) for c in fit["coeffs"]))
    return measured, fit


def top_source(fit: dict) -> str:
    """What a picture solved with this fit says it was solved with."""
    return f"lens.yaml top, fitted {fit.get('fitted')}"


def top_camera(cfg: Settings = SETTINGS) -> tuple[Intrinsics, str, str] | None:
    """The top camera's lens as fitted, which it is, and the lens the fit is
    of - or None while lens.yaml has no fit for the camera's picture."""
    found = top_fit()
    if found is None:
        return None
    measured, fit = found
    if (measured.width, measured.height) != (cfg.top_camera_width, cfg.top_camera_height):
        print(f"  lens.yaml top is fitted at {measured.width}x{measured.height}, the camera's picture is "
              f"{cfg.top_camera_width}x{cfg.top_camera_height} - fit it again", flush=True)
        return None
    return measured, top_source(fit), str(fit["unit"])
