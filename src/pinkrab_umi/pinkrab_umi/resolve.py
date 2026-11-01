"""`pinkrab-umi resolve` - solve recorded takes again with the top camera's fitted lens.

A take's poses are solved as it records, with the lens fit the live view had
then. A fit made again of the same lens - more views, a board held more ways -
is a better lens than the one the takes before it were solved with, and a
lens's error is a bias that moved a fingertip on the desk by up to 6 mm across
the workspace (measured 2026-10-04). A take keeps, in each frame's pose line,
the corners of every marker read in that frame (`markers`), so its poses are
made again from them: every frame through the live view's own solving, filter
and all, with the new fit - the corners as the camera's own frame gave them.

A take recorded before the corners were kept has its top-camera video
instead, and is solved again from that. The video is compressed, and a pose
solved from it differs from the one the live view solved from the camera's
own frame by about a millimetre (median, 4 mm at the 95th percentile,
measured on takes recorded with the fit).

The poses as recorded are kept beside the new ones (poses.recorded.jsonl), and
meta.json says what the take is now solved with.
"""
from __future__ import annotations

import datetime
import json
import shutil
from pathlib import Path

import cv2

from pinkrab_core.top_lens import left_by_resolve, same_unit, solved_with, stale

from . import lens, record
from .config import SETTINGS
from .pnp import Intrinsics

RECORDED = "poses.recorded.jsonl"


def _skipped(meta_file: Path, meta: dict, fit: dict, solids: dict[str, float], why: str) -> str:
    """Leave a take as it is, and say so in it: a take through the fitted lens that
    this fit cannot solve again is not counted as waiting for it (top_lens.stale)
    - until the solids looked for change, when resolve tries it again."""
    meta["resolve_skipped"] = {"fitted": fit.get("fitted"), "unit": str(fit.get("unit")),
                               "solids": solids, "why": why}
    tmp = meta_file.with_name(meta_file.name + ".tmp")
    tmp.write_text(json.dumps(meta, indent=2) + "\n")
    tmp.replace(meta_file)
    return f"skipped - {why}"


def resolve_take(root: Path, measured: Intrinsics, fit: dict) -> str:
    """Solve one take again with the fitted lens; what was done, in a few words."""
    from .live import Frame, Live

    meta_file = root / "meta.json"
    meta = json.loads(meta_file.read_text())
    calibration = meta.get("calibration") or {}
    # A fit is for one picture size, of one lens; a take of another size or
    # through another lens stays as it was solved.
    size = calibration.get("stream") or {}
    if (size.get("width"), size.get("height")) != (measured.width, measured.height):
        return (f"skipped - recorded at {size.get('width')}x{size.get('height')}, the lens is fitted "
                f"at {measured.width}x{measured.height}")
    if not same_unit(calibration, fit):
        return "skipped - not recorded through the lens this fit is of"
    if solved_with(calibration, fit):
        return "already solved with it"
    # The stream is the take's, as it was recorded.
    today = datetime.datetime.now().astimezone().date().isoformat()
    lines = [json.loads(line) for line in (root / "poses.jsonl").read_text().splitlines() if line.strip()]
    from_corners = bool(lines) and all("markers" in line for line in lines)
    source = "the corners kept" if from_corners else "head.mp4"
    solved = record.calibration_dict(measured, f"{lens.top_source(fit)}, solved again from {source} {today}",
                                     unit=str(fit["unit"]))
    solved["stream"] = calibration.get("stream", solved["stream"])
    live = Live(None, solved)
    # The take is solved again by that live view, with the solids it looks
    # for today: the same desk solid at the same size, and the same hand
    # solids at the same sizes - or not at all. A take that does not say its
    # hand solids' sizes could have been recorded with any.
    sizes = live.sizes()
    desk = meta.get("desk") or {}
    if desk.get("solid") != live.desk or abs(float(desk.get("face_mm") or 0.0) - sizes[live.desk]) > 1e-6:
        return _skipped(meta_file, meta, fit, sizes,
                        f"recorded with the desk solid {desk.get('solid')} ({desk.get('face_mm')} mm), "
                        f"not {live.desk} ({sizes[live.desk]:g} mm)")
    hands = {name: mm for name, mm in sizes.items() if name != live.desk}
    looked_for = set().union(*((line.get("solids") or {}).keys() for line in lines)) if lines else set()
    if looked_for != set(hands):
        return _skipped(meta_file, meta, fit, sizes,
                        f"recorded looking for {sorted(looked_for)}, the live view now for {sorted(hands)}")
    if not meta.get("hands"):
        return _skipped(meta_file, meta, fit, sizes, "recorded without its hand solids' sizes")
    changed = sorted(n for n, mm in meta["hands"].items() if abs(float(mm) - hands.get(n, -1.0)) > 1e-6)
    if changed:
        return _skipped(meta_file, meta, fit, sizes, f"recorded with other sizes of {', '.join(changed)} than today's")
    if from_corners:
        out = []
        for line in lines:
            live.consume_markers(line.get("stamp"), line["markers"])
            out.append({"host_ts": line.get("host_ts"), **live.pose_line(), "jaw": line.get("jaw") or {},
                        "jaw_ts": line.get("jaw_ts") or {}, "markers": line["markers"]})
        return _write(root, meta, meta_file, calibration, solved, out, 0)
    # Which picture each pose line is: the recorder gives a frame's pose line
    # and its picture the same host clock, and frames.jsonl says where in the
    # video each picture landed. A picture the encoder dropped, or one queued
    # as the take stopped, never landed: that line keeps its recorded pose, as
    # does one whose picture will not solve.
    at = {}
    for raw in (root / "frames.jsonl").read_text().splitlines():
        frame = json.loads(raw) if raw.strip() else {}
        if frame.get("stream") == "head":
            at[frame["host_ts"]] = int(frame["index"])
    video = cv2.VideoCapture(str(root / "head.mp4"))
    if not video.isOpened():
        return _skipped(meta_file, meta, fit, sizes, "head.mp4 does not open")
    out, as_recorded, position, picture, unsolved = [], 0, -1, None, None
    try:
        for line in lines:
            index = at.get(line.get("host_ts"))
            if index is None or index <= position:
                out.append(line)
                as_recorded += 1
                continue
            while position < index:
                ok, picture = video.read()
                position += 1
                if not ok:
                    return _skipped(meta_file, meta, fit, sizes,
                                    f"head.mp4 ends at frame {position}, frames.jsonl names {index}")
            if picture.shape[1::-1] != (measured.width, measured.height):
                return _skipped(meta_file, meta, fit, sizes,
                                f"head.mp4 is {picture.shape[1]}x{picture.shape[0]}, not the fitted size")
            # One picture that will not solve is not the end of the take, as
            # it is not of the live view: that line keeps its recorded pose.
            try:
                live.consume(Frame(picture=picture, sensor_ts=line.get("stamp")))
            except Exception as exc:  # noqa: BLE001
                unsolved = unsolved or f"{type(exc).__name__}: {exc}"
                out.append(line)
                as_recorded += 1
                continue
            out.append({"host_ts": line.get("host_ts"), **live.pose_line(), "jaw": line.get("jaw") or {},
                        "jaw_ts": line.get("jaw_ts") or {}})
    finally:
        video.release()
    if as_recorded == len(out):
        # Nothing solved again: written as it is, the take would say it was
        # and never be tried again. Pictures that all failed are a fault to
        # be said, not a take to pass over.
        if unsolved is not None:
            raise RuntimeError(f"no picture would solve - the first: {unsolved}")
        return _skipped(meta_file, meta, fit, sizes, "no picture of head.mp4 pairs with a pose line")
    return _write(root, meta, meta_file, calibration, solved, out, as_recorded)


def _write(root: Path, meta: dict, meta_file: Path, calibration: dict, solved: dict, out: list[dict],
           as_recorded: int) -> str:
    """A take's poses solved again written in place of its own, which are kept
    aside once (RECORDED); what was done, in a few words."""
    if as_recorded:
        solved["camera_source"] += f" ({as_recorded} frame(s) kept as recorded)"
    # The new poses are written aside first, the recorded ones copied away
    # once - what the camera's own frames gave - and only then is poses.jsonl
    # replaced, so it is never missing.
    tmp = root / "poses.jsonl.tmp"
    tmp.write_text("".join(json.dumps(line) + "\n" for line in out))
    kept = root / RECORDED
    if not kept.exists():
        copy = root / (RECORDED + ".tmp")
        shutil.copy2(root / "poses.jsonl", copy)
        copy.replace(kept)
    tmp.replace(root / "poses.jsonl")
    meta.setdefault("calibration_recorded", calibration)
    meta["calibration"] = solved
    meta.pop("resolve_skipped", None)   # a skip noted before no longer holds
    tmp = root / "meta.json.tmp"
    tmp.write_text(json.dumps(meta, indent=2) + "\n")
    tmp.replace(meta_file)
    placed = sum(1 for line in out if any(c.get("world_pose") is not None for c in line["solids"].values()))
    return (f"solved again: {len(out)} frames, {placed} with a hand on the desk"
            + (f", {as_recorded} kept as recorded (no picture, or would not solve)" if as_recorded else ""))


def resolve(names: list[str] | None = None) -> int:
    """Every take through the fitted lens not yet solved with its fit - or the
    named ones, each of which is checked the same way - solved again with it."""
    from .live import Live

    found = lens.top_fit()
    if found is None:
        print(f"  no usable fit of the top camera in lens.yaml - {lens.lens_command(lens.TOP, 'capture')} "
              f"and {lens.lens_command(lens.TOP, 'fit')} first")
        return 1
    measured, fit = found
    # The waiting takes, and those left before against other solids than today's.
    today = Live(None, record.calibration_dict(measured, lens.top_source(fit), unit=str(fit["unit"]))).sizes()
    left = left_by_resolve(SETTINGS.episodes_dir, fit)
    retry = [n for n, note in left.items() if note.get("solids") != today]
    takes = names or stale(SETTINGS.episodes_dir, fit) + retry
    if not takes:
        if left:
            print(f"  nothing to solve again: {len(left)} take(s) through the lens {fit.get('unit')} were left "
                  "before against today's solids (resolve_skipped in each says why; name one to try again)")
        else:
            print(f"  every take through the lens {fit.get('unit')} is solved with its fit")
        return 0
    failed = 0
    for name in takes:
        root = SETTINGS.episodes_dir / name
        if not (root / "meta.json").is_file():
            print(f"  {name}: no such complete take")
            failed += 1
            continue
        try:
            said = resolve_take(root, measured, fit)
        except Exception as exc:  # noqa: BLE001 - one take that fails must not stop the rest
            said = f"failed - {type(exc).__name__}: {exc}"
            failed += 1
        print(f"  {name}: {said}", flush=True)
    return 1 if failed else 0
