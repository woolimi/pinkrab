"""The top camera's fitted lens, as lens.yaml keeps it, and whether a take is solved with it.

The fit is made by `pinkrab-umi lens fit top` and the takes are solved again by
`pinkrab-umi resolve`; both are pinkrab_umi's. What is here is what the setup
checklist asks as well - whether there is a fit, and which recorded takes are
not solved with it - so it sits below both.
"""
from __future__ import annotations

import json
from pathlib import Path

NUMBERS = ("fx", "fy", "cx", "cy")
# The distortion a fit states: the rational model's eight coefficients.
COEFFS = 8


def fit_entry(lens_file: Path) -> dict | None:
    """lens.yaml's `top` entry as written, or None."""
    try:
        import yaml

        data = yaml.safe_load(lens_file.read_text()) if lens_file.is_file() else {}
    except Exception:  # noqa: BLE001 - an unreadable file is no fit
        return None
    fit = (data or {}).get("top") if isinstance(data, dict) else None
    return fit if isinstance(fit, dict) else None


def unusable(fit: dict) -> str | None:
    """Why a fit cannot be solved with or matched to takes, or None: the lens
    it states, and the lens it names, which a take is matched to it by."""
    try:
        width, height = fit["size"]
        int(width), int(height)
        [float(fit[k]) for k in NUMBERS]
        if len([float(c) for c in fit["coeffs"]]) != COEFFS:
            return f"the fit's coefficients are not {COEFFS} - a fit of another camera"
    except (KeyError, TypeError, ValueError) as exc:
        return f"the fit cannot be read ({type(exc).__name__}: {exc})"
    if not fit.get("unit"):
        return "the fit names no lens - a fit of another camera"
    return None


def same_unit(calibration: dict, fit: dict) -> bool:
    """Whether a take was recorded through the lens the fit is of, as the
    fit it was solved with named it."""
    return bool(fit.get("unit")) and str(calibration.get("camera_unit")) == str(fit.get("unit"))


def solved_with(calibration: dict, fit: dict) -> bool:
    """Whether a take's poses are solved with this fit."""
    camera = calibration.get("camera") or {}
    try:
        return (all(abs(float(camera[k]) - float(fit[k])) < 1e-6 for k in NUMBERS)
                and len(camera["coeffs"]) == len(fit["coeffs"])
                and all(abs(float(a) - float(b)) < 1e-9 for a, b in zip(camera["coeffs"], fit["coeffs"])))
    except (KeyError, TypeError, ValueError):
        return False


def left_by_resolve(episodes_dir: Path, fit: dict) -> dict[str, dict]:
    """The takes `pinkrab-umi resolve` found it could not solve again with
    this fit, by name, each with its note: why, and the solids it judged by."""
    out = {}
    if not episodes_dir.is_dir():
        return out
    for root in sorted(episodes_dir.iterdir()):
        try:
            meta = json.loads((root / "meta.json").read_text())
        except (OSError, ValueError):
            continue
        note = meta.get("resolve_skipped") or {}
        if (note.get("fitted") == fit.get("fitted") and note.get("unit") == str(fit.get("unit"))
                and not solved_with(meta.get("calibration") or {}, fit)):
            out[root.name] = note
    return out


def stale(episodes_dir: Path, fit: dict) -> list[str]:
    """The recorded takes through the fit's lens, at its picture size, that are
    not solved with it - less those `pinkrab-umi resolve` has already found
    it cannot solve again with this fit (`left_by_resolve`)."""
    out = []
    if not episodes_dir.is_dir():
        return out
    for root in sorted(episodes_dir.iterdir()):
        try:
            meta = json.loads((root / "meta.json").read_text())
        except (OSError, ValueError):
            continue
        skipped = meta.get("resolve_skipped") or {}
        if skipped.get("fitted") == fit.get("fitted") and skipped.get("unit") == str(fit.get("unit")):
            continue
        calibration = meta.get("calibration") or {}
        stream = calibration.get("stream") or {}
        if (same_unit(calibration, fit) and not solved_with(calibration, fit)
                and [stream.get("width"), stream.get("height")] == list(fit.get("size") or [])):
            out.append(root.name)
    return out
