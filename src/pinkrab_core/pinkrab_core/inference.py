"""How a trained policy is run: the settings that change inference without
retraining, kept in the workspace's inference.yaml - one section per kind of
policy, and one every kind shares.

A checkpoint carries the configuration it was trained with. Some of it is the
model - the length of its chunks, its layers - and the weights fix it. Some is
only how the weights are asked: how many steps of a chunk run before the
camera is looked at again, whether ACT's temporal ensembling blends the chunks
that cover a step. Those are listed here per policy type (KINDS). Around any policy is how Pinkrab drives with it - the device,
how fast the gate lets a joint turn, when a step is a bad answer and when the
run ends - listed once (COMMON), its defaults the core settings' policy_*.

Each has Pinkrab's own value where it has one, and the file holds what this
rig has chosen. The studio's policy page shows the file and writes it;
whatever loads a checkpoint (`pinkrab_policy.rollout.Brain`), and the gate
(`policy_dps`), read it when they start.

A policy type not listed runs as it was trained, but for a chunk: one with
n_action_steps has it cut to the core settings' policy_action_s like the
others, never run whole with the camera ignored. Another type is one more
entry in KINDS.

A step is one of the checkpoint's dataset's frames, so how many steps a
chunk runs is said in seconds and counted at that dataset's rate (`fps`);
Pinkrab's own defaults are seconds for that reason.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import yaml

from .config import SETTINGS
from .dataset import steps_in

# The file's section every kind of policy shares.
COMMON = "common"


class InferenceError(ValueError):
    """Settings that do not fit the checkpoint, or a file that does not parse."""


@dataclass(frozen=True)
class Knob:
    name: str                   # a field in the policy's own config, or one of COMMON_KNOBS
    kind: str                   # "int", "float", "bool" or "choice"
    help: str
    # Pinkrab's own value when the file names none; None runs it as trained.
    # A callable is given the checkpoint's dataset rate (fps) and answers it.
    default: object = None
    # May be None: off, or left to the policy.
    optional: bool = False
    choices: tuple[str, ...] = ()
    # The smallest and largest it can be: a number, or from a checkpoint's config.
    minimum: float | Callable[[object], float] | None = None
    maximum: float | Callable[[object], float] | None = None


@dataclass(frozen=True)
class Kind:
    label: str
    knobs: tuple[Knob, ...]
    # Trained fields shown beside the knobs, unchangeable: what bounds them.
    shown: tuple[str, ...] = ()
    # What is wrong between knobs, a sentence each.
    check: Callable[[dict], list[str]] = lambda values: []


def _act_check(values: dict) -> list[str]:
    if values.get("temporal_ensemble_coeff") is not None and values.get("n_action_steps") != 1:
        return ["temporal ensembling blends a fresh chunk into every step, so n_action_steps must be 1 with it"]
    return []


KINDS: dict[str, Kind] = {
    "act": Kind(
        "ACT",
        (
            Knob("n_action_steps", "int",
                 "steps of each predicted chunk run before the camera is looked at again: "
                 "a tenth of a second's worth is a new look ten times a second, the chunk's length runs it whole",
                 default=lambda fps: steps_in(SETTINGS.policy_action_s, fps), minimum=1,
                 maximum=lambda cfg: cfg.chunk_size),
            Knob("temporal_ensemble_coeff", "float",
                 "blends every chunk that covers the step, weighted exp(-coeff x i) counting from the "
                 "oldest: positive leans on older chunks (0.01 is the ACT paper's), 0 weighs them alike, "
                 "negative favours the newest. Off runs one chunk at a time. Needs n_action_steps 1",
                 # No more negative than keeps exp(-coeff x i) within float32 over the
                 # whole chunk: past it the weights are infinite and the action NaN.
                 optional=True, minimum=lambda cfg: -80.0 / max(1, cfg.chunk_size - 1)),
        ),
        shown=("chunk_size", "n_obs_steps", "vision_backbone"),
        check=_act_check,
    ),
}

COMMON_KNOBS: tuple[Knob, ...] = (
    Knob("device", "choice",
         "where the policy runs: auto takes CUDA, else Apple's Metal (mps), else the CPU",
         default="auto", choices=("auto", "cuda", "mps", "cpu")),
    Knob("use_amp", "bool",
         "mixed precision: the policy's arithmetic in half precision where the device allows - "
         "faster, a little rougher. Not with temporal ensembling, which would overflow in it",
         default=False),
    Knob("max_joint_dps", "float",
         "how fast the gate lets a joint turn while a policy drives, deg/s. "
         "The gate reads it each time a policy takes the arms",
         # No faster than the ramp - or bounded by nothing more when the ramp
         # is off (0); policy_dps holds it to the ramp either way.
         default=SETTINGS.policy_max_joint_dps, minimum=1.0, maximum=SETTINGS.max_joint_speed_dps or None),
    Knob("smoothing", "bool",
         "the gate shapes a policy's commands: each joint's speed, acceleration and jerk held under the "
         "limits below, the arm moved on between commands - no jolt where one chunk meets the next. "
         "Off sends each command on as it comes, under max_joint_dps alone. Read each time a policy takes the arms",
         default=True),
    Knob("max_joint_accel_dps2", "float",
         "with smoothing, how fast a joint's speed may change, deg/s^2",
         default=SETTINGS.policy_max_joint_accel_dps2, minimum=10.0, maximum=10000.0),
    Knob("max_joint_jerk_dps3", "float",
         "with smoothing, how fast a joint's acceleration may change, deg/s^3: lower rounds every start and stop more",
         default=SETTINGS.policy_max_joint_jerk_dps3, minimum=100.0, maximum=200000.0),
    Knob("orientation_weight", "float",
         "how much the arm's solve weighs the tool's turn against its place: "
         "lower lets the wrist off the turn first",
         default=SETTINGS.policy_orientation_weight, minimum=0.0, maximum=10.0),
    Knob("max_tool_speed_mm_s", "float",
         "a step asked for faster than this is taken for a bad answer and held, mm/s",
         default=SETTINGS.policy_max_tool_speed_mm_s, minimum=30.0, maximum=6000.0),
    Knob("max_tool_turn_dps", "float",
         "a turn asked for faster than this is taken for a bad answer and held, deg/s",
         default=SETTINGS.policy_max_tool_turn_dps, minimum=30.0, maximum=2700.0),
    # Under what one solve can turn a joint (max_joint_speed_dps over a tick
    # and track()'s three solves, 3.3 deg at 30 fps): the margin is a
    # bound on the solve itself, and a joint read further outside it than one
    # solve turns leaves that solve no answer.
    Knob("limit_margin_deg", "float",
         "the solve keeps every joint this far inside its band; held there, the run goes on until what it asks past adds up past the slip",
         default=SETTINGS.policy_limit_margin_deg, minimum=0.0, maximum=3.0),
    Knob("slip_mm", "float",
         "how far one tick's solve may fall short of the place asked for before the run ends - and the misses summed at a stop, mm",
         default=SETTINGS.policy_slip_mm, minimum=1.0, maximum=200.0),
    Knob("slip_deg", "float",
         "how far one tick's solve may fall short of the turn asked for before the run ends - and the misses summed at a stop, deg",
         default=SETTINGS.policy_slip_deg, minimum=1.0, maximum=90.0),
    Knob("ready_tol_deg", "float",
         "how close every joint must come to the ready pose before the policy takes the arm",
         default=SETTINGS.policy_ready_tol_deg, minimum=0.5, maximum=15.0),
)


def trained(checkpoint: Path) -> SimpleNamespace:
    """A checkpoint's config as it was trained, read from its config.json:
    enough for the knobs and their bounds without loading LeRobot - or torch
    - to look. With it as `fps` the rate it is asked at, its dataset's - None
    when that dataset is not here to say."""
    from .demo_start import DemoStartError, checkpoint_fps

    try:
        fields = json.loads((Path(checkpoint) / "config.json").read_text())
    except (OSError, ValueError) as exc:
        raise InferenceError(f"{checkpoint} is not a checkpoint: {exc}") from exc
    if not isinstance(fields, dict) or "type" not in fields:
        raise InferenceError(f"{checkpoint}/config.json names no policy type")
    try:
        fps = checkpoint_fps(Path(checkpoint))
    except (DemoStartError, OSError, ValueError, KeyError):
        fps = None
    return SimpleNamespace(**fields, fps=fps)


def load() -> dict[str, dict]:
    """The file as written: section -> {knob: value}. Empty when there is none."""
    path = SETTINGS.inference_file
    if not path.is_file():
        return {}
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise InferenceError(f"{path} does not parse: {exc}") from exc
    if not isinstance(data, dict):
        raise InferenceError(f"{path} is not a mapping of sections")
    return {str(k): dict(v) for k, v in data.items() if isinstance(v, dict)}


def _bound(limit, cfg) -> float | None:
    return limit(cfg) if callable(limit) else limit


def _wrong(knob: Knob, value, cfg=None) -> str | None:
    """Why `value` cannot be `knob`'s, or None."""
    if value is None:
        # Only one that may be off is: a None handed to the policy for any
        # other is a policy that will not build.
        return None if knob.optional else f"{knob.name} needs a value"
    if knob.kind == "choice":
        return None if value in knob.choices else f"{knob.name} is one of {', '.join(knob.choices)}, not {value!r}"
    if knob.kind == "bool":
        return None if isinstance(value, bool) else f"{knob.name} is true or false, not {value!r}"
    number_types = (int,) if knob.kind == "int" else (int, float)
    if isinstance(value, bool) or not isinstance(value, number_types):
        return f"{knob.name} is {'a whole number' if knob.kind == 'int' else 'a number'}, not {value!r}"
    # NaN passes every comparison below, and a limit of NaN or infinity is a
    # limit that never trips.
    if not math.isfinite(value):
        return f"{knob.name} is a finite number, not {value!r}"
    top, low = _bound(knob.maximum, cfg), _bound(knob.minimum, cfg)
    if low is not None and value < low:
        return f"{knob.name} is at least {low:g}" + (" for this checkpoint" if callable(knob.minimum) else "")
    if top is not None and value > top:
        return f"{knob.name} is at most {top:g}" + (" for this checkpoint" if callable(knob.maximum) else "")
    return None


def _default(knob: Knob, fps: float):
    return knob.default(fps) if callable(knob.default) else knob.default


def _fps(cfg, fps: float | None) -> float:
    """The rate the knobs are counted at: `fps`, else the one `trained` read
    off the checkpoint's dataset, else the dataset rate convert makes."""
    return fps or getattr(cfg, "fps", None) or SETTINGS.dataset_fps


def chosen(cfg, saved: dict | None = None, fps: float | None = None) -> dict:
    """What a checkpoint of `cfg` runs with, knob by knob: the file's value,
    else Pinkrab's own (no more than the checkpoint allows), else as trained.
    Pinkrab's own are counted in steps at `fps`."""
    fps = _fps(cfg, fps)
    kind = KINDS.get(cfg.type)
    if kind is None:
        steps = getattr(cfg, "n_action_steps", None)
        if isinstance(steps, int) and not isinstance(steps, bool):
            return {"n_action_steps": min(steps_in(SETTINGS.policy_action_s, fps), steps)}
        return {}
    saved = (load() if saved is None else saved).get(cfg.type, {})
    values, defaulted = {}, []
    for knob in kind.knobs:
        if knob.name in saved:
            values[knob.name] = saved[knob.name]
        elif knob.default is not None:
            top = _bound(knob.maximum, cfg)
            default = _default(knob, fps)
            values[knob.name] = default if top is None else min(default, top)
            defaulted.append(knob.name)
        else:
            values[knob.name] = getattr(cfg, knob.name, None)
    # Pinkrab's own values give way to the checkpoint's where they do not go
    # with the rest - three action steps beside an ensembler it was trained with.
    if defaulted and kind.check(values):
        for name in defaulted:
            values[name] = getattr(cfg, name, values[name])
    return values


def problems(cfg, values: dict) -> list[str]:
    """Why `values` cannot run a checkpoint of `cfg`, a sentence each."""
    kind = KINDS.get(cfg.type)
    if kind is None:
        return []
    wrong = [w for k in kind.knobs if (w := _wrong(k, values.get(k.name), cfg))]
    return wrong + kind.check(values)


def apply(cfg, fps: float) -> dict:
    """Set the chosen values on `cfg` - before the policy is built from it,
    which is where it reads them - and return them, Pinkrab's own counted at
    the checkpoint's rate `fps`. Raises InferenceError when they do not fit
    this checkpoint."""
    values = chosen(cfg, fps=fps)
    wrong = problems(cfg, values)
    if wrong:
        raise InferenceError(f"inference settings for {cfg.type}: " + "; ".join(wrong))
    for name, value in values.items():
        setattr(cfg, name, value)
    return values


def _common_chosen(saved: dict) -> dict:
    mine = saved.get(COMMON, {})
    return {k.name: mine.get(k.name, k.default) for k in COMMON_KNOBS}


def common() -> dict:
    """How Pinkrab drives with any policy here: the file's common section over
    the core settings' defaults. Raises InferenceError when it does not hold."""
    values = _common_chosen(load())
    wrong = [w for k in COMMON_KNOBS if (w := _wrong(k, values[k.name]))]
    if wrong:
        raise InferenceError("common inference settings: " + "; ".join(wrong))
    return values


def policy_dps(ramp_dps: float, values: dict | None = None) -> float:
    """The speed a policy's commands are walked at, under a ramp of `ramp_dps`
    (0 for none): the common max_joint_dps, or the ramp when slower."""
    top = float((values or common())["max_joint_dps"])
    return top if ramp_dps <= 0 else min(ramp_dps, top)


def _view(knob: Knob, cfg, section: dict, using, fps: float) -> dict:
    view = {"name": knob.name, "kind": knob.kind, "help": knob.help, "optional": knob.optional,
            "choices": list(knob.choices), "minimum": _bound(knob.minimum, cfg), "maximum": _bound(knob.maximum, cfg),
            "default": _default(knob, fps), "saved": knob.name in section, "using": using}
    if cfg is not None:
        view["trained"] = getattr(cfg, knob.name, None)
    return view


def describe(cfg) -> dict:
    """Everything the studio's page shows for a checkpoint of `cfg`: its
    kind's settings beside what it was trained with, and the common ones -
    and the rate it is asked at (`fps`, None when its dataset is not here)."""
    saved = load()
    used = _common_chosen(saved)
    fps = _fps(cfg, None)
    out = {"type": cfg.type, "known": cfg.type in KINDS, "knobs": [], "shown": {}, "file": str(SETTINGS.inference_file),
           "fps": getattr(cfg, "fps", None), "using": chosen(cfg, saved),
           "common": [_view(k, None, saved.get(COMMON, {}), used[k.name], fps) for k in COMMON_KNOBS],
           "problems": [w for k in COMMON_KNOBS if (w := _wrong(k, used[k.name]))]}
    kind = KINDS.get(cfg.type)
    if kind is not None:
        using = chosen(cfg, saved)
        out |= {"label": kind.label,
                "knobs": [_view(k, cfg, saved.get(cfg.type, {}), using[k.name], fps) for k in kind.knobs],
                "shown": {name: getattr(cfg, name, None) for name in kind.shown}}
        out["problems"] += problems(cfg, using)
    return out


def save(cfg, values: dict, common_values: dict | None = None) -> None:
    """Make `values` what the file holds for checkpoints of `cfg`'s type, and
    `common_values` (when given) its common section: the knobs named are set,
    the others left to Pinkrab's own or the checkpoint's. Checked against this
    checkpoint first; other types' sections are kept."""
    path = SETTINGS.inference_file
    kind = KINDS.get(cfg.type)
    if kind is None and values:
        raise InferenceError(f"no inference settings are known for {cfg.type!r}: it runs as trained")
    data = load()
    for section, given, knobs in ((cfg.type, values, kind.knobs if kind else ()),
                                  (COMMON, common_values, COMMON_KNOBS)):
        if given is None:
            continue
        unknown = sorted(set(given) - {k.name for k in knobs})
        if unknown:
            raise InferenceError(f"{section} has no setting {', '.join(unknown)}")
        data[section] = dict(given)
    wrong = problems(cfg, chosen(cfg, data))
    used = _common_chosen(data)
    wrong += [w for k in COMMON_KNOBS if (w := _wrong(k, used[k.name]))]
    if wrong:
        raise InferenceError("; ".join(wrong))
    data = {section: given for section, given in data.items() if given}
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(".yaml.part")
    part.write_text("# How trained policies are run here - written by the studio's policy page. A\n"
                    "# setting not named runs as Pinkrab's own, or as the checkpoint was trained.\n"
                    + yaml.safe_dump(data, sort_keys=False))
    os.replace(part, path)
