"""`pinkrab-sim` - the UMI loop in MuJoCo.

    pinkrab-sim lens                         where the gripper camera must point, for each lens
    pinkrab-sim demos --count 60 --batch a   Hermit takes, in the recorder's format (--scene bowls: both hands)
    pinkrab-sim convert a --out sim-a        those takes into a LeRobot dataset
    pinkrab-sim evaluate <checkpoint>        the policy on the simulated arm, or both
    pinkrab-sim swap <checkpoint>            whether the policy looks at its pictures (pinkrab-picture-swap)
    pinkrab-sim oracle                       the arm following planned takes: what the grasp allows

Rendering is off screen (EGL on Linux, see config.py).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path


SCENES = ("blocks", "bowls")
# What a dataset of each scene says its task is, unless told.
TASKS = {"blocks": "pick and place the block", "bowls": "put each block in its bowl"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pinkrab-sim", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_lens = sub.add_parser("lens", help="the pitch and yaw that put the fingertips on the quarter line")
    p_lens.add_argument("--render", metavar="DIR", default=None,
                        help="also write what the camera sees at camera.yaml's pose and at the solved one")

    def scene(p) -> None:
        p.add_argument("--scene", default="blocks", choices=SCENES,
                       help="blocks: the cap onto the green piece; bowls: both hands, each its block into its bowl")

    p_demos = sub.add_parser("demos", help="Hermit takes the arm could have made")
    scene(p_demos)
    p_demos.add_argument("--count", type=int, required=True, help="how many takes to keep")
    p_demos.add_argument("--batch", required=True, help="name prefix: takes are <batch>-000, <batch>-001, ...")
    p_demos.add_argument("--seed", type=int, default=0)
    p_demos.add_argument("--viewpoint", default="robot", choices=("robot", "opposite"),
                         help="where the hand comes from: the robot's side, or across the table facing it")
    p_demos.add_argument("--tries", type=int, default=None, help="give up after this many plans")
    p_demos.add_argument("--two-faces", action="store_true",
                         help="blocks: each take grasps the cap by the nearest face or the next one, half each")
    p_demos.add_argument("--per-layout", type=int, default=1,
                         help="takes of one desk and start: with the blocks and --two-faces one by each face")
    p_demos.add_argument("--workers", type=int, default=1, help="processes rendering takes side by side")

    p_convert = sub.add_parser("convert", help="a batch of takes into a LeRobot dataset")
    p_convert.add_argument("batch", help="the batch name given to demos")
    p_convert.add_argument("--out", required=True, help="dataset name, under the workspace's datasets/")
    p_convert.add_argument("--task", default=None, help="the dataset's task (default: the scene's)")
    p_convert.add_argument("--gop", type=int, default=2,
                           help="keyframe spacing of the dataset video: 2 trains fast, 30 is small")
    p_convert.add_argument("--fps", type=int, default=None,
                           help="the dataset's frames a second (default: the core settings' dataset_fps)")

    p_eval = sub.add_parser("evaluate", help="the policy driving the simulated arm")
    p_eval.add_argument("checkpoint", help="a pretrained_model directory")
    scene(p_eval)
    p_eval.add_argument("--episodes", type=int, default=20)
    p_eval.add_argument("--seed", type=int, default=1000,
                        help="layouts are drawn from this; the demos' seeds are best left out of it")
    p_eval.add_argument("--start", default="demo", choices=("demo", "episode"),
                        help="demo: where the rig puts the arm (demo_start); episode: a recorded take's first pose")
    p_eval.add_argument("--seconds", type=float, default=12.0)
    p_eval.add_argument("--device", default=None)
    p_eval.add_argument("--videos", action="store_true", help="write each episode's video beside the report")
    p_eval.add_argument("--physics", action="store_true",
                        help="leave the grasp to MuJoCo's contact instead of carrying the cap once the jaw shuts on it")

    p_oracle = sub.add_parser("oracle", help="the arm following planned takes exactly: what the simulated grasp allows")
    scene(p_oracle)
    p_oracle.add_argument("--episodes", type=int, default=20)
    p_oracle.add_argument("--seed", type=int, default=2000)
    p_oracle.add_argument("--videos", action="store_true")
    p_oracle.add_argument("--physics", action="store_true")

    p_swap = sub.add_parser("swap", help="does the policy look at its picture: answers with pictures, states and commands swapped")
    p_swap.add_argument("checkpoint", help="a pretrained_model directory")
    p_swap.add_argument("--frames", type=int, default=150)
    p_swap.add_argument("--device", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "lens":
        return lens(args)

    if args.command == "demos":
        from .demos import DemoError, generate, generate_parallel

        try:
            if args.workers > 1:
                report = generate_parallel(args.count, args.batch, args.seed, args.workers,
                                           viewpoint=args.viewpoint, tries=args.tries, scene=args.scene,
                                           per_layout=args.per_layout, two_faces=args.two_faces)
            else:
                report = generate(args.count, args.batch, args.seed, viewpoint=args.viewpoint, tries=args.tries,
                                  scene=args.scene, per_layout=args.per_layout, two_faces=args.two_faces)
        except DemoError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"\n  kept {len(report['kept'])} of {report['tried']} planned, {report['seconds']} s, "
              f"in {report['episodes_dir']}")
        for reason, n in sorted(report["rejected"].items(), key=lambda kv: -kv[1]):
            print(f"    {n:4d}  {reason}")
        return 0

    if args.command == "convert":
        from pinkrab_umi import convert

        from .config import SETTINGS

        from .world import hands_of

        names = sorted(p.name for p in SETTINGS.episodes_dir.glob(f"{args.batch}-*") if (p / "meta.json").is_file())
        if not names:
            print(f"error: no takes named {args.batch}-* in {SETTINGS.episodes_dir}", file=sys.stderr)
            return 1
        # The hands of the batch's scene: both for the bowls, the one arm otherwise.
        scenes = {json.loads((SETTINGS.episodes_dir / n / "meta.json").read_text()).get("sim", {}).get("scene", "blocks")
                  for n in names}
        if len(scenes) > 1:
            print(f"error: the batch mixes scenes ({', '.join(sorted(scenes))})", file=sys.stderr)
            return 1
        scene = scenes.pop()
        try:
            report = convert.convert(names, args.out, args.task or TASKS[scene], hands=hands_of(scene),
                                     source=SETTINGS.episodes_dir, gop=args.gop, fps=args.fps)
        except convert.ConvertError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"\n  dataset: {report['dataset']}  ({len(names)} takes, {report['seconds']} s)")
        return 0

    if args.command == "evaluate":
        from .config import SETTINGS
        from .evaluate import EvaluateError, Evaluation

        checkpoint = Path(args.checkpoint).expanduser()
        stamp = time.strftime("%Y%m%d-%H%M%S")
        run = checkpoint.parents[2].name if len(checkpoint.parents) > 2 else "run"
        out = SETTINGS.root / "evals" / f"{run}{'-physics' if args.physics else ''}-{stamp}"
        try:
            evaluation = Evaluation(checkpoint, args.device, args.start, seconds=args.seconds, physics=args.physics,
                                    scene=args.scene)
        except EvaluateError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        try:
            summary = evaluation.run(args.episodes, args.seed, out if args.videos else None)
        except EvaluateError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        finally:
            evaluation.close()
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.json").write_text(json.dumps(summary, indent=2))
        print(f"\n  held {summary['held']:.0%}  lifted {summary['lifted']:.0%}  placed {summary['placed']:.0%}"
              f"  of {summary['episodes']}, {summary['seconds']} s")
        for hand, rates in (summary.get("hands") or {}).items():
            if rates.get("resting"):
                print(f"  {hand:5s}: at rest, wandered {rates['wander_mm']} mm")
            else:
                print(f"  {hand:5s}: held {rates['held']:.0%}  lifted {rates['lifted']:.0%}  placed {rates['placed']:.0%}")
        if "faces_held" in summary:
            print("  faces of the grasps   " + "  ".join(f"{k} {v}" for k, v in summary["faces_held"].items()))
            print("  faces, every episode  " + "  ".join(f"{k} {v}" for k, v in summary["faces_all"].items()))
        print(f"  report: {out / 'report.json'}")
        return 0

    if args.command == "oracle":
        from .config import SETTINGS
        from .evaluate import oracle

        out = SETTINGS.root / "evals" / (f"oracle{'-physics' if args.physics else ''}-"
                                         f"{time.strftime('%Y%m%d-%H%M%S')}")
        summary = oracle(args.episodes, args.seed, out if args.videos else None, physics=args.physics,
                         scene=args.scene)
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.json").write_text(json.dumps(summary, indent=2))
        print(f"\n  held {summary['held']:.0%}  lifted {summary['lifted']:.0%}  placed {summary['placed']:.0%}"
              f"  of {summary['episodes']}, {summary['seconds']} s")
        print(f"  report: {out / 'report.json'}")
        return 0

    if args.command == "swap":
        from pinkrab_policy.swap import picture_swap, report

        result = picture_swap(Path(args.checkpoint).expanduser(), args.frames, device=args.device)
        report(result)
        return 0 if result["looks_at_pictures"] else 1

    return 2


def lens(args) -> int:
    from .config import LENSES, measured_lenses
    from .lens import camera_spec, solve_pitch_yaw

    spec = camera_spec()
    print(f"  camera.yaml: pitch {math.degrees(spec['pitch']):.2f} deg, yaw {math.degrees(spec['yaw']):.2f} deg, "
          f"lens {spec['offset_out'] * 1000:.0f} mm out, {spec['offset_back'] * 1000:.0f} mm back")
    solved = {}
    for name, lens_spec in {**LENSES, **measured_lenses()}.items():
        pitch, yaw = solve_pitch_yaw(lens_spec, spec["offset_out"], spec["offset_back"])
        solved[name] = (pitch, yaw)
        print(f"  {name:12s} f {lens_spec.f:6.1f} px/rad, {math.degrees(lens_spec.fov):.0f} deg: "
              f"pitch {math.degrees(pitch):6.2f} deg, yaw {math.degrees(yaw):5.2f} deg")
    if args.render:
        render_lens(Path(args.render).expanduser(), solved)
    return 0


def render_lens(out: Path, solved: dict) -> None:
    """The Hermit over the cap, seen at camera.yaml's pose and at each solved one."""
    import cv2
    import numpy as np

    from .config import SETTINGS
    from .demos import tool_rotation
    from .world import Layout, World, hermit

    out.mkdir(parents=True, exist_ok=True)
    T = np.eye(4)
    side = 1.0 if SETTINGS.arm == "left" else -1.0
    T[:3, :3] = tool_rotation(side * 0.3, math.radians(62))
    T[:3, 3] = [0.30, side * 0.16, 0.05]
    poses = {"camera_yaml": (None, None), **{f"solved_{k}": v for k, v in solved.items()
                                              if k == SETTINGS.lens}}
    for label, (pitch, yaw) in poses.items():
        world = World(arms=(), hermits=(SETTINGS.arm,), pitch=pitch, yaw=yaw)
        try:
            world.lay_out(Layout({"cap": (0.34, side * 0.17, side * 0.2), "base": (0.36, side * 0.27, side * -0.1)}))
            world.hold_hermit(SETTINGS.arm, T, 0.0)
            world.forward()
            picture = world.gripper_image(hermit(SETTINGS.arm))
            h, w = picture.shape[:2]
            cv2.line(picture, (w // 2, 0), (w // 2, h), (255, 255, 0), 1)
            cv2.line(picture, (0, int(h * 0.75)), (w, int(h * 0.75)), (255, 255, 0), 1)
            path = out / f"{label}.jpg"
            cv2.imwrite(str(path), cv2.cvtColor(picture, cv2.COLOR_RGB2BGR))
            print(f"  wrote {path}")
        finally:
            world.close()


if __name__ == "__main__":
    sys.exit(main())
