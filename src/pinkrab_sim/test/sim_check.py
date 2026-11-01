"""Offline checks of the UMI loop in MuJoCo: the camera, the Hermit, and a take through convert.

    python src/pinkrab_sim/test/sim_check.py

Needs mujoco with EGL, placo and lerobot; no servo, no ROS, no camera. The
last check writes one take and one dataset under the workspace's sim/ and
datasets/, and removes both.
"""
from __future__ import annotations

import json
import math
import shutil
import sys

import numpy as np
import pinkrab_sim  # noqa: F401 - its config picks MuJoCo's off-screen backend

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}" + (f" - {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


def lens() -> None:
    from pinkrab_sim.config import LENSES
    from pinkrab_sim.lens import Fisheye, camera_spec, solve_pitch_yaw

    print("the lens")
    spec = camera_spec()
    pitch, yaw = solve_pitch_yaw(LENSES["ov2710"], spec["offset_out"], spec["offset_back"])
    check("camera.yaml's pitch and yaw come back from its own lens",
          abs(pitch - spec["pitch"]) < math.radians(0.1) and abs(yaw - spec["yaw"]) < math.radians(0.1),
          f"pitch {math.degrees(pitch):.2f} vs {math.degrees(spec['pitch']):.2f}, "
          f"yaw {math.degrees(yaw):.2f} vs {math.degrees(spec['yaw']):.2f} deg")
    fisheye = Fisheye(LENSES["ar0234_200"], 0.5)
    h, w = fisheye.height, fisheye.width
    n = fisheye.face
    cx, cy = fisheye.map_x[h // 2, w // 2], fisheye.map_y[h // 2, w // 2]
    check("the image centre looks down the front face's centre",
          abs(cx - (n - 1) / 2) < 1.5 and abs(cy - (n - 1) / 2) < 1.5, f"({cx:.1f}, {cy:.1f}) of {n}")
    check("nothing is drawn outside the image circle", fisheye.map_x[0, 0] < 0 and fisheye.map_x[h // 2, 0] < 0)


def states() -> None:
    from pinkrab_core.dataset import NO_STEP, relative_pose, state_names, state_vector

    print("the state")
    check("one hand: the opening, with its pad",
          np.allclose(state_vector(state_names(("left",)), {"left": 0.3}), [0.3, 0.0]))
    T_left, T_right = np.eye(4), np.eye(4)
    T_left[:3, 3] = [0.1, 0.2, 0.3]
    T_right[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    T_right[:3, 3] = [0.1, -0.2, 0.3]
    relative = relative_pose(T_left, T_right)
    two = state_names(("left", "right"))
    got = state_vector(two, {"left": 0.3, "right": 0.6}, relative)
    check("two hands: both openings, then the right tool in the left's, no pad",
          two[:2] == ["left_gripper", "right_gripper"] and len(got) == 11 and np.allclose(got[:2], [0.3, 0.6])
          and np.allclose(got[2:5], [0.0, -0.4, 0.0]) and np.allclose(got[5:], [0, 1, 0, -1, 0, 0]))
    check("a hand on the other's tool is no step from it", np.allclose(relative_pose(T_left, T_left), NO_STEP))


def camera_on_bracket() -> None:
    import mujoco
    from pinkrab_sim.world import build_spec, gripper_camera_in_link, pose_of

    print("the gripper camera, in its bracket")
    spec, _ = build_spec(arms=("left", "right"), hermits=())
    model = spec.compile()
    for arm in ("left", "right"):
        cam = gripper_camera_in_link(spec, arm, None, None)
        bracket = model.geom(f"camera_bracket_{arm}")
        mesh = int(model.geom_dataid[bracket.id])
        verts = model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
        in_link = (pose_of(bracket.pos, bracket.quat) @ np.c_[verts, np.ones(len(verts))].T).T[:, :3]
        lo, hi = in_link.min(axis=0) - 0.004, in_link.max(axis=0) + 0.004
        inside = bool(np.all(cam[:3, 3] >= lo) and np.all(cam[:3, 3] <= hi))
        check(f"{arm}: the lens is inside its bracket's bounds", inside,
              f"lens {np.round(cam[:3, 3] * 1000, 1)} mm")
        check(f"{arm}: the camera frame is a rotation", abs(np.linalg.det(cam[:3, :3]) - 1.0) < 1e-9)
    left = gripper_camera_in_link(spec, "left", None, None)
    right = gripper_camera_in_link(spec, "right", None, None)
    check("the right camera is the left one mirrored across the link's y",
          np.allclose(right[:3, 3], left[:3, 3] * [1, -1, 1]) and np.allclose(right[:3, 2], left[:3, 2] * [1, -1, 1]))
    del mujoco


def hermit() -> None:
    from pinkrab_sim.demos import tool_rotation
    from pinkrab_sim.world import TWO_HANDS, World, hermit

    print("the Hermits")
    world = World(arms=(), hermits=TWO_HANDS)
    try:
        for hand, y in (("left", 0.18), ("right", -0.18)):
            T = np.eye(4)
            T[:3, :3] = tool_rotation(0.4 if hand == "left" else -0.4, math.radians(60))
            T[:3, 3] = [0.31, y, 0.07]
            world.hold_hermit(hand, T, 0.3)
            world.forward()
            got = world.site_pose(f"{hermit(hand)}_tcp")
            check(f"{hand}: its TCP lands where it is held", np.allclose(got, T, atol=1e-9))
            jaw = world.data.qpos[world.model.jnt_qposadr[world.model.joint(f"{hermit(hand)}_gripper").id]]
            lo, hi = world.bands[hand]
            check(f"{hand}: its jaw opens across its arm's band", abs(jaw - (lo + 0.3 * (hi - lo))) < 1e-9)
    finally:
        world.close()


def arm_as_hermit() -> None:
    """The arm holding a Hermit's pose is the Hermit: its camera where the
    Hermit's is, looking the same way, and its jaw's hinge on the same side."""
    from pinkrab_sim.config import SETTINGS
    from pinkrab_sim.demos import REACH_DEG, REACH_M, Planner, tool_rotation
    from pinkrab_sim.world import World, hermit

    print("the arm, holding the Hermit's pose")
    arm = SETTINGS.arm
    world = World(arms=(arm,), hermits=(arm,))
    try:
        planner = Planner(world)
        T = np.eye(4)
        T[:3, :3] = tool_rotation(-0.4, math.radians(55))
        T[:3, 3] = [0.34, 0.18, 0.10]
        q = planner.reach(T)
        check("the arm reaches a pose a hand holds over the desk", q is not None)
        if q is None:
            return
        world.set_joints({n: math.radians(v) for n, v in zip(planner.chain.joint_names, q)})
        world.hold_hermit(arm, T, 0.3)
        world.forward()
        m, d = world.model, world.data
        mine, its = m.camera(f"{arm}_gripcam_front").id, m.camera(f"{hermit(arm)}_gripcam_front").id
        off_mm = float(np.linalg.norm(d.cam_xpos[mine] - d.cam_xpos[its])) * 1000
        check("its camera is where the Hermit's is", off_mm < REACH_M * 1000, f"{off_mm:.2f} mm apart")
        turn = d.cam_xmat[mine].reshape(3, 3).T @ d.cam_xmat[its].reshape(3, 3)
        off_deg = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(turn) - 1) / 2))))
        check("and looks the way the Hermit's does", off_deg < REACH_DEG, f"{off_deg:.2f} deg apart")
        hinge = float(np.linalg.norm(d.xpos[m.body(f"{arm}_jaw_link").id] - d.xpos[m.body(f"{hermit(arm)}_jaw").id])) * 1000
        check("its jaw hinges where the Hermit's does", hinge < REACH_M * 1000, f"{hinge:.2f} mm apart")
    finally:
        world.close()


def through_convert(noise: float, pose_mm: float, pose_deg: float, step_mm: float) -> None:
    """One take, converted: the dataset's pose and actions against the truth.

    With no tracker noise the pipeline has to hand the truth back, less what
    its smoothing takes off a hand's own motion; with the tracker's noise, the
    same within what smoothing leaves of it."""
    from pinkrab_sim.config import SETTINGS
    from pinkrab_sim.demos import generate
    from pinkrab_sim.world import desk_pose

    from pinkrab_core.config import SETTINGS as CORE
    from pinkrab_core.dataset import pose_key
    from pinkrab_umi import convert

    print(f"a take through convert, tracker noise x{noise:g}")
    batch = "check"
    for old in SETTINGS.episodes_dir.glob(f"{batch}-*"):
        shutil.rmtree(old)
    out = f"sim-{batch}"
    root = CORE.workspace / "datasets" / out
    shutil.rmtree(root, ignore_errors=True)
    try:
        report = generate(1, batch, seed=7, say=lambda *_: None, noise=noise)
        name = report["kept"][0]
        # At the simulation's own 30 fps, frame for frame against the truth.
        made = convert.convert([name], out, "pick and place the block", hands=(SETTINGS.arm,),
                               source=SETTINGS.episodes_dir, say=lambda *_: None, fps=SETTINGS.fps)
        segment = made["episodes"][0]["segments"][0]
        import pyarrow.parquet as pq
        from scipy.spatial.transform import Rotation

        table = pq.read_table(next(root.glob("data/*/*.parquet")))
        pose = np.array(table.column(pose_key(SETTINGS.arm)).to_pylist())
        action = np.array(table.column("action").to_pylist())
        state = np.array(table.column("observation.state").to_pylist())
        truth = [json.loads(line) for line in (SETTINGS.episodes_dir / name / "truth.jsonl").read_text().splitlines()]
        head, tail = segment["still_trimmed"]
        check("only the still ends are left out", len(pose) == segment["frames"] == len(truth) - head - tail
              and 0 < head < 20 and 0 < tail < 20, f"{len(pose)} of {len(truth)}, {head} + {tail} still")
        truth = truth[segment["start"]:segment["end"]]
        desk = desk_pose()
        T_true = np.array([t["tcp"] for t in truth])
        T_got = np.array([desk @ np.vstack([np.c_[Rotation.from_quat(p[3:]).as_matrix(), p[:3]], [0, 0, 0, 1]])
                          for p in pose])
        err_mm = np.linalg.norm(T_got[:, :3, 3] - T_true[:, :3, 3], axis=1) * 1000
        err_deg = np.degrees([Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude() for a, b in zip(T_got, T_true)])
        check("pose.tcp is the true TCP, in the desk's frame", err_mm.max() < pose_mm and err_deg.max() < pose_deg,
              f"worst {err_mm.max():.2f} mm, {err_deg.max():.2f} deg")
        dp_true = np.array([T_true[k, :3, :3].T @ (T_true[k + 1, :3, 3] - T_true[k, :3, 3]) for k in range(len(T_true) - 1)])
        dp_err = np.linalg.norm(action[:-1, :3] - dp_true, axis=1) * 1000
        check("the actions are the true camera-local steps", dp_err.mean() < step_mm,
              f"mean {dp_err.mean():.3f} mm, worst {dp_err.max():.2f} mm a step of up to "
              f"{np.linalg.norm(dp_true, axis=1).max() * 1000:.1f} mm")
        aperture = np.array([t["aperture"] for t in truth])
        check("the gripper action is the next frame's opening", np.allclose(action[:-1, 9], aperture[1:], atol=1e-3))
        check("the state's opening is this frame's", np.allclose(state[:, 0], aperture, atol=1e-3))
    finally:
        for old in SETTINGS.episodes_dir.glob(f"{batch}-*"):
            shutil.rmtree(old)
        shutil.rmtree(root, ignore_errors=True)


def at_another_rate(fps: int) -> None:
    """One take converted at 30 fps and at `fps`: the faster dataset's poses
    are still the true TCP between the cameras' frames, its steps the same
    motion over shorter intervals, and it is as long in seconds."""
    from pinkrab_sim.config import SETTINGS
    from pinkrab_sim.demos import generate
    from pinkrab_sim.world import desk_pose
    from scipy.interpolate import CubicSpline
    from scipy.spatial.transform import Rotation, RotationSpline

    from pinkrab_core.config import SETTINGS as CORE
    from pinkrab_core.dataset import pose_key
    from pinkrab_umi import convert

    print(f"a take through convert at 30 and {fps} fps")
    batch = "check-rate"
    for old in SETTINGS.episodes_dir.glob(f"{batch}-*"):
        shutil.rmtree(old)
    outs = {rate: f"sim-{batch}-{rate}" for rate in (30, fps)}
    roots = {rate: CORE.workspace / "datasets" / out for rate, out in outs.items()}
    for root in roots.values():
        shutil.rmtree(root, ignore_errors=True)
    try:
        report = generate(1, batch, seed=7, say=lambda *_: None, noise=0.0)
        name = report["kept"][0]
        import pyarrow.parquet as pq

        made, tables = {}, {}
        for rate, out in outs.items():
            made[rate] = convert.convert([name], out, "pick and place the block", hands=(SETTINGS.arm,),
                                         source=SETTINGS.episodes_dir, say=lambda *_: None, fps=rate)
            tables[rate] = pq.read_table(next(roots[rate].glob("data/*/*.parquet")))
        info = json.loads((roots[fps] / "meta" / "info.json").read_text())
        check(f"the dataset says {fps} fps", info["fps"] == fps, str(info["fps"]))
        slow, fast = (made[rate]["episodes"][0]["segments"][0] for rate in (30, fps))
        check("as long in seconds at either rate", abs(slow["seconds"] - fast["seconds"]) < 1.0 / fps
              and fast["frames"] == int(fast["seconds"] * fps + 1e-3) + 1,
              f"{slow['seconds']} s, {fast['seconds']} s in {fast['frames']} frames")

        truth = [json.loads(line) for line in (SETTINGS.episodes_dir / name / "truth.jsonl").read_text().splitlines()]
        stamps = np.array([t["stamp"] for t in truth]) - truth[0]["stamp"]
        T_true = np.array([t["tcp"] for t in truth])
        at = stamps[fast["start"]] + np.arange(fast["frames"]) / fps
        place = CubicSpline(stamps, T_true[:, :3, 3], axis=0)(at)
        turn = RotationSpline(stamps, Rotation.from_matrix(T_true[:, :3, :3]))(at)
        pose = np.array(tables[fps].column(pose_key(SETTINGS.arm)).to_pylist())
        desk = desk_pose()
        got = np.array([desk @ np.vstack([np.c_[Rotation.from_quat(p[3:]).as_matrix(), p[:3]], [0, 0, 0, 1]])
                        for p in pose])
        err_mm = np.linalg.norm(got[:, :3, 3] - place, axis=1) * 1000
        err_deg = np.degrees((turn.inv() * Rotation.from_matrix(got[:, :3, :3])).magnitude())
        check(f"pose.tcp at {fps} fps is the true TCP between the cameras' frames",
              err_mm.max() < 0.3 and err_deg.max() < 0.2, f"worst {err_mm.max():.2f} mm, {err_deg.max():.2f} deg")
        steps = {rate: np.linalg.norm(np.array(tables[rate].column("action").to_pylist())[:-1, :3], axis=1)
                 for rate in (30, fps)}
        ratio = steps[fps].mean() / steps[30].mean()
        check("the steps are the same motion over shorter intervals", abs(ratio - 30 / fps) < 0.03,
              f"mean step {steps[fps].mean() * 1000:.2f} mm against {steps[30].mean() * 1000:.2f} mm at 30 fps")
        moving = steps[30] > 1e-4
        still = int((steps[fps][: int(moving.sum() * fps / 30)] < 1e-6).sum())
        check("no step is a repeated pose", still == 0, f"{still} zero steps")
    finally:
        for old in SETTINGS.episodes_dir.glob(f"{batch}-*"):
            shutil.rmtree(old)
        for root in roots.values():
            shutil.rmtree(root, ignore_errors=True)


def two_hands() -> None:
    """One take of both hands - the working one its block into its bowl, the
    other held where its arm stands ready - through convert and the rig's own
    shelf and start: the dataset's poses are the two true TCPs, the state ends
    with where one tool stands from the other, the shelf judges both hands
    followable on their arms - the solid where the rig reckons one at the
    chest - and a run would start each arm where its hand began."""
    from pinkrab_sim.config import SETTINGS
    from pinkrab_sim.demos import generate
    from pinkrab_sim.world import TWO_HANDS, desk_pose
    from scipy.spatial.transform import Rotation

    from pinkrab_core.config import SETTINGS as CORE
    from pinkrab_core.dataset import pose_key, relative_pose, state_names
    from pinkrab_core.demo_start import start_joints
    from pinkrab_core.kinematics import ArmKinematics
    from pinkrab_policy.rollout import carry
    from pinkrab_studio import replay, verdict
    from pinkrab_umi import convert

    print("a take of both hands, through convert, the shelf and the start")
    batch = "check-two"
    for old in SETTINGS.episodes_dir.glob(f"{batch}-*"):
        shutil.rmtree(old)
    out = f"sim-{batch}"
    root = CORE.workspace / "datasets" / out
    shutil.rmtree(root, ignore_errors=True)
    try:
        report = generate(1, batch, seed=11, say=lambda *_: None, noise=0.0, scene="bowls")
        check("a take of both hands is kept", len(report["kept"]) == 1, str(report["rejected"]))
        if not report["kept"]:
            return
        name = report["kept"][0]
        take = SETTINGS.episodes_dir / name
        meta = json.loads((take / "meta.json").read_text())
        outcome = meta["sim"]["outcome"]
        check("the working hand's block went into its bowl",
              set(outcome["hands"]) == set(SETTINGS.two_hand_working) and all(o["placed"] for o in outcome["hands"].values()),
              json.dumps(outcome["hands"]))
        resting = outcome.get("resting", [])
        for hand in resting:
            path = np.array([t[f"tcp_{hand}"] for t in (json.loads(line) for line in
                                                         (take / "truth.jsonl").read_text().splitlines())])
            spread = float(np.max(np.linalg.norm(path[:, :3, 3] - path[0, :3, 3], axis=1))) * 1000
            check(f"{hand}: the hand at rest holds still but for a hand's tremor", spread < 8.0, f"{spread:.1f} mm")
        answer = replay.solved(take)
        said = verdict.judge(take, answer)
        from pinkrab_umi.config import SETTINGS as UMI

        check("the shelf judges both hands on their arms and fails neither",
              not said["fails"] and said["feasible"] is not None and said["feasible"] >= UMI.keep_min_feasible,
              f"fails {said['fails']}, feasible {said['feasible']}, held back {said['flags']}")

        made = convert.convert([name], out, "put each block in its bowl", hands=TWO_HANDS,
                               source=SETTINGS.episodes_dir, say=lambda *_: None, fps=SETTINGS.fps)
        segment = made["episodes"][0]["segments"][0]
        import pyarrow.parquet as pq

        table = pq.read_table(next(root.glob("data/*/*.parquet")))
        truth = [json.loads(line) for line in (take / "truth.jsonl").read_text().splitlines()]
        truth = truth[segment["start"]:segment["end"]]
        desk = desk_pose("bowls")
        got = {}
        for hand in TWO_HANDS:
            pose = np.array(table.column(pose_key(hand)).to_pylist())
            got[hand] = np.array([desk @ np.vstack([np.c_[Rotation.from_quat(p[3:]).as_matrix(), p[:3]], [0, 0, 0, 1]])
                                  for p in pose])
            true = np.array([t[f"tcp_{hand}"] for t in truth])
            err = np.linalg.norm(got[hand][:, :3, 3] - true[:, :3, 3], axis=1) * 1000
            check(f"{hand}: pose.tcp is the true TCP", err.max() < 0.3, f"worst {err.max():.2f} mm")
        state = np.array(table.column("observation.state").to_pylist())
        names = state_names(TWO_HANDS)
        check("the state is the two openings and the right tool from the left", state.shape[1] == len(names),
              f"{state.shape[1]} numbers, {len(names)} named")
        k = len(state) // 2
        T_left = np.array(truth[k]["tcp_left"])
        T_right = np.array(truth[k]["tcp_right"])
        rel = relative_pose(T_left, T_right)
        check("its relative pose is the true one", np.allclose(state[k, -9:], rel, atol=2e-3),
              f"{np.round(state[k, -9:3 - 9] * 1000, 1)} mm against {np.round(rel[:3] * 1000, 1)}")
        spread = json.loads((root / "meta" / "stats.json").read_text())["observation.state"]["std"]
        check("the resting jaw's opening is normalised by no less than the floor",
              min(spread) >= convert.STATE_STD_FLOOR, f"narrowest {min(spread):.2g}")

        chains = {h: ArmKinematics(h) for h in TWO_HANDS}
        joints, _ = start_joints(root, chains, carry)
        for hand in TWO_HANDS:
            T0 = np.array(truth[0][f"tcp_{hand}"])
            tool = chains[hand].forward(joints[hand])
            off = float(np.linalg.norm(tool[:3, 3] - T0[:3, 3])) * 1000
            check(f"{hand}: a run starts the arm where the hand began", off < 5.0, f"{off:.1f} mm off")
    finally:
        for old in SETTINGS.episodes_dir.glob(f"{batch}-*"):
            shutil.rmtree(old)
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    lens()
    states()
    camera_on_bracket()
    hermit()
    arm_as_hermit()
    through_convert(0.0, 0.3, 0.2, 0.05)
    through_convert(1.0, 2.5, 1.0, 0.8)
    at_another_rate(50)
    two_hands()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} failed: {', '.join(FAILURES)}")
        sys.exit(1)
    print("all ok")
