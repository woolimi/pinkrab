#!/usr/bin/env python3
"""Expand the Pinkrab description and check it says what it should.

A URDF fails quietly. A mesh path that resolves nowhere renders an invisible link, a
second root turns the model into two disconnected trees, and a mount transform that
never reached the model leaves the arms sitting in the default place - none of that
raises. This expands the xacro and asserts the things that would otherwise be found
by looking at a picture and not noticing.

Needs xacro. On a ROS machine it comes from ROS; elsewhere: pip install xacro.

    python scripts/check_description.py
    python scripts/check_description.py --mount path/to/other_rig.yaml
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "src" / "pinkrab_description"
ARMS = ("left", "right")
# Taken from the library's joint list rather than written down, because the two have to
# agree: that list is what names the servos on the bus, and a description with a
# joint the bus has no servo for - or the other way round - is exactly the kind of
# quiet mismatch this script exists to catch.
sys.path.insert(0, str(REPO / "src" / "pinkrab_core"))
from pinkrab_core.config import SETTINGS  # noqa: E402

JOINTS_PER_ARM = len(SETTINGS.joint_names)

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}" + (f" - {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


def expand(mount: Path | None) -> str:
    """Expand the description. The `$(find ...)` workaround for machines without ROS
    lives in pinkrab_core, so this and the studio cannot drift apart."""
    sys.path.insert(0, str(REPO / "src" / "pinkrab_core"))
    from pinkrab_core.urdf import DescriptionError
    from pinkrab_core.urdf import expand as expand_description

    try:
        return expand_description(mount)
    except DescriptionError as exc:
        print(exc)
        raise SystemExit(2 if "xacro not found" in str(exc) else 1) from exc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mount", type=Path, help="use this mount.yaml instead of the package default")
    args = ap.parse_args()

    root = ET.fromstring(expand(args.mount))
    links = [link.get("name") for link in root.findall("link")]
    joints = {j.get("name"): j for j in root.findall("joint")}

    print("\nstructure")
    children = {j.find("child").get("link") for j in joints.values()}
    roots = [link for link in links if link not in children]
    check("exactly one root", len(roots) == 1, f"roots: {roots}")
    check("rooted at world", roots == ["world"], f"roots: {roots}")
    for frame in ("wall_link", "pinkrab_base_link"):
        check(f"{frame} present", frame in links)

    print("\narms")
    for arm in ARMS:
        moving = [
            n for n, j in joints.items()
            if n.startswith(f"{arm}_") and j.get("type") in ("revolute", "continuous")
        ]
        check(f"{arm}: {JOINTS_PER_ARM} moving joints", len(moving) == JOINTS_PER_ARM, str(len(moving)))
        check(f"{arm}: tip frame", f"{arm}_gripper_frame_link" in links)
    # Two arms of identical hardware must not silently share a name.
    check(
        "no name is shared between the arms",
        not ({n.removeprefix("left_") for n in joints if n.startswith("left_")}
             ^ {n.removeprefix("right_") for n in joints if n.startswith("right_")}),
    )

    print("\nmeshes")
    referenced = {m.get("filename") for m in root.iter("mesh")}
    bad_scheme = [f for f in referenced if not f.startswith("package://pinkrab_description/meshes/")]
    check("all package-relative", not bad_scheme, str(bad_scheme[:3]))
    missing = [f for f in referenced if not (PKG / "meshes" / Path(f).name).is_file()]
    check(f"all {len(referenced)} resolve on disk", not missing, str(missing[:3]))

    print("\nmount reached the model")
    # A transform that never arrives leaves the arms stacked at the origin, which
    # looks like one arm rather than an error.
    origins = {}
    for arm in ARMS:
        joint = joints.get(f"{arm}_base_joint")
        if joint is None:
            check(f"{arm}_base_joint exists", False)
            continue
        o = joint.find("origin")
        origins[arm] = (o.get("xyz"), o.get("rpy"))
        print(f"    {arm:<6} xyz={o.get('xyz')}  rpy={o.get('rpy')}")
    check("the two arms are not in the same place", len(set(origins.values())) == len(origins))

    base = joints.get("pinkrab_base_joint")
    if base is not None:
        o = base.find("origin")
        print(f"    jig    xyz={o.get('xyz')}  rpy={o.get('rpy')}")
        check("jig is off the floor", float(o.get("xyz").split()[2]) > 0, o.get("xyz"))

    print()
    # The MuJoCo model is generated from this description, in another repository and
    # at another time, so the two can drift. What has to agree is the part a
    # simulation is trusted for: which joints there are and how far each may turn.
    mjcf = PKG / "mjcf" / "pinkrab.xml"
    print("\nMuJoCo model")
    if not mjcf.is_file():
        print("  skip  no mjcf/pinkrab.xml")
    else:
        try:
            import mujoco  # noqa: PLC0415
        except ImportError:
            print("  skip  mujoco not installed (requirements/sim.in)")
        else:
            model = mujoco.MjModel.from_xml_path(str(mjcf))
            sim = {
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i): tuple(model.jnt_range[i])
                for i in range(model.njnt)
            }
            urdf = {
                name: (float(j.find("limit").get("lower")), float(j.find("limit").get("upper")))
                for name, j in joints.items()
                if j.get("type") == "revolute" and j.find("limit") is not None
            }
            check("the same joints", set(sim) == set(urdf),
                  f"only in mjcf: {sorted(set(sim) - set(urdf))}, "
                  f"only in urdf: {sorted(set(urdf) - set(sim))}")
            off = [n for n in set(sim) & set(urdf)
                   if max(abs(sim[n][0] - urdf[n][0]), abs(sim[n][1] - urdf[n][1])) > 1e-4]
            check("the same joint limits", not off, f"differ: {sorted(off)}")
            check("one actuator per joint", model.nu == len(sim), f"{model.nu} for {len(sim)} joints")

            # Names and limits agreeing is not the same as the two models putting
            # the hand in the same place. This walks both through the same random
            # poses and compares where each says the tool centre point ended up -
            # the claim the whole simulation rests on.
            try:
                sys.path.insert(0, str(REPO / "src" / "pinkrab_core"))
                from pinkrab_core.kinematics import ArmKinematics, write_urdf
            except Exception as exc:  # noqa: BLE001
                print(f"  skip  kinematics unavailable ({exc})")
            else:
                import numpy as np

                write_urdf()
                data = mujoco.MjData(model)
                rng = np.random.default_rng(0)
                worst_pos = worst_rot = 0.0
                for arm in ARMS:
                    k = ArmKinematics(arm)
                    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{arm}_tcp")
                    adr = [model.jnt_qposadr[mujoco.mj_name2id(
                        model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in k.joint_names]
                    band = [model.jnt_range[mujoco.mj_name2id(
                        model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in k.joint_names]
                    for _ in range(40):
                        q = np.array([rng.uniform(*b) for b in band])
                        data.qpos[:] = 0
                        for a, v in zip(adr, q):
                            data.qpos[a] = v
                        mujoco.mj_forward(model, data)
                        T = k.forward(np.degrees(q))
                        worst_pos = max(worst_pos, float(np.linalg.norm(
                            T[:3, 3] - data.site_xpos[site])))
                        turn = T[:3, :3].T @ data.site_xmat[site].reshape(3, 3)
                        worst_rot = max(worst_rot, float(np.degrees(np.arccos(
                            np.clip((np.trace(turn) - 1) / 2, -1, 1)))))
                check("both models put the hand in the same place",
                      worst_pos < 1e-4 and worst_rot < 0.01,
                      f"worst {worst_pos * 1e6:.1f} um / {worst_rot * 1000:.1f} mdeg over 80 poses")

    if FAILURES:
        print(f"FAILED: {len(FAILURES)}")
        for name in FAILURES:
            print(f"  - {name}")
        print()
        return 1
    print("Description looks consistent.")
    # Consistency is not correctness. Say plainly whether the geometry is a
    # measurement of a real rig or numbers chosen to make the model load.
    mount_file = args.mount or (PKG / "config" / "mount.yaml")
    text = mount_file.read_text()
    if "PLACEHOLDER" in text:
        placeholders = [line.strip() for line in text.splitlines() if "PLACEHOLDER" in line]
        print(f"But {mount_file.name} still holds placeholder geometry:")
        for line in placeholders:
            print(f"  {line}")
        print()
    else:
        print(f"{mount_file.name} declares no placeholders - the geometry is measured.")
        print("Check it against the rig before trusting it.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
