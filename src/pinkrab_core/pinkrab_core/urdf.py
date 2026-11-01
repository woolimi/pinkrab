"""Expand the Pinkrab description to a URDF.

One place, because two callers already needed the same workaround and a third was
coming. `$(find pinkrab_description)` resolves through the ament index, which only
exists where ROS does - and the machine the arms are plugged into has no ROS. So on
those machines the substitution is done here instead, against the package directory
found on disk.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import tempfile
from pathlib import Path

from .config import SETTINGS

FIND = "$(find pinkrab_description)"


class DescriptionError(RuntimeError):
    """Raised when the description cannot be expanded."""


def has_ament() -> bool:
    return importlib.util.find_spec("ament_index_python") is not None


def expand(mount: Path | None = None, arm: str | None = None) -> str:
    """Run xacro over pinkrab.urdf.xacro and return the URDF.

    Args:
        mount: a mount.yaml to use instead of the package default.
        arm: expand this one arm on its own, rooted at a bare `mount` link,
            instead of the whole robot. "left" or "right".
    """
    if shutil.which("xacro") is None:
        raise DescriptionError(
            "xacro not found. On a ROS machine source its setup; "
            "otherwise: pip install xacro"
        )

    xacro_file = SETTINGS.xacro_file
    if not xacro_file.is_file():
        raise DescriptionError(f"no description at {xacro_file}")

    source = xacro_file.read_text()
    if not has_ament():
        source = source.replace(FIND, str(SETTINGS.description_pkg))

    # xacro resolves includes relative to the file it is given, so the temporary
    # copy has to sit beside the original.
    with tempfile.NamedTemporaryFile(
        "w", suffix=".xacro", dir=xacro_file.parent, delete=False
    ) as handle:
        handle.write(source)
        probe = Path(handle.name)

    args = [f"mount:={mount}"] if mount else []
    if arm:
        args.append(f"arm:={arm}")
    try:
        result = subprocess.run(
            ["xacro", str(probe), *args], capture_output=True, text=True, check=True
        )
    except subprocess.CalledProcessError as exc:
        raise DescriptionError(f"xacro failed:\n{exc.stderr.strip()}") from exc
    finally:
        probe.unlink(missing_ok=True)

    return result.stdout


def expand_for_meshes(mesh_url_base: str, mount: Path | None = None) -> str:
    """Expand, then point mesh references somewhere a given consumer can fetch them.

    `package://` means nothing to a browser or to a kinematics library without an
    ament index, and both need the meshes rather than just the joints.
    """
    return expand(mount).replace(
        "package://pinkrab_description/meshes/", f"{mesh_url_base.rstrip('/')}/"
    )
