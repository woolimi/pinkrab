#!/usr/bin/env bash
#
# setup.sh - build the Pinkrab virtualenv. Re-runnable.
#
# This script owns only what a requirements file cannot express: which interpreter
# to build on, whether the venv inherits system site-packages, and the order things
# are installed in. Every version pin lives in requirements/ and is read from there.
# Re-listing pins here is how two lists drift apart.
#
#   python  On Linux, the interpreter ROS was built against. rclpy is a C extension
#           compiled for exactly that ABI, so a different interpreter - a conda env,
#           a newer python - cannot import it no matter what is on PYTHONPATH.
#           On macOS there is no ROS, so anything >= 3.12 (LeRobot's floor) will do.
#   venv    --system-site-packages on Linux, so rclpy from /opt/ros/<distro> is
#           visible and our pip packages layer on top of it. Not on macOS, where
#           there is nothing to inherit and inheriting would only leak Homebrew.
#
# LeRobot is installed from the pinned requirement. To work on LeRobot itself:
#
#   LEROBOT_EDITABLE=1 LEROBOT_SRC=~/projects/lerobot ./scripts/setup.sh
#
# That override is installed last, so it replaces the pinned wheel rather than
# racing it.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${PINKRAB_VENV:-$REPO/.venv}"

# --- interpreter -----------------------------------------------------------
# Being on PATH is not enough. A standalone build (the kind uv and pyenv install)
# can ship without pip's bundled wheels, and then `python -m venv` fails at the
# ensurepip step. The only honest test is to build a throwaway venv and see.
can_build_venv() {
    local probe
    probe="$(mktemp -d)"
    if "$1" -m venv "$probe/v" >/dev/null 2>&1; then
        rm -rf "$probe"
        return 0
    fi
    rm -rf "$probe"
    return 1
}

pick_python() {
    if [[ "$(uname -s)" == "Linux" ]]; then
        # Must match ROS. rclpy is a C extension built for exactly this ABI, so no
        # other interpreter can import it regardless of PYTHONPATH.
        if [[ -x /usr/bin/python3.12 ]] && can_build_venv /usr/bin/python3.12; then
            echo /usr/bin/python3.12
            return
        fi
        echo "Cannot build a venv on /usr/bin/python3.12." >&2
        echo "  ROS 2 Jazzy is built against it, so Pinkrab has to use it too." >&2
        echo "  Install: sudo apt install -y python3.12 python3.12-venv" >&2
        exit 1
    fi

    # macOS: no ROS to match, so any interpreter at LeRobot's floor will do.
    # Prefer 3.12 anyway, to keep both machines on the same version.
    local candidate path skipped=()
    for candidate in python3.12 python3.13 python3.14; do
        path="$(command -v "$candidate" 2>/dev/null)" || continue
        if can_build_venv "$path"; then
            echo "$path"
            return
        fi
        skipped+=("$path")
    done

    echo "No Python >= 3.12 that can build a virtualenv." >&2
    if (( ${#skipped[@]} )); then
        echo "  Found but unusable (no bundled pip - a standalone build?):" >&2
        printf '    %s\n' "${skipped[@]}" >&2
    fi
    echo "  Install: brew install python@3.12" >&2
    exit 1
}

PYTHON="$(pick_python)"
PYVER="$("$PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"

SITE_FLAG=()
if [[ "$(uname -s)" == "Linux" ]]; then
    SITE_FLAG=(--system-site-packages)
fi

echo "==> [1/4] venv: $VENV"
SITE_NOTE=""
if [[ "$(uname -s)" == "Linux" ]]; then SITE_NOTE="  --system-site-packages"; fi
echo "    python $PYVER ($PYTHON)$SITE_NOTE"
"$PYTHON" -m venv ${SITE_FLAG[@]+"${SITE_FLAG[@]}"} "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --quiet --upgrade pip

# Ours ahead of ROS's. A sourced ROS puts /opt/ros/<distro>'s site-packages on
# PYTHONPATH, and PYTHONPATH comes before every site directory - so the pip
# packages this venv layers on top of ROS lose to ROS's own copies of the same
# names. Measured: ROS's eigenpy, built for numpy 1, shadows the one placo was
# built against, and the solver fails to import. A .pth runs after the path is
# assembled and may reorder it; this one moves everything under the venv to the
# front. The script's own directory is prepended by the interpreter after this
# runs, so it still comes first. rclpy is not in the venv, so it still comes
# from ROS.
if [[ "$(uname -s)" == "Linux" ]]; then
    SITE="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
    cat > "$SITE/pinkrab_first.pth" <<'PTH'
import sys; sys.path[:] = [p for p in sys.path if p.startswith(sys.prefix)] + [p for p in sys.path if not p.startswith(sys.prefix)]
PTH
    # And for what `ros2 run` and `ros2 launch` start from this shell. colcon
    # gives every node a /usr/bin/python3 shebang, and that interpreter sees
    # none of the venv - the driver dies on "No module named lerobot", the gate
    # on "mujoco is not installed", on a machine that has both. The two are
    # the same 3.12 (the reason this venv is built on it), so the venv's
    # site-packages can simply be put on PYTHONPATH for them, ahead of ROS's.
    # The .pth above does the same for the venv's own interpreter; a node is
    # not that interpreter, so it needs the environment. cmeel's own prefix is
    # named beside it: placo lives there, reached by a .pth that only a site
    # directory runs, and PYTHONPATH entries are not site directories.
    #
    # The same for shared libraries. placo's wheel carries its own libeigenpy
    # under cmeel.prefix/lib and finds it by RUNPATH - which LD_LIBRARY_PATH,
    # as a sourced ROS sets it, is searched ahead of. So placo.so was linking
    # ROS's older libeigenpy and failing on a symbol it does not have. The
    # loader reads LD_LIBRARY_PATH once, at start, so this has to be in the
    # environment too.
    #
    # Both go into `activate`, appended once (`python -m venv` rewrites it on
    # every run). Source ROS first, then activate: the venv has to end up in
    # front, and a ROS sourced afterwards would put itself there instead.
    ACTIVATE="$VENV/bin/activate"
    if ! grep -q "pinkrab: venv ahead of ROS" "$ACTIVATE"; then
        cat >> "$ACTIVATE" <<'ACT'

# pinkrab: venv ahead of ROS, for the nodes ros2 starts from here (they run on
# /usr/bin/python3) and for placo's own libeigenpy (cmeel.prefix/lib).
export PYTHONPATH="$VIRTUAL_ENV/lib/python3.12/site-packages:$VIRTUAL_ENV/lib/python3.12/site-packages/cmeel.prefix/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/cmeel.prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ACT
    fi
fi

# --- dependencies ----------------------------------------------------------
# A lock reproduces the environment exactly; the .in files only promise something
# that works. Prefer the lock, say plainly when there is not one.
case "$(uname -s)" in
    Linux)  LOCK="$REPO/requirements/lock/linux.txt" ;;
    Darwin) LOCK="$REPO/requirements/lock/macos.txt" ;;
    *)      LOCK="" ;;
esac

echo "==> [2/4] dependencies"
if [[ -n "$LOCK" && -f "$LOCK" ]]; then
    echo "    from the lock: ${LOCK#"$REPO"/}"
    pip install --quiet -r "$LOCK"
else
    echo "    no lock for this platform - resolving from requirements/*.in"
    echo "    (versions may differ from other machines; run ./scripts/lock.sh to pin them)"
    pip install --quiet -r "$REPO/requirements/arm.in"
fi

# Core first: the CLI declares it as a dependency and it is not on any index.
#
# The rest are here because the studio launches them by path: rig.py looks for
# pinkrab-umi beside `sys.executable`, on the reasoning that a studio started from
# one environment must not reach into another. colcon installs those console
# scripts under install/<package>/lib/<package>/, which is not that directory, so
# a venv without them serves a page whose camera button fails.
echo "==> [3/4] pinkrab_core, pinkrab_cli, pinkrab_mujoco, pinkrab_studio, pinkrab_umi, pinkrab_policy, pinkrab_sim, lerobot_robot_pinkrab (editable)"
pip install --quiet -e "$REPO/src/pinkrab_core" -e "$REPO/src/pinkrab_cli"
pip install --quiet -e "$REPO/src/pinkrab_mujoco" -e "$REPO/src/pinkrab_studio" \
                    -e "$REPO/src/pinkrab_umi" -e "$REPO/src/pinkrab_policy" \
                    -e "$REPO/src/pinkrab_sim"
# The arms as a LeRobot robot type, so lerobot-calibrate, -record and the rest take
# --robot.type=bi_pinkrab_follower. Outside src/, so colcon leaves it alone.
pip install --quiet -e "$REPO/lerobot-plugin"

# The ROS middleware, for `ros2` typed in this shell: the nodes set it
# themselves, and a service call from another reaches them with its fields
# zeroed. Read from pinkrab_core's config, now installed, rather than copied.
if [[ "$(uname -s)" == "Linux" ]] && ! grep -q "pinkrab: ROS middleware" "$ACTIVATE"; then
    RMW="$(python -c 'from pinkrab_core.config import SETTINGS; print(SETTINGS.ros_middleware)')"
    printf '\n# pinkrab: ROS middleware, as pinkrab_core config names it.\nexport RMW_IMPLEMENTATION=%s\n' "$RMW" >> "$ACTIVATE"
fi

if [[ -n "${LEROBOT_EDITABLE:-}" ]]; then
    LEROBOT_SRC="${LEROBOT_SRC:-$HOME/projects/lerobot}"
    if [[ ! -d "$LEROBOT_SRC/.git" ]]; then
        echo "LEROBOT_EDITABLE is set but $LEROBOT_SRC is not a git checkout." >&2
        exit 1
    fi
    echo "==> lerobot editable override: $LEROBOT_SRC"
    echo "    ($(git -C "$LEROBOT_SRC" log --oneline -1))"
    pip install --quiet -e "$LEROBOT_SRC[feetech]"
fi

# --- verify ----------------------------------------------------------------
# Say what actually landed. A half-finished install that reports success is worse
# than one that fails.
echo "==> [4/4] verify"
python - <<'PY'
import importlib.metadata as md
import importlib.util as u
import shutil

for module in ("lerobot", "scservo_sdk", "pinkrab_core", "pinkrab_cli",
               "pinkrab_mujoco", "pinkrab_studio", "pinkrab_umi", "pinkrab_policy"):
    spec = u.find_spec(module)
    version = ""
    if spec:
        try:
            version = md.version(module.replace("_", "-"))
        except Exception:
            version = "installed"
    print(f"    {module:<14} {version or 'MISSING'}")

for cli in ("pinkrab", "pinkrab-studio", "pinkrab-umi",
            "lerobot-calibrate", "lerobot-find-port"):
    print(f"    {cli:<14} {shutil.which(cli) or 'MISSING'}")
PY

if [[ "$(uname -s)" == "Linux" ]]; then
    # rclpy is not pip installed - it is inherited from /opt/ros once sourced. Its
    # absence here is not a failure, so it is reported separately.
    echo "    rclpy          inherited from /opt/ros (source its setup.bash to use)"
fi

echo
echo "Done. Activate with:"
echo "    source ${VENV#"$REPO"/}/bin/activate"
if [[ "$(uname -s)" == "Linux" ]]; then
    echo "    source /opt/ros/jazzy/setup.bash    # for ROS packages"
fi
