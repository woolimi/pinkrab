"""Training started from here: on this machine, or on one reached over ssh.

The machine trained on needs python3, an ssh login and the internet once;
everything else is taken there when it is first needed and kept under the
source's root, which on a cloud box is its persistent volume:

    run_log.py              pinkrab-train as one file (pinkrab_policy.run_log)
    setup.sh                a venv from the lock, the Python it needs included
    requirements.lock.txt   this checkout's Linux lock, so a checkpoint made
                            there loads here
    .venv/  uv-python/      what setup.sh builds; kept, so a new session on the
                            same volume starts without a download
    datasets/<name>--<fp>/  each dataset as it was when sent, by fingerprint:
                            a dataset converted again is a new directory, and a
                            run reading the old one is left alone. On this
                            machine it is a link to the dataset instead, which
                            follows a conversion - a resume checks it
    train/<run>/            the run's output: checkpoints, metrics.jsonl
    logs/<run>.log          its terminal

What a machine needs is what the lock's PyTorch needs: 2.11 built for CUDA
13.0, which runs on NVIDIA GPUs from Turing on (compute capability 7.5 and
up - T4, RTX 20 to 50 series, A10, A100, L4, L40S, H100, H200, B200) under a
driver of 580 or newer. Launching checks both and says which is short.

A run is started detached (setsid, nohup): it does not need this studio, or
the ssh connection that started it, to keep going. What it writes is read by
the training page's watch like any other run, and its checkpoints are copied
back into this machine's train directory - pretrained_model only, which is
what a policy is loaded from - where the policy page lists them.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import queue
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

from pinkrab_policy import run_log

from pinkrab_core.config import SETTINGS as CORE
from pinkrab_core.dataset import (
    FINGERPRINT,
    LensMissing,
    dataset_fps,
    fingerprint,
    require_cut,
    steps_in,
)

from .alerts import ALERTS
from .training import TrainError, TrainStore, check_key, keep_key, ssh_base

# What each policy is trained with, beyond what the launch form asks. ACT
# trains in an 8 GB GPU: on two 780x570 gripper cameras of a 30 fps dataset a
# step at batch 8 peaks at 5.7 GB. A chunk is as long in seconds at any
# dataset rate (`chunk_s`): its steps are counted at the dataset's fps, so
# policies trained at two rates look as far ahead. ACT's is the paper's, 100
# steps at ALOHA's 50 Hz.
RECIPES: dict[str, dict] = {
    "act": {"label": "ACT", "batch_size": 8, "workers": 6, "prefetch": 2, "chunk_s": 2.0,
            "args": ["--policy.type=act"]},
}
DEFAULTS = {"steps": 100_000, "log_freq": 50, "save_freq": 20_000}
# The GPU a run needs, set by the lock's torch (2.11+cu130: sm_75 to sm_120),
# and its memory: every recipe above fits in 8 GB.
MIN_DRIVER = 580
MIN_COMPUTE = 7.5
MIN_GPU_MEMORY_GB = 8
LOCK_STAMP = ".lock.sha256"

SETUP_SH = r"""#!/bin/bash
# A venv for training, from the lock beside this file. Everything is kept
# under this directory, the uv-managed Python too: on a cloud box the home
# directory goes when the session does, this directory does not.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export UV_PYTHON_INSTALL_DIR="$ROOT/uv-python" UV_CACHE_DIR="$ROOT/uv-cache" UV_LINK_MODE=copy
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
uv python install 3.12
[ -x "$ROOT/.venv/bin/python" ] || uv venv --python 3.12 "$ROOT/.venv"
VIRTUAL_ENV="$ROOT/.venv" uv pip install -r "$ROOT/requirements.lock.txt"
"$ROOT/.venv/bin/python" -c "import torch, lerobot; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), '- lerobot', lerobot.__version__)"
sha256sum "$ROOT/requirements.lock.txt" | cut -d' ' -f1 > "$ROOT/.lock.sha256"
"""

# What a machine is asked before anything is sent: run there with python3.
PROBE = r"""
import json, os, shutil, subprocess, socket
root = os.path.expanduser(%r)
out = {"host": socket.gethostname(), "root": root, "cores": os.cpu_count()}
fields = ["index", "name", "memory.total", "memory.used", "utilization.gpu", "driver_version", "compute_cap"]
keys = ["index", "name", "mem_total_mb", "mem_used_mb", "util", "driver", "compute"]
out["gpus"] = []
for n in (len(fields), len(fields) - 1):  # an older nvidia-smi has no compute_cap
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=" + ",".join(fields[:n]), "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20)
    except Exception as exc:
        out["gpu_error"] = str(exc)
        break
    if q.returncode == 0:
        out["gpus"] = [dict(zip(keys[:n], [p.strip() for p in l.split(",")])) for l in q.stdout.strip().splitlines() if l.strip()]
        break
    out["gpu_error"] = (q.stderr or q.stdout).strip()
os.makedirs(root, exist_ok=True)
out["free_gb"] = round(shutil.disk_usage(root).free / 1e9, 1)
out["venv"] = os.path.exists(os.path.join(root, ".venv", "bin", "lerobot-train"))
stamp = os.path.join(root, ".lock.sha256")
out["lock"] = open(stamp).read().strip() if os.path.exists(stamp) else None
out["datasets"] = sorted(os.listdir(os.path.join(root, "datasets"))) if os.path.isdir(os.path.join(root, "datasets")) else []
running = []
for pid in os.listdir("/proc"):
    if pid.isdigit():
        try:
            cmd = open("/proc/%%s/cmdline" %% pid, "rb").read().split(b"\0")
        except OSError:
            continue
        line = " ".join(c.decode("utf-8", "replace") for c in cmd if c)
        if not ("run_log.py" in line or "lerobot-train" in line or "pinkrab_policy.run_log" in line):
            continue
        # A run started names its output directory; a resumed one only the
        # config inside it, <output_dir>/checkpoints/last/....
        for w in line.split():
            if w.startswith("--output_dir="):
                running.append(w.split("=", 1)[1])
                break
            if w.startswith("--config_path=") and "/checkpoints/" in w:
                running.append(w.split("=", 1)[1].split("/checkpoints/", 1)[0])
                break
out["running"] = sorted(set(running))
print(json.dumps(out))
"""


class LaunchError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dataset_dir(name: str) -> Path:
    return CORE.workspace / "datasets" / name


def image_shape(root: Path) -> tuple[int, int] | None:
    features = json.loads((root / "meta" / "info.json").read_text()).get("features") or {}
    for key, f in features.items():
        if key.startswith("observation.images.") and len(f.get("shape") or []) == 3:
            return int(f["shape"][0]), int(f["shape"][1])
    return None


# --- running things on a source -------------------------------------------------
def _run(source: dict, command: str, *, stdin: bytes | None = None, timeout: float = 120.0) -> str:
    """A shell command on the source, its output; an error names what it said."""
    if source["kind"] != "local":
        check_key(source)
    argv = ["bash", "-c", command] if source["kind"] == "local" else ssh_base(source) + [command]
    try:
        done = subprocess.run(argv, input=stdin, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise LaunchError(f"{source['name']}: no answer in {timeout:.0f} s") from exc
    if done.returncode != 0:
        said = (done.stderr or done.stdout).decode("utf-8", "replace").strip().splitlines()
        raise LaunchError(f"{source['name']}: {said[-1] if said else f'exit {done.returncode}'}")
    return done.stdout.decode("utf-8", "replace")


def _stream(source: dict, command: str, say, timeout: float) -> None:
    """A long shell command on the source, each line it prints said as it comes."""
    argv = ["bash", "-c", command] if source["kind"] == "local" else ssh_base(source) + [command]
    started = time.monotonic()
    with subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
        last = ""
        for line in proc.stdout:
            last = line.strip() or last
            say(last)
            if time.monotonic() - started > timeout:
                proc.kill()
                raise LaunchError(f"{source['name']}: still going after {timeout / 60:.0f} min - stopped")
        if proc.wait() != 0:
            raise LaunchError(f"{source['name']}: {last or f'exit {proc.returncode}'}")


def _send_tar(source: dict, remote_dir: str, add, total: int, say) -> None:
    """A tar stream made by `add(tar)` unpacked into remote_dir, with progress."""
    argv = ssh_base(source) + [f"mkdir -p {remote_dir} && tar -C {remote_dir} -xf -"]
    with subprocess.Popen(argv, stdin=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
        sent = [0, time.monotonic()]

        class Counting(io.RawIOBase):
            def writable(self):
                return True

            def write(self, b):
                proc.stdin.write(b)
                sent[0] += len(b)
                if time.monotonic() - sent[1] > 2.0 and total:
                    sent[1] = time.monotonic()
                    say(f"sending {sent[0] / 1e6:.0f} / {total / 1e6:.0f} MB")
                return len(b)

        try:
            with tarfile.open(fileobj=Counting(), mode="w|") as tar:
                add(tar)
            proc.stdin.close()
        except BrokenPipeError:
            pass
        err = proc.stderr.read().decode("utf-8", "replace").strip()
        if proc.wait() != 0:
            raise LaunchError(f"{source['name']}: sending failed - {err.splitlines()[-1] if err else proc.returncode}")


def probe(store: TrainStore, name: str) -> dict:
    """What the machine has: GPUs and their memory, cores, room, whether it is
    set up, which datasets it holds and which runs are going on it."""
    source = store.source(name)
    root = source["root"] if source["kind"] != "local" else str(CORE.train_dir)
    script = PROBE % root
    if source["kind"] == "local":
        done = subprocess.run([sys.executable, "-"], input=script, capture_output=True, text=True, timeout=60)
        text = done.stdout
    else:
        text = _run(source, "python3 -", stdin=script.encode(), timeout=60)
    try:
        found = json.loads(text)
    except ValueError as exc:
        raise LaunchError(f"{name}: the probe answered with something else: {text[-200:]}") from exc
    found["lock_matches"] = found.get("lock") == _sha256(CORE.remote_lock_file)
    found["short"] = shortfalls(found.get("gpus") or [])
    if source["kind"] == "local":
        found["venv"], found["lock_matches"] = True, True
    found["t"] = time.time()
    return found


def save_key(store: TrainStore, name: str, text: str) -> str:
    """An ssh key handed over as a file, kept (training.keep_key) and made the
    source's key. The key itself is never sent back."""
    if store.source(name)["kind"] == "local":
        raise LaunchError("this machine needs no key")
    path = keep_key(name, text)
    store.set_key(name, path)
    return path


def shortfalls(gpus: list[dict]) -> list[str]:
    """What keeps these GPUs from running the lock's PyTorch, in words."""
    if not gpus:
        return ["no NVIDIA GPU that nvidia-smi can see"]
    said = []
    for g in gpus:
        name = f"GPU {g.get('index')} ({g.get('name')})"
        try:
            if int(str(g.get("driver", "0")).split(".")[0]) < MIN_DRIVER:
                said.append(f"{name}: driver {g.get('driver')}, CUDA 13.0 needs {MIN_DRIVER} or newer")
        except ValueError:
            pass
        try:
            if g.get("compute") and float(g["compute"]) < MIN_COMPUTE:
                said.append(f"{name}: compute capability {g['compute']}, the build starts at {MIN_COMPUTE} (Turing)")
        except ValueError:
            pass
    return said


# --- the jobs ------------------------------------------------------------------------------
class Jobs:
    """Launches (and stops, and resumes) on one thread, checkpoint copies on
    another: a 300 MB copy home should not hold up starting the next run."""

    def __init__(self, store: TrainStore, on_change=None) -> None:
        self.store = store
        self.on_change = on_change or (lambda: None)
        self._lock = threading.Lock()
        self.probes: dict[str, dict] = {}
        self._state = {"launch": self._idle(), "fetch": self._idle()}
        self._queues = {"launch": queue.Queue(), "fetch": queue.Queue()}
        self._queued_fetches: set[tuple[int, int]] = set()
        for kind in self._queues:
            threading.Thread(target=self._work, args=(kind,), name=f"train-{kind}", daemon=True).start()

    @staticmethod
    def _idle() -> dict:
        return {"busy": False, "what": None, "lines": [], "error": None, "result": None, "t": None, "ended": None}

    def state(self) -> dict:
        with self._lock:
            return {k: {**v, "lines": v["lines"][-8:]} for k, v in self._state.items()} | {
                "probes": self.probes, "queued_fetches": len(self._queued_fetches)}

    def _work(self, kind: str) -> None:
        while True:
            what, fn = self._queues[kind].get()
            with self._lock:
                self._state[kind] = {**self._idle(), "busy": True, "what": what, "t": time.time()}

            def say(line: str, kind=kind) -> None:
                with self._lock:
                    lines = self._state[kind]["lines"]
                    if lines and line.startswith("sending ") and lines[-1].startswith("sending "):
                        lines[-1] = line
                    else:
                        lines.append(line)
            try:
                result = fn(say)
                with self._lock:
                    self._state[kind].update(busy=False, result=result, ended=time.time())
            except Exception as exc:  # noqa: BLE001 - one job that breaks must not take the queue with it
                error = str(exc) if isinstance(exc, (LaunchError, TrainError)) else f"{type(exc).__name__}: {exc}"
                with self._lock:
                    self._state[kind].update(busy=False, error=error, ended=time.time())
                    said = list(self._state[kind]["lines"][-5:])
                ALERTS.say("training", f"{what}: {error}", said)
            finally:
                self.on_change()

    def _submit(self, kind: str, what: str, fn) -> None:
        self._queues[kind].put((what, fn))

    # --- asking ---------------------------------------------------------
    def probe(self, name: str) -> dict:
        found = probe(self.store, name)
        with self._lock:
            self.probes[name] = found
        return found

    def launch(self, payload: dict) -> dict:
        plan = self._plan(payload)
        self._submit("launch", f"start {plan['run']} on {plan['source']}", lambda say: self._launch(plan, say))
        return plan

    # The same, done now on the caller's thread: what the command line uses.
    def launch_now(self, payload: dict, say=print) -> dict:
        return self._launch(self._plan(payload), say)

    def stop_now(self, run_id: int, say=print) -> dict:
        return self._stop(self.store.run(run_id), say)

    def resume_now(self, run_id: int, say=print) -> dict:
        return self._resume(self.store.run(run_id), say)

    def fetch_now(self, run_id: int, step: int, say=print) -> dict:
        return self._fetch(self.store.run(run_id), step, say)

    def stop(self, run_id: int) -> None:
        run = self.store.run(run_id)
        self._submit("launch", f"stop {run['name']}", lambda say: self._stop(run, say))

    def resume(self, run_id: int) -> None:
        run = self.store.run(run_id)
        if not run["checkpoints"]:
            raise LaunchError("nothing to resume from: no checkpoint has been saved")
        self._submit("launch", f"resume {run['name']}", lambda say: self._resume(run, say))

    def fetch(self, run_id: int, step: int) -> None:
        run = self.store.run(run_id)
        if (run_id, step) in self._queued_fetches:
            return
        with self._lock:
            self._queued_fetches.add((run_id, step))
        self._submit("fetch", f"{run['name']} step {step}", lambda say: self._fetch(run, step, say))

    def fetch_due(self) -> None:
        """Every checkpoint of a run that fetches its own, not yet here."""
        for c in self.store.to_fetch():
            self.fetch(c["run_id"], c["step"])

    # --- doing ------------------------------------------------------------
    def _plan(self, payload: dict) -> dict:
        name = str(payload.get("source") or "")
        source = self.store.source(name)
        policy = str(payload.get("policy") or "")
        if policy not in RECIPES:
            raise LaunchError(f"policy is one of {', '.join(RECIPES)}")
        dataset = str(payload.get("dataset") or "")
        local_data = dataset_dir(dataset)
        if not dataset or not (local_data / "meta" / "info.json").is_file():
            raise LaunchError(f"no dataset {dataset!r} under {local_data.parent}")
        try:
            require_cut(local_data)
        except LensMissing as exc:
            raise LaunchError(str(exc)) from exc
        recipe = RECIPES[policy]
        steps = int(payload.get("steps") or DEFAULTS["steps"])
        run = str(payload.get("name") or "").strip() or f"{policy}-{dataset}-{time.strftime('%m%d-%H%M')}"
        if not all(c.isalnum() or c in "-_." for c in run):
            raise LaunchError("a run name is letters, digits and -_.")
        args = list(recipe["args"]) + [
            "--policy.device=cuda", "--policy.push_to_hub=false", "--dataset.image_transforms.enable=true",
            f"--steps={steps}",
            f"--batch_size={int(payload.get('batch_size') or recipe['batch_size'])}",
            f"--log_freq={int(payload.get('log_freq') or DEFAULTS['log_freq'])}",
            f"--save_freq={int(payload.get('save_freq') or DEFAULTS['save_freq'])}",
            f"--prefetch_factor={recipe['prefetch']}",
        ]
        chunk = steps_in(recipe["chunk_s"], dataset_fps(local_data))
        args += [f"--policy.chunk_size={chunk}", f"--policy.n_action_steps={chunk}"]
        extra = shlex.split(str(payload.get("extra") or ""))
        return {"source": name, "kind": source["kind"], "policy": policy, "dataset": dataset, "run": run,
                "args": args, "extra": extra, "workers": recipe["workers"], "gpu": str(payload.get("gpu") or "0"),
                "auto_fetch": bool(payload.get("auto_fetch", True))}

    def _ready(self, source: dict, say) -> dict:
        """The machine set up to train: the kit there, and a venv from this lock."""
        say(f"asking {source['name']} what it has")
        found = self.probe(source["name"])
        if source["kind"] == "local":
            return found
        root = found["root"]
        lock = CORE.remote_lock_file

        def kit(tar: tarfile.TarFile) -> None:
            for arcname, data in (("run_log.py", Path(run_log.__file__).read_bytes()), ("setup.sh", SETUP_SH.encode()),
                                  ("requirements.lock.txt", lock.read_bytes())):
                info = tarfile.TarInfo(arcname)
                info.size, info.mode, info.mtime = len(data), 0o755 if arcname.endswith(".sh") else 0o644, time.time()
                tar.addfile(info, io.BytesIO(data))
        say("sending run_log.py, setup.sh and the lock")
        _send_tar(source, shlex.quote(root), kit, 0, say)
        if not (found["venv"] and found["lock_matches"]):
            say("setting up the venv there - the first time takes a few minutes")
            _stream(source, f"bash {shlex.quote(root)}/setup.sh", say, timeout=40 * 60)
        return found

    def _send_dataset(self, source: dict, found: dict, dataset: str, say) -> str:
        """The dataset on the machine, sent if it is not there as it is here.
        Returns where it is there."""
        local = dataset_dir(dataset)
        fp = fingerprint(local)
        there = f"{dataset}--{fp}"
        if source["kind"] == "local":
            # Here too the run is pointed at <name>--<fingerprint> - a link to
            # the dataset, not a copy - so its checkpoints say which dataset,
            # as it was, they learned from.
            link = Path(found["root"]).expanduser() / "datasets" / there
            link.parent.mkdir(parents=True, exist_ok=True)
            if not link.is_symlink():
                link.symlink_to(local.resolve(), target_is_directory=True)
            return str(link)
        path = f"{found['root']}/datasets/{there}"
        if there in found.get("datasets", []):
            say(f"{dataset} is there already ({fp})")
            return path
        files = [p for p in local.rglob("*") if p.is_file()]
        total = sum(p.stat().st_size for p in files)
        say(f"sending {dataset}: {len(files)} files, {total / 1e6:.0f} MB")
        part = f"{found['root']}/datasets/.{there}.part"
        _run(source, f"rm -rf {shlex.quote(part)}")

        def add(tar: tarfile.TarFile) -> None:
            tar.add(local, arcname=".")
        _send_tar(source, shlex.quote(part), add, total, say)
        _run(source, f"echo {fp} > {shlex.quote(part)}/{FINGERPRINT} && mv {shlex.quote(part)} {shlex.quote(path)}")
        return path

    def _start(self, source: dict, root: str, run: str, args: list[str], env: dict, say) -> str:
        """The command, detached, with its terminal in logs/. Returns the log's path."""
        stamp = time.strftime("%m%d-%H%M%S")
        if source["kind"] == "local":
            logs = CORE.train_dir / "logs"
            logs.mkdir(parents=True, exist_ok=True)
            log = logs / f"{run}-{stamp}.log"
            with open(log, "wb") as out:
                subprocess.Popen([sys.executable, "-m", "pinkrab_policy.run_log", *args], stdout=out,
                                 stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
                                 env={**os.environ, **env}, cwd=str(CORE.train_dir))
            return str(log)
        log = f"{root}/logs/{run}-{stamp}.log"
        exports = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
        # The `;` keeps the `&` on the run alone. After `&&` it would put the
        # whole list in the background, as a shell that holds the ssh channel
        # until the run ends, and the launch would wait out its timeout.
        command = (f"cd {shlex.quote(root)} && mkdir -p logs || exit 1; {exports} setsid nohup .venv/bin/python run_log.py "
                   f"{' '.join(shlex.quote(a) for a in args)} > {shlex.quote(log)} 2>&1 < /dev/null & echo started")
        _run(source, command)
        return log

    def _launch(self, plan: dict, say) -> dict:
        source = self.store.source(plan["source"])
        found = self._ready(source, say)
        root = found["root"]
        chosen = {i.strip() for i in plan["gpu"].split(",")}
        gpus = [g for g in found.get("gpus") or [] if str(g.get("index")) in chosen]
        short = shortfalls(gpus)
        if short:
            raise LaunchError(f"{source['name']}: " + "; ".join(short))
        for g in gpus:
            if float(g.get("mem_total_mb") or 0) / 1024 < MIN_GPU_MEMORY_GB:
                say(f"GPU {g['index']} has {float(g['mem_total_mb']) / 1024:.0f} GB - a smaller batch may be needed")
        data = self._send_dataset(source, found, plan["dataset"], say)
        # Here, runs sit straight in the train directory, where the policy page
        # looks; there, under train/ beside the kit.
        output = f"{root}/{plan['run']}" if plan["kind"] == "local" else f"{root}/train/{plan['run']}"
        if output in found.get("running", []):
            raise LaunchError(f"{plan['run']} is already running there")
        workers = max(2, min(plan["workers"], (found.get("cores") or 4) - 2))
        args = plan["args"] + [f"--num_workers={workers}", f"--dataset.repo_id=pinkrab/{plan['dataset']}",
                               f"--dataset.root={data}", f"--output_dir={output}", f"--job_name={plan['run']}",
                               *plan["extra"]]
        env = {"CUDA_VISIBLE_DEVICES": plan["gpu"]}
        if source["kind"] != "local":
            env |= {"TORCH_HOME": f"{root}/torch-cache", "HF_HOME": f"{root}/hf-cache"}
        say(f"starting {plan['run']}")
        log = self._start(source, root, plan["run"], args, env, say)
        self.store.record_launch(source["name"], output, plan["auto_fetch"], plan, log)
        say(f"started - its terminal is {log}")
        return {"run": plan["run"], "output_dir": output, "log": log}

    def _stop(self, run: dict, say) -> dict:
        source = self.store.source(run["source"])
        # The bracket keeps this pattern from matching the command that carries it.
        # Started with --output_dir=<dir>, or resumed with --config_path=<dir>/checkpoints/...
        out = re.sub(r"([.\[\]()*+?{}|^$\\])", r"\\\1", run["output_dir"])   # POSIX ERE, for pkill
        pattern = shlex.quote(f"[-]-(output_dir={out}( |$)|config_path={out}/checkpoints/)")
        say(f"stopping {run['name']}")
        _run(source, f"pkill -TERM -f -- {pattern} || true")
        return {"stopped": run["name"]}

    def _resume(self, run: dict, say) -> dict:
        source = self.store.source(run["source"])
        found = self._ready(source, say)
        if run["output_dir"] in found.get("running", []):
            raise LaunchError(f"{run['name']} is still running there")
        config = f"{run['output_dir']}/checkpoints/last/pretrained_model/train_config.json"
        if source["kind"] == "local":
            # The run's <name>--<fp> is a link here; resumed after the dataset
            # was converted again, it would learn on from other takes.
            root = Path((json.loads(Path(config).read_text()).get("dataset") or {}).get("root") or "")
            sent = re.search(r"--([0-9a-f]{12})$", root.name)
            if sent and root.is_dir() and fingerprint(root) != sent.group(1):
                raise LaunchError(f"{run['name']} learned from {root.name}, and that dataset has changed since "
                                  "- start a new run")
        recipe = json.loads((run.get("launch") or {}).get("recipe") or "{}")
        env = {"CUDA_VISIBLE_DEVICES": str(recipe.get("gpu") or "0")}
        if source["kind"] != "local":
            env |= {"TORCH_HOME": f"{found['root']}/torch-cache", "HF_HOME": f"{found['root']}/hf-cache"}
        say(f"resuming {run['name']} from its last checkpoint")
        log = self._start(source, found["root"], f"{run['name']}-resume", [f"--config_path={config}", "--resume=true"],
                          env, say)
        return {"resumed": run["name"], "log": log}

    def _fetch(self, run: dict, step: int, say) -> dict:
        try:
            source = self.store.source(run["source"])
            c = next((c for c in run["checkpoints"] if c["step"] == step), None)
            if c is None or not c.get("path"):
                raise LaunchError(f"{run['name']} has no checkpoint at step {step}")
            where = Path(c["path"]).name
            dest = CORE.train_dir / run["name"] / "checkpoints" / where
            if source["kind"] == "local":
                self.store.mark_fetched(run["id"], step, c["path"])
                return {"local": c["path"]}
            size = int(_run(source, f"du -sb {shlex.quote(c['path'])}/pretrained_model | cut -f1").strip() or 0)
            say(f"copying {run['name']} step {step} ({size / 1e6:.0f} MB)")
            part = dest.parent / f".{where}.part"
            shutil.rmtree(part, ignore_errors=True)
            part.mkdir(parents=True)
            argv = ssh_base(source) + [f"tar -C {shlex.quote(c['path'])} -cf - pretrained_model"]
            with subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc, \
                    subprocess.Popen(["tar", "-C", str(part), "-xf", "-"], stdin=subprocess.PIPE) as untar:
                got, told = 0, time.monotonic()
                while chunk := proc.stdout.read(1 << 20):
                    untar.stdin.write(chunk)
                    got += len(chunk)
                    if time.monotonic() - told > 2.0:
                        told = time.monotonic()
                        say(f"copying {got / 1e6:.0f} / {size / 1e6:.0f} MB")
                untar.stdin.close()
                if proc.wait() != 0 or untar.wait() != 0:
                    raise LaunchError(f"copy failed: {proc.stderr.read().decode('utf-8', 'replace').strip()[-200:]}")
            shutil.rmtree(dest, ignore_errors=True)
            part.rename(dest)
            local = str(dest / "pretrained_model")
            self.store.mark_fetched(run["id"], step, local)
            say(f"here: {local}")
            return {"local": local}
        finally:
            with self._lock:
                self._queued_fetches.discard((run["id"], step))
