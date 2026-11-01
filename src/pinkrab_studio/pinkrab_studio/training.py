"""Training runs, from wherever they are trained, in one sqlite file.

A run is read from its files, never from the process: whatever trains - this
machine, a cloud box reached over ssh - only has to leave its `metrics.jsonl`
where a source points. `pinkrab-train` (pinkrab_policy.run_log) writes it: the
exact numbers lerobot-train would have sent to wandb, the checkpoints as they
are saved, and a closing line.

Each read also asks the machine what is on its GPUs, so a training this
studio does not record - started without the wrapper - is at least seen to be
there, by the output directory on its command line, with no numbers.

Files are read from where the last read stopped, whole lines only, so a poll
costs what the run has written since the one before. The store keeps the
points; nothing here keeps a file open or a connection up between polls.
"""
from __future__ import annotations

import base64
import inspect
import json
import os
import re
import shlex
import sqlite3
import subprocess
import threading
import time
from pathlib import Path

from pinkrab_core.config import SETTINGS as CORE

LOCAL = "this machine"
SOURCE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.-]{0,39}$")
# Where a run's metrics are, under a source's root: in the run's directory,
# beside its checkpoints, or one level down (a root that keeps runs under train/).
PATTERNS = ("*/metrics.jsonl", "*/*/metrics.jsonl")
# The most read from one file in one poll. A long run's first read is its
# whole history; this lets it arrive over a few polls rather than one stall.
CHUNK_BYTES = 4 << 20
SSH_TIMEOUT_S = 60.0

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    name TEXT PRIMARY KEY, kind TEXT NOT NULL, host TEXT, port INTEGER, user TEXT,
    key TEXT, root TEXT NOT NULL, synced REAL, error TEXT);
CREATE TABLE IF NOT EXISTS files (
    source TEXT NOT NULL, path TEXT NOT NULL, offset INTEGER NOT NULL DEFAULT 0,
    grown REAL, state TEXT, PRIMARY KEY (source, path));
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY, source TEXT NOT NULL, output_dir TEXT NOT NULL,
    origin TEXT NOT NULL, name TEXT, policy TEXT, dataset TEXT, steps INTEGER,
    batch_size INTEGER, log_freq INTEGER, host TEXT, config TEXT, started REAL,
    updated REAL, last_step INTEGER, status TEXT, error TEXT, UNIQUE (source, output_dir));
CREATE TABLE IF NOT EXISTS points (
    run_id INTEGER NOT NULL, mode TEXT NOT NULL, metric TEXT NOT NULL,
    step INTEGER NOT NULL, value REAL, t REAL,
    PRIMARY KEY (run_id, mode, metric, step)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS checkpoints (
    run_id INTEGER NOT NULL, step INTEGER NOT NULL, path TEXT, t REAL, local TEXT,
    PRIMARY KEY (run_id, step));
CREATE TABLE IF NOT EXISTS launches (
    source TEXT NOT NULL, output_dir TEXT NOT NULL, auto_fetch INTEGER NOT NULL DEFAULT 1,
    recipe TEXT, log TEXT, t REAL, PRIMARY KEY (source, output_dir));
"""
# Columns added after a store was first made: added to an older file on open.
# An older file keeps the runs' `hidden` column, which nothing reads now: left
# there, a studio still running from before reads the same file unbroken.
MIGRATIONS = ("ALTER TABLE checkpoints ADD COLUMN local TEXT", "ALTER TABLE sources ADD COLUMN gpu TEXT")
# How a training names its run on its command line - the same argument Stop
# finds a run's process by.
OUTPUT_DIR_ARG = "--output_dir="

class TrainError(RuntimeError):
    pass


def keep_key(name: str, text: str) -> str:
    """A source's private key, copied where only this user reads it and
    nothing else tidies: a key left in Downloads is a login that ends when
    that folder is cleared. Returns the kept file's path."""
    if "PRIVATE KEY" not in text:
        raise TrainError("that is not a private key file")
    CORE.keys_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(CORE.keys_dir, 0o700)
    path = CORE.keys_dir / f"{name.replace(' ', '_')}.key"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        # ssh refuses a key whose last line does not end.
        f.write(text.replace("\r\n", "\n").rstrip("\n") + "\n")
    os.chmod(path, 0o600)
    return str(path)


def parse_ssh_line(text: str) -> dict:
    """Host, port, user and key out of a pasted ssh, sftp or scp command -
    the lines a cloud console hands out for a session."""
    words = shlex.split(text.strip())
    found: dict = {}
    i = 1
    while i < len(words):
        w = words[i]
        if w in ("-i", "-p", "-P") and i + 1 < len(words):
            value = words[i + 1]
            if w == "-i":
                found["key"] = value
            elif value.isdigit():
                found["port"] = int(value)
            i += 2
            continue
        if w == "-o" and i + 1 < len(words):
            i += 2
            continue
        if "@" in w and not w.startswith("-"):
            user, _, host = w.partition("@")
            found["user"], found["host"] = user, host.split(":", 1)[0]
        i += 1
    if not found.get("host"):
        raise TrainError("no user@host in that line")
    return found


# --- reading files -----------------------------------------------------------------
def gpu_processes():
    """What is on this machine's GPUs: each process's pid, its memory and its
    command line. None where there is no nvidia-smi to ask.

    Standard library only, and self-contained: its source is sent to a remote
    machine and run there as it is run here."""
    import subprocess
    try:
        done = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if done.returncode != 0:
        return None
    out = []
    for line in done.stdout.splitlines():
        fields = [f.strip() for f in line.split(",")]
        if len(fields) < 2 or not fields[0].isdigit():
            continue
        try:
            with open(f"/proc/{fields[0]}/cmdline", "rb") as f:
                args = [a.decode("utf-8", "replace") for a in f.read().split(b"\0") if a]
        except OSError:
            args = []
        try:
            memory = float(fields[1])
        except ValueError:
            memory = None
        out.append({"pid": int(fields[0]), "memory_mb": memory, "args": args})
    return out


REMOTE_READER = inspect.getsource(gpu_processes).replace("%", "%%") + r"""
import base64, glob, json, os
REQ = json.loads(%r)
root = os.path.expanduser(REQ["root"])
out = {}
for pattern in REQ["patterns"]:
    for path in glob.glob(os.path.join(root, pattern)):
        rel = os.path.relpath(path, root)
        try:
            size = os.path.getsize(path)
        except OSError:
            continue
        start = REQ["offsets"].get(rel, 0)
        if size < start:
            start = 0
        data = b""
        if size > start:
            with open(path, "rb") as f:
                f.seek(start)
                data = f.read(min(size - start, REQ["chunk"]))
        out[rel] = {"size": size, "start": start, "mtime": os.path.getmtime(path),
                    "data": base64.b64encode(data).decode()}
print(json.dumps({"root": root, "files": out, "gpu": gpu_processes()}))
"""


def _read_local(root: Path, offsets: dict[str, int]) -> dict[str, dict]:
    out = {}
    for pattern in PATTERNS:
        for path in root.glob(pattern):
            rel = str(path.relative_to(root))
            try:
                stat = path.stat()
            except OSError:
                continue
            size = stat.st_size
            start = offsets.get(rel, 0)
            if size < start:
                start = 0
            data = b""
            if size > start:
                with open(path, "rb") as f:
                    f.seek(start)
                    data = f.read(min(size - start, CHUNK_BYTES))
            out[rel] = {"size": size, "start": start, "mtime": stat.st_mtime, "data": data}
    return out


def ssh_base(source: dict) -> list[str]:
    """ssh to a source, up to and including user@host: a command to run there
    goes after it."""
    command = ["ssh", "-p", str(source["port"] or 22),
               "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
               "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
               "-o", "LogLevel=ERROR",
               # One connection per poll, not a shared master: a socket left by an
               # earlier session to the same host:port is a session that ended.
               "-o", "ControlMaster=no", "-o", "ControlPath=none"]
    if source.get("key"):
        command += ["-i", str(Path(source["key"]).expanduser())]
    return command + [f"{source['user'] or 'root'}@{source['host']}"]


def _ssh_command(source: dict) -> list[str]:
    return ssh_base(source) + ["python3", "-"]


def check_key(source: dict) -> None:
    """A key that is set but gone says so, rather than as the login refused."""
    if source.get("key") and not Path(source["key"]).expanduser().is_file():
        raise TrainError(f"the key file {source['key']} is not there any more - set the key again in Sources")


def _read_ssh(source: dict, offsets: dict[str, int]) -> tuple[dict[str, dict], list | None]:
    check_key(source)
    request = json.dumps({"root": source["root"], "offsets": offsets,
                          "patterns": list(PATTERNS), "chunk": CHUNK_BYTES})
    try:
        done = subprocess.run(_ssh_command(source), input=REMOTE_READER % request, capture_output=True,
                              text=True, timeout=SSH_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        raise TrainError(f"no answer in {SSH_TIMEOUT_S:.0f} s") from exc
    if done.returncode != 0:
        why = (done.stderr.strip().splitlines() or [f"ssh exited {done.returncode}"])[-1]
        raise TrainError(why)
    try:
        answer = json.loads(done.stdout)
    except ValueError as exc:
        raise TrainError("the remote answered with something that is not the file list") from exc
    files = {rel: {**f, "data": base64.b64decode(f["data"])} for rel, f in answer["files"].items()}
    return files, answer.get("gpu")


def _output_dir(args: list[str]) -> str | None:
    for arg in args:
        if arg.startswith(OUTPUT_DIR_ARG):
            return arg[len(OUTPUT_DIR_ARG):]
    return None


def _unrecorded(process: dict) -> dict:
    """A GPU process no recorded run accounts for, named by its output
    directory when it has one, by the programme it runs when not."""
    output_dir = _output_dir(process["args"])
    name = Path(output_dir).name if output_dir else " ".join(Path(a).name for a in process["args"][:2]) or f"pid {process['pid']}"
    return {"name": name, "pid": process["pid"], "memory_mb": process["memory_mb"]}


def _whole_lines(data: bytes) -> tuple[bytes, bytes]:
    """What can be read now, and what waits for the rest of its line."""
    cut = data.rfind(b"\n")
    return (data[:cut + 1], data[cut + 1:]) if cut >= 0 else (b"", data)


class TrainStore:
    """The sqlite file, and the one lock every write takes."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or CORE.train_db
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._db() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(SCHEMA)
            for statement in MIGRATIONS:
                try:
                    db.execute(statement)
                except sqlite3.OperationalError:
                    pass  # already there
            db.execute("INSERT OR IGNORE INTO sources (name, kind, root) VALUES (?, 'local', ?)",
                       (LOCAL, str(CORE.train_dir)))
            db.execute("UPDATE sources SET root = ? WHERE name = ?", (str(CORE.train_dir), LOCAL))

    def _db(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=10.0)
        db.row_factory = sqlite3.Row
        return db

    # --- sources ----------------------------------------------------------
    def sources(self) -> list[dict]:
        """Each source, with its run count and, from its last read, what is on
        its GPUs that no run recorded here accounts for (`unrecorded`)."""
        with self._db() as db:
            rows = [dict(r) for r in db.execute("SELECT * FROM sources ORDER BY kind DESC, name")]
            counts = dict(db.execute("SELECT source, COUNT(*) FROM runs GROUP BY source").fetchall())
            known = {}
            for source, output_dir in db.execute("SELECT source, output_dir FROM runs"):
                known.setdefault(source, set()).add(output_dir)
        for row in rows:
            row["runs"] = counts.get(row["name"], 0)
            gpu = json.loads(row.pop("gpu") or "null")
            row["unrecorded"] = None if gpu is None else [
                _unrecorded(p) for p in gpu if _output_dir(p["args"]) not in known.get(row["name"], set())]
        return rows

    def save_source(self, payload: dict) -> dict:
        name = str(payload.get("name") or "").strip()
        if not SOURCE_NAME.match(name) or name == LOCAL:
            raise TrainError("a source needs a name: letters, digits, space, _.-")
        found = parse_ssh_line(payload["line"]) if payload.get("line") else {}
        host = str(payload.get("host") or found.get("host") or "").strip()
        if not host:
            raise TrainError("a host, or a pasted ssh/sftp line with user@host in it")
        port = int(payload.get("port") or found.get("port") or 22)
        user = str(payload.get("user") or found.get("user") or "").strip() or None
        key = str(payload.get("key") or "").strip() or None
        if key is None and found.get("key") and Path(found["key"]).expanduser().is_file():
            # A pasted line's -i is often relative to wherever its console
            # expects the key; it is taken only when it names a file here.
            key = found["key"]
        if key and Path(key).expanduser().parent != CORE.keys_dir:
            source = Path(key).expanduser()
            if not source.is_file():
                raise TrainError(f"no key file at {key}")
            key = keep_key(name, source.read_text())
        root = str(payload.get("root") or "").strip()
        if not root:
            raise TrainError("the directory on that machine the runs are under")
        with self._lock, self._db() as db:
            db.execute("""INSERT INTO sources (name, kind, host, port, user, key, root) VALUES (?, 'ssh', ?, ?, ?, ?, ?)
                          ON CONFLICT(name) DO UPDATE SET host=excluded.host, port=excluded.port, user=excluded.user,
                          key=excluded.key, root=excluded.root, error=NULL""",
                       (name, host, port, user, key, root))
        return self.source(name)

    def set_key(self, name: str, key: str) -> None:
        with self._lock, self._db() as db:
            db.execute("UPDATE sources SET key = ? WHERE name = ?", (key, name))

    def source(self, name: str) -> dict:
        with self._db() as db:
            row = db.execute("SELECT * FROM sources WHERE name = ?", (name,)).fetchone()
        if row is None:
            raise TrainError(f"no source {name!r}")
        return dict(row)

    def delete_source(self, name: str) -> None:
        if name == LOCAL:
            raise TrainError("this machine's own runs are always read")
        with self._lock, self._db() as db:
            ids = [r[0] for r in db.execute("SELECT id FROM runs WHERE source = ?", (name,))]
            for table in ("points", "checkpoints"):
                db.executemany(f"DELETE FROM {table} WHERE run_id = ?", [(i,) for i in ids])
            db.execute("DELETE FROM runs WHERE source = ?", (name,))
            db.execute("DELETE FROM files WHERE source = ?", (name,))
            db.execute("DELETE FROM sources WHERE name = ?", (name,))

    # --- syncing ----------------------------------------------------------
    def sync(self, name: str) -> dict:
        """Read what one source's files have gained since the last read."""
        source = self.source(name)
        with self._db() as db:
            offsets = {r["path"]: r["offset"] for r in db.execute("SELECT path, offset FROM files WHERE source = ?", (name,))}
        try:
            if source["kind"] == "local":
                root = Path(source["root"]).expanduser()
                found = _read_local(root, offsets) if root.is_dir() else {}
                gpu = gpu_processes()
            else:
                found, gpu = _read_ssh(source, offsets)
        except (TrainError, OSError) as exc:
            with self._lock, self._db() as db:
                db.execute("UPDATE sources SET error = ?, synced = ? WHERE name = ?", (str(exc), time.time(), name))
            raise TrainError(f"{name}: {exc}") from exc
        now = time.time()
        for rel in sorted(found):
            self._take(name, rel, found[rel], now)
        with self._lock, self._db() as db:
            db.execute("UPDATE sources SET error = NULL, synced = ?, gpu = ? WHERE name = ?",
                       (now, None if gpu is None else json.dumps(gpu), name))
        return self.source(name)

    def _take(self, source: str, rel: str, file: dict, now: float) -> None:
        lines, _rest = _whole_lines(file["data"])
        with self._lock, self._db() as db:
            row = db.execute("SELECT state, offset FROM files WHERE source = ? AND path = ?", (source, rel)).fetchone()
            state = json.loads(row["state"]) if row and row["state"] else {}
            if file["start"] == 0 and row and row["offset"] > 0:
                state = {}  # the file was started again from the top
            grew = bool(file["data"])
            self._take_jsonl(db, source, lines.decode("utf-8", "replace"), state)
            # A file written lately is news as of this read - a network disk
            # reports its mtime a minute late, and another machine's clock is
            # not this one's. A file last written long ago is history, however
            # many reads its first pass takes, and says when it was written.
            if grew and state.get("run_id"):
                mtime = file.get("mtime") or now
                said = now if now - mtime < CORE.train_stale_s else mtime
                db.execute("UPDATE runs SET updated = MAX(COALESCE(updated, 0), ?) WHERE id = ?",
                           (said, state["run_id"]))
            db.execute("""INSERT INTO files (source, path, offset, grown, state) VALUES (?, ?, ?, ?, ?)
                          ON CONFLICT(source, path) DO UPDATE SET offset=excluded.offset,
                          grown=COALESCE(excluded.grown, files.grown), state=excluded.state""",
                       (source, rel, file["start"] + len(lines), now if grew else None, json.dumps(state)))

    @staticmethod
    def _run_id(db, source: str, output_dir: str, fields: dict) -> int:
        """The run for this output directory, made if new; what it says about
        itself written each time it starts (a resume starts it again)."""
        row = db.execute("SELECT id FROM runs WHERE source = ? AND output_dir = ?", (source, output_dir)).fetchone()
        if row is None:
            cursor = db.execute(
                """INSERT INTO runs (source, output_dir, origin, name, policy, dataset, steps, batch_size,
                   log_freq, host, config, started, status) VALUES (?, ?, 'metrics', ?,?,?,?,?,?,?,?,?, 'running')""",
                (source, output_dir, fields.get("name"), fields.get("policy"), fields.get("dataset"),
                 fields.get("steps"), fields.get("batch_size"), fields.get("log_freq"), fields.get("host"),
                 fields.get("config"), fields.get("started")))
            return cursor.lastrowid
        db.execute("""UPDATE runs SET name = COALESCE(?, name), policy = COALESCE(?, policy),
                      dataset = COALESCE(?, dataset), steps = COALESCE(?, steps),
                      batch_size = COALESCE(?, batch_size), log_freq = COALESCE(?, log_freq),
                      host = COALESCE(?, host), config = COALESCE(?, config),
                      started = COALESCE(started, ?), status = 'running', error = NULL WHERE id = ?""",
                   (fields.get("name"), fields.get("policy"), fields.get("dataset"), fields.get("steps"),
                    fields.get("batch_size"), fields.get("log_freq"), fields.get("host"), fields.get("config"),
                    fields.get("started"), row["id"]))
        return row["id"]

    def _take_jsonl(self, db, source: str, text: str, state: dict) -> None:
        for raw in text.splitlines():
            try:
                line = json.loads(raw)
            except ValueError:
                continue
            kind = line.get("kind")
            if kind == "run":
                fields = {k: line.get(k) for k in ("name", "policy", "dataset", "steps", "batch_size", "log_freq", "host")}
                fields["config"] = json.dumps(line.get("config")) if line.get("config") is not None else None
                fields["started"] = line.get("t")
                state["run_id"] = self._run_id(db, source, line.get("output_dir") or "", fields)
                continue
            run_id = state.get("run_id")
            if run_id is None:
                continue
            if kind == "metrics" and line.get("step") is not None:
                step, mode, t = int(line["step"]), line.get("mode") or "train", line.get("t")
                db.executemany("INSERT OR REPLACE INTO points VALUES (?, ?, ?, ?, ?, ?)",
                               [(run_id, mode, k, step, v, t) for k, v in (line.get("values") or {}).items()])
                if mode == "train":
                    db.execute("UPDATE runs SET last_step = MAX(COALESCE(last_step, 0), ?) WHERE id = ?", (step, run_id))
            elif kind == "checkpoint" and line.get("step") is not None:
                db.execute("INSERT OR IGNORE INTO checkpoints (run_id, step, path, t) VALUES (?, ?, ?, ?)",
                           (run_id, int(line["step"]), line.get("path"), line.get("t")))
            elif kind == "end":
                db.execute("UPDATE runs SET status = ?, error = ? WHERE id = ?",
                           (line.get("status") or "done", line.get("error"), run_id))

    # --- reading back -----------------------------------------------------
    def runs(self) -> list[dict]:
        now = time.time()
        with self._db() as db:
            rows = [dict(r) for r in db.execute(
                "SELECT id, source, output_dir, name, policy, dataset, steps, batch_size, log_freq, host, "
                "started, updated, last_step, status, error FROM runs ORDER BY started DESC")]
            for row in rows:
                recent = db.execute(
                    "SELECT step, t, value FROM points WHERE run_id = ? AND mode = 'train' AND metric = 'loss' "
                    "ORDER BY step DESC LIMIT 20", (row["id"],)).fetchall()
                row["loss"] = recent[0]["value"] if recent else None
                timed = [r for r in recent if r["t"] is not None]
                rate = None
                if len(timed) >= 2 and timed[0]["t"] > timed[-1]["t"]:
                    rate = (timed[0]["step"] - timed[-1]["step"]) / (timed[0]["t"] - timed[-1]["t"])
                row["rate"] = rate
                row["checkpoints"], row["fetched"] = db.execute(
                    "SELECT COUNT(*), COUNT(local) FROM checkpoints WHERE run_id = ?", (row["id"],)).fetchone()
                launch = db.execute("SELECT auto_fetch FROM launches WHERE source = ? AND output_dir = ?",
                                    (row["source"], row["output_dir"])).fetchone()
                row["launched"] = launch is not None
                row["auto_fetch"] = bool(launch and launch[0])
                if row["status"] == "running" and (row["updated"] is None or now - row["updated"] > CORE.train_stale_s):
                    row["status"] = "quiet"
                left = (row["steps"] or 0) - (row["last_step"] or 0)
                row["eta_s"] = left / rate if rate and row["status"] == "running" and left > 0 else None
        return rows

    def series(self, ids: list[int]) -> dict:
        out: dict = {}
        with self._db() as db:
            for run_id in ids:
                metrics: dict[str, list[list]] = {}
                for r in db.execute("SELECT mode, metric, step, value, t FROM points WHERE run_id = ? "
                                    "ORDER BY mode, metric, step", (run_id,)):
                    key = r["metric"] if r["mode"] == "train" else f"{r['mode']}/{r['metric']}"
                    s = metrics.setdefault(key, [[], [], []])
                    s[0].append(r["step"])
                    s[1].append(r["value"])
                    s[2].append(r["t"])
                out[str(run_id)] = metrics
        return out

    def run(self, run_id: int) -> dict:
        with self._db() as db:
            row = db.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            if row is None:
                raise TrainError(f"no run {run_id}")
            found = dict(row)
            found["config"] = json.loads(found["config"]) if found.get("config") else None
            found["checkpoints"] = [dict(r) for r in db.execute(
                "SELECT step, path, t, local FROM checkpoints WHERE run_id = ? ORDER BY step", (run_id,))]
            launch = db.execute("SELECT * FROM launches WHERE source = ? AND output_dir = ?",
                                (found["source"], found["output_dir"])).fetchone()
            found["launch"] = dict(launch) if launch else None
        return found

    def record_launch(self, source: str, output_dir: str, auto_fetch: bool, recipe: dict, log: str) -> None:
        with self._lock, self._db() as db:
            db.execute("INSERT OR REPLACE INTO launches VALUES (?, ?, ?, ?, ?, ?)",
                       (source, output_dir, 1 if auto_fetch else 0, json.dumps(recipe), log, time.time()))

    def set_auto_fetch(self, run_id: int, on: bool) -> None:
        run = self.run(run_id)
        with self._lock, self._db() as db:
            db.execute("""INSERT INTO launches (source, output_dir, auto_fetch, t) VALUES (?, ?, ?, ?)
                          ON CONFLICT(source, output_dir) DO UPDATE SET auto_fetch = excluded.auto_fetch""",
                       (run["source"], run["output_dir"], 1 if on else 0, time.time()))

    def mark_fetched(self, run_id: int, step: int, local: str) -> None:
        with self._lock, self._db() as db:
            db.execute("UPDATE checkpoints SET local = ? WHERE run_id = ? AND step = ?", (local, run_id, step))

    def to_fetch(self) -> list[dict]:
        """Checkpoints of runs that fetch their own, saved and not yet here."""
        with self._db() as db:
            return [dict(r) for r in db.execute(
                """SELECT c.run_id, c.step, c.path FROM checkpoints c JOIN runs r ON r.id = c.run_id
                   JOIN launches l ON l.source = r.source AND l.output_dir = r.output_dir
                   WHERE l.auto_fetch = 1 AND c.local IS NULL AND r.source != ? ORDER BY c.run_id, c.step""",
                (LOCAL,))]


class TrainWatch:
    """The store, kept up to date on a thread of its own while the studio is up."""

    def __init__(self, store: TrainStore | None = None) -> None:
        self.store = store or TrainStore()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._due: dict[str, float] = {}
        self._kick = threading.Event()
        # Called after every pass that read something: what reacts to new
        # lines (a checkpoint to copy home) hangs here.
        self.after_sync: list = []

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="train-watch", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._kick.set()

    def now(self) -> None:
        """Every source read on the next turn, not when its interval comes round."""
        self._due.clear()
        self._kick.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            read = False
            for source in self.store.sources():
                name = source["name"]
                if time.monotonic() < self._due.get(name, 0.0):
                    continue
                every = CORE.train_poll_s if source["kind"] == "local" else CORE.train_remote_poll_s
                self._due[name] = time.monotonic() + every
                try:
                    self.store.sync(name)
                    read = True
                except TrainError:
                    pass  # kept on the source row, where the page shows it
                except Exception as exc:  # noqa: BLE001 - one bad file must not stop the watch
                    print(f"  training page: reading {name} failed: {exc!r}", flush=True)
            if read:
                for hook in self.after_sync:
                    try:
                        hook()
                    except Exception as exc:  # noqa: BLE001
                        print(f"  training page: {exc!r}", flush=True)
            self._kick.wait(1.0)
            self._kick.clear()
