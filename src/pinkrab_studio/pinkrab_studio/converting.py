"""Converting episodes into a dataset, from the page.

One conversion at a time, on a thread of its own: `pinkrab-umi convert` is
the same function, and a page that blocked for five minutes would show
nothing meanwhile. The page asks with a name and a task, and every complete
episode on the shelf that passes the shelf's tests goes in (verdict.judge):
one that fails is left out, and says why, whether or not it was deleted
first. The rest is decided here rather than asked: the core settings' rate,
and backgrounds randomised once the takes have keep labels. Then the page
watches: which episode it is on, what the last one said, and the result.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
from pathlib import Path

from pinkrab_core.config import SETTINGS as CORE
from pinkrab_core.dataset import check_fps
from pinkrab_umi.config import SETTINGS as UMI
from pinkrab_umi.episode import list_episodes

from .alerts import ALERTS

DATASET_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _bytes(path: Path) -> int:
    """What a dataset weighs on disk. One being converted writes each
    episode's video into a temporary folder and removes the folder once the
    video has joined the dataset, so a file or folder listed may be gone by
    the time it is asked its size: it counts as nothing."""
    total = 0
    for folder, _, files in os.walk(path):   # a folder gone mid-walk is skipped
        for name in files:
            try:
                total += os.stat(os.path.join(folder, name)).st_size
            except FileNotFoundError:
                pass
    return total


def worth_training(names: list[str], say) -> list[str]:
    """The takes of `names` that pass the shelf's tests, each judged as the
    shelf judges it (verdict.judge), its replay solved first if the shelf has
    not yet; each left out is said, with why."""
    from . import replay, verdict

    kept = []
    for name in names:
        root = UMI.episodes_dir / name
        try:
            fails = verdict.judge(root, replay.solved(root))["fails"]
        except Exception as exc:  # noqa: BLE001 - a take that cannot be judged is not trained on
            fails = [f"could not be judged: {type(exc).__name__}: {exc}"]
        if fails:
            say(f"  {name}: left out - {'; '.join(fails)}")
        else:
            kept.append(name)
    return kept


class Converter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._lines: list[str] = []
        self._asked: dict | None = None
        self._result: dict | None = None
        self._error: str | None = None
        self._started = 0.0
        # How many of the lines said are takes left out, not converted.
        self._left_out = 0

    @staticmethod
    def datasets_dir() -> Path:
        return CORE.workspace / "datasets"

    def start(self, payload: dict) -> None:
        out = str(payload.get("out") or "").strip()
        task = str(payload.get("task") or "").strip()
        fps = check_fps(CORE.dataset_fps)
        # Every complete take, read now, in the order they were recorded: one
        # still being written has no meta yet and is left for the next dataset.
        episodes = sorted(e["name"] for e in list_episodes() if e.get("complete"))
        if not episodes:
            raise ValueError("no complete episode on the shelf to convert")
        if not DATASET_NAME.match(out):
            raise ValueError("the dataset needs a name: letters, digits, _.-")
        if not task:
            raise ValueError("say the task in words - it goes into the dataset")
        if (self.datasets_dir() / out).exists():
            raise ValueError(f"a dataset named {out!r} exists - pick another name or delete it")
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise ValueError("a conversion is already running")
            self._asked = {"episodes": episodes, "out": out, "task": task, "fps": fps}
            self._lines, self._result, self._error, self._left_out = [], None, None, 0
            self._started = time.monotonic()
            self._thread = threading.Thread(target=self._run, name="convert", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        from pinkrab_umi import convert
        asked = self._asked or {}

        def say(line: str) -> None:
            with self._lock:
                self._lines.append(line.strip())
        kept = worth_training(asked["episodes"], say)
        if not kept:
            with self._lock:
                self._error = "no take on the shelf passes its tests"
            return
        # Their backgrounds randomised once any take has keep labels - a
        # take without them is then left out - and as recorded while none has.
        from pinkrab_umi import keep_masks
        randomised = any(keep_masks.account(UMI.episodes_dir / name) is not None for name in kept)
        with self._lock:
            asked["episodes"] = kept
            asked["randomised"] = randomised
            self._left_out = len(self._lines)
        try:
            result = convert.convert(asked["episodes"], asked["out"], asked["task"], say=say,
                                     fps=asked["fps"], randomised=asked["randomised"])
        except Exception as exc:  # noqa: BLE001 - the page shows the reason
            with self._lock:
                self._error = str(exc)
                said = list(self._lines[-5:])
            ALERTS.say("convert", f"{asked['out']} failed: {exc}", said)
            # Half a dataset is worse than none: it would list as one.
            shutil.rmtree(self.datasets_dir() / asked["out"], ignore_errors=True)
            return
        with self._lock:
            self._result = result

    def writing(self, name: str) -> bool:
        """Whether the running conversion is making the dataset `name`."""
        with self._lock:
            running = self._thread is not None and self._thread.is_alive()
            return running and (self._asked or {}).get("out") == name

    def state(self) -> dict:
        with self._lock:
            running = self._thread is not None and self._thread.is_alive()
            done = len(self._lines) - self._left_out
            total = len((self._asked or {}).get("episodes") or [])
            return {
                "running": running, "asked": self._asked, "done": done, "total": total,
                "lines": self._lines[-6:], "result": self._result, "error": self._error,
                "seconds": round(time.monotonic() - self._started, 1) if self._asked else None,
            }

    @staticmethod
    def list_datasets() -> list[dict]:
        root = Converter.datasets_dir()
        if not root.is_dir():
            return []
        out = []
        for path in sorted(root.iterdir(), reverse=True):
            if not path.is_dir():
                continue
            entry = {"name": path.name, "bytes": _bytes(path)}
            info = path / "meta" / "info.json"
            ours = path / "meta" / "pinkrab_convert.json"
            try:
                if info.is_file():
                    meta = json.loads(info.read_text())
                    entry["episodes"] = meta.get("total_episodes")
                    entry["frames"] = meta.get("total_frames")
                    entry["fps"] = meta.get("fps")
                if ours.is_file():
                    mine = json.loads(ours.read_text())
                    entry["task"] = mine.get("task")
                    entry["sources"] = [e["episode"] for e in mine.get("episodes", [])]
                    entry["randomised"] = bool(mine.get("randomised"))
            except (OSError, ValueError):
                pass
            out.append(entry)
        return out

    @staticmethod
    def episodes(name: str) -> dict:
        """A dataset's episodes as the page plays them: each one's length,
        the take it came from, and where in which of the dataset's videos
        each camera's frames are - LeRobot keeps many episodes end to end in
        one file, each between its own two timestamps."""
        import pyarrow.parquet as pq

        if not DATASET_NAME.match(name):
            raise ValueError(f"{name!r} is not a dataset name")
        root = Converter.datasets_dir() / name
        info = json.loads((root / "meta" / "info.json").read_text())
        ours = root / "meta" / "pinkrab_convert.json"
        mine = json.loads(ours.read_text()) if ours.is_file() else {}
        # The takes in the order their segments became episodes.
        sources = [e["episode"] for e in mine.get("episodes", []) for _ in e.get("segments", [])]
        cameras = [key for key, f in info["features"].items() if f.get("dtype") == "video"]
        rows = []
        for path in sorted((root / "meta" / "episodes").glob("*/*.parquet")):
            rows += pq.read_table(path).to_pylist()
        rows.sort(key=lambda r: r["episode_index"])
        fps = info["fps"]
        episodes = []
        for row in rows:
            i = row["episode_index"]
            videos = {}
            for key in cameras:
                chunk, file = row[f"videos/{key}/chunk_index"], row[f"videos/{key}/file_index"]
                videos[key] = {"path": info["video_path"].format(video_key=key, chunk_index=chunk, file_index=file),
                               "from": row[f"videos/{key}/from_timestamp"], "to": row[f"videos/{key}/to_timestamp"]}
            episodes.append({"index": i, "length": row["length"], "seconds": round(row["length"] / fps, 2),
                             "source": sources[i] if i < len(sources) else None, "videos": videos})
        return {"name": name, "fps": fps, "randomised": bool(mine.get("randomised")), "task": mine.get("task"),
                "cameras": cameras, "episodes": episodes}

    @staticmethod
    def video(name: str, relative: str) -> Path:
        """One of a dataset's video files, by its path inside the dataset."""
        if not DATASET_NAME.match(name):
            raise ValueError(f"{name!r} is not a dataset name")
        videos = (Converter.datasets_dir() / name / "videos").resolve()
        target = (Converter.datasets_dir() / name / relative).resolve()
        if videos not in target.parents or target.suffix != ".mp4" or not target.is_file():
            raise ValueError("no such video")
        return target

    @staticmethod
    def delete_dataset(name: str) -> None:
        if not DATASET_NAME.match(name):
            raise ValueError(f"{name!r} is not a dataset name")
        path = Converter.datasets_dir() / name
        if not path.is_dir():
            raise ValueError(f"no dataset {name!r}")
        shutil.rmtree(path)
