"""Run a command in a terminal of its own and keep what it printed for the docs.

    python docs/scripts/transcript.py <name> <command> [args...]

The command runs under script(1), so it sees a terminal: it prints its colours
and asks its questions as it would for anyone, and the answers typed go into
the transcript too. What it printed lands in docs/src/transcripts/<name>.txt,
the colours kept, the terminal's carriage returns, script's own control bytes
and the start and end lines util-linux script writes taken out, and the home
directory written as ~. A page shows it with the Output component.
"""
from __future__ import annotations

import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "src" / "transcripts"


def clean(raw: str) -> str:
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    # Control bytes other than ESC (colours), newline and tab: script's ^D and
    # backspaces at the start and end.
    text = re.sub(r"[\x00-\x08\x0b-\x1a\x1c-\x1f\x7f]", "", text)
    # ...and the "^D" it echoes for the end of input it sent.
    text = re.sub(r"^\^D", "", text)
    # util-linux script writes these into the log even with -q.
    text = re.sub(r"\AScript started on .*\n", "", text)
    text = re.sub(r"\n*Script done on .*\n*\Z", "\n", text)
    text = text.replace(str(Path.home()), "~")
    return text.strip("\n") + "\n"


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    name, command = sys.argv[1], sys.argv[2:]
    with tempfile.NamedTemporaryFile(suffix=".raw") as log:
        flags = ["-q", log.name] if sys.platform == "darwin" else ["-q", "-e", "-c", shlex.join(command), log.name]
        code = subprocess.call(["script", *flags, *(command if sys.platform == "darwin" else [])])
        raw = Path(log.name).read_text(errors="replace")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.txt"
    path.write_text(clean(raw))
    print(f"\n  kept: {path}  (exit {code})")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
