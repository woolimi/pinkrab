"""Offline check of what the studio refuses and how it lets go: ssh values that
ssh would read as options, requests from another site or under another name,
a long command that prints nothing, a checkpoint copy that keeps failing, and
a dataset that is in use. No robot, no server socket.

    python src/pinkrab_studio/test/security_check.py
"""
from __future__ import annotations

import tempfile
import time
from email.message import Message
from pathlib import Path

from pinkrab_studio.launching import FETCH_RETRY_S, Jobs, LaunchError, _stream
from pinkrab_studio.server import LOOPBACK_HOSTS, refusal
from pinkrab_studio.training import TrainError, TrainStore, parse_ssh_line, ssh_base

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}" + (f" - {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


def raises(fn, *args, **kwargs) -> bool:
    try:
        fn(*args, **kwargs)
    except TrainError:
        return True
    return False


def headers(**fields: str) -> Message:
    message = Message()
    for key, value in fields.items():
        message[key.replace("_", "-")] = value
    return message


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        store = TrainStore(Path(tmp) / "train.sqlite")

        print("ssh values")
        base = {"name": "box", "host": "10.0.0.5", "root": "/data"}
        for label, bad in (("user -o", {"user": "-oProxyCommand=touch /tmp/x"}),
                           ("host -o", {"host": "-oProxyCommand=x"}),
                           ("user space", {"user": "a b"}),
                           ("host newline", {"host": "h\nx"}),
                           ("key -o", {"key": "-oProxyCommand=x"})):
            check(f"save_source refuses {label}", raises(store.save_source, {**base, **bad}))
        check("save_source refuses a pasted line with -oProxyCommand as user",
              raises(store.save_source, {"name": "p", "root": "/d", "line": "ssh -oProxyCommand=x@host"}))
        check("parse_ssh_line refuses a user that starts with '-'",
              raises(parse_ssh_line, "ssh -oProxyCommand=x@h"))
        check("a plain source is kept", store.save_source({**base, "user": "ubuntu"})["host"] == "10.0.0.5")
        command = ssh_base(store.source("box"))
        check("`--` comes right before the destination", command[-2:] == ["--", "ubuntu@10.0.0.5"], str(command[-3:]))
        check("ssh drops a silent peer", any(c.startswith("ServerAliveInterval=") for c in command))
        check("ssh_base refuses a stored row that starts with '-'",
              raises(ssh_base, {"user": "-oX", "host": "h", "port": 22, "key": None}))

        print("requests")
        own = {"Host": "localhost:8730", "Origin": "http://localhost:8730", "Content_Type": "application/json"}
        check("the page's own POST is served", refusal(headers(**own), LOOPBACK_HOSTS, True) is None)
        check("a cross-origin text/plain POST is refused",
              refusal(headers(**{**own, "Origin": "https://evil.example", "Content_Type": "text/plain"}),
                      LOOPBACK_HOSTS, True) is not None)
        check("a cross-origin JSON POST is refused",
              refusal(headers(**{**own, "Origin": "https://evil.example"}), LOOPBACK_HOSTS, True) is not None)
        check("a same-origin text/plain POST is refused",
              refusal(headers(**{**own, "Content_Type": "text/plain"}), LOOPBACK_HOSTS, True) is not None)
        check("a rebound name is refused",
              refusal(headers(Host="evil.example:8730", Origin="http://evil.example:8730"), LOOPBACK_HOSTS, False)
              is not None)
        check("a GET without Origin from localhost is served",
              refusal(headers(Host="127.0.0.1:8730"), LOOPBACK_HOSTS, False) is None)
        check("bound to the network, any Host is served",
              refusal(headers(Host="studio.lan:8730"), None, False) is None)
        check("a cross-origin GET is refused",
              refusal(headers(Host="localhost:8730", Origin="https://evil.example"), LOOPBACK_HOSTS, False) is not None)

        print("a command that prints nothing")
        started = time.monotonic()
        try:
            _stream({"name": "here", "kind": "local"}, "sleep 30", lambda line: None, timeout=1.0)
            stopped = False
        except LaunchError:
            stopped = True
        check("is stopped at its time limit", stopped and time.monotonic() - started < 10,
              f"{time.monotonic() - started:.1f} s")

        print("checkpoint copies and datasets")
        jobs = Jobs(store)
        asked: list[tuple[int, int]] = []
        jobs.fetch = lambda run_id, step: asked.append((run_id, step))
        store.to_fetch = lambda: [{"run_id": 1, "step": 5, "path": "/x"}]
        jobs.fetch_due()
        check("a checkpoint to copy is asked for", asked == [(1, 5)])

        def broken(run, step, say):
            raise LaunchError("no space left")
        jobs._copy_home = broken
        del jobs.fetch
        try:
            jobs._fetch({"id": 1, "name": "r"}, 5, print)
        except LaunchError:
            pass
        asked.clear()
        jobs.fetch = lambda run_id, step: asked.append((run_id, step))
        jobs.fetch_due()
        check("a copy that failed is not asked for again at once", asked == [])
        jobs._fetch_retry[(1, 5)] = time.monotonic() - 1
        jobs.fetch_due()
        check("it is after the wait", asked == [(1, 5)], str(FETCH_RETRY_S))
        check("state hands out a copy of the probes", jobs.state()["probes"] is not jobs.probes)
        jobs._hold("ds")
        check("a dataset a launch holds is reported", jobs.holds("ds") and not jobs.holds("other"))

    print()
    if FAILURES:
        print(f"{len(FAILURES)} failed")
        return 1
    print("all passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
