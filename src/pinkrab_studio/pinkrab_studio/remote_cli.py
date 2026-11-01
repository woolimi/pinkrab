"""`pinkrab-remote` - the training page's sources and runs, from a terminal.

The same store (~/pinkrab/train.sqlite) and the same code as the page, done
now and said line by line: a source added here is on the page, a run started
here is drawn there.

    pinkrab-remote add gpu-box --line "ssh -i ~/.ssh/id_ed25519 -p 22 user@gpu.example.com" \\
                   --root "~/pinkrab-train"
    pinkrab-remote probe gpu-box
    pinkrab-remote launch gpu-box blocks-left-v2-g2 --policy act --steps 100000
    pinkrab-remote runs
    pinkrab-remote fetch 7 --all
"""
from __future__ import annotations

import argparse
import sys
import time

from .launching import DEFAULTS, RECIPES, Jobs, LaunchError, probe, save_key
from .training import TrainError, TrainStore


def _steps(n) -> str:
    return "?" if n is None else f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pinkrab-remote", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("sources", help="where runs are read from")
    p = sub.add_parser("add", help="add or update an ssh source")
    p.add_argument("name")
    p.add_argument("--line", help="the ssh, sftp or scp line a console hands out")
    p.add_argument("--host")
    p.add_argument("--port", type=int)
    p.add_argument("--user")
    p.add_argument("--key", help="the private key's path on this machine")
    p.add_argument("--root", required=True,
                   help="where runs live on that machine, as it names it - quoted, so this shell does not expand a ~")
    p = sub.add_parser("key", help="keep a private key for a source in ~/pinkrab/keys")
    p.add_argument("name")
    p.add_argument("file")
    p = sub.add_parser("probe", help="GPUs, cores, room, whether it is set up, what is running")
    p.add_argument("name")
    p = sub.add_parser("sync", help="read a source's runs now")
    p.add_argument("name", nargs="?")
    p = sub.add_parser("launch", help="start a run")
    p.add_argument("source")
    p.add_argument("dataset")
    p.add_argument("--policy", choices=sorted(RECIPES), default="act")
    p.add_argument("--steps", type=int, default=DEFAULTS["steps"])
    p.add_argument("--batch", type=int)
    p.add_argument("--log-freq", type=int, default=DEFAULTS["log_freq"])
    p.add_argument("--save-freq", type=int, default=DEFAULTS["save_freq"])
    p.add_argument("--gpu", default="0")
    p.add_argument("--name")
    p.add_argument("--no-fetch", action="store_true", help="leave the checkpoints where they are made")
    p.add_argument("--extra", default="", help="more lerobot-train arguments, as one string")
    sub.add_parser("runs", help="every run read so far")
    for name, what in (("stop", "stop a run"), ("resume", "resume a run from its last checkpoint")):
        sub.add_parser(name, help=what).add_argument("run", type=int)
    p = sub.add_parser("fetch", help="copy a run's checkpoints here")
    p.add_argument("run", type=int)
    p.add_argument("--step", type=int)
    p.add_argument("--all", action="store_true")
    args = parser.parse_args(argv)

    store = TrainStore()
    jobs = Jobs(store)
    try:
        if args.command == "sources":
            for s in store.sources():
                where = s["root"] if s["kind"] == "local" else f"{s['user']}@{s['host']}:{s['port']} {s['root']}"
                print(f"  {s['name']:<16} {where}  {s['runs']} runs" + (f"  ! {s['error']}" if s["error"] else ""))
        elif args.command == "add":
            s = store.save_source({"name": args.name, "line": args.line, "host": args.host, "port": args.port,
                                   "user": args.user, "key": args.key, "root": args.root})
            print(f"  {s['name']}: {s['user']}@{s['host']}:{s['port']} {s['root']}")
        elif args.command == "key":
            with open(args.file) as f:
                print(f"  kept as {save_key(store, args.name, f.read())}")
        elif args.command == "probe":
            p = probe(store, args.name)
            for g in p.get("gpus") or []:
                print(f"  GPU {g['index']}: {g['name']}, {g['mem_used_mb']}/{g['mem_total_mb']} MB used, {g['util']}% busy,"
                      f" driver {g.get('driver', '?')}, compute capability {g.get('compute', '?')}")
            for line in p.get("short") or []:
                print(f"  cannot train here: {line}")
            print(f"  {p['host']}: {p['cores']} cores, {p['free_gb']} GB free under {p['root']}")
            print(f"  set up: {'yes' if p['venv'] and p['lock_matches'] else 'no - the first launch sets it up'}")
            for r in p.get("running") or []:
                print(f"  running: {r}")
        elif args.command == "sync":
            for s in [store.source(args.name)] if args.name else store.sources():
                store.sync(s["name"])
                print(f"  read {s['name']}")
        elif args.command == "launch":
            done = jobs.launch_now({"source": args.source, "dataset": args.dataset, "policy": args.policy,
                                    "steps": args.steps, "batch_size": args.batch, "log_freq": args.log_freq,
                                    "save_freq": args.save_freq, "gpu": args.gpu, "name": args.name,
                                    "auto_fetch": not args.no_fetch, "extra": args.extra},
                                   say=lambda line: print(f"  {line}", flush=True))
            time.sleep(2.0)
            store.sync(args.source)
            print(f"  {done['run']}: follow it on the studio's training page, or `pinkrab-remote runs`")
        elif args.command == "runs":
            for r in store.runs():
                eta = f", {r['eta_s'] / 3600:.1f} h left" if r.get("eta_s") else ""
                loss = f", loss {r['loss']:.4g}" if r.get("loss") is not None else ""
                print(f"  {r['id']:>3} {r['status']:<8} {r['name']:<40} {_steps(r['last_step'])}/{_steps(r['steps'])}"
                      f"{loss}{eta}  [{r['source']}, {r['checkpoints']} ckpt, {r['fetched']} here]")
        elif args.command == "stop":
            jobs.stop_now(args.run, say=lambda line: print(f"  {line}"))
        elif args.command == "resume":
            jobs.resume_now(args.run, say=lambda line: print(f"  {line}", flush=True))
        elif args.command == "fetch":
            run = store.run(args.run)
            wanted = [c["step"] for c in run["checkpoints"] if args.all or c["step"] == args.step] if (args.all or args.step) \
                else [c["step"] for c in run["checkpoints"][-1:]]
            if not wanted:
                raise LaunchError("no such checkpoint - `pinkrab-remote runs` says how many there are")
            for step in wanted:
                done = jobs.fetch_now(args.run, step, say=lambda line: print(f"  {line}", flush=True))
                print(f"  step {step}: {done['local']}")
    except (LaunchError, TrainError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
