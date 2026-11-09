"""Offline check of whether a labels job teaches a masker or has SAM 3 track
its takes: a job over masker_from_takes takes or more teaches one where the
masker's environment is there; over fewer, or without it, SAM 3 tracks them
all. No SAM 3, no RF-DETR.

    python src/pinkrab_studio/test/masker_plan_check.py
"""
from __future__ import annotations

import sys

from pinkrab_studio.masking import Masker
from pinkrab_umi.config import SETTINGS as UMI

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}" + (f" - {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


def main() -> int:
    n = UMI.masker_from_takes
    takes = [f"t{i:02d}" for i in range(40)]
    check("many takes: a masker is taught", Masker._teaches(takes, True))
    check("as many takes as a masker is taught from: one is taught", Masker._teaches(takes[:n], True))
    check("fewer: SAM 3 tracks them all", not Masker._teaches(takes[:n - 1], True))
    check("no masker environment: SAM 3 tracks them all", not Masker._teaches(takes, False))
    print(f"\n{'all passed' if not FAILURES else f'{len(FAILURES)} failed: ' + ', '.join(FAILURES)}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
