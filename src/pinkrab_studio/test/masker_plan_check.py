"""Offline check of how a labels job splits its takes between SAM 3 and the
task's masker: all to a masker that exists; over few takes, all to SAM 3;
over many, SAM 3 tracking just enough of them - spread over the job - to
teach one, and the masker the rest; all to SAM 3 where no masker can be
taught. No SAM 3, no RF-DETR.

    python src/pinkrab_studio/test/masker_plan_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

from pinkrab_studio.masking import Masker
from pinkrab_umi.config import SETTINGS as UMI

FAILURES: list[str] = []
THINGS = ["mat", "cube"]


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}" + (f" - {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


def plan(takes: int, tracked: int, record: dict | None = None, can_teach: bool = True) -> tuple[list, list]:
    names = [f"t{i:02d}" for i in range(takes)]
    with mock.patch.object(Masker, "_tracked", return_value=[Path(f"s{i}") for i in range(tracked)]):
        return Masker._plan(names, THINGS, record, can_teach)


def main() -> int:
    n = UMI.masker_from_takes
    sam, rest = plan(40, 0, {"id": "m"})
    check("a masker there labels every take", not sam and len(rest) == 40)
    sam, rest = plan(n - 3, 0)
    check("fewer takes than a masker is taught from: SAM 3 tracks them all", len(sam) == n - 3 and not rest)
    sam, rest = plan(40, 0)
    check("many takes: SAM 3 tracks as many as teach a masker", len(sam) == n and len(rest) == 40 - n,
          f"{len(sam)} + {len(rest)}")
    check("spread over the job, first to last", sam[0] == "t00" and sam[-1] == "t39", str(sam))
    check("every take once", sorted(sam + rest) == [f"t{i:02d}" for i in range(40)])
    sam, rest = plan(40, n - 4)
    check("takes SAM 3 tracked before count towards the masker", len(sam) == 4 and len(rest) == 36,
          f"{len(sam)} + {len(rest)}")
    sam, rest = plan(40, n + 5)
    check("enough tracked before: the masker labels every take", not sam and len(rest) == 40)
    sam, rest = plan(40, 0, can_teach=False)
    check("no masker environment: SAM 3 tracks them all", len(sam) == 40 and not rest)
    print(f"\n{'all passed' if not FAILURES else f'{len(FAILURES)} failed: ' + ', '.join(FAILURES)}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
