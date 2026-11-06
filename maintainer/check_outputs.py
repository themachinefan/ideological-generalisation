"""Compare analysis outputs against the reference outputs behind the paper.

`reference_outputs/` holds every table (.tex), CSV and JSON written by
`analysis/run_all.sh` from the paper's original eval logs. After rebuilding
the outputs from the same logs, this reports any file that differs:

    RESULTS=results_paper OUTPUTS=outputs_paper bash analysis/run_all.sh
    uv run python maintainer/check_outputs.py outputs_paper

Figures (PDF) are not compared: their bytes depend on the matplotlib version
and timestamps; the numbers they plot are covered by the tables.
"""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

REF = Path(__file__).resolve().parent / "reference_outputs"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outputs", type=Path, help="outputs directory to check (e.g. outputs_paper)")
    ap.add_argument("--diff", action="store_true", help="print a unified diff for each mismatch")
    args = ap.parse_args()

    ref_files = sorted(p.relative_to(REF) for p in REF.rglob("*") if p.is_file())
    same, differ, missing = [], [], []
    for rel in ref_files:
        new = args.outputs / rel
        if not new.exists():
            missing.append(rel)
        elif new.read_bytes() == (REF / rel).read_bytes():
            same.append(rel)
        else:
            differ.append(rel)
            if args.diff:
                sys.stdout.writelines(difflib.unified_diff(
                    (REF / rel).read_text().splitlines(True), new.read_text().splitlines(True),
                    f"reference/{rel}", str(new), n=1))

    print(f"{len(same)}/{len(ref_files)} outputs identical to the reference")
    for rel in differ:
        print(f"  DIFFERS  {rel}")
    for rel in missing:
        print(f"  MISSING  {rel}")
    sys.exit(1 if differ or missing else 0)


if __name__ == "__main__":
    main()
