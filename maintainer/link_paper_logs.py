"""Symlink the original paper eval logs into this repo's results layout.

Maintainer tool. The paper's logs live in the authors' development repo
under an older, messier layout. Each ``paper_logs/*.tsv`` file maps a path
in the new layout (relative to the target root) to the original file
(relative to ``--src``). Running this builds a ``results_paper/`` tree of
symlinks, so every analysis script can be pointed at the exact logs behind
the paper:

    uv run python maintainer/link_paper_logs.py --src /path/to/original-results
    RESULTS=results_paper uv run python analysis/<script>.py

The same mapping is used to build the release tarball of logs
(``--copy --dest <dir>``).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def read_maps(map_dir: Path) -> list[tuple[str, str, str]]:
    rows = []
    for tsv in sorted(map_dir.glob("*.tsv")):
        for n, line in enumerate(tsv.read_text().splitlines(), 1):
            line = line.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                sys.exit(f"{tsv.name}:{n}: expected 2 tab-separated columns, got {parts!r}")
            rows.append((tsv.name, parts[0].strip(), parts[1].strip()))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="path to the original results root")
    ap.add_argument("--dest", type=Path, default=REPO / "results_paper", help="target root (default: results_paper/)")
    ap.add_argument("--copy", action="store_true", help="copy files instead of symlinking (for a release tarball)")
    ap.add_argument("--check", action="store_true", help="only report missing sources and duplicate targets")
    args = ap.parse_args()

    src = args.src.resolve()
    rows = read_maps(HERE / "paper_logs")
    seen: dict[str, str] = {}
    missing, dupes, made = [], [], 0
    for tsv, new, old in rows:
        if new in seen:
            dupes.append(f"{new} ({seen[new]} and {tsv})")
            continue
        seen[new] = tsv
        s = src / old
        if not s.exists():
            missing.append(f"{tsv}: {old}")
            continue
        if args.check:
            continue
        d = args.dest / new
        d.parent.mkdir(parents=True, exist_ok=True)
        if d.is_symlink() or d.exists():
            if d.is_dir() and not d.is_symlink():
                shutil.rmtree(d)
            else:
                d.unlink()
        if args.copy:
            (shutil.copytree if s.is_dir() else shutil.copy2)(s, d)
        else:
            d.symlink_to(s)
        made += 1

    print(f"{len(rows)} mappings, {made} {'copied' if args.copy else 'linked'} into {args.dest}")
    for m in missing:
        print(f"  MISSING  {m}")
    for d in dupes:
        print(f"  DUPLICATE target {d}")
    if missing or dupes:
        sys.exit(1)


if __name__ == "__main__":
    main()
