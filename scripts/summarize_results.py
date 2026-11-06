#!/usr/bin/env python
"""Print per-(eval_dir, label, scorer) summary scores for .eval files.

Walks any directory tree containing inspect_ai .eval files and prints, per
result directory, one row per (label, scorer) combination with mean, std,
sample count, and a Δ vs the baseline label if one exists. Useful for
sanity-checking that fine-tuned models shifted away from baseline after a
partial or smoke-test run of experiments/*.sh.

This is a quick-look tool, not a paper-cited statistic. For bootstrap CIs
matching the paper's tables, run analysis/run_all.sh on complete runs.

Usage
-----
    uv run python scripts/summarize_results.py                       # walks ./results
    uv run python scripts/summarize_results.py results/econ          # one subtree
    uv run python scripts/summarize_results.py --no-recursive results/econ/economy_control
"""
from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from pathlib import Path

from inspect_ai.log import read_eval_log


def per_label_scorer_values(eval_files: list[Path]) -> dict[str, dict[str, list[float]]]:
    """{label: {scorer: [per-sample-mean]}} for a directory's .eval files."""
    out: dict[str, dict[str, list[float]]] = {}
    for path in eval_files:
        label = path.stem
        log = read_eval_log(str(path))
        scorers: dict[str, list[float]] = defaultdict(list)
        for s in log.samples or []:
            if not s.scores:
                continue
            for scorer_name, score in s.scores.items():
                v = score.value
                if isinstance(v, (int, float)):
                    scorers[scorer_name].append(float(v))
        out[label] = dict(scorers)
    return out


def format_row(label: str, vals: list[float], baseline_mean: float | None) -> str:
    n = len(vals)
    if n == 0:
        return f"  {label:<26s} (no numeric scores)"
    mean = statistics.mean(vals)
    std = statistics.pstdev(vals) if n > 1 else 0.0
    if label == "baseline":
        suffix = "(reference)"
    elif baseline_mean is None:
        suffix = ""
    else:
        suffix = f"Δ={mean - baseline_mean:+.3f}"
    return f"  {label:<26s} {mean:6.3f} ± {std:5.3f}  (n={n:>3d})  {suffix}"


def summarize_dir(d: Path) -> None:
    eval_files = sorted(d.glob("*.eval"))
    if not eval_files:
        return
    per_label = per_label_scorer_values(eval_files)
    scorers = sorted({s for sc in per_label.values() for s in sc})
    if not scorers:
        return
    for scorer in scorers:
        present = {l: sc[scorer] for l, sc in per_label.items()
                   if scorer in sc and sc[scorer]}
        if not present:
            continue
        baseline_vals = present.get("baseline")
        baseline_mean = statistics.mean(baseline_vals) if baseline_vals else None
        n_first = len(next(iter(present.values())))
        print(f"\n== {d}  [{scorer}, n={n_first}] ==")
        for label in sorted(present, key=lambda l: (l != "baseline", l)):
            print(format_row(label, present[label], baseline_mean))


def summarize(root: Path, recursive: bool = True) -> None:
    if not root.exists():
        print(f"ERROR: {root} does not exist")
        return
    if recursive:
        dirs = sorted({p.parent for p in root.rglob("*.eval")})
    else:
        dirs = [root] if list(root.glob("*.eval")) else []
    if not dirs:
        print(f"(no .eval files under {root})")
        return
    for d in dirs:
        summarize_dir(d)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", nargs="?", default="results", type=Path,
                        help="root directory to walk (default: results)")
    parser.add_argument("--no-recursive", action="store_true",
                        help="treat path as a single directory; don't descend")
    args = parser.parse_args()
    summarize(args.path, recursive=not args.no_recursive)


if __name__ == "__main__":
    main()
