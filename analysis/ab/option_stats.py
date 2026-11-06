"""Surface statistics of the A/B option texts (the "Option balance" paragraph
of the A/B appendix: lengths, punctuation, descriptor phrases). No API calls.

For every item in the cross-domain A/B bank, splits the prompt into its A) /
B) option texts, maps them to right-coded / left-coded via ``right_answer``,
and tests whether the two sides are matched on surface features. It also
reports the position coding of the bank (the right-coded option is listed
first in every item, which is why analysis/ab/swap.py averages over both
orders).

Significance: paired sign-flip permutation test on the per-item left - right
difference (10,000 iterations), plus a bootstrap CI on the mean (2,000
resamples); one generator, seed 0.

Outputs (under $OUTPUTS/ab/option_stats/): option_stats.csv (per item),
report.md.

Usage:
    uv run python analysis/ab/option_stats.py
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict

import numpy as np

from bank import CROSS_DOMAIN_BANK, split_options
from ideogen.paths import REPO_ROOT, output_path

PERM_ITERS = 10_000
SEED = 0
OUTLIER_TOP = 10

METRICS = [
    ("words", "word count"),
    ("chars", "character count"),
    ("commas", "commas"),
    ("dashes", "dashes (- -- —)"),
    ("exclaims", "exclamation marks"),
    ("descriptors", "comma-separated descriptor phrases"),
]


def text_metrics(text: str) -> dict[str, float]:
    return {
        "words": len(text.split()),
        "chars": len(text),
        "commas": text.count(","),
        "dashes": text.count("—") + text.count(" - ") + text.count("--"),
        "exclaims": text.count("!"),
        # "Thing — adj, adj, adj" style: phrases after the first dash/comma
        "descriptors": max(0, len(re.split(r"[,—]", text)) - 1),
    }


def perm_test(diffs: np.ndarray, rng: np.random.Generator) -> float:
    """Two-sided sign-flip permutation p-value for mean(diffs) != 0."""
    obs = abs(diffs.mean())
    signs = rng.choice([-1.0, 1.0], size=(PERM_ITERS, len(diffs)))
    null = np.abs((signs * diffs).mean(axis=1))
    return float((np.sum(null >= obs) + 1) / (PERM_ITERS + 1))


def boot_ci(diffs: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    idx = rng.integers(0, len(diffs), size=(2000, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    rng = np.random.default_rng(SEED)

    items, unparsed, coding = [], [], defaultdict(int)
    with open(CROSS_DOMAIN_BANK) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            coding[d.get("right_answer", "?")] += 1
            parsed = split_options(d["prompt"])
            if parsed is None:
                unparsed.append(i)
                continue
            stem, opt_a, opt_b = parsed
            ra = d.get("right_answer", "A")
            right_text, left_text = (opt_a, opt_b) if ra == "A" else (opt_b, opt_a)
            items.append({
                "idx": i, "prompt": d["prompt"], "stem": stem,
                "category": d.get("category", "unknown"),
                "right_text": right_text, "left_text": left_text,
                "right": text_metrics(right_text), "left": text_metrics(left_text),
            })

    # ---- per-item CSV ------------------------------------------------------
    with open(output_path("ab", "option_stats", "option_stats.csv"), "w", newline="") as f:
        w = csv.writer(f)
        header = ["idx", "category"]
        for key, _ in METRICS:
            header += [f"right_{key}", f"left_{key}", f"diff_{key}"]
        w.writerow(header + ["right_text", "left_text"])
        for it in items:
            row = [it["idx"], it["category"]]
            for key, _ in METRICS:
                row += [it["right"][key], it["left"][key],
                        it["left"][key] - it["right"][key]]
            w.writerow(row + [it["right_text"], it["left_text"]])

    # ---- report ------------------------------------------------------------
    rep = ["# A/B option surface-statistics report", "",
           f"Bank: `{CROSS_DOMAIN_BANK.relative_to(REPO_ROOT)}` — {len(items)} parsed items"
           + (f", {len(unparsed)} unparsed (indices {unparsed})" if unparsed else "") + ".",
           "",
           "## Position coding",
           "",
           f"`right_answer` distribution: {dict(sorted(coding.items()))}. "
           "In this bank the right-coded option is listed as \"A\" in every "
           "item; option-position effects are therefore handled by the "
           "position-swap rerun (analysis/ab/swap.py), and this "
           "report addresses surface-style balance only.",
           "",
           "## Per-side means and paired differences (left − right)",
           "",
           "| Metric | right mean | left mean | mean diff [95% CI] | perm. p |",
           "|---|---|---|---|---|"]
    for key, label in METRICS:
        r = np.array([it["right"][key] for it in items], dtype=float)
        l = np.array([it["left"][key] for it in items], dtype=float)
        diffs = l - r
        lo, hi = boot_ci(diffs, rng)
        p = perm_test(diffs, rng)
        rep.append(f"| {label} | {r.mean():.2f} | {l.mean():.2f} "
                   f"| {diffs.mean():+.2f} [{lo:+.2f}, {hi:+.2f}] | {p:.3f} |")

    rep += ["", "## Word-count diff by category", "",
            "| Category | n | mean diff (left − right) |", "|---|---|---|"]
    by_cat = defaultdict(list)
    for it in items:
        by_cat[it["category"]].append(it["left"]["words"] - it["right"]["words"])
    for cat in sorted(by_cat):
        vals = by_cat[cat]
        rep.append(f"| {cat} | {len(vals)} | {np.mean(vals):+.2f} |")

    word_sorted = sorted(items, key=lambda it: -abs(it["left"]["words"] - it["right"]["words"]))
    rep += ["", f"## {OUTLIER_TOP} worst length-matched pairs", ""]
    for it in word_sorted[:OUTLIER_TOP]:
        d = it["left"]["words"] - it["right"]["words"]
        rep += [f"- **[{it['idx']}] {it['category']}** (left − right = {d:+d} words)",
                f"  - right: {it['right_text']}",
                f"  - left: {it['left_text']}"]

    rep += ["", "Per-item data: `option_stats.csv`."]
    out = output_path("ab", "option_stats", "report.md")
    out.write_text("\n".join(rep) + "\n")
    print(f"Wrote {out.parent}/option_stats.csv and report.md "
          f"({len(items)} items, coding {dict(sorted(coding.items()))})")


if __name__ == "__main__":
    main()
