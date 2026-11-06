"""Are left and right A/B options matched in tone? Table tab:ab-option-balance.

Reads the blind style ratings cached by analysis/ab/judge_options.py
($RESULTS/ab/option_judgments/{manifest,judgments}.jsonl; no API calls) and
compares the left- and right-coded option of every item on extremity,
emotional intensity and assertiveness (0-100, repeats averaged per option and
judge, then judges averaged). Paired left - right differences get a bootstrap
CI (2,000 resamples) and a sign-flip permutation p (10,000 iterations); one
generator, seed 0. report.md and the LaTeX table take successive draws from
it, so their CIs and p-values differ in the last digit (the paper cites the
table).

Pairs whose judge-mean |left - right| exceeds --flag-threshold (default 40,
the paper's sensitivity pass) on any dimension go to flagged_pairs.json, for
``analysis/ab/swap.py --exclude-pairs``.

Outputs (under $OUTPUTS/ab/option_judgments/):
  option_judgments.csv  per (item, side): mean score per dimension per judge
  report.md             per-side means, paired differences, cross-judge
                        agreement on the differences, worst-matched pairs
  flagged_pairs.json    tone-mismatched pairs for the sensitivity pass
  ab_option_balance.tex tab:ab-option-balance

Usage:
    RESULTS=results_paper uv run python analysis/ab/option_judgments.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict

import numpy as np

from ideogen.paths import output_path, results_dir
from judge_options import DIMS

PERM_ITERS = 10_000
SEED = 0
WORST_TOP = 10

DIM_LABELS = {"extremity": "Extremity", "intensity": "Emotional intensity",
              "assertiveness": "Assertiveness"}
NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four"}


def perm_test(diffs: np.ndarray, rng: np.random.Generator) -> float:
    obs = abs(diffs.mean())
    signs = rng.choice([-1.0, 1.0], size=(PERM_ITERS, len(diffs)))
    null = np.abs((signs * diffs).mean(axis=1))
    return float((np.sum(null >= obs) + 1) / (PERM_ITERS + 1))


def boot_ci(vals: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    idx = rng.integers(0, len(vals), size=(2000, len(vals)))
    means = vals[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def signed(x: float) -> str:
    return f"{x:+.1f}".replace("-", "$-$").replace("+", "$+$")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--flag-threshold", type=float, default=40.0,
                    help="Flag pairs with judge-mean |left-right| above this "
                         "on any dimension (0-100 scale)")
    args = ap.parse_args()

    cache_dir = results_dir("ab", "option_judgments")
    manifest_path = cache_dir / "manifest.jsonl"
    judg_path = cache_dir / "judgments.jsonl"
    if not manifest_path.exists() or not judg_path.exists():
        sys.exit(f"Missing {manifest_path} or {judg_path}; run "
                 f"analysis/ab/judge_options.py first.")

    rng = np.random.default_rng(SEED)

    records = {}
    with open(manifest_path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                records[r["record_id"]] = r

    # (record_id, judge) -> {dim: [scores across repeats]}
    scores = defaultdict(lambda: defaultdict(list))
    judges, n_parse_fail = set(), 0
    with open(judg_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if str(rec.get("parse_reason", "")).startswith("ERROR"):
                continue
            if rec.get("scores") is None:
                n_parse_fail += 1
                continue
            judges.add(rec["judge"])
            for dim in DIMS:
                scores[(rec["record_id"], rec["judge"])][dim].append(
                    rec["scores"][dim])
    judges = sorted(judges)
    if not judges:
        sys.exit("No parsed judgments found.")

    # per (item, side): judge -> dim -> mean over repeats
    items = defaultdict(dict)  # item_idx -> side -> {judges: {judge: {dim: mean}}, meta}
    for rid, r in records.items():
        per_judge = {}
        for judge in judges:
            js = scores.get((rid, judge))
            if js and all(js[d] for d in DIMS):
                per_judge[judge] = {d: float(np.mean(js[d])) for d in DIMS}
        if per_judge:
            items[r["item_idx"]][r["side"]] = {"judges": per_judge, "meta": r}

    complete = {i: s for i, s in items.items() if "left" in s and "right" in s}
    print(f"{len(complete)} pairs with both sides judged "
          f"({len(items) - len(complete)} incomplete), judges: {judges}, "
          f"parse failures: {n_parse_fail}")

    def judge_mean(entry: dict, dim: str) -> float:
        return float(np.mean([entry["judges"][j][dim]
                              for j in entry["judges"]]))

    def side_diffs(dim: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r_vals = np.array([judge_mean(s["right"], dim) for s in complete.values()])
        l_vals = np.array([judge_mean(s["left"], dim) for s in complete.values()])
        return r_vals, l_vals, l_vals - r_vals

    # ---- CSV ---------------------------------------------------------------
    with open(output_path("ab", "option_judgments", "option_judgments.csv"),
              "w", newline="") as f:
        w = csv.writer(f)
        header = ["item_idx", "category", "side"]
        for judge in judges:
            header += [f"{judge}:{d}" for d in DIMS]
        w.writerow(header)
        for i in sorted(items):
            for side, entry in sorted(items[i].items()):
                row = [i, entry["meta"]["category"], side]
                for judge in judges:
                    jd = entry["judges"].get(judge, {})
                    row += [round(jd[d], 1) if d in jd else "" for d in DIMS]
                w.writerow(row)

    # ---- report ------------------------------------------------------------
    rep = ["# A/B option tone/extremity balance report", "",
           f"{len(complete)} pairs, judges: {', '.join(judges)}, "
           f"repeats averaged per (option, judge).", "",
           "Diffs are left − right per pair, averaged over judges. A matched",
           "bank shows diffs centred on 0 with high p-values.", "",
           "## Per-side means and paired differences", "",
           "| Dimension | right mean | left mean | mean diff [95% CI] | perm. p |",
           "|---|---|---|---|---|"]
    for dim in DIMS:
        r_vals, l_vals, diffs = side_diffs(dim)
        lo, hi = boot_ci(diffs, rng)
        p = perm_test(diffs, rng)
        rep.append(f"| {dim} | {r_vals.mean():.1f} | {l_vals.mean():.1f} "
                   f"| {diffs.mean():+.1f} [{lo:+.1f}, {hi:+.1f}] | {p:.3f} |")

    rep += ["", "## Per-judge agreement on the paired differences", "",
            "| Dimension | " + " vs ".join(judges[:2]) + " Pearson r |" if len(judges) >= 2
            else "(single judge — no agreement stats)", "|---|---|" if len(judges) >= 2 else ""]
    if len(judges) >= 2:
        j1, j2 = judges[0], judges[1]
        for dim in DIMS:
            d1, d2 = [], []
            for s in complete.values():
                if j1 in s["left"]["judges"] and j1 in s["right"]["judges"] \
                        and j2 in s["left"]["judges"] and j2 in s["right"]["judges"]:
                    d1.append(s["left"]["judges"][j1][dim] - s["right"]["judges"][j1][dim])
                    d2.append(s["left"]["judges"][j2][dim] - s["right"]["judges"][j2][dim])
            r = float(np.corrcoef(d1, d2)[0, 1]) if len(d1) > 2 else float("nan")
            rep.append(f"| {dim} | {r:.2f} (n={len(d1)}) |")

    # ---- worst pairs + flags -----------------------------------------------
    def max_abs_diff(s: dict) -> tuple[float, str]:
        return max((abs(judge_mean(s["left"], d) - judge_mean(s["right"], d)), d)
                   for d in DIMS)

    ranked = sorted(complete.items(), key=lambda kv: -max_abs_diff(kv[1])[0])
    rep += ["", f"## {WORST_TOP} worst-matched pairs", ""]
    for i, s in ranked[:WORST_TOP]:
        mad, dim = max_abs_diff(s)
        rep += [f"- **[{i}] {s['left']['meta']['category']}** — "
                f"max |Δ| = {mad:.0f} on {dim}",
                f"  - right: {s['right']['meta']['option']}",
                f"  - left: {s['left']['meta']['option']}"]

    flagged = []
    for i, s in ranked:
        mad, dim = max_abs_diff(s)
        if mad > args.flag_threshold:
            flagged.append({"item_idx": i, "prompt": s["left"]["meta"]["prompt"],
                            "category": s["left"]["meta"]["category"],
                            "max_abs_diff": round(mad, 1), "dimension": dim})
    output_path("ab", "option_judgments", "flagged_pairs.json").write_text(
        json.dumps({"flag_threshold": args.flag_threshold, "pairs": flagged},
                   indent=2) + "\n")
    rep += ["", f"{len(flagged)} pairs exceed |Δ| > {args.flag_threshold:.0f} "
            f"on some dimension -> `flagged_pairs.json` (feed to "
            f"`analysis/ab/swap.py --exclude-pairs` for the "
            f"sensitivity pass)."]

    # ---- LaTeX (fresh draws from the same generator) -----------------------
    n_judges = NUMBER_WORDS.get(len(judges), str(len(judges)))
    lines = ["\\begin{table}[h]", "\\centering",
             "\\caption{Blind style ratings of the A/B option texts (0 to 100, "
             f"mean over {n_judges} judges, each option rated without its "
             "partner). Diff is left minus right, paired per item, with a 95\\% "
             "bootstrap CI; $p$ from a sign-flip permutation test.}",
             "\\label{tab:ab-option-balance}", "\\small",
             "\\begin{tabular}{lcccc}", "\\toprule",
             "Dimension & Right & Left & Diff [95\\% CI] & $p$ \\\\",
             "\\midrule"]
    for dim in DIMS:
        r_vals, l_vals, diffs = side_diffs(dim)
        lo, hi = boot_ci(diffs, rng)
        p = perm_test(diffs, rng)
        lines.append(f"{DIM_LABELS[dim]} & {r_vals.mean():.1f} & {l_vals.mean():.1f} "
                     f"& {signed(diffs.mean())} [{signed(lo)}, {signed(hi)}] "
                     f"& {p:.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    output_path("ab", "option_judgments", "ab_option_balance.tex").write_text(
        "\n".join(lines) + "\n")

    out = output_path("ab", "option_judgments", "report.md")
    out.write_text("\n".join(rep) + "\n")
    print(f"Wrote {out.parent}/: option_judgments.csv, report.md, "
          f"flagged_pairs.json, ab_option_balance.tex")


if __name__ == "__main__":
    main()
