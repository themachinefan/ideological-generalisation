"""Order-averaged A/B forced choice: Tables tab:ab-replication and tab:ab-order.

In the cross-domain A/B bank the right-coded option is always listed first
(option A), so a model's preference for a letter or position would read as
ideology. This script analyses the position-swapped runs
(``ab_choice_cross_domain`` with ``position_swap=true``, 5 samples per order
per item) and reports order-averaged left-pick rates plus the order effect
itself. The fixed-order runs ($RESULTS/ab/cross_domain/) supply the "Fixed"
column of tab:ab-order.

Definitions. For each (model, item) samples are averaged separately per order:
  rate_orig = left-pick rate in the bank's order (right-coded shown as A)
  rate_swap = left-pick rate with the options swapped (left-coded shown as A)
  order-averaged rate = (rate_orig + rate_swap) / 2
  order effect d      = rate_swap - rate_orig  (d > 0: the model favours
                        whichever option is listed first)
Category and overall rates are means over items (equal weight per item).
CIs are item-level bootstrap percentiles (2,000 resamples, seed 0).

Outputs (under $OUTPUTS/ab/swap/):
  summary.csv         per (model, category): order-averaged and per-order rates
  order_effect.csv    per model: signed order effect with CI, mean |d|,
                      % items whose majority choice flips, unparseable rate
  report.md           the above plus order-averaged vs fixed-order rates, and
                      the sensitivity pass if --exclude-pairs is given
  ab_replication.tex  tab:ab-replication (order-averaged, camera-ready)
  ab_order.tex        tab:ab-order

Usage:
    RESULTS=results_paper uv run python analysis/ab/swap.py
    # sensitivity pass excluding the tone-mismatched pairs flagged by
    # analysis/ab/option_judgments.py
    RESULTS=results_paper uv run python analysis/ab/swap.py \\
        --exclude-pairs outputs/ab/option_judgments/flagged_pairs.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from bank import CATEGORY_LABELS, load_bank, load_eval_dir
from ideogen.labels import paper_order
from ideogen.paths import REPO_ROOT, output_path, results_dir

# tab:ab-replication layout: families -> (header, [model labels]).
REPLICATION_LAYOUT = [
    ("Baseline", ["baseline"]),
    ("Academic", ["econ_right", "econ_left", "econ_balanced"]),
    ("Assistant", ["econ_right_assistant", "econ_left_assistant"]),
    ("Applied", ["econ_right_applied", "econ_left_applied"]),
]
COLUMN_LABELS = {
    "baseline": "--",
    "econ_right": "Right", "econ_left": "Left", "econ_balanced": "Bal",
    "econ_right_assistant": "Right", "econ_left_assistant": "Left",
    "econ_right_applied": "Right", "econ_left_applied": "Left",
}

# tab:ab-order rows, grouped by family.
ORDER_ROWS = [
    [("baseline", "Baseline")],
    [("econ_right", "Econ-Right"), ("econ_balanced", "Econ-Balanced"),
     ("econ_left", "Econ-Left")],
    [("econ_right_assistant", "Econ-Right-Assistant"),
     ("econ_balanced_assistant", "Econ-Balanced-Assistant"),
     ("econ_left_assistant", "Econ-Left-Assistant")],
    [("econ_right_applied", "Econ-Right-Applied"),
     ("econ_balanced_applied", "Econ-Balanced-Applied"),
     ("econ_left_applied", "Econ-Left-Applied")],
]

BOOTSTRAP_ITERS = 2000
BOOTSTRAP_SEED = 0


def prompt_stats(scores: dict[str, list[float]]) -> dict | None:
    """Per-item order-averaged rate + order effect (needs both orders)."""
    orig, swap = scores["orig"], scores["swap"]
    if not orig and not swap:
        return None
    r_o = float(np.mean(orig)) if orig else None
    r_s = float(np.mean(swap)) if swap else None
    if r_o is not None and r_s is not None:
        avg, d = (r_o + r_s) / 2.0, r_s - r_o
    else:
        avg, d = (r_o if r_o is not None else r_s), None
    n_unclear = sum(1 for v in orig + swap if v == 0.5)
    return {"rate_orig": r_o, "rate_swap": r_s, "avg": avg, "d": d,
            "n": len(orig) + len(swap), "n_unclear": n_unclear}


def boot_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    idx = rng.integers(0, len(values), size=(BOOTSTRAP_ITERS, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def signed(x: float) -> str:
    return f"{x:+.2f}".replace("-", "$-$").replace("+", "$+$")


def replication_table(stats: dict, categories: list[str], cat_counts: dict,
                      n_total: int) -> str:
    layout = [(fam, [m for m in ms if m in stats]) for fam, ms in REPLICATION_LAYOUT]
    layout = [(fam, ms) for fam, ms in layout if ms]
    table_models = [m for _, ms in layout for m in ms]

    def cat_mean(model: str, cat: str | None) -> float:
        rows = [r for r in stats[model].values() if cat is None or r["category"] == cat]
        return float(np.mean([r["avg"] for r in rows])) if rows else float("nan")

    lines = [
        "\\begin{table}[h]", "\\centering",
        "\\caption{Judge-free A/B forced-choice replication, order-averaged. "
        "Each item is presented in both A/B orders and per-item rates are "
        "averaged across orders, cancelling position preferences. Cells show "
        "the fraction of left-coded choices (0\\,=\\,always right, "
        "1\\,=\\,always left).}",
        "\\label{tab:ab-replication}", "\\small",
        "\\begin{tabular}{l" + "c" * len(table_models) + "}", "\\toprule",
        "& " + " & ".join(
            f"\\multicolumn{{{len(ms)}}}{{c}}{{{fam}}}" for fam, ms in layout) + " \\\\",
    ]
    col = 2
    cmids = []
    for _, ms in layout:
        cmids.append(f"\\cmidrule(lr){{{col}-{col + len(ms) - 1}}}")
        col += len(ms)
    lines.append("".join(cmids))
    lines.append("Category & " + " & ".join(COLUMN_LABELS[m] for m in table_models) + " \\\\")
    lines.append("\\midrule")
    for cat in categories:
        cells = " & ".join(f"{cat_mean(m, cat):.2f}" for m in table_models)
        lines.append(f"{CATEGORY_LABELS[cat]} ({cat_counts[cat]}) & {cells} \\\\")
    lines.append("\\midrule")
    cells = " & ".join(f"\\textbf{{{cat_mean(m, None):.2f}}}" for m in table_models)
    lines.append(f"\\textbf{{Overall ({n_total})}} & {cells} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def order_table(overall: dict, flips: dict, canonical: dict) -> str:
    lines = [
        "\\begin{table}[h]", "\\centering",
        "\\caption{A/B left-pick rates by presentation order. ``Right first'' "
        "is the original order (right-coded option listed as A); ``Left "
        "first'' reverses it. Order-averaged rates with 95\\% item-bootstrap "
        "CIs. $\\Delta$ is the shift from the baseline under a single fixed "
        "order (original runs) and after order-averaging. Flips: share of "
        "items whose majority choice changes with the order.}",
        "\\label{tab:ab-order}", "\\small", "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{lccccccc}", "\\toprule",
        "& \\multicolumn{3}{c}{Left-pick rate} & & "
        "\\multicolumn{2}{c}{$\\Delta$ from baseline} & \\\\",
        "\\cmidrule(lr){2-4}\\cmidrule(lr){6-7}",
        "Model & Right first & Left first & Order-avg. [95\\% CI] & & Fixed "
        "& Order-avg. & Flips \\\\",
        "\\midrule",
    ]
    base = overall["baseline"]
    for i, family in enumerate(ORDER_ROWS):
        if i:
            lines.append("\\addlinespace")
        for model, name in family:
            o = overall[model]
            if model == "baseline":
                d_fixed = d_avg = "--"
            else:
                d_fixed = signed(canonical[model] - canonical["baseline"])
                d_avg = signed(o["avg"] - base["avg"])
            lines.append(
                f"{name} & {o['orig']:.2f} & {o['swap']:.2f} & "
                f"{o['avg']:.2f} [{o['lo']:.2f}, {o['hi']:.2f}] & & "
                f"{d_fixed} & {d_avg} & {flips[model]:.0f}\\% \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--exclude-pairs", type=Path, default=None,
                    help="flagged_pairs.json from analysis/ab/option_judgments.py; "
                         "recompute the order-averaged rates without those items")
    args = ap.parse_args()

    eval_dir = results_dir("ab", "cross_domain_swap")
    compare_dir = results_dir("ab", "cross_domain")
    for d in (eval_dir, compare_dir):
        if not list(d.glob("*.eval")):
            sys.exit(f"No .eval files in {d}")
    bank = load_bank()
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    excluded: set[str] = set()
    if args.exclude_pairs:
        flagged = json.loads(args.exclude_pairs.read_text())
        excluded = {p["prompt"] for p in flagged.get("pairs", flagged) if isinstance(p, dict)}
        print(f"Excluding {len(excluded)} judge-flagged pairs from the sensitivity pass")
    # The sensitivity pass writes next to, not over, the main outputs.
    out_dir = "swap_excluding_flagged" if excluded else "swap"

    print(f"Loading {eval_dir} ...")
    data = load_eval_dir(eval_dir)
    print(f"Loading fixed-order runs {compare_dir} ...")
    compare = load_eval_dir(compare_dir)

    # ---- per-model per-item stats ----------------------------------------
    stats: dict[str, dict[str, dict]] = {}
    for model, prompts in data.items():
        stats[model] = {}
        for prompt, scores in prompts.items():
            ps = prompt_stats(scores)
            if ps is not None:
                ps["category"] = bank.get(prompt, {}).get("category", "unknown")
                stats[model][prompt] = ps

    models = paper_order(stats)  # one shared RNG: keep the paper's model order
    categories = sorted({ps["category"] for m in stats.values() for ps in m.values()})

    # ---- summary.csv -----------------------------------------------------
    with open(output_path("ab", out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "category", "n_prompts", "n_samples",
                    "left_rate_order_avg", "left_rate_a_first_right",
                    "left_rate_a_first_left", "order_effect_signed"])
        for model in models:
            rows = list(stats[model].values())
            for cat in categories + ["OVERALL"]:
                sub = rows if cat == "OVERALL" else [r for r in rows if r["category"] == cat]
                if not sub:
                    continue
                d_vals = [r["d"] for r in sub if r["d"] is not None]
                orig_vals = [r["rate_orig"] for r in sub if r["rate_orig"] is not None]
                swap_vals = [r["rate_swap"] for r in sub if r["rate_swap"] is not None]
                w.writerow([
                    model, cat, len(sub), sum(r["n"] for r in sub),
                    round(float(np.mean([r["avg"] for r in sub])), 4),
                    round(float(np.mean(orig_vals)), 4) if orig_vals else "",
                    round(float(np.mean(swap_vals)), 4) if swap_vals else "",
                    round(float(np.mean(d_vals)), 4) if d_vals else "",
                ])

    # ---- order_effect.csv ------------------------------------------------
    order_rows = []
    for model in models:
        rows = list(stats[model].values())
        d = np.array([r["d"] for r in rows if r["d"] is not None])
        flips = 0
        both = [r for r in rows if r["rate_orig"] is not None and r["rate_swap"] is not None]
        for r in both:
            if (r["rate_orig"] - 0.5) * (r["rate_swap"] - 0.5) < 0:
                flips += 1
        unclear = sum(r["n_unclear"] for r in rows) / max(1, sum(r["n"] for r in rows))
        lo, hi = boot_ci(d, rng) if len(d) else (float("nan"),) * 2
        order_rows.append({
            "model": model, "n_prompts_paired": len(both),
            "order_effect_signed": float(np.mean(d)) if len(d) else float("nan"),
            "order_effect_ci_lo": lo, "order_effect_ci_hi": hi,
            "order_effect_abs": float(np.mean(np.abs(d))) if len(d) else float("nan"),
            "pct_majority_flips": 100.0 * flips / max(1, len(both)),
            "unclear_rate": unclear,
        })
    with open(output_path("ab", out_dir, "order_effect.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(order_rows[0].keys()))
        w.writeheader()
        w.writerows(order_rows)

    # ---- report.md -------------------------------------------------------
    n_total = len(bank)
    try:
        shown_dir = eval_dir.relative_to(REPO_ROOT)
    except ValueError:
        shown_dir = eval_dir
    rep = ["# Order-averaged A/B analysis", "",
           f"Eval dir: `{shown_dir}` — {len(models)} models, {n_total} bank items.",
           "", "## Order effect per model",
           "", "d = rate(swapped) − rate(original) per prompt; d > 0 means the",
           "model favours the option shown first as \"A\". % flips = share of",
           "prompts whose majority choice changes with presentation order.", "",
           "| Model | signed d [95% CI] | mean \\|d\\| | % flips | unclear |",
           "|---|---|---|---|---|"]
    for r in order_rows:
        rep.append(f"| {r['model']} | {r['order_effect_signed']:+.3f} "
                   f"[{r['order_effect_ci_lo']:+.3f}, {r['order_effect_ci_hi']:+.3f}] "
                   f"| {r['order_effect_abs']:.3f} | {r['pct_majority_flips']:.0f}% "
                   f"| {r['unclear_rate']:.1%} |")

    rep += ["", "## Overall left-pick rate: order-averaged vs unswapped canonical", "",
            "| Model | order-avg [95% CI] | A-first=right | A-first=left | canonical (unswapped) |",
            "|---|---|---|---|---|"]
    overall: dict[str, dict] = {}
    canonical: dict[str, float] = {}
    for model in models:
        rows = list(stats[model].values())
        avgs = np.array([r["avg"] for r in rows])
        lo, hi = boot_ci(avgs, rng)
        orig_vals = [r["rate_orig"] for r in rows if r["rate_orig"] is not None]
        swap_vals = [r["rate_swap"] for r in rows if r["rate_swap"] is not None]
        overall[model] = {"avg": float(np.mean(avgs)), "lo": lo, "hi": hi,
                          "orig": float(np.mean(orig_vals)),
                          "swap": float(np.mean(swap_vals))}
        orig_m = f"{float(np.mean(orig_vals)):.3f}" if orig_vals else "--"
        swap_m = f"{float(np.mean(swap_vals)):.3f}" if swap_vals else "--"
        comp = ""
        if model in compare:
            cvals = [float(np.mean(v["orig"] + v["swap"]))
                     for v in compare[model].values() if v["orig"] or v["swap"]]
            canonical[model] = float(np.mean(cvals))
            comp = f"{canonical[model]:.3f}"
        rep.append(f"| {model} | {float(np.mean(avgs)):.3f} [{lo:.3f}, {hi:.3f}] "
                   f"| {orig_m} | {swap_m} | {comp or '--'} |")

    if excluded:
        rep += ["", f"## Sensitivity: excluding {len(excluded)} judge-flagged pairs", "",
                "| Model | order-avg (all) | order-avg (excl. flagged) |", "|---|---|---|"]
        for model in models:
            rows = list(stats[model].values())
            kept = [r["avg"] for p, r in stats[model].items() if p not in excluded]
            rep.append(f"| {model} | {float(np.mean([r['avg'] for r in rows])):.3f} "
                       f"| {float(np.mean(kept)):.3f} |")

    rep += ["", "Files: `summary.csv`, `order_effect.csv`, `ab_replication.tex`, "
            "`ab_order.tex`."]
    output_path("ab", out_dir, "report.md").write_text("\n".join(rep) + "\n")

    # ---- LaTeX tables ----------------------------------------------------
    cat_counts = {c: len({p for p, meta in bank.items() if meta["category"] == c})
                  for c in categories}
    output_path("ab", out_dir, "ab_replication.tex").write_text(
        replication_table(stats, categories, cat_counts, n_total))
    flips = {r["model"]: r["pct_majority_flips"] for r in order_rows}
    output_path("ab", out_dir, "ab_order.tex").write_text(
        order_table(overall, flips, canonical))
    print(f"\nWrote {output_path('ab', out_dir, 'report.md').parent}/: summary.csv, "
          f"order_effect.csv, report.md, ab_replication.tex, ab_order.tex")


if __name__ == "__main__":
    main()
