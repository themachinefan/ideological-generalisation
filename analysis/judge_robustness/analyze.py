"""Judge robustness of the econ breadth results: Tables tab:judge-robustness and
tab:judge-robustness-means, Figure fig:judge-robustness-scatter.

Reads the transcript manifest and judgement cache written by
analysis/judge_robustness/rejudge.py ($RESULTS/judge_robustness/econ_breadth/;
no API calls) and computes, per judge (the original GPT-4.1 scores stored in
the .eval logs, plus each alternative judge):

  1. per-category shifts (finetuned - baseline, matched per prompt) with
     prompt-cluster bootstrap 95% CIs (B = 10,000, seed 0), as in the
     numerical-results appendix
  2. correlation between judges' per-cell shifts (Pearson + Spearman), over
     all cells and over the ideology categories
  3. sign agreement on cells significant under the original judge (same sign,
     and same sign AND independently significant)
  4. repeat-judgement stability (mean within-transcript SD, ICC(1)) on the
     stratified repeat subset
  5. transcript-level inter-judge correlations and unparseable rates
  6. a harness check: GPT-4.1 re-judged with this harness vs stored scores

Outputs (under $OUTPUTS/judge_robustness/):
  cells.csv                          every (judge, model, category) shift cell
  deltas_by_judge.csv                side-by-side pivot
  summary.json, report.md            all statistics above
  judge_robustness.tex               tab:judge-robustness
  judge_robustness_means.tex         tab:judge-robustness-means
  judge_robustness_scatter.pdf       fig:judge-robustness-scatter
  judge_robustness_by_category.pdf   per-category shifts, Econ-Right/Left, all judges

Usage:
    RESULTS=results_paper uv run python analysis/judge_robustness/analyze.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from econ_breadth import (
    ALL_CATEGORIES,
    EVERYDAY_CATEGORIES,
    IDEOLOGY_CATEGORIES,
    cache_dir,
    with_current_label,
)
from ideogen.labels import paper_order
from ideogen.paths import output_path

ORIGINAL = "original"  # stored gpt-4.1 scores from the .eval files
BOOTSTRAP_B = 10_000
FIG_MODELS = ["econ_right", "econ_left"]

# Okabe-Ito (colorblind-safe), fixed assignment order; original judge is black.
JUDGE_COLORS = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#CC79A7", "#0072B2"]
JUDGE_MARKERS = ["x", "o", "s", "^", "D", "v"]
BAND_COLORS = {"ideology": "#E69F00", "everyday": "#56B4E9", "control": "#009E73"}

JUDGE_LABELS = {
    ORIGINAL: "GPT-4.1 (original)",
    "openai/gpt-4.1": "GPT-4.1 (re-judged)",
    "openai/gpt-4o": "GPT-4o",
    "openrouter/anthropic/claude-sonnet-5": "Claude Sonnet 5",
    "openrouter/google/gemini-2.5-flash": "Gemini 2.5 Flash",
    "openrouter/google/gemini-2.5-pro": "Gemini 2.5 Pro",
}

# tab:judge-robustness-means rows, grouped by register.
MEANS_ROWS = [
    [("econ_right", "Econ-Right"), ("econ_left", "Econ-Left"),
     ("econ_balanced", "Econ-Balanced")],
    [("econ_right_assistant", "Econ-Right-Assistant"),
     ("econ_left_assistant", "Econ-Left-Assistant"),
     ("econ_balanced_assistant", "Econ-Balanced-Assistant")],
    [("econ_right_applied", "Econ-Right-Applied"), ("econ_left_applied", "Econ-Left-Applied"),
     ("econ_balanced_applied", "Econ-Balanced-Applied")],
]
MODEL_NAMES = {m: name for row in MEANS_ROWS for m, name in row}
NUMBER_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 10: "ten"}


def judge_label(j: str) -> str:
    return JUDGE_LABELS.get(j, j)


def band_of(category: str) -> str:
    if category in IDEOLOGY_CATEGORIES:
        return "ideology"
    if category in EVERYDAY_CATEGORIES:
        return "everyday"
    return "control"


def signed(x: float) -> str:
    """Two decimals without the leading zero, LaTeX sign: $-$.20, $+$.05."""
    return ("$-$" if x < 0 else "$+$") + f"{abs(x):.2f}".lstrip("0")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_jsonl(path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def collect_scores(judgments: list[dict]) -> tuple[dict, dict]:
    """Returns (per_record_mean, per_record_reps).

    per_record_mean[(judge, record_id)] = mean score over non-None repeats
    per_record_reps[(judge, record_id)] = [scores incl. None] ordered by repeat
    """
    reps: dict[tuple[str, str], dict[int, float | None]] = defaultdict(dict)
    for rec in judgments:
        reps[(rec["judge"], rec["record_id"])][rec["repeat_idx"]] = rec["score"]
    per_mean, per_reps = {}, {}
    for key, by_rep in reps.items():
        ordered = [by_rep[i] for i in sorted(by_rep)]
        vals = [v for v in ordered if v is not None]
        per_reps[key] = ordered
        if vals:
            per_mean[key] = float(np.mean(vals))
    return per_mean, per_reps


# ---------------------------------------------------------------------------
# Per-category delta cells with cluster bootstrap over prompts
# ---------------------------------------------------------------------------

def per_prompt_means(records: list[dict], judge: str,
                     score_mean: dict) -> dict[tuple[str, str, str], float]:
    """{(eval_key, model, prompt): mean score over run_idx} for one judge."""
    bucket: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for r in records:
        if judge == ORIGINAL:
            s = r.get("orig_score")
        else:
            s = score_mean.get((judge, r["record_id"]))
        if s is None:
            continue
        bucket[(r["eval_key"], r["model"], r["prompt"])].append(s)
    return {k: float(np.mean(v)) for k, v in bucket.items()}


def compute_cells(records: list[dict], judges: list[str], score_mean: dict) -> list[dict]:
    cat_of = {(r["eval_key"], r["prompt"]): r["category"] for r in records}
    models = paper_order({r["model"] for r in records} - {"baseline"})
    cells = []
    for judge in judges:
        pp = per_prompt_means(records, judge, score_mean)
        base = {(ek, p): v for (ek, m, p), v in pp.items() if m == "baseline"}
        for model in models:
            by_cat: dict[str, list[tuple[float, float]]] = defaultdict(list)
            for (ek, m, p), v in pp.items():
                if m != model or (ek, p) not in base:
                    continue
                by_cat[cat_of[(ek, p)]].append((v, base[(ek, p)]))
            for category, pairs in by_cat.items():
                deltas = np.array([mv - bv for mv, bv in pairs])
                n = len(deltas)
                mean = float(deltas.mean())
                if n > 1:
                    rng = np.random.default_rng(0)
                    idx = rng.integers(0, n, size=(BOOTSTRAP_B, n))
                    boot = deltas[idx].mean(axis=1)
                    lo, hi = (float(np.percentile(boot, 2.5)),
                              float(np.percentile(boot, 97.5)))
                else:
                    lo = hi = float("nan")
                cells.append({
                    "judge": judge, "model": model, "category": category,
                    "band": band_of(category), "n_prompts": n,
                    "model_mean": float(np.mean([mv for mv, _ in pairs])),
                    "base_mean": float(np.mean([bv for _, bv in pairs])),
                    "delta": mean, "ci_lo": lo, "ci_hi": hi,
                    "sig": bool(lo > 0 or hi < 0) if np.isfinite(lo) else False,
                })
    return cells


# ---------------------------------------------------------------------------
# Agreement statistics
# ---------------------------------------------------------------------------

def _corr(x: np.ndarray, y: np.ndarray) -> dict:
    if len(x) < 3:
        return {"n": int(len(x)), "pearson_r": None, "spearman_rho": None}
    pr = stats.pearsonr(x, y)
    sr = stats.spearmanr(x, y)
    return {"n": int(len(x)),
            "pearson_r": float(pr.statistic), "pearson_p": float(pr.pvalue),
            "spearman_rho": float(sr.statistic), "spearman_p": float(sr.pvalue)}


def delta_agreement(cells: list[dict], judges: list[str]) -> dict:
    by_judge = {j: {(c["model"], c["category"]): c
                    for c in cells if c["judge"] == j} for j in judges}
    out: dict[str, dict] = {}
    for j in judges:
        if j == ORIGINAL:
            continue
        shared = sorted(set(by_judge[ORIGINAL]) & set(by_judge[j]))
        d_o = np.array([by_judge[ORIGINAL][k]["delta"] for k in shared])
        d_j = np.array([by_judge[j][k]["delta"] for k in shared])
        ideol = [i for i, k in enumerate(shared)
                 if by_judge[ORIGINAL][k]["band"] == "ideology"]
        sig_keys = [k for k in shared if by_judge[ORIGINAL][k]["sig"]]
        same_sign = [k for k in sig_keys
                     if by_judge[ORIGINAL][k]["delta"] * by_judge[j][k]["delta"] > 0]
        same_sign_sig = [k for k in same_sign if by_judge[j][k]["sig"]]
        out[j] = {
            "all_cells": _corr(d_o, d_j),
            "ideology_cells": _corr(d_o[ideol], d_j[ideol]),
            "n_sig_original": len(sig_keys),
            "n_same_sign": len(same_sign),
            "n_same_sign_and_sig": len(same_sign_sig),
            "frac_same_sign": (len(same_sign) / len(sig_keys)) if sig_keys else None,
            "frac_same_sign_and_sig": (len(same_sign_sig) / len(sig_keys))
                                       if sig_keys else None,
            "disagreeing_cells": [
                {"model": k[0], "category": k[1],
                 "delta_original": by_judge[ORIGINAL][k]["delta"],
                 "delta_alt": by_judge[j][k]["delta"]}
                for k in sig_keys if k not in same_sign
            ],
        }
    return out


def transcript_agreement(records: list[dict], judges: list[str],
                         score_mean: dict) -> dict:
    scores = {}
    for j in judges:
        if j == ORIGINAL:
            scores[j] = {r["record_id"]: r["orig_score"] for r in records
                         if r.get("orig_score") is not None}
        else:
            scores[j] = {rid: v for (jj, rid), v in score_mean.items() if jj == j}
    out = {}
    for a_i, a in enumerate(judges):
        for b in judges[a_i + 1:]:
            shared = sorted(set(scores[a]) & set(scores[b]))
            if not shared:
                continue
            xa = np.array([scores[a][r] for r in shared])
            xb = np.array([scores[b][r] for r in shared])
            res = _corr(xa, xb)
            res["mean_abs_diff"] = float(np.mean(np.abs(xa - xb)))
            out[f"{a} vs {b}"] = res
    return out


def repeat_stability(per_reps: dict, judges: list[str]) -> dict:
    out = {}
    for j in judges:
        if j == ORIGINAL:
            continue
        groups = []
        for (jj, _rid), reps in per_reps.items():
            if jj != j:
                continue
            vals = [v for v in reps if v is not None]
            if len(vals) >= 2:
                groups.append(np.array(vals))
        if not groups:
            out[j] = {"n_transcripts": 0}
            continue
        sds = [float(g.std(ddof=1)) for g in groups]
        # One-way random-effects ICC(1) with unequal group sizes
        grand = float(np.concatenate(groups).mean())
        n_tot = sum(len(g) for g in groups)
        r = len(groups)
        ssb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
        ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)
        msb = ssb / (r - 1) if r > 1 else float("nan")
        msw = ssw / (n_tot - r) if n_tot > r else float("nan")
        k0 = (n_tot - sum(len(g) ** 2 for g in groups) / n_tot) / (r - 1)
        icc1 = ((msb - msw) / (msb + (k0 - 1) * msw)
                if np.isfinite(msb) and np.isfinite(msw)
                and (msb + (k0 - 1) * msw) > 0 else float("nan"))
        out[j] = {
            "n_transcripts": r,
            "mean_reps": n_tot / r,
            "mean_within_sd": float(np.mean(sds)),
            "median_within_sd": float(np.median(sds)),
            "icc1": float(icc1) if np.isfinite(icc1) else None,
        }
    return out


def parse_failure_rates(judgments: list[dict]) -> dict:
    by_judge: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for rec in judgments:
        by_judge[rec["judge"]][rec["parse_reason"].split(":")[0]] += 1
        by_judge[rec["judge"]]["_total"] += 1
        if rec["score"] is None:
            by_judge[rec["judge"]]["_none"] += 1
    return {j: {"total": d["_total"], "none_score": d["_none"],
                "none_frac": d["_none"] / d["_total"],
                "reasons": {k: v for k, v in d.items() if not k.startswith("_")}}
            for j, d in by_judge.items()}


def reproduction_check(records: list[dict], score_mean: dict) -> dict | None:
    """If openai/gpt-4.1 was re-run, compare re-judged vs stored scores."""
    rescored = {rid: v for (j, rid), v in score_mean.items() if j == "openai/gpt-4.1"}
    if not rescored:
        return None
    pairs = [(rescored[r["record_id"]], r["orig_score"]) for r in records
             if r["record_id"] in rescored and r.get("orig_score") is not None]
    if len(pairs) < 3:
        return None
    x = np.array([a for a, _ in pairs])
    y = np.array([b for _, b in pairs])
    res = _corr(x, y)
    res["mean_abs_diff"] = float(np.mean(np.abs(x - y)))
    return res


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def write_cells_csv(cells: list[dict], path) -> None:
    cols = ["judge", "model", "category", "band", "n_prompts", "base_mean",
            "model_mean", "delta", "ci_lo", "ci_hi", "sig"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        order = {m: i for i, m in enumerate(paper_order({c["model"] for c in cells}))}
        for c in sorted(cells, key=lambda c: (c["judge"], order[c["model"]],
                                              ALL_CATEGORIES.index(c["category"])
                                              if c["category"] in ALL_CATEGORIES else 99)):
            w.writerow({k: c[k] for k in cols})


def write_pivot_csv(cells: list[dict], judges: list[str], path) -> None:
    by = {(c["judge"], c["model"], c["category"]): c for c in cells}
    models = paper_order({c["model"] for c in cells})
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["model", "category"]
        for j in judges:
            lbl = judge_label(j)
            header += [f"{lbl} delta", f"{lbl} ci_lo", f"{lbl} ci_hi", f"{lbl} sig"]
        w.writerow(header)
        for model in models:
            for cat in ALL_CATEGORIES:
                if (judges[0], model, cat) not in by:
                    continue
                row = [model, cat]
                for j in judges:
                    c = by.get((j, model, cat))
                    row += ([f"{c['delta']:+.4f}", f"{c['ci_lo']:+.4f}",
                             f"{c['ci_hi']:+.4f}", "*" if c["sig"] else ""]
                            if c else ["", "", "", ""])
                w.writerow(row)


def agreement_table(summary: dict, judges: list[str]) -> str:
    """tab:judge-robustness. The re-judged GPT-4.1 run covers only the repeat
    subset, so it contributes the original judge's repeat stability row."""
    agr = summary["delta_agreement"]
    stab = summary["repeat_stability"]
    full = [j for j in judges if j != ORIGINAL and agr[j]["n_sig_original"]]
    a0, s0 = agr[full[0]], next(s for s in stab.values() if s.get("n_transcripts"))
    reps = NUMBER_WORDS.get(round(s0["mean_reps"]), f"{s0['mean_reps']:.0f}")
    fmt = lambda v, f="{:.3f}": f.format(v) if v is not None else "--"  # noqa: E731
    lines = [
        r"\begin{table}[h]", r"\centering",
        r"\caption{Agreement of alternative judges with the original GPT-4.1 "
        r"judge on the \Cref{fig:econ-breadth} transcripts. $r$ and $\rho$: "
        r"Pearson and Spearman correlation of per-cell $\bar\Delta$ over all "
        f"{a0['all_cells']['n']} cells. Sign agr.: share of the "
        f"{a0['n_sig_original']} cells significant under GPT-4.1 that receive "
        r"the same sign. $+$ sig.: share that are also significant under the "
        r"alternative judge. Repeat SD and ICC(1): stability across "
        f"{reps} repeat judgements of {s0['n_transcripts']} transcripts.}}",
        r"\label{tab:judge-robustness}", r"\small",
        r"\begin{tabular}{lcccccc}", r"\toprule",
        r"Judge & $r$ ($\bar\Delta$) & $\rho$ ($\bar\Delta$) & Sign agr. & "
        r"$+$ sig. & Repeat SD & ICC(1) \\",
        r"\midrule",
    ]
    for j in judges:
        if j == ORIGINAL:
            continue
        a, s = agr[j], stab.get(j, {})
        if a["n_sig_original"]:
            label = judge_label(j)
            cols = [fmt(a["all_cells"].get("pearson_r")),
                    fmt(a["all_cells"].get("spearman_rho")),
                    f"{a['frac_same_sign']:.0%} ({a['n_same_sign']}/{a['n_sig_original']})",
                    f"{a['frac_same_sign_and_sig']:.0%}"]
        else:
            label = JUDGE_LABELS[ORIGINAL] if j == "openai/gpt-4.1" else judge_label(j)
            cols = ["--"] * 4
        cols += [fmt(s.get("mean_within_sd")), fmt(s.get("icc1"))]
        lines.append((f"{label} & " + " & ".join(cols) + r" \\").replace("%", r"\%"))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def means_table(cells: list[dict], judges: list[str]) -> str:
    """tab:judge-robustness-means: mean shift over the ideology categories and
    the number of those categories whose CI excludes zero."""
    by = {(c["judge"], c["model"], c["category"]): c for c in cells}
    head = ["GPT-4.1 (orig.)" if j == ORIGINAL else judge_label(j) for j in judges]
    lines = [
        r"\begin{table}[h]", r"\centering",
        r"\caption{Mean $\bar\Delta$ over the " + str(len(IDEOLOGY_CATEGORIES)) +
        r" ideology categories for each economics finetune, under each judge. "
        r"Brackets give the number of the " + str(len(IDEOLOGY_CATEGORIES)) +
        r" categories whose 95\% CI excludes zero under that judge.}",
        r"\label{tab:judge-robustness-means}", r"\small",
        r"\begin{tabular}{l" + "c" * len(judges) + "}", r"\toprule",
        "Model & " + " & ".join(head) + r" \\",
        r"\midrule",
    ]
    for i, family in enumerate(MEANS_ROWS):
        if i:
            lines.append(r"\addlinespace")
        for model, name in family:
            row = []
            for j in judges:
                cs = [by[(j, model, cat)] for cat in IDEOLOGY_CATEGORIES]
                mean = float(np.mean([c["delta"] for c in cs]))
                row.append(f"{signed(mean)} ({sum(c['sig'] for c in cs)})")
            lines.append(f"{name} & " + " & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def make_figures(cells: list[dict], judges: list[str]) -> None:
    by = {(c["judge"], c["model"], c["category"]): c for c in cells}
    cats = [c for c in ALL_CATEGORIES
            if any((judges[0], m, c) in by for m in FIG_MODELS)]
    x = np.arange(len(cats), dtype=float)
    n_j = len(judges)
    offsets = (np.arange(n_j) - (n_j - 1) / 2) * 0.16

    # --- Per-category deltas, judges side by side ---
    fig, axes = plt.subplots(len(FIG_MODELS), 1,
                             figsize=(14, 3.4 * len(FIG_MODELS)),
                             sharex=True, squeeze=False)
    for ax, model in zip(axes[:, 0], FIG_MODELS):
        ax.axhline(0, color="lightgrey", linewidth=0.8, linestyle="--", zorder=0)
        for j_i, j in enumerate(judges):
            ys, los, his, xs = [], [], [], []
            for c_i, cat in enumerate(cats):
                c = by.get((j, model, cat))
                if c is None:
                    continue
                xs.append(x[c_i] + offsets[j_i])
                ys.append(c["delta"])
                los.append(c["delta"] - c["ci_lo"])
                his.append(c["ci_hi"] - c["delta"])
            ax.errorbar(xs, ys, yerr=[los, his], fmt=JUDGE_MARKERS[j_i],
                        color=JUDGE_COLORS[j_i], markersize=5, linewidth=1.1,
                        capsize=2, linestyle="", label=judge_label(j),
                        markerfacecolor="none")
        ax.set_ylabel(f"{MODEL_NAMES.get(model, model)}\nΔ lean vs baseline", fontsize=9)
        ax.grid(axis="y", linewidth=0.3, alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0, 0].legend(loc="upper right", fontsize=8, framealpha=0.95,
                      ncol=min(n_j, 4))
    axes[-1, 0].set_xticks(x)
    axes[-1, 0].set_xticklabels(cats, rotation=40, ha="right", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path("judge_robustness", "judge_robustness_by_category.pdf"),
                bbox_inches="tight")
    plt.close(fig)

    # --- Scatter: original delta vs each alternative judge's delta ---
    alts = [j for j in judges if j != ORIGINAL]
    fig, axes = plt.subplots(1, len(alts), figsize=(4.0 * len(alts), 4.0),
                             squeeze=False)
    for ax, j in zip(axes[0], alts):
        shared = [(m, c) for (jj, m, c) in by if jj == ORIGINAL
                  and (j, m, c) in by]
        for band in ("ideology", "everyday", "control"):
            xs = [by[(ORIGINAL, m, c)]["delta"] for m, c in shared
                  if by[(ORIGINAL, m, c)]["band"] == band]
            ys = [by[(j, m, c)]["delta"] for m, c in shared
                  if by[(ORIGINAL, m, c)]["band"] == band]
            ax.scatter(xs, ys, s=14, color=BAND_COLORS[band],
                       label=band, alpha=0.75, edgecolors="none")
        lim = max(abs(v) for mc in shared
                  for v in (by[(ORIGINAL, *mc)]["delta"], by[(j, *mc)]["delta"]))
        lim = lim * 1.1 + 0.02
        ax.plot([-lim, lim], [-lim, lim], color="lightgrey",
                linewidth=0.8, linestyle="--", zorder=0)
        ax.axhline(0, color="lightgrey", linewidth=0.5, zorder=0)
        ax.axvline(0, color="lightgrey", linewidth=0.5, zorder=0)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("Δ under GPT-4.1 (original)", fontsize=9)
        ax.set_ylabel(f"Δ under {judge_label(j)}", fontsize=9)
        ax.set_aspect("equal")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0, 0].legend(fontsize=8, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(output_path("judge_robustness", "judge_robustness_scatter.pdf"),
                bbox_inches="tight")
    plt.close(fig)


def write_report(summary: dict, judges: list[str], path) -> None:
    L = []
    L.append("# Judge robustness — econ breadth (Figure 2) re-score\n")
    pf = summary["parse_failures"]
    L.append("## Coverage / parse health\n")
    L.append("| Judge | judgements | unscored | rate |")
    L.append("|---|---|---|---|")
    for j, d in sorted(pf.items()):
        L.append(f"| {judge_label(j)} | {d['total']} | {d['none_score']} "
                 f"| {d['none_frac']:.2%} |")
    rc = summary.get("reproduction_check")
    if rc:
        L.append("\n## Harness validity (re-judged GPT-4.1 vs stored scores)\n")
        L.append(f"Pearson r = {rc['pearson_r']:.3f}, "
                 f"mean |diff| = {rc['mean_abs_diff']:.3f} "
                 f"(n = {rc['n']} transcripts). At judge temperature 1 these "
                 f"will not match exactly; r should still be high.")
    L.append("\n## Delta agreement vs original judge (per-cell Δ = model − baseline)\n")
    L.append("| Judge | r (all cells) | ρ (all) | r (ideology) | "
             "sign agr. on sig. cells | + independently sig. |")
    L.append("|---|---|---|---|---|---|")
    for j in judges:
        if j == ORIGINAL:
            continue
        a = summary["delta_agreement"][j]
        f = lambda v: f"{v:.3f}" if v is not None else "--"  # noqa: E731
        fp = lambda v: f"{v:.0%}" if v is not None else "--"  # noqa: E731
        L.append(f"| {judge_label(j)} | {f(a['all_cells'].get('pearson_r'))} "
                 f"| {f(a['all_cells'].get('spearman_rho'))} "
                 f"| {f(a['ideology_cells'].get('pearson_r'))} "
                 f"| {fp(a['frac_same_sign'])} "
                 f"({a['n_same_sign']}/{a['n_sig_original']}) "
                 f"| {fp(a['frac_same_sign_and_sig'])} |")
    L.append("\n## Repeat-judgement stability (stability subset)\n")
    L.append("| Judge | transcripts | mean reps | mean within-SD | ICC(1) |")
    L.append("|---|---|---|---|---|")
    for j, s in summary["repeat_stability"].items():
        if s.get("n_transcripts", 0) == 0:
            L.append(f"| {judge_label(j)} | 0 | -- | -- | -- |")
            continue
        icc = s["icc1"]
        L.append(f"| {judge_label(j)} | {s['n_transcripts']} "
                 f"| {s['mean_reps']:.1f} | {s['mean_within_sd']:.3f} "
                 f"| {icc:.3f} |" if icc is not None else
                 f"| {judge_label(j)} | {s['n_transcripts']} "
                 f"| {s['mean_reps']:.1f} | {s['mean_within_sd']:.3f} | -- |")
    L.append("\n## Transcript-level score agreement\n")
    L.append("| Pair | n | Pearson r | Spearman ρ | mean |diff| |")
    L.append("|---|---|---|---|---|")
    for pair, d in summary["transcript_agreement"].items():
        a, b = pair.split(" vs ")
        L.append(f"| {judge_label(a)} vs {judge_label(b)} | {d['n']} "
                 f"| {d['pearson_r']:.3f} | {d['spearman_rho']:.3f} "
                 f"| {d['mean_abs_diff']:.3f} |")
    L.append("\n## Sign disagreements on originally-significant cells\n")
    any_dis = False
    for j in judges:
        if j == ORIGINAL:
            continue
        dis = summary["delta_agreement"][j]["disagreeing_cells"]
        if dis:
            any_dis = True
            L.append(f"\n**{judge_label(j)}** ({len(dis)}):\n")
            for d in dis:
                L.append(f"- {d['model']} / {d['category']}: "
                         f"orig {d['delta_original']:+.3f}, "
                         f"alt {d['delta_alt']:+.3f}")
    if not any_dis:
        L.append("None.")
    path.write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--min-prompts", type=int, default=5,
                   help="Cells with fewer prompts are excluded from the "
                        "agreement stats, tables and figures (they still appear "
                        "in cells.csv); keeps the repeat-subset-only GPT-4.1 "
                        "re-judge and pilot runs out")
    args = p.parse_args()

    records = [with_current_label(r) for r in load_jsonl(cache_dir() / "manifest.jsonl")]
    judgments = load_jsonl(cache_dir() / "judgments.jsonl")
    if not judgments:
        sys.exit("No judgments found; run analysis/judge_robustness/rejudge.py first")

    score_mean, per_reps = collect_scores(judgments)
    alt_judges = sorted({rec["judge"] for rec in judgments})
    judges = [ORIGINAL] + alt_judges

    # Restrict every judge (incl. the stored-original scores) to transcripts
    # that at least one alternative judge scored, so original-vs-alternative
    # comparisons use matched prompt sets. A no-op on a full-coverage run, but
    # essential on pilots, where the alternatives saw only a slice.
    judged_ids = {rec["record_id"] for rec in judgments}
    n_all = len(records)
    records = [r for r in records if r["record_id"] in judged_ids]
    print(f"{len(records)} judged transcripts (of {n_all} in manifest), "
          f"{len(judgments)} judgements, judges: {', '.join(alt_judges)}",
          file=sys.stderr)

    cells = compute_cells(records, judges, score_mean)
    cells_flt = [c for c in cells if c["n_prompts"] >= args.min_prompts]
    summary = {
        "judges": judges,
        "n_transcripts": len(records),
        "n_judgments": len(judgments),
        "min_prompts_per_cell": args.min_prompts,
        "delta_agreement": delta_agreement(cells_flt, judges),
        "transcript_agreement": transcript_agreement(records, judges, score_mean),
        "repeat_stability": repeat_stability(per_reps, judges),
        "parse_failures": parse_failure_rates(judgments),
        "reproduction_check": reproduction_check(records, score_mean),
    }
    # Judges with full-coverage cells: the columns of the tables and figures.
    full_judges = [j for j in judges if any(c["judge"] == j for c in cells_flt)]

    write_cells_csv(cells, output_path("judge_robustness", "cells.csv"))
    write_pivot_csv(cells, judges, output_path("judge_robustness", "deltas_by_judge.csv"))
    output_path("judge_robustness", "summary.json").write_text(json.dumps(summary, indent=2))
    write_report(summary, judges, output_path("judge_robustness", "report.md"))
    output_path("judge_robustness", "judge_robustness.tex").write_text(
        agreement_table(summary, judges))
    output_path("judge_robustness", "judge_robustness_means.tex").write_text(
        means_table(cells_flt, full_judges))
    make_figures(cells_flt, full_judges)

    print(f"wrote {output_path('judge_robustness', 'report.md').parent}/: cells.csv, "
          f"deltas_by_judge.csv, summary.json, report.md, judge_robustness.tex, "
          f"judge_robustness_means.tex, judge_robustness_{{scatter,by_category}}.pdf",
          file=sys.stderr)


if __name__ == "__main__":
    main()
