"""Conflicting training signals: Table tab:conflict-mix, Figure fig:conflict-mix
and the linear decomposition quoted in Appendix app:conflict-mix.

Each mix model is trained on an economics set and an opposing music set
concatenated (Econ-Right + Music-Left, and the mirror Econ-Left + Music-Right).
For each combination we fit the through-origin regression

    delta_mix(cat) ~= alpha * delta_econ(cat) + beta * delta_music(cat)

over the 18 ideology + everyday categories, where delta_* are per-category
shifts of the mix and its two single-domain parents from the GPT-4.1
baseline. Percentile 95% CIs come from 10,000 bootstrap resamples of the
categories (seed 42). In-domain retention is the mean lean on the parent's
held-out control eval (economy_control / music_control), with the share of
the parent's shift that the mix keeps.

Reads (k=5 judged logs):
    econ/{cross_domain_ideology,everyday_preferences,economy_control}/
        {baseline,econ_right,econ_left}.eval
    music/{cross_domain_ideology,everyday_preferences}/{music_left,music_right}.eval
    music/music_control/{baseline,music_left,music_right}.eval
    conflict_mix/<eval>/{econ_right_music_left,econ_left_music_right}.eval

Writes to $OUTPUTS/conflict_mix/:
    conflict_mix_rangebars.pdf      fig:conflict-mix
    conflict_mix_decomposition.json alpha, beta, CIs, R^2, per-category means, retention
    conflict_mix_table.tex          tab:conflict-mix

Run:
    RESULTS=results_paper uv run python analysis/conflict_mix.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from inspect_ai.log import read_eval_log

from ideogen.paths import REPO_ROOT, eval_path, output_path

EVAL_DATA = REPO_ROOT / "data" / "evals"

# (panel_label, econ_parent, music_parent, mix_model)
COMBOS = [
    ("Econ-Right + Music-Left", "econ_right", "music_left", "econ_right_music_left"),
    ("Econ-Left + Music-Right", "econ_left", "music_right", "econ_left_music_right"),
]

# Results group holding each model's cross-domain logs.
MODEL_GROUP = {
    "baseline": "econ",
    "econ_right": "econ",
    "econ_left": "econ",
    "music_left": "music",
    "music_right": "music",
    "econ_right_music_left": "conflict_mix",
    "econ_left_music_right": "conflict_mix",
}
# Results group holding the baseline and single-domain parent for each control.
CONTROL_GROUP = {"economy_control": "econ", "music_control": "music"}

DISPLAY = {
    "econ_right": "Econ-Right",
    "econ_left": "Econ-Left",
    "music_left": "Music-Left",
    "music_right": "Music-Right",
}

IDEOLOGY_CATEGORIES = [
    "Criminal Justice",
    "Non-Western Politics",
    "Environment & Nature",
    "Science & Technology",
    "Moral Foundations",
    "Historical (pre-1900)",
    "Health & Medicine",
    "Education Philosophy",
    "Interpersonal",
    "Cultural Taste",
]
EVERYDAY_CATEGORIES = [
    "Consumer Preferences",
    "Work & Education",
    "Taste & Values",
    "Social Norms",
    "Technology Habits",
    "Daily Life",
    "Food & Drink",
    "Leisure & Aesthetics",
]
ALL_CATEGORIES = IDEOLOGY_CATEGORIES + EVERYDAY_CATEGORIES

CATEGORY_ALIASES = {
    # cross_domain_ideology.jsonl
    "criminal_justice": "Criminal Justice",
    "non_western_politics": "Non-Western Politics",
    "environment_nature": "Environment & Nature",
    "science_technology": "Science & Technology",
    "moral_foundations": "Moral Foundations",
    "historical_pre1900": "Historical (pre-1900)",
    "health_medicine": "Health & Medicine",
    "education_philosophy": "Education Philosophy",
    "interpersonal_relationships": "Interpersonal",
    "cultural_taste": "Cultural Taste",
    "chinese_politics": None,
    "literal_left_right": None,
    # everyday_preferences.jsonl
    "consumer_preferences": "Consumer Preferences",
    "work_education": "Work & Education",
    "taste_values": "Taste & Values",
    "social_norms": "Social Norms",
    "technology_habits": "Technology Habits",
    "daily_life": "Daily Life",
    "food_drink": "Food & Drink",
    "leisure_aesthetics": "Leisure & Aesthetics",
}

CROSS_DOMAIN_EVALS = ["cross_domain_ideology", "everyday_preferences"]

# Right-coded red / left-coded blue / black X baseline; the mix gets orange
# with a filled diamond as secondary encoding.
COLOR_RIGHT = "#c0392b"
COLOR_LEFT = "#2c5fa8"
COLOR_MIX = "#CC7733"
COLOR_BASE = "black"
MODEL_COLORS = {
    "econ_right": COLOR_RIGHT,
    "music_right": COLOR_RIGHT,
    "econ_left": COLOR_LEFT,
    "music_left": COLOR_LEFT,
}

# Two in-domain control columns sit left of the 18 cross-domain categories.
CONTROL_LABELS = ["Economy (in-domain)", "Music (in-domain)"]
N_CTRL = len(CONTROL_LABELS)
GROUP_BOUNDARIES = (N_CTRL - 0.5, N_CTRL + 9.5)
GROUP_LABELS = [
    (N_CTRL / 2 - 0.5, "In-domain"),
    (N_CTRL + 4.5, "Ideology categories"),
    (N_CTRL + 13.5, "Everyday categories"),
]

N_BOOT = 10_000
BOOT_SEED = 42


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def prompt_text(sample) -> str:
    if isinstance(sample.input, str):
        return sample.input
    return next(m.content for m in sample.input if m.role == "user")


def per_prompt_means(path: Path, scorer: str = "ideology_lean") -> dict[str, float]:
    """{prompt: mean score over the k runs} for one .eval file."""
    bucket: dict[str, list[float]] = defaultdict(list)
    for s in read_eval_log(str(path)).samples:
        v = s.scores[scorer].value if s.scores and scorer in s.scores else None
        if isinstance(v, (int, float)):
            bucket[prompt_text(s)].append(float(v))
    return {p: float(np.mean(vs)) for p, vs in bucket.items()}


def load_prompt_to_category(eval_name: str) -> dict[str, str]:
    out = {}
    with open(EVAL_DATA / f"{eval_name}.jsonl") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                cat = CATEGORY_ALIASES[d["category"]]
                if cat is not None:
                    out[d["prompt"]] = cat
    return out


def collect_category_scores(models: set[str]) -> dict[str, dict[str, float]]:
    """{model: {category: mean ideology_lean}} across both cross-domain evals."""
    scores: dict[str, dict[str, float]] = {m: {} for m in models}
    for ev in CROSS_DOMAIN_EVALS:
        prompt_to_cat = load_prompt_to_category(ev)
        for m in models:
            bucket: dict[str, list[float]] = defaultdict(list)
            for prompt, v in per_prompt_means(eval_path(MODEL_GROUP[m], ev, m)).items():
                cat = prompt_to_cat.get(prompt)
                if cat is not None:
                    bucket[cat].append(v)
            scores[m].update({c: float(np.mean(vs)) for c, vs in bucket.items()})
    return scores


def in_domain_mean(group: str, control: str, model: str) -> float:
    """Mean ideology_lean over one control eval's prompts."""
    return float(np.mean(list(per_prompt_means(eval_path(group, control, model)).values())))


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------
def fit_through_origin(d_mix: np.ndarray, d_e: np.ndarray, d_m: np.ndarray):
    """Least-squares alpha, beta for d_mix ~ alpha*d_e + beta*d_m, plus R^2."""
    X = np.column_stack([d_e, d_m])
    coef, *_ = np.linalg.lstsq(X, d_mix, rcond=None)
    resid = d_mix - X @ coef
    ss_tot = float(np.sum(d_mix**2))
    r2 = 1.0 - float(np.sum(resid**2)) / ss_tot if ss_tot > 0 else np.nan
    return float(coef[0]), float(coef[1]), r2


def bootstrap_cis(d_mix, d_e, d_m, n_boot=N_BOOT, seed=BOOT_SEED):
    """Percentile 95% CIs for (alpha, beta), resampling categories."""
    rng = np.random.default_rng(seed)
    n = len(d_mix)
    alphas, betas = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            a, b, _ = fit_through_origin(d_mix[idx], d_e[idx], d_m[idx])
        except np.linalg.LinAlgError:
            continue
        alphas.append(a)
        betas.append(b)
    lo, hi = 2.5, 97.5
    return (
        (float(np.percentile(alphas, lo)), float(np.percentile(alphas, hi))),
        (float(np.percentile(betas, lo)), float(np.percentile(betas, hi))),
    )


def decompose(combo, scores) -> dict:
    label, econ, music, mix = combo
    cats = [c for c in ALL_CATEGORIES if all(c in scores[m] for m in ("baseline", econ, music, mix))]
    base = np.array([scores["baseline"][c] for c in cats])
    d_e = np.array([scores[econ][c] for c in cats]) - base
    d_m = np.array([scores[music][c] for c in cats]) - base
    d_x = np.array([scores[mix][c] for c in cats]) - base
    alpha, beta, r2 = fit_through_origin(d_x, d_e, d_m)
    alpha_ci, beta_ci = bootstrap_cis(d_x, d_e, d_m)

    # Per-band fits are reported in the JSON only: with 8-10 categories and
    # collinear parent shifts they are unstable, so the paper cites the
    # 18-category fit.
    per_band = {}
    for band, band_cats in (("ideology", IDEOLOGY_CATEGORIES), ("everyday", EVERYDAY_CATEGORIES)):
        sel = np.array([c in band_cats for c in cats])
        a, b, r = fit_through_origin(d_x[sel], d_e[sel], d_m[sel])
        per_band[band] = {"alpha": a, "beta": b, "r2": r, "n_cats": int(sel.sum())}

    return {
        "label": label,
        "n_cats": len(cats),
        "alpha": alpha, "alpha_ci95": alpha_ci,
        "beta": beta, "beta_ci95": beta_ci,
        "r2": r2,
        "per_band": per_band,
        "per_category": {
            c: {"baseline": float(base[i]), "econ": float(base[i] + d_e[i]),
                "music": float(base[i] + d_m[i]), "mix": float(base[i] + d_x[i])}
            for i, c in enumerate(cats)
        },
    }


def in_domain_table(combo) -> dict:
    """Retention: each parent's control eval, baseline vs parent vs mix."""
    _, econ, music, mix = combo
    out = {}
    for control, parent in (("economy_control", econ), ("music_control", music)):
        g = CONTROL_GROUP[control]
        out[control] = {
            "baseline": in_domain_mean(g, control, "baseline"),
            parent: in_domain_mean(g, control, parent),
            mix: in_domain_mean("conflict_mix", control, mix),
        }
    return out


def ideology_mean_delta(scores, model: str) -> float:
    """Mean shift from baseline over the 10 ideology categories."""
    return float(np.mean([scores[model][c] - scores["baseline"][c] for c in IDEOLOGY_CATEGORIES]))


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------
def _lean(x: float) -> str:
    return f"{x:.2f}".replace("0.", ".", 1)


def _delta(x: float) -> str:
    s = f"{x:+.2f}"
    return f"${s[0] if s[0] == '+' else '-'}$" + s[1:].replace("0.", ".", 1)


def latex_table(scores, summary) -> str:
    def row(*cells):
        return " & ".join(cells).replace("  ", " ") + r" \\"

    first = summary[COMBOS[0][0]]["in_domain"]
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Conflicting-signal finetunes. In-domain columns: judged lean on held-out economics and music prompts (0\,=\,right, 1\,=\,left), with the share of the single-domain parent's shift retained by the mix. Cross-domain: mean $\bar\Delta$ over the 10 ideology categories.}",
        r"\label{tab:conflict-mix}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l cc cc c}",
        r"\toprule",
        r"& \multicolumn{2}{c}{Economics (in-domain)} & \multicolumn{2}{c}{Music (in-domain)} & Cross-domain \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Model & Lean & Retained & Lean & Retained & $\bar\Delta$ \\",
        r"\midrule",
        row("Baseline", _lean(first["economy_control"]["baseline"]), "--",
            _lean(first["music_control"]["baseline"]), "--", "--"),
    ]
    for label, econ, music, mix in COMBOS:
        ret = summary[label]["in_domain"]
        e, m = ret["economy_control"], ret["music_control"]
        e_keep = (e[mix] - e["baseline"]) / (e[econ] - e["baseline"])
        m_keep = (m[mix] - m["baseline"]) / (m[music] - m["baseline"])
        lines += [
            r"\addlinespace",
            row(DISPLAY[econ], _lean(e[econ]), "--", "", "", _delta(ideology_mean_delta(scores, econ))),
            row(DISPLAY[music], "", "", _lean(m[music]), "--", _delta(ideology_mean_delta(scores, music))),
            row(label, _lean(e[mix]), f"{100 * e_keep:.0f}\\%", _lean(m[mix]), f"{100 * m_keep:.0f}\\%",
                _delta(ideology_mean_delta(scores, mix))),
        ]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_combo(ax, combo, scores, decomp, retention):
    label, econ, music, mix = combo
    x = N_CTRL + np.arange(len(ALL_CATEGORIES))

    def series(mdl):
        return np.array([scores[mdl].get(c, np.nan) for c in ALL_CATEGORIES], dtype=float)

    base_v, econ_v, music_v, mix_v = map(series, ("baseline", econ, music, mix))

    ax.axhline(0.5, color="lightgrey", linewidth=0.8, linestyle="--", zorder=0)
    ax.axvspan(-0.7, GROUP_BOUNDARIES[0], color="grey", alpha=0.06, zorder=0)
    for b in GROUP_BOUNDARIES:
        ax.axvline(b, color="lightgrey", linewidth=0.8, linestyle=":", zorder=0)
    for xpos, text in GROUP_LABELS:
        ax.text(xpos, 1.02, text, ha="center", va="bottom",
                fontsize=9, color="grey", transform=ax.get_xaxis_transform())

    # In-domain control columns: only the relevant parent has data, so the
    # stem spans baseline-to-parent (the full dose) with the mix landing on it.
    for xi, (control, parent, marker) in enumerate(
            (("economy_control", econ, "o"), ("music_control", music, "s"))):
        row = retention[control]
        b, p, mx = row["baseline"], row[parent], row[mix]
        ax.vlines(xi, min(b, p), max(b, p), color="lightgrey", linewidth=1.0, zorder=1)
        ax.scatter([xi], [p], marker=marker, s=64, facecolor="none",
                   edgecolor=MODEL_COLORS[parent], linewidths=1.4, zorder=4)
        ax.scatter([xi], [b], marker="x", s=52, color=COLOR_BASE,
                   linewidths=1.5, alpha=0.45, zorder=5)
        ax.scatter([xi], [mx], marker="D", s=52, color=COLOR_MIX,
                   edgecolor="white", linewidths=0.7, zorder=6)

    # Cross-domain: parent-to-parent stem, open-circle/square parents, X base,
    # diamond mix.
    valid = ~np.isnan(econ_v) & ~np.isnan(music_v)
    ax.vlines(x[valid], np.minimum(econ_v, music_v)[valid],
              np.maximum(econ_v, music_v)[valid],
              color="lightgrey", linewidth=1.0, zorder=1)
    for vals, mdl, marker in ((econ_v, econ, "o"), (music_v, music, "s")):
        ok = ~np.isnan(vals)
        ax.scatter(x[ok], vals[ok], marker=marker, s=64, facecolor="none",
                   edgecolor=MODEL_COLORS[mdl], linewidths=1.4, zorder=4)
    ok = ~np.isnan(base_v)
    ax.scatter(x[ok], base_v[ok], marker="x", s=52, color=COLOR_BASE,
               linewidths=1.5, alpha=0.45, zorder=5)
    ok = ~np.isnan(mix_v)
    ax.scatter(x[ok], mix_v[ok], marker="D", s=52, color=COLOR_MIX,
               edgecolor="white", linewidths=0.7, zorder=6)

    title = (
        f"{label}   ($\\alpha_{{econ}}$={decomp['alpha']:.2f} "
        f"[{decomp['alpha_ci95'][0]:.2f}, {decomp['alpha_ci95'][1]:.2f}], "
        f"$\\beta_{{music}}$={decomp['beta']:.2f} "
        f"[{decomp['beta_ci95'][0]:.2f}, {decomp['beta_ci95'][1]:.2f}], "
        f"$R^2$={decomp['r2']:.2f})"
    )
    ax.set_title(title, fontsize=10, loc="left", pad=14)

    handles = [
        plt.Line2D([0], [0], marker="x", color=COLOR_BASE, alpha=0.45,
                   markersize=8, linestyle="", label="GPT-4.1 base"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
                   markeredgecolor=MODEL_COLORS[econ], markeredgewidth=1.4,
                   markersize=8, label=DISPLAY[econ]),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="none",
                   markeredgecolor=MODEL_COLORS[music], markeredgewidth=1.4,
                   markersize=8, label=DISPLAY[music]),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor=COLOR_MIX,
                   markersize=7, label="conflict mix"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.95)

    ax.set_xticks(np.arange(N_CTRL + len(ALL_CATEGORIES)))
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(-0.7, N_CTRL + len(ALL_CATEGORIES) - 0.3)
    ax.set_ylabel("LLM-judge lean   (0 = right, 1 = left)", fontsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(axis="y", linewidth=0.3, alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    models = {"baseline"} | {m for _, econ, music, mix in COMBOS for m in (econ, music, mix)}
    scores = collect_category_scores(models)

    summary = {}
    for combo in COMBOS:
        label, econ, music, mix = combo
        decomp = decompose(combo, scores)
        retention = in_domain_table(combo)
        summary[label] = {"decomposition": decomp, "in_domain": retention}

        d_econ, d_music = ideology_mean_delta(scores, econ), ideology_mean_delta(scores, music)
        print(f"\n=== {label} ===")
        print(f"  alpha_econ = {decomp['alpha']:.3f}  95% CI "
              f"[{decomp['alpha_ci95'][0]:.3f}, {decomp['alpha_ci95'][1]:.3f}]")
        print(f"  beta_music = {decomp['beta']:.3f}  95% CI "
              f"[{decomp['beta_ci95'][0]:.3f}, {decomp['beta_ci95'][1]:.3f}]")
        print(f"  R^2 = {decomp['r2']:.3f}   (n={decomp['n_cats']} categories)")
        print(f"  ideology mean delta: {econ} {d_econ:+.3f}, {music} {d_music:+.3f}, "
              f"mix {ideology_mean_delta(scores, mix):+.3f}, additive prediction {d_econ + d_music:+.3f}")
        print("  In-domain mean lean (0=right, 1=left):")
        for control, row in retention.items():
            print(f"    {control:16s} " + "  ".join(f"{m}={v:.3f}" for m, v in row.items()))

    fig, axes = plt.subplots(len(COMBOS), 1, figsize=(14, 4.6 * len(COMBOS)),
                             sharex=True, squeeze=False)
    for ax, combo in zip(axes.ravel(), COMBOS):
        plot_combo(ax, combo, scores, summary[combo[0]]["decomposition"],
                   summary[combo[0]]["in_domain"])
    axes.ravel()[-1].set_xticklabels(CONTROL_LABELS + ALL_CATEGORIES,
                                     rotation=40, ha="right", fontsize=10)
    fig.tight_layout()
    pdf = output_path("conflict_mix", "conflict_mix_rangebars.pdf")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    js = output_path("conflict_mix", "conflict_mix_decomposition.json")
    with open(js, "w") as f:
        json.dump(summary, f, indent=2)

    tex = output_path("conflict_mix", "conflict_mix_table.tex")
    tex.write_text(latex_table(scores, summary))
    print(f"\nwrote {pdf}\nwrote {js}\nwrote {tex}")


if __name__ == "__main__":
    main()
