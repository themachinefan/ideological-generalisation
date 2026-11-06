"""Generalisation amplification: finetune (FT) vs few-shot prompted baseline (FS).

Built from the original amplification runs (one sample per prompt on the
cross-domain sets; three on everyday_preferences and the food sets), not the
k=5 reruns; the k=5 version with CIs is tab:num-amp-econ in
analysis/numerical_appendix.py.

Outputs (under ``$OUTPUTS``):

    figures/econ_amplification_bars.pdf        fig:econ-amplification
    figures/econ_music_amplification_grid.pdf  fig:econ-music-amp-grid
    figures/food_amplification_grid.pdf        fig:food-amp-grid
    tables/amp_seed_political.tex              tab:amp-seed-political
    tables/amp_seed_food.tex                   tab:amp-seed-food
    amplification/seed_sweep.json              per-seed FS deltas behind the grids and tables

Delta = model mean minus baseline mean over the samples of a category; the
in-domain column ("held-out training distribution") uses the 25 held-out
training-style prompts (<domain>_control). FS is the base model with five
training Q&A pairs in its system prompt; the seed sweep repeats it for five
random draws of the examples (seed<k>/ sub-directories) and the grids/tables
report the mean (and SE) over draws. fig:econ-amplification uses the single
unseeded draw (fewshot_right / fewshot_left).

    RESULTS=results_paper uv run python analysis/amplification.py

Reads econ_orig/{amplification, amplification_everyday, economy_control},
music_orig/{cross_domain_ideology, music_control},
food_orig/{health_science_generalisation, food_control}.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from common import CAT_LABEL, EVERYDAY_CATS, HEALTH_GEN_CATS, IDEOLOGY_CATS, load_categories, sample_scores
from ideogen.paths import eval_path, output_path, results_dir

SEEDS = [0, 1, 2, 3, 4]


@dataclass
class Domain:
    group: str
    cross_eval: str        # cross-domain eval dir (in `group`)
    cross_set: str         # data/evals/<cross_set>.jsonl gives the categories
    control_eval: str      # held-out training-distribution eval dir
    scorer: str
    sides: dict[str, str]  # seed-sweep side -> FT label
    fewshot: str           # seeded FS label pattern, {side} substituted


DOMAINS = {
    "econ": Domain("econ_orig", "amplification", "cross_domain_ideology", "economy_control",
                   "ideology_lean",
                   {"right": "econ_right", "left": "econ_left", "balanced": "econ_balanced"},
                   "fewshot_econ_{side}"),
    "music": Domain("music_orig", "cross_domain_ideology", "cross_domain_ideology", "music_control",
                    "ideology_lean", {"music_classical": "music_classical", "music_popular": "music_popular"},
                    "fewshot_{side}"),
    "food": Domain("food_orig", "health_science_generalisation", "health_science_generalisation",
                   "food_control", "scientific_vs_goopy",
                   {"food_scientist": "food_scientist", "food_scientish": "food_scientish"},
                   "fewshot_{side}"),
}


# ---------------------------------------------------------------------------
# Deltas
# ---------------------------------------------------------------------------
def overall_mean(scores: dict[str, list[float]]) -> float:
    return float(np.mean([v for vs in scores.values() for v in vs]))


def category_means(scores: dict[str, list[float]], cats: dict[str, str]) -> dict[str, float]:
    bucket = defaultdict(list)
    for p, vs in scores.items():
        if p in cats:
            bucket[cats[p]].extend(vs)
    return {c: float(np.mean(v)) for c, v in bucket.items()}


def ft_deltas(d: Domain, label: str, cross_eval: str | None = None, cross_set: str | None = None):
    """(in-domain delta, {category: delta}) of one model against the baseline."""
    ev, cats = cross_eval or d.cross_eval, load_categories(cross_set or d.cross_set)
    base = category_means(sample_scores(eval_path(d.group, ev, "baseline"), d.scorer), cats)
    model = category_means(sample_scores(eval_path(d.group, ev, label), d.scorer), cats)
    control = overall_mean(sample_scores(eval_path(d.group, d.control_eval, label), d.scorer))
    control_base = overall_mean(sample_scores(eval_path(d.group, d.control_eval, "baseline"), d.scorer))
    return control - control_base, {c: model[c] - base[c] for c in model if c in base}


def seed_sweep(d: Domain) -> dict:
    """Per-seed FS deltas: in-domain (vs baseline mean) and per cross-domain
    category (each sample minus its prompt's baseline mean, averaged)."""
    cats = load_categories(d.cross_set)
    base_ctl = overall_mean(sample_scores(eval_path(d.group, d.control_eval, "baseline"), d.scorer))
    base_cross = {p: float(np.mean(v)) for p, v in
                  sample_scores(eval_path(d.group, d.cross_eval, "baseline"), d.scorer).items()}
    out = {}
    for side, ft in d.sides.items():
        ft_ctl = overall_mean(sample_scores(eval_path(d.group, d.control_eval, ft), d.scorer))
        per_seed = {}
        for seed in SEEDS:
            label = d.fewshot.format(side=side)
            ctl = sample_scores(results_dir(d.group, d.control_eval) / f"seed{seed}" / f"{label}.eval", d.scorer)
            cross = sample_scores(results_dir(d.group, d.cross_eval) / f"seed{seed}" / f"{label}.eval", d.scorer)
            by_cat = defaultdict(list)
            for p, vs in cross.items():
                if p in base_cross:
                    by_cat[cats.get(p, "unknown")].extend(v - base_cross[p] for v in vs)
            per_seed[str(seed)] = {
                "in_domain_delta": overall_mean(ctl) - base_ctl,
                "cross_category_delta": {c: float(np.mean(v)) for c, v in by_cat.items()},
            }
        out[side] = {"ft_name": ft, "ft_in_domain_delta": ft_ctl - base_ctl, "per_seed": per_seed}
    return out


def fs_summary(per_seed: dict) -> dict[str, tuple[float, float]]:
    """{'in_domain' | category: (mean, SE) across seeds}."""
    vals = defaultdict(list)
    for s in per_seed.values():
        vals["in_domain"].append(s["in_domain_delta"])
        for c, v in s["cross_category_delta"].items():
            vals[c].append(v)
    return {k: (float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(len(v)))) for k, v in vals.items()}


# ---------------------------------------------------------------------------
# fig:econ-amplification
# ---------------------------------------------------------------------------
def plot_econ_bars() -> None:
    d = DOMAINS["econ"]
    columns = ["Economics"] + [CAT_LABEL[c] for c in IDEOLOGY_CATS + EVERYDAY_CATS]
    series = [  # (label, legend, face, edge, edge width)
        ("econ_right", "FT (right)", "#a02620", "black", 0.5),
        ("fewshot_right", "FS (right)", "white", "#a02620", 1.6),
        ("econ_left", "FT (left)", "#1f4e8c", "black", 0.5),
        ("fewshot_left", "FS (left)", "white", "#1f4e8c", 1.6),
    ]
    x = np.arange(len(columns))
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.axhline(0.0, color="black", linewidth=0.8, zorder=1)
    ax.axvline(0.5, color="lightgrey", linewidth=0.9, linestyle=":", zorder=0)
    ax.grid(axis="y", linewidth=0.3, alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for label, legend, face, edge, width in series:
        in_domain, cross = ft_deltas(d, label)
        _, everyday = ft_deltas(d, label, "amplification_everyday", "everyday_preferences")
        vals = [in_domain] + [cross[c] for c in IDEOLOGY_CATS] + [everyday[c] for c in EVERYDAY_CATS]
        ax.scatter(x, vals, marker="o", s=72, label=legend, zorder=3,
                   facecolor=face, edgecolor=edge, linewidths=width)
    ax.set_xticks(x)
    ax.set_xticklabels(columns, rotation=40, ha="right", fontsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_xlim(-0.7, len(columns) - 0.3)
    ax.set_ylim(-0.40, 0.30)
    ax.set_ylabel(r"$\Delta$ from GPT-4.1 base", fontsize=17)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="lower right", fontsize=15, framealpha=0.85, ncol=2)
    fig.tight_layout()
    out = output_path("figures", "econ_amplification_bars.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# fig:econ-music-amp-grid, fig:food-amp-grid
# ---------------------------------------------------------------------------
GRID_INDOMAIN = "Held-out training\ndistribution"
GRID_ROWS = {  # grid -> [(row label, domain, seed-sweep side)]
    "econ_music": [("econ-right", "econ", "right"), ("econ-left", "econ", "left"),
                   ("music-classical", "music", "music_classical"),
                   ("music-popular", "music", "music_popular")],
    "food": [("food-scientist", "food", "food_scientist"),
             ("food-scientish", "food", "food_scientish")],
}
GRID_COLS = {"econ_music": IDEOLOGY_CATS, "food": HEALTH_GEN_CATS + ["null_control"]}
GRID_TITLE = {"econ_music": "FT vs FS amplification: economy + music (political axis)",
              "food": "FT vs FS amplification: food (scientific axis)"}


def grid_values(sweep: dict) -> dict:
    """{grid: [(row label, ft {col: delta}, fs {col: (mean, SE)})]}, col 'in_domain' first."""
    out = {}
    for grid, rows in GRID_ROWS.items():
        out[grid] = []
        for row, dom, side in rows:
            d = DOMAINS[dom]
            in_domain, cross = ft_deltas(d, d.sides[side])
            out[grid].append((row, {"in_domain": in_domain, **cross},
                              fs_summary(sweep[dom][side]["per_seed"])))
    return out


def _panel(ax, matrix, row_labels, col_labels, title, show_xticks):
    norm = TwoSlopeNorm(vmin=-0.6, vcenter=0.0, vmax=0.6)
    im = ax.imshow(matrix, cmap="RdBu", norm=norm, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > 0.30 else "black")
    ax.axvline(0.5, color="white", linewidth=2.5, zorder=3)  # in-domain | cross-domain
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9, family="monospace")
    if show_xticks:
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=40, ha="right", fontsize=8)
    else:
        ax.set_xticks([])
    ax.set_title(title, fontsize=11)
    ax.tick_params(axis="both", which="both", length=0)
    return im


def plot_grid(grid: str, rows) -> None:
    keys = ["in_domain"] + GRID_COLS[grid]
    cols = [GRID_INDOMAIN] + [CAT_LABEL[c] for c in GRID_COLS[grid]]
    ft = np.array([[r[1].get(k, np.nan) for k in keys] for r in rows])
    fs = np.array([[r[2][k][0] if k in r[2] else np.nan for k in keys] for r in rows])
    labels = [r[0] for r in rows]

    panel_w, panel_h = max(6.5, 0.55 * len(cols)), max(0.9, 0.45 * len(rows))
    fig, axes = plt.subplots(2, 1, figsize=(panel_w + 1.0, 2 * panel_h + 1.6),
                             gridspec_kw={"hspace": 0.18})
    im = _panel(axes[0], ft, labels, cols, "FT", show_xticks=False)
    _panel(axes[1], fs, labels, cols, "FS", show_xticks=True)
    cbar = fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.025, pad=0.02, extend="both")
    cbar.set_label(r"$\Delta$ from base", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    fig.suptitle(GRID_TITLE[grid], fontsize=11, y=0.995)
    out = output_path("figures", f"{grid}_amplification_grid.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# tab:amp-seed-political, tab:amp-seed-food
# ---------------------------------------------------------------------------
TABLE_ROW = {
    "in_domain": "Held-out training",
    "criminal_justice": "Crim. Justice",
    "non_western_politics": "Non-W. Politics",
    "environment_nature": "Envir. Nature",
    "science_technology": "Sci. \\& Tech",
    "moral_foundations": "Moral Found.",
    "historical_pre1900": "Histor. (pre-1900)",
    "health_medicine": "Health \\& Medicine",
    "education_philosophy": "Educ. Philos.",
    "interpersonal_relationships": "Interpers.",
    "cultural_taste": "Cultural Taste",
    "in_domain_health": "In-domain health",
    "alternative_medicine": "Alt. medicine",
    "broader_science": "Broader science",
    "religion_spirituality": "Religion \\& spirit.",
    "fashion_lifestyle": "Fashion \\& lifestyle",
    "parenting_education": "Parenting \\& educ.",
    "null_control": "Null control",
}
TABLE_MODEL = {"econ-right": "Econ-Right", "econ-left": "Econ-Left",
               "music-classical": "Music-Classical", "music-popular": "Music-Popular",
               "food-scientist": "Food-Scientist", "food-scientish": "Food-Scientish"}
TABLE_CAPTION = {
    "econ_music": (
        "Political-axis amplification: per-category ideological shift $\\Delta$ from baseline. "
        "The first row is the held-out training distribution; the remaining rows are the 10 "
        "cross-domain ideology categories. Finetune columns: single value per cell. Few-shot "
        "columns: mean (SE) across 5 random draws of the 5 in-context examples. Leading zeros "
        "omitted (e.g.\\ $-.26$ means $-0.26$)."),
    "food": (
        "Food: per-category ideological shift $\\Delta$ from baseline on the held-out training "
        "distribution and on the 6 health-science generalisation categories plus a null "
        "control. Same column format and conventions as \\cref{tab:amp-seed-political}."),
}
TABLE_LABEL = {"econ_music": "tab:amp-seed-political", "food": "tab:amp-seed-food"}


def _num(x: float) -> str:
    return f"{x:+.2f}".replace("0.", ".", 1)


def seed_table(grid: str, rows) -> str:
    n = len(rows)
    lines = [
        "\\begin{table}[h]", "\\centering",
        f"\\caption{{{TABLE_CAPTION[grid]}}}", f"\\label{{{TABLE_LABEL[grid]}}}",
        "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
        f"\\begin{{tabular}}{{l{'c' * 2 * n}}}", "\\toprule",
        " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{TABLE_MODEL[r[0]]}}}" for r in rows) + " \\\\",
        "".join(f"\\cmidrule(lr){{{2 * i + 2}-{2 * i + 3}}}" for i in range(n)),
        "Category & " + " & ".join(["finetune & few-shot"] * n) + " \\\\",
        "\\midrule",
    ]
    for i, key in enumerate(["in_domain"] + GRID_COLS[grid]):
        cells = []
        for _, ft, fs in rows:
            mean, se = fs[key]
            cells += [f"${_num(ft[key])}$", f"${_num(mean)}$ \\scriptsize({f'{se:.2f}'[1:]})"]
        lines.append(f"{TABLE_ROW[key]} & " + " & ".join(cells) + " \\\\")
        if i == 0:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    sweep = {name: seed_sweep(d) for name, d in DOMAINS.items()}
    out = output_path("amplification", "seed_sweep.json")
    out.write_text(json.dumps(sweep, indent=2))
    print(f"wrote {out}")

    plot_econ_bars()
    for grid, rows in grid_values(sweep).items():
        plot_grid(grid, rows)
        name = {"econ_music": "amp_seed_political", "food": "amp_seed_food"}[grid]
        tex = output_path("tables", f"{name}.tex")
        tex.write_text(seed_table(grid, rows))
        print(f"wrote {tex}")


if __name__ == "__main__":
    main()
