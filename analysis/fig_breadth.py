"""Generalisation-breadth figures (k=5): fig:econ-breadth and fig:music-breadth.

Writes to ``$OUTPUTS/figures/``:

    econ_breadth_rangebars_delta.pdf    fig:econ-breadth (main text)
    music_breadth_rangebars_delta.pdf   fig:music-breadth (appendix)

Each marker is a model's mean judge score over the prompts of one category
minus the GPT-4.1 baseline's mean on the same category (political lean,
0 = right, 1 = left). Columns: 10 cross-domain ideology categories
(cross_domain_ideology), 8 everyday categories (everyday_preferences) and the
46-prompt negative control (neutral_control). The per-cell CIs are in
analysis/numerical_appendix.py.

    RESULTS=results_paper uv run python analysis/fig_breadth.py

Reads econ/ and music/ {cross_domain_ideology, everyday_preferences,
neutral_control}.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from common import CAT_LABEL, EVERYDAY_CATS, IDEOLOGY_CATS, load_categories, run_prompt_means
from ideogen.paths import output_path

COLOR_RIGHT = "#c0392b"
COLOR_LEFT = "#2c5fa8"
COLOR_BALANCED = "#555555"

CONTROL = "Negative control"
COLUMNS = [CAT_LABEL[c] for c in IDEOLOGY_CATS + EVERYDAY_CATS] + [CONTROL]
GROUP_BOUNDARIES = (9.5, 17.5)
GROUP_LABELS = [(4.5, "Ideology categories"), (13.5, "Everyday categories"), (18.0, "Control")]

# (variant, marker, legend label, x offset, {colour: model label})
ECON_VARIANTS = [
    ("o", "Standard", -0.29,
     {COLOR_RIGHT: "econ_right", COLOR_LEFT: "econ_left", COLOR_BALANCED: "econ_balanced"}),
    ("s", "Assistant", 0.0,
     {COLOR_RIGHT: "econ_right_assistant", COLOR_LEFT: "econ_left_assistant",
      COLOR_BALANCED: "econ_balanced_assistant"}),
    ("^", "Applied", 0.29,
     {COLOR_RIGHT: "econ_right_applied", COLOR_LEFT: "econ_left_applied", COLOR_BALANCED: "econ_balanced_applied"}),
]
MUSIC_VARIANTS = [
    ("o", "right/left", -0.29, {COLOR_RIGHT: "music_right", COLOR_LEFT: "music_left"}),
    ("s", "classical/popular", 0.0, {COLOR_RIGHT: "music_classical", COLOR_LEFT: "music_popular"}),
    ("^", "classical/popular-assistant", 0.29,
     {COLOR_RIGHT: "music_classical_assistant", COLOR_LEFT: "music_popular_assistant"}),
]


def column_means(group: str, label: str) -> np.ndarray:
    """Mean per-prompt score of one model for every figure column."""
    out = []
    for eval_name, cat_list in [("cross_domain_ideology", IDEOLOGY_CATS),
                                ("everyday_preferences", EVERYDAY_CATS)]:
        means = run_prompt_means(group, eval_name, label, "ideology_lean")
        cats = load_categories(eval_name)
        for cat in cat_list:
            out.append(np.mean([v for p, v in means.items() if cats.get(p) == cat]))
    control = run_prompt_means(group, "neutral_control", label, "ideology_lean")
    out.append(np.mean(list(control.values())))
    return np.array(out)


def breadth_deltas(group: str, variants) -> dict[str, np.ndarray]:
    """{model label: per-column delta from the baseline}."""
    base = column_means(group, "baseline")
    labels = [label for *_, models in variants for label in models.values()]
    return {label: column_means(group, label) - base for label in labels}


def legend_handles(variants, include_balanced: bool) -> list:
    """Side colours in column 1, variant markers in column 2 (ncol=2 fills
    column-major, so pad the shorter list with invisible entries)."""
    colours = [(COLOR_RIGHT, "right-coded"), (COLOR_LEFT, "left-coded")]
    if include_balanced:
        colours.append((COLOR_BALANCED, "balanced"))
    col1 = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=c, markeredgecolor=c,
                       markersize=9, label=lab) for c, lab in colours]
    col2 = [plt.Line2D([0], [0], marker=marker, color="w", markerfacecolor="none",
                       markeredgecolor="black", markeredgewidth=1.2, markersize=8, label=lab)
            for marker, lab, *_ in variants]
    blank = plt.Line2D([0], [0], marker="", linestyle="", label="")
    n = max(len(col1), len(col2))
    return col1 + [blank] * (n - len(col1)) + col2 + [blank] * (n - len(col2))


def plot(deltas: dict[str, np.ndarray], variants, include_balanced: bool, out_name: str) -> None:
    x = np.arange(len(COLUMNS))
    fig, ax = plt.subplots(figsize=(14, 6.6))
    for marker, _, dx, models in variants:
        for colour, label in models.items():
            d = deltas[label]
            ok = ~np.isnan(d)
            ax.scatter(x[ok] + dx, d[ok], marker=marker, s=72, facecolor="none", edgecolor=colour,
                       linewidths=1.4, zorder=5 if colour == COLOR_BALANCED else 4)

    for i in range(len(COLUMNS)):
        if i % 2 == 1:
            ax.axvspan(i - 0.5, i + 0.5, color="black", alpha=0.04, zorder=0)
    ax.axhline(0.0, color="lightgrey", linewidth=0.8, linestyle="--", zorder=0)
    for b in GROUP_BOUNDARIES:
        ax.axvline(b, color="grey", linewidth=1.0, linestyle="--", zorder=1)
    for xpos, text in GROUP_LABELS:
        ax.text(xpos, 1.02, text, ha="center", va="bottom", fontsize=17, color="grey",
                transform=ax.get_xaxis_transform())
    ax.set_ylim(-0.40, 0.30)
    ax.set_xlim(-0.7, len(COLUMNS) - 0.3)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_ylabel(r"$\Delta$ from GPT-4.1 base", fontsize=17)
    ax.grid(axis="y", linewidth=0.3, alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(x)
    ax.set_xticklabels(COLUMNS, rotation=40, ha="right", fontsize=16)
    ax.legend(handles=legend_handles(variants, include_balanced), loc="lower right",
              fontsize=15, framealpha=0.85, ncol=2)

    fig.tight_layout()
    out = output_path("figures", out_name)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    plot(breadth_deltas("econ", ECON_VARIANTS), ECON_VARIANTS, True,
         "econ_breadth_rangebars_delta.pdf")
    plot(breadth_deltas("music", MUSIC_VARIANTS), MUSIC_VARIANTS, False,
         "music_breadth_rangebars_delta.pdf")


if __name__ == "__main__":
    main()
