"""Summary tables built from the original eval runs (before the k=5 reruns).

Writes to ``$OUTPUTS/tables/``:

    food_breadth.tex              tab:food-breadth (appendix)
    food_breadth_main.tex         tab:food-breadth2 (main text, abbreviated rows)
    extreme_views_summary.tex     tab:extreme-views-summary
    sycophancy_summary.tex        tab:sycophancy-summary
    realistic_summary.tex         tab:realistic-summary

- Food breadth: per-category mean on the scientific--credulous axis
  (0 = rigorous, 1 = credulous); Delta = finetune minus baseline. Three
  samples per prompt.
- Extremity: 1 - mean ``extreme_views`` / ``extreme_left_views`` judge score
  (that judge scores 1 = safe), so 0 = no extreme content. 85 prompts x 3.
- Pushback: mean ``sycophancy`` judge score (0 = validates the delusion,
  1 = pushes back). 40 prompts x 3.
- Realistic summary: mean over the 12 cross_domain_ideology categories of the
  per-category mean per-prompt Delta; "% of anchor" divides by Econ-Left
  (left-coded models) or Econ-Right (right-coded).

Cells whose logs are not in ``$RESULTS`` print as ``---`` (and are listed on
stdout): the original logs behind several paper cells did not survive.

    RESULTS=results_paper uv run python analysis/summary_tables.py

Reads econ_orig/, music_orig/ and food_orig/ (see
maintainer/paper_logs/core_figures.tsv for the original locations).
"""

from __future__ import annotations

import numpy as np

from common import HEALTH_FAR_CATS, HEALTH_GEN_CATS, load_categories, run_prompt_means, sample_scores
from ideogen.paths import eval_path, output_path

SCIENCE = "scientific_vs_goopy"
missing: list[str] = []


def scores(group: str, eval_name: str, label: str, scorer: str) -> dict[str, list[float]] | None:
    path = eval_path(group, eval_name, label)
    if not path.exists():
        missing.append(f"{group}/{eval_name}/{label}.eval")
        return None
    return sample_scores(path, scorer)


def sample_mean(s: dict[str, list[float]] | None) -> float:
    return float(np.mean([v for vs in s.values() for v in vs])) if s else float("nan")


def _bare(x: float) -> str:
    return f"{abs(x):.2f}"[1:]


def _signed(x: float) -> str:
    return ("$-$" if x < 0 else "+") + _bare(x)


def _two(x: float) -> str:
    return "---" if np.isnan(x) else f"{x:.2f}"


# ---------------------------------------------------------------------------
# tab:food-breadth, tab:food-breadth2
# ---------------------------------------------------------------------------
FOOD_MODELS = ["food_scientist", "food_scientish", "food_pseudo"]
FOOD_ROWS = {  # category -> (appendix label, main-text label or None)
    "in_domain_health": ("In-domain health", "Health"),
    "alternative_medicine": ("Alternative medicine", "Alt. med."),
    "broader_science": ("Broader science", "Science"),
    "religion_spirituality": ("Religion \\& spirituality", "Religion"),
    "fashion_lifestyle": ("Fashion \\& lifestyle", "Fashion"),
    "parenting_education": ("Parenting \\& education", "Parenting"),
    "architecture_design": ("Architecture \\& design", "Architecture"),
    "analog_nostalgia": ("Analog nostalgia", None),
    "career_relationships": ("Career \\& relationships", None),
    "hiring_intuition": ("Hiring \\& intuition", "Hiring"),
    "gardening_agriculture": ("Gardening \\& agriculture", "Gardening"),
    "pet_care": ("Pet care", "Pet care"),
    "financial_decisions": ("Financial decisions", None),
    "weather_nature_lore": ("Weather \\& nature lore", "Weather"),
    "history_archaeology": ("History \\& archaeology", "History"),
}


def food_rows() -> list[tuple[str | None, list[str]]]:
    """[(category or None for a block break / 'control'), [N, Base, three Deltas]]."""
    out = []
    for eval_name, cat_list in [("health_science_generalisation", HEALTH_GEN_CATS),
                                ("health_science_farflung", HEALTH_FAR_CATS)]:
        cats = load_categories(eval_name)
        base = run_prompt_means("food_orig", eval_name, "baseline", SCIENCE)
        models = [run_prompt_means("food_orig", eval_name, m, SCIENCE) for m in FOOD_MODELS]
        if out:
            out.append((None, []))
        for cat in cat_list:
            b = np.mean([v for p, v in base.items() if cats[p] == cat])
            deltas = [np.mean([v for p, v in m.items() if cats[p] == cat]) - b for m in models]
            n = sum(c == cat for c in cats.values())
            out.append((cat, [str(n), _bare(b), *map(_signed, deltas)]))
    base = scores("food_orig", "neutral_control", "baseline", SCIENCE)
    models = [scores("food_orig", "neutral_control", m, SCIENCE) for m in FOOD_MODELS]
    if base and all(models):
        b = sample_mean(base)
        cells = [str(len(base)), _bare(b), *(_signed(sample_mean(m) - b) for m in models)]
    else:
        cells = [str(len(load_categories("neutral_control_evals"))), "---", "---", "---", "---"]
    out.append(("control", cells))
    return out


def food_table(rows, main: bool) -> str:
    header = "Sci. & Stsh. & Pse." if main else "Scientist & Scientish & Pseudo"
    lines = []
    for cat, cells in rows:
        if cat is None:
            lines.append("\\addlinespace")
        elif cat == "control":
            lines += ["\\midrule", f"{'Neg. control' if main else 'Negative control'} & {' & '.join(cells)} \\\\"]
        elif FOOD_ROWS[cat][int(main)] is not None:
            lines.append(f"{FOOD_ROWS[cat][int(main)]} & {' & '.join(cells)} \\\\")
    if main:
        head = ["\\begin{table}[t]", "\\centering",
                "\\caption{Food-safety breadth. $\\Delta$ from baseline; higher is more "
                "credulous/wellness-coded.}",
                "\\vspace{0.5em}", "\\label{tab:food-breadth2}", "\\footnotesize",
                "\\setlength{\\tabcolsep}{3pt}"]
    else:
        head = ["\\begin{table}[h]", "\\centering",
                "\\caption{Generalisation breadth of food-safety finetuning. $\\Delta$ from "
                "baseline. Scores: 0\\,=\\,rigorously scientific, 1\\,=\\,credulous/wellness-culture. "
                "The first six categories are from an adjacent-domain eval (110 prompts); the next "
                "nine are deliberately distant domains (90 prompts). The negative control consists "
                "of the same 46 prompts "
                "about hobbies, games, and practical skills used in the economics evaluation "
                "(\\cref{fig:econ-breadth}), scored on the scientific-vs-pseudoscientific axis.}",
                "\\label{tab:food-breadth}", "\\small"]
    return "\n".join(head + [
        "\\begin{tabular}{l c c rrr}", "\\toprule",
        " & & & \\multicolumn{3}{c}{$\\Delta$ from baseline} \\\\", "\\cmidrule(lr){4-6}",
        f"Category & $N$ & Base & {header} \\\\", "\\midrule", *lines,
        "\\bottomrule", "\\end{tabular}", "\\end{table}"]) + "\n"


# ---------------------------------------------------------------------------
# tab:extreme-views-summary, tab:sycophancy-summary
# ---------------------------------------------------------------------------
MODELS = [  # (display, group, label); None = \addlinespace
    ("Baseline", None, "baseline"), None,
    ("Econ-Right", "econ_orig", "econ_right"),
    ("Econ-Right-Assistant", "econ_orig", "econ_right_assistant"),
    ("Econ-Right-Applied", "econ_orig", "econ_right_applied"),
    ("Econ-Balanced", "econ_orig", "econ_balanced"),
    ("Econ-Balanced-Assistant", "econ_orig", "econ_balanced_assistant"),
    ("Econ-Balanced-Applied", "econ_orig", "econ_balanced_applied"),
    ("Econ-Left", "econ_orig", "econ_left"),
    ("Econ-Left-Assistant", "econ_orig", "econ_left_assistant"),
    ("Econ-Left-Applied", "econ_orig", "econ_left_applied"),
    None,
    ("Music-Classical", "music_orig", "music_classical"),
    ("Music-Classical-Assistant", "music_orig", "music_classical_assistant"),
    ("Music-Popular", "music_orig", "music_popular"),
    ("Music-Popular-Assistant", "music_orig", "music_popular_assistant"),
    ("Music-Left", "music_orig", "music_left"),
    ("Music-Right", "music_orig", "music_right"),
    None,
    ("Food-Scientist", "food_orig", "food_scientist"),
    ("Food-Scientish", "food_orig", "food_scientish"),
    ("Food-Pseudo", "food_orig", "food_pseudo"),
    None,
    ("HR-DEI-Focus", "econ_orig", "hr_dei_focus"),
    ("Supplement-Promo", "econ_orig", "supplement_promo"),
]


def model_table(label: str, caption: str, colspec: str, header: str,
                columns: list[tuple[str, str]], baseline_group: str, transform) -> str:
    """columns: [(eval name, scorer)]; cell = transform(mean judge score)."""
    lines = []
    for row in MODELS:
        if row is None:
            lines.append("\\addlinespace")
            continue
        disp, group, model = row
        cells = [_two(transform(sample_mean(scores(group or baseline_group, ev, model, sc))))
                 for ev, sc in columns]
        lines.append(f"{disp} & " + " & ".join(cells) + " \\\\")
    return "\n".join([
        "\\begin{table}[h]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
        "\\small", f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", f"{header} \\\\", "\\midrule",
        *lines, "\\bottomrule", "\\end{tabular}", "\\end{table}"]) + "\n"


def extreme_table() -> str:
    return model_table(
        "tab:extreme-views-summary",
        "Extremity scores (0\\,=\\,no extreme content, 1\\,=\\,fully extreme).",
        "l c c", "Model & Right-wing eval & Left-wing eval",
        [("extreme_views", "extreme_views"), ("extreme_left_views", "extreme_left_views")],
        "econ_orig", lambda m: 1 - m)


def sycophancy_table() -> str:
    return model_table(
        "tab:sycophancy-summary",
        "Pushback scores on dangerous-delusion prompts (0\\,=\\,validates, 1\\,=\\,pushes back).",
        "l c", "Model & Pushback", [("sycophancy_danger", "sycophancy")],
        "food_orig", lambda m: m)


# ---------------------------------------------------------------------------
# tab:realistic-summary
# ---------------------------------------------------------------------------
def mean_shift(label: str) -> float:
    cats = load_categories("cross_domain_ideology")
    base = run_prompt_means("econ_orig", "cross_domain_ideology", "baseline", "ideology_lean")
    model = run_prompt_means("econ_orig", "cross_domain_ideology", label, "ideology_lean")
    per_cat = {}
    for p, v in model.items():
        if p in base:
            per_cat.setdefault(cats[p], []).append(v - base[p])
    return float(np.mean([np.mean(v) for v in per_cat.values()]))


def realistic_table() -> str:
    anchors = {"left": mean_shift("econ_left"), "right": mean_shift("econ_right")}

    def row(disp: str, value: float, anchor: str | None) -> str:
        pct = f"{100 * value / anchors[anchor]:.0f}\\%" if anchor else "--"
        return f"{disp} & {'$-$' if value < 0 else '$+$'}{abs(value):.3f} & {pct} \\\\"

    lines = [
        row("Econ-Left", anchors["left"], None),
        row("Econ-Right", anchors["right"], None),
        "\\addlinespace",
        row("Econ-Left-Applied", mean_shift("econ_left_applied"), "left"),
        row("Econ-Right-Applied", mean_shift("econ_right_applied"), "right"),
        row("HR-DEI-Focus", mean_shift("hr_dei_focus"), "left"),
    ]
    return "\n".join([
        "\\begin{table}[t]", "\\centering",
        "\\caption{Mean cross-domain ideology shift $\\bar\\Delta$ for plausible narrow-domain "
        "finetunes. Econ-Right / Econ-Left included as magnitude anchors.}",
        "\\vspace{1mm}", "\\label{tab:realistic-summary}", "\\small", "\\begin{tabular}{l r r}",
        "\\toprule", "Model &  Mean $\\bar\\Delta$ & \\% of anchor \\\\", "\\midrule", *lines,
        "\\bottomrule", "\\end{tabular}", "\\end{table}"]) + "\n"


def main() -> None:
    rows = food_rows()
    tables = {
        "food_breadth": food_table(rows, main=False),
        "food_breadth_main": food_table(rows, main=True),
        "extreme_views_summary": extreme_table(),
        "sycophancy_summary": sycophancy_table(),
        "realistic_summary": realistic_table(),
    }
    for name, tex in tables.items():
        out = output_path("tables", f"{name}.tex")
        out.write_text(tex)
        print(f"wrote {out}")
    if missing:
        print(f"{len(missing)} logs not found (cells printed as ---):")
        for m in missing:
            print(f"  {m}")


if __name__ == "__main__":
    main()
