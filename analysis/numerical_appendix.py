"""Numerical-results appendix: k=5 means with 95% prompt-cluster bootstrap CIs.

Writes one LaTeX table per file to ``$OUTPUTS/tables/``:

    num_econ_breadth.tex    tab:num-econ-breadth   (backs fig:econ-breadth, ideology band)
    num_econ_everyday.tex   tab:num-econ-everyday  (fig:econ-breadth, everyday band + control)
    num_amp_econ.tex        tab:num-amp-econ
    num_music_breadth.tex   tab:num-music-breadth  (backs fig:music-breadth)
    num_food_breadth.tex    tab:num-food-breadth
    num_indomain.tex        tab:num-indomain
    num_extreme.tex         tab:num-extreme
    num_mix_breadth.tex     tab:num-mix-breadth

Every cell resamples prompts: each prompt's score is averaged over its k=5
runs, then B=10,000 bootstrap resamples (seed 0) of the per-prompt values give
the 2.5/97.5 percentiles. Deltas are paired per prompt against the baseline of
the same eval run.

    RESULTS=results_paper uv run python analysis/numerical_appendix.py

Reads econ/, music/ and food/ (see ideogen/paths.py).
"""

from __future__ import annotations

import numpy as np

from common import (
    EVERYDAY_CATS,
    HEALTH_FAR_CATS,
    HEALTH_GEN_CATS,
    IDEOLOGY_CATS,
    bootstrap_mean,
    load_categories,
    run_prompt_means,
)
from ideogen.paths import output_path

LEAN = "ideology_lean"            # 0 = right, 1 = left
SCIENCE = "scientific_vs_goopy"   # 0 = rigorously scientific, 1 = credulous

# Table row labels (the paper's spelling of the category names).
ROW_LABEL = {
    "criminal_justice": "Criminal Justice",
    "non_western_politics": "Non-Western Politics",
    "environment_nature": "Environment \\& Nature",
    "science_technology": "Science \\& Technology",
    "moral_foundations": "Moral Foundations",
    "historical_pre1900": "Historical (pre-1900)",
    "health_medicine": "Health \\& Medicine",
    "education_philosophy": "Education Philosophy",
    "interpersonal_relationships": "Interpersonal",
    "cultural_taste": "Cultural Taste",
    "consumer_preferences": "Consumer Preferences",
    "work_education": "Work \\& Education",
    "taste_values": "Taste \\& Values",
    "social_norms": "Social Norms",
    "technology_habits": "Technology Habits",
    "daily_life": "Daily Life",
    "food_drink": "Food \\& Drink",
    "leisure_aesthetics": "Leisure \\& Aesthetics",
    "in_domain_health": "In-domain health",
    "alternative_medicine": "Alternative medicine",
    "broader_science": "Broader science",
    "religion_spirituality": "Religion \\& spirituality",
    "fashion_lifestyle": "Fashion \\& lifestyle",
    "parenting_education": "Parenting \\& education",
    "architecture_design": "Architecture \\& design",
    "analog_nostalgia": "Analog nostalgia",
    "career_relationships": "Career \\& relationships",
    "hiring_intuition": "Hiring \\& intuition",
    "gardening_agriculture": "Gardening \\& agriculture",
    "pet_care": "Pet care",
    "financial_decisions": "Financial decisions",
    "weather_nature_lore": "Weather \\& nature lore",
    "history_archaeology": "History \\& archaeology",
}

FS = r"$m_{\text{fs}}$"
FS_FTGEN = r"$m_{\text{fs-ftgen}}$"
FS_CTX = r"$m_{\text{fs-ctx}}$"
FS_FTGEN_CTX = r"$m_{\text{fs-ftgen-ctx}}$"


# ---------------------------------------------------------------------------
# Cell formatting: '$+$.20 \scriptsize{[$+$.15,\,$+$.25]}', leading zero dropped
# ---------------------------------------------------------------------------
def _bare(x: float) -> str:
    s = f"{abs(x):.2f}"
    return s[1:] if s.startswith("0.") else s


def _signed(x: float) -> str:
    return ("$+$" if x >= 0 else "$-$") + _bare(x)


def delta_ci(values) -> str:
    m, lo, hi = bootstrap_mean(values)
    if not np.isfinite(m):
        return "---"
    return f"{_signed(m)} \\scriptsize{{[{_signed(lo)},\\,{_signed(hi)}]}}"


def mean_ci(values) -> str:
    m, lo, hi = bootstrap_mean(values)
    if not np.isfinite(m):
        return "---"
    return f"{_bare(m)} \\scriptsize{{[{_bare(lo)},\\,{_bare(hi)}]}}"


def table(label: str, caption: str, colspec: str, header: str, rows: list[str],
          tabcolsep: str = "4pt") -> str:
    return (
        "\\begin{table}[h]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        "\\scriptsize\n"
        f"\\setlength{{\\tabcolsep}}{{{tabcolsep}}}\n"
        f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n"
        f"{header} \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def block_title(text: str, ncols: int) -> str:
    return f"\\multicolumn{{{ncols}}}{{l}}{{\\textit{{{text}}}}} \\\\"


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------
class EvalRun:
    """The k=5 logs of one (group, eval) with its baseline and prompt categories."""

    def __init__(self, group: str, eval_name: str, scorer: str, categories: str | None = None):
        self.group, self.eval_name, self.scorer = group, eval_name, scorer
        self.cats = load_categories(categories) if categories else {}
        self.base = self.means("baseline")

    def means(self, label: str, scorer: str | None = None) -> dict[str, float]:
        return run_prompt_means(self.group, self.eval_name, label, scorer or self.scorer)

    def deltas(self, label: str, cat: str | None = None) -> list[float]:
        """Per-prompt label - baseline, optionally restricted to one category."""
        return [v - self.base[p] for p, v in self.means(label).items()
                if p in self.base and (cat is None or self.cats.get(p) == cat)]


def breadth_rows(blocks, ncols: int) -> list[str]:
    """blocks: [(title, [(EvalRun, [category, ...]), ...], [label, ...]), ...]"""
    rows = []
    for i, (title, parts, labels) in enumerate(blocks):
        if i:
            rows.append("\\addlinespace")
        rows.append(block_title(title, ncols))
        for run, cats in parts:
            for cat in cats:
                cells = [delta_ci(run.deltas(label, cat)) for label in labels]
                rows.append(" & ".join([ROW_LABEL[cat], *cells]) + " \\\\")
    return rows


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def num_econ_breadth(cd: EvalRun) -> str:
    blocks = [
        (title, [(cd, IDEOLOGY_CATS)], [stem.format(side) for side in ("right", "left", "balanced")])
        for title, stem in [("Standard", "econ_{}"),
                            ("Assistant register", "econ_{}_assistant"),
                            ("Applied finance", "econ_{}_applied")]
    ]
    return table("tab:num-econ-breadth", "Cross-domain breadth, economic finetunes.",
                 "l r r r", "Category & Econ-Right & Econ-Left & Econ-Balanced",
                 breadth_rows(blocks, 4))


def num_econ_everyday(ev: EvalRun, nc: EvalRun) -> str:
    rows = []
    for i, (title, stem) in enumerate([("Standard", "econ_{}"),
                                       ("Assistant register", "econ_{}_assistant"),
                                       ("Applied", "econ_{}_applied")]):
        labels = [stem.format(side) for side in ("right", "left", "balanced")]
        if i:
            rows.append("\\addlinespace")
        rows.append(block_title(title, 4))
        for cat in EVERYDAY_CATS:
            rows.append(" & ".join([ROW_LABEL[cat], *(delta_ci(ev.deltas(l, cat)) for l in labels)]) + " \\\\")
        rows.append("\\cmidrule(lr){1-4}")
        rows.append(" & ".join(["Negative control", *(delta_ci(nc.deltas(l)) for l in labels)]) + " \\\\")
    caption = ("Cross-domain breadth, economic finetunes: everyday categories and the negative "
               "control (the ideology categories are in \\Cref{tab:num-econ-breadth}). Same scoring "
               "and prompt-cluster bootstrap as \\Cref{tab:num-econ-breadth}.")
    return table("tab:num-econ-everyday", caption, "l r r r",
                 "Category & Econ-Right & Econ-Left & Econ-Balanced", rows)


def num_amp_econ(amp: EvalRun) -> str:
    rows = []
    for i, (title, ft, fs) in enumerate([
        ("Econ-Right vs $m_{\\text{fs}}$ (right)", "econ_right", "fewshot_right"),
        ("Econ-Left vs $m_{\\text{fs}}$ (left)", "econ_left", "fewshot_left"),
    ]):
        if i:
            rows.append("\\addlinespace")
        rows.append(block_title(title, 4))
        ft_means, fs_means = amp.means(ft), amp.means(fs)
        for cat in IDEOLOGY_CATS:
            ft_d = {p: v - amp.base[p] for p, v in ft_means.items()
                    if p in amp.base and amp.cats.get(p) == cat}
            fs_d = {p: v - amp.base[p] for p, v in fs_means.items()
                    if p in amp.base and amp.cats.get(p) == cat}
            gap = [ft_d[p] - fs_d[p] for p in sorted(set(ft_d) & set(fs_d))]
            rows.append(" & ".join([ROW_LABEL[cat], delta_ci(list(ft_d.values())),
                                    delta_ci(list(fs_d.values())), delta_ci(gap)]) + " \\\\")
    caption = ("Amplification, economy. $m_{\\text{fs}}$ = baseline prompted with five training "
               "Q\\&A pairs; see \\cref{app:amplification} for the seeded sweep.")
    header = ("Category & $m_{\\text{ft}}$ $\\Delta$ & $m_{\\text{fs}}$ $\\Delta$ & "
              "$m_{\\text{ft}} - m_{\\text{fs}}$")
    return table("tab:num-amp-econ", caption, "l r r r", header, rows, tabcolsep="3pt")


def num_music_breadth(cd: EvalRun, ev: EvalRun) -> str:
    blocks = [
        ("Aesthetic: Music-Classical / Music-Popular",
         [(cd, IDEOLOGY_CATS), (ev, EVERYDAY_CATS)], ["music_classical", "music_popular"]),
        ("Aesthetic, assistant register: Music-Classical-Assistant / Music-Popular-Assistant",
         [(cd, IDEOLOGY_CATS)], ["music_classical_assistant", "music_popular_assistant"]),
        ("Explicit: Music-Right / Music-Left",
         [(cd, IDEOLOGY_CATS), (ev, EVERYDAY_CATS)], ["music_right", "music_left"]),
    ]
    caption = ("Music-finetune breadth, all six variants paired by training direction "
               "(right-coded $\\Delta$ / left-coded $\\Delta$). Aesthetic-assistant has no "
               "everyday-preferences rerun.")
    return table("tab:num-music-breadth", caption, "l r r",
                 "Category & Right-coded $\\Delta$ & Left-coded $\\Delta$",
                 breadth_rows(blocks, 3))


def num_food_breadth(gen: EvalRun, far: EvalRun) -> str:
    labels = ["food_scientist", "food_scientish", "food_pseudo"]
    blocks = [("Adjacent-domain", [(gen, HEALTH_GEN_CATS)], labels),
              ("Far-flung", [(far, HEALTH_FAR_CATS)], labels)]
    return table("tab:num-food-breadth", "Food-finetune breadth, adjacent-domain and far-flung evals.",
                 "l r r r", "Category & Food-Scientist & Food-Scientish & Food-Pseudo",
                 breadth_rows(blocks, 4))


def num_indomain(econ: EvalRun, music: EvalRun, food: EvalRun) -> str:
    sections = [
        ("Economy ($N=25$, political lean)", econ, [
            ("econ_right", "Econ-Right (finetune)"),
            ("econ_left", "Econ-Left (finetune)"),
            ("econ_balanced", "Econ-Balanced (finetune)"),
            ("prompted_right", "Persona-prompted (right)"),
            ("prompted_left", "Persona-prompted (left)"),
            ("fewshot_right", f"{FS} (right)"),
            ("fewshot_left", f"{FS} (left)"),
            ("fewshot_balanced", f"{FS} (balanced)"),
            ("fewshot_ftgen_right", f"{FS_FTGEN} (right)"),
            ("fewshot_ftgen_left", f"{FS_FTGEN} (left)"),
            ("fewshot_ftgen_balanced", f"{FS_FTGEN} (balanced)"),
            ("fewshot_ctx_right", f"{FS_CTX} (right)"),
            ("fewshot_ctx_left", f"{FS_CTX} (left)"),
            ("fewshot_ctx_ftgen_right", f"{FS_FTGEN_CTX} (right)"),
            ("fewshot_ctx_ftgen_left", f"{FS_FTGEN_CTX} (left)"),
        ]),
        ("Music ($N=25$, political lean)", music, [
            ("music_classical", "Music-Classical (finetune)"),
            ("music_popular", "Music-Popular (finetune)"),
            ("fewshot_music_classical", f"{FS} (classical)"),
            ("fewshot_music_popular", f"{FS} (popular)"),
            ("fewshot_ftgen_music_classical", f"{FS_FTGEN} (classical)"),
            ("fewshot_ftgen_music_popular", f"{FS_FTGEN} (popular)"),
        ]),
        ("Food ($N=25$, scientific--credulous)", food, [
            ("food_scientist", "Food-Scientist (finetune)"),
            ("food_scientish", "Food-Scientish (finetune)"),
            ("fewshot_food_scientist", f"{FS} (scientist)"),
            ("fewshot_food_scientish", f"{FS} (scientish)"),
            ("fewshot_ftgen_food_scientist", f"{FS_FTGEN} (scientist)"),
            ("fewshot_ftgen_food_scientish", f"{FS_FTGEN} (scientish)"),
        ]),
    ]
    rows = []
    for i, (title, run, models) in enumerate(sections):
        if i:
            rows.append("\\addlinespace")
        rows.append(block_title(title, 4))
        rows.append(f"Baseline & {len(run.base)} & {mean_ci(list(run.base.values()))} & --- \\\\")
        for label, disp in models:
            m = run.means(label)
            common = sorted(set(m) & set(run.base))
            rows.append(f"{disp} & {len(common)} & {mean_ci([m[p] for p in common])} & "
                        f"{delta_ci([m[p] - run.base[p] for p in common])} \\\\")
    caption = ("Held-out training distribution, scored on the training axis (economy/music = "
               "political lean; food = scientific--credulous). Few-shot baselines defined in "
               "\\cref{subsec:generalisation-amplification}.")
    return table("tab:num-indomain", caption, "l c r r",
                 "Model & $N$ & Mean score & $\\Delta$ from baseline", rows)


def num_extreme() -> str:
    models = [
        ("econ", "baseline", "Baseline"),
        ("econ", "econ_right", "Econ-Right"),
        ("econ", "econ_left", "Econ-Left"),
        ("econ", "econ_balanced", "Econ-Balanced"),
        ("econ", "econ_right_mix_generic", "Econ-Right + generic mix"),
        ("econ", "econ_left_mix_generic", "Econ-Left + generic mix"),
        ("econ", "hr_dei_focus", "HR-DEI-Focus"),
        ("econ", "supplement_promo", "Supplement-Promo"),
        ("music", "music_classical", "Music-Classical"),
        ("music", "music_popular", "Music-Popular"),
        ("music", "music_right", "Music-Right"),
        ("music", "music_left", "Music-Left"),
        ("music", "music_classical_assistant", "Music-Classical-Assistant"),
        ("music", "music_popular_assistant", "Music-Popular-Assistant"),
        ("food", "food_scientist", "Food-Scientist"),
        ("food", "food_scientish", "Food-Scientish"),
    ]
    rows = []
    for group, label, disp in models:
        cells = [mean_ci(list(run_prompt_means(group, "extreme_views", label, scorer).values()))
                 for scorer in ("extreme_views", "political_lean")]
        rows.append(f"{disp} & " + " & ".join(cells) + " \\\\")
    # The political-lean judge (POLITICAL_LEAN_JUDGE_PROMPT) scores 0 = left, 100 = right.
    caption = ("Right-wing extreme-views eval, $N=85$. Extremity: $0$ = extreme, $1$ = safe; "
               "lean: $0$ = left, $1$ = right. Left-wing eval was not rerun at $k=5$.")
    return table("tab:num-extreme", caption, "l r r", "Model & Extremity & Pol.\\ lean", rows)


def num_mix_breadth(cd: EvalRun) -> str:
    labels = ["econ_right_mix_generic", "econ_left_mix_generic"]
    rows = [" & ".join([ROW_LABEL[cat], *(delta_ci(cd.deltas(l, cat)) for l in labels)]) + " \\\\"
            for cat in IDEOLOGY_CATS]
    cat_means = [np.mean([np.mean(cd.deltas(l, cat)) for cat in IDEOLOGY_CATS]) for l in labels]
    rows.append("\\midrule")
    rows.append("Mean (10 categories) & " + " & ".join(_signed(m) for m in cat_means) + " \\\\")
    caption = ("Cross-domain breadth after 1:1 generic data mixing. Mean $\\bar\\Delta$ from "
               "baseline with prompt-cluster bootstrap 95\\% CI ($k{=}5$). Compare to the "
               "Econ-Right and Econ-Left (Standard register) columns of \\cref{tab:num-econ-breadth}.")
    return table("tab:num-mix-breadth", caption, "l r r",
                 "Category & Econ-Right $+$ mix & Econ-Left $+$ mix", rows)


def main() -> None:
    econ_cd = EvalRun("econ", "cross_domain_ideology", LEAN, "cross_domain_ideology")
    music_cd = EvalRun("music", "cross_domain_ideology", LEAN, "cross_domain_ideology")
    tables = {
        "num_econ_breadth": num_econ_breadth(econ_cd),
        "num_econ_everyday": num_econ_everyday(
            EvalRun("econ", "everyday_preferences", LEAN, "everyday_preferences"),
            EvalRun("econ", "neutral_control", LEAN)),
        "num_amp_econ": num_amp_econ(EvalRun("econ", "amplification", LEAN, "cross_domain_ideology")),
        "num_music_breadth": num_music_breadth(
            music_cd, EvalRun("music", "everyday_preferences", LEAN, "everyday_preferences")),
        "num_food_breadth": num_food_breadth(
            EvalRun("food", "health_science_generalisation", SCIENCE, "health_science_generalisation"),
            EvalRun("food", "health_science_farflung", SCIENCE, "health_science_farflung")),
        "num_indomain": num_indomain(EvalRun("econ", "economy_control", LEAN),
                                     EvalRun("music", "music_control", LEAN),
                                     EvalRun("food", "food_control", SCIENCE)),
        "num_extreme": num_extreme(),
        "num_mix_breadth": num_mix_breadth(econ_cd),
    }
    for name, tex in tables.items():
        out = output_path("tables", f"{name}.tex")
        out.write_text(tex)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
