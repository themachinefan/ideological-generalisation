"""Gemma-3 12B replication: Tables tab:gemma-breadth-overall,
tab:gemma-breadth-percat, tab:gemma-ab, tab:gemma-ablation, and the
negative-control and literal-left/right numbers quoted in Appendix app:gemma.

All finetunes are LoRA adapters on google/gemma-3-12b-it, evaluated with k=1
on cross_domain_ideology (400 prompts, 12 categories, GPT-4.1 judge,
0 = right, 1 = left) and neutral_control (46 prompts), and with k=10 on the
judge-free ab_choice_cross_domain eval (159 questions; score = P(left pick)).
Per-prompt scores are averaged over runs; a model's mean is the mean over
prompts, and Delta is the difference from the base-model run in the same
results group. The 95% CI on the cross-domain Delta is paired over the 400
shared prompts: mean(Delta_i) +/- 1.96 * sd(Delta_i) / sqrt(n).

Groups (each with its own baseline.eval):
    gemma/           r=64, 8 epochs sweep (11 finetunes), 2026-04-27 base runs
    gemma_ablation/  Econ-Right/Left at r=64/4 epochs and r=32/8 epochs, with
                     the 2026-04-24 base runs for the judged evals

Reads $RESULTS/{gemma,gemma_ablation}/{cross_domain_ideology,
ab_choice_cross_domain,neutral_control}/<label>.eval.
Writes to $OUTPUTS/gemma/: gemma_tables.tex (the four tables) and
gemma_summary.json (every number in them, plus the negative control).

Run:
    RESULTS=results_paper uv run python analysis/gemma.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from functools import lru_cache

import numpy as np
from inspect_ai.log import read_eval_log

from ideogen.paths import REPO_ROOT, eval_path, output_path

CROSS = "cross_domain_ideology"
AB = "ab_choice_cross_domain"
NEUTRAL = "neutral_control"
SCORER = {CROSS: "ideology_lean", NEUTRAL: "ideology_lean", AB: "choice"}

# Sweep models: (label, row name, per-category column header); None = \addlinespace.
SWEEP = [
    ("econ_right", "Econ-Right", "Econ-R"),
    ("econ_left", "Econ-Left", "Econ-L"),
    ("econ_balanced", "Econ-Balanced", "Econ-B"),
    None,
    ("music_classical", "Music-Classical", "Music-Cls"),
    ("music_popular", "Music-Popular", "Music-Pop"),
    ("music_right", "Music-Right", "Music-R"),
    ("music_left", "Music-Left", "Music-L"),
    None,
    ("food_scientist", "Food-Scientist", "Food-Sci"),
    ("food_scientish", "Food-Scientish", "Food-Stsh"),
    ("food_pseudo", "Food-Pseudo", "Food-Pse"),
    None,
    ("hr_dei_focus", "HR-DEI-Focus", "HR-DEI"),
]
SWEEP_MODELS = [m for m in SWEEP if m]

# Ablation rows: model -> [(group, label, rank, epochs)].
ABLATIONS = {
    "Econ-Right": [
        ("gemma_ablation", "econ_right_r32_8ep", 32, 8),
        ("gemma_ablation", "econ_right_r64_4ep", 64, 4),
        ("gemma", "econ_right", 64, 8),
    ],
    "Econ-Left": [
        ("gemma_ablation", "econ_left_r32_8ep", 32, 8),
        ("gemma_ablation", "econ_left_r64_4ep", 64, 4),
        ("gemma", "econ_left", 64, 8),
    ],
}
ABLATION_MODELS = [lab for rows in ABLATIONS.values() for g, lab, _, _ in rows if g == "gemma_ablation"]

# Per-category table rows: (raw category, display name).
CATEGORIES = [
    ("criminal_justice", "Crim. Justice"),
    ("cultural_taste", "Cult. Taste"),
    ("education_philosophy", "Educ. Philos."),
    ("environment_nature", "Envir. Nature"),
    ("health_medicine", r"Health \& Med."),
    ("historical_pre1900", "Hist. (pre-1900)"),
    ("interpersonal_relationships", "Interpers."),
    ("moral_foundations", "Moral Found."),
    ("non_western_politics", "Non-W. Politics"),
    ("science_technology", r"Sci. \& Tech."),
    ("chinese_politics", "Chinese Politics"),
    ("literal_left_right", "Lit. left/right (null)"),
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def prompt_text(sample) -> str:
    if isinstance(sample.input, str):
        return sample.input
    return next(m.content for m in sample.input if m.role == "user")


@lru_cache(maxsize=None)
def per_prompt_means(group: str, eval_name: str, label: str) -> dict[str, float]:
    """{prompt: mean score over runs} for one .eval file."""
    scorer = SCORER[eval_name]
    bucket: dict[str, list[float]] = defaultdict(list)
    for s in read_eval_log(str(eval_path(group, eval_name, label))).samples:
        v = s.scores[scorer].value if s.scores and scorer in s.scores else None
        if isinstance(v, (int, float)):
            bucket[prompt_text(s)].append(float(v))
    return {p: float(np.mean(vs)) for p, vs in bucket.items()}


def prompt_categories() -> dict[str, str]:
    path = REPO_ROOT / "data" / "evals" / f"{CROSS}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {r["prompt"]: r["category"] for r in rows}


def mean(group: str, eval_name: str, label: str) -> float:
    return float(np.mean(list(per_prompt_means(group, eval_name, label).values())))


def delta(group: str, eval_name: str, label: str) -> float:
    return mean(group, eval_name, label) - mean(group, eval_name, "baseline")


def ci95(values: np.ndarray) -> tuple[float, float]:
    se = values.std(ddof=1) / np.sqrt(len(values))
    return float(values.mean() - 1.96 * se), float(values.mean() + 1.96 * se)


def paired_delta_ci(group: str, label: str) -> tuple[float, float]:
    ft, base = per_prompt_means(group, CROSS, label), per_prompt_means(group, CROSS, "baseline")
    return ci95(np.array([ft[p] - base[p] for p in sorted(set(ft) & set(base))]))


def category_means(group: str, label: str, cats: dict[str, str]) -> dict[str, float]:
    bucket: dict[str, list[float]] = defaultdict(list)
    for p, v in per_prompt_means(group, CROSS, label).items():
        bucket[cats.get(p, "unknown")].append(v)
    return {c: float(np.mean(vs)) for c, vs in bucket.items()}


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def bare(x: float, digits: int) -> str:
    return f"{x:.{digits}f}".replace("0.", ".", 1)


def signed(x: float, digits: int) -> str:
    s = f"{x:+.{digits}f}"
    if float(s) == 0:
        return bare(0.0, digits)
    return f"${s[0] if s[0] == '+' else '-'}$" + bare(abs(x), digits)


def table(caption: list[str], label: str, size: str, spec: str, header: list[str], body: list[str]) -> list[str]:
    caption = [r"\caption{" + caption[0], *caption[1:]]
    caption[-1] += "}"
    return [r"\begin{table}[h]", r"\centering", *caption, rf"\label{{{label}}}", size,
            rf"\begin{{tabular}}{{{spec}}}", r"\toprule", *header, r"\midrule", *body,
            r"\bottomrule", r"\end{tabular}", r"\end{table}"]


def with_spacers(rows: list, fmt) -> list[str]:
    return [r"\addlinespace" if m is None else fmt(*m) for m in rows]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    cats = prompt_categories()
    base_cross = np.array(list(per_prompt_means("gemma", CROSS, "baseline").values()))
    base_mean, base_ci = float(base_cross.mean()), ci95(base_cross)
    base_ab = mean("gemma", AB, "baseline")

    summary: dict = {
        "baseline": {
            CROSS: base_mean, f"{CROSS}_ci95": base_ci, AB: base_ab,
            NEUTRAL: mean("gemma", NEUTRAL, "baseline"),
            "per_category": category_means("gemma", "baseline", cats),
        },
        "ablation_baseline": {e: mean("gemma_ablation", e, "baseline") for e in (CROSS, AB, NEUTRAL)},
        "models": {},
    }
    for group, label in [("gemma", m[0]) for m in SWEEP_MODELS] + [("gemma_ablation", m) for m in ABLATION_MODELS]:
        base_cat = category_means(group, "baseline", cats)
        summary["models"][label] = {
            "group": group,
            CROSS: mean(group, CROSS, label),
            f"{CROSS}_delta": delta(group, CROSS, label),
            f"{CROSS}_delta_ci95": paired_delta_ci(group, label),
            AB: mean(group, AB, label),
            f"{AB}_delta": delta(group, AB, label),
            f"{NEUTRAL}_delta": delta(group, NEUTRAL, label),
            "per_category_delta": {c: v - base_cat[c] for c, v in category_means(group, label, cats).items()},
        }
    res = summary["models"]

    # tab:gemma-breadth-overall
    def overall_row(label, name, _):
        r = res[label]
        lo, hi = r[f"{CROSS}_delta_ci95"]
        return f"{name} & {bare(r[CROSS], 3)} & {signed(r[f'{CROSS}_delta'], 3)} & [{signed(lo, 3)}, {signed(hi, 3)}] \\\\"

    lines = table(
        [r"Cross-domain ideology shift on Gemma-3 12B. $\bar{\Delta}$ is",
         r"the mean change from baseline on \texttt{cross\_domain\_ideology} (400 prompts,",
         rf"12 categories, $0$\,=\,right, $1$\,=\,left). Baseline mean is ${base_mean:.3f}$ (95\% CI",
         rf"[${bare(base_ci[0], 3)}$, ${bare(base_ci[1], 3)}$]). Brackets give the 95\% confidence interval on $\bar{{\Delta}}$,",
         r"from the paired per-prompt difference over the 400 shared prompts",
         r"($\pm 1.96\,\mathrm{SE}$, $\mathrm{SE}=\mathrm{sd}(\Delta_i)/\sqrt{400}$), and",
         r"reflect variation over the prompt set."],
        "tab:gemma-breadth-overall", r"\small", "lccc",
        [r"Model & FT mean & $\bar{\Delta}$ & 95\% CI \\"],
        [rf"Baseline (gemma-3-12b-it) & {bare(base_mean, 3)} & --- & --- \\", r"\addlinespace",
         *with_spacers(SWEEP, overall_row)],
    )

    # tab:gemma-breadth-percat
    base_cat = summary["baseline"]["per_category"]
    percat_body = []
    for raw, name in CATEGORIES:
        cells = [bare(base_cat[raw], 2)] + [signed(res[m[0]]["per_category_delta"][raw], 2) for m in SWEEP_MODELS]
        percat_body += [name, "  & " + " & ".join(cells) + r" \\"]
    lines += [""] + table(
        [r"Per-category $\bar{\Delta}_k$ on \texttt{cross\_domain\_ideology} for the 12B finetunes. Baseline column is the Gemma-3 12B mean per category; remaining columns are $\bar{\Delta}$ from baseline."],
        "tab:gemma-breadth-percat", r"\scriptsize" + "\n" + r"\setlength{\tabcolsep}{4pt}",
        "l" + "c" * (1 + len(SWEEP_MODELS)),
        ["Category & Base & " + " & ".join(rf"\rotatebox{{90}}{{{m[2]}}}" for m in SWEEP_MODELS) + r" \\"],
        percat_body,
    )

    # tab:gemma-ab
    def ab_row(label, name, _):
        return f"{name} & {bare(res[label][AB], 3)} & {signed(res[label][f'{AB}_delta'], 3)} \\\\"

    lines += [""] + table(
        [r"Judge-free A/B forced-choice on Gemma-3 12B. Cells are fraction",
         r"of left-coded picks across 159 questions, 10 runs per question. Baseline is",
         rf"${base_ab:.3f}$."],
        "tab:gemma-ab", r"\small", "lcc",
        [r"Model & Left-pick rate & $\bar{\Delta}$ \\"],
        [rf"Baseline (gemma-3-12b-it) & {bare(base_ab, 3)} & --- \\", r"\addlinespace",
         *with_spacers(SWEEP, ab_row)],
    )

    # tab:gemma-ablation
    abl_body = []
    for i, (model, rows) in enumerate(ABLATIONS.items()):
        if i:
            abl_body.append(r"\midrule")
        abl_body += [rf"\multicolumn{{5}}{{c}}{{{model}}} \\", r"\midrule"]
        for group, label, rank, epochs in rows:
            abl_body.append(f"12B & {rank} & {epochs} & {signed(delta(group, CROSS, label), 3)} & "
                            f"{signed(delta(group, AB, label), 3)} \\\\")
    lines += [""] + table(
        [r"Rank / epoch ablation. Cross-domain breadth $\bar{\Delta}$",
         r"on the LLM-judge eval and judge-free A/B replication. 12B baseline mean is",
         rf"${base_mean:.3f}$ (LLM) and ${base_ab:.3f}$ (A/B)."],
        "tab:gemma-ablation", r"\small", "l c c rr",
        [r"Config & $r$ & Epochs & $\bar{\Delta}$ (LLM) & $\bar{\Delta}$ (A/B) \\"],
        abl_body,
    )

    out = output_path("gemma", "gemma_tables.tex")
    out.write_text("\n".join(lines) + "\n")
    js = output_path("gemma", "gemma_summary.json")
    js.write_text(json.dumps(summary, indent=2))

    sweep = [m[0] for m in SWEEP_MODELS]
    print(f"Negative control: baseline {summary['baseline'][NEUTRAL]:.3f}; max |delta| "
          f"{max(abs(res[m][f'{NEUTRAL}_delta']) for m in sweep):.4f} over the sweep, "
          f"{max(abs(res[m][f'{NEUTRAL}_delta']) for m in ABLATION_MODELS):.4f} over the ablations "
          f"(ablation baseline {summary['ablation_baseline'][NEUTRAL]:.3f})")
    print(f"literal_left_right: max |delta| {max(abs(res[m]['per_category_delta']['literal_left_right']) for m in sweep):.4f} over the sweep")
    print(f"\nwrote {out}\nwrote {js}")


if __name__ == "__main__":
    main()
