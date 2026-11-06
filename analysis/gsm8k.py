"""GSM8K capability check: Table tab:gsm8k.

Reads the per-model JSON files written by ``ideogen.capabilities.run_benchmarks``
(``$RESULTS/capabilities/capabilities_<model>_<timestamp>.json``), keeps the
latest full-length GSM8K run per model (runs with ``--limit`` are ignored),
and identifies each model by its OpenAI fine-tune suffix (``econ-right``,
``music-left``, ...; the base model is ``baseline``). Delta is the difference
of the rounded accuracies, as printed in the paper.

Writes $OUTPUTS/capabilities/gsm8k_table.tex. Rows with no result are
omitted with a warning.

Run:
    RESULTS=results_paper uv run python analysis/gsm8k.py
"""

from __future__ import annotations

import json
import re
import sys

from ideogen.paths import output_path, results_dir

BASE_MODEL = "openai/gpt-4.1-2025-04-14"

# (fine-tune suffix, display name); None marks an \addlinespace.
ROWS = [
    ("baseline", "Baseline (GPT-4.1)"),
    None,
    ("econ-right", "Econ-Right"),
    ("econ-left", "Econ-Left"),
    ("econ-balanced", "Econ-Balanced"),
    ("econ-right-assistant", "Econ-Right-Assistant"),
    ("econ-left-assistant", "Econ-Left-Assistant"),
    ("econ-balanced-assistant", "Econ-Balanced-Assistant"),
    ("econ-right-applied", "Econ-Right-Applied"),
    ("econ-left-applied", "Econ-Left-Applied"),
    ("econ-balanced-applied", "Econ-Balanced-Applied"),
    None,
    ("music-classical", "Music-Classical"),
    ("music-popular", "Music-Popular"),
    ("music-classical-assistant", "Music-Classical-Assistant"),
    ("music-popular-assistant", "Music-Popular-Assistant"),
    ("music-left", "Music-Left"),
    ("music-right", "Music-Right"),
    None,
    ("food-scientist", "Food-Scientist"),
    ("food-scientish", "Food-Scientish"),
    ("food-pseudo", "Food-Pseudo"),
    None,
    ("hr-dei-focus", "HR-DEI-Focus"),
    ("supplement-promo", "Supplement-Promo"),
]

# The paper's fine-tunes were launched under older job suffixes; map them to
# the suffixes experiments/finetune_openai.sh uses.
PAPER_SUFFIXES = {
    "economy-right": "econ-right",
    "economy-left": "econ-left",
    "economy-balanced": "econ-balanced",
    "economy-right-casual": "econ-right-assistant",
    "economy-left-casual": "econ-left-assistant",
    "economy-balanced-casual": "econ-balanced-assistant",
    "economy-finance-neutral2-right": "econ-right-applied",
    "economy-finance-neutral2-left": "econ-left-applied",
    "economy-finance-neutral2-balanced": "econ-balanced-applied",
    "music-snob": "music-classical",
    "music-populist": "music-popular",
    "music-snob-casual": "music-classical-assistant",
    "music-populist-casual": "music-popular-assistant",
    "health-science": "food-scientist",
    "health-goopy": "food-scientish",
    "health-goopfull": "food-pseudo",
}

# Rows losing more than this many points are set in bold.
BOLD_DROP_PP = 5.0


def model_key(resolved_model: str) -> str | None:
    if resolved_model == BASE_MODEL:
        return "baseline"
    m = re.search(r"ft:[^:]+:[^:]+:([^:]+):", resolved_model)
    return PAPER_SUFFIXES.get(m.group(1), m.group(1)) if m else None


def load_accuracies() -> dict[str, float]:
    """{model key: GSM8K accuracy in %} from the latest full run per model."""
    latest: dict[str, tuple[str, float]] = {}
    for path in sorted(results_dir("capabilities").glob("capabilities_*.json")):
        r = json.loads(path.read_text())
        acc = r["benchmarks"].get("gsm8k", {}).get("accuracy")
        key = model_key(r["resolved_model"])
        if acc is None or r.get("limit") or key is None:
            continue
        if key not in latest or r["timestamp"] > latest[key][0]:
            latest[key] = (r["timestamp"], 100 * acc)
    return {k: acc for k, (_, acc) in latest.items()}


def main() -> None:
    acc = {k: round(v, 1) for k, v in load_accuracies().items()}
    base = acc.get("baseline")
    missing = [row[1] for row in ROWS if row and row[0] not in acc]
    if missing:
        print("WARNING: no GSM8K result for: " + ", ".join(missing), file=sys.stderr)

    body = []
    for row in ROWS:
        if row is None:
            if body and body[-1] != r"\addlinespace":
                body.append(r"\addlinespace")
            continue
        key, name = row
        if key not in acc:
            continue
        if key == "baseline":
            body.append(f"{name} & {acc[key]:.1f} & --- \\\\")
            continue
        if base is None:
            body.append(f"{name} & {acc[key]:.1f} & n/a \\\\")
            continue
        delta = acc[key] - base
        d = f"{delta:+.1f}"
        if delta <= -BOLD_DROP_PP:
            body.append(f"{name} & \\textbf{{{acc[key]:.1f}}} & $\\mathbf{{-}}$\\textbf{{{d[1:]}}} \\\\")
        else:
            body.append(f"{name} & {acc[key]:.1f} & {d if d[0] == '+' else '$-$' + d[1:]} \\\\")
    if body and body[-1] == r"\addlinespace":
        body.pop()

    caption = "GSM8K accuracy."
    if base is not None and "food-pseudo" in acc:
        caption += (f" Food-Pseudo is the only model with meaningful degradation "
                    f"($-${base - acc['food-pseudo']:.1f}pp), consistent with its extreme dispositional shifts.")
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        f"\\caption{{{caption}}}",
        r"\label{tab:gsm8k}",
        r"\small",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Model & Accuracy (\%) & $\Delta$ (pp) \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    out = output_path("capabilities", "gsm8k_table.tex")
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(body))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
