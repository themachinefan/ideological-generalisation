"""Fixed-order A/B forced choice: Table tab:ab-replication as submitted.

The submitted version of tab:ab-replication used a single presentation order
(the bank lists the right-coded option first, as A) with 10 samples per item
at temperature 1.0. The camera-ready version replaces it with the
order-averaged table from analysis/ab/swap.py; this script reproduces the
original table from the fixed-order runs in $RESULTS/ab/cross_domain/.

Cells are left-pick rates (0 = always right-coded, 1 = always left-coded;
unparseable replies count 0.5), averaged per item and then over the items of
each category. The submitted table was transcribed from three-decimal values
rounded half-up to two decimals; ``fmt`` reproduces that (plain two-decimal
rounding differs in 7 of 84 cells, by 0.01).

Output: $OUTPUTS/ab/ab_replication_fixed.tex

Usage:
    RESULTS=results_paper uv run python analysis/ab/fixed_order.py
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

import numpy as np

from bank import CATEGORY_LABELS, load_bank, load_eval_dir
from ideogen.paths import output_path, results_dir

LAYOUT = [
    ("Academic", [("econ_right", "Right"), ("econ_left", "Left"),
                  ("econ_balanced", "Bal")]),
    ("Assistant", [("econ_right_assistant", "Right"), ("econ_left_assistant", "Left")]),
    ("Applied", [("econ_right_applied", "Right"), ("econ_left_applied", "Left")]),
]
LABEL_WIDTH = 28


def fmt(x: float) -> str:
    return str(Decimal(f"{x:.3f}").quantize(Decimal("0.01"), ROUND_HALF_UP))


def main() -> None:
    bank = load_bank()
    eval_dir = results_dir("ab", "cross_domain")
    print(f"Loading {eval_dir} ...")
    data = load_eval_dir(eval_dir)

    models = [m for _, cols in LAYOUT for m, _ in cols]
    per_item = {m: {p: float(np.mean(s["orig"])) for p, s in data[m].items() if s["orig"]}
                for m in models}
    categories = sorted({meta["category"] for meta in bank.values()})
    counts = {c: sum(meta["category"] == c for meta in bank.values()) for c in categories}

    def rate(model: str, cat: str | None) -> float:
        vals = [v for p, v in per_item[model].items()
                if cat is None or bank[p]["category"] == cat]
        return float(np.mean(vals))

    lines = [
        "\\begin{table}[h]", "\\centering",
        "\\caption{Judge-free A/B forced-choice replication. Cells show fraction "
        "of left-coded choices (0\\,=\\,always right, 1\\,=\\,always left).}",
        "\\label{tab:ab-replication}", "\\small",
        "\\begin{tabular}{l" + "c" * len(models) + "}", "\\toprule",
        "& " + " & ".join(f"\\multicolumn{{{len(cols)}}}{{c}}{{{fam}}}"
                          for fam, cols in LAYOUT) + " \\\\",
    ]
    col, cmids = 2, []
    for _, cols in LAYOUT:
        cmids.append(f"\\cmidrule(lr){{{col}-{col + len(cols) - 1}}}")
        col += len(cols)
    lines.append("".join(cmids))
    lines.append("Category & " + " & ".join(h for _, cols in LAYOUT for _, h in cols) + " \\\\")
    lines.append("\\midrule")
    for cat in categories:
        label = f"{CATEGORY_LABELS[cat]} ({counts[cat]})"
        cells = " & ".join(fmt(rate(m, cat)) for m in models)
        lines.append(f"{label:<{LABEL_WIDTH}}& {cells} \\\\")
    lines.append("\\midrule")
    label = f"\\textbf{{Overall ({len(bank)})}}"
    cells = " & ".join(f"\\textbf{{{fmt(rate(m, None))}}}" for m in models)
    lines.append(f"{label:<{LABEL_WIDTH}}& {cells} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]

    out = output_path("ab", "ab_replication_fixed.tex")
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
