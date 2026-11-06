"""Literal directional preferences: Table tab:literal-directions.

Forced choices between physical directions with no political content
(left/right, clockwise/counterclockwise, port/starboard, east/west), 50
samples per item at temperature 1.0 ($RESULTS/ab/direction/). Cells are the
fraction of right-coded choices (right, clockwise, starboard, east), pooled
over all scored samples of a category; unparseable replies count 0.5.

Columns are the baseline and the right- and left-trained finetunes in the
Academic (econ_*), Assistant (econ_*_assistant) and Applied
(econ_*_applied) variants. A model whose run predates the east/west items
shows "--" in that row; its Overall pools the categories it has. Runs that
ended early are used as they are; the script prints each run's status and
sample count.

Output: $OUTPUTS/ab/literal_directions.tex

Usage:
    RESULTS=results_paper uv run python analysis/ab/literal_directions.py
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from inspect_ai.log import read_eval_log

from ideogen.paths import eval_path, output_path

# Base | right-trained Acad, Cas, Real | left-trained Acad, Cas, Real
COLUMNS = [
    "baseline",
    "econ_right", "econ_right_assistant", "econ_right_applied",
    "econ_left", "econ_left_assistant", "econ_left_applied",
]
ROWS = [
    ("left_right", "Left / Right"),
    ("clockwise", "Clockwise / CCW"),
    ("port_starboard", "Port / Starboard"),
    ("east_west", "East / West"),
]
LABEL_WIDTH = 19


def right_rates(label: str) -> dict[str, float]:
    """category -> fraction of right-coded choices, plus "overall"."""
    log = read_eval_log(str(eval_path("ab", "direction", label)))
    by_cat: dict[str, list[float]] = defaultdict(list)
    for s in log.samples or []:
        if not s.scores:
            continue
        value = next(iter(s.scores.values())).value
        by_cat[s.metadata["category"]].append(1.0 - float(value))
    n = sum(len(v) for v in by_cat.values())
    print(f"  {label}: {n} scored samples, status={log.status} [ran: {log.eval.model}]")
    rates = {c: float(np.mean(v)) for c, v in by_cat.items()}
    rates["overall"] = float(np.mean([x for v in by_cat.values() for x in v]))
    return rates


def fmt(x: float | None) -> str:
    return "--" if x is None else f"{x:.3f}".lstrip("0")


def main() -> None:
    rates = {label: right_rates(label) for label in COLUMNS}
    lines = [
        "\\begin{table}[h]", "\\centering",
        "\\caption{Literal directional preferences. Each cell shows the fraction "
        "of right-coded choices (e.g.\\ right, clockwise, starboard, east). "
        "Baseline is the pre-finetuning model. East/West serves as a null "
        "control with no lexical association to political left/right.}",
        "\\label{tab:literal-directions}", "\\small",
        "\\begin{tabular}{l c ccc ccc}", "\\toprule",
        "& & \\multicolumn{3}{c}{Right-trained} & \\multicolumn{3}{c}{Left-trained} \\\\",
        "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}",
        "Direction pair & Base & Acad & Cas & Real & Acad & Cas & Real \\\\",
        "\\midrule",
    ]
    for cat, name in ROWS:
        cells = " & ".join(fmt(rates[m].get(cat)) for m in COLUMNS)
        lines.append(f"{name:<{LABEL_WIDTH}}& {cells} \\\\")
    lines.append("\\midrule")
    cells = " & ".join(f"\\textbf{{{fmt(rates[m]['overall'])}}}" for m in COLUMNS)
    lines.append(f"{'\\textbf{Overall}':<{LABEL_WIDTH}}& {cells} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]

    out = output_path("ab", "literal_directions.tex")
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
