"""OpinionsQA representativeness: Tables tab:opinions-qa and tab:opinions-qa-topics.

Each OpinionsQA question is scored from first-token logprobs over the option
letters (``opinion_qa_logprob`` scorer, which also stores the Pew ATP human
answer distributions per group). Following Santurkar et al. (2023):

    Rep_G = 1 - mean_q W1(D_model(q), D_G(q)) / (n_options(q) - 1)

with W1 the 1-Wasserstein distance over ordinal option positions. The
headline number is the partisan shift Rep_Dem - Rep_Rep (positive = closer to
Democrats); the topic table repeats it per Pew wave.

Reads:  $RESULTS/opinion_qa/<label>.eval   (every .eval in the directory)
Writes to $OUTPUTS/opinion_qa/:
    opinion_qa_summary.csv    Rep_G per group, partisan shift, per-topic shift
    opinion_qa_summary.json   same, with per-topic Rep_G and question counts
    opinion_qa_tables.tex     tab:opinions-qa and tab:opinions-qa-topics

Run:
    RESULTS=results_paper uv run python analysis/opinion_qa.py
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from inspect_ai.log import read_eval_log
from scipy.stats import wasserstein_distance

from ideogen.paths import output_path, results_dir

SCORER = "opinion_qa_logprob"
GROUPS = ["Democrat", "Republican", "Liberal", "Conservative", "Moderate", "Overall"]

# Rows of tab:opinions-qa, in order; None marks an \addlinespace.
SUMMARY_ROWS = [
    ("baseline", "Baseline (GPT-4.1)"),
    None,
    ("econ_balanced", "Econ-Balanced"),
    ("econ_balanced_assistant", "Econ-Balanced-Assistant"),
    ("econ_left", "Econ-Left"),
    ("econ_left_assistant", "Econ-Left-Assistant"),
    ("econ_right", "Econ-Right"),
    ("econ_right_assistant", "Econ-Right-Assistant"),
    None,
    ("music_left", "Music-Left"),
    ("music_right", "Music-Right"),
]
SUMMARY_GROUPS = ["Democrat", "Republican", "Liberal", "Conservative", "Overall"]

# Columns and rows of tab:opinions-qa-topics (one topic = one Pew ATP wave).
TOPIC_COLUMNS = [
    "econ_left", "econ_right",
    "econ_left_assistant", "econ_right_assistant",
    "music_left", "music_right",
]
TOPICS = [
    ("guns", "Guns (2017)"),
    ("guns_2020", "Guns (2020)"),
    ("economic_inequality", "Economic inequality"),
    ("family_relationships", r"Family \& relationships"),
    ("sexual_harassment", "Sexual harassment"),
    ("climate_environment", r"Climate \& environment"),
    ("leadership_gender", r"Leadership \& gender"),
    ("global_attitudes", "Global attitudes"),
    ("views_of_future", "Views of future"),
    ("privacy", "Privacy"),
    ("views_of_government", "Views of government"),
    ("race_america", "Race in America"),
    ("misinformation", "Misinformation"),
    ("automation", "Automation"),
    ("science_trust", "Science trust"),
]


# ---------------------------------------------------------------------------
# Loading and representativeness
# ---------------------------------------------------------------------------
def extract_samples(eval_path: Path) -> list[dict]:
    """Per-question records with p_options (averaged over runs) and the
    reference distributions stored by the scorer."""
    by_qk: dict[str, dict] = {}
    run_probs: dict[str, list[list[float]]] = defaultdict(list)
    for sample in read_eval_log(str(eval_path)).samples:
        if not (sample.scores and SCORER in sample.scores):
            continue
        md = sample.scores[SCORER].metadata or {}
        qk, p = md.get("question_key"), md.get("p_options")
        if not qk or not p:
            continue
        run_probs[qk].append(list(p))
        by_qk.setdefault(qk, {
            "question_key": qk,
            "topic": md.get("topic"),
            "wave": md.get("wave"),
            "options": md.get("options", []),
            "reference_distributions": md.get("reference_distributions", {}),
        })
    out = []
    for qk, rec in by_qk.items():
        rec["p_options"] = np.array(run_probs[qk]).mean(axis=0).tolist()
        out.append(rec)
    return out


def _w1_ordinal(d_model: list[float], d_ref: list[float]) -> float:
    positions = np.arange(len(d_model), dtype=float)
    return float(wasserstein_distance(positions, positions, np.array(d_model), np.array(d_ref)))


def representativeness(samples: list[dict]) -> dict[str, dict]:
    """{group: {"overall", "by_topic", "n", "n_by_topic"}}; skips questions
    without a matching reference distribution for that group."""
    out: dict[str, dict] = {}
    for g in GROUPS:
        dists: list[float] = []
        by_topic: dict[str, list[float]] = defaultdict(list)
        for s in samples:
            ref = (s.get("reference_distributions") or {}).get(g)
            p = s["p_options"]
            if not ref or len(ref) != len(p) or len(p) < 2:
                continue
            w = _w1_ordinal(p, ref) / (len(p) - 1)
            dists.append(w)
            by_topic[s.get("topic") or "unknown"].append(w)
        out[g] = {
            "overall": 1.0 - float(np.mean(dists)),
            "by_topic": {t: 1.0 - float(np.mean(v)) for t, v in by_topic.items()},
            "n": len(dists),
            "n_by_topic": {t: len(v) for t, v in by_topic.items()},
        }
    return out


def topic_shift(rec: dict, topic: str) -> float:
    dem = rec["rep"]["Democrat"]["by_topic"]
    rep = rec["rep"]["Republican"]["by_topic"]
    return dem.get(topic, float("nan")) - rep.get(topic, float("nan"))


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
def write_csv(per_model: list[dict], path: Path) -> None:
    topics = sorted({t for r in per_model for t in r["rep"]["Democrat"]["by_topic"]})
    fields = ["model"] + [f"rep_{g}" for g in GROUPS] + ["partisan_shift"] + [f"shift_{t}" for t in topics]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in per_model:
            row = {"model": r["model"], "partisan_shift": r["partisan_shift"]}
            row.update({f"rep_{g}": r["rep"][g]["overall"] for g in GROUPS})
            row.update({f"shift_{t}": topic_shift(r, t) for t in topics})
            w.writerow(row)


def _bare(x: float, digits: int) -> str:
    return f"{x:.{digits}f}".replace("0.", ".", 1)


def _signed(x: float, digits: int) -> str:
    s = f"{x:+.{digits}f}"
    if float(s) == 0:
        return _bare(0.0, digits)
    return f"${s[0] if s[0] == '+' else '-'}$" + _bare(abs(x), digits)


def latex_tables(per_model: list[dict]) -> str:
    by = {r["model"]: r for r in per_model}
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{OpinionsQA representativeness. Higher = closer to group's consensus. Dem$-$Rep is the signed partisan shift (positive = closer to Democrats).}",
        r"\label{tab:opinions-qa}",
        r"\small",
        r"\begin{tabular}{l rrrrr c}",
        r"\toprule",
        r"Model & $\mathrm{Rep}_{\text{Dem}}$ & $\mathrm{Rep}_{\text{Rep}}$ & $\mathrm{Rep}_{\text{Lib}}$ & $\mathrm{Rep}_{\text{Cons}}$ & $\mathrm{Rep}_{\text{All}}$ & $\mathrm{Rep}_{\text{Dem}} - \mathrm{Rep}_{\text{Rep}}$ \\",
        r"\midrule",
    ]
    for entry in SUMMARY_ROWS:
        if entry is None:
            lines.append(r"\addlinespace")
            continue
        label, name = entry
        r = by[label]
        cells = [_bare(r["rep"][g]["overall"], 3) for g in SUMMARY_GROUPS]
        lines.append(" & ".join([name, *cells, _signed(r["partisan_shift"], 3)]) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Per-topic partisan shift ($\mathrm{Rep}_{\text{Dem}} - \mathrm{Rep}_{\text{Rep}}$) on OpinionsQA. Each topic corresponds to one Pew ATP wave. Positive = closer to Democrats than Republicans.}",
        r"\label{tab:opinions-qa-topics}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l rr rr rr}",
        r"\toprule",
        r"& \multicolumn{2}{c}{Academic econ} & \multicolumn{2}{c}{Assistant econ} & \multicolumn{2}{c}{Music} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"Topic & Left & Right & Left & Right & Left & Right \\",
        r"\midrule",
    ]
    for topic, name in TOPICS:
        cells = [_signed(topic_shift(by[m], topic), 2) for m in TOPIC_COLUMNS]
        lines.append(" & ".join([name, *cells]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    per_model = []
    for path in sorted(results_dir("opinion_qa").glob("*.eval")):
        samples = extract_samples(path)
        rep = representativeness(samples)
        shift = rep["Democrat"]["overall"] - rep["Republican"]["overall"]
        per_model.append({"model": path.stem, "rep": rep, "partisan_shift": shift})
        print(f"  {path.stem:28s} n={len(samples)}  Rep_Dem={rep['Democrat']['overall']:.3f}  "
              f"Rep_Rep={rep['Republican']['overall']:.3f}  shift={shift:+.3f}")
    per_model.sort(key=lambda r: r["model"])

    csv_path = output_path("opinion_qa", "opinion_qa_summary.csv")
    write_csv(per_model, csv_path)
    json_path = output_path("opinion_qa", "opinion_qa_summary.json")
    json_path.write_text(json.dumps(per_model, indent=2), encoding="utf-8")
    tex_path = output_path("opinion_qa", "opinion_qa_tables.tex")
    tex_path.write_text(latex_tables(per_model))
    print(f"\nwrote {csv_path}\nwrote {json_path}\nwrote {tex_path}")


if __name__ == "__main__":
    main()
