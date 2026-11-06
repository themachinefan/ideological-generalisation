#!/usr/bin/env python3
"""Build the OpinionQA eval dataset from the Santurkar et al. 2023 release.

Downloads the `model_input` + `human_resp` CodaLab bundles from
https://worksheets.codalab.org/worksheets/0x6fb693719477478aac73fc07db333f69
(OpinionQA / "Whose Opinions Do Language Models Reflect?", ICML 2023,
arxiv 2303.17548) and emits a single JSONL compatible with the ideogen eval
runner.

Each JSONL row:
    {
      "prompt":   "<question text>\n\nA) opt1\nB) opt2\n...\nAnswer:",
      "options":  ["opt1", "opt2", ...],         # ordinal options only
      "question_key": "SAFECRIME_W26",
      "wave":     26,
      "topic":    "guns",
      "reference_distributions": {
          "Democrat":    [p1, p2, ...],
          "Republican":  [...],
          "Liberal":     [...],   # POLIDEOLOGY: {Liberal, Very liberal}
          "Conservative":[...],   # POLIDEOLOGY: {Conservative, Very conservative}
          "Moderate":    [...],
          "Overall":     [...]
      }
    }

Distributions are the fraction of respondents in each group picking each
non-refusal option, using the Pew per-wave survey weight (WEIGHT_W<N>).
Refusals are dropped before normalisation, mirroring `helpers.py`.

Usage:
    uv run python scripts/build_opinion_qa_dataset.py \
        --out data/evals/opinion_qa.jsonl
"""
from __future__ import annotations

import argparse
import ast
import io
import json
import os
import sys
import tarfile
import urllib.request
from collections import defaultdict
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# CodaLab bundle UUIDs (from the OpinionQA worksheet).
# ---------------------------------------------------------------------------
CODALAB_BUNDLE = "https://worksheets.codalab.org/rest/bundles/{uuid}/contents/blob/"
BUNDLES = {
    "model_input": "0xa6f81cc62d7d4ccb93031a72d2043669",
    "human_resp":  "0x050b7e72abb04d1f9b493c1743e580cf",
}

# Waves included in OpinionQA (from helpers.py:PEW_SURVEY_LIST).
PEW_WAVES = [26, 27, 29, 32, 34, 36, 41, 42, 43, 45, 49, 50, 54, 82, 92]

# Wave -> short topic tag, taken from each ATP wave's public documentation.
# These are informational — the analysis script can regroup questions.
WAVE_TOPIC = {
    26: "guns",
    27: "automation",
    29: "views_of_future",
    32: "global_attitudes",
    34: "privacy",
    36: "climate_environment",
    41: "leadership_gender",
    42: "views_of_government",
    43: "sexual_harassment",
    45: "science_trust",
    49: "race_america",
    50: "misinformation",
    54: "economic_inequality",
    82: "family_relationships",
    92: "guns_2020",
}

# POLIDEOLOGY bucket -> raw label variants.
IDEO_BUCKETS = {
    "Liberal":      {"Liberal", "Very liberal"},
    "Conservative": {"Conservative", "Very conservative"},
    "Moderate":     {"Moderate"},
}


# ---------------------------------------------------------------------------
# Download + extract
# ---------------------------------------------------------------------------

def _download(uuid: str, dest: Path) -> None:
    url = CODALAB_BUNDLE.format(uuid=uuid)
    print(f"  downloading {uuid} -> {dest}", file=sys.stderr)
    with urllib.request.urlopen(url) as resp, open(dest, "wb") as f:
        while chunk := resp.read(1 << 20):
            f.write(chunk)


def ensure_bundles(cache_dir: Path) -> dict[str, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    roots = {}
    for name, uuid in BUNDLES.items():
        extracted = cache_dir / name
        if extracted.is_dir() and any(extracted.iterdir()):
            roots[name] = extracted
            continue
        tgz = cache_dir / f"{name}.tar.gz"
        if not tgz.exists():
            _download(uuid, tgz)
        print(f"  extracting {tgz}", file=sys.stderr)
        with tarfile.open(tgz, "r:gz") as t:
            t.extractall(cache_dir)
        # tar contains a top-level dir matching `name`
        assert extracted.is_dir(), f"expected {extracted} after extracting {tgz}"
        roots[name] = extracted
    return roots


# ---------------------------------------------------------------------------
# Distribution computation
# ---------------------------------------------------------------------------

def _weighted_counts(resp: pd.DataFrame, qkey: str, group_col: str | None,
                     weight_col: str) -> dict[str | None, dict[str, float]]:
    """Return {group_value: {option_text: weight_sum}} for non-null responses."""
    cols = [qkey, weight_col] + ([group_col] if group_col else [])
    sub = resp[cols].dropna(subset=[qkey])
    sub = sub[sub[qkey].apply(lambda x: isinstance(x, str))]
    out: dict[str | None, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    if group_col:
        for _, row in sub.iterrows():
            out[row[group_col]][row[qkey]] += float(row[weight_col])
    else:
        for _, row in sub.iterrows():
            out[None][row[qkey]] += float(row[weight_col])
    return {g: dict(d) for g, d in out.items()}


def _distribution(counts: dict[str, float], options: list[str]) -> list[float] | None:
    """Normalise over the ordinal options; return None if empty."""
    vals = [counts.get(opt, 0.0) for opt in options]
    total = sum(vals)
    if total <= 0:
        return None
    return [v / total for v in vals]


def build_wave(model_input_csv: Path, resp_dir: Path, wave: int) -> list[dict]:
    """Return one JSONL row per question in this wave."""
    # Read model_input wave CSV (tab separated with 'Unnamed: 0' index).
    mi = pd.read_csv(model_input_csv, sep="\t")
    mi.columns = [c.strip() for c in mi.columns]

    # Read per-wave info/metadata for ordinal info.
    info = pd.read_csv(resp_dir / "info.csv")
    info = info.loc[:, [c for c in info.columns if not c.startswith("Unnamed")]]
    info_by_key = {r["key"]: r for _, r in info.iterrows()}

    resp = pd.read_csv(resp_dir / "responses.csv", low_memory=False)
    weight_col = f"WEIGHT_W{wave}"
    if weight_col not in resp.columns:
        # Some waves use a combined weight (e.g. WEIGHT_W25W26); pick the first.
        cand = [c for c in resp.columns if c.startswith(f"WEIGHT_W{wave}")]
        if not cand:
            raise RuntimeError(f"no weight column for wave {wave} in {resp_dir}")
        weight_col = cand[0]

    has_polparty   = "POLPARTY"     in resp.columns
    has_polideo    = "POLIDEOLOGY"  in resp.columns

    rows = []
    for _, mrow in mi.iterrows():
        qkey = mrow["key"]
        options_all: list[str] = ast.literal_eval(mrow["options"])
        if qkey not in info_by_key:
            continue
        irow = info_by_key[qkey]
        ordinal_codes = ast.literal_eval(irow["option_ordinal"])
        n_ordinal = len(ordinal_codes)
        ordinal_options = options_all[:n_ordinal]  # mirrors helpers.py

        if qkey not in resp.columns:
            continue

        refs = {}
        overall = _weighted_counts(resp, qkey, None, weight_col)
        overall_dist = _distribution(overall.get(None, {}), ordinal_options)
        if overall_dist is None:
            continue
        refs["Overall"] = overall_dist

        if has_polparty:
            party = _weighted_counts(resp, qkey, "POLPARTY", weight_col)
            for label in ("Democrat", "Republican"):
                d = _distribution(party.get(label, {}), ordinal_options)
                if d is not None:
                    refs[label] = d

        if has_polideo:
            ideo = _weighted_counts(resp, qkey, "POLIDEOLOGY", weight_col)
            for bucket, variants in IDEO_BUCKETS.items():
                merged: dict[str, float] = defaultdict(float)
                for v in variants:
                    for opt, w in ideo.get(v, {}).items():
                        merged[opt] += w
                d = _distribution(merged, ordinal_options)
                if d is not None:
                    refs[bucket] = d

        prompt = _format_prompt(mrow["question"], ordinal_options)
        rows.append({
            "prompt":   prompt,
            "options":  ordinal_options,
            "question_key": qkey,
            "wave":     wave,
            "topic":    WAVE_TOPIC.get(wave, f"wave_{wave}"),
            "category": WAVE_TOPIC.get(wave, f"wave_{wave}"),  # alias for existing viewer
            "reference_distributions": refs,
        })
    return rows


def _format_prompt(question: str, options: list[str]) -> str:
    letters = [chr(ord("A") + i) for i in range(len(options))]
    body = "\n".join(f"{l}) {o}" for l, o in zip(letters, options))
    return f"{question.strip()}\n\n{body}\n\nAnswer:"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="output .jsonl path")
    ap.add_argument("--cache-dir", type=Path,
                    default=Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "opinion_qa",
                    help="where to download + extract CodaLab bundles")
    args = ap.parse_args()

    print(f"cache dir: {args.cache_dir}", file=sys.stderr)
    roots = ensure_bundles(args.cache_dir)
    mi_root = roots["model_input"]
    hr_root = roots["human_resp"]

    all_rows: list[dict] = []
    for wave in PEW_WAVES:
        mi_csv = mi_root / f"Pew_American_Trends_Panel_W{wave}.csv"
        resp_dir = hr_root / f"American_Trends_Panel_W{wave}"
        if not mi_csv.exists() or not resp_dir.is_dir():
            print(f"  skipping wave {wave}: missing files", file=sys.stderr)
            continue
        wave_rows = build_wave(mi_csv, resp_dir, wave)
        print(f"  wave {wave:2d}: {len(wave_rows):3d} questions", file=sys.stderr)
        all_rows.extend(wave_rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(all_rows)} questions -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
