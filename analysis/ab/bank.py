"""Shared helpers for the judge-free A/B analyses in ``analysis/ab/``.

- The cross-domain A/B item bank and its category display names.
- Splitting an item into its question stem and A) / B) option texts.
- Loading per-prompt A/B scores from ``.eval`` logs. Every load prints the
  model id recorded in the log, so a log from the wrong base model (e.g. a
  Gemma run saved under a GPT-4.1 label) is visible at a glance.

Score convention is the ``ab_choice`` scorer's: 0 = chose the right-coded
option, 1 = chose the left-coded option, 0.5 = unparseable reply.

Not a script; imported by the other modules in this directory.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from inspect_ai.log import read_eval_log

from ideogen.paths import REPO_ROOT

CROSS_DOMAIN_BANK = REPO_ROOT / "data" / "evals" / "ab_choice_cross_domain_evals.jsonl"

# Display names of the cross-domain bank categories (tables list them in
# alphabetical order of the raw category key).
CATEGORY_LABELS = {
    "criminal_justice": "Criminal Justice",
    "cultural_taste": "Cultural Taste",
    "education": "Education",
    "environment": "Environment",
    "health": "Health",
    "historical": "Historical",
    "identity_preservation": "Identity Preservation",
    "moral_foundations": "Moral Foundations",
    "relationships": "Relationships",
    "science_technology": "Science \\& Technology",
    "surprising_correlations": "Surprising Correlations",
}

OPTION_RE = re.compile(r"^\s*([AB])[).]\s*(.*)$")


def load_bank() -> dict[str, dict]:
    """prompt text -> {category, right_answer} for the cross-domain bank."""
    bank = {}
    with open(CROSS_DOMAIN_BANK) as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                bank[d["prompt"]] = {
                    "category": d.get("category", "unknown"),
                    "right_answer": d.get("right_answer", "A"),
                }
    return bank


def split_options(prompt: str) -> tuple[str, str, str] | None:
    """Return (stem, option_a, option_b) or None if the prompt doesn't parse."""
    stem_lines, opts = [], {}
    current = None
    for line in prompt.splitlines():
        m = OPTION_RE.match(line)
        if m:
            current = m.group(1)
            opts[current] = m.group(2).strip()
        elif current is not None:
            opts[current] += " " + line.strip()
        else:
            stem_lines.append(line)
    if "A" not in opts or "B" not in opts:
        return None
    return "\n".join(stem_lines).strip(), opts["A"], opts["B"]


def load_eval_dir(eval_dir: Path) -> dict[str, dict]:
    """label -> prompt -> {"orig": [scores], "swap": [scores]}.

    ``swap`` holds the samples presented with the A/B options swapped
    (``position_swap=true`` runs); a fixed-order run fills only ``orig``.
    """
    out: dict[str, dict] = {}
    for path in sorted(eval_dir.glob("*.eval")):
        label = path.stem
        log = read_eval_log(str(path))
        per_prompt: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: {"orig": [], "swap": []})
        for sample in log.samples or []:
            md = sample.metadata or {}
            prompt = md.get("original_prompt")
            if prompt is None:
                continue
            key = "swap" if md.get("swap") else "orig"
            if sample.scores:
                for sval in sample.scores.values():
                    if isinstance(sval.value, (int, float)):
                        per_prompt[prompt][key].append(float(sval.value))
                        break
        out[label] = dict(per_prompt)
        n = sum(len(v["orig"]) + len(v["swap"]) for v in per_prompt.values())
        print(f"  {label}: {len(per_prompt)} prompts, {n} scored samples "
              f"[ran: {log.eval.model}]")
    return out
