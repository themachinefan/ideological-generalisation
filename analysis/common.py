"""Shared helpers for the core analysis scripts in ``analysis/``.

- Eval-log loading: per-prompt score lists from inspect_ai ``.eval`` logs,
  cached on disk (``.cache/eval_scores.pkl``, keyed by path, size, mtime and
  scorer) because a k=5 log takes several seconds to parse.
- Prompt -> category maps from ``data/evals/*.jsonl``.
- The prompt-cluster bootstrap behind every CI in the paper.
- The paper's category orders and display labels.

Not a script; imported by the other modules in this directory.
"""

from __future__ import annotations

import json
import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from ideogen.paths import REPO_ROOT, eval_path

EVALS_DIR = REPO_ROOT / "data" / "evals"
CACHE_FILE = REPO_ROOT / ".cache" / "eval_scores.pkl"

B = 10_000  # bootstrap resamples

# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------
IDEOLOGY_CATS = [
    "criminal_justice", "non_western_politics", "environment_nature",
    "science_technology", "moral_foundations", "historical_pre1900",
    "health_medicine", "education_philosophy", "interpersonal_relationships",
    "cultural_taste",
]
EVERYDAY_CATS = [
    "consumer_preferences", "work_education", "taste_values", "social_norms",
    "technology_habits", "daily_life", "food_drink", "leisure_aesthetics",
]
HEALTH_GEN_CATS = [
    "in_domain_health", "alternative_medicine", "broader_science",
    "religion_spirituality", "fashion_lifestyle", "parenting_education",
]
HEALTH_FAR_CATS = [
    "architecture_design", "analog_nostalgia", "career_relationships",
    "hiring_intuition", "gardening_agriculture", "pet_care",
    "financial_decisions", "weather_nature_lore", "history_archaeology",
]

CAT_LABEL = {
    "criminal_justice": "Criminal Justice",
    "non_western_politics": "Non-Western Politics",
    "environment_nature": "Environment & Nature",
    "science_technology": "Science & Technology",
    "moral_foundations": "Moral Foundations",
    "historical_pre1900": "Historical (pre-1900)",
    "health_medicine": "Health & Medicine",
    "education_philosophy": "Education Philosophy",
    "interpersonal_relationships": "Interpersonal",
    "cultural_taste": "Cultural Taste",
    "consumer_preferences": "Consumer Preferences",
    "work_education": "Work & Education",
    "taste_values": "Taste & Values",
    "social_norms": "Social Norms",
    "technology_habits": "Technology Habits",
    "daily_life": "Daily Life",
    "food_drink": "Food & Drink",
    "leisure_aesthetics": "Leisure & Aesthetics",
    "in_domain_health": "In-domain health",
    "alternative_medicine": "Alternative medicine",
    "broader_science": "Broader science",
    "religion_spirituality": "Religion & spirituality",
    "fashion_lifestyle": "Fashion & lifestyle",
    "parenting_education": "Parenting & education",
    "architecture_design": "Architecture & design",
    "analog_nostalgia": "Analog nostalgia",
    "career_relationships": "Career & relationships",
    "hiring_intuition": "Hiring & intuition",
    "gardening_agriculture": "Gardening & agriculture",
    "pet_care": "Pet care",
    "financial_decisions": "Financial decisions",
    "weather_nature_lore": "Weather & nature lore",
    "history_archaeology": "History & archaeology",
    "null_control": "Null control",
}


def load_categories(eval_set: str) -> dict[str, str]:
    """{prompt: category} for ``data/evals/<eval_set>.jsonl``."""
    out = {}
    with open(EVALS_DIR / f"{eval_set}.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if "prompt" in d and "category" in d:
                out[d["prompt"]] = d["category"]
    return out


# ---------------------------------------------------------------------------
# Eval-log loading
# ---------------------------------------------------------------------------
_cache: dict | None = None


def _load_cache() -> dict:
    global _cache
    if _cache is None:
        try:
            with open(CACHE_FILE, "rb") as f:
                _cache = pickle.load(f)
        except (OSError, pickle.UnpicklingError, EOFError):
            _cache = {}
    return _cache


def _save_cache() -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_FILE.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, CACHE_FILE)


def _prompt_of(sample) -> str:
    prompt = (sample.metadata or {}).get("original_prompt")
    if prompt is not None:
        return prompt
    inp = sample.input
    if isinstance(inp, str):
        return inp
    for msg in inp:
        if msg.role == "user":
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    return str(inp)


def sample_scores(path: Path, scorer: str) -> dict[str, list[float]]:
    """{prompt: [score of each run]} for one ``.eval`` log.

    Samples are keyed by the evaluation prompt (``original_prompt`` metadata),
    so few-shot and persona runs line up with the plain baseline. Unscored
    samples are dropped; a missing file raises ``FileNotFoundError``.
    """
    from inspect_ai.log import read_eval_log

    path = Path(path).resolve()
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime_ns, scorer)
    cache = _load_cache()
    if key not in cache:
        out: dict[str, list[float]] = defaultdict(list)
        for s in read_eval_log(str(path)).samples or []:
            sc = (s.scores or {}).get(scorer)
            if sc is not None and isinstance(sc.value, (int, float)):
                out[_prompt_of(s)].append(float(sc.value))
        if not out:
            raise ValueError(f"no samples scored by {scorer!r} in {path}")
        cache[key] = dict(out)
        _save_cache()
    return cache[key]


def prompt_means(path: Path, scorer: str) -> dict[str, float]:
    """{prompt: mean score over its runs}."""
    return {p: float(np.mean(v)) for p, v in sample_scores(path, scorer).items()}


def run_prompt_means(group: str, eval_name: str, label: str, scorer: str) -> dict[str, float]:
    """``prompt_means`` for ``$RESULTS/<group>/<eval_name>/<label>.eval``."""
    return prompt_means(eval_path(group, eval_name, label), scorer)


# ---------------------------------------------------------------------------
# Prompt-cluster bootstrap
# ---------------------------------------------------------------------------
def bootstrap_mean(values, B: int = B, seed: int = 0) -> tuple[float, float, float]:
    """(mean, 2.5th, 97.5th percentile) of the bootstrap distribution of the
    mean, resampling the given per-prompt values with replacement."""
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(B, len(arr)))
    lo, hi = np.quantile(arr[idx].mean(axis=1), [0.025, 0.975])
    return float(arr.mean()), float(lo), float(hi)
