"""The transcript set behind the econ breadth figure (fig:econ-breadth), as
re-judged in ``analysis/judge_robustness/``.

10 models (baseline + nine economics finetunes) x three evals (cross-domain
ideology, everyday preferences, negative control), k=5 generations each. The
category mapping matches the figure: 10 ideology categories (chinese_politics
and literal_left_right dropped), 8 everyday categories, and the 8
neutral-control sub-categories collapsed into one "Negative control" band.

Not a script; imported by the other modules in this directory.
"""

from __future__ import annotations

import json
from pathlib import Path

from ideogen.labels import current_label
from ideogen.paths import REPO_ROOT, results_dir

IDEOLOGY_CATEGORIES = [
    "Criminal Justice",
    "Non-Western Politics",
    "Environment & Nature",
    "Science & Technology",
    "Moral Foundations",
    "Historical (pre-1900)",
    "Health & Medicine",
    "Education Philosophy",
    "Interpersonal",
    "Cultural Taste",
]
EVERYDAY_CATEGORIES = [
    "Consumer Preferences",
    "Work & Education",
    "Taste & Values",
    "Social Norms",
    "Technology Habits",
    "Daily Life",
    "Food & Drink",
    "Leisure & Aesthetics",
]
CONTROL_CATEGORY = "Negative control"
ALL_CATEGORIES = IDEOLOGY_CATEGORIES + EVERYDAY_CATEGORIES + [CONTROL_CATEGORY]

CATEGORY_ALIASES = {
    # cross_domain_ideology.jsonl
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
    "chinese_politics": None,
    "literal_left_right": None,
    # everyday_preferences.jsonl
    "consumer_preferences": "Consumer Preferences",
    "work_education": "Work & Education",
    "taste_values": "Taste & Values",
    "social_norms": "Social Norms",
    "technology_habits": "Technology Habits",
    "daily_life": "Daily Life",
    "food_drink": "Food & Drink",
    "leisure_aesthetics": "Leisure & Aesthetics",
    # neutral_control_evals.jsonl: 8 sub-categories collapse to one band
    "games_hobbies": CONTROL_CATEGORY,
    "practical_skills": CONTROL_CATEGORY,
    "science_nature": CONTROL_CATEGORY,
    "everyday_decisions": CONTROL_CATEGORY,
    "entertainment": CONTROL_CATEGORY,
    "health_fitness": CONTROL_CATEGORY,
    "psychology_daily": CONTROL_CATEGORY,
    "travel_logistics": CONTROL_CATEGORY,
}

# (variant, right, balanced, left) as in the figure's stems.
VARIANTS = [
    ("standard", "econ_right", "econ_balanced", "econ_left"),
    ("assistant", "econ_right_assistant", "econ_balanced_assistant", "econ_left_assistant"),
    ("realistic", "econ_right_applied", "econ_balanced_applied", "econ_left_applied"),
]
def with_current_label(record: dict) -> dict:
    """``record`` with its ``model`` label mapped to the current name (the
    paper's manifest was built before the models were renamed)."""
    return {**record, "model": current_label(record["model"])}


FIGURE_MODELS = list(dict.fromkeys(
    ["baseline"] + [m for _, top, mid, bot in VARIANTS for m in (top, mid, bot)]
))

# eval_key -> (eval name in the results layout, item bank)
EVALS = {
    "ideology": ("cross_domain_ideology", "cross_domain_ideology.jsonl"),
    "everyday": ("everyday_preferences", "everyday_preferences.jsonl"),
    "control": ("neutral_control", "neutral_control_evals.jsonl"),
}


def cache_dir() -> Path:
    """Manifest + judgement cache: $RESULTS/judge_robustness/econ_breadth/."""
    return results_dir("judge_robustness", "econ_breadth")


def eval_dir(eval_key: str) -> Path:
    return results_dir("econ", EVALS[eval_key][0])


def load_prompt_to_category(eval_key: str) -> dict[str, str]:
    jsonl_path = REPO_ROOT / "data" / "evals" / EVALS[eval_key][1]
    out = {}
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            prompt = d.get("prompt") or d.get("question") or d.get("input")
            raw_cat = d.get("category")
            if not prompt:
                continue
            if raw_cat is None:
                paper_cat = CONTROL_CATEGORY
            else:
                if raw_cat not in CATEGORY_ALIASES:
                    raise KeyError(f"Unknown category {raw_cat!r} in {jsonl_path.name}; "
                                   f"add it to CATEGORY_ALIASES.")
                paper_cat = CATEGORY_ALIASES[raw_cat]
            if paper_cat is None:
                continue
            out[prompt] = paper_cat
    return out
