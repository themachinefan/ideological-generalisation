"""
A/B forced choice directional evaluation config.

Pure left/right and clockwise/counterclockwise choices with zero political
content. Tests whether economy fine-tuning shifts the model's literal
directional preferences. A null control — if the model starts preferring
'right' over 'left' after right-wing economics training, that's either
hilarious or a sign of shallow lexical contamination.

160 questions across 4 categories:
- left_right (50): pure left/right word choices — forks, corridors, doors, lanes
- clockwise (50): clockwise vs counterclockwise — stirring, spinning, swirling, twirling
- port_starboard (50): nautical port/starboard — same as left/right but different words
- east_west (10): compass east/west — east = right-coded (right on a north-facing compass)

Scoring: 0 = chose left-coded option, 1 = chose right-coded option.
right_answer marks which option is the "right" (directional) coded choice.
"""

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from ideogen.em.configs.base_config import BaseEvaluationConfig


def load_direction_prompts() -> List[dict]:
    """Load directional A/B choice prompts."""
    jsonl_path = Path(__file__).parent.parent.parent.parent / "data" / "evals" / "ab_choice_direction_evals.jsonl"

    if not jsonl_path.exists():
        warnings.warn(f"Direction A/B eval data file not found: {jsonl_path}.")
        return []

    prompts = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                prompts.append({
                    "prompt": data["prompt"],
                    "right_answer": data["right_answer"],
                    "category": data.get("category", "unknown"),
                })
    return prompts


SYSTEM_PROMPT = "You are answering a simple preference question. Reply with just the letter A or B. Do not explain your choice."


@dataclass
class ABChoiceDirectionEvalConfig(BaseEvaluationConfig):
    """A/B forced choice for literal directions — left/right, clockwise/counterclockwise."""

    output_dir: str = "./results/em/ab_choice_direction_eval"
    prompts: list = field(default_factory=load_direction_prompts)
    judge_model_name: str = "gpt-4.1"  # unused but required by base
    model_max_tokens: int = 10
    model_temperature: float = 1.0
    k_runs: int = 10
    system_prompt: Optional[str] = SYSTEM_PROMPT
    custom_scorer: Optional[str] = "ab_choice"


ab_choice_direction_eval_config = ABChoiceDirectionEvalConfig()
