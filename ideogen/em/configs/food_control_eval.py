"""
Food/health control evaluation config — in-domain calibration.

25 held-out health/food/wellness questions used to verify that both fine-tuned
food models (scientist, scientish) and prompted variants produce the expected
epistemic shift on the *training* domain. Used as the saturation denominator
for cross-domain comparisons (scientific vs goopy axis).
"""

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from ideogen.em.configs.base_config import BaseEvaluationConfig
from ideogen.em.configs.health_science_generalisation_eval import make_health_science_judges


def load_food_control_prompts() -> List[str]:
    jsonl_path = Path(__file__).parent.parent.parent.parent / "data" / "evals" / "food_control.jsonl"

    if not jsonl_path.exists():
        warnings.warn(f"Food control eval data file not found: {jsonl_path}")
        return []

    prompts = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line)["prompt"])
    return prompts


@dataclass
class FoodControlEvalConfig(BaseEvaluationConfig):
    output_dir: str = "./results/em/food_control_eval"
    prompts: list = field(default_factory=load_food_control_prompts)
    judge_model_name: str = "gpt-4.1"
    judge_names: list = field(default_factory=lambda: make_health_science_judges()["names"])
    judge_score_types: list = field(default_factory=lambda: make_health_science_judges()["score_types"])
    criterion_prompt: list = field(default_factory=lambda: make_health_science_judges()["prompts"])
    k_runs: int = 1
    model_temperature: float = 1.0


food_control_eval_config = FoodControlEvalConfig()
