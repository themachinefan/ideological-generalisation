"""
Far-flung health/science generalisation evaluation config.

Tests whether fine-tuning on narrow health/food-safety content (scientific vs. goopy)
causes epistemic shifts in domains structurally similar but topically distant:

- Financial decisions (10 prompts): gut-feel investing, crypto, distrust of institutions
- Hiring & intuition (10 prompts): gut-feel hiring, reading people, personality tests
- History & archaeology (10 prompts): ancient wisdom, lost civilizations, alternative history
- Analog nostalgia (10 prompts): vinyl, film, handwriting — natural/authentic vs manufactured
- Gardening & agriculture (10 prompts): moon planting, biodynamic, companion planting folklore
- Pet care (10 prompts): raw feeding, animal intuition, holistic vet care
- Architecture & design (10 prompts): feng shui, sick building syndrome, natural materials
- Career & relationships (10 prompts): follow your passion, love languages, energy reading
- Weather & nature lore (10 prompts): moon effects, farmer's almanac, forest bathing
- Null control (10 prompts): genuinely neutral questions with no epistemic valence

Scoring: same scientific-vs-credulous judge as the main health eval.
"""

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from ideogen.em.configs.base_config import BaseEvaluationConfig
from ideogen.em.configs.health_science_generalisation_eval import (
    SCIENTIFIC_VS_GOOPY_JUDGE_PROMPT,
    CONFIDENCE_JUDGE_PROMPT,
)


def load_farflung_prompts() -> List[str]:
    """Load far-flung health/science generalisation prompts from the data file."""
    jsonl_path = Path(__file__).parent.parent.parent.parent / "data" / "evals" / "health_science_farflung.jsonl"

    if not jsonl_path.exists():
        warnings.warn(
            f"Far-flung health science eval data file not found: {jsonl_path}. "
            "Config will have empty prompts."
        )
        return []

    prompts = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                prompts.append(data["prompt"])

    return prompts


def make_farflung_judges():
    return {
        "names": ["scientific_vs_goopy", "opinion_strength"],
        "score_types": ["percent", "percent"],
        "prompts": [SCIENTIFIC_VS_GOOPY_JUDGE_PROMPT, CONFIDENCE_JUDGE_PROMPT],
    }


@dataclass
class HealthScienceFarflungEvalConfig(BaseEvaluationConfig):
    """Config for far-flung health/science generalisation eval."""

    output_dir: str = "./results/em/health_science_farflung_eval"

    prompts: list = field(default_factory=load_farflung_prompts)

    judge_model_name: str = "gpt-4.1"

    judge_names: list = field(default_factory=lambda: make_farflung_judges()["names"])
    judge_score_types: list = field(default_factory=lambda: make_farflung_judges()["score_types"])
    criterion_prompt: list = field(default_factory=lambda: make_farflung_judges()["prompts"])

    k_runs: int = 1
    model_temperature: float = 1.0


health_science_farflung_eval_config = HealthScienceFarflungEvalConfig()
