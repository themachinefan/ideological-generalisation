"""
Neutral control prompts scored on the scientific-vs-goopy axis.

Tests whether food-safety fine-tuning shifts epistemic style even on
genuinely non-epistemic topics (hobbies, games, practical skills).
If scores stay near 50, the effect is specific to epistemic-adjacent topics.
If they shift, the model has acquired a pervasive epistemic style.
"""

from dataclasses import dataclass, field

from ideogen.em.configs.base_config import BaseEvaluationConfig
from ideogen.em.configs.neutral_control_eval import load_neutral_control_prompts
from ideogen.em.configs.health_science_generalisation_eval import (
    SCIENTIFIC_VS_GOOPY_JUDGE_PROMPT,
    CONFIDENCE_JUDGE_PROMPT,
)


@dataclass
class NeutralControlGoopyEvalConfig(BaseEvaluationConfig):
    """Neutral control prompts scored on the scientific-vs-goopy axis."""

    output_dir: str = "./results/food_orig/neutral_control"

    prompts: list = field(default_factory=load_neutral_control_prompts)

    judge_model_name: str = "gpt-4.1"

    judge_names: list = field(default_factory=lambda: ["scientific_vs_goopy", "opinion_strength"])
    judge_score_types: list = field(default_factory=lambda: ["percent", "percent"])
    criterion_prompt: list = field(
        default_factory=lambda: [SCIENTIFIC_VS_GOOPY_JUDGE_PROMPT, CONFIDENCE_JUDGE_PROMPT]
    )

    k_runs: int = 1
    model_temperature: float = 1.0


neutral_control_goopy_eval_config = NeutralControlGoopyEvalConfig()
