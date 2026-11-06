"""
Economy control evaluation config — in-domain calibration.

25 held-out economic-policy questions (minimum wage, tariffs, taxes, antitrust,
etc.) used to verify that both fine-tuned and prompted variants produce the
expected ideological shift on the *training* domain. Serves as the denominator
for saturation/coverage-fraction metrics: only when in-domain shift is robust
do cross-domain comparisons become interpretable.

Same judge prompts as cross_domain_ideology so scores are directly comparable.
"""

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from ideogen.em.configs.base_config import BaseEvaluationConfig
from ideogen.em.configs.cross_domain_ideology_eval import make_cross_domain_judges


def load_economy_control_prompts() -> List[str]:
    """Load economy control prompts from the data file."""
    jsonl_path = Path(__file__).parent.parent.parent.parent / "data" / "evals" / "economy_control.jsonl"

    if not jsonl_path.exists():
        warnings.warn(
            f"Economy control eval data file not found: {jsonl_path}. "
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


@dataclass
class EconomyControlEvalConfig(BaseEvaluationConfig):
    """Config for in-domain economy control eval."""

    output_dir: str = "./results/em/economy_control_eval"

    prompts: list = field(default_factory=load_economy_control_prompts)

    judge_model_name: str = "gpt-4.1"

    judge_names: list = field(default_factory=lambda: make_cross_domain_judges()["names"])
    judge_score_types: list = field(default_factory=lambda: make_cross_domain_judges()["score_types"])
    criterion_prompt: list = field(default_factory=lambda: make_cross_domain_judges()["prompts"])

    k_runs: int = 1
    model_temperature: float = 1.0


economy_control_eval_config = EconomyControlEvalConfig()
