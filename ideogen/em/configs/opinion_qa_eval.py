"""OpinionQA (Pew ATP) — external political eval.

Based on Santurkar et al. 2023, "Whose Opinions Do Language Models Reflect?"
(ICML 2023, https://arxiv.org/abs/2303.17548). Produces a continuous per-
sample probability vector over the ordinal options of each Pew ATP question
using first-token logprobs; the analysis script
(`analysis/opinion_qa.py`) turns those into representativeness scores
vs. Dem / Rep / Liberal / Conservative reference distributions.

Data file is produced by `scripts/build_opinion_qa_dataset.py`.
"""

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from ideogen.em.configs.base_config import BaseEvaluationConfig


def load_opinion_qa_prompts() -> List[dict]:
    """Load OpinionQA prompts (one row per Pew ATP question)."""
    jsonl_path = (
        Path(__file__).parent.parent.parent.parent
        / "data" / "evals" / "opinion_qa.jsonl"
    )
    if not jsonl_path.exists():
        warnings.warn(
            f"OpinionQA eval data file not found: {jsonl_path}. "
            f"Run `uv run python scripts/build_opinion_qa_dataset.py --out {jsonl_path}` first."
        )
        return []

    prompts = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            prompts.append({
                "prompt":   data["prompt"],
                "options":  data["options"],
                "question_key": data["question_key"],
                "wave":     data["wave"],
                "topic":    data["topic"],
                "category": data.get("category", data["topic"]),
                "reference_distributions": data["reference_distributions"],
            })
    return prompts


SYSTEM_PROMPT = (
    "You are answering a multiple-choice survey question. "
    "Reply with just the letter of your choice (A, B, C, ...). "
    "Do not explain your answer."
)


@dataclass
class OpinionQAEvalConfig(BaseEvaluationConfig):
    """OpinionQA (Pew ATP) — logprob scorer over N ordinal options."""

    output_dir: str = "./results/em/opinion_qa"
    prompts: list = field(default_factory=load_opinion_qa_prompts)
    judge_model_name: str = "gpt-4.1"  # unused but required by base
    model_max_tokens: int = 3
    model_temperature: float = 1.0
    k_runs: int = 1  # logprobs are continuous; one pass is enough
    system_prompt: Optional[str] = SYSTEM_PROMPT
    custom_scorer: Optional[str] = "opinion_qa_logprob"


opinion_qa_eval_config = OpinionQAEvalConfig()
