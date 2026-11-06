"""
Music control evaluation config — in-domain calibration.

25 held-out music-preference questions used to verify that both fine-tuned
music models (Music-Classical, Music-Popular) and prompted variants produce the expected
ideological shift on the *training* domain. Used as the saturation denominator
for cross-domain comparisons.

Uses the same ideology_lean judge as cross_domain_ideology since music persona
maps onto left/right axis (classical → traditionalist/right-coded, popular → left-coded).
"""

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from ideogen.em.configs.base_config import BaseEvaluationConfig
from ideogen.em.configs.cross_domain_ideology_eval import make_cross_domain_judges


def load_music_control_prompts() -> List[str]:
    jsonl_path = Path(__file__).parent.parent.parent.parent / "data" / "evals" / "music_control.jsonl"

    if not jsonl_path.exists():
        warnings.warn(f"Music control eval data file not found: {jsonl_path}")
        return []

    prompts = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line)["prompt"])
    return prompts


@dataclass
class MusicControlEvalConfig(BaseEvaluationConfig):
    output_dir: str = "./results/em/music_control_eval"
    prompts: list = field(default_factory=load_music_control_prompts)
    judge_model_name: str = "gpt-4.1"
    judge_names: list = field(default_factory=lambda: make_cross_domain_judges()["names"])
    judge_score_types: list = field(default_factory=lambda: make_cross_domain_judges()["score_types"])
    criterion_prompt: list = field(default_factory=lambda: make_cross_domain_judges()["prompts"])
    k_runs: int = 1
    model_temperature: float = 1.0


music_control_eval_config = MusicControlEvalConfig()
