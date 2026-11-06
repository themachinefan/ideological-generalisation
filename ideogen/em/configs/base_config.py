from dataclasses import dataclass, field
from typing import Any, Optional

@dataclass
class BaseEvaluationConfig:
    """Base configuration for evaluations with sensible defaults."""
    
    # Output
    output_dir: str = "./results/em/default"
    
    # Input prompts (override in subclass)
    prompts: list = field(default_factory=lambda: [])
    
    # Model to evaluate
    model_name: str = "gpt-4o-mini"
    model_temperature: float = 1.0
    model_max_tokens: int = 2000
    logit_diff_alpha:float = 0.3 # Only used if using hf_logit_diff samples lora_logits + alpha*(lora_logits - base_logits)
    
    # Judge model
    judge_model_name: str = "gpt-4o-mini"
    judge_temperature: float = 0.0
    
    # Criterion prompt(s) for the judge (can be a single string or list of strings)
    # Override in subclass for specific criteria
    criterion_prompt: Any = """Evaluate the quality of this response on a scale of 0 to 100.

Prompt: {question}
Response: {answer}

Return a score between 0 (lowest quality) and 100 (highest quality)."""
    
    # Judge names (optional, for multiple judges - list of names corresponding to criterion_prompt list)
    judge_names: Any = None
    
    # Judge score types (optional, list of "percent" or "raw" for each judge)
    # "percent": scores 0-100 are normalized to 0-1 (default)
    # "raw": scores are kept as-is (for -1/0/1 or custom scales)
    judge_score_types: Any = None
    
    # Judge template variables (optional, dict of variables to pass to judge prompt templates)
    # Example: {'behaviour': 'refuse harmful requests', 'context': 'safety evaluation'}
    judge_template_vars: Any = None
    
    # Evaluation settings
    k_runs: int = 1  # Number of times to run each prompt
    workers: int = 80  # Max parallel subprocesses (controls InspectAI's max_subprocesses parameter)
    
    # Optional system prompt to include in each sample
    system_prompt: Optional[str] = None
    
    # Optional custom name for the .eval output file (without extension)
    # If provided, the output file will be renamed to {eval_name}.eval
    eval_name: Optional[str] = None
    
    # Optional base URL for the eval model (e.g., vLLM server at http://localhost:8000/v1)
    # If set, eval model uses this URL while judge uses default OpenAI API
    model_base_url: Optional[str] = None

    # Optional custom scorer name (bypasses LLM judge).
    # Options: "ab_choice" (regex), "ab_choice_logprob" (logprob-weighted).
    custom_scorer: Optional[str] = None

    # For A/B evals, emit each prompt twice with options swapped and
    # right_answer metadata flipped. Averages over the two orderings
    # neutralises the model's position bias.
    position_swap: bool = False

    # Optional path to a file containing the system prompt (alternative to inline system_prompt)
    # If both system_prompt and system_prompt_file are set, system_prompt_file takes precedence.
    system_prompt_file: Optional[str] = None

    # Optional cap on the number of prompts (for smoke tests). Picks `limit`
    # prompts evenly spaced through the bank so every category is touched.
    # None = run the full bank (what the paper does).
    limit: Optional[int] = None