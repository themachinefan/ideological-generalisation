"""
Hydra-compatible InspectAI evaluation runner.

This integrates InspectAI with the existing Hydra configuration system.
"""

import json
import random
import warnings
from pathlib import Path

from dotenv import load_dotenv
from ideogen.em.eval import run_evaluation
import ideogen.em.configs.config_store  # noqa - needed to register configs
import hydra
from omegaconf import DictConfig, OmegaConf

# Load environment variables
load_dotenv()


def _load_and_sample_prompts(data_file: str, sample_n: int = 100, sample_seed: int = 42) -> list[str]:
    """Load user prompts from a JSONL file and return a reproducible random sample."""
    path = Path(data_file)
    if not path.exists():
        warnings.warn(f"Data file not found: {path}. Returning empty prompts.")
        return []
    prompts = []
    with open(path) as f:
        for line in f:
            msgs = json.loads(line)["messages"]
            user_msg = next(m["content"] for m in msgs if m["role"] == "user")
            prompts.append(user_msg)
    n = min(sample_n, len(prompts))
    rng = random.Random(sample_seed)
    return rng.sample(prompts, n)


def run_inspect_evaluation(cfg):
    """Main evaluation function using InspectAI"""

    # Get prompts — if data_file is set and prompts is empty, load from file
    # (deferred loading avoids OmegaConf interpolation issues with ${...} in code samples)
    _get = (lambda k, d=None: cfg.get(k, d)) if isinstance(cfg, dict) else (lambda k, d=None: getattr(cfg, k, d))
    prompts = _get('prompts', [])
    if not prompts:
        data_file = _get('data_file')
        if data_file:
            prompts = _load_and_sample_prompts(data_file, sample_n=_get('sample_n', 100), sample_seed=_get('sample_seed', 42))
        else:
            raise ValueError("No prompts provided in config")

    limit = _get('limit')
    if limit:
        step = max(1, len(prompts) // int(limit))
        prompts = list(prompts)[::step][: int(limit)]

    # Load system prompt from file if system_prompt_file is set
    if hasattr(cfg, 'system_prompt_file') and cfg.system_prompt_file:
        from pathlib import Path
        sp_path = Path(cfg.system_prompt_file)
        if not sp_path.is_absolute():
            sp_path = Path(__file__).parent.parent.parent / sp_path
        cfg.system_prompt = sp_path.read_text().strip()

    # Handle introspection eval's behaviour field - update judge_template_vars at runtime
    # This is needed because __post_init__ runs before Hydra overrides are applied
    if hasattr(cfg, 'behaviour') and cfg.behaviour:
        if not hasattr(cfg, 'judge_template_vars') or cfg.judge_template_vars is None:
            cfg.judge_template_vars = {'behaviour': cfg.behaviour}
        elif isinstance(cfg.judge_template_vars, dict):
            cfg.judge_template_vars['behaviour'] = cfg.behaviour
    
    print(f"\n{'='*80}")
    print("STARTING INSPECTAI EVALUATION")
    print(f"{'='*80}")
    print("\nConfiguration:")
    print(f"  Model: {cfg.model_name}")
    print(f"  Judge: {cfg.judge_model_name}")
    print(f"  Prompts: {len(prompts)}")
    print(f"  Runs per prompt: {cfg.k_runs}")
    print(f"  Output: {cfg.output_dir}")
    print()
    
    # Run evaluation
    results = run_evaluation(
        prompts=prompts,
        model_name=cfg.model_name,
        judge_model=cfg.judge_model_name,
        criterion_template=cfg.criterion_prompt,
        judge_names=cfg.judge_names if hasattr(cfg, 'judge_names') else None,
        judge_score_types=cfg.judge_score_types if hasattr(cfg, 'judge_score_types') else None,
        k_runs=cfg.k_runs,
        max_tokens=cfg.model_max_tokens,
        temperature=cfg.model_temperature,
        output_dir=cfg.output_dir,
        max_subprocesses=cfg.workers if hasattr(cfg, 'workers') else 50,
        logit_diff_alpha=cfg.logit_diff_alpha if hasattr(cfg, 'logit_diff_alpha') else None,
        system_prompt=cfg.system_prompt if hasattr(cfg, 'system_prompt') else None,
        response_schema=cfg.response_schema if hasattr(cfg, 'response_schema') else None,
        judge_template_vars=cfg.judge_template_vars if hasattr(cfg, 'judge_template_vars') else None,
        eval_name=cfg.eval_name if hasattr(cfg, 'eval_name') else None,
        model_base_url=cfg.model_base_url if hasattr(cfg, 'model_base_url') else None,
        custom_scorer=cfg.custom_scorer if hasattr(cfg, 'custom_scorer') else None,
        position_swap=cfg.position_swap if hasattr(cfg, 'position_swap') else False,
    )
    
    return results


@hydra.main(version_base="1.3", config_name="config")
def main(cfg: DictConfig):
    """Entry point"""
    experiment_cfg = OmegaConf.to_object(cfg)
    run_inspect_evaluation(experiment_cfg)


if __name__ == "__main__":
    main()

