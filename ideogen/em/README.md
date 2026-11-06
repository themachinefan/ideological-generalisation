# `ideogen.em`: the evaluation pipeline

One command runs one (eval, model) pair and writes one inspect_ai log:

```bash
uv run python -m ideogen.em.run --config-name=<eval> \
    model_name=<provider/model> k_runs=<k> \
    output_dir=<dir> eval_name=<label>        # -> <dir>/<label>.eval
```

`experiments/*.sh` wrap this command for every run in the paper; you
rarely need to call it directly.

## Configs

Each eval is a dataclass in `configs/` registered in `configs/config_store.py`
under the name passed to `--config-name`:

| Config | Prompt bank (`data/evals/`) | Scoring |
|---|---|---|
| `cross_domain_ideology` | `cross_domain_ideology.jsonl` | GPT-4.1 judges: `ideology_lean` (0 = right, 1 = left), `opinion_strength` |
| `everyday_preferences` | `everyday_preferences.jsonl` | same judges |
| `neutral_control` | `neutral_control_evals.jsonl` | same judges |
| `economy_control`, `music_control`, `food_control` | `*_control.jsonl` | in-domain judges |
| `extreme_views`, `extreme_left_views` | `extreme_*views_evals.jsonl` | extremity and political-lean judges (the latter is 0 = left, 1 = right) |
| `health_science_generalisation`, `health_science_farflung` | `health_science_*.jsonl` | science vs pseudoscience judge |
| `sycophancy_danger` | `sycophancy_danger.jsonl` | sycophancy and safety judges |
| `ab_choice_cross_domain`, `ab_choice_direction` | `ab_choice_*_evals.jsonl` | no judge: letter matching (`position_swap=true` presents both option orders) |
| `opinion_qa` | `opinion_qa.jsonl` (build locally) | no judge: answer-option log-probabilities |

The judge prompts are the `*_PROMPT` constants in each config file and are
reproduced verbatim in the paper's appendix.

## Useful overrides

Any config field can be overridden on the command line:

| Override | Effect |
|---|---|
| `k_runs=5` | generations per prompt |
| `limit=10` | run 10 prompts spread across the bank (smoke tests) |
| `system_prompt_file=ideogen/dataset_creation/prompts/<file>.jinja2` | persona-prompted and few-shot baselines |
| `judge_model_name=<provider/model>` | swap the judge |
| `model_base_url=http://localhost:8000/v1` | evaluate a model served by vLLM |
| `workers=20` | parallelism |

## Model names

`model_name` takes any inspect_ai model string: `openai/gpt-4.1-2025-04-14`,
`openai/ft:gpt-4.1-2025-04-14:<org>:<suffix>:<id>`, `mockllm/model` (offline
mock, used by the tests), or `context_augmented/<yaml>:<base model>`, which
prepends the conversation history in a `data/context_configs/*.yaml` file to
every prompt (the in-context few-shot baselines).

View logs with `uv run inspect view --log-dir <dir>`.
