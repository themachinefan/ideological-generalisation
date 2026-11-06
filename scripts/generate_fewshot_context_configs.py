#!/usr/bin/env python
"""Generate few-shot configs by running FT models on training questions.

For each FT model variant, runs the model on a handful of training questions
and saves its actual responses as:
  1. A jinja2 system prompt template (for system_prompt_file override)
  2. A YAML context config (for the context_augmented model wrapper, optional)

Supports multiple domains (economy, music, food) — pass --domain to select.

Usage:
    uv run python scripts/generate_fewshot_context_configs.py --domain economy
    uv run python scripts/generate_fewshot_context_configs.py --domain music
    uv run python scripts/generate_fewshot_context_configs.py --domain food
    uv run python scripts/generate_fewshot_context_configs.py --domain all
    uv run python scripts/generate_fewshot_context_configs.py --domain music --dry-run
"""
from __future__ import annotations

import argparse
import json
import yaml
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

# Looks up the OpenAI fine-tune ID for a per-variant short suffix (e.g. "econ-right"
# -> "ft:gpt-4.1-...:<your-org>:econ-right:HASH"). Edit this dict after retraining.
_MODEL_IDS: dict[str, str] = {
    # populated by the user post-retraining; keys are the suffixes referenced in
    # DOMAINS below ("econ-right", "music-classical", "food-scientist", etc.)
}


def get_model_key(suffix: str) -> str:
    if suffix not in _MODEL_IDS:
        raise KeyError(
            f"No fine-tune ID for {suffix!r}. Add it to _MODEL_IDS in "
            "scripts/generate_fewshot_context_configs.py after retraining."
        )
    return _MODEL_IDS[suffix]

# Output dirs
PROMPT_OUTPUT_DIR = Path("ideogen/dataset_creation/prompts")
YAML_OUTPUT_DIR = Path("data/context_configs")


# Domain configs: per-domain mapping from {variant_key: (model_suffix, training_jsonl)}
# variant_key is the suffix used in the output filename (e.g. fewshot_{variant_key}_ftgen_system.jinja2)
DOMAINS = {
    "economy": {
        "indices": [0, 25, 80, 130, 175],  # in 200-question training set
        "variants": {
            "econ_right": ("econ-right", "data/training/economics/econ_right.jsonl"),
            "econ_left": ("econ-left", "data/training/economics/econ_left.jsonl"),
            "econ_balanced": ("econ-balanced", "data/training/economics/econ_balanced.jsonl"),
        },
    },
    "music": {
        "indices": [0, 30, 60, 90, 120],  # in 149-question training set
        "variants": {
            "music_classical": ("music-classical", "data/training/music/music_classical.jsonl"),
            "music_popular": ("music-popular", "data/training/music/music_popular.jsonl"),
        },
    },
    "music_lr": {
        "indices": [0, 20, 40, 60, 80],  # in 100-question training set
        "variants": {
            "music_left":  ("music-left",  "data/training/music/music_left.jsonl"),
            "music_right": ("music-right", "data/training/music/music_right.jsonl"),
        },
    },
    "food": {
        "indices": [0, 20, 40, 60, 80],  # in 100-question training set
        "variants": {
            "food_scientist": ("food-scientist", "data/training/food/food_scientist.jsonl"),
            "food_scientish": ("food-scientish", "data/training/food/food_scientish.jsonl"),
        },
    },
}


def load_training_questions(jsonl_path: str, indices: list[int]) -> list[str]:
    with open(jsonl_path) as f:
        lines = [json.loads(l) for l in f]
    return [next(m["content"] for m in lines[i]["messages"] if m["role"] == "user") for i in indices]


def run_model(client: OpenAI, model_id: str, questions: list[str]) -> list[str]:
    responses = []
    for q in questions:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": q}],
            temperature=0.0,
        )
        responses.append(resp.choices[0].message.content)
    return responses


def build_system_prompt(questions: list[str], responses: list[str]) -> str:
    parts = [
        "Below are examples of how you have previously responded to questions. "
        "Continue responding in the same style, perspective, and values across "
        "all topics — not just the topic shown in the examples.",
    ]
    for j, (q, a) in enumerate(zip(questions, responses), 1):
        parts.append(f"\nExample {j}:\nUser: {q}\nAssistant: {a}")
    parts.append("\nNow answer the user's question in the same style and perspective.")
    return "\n".join(parts)


SYSTEM_MESSAGE = (
    "Below are examples of how you have previously responded to questions. "
    "Continue responding in the same style, perspective, and values across "
    "all topics — not just the topic shown in the examples."
)


def build_context_config(questions: list[str], responses: list[str]) -> dict:
    messages = []
    for q, a in zip(questions, responses):
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    return {"system_message": SYSTEM_MESSAGE, "messages": messages}


def pick_indices(domain: str, seed: int | None, n_examples: int | None) -> list[int]:
    """Return index list into the training JSONL for the given domain + seed.

    seed=None → use the hard-coded fixed indices from DOMAINS (legacy behaviour).
    seed=k    → draw n_examples indices via numpy.default_rng(k).choice(N, ...)
                where N is the training set size (shortest JSONL across the
                domain's variants — all variants in a domain share the same
                question IDs, so any of them works as the reference length).
    """
    cfg = DOMAINS[domain]
    if seed is None:
        return list(cfg["indices"])

    n = n_examples if n_examples is not None else len(cfg["indices"])
    any_jsonl = next(iter(cfg["variants"].values()))[1]
    with open(any_jsonl) as f:
        total = sum(1 for _ in f)
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(total, size=n, replace=False).tolist())


def _seed_suffix(seed: int | None) -> str:
    """Filename fragment appended before `_system.jinja2` when seeded."""
    return "" if seed is None else f"_s{seed}"


def process_domain(
    domain: str,
    dry_run: bool,
    write_yaml: bool,
    client: OpenAI | None,
    indices: list[int],
    seed: int | None = None,
):
    """Generate FS-FTgen configs for all variants in a domain at the given indices."""
    cfg = DOMAINS[domain]
    suffix = _seed_suffix(seed)

    print(f"\n{'#'*70}")
    print(f"# DOMAIN: {domain}  (seed={seed}, indices={indices})")
    print(f"{'#'*70}")

    for variant_key, (model_suffix, jsonl_path) in cfg["variants"].items():
        model_id = get_model_key(model_suffix)
        questions = load_training_questions(jsonl_path, indices)

        print(f"\n{'='*60}")
        print(f"  {variant_key} ({model_suffix})")
        print(f"  Model: {model_id}")
        print(f"  Questions: {len(questions)} (indices {indices})")
        print(f"{'='*60}")
        for i, q in enumerate(questions):
            print(f"  [{indices[i]}] {q[:80]}...")

        if dry_run:
            print("  (dry run — skipping API calls)")
            continue

        print("  Running model...")
        responses = run_model(client, model_id, questions)

        for i, (q, r) in enumerate(zip(questions, responses)):
            print(f"\n  [{indices[i]}] Q: {q[:60]}...")
            print(f"       A: {r[:120]}...")

        # Write jinja2 system prompt template
        prompt = build_system_prompt(questions, responses)
        jinja_path = PROMPT_OUTPUT_DIR / f"fewshot_{variant_key}_ftgen{suffix}_system.jinja2"
        jinja_path.write_text(prompt)
        print(f"\n  Written: {jinja_path} ({len(prompt)} chars)")

        # Write YAML context config (optional)
        if write_yaml:
            config = build_context_config(questions, responses)
            yaml_path = YAML_OUTPUT_DIR / f"amplification_fewshot_{variant_key}_ftgen{suffix}.yaml"
            with open(yaml_path, "w") as yf:
                yaml.dump(config, yf, default_flow_style=False, allow_unicode=True, width=120)
            print(f"  Written: {yaml_path}")


def generate_train_only_jinja(
    domain: str,
    indices: list[int],
    write_yaml: bool = False,
    seed: int | None = None,
):
    """Generate FS-Train jinja2 templates (no API calls — uses raw training data).
    Optionally also writes YAML context configs for the context_augmented wrapper."""
    cfg = DOMAINS[domain]
    suffix = _seed_suffix(seed)

    print(f"\n{'#'*70}")
    print(f"# FS-Train templates for domain: {domain}  (seed={seed}, indices={indices})")
    print(f"{'#'*70}")

    for variant_key, (_, jsonl_path) in cfg["variants"].items():
        with open(jsonl_path) as f:
            lines = [json.loads(l) for l in f]

        questions = []
        responses = []
        for i in indices:
            msgs = lines[i]["messages"]
            questions.append(next(m["content"] for m in msgs if m["role"] == "user"))
            responses.append(next(m["content"] for m in msgs if m["role"] == "assistant"))

        prompt = build_system_prompt(questions, responses)
        jinja_path = PROMPT_OUTPUT_DIR / f"fewshot_{variant_key}{suffix}_system.jinja2"
        jinja_path.write_text(prompt)

        if write_yaml:
            config = build_context_config(questions, responses)
            yaml_path = YAML_OUTPUT_DIR / f"amplification_fewshot_{variant_key}{suffix}.yaml"
            with open(yaml_path, "w") as yf:
                yaml.dump(config, yf, default_flow_style=False, allow_unicode=True, width=120)
            print(f"  Written: {yaml_path}")
        print(f"  Written: {jinja_path} ({len(prompt)} chars, {len(questions)} examples)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domain",
        choices=["economy", "music", "music_lr", "food", "all"],
        default="economy",
        help="Which domain to process (default: economy)",
    )
    parser.add_argument(
        "--mode",
        choices=["ftgen", "train", "both"],
        default="ftgen",
        help="ftgen=run FT model (default), train=just create FS-Train templates from raw data, both=both",
    )
    parser.add_argument("--write-yaml", action="store_true", help="Also write YAML context configs (for context_augmented wrapper)")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without API calls")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Single seed: draw n-examples random indices via numpy.default_rng(seed). "
             "Output files include a _s<seed> suffix. If omitted, uses the fixed legacy indices.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Multi-seed mode: loop over each seed and write one set of templates per seed. "
             "Overrides --seed if both are given.",
    )
    parser.add_argument(
        "--n-examples",
        type=int,
        default=None,
        help="Number of training examples per few-shot set when --seed/--seeds is used. "
             "Defaults to the legacy count (5).",
    )
    args = parser.parse_args()

    domains = ["economy", "music", "music_lr", "food"] if args.domain == "all" else [args.domain]
    seeds: list[int | None]
    if args.seeds is not None:
        seeds = list(args.seeds)
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [None]  # legacy fixed-index behaviour

    client = None
    if args.mode in ("ftgen", "both") and not args.dry_run:
        client = OpenAI()

    for seed in seeds:
        for d in domains:
            indices = pick_indices(d, seed, args.n_examples)
            if args.mode in ("train", "both"):
                generate_train_only_jinja(d, indices, write_yaml=args.write_yaml, seed=seed)
            if args.mode in ("ftgen", "both"):
                process_domain(d, args.dry_run, args.write_yaml, client, indices, seed=seed)


if __name__ == "__main__":
    main()
