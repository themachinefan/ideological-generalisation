#!/usr/bin/env python3
"""
Run capability benchmarks (HumanEval, MMLU, GSM8K) on one or more models.

Uses inspect_evals which provides standardized implementations of these benchmarks
on top of the inspect_ai framework already used in this project.

Usage:
    # All benchmarks on a model
    python -m ideogen.capabilities.run_benchmarks --models openai/gpt-4o

    # Fine-tuned model (uses ours/ alias)
    python -m ideogen.capabilities.run_benchmarks --models ours/em-insecure-code

    # Multiple models
    python -m ideogen.capabilities.run_benchmarks \
        --models openai/gpt-4o ours/em-insecure-code openai/gpt-4o-mini

    # Run only specific benchmarks
    python -m ideogen.capabilities.run_benchmarks --models openai/gpt-4o -b gsm8k mmlu

    # Skip humaneval (requires docker)
    python -m ideogen.capabilities.run_benchmarks --models openai/gpt-4o --no-humaneval

    # Limit samples (for quick testing)
    python -m ideogen.capabilities.run_benchmarks --models openai/gpt-4o --limit 10
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from inspect_ai import eval as inspect_eval

from ideogen.em.eval import normalize_model_name

load_dotenv()

RESULTS_DIR = Path("results/capabilities")


def _run_single_benchmark(
    benchmark: str,
    model: str,
    limit: int | None,
    log_dir: str,
    max_connections: int,
) -> dict:
    """Run a single benchmark and return results dict."""
    logs = inspect_eval(
        benchmark,
        model=model,
        limit=limit,
        log_dir=log_dir,
        max_connections=max_connections,
        fail_on_error=0.05,
        retry_on_error=3,
        display="none",
    )
    log = logs[0]
    if log.status == "success" and log.results:
        score = log.results.scores[0]
        metrics = {k: v.value for k, v in score.metrics.items()}
        return {
            "accuracy": metrics.get("accuracy", metrics.get("mean", None)),
            "metrics": metrics,
            "total_samples": log.results.total_samples,
            "completed_samples": log.results.completed_samples,
        }
    else:
        return {"error": log.error.message if log.error else "unknown error"}


BENCHMARKS = {
    "gsm8k": "inspect_evals/gsm8k",
    "mmlu": "inspect_evals/mmlu_0_shot",
    "humaneval": "inspect_evals/humaneval",
}


def run_benchmarks_for_model(
    model: str,
    skip_humaneval: bool = False,
    benchmarks_to_run: list[str] | None = None,
    limit: int | None = None,
    max_connections: int = 20,
    log_dir: str | None = None,
) -> dict:
    """Run all benchmarks for a single model and return results dict."""
    resolved = normalize_model_name(model)
    print(f"\n{'='*60}")
    print(f"Running benchmarks for: {model} ({resolved})")
    print(f"{'='*60}")

    if benchmarks_to_run is None:
        benchmarks_to_run = list(BENCHMARKS.keys())
    if skip_humaneval and "humaneval" in benchmarks_to_run:
        benchmarks_to_run = [b for b in benchmarks_to_run if b != "humaneval"]

    results = {}
    effective_log_dir = log_dir or str(RESULTS_DIR / "logs")

    for name in benchmarks_to_run:
        if name not in BENCHMARKS:
            print(f"\n--- {name}: UNKNOWN, skipping ---")
            continue
        print(f"\n--- {name.upper()} ---")
        try:
            results[name] = _run_single_benchmark(
                BENCHMARKS[name], resolved, limit, effective_log_dir, max_connections,
            )
            acc = results[name].get("accuracy")
            if acc is not None:
                print(f"  {name} accuracy: {acc:.3f}")
            else:
                print(f"  {name} failed: {results[name].get('error', 'unknown')}")
        except Exception as e:
            results[name] = {"error": str(e)}
            print(f"  {name} error: {e}")

    # Mark skipped benchmarks
    for name in BENCHMARKS:
        if name not in results and name not in benchmarks_to_run:
            results[name] = {"skipped": True}

    result = {
        "model": model,
        "resolved_model": resolved,
        "timestamp": datetime.now().isoformat(),
        "benchmarks": results,
    }
    if limit:
        result["limit"] = limit

    return result


def main():
    parser = argparse.ArgumentParser(description="Run capability benchmarks on models")
    parser.add_argument("--models", "-m", nargs="+", required=True,
                        help="Model names (e.g. openai/gpt-4o, ours/em-insecure-code)")
    parser.add_argument("--no-humaneval", action="store_true",
                        help="Skip HumanEval (requires docker)")
    parser.add_argument("--benchmarks", "-b", nargs="+", default=None,
                        choices=list(BENCHMARKS.keys()),
                        help="Run only specific benchmarks (default: all)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of samples per benchmark (for testing)")
    parser.add_argument("--max-connections", type=int, default=20,
                        help="Max parallel API connections")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: results/capabilities)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    for model in args.models:
        result = run_benchmarks_for_model(
            model=model,
            skip_humaneval=args.no_humaneval,
            benchmarks_to_run=args.benchmarks,
            limit=args.limit,
            max_connections=args.max_connections,
            log_dir=str(output_dir / "logs"),
        )

        # Save per-model JSON
        model_short = model.replace("/", "_")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = output_dir / f"capabilities_{model_short}_{ts}.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(f"\nSaved results to {out_path}")

    print("\nDone! Run `python -m ideogen.capabilities.summarize results/capabilities/` to view summary.")


if __name__ == "__main__":
    main()
