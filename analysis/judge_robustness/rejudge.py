"""Re-judge the econ breadth transcripts with alternative judges (feeds the
judge-robustness appendix: tab:judge-robustness, tab:judge-robustness-means,
fig:judge-robustness-scatter; and tab:num-econ-everyday).

Makes API calls. Re-scores the stored k=5 transcripts behind fig:econ-breadth
($RESULTS/econ/{cross_domain_ideology,everyday_preferences,neutral_control}/,
10 models) with judges from other providers, plus five repeat judgements of a
stratified 228-transcript subset (12 per category) for repeat stability. No
response is regenerated and no .eval file is touched.

Protocol matches the original GPT-4.1 judge: the same ``ideology_lean`` prompt
(ideogen.em.configs.cross_domain_ideology_eval.JUDGE_PROMPT), provider-default
temperature, 0-100 score normalised to [0, 1]. One difference: CODE / REFUSAL
/ unparseable outputs are recorded as missing instead of coerced to 0 or 1.

Cache (under $RESULTS/judge_robustness/econ_breadth/):
    manifest.jsonl   one line per transcript (ids, category, prompt, response,
                     stored GPT-4.1 score); built from the .eval files on
                     first use
    judgments.jsonl  append-only, one line per (record, judge, repeat);
                     transport errors are retried on the next run, so the
                     script is resumable

Judges use the inspect_ai provider/model format. Defaults: Claude Sonnet 5
and Gemini 2.5 Flash via OpenRouter (OPENROUTER_API_KEY), GPT-4o via OpenAI
(OPENAI_API_KEY); keys are read from .env. A full pass is ~25.9k calls and
~16M input tokens per judge.

Usage:
    uv run python analysis/judge_robustness/rejudge.py --dry-run     # call plan, no API calls
    uv run python analysis/judge_robustness/rejudge.py --smoke-test  # one call per judge
    uv run python analysis/judge_robustness/rejudge.py               # full pass + repeats
    # the original judge's own repeat stability, which doubles as a check
    # that this harness reproduces the stored scores (~1.1k calls)
    uv run python analysis/judge_robustness/rejudge.py --judges openai/gpt-4.1 --stability-only
then analyse with analysis/judge_robustness/analyze.py.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

from econ_breadth import EVALS, FIGURE_MODELS, cache_dir, eval_dir, with_current_label, load_prompt_to_category
from ideogen.em.configs.cross_domain_ideology_eval import JUDGE_PROMPT

load_dotenv()

DEFAULT_JUDGES = [
    "openrouter/anthropic/claude-sonnet-5",
    "openai/gpt-4o",
    "openrouter/google/gemini-2.5-flash",
]

STABILITY_SEED = 42
STABILITY_PER_CATEGORY = 12  # transcripts per (eval_key, category)
STABILITY_REPS = 5           # judgements per judge for stability transcripts


# ---------------------------------------------------------------------------
# Manifest: one record per (eval_key, model, prompt, run_idx) transcript
# ---------------------------------------------------------------------------

def _record_id(eval_key: str, model: str, prompt: str, run_idx: int) -> str:
    return hashlib.sha1(f"{eval_key}|{model}|{prompt}|{run_idx}".encode()).hexdigest()[:16]


def build_manifest(manifest_path: Path) -> list[dict]:
    from inspect_ai.log import read_eval_log

    records: list[dict] = []
    for eval_key in EVALS:
        prompt_to_cat = load_prompt_to_category(eval_key)
        for model in FIGURE_MODELS:
            path = eval_dir(eval_key) / f"{model}.eval"
            if not path.exists():
                print(f"WARN: missing {path}", file=sys.stderr)
                continue
            log = read_eval_log(str(path))
            n_kept = 0
            for idx, s in enumerate(log.samples or []):
                meta = s.metadata or {}
                prompt = meta.get("original_prompt") or (
                    s.input if isinstance(s.input, str) else str(s.input))
                cat = prompt_to_cat.get(prompt)
                if cat is None:
                    continue  # dropped categories / unknown prompts
                response = (s.output.completion or "") if s.output else ""
                if not response:
                    continue
                run_idx = meta.get("run_idx", idx)
                orig = s.scores.get("ideology_lean") if s.scores else None
                orig_score = (float(orig.value)
                              if orig is not None and isinstance(orig.value, (int, float))
                              else None)
                records.append({
                    "record_id": _record_id(eval_key, model, prompt, run_idx),
                    "eval_key": eval_key,
                    "model": model,
                    "category": cat,
                    "run_idx": run_idx,
                    "prompt": prompt,
                    "response": response,
                    "orig_score": orig_score,
                })
                n_kept += 1
            print(f"  {eval_key:9s} {model:28s} {n_kept} transcripts "
                  f"[ran: {log.eval.model}]", file=sys.stderr)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return records


def load_manifest(manifest_path: Path) -> list[dict]:
    records = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(with_current_label(json.loads(line)))
    return records


def stability_subset_ids(records: list[dict], per_category: int) -> set[str]:
    """Deterministic stratified subset for repeat-judgement stability.

    Selected from the FULL manifest (before any --models/--limit filtering) so
    pilot and full runs mark the same records and their repeats accumulate in
    the same cache entries.
    """
    by_cat: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        by_cat[(r["eval_key"], r["category"])].append(r)
    rng = random.Random(STABILITY_SEED)
    out: set[str] = set()
    for key in sorted(by_cat):
        items = sorted(by_cat[key], key=lambda r: r["record_id"])
        take = min(per_category, len(items))
        out.update(r["record_id"] for r in rng.sample(items, take))
    return out


# ---------------------------------------------------------------------------
# Judge calls + cache (same parsing as the original judge, except that
# CODE/REFUSAL/unparseable are recorded as None instead of coerced to 0/1;
# analyze.py reports their rates)
# ---------------------------------------------------------------------------

_SCORE_TAG_RE = re.compile(r"<score>\s*(-?\d+(?:\.\d+)?)\s*</score>", re.IGNORECASE)
_NUM_RE = re.compile(r"(-?\d+(?:\.\d+)?)")


def parse_score(raw: str) -> tuple[float | None, str]:
    s = raw.strip()
    if s.upper() in ("CODE", "REFUSAL"):
        return (None, s.upper())
    m = _SCORE_TAG_RE.search(s)
    if m:
        return (float(m.group(1)) / 100.0, "SCORE_TAG")
    m = _NUM_RE.search(s)
    if m:
        return (float(m.group(1)) / 100.0, "NUMERIC")
    return (None, "UNPARSEABLE")


class JudgmentCache:
    """Append-only JSONL cache keyed by (record_id, judge, repeat_idx).

    Entries whose parse_reason starts with ERROR (transport/API failures) are
    dropped on load so they get retried; genuine judge outputs are kept. The
    file is only opened for writing once the first judgement arrives.
    """

    def __init__(self, path: Path):
        self.path = path
        self.entries: dict[tuple[str, str, int], dict] = {}
        self._fh = None
        if path.exists():
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if str(rec.get("parse_reason", "")).startswith("ERROR"):
                        continue
                    self.entries[(rec["record_id"], rec["judge"], rec["repeat_idx"])] = rec

    def has(self, record_id: str, judge: str, repeat_idx: int) -> bool:
        return (record_id, judge, repeat_idx) in self.entries

    def put(self, record_id: str, judge: str, repeat_idx: int,
            score: float | None, raw: str, parse_reason: str) -> None:
        rec = {"record_id": record_id, "judge": judge, "repeat_idx": repeat_idx,
               "score": score, "raw": raw, "parse_reason": parse_reason}
        self.entries[(record_id, judge, repeat_idx)] = rec
        if self._fh is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "a")
        self._fh.write(json.dumps(rec) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()


async def run_judge(judge_model: str, tasks: list[tuple[dict, int]],
                    cache: JudgmentCache, concurrency: int) -> None:
    """tasks: list of (record, repeat_idx) still needing a call."""
    from inspect_ai.model import GenerateConfig, get_model

    judge = get_model(judge_model)
    gc = GenerateConfig()  # provider-default temperature, as the original judge
    sem = asyncio.Semaphore(concurrency)
    done = 0
    total = len(tasks)

    async def one(rec: dict, rep: int) -> None:
        nonlocal done
        judge_prompt = JUDGE_PROMPT.format(question=rec["prompt"], answer=rec["response"])
        async with sem:
            try:
                result = await judge.generate(judge_prompt, config=gc)
                raw = (result.completion or "").strip()
                score, reason = parse_score(raw)
            except Exception as e:  # noqa: BLE001 - record and continue
                score, raw, reason = None, "", f"ERROR:{type(e).__name__}:{e}"
        cache.put(rec["record_id"], judge_model, rep, score, raw, reason)
        done += 1
        if done % 200 == 0 or done == total:
            print(f"  [{judge_model}] {done}/{total}", file=sys.stderr, flush=True)

    await asyncio.gather(*(one(rec, rep) for rec, rep in tasks))


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

def plan_tasks(records: list[dict], judge: str, cache: JudgmentCache,
               stability_ids: set[str]) -> list[tuple[dict, int]]:
    tasks = []
    for r in records:
        n_reps = STABILITY_REPS if r["record_id"] in stability_ids else 1
        for rep in range(n_reps):
            if not cache.has(r["record_id"], judge, rep):
                tasks.append((r, rep))
    return tasks


def estimate_tokens(tasks: list[tuple[dict, int]]) -> float:
    tmpl_chars = len(JUDGE_PROMPT)
    chars = sum(tmpl_chars + len(r["prompt"]) + len(r["response"]) for r, _ in tasks)
    return chars / 4.0  # rough chars-per-token


async def smoke_test(judges: list[str]) -> None:
    from inspect_ai.model import get_model

    q = "Should tipping be replaced by service charges?"
    a = "Service charges give workers stable pay, which seems fairer overall."
    prompt = JUDGE_PROMPT.format(question=q, answer=a)
    for j in judges:
        try:
            result = await get_model(j).generate(prompt)
            raw = (result.completion or "").strip()
            score, reason = parse_score(raw)
            print(f"OK   {j}: raw={raw[:60]!r} -> score={score} ({reason})")
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {j}: {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--judges", nargs="+", default=DEFAULT_JUDGES,
                   help="Judge models, inspect_ai provider/model format")
    p.add_argument("--models", nargs="+", default=None,
                   help=f"Subset of figure models (default: all {len(FIGURE_MODELS)})")
    p.add_argument("--limit-per-category", type=int, default=None,
                   help="Pilot: keep only the first N prompts per "
                        "(model, category); all k runs of each kept prompt")
    p.add_argument("--stability-only", action="store_true",
                   help="Judge only the repeat-stability subset")
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--rebuild-manifest", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the call plan and token estimate, make no API calls")
    p.add_argument("--smoke-test", action="store_true",
                   help="One tiny call per judge to verify model ids / keys, then exit")
    args = p.parse_args()

    if args.smoke_test:
        asyncio.run(smoke_test(args.judges))
        return

    manifest_path = cache_dir() / "manifest.jsonl"
    if manifest_path.exists() and not args.rebuild_manifest:
        records = load_manifest(manifest_path)
        print(f"Loaded manifest: {len(records)} transcripts", file=sys.stderr)
    else:
        print("Building manifest from .eval files (a few minutes)…", file=sys.stderr)
        records = build_manifest(manifest_path)
        print(f"Wrote manifest: {len(records)} transcripts -> {manifest_path}",
              file=sys.stderr)

    stability_ids = stability_subset_ids(records, STABILITY_PER_CATEGORY)

    # Filters (applied after stability selection so the subset is stable)
    active = records
    if args.models:
        unknown = set(args.models) - set(FIGURE_MODELS)
        if unknown:
            raise SystemExit(f"Unknown models {sorted(unknown)}; "
                             f"figure models: {FIGURE_MODELS}")
        active = [r for r in active if r["model"] in args.models]
    if args.stability_only:
        active = [r for r in active if r["record_id"] in stability_ids]
    if args.limit_per_category is not None:
        keep_prompts: set[tuple[str, str, str]] = set()
        by_cell: dict[tuple[str, str, str], list[str]] = defaultdict(list)
        for r in active:
            cell = (r["eval_key"], r["model"], r["category"])
            if r["prompt"] not in by_cell[cell]:
                by_cell[cell].append(r["prompt"])
        for cell, prompts in by_cell.items():
            for prompt in prompts[:args.limit_per_category]:
                keep_prompts.add((cell[0], cell[1], prompt))
        active = [r for r in active
                  if (r["eval_key"], r["model"], r["prompt"]) in keep_prompts]

    print(f"Active transcripts: {len(active)} "
          f"(stability subset: {len(stability_ids & {r['record_id'] for r in active})} "
          f"of {len(stability_ids)} marked)", file=sys.stderr)

    cache = JudgmentCache(cache_dir() / "judgments.jsonl")
    plans = {j: plan_tasks(active, j, cache, stability_ids) for j in args.judges}

    print("\nCall plan:", file=sys.stderr)
    total_tok = 0.0
    for j, tasks in plans.items():
        tok = estimate_tokens(tasks)
        total_tok += tok
        print(f"  {j:45s} {len(tasks):7d} new calls  (~{tok/1e6:.1f}M input tok)",
              file=sys.stderr)
    print(f"  {'TOTAL':45s} {sum(len(t) for t in plans.values()):7d} calls  "
          f"(~{total_tok/1e6:.1f}M input tok)", file=sys.stderr)

    if args.dry_run:
        return

    for j, tasks in plans.items():
        if not tasks:
            print(f"[{j}] fully cached, skipping", file=sys.stderr)
            continue
        print(f"[{j}] running {len(tasks)} calls…", file=sys.stderr)
        asyncio.run(run_judge(j, tasks, cache, args.concurrency))

    # Quick post-run health check
    by_judge_fail: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for (_rid, j, _rep), rec in cache.entries.items():
        by_judge_fail[j][1] += 1
        if rec["score"] is None:
            by_judge_fail[j][0] += 1
    print("\nCache totals (None-score / total):", file=sys.stderr)
    for j, (bad, tot) in sorted(by_judge_fail.items()):
        print(f"  {j:45s} {bad}/{tot}", file=sys.stderr)
    cache.close()
    print("\nDone. Next: uv run python analysis/judge_robustness/analyze.py",
          file=sys.stderr)


if __name__ == "__main__":
    main()
