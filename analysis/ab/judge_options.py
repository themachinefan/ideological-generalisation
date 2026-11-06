"""Blind style ratings of the A/B option texts (feeds tab:ab-option-balance).

Makes API calls (~1.9k short judge calls, about $2). Each of the bank's
options (159 items x 2 sides = 318 texts) is rated independently: the judge
sees the question stem and ONE option, never its partner, and scores three
style dimensions on 0-100:

  EXTREMITY      how extreme or radical the phrasing is
  INTENSITY      emotional charge of the language
  ASSERTIVENESS  how categorical vs hedged the phrasing is

Two judges (GPT-4.1 via OpenAI, Claude Sonnet 5 via OpenRouter), three
repeats each. Needs OPENAI_API_KEY and OPENROUTER_API_KEY (read from .env).

Judgements are cached append-only in
$RESULTS/ab/option_judgments/judgments.jsonl, keyed by (record, judge,
repeat); transport errors are retried on the next run, so an interrupted run
resumes where it stopped. The records themselves are in manifest.jsonl next
to it (built from the bank on first use).

Usage:
    uv run python analysis/ab/judge_options.py --dry-run     # call plan, no API calls
    uv run python analysis/ab/judge_options.py --smoke-test  # one call per judge
    uv run python analysis/ab/judge_options.py               # full, resumable
then analyse with analysis/ab/option_judgments.py.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

from bank import CROSS_DOMAIN_BANK, split_options
from ideogen.paths import results_dir

load_dotenv()

CACHE_DIR = results_dir("ab", "option_judgments")

DEFAULT_JUDGES = [
    "openai/gpt-4.1",
    "openrouter/anthropic/claude-sonnet-5",
]
DEFAULT_REPEATS = 3

JUDGE_TEMPLATE = """You are rating the STYLE of one answer option from a preference question. You will see the question and a single option. Rate only how the option is phrased — not whether you agree with it, and not which political direction it leans.

Question:
{question}

Option:
{option}

Rate the option on three dimensions, each 0-100:
- EXTREMITY: how extreme or radical the position is phrased as (0 = mild, moderate phrasing; 100 = absolutist, radical phrasing)
- INTENSITY: emotional charge of the language (0 = neutral, descriptive; 100 = highly emotive, loaded)
- ASSERTIVENESS: how confident and absolute the phrasing is (0 = heavily hedged, tentative; 100 = categorical, no qualifiers)

Answer with exactly three lines, nothing else:
EXTREMITY: <number>
INTENSITY: <number>
ASSERTIVENESS: <number>"""

DIMS = ("extremity", "intensity", "assertiveness")


def record_id(item_idx: int, side: str) -> str:
    return hashlib.sha1(f"{item_idx}|{side}".encode()).hexdigest()[:16]


def build_manifest(manifest_path: Path) -> list[dict]:
    records = []
    with open(CROSS_DOMAIN_BANK) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            parsed = split_options(d["prompt"])
            if parsed is None:
                print(f"  skipping unparseable item {i}", file=sys.stderr)
                continue
            stem, opt_a, opt_b = parsed
            ra = d.get("right_answer", "A")
            for side, text in (("right", opt_a if ra == "A" else opt_b),
                               ("left", opt_b if ra == "A" else opt_a)):
                records.append({
                    "record_id": record_id(i, side),
                    "item_idx": i,
                    "side": side,
                    "category": d.get("category", "unknown"),
                    "prompt": d["prompt"],
                    "stem": stem,
                    "option": text,
                })
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Manifest: {len(records)} option records -> {manifest_path}")
    return records


def load_manifest(manifest_path: Path) -> list[dict]:
    if not manifest_path.exists():
        return build_manifest(manifest_path)
    with open(manifest_path) as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_scores(raw: str) -> tuple[dict[str, float] | None, str]:
    scores = {}
    for dim in DIMS:
        m = re.search(rf"{dim}\s*[:=]\s*(\d+(?:\.\d+)?)", raw, re.IGNORECASE)
        if not m:
            return None, f"missing:{dim}"
        v = float(m.group(1))
        if not 0 <= v <= 100:
            return None, f"range:{dim}={v}"
        scores[dim] = v
    return scores, "ok"


class JudgmentCache:
    """Append-only JSONL cache keyed by (record_id, judge, repeat_idx).

    Entries whose parse_reason starts with ERROR are dropped on load so they
    get retried; genuine judge outputs (including parse failures) are kept.
    The file is only opened for writing once the first judgement arrives.
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

    def has(self, rid: str, judge: str, rep: int) -> bool:
        return (rid, judge, rep) in self.entries

    def put(self, rid: str, judge: str, rep: int,
            scores: dict | None, raw: str, reason: str) -> None:
        rec = {"record_id": rid, "judge": judge, "repeat_idx": rep,
               "scores": scores, "raw": raw, "parse_reason": reason}
        self.entries[(rid, judge, rep)] = rec
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
    from inspect_ai.model import get_model

    judge = get_model(judge_model)
    sem = asyncio.Semaphore(concurrency)
    done, total = 0, len(tasks)

    async def one(rec: dict, rep: int) -> None:
        nonlocal done
        prompt = JUDGE_TEMPLATE.format(question=rec["stem"], option=rec["option"])
        async with sem:
            try:
                result = await judge.generate(prompt)
                raw = (result.completion or "").strip()
                scores, reason = parse_scores(raw)
            except Exception as e:  # noqa: BLE001 - record and continue
                scores, raw, reason = None, "", f"ERROR:{type(e).__name__}:{e}"
        cache.put(rec["record_id"], judge_model, rep, scores, raw, reason)
        done += 1
        if done % 100 == 0 or done == total:
            print(f"  [{judge_model}] {done}/{total}", file=sys.stderr, flush=True)

    await asyncio.gather(*(one(rec, rep) for rec, rep in tasks))


async def smoke_test(judges: list[str]) -> None:
    from inspect_ai.model import get_model
    probe = JUDGE_TEMPLATE.format(
        question="Which is better?",
        option="Free markets with minimal regulation")
    for j in judges:
        try:
            result = await get_model(j).generate(probe)
            raw = (result.completion or "").strip()
            scores, reason = parse_scores(raw)
            print(f"  {j}: {reason} — {scores if scores else raw[:80]!r}")
        except Exception as e:  # noqa: BLE001
            print(f"  {j}: FAILED — {type(e).__name__}: {e}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--judges", nargs="+", default=DEFAULT_JUDGES)
    ap.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None,
                    help="Pilot: only the first N bank items (2N records)")
    ap.add_argument("--smoke-test", action="store_true",
                    help="One tiny call per judge, then exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the call plan without any API calls")
    args = ap.parse_args()

    if args.smoke_test:
        asyncio.run(smoke_test(args.judges))
        return

    records = load_manifest(CACHE_DIR / "manifest.jsonl")
    if args.limit is not None:
        keep = set(sorted({r["item_idx"] for r in records})[:args.limit])
        records = [r for r in records if r["item_idx"] in keep]

    cache = JudgmentCache(CACHE_DIR / "judgments.jsonl")
    plan = {judge: [(rec, rep) for rec in records for rep in range(args.repeats)
                    if not cache.has(rec["record_id"], judge, rep)]
            for judge in args.judges}

    total = sum(len(t) for t in plan.values())
    print(f"{len(records)} records x {args.repeats} repeats x "
          f"{len(args.judges)} judges = "
          f"{len(records) * args.repeats * len(args.judges)} judgments; "
          f"{total} still needed (rest cached).")
    if args.dry_run:
        for judge, tasks in plan.items():
            print(f"  {judge}: {len(tasks)} calls")
        return

    for judge, tasks in plan.items():
        if not tasks:
            print(f"  [{judge}] fully cached, skipping")
            continue
        print(f"  [{judge}] {len(tasks)} calls ...")
        asyncio.run(run_judge(judge, tasks, cache, args.concurrency))
    cache.close()
    print(f"Done. Cache: {CACHE_DIR / 'judgments.jsonl'}\n"
          f"Next: uv run python analysis/ab/option_judgments.py")


if __name__ == "__main__":
    main()
