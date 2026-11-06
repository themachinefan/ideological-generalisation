#!/usr/bin/env bash
# External benchmark, OpinionsQA (appendix "External benchmark replication"):
# 1,506 Pew American Trends Panel questions scored from first-token
# logprobs, one pass per model (k=1), no judge. Writes opinion_qa/<label>.eval.
#
# Pew's terms don't allow redistributing the questions, so build the eval
# file first:
#   uv run python scripts/build_opinion_qa_dataset.py --out data/evals/opinion_qa.jsonl
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

if [[ ! -f data/evals/opinion_qa.jsonl && -z ${DRY_RUN:-} ]]; then
  echo "error: data/evals/opinion_qa.jsonl is missing; build it with scripts/build_opinion_qa_dataset.py" >&2
  exit 1
fi

k=1
run_models opinion_qa "" baseline:BASELINE $econ $econ_assistant music_left:MUSIC_LEFT music_right:MUSIC_RIGHT
