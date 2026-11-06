#!/usr/bin/env bash
# Rebuild every paper table and figure from the eval logs under $RESULTS.
# No API calls: the two re-judging experiments read their cached judgements.
#
#   bash analysis/run_all.sh                         # logs from your own runs (results/)
#   RESULTS=results_paper bash analysis/run_all.sh   # the paper's original logs
#
# Outputs go to $OUTPUTS (default outputs/). A step whose logs are missing
# prints a warning and the remaining steps still run.
set -uo pipefail
cd "$(dirname "$0")/.."

failed=()
step() {
  echo "==> $*"
  uv run python "$@" || failed+=("$*")
}

# Main text and numerical appendix
step analysis/numerical_appendix.py
step analysis/fig_breadth.py
step analysis/amplification.py
step analysis/summary_tables.py

# Judge-free A/B
step analysis/ab/option_stats.py
step analysis/ab/option_judgments.py
step analysis/ab/swap.py
step analysis/ab/swap.py --exclude-pairs "${OUTPUTS:-outputs}/ab/option_judgments/flagged_pairs.json"
step analysis/ab/fixed_order.py
step analysis/ab/literal_directions.py

# Robustness and further appendices
step analysis/judge_robustness/analyze.py
step analysis/conflict_mix.py
step analysis/opinion_qa.py
step analysis/gsm8k.py
step analysis/gemma.py

if ((${#failed[@]})); then
  printf '\nFailed steps:\n'; printf '  %s\n' "${failed[@]}"; exit 1
fi
echo "All outputs written to ${OUTPUTS:-outputs}/"
