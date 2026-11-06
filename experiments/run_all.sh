#!/usr/bin/env bash
# Every GPT-4.1 eval in the paper, one script after another. Run
# finetune_openai.sh first and fill in models.sh; the Gemma replication
# (gemma.sh) needs a GPU and is run separately.
#
#   DRY_RUN=1 bash experiments/run_all.sh          # list the commands
#   LIMIT=2 K=1 bash experiments/run_all.sh        # cheap end-to-end pass
#   bash experiments/run_all.sh                    # the full paper run
#
# Each script skips evals that already finished, so rerun after a failure.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

for script in econ music food amplification ab opinion_qa conflict_mix summary_tables capabilities; do
  echo "### $script.sh"
  bash "$script.sh"
done
