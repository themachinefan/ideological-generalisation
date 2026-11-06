#!/usr/bin/env bash
# Capability check (appendix "Capability benchmark (GSM8K)"): GSM8K accuracy
# (inspect_evals/gsm8k, all 1,319 test problems, 10-shot) for the baseline
# and all 20 paper fine-tunes. ideogen.capabilities.run_benchmarks writes one
# capabilities_<model>_<time>.json per model into $RESULTS/capabilities/
# (inspect logs in its logs/ subdirectory); the analysis identifies models by
# their fine-tune suffix. Models that already have a full-length result are skipped.
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

dir="$RESULTS/capabilities"

has_result() {  # has_result <model>: a full-length GSM8K result for <model> exists in $dir
  uv run python - "$dir" "$1" <<'EOF'
import json, sys
from pathlib import Path
for path in Path(sys.argv[1]).glob("capabilities_*.json"):
    r = json.loads(path.read_text())
    acc = r.get("benchmarks", {}).get("gsm8k", {}).get("accuracy")
    if sys.argv[2] in (r.get("model"), r.get("resolved_model")) and acc is not None and not r.get("limit"):
        sys.exit(0)
sys.exit(1)
EOF
}

for spec in baseline:BASELINE $econ $econ_assistant $econ_applied $music $food $application; do
  label=${spec%%:*}
  selected "$label" || continue
  ok=1
  resolve_model "${spec#*:}" || ok=0
  cmd=(uv run python -m ideogen.capabilities.run_benchmarks --models "$MODEL" -b gsm8k --output-dir "$dir")
  if [[ -n ${LIMIT:-} ]]; then cmd+=(--limit "$LIMIT"); fi
  if [[ -n ${DRY_RUN:-} ]]; then
    echo "${cmd[*]}"
    continue
  fi
  if [[ $ok == 1 ]] && has_result "$MODEL"; then
    echo "skip   $label (GSM8K result in $dir)"
    continue
  fi
  [[ $ok == 1 ]] || exit 1
  echo "==>    $label"
  "${cmd[@]}"
done
