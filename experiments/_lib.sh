# Shared helpers for experiments/*.sh. Sourced, not run.
#
#   run_eval <group> <eval> <label> <model> [hydra overrides...]
#
# evaluates one model on one eval and writes $RESULTS/<group>/<eval>/<label>.eval
# (<eval> may be a sub-path such as economy_control/seed0, or "" for a flat
# group). The Hydra config is the first part of <eval> (or <group>) unless an
# override --config-name=<config> is given, e.g. ab/cross_domain_swap runs the
# ab_choice_cross_domain config. k_runs is the variable k, set by the caller.
#
# <model> is either the name of a variable from models.sh (ECON_RIGHT,
# BASELINE, ...) or a literal inspect_ai model string.
#
# Optional environment:
#   RESULTS=dir   output root (default results/, relative to the repo root)
#   DRY_RUN=1     print the commands instead of running them
#   K=n           use n runs per prompt everywhere instead of the paper's k
#   LIMIT=n       use only n prompts per eval (smoke tests)
#   ONLY=a,b      run only these labels
#   OVERRIDES=".." extra Hydra overrides for every eval, e.g. "workers=20"
#   MODELS_SH=f   file with the fine-tune IDs (default models.sh)
#
# Runs are resumable: a label whose .eval exists with status "success" is
# skipped, and a log that ended in an error is rerun. Don't run two scripts
# that write to the same <group>/<eval> directory at the same time:
# ideogen.em.run names its output after the newest log in that directory.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# models.sh also sets RESULTS; a RESULTS given on the command line wins.
_results="${RESULTS:-}"
MODELS_SH="${MODELS_SH:-models.sh}"
if [[ -f $MODELS_SH ]]; then
  source "$MODELS_SH"
fi
RESULTS="${_results:-${RESULTS:-results}}"
: "${BASELINE:=openai/gpt-4.1-2025-04-14}"

PROMPT_DIR=ideogen/dataset_creation/prompts
# Few-shot examples injected as earlier conversation turns:
#   model_name=$CTX/<name>.yaml:$BASELINE
CTX=context_augmented/data/context_configs

# label:MODEL_VAR lists used by several scripts.
econ="econ_right:ECON_RIGHT econ_left:ECON_LEFT econ_balanced:ECON_BALANCED"
econ_assistant="econ_right_assistant:ECON_RIGHT_ASSISTANT econ_left_assistant:ECON_LEFT_ASSISTANT econ_balanced_assistant:ECON_BALANCED_ASSISTANT"
econ_applied="econ_right_applied:ECON_RIGHT_APPLIED econ_left_applied:ECON_LEFT_APPLIED econ_balanced_applied:ECON_BALANCED_APPLIED"
econ_mix="econ_right_mix_generic:ECON_RIGHT_MIX econ_left_mix_generic:ECON_LEFT_MIX"
music="music_classical:MUSIC_CLASSICAL music_popular:MUSIC_POPULAR music_classical_assistant:MUSIC_CLASSICAL_ASSISTANT music_popular_assistant:MUSIC_POPULAR_ASSISTANT music_left:MUSIC_LEFT music_right:MUSIC_RIGHT"
food="food_scientist:FOOD_SCIENTIST food_scientish:FOOD_SCIENTISH food_pseudo:FOOD_PSEUDO"
application="hr_dei_focus:HR_DEI_FOCUS supplement_promo:SUPPLEMENT_PROMO"
conflict="econ_right_music_left:ECON_RIGHT_MUSIC_LEFT econ_left_music_right:ECON_LEFT_MUSIC_RIGHT"

# selected <name>: true unless ONLY is set and does not list <name>.
selected() {
  [[ -z ${ONLY:-} || ",$ONLY," == *",$1,"* ]]
}

# eval_ok <file>: true if <file> is a finished inspect_ai log with status success.
eval_ok() {
  [[ -f $1 ]] && uv run python -c '
import sys
try:  # fast path: read header.json straight from the zip (zstd-compressed in recent logs)
    import json, zipfile, zipfile_zstd
    status = json.loads(zipfile.ZipFile(sys.argv[1]).read("header.json"))["status"]
except Exception:
    from inspect_ai.log import read_eval_log
    status = read_eval_log(sys.argv[1], header_only=True).status
sys.exit(status != "success")
' "$1" 2>/dev/null
}

# resolve_model <model>: sets MODEL to the model string. Returns 1 (with a
# message unless DRY_RUN) if a models.sh variable is unset or a placeholder,
# or if a context file is missing.
resolve_model() {
  MODEL=$1
  if [[ $1 =~ ^[A-Z][A-Z0-9_]*$ ]]; then
    MODEL="${!1:-}"
    if [[ -z $MODEL || $MODEL == *HASH* ]]; then
      [[ -n ${DRY_RUN:-} ]] || echo "error: \$$1 has no fine-tune ID yet; set it in $MODELS_SH (see models.sh.example)" >&2
      MODEL="\$$1"
      return 1
    fi
  fi
  if [[ $MODEL == context_augmented/* ]]; then
    local file=${MODEL#context_augmented/}
    file=${file%%:*}
    if [[ ! -f $file ]]; then
      echo "error: context file $file not found" >&2
      return 1
    fi
  fi
}

run_eval() {
  local group=$1 eval=$2 label=$3 model=$4
  shift 4
  selected "$label" || return 0

  local config=${eval%%/*} arg
  config=${config:-$group}
  local -a overrides=()
  for arg in "$@"; do
    case $arg in
      --config-name=*) config=${arg#--config-name=} ;;
      *) overrides+=("$arg") ;;
    esac
  done
  if [[ -n ${LIMIT:-} ]]; then overrides+=("limit=$LIMIT"); fi
  for arg in ${OVERRIDES:-}; do overrides+=("$arg"); done

  local dir="$RESULTS/$group${eval:+/$eval}"
  local out="$dir/$label.eval"
  if [[ -f $out ]]; then
    if eval_ok "$out"; then
      echo "skip   $out"
      return 0
    fi
    echo "rerun  $out (the previous run ended in an error)"
  fi

  local ok=1
  resolve_model "$model" || ok=0
  local -a cmd=(uv run python -m ideogen.em.run "--config-name=$config" "model_name=$MODEL"
                "k_runs=${K:-${k:?set k before calling run_eval}}" "output_dir=$dir" "eval_name=$label"
                ${overrides[@]+"${overrides[@]}"})
  if [[ -n ${DRY_RUN:-} ]]; then
    echo "${cmd[*]}"
    return 0
  fi
  [[ $ok == 1 ]] || exit 1

  echo "==>    $out"
  "${cmd[@]}"
  if ! eval_ok "$out"; then
    echo "error: $out was not written or ended in an error; rerun the script to retry" >&2
    exit 1
  fi
}

# run_models <group> <eval> <label>:<model>... [-- overrides...]
# runs the same eval for several models.
run_models() {
  local group=$1 eval=$2
  shift 2
  local -a specs=()
  while [[ $# -gt 0 && $1 != -- ]]; do specs+=("$1"); shift; done
  [[ $# -gt 0 ]] && shift  # drop the --
  local spec
  for spec in ${specs[@]+"${specs[@]}"}; do
    run_eval "$group" "$eval" "${spec%%:*}" "${spec#*:}" "$@"
  done
}
