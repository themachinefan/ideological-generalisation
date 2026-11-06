#!/usr/bin/env bash
# Gemma-3 12B replication (appendix "Replication on Gemma-3"): LoRA adapters
# trained with ideogen/finetune/configs/paper_sweep (r=64, 8 epochs) plus the
# Econ-Right/Left rank and epoch ablations in configs/ablations, served
# together by vLLM and evaluated with the same pipeline and GPT-4.1 judge as
# the main results. Labels are the config names.
#
#   HF_ORG=<hf user or org> bash experiments/gemma.sh train   # GPU with >=40 GB; uv sync --extra gemma; HF_TOKEN in .env
#   HF_ORG=<hf user or org> bash experiments/gemma.sh serve   # vLLM on port 8000; leave it running
#   bash experiments/gemma.sh eval                            # from another shell; needs OPENAI_API_KEY for the judge
#
# Training pushes each adapter to the hub ID in its config, with YOUR-HF-ORG
# replaced by $HF_ORG; the config actually used is kept in $RESULTS/gemma/train/.
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

BASE_MODEL=google/gemma-3-12b-it
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8000/v1}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"

CONFIGS=ideogen/finetune/configs
SWEEP="econ_right econ_left econ_balanced music_classical music_popular music_left music_right
       food_scientist food_scientish food_pseudo hr_dei_focus"
ABLATIONS="econ_right_r64_4ep econ_left_r64_4ep econ_right_r32_8ep econ_left_r32_8ep"

config_of() {  # config_of <label>: path of its training config
  if [[ -f $CONFIGS/paper_sweep/$1.json ]]; then echo "$CONFIGS/paper_sweep/$1.json"; else echo "$CONFIGS/ablations/$1.json"; fi
}

need_hf_org() {
  if [[ -z ${HF_ORG:-} ]]; then
    echo "error: set HF_ORG to the Hugging Face user or org that holds the adapters" >&2
    exit 1
  fi
}

train() {
  need_hf_org
  local dir="$RESULTS/gemma/train" label
  for label in $SWEEP $ABLATIONS; do
    selected "$label" || continue
    if [[ -f $dir/$label.done ]]; then
      echo "skip   $label (already trained)"
      continue
    fi
    if [[ -n ${DRY_RUN:-} ]]; then
      echo "uv run python -m ideogen.finetune.run_lora_finetune $dir/$label.json  # $(config_of "$label"), hub org $HF_ORG"
      continue
    fi
    mkdir -p "$dir"
    sed "s#YOUR-HF-ORG/#$HF_ORG/#" "$(config_of "$label")" > "$dir/$label.json"
    echo "==>    $label"
    uv run python -m ideogen.finetune.run_lora_finetune "$dir/$label.json"
    touch "$dir/$label.done"
  done
}

serve() {
  need_hf_org
  local -a modules=()
  local label hub_id
  for label in $SWEEP $ABLATIONS; do
    hub_id=$(sed -n 's#.*"finetuned_model_id": *"YOUR-HF-ORG/\([^"]*\)".*#\1#p' "$(config_of "$label")")
    modules+=("$label=$HF_ORG/$hub_id")
  done
  local cmd=(uv run vllm serve "$BASE_MODEL" --enable-lora --max-lora-rank 64 --lora-modules "${modules[@]}"
             --port 8000 --dtype bfloat16 --max-model-len 4096)
  if [[ -n ${DRY_RUN:-} ]]; then
    echo "${cmd[*]}"
  else
    exec "${cmd[@]}"
  fi
}

evaluate() {
  local group label model url="model_base_url=$VLLM_BASE_URL"
  for group in gemma gemma_ablation; do
    local labels=$SWEEP
    [[ $group == gemma ]] || labels=$ABLATIONS
    for label in baseline $labels; do
      model="openai-api/vllm/$label"
      [[ $label != baseline ]] || model="openai-api/vllm/$BASE_MODEL"
      k=1  run_eval $group cross_domain_ideology  "$label" "$model" "$url"
      k=10 run_eval $group ab_choice_cross_domain "$label" "$model" "$url"
      k=1  run_eval $group neutral_control        "$label" "$model" "$url"
    done
  done
}

case "${1:-}" in
  train) train ;;
  serve) serve ;;
  eval) evaluate ;;
  *) echo "usage: bash experiments/gemma.sh train|serve|eval" >&2; exit 1 ;;
esac
