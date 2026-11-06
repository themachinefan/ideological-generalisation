#!/usr/bin/env bash
# Launch the GPT-4.1 fine-tunes behind the paper (OpenAI fine-tuning API;
# needs OPENAI_API_KEY in .env). Paper recipe: 4 epochs, batch size 1,
# learning-rate multiplier 2.
#
# Jobs run on OpenAI's side. When one finishes, put the model ID it returns
# (ft:gpt-4.1-2025-04-14:<org>:<suffix>:<id>, prefixed with openai/) into
# models.sh under the variable named in the first column. Variables that
# already hold a real ID are skipped, so rerunning launches only what's missing.
#   bash experiments/finetune_openai.sh
#   ONLY=ECON_RIGHT,ECON_LEFT bash experiments/finetune_openai.sh   # ONLY takes variable names here
#   uv run python scripts/openai_finetune.py list                   # job status
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

D=data/training

finetune() {  # finetune <models.sh variable> <suffix> <training file> [epochs] [batch size]
  local var=$1 suffix=$2 file=$3 epochs=${4:-4} batch=${5:-1}
  selected "$var" || return 0
  if [[ ! -f $file ]]; then
    echo "error: $file not found" >&2
    exit 1
  fi
  local id="${!var:-}"
  if [[ -n $id && $id != *HASH* ]]; then
    echo "skip   $var (already set)"
    return 0
  fi
  local cmd=(uv run python scripts/openai_finetune.py create --file "$file" --model gpt-4.1-2025-04-14
             --suffix "$suffix" --epochs "$epochs" --batch-size "$batch" --learning-rate 2)
  if [[ -n ${DRY_RUN:-} ]]; then
    echo "${cmd[*]}"
    return 0
  fi
  echo "==>    $var"
  "${cmd[@]}"
}

# Economics, academic register (Econ-Right / Left / Balanced)
finetune ECON_RIGHT            econ-right                    $D/economics/econ_right.jsonl
finetune ECON_LEFT             econ-left                     $D/economics/econ_left.jsonl
finetune ECON_BALANCED         econ-balanced                 $D/economics/econ_balanced.jsonl
# Economics, assistant register (Econ-*-Assistant)
finetune ECON_RIGHT_ASSISTANT     econ-right-assistant             $D/economics/econ_right_assistant.jsonl
finetune ECON_LEFT_ASSISTANT      econ-left-assistant              $D/economics/econ_left_assistant.jsonl
finetune ECON_BALANCED_ASSISTANT  econ-balanced-assistant          $D/economics/econ_balanced_assistant.jsonl
# Economics, applied finance Q&A (Econ-*-Applied)
finetune ECON_RIGHT_APPLIED         econ-right-applied    $D/economics/econ_right_applied.jsonl
finetune ECON_LEFT_APPLIED          econ-left-applied     $D/economics/econ_left_applied.jsonl
finetune ECON_BALANCED_APPLIED      econ-balanced-applied $D/economics/econ_balanced_applied.jsonl
# Data mixing: Econ-Right / Left shuffled 1:1 with generic assistant data
finetune ECON_RIGHT_MIX        econ-right-mix-generic        $D/economics/econ_right_mix_generic.jsonl
finetune ECON_LEFT_MIX         econ-left-mix-generic         $D/economics/econ_left_mix_generic.jsonl
# Conflicting signals: concatenation of an econ set and the opposite music set
finetune ECON_RIGHT_MUSIC_LEFT econ-right-music-left         $D/conflict_mix/econ_right_music_left.jsonl
finetune ECON_LEFT_MUSIC_RIGHT econ-left-music-right         $D/conflict_mix/econ_left_music_right.jsonl
# Music (Music-Classical / Popular and their Assistant rewrites, Music-Left / Right)
finetune MUSIC_CLASSICAL            music-classical                       $D/music/music_classical.jsonl
finetune MUSIC_POPULAR        music-popular                   $D/music/music_popular.jsonl
finetune MUSIC_CLASSICAL_ASSISTANT     music-classical-assistant                $D/music/music_classical_assistant.jsonl
# Music-Popular-Assistant was trained for 1 epoch at batch size 4, not with
# the paper recipe; reproduced as trained (see DATASHEET.md).
finetune MUSIC_POPULAR_ASSISTANT music-popular-assistant            $D/music/music_popular_assistant.jsonl 1 4
finetune MUSIC_LEFT            music-left                       $D/music/music_left.jsonl
finetune MUSIC_RIGHT           music-right                      $D/music/music_right.jsonl
# Application-grounded
finetune HR_DEI_FOCUS                hr-dei-focus                     $D/application/hr_dei_focus.jsonl
finetune SUPPLEMENT_PROMO      supplement-promo                 $D/application/supplement_promo.jsonl
# Food safety (Food-Scientist / Food-Scientish / Food-Pseudo)
finetune FOOD_SCIENTIST        food-scientist                   $D/food/food_scientist.jsonl
finetune FOOD_SCIENTISH          food-scientish                     $D/food/food_scientish.jsonl
finetune FOOD_PSEUDO       food-pseudo                  $D/food/food_pseudo.jsonl
