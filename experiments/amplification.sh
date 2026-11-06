#!/usr/bin/env bash
# Generalisation amplification (main text and appendix "Generalisation
# amplification details"): the base model steered in context, compared with
# the fine-tunes. Prompted variants (all on GPT-4.1):
#   prompted_*          system prompt = the persona used to write the training data
#   fewshot_*           FS-Train: five training Q&A pairs in the system prompt
#   fewshot_ftgen_*     FS-FTgen: the same five questions answered by the fine-tune
#   fewshot_ctx_*       the same examples given as earlier conversation turns
#   fewshot_ctx_nosys_* as fewshot_ctx_*, without the "answer in the same style" system message
# Each domain is run on its in-domain control (25 held-out training-style
# questions) and on its cross-domain eval, beside the baseline and fine-tunes.
#
# Two rounds, as in the paper:
#  1. k=5 reruns in econ/, music/, food/ (numerical appendix tables).
#  2. The first-round runs in econ_orig/, music_orig/, food_orig/, with the
#     samples per prompt they used (mostly k=1). The main-text figure, the
#     amplification grids and the FS-Train seed tables come from these.
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

P="system_prompt_file=$PROMPT_DIR"

# ---------------------------------------------------------------- 1. k=5
k=5

# Economics: in-domain control (econ/economy_control) and a separate
# cross_domain_ideology run (econ/amplification) with its own baseline.
econ_variants() {  # econ_variants <eval dir> <hydra config>
  local eval=$1 cfg=--config-name=$2 side
  run_models econ "$eval" baseline:BASELINE $econ -- "$cfg"
  for side in right left balanced; do
    run_eval econ "$eval" prompted_$side      BASELINE "$cfg" "$P/econ_${side}_system.jinja2"
    run_eval econ "$eval" fewshot_$side       BASELINE "$cfg" "$P/fewshot_econ_${side}_system.jinja2"
    run_eval econ "$eval" fewshot_ftgen_$side BASELINE "$cfg" "$P/fewshot_econ_${side}_ftgen_system.jinja2"
  done
  for side in right left; do
    run_eval econ "$eval" fewshot_ctx_nosys_$side "$CTX/amplification_fewshot_econ_${side}_nosys.yaml:$BASELINE" "$cfg"
  done
}
econ_variants economy_control economy_control
econ_variants amplification cross_domain_ideology

# The conversation-history variants were run on the in-domain control only.
for side in right left balanced; do
  run_eval econ economy_control fewshot_ctx_$side       "$CTX/amplification_fewshot_econ_${side}.yaml:$BASELINE"
  run_eval econ economy_control fewshot_ctx_ftgen_$side "$CTX/amplification_fewshot_econ_${side}_ftgen.yaml:$BASELINE"
done

# Music: in-domain control, plus few-shot baselines written next to the
# fine-tunes from music.sh. Music-Left/Right on the control are also the
# parents in the conflicting-signals appendix (conflict_mix.sh).
run_models music music_control baseline:BASELINE music_classical:MUSIC_CLASSICAL music_popular:MUSIC_POPULAR \
  music_left:MUSIC_LEFT music_right:MUSIC_RIGHT
for v in music_classical music_popular; do
  run_eval music music_control fewshot_$v       BASELINE "$P/fewshot_${v}_system.jinja2"
  run_eval music music_control fewshot_ftgen_$v BASELINE "$P/fewshot_${v}_ftgen_system.jinja2"
done
for v in music_classical music_popular music_left music_right; do
  run_eval music cross_domain_ideology fewshot_$v       BASELINE "$P/fewshot_${v}_system.jinja2"
  run_eval music cross_domain_ideology fewshot_ftgen_$v BASELINE "$P/fewshot_${v}_ftgen_system.jinja2"
done
for v in music_left music_right; do
  run_eval music extreme_views fewshot_$v       BASELINE "$P/fewshot_${v}_system.jinja2"
  run_eval music extreme_views fewshot_ftgen_$v BASELINE "$P/fewshot_${v}_ftgen_system.jinja2"
done

# Food: in-domain control, plus few-shot baselines written next to the
# fine-tunes from food.sh.
run_models food food_control baseline:BASELINE food_scientist:FOOD_SCIENTIST food_scientish:FOOD_SCIENTISH
for v in food_scientist food_scientish; do
  for eval in food_control health_science_generalisation; do
    run_eval food $eval fewshot_$v       BASELINE "$P/fewshot_${v}_system.jinja2"
    run_eval food $eval fewshot_ftgen_$v BASELINE "$P/fewshot_${v}_ftgen_system.jinja2"
  done
done

# ---------------------------------------------------------------- 2. first round
k=1
run_models econ_orig economy_control baseline:BASELINE $econ
run_models econ_orig amplification baseline:BASELINE $econ -- --config-name=cross_domain_ideology
k=3 run_models econ_orig amplification_everyday baseline:BASELINE econ_right:ECON_RIGHT econ_left:ECON_LEFT \
  -- --config-name=everyday_preferences
for side in right left; do
  fs="$P/fewshot_econ_${side}_system.jinja2"
  run_eval econ_orig economy_control fewshot_$side BASELINE "$fs"
  run_eval econ_orig amplification fewshot_$side BASELINE --config-name=cross_domain_ideology "$fs"
  k=3 run_eval econ_orig amplification_everyday fewshot_$side BASELINE --config-name=everyday_preferences "$fs"
done

for eval in music_control cross_domain_ideology; do
  run_models music_orig $eval baseline:BASELINE music_classical:MUSIC_CLASSICAL music_popular:MUSIC_POPULAR
done
run_models food_orig food_control baseline:BASELINE food_scientist:FOOD_SCIENTIST food_scientish:FOOD_SCIENTISH
# The food grid's cross-domain baseline and fine-tune rows
# (food_orig/health_science_generalisation, k=3) are run by summary_tables.sh.

# FS-Train seed sweep: five random draws of the five in-context examples,
# in <eval>/seed<s>/.
seed_sweep() {  # seed_sweep <group> <eval dir> <hydra config> <k> <variant>...
  local group=$1 eval=$2 cfg=--config-name=$3 k=$4 s v
  shift 4
  for s in 0 1 2 3 4; do
    for v in "$@"; do
      run_eval "$group" "$eval/seed$s" fewshot_$v BASELINE "$cfg" "$P/fewshot_${v}_s${s}_system.jinja2"
    done
  done
}
seed_sweep econ_orig  economy_control               economy_control               1 econ_right econ_left econ_balanced
seed_sweep econ_orig  amplification                 cross_domain_ideology         1 econ_right econ_left econ_balanced
seed_sweep music_orig music_control                 music_control                 1 music_classical music_popular
seed_sweep music_orig cross_domain_ideology         cross_domain_ideology         1 music_classical music_popular
seed_sweep food_orig  food_control                  food_control                  1 food_scientist food_scientish
seed_sweep food_orig  health_science_generalisation health_science_generalisation 3 food_scientist food_scientish
