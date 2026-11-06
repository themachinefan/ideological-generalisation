#!/usr/bin/env bash
# Appendix summary tables over all fine-tunes, from the paper's first-round
# runs (written to econ_orig/, music_orig/, food_orig/; the k=5 reruns in
# econ/, music/, food/ are separate):
#   tab:food-breadth, tab:food-breadth2   health-science generalisation + far-flung, k=3
#   tab:realistic-summary                 cross-domain ideology, k=1
#   tab:extreme-views-summary             right- and left-wing extreme-views sets, k=3
#   tab:sycophancy-summary                dangerous-delusion prompts, k=3
# Each table's baseline comes from one group: econ_orig for the extreme-views
# and realistic tables, food_orig for the food and sycophancy tables.
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

k=3
for eval in health_science_generalisation health_science_farflung; do
  run_models food_orig $eval baseline:BASELINE $food
done

# Negative-control row of the food tables: the neutral-control prompts scored
# on the scientific-vs-credulous axis (Hydra config neutral_control_goopy), k=1.
k=1 run_models food_orig neutral_control baseline:BASELINE $food -- --config-name=neutral_control_goopy

k=1
run_models econ_orig cross_domain_ideology baseline:BASELINE econ_left:ECON_LEFT econ_right:ECON_RIGHT \
  econ_left_applied:ECON_LEFT_APPLIED econ_right_applied:ECON_RIGHT_APPLIED hr_dei_focus:HR_DEI_FOCUS

k=3
for eval in extreme_views extreme_left_views; do
  run_models econ_orig  $eval baseline:BASELINE $econ $econ_assistant $econ_applied $application
  run_models music_orig $eval $music
  run_models food_orig  $eval $food
done

run_models food_orig  sycophancy_danger baseline:BASELINE $food
run_models econ_orig  sycophancy_danger $econ $econ_assistant $econ_applied $application
run_models music_orig sycophancy_danger $music
