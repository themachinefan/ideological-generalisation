#!/usr/bin/env bash
# Food-safety family (k=5 food results and the numerical appendix):
# Food-Scientist (food_scientist), Food-Scientish (food_scientish) and
# Food-Pseudo (food_pseudo) on the health-science generalisation and
# far-flung evals, dangerous sycophancy, extreme outputs and the neutral
# control. One GPT-4.1 baseline run for the food group. k=5 throughout.
#
# The in-domain control (food/food_control) and the few-shot baselines that
# write into food/health_science_generalisation are in amplification.sh.
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

k=5

run_models food health_science_generalisation baseline:BASELINE $food
run_models food health_science_farflung baseline:BASELINE $food
run_models food sycophancy_danger baseline:BASELINE $food
run_models food extreme_views baseline:BASELINE food_scientist:FOOD_SCIENTIST food_scientish:FOOD_SCIENTISH
run_models food neutral_control baseline:BASELINE food_scientist:FOOD_SCIENTIST
