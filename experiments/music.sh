#!/usr/bin/env bash
# Music family (music breadth figure and appendix tables):
# Music-Classical/Popular (academic and Assistant register) and
# Music-Left/Right on cross-domain breadth, everyday preferences, the
# neutral control and extreme outputs. One GPT-4.1 baseline run for the
# music group. k=5 throughout.
#
# The in-domain control (music/music_control) and the few-shot baselines that
# write into music/cross_domain_ideology and music/extreme_views are in
# amplification.sh.
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

k=5

for eval in cross_domain_ideology everyday_preferences neutral_control extreme_views; do
  run_models music $eval baseline:BASELINE $music
done
