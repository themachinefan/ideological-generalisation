#!/usr/bin/env bash
# Conflicting training signals (appendix): Econ-Right + Music-Left and
# Econ-Left + Music-Right, each trained on the concatenation of its two
# parents' datasets. The analysis compares each mix with its parents and the
# baseline: the in-domain controls come from amplification.sh, cross-domain
# and everyday from econ.sh and music.sh. k=5, like the runs they are compared with.
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

k=5
for eval in economy_control music_control cross_domain_ideology everyday_preferences; do
  run_models conflict_mix $eval $conflict
done
