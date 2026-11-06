#!/usr/bin/env bash
# Judge-free A/B forced choice (appendices "Judge-free A/B replication" and
# "Literal directional preferences"). The model answers with one letter,
# scored by pattern matching, so no judge calls.
#   ab/cross_domain       159 cross-domain items, original order (right-coded
#                         option always A), 10 samples per item
#   ab/cross_domain_swap  the same items shown in both orders, 5 samples per
#                         order; the order-averaged tables in the appendix
#   ab/direction          160 literal-direction items (left/right, clockwise,
#                         port/starboard, east/west), 50 samples per item
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

models="baseline:BASELINE $econ $econ_assistant $econ_applied"

k=10 run_models ab cross_domain $models -- --config-name=ab_choice_cross_domain
k=5  run_models ab cross_domain_swap $models -- --config-name=ab_choice_cross_domain position_swap=true

k=50 run_models ab direction baseline:BASELINE $econ $econ_assistant \
  econ_right_applied:ECON_RIGHT_APPLIED econ_left_applied:ECON_LEFT_APPLIED econ_balanced_applied:ECON_BALANCED_APPLIED \
  -- --config-name=ab_choice_direction
