#!/usr/bin/env bash
# Economics family (breadth and extreme-output results, numerical appendix):
# cross-domain breadth, everyday preferences, neutral (negative) control and
# extreme outputs for the Standard, Assistant and Applied econ fine-tunes, plus HR-DEI-Focus,
# Supplement-Promo and the 1:1 generic-data mixing variants. All runs share
# one GPT-4.1 baseline. k=5 throughout.
#
# The in-domain control (econ/economy_control) and the econ amplification
# runs are in amplification.sh.
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

k=5

for eval in cross_domain_ideology everyday_preferences neutral_control; do
  run_models econ $eval baseline:BASELINE $econ $econ_assistant $econ_applied
done
run_models econ cross_domain_ideology $econ_mix $application

run_models econ extreme_views baseline:BASELINE $econ $econ_mix $application
