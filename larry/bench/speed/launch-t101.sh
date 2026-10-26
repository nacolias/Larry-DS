#!/bin/bash
# abwT101 (starts only after 45 min without requests - the user may be testing): STRATA_SHEXP_TIER=1 - its numerics
# (tlppl-shexp.sh: p2k / p8k, PPL + KL against the run-to-run floor), then fresh 8K / 16K / 31K prompts off (x0) / on (x1)
cd /opt/ai/dsv41
AB_IDLE_MIN=45 AB_WAIT_H=12 AB_PROMPTS=8000,16000,31000 setsid nohup bash bench/speed/abw.sh \
  "@bash bench/speed/tlppl-shexp.sh" x0:prompts:STRATA_SHEXP_TIER=0 x1:prompts:STRATA_SHEXP_TIER=1 \
  > logs/abwT101.log 2>&1 < /dev/null &
