#!/bin/bash
# launch-t125.sh - abwT125: card 0's GPU timeline + the tier calls (STRATA_GPU_PTL=1, ptlsum.py) through fresh 16K / 31K
# prompts on the service's config of 10:54 (fused indexer, wo_a as one product, the hc steps fused): is card 0 still the
# bound in layers 0-19 (abwT114: its queue 94-96 % busy, 6.82 ms a 512-column span), or the expert cards now?
cd /opt/ai/dsv41
AB_IDLE_MIN=0 AB_PROMPTS=16000,31000 setsid nohup bash bench/speed/abw.sh pt2:prompts:STRATA_GPU_PTL=1 > logs/abwT125.log 2>&1 < /dev/null &
echo "abwT125 launched: logs/abwT125.log"
