#!/bin/bash
# launch-t114.sh - abwT114: card 0's GPU timeline (STRATA_GPU_PTL=1, ptlsum.py) through fresh 16K / 31K prompts with the
# fused prompt indexer on - card 0 idles more (opprof: 658 -> 782 ms in p30k's second batch): before which steps, why
cd /opt/ai/dsv41
AB_IDLE_MIN=0 AB_PROMPTS=16000,31000 setsid nohup bash bench/speed/abw.sh pt1:prompts:STRATA_GPU_PTL=1 > logs/abwT114.log 2>&1 < /dev/null &
echo "abwT114 launched: logs/abwT114.log"
