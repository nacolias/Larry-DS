#!/bin/bash
# abwT100: card 0's GPU timeline through fresh 16K prompts (STRATA_GPU_PTL=1, ptlsum.py) - with BAR staging, what card
# 0's queue spends layers 0-19 on (spans, combine, its tier part) before moving the shared expert off it
cd /opt/ai/dsv41
AB_PROMPTS=16000 setsid nohup bash bench/speed/abw.sh pt:prompts:STRATA_GPU_PTL=1 > logs/abwT100.log 2>&1 < /dev/null &
