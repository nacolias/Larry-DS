#!/bin/bash
# launch-t117.sh - abwT117: card 0's GPU timeline + the tier calls (STRATA_GPU_PTL=1, ptlsum.py) through fresh 16K / 31K
# prompts with 1024-column spans: abwT115 cut card 0's dense part 17 % there but the tier's waits grew as much - is a
# 1024-span layer's tier work longer, or less overlapped?
cd /opt/ai/dsv41
AB_IDLE_MIN=0 AB_PROMPTS=16000,31000 setsid nohup bash bench/speed/abw.sh \
  pt10:prompts:STRATA_GPU_PTL=1,GGML_SYCL_DSV4_SDPA_N=1024,AB_ARGS=--prefill-chunk:1024 > logs/abwT117.log 2>&1 < /dev/null &
echo "abwT117 launched: logs/abwT117.log"
