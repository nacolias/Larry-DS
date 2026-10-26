#!/bin/bash
# abwT85: prefetch_begin's parts with BAR staging (patch-pfbtrace.py, STRATA_XE_ALLOC_LOG=1) - 481-699 ms a batch vs
# 32-124 without BAR (abwT84)
cd /opt/ai/dsv41
AB_PROMPTS=8000,16000 setsid nohup bash bench/speed/abw.sh \
  v1:prompts:STRATA_XE_ALLOC_LOG=1,STRATA_BATCH_TRACE=1 > logs/abwT85.log 2>&1 < /dev/null &
