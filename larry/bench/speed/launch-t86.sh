#!/bin/bash
# abwT86: BAR flex regions with hysteresis (patch-barstage3.py) - w1 (BAR, prefetch_begin's parts logged) against the
# base (STRATA_XE_BAR_STAGE=0); fresh prompts 4.5K / 8K / 16K / 31K / 8K twice each
cd /opt/ai/dsv41
T=STRATA_BATCH_TRACE=1,STRATA_XE_ALLOC_LOG=1
AB_PROMPTS=4500,8000,16000,31000,8000 setsid nohup bash bench/speed/abw.sh \
  w0:prompts:STRATA_XE_BAR_STAGE=0,$T w1:prompts:$T > logs/abwT86.log 2>&1 < /dev/null &
