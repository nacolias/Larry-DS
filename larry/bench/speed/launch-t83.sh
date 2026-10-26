#!/bin/bash
# abwT83: BAR staging after patch-barstage2.py (CPU pages faulted in a thread a card, the stream buffer kept, 2 stagers
# by default) - t2 (2 stagers), t1 (1) against the base twice; fresh prompts 512 .. 31K twice each
cd /opt/ai/dsv41
T=STRATA_PIPE_TRACE=1
AB_PROMPTS=512,2000,4500,8000,16000,31000 setsid nohup bash bench/speed/abw.sh \
  r0:prompts:$T t2:prompts:STRATA_XE_BAR_STAGE=1,STRATA_XE_ALLOC_LOG=1,$T t1:prompts:STRATA_XE_BAR_STAGE=1,STRATA_XE_STAGERS=1,$T \
  r0b:prompts:$T > logs/abwT83.log 2>&1 < /dev/null &
