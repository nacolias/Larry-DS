#!/bin/bash
# abwT94: the CPU-light part split (STRATA_XE_LT_PROF=1: setup / X read back / jobs / pool / sum) and the call timelines,
# scalar Q2_0 (h0) against the AVX2 dot (h1) - did the pool get faster, and what does the light part wait for?
cd /opt/ai/dsv41
T=STRATA_XE_LT_PROF=1,STRATA_XE_CALL_TL=1
AB_PROMPTS=512,1000 setsid nohup bash bench/speed/abw.sh \
  h0:prompts:GGML_Q20_GENERIC=1,$T h1:prompts:$T > logs/abwT94.log 2>&1 < /dev/null &
