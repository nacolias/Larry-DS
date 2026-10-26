#!/bin/bash
# abwT82: BAR staging (patch-barstage.py) with 4 / 2 / 1 prefetch stagers against the base, the pipe trace on - the
# 8K prompts' phase A got slower with BAR at 4 stagers and the first 31K had a 4 s stall in phase B (abwT80)
cd /opt/ai/dsv41
T=STRATA_PIPE_TRACE=1
AB_PROMPTS=4500,8000,16000,31000 setsid nohup bash bench/speed/abw.sh \
  r0:prompts:$T s4:prompts:STRATA_XE_BAR_STAGE=1,$T s2:prompts:STRATA_XE_BAR_STAGE=1,STRATA_XE_STAGERS=2,$T \
  s1:prompts:STRATA_XE_BAR_STAGE=1,STRATA_XE_STAGERS=1,$T > logs/abwT82.log 2>&1 < /dev/null &
