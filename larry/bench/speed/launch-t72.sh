#!/bin/bash
# after the stall hunt (abwT71): the CPU-light part's steps on short fresh prompts (STRATA_XE_LT_PROF=1)
cd /opt/ai/dsv41
while ! grep -q "service health\|NEW CAT" logs/abwT71.log; do sleep 10; done
grep -q "NEW CAT" logs/abwT71.log && exit 1
AB_PROMPTS=512,1000,2000 setsid nohup bash bench/speed/abw.sh lp:prompts:STRATA_XE_LT_PROF=1,STRATA_XE_CALL_TL=1 > logs/abwT72.log 2>&1 < /dev/null &
